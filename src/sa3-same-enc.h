// sa3-same-enc.h: Stable Audio 3 SAME (taae_v2) autoencoder ENCODER via ggml.
//
// encode(audio[2, 4096*T]) -> latent[256, T].  Exact mirror of the decoder in sa3-same.h:
//   patch stereo 44.1kHz into [512, 16*T] (channel-major, inverse of the decoder unpatch) ->
//   weightnorm 1x1 conv 512->1536 -> per 16-audio-token group append 1 learned new_token => N=17*T ->
//   12 DynamicTanh differential-attn transformer blocks (banded |i-j|<=17 + partial NEOX RoPE(32), GLU
//   FF with SiLU on ALL blocks -- encoder has no sin gates) -> keep the LAST token of each 17-group
//   (the new_token's output) => T frames -> Linear 1536->256 -> softnorm bottleneck encode
//   (z = (x*scaling_factor + bias) / running_std).  Shares all block helpers with the decoder.
//   Validated vs enc_latent.npy golden (reference pretransform.encode of the same audio).
#pragma once
#include "backend.h"
#include "gguf-weights.h"
#include "sa3-attn.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#define SA3_ENC_MAX_LAYERS 16

struct SA3EncConfig {
    int   latent_dim = 256, embed = 1536, depth = 12, n_heads = 24, head_dim = 64;
    int   stride = 16, sub_chunk = 17, band = 17, in_ch = 512, patch = 256, audio_ch = 2;
    int   rope_ndims = 32;
    float rope_theta = 10000.0f, attn_scale = 0.125f;
};

struct SA3EncDyt { struct ggml_tensor *a, *g, *b; };
struct SA3EncLayer {
    SA3EncDyt pre_norm, ff_norm, sa_qn, sa_kn;
    struct ggml_tensor *sa_qkv, *sa_out, *ff_in_w, *ff_in_b, *ff_out_w, *ff_out_b;
};

struct SA3Enc {
    SA3EncConfig cfg;
    SA3EncLayer  layers[SA3_ENC_MAX_LAYERS];
    struct ggml_tensor *map_w, *map_b, *new_tokens, *out_w, *out_b;
    struct ggml_tensor *bn_scale, *bn_bias;   // softnorm scaling_factor / bias [latent_dim]
    float running_std;

    ggml_backend_t backend, cpu_backend;
    ggml_backend_sched_t sched;
    WeightCtx wctx;
};

static bool sa3enc_load(SA3Enc * m, const char * path) {
    BackendPair bp = backend_init("SA3-ENC");
    m->backend = bp.backend; m->cpu_backend = bp.cpu_backend;
    m->sched = backend_sched_new(bp, 32768);
    sa3_set_flash(bp.has_gpu);  // flash default-on for GPU (see sa3-attn.h)
    m->cfg = {};

    GGUFModel gf;
    if (!gf_load(&gf, path)) return false;
    wctx_init(&m->wctx, 256);
    auto T  = [&](const std::string & n) { return gf_load_tensor(&m->wctx, gf, n); };
    auto Tf = [&](const std::string & n) { return gf_load_tensor_f32(&m->wctx, gf, n); };

    const float * rs = (const float *) gf_get_data(gf, "bn.running_std");
    m->running_std = rs ? rs[0] : 1.0f;
    m->bn_scale = Tf("bn.scale"); m->bn_bias = Tf("bn.bias");

    m->map_w = T("enc.map.w"); m->map_b = Tf("enc.map.b");
    m->new_tokens = Tf("enc.new_tokens");
    m->out_w = T("enc.out.w"); m->out_b = Tf("enc.out.b");

    for (int i = 0; i < m->cfg.depth; i++) {
        char p[32]; snprintf(p, sizeof(p), "enc.blk.%d.", i);
        std::string s(p);
        SA3EncLayer * ly = &m->layers[i];
        ly->pre_norm = { Tf(s+"pre_norm.a"), Tf(s+"pre_norm.g"), Tf(s+"pre_norm.b") };
        ly->ff_norm  = { Tf(s+"ff_norm.a"),  Tf(s+"ff_norm.g"),  Tf(s+"ff_norm.b") };
        ly->sa_qn    = { Tf(s+"sa_qn.a"), Tf(s+"sa_qn.g"), Tf(s+"sa_qn.b") };
        ly->sa_kn    = { Tf(s+"sa_kn.a"), Tf(s+"sa_kn.g"), Tf(s+"sa_kn.b") };
        ly->sa_qkv = T(s+"sa_qkv"); ly->sa_out = T(s+"sa_out");
        ly->ff_in_w = T(s+"ff_in.w"); ly->ff_in_b = Tf(s+"ff_in.b");
        ly->ff_out_w = T(s+"ff_out.w"); ly->ff_out_b = Tf(s+"ff_out.b");
        // encoder uses SiLU on all blocks (no sinusoidal gates)
    }
    if (!wctx_alloc(&m->wctx, m->backend)) { gf_close(&gf); return false; }
    gf_close(&gf);
    fprintf(stderr, "[SA3-ENC] loaded %dL embed=%d running_std=%.6f\n", m->cfg.depth, m->cfg.embed, m->running_std);
    return true;
}

// encode audio [2, Nsamp] planar (c*Nsamp+s), Nsamp must be a multiple of patch*stride (4096).
// Returns T (= Nsamp/4096); latent_out filled [latent_dim * T] token-major (ggml [C,T], t*C+c).
static int sa3enc_encode(SA3Enc * m, const float * audio, int Nsamp, std::vector<float> & latent_out) {
    const SA3EncConfig & cf = m->cfg;
    int E = cf.embed, D = cf.head_dim, Nh = cf.n_heads, C = cf.latent_dim;
    int P = cf.patch, ICH = cf.in_ch;                 // 256, 512
    int Taud = Nsamp / P;                             // audio patch tokens = 16*T
    int Tlat = Taud / cf.stride;                      // latent frames
    int sc = cf.sub_chunk, N = sc * Tlat;             // packed seq 17*T

    // ---- host-side patch: [in_ch=512, Taud] token-major, inverse of decoder unpatch ----
    // patch[token l, channel c*P+h] = audio[c*Nsamp + l*P + h]
    std::vector<float> patch((size_t) ICH * Taud);
    for (int l = 0; l < Taud; l++)
        for (int c = 0; c < cf.audio_ch; c++)
            for (int h = 0; h < P; h++)
                patch[(size_t) l * ICH + (size_t) c * P + h] = audio[(size_t) c * Nsamp + (size_t) l * P + h];

    size_t ctx_size = 8192 * ggml_tensor_overhead() + ggml_graph_overhead_custom(65536, false);
    struct ggml_init_params gp = { ctx_size, NULL, true };
    struct ggml_context * ctx = ggml_init(gp);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 65536, false);

    struct ggml_tensor * in_p   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ICH, Taud); ggml_set_input(in_p);
    struct ggml_tensor * in_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);         ggml_set_input(in_pos);
    struct ggml_tensor * in_mask= ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, N);      ggml_set_input(in_mask);

    // mapping conv 1x1: 512 -> 1536
    struct ggml_tensor * mw = ggml_reshape_2d(ctx, m->map_w, m->map_w->ne[1], m->map_w->ne[2]);  // [512,1536]
    struct ggml_tensor * x = ggml_add(ctx, ggml_mul_mat(ctx, mw, in_p), m->map_b);   // [E, Taud]

    // group every 16 audio tokens, append 1 new_token at the END => [E,17,T]
    struct ggml_tensor * x3 = ggml_reshape_3d(ctx, x, E, cf.stride, Tlat);           // [E,16,T]
    struct ggml_tensor * nt = ggml_reshape_3d(ctx, m->new_tokens, E, 1, 1);
    nt = ggml_repeat_4d(ctx, nt, E, 1, Tlat, 1);                                     // [E,1,T]
    struct ggml_tensor * packed = ggml_concat(ctx, x3, nt, 1);                       // [E,17,T]
    x = ggml_reshape_2d(ctx, ggml_cont(ctx, packed), E, N);                          // [E, 17*T]

    for (int i = 0; i < cf.depth; i++) {
        SA3EncLayer * ly = &m->layers[i];
        struct ggml_tensor * res = x;
        struct ggml_tensor * h = sa3_dyt(ctx, x, ly->pre_norm.a, ly->pre_norm.g, ly->pre_norm.b);
        struct ggml_tensor * qkv = ggml_mul_mat(ctx, ly->sa_qkv, h);                 // [5E,N]
        struct ggml_tensor * q  = sa3_heads_dyt(ctx, sa3_chunk(ctx,qkv,E,N,0),   D,Nh,N, ly->sa_qn.a,ly->sa_qn.g,ly->sa_qn.b, in_pos, cf.rope_ndims, cf.rope_theta);
        struct ggml_tensor * k  = sa3_heads_dyt(ctx, sa3_chunk(ctx,qkv,E,N,E),   D,Nh,N, ly->sa_kn.a,ly->sa_kn.g,ly->sa_kn.b, in_pos, cf.rope_ndims, cf.rope_theta);
        struct ggml_tensor * v  = sa3_heads_dyt(ctx, sa3_chunk(ctx,qkv,E,N,2*E), D,Nh,N, NULL,NULL,NULL, NULL,0,0);
        struct ggml_tensor * qd = sa3_heads_dyt(ctx, sa3_chunk(ctx,qkv,E,N,3*E), D,Nh,N, ly->sa_qn.a,ly->sa_qn.g,ly->sa_qn.b, in_pos, cf.rope_ndims, cf.rope_theta);
        struct ggml_tensor * kd = sa3_heads_dyt(ctx, sa3_chunk(ctx,qkv,E,N,4*E), D,Nh,N, ly->sa_kn.a,ly->sa_kn.g,ly->sa_kn.b, in_pos, cf.rope_ndims, cf.rope_theta);
        struct ggml_tensor * attn = sa3_diff_attn(ctx, q,qd,k,kd,v, in_mask, D,Nh,N, cf.attn_scale);
        attn = ggml_mul_mat(ctx, ly->sa_out, attn);
        x = ggml_add(ctx, res, attn);

        struct ggml_tensor * res2 = x;
        struct ggml_tensor * hf = sa3_dyt(ctx, x, ly->ff_norm.a, ly->ff_norm.g, ly->ff_norm.b);
        struct ggml_tensor * proj = ggml_add(ctx, ggml_mul_mat(ctx, ly->ff_in_w, hf), ly->ff_in_b);  // [2*inter,N]
        int inter = (int) ly->ff_out_w->ne[0];
        struct ggml_tensor * val  = sa3_chunk(ctx, proj, inter, N, 0);
        struct ggml_tensor * gate = sa3_chunk(ctx, proj, inter, N, inter);
        struct ggml_tensor * ff = ggml_mul(ctx, val, ggml_silu(ctx, gate));          // SiLU on all enc blocks
        ff = ggml_add(ctx, ggml_mul_mat(ctx, ly->ff_out_w, ff), ly->ff_out_b);       // [E,N]
        x = ggml_add(ctx, res2, ff);
    }

    // keep the LAST token of each 17-group (index 16 = the new_token output): [E,17,T] -> [E,T]
    struct ggml_tensor * x3o = ggml_reshape_3d(ctx, x, E, sc, Tlat);
    struct ggml_tensor * last = ggml_cont(ctx, ggml_view_3d(ctx, x3o, E, 1, Tlat,
                                          x3o->nb[1], x3o->nb[2], (size_t) cf.stride * x3o->nb[1]));  // [E,1,T]
    last = ggml_reshape_2d(ctx, last, E, Tlat);                                      // [E,T]

    // project 1536 -> 256, then softnorm encode: (x*scale + bias) / running_std
    struct ggml_tensor * z = ggml_add(ctx, ggml_mul_mat(ctx, m->out_w, last), m->out_b);  // [C,T]
    z = ggml_add(ctx, ggml_mul(ctx, z, m->bn_scale), m->bn_bias);
    z = ggml_scale(ctx, z, 1.0f / m->running_std);
    ggml_set_output(z);
    ggml_build_forward_expand(gf, z);

    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) { fprintf(stderr,"[SA3-ENC] alloc fail\n"); exit(1); }
    if (getenv("SA3_PROBE")) {
        fprintf(stderr, "[probe] ENC Taud=%d N=%d compute-buffer = %.0f MiB\n", Taud, N,
                ggml_backend_sched_get_buffer_size(m->sched, m->backend) / 1048576.0);
        ggml_backend_sched_reset(m->sched); ggml_free(ctx); latent_out.assign((size_t)C*Tlat,0.f); return Tlat;
    }

    ggml_backend_tensor_set(in_p, patch.data(), 0, patch.size() * sizeof(float));
    std::vector<int> pos(N); for (int i = 0; i < N; i++) pos[i] = i;
    ggml_backend_tensor_set(in_pos, pos.data(), 0, pos.size() * sizeof(int));
    std::vector<float> md((size_t) N * N);
    for (int qi = 0; qi < N; qi++)
        for (int kj = 0; kj < N; kj++)
            md[(size_t) qi * N + kj] = (abs(qi - kj) <= cf.band) ? 0.0f : -INFINITY;
    ggml_backend_tensor_set(in_mask, md.data(), 0, md.size() * sizeof(float));

    ggml_backend_sched_graph_compute(m->sched, gf);
    latent_out.assign((size_t) C * Tlat, 0.0f);
    ggml_backend_tensor_get(z, latent_out.data(), 0, latent_out.size() * sizeof(float));
    ggml_backend_sched_reset(m->sched);
    ggml_free(ctx);
    return Tlat;
}

// Chunked encode: split audio into overlapping windows (bounds the N=17*T attention memory) and
// stitch latents. Band is +-17 packed tokens (~+-1 latent frame); overlap >> 1 -> seamless. Mirrors
// sa3same_decode_chunked. audio [2,Nsamp] planar. latent_out [C, Tlat] token-major. Returns Tlat.
static int sa3enc_encode_chunked(SA3Enc * m, const float * audio, int Nsamp, std::vector<float> & latent_out,
                                 int chunk = 128, int overlap = 32) {
    const int C = m->cfg.latent_dim, P = m->cfg.patch * m->cfg.stride;  // samples per latent frame = 4096
    int Ttot = Nsamp / P;
    if (Ttot <= chunk) return sa3enc_encode(m, audio, Nsamp, latent_out);
    latent_out.assign((size_t) C * Ttot, 0.0f);
    int step = chunk - overlap, ho = overlap / 2;
    std::vector<float> seg, segaudio;
    for (int a = 0; a < Ttot; a += step) {
        int b = a + chunk; if (b > Ttot) b = Ttot;
        int ks = (a == 0) ? 0 : a + ho;
        int ke = (b == Ttot) ? b : b - ho;
        int segNsamp = (b - a) * P;
        segaudio.assign((size_t) 2 * segNsamp, 0.0f);
        for (int c = 0; c < 2; c++)
            for (int s = 0; s < segNsamp; s++)
                segaudio[(size_t) c * segNsamp + s] = audio[(size_t) c * Nsamp + (size_t) a * P + s];
        sa3enc_encode(m, segaudio.data(), segNsamp, seg);  // [C, b-a] token-major
        for (int g = ks; g < ke; g++) {
            int loc = g - a;
            for (int c = 0; c < C; c++)
                latent_out[(size_t) g * C + c] = seg[(size_t) loc * C + c];
        }
        fprintf(stderr, "[SA3-ENC] chunk [%d,%d) kept [%d,%d)\n", a, b, ks, ke);
        if (b == Ttot) break;
    }
    return Ttot;
}

static void sa3enc_free(SA3Enc * m) {
    if (m->sched) ggml_backend_sched_free(m->sched);
    backend_release(m->backend, m->cpu_backend);
    wctx_free(&m->wctx);
    *m = {};
}
