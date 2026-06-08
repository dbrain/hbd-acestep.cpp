// sa3-same.h: Stable Audio 3 SAME (taae_v2) autoencoder DECODER via ggml.
//
// decode(latent[256,T]) -> stereo audio[2, 4096*T] @ 44.1kHz.
// Pipeline (see subagent spec): softnorm un-normalize (z*running_std) -> Linear 256->1536 ->
// per latent frame build a 17-token group (1 real + 16 learned new_tokens) packed to seq N=17*T ->
// 12 DynamicTanh differential-attn transformer blocks with banded attention (|i-j|<=17) + partial
// NEOX RoPE(32) + GLU FF (Sin(pi*) for blocks 5..11, SiLU for 0..4) -> keep last 16 of each group
// (drop real token) => 16*T frames -> weight-norm 1x1 conv 1536->512 -> channel-major unpatch to [2,4096*T].
// No AdaLN / cross-attn / global cond.  Validated vs latent_final.npy -> audio.npy golden.
#pragma once
#include "backend.h"
#include "gguf-weights.h"
#include "sa3-attn.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#define SA3_SAME_MAX_LAYERS 16

struct SA3SameConfig {
    int   latent_dim = 256, embed = 1536, depth = 12, n_heads = 24, head_dim = 64;
    int   stride = 16, sub_chunk = 17, band = 17, out_ch = 512, patch = 256, audio_ch = 2;
    int   rope_ndims = 32, sinusoidal_blocks = 8;
    float rope_theta = 10000.0f, attn_scale = 0.125f;
};

struct SA3SameDyt { struct ggml_tensor *a, *g, *b; };
struct SA3SameLayer {
    SA3SameDyt pre_norm, ff_norm, sa_qn, sa_kn;
    struct ggml_tensor *sa_qkv, *sa_out, *ff_in_w, *ff_in_b, *ff_out_w, *ff_out_b;
    bool sinusoidal;
};

struct SA3Same {
    SA3SameConfig cfg;
    SA3SameLayer  layers[SA3_SAME_MAX_LAYERS];
    struct ggml_tensor *dec_in_w, *dec_in_b, *new_tokens, *map_w, *map_b;
    float running_std;

    ggml_backend_t backend, cpu_backend;
    ggml_backend_sched_t sched;
    WeightCtx wctx;
};

static bool sa3same_load(SA3Same * m, const char * path) {
    BackendPair bp = backend_init("SA3-SAME");
    m->backend = bp.backend; m->cpu_backend = bp.cpu_backend;
    m->sched = backend_sched_new(bp, 32768);
    sa3_set_flash(bp.has_gpu);  // flash default-on for GPU (see sa3-attn.h)
    m->cfg = {};

    GGUFModel gf;
    if (!gf_load(&gf, path)) return false;
    wctx_init(&m->wctx, 256);
    auto T  = [&](const std::string & n) { return gf_load_tensor(&m->wctx, gf, n); };
    auto Tf = [&](const std::string & n) { return gf_load_tensor_f32(&m->wctx, gf, n); };

    // running_std scalar read directly from mmap (f32)
    const float * rs = (const float *) gf_get_data(gf, "bn.running_std");
    m->running_std = rs ? rs[0] : 1.0f;

    m->dec_in_w = T("dec.in.w"); m->dec_in_b = Tf("dec.in.b");
    m->new_tokens = Tf("dec.new_tokens");
    m->map_w = T("dec.map.w"); m->map_b = Tf("dec.map.b");

    for (int i = 0; i < m->cfg.depth; i++) {
        char p[32]; snprintf(p, sizeof(p), "dec.blk.%d.", i);
        std::string s(p);
        SA3SameLayer * ly = &m->layers[i];
        ly->pre_norm = { Tf(s+"pre_norm.a"), Tf(s+"pre_norm.g"), Tf(s+"pre_norm.b") };
        ly->ff_norm  = { Tf(s+"ff_norm.a"),  Tf(s+"ff_norm.g"),  Tf(s+"ff_norm.b") };
        ly->sa_qn    = { Tf(s+"sa_qn.a"), Tf(s+"sa_qn.g"), Tf(s+"sa_qn.b") };
        ly->sa_kn    = { Tf(s+"sa_kn.a"), Tf(s+"sa_kn.g"), Tf(s+"sa_kn.b") };
        ly->sa_qkv = T(s+"sa_qkv"); ly->sa_out = T(s+"sa_out");
        ly->ff_in_w = T(s+"ff_in.w"); ly->ff_in_b = Tf(s+"ff_in.b");
        ly->ff_out_w = T(s+"ff_out.w"); ly->ff_out_b = Tf(s+"ff_out.b");
        // sinusoidal selection: (depth - i) < sinusoidal_blocks  => Sin(pi*) gate
        ly->sinusoidal = (m->cfg.depth - i) < m->cfg.sinusoidal_blocks;
    }
    if (!wctx_alloc(&m->wctx, m->backend)) { gf_close(&gf); return false; }
    gf_close(&gf);
    fprintf(stderr, "[SA3-SAME] loaded %dL embed=%d running_std=%.6f\n", m->cfg.depth, m->cfg.embed, m->running_std);
    return true;
}

// decode latent z_t [latent_dim * T] in ggml [C=256, T] token-major. Returns audio [2 * (4096*T)] (c*N+s).
static int sa3same_decode(SA3Same * m, const float * z_t, int T, std::vector<float> & audio_out) {
    const SA3SameConfig & cf = m->cfg;
    int E = cf.embed, D = cf.head_dim, Nh = cf.n_heads, C = cf.latent_dim;
    int sc = cf.sub_chunk, N = sc * T, Lup = cf.stride * T;

    size_t ctx_size = 8192 * ggml_tensor_overhead() + ggml_graph_overhead_custom(65536, false);
    struct ggml_init_params gp = { ctx_size, NULL, true };
    struct ggml_context * ctx = ggml_init(gp);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 65536, false);

    struct ggml_tensor * in_z   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, T);   ggml_set_input(in_z);
    struct ggml_tensor * in_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);       ggml_set_input(in_pos);
    struct ggml_tensor * in_mask= ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, N);    ggml_set_input(in_mask);

    // bottleneck + linear-in
    struct ggml_tensor * z = ggml_scale(ctx, in_z, m->running_std);                // [C,T]
    struct ggml_tensor * x = ggml_add(ctx, ggml_mul_mat(ctx, m->dec_in_w, z), m->dec_in_b);  // [E,T]

    // build packed seq [E, 17, T]: real token first, then 16 new_tokens
    struct ggml_tensor * x3 = ggml_reshape_3d(ctx, x, E, 1, T);
    struct ggml_tensor * nt = ggml_reshape_3d(ctx, m->new_tokens, E, 1, 1);
    nt = ggml_repeat_4d(ctx, nt, E, cf.stride, T, 1);                              // [E,16,T]
    struct ggml_tensor * packed = ggml_concat(ctx, x3, nt, 1);                     // [E,17,T]
    x = ggml_reshape_2d(ctx, ggml_cont(ctx, packed), E, N);                        // [E, 17*T]

    for (int i = 0; i < cf.depth; i++) {
        SA3SameLayer * ly = &m->layers[i];
        // self-attn (DynamicTanh pre-norm, differential, banded, partial rope)
        struct ggml_tensor * res = x;
        struct ggml_tensor * h = sa3_dyt(ctx, x, ly->pre_norm.a, ly->pre_norm.g, ly->pre_norm.b);
        struct ggml_tensor * qkv = ggml_mul_mat(ctx, ly->sa_qkv, h);              // [5E, N]
        struct ggml_tensor * q  = sa3_heads_dyt(ctx, sa3_chunk(ctx,qkv,E,N,0),   D,Nh,N, ly->sa_qn.a,ly->sa_qn.g,ly->sa_qn.b, in_pos, cf.rope_ndims, cf.rope_theta);
        struct ggml_tensor * k  = sa3_heads_dyt(ctx, sa3_chunk(ctx,qkv,E,N,E),   D,Nh,N, ly->sa_kn.a,ly->sa_kn.g,ly->sa_kn.b, in_pos, cf.rope_ndims, cf.rope_theta);
        struct ggml_tensor * v  = sa3_heads_dyt(ctx, sa3_chunk(ctx,qkv,E,N,2*E), D,Nh,N, NULL,NULL,NULL, NULL,0,0);
        struct ggml_tensor * qd = sa3_heads_dyt(ctx, sa3_chunk(ctx,qkv,E,N,3*E), D,Nh,N, ly->sa_qn.a,ly->sa_qn.g,ly->sa_qn.b, in_pos, cf.rope_ndims, cf.rope_theta);
        struct ggml_tensor * kd = sa3_heads_dyt(ctx, sa3_chunk(ctx,qkv,E,N,4*E), D,Nh,N, ly->sa_kn.a,ly->sa_kn.g,ly->sa_kn.b, in_pos, cf.rope_ndims, cf.rope_theta);
        struct ggml_tensor * attn = sa3_diff_attn(ctx, q,qd,k,kd,v, in_mask, D,Nh,N, cf.attn_scale);
        attn = ggml_mul_mat(ctx, ly->sa_out, attn);
        x = ggml_add(ctx, res, attn);

        // FF (DynamicTanh, GLU; activation Sin(pi*) or SiLU per block)
        struct ggml_tensor * res2 = x;
        struct ggml_tensor * hf = sa3_dyt(ctx, x, ly->ff_norm.a, ly->ff_norm.g, ly->ff_norm.b);
        struct ggml_tensor * proj = ggml_add(ctx, ggml_mul_mat(ctx, ly->ff_in_w, hf), ly->ff_in_b);  // [2*inter,N]
        int inter = (int) ly->ff_out_w->ne[0];
        struct ggml_tensor * val  = sa3_chunk(ctx, proj, inter, N, 0);
        struct ggml_tensor * gate = sa3_chunk(ctx, proj, inter, N, inter);
        struct ggml_tensor * act = ly->sinusoidal ? ggml_sin(ctx, ggml_scale(ctx, gate, (float) M_PI))
                                                   : ggml_silu(ctx, gate);
        struct ggml_tensor * ff = ggml_mul(ctx, val, act);
        ff = ggml_add(ctx, ggml_mul_mat(ctx, ly->ff_out_w, ff), ly->ff_out_b);    // [E,N]
        x = ggml_add(ctx, res2, ff);
    }

    // keep last 16 of each 17-group: reshape [E,17,T] -> view [:,1:17,:] -> [E,16,T] -> [E,16*T]
    struct ggml_tensor * x3o = ggml_reshape_3d(ctx, x, E, sc, T);
    struct ggml_tensor * up = ggml_cont(ctx, ggml_view_3d(ctx, x3o, E, cf.stride, T,
                                       x3o->nb[1], x3o->nb[2], (size_t) 1 * x3o->nb[1]));  // [E,16,T]
    up = ggml_reshape_2d(ctx, up, E, Lup);                                        // [E, 16*T]
    // mapping conv 1x1: 1536->512
    struct ggml_tensor * w2 = ggml_reshape_2d(ctx, m->map_w, m->map_w->ne[1], m->map_w->ne[2]);  // [1536,512]
    struct ggml_tensor * dec = ggml_add(ctx, ggml_mul_mat(ctx, w2, up), m->map_b);  // [512, 16*T]
    ggml_set_output(dec);
    ggml_build_forward_expand(gf, dec);

    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) { fprintf(stderr,"[SA3-SAME] alloc fail\n"); exit(1); }
    if (getenv("SA3_PROBE")) {
        fprintf(stderr, "[probe] SAME T=%d N=%d compute-buffer = %.0f MiB\n", T, N,
                ggml_backend_sched_get_buffer_size(m->sched, m->backend) / 1048576.0);
        ggml_backend_sched_reset(m->sched); ggml_free(ctx); audio_out.assign((size_t)2*Lup*cf.patch,0.f); return Lup*cf.patch;
    }

    ggml_backend_tensor_set(in_z, z_t, 0, (size_t) C * T * sizeof(float));
    std::vector<int> pos(N); for (int i = 0; i < N; i++) pos[i] = i;
    ggml_backend_tensor_set(in_pos, pos.data(), 0, pos.size() * sizeof(int));
    std::vector<float> md((size_t) N * N);
    for (int qi = 0; qi < N; qi++)
        for (int kj = 0; kj < N; kj++)
            md[(size_t) qi * N + kj] = (abs(qi - kj) <= cf.band) ? 0.0f : -INFINITY;
    ggml_backend_tensor_set(in_mask, md.data(), 0, md.size() * sizeof(float));

    ggml_backend_sched_graph_compute(m->sched, gf);

    // read [512, 16*T] (ggml token-major: l*512 + k) and unpatch channel-major
    std::vector<float> decbuf((size_t) cf.out_ch * Lup);
    ggml_backend_tensor_get(dec, decbuf.data(), 0, decbuf.size() * sizeof(float));
    ggml_backend_sched_reset(m->sched);
    ggml_free(ctx);

    int Nsamp = Lup * cf.patch;  // 16*T * 256 = 4096*T
    audio_out.assign((size_t) cf.audio_ch * Nsamp, 0.0f);
    for (int c = 0; c < cf.audio_ch; c++)
        for (int l = 0; l < Lup; l++)
            for (int h = 0; h < cf.patch; h++)
                audio_out[(size_t) c * Nsamp + (size_t) l * cf.patch + h] =
                    decbuf[(size_t) l * cf.out_ch + (size_t) c * cf.patch + h];
    return Nsamp;
}

// Chunked decode: split latent into overlapping windows (bounds the N=17*T attention memory) and
// stitch. Band is +-17 packed tokens (~+-1 latent frame), so overlap >> 1 -> seamless. Matches the
// reference chunked_decode. z_t [C,T] token-major. Returns total samples; audio_out [2,N] planar.
static int sa3same_decode_chunked(SA3Same * m, const float * z_t, int T, std::vector<float> & audio_out,
                                  int chunk = 128, int overlap = 32) {
    // SA3_SAME_CHUNK / SA3_SAME_OVERLAP override the defaults (chunk-size sweep / tuning).
    if (const char * e = getenv("SA3_SAME_CHUNK"))   { int v = atoi(e); if (v > 0) chunk = v; }
    if (const char * e = getenv("SA3_SAME_OVERLAP")) { int v = atoi(e); if (v >= 0) overlap = v; }
    const int C = m->cfg.latent_dim, P = m->cfg.patch * m->cfg.stride;  // samples per latent frame = 4096
    if (T <= chunk) return sa3same_decode(m, z_t, T, audio_out);
    int Ntot = T * P;
    audio_out.assign((size_t) 2 * Ntot, 0.0f);
    int step = chunk - overlap, ho = overlap / 2;
    std::vector<float> seg;
    for (int a = 0; a < T; a += step) {
        int b = a + chunk; if (b > T) b = T;
        int ks = (a == 0) ? 0 : a + ho;
        int ke = (b == T) ? b : b - ho;
        sa3same_decode(m, z_t + (size_t) a * C, b - a, seg);  // [2,(b-a)*P]
        int segN = (b - a) * P;
        for (int c = 0; c < 2; c++)
            for (int g = ks; g < ke; g++) {
                int loc = g - a;
                for (int s = 0; s < P; s++)
                    audio_out[(size_t) c * Ntot + (size_t) g * P + s] = seg[(size_t) c * segN + (size_t) loc * P + s];
            }
        fprintf(stderr, "[SA3-SAME] chunk [%d,%d) kept [%d,%d)\n", a, b, ks, ke);
        if (b == T) break;
    }
    return Ntot;
}

static void sa3same_free(SA3Same * m) {
    if (m->sched) ggml_backend_sched_free(m->sched);
    backend_release(m->backend, m->cpu_backend);
    wctx_free(&m->wctx);
    *m = {};
}
