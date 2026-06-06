// songgen-htdemucs.h : HTDemucs (hybrid transformer Demucs) source separator (GGML).
//
// Separates a 44.1kHz stereo mix into 4 stems [drums,bass,other,vocals] via two parallel
// U-Nets (spectrogram + time) fused by a 5-layer cross-transformer bottleneck.
//
// Pipeline (single 7.8s segment = 343980 samples):
//   [host]  STFT(nfft4096,hop1024,center,reflect,normalized) -> complex z[2,2048,336]
//           CAC: mag = [re0,im0,re1,im1] -> [4,2048,336]; per-tensor normalize.
//   [ggml]  spec encoder 0..3 (Conv2d kernel[8,1] s[4,1] p[2,0] -> GLU rewrite -> DConv)
//           time encoder 0..3 (Conv1d k8 s4 p2 -> GLU rewrite -> DConv)
//           + freq embedding after encoder0.
//   [ggml]  channel up 384->512 ; CrossTransformerEncoder(5 layers [self,cross,...]) ; down 512->384.
//   [ggml]  decoder 0..3 (rewrite GLU -> ConvTranspose2d, skip add, NO dconv) for spec,
//           tdecoder 0..3 for time.
//   [host]  iSTFT spec branch -> wave ; denormalize ; sum time+spec -> 4 stems [4,2,343980].
//
// dconv_mode=1: DConv in ENCODERS only. tencoder/decoder/tdecoder dconv keys are unused.
#pragma once

#include "backend.h"
#include "gguf-weights.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

// ---------------- weight structs ----------------
struct HtdDConvLayer {
    struct ggml_tensor *c1w, *c1b;      // conv1d [hidden, C, 3] -> ggml conv weight
    struct ggml_tensor *gn1w, *gn1b;    // groupnorm(1, hidden)
    struct ggml_tensor *c3w, *c3b;      // conv1d [2C, hidden, 1]
    struct ggml_tensor *gn4w, *gn4b;    // groupnorm(1, 2C)
    struct ggml_tensor *scale;          // layerscale [C]
    int                 dilation;
};

struct HtdEncLayer {
    struct ggml_tensor *conv_w, *conv_b;      // conv (spec: 2d [1,8,IC,OC]; time: 1d)
    struct ggml_tensor *rw_w, *rw_b;          // rewrite 1x1 -> 2*OC
    HtdDConvLayer       dconv[2];
    int                 chin, chout;
    bool                freq;
};

struct HtdDecLayer {
    struct ggml_tensor *ctr_w, *ctr_b;        // conv_transpose (col2im GEMM layout [IC, K*OC])
    struct ggml_tensor *rw_w, *rw_b;          // rewrite (spec 3x3 / time 3)
    int                 chin, chout, ktr;
    bool                freq, last;
};

struct HtdTLayer {
    bool                is_cross;
    struct ggml_tensor *in_proj_w, *in_proj_b;   // [3*dim, dim]
    struct ggml_tensor *out_proj_w, *out_proj_b;
    struct ggml_tensor *l1w, *l1b, *l2w, *l2b;
    struct ggml_tensor *n1w, *n1b, *n2w, *n2b, *n3w, *n3b;  // n3 only for cross
    struct ggml_tensor *nout_w, *nout_b;
    struct ggml_tensor *g1, *g2;                 // layerscale gamma
};

struct SonggenHTDemucs {
    int   depth, t_layers, t_dim, t_heads, t_ffn, transformer_ch, nfft, hop, sample_rate, segment_samples;
    float freq_emb_scale, max_period;

    HtdEncLayer enc[4], tenc[4];
    HtdDecLayer dec[4], tdec[4];
    HtdTLayer   tx[5], tx_t[5];

    struct ggml_tensor *freq_emb;                // [48, 512] -> stored [emb_dim, num]
    struct ggml_tensor *up_w, *up_b, *down_w, *down_b, *up_t_w, *up_t_b, *down_t_w, *down_t_b;
    struct ggml_tensor *normin_w, *normin_b, *normin_t_w, *normin_t_b;

    WeightCtx            wctx;
    ggml_backend_t       backend;
    ggml_backend_t       cpu_backend;
    ggml_backend_sched_t sched;
};

// ---------------- conv weight loaders (torch -> ggml f32) ----------------
// Conv2d torch [OC,IC,KH,KW] -> ggml [KW,KH,IC,OC] f32.
static struct ggml_tensor * htd_conv2d_w(WeightCtx * w, const GGUFModel & gf, const std::string & name) {
    struct ggml_tensor * src = ggml_get_tensor(gf.meta, name.c_str());
    int oc = (int) src->ne[3], ic = (int) src->ne[2], kh = (int) src->ne[1], kw = (int) src->ne[0];
    const ggml_bf16_t * v = (const ggml_bf16_t *) gf_get_data(gf, name.c_str());
    auto buf = std::make_unique<float[]>((size_t) kw * kh * ic * oc);
    for (int o = 0; o < oc; o++)
        for (int i = 0; i < ic; i++)
            for (int a = 0; a < kh; a++)
                for (int b = 0; b < kw; b++)
                    buf[(((size_t) o * ic + i) * kh + a) * kw + b] =
                        ggml_bf16_to_fp32(v[(((size_t) o * ic + i) * kh + a) * kw + b]);
    int64_t ne[4] = { kw, kh, ic, oc };
    struct ggml_tensor * t = ggml_new_tensor(w->ctx, GGML_TYPE_F32, 4, ne);
    ggml_set_name(t, name.c_str());
    w->pending.push_back({ t, buf.get(), (size_t) kw * kh * ic * oc * sizeof(float), 0 });
    w->staging.push_back(std::move(buf));
    return t;
}

// Conv1d torch [OC,IC,K] -> ggml [K,IC,OC] f32.
static struct ggml_tensor * htd_conv1d_w(WeightCtx * w, const GGUFModel & gf, const std::string & name) {
    struct ggml_tensor * src = ggml_get_tensor(gf.meta, name.c_str());
    int oc = (int) src->ne[2], ic = (int) src->ne[1], k = (int) src->ne[0];
    const ggml_bf16_t * v = (const ggml_bf16_t *) gf_get_data(gf, name.c_str());
    auto buf = std::make_unique<float[]>((size_t) k * ic * oc);
    for (size_t i = 0; i < (size_t) k * ic * oc; i++) buf[i] = ggml_bf16_to_fp32(v[i]);
    int64_t ne[3] = { k, ic, oc };
    struct ggml_tensor * t = ggml_new_tensor(w->ctx, GGML_TYPE_F32, 3, ne);
    ggml_set_name(t, name.c_str());
    w->pending.push_back({ t, buf.get(), (size_t) k * ic * oc * sizeof(float), 0 });
    w->staging.push_back(std::move(buf));
    return t;
}

// Spec conv as 1D-over-freq: torch [OC,IC,KH,1] -> ggml conv1d kernel [KH, IC, OC] f32.
// (KW=1 makes the spec Conv2d a pure freq conv; reuse the validated conv1d path.)
static struct ggml_tensor * htd_specconv1d_w(WeightCtx * w, const GGUFModel & gf, const std::string & name) {
    struct ggml_tensor * src = ggml_get_tensor(gf.meta, name.c_str());  // ggml ne=[1,KH,IC,OC]
    int kh = (int) src->ne[1], ic = (int) src->ne[2], oc = (int) src->ne[3];
    const ggml_bf16_t * v = (const ggml_bf16_t *) gf_get_data(gf, name.c_str());  // physical [OC][IC][KH][1]
    auto buf = std::make_unique<float[]>((size_t) kh * ic * oc);
    for (int o = 0; o < oc; o++)
        for (int i = 0; i < ic; i++)
            for (int a = 0; a < kh; a++)
                buf[((size_t) o * ic + i) * kh + a] = ggml_bf16_to_fp32(v[((size_t) o * ic + i) * kh + a]);
    int64_t ne[3] = { kh, ic, oc };
    struct ggml_tensor * t = ggml_new_tensor(w->ctx, GGML_TYPE_F32, 3, ne);
    ggml_set_name(t, name.c_str());
    w->pending.push_back({ t, buf.get(), (size_t) kh * ic * oc * sizeof(float), 0 });
    w->staging.push_back(std::move(buf));
    return t;
}

// ConvTranspose1d torch [IC,OC,K(,1)] -> col2im GEMM layout dst[IC, K*OC] col index oc*K+k.
// Works for time (3D [IC,OC,K]) and spec ([IC,OC,K,1], trailing 1 squeezed; freq-only transpose).
static struct ggml_tensor * htd_convt_gemm_w(WeightCtx * w, const GGUFModel & gf, const std::string & name) {
    struct ggml_tensor * src = ggml_get_tensor(gf.meta, name.c_str());
    int nd = ggml_n_dims(src);
    int ic, oc, k;
    if (nd >= 4) { ic = (int) src->ne[3]; oc = (int) src->ne[2]; k = (int) src->ne[1]; }  // gguf [1,K,OC,IC]
    else { ic = (int) src->ne[2]; oc = (int) src->ne[1]; k = (int) src->ne[0]; }          // gguf [K,OC,IC]
    const ggml_bf16_t * v = (const ggml_bf16_t *) gf_get_data(gf, name.c_str());  // physical [IC][OC][K]
    int KOC = k * oc;
    auto buf = std::make_unique<float[]>((size_t) ic * KOC);
    for (int i = 0; i < ic; i++)
        for (int o = 0; o < oc; o++)
            for (int kk = 0; kk < k; kk++)
                buf[(size_t) (o * k + kk) * ic + i] =
                    ggml_bf16_to_fp32(v[((size_t) i * oc + o) * k + kk]);
    int64_t ne[2] = { ic, KOC };
    struct ggml_tensor * t = ggml_new_tensor(w->ctx, GGML_TYPE_F32, 2, ne);
    ggml_set_name(t, name.c_str());
    w->pending.push_back({ t, buf.get(), (size_t) ic * KOC * sizeof(float), 0 });
    w->staging.push_back(std::move(buf));
    return t;
}

static struct ggml_tensor * htd_f32(WeightCtx * w, const GGUFModel & gf, const std::string & n) {
    return gf_load_tensor_f32(w, gf, n);
}
static struct ggml_tensor * htd_bf16(WeightCtx * w, const GGUFModel & gf, const std::string & n) {
    return gf_load_tensor(w, gf, n);
}

static void htd_load_dconv(HtdEncLayer * E, WeightCtx * w, const GGUFModel & gf, const std::string & prefix) {
    for (int s = 0; s < 2; s++) {
        HtdDConvLayer & D = E->dconv[s];
        std::string p = prefix + ".dconv.layers." + std::to_string(s);
        D.c1w  = htd_conv1d_w(w, gf, p + ".0.weight");
        D.c1b  = htd_f32(w, gf, p + ".0.bias");
        D.gn1w = htd_f32(w, gf, p + ".1.weight");
        D.gn1b = htd_f32(w, gf, p + ".1.bias");
        D.c3w  = htd_conv1d_w(w, gf, p + ".3.weight");
        D.c3b  = htd_f32(w, gf, p + ".3.bias");
        D.gn4w = htd_f32(w, gf, p + ".4.weight");
        D.gn4b = htd_f32(w, gf, p + ".4.bias");
        D.scale = htd_f32(w, gf, p + ".6.scale");
        D.dilation = 1 << s;  // 1, 2
    }
}

static void htd_load_tlayer(HtdTLayer * L, WeightCtx * w, const GGUFModel & gf, const std::string & p, bool is_cross) {
    L->is_cross = is_cross;
    std::string an = is_cross ? ".cross_attn" : ".self_attn";
    L->in_proj_w  = htd_bf16(w, gf, p + an + ".in_proj_weight");
    L->in_proj_b  = htd_f32(w, gf, p + an + ".in_proj_bias");
    L->out_proj_w = htd_bf16(w, gf, p + an + ".out_proj.weight");
    L->out_proj_b = htd_f32(w, gf, p + an + ".out_proj.bias");
    L->l1w = htd_bf16(w, gf, p + ".linear1.weight");
    L->l1b = htd_f32(w, gf, p + ".linear1.bias");
    L->l2w = htd_bf16(w, gf, p + ".linear2.weight");
    L->l2b = htd_f32(w, gf, p + ".linear2.bias");
    L->n1w = htd_f32(w, gf, p + ".norm1.weight");
    L->n1b = htd_f32(w, gf, p + ".norm1.bias");
    L->n2w = htd_f32(w, gf, p + ".norm2.weight");
    L->n2b = htd_f32(w, gf, p + ".norm2.bias");
    if (is_cross) {
        L->n3w = htd_f32(w, gf, p + ".norm3.weight");
        L->n3b = htd_f32(w, gf, p + ".norm3.bias");
    } else {
        L->n3w = L->n3b = nullptr;
    }
    L->nout_w = htd_f32(w, gf, p + ".norm_out.weight");
    L->nout_b = htd_f32(w, gf, p + ".norm_out.bias");
    L->g1 = htd_f32(w, gf, p + ".gamma_1.scale");
    L->g2 = htd_f32(w, gf, p + ".gamma_2.scale");
}

static bool htd_load(SonggenHTDemucs * m, const char * gguf_path) {
    *m = {};
    BackendPair bp = backend_init("HTDEMUCS");
    m->backend     = bp.backend;
    m->cpu_backend = bp.cpu_backend;
    m->sched       = backend_sched_new(bp, 200000);

    GGUFModel gf;
    if (!gf_load(&gf, gguf_path)) return false;

    const char * A   = "songgen-htdemucs";
    auto u32 = [&](const char * k) { return (int) gf_get_u32(gf, (std::string(A) + "." + k).c_str()); };
    auto f32 = [&](const char * k) { return gf_get_f32(gf, (std::string(A) + "." + k).c_str()); };
    m->depth          = u32("depth");
    m->t_layers       = u32("t_layers");
    m->t_dim          = u32("t_dim");
    m->t_heads        = u32("t_heads");
    m->t_ffn          = u32("t_ffn");
    m->transformer_ch = u32("transformer_ch");
    m->nfft           = u32("nfft");
    m->hop            = u32("hop");
    m->sample_rate    = u32("sample_rate");
    m->segment_samples= u32("segment_samples");
    m->freq_emb_scale = f32("freq_emb_scale");
    m->max_period     = f32("max_period");
    fprintf(stderr, "[HTDEMUCS] depth=%d t_layers=%d dim=%d heads=%d nfft=%d hop=%d seg=%d\n", m->depth, m->t_layers,
            m->t_dim, m->t_heads, m->nfft, m->hop, m->segment_samples);

    wctx_init(&m->wctx, 700);

    int ch[5] = { 48, 96, 192, 384, 768 };
    // spec encoders
    for (int i = 0; i < 4; i++) {
        HtdEncLayer & E = m->enc[i];
        std::string p = "encoder." + std::to_string(i);
        E.freq = true;
        E.chin = (i == 0) ? 4 : ch[i - 1];
        E.chout = ch[i];
        E.conv_w = htd_specconv1d_w(&m->wctx, gf, p + ".conv.weight");    // [8,IC,OC]
        E.conv_b = htd_f32(&m->wctx, gf, p + ".conv.bias");
        E.rw_w = htd_specconv1d_w(&m->wctx, gf, p + ".rewrite.weight");   // [1,OC,2OC]
        E.rw_b = htd_f32(&m->wctx, gf, p + ".rewrite.bias");
        htd_load_dconv(&E, &m->wctx, gf, p);
    }
    // time encoders
    for (int i = 0; i < 4; i++) {
        HtdEncLayer & E = m->tenc[i];
        std::string p = "tencoder." + std::to_string(i);
        E.freq = false;
        E.chin = (i == 0) ? 2 : ch[i - 1];
        E.chout = ch[i];
        E.conv_w = htd_conv1d_w(&m->wctx, gf, p + ".conv.weight");
        E.conv_b = htd_f32(&m->wctx, gf, p + ".conv.bias");
        E.rw_w = htd_conv1d_w(&m->wctx, gf, p + ".rewrite.weight");
        E.rw_b = htd_f32(&m->wctx, gf, p + ".rewrite.bias");
        htd_load_dconv(&E, &m->wctx, gf, p);
    }
    m->freq_emb = htd_f32(&m->wctx, gf, "freq_emb.embedding.weight");  // [48,512]

    m->up_w   = htd_bf16(&m->wctx, gf, "channel_upsampler.weight");
    m->up_b   = htd_f32(&m->wctx, gf, "channel_upsampler.bias");
    m->down_w = htd_bf16(&m->wctx, gf, "channel_downsampler.weight");
    m->down_b = htd_f32(&m->wctx, gf, "channel_downsampler.bias");
    m->up_t_w   = htd_bf16(&m->wctx, gf, "channel_upsampler_t.weight");
    m->up_t_b   = htd_f32(&m->wctx, gf, "channel_upsampler_t.bias");
    m->down_t_w = htd_bf16(&m->wctx, gf, "channel_downsampler_t.weight");
    m->down_t_b = htd_f32(&m->wctx, gf, "channel_downsampler_t.bias");

    m->normin_w   = htd_f32(&m->wctx, gf, "crosstransformer.norm_in.weight");
    m->normin_b   = htd_f32(&m->wctx, gf, "crosstransformer.norm_in.bias");
    m->normin_t_w = htd_f32(&m->wctx, gf, "crosstransformer.norm_in_t.weight");
    m->normin_t_b = htd_f32(&m->wctx, gf, "crosstransformer.norm_in_t.bias");

    for (int i = 0; i < 5; i++) {
        bool is_cross = (i % 2 == 1);
        htd_load_tlayer(&m->tx[i], &m->wctx, gf, "crosstransformer.layers." + std::to_string(i), is_cross);
        htd_load_tlayer(&m->tx_t[i], &m->wctx, gf, "crosstransformer.layers_t." + std::to_string(i), is_cross);
    }

    // decoders (spec)
    for (int i = 0; i < 4; i++) {
        HtdDecLayer & D = m->dec[i];
        std::string p = "decoder." + std::to_string(i);
        D.freq = true;
        D.last = (i == 3);
        struct ggml_tensor * cm = ggml_get_tensor(gf.meta, (p + ".conv_tr.weight").c_str());  // gguf [1,K,OC,IC]
        D.ktr = (int) cm->ne[1]; D.chout = (int) cm->ne[2]; D.chin = (int) cm->ne[3];
        D.ctr_w = htd_convt_gemm_w(&m->wctx, gf, p + ".conv_tr.weight");
        D.ctr_b = htd_f32(&m->wctx, gf, p + ".conv_tr.bias");
        D.rw_w = htd_conv2d_w(&m->wctx, gf, p + ".rewrite.weight");
        D.rw_b = htd_f32(&m->wctx, gf, p + ".rewrite.bias");
    }
    // tdecoders (time)
    for (int i = 0; i < 4; i++) {
        HtdDecLayer & D = m->tdec[i];
        std::string p = "tdecoder." + std::to_string(i);
        D.freq = false;
        D.last = (i == 3);
        struct ggml_tensor * cm = ggml_get_tensor(gf.meta, (p + ".conv_tr.weight").c_str());  // gguf [K,OC,IC]
        D.ktr = (int) cm->ne[0]; D.chout = (int) cm->ne[1]; D.chin = (int) cm->ne[2];
        D.ctr_w = htd_convt_gemm_w(&m->wctx, gf, p + ".conv_tr.weight");
        D.ctr_b = htd_f32(&m->wctx, gf, p + ".conv_tr.bias");
        D.rw_w = htd_conv1d_w(&m->wctx, gf, p + ".rewrite.weight");
        D.rw_b = htd_f32(&m->wctx, gf, p + ".rewrite.bias");
    }

    if (!wctx_alloc(&m->wctx, m->backend)) { gf_close(&gf); return false; }
    gf_close(&gf);
    return true;
}

static void htd_free(SonggenHTDemucs * m) {
    if (m->sched) ggml_backend_sched_free(m->sched);
    backend_release(m->backend, m->cpu_backend);
    wctx_free(&m->wctx);
    *m = {};
}

// ================= host STFT / CAC / iSTFT =================
// Hann window 0.5 - 0.5*cos(2*pi*n/N).
static void htd_hann(int N, std::vector<float> & w) {
    w.resize(N);
    for (int n = 0; n < N; n++) w[n] = 0.5f - 0.5f * (float) std::cos(2.0 * M_PI * n / N);
}

// Iterative radix-2 FFT in place (n power of 2). re/im length n. sign=-1 forward, +1 inverse.
static void htd_fft(double * re, double * im, int n, int sign) {
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = sign * 2.0 * M_PI / len;
        double wr = std::cos(ang), wi = std::sin(ang);
        for (int i = 0; i < n; i += len) {
            double cr = 1.0, ci = 0.0;
            for (int k = 0; k < len / 2; k++) {
                double ur = re[i + k], ui = im[i + k];
                double vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
                double vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
                re[i + k] = ur + vr; im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
                double ncr = cr * wr - ci * wi, nci = cr * wi + ci * wr;
                cr = ncr; ci = nci;
            }
        }
    }
}

// reflect pad (no boundary repeat): torch reflect.
static void htd_reflect_pad(const float * x, int N, int lp, int rp, std::vector<float> & out) {
    out.resize((size_t) N + lp + rp);
    for (int i = 0; i < N; i++) out[lp + i] = x[i];
    for (int i = 0; i < lp; i++) out[lp - 1 - i] = x[i + 1];
    for (int i = 0; i < rp; i++) out[lp + N + i] = x[N - 2 - i];
}

// Demucs _spec: reflect-pad (hl/2*3, ...), STFT(nfft,hop,center,reflect,normalized), drop last freq, z[...,2:2+le].
// Produces CAC mag [4, 2048, 336] channel-major (ch,freq,time) per the layout [re0,im0,re1,im1].
// Returns mag flat [4*2048*336] in order ch-major-then-freq-then-time (matches torch [1,4,2048,336] C order).
static void htd_spec_cac(const SonggenHTDemucs * m, const float * mix, int N, int AC, std::vector<float> & mag_out,
                         std::vector<float> & z_re, std::vector<float> & z_im, int * Fr_out, int * T_out) {
    const int nfft = m->nfft, hl = m->hop;
    const int full_freq = nfft / 2 + 1;   // 2049
    const int Fr = nfft / 2;              // 2048 (drop last)
    int le = (int) std::ceil((double) N / hl);
    int pad = hl / 2 * 3;
    int T = le;                           // 336
    *Fr_out = Fr; *T_out = T;

    std::vector<float> win; htd_hann(nfft, win);
    int cpad = nfft / 2;

    // per audio channel
    z_re.assign((size_t) AC * Fr * T, 0.f);
    z_im.assign((size_t) AC * Fr * T, 0.f);
    mag_out.assign((size_t) 2 * AC * Fr * T, 0.f);

    const float inv = 1.0f / std::sqrt((float) nfft);
    (void) full_freq;

    for (int c = 0; c < AC; c++) {
        // demucs re-pad then torch.stft does center reflect pad again
        std::vector<float> xp;
        htd_reflect_pad(mix + (size_t) c * N, N, pad, pad + le * hl - N, xp);
        int Lp = (int) xp.size();
        std::vector<float> xx;
        htd_reflect_pad(xp.data(), Lp, cpad, cpad, xx);
        std::vector<double> re(nfft), im(nfft);
        for (int t = 0; t < T; t++) {
            int base = (t + 2) * hl;  // z[...,2:2+le]
            for (int n = 0; n < nfft; n++) { re[n] = (double) xx[base + n] * win[n]; im[n] = 0.0; }
            htd_fft(re.data(), im.data(), nfft, -1);
            for (int k = 0; k < Fr; k++) {  // drop last freq -> k<2048
                float rf = (float) re[k] * inv, imf = (float) im[k] * inv;
                z_re[((size_t) c * Fr + k) * T + t] = rf;
                z_im[((size_t) c * Fr + k) * T + t] = imf;
                mag_out[((size_t) (2 * c + 0) * Fr + k) * T + t] = rf;
                mag_out[((size_t) (2 * c + 1) * Fr + k) * T + t] = imf;
            }
        }
    }
}

// iSTFT for one channel. z_re/z_im: [Fr=2048, T=336] for THIS channel. _ispec pads freq to 2049,
// pads time by 2 each side, istft normalized, trims center pad, returns [length].
static void htd_ispec_chan(const SonggenHTDemucs * m, const float * zr, const float * zi, int Fr, int T, int length,
                           std::vector<float> & out) {
    const int nfft = m->nfft, hl = m->hop;
    int freqs = Fr + 1;                  // 2049
    int Tp = T + 4;                      // pad (2,2) on time
    // build padded complex [freqs, Tp]; freq pad adds a zero row at top (F.pad(z,(0,0,0,1)))
    std::vector<double> cre((size_t) freqs * Tp, 0.0), cim((size_t) freqs * Tp, 0.0);
    for (int k = 0; k < Fr; k++)
        for (int t = 0; t < T; t++) {
            cre[(size_t) k * Tp + (t + 2)] = zr[(size_t) k * T + t];
            cim[(size_t) k * Tp + (t + 2)] = zi[(size_t) k * T + t];
        }
    // last freq row (k=Fr) stays 0.
    std::vector<float> win; htd_hann(nfft, win);
    int cpad = nfft / 2;
    int le = hl * (int) std::ceil((double) length / hl) + 2 * (hl / 2 * 3);
    // istft total length from Tp frames (center): n_out = (Tp-1)*hl + nfft, then trim center pad and to le.
    int n_out = (Tp - 1) * hl + nfft;
    std::vector<double> y((size_t) n_out, 0.0), wsum((size_t) n_out, 0.0);
    const float scale = std::sqrt((float) nfft);  // undo normalized=True
    std::vector<double> fr_re(nfft), fr_im(nfft);
    for (int t = 0; t < Tp; t++) {
        // build full hermitian spectrum then inverse FFT: x[n] = (1/N) sum_k Xk e^{+i2pi kn/N}
        for (int k = 0; k < freqs; k++) { fr_re[k] = cre[(size_t) k * Tp + t]; fr_im[k] = cim[(size_t) k * Tp + t]; }
        for (int k = freqs; k < nfft; k++) { fr_re[k] = fr_re[nfft - k]; fr_im[k] = -fr_im[nfft - k]; }
        htd_fft(fr_re.data(), fr_im.data(), nfft, +1);
        for (int n = 0; n < nfft; n++) {
            double s = fr_re[n] / nfft;
            y[(size_t) t * hl + n] += s * win[n] * scale;
            wsum[(size_t) t * hl + n] += (double) win[n] * win[n];
        }
    }
    for (size_t i = 0; i < y.size(); i++) if (wsum[i] > 1e-8) y[i] /= wsum[i];
    // trim: torch istft with center removes nfft/2 each side, giving le samples; then _ispec trims pad: x[pad:pad+length]
    int pad = hl / 2 * 3;
    out.resize(length);
    for (int i = 0; i < length; i++) {
        int idx = cpad + pad + i;
        out[i] = (idx >= 0 && idx < n_out) ? (float) y[idx] : 0.f;
    }
    (void) le;
}

// ================= ggml graph helpers =================
// conv1d with F32 im2col (ggml_conv_1d hardcodes F16 dst, needing F16 kernels). a:[K,IC,OC] f32,
// b:[L,IC,N] -> [OL, OC, N].
static struct ggml_tensor * htd_conv1d(struct ggml_context * c, struct ggml_tensor * a, struct ggml_tensor * b,
                                       int s0, int p0, int d0) {
    struct ggml_tensor * im = ggml_im2col(c, a, b, s0, 0, p0, 0, d0, 0, false, GGML_TYPE_F32);  // [IC*K, OL, N]
    struct ggml_tensor * r = ggml_mul_mat(c, ggml_reshape_2d(c, a, a->ne[0] * a->ne[1], a->ne[2]),
                                          ggml_reshape_2d(c, im, im->ne[0], im->ne[1] * im->ne[2]));
    r = ggml_reshape_3d(c, r, a->ne[2], im->ne[1], im->ne[2]);  // [OC, OL, N]
    r = ggml_cont(c, ggml_permute(c, r, 1, 0, 2, 3));           // [OL, OC, N]
    return r;
}

// LayerNorm over ne0.
static struct ggml_tensor * htd_ln(struct ggml_context * c, struct ggml_tensor * x, struct ggml_tensor * w,
                                   struct ggml_tensor * b, float eps) {
    x = ggml_norm(c, x, eps);
    x = ggml_mul(c, x, w);
    x = ggml_add(c, x, b);
    return x;
}

// GroupNorm(1, C) for conv features laid out [L, C, B] (ne0=L spatial, ne1=C, ne2=B batch).
// Normalize over (L,C) jointly per B, then affine per-channel w/b [C].
static struct ggml_tensor * htd_gn1(struct ggml_context * c, struct ggml_tensor * x, struct ggml_tensor * w,
                                    struct ggml_tensor * b, float eps) {
    int64_t L = x->ne[0], C = x->ne[1], B = x->ne[2];
    struct ggml_tensor * f = ggml_reshape_3d(c, ggml_cont(c, x), L * C, 1, B);  // [L*C,1,B]
    f = ggml_norm(c, f, eps);
    f = ggml_reshape_3d(c, f, L, C, B);
    struct ggml_tensor * ww = ggml_reshape_3d(c, w, 1, C, 1);
    struct ggml_tensor * bb = ggml_reshape_3d(c, b, 1, C, 1);
    f = ggml_add(c, ggml_mul(c, f, ww), bb);
    return f;
}

// DConv branch on data laid out [L=time, C, B=freq(or1)]. residual add inside.
static struct ggml_tensor * htd_dconv(struct ggml_context * c, HtdEncLayer * E, struct ggml_tensor * y) {
    // y: [T, C, B]
    for (int s = 0; s < 2; s++) {
        HtdDConvLayer & D = E->dconv[s];
        int dil = D.dilation, pad = dil * 1;  // kernel 3 -> pad = dil*(3//2)=dil
        struct ggml_tensor * h = htd_conv1d(c, D.c1w, y, 1, pad, dil);  // [T, hidden, B]
        h = ggml_add(c, h, ggml_reshape_3d(c, D.c1b, 1, D.c1b->ne[0], 1));
        h = htd_gn1(c, h, D.gn1w, D.gn1b, 1e-5f);
        h = ggml_gelu(c, h);
        struct ggml_tensor * z = htd_conv1d(c, D.c3w, h, 1, 0, 1);  // [T, 2C, B]
        z = ggml_add(c, z, ggml_reshape_3d(c, D.c3b, 1, D.c3b->ne[0], 1));
        z = htd_gn1(c, z, D.gn4w, D.gn4b, 1e-5f);
        // GLU dim=channels (ne1): split 2C -> a*sigmoid(b)
        int64_t T = z->ne[0], C2 = z->ne[1], B = z->ne[2];
        int64_t C = C2 / 2;
        struct ggml_tensor * a = ggml_cont(c, ggml_view_3d(c, z, T, C, B, z->nb[1], z->nb[2], 0));
        struct ggml_tensor * g = ggml_cont(c, ggml_view_3d(c, z, T, C, B, z->nb[1], z->nb[2], C * z->nb[1]));
        struct ggml_tensor * o = ggml_mul(c, a, ggml_sigmoid(c, g));  // [T, C, B]
        // layerscale per channel
        o = ggml_mul(c, o, ggml_reshape_3d(c, D.scale, 1, C, 1));
        y = ggml_add(c, y, o);
    }
    return y;
}

// ---- spec encoder layer. x: [T=time, F=freq, C]. returns [T, F/4, OC]. ----
// Spec conv/rewrite are freq-1D convs (kw=1). Run as conv1d over freq with batch=time:
// permute [T,F,C] -> [F,C,T], conv1d, back.
static struct ggml_tensor * htd_enc_spec(struct ggml_context * c, HtdEncLayer * E, struct ggml_tensor * x) {
    int64_t T = x->ne[0];
    struct ggml_tensor * xf = ggml_cont(c, ggml_permute(c, x, 2, 0, 1, 3));  // [F, C, T]
    struct ggml_tensor * y = htd_conv1d(c, E->conv_w, xf, 4, 2, 1);          // [F/4, OC, T]
    y = ggml_add(c, y, ggml_reshape_3d(c, E->conv_b, 1, E->chout, 1));
    y = ggml_gelu(c, y);
    int64_t Fp = y->ne[0], OC = y->ne[1];
    // dconv operates per-freq along time -> need [T, OC, F']. y is [F', OC, T] -> permute.
    struct ggml_tensor * yd = ggml_cont(c, ggml_permute(c, y, 2, 1, 0, 3));  // [T, OC, F']
    yd = htd_dconv(c, E, yd);
    struct ggml_tensor * yb = ggml_cont(c, ggml_permute(c, yd, 2, 1, 0, 3)); // [F', OC, T]
    // rewrite 1x1 over channels -> 2OC, GLU over channels
    struct ggml_tensor * z = htd_conv1d(c, E->rw_w, yb, 1, 0, 1);            // [F', 2OC, T]
    z = ggml_add(c, z, ggml_reshape_3d(c, E->rw_b, 1, 2 * OC, 1));
    struct ggml_tensor * a = ggml_cont(c, ggml_view_3d(c, z, Fp, OC, T, z->nb[1], z->nb[2], 0));
    struct ggml_tensor * g = ggml_cont(c, ggml_view_3d(c, z, Fp, OC, T, z->nb[1], z->nb[2], OC * z->nb[1]));
    struct ggml_tensor * o = ggml_mul(c, a, ggml_sigmoid(c, g));             // [F', OC, T]
    return ggml_cont(c, ggml_permute(c, o, 1, 2, 0, 3));                     // src F'->1,OC->2,T->0 => [T,F',OC]
}

// ---- time encoder layer. xt: [Tt, C]. returns [Tt/4, OC]. ----
static struct ggml_tensor * htd_enc_time(struct ggml_context * c, HtdEncLayer * E, struct ggml_tensor * xt) {
    int Tt = (int) xt->ne[0];
    int stride = 4;
    // torch pads time if not divisible by stride (HEncLayer time path)
    int rem = Tt % stride;
    if (rem != 0) xt = ggml_pad(c, xt, stride - rem, 0, 0, 0);
    // conv1d k8 s4 p2: data [L, IC], kernel [8,IC,OC]
    struct ggml_tensor * y = htd_conv1d(c, E->conv_w, xt, 4, 2, 1);  // [L', OC]
    y = ggml_add(c, y, ggml_reshape_2d(c, E->conv_b, 1, E->chout));
    y = ggml_gelu(c, y);
    int64_t Lp = y->ne[0], OC = y->ne[1];
    // dconv on [L', OC, 1]
    struct ggml_tensor * yd = ggml_reshape_3d(c, y, Lp, OC, 1);
    yd = htd_dconv(c, E, yd);
    y = ggml_reshape_2d(c, yd, Lp, OC);
    // rewrite 1x1 -> 2OC: conv1d k1
    struct ggml_tensor * z = htd_conv1d(c, E->rw_w, y, 1, 0, 1);  // [L', 2OC]
    z = ggml_add(c, z, ggml_reshape_2d(c, E->rw_b, 1, 2 * OC));
    struct ggml_tensor * a = ggml_cont(c, ggml_view_2d(c, z, Lp, OC, z->nb[1], 0));
    struct ggml_tensor * g = ggml_cont(c, ggml_view_2d(c, z, Lp, OC, z->nb[1], OC * z->nb[1]));
    return ggml_mul(c, a, ggml_sigmoid(c, g));  // [L', OC]
}

// ================= encoder-only forward (validation stage 1) =================
// Runs STFT/CAC -> normalize -> spec & time encoders. Outputs (channel-major torch layout):
//   spec enc i: [chout_i, F_i, T]   (flat ch-major-freq-time)
//   time enc i: [chout_i, Lt_i]
// plus freq-embedded enc0 used downstream (returned separately as enc0_fe).
struct HtdEncOut {
    std::vector<float> spec[4];   // enc0..3 (enc0 is PRE freq_emb, matching shipped golden)
    std::vector<float> spec_fe0;  // enc0 AFTER freq_emb
    std::vector<float> time[4];   // tenc0..3
    int spec_ch[4], spec_F[4], spec_T;
    int time_ch[4], time_L[4];
};

static void htd_encode_encoders(SonggenHTDemucs * m, const float * mix, int N, int AC, HtdEncOut * out) {
    std::vector<float> mag, z_re, z_im;
    int Fr = 0, T = 0;
    htd_spec_cac(m, mix, N, AC, mag, z_re, z_im, &Fr, &T);  // mag [2*AC, Fr, T] = [4,2048,336]
    int Cspec = 2 * AC;

    // normalize spec (per-tensor mean/std over all)
    {
        double s = 0, s2 = 0; size_t n = mag.size();
        for (float v : mag) { s += v; s2 += (double) v * v; }
        double mean = s / n, var = s2 / n - mean * mean;
        // torch std is unbiased (N-1)
        var = (s2 - (double) n * mean * mean) / (double) (n - 1);
        double sd = std::sqrt(var);
        float inv = 1.0f / (float) (1e-5 + sd);
        for (float & v : mag) v = (float) ((v - mean) * inv);
    }
    // normalize time (per-tensor over all AC*N)
    std::vector<float> xt((size_t) AC * N);
    {
        double s = 0, s2 = 0; size_t n = (size_t) AC * N;
        for (int c = 0; c < AC; c++) for (int i = 0; i < N; i++) { float v = mix[(size_t) c * N + i]; s += v; s2 += (double) v * v; }
        double mean = s / n;
        double var = (s2 - (double) n * mean * mean) / (double) (n - 1);
        double sd = std::sqrt(var);
        float inv = 1.0f / (float) (1e-5 + sd);
        for (int c = 0; c < AC; c++) for (int i = 0; i < N; i++) xt[(size_t) c * N + i] = (float) ((mix[(size_t) c * N + i] - mean) * inv);
    }

    size_t ctx_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false);
    uint8_t * gbuf = (uint8_t *) malloc(ctx_size);
    struct ggml_init_params gp = { ctx_size, gbuf, true };
    struct ggml_context * c = ggml_init(gp);

    // spec input: ggml [T, Fr, Cspec] (ne0=time). mag is [Cspec, Fr, T] (time innermost) -> reorder.
    struct ggml_tensor * x = ggml_new_tensor_3d(c, GGML_TYPE_F32, T, Fr, Cspec);
    ggml_set_input(x); ggml_set_name(x, "spec_in");
    struct ggml_tensor * xtin = ggml_new_tensor_2d(c, GGML_TYPE_F32, N, AC);
    ggml_set_input(xtin); ggml_set_name(xtin, "time_in");

    std::vector<struct ggml_tensor *> outs;
    // spec encoders
    struct ggml_tensor * sx = x;
    struct ggml_tensor * enc_pre[4]; struct ggml_tensor * enc_post[4];
    for (int i = 0; i < 4; i++) {
        sx = htd_enc_spec(c, &m->enc[i], sx);  // [T, F', OC]
        enc_pre[i] = sx;
        if (i == 0) {
            // freq emb: emb[F',OC] added scaled. freq_emb weight [emb_dim=48? , 512]; here OC=48, F'=512.
            // m->freq_emb is f32 ggml [48, 512] = [emb_dim, num]; need [num=512(=F'), emb_dim=48(=OC)] per freq.
            // torch: emb = freq_emb(arange(F')).t() -> [OC=48, F'=512]; added as [1,OC,F',1] broadcast over T.
            // our sx is [T, F', OC]. emb per (F',OC). freq_emb.weight is Embedding [num=512, dim=48] stored ggml [48,512].
            struct ggml_tensor * fe = m->freq_emb;  // ne0=48(dim=OC), ne1=512(num=F')
            // want add tensor [T, F', OC] broadcasting over T: shape [1, F', OC]
            struct ggml_tensor * fet = ggml_cont(c, ggml_transpose(c, fe));  // [512(F'), 48(OC)]
            fet = ggml_reshape_3d(c, fet, 1, fet->ne[0], fet->ne[1]);        // [1, F', OC]
            // ScaledEmbedding forward multiplies stored weight by emb_scale=10; total = 0.2 * 10.
            struct ggml_tensor * fe_sc = ggml_scale(c, fet, m->freq_emb_scale * 10.0f);
            sx = ggml_add(c, sx, fe_sc);
        }
        enc_post[i] = sx;
    }
    // time encoders
    struct ggml_tensor * tx = xtin;
    struct ggml_tensor * tenc[4];
    for (int i = 0; i < 4; i++) { tx = htd_enc_time(c, &m->tenc[i], tx); tenc[i] = tx; }

    for (int i = 0; i < 4; i++) { ggml_set_output(enc_pre[i]); outs.push_back(enc_pre[i]); }
    ggml_set_output(enc_post[0]); outs.push_back(enc_post[0]);
    for (int i = 0; i < 4; i++) { ggml_set_output(tenc[i]); outs.push_back(tenc[i]); }

    struct ggml_cgraph * g = ggml_new_graph_custom(c, 16384, false);
    for (auto * o : outs) ggml_build_forward_expand(g, o);
    if (!ggml_backend_sched_alloc_graph(m->sched, g)) { fprintf(stderr, "[HTDEMUCS] enc alloc failed\n"); exit(1); }

    // fill spec_in: ggml layout ne0=T,ne1=Fr,ne2=Cspec contiguous => index (cc*Fr+ff)*T+tt.
    // mag layout is [Cspec][Fr][T] = (cc*Fr+ff)*T+tt -> SAME. direct copy.
    ggml_backend_tensor_set(x, mag.data(), 0, mag.size() * sizeof(float));
    // time_in ggml ne0=N,ne1=AC contiguous => (c*N+i). xt same. direct.
    ggml_backend_tensor_set(xtin, xt.data(), 0, xt.size() * sizeof(float));

    ggml_backend_sched_graph_compute(m->sched, g);

    auto fetch = [&](struct ggml_tensor * t) {
        std::vector<float> v((size_t) ggml_nelements(t));
        ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
        return v;
    };
    // spec enc out ggml [T, F', OC] -> torch [OC, F', T] channel-major. reorder.
    auto to_cm_spec = [&](struct ggml_tensor * t) {
        int Tt = (int) t->ne[0], Ff = (int) t->ne[1], OC = (int) t->ne[2];
        std::vector<float> raw = fetch(t);
        std::vector<float> r((size_t) OC * Ff * Tt);
        for (int o = 0; o < OC; o++) for (int f = 0; f < Ff; f++) for (int tt = 0; tt < Tt; tt++)
            r[((size_t) o * Ff + f) * Tt + tt] = raw[((size_t) o * Ff + f) * Tt + tt];
        return r;  // ggml contiguous [T,F',OC] index (o*Ff+f)*Tt+tt == torch [OC,F',T] -> identical
    };
    auto to_cm_time = [&](struct ggml_tensor * t) {
        int Lt = (int) t->ne[0], OC = (int) t->ne[1];
        std::vector<float> raw = fetch(t);
        std::vector<float> r((size_t) OC * Lt);
        for (int o = 0; o < OC; o++) for (int l = 0; l < Lt; l++) r[(size_t) o * Lt + l] = raw[(size_t) o * Lt + l];
        return r;  // [Lt,OC] ggml == torch [OC,Lt]
    };
    for (int i = 0; i < 4; i++) {
        out->spec[i] = to_cm_spec(enc_pre[i]);
        out->spec_ch[i] = (int) enc_pre[i]->ne[2];
        out->spec_F[i] = (int) enc_pre[i]->ne[1];
        out->time[i] = to_cm_time(tenc[i]);
        out->time_ch[i] = (int) tenc[i]->ne[1];
        out->time_L[i] = (int) tenc[i]->ne[0];
    }
    out->spec_fe0 = to_cm_spec(enc_post[0]);
    out->spec_T = T;

    fprintf(stderr, "[HTDEMUCS] encoders graph %d nodes\n", ggml_graph_n_nodes(g));
    ggml_backend_sched_reset(m->sched);
    ggml_free(c);
    free(gbuf);
}

// ================= cross-transformer =================
// 1x1 conv (channel projection) on data [S, Cin, ...] -> [S, Cout, ...]: per-position matmul.
// w is bf16 ggml [Cin, Cout] (torch [Cout,Cin,1] -> [Cout,Cin] -> gguf ggml [Cin,Cout]).
// x: [feat=Cin, S] (ne0=Cin). returns [Cout, S].
static struct ggml_tensor * htd_proj(struct ggml_context * c, struct ggml_tensor * w, struct ggml_tensor * b,
                                     struct ggml_tensor * x) {
    struct ggml_tensor * y = ggml_mul_mat(c, w, x);  // [Cout, S]
    y = ggml_add(c, y, b);
    return y;
}

// Multi-head attention (batch_first, single batch). q:[C, Tq], kv:[C, Tk]. in_proj fused [3C,C].
// Returns [C, Tq].
static struct ggml_tensor * htd_mha(struct ggml_context * c, HtdTLayer * L, struct ggml_tensor * q_in,
                                    struct ggml_tensor * kv_in, int nh) {
    int C = (int) q_in->ne[0];
    int Tq = (int) q_in->ne[1];
    int Tk = (int) kv_in->ne[1];
    int dh = C / nh;
    float scale = 1.0f / std::sqrt((float) dh);

    // in_proj_w [3C, C]; split rows: q rows [0,C), k [C,2C), v [2C,3C).
    struct ggml_tensor * wq = ggml_view_2d(c, L->in_proj_w, C, C, L->in_proj_w->nb[1], 0);
    struct ggml_tensor * wk = ggml_view_2d(c, L->in_proj_w, C, C, L->in_proj_w->nb[1], (size_t) C * L->in_proj_w->nb[1]);
    struct ggml_tensor * wv = ggml_view_2d(c, L->in_proj_w, C, C, L->in_proj_w->nb[1], (size_t) 2 * C * L->in_proj_w->nb[1]);
    struct ggml_tensor * bq = ggml_view_1d(c, L->in_proj_b, C, 0);
    struct ggml_tensor * bk = ggml_view_1d(c, L->in_proj_b, C, (size_t) C * L->in_proj_b->nb[0]);
    struct ggml_tensor * bv = ggml_view_1d(c, L->in_proj_b, C, (size_t) 2 * C * L->in_proj_b->nb[0]);

    struct ggml_tensor * q = ggml_add(c, ggml_mul_mat(c, ggml_cont(c, wq), q_in), bq);   // [C, Tq]
    struct ggml_tensor * k = ggml_add(c, ggml_mul_mat(c, ggml_cont(c, wk), kv_in), bk);  // [C, Tk]
    struct ggml_tensor * v = ggml_add(c, ggml_mul_mat(c, ggml_cont(c, wv), kv_in), bv);  // [C, Tk]

    q = ggml_reshape_3d(c, q, dh, nh, Tq);
    k = ggml_reshape_3d(c, k, dh, nh, Tk);
    v = ggml_reshape_3d(c, v, dh, nh, Tk);
    q = ggml_permute(c, q, 0, 2, 1, 3);  // [dh, Tq, nh]
    k = ggml_permute(c, k, 0, 2, 1, 3);  // [dh, Tk, nh]
    v = ggml_permute(c, v, 0, 2, 1, 3);  // [dh, Tk, nh]

    struct ggml_tensor * scores = ggml_mul_mat(c, ggml_cont(c, k), ggml_cont(c, q));  // [Tk, Tq, nh]
    scores = ggml_soft_max_ext(c, scores, NULL, scale, 0.0f);
    struct ggml_tensor * vt = ggml_cont(c, ggml_transpose(c, v));   // [Tk, dh, nh]
    struct ggml_tensor * o = ggml_mul_mat(c, vt, scores);          // [dh, Tq, nh]
    o = ggml_cont(c, ggml_permute(c, o, 0, 2, 1, 3));              // [dh, nh, Tq]
    o = ggml_reshape_2d(c, o, C, Tq);
    o = ggml_add(c, ggml_mul_mat(c, L->out_proj_w, o), L->out_proj_b);  // [C, Tq]
    return o;
}

// GroupNorm(1,C) over feature dim for transformer tokens x:[C, T] (MyGroupNorm: norm over all C,T per batch).
static struct ggml_tensor * htd_gn_all(struct ggml_context * c, struct ggml_tensor * x, struct ggml_tensor * w,
                                       struct ggml_tensor * b, float eps) {
    int64_t C = x->ne[0], T = x->ne[1];
    struct ggml_tensor * f = ggml_reshape_2d(c, ggml_cont(c, x), C * T, 1);
    f = ggml_norm(c, f, eps);
    f = ggml_reshape_2d(c, f, C, T);
    f = ggml_add(c, ggml_mul(c, f, w), b);  // affine per-channel
    return f;
}

// FFN: linear1 -> gelu -> linear2. x:[C,T].
static struct ggml_tensor * htd_ffn(struct ggml_context * c, HtdTLayer * L, struct ggml_tensor * x) {
    x = ggml_add(c, ggml_mul_mat(c, L->l1w, x), L->l1b);
    x = ggml_gelu(c, x);
    x = ggml_add(c, ggml_mul_mat(c, L->l2w, x), L->l2b);
    return x;
}

// Self-attn encoder layer (MyTransformerEncoderLayer, norm_first, layer_scale, norm_out).
// x:[C,T]. returns [C,T].
static struct ggml_tensor * htd_tx_self(struct ggml_context * c, HtdTLayer * L, struct ggml_tensor * x, int nh,
                                        float eps) {
    struct ggml_tensor * n = htd_ln(c, x, L->n1w, L->n1b, eps);
    struct ggml_tensor * sa = htd_mha(c, L, n, n, nh);
    sa = ggml_mul(c, sa, L->g1);  // LayerScale (channel-last -> per ne0)
    x = ggml_add(c, x, sa);
    struct ggml_tensor * n2 = htd_ln(c, x, L->n2w, L->n2b, eps);
    struct ggml_tensor * ff = htd_ffn(c, L, n2);
    ff = ggml_mul(c, ff, L->g2);
    x = ggml_add(c, x, ff);
    x = htd_gn_all(c, x, L->nout_w, L->nout_b, eps);
    return x;
}

// Cross-attn encoder layer (CrossTransformerEncoderLayer). q stream attends to k stream.
// q:[C,Tq], kv:[C,Tk]. returns [C,Tq].
static struct ggml_tensor * htd_tx_cross(struct ggml_context * c, HtdTLayer * L, struct ggml_tensor * q,
                                         struct ggml_tensor * kv, int nh, float eps) {
    struct ggml_tensor * nq = htd_ln(c, q, L->n1w, L->n1b, eps);
    struct ggml_tensor * nk = htd_ln(c, kv, L->n2w, L->n2b, eps);
    struct ggml_tensor * ca = htd_mha(c, L, nq, nk, nh);
    ca = ggml_mul(c, ca, L->g1);
    struct ggml_tensor * x = ggml_add(c, q, ca);
    struct ggml_tensor * n3 = htd_ln(c, x, L->n3w, L->n3b, eps);
    struct ggml_tensor * ff = htd_ffn(c, L, n3);
    ff = ggml_mul(c, ff, L->g2);
    x = ggml_add(c, x, ff);
    x = htd_gn_all(c, x, L->nout_w, L->nout_b, eps);
    return x;
}

// Host: 2D sin pos embedding (create_2d_sin_embedding) flattened to token order (t1 fr) -> [C, n_tok].
// matches transformer.create_2d_sin_embedding then rearrange "b c fr t1 -> b (t1 fr) c".
static void htd_pos2d(int C, int Fr, int T1, float max_period, std::vector<float> & out) {
    // pe[C, Fr, T1] per the torch fn; then token = t1*Fr+fr, feature = c.
    std::vector<float> pe((size_t) C * Fr * T1, 0.f);
    int d = C / 2;
    std::vector<double> div(d / 2);  // arange(0,d,2)/... -> d/2 terms
    for (int i = 0; i < d / 2; i++) div[i] = std::exp((double) (2 * i) * -(std::log((double) max_period) / d));
    auto idx = [&](int cc, int f, int t) { return ((size_t) cc * Fr + f) * T1 + t; };
    for (int w = 0; w < T1; w++)
        for (int h = 0; h < Fr; h++) {
            for (int i = 0; i < d / 2; i++) {
                pe[idx(2 * i + 0, h, w)]       = (float) std::sin(w * div[i]);   // pe[0:d:2] = sin(pos_w*div)
                pe[idx(2 * i + 1, h, w)]       = (float) std::cos(w * div[i]);   // pe[1:d:2] = cos(pos_w*div)
                pe[idx(d + 2 * i + 0, h, w)]   = (float) std::sin(h * div[i]);   // pe[d::2]  = sin(pos_h*div)
                pe[idx(d + 2 * i + 1, h, w)]   = (float) std::cos(h * div[i]);   // pe[d+1::2]= cos(pos_h*div)
            }
        }
    // flatten to ggml [C(ne0), n_tok(ne1)] with token = t1*Fr+fr -> index tok*C + c
    int nt = Fr * T1;
    out.assign((size_t) C * nt, 0.f);
    for (int cc = 0; cc < C; cc++)
        for (int t1 = 0; t1 < T1; t1++)
            for (int f = 0; f < Fr; f++)
                out[(size_t) (t1 * Fr + f) * C + cc] = pe[idx(cc, f, t1)];
}

// Host: 1D sin pos embedding (create_sin_embedding, shift=0) -> ggml [C(ne0), T(ne1)] index t*C + c.
// phase = pos / max_period^(adim/(half-1)); cat[cos, sin].
static void htd_pos1d(int C, int T, float max_period, std::vector<float> & out) {
    int half = C / 2;
    out.assign((size_t) C * T, 0.f);
    for (int t = 0; t < T; t++)
        for (int i = 0; i < half; i++) {
            double ph = (double) t / std::pow((double) max_period, (double) i / (half - 1));
            out[(size_t) t * C + i]        = (float) std::cos(ph);
            out[(size_t) t * C + half + i] = (float) std::sin(ph);
        }
}

// ================= decoder layers =================
// spec decoder. x:[T,F,C] (+skip same shape). conv_tr over freq (kh=8,kw=1,stride[4,1]), trim pad.
// rewrite is 3x3 over (freq,time). returns (out[T,F*4,Cout], pre[T,F,Cin]) ; pre used for branch split.
static struct ggml_tensor * htd_dec_spec(struct ggml_context * c, HtdDecLayer * D, struct ggml_tensor * x,
                                         struct ggml_tensor * skip, struct ggml_tensor ** pre_out) {
    x = ggml_add(c, x, skip);                       // [T, F, Cin]
    int64_t T = x->ne[0], F = x->ne[1], Cin = x->ne[2];
    // rewrite 3x3 -> 2Cin, GLU over channels. conv2d data [W=T,H=F,Cin], kernel [3,3,Cin,2Cin], s1 p1.
    struct ggml_tensor * z = ggml_conv_2d(c, D->rw_w, x, 1, 1, 1, 1, 1, 1);  // [T, F, 2Cin]
    z = ggml_add(c, z, ggml_reshape_3d(c, D->rw_b, 1, 1, 2 * Cin));
    struct ggml_tensor * a = ggml_cont(c, ggml_view_3d(c, z, T, F, Cin, z->nb[1], z->nb[2], 0));
    struct ggml_tensor * g = ggml_cont(c, ggml_view_3d(c, z, T, F, Cin, z->nb[1], z->nb[2], Cin * z->nb[2]));
    struct ggml_tensor * y = ggml_mul(c, a, ggml_sigmoid(c, g));  // [T, F, Cin] = pre
    *pre_out = y;
    // conv_tr over freq, vectorized (K=8=2*stride=2*4). GEMM batched over time:
    //   data [Cin, F*T] -> col [K*Cout, F*T] -> [K=8, Cout, F, T]. Output pos = i*4+k.
    //   low taps k0..3 -> A[4F]; high taps k4..7 -> B[4F] shifted +4; out = pad(A,r4)+pad(B,l4).
    int Cout = D->chout;
    int K = D->ktr;          // 8
    int S = 4;
    // y [T,F,Cin] -> [Cin, F, T]: src Cin(2)->0, F(1)->1, T(0)->2 => permute(2,1,0)
    struct ggml_tensor * yc = ggml_cont(c, ggml_permute(c, y, 2, 1, 0, 3));   // [Cin, F, T]
    struct ggml_tensor * yflat = ggml_reshape_2d(c, yc, Cin, (int64_t) F * T);  // [Cin, F*T]
    struct ggml_tensor * col = ggml_mul_mat(c, D->ctr_w, yflat);   // [K*Cout, F*T]
    col = ggml_reshape_4d(c, col, K, Cout, F, T);                  // [K, Cout, F, T] (k inner of K*Cout=oc*K+k)
    // low k0..3, high k4..7
    struct ggml_tensor * lo = ggml_cont(c, ggml_view_4d(c, col, S, Cout, F, T, col->nb[1], col->nb[2], col->nb[3], 0));
    struct ggml_tensor * hi = ggml_cont(c, ggml_view_4d(c, col, S, Cout, F, T, col->nb[1], col->nb[2], col->nb[3], (size_t) S * col->nb[0]));
    // interleave (k,F) -> pos=f*S+k : need [k, F, Cout, T] then reshape [S*F, Cout, T]
    lo = ggml_cont(c, ggml_permute(c, lo, 0, 2, 1, 3));  // [S, F, Cout, T] (src k->0,Cout->2,F->1)
    hi = ggml_cont(c, ggml_permute(c, hi, 0, 2, 1, 3));
    int SF = S * (int) F;
    lo = ggml_reshape_3d(c, lo, SF, Cout, T);            // [S*F, Cout, T] pos=f*S+k
    hi = ggml_reshape_3d(c, hi, SF, Cout, T);
    // out length = S*F + S = (F-1)*S + 2S = (F-1)*4+8. A at [0,SF), B at [S, S+SF).
    struct ggml_tensor * A = ggml_pad_ext(c, lo, 0, S, 0, 0, 0, 0, 0, 0);  // right pad S -> [SF+S, Cout, T]
    struct ggml_tensor * B = ggml_pad_ext(c, hi, S, 0, 0, 0, 0, 0, 0, 0);  // left pad S -> [S+SF, Cout, T]
    struct ggml_tensor * o = ggml_add(c, A, B);                  // [(F-1)*4+8, Cout, T]
    int pad = 2;
    int Ffull = (int) o->ne[0];
    o = ggml_cont(c, ggml_view_3d(c, o, Ffull - 2 * pad, Cout, T, o->nb[1], o->nb[2], (size_t) pad * o->nb[0]));
    o = ggml_add(c, o, ggml_reshape_3d(c, D->ctr_b, 1, Cout, 1));
    if (!D->last) o = ggml_gelu(c, o);
    o = ggml_cont(c, ggml_permute(c, o, 1, 2, 0, 3));  // src Fo->1,Cout->2,T->0 => [T,Fo,Cout]
    return o;
}

// time decoder. xt:[Tt, C] (+skip). conv_tr k8 s4 via col2im, trim to length. rewrite k3 pad1.
static struct ggml_tensor * htd_dec_time(struct ggml_context * c, HtdDecLayer * D, struct ggml_tensor * xt,
                                         struct ggml_tensor * skip, int length) {
    xt = ggml_add(c, xt, skip);  // [Tt, C]
    int64_t Cin = xt->ne[1];
    struct ggml_tensor * z = htd_conv1d(c, D->rw_w, xt, 1, 1, 1);  // [Tt, 2Cin]
    z = ggml_add(c, z, ggml_reshape_2d(c, D->rw_b, 1, 2 * Cin));
    int64_t Tt = z->ne[0];
    struct ggml_tensor * a = ggml_cont(c, ggml_view_2d(c, z, Tt, Cin, z->nb[1], 0));
    struct ggml_tensor * g = ggml_cont(c, ggml_view_2d(c, z, Tt, Cin, z->nb[1], Cin * z->nb[1]));
    struct ggml_tensor * y = ggml_mul(c, a, ggml_sigmoid(c, g));  // [Tt, Cin]
    // conv_tr1d k8 s4 via col2im: x[Tt,Cin]->[Cin,Tt] GEMM w[Cin,K*Cout]->[K*Cout,Tt] col2im->[OL,Cout]
    struct ggml_tensor * yt = ggml_cont(c, ggml_transpose(c, y));   // [Cin, Tt]
    struct ggml_tensor * col = ggml_mul_mat(c, D->ctr_w, yt);       // [K*Cout, Tt]
    int Cout = D->chout, pad = 2;
    struct ggml_tensor * o = ggml_col2im_1d(c, col, 4, Cout, pad);  // [OL, Cout] (pad cropped)
    o = ggml_add(c, o, ggml_reshape_2d(c, D->ctr_b, 1, Cout));
    // col2im crops pad both sides giving OL=(Tt-1)*4+8-4; torch trims to [pad:pad+length] then asserts == length.
    o = ggml_cont(c, ggml_view_2d(c, o, length, Cout, o->nb[1], 0));
    if (!D->last) o = ggml_gelu(c, o);
    return o;
}

// ================= full single-segment forward =================
struct HtdForwardOut {
    // transformer per-layer outputs (channel-major torch [C,F,T] for x, [C,T] for xt)
    std::vector<float> ct_x_layer[5], ct_xt_layer[5];
    int ct_F, ct_T1, ct_T2;
    // decoder outputs (channel-major)
    std::vector<float> dec[4], tdec[4];
    int dec_ch[4], dec_F[4], dec_T;
    int tdec_ch[4], tdec_L[4];
    // final stems [4, AC, length] channel-major (drums,bass,other,vocals)
    std::vector<float> stems;  // [4*AC*length]
    int length, AC;
};

static void htd_forward(SonggenHTDemucs * m, const float * mix, int N, int AC, HtdForwardOut * out) {
    const bool htd_timing = getenv("HTD_TIMING") != nullptr;
    auto       tnow       = [] { return std::chrono::steady_clock::now(); };
    auto       tms        = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    auto       t0         = tnow();
    std::vector<float> mag, z_re, z_im;
    int Fr = 0, T = 0;
    htd_spec_cac(m, mix, N, AC, mag, z_re, z_im, &Fr, &T);
    auto t_after_stft = tnow();
    int Cspec = 2 * AC;

    double smean, sstd, tmean, tstd;
    {
        double s = 0, s2 = 0; size_t n = mag.size();
        for (float v : mag) { s += v; s2 += (double) v * v; }
        smean = s / (double) n;
        sstd = std::sqrt((s2 - (double) n * smean * smean) / (double) (n - 1));
        float inv = 1.0f / (float) (1e-5 + sstd);
        for (float & v : mag) v = (float) ((v - smean) * inv);
    }
    std::vector<float> xt((size_t) AC * N);
    {
        double s = 0, s2 = 0; size_t n = (size_t) AC * N;
        for (int c = 0; c < AC; c++) for (int i = 0; i < N; i++) { float v = mix[(size_t) c * N + i]; s += v; s2 += (double) v * v; }
        tmean = s / (double) n;
        tstd = std::sqrt((s2 - (double) n * tmean * tmean) / (double) (n - 1));
        float inv = 1.0f / (float) (1e-5 + tstd);
        for (int c = 0; c < AC; c++) for (int i = 0; i < N; i++) xt[(size_t) c * N + i] = (float) ((mix[(size_t) c * N + i] - tmean) * inv);
    }

    size_t ctx_size = ggml_tensor_overhead() * 200000 + ggml_graph_overhead_custom(131072, false);
    uint8_t * gbuf = (uint8_t *) malloc(ctx_size);
    struct ggml_init_params gp = { ctx_size, gbuf, true };
    struct ggml_context * c = ggml_init(gp);

    struct ggml_tensor * x = ggml_new_tensor_3d(c, GGML_TYPE_F32, T, Fr, Cspec);
    ggml_set_input(x); ggml_set_name(x, "spec_in");
    struct ggml_tensor * xtin = ggml_new_tensor_2d(c, GGML_TYPE_F32, N, AC);
    ggml_set_input(xtin); ggml_set_name(xtin, "time_in");

    // ---- encoders, save skips ----
    struct ggml_tensor * saved[4];
    struct ggml_tensor * saved_t[4];
    struct ggml_tensor * sx = x;
    for (int i = 0; i < 4; i++) {
        sx = htd_enc_spec(c, &m->enc[i], sx);
        if (i == 0) {
            struct ggml_tensor * fet = ggml_cont(c, ggml_transpose(c, m->freq_emb));
            fet = ggml_reshape_3d(c, fet, 1, fet->ne[0], fet->ne[1]);
            sx = ggml_add(c, sx, ggml_scale(c, fet, m->freq_emb_scale * 10.0f));
        }
        saved[i] = sx;
    }
    struct ggml_tensor * tx = xtin;
    for (int i = 0; i < 4; i++) { tx = htd_enc_time(c, &m->tenc[i], tx); saved_t[i] = tx; }
    // sx: [T=336, F=8, C=384]; tx: [Tt=1344, C=384]

    int T1 = (int) sx->ne[0], Ffr = (int) sx->ne[1];
    int Cx = (int) sx->ne[2];
    int Tt = (int) tx->ne[0];

    // ---- channel upsample 384->512 ----
    // spec: rearrange b c f t -> b c (f t) then Conv1d 1x1. operates per (f,t). need feature=C on ne0.
    // sx [T,F,C] -> tokens (f t) with feature C. Conv1d weight up_w bf16 ggml [384,512].
    // put sx as [C, (f t)] then proj. token order torch (f t) = f*T1+t? rearrange "b c f t -> b c (f t)": last dim t fastest -> idx f*T1+t.
    struct ggml_tensor * sx_ct = ggml_cont(c, ggml_permute(c, sx, 1, 2, 0, 3));  // src T->1,F->2,C->0 => [C, T, F]
    sx_ct = ggml_reshape_2d(c, sx_ct, Cx, (int64_t) T1 * Ffr);   // [C, T*F] token=f*T1+t? our order: ne1 iterates T then F => token = f*T1 + t
    struct ggml_tensor * xup = htd_proj(c, m->up_w, m->up_b, sx_ct);  // [512, T*F]
    struct ggml_tensor * tx_ct = ggml_cont(c, ggml_transpose(c, tx));  // [C, Tt]
    struct ggml_tensor * xtup = htd_proj(c, m->up_t_w, m->up_t_b, tx_ct);  // [512, Tt]

    int Cd = m->t_dim;  // 512
    // ---- transformer ----
    // pos embeddings (host) -> input tensors
    struct ggml_tensor * pos2d = ggml_new_tensor_2d(c, GGML_TYPE_F32, Cd, (int64_t) T1 * Ffr);
    ggml_set_input(pos2d); ggml_set_name(pos2d, "pos2d");
    struct ggml_tensor * pos1d = ggml_new_tensor_2d(c, GGML_TYPE_F32, Cd, Tt);
    ggml_set_input(pos1d); ggml_set_name(pos1d, "pos1d");

    // The transformer token order for x is (t1 fr): token = t1*Fr + fr. Our xup token = f*T1 + t.
    // Need to REORDER xup tokens to (t1 fr). Build a permuted view via reshape+permute.
    // xup currently [Cd, F*T1] with token = f*T1 + t (f outer, t inner). Want token = t*Fr + f (t outer, f inner).
    struct ggml_tensor * xup3 = ggml_reshape_3d(c, xup, Cd, T1, Ffr);  // [Cd, t, f]
    xup3 = ggml_cont(c, ggml_permute(c, xup3, 0, 2, 1, 3));            // [Cd, f, t]
    struct ggml_tensor * xs = ggml_reshape_2d(c, xup3, Cd, (int64_t) Ffr * T1);  // token = t*Fr + f

    // norm_in + pos
    xs = htd_ln(c, xs, m->normin_w, m->normin_b, 1e-5f);
    xs = ggml_add(c, xs, ggml_scale(c, pos2d, 1.0f));
    struct ggml_tensor * xtt = htd_ln(c, xtup, m->normin_t_w, m->normin_t_b, 1e-5f);
    xtt = ggml_add(c, xtt, ggml_scale(c, pos1d, 1.0f));

    struct ggml_tensor * ctx_layer[5]; struct ggml_tensor * ctxt_layer[5];
    for (int i = 0; i < 5; i++) {
        bool is_cross = (i % 2 == 1);
        if (!is_cross) {
            xs = htd_tx_self(c, &m->tx[i], xs, m->t_heads, 1e-5f);
            xtt = htd_tx_self(c, &m->tx_t[i], xtt, m->t_heads, 1e-5f);
        } else {
            struct ggml_tensor * oldx = xs;
            xs = htd_tx_cross(c, &m->tx[i], xs, xtt, m->t_heads, 1e-5f);
            xtt = htd_tx_cross(c, &m->tx_t[i], xtt, oldx, m->t_heads, 1e-5f);
        }
        ctx_layer[i] = xs; ctxt_layer[i] = xtt;
    }

    // ---- channel downsample 512->384 ----
    // xs token order (t fr). downsample expects (f t) order for rearrange back. Reorder tokens (t fr)->(f t).
    struct ggml_tensor * xs3 = ggml_reshape_3d(c, xs, Cd, Ffr, T1);  // [Cd, f, t]
    xs3 = ggml_cont(c, ggml_permute(c, xs3, 0, 2, 1, 3));            // [Cd, t, f]
    struct ggml_tensor * xs_ft = ggml_reshape_2d(c, xs3, Cd, (int64_t) T1 * Ffr);  // token = f*T1+t
    struct ggml_tensor * xdown = htd_proj(c, m->down_w, m->down_b, xs_ft);  // [384, F*T1]
    struct ggml_tensor * xtdown = htd_proj(c, m->down_t_w, m->down_t_b, xtt);  // [384, Tt]

    // reshape spec back to [T, F, C]: xdown [Cx, F*T1] token=f*T1+t -> [Cx, t, f] -> [T,F,C]
    struct ggml_tensor * xd3 = ggml_reshape_3d(c, xdown, Cx, T1, Ffr);  // [Cx, t, f]
    struct ggml_tensor * xdec = ggml_cont(c, ggml_permute(c, xd3, 2, 0, 1, 3));  // src Cx->2,t->0,f->1 => [t, f, Cx]
    struct ggml_tensor * xtdec = ggml_cont(c, ggml_transpose(c, xtdown));  // [Tt, 384]

    // ---- decoders ----
    // input length to tenc[i] (torch lengths_t): tenc0 input = N; tenc[i] input = tenc[i-1] output.
    int in_len_t[4] = { N, (int) saved_t[0]->ne[0], (int) saved_t[1]->ne[0], (int) saved_t[2]->ne[0] };
    struct ggml_tensor * dec_out[4]; struct ggml_tensor * tdec_out[4];
    struct ggml_tensor * dx = xdec;
    struct ggml_tensor * dxt = xtdec;
    for (int i = 0; i < 4; i++) {
        struct ggml_tensor * skip = saved[3 - i];
        struct ggml_tensor * pre = nullptr;
        dx = htd_dec_spec(c, &m->dec[i], dx, skip, &pre);
        dec_out[i] = dx;
        // tdecoder[i] pops skip=saved_t[3-i], output length = in_len_t[3-i] (input len to that tenc).
        struct ggml_tensor * tskip = saved_t[3 - i];
        int len_t = in_len_t[3 - i];
        dxt = htd_dec_time(c, &m->tdec[i], dxt, tskip, len_t);
        tdec_out[i] = dxt;
    }

    // collect outputs
    std::vector<struct ggml_tensor *> outs;
    for (int i = 0; i < 5; i++) { outs.push_back(ctx_layer[i]); outs.push_back(ctxt_layer[i]); }
    for (int i = 0; i < 4; i++) { outs.push_back(dec_out[i]); outs.push_back(tdec_out[i]); }
    for (auto * o : outs) ggml_set_output(o);

    struct ggml_cgraph * g = ggml_new_graph_custom(c, 131072, false);
    for (auto * o : outs) ggml_build_forward_expand(g, o);
    if (!ggml_backend_sched_alloc_graph(m->sched, g)) { fprintf(stderr, "[HTDEMUCS] fwd alloc failed\n"); exit(1); }

    ggml_backend_tensor_set(x, mag.data(), 0, mag.size() * sizeof(float));
    ggml_backend_tensor_set(xtin, xt.data(), 0, xt.size() * sizeof(float));
    std::vector<float> p2, p1;
    htd_pos2d(Cd, Ffr, T1, m->max_period, p2);
    htd_pos1d(Cd, Tt, m->max_period, p1);
    ggml_backend_tensor_set(pos2d, p2.data(), 0, p2.size() * sizeof(float));
    ggml_backend_tensor_set(pos1d, p1.data(), 0, p1.size() * sizeof(float));

    auto t_before_gpu = tnow();
    ggml_backend_sched_graph_compute(m->sched, g);
    if (htd_timing) ggml_backend_sched_synchronize(m->sched);
    auto t_after_gpu = tnow();

    auto fetch = [&](struct ggml_tensor * t) {
        std::vector<float> v((size_t) ggml_nelements(t));
        ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
        return v;
    };
    // ct layer x: ggml [Cd, token=(t fr)] -> torch [C, Fr, T1]: element (cc, f, t1) = tok t1*Fr+f.
    auto ct_x_cm = [&](struct ggml_tensor * t) {
        std::vector<float> raw = fetch(t);  // [Cd, nt]
        int nt = (int) t->ne[1];
        std::vector<float> r((size_t) Cd * Ffr * T1);
        for (int cc = 0; cc < Cd; cc++)
            for (int t1 = 0; t1 < T1; t1++)
                for (int f = 0; f < Ffr; f++)
                    r[((size_t) cc * Ffr + f) * T1 + t1] = raw[(size_t) (t1 * Ffr + f) * Cd + cc];
        (void) nt;
        return r;
    };
    auto ct_xt_cm = [&](struct ggml_tensor * t) {
        std::vector<float> raw = fetch(t);  // [Cd, Tt]
        std::vector<float> r((size_t) Cd * Tt);
        for (int cc = 0; cc < Cd; cc++) for (int tt = 0; tt < Tt; tt++) r[(size_t) cc * Tt + tt] = raw[(size_t) tt * Cd + cc];
        return r;
    };
    for (int i = 0; i < 5; i++) { out->ct_x_layer[i] = ct_x_cm(ctx_layer[i]); out->ct_xt_layer[i] = ct_xt_cm(ctxt_layer[i]); }
    out->ct_F = Ffr; out->ct_T1 = T1; out->ct_T2 = Tt;

    // decoder spec out ggml [T, F', C] -> torch [C, F', T] (identical contiguous)
    auto dec_cm = [&](struct ggml_tensor * t) {
        return fetch(t);  // [T,F',C] ggml == [C,F',T] torch C-order (index (c*F'+f)*T+t)
    };
    for (int i = 0; i < 4; i++) {
        out->dec[i] = dec_cm(dec_out[i]);
        out->dec_ch[i] = (int) dec_out[i]->ne[2];
        out->dec_F[i] = (int) dec_out[i]->ne[1];
        out->tdec[i] = fetch(tdec_out[i]);  // [L, C] == [C, L] torch
        out->tdec_ch[i] = (int) tdec_out[i]->ne[1];
        out->tdec_L[i] = (int) tdec_out[i]->ne[0];
    }
    out->dec_T = T;

    // ---- final spec branch: dec_out[3] is [T, F=2048, C=16]; reshape [S=4, 4cac, F, T], denorm, iSTFT ----
    std::vector<float> xspec = out->dec[3];  // [16, 2048, 336] channel-major (c*F+f)*T+t
    int Cfinal = out->dec_ch[3];  // 16
    int Ff2 = out->dec_F[3];      // 2048
    int S = 4;
    int training_length = m->segment_samples;
    out->length = training_length;
    out->AC = AC;
    out->stems.assign((size_t) S * AC * training_length, 0.f);

    // denorm: x*std + mean ; then CAC: for source s, audio ch a -> complex (re,im) = chan (s*4 + 2a, s*4 + 2a+1)
    // The S*AC=8 channel reconstructions are independent (disjoint output slices, own buffers, m read-only),
    // and the host iSTFT (radix-2 FFT) is the largest forward phase — parallelize across worker threads.
    auto t_before_istft = tnow();
    auto istft_one = [&](int s, int a) {
        int cre = s * (2 * AC) + 2 * a, cim = s * (2 * AC) + 2 * a + 1;
        std::vector<float> zr((size_t) Ff2 * T), zi((size_t) Ff2 * T);
        for (int f = 0; f < Ff2; f++)
            for (int t = 0; t < T; t++) {
                zr[(size_t) f * T + t] = xspec[((size_t) cre * Ff2 + f) * T + t] * (float) sstd + (float) smean;
                zi[(size_t) f * T + t] = xspec[((size_t) cim * Ff2 + f) * T + t] * (float) sstd + (float) smean;
            }
        std::vector<float> wav;
        htd_ispec_chan(m, zr.data(), zi.data(), Ff2, T, training_length, wav);
        for (int t = 0; t < training_length; t++) out->stems[((size_t) s * AC + a) * training_length + t] = wav[t];
    };
    {
        int njobs = S * AC;
        int nthreads = (int) std::thread::hardware_concurrency();
        if (nthreads <= 0) nthreads = 4;
        if (nthreads > njobs) nthreads = njobs;
        std::vector<std::thread> pool;
        std::atomic<int>         next{ 0 };
        for (int w = 0; w < nthreads; w++)
            pool.emplace_back([&] {
                for (int job = next.fetch_add(1); job < njobs; job = next.fetch_add(1))
                    istft_one(job / AC, job % AC);
            });
        for (auto & th : pool) th.join();
    }
    // ---- time branch: tdec_out[3] is [length, S*AC=8]; reshape [S, AC, length], denorm, add ----
    std::vector<float> xtime = out->tdec[3];  // [8, length] channel-major
    int Ctime = out->tdec_ch[3];  // 8
    (void) Ctime;
    for (int s = 0; s < S; s++)
        for (int a = 0; a < AC; a++) {
            int ch = s * AC + a;
            for (int t = 0; t < training_length; t++) {
                float tv = xtime[(size_t) ch * training_length + t] * (float) tstd + (float) tmean;
                out->stems[((size_t) s * AC + a) * training_length + t] += tv;
            }
        }

    if (getenv("HTD_DEBUG_RMS")) {
        // RMS of each branch's contribution in stem 3 (vocal), pre-sum, to see which blows up.
        double spec_sq = 0, tot_sq = 0; int a0 = 0; size_t cnt = training_length;
        for (int t = 0; t < training_length; t++) {
            double tot = out->stems[((size_t) 3 * AC + a0) * training_length + t];
            tot_sq += tot * tot;
        }
        for (int t = 0; t < (int) cnt; t++) (void) spec_sq;
        fprintf(stderr, "[HTD_NORM] smean=%.5f sstd=%.5f tmean=%.5f tstd=%.5f | stem3ch0 rms=%.4f\n",
                smean, sstd, tmean, tstd, sqrt(tot_sq / (double) cnt));
    }
    auto t_after_istft = tnow();
    if (htd_timing) {
        fprintf(stderr,
                "[HTD_TIMING] stft=%.1fms graph_build=%.1fms gpu=%.1fms istft=%.1fms total=%.1fms\n",
                tms(t0, t_after_stft), tms(t_after_stft, t_before_gpu), tms(t_before_gpu, t_after_gpu),
                tms(t_before_istft, t_after_istft), tms(t0, t_after_istft));
    }
    fprintf(stderr, "[HTDEMUCS] forward graph %d nodes\n", ggml_graph_n_nodes(g));
    ggml_backend_sched_reset(m->sched);
    ggml_free(c);
    free(gbuf);
}
