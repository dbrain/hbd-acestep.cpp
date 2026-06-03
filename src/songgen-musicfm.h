// songgen-musicfm.h : SongGeneration MusicFM_95M (BESTRQ) audio encoder (GGML).
//
// features_only path (extract_bestrq_embeds): 24k mono audio ->
//   [host] mel front-end: STFT(n_fft=2048,hop=240,center,reflect) -> power -> mel fb[1025,128]
//          -> AmplitudeToDB(power,top_db=80,amin=1e-10) -> (x-6.7684)/18.4179 -> drop last frame
//   [ggml] Conv2dSubsampling: 2x Res2dModule(Conv2d3x3+BN+ReLU, strided residual) 4x down in
//          freq&time -> rearrange b c f t -> b t (c f) -> Linear(16384->1024)
//   [ggml] Wav2Vec2ConformerEncoder (do_stable_layer_norm, rotary): 12 macaron layers
//          0.5*FFN1 -> self_attn(rotary NEOX, 16h hd64) -> conv_module(LN,pw1,GLU,dw k31,BN,
//          swish,pw2) -> 0.5*FFN2 -> final_LN; final encoder LN.
//   layer_results[0]=conv-stem output (pre layer0), [i+1]=output of layer i, [12]=final-LN'd.
// pos_conv_embed is UNUSED in the HF encoder forward (rotary path) -> skipped.
#pragma once

#include "backend.h"
#include "gguf-weights.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// BatchNorm folded at load to affine: y = x*scale + shift.
struct SgMfBN {
    struct ggml_tensor *scale, *shift;  // [C]
};

struct SgMfRes2d {
    struct ggml_tensor *c1w, *c1b, *c2w, *c2b, *c3w, *c3b;
    SgMfBN              bn1, bn2, bn3;
    int                 in_ch, out_ch;
};

struct SgMfLayer {
    struct ggml_tensor *ffn1_ln_w, *ffn1_ln_b, *ffn1_id_w, *ffn1_id_b, *ffn1_od_w, *ffn1_od_b;
    struct ggml_tensor *attn_ln_w, *attn_ln_b;
    struct ggml_tensor *q_w, *q_b, *k_w, *k_b, *v_w, *v_b, *o_w, *o_b;
    struct ggml_tensor *cm_ln_w, *cm_ln_b, *cm_pw1_w, *cm_dw_w, *cm_pw2_w;
    SgMfBN              cm_bn;
    struct ggml_tensor *ffn2_ln_w, *ffn2_ln_b, *ffn2_id_w, *ffn2_id_b, *ffn2_od_w, *ffn2_od_b;
    struct ggml_tensor *final_ln_w, *final_ln_b;
};

struct SonggenMusicFM {
    int   n_layers, hidden, heads, head_dim, ffn, conv_dim, n_mels, n_fft, hop, sample_rate, depthwise_k;
    int   rotary_base;
    float bn_eps, ln_eps, mel_mean, mel_std;

    std::vector<float> mel_fb;      // [1025*128] row-major [freq][mel]
    std::vector<float> mel_window;  // [n_fft]

    SgMfRes2d           res[2];
    struct ggml_tensor *lin_w, *lin_b;
    struct ggml_tensor *enc_ln_w, *enc_ln_b;
    SgMfLayer           layer[12];

    WeightCtx            wctx;
    ggml_backend_t       backend;
    ggml_backend_t       cpu_backend;
    ggml_backend_sched_t sched;
};

static struct ggml_tensor * sgmf_f32(WeightCtx * w, const GGUFModel & gf, const std::string & name) {
    return gf_load_tensor_f32(w, gf, name);
}
static struct ggml_tensor * sgmf_bf16(WeightCtx * w, const GGUFModel & gf, const std::string & name) {
    return gf_load_tensor(w, gf, name);
}

// Conv2d weight torch [OC,IC,KH,KW] (bf16 in gguf) -> ggml [KW,KH,IC,OC] f32 for ggml_conv_2d.
static struct ggml_tensor * sgmf_conv2d_w(WeightCtx * w, const GGUFModel & gf, const std::string & name, int oc,
                                          int ic, int kh, int kw) {
    const ggml_bf16_t * v   = (const ggml_bf16_t *) gf_get_data(gf, name.c_str());  // [oc][ic][kh][kw]
    auto                buf = std::make_unique<float[]>((size_t) kw * kh * ic * oc);
    float *             d   = buf.get();
    for (int o = 0; o < oc; o++)
        for (int i = 0; i < ic; i++)
            for (int a = 0; a < kh; a++)
                for (int b = 0; b < kw; b++) {
                    float val = ggml_bf16_to_fp32(v[(((size_t) o * ic + i) * kh + a) * kw + b]);
                    // ggml layout ne0=kw, ne1=kh, ne2=ic, ne3=oc
                    d[(((size_t) o * ic + i) * kh + a) * kw + b] = val;
                }
    int64_t              ne[4]  = { kw, kh, ic, oc };
    struct ggml_tensor * tensor = ggml_new_tensor(w->ctx, GGML_TYPE_F32, 4, ne);
    ggml_set_name(tensor, name.c_str());
    w->pending.push_back({ tensor, d, (size_t) kw * kh * ic * oc * sizeof(float), 0 });
    w->staging.push_back(std::move(buf));
    return tensor;
}

// Read a raw f32 tensor from gguf into a host vector.
static void sgmf_read_f32(const GGUFModel & gf, const std::string & name, std::vector<float> & out, int n) {
    out.resize(n);
    memcpy(out.data(), gf_get_data(gf, name.c_str()), (size_t) n * sizeof(float));
}

static void sgmf_load_bn(SgMfBN * bn, WeightCtx * w, const GGUFModel & gf, const std::string & p, int C, float eps) {
    std::vector<float> g, b, rm, rv;
    sgmf_read_f32(gf, p + ".weight", g, C);
    sgmf_read_f32(gf, p + ".bias", b, C);
    sgmf_read_f32(gf, p + ".running_mean", rm, C);
    sgmf_read_f32(gf, p + ".running_var", rv, C);
    auto    sbuf = std::make_unique<float[]>(C);
    auto    hbuf = std::make_unique<float[]>(C);
    float * sp   = sbuf.get();
    float * hp   = hbuf.get();
    for (int c = 0; c < C; c++) {
        float s = g[c] / std::sqrt(rv[c] + eps);
        sp[c]   = s;
        hp[c]   = b[c] - rm[c] * s;
    }
    int64_t ne[1] = { C };
    bn->scale     = ggml_new_tensor(w->ctx, GGML_TYPE_F32, 1, ne);
    bn->shift     = ggml_new_tensor(w->ctx, GGML_TYPE_F32, 1, ne);
    ggml_set_name(bn->scale, (p + ".scale").c_str());
    ggml_set_name(bn->shift, (p + ".shift").c_str());
    w->pending.push_back({ bn->scale, sp, (size_t) C * sizeof(float), 0 });
    w->pending.push_back({ bn->shift, hp, (size_t) C * sizeof(float), 0 });
    w->staging.push_back(std::move(sbuf));
    w->staging.push_back(std::move(hbuf));
}

static bool sgmf_load(SonggenMusicFM * m, const char * gguf_path) {
    *m = {};
    BackendPair bp = backend_init("MUSICFM");
    m->backend     = bp.backend;
    m->cpu_backend = bp.cpu_backend;
    m->sched       = backend_sched_new(bp, 16384);

    GGUFModel gf;
    if (!gf_load(&gf, gguf_path)) return false;

    const char * A      = "songgen-musicfm";
    auto         u32    = [&](const char * k) { return (int) gf_get_u32(gf, (std::string(A) + "." + k).c_str()); };
    auto         f32kv  = [&](const char * k) { return gf_get_f32(gf, (std::string(A) + "." + k).c_str()); };
    m->n_layers         = u32("n_layers");
    m->hidden           = u32("hidden");
    m->heads            = u32("heads");
    m->head_dim         = u32("head_dim");
    m->ffn              = u32("ffn");
    m->conv_dim         = u32("conv_dim");
    m->n_mels           = u32("n_mels");
    m->n_fft            = u32("n_fft");
    m->hop              = u32("hop");
    m->sample_rate      = u32("sample_rate");
    m->depthwise_k      = u32("depthwise_kernel");
    m->rotary_base      = u32("rotary_base");
    m->bn_eps           = f32kv("bn_eps");
    m->ln_eps           = f32kv("ln_eps");
    m->mel_mean         = f32kv("mel_mean");
    m->mel_std          = f32kv("mel_std");
    if (m->n_layers <= 0) m->n_layers = 12;

    fprintf(stderr, "[MUSICFM] L=%d H=%d heads=%d hd=%d ffn=%d nmels=%d nfft=%d hop=%d\n", m->n_layers, m->hidden,
            m->heads, m->head_dim, m->ffn, m->n_mels, m->n_fft, m->hop);

    // mel buffers (read raw f32 from gguf, keep host-side)
    {
        int freq = m->n_fft / 2 + 1;
        m->mel_fb.resize((size_t) freq * m->n_mels);
        memcpy(m->mel_fb.data(), gf_get_data(gf, "mel.fb"), m->mel_fb.size() * sizeof(float));
        m->mel_window.resize(m->n_fft);
        memcpy(m->mel_window.data(), gf_get_data(gf, "mel.window"), m->mel_window.size() * sizeof(float));
    }

    wctx_init(&m->wctx, 600);

    // conv stem res2d
    for (int r = 0; r < 2; r++) {
        SgMfRes2d & R = m->res[r];
        R.in_ch       = r == 0 ? 1 : m->conv_dim;
        R.out_ch      = m->conv_dim;
        std::string o = "conv.res." + std::to_string(r);
        R.c1w         = sgmf_conv2d_w(&m->wctx, gf, o + ".conv1.weight", R.out_ch, R.in_ch, 3, 3);
        R.c1b         = sgmf_f32(&m->wctx, gf, o + ".conv1.bias");
        R.c2w         = sgmf_conv2d_w(&m->wctx, gf, o + ".conv2.weight", R.out_ch, R.out_ch, 3, 3);
        R.c2b         = sgmf_f32(&m->wctx, gf, o + ".conv2.bias");
        R.c3w         = sgmf_conv2d_w(&m->wctx, gf, o + ".conv3.weight", R.out_ch, R.in_ch, 3, 3);
        R.c3b         = sgmf_f32(&m->wctx, gf, o + ".conv3.bias");
        sgmf_load_bn(&R.bn1, &m->wctx, gf, o + ".bn1", R.out_ch, m->bn_eps);
        sgmf_load_bn(&R.bn2, &m->wctx, gf, o + ".bn2", R.out_ch, m->bn_eps);
        sgmf_load_bn(&R.bn3, &m->wctx, gf, o + ".bn3", R.out_ch, m->bn_eps);
    }
    m->lin_w    = sgmf_bf16(&m->wctx, gf, "conv.linear.weight");  // [16384,1024]
    m->lin_b    = sgmf_f32(&m->wctx, gf, "conv.linear.bias");
    m->enc_ln_w = sgmf_f32(&m->wctx, gf, "conformer.layer_norm.weight");
    m->enc_ln_b = sgmf_f32(&m->wctx, gf, "conformer.layer_norm.bias");

    for (int li = 0; li < m->n_layers; li++) {
        SgMfLayer & L  = m->layer[li];
        std::string lp = "conformer.layers." + std::to_string(li);
        L.ffn1_ln_w    = sgmf_f32(&m->wctx, gf, lp + ".ffn1_layer_norm.weight");
        L.ffn1_ln_b    = sgmf_f32(&m->wctx, gf, lp + ".ffn1_layer_norm.bias");
        L.ffn1_id_w    = sgmf_bf16(&m->wctx, gf, lp + ".ffn1.intermediate_dense.weight");
        L.ffn1_id_b    = sgmf_f32(&m->wctx, gf, lp + ".ffn1.intermediate_dense.bias");
        L.ffn1_od_w    = sgmf_bf16(&m->wctx, gf, lp + ".ffn1.output_dense.weight");
        L.ffn1_od_b    = sgmf_f32(&m->wctx, gf, lp + ".ffn1.output_dense.bias");
        L.attn_ln_w    = sgmf_f32(&m->wctx, gf, lp + ".self_attn_layer_norm.weight");
        L.attn_ln_b    = sgmf_f32(&m->wctx, gf, lp + ".self_attn_layer_norm.bias");
        L.q_w          = sgmf_bf16(&m->wctx, gf, lp + ".self_attn.linear_q.weight");
        L.q_b          = sgmf_f32(&m->wctx, gf, lp + ".self_attn.linear_q.bias");
        L.k_w          = sgmf_bf16(&m->wctx, gf, lp + ".self_attn.linear_k.weight");
        L.k_b          = sgmf_f32(&m->wctx, gf, lp + ".self_attn.linear_k.bias");
        L.v_w          = sgmf_bf16(&m->wctx, gf, lp + ".self_attn.linear_v.weight");
        L.v_b          = sgmf_f32(&m->wctx, gf, lp + ".self_attn.linear_v.bias");
        L.o_w          = sgmf_bf16(&m->wctx, gf, lp + ".self_attn.linear_out.weight");
        L.o_b          = sgmf_f32(&m->wctx, gf, lp + ".self_attn.linear_out.bias");
        L.cm_ln_w      = sgmf_f32(&m->wctx, gf, lp + ".conv_module.layer_norm.weight");
        L.cm_ln_b      = sgmf_f32(&m->wctx, gf, lp + ".conv_module.layer_norm.bias");
        L.cm_pw1_w     = sgmf_bf16(&m->wctx, gf, lp + ".conv_module.pointwise_conv1.weight");  // [1024,2048]
        L.cm_dw_w      = sgmf_f32(&m->wctx, gf, lp + ".conv_module.depthwise_conv.weight");   // [31,1024] f32 for elementwise
        sgmf_load_bn(&L.cm_bn, &m->wctx, gf, lp + ".conv_module.batch_norm", m->hidden, m->bn_eps);
        L.cm_pw2_w  = sgmf_bf16(&m->wctx, gf, lp + ".conv_module.pointwise_conv2.weight");  // [1024,1024]
        L.ffn2_ln_w = sgmf_f32(&m->wctx, gf, lp + ".ffn2_layer_norm.weight");
        L.ffn2_ln_b = sgmf_f32(&m->wctx, gf, lp + ".ffn2_layer_norm.bias");
        L.ffn2_id_w = sgmf_bf16(&m->wctx, gf, lp + ".ffn2.intermediate_dense.weight");
        L.ffn2_id_b = sgmf_f32(&m->wctx, gf, lp + ".ffn2.intermediate_dense.bias");
        L.ffn2_od_w = sgmf_bf16(&m->wctx, gf, lp + ".ffn2.output_dense.weight");
        L.ffn2_od_b = sgmf_f32(&m->wctx, gf, lp + ".ffn2.output_dense.bias");
        L.final_ln_w = sgmf_f32(&m->wctx, gf, lp + ".final_layer_norm.weight");
        L.final_ln_b = sgmf_f32(&m->wctx, gf, lp + ".final_layer_norm.bias");
    }

    if (!wctx_alloc(&m->wctx, m->backend)) {
        gf_close(&gf);
        return false;
    }
    gf_close(&gf);
    return true;
}

// ---------------- host mel front-end ----------------
// Returns mel[n_mels * T] row-major [mel][t] (channel-major), and sets T.
// Matches golden_clone_bestrq.py: STFT n_fft, hop, center+reflect, power=2, mel fb, AmpToDB
// (10*log10, amin=1e-10, top_db=80), normalize (x-mean)/std, drop last frame.
static void sgmf_mel(const SonggenMusicFM * m, const float * audio, int N, std::vector<float> & mel_out, int * T_out) {
    const int n_fft = m->n_fft, hop = m->hop, freq = n_fft / 2 + 1, nm = m->n_mels;
    const int pad   = n_fft / 2;  // center reflect pad

    // reflect-pad the signal
    std::vector<float> x((size_t) N + 2 * pad);
    for (int i = 0; i < N; i++) x[pad + i] = audio[i];
    for (int i = 0; i < pad; i++) {
        x[pad - 1 - i]         = audio[i + 1];       // reflect (exclude boundary)
        x[pad + N + i]         = audio[N - 2 - i];
    }
    int Lp     = N + 2 * pad;
    int frames = 1 + (Lp - n_fft) / hop;  // center=True frame count

    // precompute DFT basis cos/sin for rFFT (k=0..freq-1, n=0..n_fft-1)
    // power[k] = (sum_n xw[n]*cos)^2 + (sum_n xw[n]*sin)^2, sin negated for e^{-i}
    // Build per-frame; reuse cos/sin tables.
    static thread_local std::vector<float> cosb, sinb;
    if ((int) cosb.size() != freq * n_fft) {
        cosb.assign((size_t) freq * n_fft, 0.f);
        sinb.assign((size_t) freq * n_fft, 0.f);
        const double tp = 2.0 * M_PI / n_fft;
        for (int k = 0; k < freq; k++)
            for (int n = 0; n < n_fft; n++) {
                cosb[(size_t) k * n_fft + n] = (float) std::cos(tp * k * n);
                sinb[(size_t) k * n_fft + n] = (float) std::sin(tp * k * n);
            }
    }

    // power spectrogram [freq][frames]
    std::vector<float> power((size_t) freq * frames);
    std::vector<float> win(n_fft);
    for (int n = 0; n < n_fft; n++) win[n] = m->mel_window[n];
    std::vector<float> xw(n_fft);
    for (int t = 0; t < frames; t++) {
        int base = t * hop;
        for (int n = 0; n < n_fft; n++) xw[n] = x[base + n] * win[n];
        for (int k = 0; k < freq; k++) {
            const float * cb = &cosb[(size_t) k * n_fft];
            const float * sb = &sinb[(size_t) k * n_fft];
            double        re = 0, im = 0;
            for (int n = 0; n < n_fft; n++) {
                re += xw[n] * cb[n];
                im -= xw[n] * sb[n];
            }
            power[(size_t) k * frames + t] = (float) (re * re + im * im);
        }
    }

    // mel = fb^T @ power : mel[mch][t] = sum_k power[k][t] * fb[k][mch]
    std::vector<float> mel((size_t) nm * frames);
    for (int c = 0; c < nm; c++)
        for (int t = 0; t < frames; t++) {
            double s = 0;
            for (int k = 0; k < freq; k++) s += (double) power[(size_t) k * frames + t] * m->mel_fb[(size_t) k * nm + c];
            mel[(size_t) c * frames + t] = (float) s;
        }

    // AmplitudeToDB(power): 10*log10(max(mel,amin)); then clamp to (max - top_db)
    const float amin = 1e-10f;
    float       gmax = -INFINITY;
    for (size_t i = 0; i < mel.size(); i++) {
        float v = 10.0f * std::log10(std::max(mel[i], amin));
        mel[i]  = v;
        if (v > gmax) gmax = v;
    }
    float floor_db = gmax - 80.0f;
    for (size_t i = 0; i < mel.size(); i++) {
        float v = mel[i] < floor_db ? floor_db : mel[i];
        mel[i]  = (v - m->mel_mean) / m->mel_std;
    }

    // drop last frame
    int T = frames - 1;
    mel_out.resize((size_t) nm * T);
    for (int c = 0; c < nm; c++)
        for (int t = 0; t < T; t++) mel_out[(size_t) c * T + t] = mel[(size_t) c * frames + t];
    *T_out = T;
}

// ---------------- ggml helpers ----------------
static struct ggml_tensor * sgmf_ln(struct ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * w,
                                    struct ggml_tensor * b, float eps) {
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, w);
    x = ggml_add(ctx, x, b);
    return x;
}

// batchnorm eval (folded affine) over channel dim. x:[C, T] (ne0=C). scale/shift:[C].
static struct ggml_tensor * sgmf_bn1d(struct ggml_context * ctx, struct ggml_tensor * x, SgMfBN * bn) {
    struct ggml_tensor * s = ggml_reshape_2d(ctx, bn->scale, bn->scale->ne[0], 1);
    struct ggml_tensor * h = ggml_reshape_2d(ctx, bn->shift, bn->shift->ne[0], 1);
    return ggml_add(ctx, ggml_mul(ctx, x, s), h);
}

static struct ggml_tensor * sgmf_swish(struct ggml_context * ctx, struct ggml_tensor * x) {
    return ggml_silu(ctx, x);  // x*sigmoid(x)
}

// batchnorm (folded affine) for conv2d feature maps. x:[W,H,C] (ne2=C). scale/shift:[C].
static struct ggml_tensor * sgmf_bn2d(struct ggml_context * ctx, struct ggml_tensor * x, SgMfBN * bn) {
    int64_t              C = bn->scale->ne[0];
    struct ggml_tensor * s = ggml_reshape_3d(ctx, bn->scale, 1, 1, C);
    struct ggml_tensor * h = ggml_reshape_3d(ctx, bn->shift, 1, 1, C);
    return ggml_add(ctx, ggml_mul(ctx, x, s), h);
}

// Res2dModule. x:[W=T,H=freq,C]. conv1 stride(2,2), conv2 stride 1; residual conv3 stride(2,2).
static struct ggml_tensor * sgmf_res2d(struct ggml_context * ctx, SgMfRes2d * R, struct ggml_tensor * x) {
    struct ggml_tensor * cb1 = ggml_reshape_3d(ctx, R->c1b, 1, 1, R->out_ch);
    struct ggml_tensor * cb2 = ggml_reshape_3d(ctx, R->c2b, 1, 1, R->out_ch);
    struct ggml_tensor * cb3 = ggml_reshape_3d(ctx, R->c3b, 1, 1, R->out_ch);

    // out = relu(bn1(conv1(x, s=2, p=1)))
    struct ggml_tensor * out = ggml_conv_2d(ctx, R->c1w, x, 2, 2, 1, 1, 1, 1);
    out                      = ggml_add(ctx, out, cb1);
    out                      = sgmf_bn2d(ctx, out, &R->bn1);
    out                      = ggml_relu(ctx, out);
    // out = bn2(conv2(out, s=1, p=1))
    out = ggml_conv_2d(ctx, R->c2w, out, 1, 1, 1, 1, 1, 1);
    out = ggml_add(ctx, out, cb2);
    out = sgmf_bn2d(ctx, out, &R->bn2);
    // residual = bn3(conv3(x, s=2, p=1))
    struct ggml_tensor * res = ggml_conv_2d(ctx, R->c3w, x, 2, 2, 1, 1, 1, 1);
    res                      = ggml_add(ctx, res, cb3);
    res                      = sgmf_bn2d(ctx, res, &R->bn3);
    out                      = ggml_add(ctx, res, out);
    out                      = ggml_relu(ctx, out);
    return out;
}

// One conformer macaron layer. h:[H, T] (ne0=H, ne1=T). positions:[T] i32.
static struct ggml_tensor * sgmf_conformer_layer(struct ggml_context * ctx, SonggenMusicFM * m, SgMfLayer * L,
                                                 struct ggml_tensor * h, struct ggml_tensor * positions) {
    const float eps   = m->ln_eps;
    const int   H     = m->hidden;
    const int   nh    = m->heads;
    const int   dh    = m->head_dim;
    const int   T     = (int) h->ne[1];
    const float scale = 1.0f / std::sqrt((float) dh);

    // FFN1 (macaron *0.5)
    {
        struct ggml_tensor * x = sgmf_ln(ctx, h, L->ffn1_ln_w, L->ffn1_ln_b, eps);
        x                      = ggml_mul_mat(ctx, L->ffn1_id_w, x);
        x                      = ggml_add(ctx, x, L->ffn1_id_b);
        x                      = sgmf_swish(ctx, x);
        x                      = ggml_mul_mat(ctx, L->ffn1_od_w, x);
        x                      = ggml_add(ctx, x, L->ffn1_od_b);
        h                      = ggml_add(ctx, h, ggml_scale(ctx, x, 0.5f));
    }

    // Self-attention. HF applies rotary to the LN'd hidden states BEFORE q/k projections;
    // value uses the UN-rotated hidden states. rotary = NEOX rotate-half over head_dim.
    {
        struct ggml_tensor * x = sgmf_ln(ctx, h, L->attn_ln_w, L->attn_ln_b, eps);  // [H,T]
        // rotate the hidden states viewed as [hd, nh, T]
        struct ggml_tensor * xr = ggml_reshape_3d(ctx, x, dh, nh, T);
        xr = ggml_rope_ext(ctx, xr, positions, NULL, dh, GGML_ROPE_TYPE_NEOX, 0, (float) m->rotary_base, 1.0f, 0.0f,
                           1.0f, 0.0f, 0.0f);
        struct ggml_tensor * qk = ggml_reshape_2d(ctx, ggml_cont(ctx, xr), H, T);  // rotated [H,T]

        struct ggml_tensor * q = ggml_add(ctx, ggml_mul_mat(ctx, L->q_w, qk), L->q_b);  // [H,T]
        struct ggml_tensor * k = ggml_add(ctx, ggml_mul_mat(ctx, L->k_w, qk), L->k_b);
        struct ggml_tensor * v = ggml_add(ctx, ggml_mul_mat(ctx, L->v_w, x), L->v_b);  // un-rotated

        q = ggml_reshape_3d(ctx, q, dh, nh, T);  // [dh, nh, T]
        k = ggml_reshape_3d(ctx, k, dh, nh, T);
        v = ggml_reshape_3d(ctx, v, dh, nh, T);

        q = ggml_permute(ctx, q, 0, 2, 1, 3);  // [dh, T, nh]
        k = ggml_permute(ctx, k, 0, 2, 1, 3);
        v = ggml_permute(ctx, v, 0, 2, 1, 3);

        struct ggml_tensor * scores = ggml_mul_mat(ctx, ggml_cont(ctx, k), ggml_cont(ctx, q));  // [T(k),T(q),nh]
        scores                      = ggml_soft_max_ext(ctx, scores, NULL, scale, 0.0f);
        struct ggml_tensor * vt     = ggml_cont(ctx, ggml_transpose(ctx, v));     // [T, dh, nh]
        struct ggml_tensor * o      = ggml_mul_mat(ctx, vt, scores);              // [dh, T(q), nh]
        o                           = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));  // [dh, nh, T]
        o                           = ggml_reshape_2d(ctx, o, H, T);
        o                           = ggml_add(ctx, ggml_mul_mat(ctx, L->o_w, o), L->o_b);
        h                           = ggml_add(ctx, h, o);
    }

    // Conv module
    {
        struct ggml_tensor * x = sgmf_ln(ctx, h, L->cm_ln_w, L->cm_ln_b, eps);  // [H,T]
        // pointwise_conv1: 1x1 conv H->2H == matmul. [2H,T]
        x = ggml_mul_mat(ctx, L->cm_pw1_w, x);
        // GLU dim=1 (channels): split 2H into a,b -> a*sigmoid(b). x:[2H,T]
        struct ggml_tensor * a = ggml_cont(ctx, ggml_view_2d(ctx, x, H, T, x->nb[1], 0));
        struct ggml_tensor * b = ggml_cont(ctx, ggml_view_2d(ctx, x, H, T, x->nb[1], (size_t) H * x->nb[0]));
        x                      = ggml_mul(ctx, a, ggml_sigmoid(ctx, b));  // [H,T]
        // depthwise conv k=31 pad=15 groups=H. x:[H,T] -> per-channel 1d conv along T.
        // ggml_conv_1d_dw expects data [T, C, N] and kernel [K,1,C]. Transpose x to [T,H].
        // depthwise conv k=31 pad=15 groups=H, F32 im2col (avoid F16 mul_mat split issue).
        struct ggml_tensor * xt   = ggml_cont(ctx, ggml_transpose(ctx, x));               // [T, H]
        xt                        = ggml_reshape_4d(ctx, xt, T, 1, H, 1);                 // [T,1,H,N]
        struct ggml_tensor * dw   = ggml_reshape_3d(ctx, L->cm_dw_w, m->depthwise_k, 1, H);  // [K,1,H]
        struct ggml_tensor * cols = ggml_im2col(ctx, dw, xt, 1, 0, (m->depthwise_k - 1) / 2, 0, 1, 0, false,
                                                GGML_TYPE_F32);  // [K, T, H, 1]
        struct ggml_tensor * wc   = ggml_mul(ctx, cols, dw);          // broadcast [K,1,H] -> [K,T,H]
        struct ggml_tensor * yc   = ggml_sum_rows(ctx, wc);          // [1,T,H]
        yc                        = ggml_reshape_2d(ctx, yc, T, H);  // [T,H]
        x                         = ggml_cont(ctx, ggml_transpose(ctx, yc));  // [H,T]
        x                       = sgmf_bn1d(ctx, x, &L->cm_bn);
        x                       = sgmf_swish(ctx, x);
        x                       = ggml_mul_mat(ctx, L->cm_pw2_w, x);  // [H,T]
        h                       = ggml_add(ctx, h, x);
    }

    // FFN2 (macaron *0.5)
    {
        struct ggml_tensor * x = sgmf_ln(ctx, h, L->ffn2_ln_w, L->ffn2_ln_b, eps);
        x                      = ggml_mul_mat(ctx, L->ffn2_id_w, x);
        x                      = ggml_add(ctx, x, L->ffn2_id_b);
        x                      = sgmf_swish(ctx, x);
        x                      = ggml_mul_mat(ctx, L->ffn2_od_w, x);
        x                      = ggml_add(ctx, x, L->ffn2_od_b);
        h                      = ggml_add(ctx, h, ggml_scale(ctx, x, 0.5f));
    }

    // final LN
    h = sgmf_ln(ctx, h, L->final_ln_w, L->final_ln_b, eps);
    return h;
}

// Build the full graph. mel:[n_mels, T_mel] input tensor (ne0=n_mels? no: ne0=T_mel? see encode).
// We set up conv stem then conformer. Returns vector of 13 layer_result tensors [H, T].
struct SgMfGraph {
    struct ggml_tensor * results[13];
    int                  n;
};

// audio[N] mono 24k -> writes layer_results. Each result [H, T_frames] -> caller permutes to [H,T].
static void sgmf_encode(SonggenMusicFM * m, const float * audio, int N, std::vector<std::vector<float>> & out_layers,
                        int * T_out) {
    std::vector<float> mel;
    int                T_mel = 0;
    sgmf_mel(m, audio, N, mel, &T_mel);  // [n_mels, T_mel] channel-major
    int nm = m->n_mels;

    size_t                  ctx_size = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(16384, false);
    uint8_t *               gbuf     = (uint8_t *) malloc(ctx_size);
    struct ggml_init_params gp       = { ctx_size, gbuf, true };
    struct ggml_context *   ctx      = ggml_init(gp);

    // mel input: torch [B,1,freq=128(H),time=T_mel(W)]. ggml data layout [W=T_mel, H=128, C=1].
    struct ggml_tensor * inp = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T_mel, nm, 1);  // [W,H,C]
    ggml_set_name(inp, "mel_input");
    ggml_set_input(inp);

    // conv stem
    struct ggml_tensor * x = sgmf_res2d(ctx, &m->res[0], inp);  // [W/2, H/2, 512]
    x                      = sgmf_res2d(ctx, &m->res[1], x);    // [W/4, H/4, 512]
    int Tt                 = (int) x->ne[0];                               // time frames
    int Ff                 = (int) x->ne[1];                               // freq bands = 32
    int Cc                 = (int) x->ne[2];                               // 512

    // rearrange b c f t -> b t (c f): feature vector for time t indexed by (c*Ff + f).
    // x src axes: 0=t, 1=f, 2=c. Want dst [f, c, t] so reshape gives feat=(c*Ff+f) inner.
    // ggml_permute(x, p0,p1,p2,p3): src axis i moves to position p_i. t->2, f->0, c->1.
    struct ggml_tensor * xr = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3));  // [f, c, t]
    xr                      = ggml_reshape_2d(ctx, xr, (int64_t) Cc * Ff, Tt);    // [(c*Ff+f), t] = [16384, t]
    struct ggml_tensor * h  = ggml_add(ctx, ggml_mul_mat(ctx, m->lin_w, xr), m->lin_b);  // [1024, t]

    int T = Tt;

    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    std::vector<struct ggml_tensor *> results;
    results.push_back(h);  // layer_results[0] = conv stem output (pre layer0)
    for (int li = 0; li < m->n_layers; li++) {
        h = sgmf_conformer_layer(ctx, m, &m->layer[li], h, positions);
        results.push_back(h);
    }
    // final encoder LN applied to the LAST hidden, replacing results[n_layers]
    struct ggml_tensor * final_h = sgmf_ln(ctx, results.back(), m->enc_ln_w, m->enc_ln_b, m->ln_eps);
    results.back()               = final_h;

    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16384, false);
    for (auto * r : results) {
        ggml_set_output(r);
        ggml_build_forward_expand(graph, r);
    }

    if (!ggml_backend_sched_alloc_graph(m->sched, graph)) {
        fprintf(stderr, "[MUSICFM] FATAL: graph alloc failed (T_mel=%d)\n", T_mel);
        exit(1);
    }

    ggml_backend_tensor_set(inp, mel.data(), 0, mel.size() * sizeof(float));
    std::vector<int32_t> pos(T);
    for (int i = 0; i < T; i++) pos[i] = i;
    ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(int32_t));

    ggml_backend_sched_graph_compute(m->sched, graph);

    out_layers.resize(results.size());
    for (size_t i = 0; i < results.size(); i++) {
        out_layers[i].resize((size_t) m->hidden * T);
        ggml_backend_tensor_get(results[i], out_layers[i].data(), 0, out_layers[i].size() * sizeof(float));
    }
    *T_out = T;

    fprintf(stderr, "[MUSICFM] graph %d nodes, N=%d -> T_mel=%d -> T=%d, %zu layer_results\n",
            ggml_graph_n_nodes(graph), N, T_mel, T, results.size());

    ggml_backend_sched_reset(m->sched);
    ggml_free(ctx);
    free(gbuf);
}

static void sgmf_free(SonggenMusicFM * m) {
    if (m->sched) ggml_backend_sched_free(m->sched);
    backend_release(m->backend, m->cpu_backend);
    wctx_free(&m->wctx);
    *m = {};
}
