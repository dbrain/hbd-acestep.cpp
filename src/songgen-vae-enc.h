// songgen-vae-enc.h : SongGeneration Stable-Audio Oobleck VAE ENCODER (GGML).
//
// Mirror of the ported decoder (songgen-vae.h), structure reversed:
//   conv_in(2->128,k7,p3) -> 5x EncoderBlock -> snake_out -> conv_out(2048->128,k3,p1)
//   EncoderBlock: 3x ResidualUnit(dil 1,3,9) -> snake -> strided WNConv1d downsample
//   ResidualUnit: snake -> conv(k7,dil,pad=3*dil) -> snake -> conv(k1) + residual
//   strided conv: k=2*stride, pad=ceil(stride/2), strides[2,4,4,6,10] -> 1920x down.
//   Bottleneck(vae): mean,scale = chunk(2) over channels (64 each); deterministic latent = mean.
//   Snake (SnakeBeta, alpha_logscale) + weight_norm folded at convert time.
#pragma once

#include "backend.h"
#include "gguf-weights.h"
#include "songgen-cache.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

struct SgVeResUnit {
    struct ggml_tensor *s1a, *s1bi;
    struct ggml_tensor *c1w, *c1b;
    struct ggml_tensor *s2a, *s2bi;
    struct ggml_tensor *c2w, *c2b;
    int                 dilation;
};

struct SgVeBlock {
    SgVeResUnit         ru[3];
    struct ggml_tensor *sa, *sbi;    // pre-downsample snake [in_ch]
    struct ggml_tensor *cdw, *cdb;   // strided conv weight [k,in,out], bias [out]
    int                 in_ch, out_ch, stride, kernel;
};

struct SonggenVaeEnc {
    int n_blocks;
    int latent_dim;

    struct ggml_tensor *cin_w, *cin_b;   // conv_in [7,2,128], bias [128]
    SgVeBlock           blk[8];
    struct ggml_tensor *sa, *sbi;        // final snake [2048]
    struct ggml_tensor *cout_w, *cout_b; // conv_out [3,2048,128], bias [128]

    WeightCtx            wctx;
    ggml_backend_t       backend;
    ggml_backend_t       cpu_backend;
    ggml_backend_sched_t sched;
};

// Conv weight (torch [out,in,k] bf16) -> ggml [k,in,out] f32 for sgve_conv1d.
static struct ggml_tensor * sgve_conv(WeightCtx * w, const GGUFModel & gf, const std::string & name, int out_ch,
                                      int in_ch, int kernel) {
    struct ggml_tensor * src = ggml_get_tensor(gf.meta, name.c_str());
    if (!src) {
        fprintf(stderr, "[VAE-ENC] FATAL: conv tensor '%s' not found\n", name.c_str());
        exit(1);
    }
    const ggml_bf16_t * v   = (const ggml_bf16_t *) gf_get_data(gf, name.c_str());  // [out][in][k]
    auto                buf = std::make_unique<float[]>((size_t) kernel * in_ch * out_ch);
    float *             d   = buf.get();
    for (int oc = 0; oc < out_ch; oc++) {
        for (int ic = 0; ic < in_ch; ic++) {
            for (int k = 0; k < kernel; k++) {
                float val = ggml_bf16_to_fp32(v[((size_t) oc * in_ch + ic) * kernel + k]);
                // ggml conv weight layout: ne0=k, ne1=in, ne2=out
                d[((size_t) oc * in_ch + ic) * kernel + k] = val;
            }
        }
    }
    int64_t              ne[3]  = { kernel, in_ch, out_ch };
    struct ggml_tensor * tensor = ggml_new_tensor(w->ctx, GGML_TYPE_F32, 3, ne);
    ggml_set_name(tensor, name.c_str());
    w->pending.push_back({ tensor, d, (size_t) kernel * in_ch * out_ch * sizeof(float), 0 });
    w->staging.push_back(std::move(buf));
    return tensor;
}

static struct ggml_tensor * sgve_f32(WeightCtx * w, const GGUFModel & gf, const std::string & name) {
    return gf_load_tensor_f32(w, gf, name);
}

// Load a snake gain ([C]) as [1, C] so the snake graph needs no in-chain reshape
// (a reshape node breaks the MUL,SIN,SQR,MUL,ADD snake autofusion). See songgen-vae.h.
static struct ggml_tensor * sgve_snake_gain(WeightCtx * w, const GGUFModel & gf, const std::string & name) {
    struct ggml_tensor * t = gf_load_tensor_f32(w, gf, name);
    GGML_ASSERT(ggml_n_dims(t) == 1 && ggml_is_contiguous(t));
    t->ne[1] = t->ne[0];   // [C] -> [1, C]
    t->ne[0] = 1;
    t->nb[1] = t->nb[0];
    return t;
}

static bool sgve_load_real(SonggenVaeEnc * m, const char * gguf_path) {
    *m = {};

    BackendPair bp = backend_init("VAE-ENC");
    m->backend     = bp.backend;
    m->cpu_backend = bp.cpu_backend;
    m->sched       = backend_sched_new(bp, 8192);

    GGUFModel gf;
    if (!gf_load(&gf, gguf_path)) {
        return false;
    }

    m->n_blocks   = (int) gf_get_u32(gf, "songgen-vae-enc.n_blocks");
    m->latent_dim = (int) gf_get_u32(gf, "songgen-vae-enc.latent_dim");
    if (m->n_blocks <= 0) m->n_blocks = 5;
    if (m->latent_dim <= 0) m->latent_dim = 64;

    static const int strides[] = { 2, 4, 4, 6, 10 };
    static const int in_ch[]   = { 128, 128, 256, 512, 1024 };
    static const int out_ch[]  = { 128, 256, 512, 1024, 2048 };
    static const int dils[]    = { 1, 3, 9 };

    fprintf(stderr, "[VAE-ENC] blocks=%d latent=%d\n", m->n_blocks, m->latent_dim);

    wctx_init(&m->wctx, 200);

    m->cin_w = sgve_conv(&m->wctx, gf, "conv_in.weight", 128, 2, 7);
    m->cin_b = sgve_f32(&m->wctx, gf, "conv_in.bias");

    for (int i = 0; i < m->n_blocks; i++) {
        SgVeBlock & b = m->blk[i];
        b.in_ch       = in_ch[i];
        b.out_ch      = out_ch[i];
        b.stride      = strides[i];
        b.kernel      = strides[i] * 2;
        std::string bpfx = "block." + std::to_string(i);
        for (int r = 0; r < 3; r++) {
            SgVeResUnit & ru = b.ru[r];
            ru.dilation      = dils[r];
            std::string rp   = bpfx + ".res." + std::to_string(r);
            ru.s1a           = sgve_snake_gain(&m->wctx, gf, rp + ".snake1.alpha");
            ru.s1bi          = sgve_snake_gain(&m->wctx, gf, rp + ".snake1.beta_inv");
            ru.c1w           = sgve_conv(&m->wctx, gf, rp + ".conv1.weight", b.in_ch, b.in_ch, 7);
            ru.c1b           = sgve_f32(&m->wctx, gf, rp + ".conv1.bias");
            ru.s2a           = sgve_snake_gain(&m->wctx, gf, rp + ".snake2.alpha");
            ru.s2bi          = sgve_snake_gain(&m->wctx, gf, rp + ".snake2.beta_inv");
            ru.c2w           = sgve_conv(&m->wctx, gf, rp + ".conv2.weight", b.in_ch, b.in_ch, 1);
            ru.c2b           = sgve_f32(&m->wctx, gf, rp + ".conv2.bias");
        }
        b.sa  = sgve_snake_gain(&m->wctx, gf, bpfx + ".snake.alpha");
        b.sbi = sgve_snake_gain(&m->wctx, gf, bpfx + ".snake.beta_inv");
        b.cdw = sgve_conv(&m->wctx, gf, bpfx + ".conv_d.weight", b.out_ch, b.in_ch, b.kernel);
        b.cdb = sgve_f32(&m->wctx, gf, bpfx + ".conv_d.bias");
    }

    m->sa     = sgve_snake_gain(&m->wctx, gf, "snake_out.alpha");
    m->sbi    = sgve_snake_gain(&m->wctx, gf, "snake_out.beta_inv");
    m->cout_w = sgve_conv(&m->wctx, gf, "conv_out.weight", 2 * m->latent_dim, 2048, 3);
    m->cout_b = sgve_f32(&m->wctx, gf, "conv_out.bias");

    if (!wctx_alloc(&m->wctx, m->backend)) {
        gf_close(&gf);
        return false;
    }
    gf_close(&gf);
    return true;
}

// Snake: y = x + sin(alpha*x)^2 * beta_inv.  x:[T,C]  alpha/beta_inv:[C].
static struct ggml_tensor * sgve_snake(struct ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * alpha,
                                       struct ggml_tensor * beta_inv) {
    // alpha/beta_inv loaded as [1,C] (sgve_snake_gain) -> chain is MUL,SIN,SQR,MUL,ADD
    // with no interleaved reshape, so ggml-cuda snake autofusion fires.
    struct ggml_tensor * s = ggml_sin(ctx, ggml_mul(ctx, x, alpha));
    struct ggml_tensor * d = ggml_mul(ctx, ggml_sqr(ctx, s), beta_inv);
    return ggml_add(ctx, x, d);
}

// Conv1d + optional bias.  w:[K,IC,OC] F32  x:[T,IC] -> [T_out,OC].
static struct ggml_tensor * sgve_conv1d(struct ggml_context * ctx, struct ggml_tensor * w, struct ggml_tensor * b,
                                        struct ggml_tensor * x, int stride, int pad, int dilation) {
    struct ggml_tensor * im2col = ggml_im2col(ctx, w, x, stride, 0, pad, 0, dilation, 0, false, GGML_TYPE_F32);
    struct ggml_tensor * y      = ggml_mul_mat(ctx,
                                         ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
                                         ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]));
    y = ggml_reshape_2d(ctx, y, im2col->ne[1], w->ne[2]);  // [T_out, OC]
    if (b) {
        y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
    }
    return y;
}

static struct ggml_tensor * sgve_res_unit(struct ggml_context * ctx, SgVeResUnit * ru, struct ggml_tensor * x) {
    struct ggml_tensor * skip = x;
    int                  pad  = 3 * ru->dilation;  // (7-1)*dil/2
    x                         = sgve_snake(ctx, x, ru->s1a, ru->s1bi);
    x                         = sgve_conv1d(ctx, ru->c1w, ru->c1b, x, 1, pad, ru->dilation);
    x                         = sgve_snake(ctx, x, ru->s2a, ru->s2bi);
    x                         = sgve_conv1d(ctx, ru->c2w, ru->c2b, x, 1, 0, 1);
    return ggml_add(ctx, skip, x);
}

// audio[T,2] -> pre_bottleneck[T/1920, 128]
static struct ggml_tensor * sgve_build(struct ggml_context * ctx, SonggenVaeEnc * m, struct ggml_tensor * audio) {
    struct ggml_tensor * x = sgve_conv1d(ctx, m->cin_w, m->cin_b, audio, 1, 3, 1);
    for (int i = 0; i < m->n_blocks; i++) {
        SgVeBlock & b = m->blk[i];
        for (int r = 0; r < 3; r++) {
            x = sgve_res_unit(ctx, &b.ru[r], x);
        }
        x       = sgve_snake(ctx, x, b.sa, b.sbi);
        int pad = (b.stride + 1) / 2;  // ceil(stride/2)
        x       = sgve_conv1d(ctx, b.cdw, b.cdb, x, b.stride, pad, 1);
    }
    x = sgve_snake(ctx, x, m->sa, m->sbi);
    x = sgve_conv1d(ctx, m->cout_w, m->cout_b, x, 1, 1, 1);  // k3 pad1
    return x;  // [T_out, 128]
}

// audio: [2, T] channel-major (npy [1,2,T]) -> mean[64, T/1920] channel-major.
// Also returns scale[64, T/1920] if scale_out != nullptr. Returns T_latent.
static int sgve_encode(SonggenVaeEnc * m, const float * audio, int T_audio, std::vector<float> & mean_out,
                       std::vector<float> * scale_out = nullptr) {
    size_t                  ctx_size = ggml_tensor_overhead() * 2048 + ggml_graph_overhead_custom(8192, false);
    uint8_t *               gbuf     = (uint8_t *) malloc(ctx_size);
    struct ggml_init_params gp       = { ctx_size, gbuf, true };
    struct ggml_context *   ctx      = ggml_init(gp);

    struct ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_audio, 2);  // [T, C]
    ggml_set_name(inp, "vae_enc_input");
    ggml_set_input(inp);

    struct ggml_tensor * out = sgve_build(ctx, m, inp);  // [T_lat, 128]
    ggml_set_output(out);

    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(graph, out);

    if (!ggml_backend_sched_alloc_graph(m->sched, graph)) {
        fprintf(stderr, "[VAE-ENC] FATAL: graph alloc failed (T=%d)\n", T_audio);
        exit(1);
    }

    // npy is [2, T] channel-major; ggml inp is [T, C] -> already same memory layout (ch0 then ch1).
    ggml_backend_tensor_set(inp, audio, 0, (size_t) 2 * T_audio * sizeof(float));

    ggml_backend_sched_graph_compute(m->sched, graph);

    int T_lat = (int) out->ne[0];
    int latd  = m->latent_dim;
    // out is [T_lat, 128] (ne0=T_lat, ne1=128). First 64 channels = mean, next 64 = scale.
    std::vector<float> pre((size_t) 128 * T_lat);
    ggml_backend_tensor_get(out, pre.data(), 0, pre.size() * sizeof(float));

    mean_out.resize((size_t) latd * T_lat);
    for (int c = 0; c < latd; c++) {
        for (int t = 0; t < T_lat; t++) {
            mean_out[(size_t) c * T_lat + t] = pre[(size_t) c * T_lat + t];
        }
    }
    if (scale_out) {
        scale_out->resize((size_t) latd * T_lat);
        for (int c = 0; c < latd; c++) {
            for (int t = 0; t < T_lat; t++) {
                (*scale_out)[(size_t) c * T_lat + t] = pre[(size_t) (c + latd) * T_lat + t];
            }
        }
    }

    fprintf(stderr, "[VAE-ENC] graph %d nodes, T_audio=%d -> T_latent=%d\n", ggml_graph_n_nodes(graph), T_audio, T_lat);

    ggml_backend_sched_reset(m->sched);
    ggml_free(ctx);
    free(gbuf);
    return T_lat;
}

static bool sgve_load(SonggenVaeEnc * m, const char * gguf_path) {
    return sgcache_load(gguf_path, m, sgve_load_real);
}

static void sgve_free(SonggenVaeEnc * m) {
    if (m->sched) ggml_backend_sched_free(m->sched);
    backend_release(m->backend, m->cpu_backend);
    wctx_free(&m->wctx);
    *m = {};
}
