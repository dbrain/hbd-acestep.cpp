// songgen-vae.h : SongGeneration Stable-Audio Oobleck VAE decoder (GGML)
//
//   conv_in(64->2048,k7,p3) -> 5x DecoderBlock -> snake_out -> conv_out(128->2,k7,p3)
//   DecoderBlock: snake -> ConvTranspose1d(stride s, k=2s, pad=ceil(s/2))
//                 -> 3x ResidualUnit(dil 1,3,9)
//   ResidualUnit: snake -> conv(k7,dil,pad=3*dil) -> snake -> conv(k1) + residual
//   Snake (SnakeBeta, alpha_logscale): y = x + (1/exp(b)) * sin(exp(a)*x)^2
//   weight_norm folded at convert time; snake exp() pre-applied at convert time.
//   strides[10,6,4,4,2] -> 1920x upsample, 48kHz stereo.
//
// ConvTranspose1d is done as GEMM(W[IC,K*OC] . xT[IC,T]) -> col2im_1d, matching the
// proven acestep VAE path (ggml_conv_transpose_1d's naive kernel is slower/F16-lossy).
#pragma once

#include "backend.h"
#include "gguf-weights.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

struct SgVaeResUnit {
    struct ggml_tensor *s1a, *s1bi;  // snake1 alpha, beta_inv [C]
    struct ggml_tensor *c1w, *c1b;   // conv1 [7,C,C], bias [C]
    struct ggml_tensor *s2a, *s2bi;  // snake2
    struct ggml_tensor *c2w, *c2b;   // conv2 [1,C,C], bias [C]
    int                 dilation;
};

struct SgVaeBlock {
    struct ggml_tensor *sa, *sbi;    // pre-upsample snake [in_ch]
    struct ggml_tensor *ctw, *ctb;   // conv_t GEMM weight [IC, K*OC], bias [OC]
    int                 in_ch, out_ch, stride, kernel;
    SgVaeResUnit        ru[3];
};

struct SonggenVae {
    int n_blocks;

    struct ggml_tensor * cin_w, *cin_b;   // conv_in [7,64,2048], bias [2048]
    SgVaeBlock           blk[8];
    struct ggml_tensor * sa, *sbi;        // final snake [128]
    struct ggml_tensor * cout_w;          // conv_out [7,128,2], no bias

    WeightCtx            wctx;
    ggml_backend_t       backend;
    ggml_backend_t       cpu_backend;
    ggml_backend_sched_t sched;
};

// ---- custom loaders (data already folded at convert time) ----

// Load a tensor as F32 (snake gains, biases).
static struct ggml_tensor * sgvae_f32(WeightCtx * w, const GGUFModel & gf, const std::string & name) {
    return gf_load_tensor_f32(w, gf, name);
}

// Load ConvTranspose1d weight (torch [in,out,k]) into GEMM layout dst[IC, K*OC] with
// column index oc*K+k (oc-major, matching ggml_col2im_1d), dst ne[0]=IC.
static struct ggml_tensor * sgvae_conv_t(WeightCtx * w, const GGUFModel & gf, const std::string & name,
                                         int in_ch, int out_ch, int kernel) {
    struct ggml_tensor * src = ggml_get_tensor(gf.meta, name.c_str());
    if (!src) {
        fprintf(stderr, "[VAE] FATAL: conv_t tensor '%s' not found\n", name.c_str());
        exit(1);
    }
    const uint16_t * v   = (const uint16_t *) gf_get_data(gf, name.c_str());  // bf16, [in][out][k]
    int              KOC = kernel * out_ch;
    auto             buf = std::make_unique<float[]>((size_t) in_ch * KOC);
    float *          d   = buf.get();
    for (int ic = 0; ic < in_ch; ic++) {
        for (int oc = 0; oc < out_ch; oc++) {
            for (int k = 0; k < kernel; k++) {
                float val            = ggml_bf16_to_fp32(*(const ggml_bf16_t *) &v[((size_t) ic * out_ch + oc) * kernel + k]);
                int   koc            = oc * kernel + k;          // col2im column ordering: oc-major
                d[(size_t) koc * in_ch + ic] = val;              // dst ne[0]=IC
            }
        }
    }
    int64_t              ne[2]  = { in_ch, KOC };
    struct ggml_tensor * tensor = ggml_new_tensor(w->ctx, GGML_TYPE_F32, 2, ne);
    ggml_set_name(tensor, name.c_str());
    w->pending.push_back({ tensor, d, (size_t) in_ch * KOC * sizeof(float), 0 });
    w->staging.push_back(std::move(buf));
    return tensor;
}

static bool sgvae_load(SonggenVae * m, const char * gguf_path) {
    *m = {};

    BackendPair bp = backend_init("VAE");
    m->backend     = bp.backend;
    m->cpu_backend = bp.cpu_backend;
    m->sched       = backend_sched_new(bp, 8192);

    GGUFModel gf;
    if (!gf_load(&gf, gguf_path)) {
        return false;
    }

    m->n_blocks         = (int) gf_get_u32(gf, "songgen-vae.n_blocks");
    int latent_dim      = (int) gf_get_u32(gf, "songgen-vae.latent_dim");
    int out_channels    = (int) gf_get_u32(gf, "songgen-vae.out_channels");
    int ds_ratio        = (int) gf_get_u32(gf, "songgen-vae.downsampling_ratio");
    if (m->n_blocks <= 0) m->n_blocks = 5;

    static const int strides[] = { 10, 6, 4, 4, 2 };
    static const int in_ch[]   = { 2048, 1024, 512, 256, 128 };
    static const int out_ch[]  = { 1024, 512, 256, 128, 128 };
    static const int dils[]    = { 1, 3, 9 };

    fprintf(stderr, "[VAE] blocks=%d latent=%d out_ch=%d upsample=%dx\n", m->n_blocks, latent_dim, out_channels,
            ds_ratio);

    wctx_init(&m->wctx, 200);

    m->cin_w = sgvae_f32(&m->wctx, gf, "conv_in.weight");
    m->cin_b = sgvae_f32(&m->wctx, gf, "conv_in.bias");

    for (int i = 0; i < m->n_blocks; i++) {
        SgVaeBlock & b = m->blk[i];
        b.in_ch        = in_ch[i];
        b.out_ch       = out_ch[i];
        b.stride       = strides[i];
        b.kernel       = strides[i] * 2;
        std::string bpfx = "block." + std::to_string(i);
        b.sa           = sgvae_f32(&m->wctx, gf, bpfx + ".snake.alpha");
        b.sbi          = sgvae_f32(&m->wctx, gf, bpfx + ".snake.beta_inv");
        b.ctw          = sgvae_conv_t(&m->wctx, gf, bpfx + ".conv_t.weight", b.in_ch, b.out_ch, b.kernel);
        b.ctb          = sgvae_f32(&m->wctx, gf, bpfx + ".conv_t.bias");
        for (int r = 0; r < 3; r++) {
            SgVaeResUnit & ru = b.ru[r];
            ru.dilation       = dils[r];
            std::string rp    = bpfx + ".res." + std::to_string(r);
            ru.s1a            = sgvae_f32(&m->wctx, gf, rp + ".snake1.alpha");
            ru.s1bi           = sgvae_f32(&m->wctx, gf, rp + ".snake1.beta_inv");
            ru.c1w            = sgvae_f32(&m->wctx, gf, rp + ".conv1.weight");
            ru.c1b            = sgvae_f32(&m->wctx, gf, rp + ".conv1.bias");
            ru.s2a            = sgvae_f32(&m->wctx, gf, rp + ".snake2.alpha");
            ru.s2bi           = sgvae_f32(&m->wctx, gf, rp + ".snake2.beta_inv");
            ru.c2w            = sgvae_f32(&m->wctx, gf, rp + ".conv2.weight");
            ru.c2b            = sgvae_f32(&m->wctx, gf, rp + ".conv2.bias");
        }
    }

    m->sa     = sgvae_f32(&m->wctx, gf, "snake_out.alpha");
    m->sbi    = sgvae_f32(&m->wctx, gf, "snake_out.beta_inv");
    m->cout_w = sgvae_f32(&m->wctx, gf, "conv_out.weight");

    if (!wctx_alloc(&m->wctx, m->backend)) {
        gf_close(&gf);
        return false;
    }
    gf_close(&gf);
    return true;
}

// ---- graph helpers ----

// Snake: y = x + sin(alpha*x)^2 * beta_inv.  x:[T,C]  alpha/beta_inv:[C] (broadcast over T)
static struct ggml_tensor * sgvae_snake(struct ggml_context * ctx, struct ggml_tensor * x,
                                        struct ggml_tensor * alpha, struct ggml_tensor * beta_inv) {
    struct ggml_tensor * a2 = ggml_reshape_2d(ctx, alpha, 1, alpha->ne[0]);
    struct ggml_tensor * b2 = ggml_reshape_2d(ctx, beta_inv, 1, beta_inv->ne[0]);
    struct ggml_tensor * s  = ggml_sin(ctx, ggml_mul(ctx, x, a2));
    struct ggml_tensor * d  = ggml_mul(ctx, ggml_sqr(ctx, s), b2);
    return ggml_add(ctx, x, d);
}

// Conv1d + optional bias.  w:[K,IC,OC] F32  x:[T,IC] -> [T_out,OC]
// im2col forced to F32 dst (ggml_conv_1d hardcodes F16) so the whole conv stays fp32.
static struct ggml_tensor * sgvae_conv1d(struct ggml_context * ctx, struct ggml_tensor * w, struct ggml_tensor * b,
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

// ConvTranspose1d via GEMM + col2im.  w:[IC,K*OC]  x:[T_in,IC] -> [T_out,OC]
static struct ggml_tensor * sgvae_conv_t1d(struct ggml_context * ctx, struct ggml_tensor * w, struct ggml_tensor * b,
                                           struct ggml_tensor * x, int stride, int pad, int oc) {
    struct ggml_tensor * xt  = ggml_cont(ctx, ggml_transpose(ctx, x));  // [IC, T_in]
    struct ggml_tensor * col = ggml_mul_mat(ctx, w, xt);                // [K*OC, T_in]
    struct ggml_tensor * y   = ggml_col2im_1d(ctx, col, stride, oc, pad);
    if (b) {
        y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
    }
    return y;
}

static struct ggml_tensor * sgvae_res_unit(struct ggml_context * ctx, SgVaeResUnit * ru, struct ggml_tensor * x) {
    struct ggml_tensor * skip = x;
    int                  pad  = 3 * ru->dilation;  // (7-1)*dil/2
    x                         = sgvae_snake(ctx, x, ru->s1a, ru->s1bi);
    x                         = sgvae_conv1d(ctx, ru->c1w, ru->c1b, x, 1, pad, ru->dilation);
    x                         = sgvae_snake(ctx, x, ru->s2a, ru->s2bi);
    x                         = sgvae_conv1d(ctx, ru->c2w, ru->c2b, x, 1, 0, 1);
    return ggml_add(ctx, skip, x);
}

// latent[T,64] -> audio[T*1920, 2]
static struct ggml_tensor * sgvae_build(struct ggml_context * ctx, SonggenVae * m, struct ggml_tensor * latent) {
    struct ggml_tensor * x = sgvae_conv1d(ctx, m->cin_w, m->cin_b, latent, 1, 3, 1);
    for (int i = 0; i < m->n_blocks; i++) {
        SgVaeBlock & b   = m->blk[i];
        x                = sgvae_snake(ctx, x, b.sa, b.sbi);
        int          pad = (b.stride + 1) / 2;  // ceil(stride/2)
        x                = sgvae_conv_t1d(ctx, b.ctw, b.ctb, x, b.stride, pad, b.out_ch);
        for (int r = 0; r < 3; r++) {
            x = sgvae_res_unit(ctx, &b.ru[r], x);
        }
    }
    x = sgvae_snake(ctx, x, m->sa, m->sbi);
    x = sgvae_conv1d(ctx, m->cout_w, NULL, x, 1, 3, 1);
    return x;  // [T_audio, 2]
}

// latent: [64, T] channel-major (npy [1,64,T]) -> audio_out [2, T*1920] (ch0 then ch1).
// Returns T_audio.
static int sgvae_decode(SonggenVae * m, const float * latent, int T_latent, std::vector<float> & audio_out) {
    size_t                  ctx_size = ggml_tensor_overhead() * 2048 + ggml_graph_overhead_custom(8192, false);
    uint8_t *               gbuf     = (uint8_t *) malloc(ctx_size);
    struct ggml_init_params gp       = { ctx_size, gbuf, true };
    struct ggml_context *   ctx      = ggml_init(gp);

    struct ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_latent, 64);  // [T, C]
    ggml_set_name(inp, "vae_input");
    ggml_set_input(inp);

    struct ggml_tensor * out = sgvae_build(ctx, m, inp);
    ggml_set_output(out);

    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(graph, out);

    if (!ggml_backend_sched_alloc_graph(m->sched, graph)) {
        fprintf(stderr, "[VAE] FATAL: graph alloc failed (T=%d)\n", T_latent);
        exit(1);
    }

    // input npy is [64, T] channel-major; ggml inp is [T, C] (ne0=T). Transpose.
    std::vector<float> tin((size_t) 64 * T_latent);
    for (int c = 0; c < 64; c++) {
        for (int t = 0; t < T_latent; t++) {
            tin[(size_t) c * T_latent + t] = latent[(size_t) c * T_latent + t];
        }
    }
    ggml_backend_tensor_set(inp, tin.data(), 0, tin.size() * sizeof(float));

    ggml_backend_sched_graph_compute(m->sched, graph);

    int T_audio = (int) out->ne[0];
    audio_out.resize((size_t) 2 * T_audio);
    // out is [T_audio, 2] (ne0=T_audio, ne1=2) -> read both channels contiguous.
    ggml_backend_tensor_get(out, audio_out.data(), 0, (size_t) 2 * T_audio * sizeof(float));

    fprintf(stderr, "[VAE] graph %d nodes, T_latent=%d -> T_audio=%d\n", ggml_graph_n_nodes(graph), T_latent, T_audio);

    ggml_backend_sched_reset(m->sched);
    ggml_free(ctx);
    free(gbuf);
    return T_audio;
}

static void sgvae_free(SonggenVae * m) {
    if (m->sched) ggml_backend_sched_free(m->sched);
    backend_release(m->backend, m->cpu_backend);
    wctx_free(&m->wctx);
    *m = {};
}
