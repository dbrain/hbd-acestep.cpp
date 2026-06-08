// sa3-t5gemma-enc.h: T5Gemma-b-b-ul2 text ENCODER (Gemma2-style) via ggml.
//
// Stable Audio 3 text conditioner. Encoder-only, bidirectional. Output 768-dim
// last_hidden_state, fed (after learned-padding) as DiT cross-attn context.
//
// Gemma2 specifics vs a plain transformer (all load-bearing, validated vs golden):
//   - RMSNorm applied as x_normed * (1 + w)  [Gemma adds 1 to the norm weight]
//   - Sandwich norms: residual + post_norm(sublayer(pre_norm(x)))  (4 norms/layer)
//   - MLP = down( gelu_tanh(gate) * up )   (GeGLU, gelu_pytorch_tanh)
//   - Token embeddings scaled by sqrt(hidden_size)
//   - Bidirectional; key-padding mask zeroes attention to pad tokens
//   - Optional attn logit softcapping (config 50.0; transformers SDPA path disables it -> default OFF)
#pragma once
#include "backend.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#define SA3_T5G_MAX_LAYERS 16

struct SA3T5GConfig {
    int   hidden_size   = 768;
    int   intermediate  = 2048;
    int   n_heads       = 12;
    int   head_dim      = 64;
    int   n_layers      = 12;
    float rope_theta    = 10000.0f;
    float rms_eps       = 1e-6f;
    float query_scale   = 0.125f;   // query_pre_attn_scalar^-0.5 = 64^-0.5
    float embed_scale   = 0.0f;     // sqrt(hidden_size), set at load
    float attn_softcap  = 0.0f;     // 0 = disabled (SDPA path); 50.0 to enable
};

struct SA3T5GLayer {
    struct ggml_tensor * pre_attn_norm;
    struct ggml_tensor * post_attn_norm;
    struct ggml_tensor * pre_ffn_norm;
    struct ggml_tensor * post_ffn_norm;
    struct ggml_tensor * q, *k, *v, *o;
    struct ggml_tensor * gate, *up, *down;
};

struct SA3T5GModel {
    SA3T5GConfig cfg;
    SA3T5GLayer  layers[SA3_T5G_MAX_LAYERS];
    struct ggml_tensor * tok_embd;
    struct ggml_tensor * output_norm;

    ggml_backend_t       backend;
    ggml_backend_t       cpu_backend;
    ggml_backend_sched_t sched;
    WeightCtx            wctx;
};

static struct ggml_tensor * sa3t5g_f32(struct ggml_context * ctx, struct ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 ? t : ggml_cast(ctx, t, GGML_TYPE_F32);
}

// Gemma RMSNorm: rms(x) * (1 + w) = rms(x)*w + rms(x)
static struct ggml_tensor * sa3t5g_norm(struct ggml_context * ctx, struct ggml_tensor * x,
                                        struct ggml_tensor * w, float eps) {
    struct ggml_tensor * n = ggml_rms_norm(ctx, x, eps);
    return ggml_add(ctx, ggml_mul(ctx, n, sa3t5g_f32(ctx, w)), n);
}

// Manual bidirectional attention with optional logit softcapping + key-padding mask.
// q,k,v: [D, S, Nh]; mask: [S, S] f32 additive (0/-inf) or NULL.
static struct ggml_tensor * sa3t5g_attn(struct ggml_context * ctx, struct ggml_tensor * q,
                                        struct ggml_tensor * k, struct ggml_tensor * v,
                                        struct ggml_tensor * mask, float scale, float softcap) {
    struct ggml_tensor * scores = ggml_mul_mat(ctx, k, q);  // [S_k, S_q, Nh]
    if (softcap > 0.0f) {
        scores = ggml_scale(ctx, scores, scale);
        scores = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, scores, 1.0f / softcap)), softcap);
        scores = ggml_soft_max_ext(ctx, scores, mask, 1.0f, 0.0f);
    } else {
        scores = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);
    }
    struct ggml_tensor * vt  = ggml_cont(ctx, ggml_transpose(ctx, v));  // [S, D, Nh]
    struct ggml_tensor * out = ggml_mul_mat(ctx, vt, scores);           // [D, S_q, Nh]
    return out;
}

static struct ggml_tensor * sa3t5g_build_layer(struct ggml_context * ctx, const SA3T5GConfig & c,
                                               SA3T5GLayer * ly, struct ggml_tensor * h,
                                               struct ggml_tensor * positions, struct ggml_tensor * mask, int S) {
    int D = c.head_dim, Nh = c.n_heads;

    // ---- self-attention (sandwich) ----
    struct ggml_tensor * x = sa3t5g_norm(ctx, h, ly->pre_attn_norm, c.rms_eps);
    struct ggml_tensor * q = ggml_mul_mat(ctx, ly->q, x);
    struct ggml_tensor * k = ggml_mul_mat(ctx, ly->k, x);
    struct ggml_tensor * v = ggml_mul_mat(ctx, ly->v, x);
    q = ggml_reshape_3d(ctx, q, D, Nh, S);
    k = ggml_reshape_3d(ctx, k, D, Nh, S);
    v = ggml_reshape_3d(ctx, v, D, Nh, S);
    q = ggml_rope_ext(ctx, q, positions, NULL, D, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, positions, NULL, D, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    q = ggml_permute(ctx, q, 0, 2, 1, 3);  // [D, S, Nh]
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);
    struct ggml_tensor * attn = sa3t5g_attn(ctx, q, k, v, mask, c.query_scale, c.attn_softcap);  // [D, S, Nh]
    attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));  // [D, Nh, S]
    attn = ggml_reshape_2d(ctx, attn, Nh * D, S);
    attn = ggml_mul_mat(ctx, ly->o, attn);
    attn = sa3t5g_norm(ctx, attn, ly->post_attn_norm, c.rms_eps);
    h = ggml_add(ctx, h, attn);

    // ---- feedforward (sandwich, GeGLU) ----
    x = sa3t5g_norm(ctx, h, ly->pre_ffn_norm, c.rms_eps);
    struct ggml_tensor * g  = ggml_gelu(ctx, ggml_mul_mat(ctx, ly->gate, x));  // gelu_tanh
    struct ggml_tensor * u  = ggml_mul_mat(ctx, ly->up, x);
    struct ggml_tensor * ff = ggml_mul_mat(ctx, ly->down, ggml_mul(ctx, g, u));
    ff = sa3t5g_norm(ctx, ff, ly->post_ffn_norm, c.rms_eps);
    h = ggml_add(ctx, h, ff);
    return h;
}

static bool sa3t5g_load(SA3T5GModel * m, const char * gguf_path) {
    BackendPair bp = backend_init("SA3-T5Gemma");
    m->backend = bp.backend; m->cpu_backend = bp.cpu_backend;
    m->sched = backend_sched_new(bp, 4096);

    m->cfg = {};
    m->cfg.embed_scale = sqrtf((float) m->cfg.hidden_size);
    if (const char * e = std::getenv("SA3_T5G_SOFTCAP")) m->cfg.attn_softcap = (float) atof(e);

    GGUFModel gf;
    if (!gf_load(&gf, gguf_path)) return false;

    int n_tensors = 2 + m->cfg.n_layers * 11;
    wctx_init(&m->wctx, n_tensors);
    m->tok_embd    = gf_load_tensor(&m->wctx, gf, "tok_embd");
    m->output_norm = gf_load_tensor_f32(&m->wctx, gf, "output_norm");
    for (int i = 0; i < m->cfg.n_layers; i++) {
        char p[32]; snprintf(p, sizeof(p), "blk.%d.", i);
        SA3T5GLayer * ly = &m->layers[i];
        ly->pre_attn_norm  = gf_load_tensor_f32(&m->wctx, gf, std::string(p) + "attn_norm");
        ly->post_attn_norm = gf_load_tensor_f32(&m->wctx, gf, std::string(p) + "post_attn_norm");
        ly->pre_ffn_norm   = gf_load_tensor_f32(&m->wctx, gf, std::string(p) + "ffn_norm");
        ly->post_ffn_norm  = gf_load_tensor_f32(&m->wctx, gf, std::string(p) + "post_ffn_norm");
        ly->q = gf_load_tensor(&m->wctx, gf, std::string(p) + "attn_q");
        ly->k = gf_load_tensor(&m->wctx, gf, std::string(p) + "attn_k");
        ly->v = gf_load_tensor(&m->wctx, gf, std::string(p) + "attn_v");
        ly->o = gf_load_tensor(&m->wctx, gf, std::string(p) + "attn_o");
        ly->gate = gf_load_tensor(&m->wctx, gf, std::string(p) + "ffn_gate");
        ly->up   = gf_load_tensor(&m->wctx, gf, std::string(p) + "ffn_up");
        ly->down = gf_load_tensor(&m->wctx, gf, std::string(p) + "ffn_down");
    }
    if (!wctx_alloc(&m->wctx, m->backend)) { gf_close(&gf); return false; }
    gf_close(&gf);
    fprintf(stderr, "[SA3-T5Gemma] loaded %dL H=%d softcap=%.1f\n", m->cfg.n_layers, m->cfg.hidden_size, m->cfg.attn_softcap);
    return true;
}

// token_ids[S], valid[S] (1=real token, 0=pad). output: [hidden*S] f32 ggml layout.
static void sa3t5g_encode(SA3T5GModel * m, const int * token_ids, const int * valid, int S, float * output) {
    const SA3T5GConfig & c = m->cfg;
    int H = c.hidden_size;

    size_t ctx_size = 4096 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8192, false);
    struct ggml_init_params gp = { ctx_size, NULL, true };
    struct ggml_context * ctx = ggml_init(gp);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

    struct ggml_tensor * t_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    ggml_set_input(t_ids);
    struct ggml_tensor * h = ggml_get_rows(ctx, m->tok_embd, t_ids);  // [H,S]
    h = ggml_scale(ctx, h, c.embed_scale);

    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    ggml_set_input(positions);

    // key-padding mask [S_k, S_q]: 0 if key valid else -inf (same for all queries)
    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S, S);
    ggml_set_input(mask);

    for (int i = 0; i < c.n_layers; i++) {
        h = sa3t5g_build_layer(ctx, c, &m->layers[i], h, positions, mask, S);
    }
    struct ggml_tensor * out = sa3t5g_norm(ctx, h, m->output_norm, c.rms_eps);
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) {
        fprintf(stderr, "[SA3-T5Gemma] FATAL: alloc graph (%d tok)\n", S); exit(1);
    }
    ggml_backend_tensor_set(t_ids, token_ids, 0, S * sizeof(int));
    std::vector<int> pos(S);
    for (int i = 0; i < S; i++) pos[i] = i;
    ggml_backend_tensor_set(positions, pos.data(), 0, S * sizeof(int));
    std::vector<float> md((size_t) S * S);
    for (int qi = 0; qi < S; qi++)
        for (int kj = 0; kj < S; kj++)
            md[(size_t) qi * S + kj] = (valid[kj] != 0) ? 0.0f : -INFINITY;
    ggml_backend_tensor_set(mask, md.data(), 0, md.size() * sizeof(float));

    ggml_backend_sched_graph_compute(m->sched, gf);
    ggml_backend_tensor_get(out, output, 0, (size_t) H * S * sizeof(float));
    ggml_backend_sched_reset(m->sched);
    ggml_free(ctx);
}

static void sa3t5g_free(SA3T5GModel * m) {
    if (m->sched) ggml_backend_sched_free(m->sched);
    backend_release(m->backend, m->cpu_backend);
    wctx_free(&m->wctx);
    *m = {};
}
