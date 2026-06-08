// sa3-attn.h: shared building blocks for the Stable Audio 3 transformer (DiT + SAME AE).
//
// Differential attention (Diff-Transformer, plain subtract, shared V):
//   out = softmax(qk^T*s)·v  -  softmax(qd kd^T*s)·v
// Self-attn weight to_qkv -> 5 chunks (q,k,v,q_diff,k_diff); cross-attn to_q->2 (q,q_diff),
// to_kv->3 (k,k_diff,v). qk-norm applied identically to base+diff; RoPE (partial, NEOX) on self only.
//
// Norm helpers: RMS (DiT, gamma only) and DynamicTanh (SAME AE: tanh(alpha*x)*gamma+beta).
#pragma once
#include "ggml.h"
#include <cmath>
#include <cstdlib>

// ggml_flash_attn_ext (fused QK^T/softmax/·V, O(S) memory) replaces the materialized-scores
// differential attention. Default-ON for GPU (faster + far less VRAM, near-lossless), like
// dit.h's use_flash_attn=has_gpu. OFF for CPU — flash accumulates in F16 there and drifts, so
// the materialized path stays the F32 validation oracle. SA3_FLASH=0/1 overrides. The model
// load sets this flag from its backend (see sa3_set_flash).
static inline bool & sa3_flash_flag() { static bool f = false; return f; }
static inline void sa3_set_flash(bool has_gpu) {
    const char * e = std::getenv("SA3_FLASH");
    sa3_flash_flag() = e ? (std::atoi(e) != 0) : has_gpu;
}
static inline bool sa3_flash_enabled() { return sa3_flash_flag(); }

// One flash attention: q [D,Tq,Nh], k/v [D,Tk,Nh] -> [D,Nh,Tq] (flash layout). mask F16 or NULL.
// K/V cast to F16 (the fast CUDA FA path); F32 accumulation via set_prec.
static inline struct ggml_tensor * sa3_flash_once(struct ggml_context * c, struct ggml_tensor * q,
                                                  struct ggml_tensor * k, struct ggml_tensor * v,
                                                  struct ggml_tensor * mask, float scale) {
    if (k->type != GGML_TYPE_F16) k = ggml_cast(c, k, GGML_TYPE_F16);
    if (v->type != GGML_TYPE_F16) v = ggml_cast(c, v, GGML_TYPE_F16);
    struct ggml_tensor * o = ggml_flash_attn_ext(c, q, k, v, mask, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
    return o;  // [D, Nh, Tq]
}

static inline struct ggml_tensor * sa3_f32(struct ggml_context * c, struct ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 ? t : ggml_cast(c, t, GGML_TYPE_F32);
}

// RMSNorm * gamma  (DiT). x: [C, ...]
static inline struct ggml_tensor * sa3_rms(struct ggml_context * c, struct ggml_tensor * x,
                                           struct ggml_tensor * g, float eps) {
    return ggml_mul(c, ggml_rms_norm(c, x, eps), sa3_f32(c, g));
}

// DynamicTanh: tanh(alpha * x) * gamma + beta   (SAME AE). alpha [1], gamma/beta [C].
static inline struct ggml_tensor * sa3_dyt(struct ggml_context * c, struct ggml_tensor * x,
                                           struct ggml_tensor * alpha, struct ggml_tensor * gamma,
                                           struct ggml_tensor * beta) {
    // alpha is a learned scalar tensor; multiply elementwise (broadcast)
    struct ggml_tensor * t = ggml_tanh(c, ggml_mul(c, x, sa3_f32(c, alpha)));
    return ggml_add(c, ggml_mul(c, t, sa3_f32(c, gamma)), sa3_f32(c, beta));
}

// One attention: q,k:[D,Tq/Tk,Nh] v:[D,Tk,Nh] -> out [D,Tq,Nh]. mask optional additive.
static inline struct ggml_tensor * sa3_attn_once(struct ggml_context * c, struct ggml_tensor * q,
                                                 struct ggml_tensor * k, struct ggml_tensor * v,
                                                 struct ggml_tensor * mask, float scale) {
    struct ggml_tensor * scores = ggml_mul_mat(c, k, q);                 // [Tk, Tq, Nh]
    scores = ggml_soft_max_ext(c, scores, mask, scale, 0.0f);
    struct ggml_tensor * vt  = ggml_cont(c, ggml_transpose(c, v));       // [Tk, D, Nh]
    return ggml_mul_mat(c, vt, scores);                                  // [D, Tq, Nh]
}

// Reshape a projection [E, T] (E = Nh*D) into [D, T, Nh] with per-head rms qk-norm and optional partial RoPE.
// norm_g: [D] rms gamma (or NULL). If positions != NULL, apply NEOX rope over rope_ndims with rope_theta.
static inline struct ggml_tensor * sa3_heads(struct ggml_context * c, struct ggml_tensor * proj,
                                             int D, int Nh, int T, struct ggml_tensor * norm_g, float eps,
                                             struct ggml_tensor * positions, int rope_ndims, float rope_theta) {
    struct ggml_tensor * h = ggml_reshape_3d(c, proj, D, Nh, T);  // [D, Nh, T]
    if (norm_g) h = ggml_mul(c, ggml_rms_norm(c, h, eps), sa3_f32(c, norm_g));
    if (positions) {
        h = ggml_rope_ext(c, h, positions, NULL, rope_ndims, GGML_ROPE_TYPE_NEOX, 0,
                          rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    }
    return ggml_permute(c, h, 0, 2, 1, 3);  // [D, T, Nh]
}

// Like sa3_heads but with DynamicTanh qk-norm (SAME AE): tanh(alpha*h)*gamma+beta over head_dim.
static inline struct ggml_tensor * sa3_heads_dyt(struct ggml_context * c, struct ggml_tensor * proj,
                                                 int D, int Nh, int T,
                                                 struct ggml_tensor * a, struct ggml_tensor * g, struct ggml_tensor * b,
                                                 struct ggml_tensor * positions, int rope_ndims, float rope_theta) {
    struct ggml_tensor * h = ggml_reshape_3d(c, proj, D, Nh, T);  // [D, Nh, T]
    if (a) h = ggml_add(c, ggml_mul(c, ggml_tanh(c, ggml_mul(c, h, sa3_f32(c, a))), sa3_f32(c, g)), sa3_f32(c, b));
    if (positions) {
        h = ggml_rope_ext(c, h, positions, NULL, rope_ndims, GGML_ROPE_TYPE_NEOX, 0,
                          rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    }
    return ggml_permute(c, h, 0, 2, 1, 3);  // [D, T, Nh]
}

// Differential attention out [E=Nh*D, Tq] given base/diff q,k and shared v (each [D,T,Nh]).
static inline struct ggml_tensor * sa3_diff_attn(struct ggml_context * c,
                                                 struct ggml_tensor * q, struct ggml_tensor * qd,
                                                 struct ggml_tensor * k, struct ggml_tensor * kd,
                                                 struct ggml_tensor * v, struct ggml_tensor * mask,
                                                 int D, int Nh, int Tq, float scale) {
    if (sa3_flash_enabled()) {
        // flash needs a F16 mask; cast once and share V's F16 across both calls.
        struct ggml_tensor * mf = (mask && mask->type != GGML_TYPE_F16) ? ggml_cast(c, mask, GGML_TYPE_F16) : mask;
        struct ggml_tensor * vf = v->type == GGML_TYPE_F16 ? v : ggml_cast(c, v, GGML_TYPE_F16);
        struct ggml_tensor * a1 = sa3_flash_once(c, q,  k,  vf, mf, scale);  // [D,Nh,Tq]
        struct ggml_tensor * a2 = sa3_flash_once(c, qd, kd, vf, mf, scale);  // [D,Nh,Tq]
        struct ggml_tensor * o  = ggml_sub(c, a1, a2);                       // [D,Nh,Tq]
        return ggml_reshape_2d(c, ggml_cont(c, o), D * Nh, Tq);              // [E, Tq]
    }
    struct ggml_tensor * a1 = sa3_attn_once(c, q, k, v, mask, scale);     // [D,Tq,Nh]
    struct ggml_tensor * a2 = sa3_attn_once(c, qd, kd, v, mask, scale);   // [D,Tq,Nh]
    struct ggml_tensor * o  = ggml_sub(c, a1, a2);                        // [D,Tq,Nh]
    o = ggml_cont(c, ggml_permute(c, o, 0, 2, 1, 3));                     // [D,Nh,Tq]
    return ggml_reshape_2d(c, o, D * Nh, Tq);                             // [E, Tq]
}

// View a contiguous slice [n, T] out of a packed projection [E, T] at row offset `off`.
static inline struct ggml_tensor * sa3_chunk(struct ggml_context * c, struct ggml_tensor * proj, int n, int T, int off) {
    return ggml_cont(c, ggml_view_2d(c, proj, n, T, proj->nb[1], (size_t) off * proj->nb[0]));
}
