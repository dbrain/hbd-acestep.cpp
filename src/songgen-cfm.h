// songgen-cfm.h : SongGeneration CFM reflow ESTIMATOR (custom GPT2) graph (GGML)
//
// Custom time-conditioned NON-causal GPT2 (PixArt-style adaLN-single), NOT vanilla
// HF GPT2. Source: models_gpt.models.gpt2_rope2_time_new_correct_mask_noncasual_reflow.
//
// Forward(inputs_embeds[H,T] per batch row, time_step float):
//   h = inputs_embeds + wpe[0:T]                              (position embeddings)
//   embedded_timestep = TimestepEmbedder(sinusoid(ts))       (512 -> SiLU -> 2200)
//   time6 = adaln.linear(SiLU(embedded_timestep))            ([6*H], per block)
//   per block i (16x):
//     (shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp)
//         = block.scale_shift_table[6,H] + time6.reshape(6,H)
//     a = ln_1(h)*(1+scale_msa)+shift_msa
//     a = c_proj( noncausal_attn_rope(c_attn(a)) )           (RoPE2 interleaved, head_dim=110)
//     h = h + gate_msa * a
//     m = ln_2(h)*(1+scale_mlp)+shift_mlp
//     m = c_proj( gelu_new( c_fc(m) ) )                       (gelu tanh approx)
//     h = h + gate_mlp * m
//   (shift,scale) = scale_shift_table[2,H] + embedded_timestep
//   h = proj_out( ln_f(h)*(1+scale)+shift )
//
// GPT2 Conv1D weights were TRANSPOSED at convert time ([in,out]->[out,in]) so the
// ggml graph mul_mat's cleanly. LayerNorm (with bias, eps 1e-5) — NOT RMSNorm.
// Attention scale = 1/sqrt(head_dim). c_attn output split q|k|v (each H) along dim0.
#pragma once

#include "backend.h"
#include "gguf-weights.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define SGCFM_MAX_LAYERS 32

struct SgCfmConfig {
    int   n_layer;
    int   n_head;
    int   n_embd;
    int   n_inner;
    int   n_positions;
    int   head_dim;
    int   flow_t_size;
    float ln_eps;
    float rope_theta;
};

struct SgCfmBlock {
    struct ggml_tensor *ln1_w, *ln1_b;
    struct ggml_tensor *ln2_w, *ln2_b;
    struct ggml_tensor *c_attn_w, *c_attn_b;  // [H, 3H] (mul_mat) / [3H]
    struct ggml_tensor *c_proj_w, *c_proj_b;  // [H, H]
    struct ggml_tensor *c_fc_w, *c_fc_b;      // [H, FFN]
    struct ggml_tensor *mlp_proj_w, *mlp_proj_b;  // [FFN, H]
    struct ggml_tensor *sst;                  // [H, 6] (ggml) per-block scale_shift_table
};

struct SonggenCfm {
    SgCfmConfig cfg;

    struct ggml_tensor * wpe;          // [H, n_positions]
    struct ggml_tensor * te_l1_w;      // [flow_t, H]
    struct ggml_tensor * te_l1_b;
    struct ggml_tensor * te_l2_w;      // [H, H]
    struct ggml_tensor * te_l2_b;
    struct ggml_tensor * adaln_lin_w;  // [H, 6H]
    struct ggml_tensor * adaln_lin_b;  // [6H]
    struct ggml_tensor * sst;          // [H, 2] global scale_shift_table
    struct ggml_tensor * ln_f_w, *ln_f_b;
    struct ggml_tensor * proj_out_w, *proj_out_b;

    SgCfmBlock blk[SGCFM_MAX_LAYERS];

    WeightCtx            wctx;
    ggml_backend_t       backend;
    ggml_backend_t       cpu_backend;
    ggml_backend_sched_t sched;
};

// ---- load ----

static bool sgcfm_load(SonggenCfm * m, const char * gguf_path) {
    *m = {};

    BackendPair bp = backend_init("CFM");
    m->backend     = bp.backend;
    m->cpu_backend = bp.cpu_backend;
    m->sched       = backend_sched_new(bp, 8192);

    GGUFModel gf;
    if (!gf_load(&gf, gguf_path)) {
        return false;
    }

    SgCfmConfig c   = {};
    c.n_layer       = (int) gf_get_u32(gf, "songgen-cfm.n_layer");
    c.n_head        = (int) gf_get_u32(gf, "songgen-cfm.n_head");
    c.n_embd        = (int) gf_get_u32(gf, "songgen-cfm.n_embd");
    c.n_inner       = (int) gf_get_u32(gf, "songgen-cfm.n_inner");
    c.n_positions   = (int) gf_get_u32(gf, "songgen-cfm.n_positions");
    c.head_dim      = (int) gf_get_u32(gf, "songgen-cfm.head_dim");
    c.flow_t_size   = (int) gf_get_u32(gf, "songgen-cfm.flow_t_size");
    c.ln_eps        = gf_get_f32(gf, "songgen-cfm.layer_norm_eps");
    c.rope_theta    = gf_get_f32(gf, "songgen-cfm.rope_theta");
    if (c.ln_eps <= 0.0f) c.ln_eps = 1e-5f;
    if (c.rope_theta <= 0.0f) c.rope_theta = 10000.0f;
    m->cfg = c;

    fprintf(stderr, "[CFM] L=%d heads=%d H=%d FFN=%d hd=%d flow_t=%d eps=%.1e theta=%.0f\n", c.n_layer, c.n_head,
            c.n_embd, c.n_inner, c.head_dim, c.flow_t_size, c.ln_eps, c.rope_theta);

    if (c.n_layer > SGCFM_MAX_LAYERS) {
        fprintf(stderr, "[CFM] FATAL: too many layers\n");
        gf_close(&gf);
        return false;
    }

    int n_tensors = c.n_layer * 13 + 16;
    wctx_init(&m->wctx, n_tensors);

    m->wpe         = gf_load_tensor(&m->wctx, gf, "wpe.weight");
    m->te_l1_w     = gf_load_tensor(&m->wctx, gf, "adaln.te.l1.weight");
    m->te_l1_b     = gf_load_tensor_f32(&m->wctx, gf, "adaln.te.l1.bias");
    m->te_l2_w     = gf_load_tensor(&m->wctx, gf, "adaln.te.l2.weight");
    m->te_l2_b     = gf_load_tensor_f32(&m->wctx, gf, "adaln.te.l2.bias");
    m->adaln_lin_w = gf_load_tensor(&m->wctx, gf, "adaln.linear.weight");
    m->adaln_lin_b = gf_load_tensor_f32(&m->wctx, gf, "adaln.linear.bias");
    m->sst         = gf_load_tensor_f32(&m->wctx, gf, "scale_shift_table");
    m->ln_f_w      = gf_load_tensor_f32(&m->wctx, gf, "ln_f.weight");
    m->ln_f_b      = gf_load_tensor_f32(&m->wctx, gf, "ln_f.bias");
    m->proj_out_w  = gf_load_tensor(&m->wctx, gf, "proj_out.weight");
    m->proj_out_b  = gf_load_tensor_f32(&m->wctx, gf, "proj_out.bias");

    for (int i = 0; i < c.n_layer; i++) {
        SgCfmBlock * b = &m->blk[i];
        std::string  p = "blk." + std::to_string(i) + ".";
        b->ln1_w       = gf_load_tensor_f32(&m->wctx, gf, p + "ln_1.weight");
        b->ln1_b       = gf_load_tensor_f32(&m->wctx, gf, p + "ln_1.bias");
        b->ln2_w       = gf_load_tensor_f32(&m->wctx, gf, p + "ln_2.weight");
        b->ln2_b       = gf_load_tensor_f32(&m->wctx, gf, p + "ln_2.bias");
        b->c_attn_w    = gf_load_tensor(&m->wctx, gf, p + "attn.c_attn.weight");
        b->c_attn_b    = gf_load_tensor_f32(&m->wctx, gf, p + "attn.c_attn.bias");
        b->c_proj_w    = gf_load_tensor(&m->wctx, gf, p + "attn.c_proj.weight");
        b->c_proj_b    = gf_load_tensor_f32(&m->wctx, gf, p + "attn.c_proj.bias");
        b->c_fc_w      = gf_load_tensor(&m->wctx, gf, p + "mlp.c_fc.weight");
        b->c_fc_b      = gf_load_tensor_f32(&m->wctx, gf, p + "mlp.c_fc.bias");
        b->mlp_proj_w  = gf_load_tensor(&m->wctx, gf, p + "mlp.c_proj.weight");
        b->mlp_proj_b  = gf_load_tensor_f32(&m->wctx, gf, p + "mlp.c_proj.bias");
        b->sst         = gf_load_tensor_f32(&m->wctx, gf, p + "scale_shift_table");
    }

    if (!wctx_alloc(&m->wctx, m->backend)) {
        gf_close(&gf);
        return false;
    }
    gf_close(&gf);
    return true;
}

// ---- graph helpers ----

static struct ggml_tensor * sgcfm_layernorm(struct ggml_context * ctx,
                                            struct ggml_tensor *  x,
                                            struct ggml_tensor *  w,
                                            struct ggml_tensor *  b,
                                            float                 eps) {
    struct ggml_tensor * n = ggml_norm(ctx, x, eps);  // mean/var over ne0
    n                      = ggml_mul(ctx, n, w);
    n                      = ggml_add(ctx, n, b);
    return n;
}

// modulate: x*(1+scale)+shift, scale/shift are [H] broadcast over T columns.
static struct ggml_tensor * sgcfm_modulate(struct ggml_context * ctx,
                                           struct ggml_tensor *  x,      // [H, T]
                                           struct ggml_tensor *  scale,  // [H]
                                           struct ggml_tensor *  shift)  // [H]
{
    // x*(1+scale)+shift = x + x*scale + shift  (avoids needing a literal tensor)
    struct ggml_tensor * xs = ggml_mul(ctx, x, scale);
    return ggml_add(ctx, ggml_add(ctx, x, xs), shift);
}

// Non-causal multi-head self-attention with RoPE2 (interleaved / GPT-J style).
// x:[H,T] -> [H,T]. No mask (golden mask is all-ones).
static struct ggml_tensor * sgcfm_attn(struct ggml_context * ctx,
                                       const SgCfmConfig &   c,
                                       SgCfmBlock *          b,
                                       struct ggml_tensor *  x,
                                       struct ggml_tensor *  positions,
                                       int                   T) {
    int H  = c.n_embd;
    int D  = c.head_dim;
    int Nh = c.n_head;

    struct ggml_tensor * qkv = ggml_mul_mat(ctx, b->c_attn_w, x);  // [3H, T]
    qkv                      = ggml_add(ctx, qkv, b->c_attn_b);

    struct ggml_tensor * q = ggml_cont(ctx, ggml_view_2d(ctx, qkv, H, T, qkv->nb[1], 0));
    struct ggml_tensor * k = ggml_cont(ctx, ggml_view_2d(ctx, qkv, H, T, qkv->nb[1], (size_t) H * qkv->nb[0]));
    struct ggml_tensor * v = ggml_cont(ctx, ggml_view_2d(ctx, qkv, H, T, qkv->nb[1], (size_t) 2 * H * qkv->nb[0]));

    q = ggml_reshape_3d(ctx, q, D, Nh, T);
    k = ggml_reshape_3d(ctx, k, D, Nh, T);
    v = ggml_reshape_3d(ctx, v, D, Nh, T);

    // RoPE2: interleaved complex pairs (q0,q1),(q2,q3),... -> ggml mode 0 (NORMAL).
    q = ggml_rope_ext(ctx, q, positions, NULL, D, GGML_ROPE_TYPE_NORMAL, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, positions, NULL, D, GGML_ROPE_TYPE_NORMAL, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    q = ggml_permute(ctx, q, 0, 2, 1, 3);  // [D, T, Nh]
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    float                scale  = 1.0f / sqrtf((float) D);
    struct ggml_tensor * scores = ggml_mul_mat(ctx, ggml_cont(ctx, k), ggml_cont(ctx, q));  // [T(k), T(q), Nh]
    scores                      = ggml_soft_max_ext(ctx, scores, NULL, scale, 0.0f);
    struct ggml_tensor * vt     = ggml_cont(ctx, ggml_transpose(ctx, v));  // [T, D, Nh]
    struct ggml_tensor * out    = ggml_mul_mat(ctx, vt, scores);            // [D, T(q), Nh]
    out                         = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));  // [D, Nh, T]
    out                         = ggml_reshape_2d(ctx, out, H, T);

    out = ggml_mul_mat(ctx, b->c_proj_w, out);
    out = ggml_add(ctx, out, b->c_proj_b);
    return out;
}

// ---- forward (batch processed row-by-row; B rows share the time embedding) ----
// inputs_embeds: [B*T*H] flat in numpy order (row-major: b, t, h), f32.
// time_step: scalar (same for all rows in golden). out: [B*T*H] same order.
static void sgcfm_forward(SonggenCfm * m, const float * inputs_embeds, int B, int T, float time_step, float * out) {
    const SgCfmConfig & c   = m->cfg;
    int                 H   = c.n_embd;
    int                 FT  = c.flow_t_size;

    size_t                  ctx_size = (size_t) 8192 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8192, false);
    struct ggml_init_params gp       = { ctx_size, NULL, true };
    struct ggml_context *   ctx      = ggml_init(gp);
    struct ggml_cgraph *    gf       = ggml_new_graph_custom(ctx, 8192, false);

    // ---- inputs ----
    // sinusoidal timestep embedding (precomputed on host): cat([cos(args),sin(args)]), len FT.
    struct ggml_tensor * t_sin = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, FT);
    ggml_set_name(t_sin, "t_sin");
    ggml_set_input(t_sin);

    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    struct ggml_tensor * emb_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H, T, B);  // [H,T,B]
    ggml_set_name(emb_in, "inputs_embeds");
    ggml_set_input(emb_in);

    // ---- time embedding: TimestepEmbedder(sinusoid) ----
    struct ggml_tensor * et = ggml_mul_mat(ctx, m->te_l1_w, t_sin);  // [H]
    et                      = ggml_add(ctx, et, m->te_l1_b);
    et                      = ggml_silu(ctx, et);
    et                      = ggml_mul_mat(ctx, m->te_l2_w, et);     // [H] embedded_timestep
    et                      = ggml_add(ctx, et, m->te_l2_b);

    // adaln_single.linear(SiLU(embedded_timestep)) -> [6H]
    struct ggml_tensor * time6 = ggml_mul_mat(ctx, m->adaln_lin_w, ggml_silu(ctx, et));  // [6H]
    time6                      = ggml_add(ctx, time6, m->adaln_lin_b);
    time6                      = ggml_reshape_2d(ctx, time6, H, 6);  // [H,6] columns: shiftmsa,scalemsa,gatemsa,shiftmlp,scalemlp,gatemlp

    // per-block modulation = block.sst[H,6] + time6[H,6]
    // final modulation = sst[H,2] + et (broadcast et over both columns)
    struct ggml_tensor * et2 = ggml_reshape_2d(ctx, et, H, 1);

    // wpe[0:T] : [H, T]
    struct ggml_tensor * pos_emb =
        ggml_cont(ctx, ggml_view_2d(ctx, m->wpe, H, T, m->wpe->nb[1], 0));

    auto col = [&](struct ggml_tensor * t, int j) {
        return ggml_view_1d(ctx, t, H, (size_t) j * t->nb[1]);
    };

    std::vector<struct ggml_tensor *> outs(B);
    for (int bi = 0; bi < B; bi++) {
        struct ggml_tensor * h =
            ggml_cont(ctx, ggml_view_2d(ctx, emb_in, H, T, emb_in->nb[1], (size_t) bi * emb_in->nb[2]));  // [H,T]
        h = ggml_add(ctx, h, pos_emb);

        for (int i = 0; i < c.n_layer; i++) {
            SgCfmBlock *         b   = &m->blk[i];
            struct ggml_tensor * mod = ggml_add(ctx, b->sst, time6);  // [H,6]
            struct ggml_tensor * shift_msa = col(mod, 0), *scale_msa = col(mod, 1), *gate_msa = col(mod, 2);
            struct ggml_tensor * shift_mlp = col(mod, 3), *scale_mlp = col(mod, 4), *gate_mlp = col(mod, 5);

            struct ggml_tensor * a = sgcfm_layernorm(ctx, h, b->ln1_w, b->ln1_b, c.ln_eps);
            a                      = sgcfm_modulate(ctx, a, scale_msa, shift_msa);
            a                      = sgcfm_attn(ctx, c, b, a, positions, T);
            a                      = ggml_mul(ctx, a, gate_msa);
            h                      = ggml_add(ctx, h, a);

            struct ggml_tensor * mm = sgcfm_layernorm(ctx, h, b->ln2_w, b->ln2_b, c.ln_eps);
            mm                      = sgcfm_modulate(ctx, mm, scale_mlp, shift_mlp);
            mm                      = ggml_mul_mat(ctx, b->c_fc_w, mm);  // [FFN,T]
            mm                      = ggml_add(ctx, mm, b->c_fc_b);
            mm                      = ggml_gelu(ctx, mm);  // gelu_new (tanh approx)
            mm                      = ggml_mul_mat(ctx, b->mlp_proj_w, mm);  // [H,T]
            mm                      = ggml_add(ctx, mm, b->mlp_proj_b);
            mm                      = ggml_mul(ctx, mm, gate_mlp);
            h                       = ggml_add(ctx, h, mm);
        }

        // final: (shift,scale) = sst[H,2] + et ; ln_f then modulate then proj_out
        struct ggml_tensor * fmod  = ggml_add(ctx, m->sst, et2);  // [H,2] (et broadcast)
        struct ggml_tensor * fshift = col(fmod, 0);
        struct ggml_tensor * fscale = col(fmod, 1);
        h                          = sgcfm_layernorm(ctx, h, m->ln_f_w, m->ln_f_b, c.ln_eps);
        h                          = sgcfm_modulate(ctx, h, fscale, fshift);
        h                          = ggml_mul_mat(ctx, m->proj_out_w, h);
        h                          = ggml_add(ctx, h, m->proj_out_b);  // [H,T]
        outs[bi]                   = h;
    }

    // stack outs along a 3rd dim -> [H,T,B]
    struct ggml_tensor * stacked = outs[0];
    for (int bi = 1; bi < B; bi++) {
        stacked = ggml_concat(ctx, stacked, outs[bi], 2);
    }
    ggml_set_output(stacked);
    ggml_build_forward_expand(gf, stacked);

    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) {
        fprintf(stderr, "[CFM] FATAL: alloc graph failed\n");
        exit(1);
    }

    // ---- set inputs ----
    {
        // sinusoid: half=FT/2, freqs=exp(-log(1e4)*arange(half)/half), args=ts*freqs*1000
        int                half = FT / 2;
        std::vector<float> sv(FT);
        for (int j = 0; j < half; j++) {
            float freq = expf(-logf(10000.0f) * (float) j / (float) half);
            float arg  = time_step * freq * 1000.0f;
            sv[j]      = cosf(arg);
            sv[half + j] = sinf(arg);
        }
        ggml_backend_tensor_set(t_sin, sv.data(), 0, FT * sizeof(float));
    }
    {
        std::vector<int32_t> pos(T);
        for (int i = 0; i < T; i++) pos[i] = i;
        ggml_backend_tensor_set(positions, pos.data(), 0, T * sizeof(int32_t));
    }
    // inputs_embeds numpy order [B,T,H] row-major == ggml [H,T,B] contiguous. direct copy.
    ggml_backend_tensor_set(emb_in, inputs_embeds, 0, (size_t) B * T * H * sizeof(float));

    ggml_backend_sched_graph_compute(m->sched, gf);

    // stacked [H,T,B] contiguous == numpy [B,T,H]. direct copy.
    ggml_backend_tensor_get(stacked, out, 0, (size_t) B * T * H * sizeof(float));

    ggml_backend_sched_reset(m->sched);
    ggml_free(ctx);
}

static void sgcfm_free(SonggenCfm * m) {
    if (m->sched) ggml_backend_sched_free(m->sched);
    backend_release(m->backend, m->cpu_backend);
    wctx_free(&m->wctx);
    *m = {};
}
