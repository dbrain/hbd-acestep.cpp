// songgen-lelm-gen.h : KV-cache autoregressive generation for the SongGeneration LeLM.
//
// Reuses songgen-lelm.h (weights, config, conditioning/code embedding assembly, building
// blocks). sglm_forward in songgen-lelm.h stays the full-prefill ORACLE; this file adds a
// dual KV-cache decode path (main 36L + sub 12L), one cache set per CFG row.
//
// Per AR step we run ONE CFG row: embed code token(s) -> (step0: prepend conditioning) ->
// main 36L (KV) -> h1 -> cb0; bridge cat([fin2,h1]) -> sub 12L (KV) -> h2 -> cb1,cb2.
// Positions are continuous 0..S-1 across [prepend ++ codes] (NEOX RoPE, theta=500000).
//
// KV cache layout per (set, layer): [D, max_seq, Nh] f16 (MHA, n_kv=n_heads=16).
#pragma once

#include "songgen-lelm.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define SGGEN_MAX_SETS 4  // cond/uncond for main + cond/uncond for sub kept separate per stack

// Upstream SongGeneration-v2-large generates to the model's max length and stops on EOS — there
// is no duration target. config.yaml max_dur = 270 s @ 25 fps frame rate = 6750 code frames.
#define SGGEN_MODEL_MAX_FRAMES 6750  // 270 s ceiling (EOS terminates earlier for most prompts)

struct SonggenLeLMGen {
    SonggenLeLM * m;
    int           max_seq;  // current KV capacity (allocated)
    int           cap_seq;  // hard ceiling: KV never grows past this (== max_seq for static alloc)
    int           n_sets;   // 2 (cond, uncond)

    struct ggml_context * kv_ctx;
    ggml_backend_buffer_t kv_buf;
    // 4D batched caches per layer: [D, max_seq, Nh, n_sets]. cond=set0, uncond=set1.
    struct ggml_tensor * main_k4[SGLM_MAX_LAYERS];
    struct ggml_tensor * main_v4[SGLM_MAX_LAYERS];
    struct ggml_tensor * sub_k4[SGLM_MAX_LAYERS];
    struct ggml_tensor * sub_v4[SGLM_MAX_LAYERS];
    // 3D per-set views into the 4D caches (for the single-row sggen_step path / gate test).
    struct ggml_tensor * main_k[2][SGLM_MAX_LAYERS];
    struct ggml_tensor * main_v[2][SGLM_MAX_LAYERS];
    struct ggml_tensor * sub_k[2][SGLM_MAX_LAYERS];
    struct ggml_tensor * sub_v[2][SGLM_MAX_LAYERS];
    int                  main_pos[2];
    int                  sub_pos[2];
};

// KV cache element type. F16 default; SG_KV_TYPE=q8_0|q4_0 quantizes the cache (FA reads it
// directly). D=head_dim must be a multiple of the quant block (32) — 128/32=4, ok. Quantized KV
// halves (Q8_0) or quarters (Q4_0) the LM-stage KV footprint, the binding VRAM term at length.
static enum ggml_type sggen_kv_type() {
    const char * e = getenv("SG_KV_TYPE");
    if (e) {
        if (!strcmp(e, "q8_0")) return GGML_TYPE_Q8_0;
        if (!strcmp(e, "q4_0")) return GGML_TYPE_Q4_0;
        if (!strcmp(e, "f16"))  return GGML_TYPE_F16;
    }
    return GGML_TYPE_F16;
}

static size_t sggen_kv_bytes(const SonggenConfig & c, int n_sets, int S, enum ggml_type kvt) {
    return (size_t) n_sets * (c.n_layers + c.n_layers_sub) * 2 *
           ((size_t) c.head_dim * S * c.n_heads / ggml_blck_size(kvt)) * ggml_type_size(kvt);
}

// (Re)build the KV tensors+views+buffer at capacity `S` into a fresh ctx. Leaves positions
// untouched (so a grow preserves main_pos/sub_pos). Sets g->max_seq=S.
static void sggen_build_kv_tensors(SonggenLeLMGen * g, int S) {
    SonggenLeLM *         m = g->m;
    const SonggenConfig & c = m->cfg;
    int D = c.head_dim, Nh = c.n_heads, Lm = c.n_layers, Ls = c.n_layers_sub;
    int n_sets = g->n_sets;
    enum ggml_type kvt = sggen_kv_type();
    g->max_seq = S;

    // (Lm+Ls)*2 4D tensors + (Lm+Ls)*2*n_sets 3D views (views still consume ctx object space).
    int                     n_tensors = (Lm + Ls) * 2 * (1 + n_sets);
    size_t                  ctx_size  = (size_t) n_tensors * ggml_tensor_overhead() + 1024;
    struct ggml_init_params gp        = { ctx_size, NULL, true };
    g->kv_ctx                         = ggml_init(gp);

    auto mk4 = [&](struct ggml_tensor *(&k4)[SGLM_MAX_LAYERS], struct ggml_tensor *(&v4)[SGLM_MAX_LAYERS],
                   struct ggml_tensor *(&k3)[2][SGLM_MAX_LAYERS], struct ggml_tensor *(&v3)[2][SGLM_MAX_LAYERS], int L) {
        for (int l = 0; l < L; l++) {
            k4[l] = ggml_new_tensor_4d(g->kv_ctx, kvt, D, S, Nh, n_sets);
            v4[l] = ggml_new_tensor_4d(g->kv_ctx, kvt, D, S, Nh, n_sets);
            for (int s = 0; s < n_sets; s++) {
                size_t off = (size_t) s * k4[l]->nb[3];
                k3[s][l]   = ggml_view_3d(g->kv_ctx, k4[l], D, S, Nh, k4[l]->nb[1], k4[l]->nb[2], off);
                v3[s][l]   = ggml_view_3d(g->kv_ctx, v4[l], D, S, Nh, v4[l]->nb[1], v4[l]->nb[2], off);
            }
        }
    };
    mk4(g->main_k4, g->main_v4, g->main_k, g->main_v, Lm);
    mk4(g->sub_k4, g->sub_v4, g->sub_k, g->sub_v, Ls);

    g->kv_buf = ggml_backend_alloc_ctx_tensors(g->kv_ctx, m->backend);
    if (!g->kv_buf) {
        fprintf(stderr, "[LeLM-gen] FATAL: KV alloc failed (S=%d)\n", S);
        exit(1);
    }
    ggml_backend_buffer_clear(g->kv_buf, 0);
}

// Dynamic alloc: start at init_seq, allow growth up to cap_seq. Static alloc (cap==init)
// reproduces the old behaviour exactly (no growth, one buffer sized to the full song).
static void sggen_alloc_kv_dyn(SonggenLeLMGen * g, SonggenLeLM * m, int init_seq, int cap_seq, int n_sets) {
    *g        = {};
    g->m      = m;
    g->n_sets = n_sets;
    if (init_seq > cap_seq) init_seq = cap_seq;
    if (init_seq < 1) init_seq = 1;
    g->cap_seq = cap_seq;
    sggen_build_kv_tensors(g, init_seq);
    for (int s = 0; s < n_sets; s++) {
        g->main_pos[s] = 0;
        g->sub_pos[s]  = 0;
    }
    const SonggenConfig & c = m->cfg;
    enum ggml_type        kvt = sggen_kv_type();
    fprintf(stderr, "[LeLM-gen] KV: %d sets, main %dL + sub %dL, type=%s, S=%d (cap %d) -> %.1f MB (cap %.1f MB)\n",
            n_sets, c.n_layers, c.n_layers_sub, ggml_type_name(kvt), init_seq, cap_seq,
            (float) sggen_kv_bytes(c, n_sets, init_seq, kvt) / (1024 * 1024),
            (float) sggen_kv_bytes(c, n_sets, cap_seq, kvt) / (1024 * 1024));
}

static void sggen_alloc_kv(SonggenLeLMGen * g, SonggenLeLM * m, int max_seq, int n_sets) {
    sggen_alloc_kv_dyn(g, m, max_seq, max_seq, n_sets);  // static: init == cap (never grows)
}

// Stage the populated [0,used) prefix of every cache tensor to HOST, free the old GPU buffer,
// build the new (bigger) buffer, then upload. Going via host means the old and new GPU buffers are
// never co-resident — the grow transient is weights + max(old,new) KV, not old+new KV (which would
// briefly ~double the KV term and spike the peak). nb[1] (D-row bytes) is capacity-independent, so
// each (set,head) slab is a contiguous `used*nb[1]` run; raw byte move, KV-type-agnostic (no dequant).
static void sggen_grow_kv(SonggenLeLMGen * g, int new_seq) {
    if (new_seq > g->cap_seq) new_seq = g->cap_seq;
    if (new_seq <= g->max_seq) return;
    SonggenLeLM *         m  = g->m;
    const SonggenConfig & c  = m->cfg;
    int                   Lm = c.n_layers, Ls = c.n_layers_sub;
    int                   used_main = g->main_pos[0];
    int                   used_sub  = g->sub_pos[0];
    for (int s = 1; s < g->n_sets; s++) {
        if (g->main_pos[s] > used_main) used_main = g->main_pos[s];
        if (g->sub_pos[s] > used_sub) used_sub = g->sub_pos[s];
    }
    int     old_seq = g->max_seq;
    int64_t t0      = ggml_time_us();

    // 1) snapshot old GPU tensors → host (one staging buffer per tensor, freed as we go).
    auto stage_down = [&](struct ggml_tensor *(&arr)[SGLM_MAX_LAYERS], int L, int used,
                          std::vector<std::vector<char>> & host) {
        int Nh = (int) arr[0]->ne[2], n_sets = (int) arr[0]->ne[3];
        size_t seg = (size_t) used * arr[0]->nb[1];
        for (int l = 0; l < L; l++) {
            host[l].resize((size_t) n_sets * Nh * seg);
            for (int s = 0; s < n_sets; s++)
                for (int h = 0; h < Nh; h++)
                    ggml_backend_tensor_get(arr[l], host[l].data() + ((size_t) s * Nh + h) * seg,
                                            (size_t) s * arr[l]->nb[3] + (size_t) h * arr[l]->nb[2], seg);
        }
    };
    std::vector<std::vector<char>> hmk(Lm), hmv(Lm), hsk(Ls), hsv(Ls);
    if (used_main > 0) { stage_down(g->main_k4, Lm, used_main, hmk); stage_down(g->main_v4, Lm, used_main, hmv); }
    if (used_sub > 0)  { stage_down(g->sub_k4, Ls, used_sub, hsk);  stage_down(g->sub_v4, Ls, used_sub, hsv); }

    // 2) free old GPU buffer BEFORE allocating the new one (no co-residence).
    struct ggml_context * old_ctx = g->kv_ctx;
    ggml_backend_buffer_t old_buf = g->kv_buf;
    ggml_backend_buffer_free(old_buf);
    ggml_free(old_ctx);

    // 3) build new (bigger) buffer and upload from host.
    sggen_build_kv_tensors(g, new_seq);  // rebuilds g->kv_ctx/kv_buf/main_k4/.../views, sets max_seq
    auto stage_up = [&](struct ggml_tensor *(&arr)[SGLM_MAX_LAYERS], int L, int used,
                        std::vector<std::vector<char>> & host) {
        int Nh = (int) arr[0]->ne[2], n_sets = (int) arr[0]->ne[3];
        size_t seg = (size_t) used * arr[0]->nb[1];
        for (int l = 0; l < L; l++)
            for (int s = 0; s < n_sets; s++)
                for (int h = 0; h < Nh; h++)
                    ggml_backend_tensor_set(arr[l], host[l].data() + ((size_t) s * Nh + h) * seg,
                                            (size_t) s * arr[l]->nb[3] + (size_t) h * arr[l]->nb[2], seg);
    };
    if (used_main > 0) { stage_up(g->main_k4, Lm, used_main, hmk); stage_up(g->main_v4, Lm, used_main, hmv); }
    if (used_sub > 0)  { stage_up(g->sub_k4, Ls, used_sub, hsk);  stage_up(g->sub_v4, Ls, used_sub, hsv); }

    enum ggml_type kvt = sggen_kv_type();
    fprintf(stderr, "[LeLM-gen] KV grow %d -> %d (used main=%d sub=%d, %.1f MB, %.1f ms, host-bridged)\n", old_seq,
            new_seq, used_main, used_sub, (float) sggen_kv_bytes(c, g->n_sets, new_seq, kvt) / (1024 * 1024),
            (ggml_time_us() - t0) / 1000.0f);
}

// Ensure capacity for `need` total KV length. Grow geometrically (≈doubling) for ~linear total copy
// cost, but cap the per-grow increment at SG_KV_GROW_BLOCK frames so we don't over-allocate (and
// over-spike the grow transient) near the ceiling. Default 1500 (~60 s ≈ 600 MB q8_0 overshoot cap).
static inline void sggen_ensure_kv(SonggenLeLMGen * g, int need) {
    if (need <= g->max_seq) return;
    static int blk = []() {
        const char * e = getenv("SG_KV_GROW_BLOCK");
        int          v = e ? atoi(e) : 1500;
        return v > 0 ? v : 1500;
    }();
    int target = g->max_seq * 2;
    if (target > g->max_seq + blk) target = g->max_seq + blk;
    if (target < need) target = need;
    sggen_grow_kv(g, target);
}

static void sggen_free_kv(SonggenLeLMGen * g) {
    if (g->kv_buf) ggml_backend_buffer_free(g->kv_buf);
    if (g->kv_ctx) ggml_free(g->kv_ctx);
    *g = {};
}

// Attention with KV cache write@kv_pos + read[0..kv_len], NO QK-norm, NEOX RoPE theta.
// x:[H,S_new]; cache_k/v:[D,max_seq,Nh]; mask:[kv_len,S_new] f16 or NULL.
static struct ggml_tensor * sggen_attn(struct ggml_context * ctx,
                                       struct ggml_cgraph *  gf,
                                       const SonggenConfig & c,
                                       Qwen3Layer *          ly,
                                       struct ggml_tensor *  x,
                                       struct ggml_tensor *  positions,
                                       struct ggml_tensor *  mask,
                                       struct ggml_tensor *  cache_k,
                                       struct ggml_tensor *  cache_v,
                                       int                   kv_pos,
                                       int                   kv_len,
                                       int                   max_seq,
                                       float                 theta,
                                       int                   S) {
    int D  = c.head_dim;
    int Nh = c.n_heads;

    struct ggml_tensor * q = qwen3_linear(ctx, ly->q_proj, x);
    struct ggml_tensor * k = qwen3_linear(ctx, ly->k_proj, x);
    struct ggml_tensor * v = qwen3_linear(ctx, ly->v_proj, x);

    q = ggml_reshape_3d(ctx, q, D, Nh, S);
    k = ggml_reshape_3d(ctx, k, D, Nh, S);
    v = ggml_reshape_3d(ctx, v, D, Nh, S);

    q = ggml_rope_ext(ctx, q, positions, NULL, D, 2, 0, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, positions, NULL, D, 2, 0, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // -> [D, S, Nh]
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    // write K,V into cache at kv_pos. Use the cache tensor's OWN strides (nb[1]=seq, nb[2]=head)
    // — these are correct for whatever element type the cache was allocated with (F16 or Q8_0).
    // (Previously hardcoded F16 strides, which silently corrupted the prefix KV when the cache is
    // quantized: writes landed at F16 offsets but the batched decoder reads at Q8_0 offsets.)
    size_t nb1 = cache_k->nb[1];
    size_t nb2 = cache_k->nb[2];
    size_t off = (size_t) kv_pos * nb1;
    struct ggml_tensor * k_dst = ggml_view_3d(ctx, cache_k, D, S, Nh, nb1, nb2, off);
    struct ggml_tensor * v_dst = ggml_view_3d(ctx, cache_v, D, S, Nh, cache_v->nb[1], cache_v->nb[2],
                                              (size_t) kv_pos * cache_v->nb[1]);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, k, k_dst));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, v, v_dst));

    // read full KV [0..kv_len]
    struct ggml_tensor * k_full = ggml_view_3d(ctx, cache_k, D, kv_len, Nh, nb1, nb2, 0);
    struct ggml_tensor * v_full = ggml_view_3d(ctx, cache_v, D, kv_len, Nh, cache_v->nb[1], cache_v->nb[2], 0);

    float                scale = 1.0f / sqrtf((float) D);
    static const bool    no_fa = getenv("SGLM_NO_FA") != NULL;
    struct ggml_tensor * attn;
    if (!no_fa) {
        // FA over the F16 KV-cache views; q [D,S,Nh,1], k/v [D,kv_len,Nh,1], mask [kv_len,S].
        attn = ggml_flash_attn_ext(ctx, ggml_cont(ctx, q), k_full, v_full, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    } else {
        attn = qwen3_attn_f32(ctx, ggml_cont(ctx, q), k_full, v_full, mask, scale);
    }
    attn = ggml_reshape_2d(ctx, attn, Nh * D, S);
    return qwen3_linear(ctx, ly->o_proj, attn);
}

static struct ggml_tensor * sggen_layer(struct ggml_context * ctx,
                                        struct ggml_cgraph *  gf,
                                        const SonggenConfig & c,
                                        Qwen3Layer *          ly,
                                        struct ggml_tensor *  h,
                                        struct ggml_tensor *  positions,
                                        struct ggml_tensor *  mask,
                                        struct ggml_tensor *  cache_k,
                                        struct ggml_tensor *  cache_v,
                                        int                   kv_pos,
                                        int                   kv_len,
                                        int                   max_seq,
                                        float                 theta,
                                        int                   S) {
    struct ggml_tensor * n = qwen3_rms_norm(ctx, h, ly->input_layernorm, c.rms_eps);
    struct ggml_tensor * a =
        sggen_attn(ctx, gf, c, ly, n, positions, mask, cache_k, cache_v, kv_pos, kv_len, max_seq, theta, S);
    h                        = ggml_add(ctx, h, a);
    n                        = qwen3_rms_norm(ctx, h, ly->post_attn_layernorm, c.rms_eps);
    struct ggml_tensor * mlp = qwen3_build_mlp(ctx, ly, n, S);
    return ggml_add(ctx, h, mlp);
}

// ---- one CFG-row decode step ----
// in: conditioning ids (used only when prepend==true, i.e. step 0) + the S_code new code tokens.
//     seq0/seq1/seq2 hold the new code tokens for THIS step (1 at decode, or the step-0 code at prefill).
// At step 0 (prepend), the conditioning P tokens are prepended; positions span [0..P+S_code-1].
// At later steps, kv_pos = main_pos[set]; positions span [kv_pos .. kv_pos+S_code-1].
// Fills cb0/cb1/cb2 with last-position logits (card floats each).
static void sggen_step(SonggenLeLMGen * g,
                       int               set,
                       const SonggenCondInput & in,
                       bool              prepend,
                       float *           cb0,
                       float *           cb1,
                       float *           cb2) {
    SonggenLeLM *         m = g->m;
    const SonggenConfig & c = m->cfg;
    int H      = c.dim;
    int n_code = (int) in.seq0.size();
    int Pdesc  = c.desc_len;
    int Paud   = c.audio_len;
    int Ptype  = c.type_len;
    int audN   = c.audio_len - 1;
    int P      = prepend ? (Pdesc + Paud + Ptype) : 0;
    int S      = P + n_code;  // tokens processed this step

    int kv_pos_main = g->main_pos[set];
    int kv_pos_sub  = g->sub_pos[set];
    int kv_len_main = kv_pos_main + S;
    int kv_len_sub  = kv_pos_sub + S;
    sggen_ensure_kv(g, kv_len_main > kv_len_sub ? kv_len_main : kv_len_sub);
    if (kv_len_main > g->max_seq) {
        fprintf(stderr, "[LeLM-gen] FATAL: kv_len %d > cap %d\n", kv_len_main, g->cap_seq);
        exit(1);
    }

    static bool    _prof = getenv("SGLM_PROFILE") != NULL;
    static int64_t _acc_build = 0, _acc_alloc = 0, _acc_comp = 0, _acc_read = 0;
    static int     _acc_n = 0;
    int64_t        _tb0 = _prof ? ggml_time_us() : 0;

    size_t                  ctx_size = (size_t) 16384 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16384, false);
    struct ggml_init_params gp       = { ctx_size, NULL, true };
    struct ggml_context *   ctx      = ggml_init(gp);
    struct ggml_cgraph *    gf       = ggml_new_graph_custom(ctx, 16384, false);

    auto mk_ids = [&](const char * name, int n) {
        struct ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
        ggml_set_name(t, name);
        ggml_set_input(t);
        return t;
    };
    struct ggml_tensor * seq0_t    = mk_ids("seq0", n_code);
    struct ggml_tensor * seq1_t    = mk_ids("seq1", n_code);
    struct ggml_tensor * seq2_t    = mk_ids("seq2", n_code);
    struct ggml_tensor * positions = mk_ids("positions", S);

    struct ggml_tensor *desc_ids_t = NULL, *desc_cover_t = NULL, *type_ids_t = NULL;
    struct ggml_tensor *aud_pmt_t = NULL, *aud_voc_t = NULL, *aud_bgm_t = NULL;
    if (prepend) {
        desc_ids_t   = mk_ids("desc_ids", Pdesc);
        desc_cover_t = mk_ids("desc_cover", Pdesc);
        type_ids_t   = mk_ids("type_ids", Ptype);
        aud_pmt_t    = mk_ids("audio_ids_pmt", audN);
        aud_voc_t    = mk_ids("audio_ids_vocal", audN);
        aud_bgm_t    = mk_ids("audio_ids_bgm", audN);
    }

    // masks: main and sub share identical causal layout (same S, kv_len), so one mask suffices.
    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_len_main, S);
    ggml_set_name(mask, "mask");
    ggml_set_input(mask);

    // --- code embeddings for the new tokens ---
    struct ggml_tensor * code1 = ggml_get_rows(ctx, m->emb0, seq0_t);  // [H, n_code]
    struct ggml_tensor * code2 = ggml_add(ctx, ggml_get_rows(ctx, m->layer2_emb1, seq1_t),
                                          ggml_get_rows(ctx, m->layer2_emb2, seq2_t));  // [H, n_code]

    struct ggml_tensor * fin1 = code1;
    struct ggml_tensor * fin2 = code2;
    if (prepend) {
        struct ggml_tensor * eot1 = ggml_reshape_2d(ctx, qwen3_f32(ctx, m->eot), H, 1);
        struct ggml_tensor * eot2 = ggml_reshape_2d(ctx, qwen3_f32(ctx, m->layer2_eot), H, 1);
        struct ggml_tensor * desc = ggml_add(ctx, ggml_get_rows(ctx, m->desc_proj, desc_ids_t),
                                             ggml_get_rows(ctx, m->desc_struct, desc_cover_t));  // [H,600]
        struct ggml_tensor * type = ggml_get_rows(ctx, m->type_proj, type_ids_t);                // [H,100]
        struct ggml_tensor * aud1 =
            ggml_concat(ctx, eot1, ggml_get_rows(ctx, m->audio_emb0, aud_pmt_t), 1);  // [H,252]
        struct ggml_tensor * aud2_codes = ggml_add(ctx, ggml_get_rows(ctx, m->audio_emb1, aud_voc_t),
                                                   ggml_get_rows(ctx, m->audio_emb2, aud_bgm_t));
        struct ggml_tensor * aud2 = ggml_concat(ctx, eot2, aud2_codes, 1);  // [H,252]
        fin1 = ggml_concat(ctx, ggml_concat(ctx, ggml_concat(ctx, desc, aud1, 1), type, 1), code1, 1);
        fin2 = ggml_concat(ctx, ggml_concat(ctx, ggml_concat(ctx, desc, aud2, 1), type, 1), code2, 1);
    }

    // --- main transformer (KV) ---
    struct ggml_tensor * h1 = fin1;
    for (int l = 0; l < c.n_layers; l++) {
        h1 = sggen_layer(ctx, gf, c, &m->main_layers[l], h1, positions, mask, g->main_k[set][l], g->main_v[set][l],
                         kv_pos_main, kv_len_main, g->max_seq, c.rope_theta, S);
    }
    h1 = qwen3_rms_norm(ctx, h1, m->main_norm, c.rms_eps);

    // --- bridge: cat([fin2, h1], feature) -> mlp ---
    struct ggml_tensor * cat = ggml_concat(ctx, fin2, h1, 0);  // [2H,S]
    struct ggml_tensor * br  = ggml_mul_mat(ctx, m->mlp0_w, cat);
    br                       = ggml_add(ctx, br, m->mlp0_b);
    br                       = ggml_gelu_erf(ctx, br);
    br                       = ggml_mul_mat(ctx, m->mlp2_w, br);
    br                       = ggml_add(ctx, br, m->mlp2_b);  // [H,S]

    // --- sub transformer (KV) ---
    struct ggml_tensor * h2 = br;
    for (int l = 0; l < c.n_layers_sub; l++) {
        h2 = sggen_layer(ctx, gf, c, &m->sub_layers[l], h2, positions, mask, g->sub_k[set][l], g->sub_v[set][l],
                         kv_pos_sub, kv_len_sub, g->max_seq, c.rope_theta_sub, S);
    }
    h2 = qwen3_rms_norm(ctx, h2, m->sub_norm, c.rms_eps);

    // --- last-position logits ---
    struct ggml_tensor * h1_last = ggml_view_2d(ctx, h1, H, 1, h1->nb[1], (size_t) (S - 1) * h1->nb[1]);
    struct ggml_tensor * h2_last = ggml_view_2d(ctx, h2, H, 1, h2->nb[1], (size_t) (S - 1) * h2->nb[1]);
    struct ggml_tensor * lg0     = ggml_mul_mat(ctx, m->lm_head, h1_last);
    struct ggml_tensor * lg1     = ggml_mul_mat(ctx, m->linears[0], h2_last);
    struct ggml_tensor * lg2     = ggml_mul_mat(ctx, m->linears[1], h2_last);
    ggml_set_output(lg0);
    ggml_set_output(lg1);
    ggml_set_output(lg2);
    ggml_build_forward_expand(gf, lg0);
    ggml_build_forward_expand(gf, lg1);
    ggml_build_forward_expand(gf, lg2);

    int64_t _ta0 = _prof ? ggml_time_us() : 0;
    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) {
        fprintf(stderr, "[LeLM-gen] FATAL: alloc graph failed (S=%d)\n", S);
        exit(1);
    }

    ggml_backend_tensor_set(seq0_t, in.seq0.data(), 0, n_code * sizeof(int32_t));
    ggml_backend_tensor_set(seq1_t, in.seq1.data(), 0, n_code * sizeof(int32_t));
    ggml_backend_tensor_set(seq2_t, in.seq2.data(), 0, n_code * sizeof(int32_t));
    if (prepend) {
        ggml_backend_tensor_set(desc_ids_t, in.desc_ids.data(), 0, Pdesc * sizeof(int32_t));
        ggml_backend_tensor_set(desc_cover_t, in.desc_cover.data(), 0, Pdesc * sizeof(int32_t));
        ggml_backend_tensor_set(type_ids_t, in.type_ids.data(), 0, Ptype * sizeof(int32_t));
        const std::vector<int32_t> & ap = in.audio_ids_pmt.empty() ? in.audio_ids : in.audio_ids_pmt;
        const std::vector<int32_t> & av = in.audio_ids_vocal.empty() ? in.audio_ids : in.audio_ids_vocal;
        const std::vector<int32_t> & ab = in.audio_ids_bgm.empty() ? in.audio_ids : in.audio_ids_bgm;
        ggml_backend_tensor_set(aud_pmt_t, ap.data(), 0, audN * sizeof(int32_t));
        ggml_backend_tensor_set(aud_voc_t, av.data(), 0, audN * sizeof(int32_t));
        ggml_backend_tensor_set(aud_bgm_t, ab.data(), 0, audN * sizeof(int32_t));
    }
    {
        std::vector<int32_t> pos(S);
        for (int i = 0; i < S; i++) pos[i] = kv_pos_main + i;  // main and sub advance identically
        ggml_backend_tensor_set(positions, pos.data(), 0, S * sizeof(int32_t));
    }
    {
        // causal mask [key=kv_len, query=S]: query i (abs pos kv_pos+i) attends keys j<=kv_pos+i.
        int                   KL = kv_len_main;
        std::vector<uint16_t> md((size_t) KL * S, ggml_fp32_to_fp16(-INFINITY));
        for (int i = 0; i < S; i++) {
            int qpos = kv_pos_main + i;
            for (int j = 0; j <= qpos; j++) md[(size_t) i * KL + j] = ggml_fp32_to_fp16(0.0f);
        }
        ggml_backend_tensor_set(mask, md.data(), 0, (size_t) KL * S * sizeof(uint16_t));
    }

    int64_t _tc0 = _prof ? ggml_time_us() : 0;
    ggml_backend_sched_graph_compute(m->sched, gf);
    int64_t _tr0 = _prof ? ggml_time_us() : 0;

    ggml_backend_tensor_get(lg0, cb0, 0, c.card * sizeof(float));
    ggml_backend_tensor_get(lg1, cb1, 0, c.card * sizeof(float));
    ggml_backend_tensor_get(lg2, cb2, 0, c.card * sizeof(float));

    g->main_pos[set] += S;
    g->sub_pos[set] += S;

    ggml_backend_sched_reset(m->sched);
    ggml_free(ctx);

    static int64_t _w_comp = 0, _w_tot = 0;
    if (_prof) {
        int64_t _te = ggml_time_us();
        int64_t _step = _te - _tb0;
        _acc_build += _ta0 - _tb0;
        _acc_alloc += _tc0 - _ta0;
        _acc_comp  += _tr0 - _tc0;
        _acc_read  += _te - _tr0;
        _w_comp += _tr0 - _tc0;
        _w_tot  += _step;
        if (++_acc_n % 50 == 0) {
            fprintf(stderr,
                    "[SGLM-PROF] %d fwd-calls marginal-last50: compute=%.1fms total=%.1fms/call\n",
                    _acc_n, _w_comp / 1000.0 / 50.0, _w_tot / 1000.0 / 50.0);
            _w_comp = 0; _w_tot = 0;
        }
    }
}

// ============================================================================
// CFG-batched decode step: process N rows (cond=set0, uncond=set1) in ONE graph.
// Requires S_code==1 per row (true at every decode step) and identical kv_pos
// across rows (the two caches advance in lockstep). Numerically identical to two
// separate sggen_step() calls: same per-row K/V written, each row attends only
// its own cache, masks/positions match. Use sggen_step(prepend=true) for step 0.
// ============================================================================

// Batched attention over N rows, S=1 each. x:[H,1,N]; cache4 k/v:[D,max_seq,Nh,n_sets].
// mask:[kv_len,1,1,N] f16. Returns [Nh*D,1,N].
static struct ggml_tensor * sggen_attn_batch(struct ggml_context * ctx,
                                             struct ggml_cgraph *  gf,
                                             const SonggenConfig & c,
                                             Qwen3Layer *          ly,
                                             struct ggml_tensor *  x,
                                             struct ggml_tensor *  positions,
                                             struct ggml_tensor *  mask,
                                             struct ggml_tensor *  cache_k4,
                                             struct ggml_tensor *  cache_v4,
                                             int                   kv_pos,
                                             int                   kv_len,
                                             float                 theta,
                                             int                   N) {
    int D  = c.head_dim;
    int Nh = c.n_heads;

    struct ggml_tensor * q = qwen3_linear(ctx, ly->q_proj, x);  // [Nh*D, 1, N]
    struct ggml_tensor * k = qwen3_linear(ctx, ly->k_proj, x);
    struct ggml_tensor * v = qwen3_linear(ctx, ly->v_proj, x);

    // [Nh*D,1,N] -> [D, Nh, N]
    q = ggml_reshape_3d(ctx, q, D, Nh, N);
    k = ggml_reshape_3d(ctx, k, D, Nh, N);
    v = ggml_reshape_3d(ctx, v, D, Nh, N);

    q = ggml_rope_ext(ctx, q, positions, NULL, D, 2, 0, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, positions, NULL, D, 2, 0, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    k = ggml_cont(ctx, k);
    v = ggml_cont(ctx, v);

    // Write each row's new K,V into its set's slice of the 4D cache at kv_pos.
    for (int i = 0; i < N; i++) {
        size_t off = (size_t) i * cache_k4->nb[3] + (size_t) kv_pos * cache_k4->nb[1];
        // row i: [D, Nh, 1] from [D, Nh, N]
        struct ggml_tensor * ki = ggml_view_3d(ctx, k, D, Nh, 1, k->nb[1], k->nb[2], (size_t) i * k->nb[2]);
        struct ggml_tensor * vi = ggml_view_3d(ctx, v, D, Nh, 1, v->nb[1], v->nb[2], (size_t) i * v->nb[2]);
        // [D, Nh, 1] -> [D, 1, Nh] (cache layout [D, seq, Nh])
        ki = ggml_cont(ctx, ggml_permute(ctx, ki, 0, 2, 1, 3));
        vi = ggml_cont(ctx, ggml_permute(ctx, vi, 0, 2, 1, 3));
        struct ggml_tensor * k_dst =
            ggml_view_3d(ctx, cache_k4, D, 1, Nh, cache_k4->nb[1], cache_k4->nb[2], off);
        struct ggml_tensor * v_dst =
            ggml_view_3d(ctx, cache_v4, D, 1, Nh, cache_v4->nb[1], cache_v4->nb[2], off);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ki, k_dst));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, vi, v_dst));
    }

    // q: [D, Nh, N] -> [D, 1, Nh, N]
    struct ggml_tensor * q4 = ggml_reshape_4d(ctx, ggml_cont(ctx, q), D, 1, Nh, N);

    // batched KV read: [D, kv_len, Nh, N] view over all sets (sets 0..N-1 consecutive in ne3).
    struct ggml_tensor * k_batch = ggml_view_4d(ctx, cache_k4, D, kv_len, Nh, N, cache_k4->nb[1], cache_k4->nb[2],
                                                cache_k4->nb[3], 0);
    struct ggml_tensor * v_batch = ggml_view_4d(ctx, cache_v4, D, kv_len, Nh, N, cache_v4->nb[1], cache_v4->nb[2],
                                                cache_v4->nb[3], 0);

    float                scale = 1.0f / sqrtf((float) D);
    // Flash attention (default): one fused kernel over the strided KV-cache views instead of the
    // ~6-kernel manual QKᵀ/softmax/transpose/×V chain — fewer launches (decode is latency-bound) and
    // it consumes K/V directly in cache layout (un-transposed V == what FA wants), which also lets the
    // cache be quantized. Validated against the prefill oracle by lelm-gen-test. SGLM_NO_FA reverts.
    static const bool no_fa = getenv("SGLM_NO_FA") != NULL;
    struct ggml_tensor * attn;
    if (!no_fa) {
        attn = ggml_flash_attn_ext(ctx, q4, k_batch, v_batch, mask, scale, 0.0f, 0.0f);  // [D, Nh, 1, N]
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    } else {
        attn = qwen3_attn_f32(ctx, q4, k_batch, v_batch, mask, scale);  // [D, Nh, 1, N]
    }
    attn = ggml_reshape_3d(ctx, attn, Nh * D, 1, N);
    return qwen3_linear(ctx, ly->o_proj, attn);
}

static struct ggml_tensor * sggen_layer_batch(struct ggml_context * ctx,
                                              struct ggml_cgraph *  gf,
                                              const SonggenConfig & c,
                                              Qwen3Layer *          ly,
                                              struct ggml_tensor *  h,
                                              struct ggml_tensor *  positions,
                                              struct ggml_tensor *  mask,
                                              struct ggml_tensor *  cache_k4,
                                              struct ggml_tensor *  cache_v4,
                                              int                   kv_pos,
                                              int                   kv_len,
                                              float                 theta,
                                              int                   N) {
    struct ggml_tensor * n = qwen3_rms_norm(ctx, h, ly->input_layernorm, c.rms_eps);
    struct ggml_tensor * a =
        sggen_attn_batch(ctx, gf, c, ly, n, positions, mask, cache_k4, cache_v4, kv_pos, kv_len, theta, N);
    h                        = ggml_add(ctx, h, a);
    n                        = qwen3_rms_norm(ctx, h, ly->post_attn_layernorm, c.rms_eps);
    struct ggml_tensor * mlp = qwen3_build_mlp(ctx, ly, n, N);
    return ggml_add(ctx, h, mlp);
}

// One CFG-batched decode step. ins[i] supplies the single code token for row i (set i).
// Fills cb0/cb1/cb2 per row: cb[k] has N*card floats (row-major over rows).
static void sggen_step_batch(SonggenLeLMGen *                      g,
                             const std::vector<SonggenCondInput> & ins,
                             float *                               cb0,
                             float *                               cb1,
                             float *                               cb2) {
    SonggenLeLM *         m = g->m;
    const SonggenConfig & c = m->cfg;
    int                   H = c.dim;
    int                   N = (int) ins.size();

    int kv_pos_main = g->main_pos[0];
    int kv_pos_sub  = g->sub_pos[0];
    for (int i = 0; i < N; i++) {
        if (g->main_pos[i] != kv_pos_main || g->sub_pos[i] != kv_pos_sub || (int) ins[i].seq0.size() != 1) {
            fprintf(stderr, "[LeLM-gen] FATAL: sggen_step_batch requires lockstep kv_pos and S_code==1\n");
            exit(1);
        }
    }
    int kv_len_main = kv_pos_main + 1;
    int kv_len_sub  = kv_pos_sub + 1;
    int kv_len      = kv_len_main > kv_len_sub ? kv_len_main : kv_len_sub;
    // Round the FA KV-read up to a fixed 256 stride so the CUDA flash-attn decode kernel hits its
    // 256-aligned fast path (~17% faster than a ragged length; ported from the acestep LM, qwen3-lm.h).
    // The cache is zeroed at alloc/grow and the mask sets padded positions [kv_len, kv_read) to -inf,
    // so the tail is numerically inert. Bucketing changes FP block-accumulation order → at temp=0 greedy
    // a borderline argmax can flip (different-but-equivalent); SG_KV_BUCKET=1 reverts to ragged.
    static const int KV_BUCKET = []() { const char * e = getenv("SG_KV_BUCKET"); return e ? atoi(e) : 256; }();
    int kv_read = (KV_BUCKET <= 1) ? kv_len : ((kv_len + KV_BUCKET - 1) / KV_BUCKET) * KV_BUCKET;
    sggen_ensure_kv(g, kv_read);                       // buffer must cover the padded read
    if (kv_read > g->max_seq) kv_read = g->max_seq;    // clamp at the hard ceiling (last bucket may be ragged)
    if (kv_len_main > g->max_seq) {
        fprintf(stderr, "[LeLM-gen] FATAL: kv_len %d > cap %d\n", kv_len_main, g->cap_seq);
        exit(1);
    }

    static bool    _prof = getenv("SGLM_PROFILE") != NULL;
    static int64_t _acc_build = 0, _acc_alloc = 0, _acc_comp = 0, _acc_read = 0;
    static int     _acc_n = 0;
    int64_t        _tb0 = _prof ? ggml_time_us() : 0;

    size_t                  ctx_size = (size_t) 16384 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16384, false);
    struct ggml_init_params gp       = { ctx_size, NULL, true };
    struct ggml_context *   ctx      = ggml_init(gp);
    struct ggml_cgraph *    gf       = ggml_new_graph_custom(ctx, 16384, false);

    auto mk_ids = [&](const char * name, int n) {
        struct ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
        ggml_set_name(t, name);
        ggml_set_input(t);
        return t;
    };
    // per-row code ids laid out [N] (one token each)
    struct ggml_tensor * seq0_t    = mk_ids("seq0", N);
    struct ggml_tensor * seq1_t    = mk_ids("seq1", N);
    struct ggml_tensor * seq2_t    = mk_ids("seq2", N);
    struct ggml_tensor * positions = mk_ids("positions", N);

    struct ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv_read, 1, 1, N);
    ggml_set_name(mask, "mask");
    ggml_set_input(mask);

    // code embeddings: [H, N]
    struct ggml_tensor * code1 = ggml_get_rows(ctx, m->emb0, seq0_t);
    struct ggml_tensor * code2 = ggml_add(ctx, ggml_get_rows(ctx, m->layer2_emb1, seq1_t),
                                          ggml_get_rows(ctx, m->layer2_emb2, seq2_t));
    // [H, N] -> [H, 1, N]
    struct ggml_tensor * fin1 = ggml_reshape_3d(ctx, code1, H, 1, N);
    struct ggml_tensor * fin2 = ggml_reshape_3d(ctx, code2, H, 1, N);

    struct ggml_tensor * h1 = fin1;
    for (int l = 0; l < c.n_layers; l++) {
        h1 = sggen_layer_batch(ctx, gf, c, &m->main_layers[l], h1, positions, mask, g->main_k4[l], g->main_v4[l],
                               kv_pos_main, kv_read, c.rope_theta, N);
    }
    h1 = qwen3_rms_norm(ctx, h1, m->main_norm, c.rms_eps);

    struct ggml_tensor * cat = ggml_concat(ctx, fin2, h1, 0);  // [2H,1,N]
    struct ggml_tensor * br  = ggml_mul_mat(ctx, m->mlp0_w, cat);
    br                       = ggml_add(ctx, br, m->mlp0_b);
    br                       = ggml_gelu_erf(ctx, br);
    br                       = ggml_mul_mat(ctx, m->mlp2_w, br);
    br                       = ggml_add(ctx, br, m->mlp2_b);  // [H,1,N]

    struct ggml_tensor * h2 = br;
    for (int l = 0; l < c.n_layers_sub; l++) {
        h2 = sggen_layer_batch(ctx, gf, c, &m->sub_layers[l], h2, positions, mask, g->sub_k4[l], g->sub_v4[l],
                               kv_pos_sub, kv_read, c.rope_theta_sub, N);
    }
    h2 = qwen3_rms_norm(ctx, h2, m->sub_norm, c.rms_eps);

    // S==1 so every position IS the last position. [H,1,N] -> [H,N]
    struct ggml_tensor * h1f = ggml_reshape_2d(ctx, h1, H, N);
    struct ggml_tensor * h2f = ggml_reshape_2d(ctx, h2, H, N);
    struct ggml_tensor * lg0 = ggml_mul_mat(ctx, m->lm_head, h1f);     // [card, N]
    struct ggml_tensor * lg1 = ggml_mul_mat(ctx, m->linears[0], h2f);  // [card, N]
    struct ggml_tensor * lg2 = ggml_mul_mat(ctx, m->linears[1], h2f);
    ggml_set_output(lg0);
    ggml_set_output(lg1);
    ggml_set_output(lg2);
    ggml_build_forward_expand(gf, lg0);
    ggml_build_forward_expand(gf, lg1);
    ggml_build_forward_expand(gf, lg2);

    int64_t _ta0 = _prof ? ggml_time_us() : 0;
    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) {
        fprintf(stderr, "[LeLM-gen] FATAL: batch alloc graph failed (N=%d)\n", N);
        exit(1);
    }

    {
        std::vector<int32_t> s0(N), s1(N), s2(N), pos(N);
        for (int i = 0; i < N; i++) {
            s0[i]  = ins[i].seq0[0];
            s1[i]  = ins[i].seq1[0];
            s2[i]  = ins[i].seq2[0];
            pos[i] = kv_pos_main;  // identical across rows
        }
        ggml_backend_tensor_set(seq0_t, s0.data(), 0, N * sizeof(int32_t));
        ggml_backend_tensor_set(seq1_t, s1.data(), 0, N * sizeof(int32_t));
        ggml_backend_tensor_set(seq2_t, s2.data(), 0, N * sizeof(int32_t));
        ggml_backend_tensor_set(positions, pos.data(), 0, N * sizeof(int32_t));
    }
    {
        // mask [key=kv_read, 1, 1, N]: the single query (abs pos kv_pos) attends real keys [0,kv_len);
        // bucketed-pad positions [kv_len, kv_read) are -inf so the 256-aligned read tail is inert.
        const uint16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
        std::vector<uint16_t> md((size_t) kv_read * N);
        for (int n = 0; n < N; n++)
            for (int j = 0; j < kv_read; j++) md[(size_t) n * kv_read + j] = (j < kv_len) ? z : ninf;
        ggml_backend_tensor_set(mask, md.data(), 0, (size_t) kv_read * N * sizeof(uint16_t));
    }

    int64_t _tc0 = _prof ? ggml_time_us() : 0;
    ggml_backend_sched_graph_compute(m->sched, gf);
    int64_t _tr0 = _prof ? ggml_time_us() : 0;

    ggml_backend_tensor_get(lg0, cb0, 0, (size_t) c.card * N * sizeof(float));
    ggml_backend_tensor_get(lg1, cb1, 0, (size_t) c.card * N * sizeof(float));
    ggml_backend_tensor_get(lg2, cb2, 0, (size_t) c.card * N * sizeof(float));

    for (int i = 0; i < N; i++) {
        g->main_pos[i] += 1;
        g->sub_pos[i] += 1;
    }

    ggml_backend_sched_reset(m->sched);
    ggml_free(ctx);

    static int64_t _w_comp = 0, _w_tot = 0;
    if (_prof) {
        int64_t _te = ggml_time_us();
        int64_t _step = _te - _tb0;
        _acc_build += _ta0 - _tb0;
        _acc_alloc += _tc0 - _ta0;
        _acc_comp  += _tr0 - _tc0;
        _acc_read  += _te - _tr0;
        _w_comp += _tr0 - _tc0;
        _w_tot  += _step;
        if (++_acc_n % 50 == 0) {
            fprintf(stderr,
                    "[SGLM-BATCH-PROF] step %d (N=%d) marginal-last50: compute=%.1fms total=%.1fms/step\n",
                    _acc_n, N, _w_comp / 1000.0 / 50.0, _w_tot / 1000.0 / 50.0);
            _w_comp = 0; _w_tot = 0;
        }
    }
}
