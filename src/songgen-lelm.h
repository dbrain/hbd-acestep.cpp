// songgen-lelm.h : SongGeneration (LeVo 2) hierarchical LeLM graph (GGML)
//
// Hierarchical music language model:
//   embedding-gather conditioning (NO text transformer) -> prepend
//   main transformer (36L Llama: MHA, SwiGLU, RMSNorm, RoPE theta=500000) -> cb0 logits + hidden
//   bridge MLP (Linear 4096->2048, GELU(erf), Linear 2048->2048)
//   sub transformer (12L Llama) -> linears[0,1] -> cb1/cb2 logits
//
// This is the validation/full-prefill path (no KV cache): given the prepend conditioning
// ids + a code sequence of n_code positions, run both stacks over the full causal sequence
// and return the LAST-position logits for all 3 codebooks. Matches golden lelm_logits.
//
// Plain Llama => NO QK-norm (unlike qwen3-enc.h). SwiGLU MLP is reused from qwen3-enc.h.
#pragma once

#include "backend.h"
#include "gguf-weights.h"
#include "qwen3-enc.h"  // Qwen3Layer, qwen3_linear, qwen3_rms_norm, qwen3_build_mlp, qwen3_attn_f32
#include "songgen-cache.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define SGLM_MAX_LAYERS 48

struct SonggenConfig {
    int   dim;            // 2048
    int   ffn;            // 11008
    int   n_heads;        // 16
    int   head_dim;       // 128
    int   n_layers;       // 36 main
    int   n_layers_sub;   // 12 sub
    int   code_depth;     // 3
    int   code_size;      // 16384 (card = code_size+1 = 16385)
    int   card;           // 16385  logits per codebook
    int   input_emb_dim;  // 16386
    float rope_theta;     // 500000
    float rope_theta_sub; // 500000
    float rms_eps;        // 1e-5
    int   special_token;  // 16385 (EOP/mask)
    // conditioner padded lengths
    int desc_len;   // 600
    int audio_len;  // 252 (EOT + 251 code positions)
    int type_len;   // 100
};

struct SonggenLeLM {
    SonggenConfig cfg;

    // main transformer
    Qwen3Layer           main_layers[SGLM_MAX_LAYERS];
    struct ggml_tensor * main_norm;     // [H]  transformer.model.norm.weight
    struct ggml_tensor * lm_head;       // [H, card] transformer.lm_head.weight (cb0)
    // sub transformer
    Qwen3Layer           sub_layers[SGLM_MAX_LAYERS];
    struct ggml_tensor * sub_norm;      // [H]  transformer2.model.norm.weight
    struct ggml_tensor * linears[2];    // [H, card] cb1, cb2

    // bridge MLP
    struct ggml_tensor * mlp0_w;  // [4096, 2048]
    struct ggml_tensor * mlp0_b;  // [2048]
    struct ggml_tensor * mlp2_w;  // [2048, 2048]
    struct ggml_tensor * mlp2_b;  // [2048]

    // code input embeddings
    struct ggml_tensor * emb0;        // [H, 16386]  cb0 input embed
    struct ggml_tensor * layer2_emb1; // [H, 16386]  cb1 input embed
    struct ggml_tensor * layer2_emb2; // [H, 16386]  cb2 input embed

    // conditioners
    struct ggml_tensor * desc_proj;    // [H, 151659]
    struct ggml_tensor * desc_struct;  // [H, 200]
    struct ggml_tensor * type_proj;    // [H, 151652]
    struct ggml_tensor * audio_emb0;   // [H, 16386]
    struct ggml_tensor * audio_emb1;   // [H, 16386]
    struct ggml_tensor * audio_emb2;   // [H, 16386]
    struct ggml_tensor * eot;          // [H]
    struct ggml_tensor * layer2_eot;   // [H]

    WeightCtx            wctx;
    ggml_backend_t       backend;
    ggml_backend_t       cpu_backend;
    ggml_backend_sched_t sched;
    bool                 use_flash_attn;
};

// ---- load ----

static void sglm_load_layer(WeightCtx * w, const GGUFModel & gf, Qwen3Layer * ly, const std::string & p) {
    *ly                     = {};
    ly->input_layernorm     = gf_load_tensor_f32(w, gf, p + ".input_layernorm.weight");
    ly->post_attn_layernorm = gf_load_tensor_f32(w, gf, p + ".post_attention_layernorm.weight");
    ly->q_proj              = gf_load_tensor(w, gf, p + ".self_attn.q_proj.weight");
    ly->k_proj              = gf_load_tensor(w, gf, p + ".self_attn.k_proj.weight");
    ly->v_proj              = gf_load_tensor(w, gf, p + ".self_attn.v_proj.weight");
    ly->o_proj              = gf_load_tensor(w, gf, p + ".self_attn.o_proj.weight");
    ly->gate_proj           = gf_load_tensor(w, gf, p + ".mlp.gate_proj.weight");
    ly->up_proj             = gf_load_tensor(w, gf, p + ".mlp.up_proj.weight");
    ly->down_proj           = gf_load_tensor(w, gf, p + ".mlp.down_proj.weight");
    // q_norm/k_norm stay NULL: plain Llama, no QK-norm.
}

static bool sglm_load_real(SonggenLeLM * m, const char * gguf_path) {
    *m = {};

    BackendPair bp    = backend_init("LeLM");
    m->backend        = bp.backend;
    m->cpu_backend    = bp.cpu_backend;
    m->sched          = backend_sched_new(bp, 8192);
    m->use_flash_attn = bp.has_gpu;

    GGUFModel gf;
    if (!gf_load(&gf, gguf_path)) {
        return false;
    }

    SonggenConfig c = {};
    c.dim           = (int) gf_get_u32(gf, "songgen-lelm.embedding_length");
    c.ffn           = (int) gf_get_u32(gf, "songgen-lelm.feed_forward_length");
    c.n_heads       = (int) gf_get_u32(gf, "songgen-lelm.attention.head_count");
    c.n_layers      = (int) gf_get_u32(gf, "songgen-lelm.block_count");
    c.n_layers_sub  = (int) gf_get_u32(gf, "songgen.num_layers_sub");
    c.code_depth    = (int) gf_get_u32(gf, "songgen.code_depth");
    c.code_size     = (int) gf_get_u32(gf, "songgen.code_size");
    c.rope_theta    = gf_get_f32(gf, "songgen-lelm.rope.freq_base");
    c.rope_theta_sub= gf_get_f32(gf, "songgen.rope_theta_sub");
    c.rms_eps       = gf_get_f32(gf, "songgen-lelm.attention.layer_norm_rms_epsilon");
    int prompt_len  = (int) gf_get_u32(gf, "songgen.prompt_len");
    int frame_rate  = (int) gf_get_u32(gf, "songgen.frame_rate");

    if (c.rms_eps <= 0.0f) c.rms_eps = 1e-5f;
    c.head_dim      = c.dim / c.n_heads;
    c.card          = c.code_size + 1;   // 16385
    c.input_emb_dim = c.code_size + 2;   // 16386
    c.special_token = c.code_size + 1;   // 16385
    c.desc_len      = 600;
    c.audio_len     = prompt_len * frame_rate + 2;  // 252
    c.type_len      = 100;
    m->cfg          = c;

    fprintf(stderr,
            "[LeLM] dim=%d ffn=%d heads=%d hd=%d L=%d+%d code_depth=%d card=%d theta=%.0f/%.0f eps=%.2e\n", c.dim,
            c.ffn, c.n_heads, c.head_dim, c.n_layers, c.n_layers_sub, c.code_depth, c.card, c.rope_theta,
            c.rope_theta_sub, c.rms_eps);

    if (c.n_layers > SGLM_MAX_LAYERS || c.n_layers_sub > SGLM_MAX_LAYERS) {
        fprintf(stderr, "[LeLM] FATAL: layer count exceeds max\n");
        gf_close(&gf);
        return false;
    }

    // tensor count budget (generous): per layer 9 tensors (qkv o, gate up down, 2 norms)
    int n_tensors = (c.n_layers + c.n_layers_sub) * 9 + 32;
    wctx_init(&m->wctx, n_tensors);

    for (int i = 0; i < c.n_layers; i++) {
        sglm_load_layer(&m->wctx, gf, &m->main_layers[i], "transformer.model.layers." + std::to_string(i));
    }
    m->main_norm = gf_load_tensor_f32(&m->wctx, gf, "transformer.model.norm.weight");
    m->lm_head   = gf_load_tensor(&m->wctx, gf, "transformer.lm_head.weight");

    for (int i = 0; i < c.n_layers_sub; i++) {
        sglm_load_layer(&m->wctx, gf, &m->sub_layers[i], "transformer2.model.layers." + std::to_string(i));
    }
    m->sub_norm   = gf_load_tensor_f32(&m->wctx, gf, "transformer2.model.norm.weight");
    m->linears[0] = gf_load_tensor(&m->wctx, gf, "linears.0.weight");
    m->linears[1] = gf_load_tensor(&m->wctx, gf, "linears.1.weight");

    m->mlp0_w = gf_load_tensor(&m->wctx, gf, "mlp.0.weight");
    m->mlp0_b = gf_load_tensor_f32(&m->wctx, gf, "mlp.0.bias");
    m->mlp2_w = gf_load_tensor(&m->wctx, gf, "mlp.2.weight");
    m->mlp2_b = gf_load_tensor_f32(&m->wctx, gf, "mlp.2.bias");

    m->emb0        = gf_load_tensor(&m->wctx, gf, "emb.0.weight");
    m->layer2_emb1 = gf_load_tensor(&m->wctx, gf, "layer2_emb.1.weight");
    m->layer2_emb2 = gf_load_tensor(&m->wctx, gf, "layer2_emb.2.weight");

    m->desc_proj   = gf_load_tensor(&m->wctx, gf, "condition_provider.conditioners.description.output_proj.weight");
    m->desc_struct = gf_load_tensor(&m->wctx, gf, "condition_provider.conditioners.description.structure_emb.weight");
    m->type_proj   = gf_load_tensor(&m->wctx, gf, "condition_provider.conditioners.type_info.output_proj.weight");
    m->audio_emb0  = gf_load_tensor(&m->wctx, gf, "condition_provider.conditioners.prompt_audio.emb.0.weight");
    m->audio_emb1  = gf_load_tensor(&m->wctx, gf, "condition_provider.conditioners.prompt_audio.emb.1.weight");
    m->audio_emb2  = gf_load_tensor(&m->wctx, gf, "condition_provider.conditioners.prompt_audio.emb.2.weight");
    m->eot         = gf_load_tensor_f32(&m->wctx, gf, "condition_provider.conditioners.prompt_audio.EOT_emb");
    m->layer2_eot  = gf_load_tensor_f32(&m->wctx, gf, "condition_provider.conditioners.prompt_audio.layer2_EOT_emb");

    if (!wctx_alloc(&m->wctx, m->backend)) {
        gf_close(&gf);
        return false;
    }
    gf_close(&gf);
    return true;
}

// ---- graph helpers ----

// Self-attention WITHOUT QK-norm. x:[H,S] -> [H,S]. Causal mask [S,S] (fp16) or NULL.
static struct ggml_tensor * sglm_attn(struct ggml_context * ctx,
                                      const SonggenConfig & c,
                                      Qwen3Layer *          ly,
                                      struct ggml_tensor *  x,
                                      struct ggml_tensor *  positions,
                                      struct ggml_tensor *  mask,
                                      float                 theta,
                                      int                   S,
                                      bool                  use_flash_attn) {
    int D  = c.head_dim;
    int Nh = c.n_heads;

    struct ggml_tensor * q = qwen3_linear(ctx, ly->q_proj, x);
    struct ggml_tensor * k = qwen3_linear(ctx, ly->k_proj, x);
    struct ggml_tensor * v = qwen3_linear(ctx, ly->v_proj, x);

    q = ggml_reshape_3d(ctx, q, D, Nh, S);
    k = ggml_reshape_3d(ctx, k, D, Nh, S);
    v = ggml_reshape_3d(ctx, v, D, Nh, S);

    // RoPE NEOX (mode 2), full head_dim
    q = ggml_rope_ext(ctx, q, positions, NULL, D, 2, 0, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, positions, NULL, D, 2, 0, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    q = ggml_permute(ctx, q, 0, 2, 1, 3);  // [D, S, Nh]
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    float scale = 1.0f / sqrtf((float) D);
    struct ggml_tensor * attn;
    if (use_flash_attn) {
        k = ggml_cont(ctx, k);
        v = ggml_cont(ctx, v);
        if (k->type == GGML_TYPE_F32) k = ggml_cast(ctx, k, GGML_TYPE_F16);
        if (v->type == GGML_TYPE_F32) v = ggml_cast(ctx, v, GGML_TYPE_F16);
        attn = ggml_flash_attn_ext(ctx, ggml_cont(ctx, q), k, v, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    } else {
        attn = qwen3_attn_f32(ctx, q, k, v, mask, scale);
    }
    attn = ggml_reshape_2d(ctx, attn, Nh * D, S);
    return qwen3_linear(ctx, ly->o_proj, attn);
}

// One Llama layer (no QK-norm).
static struct ggml_tensor * sglm_layer(struct ggml_context * ctx,
                                       const SonggenConfig & c,
                                       Qwen3Layer *          ly,
                                       struct ggml_tensor *  h,
                                       struct ggml_tensor *  positions,
                                       struct ggml_tensor *  mask,
                                       float                 theta,
                                       int                   S,
                                       bool                  use_flash_attn) {
    struct ggml_tensor * n    = qwen3_rms_norm(ctx, h, ly->input_layernorm, c.rms_eps);
    struct ggml_tensor * attn = sglm_attn(ctx, c, ly, n, positions, mask, theta, S, use_flash_attn);
    h                         = ggml_add(ctx, h, attn);
    n                         = qwen3_rms_norm(ctx, h, ly->post_attn_layernorm, c.rms_eps);
    struct ggml_tensor * mlp  = qwen3_build_mlp(ctx, ly, n, S);
    h                         = ggml_add(ctx, h, mlp);
    return h;
}

static struct ggml_tensor * sglm_stack(struct ggml_context * ctx,
                                       const SonggenConfig & c,
                                       Qwen3Layer *          layers,
                                       int                   n_layers,
                                       struct ggml_tensor *  final_norm,
                                       struct ggml_tensor *  h,
                                       struct ggml_tensor *  positions,
                                       struct ggml_tensor *  mask,
                                       float                 theta,
                                       int                   S,
                                       bool                  use_flash_attn) {
    for (int i = 0; i < n_layers; i++) {
        h = sglm_layer(ctx, c, &layers[i], h, positions, mask, theta, S, use_flash_attn);
    }
    return qwen3_rms_norm(ctx, h, final_norm, c.rms_eps);
}

// ---- conditioning input ids (one CFG row) ----
struct SonggenCondInput {
    std::vector<int32_t> desc_ids;    // [desc_len]
    std::vector<int32_t> desc_cover;  // [desc_len]
    std::vector<int32_t> type_ids;    // [type_len]
    std::vector<int32_t> audio_ids;   // [audio_len-1]  no-prompt: all special; EOT prepended in graph
    // Clone path: per-codebook prompt audio ids (cb0=pmt, cb1=vocal, cb2=bgm). Each [audio_len-1] =
    // [eos_token_id(16384), prompt_codes..., special(16385) padding]. When empty, audio_ids is used
    // for all three (the no-prompt all-16385 case).
    std::vector<int32_t> audio_ids_pmt;
    std::vector<int32_t> audio_ids_vocal;
    std::vector<int32_t> audio_ids_bgm;
    // code positions (n_code each)
    std::vector<int32_t> seq0;
    std::vector<int32_t> seq1;
    std::vector<int32_t> seq2;
};

// ---- forward (full prefill) ----
// Fills cb0/cb1/cb2 with the LAST-position logits (card floats each).
static void sglm_forward(SonggenLeLM * m, const SonggenCondInput & in, float * cb0, float * cb1, float * cb2) {
    const SonggenConfig & c      = m->cfg;
    int                   H      = c.dim;
    int                   n_code = (int) in.seq0.size();
    int                   Pdesc  = c.desc_len;
    int                   Paud   = c.audio_len;       // 252
    int                   Ptype  = c.type_len;
    int                   audN   = c.audio_len - 1;    // 251
    int                   S      = Pdesc + Paud + Ptype + n_code;

    size_t                  ctx_size = (size_t) 8192 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8192, false);
    struct ggml_init_params gp       = { ctx_size, NULL, true };
    struct ggml_context *   ctx      = ggml_init(gp);
    struct ggml_cgraph *    gf       = ggml_new_graph_custom(ctx, 8192, false);

    // --- input id tensors ---
    auto mk_ids = [&](const char * name, int n) {
        struct ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
        ggml_set_name(t, name);
        ggml_set_input(t);
        return t;
    };
    struct ggml_tensor * desc_ids_t   = mk_ids("desc_ids", Pdesc);
    struct ggml_tensor * desc_cover_t = mk_ids("desc_cover", Pdesc);
    struct ggml_tensor * type_ids_t   = mk_ids("type_ids", Ptype);
    struct ggml_tensor * aud_pmt_t    = mk_ids("audio_ids_pmt", audN);
    struct ggml_tensor * aud_voc_t    = mk_ids("audio_ids_vocal", audN);
    struct ggml_tensor * aud_bgm_t    = mk_ids("audio_ids_bgm", audN);
    struct ggml_tensor * seq0_t       = mk_ids("seq0", n_code);
    struct ggml_tensor * seq1_t       = mk_ids("seq1", n_code);
    struct ggml_tensor * seq2_t       = mk_ids("seq2", n_code);
    struct ggml_tensor * positions    = mk_ids("positions", S);

    struct ggml_tensor * mask = NULL;
    if (!m->use_flash_attn) {
        mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, S, S);  // [key, query]
        ggml_set_name(mask, "mask");
        ggml_set_input(mask);
    } else {
        // flash_attn needs mask query-dim padded; pad to multiple of 64.
        int pad = ((S + 63) / 64) * 64;
        mask    = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, S, pad);
        ggml_set_name(mask, "mask");
        ggml_set_input(mask);
    }

    // --- conditioning embeddings ---
    struct ggml_tensor * eot1 = ggml_reshape_2d(ctx, qwen3_f32(ctx, m->eot), H, 1);
    struct ggml_tensor * eot2 = ggml_reshape_2d(ctx, qwen3_f32(ctx, m->layer2_eot), H, 1);

    // description: content + structure (shared input_1/input_2)
    struct ggml_tensor * desc = ggml_add(ctx, ggml_get_rows(ctx, m->desc_proj, desc_ids_t),
                                         ggml_get_rows(ctx, m->desc_struct, desc_cover_t));  // [H, 600]
    // type_info (shared)
    struct ggml_tensor * type = ggml_get_rows(ctx, m->type_proj, type_ids_t);  // [H, 100]

    // prompt_audio: emb0 path (cb0=pmt) and emb1+emb2 path (cb1=vocal + cb2=bgm). Each codebook
    // gets its OWN id stream (clone); for the no-prompt path all three are identical (all-16385).
    struct ggml_tensor * aud1 = ggml_concat(ctx, eot1, ggml_get_rows(ctx, m->audio_emb0, aud_pmt_t), 1);  // [H,252]
    struct ggml_tensor * aud2_codes =
        ggml_add(ctx, ggml_get_rows(ctx, m->audio_emb1, aud_voc_t), ggml_get_rows(ctx, m->audio_emb2, aud_bgm_t));
    struct ggml_tensor * aud2 = ggml_concat(ctx, eot2, aud2_codes, 1);  // [H,252]

    // code embeddings
    struct ggml_tensor * code1 = ggml_get_rows(ctx, m->emb0, seq0_t);  // [H, n_code]
    struct ggml_tensor * code2 = ggml_add(ctx, ggml_get_rows(ctx, m->layer2_emb1, seq1_t),
                                          ggml_get_rows(ctx, m->layer2_emb2, seq2_t));  // [H, n_code]

    // fused_input1 = [desc, aud1, type, code1] ; fused_input2 = [desc, aud2, type, code2]
    struct ggml_tensor * fin1 = ggml_concat(ctx, ggml_concat(ctx, ggml_concat(ctx, desc, aud1, 1), type, 1), code1, 1);
    struct ggml_tensor * fin2 = ggml_concat(ctx, ggml_concat(ctx, ggml_concat(ctx, desc, aud2, 1), type, 1), code2, 1);

    // --- main transformer ---
    struct ggml_tensor * h1 =
        sglm_stack(ctx, c, m->main_layers, c.n_layers, m->main_norm, fin1, positions, mask, c.rope_theta, S,
                   m->use_flash_attn);  // [H, S] post final-norm

    // --- bridge: cat([fin2, h1], feature) -> mlp ---
    struct ggml_tensor * cat = ggml_concat(ctx, fin2, h1, 0);  // [2H, S]
    struct ggml_tensor * br  = ggml_mul_mat(ctx, m->mlp0_w, cat);
    br                       = ggml_add(ctx, br, m->mlp0_b);
    br                       = ggml_gelu_erf(ctx, br);  // torch nn.GELU() default = exact erf
    br                       = ggml_mul_mat(ctx, m->mlp2_w, br);
    br                       = ggml_add(ctx, br, m->mlp2_b);  // [H, S]

    // --- sub transformer ---
    struct ggml_tensor * h2 = sglm_stack(ctx, c, m->sub_layers, c.n_layers_sub, m->sub_norm, br, positions, mask,
                                         c.rope_theta_sub, S, m->use_flash_attn);  // [H, S]

    // --- last-position logits ---
    struct ggml_tensor * h1_last = ggml_view_2d(ctx, h1, H, 1, h1->nb[1], (size_t) (S - 1) * h1->nb[1]);
    struct ggml_tensor * h2_last = ggml_view_2d(ctx, h2, H, 1, h2->nb[1], (size_t) (S - 1) * h2->nb[1]);
    struct ggml_tensor * lg0     = ggml_mul_mat(ctx, m->lm_head, h1_last);     // [card,1]
    struct ggml_tensor * lg1     = ggml_mul_mat(ctx, m->linears[0], h2_last);  // [card,1]
    struct ggml_tensor * lg2     = ggml_mul_mat(ctx, m->linears[1], h2_last);
    ggml_set_output(lg0);
    ggml_set_output(lg1);
    ggml_set_output(lg2);
    ggml_build_forward_expand(gf, lg0);
    ggml_build_forward_expand(gf, lg1);
    ggml_build_forward_expand(gf, lg2);

    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) {
        fprintf(stderr, "[LeLM] FATAL: alloc graph failed (S=%d)\n", S);
        exit(1);
    }

    // --- set inputs ---
    ggml_backend_tensor_set(desc_ids_t, in.desc_ids.data(), 0, Pdesc * sizeof(int32_t));
    ggml_backend_tensor_set(desc_cover_t, in.desc_cover.data(), 0, Pdesc * sizeof(int32_t));
    ggml_backend_tensor_set(type_ids_t, in.type_ids.data(), 0, Ptype * sizeof(int32_t));
    {
        const std::vector<int32_t> & ap = in.audio_ids_pmt.empty() ? in.audio_ids : in.audio_ids_pmt;
        const std::vector<int32_t> & av = in.audio_ids_vocal.empty() ? in.audio_ids : in.audio_ids_vocal;
        const std::vector<int32_t> & ab = in.audio_ids_bgm.empty() ? in.audio_ids : in.audio_ids_bgm;
        ggml_backend_tensor_set(aud_pmt_t, ap.data(), 0, audN * sizeof(int32_t));
        ggml_backend_tensor_set(aud_voc_t, av.data(), 0, audN * sizeof(int32_t));
        ggml_backend_tensor_set(aud_bgm_t, ab.data(), 0, audN * sizeof(int32_t));
    }
    ggml_backend_tensor_set(seq0_t, in.seq0.data(), 0, n_code * sizeof(int32_t));
    ggml_backend_tensor_set(seq1_t, in.seq1.data(), 0, n_code * sizeof(int32_t));
    ggml_backend_tensor_set(seq2_t, in.seq2.data(), 0, n_code * sizeof(int32_t));
    {
        std::vector<int32_t> pos(S);
        for (int i = 0; i < S; i++) pos[i] = i;
        ggml_backend_tensor_set(positions, pos.data(), 0, S * sizeof(int32_t));
    }
    // causal mask: row=query i, col=key j -> 0 if j<=i else -inf. Layout [ne0=key, ne1=query].
    {
        int                   rows = (int) mask->ne[1];
        std::vector<uint16_t> md((size_t) S * rows, ggml_fp32_to_fp16(-INFINITY));
        for (int i = 0; i < S; i++) {
            for (int j = 0; j <= i; j++) md[(size_t) i * S + j] = ggml_fp32_to_fp16(0.0f);
        }
        ggml_backend_tensor_set(mask, md.data(), 0, (size_t) S * rows * sizeof(uint16_t));
    }

    ggml_backend_sched_graph_compute(m->sched, gf);

    ggml_backend_tensor_get(lg0, cb0, 0, c.card * sizeof(float));
    ggml_backend_tensor_get(lg1, cb1, 0, c.card * sizeof(float));
    ggml_backend_tensor_get(lg2, cb2, 0, c.card * sizeof(float));

    ggml_backend_sched_reset(m->sched);
    ggml_free(ctx);
}

static bool sglm_load(SonggenLeLM * m, const char * gguf_path) {
    return sgcache_load(gguf_path, m, sglm_load_real);
}

static void sglm_free(SonggenLeLM * m) {
    if (m->sched) ggml_backend_sched_free(m->sched);
    backend_release(m->backend, m->cpu_backend);
    wctx_free(&m->wctx);
    *m = {};
}
