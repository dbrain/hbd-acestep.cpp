// sa3-dit.h: Stable Audio 3 DiT (rf_denoiser velocity model) via ggml.
//
// 24-layer transformer, embed 1536 / 24 heads / D=64. Per block: AdaLN self-attn (differential,
// rms qk-norm, partial NEOX RoPE n_dims=32) + cross-attn to [prompt;seconds] (differential, no RoPE,
// no AdaLN) + local-additive inpaint cond + GLU(SiLU) FF with AdaLN. 64 learned memory tokens
// prepended (stripped before project_out). Conditioning: cross via to_cond_embed; global = to_global_embed(
// seconds) + to_timestep_embed(ExpoFourier(t)); per-layer modulation = to_scale_shift_gate + global_cond_embedder(global).
//
// forward(x[256,L], t, cross_cond[768,257], global_cond[768]) -> velocity[256,L].  Validated vs step{i}_vout golden.
#pragma once
#include "backend.h"
#include "gguf-weights.h"
#include "sa3-attn.h"
#include "sa3-imatrix.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#define SA3_DIT_MAX_LAYERS 32

struct SA3DiTConfig {
    int   embed = 1536, depth = 24, n_heads = 24, head_dim = 64;
    int   io_channels = 256, cond_dim = 768;
    int   mem_tokens = 64, local_cond_dim = 257;
    int   rope_ndims = 32;            // max(head_dim/2, 32)
    float rope_theta = 10000.0f, rms_eps = 1e-6f;
    float attn_scale = 0.125f;        // 1/sqrt(head_dim)
    int   ts_feat_dim = 256;          // ExpoFourier dim
    float ts_min = 0.5f, ts_max = 10000.0f;
};

struct SA3DiTLayer {
    struct ggml_tensor *pre_norm, *ff_norm, *ca_norm, *ssg;
    struct ggml_tensor *sa_qkv, *sa_out, *sa_qn, *sa_kn;
    struct ggml_tensor *ca_q, *ca_kv, *ca_out, *ca_qn, *ca_kn;
    struct ggml_tensor *ff_in_w, *ff_in_b, *ff_out_w, *ff_out_b;
    struct ggml_tensor *loc0_w, *loc0_b, *loc2_w, *loc2_b;
};

struct SA3DiT {
    SA3DiTConfig cfg;
    SA3DiTLayer  layers[SA3_DIT_MAX_LAYERS];
    struct ggml_tensor *preprocess_conv, *postprocess_conv;
    struct ggml_tensor *to_cond0, *to_cond2, *to_global0, *to_global2;
    struct ggml_tensor *to_ts0_w, *to_ts0_b, *to_ts2_w, *to_ts2_b;
    struct ggml_tensor *gce0_w, *gce0_b, *gce2_w, *gce2_b;
    struct ggml_tensor *mem_tokens, *project_in, *project_out;

    ggml_backend_t backend, cpu_backend;
    ggml_backend_sched_t sched;
    WeightCtx wctx;
};

static struct ggml_tensor * sa3_silu(struct ggml_context * c, struct ggml_tensor * x) { return ggml_silu(c, x); }

// conv1d k=1 stored [k=1,in,out] -> mul_mat weight [in,out]
static struct ggml_tensor * sa3_conv1(struct ggml_context * c, struct ggml_tensor * w, struct ggml_tensor * x) {
    struct ggml_tensor * w2 = ggml_reshape_2d(c, w, w->ne[1], w->ne[2]);
    return ggml_mul_mat(c, w2, x);
}

static bool sa3dit_load(SA3DiT * m, const char * path) {
    BackendPair bp = backend_init("SA3-DiT");
    m->backend = bp.backend; m->cpu_backend = bp.cpu_backend;
    m->sched = backend_sched_new(bp, 16384);
    sa3_set_flash(bp.has_gpu);  // flash default-on for GPU (see sa3-attn.h)
    m->cfg = {};

    GGUFModel gf;
    if (!gf_load(&gf, path)) return false;
    wctx_init(&m->wctx, 600);
    auto T  = [&](const char * n) { return gf_load_tensor(&m->wctx, gf, n); };
    auto Tf = [&](const char * n) { return gf_load_tensor_f32(&m->wctx, gf, n); };

    m->preprocess_conv  = T("preprocess_conv");
    m->postprocess_conv = T("postprocess_conv");
    m->to_cond0 = T("to_cond.0"); m->to_cond2 = T("to_cond.2");
    m->to_global0 = T("to_global.0"); m->to_global2 = T("to_global.2");
    m->to_ts0_w = T("to_tstep.0.w"); m->to_ts0_b = Tf("to_tstep.0.b");
    m->to_ts2_w = T("to_tstep.2.w"); m->to_ts2_b = Tf("to_tstep.2.b");
    m->gce0_w = T("gce.0.w"); m->gce0_b = Tf("gce.0.b");
    m->gce2_w = T("gce.2.w"); m->gce2_b = Tf("gce.2.b");
    m->mem_tokens = Tf("mem_tokens");
    m->project_in = T("project_in"); m->project_out = T("project_out");

    for (int i = 0; i < m->cfg.depth; i++) {
        char p[32]; snprintf(p, sizeof(p), "blk.%d.", i);
        std::string s(p);
        SA3DiTLayer * ly = &m->layers[i];
        ly->pre_norm = Tf((s+"pre_norm").c_str()); ly->ff_norm = Tf((s+"ff_norm").c_str());
        ly->ca_norm = Tf((s+"ca_norm").c_str());   ly->ssg = Tf((s+"ssg").c_str());
        ly->sa_qkv = T((s+"sa_qkv").c_str()); ly->sa_out = T((s+"sa_out").c_str());
        ly->sa_qn = Tf((s+"sa_qn").c_str());  ly->sa_kn = Tf((s+"sa_kn").c_str());
        ly->ca_q = T((s+"ca_q").c_str()); ly->ca_kv = T((s+"ca_kv").c_str()); ly->ca_out = T((s+"ca_out").c_str());
        ly->ca_qn = Tf((s+"ca_qn").c_str()); ly->ca_kn = Tf((s+"ca_kn").c_str());
        ly->ff_in_w = T((s+"ff_in.w").c_str()); ly->ff_in_b = Tf((s+"ff_in.b").c_str());
        ly->ff_out_w = T((s+"ff_out.w").c_str()); ly->ff_out_b = Tf((s+"ff_out.b").c_str());
        ly->loc0_w = T((s+"loc.0.w").c_str()); ly->loc0_b = Tf((s+"loc.0.b").c_str());
        ly->loc2_w = T((s+"loc.2.w").c_str()); ly->loc2_b = Tf((s+"loc.2.b").c_str());
    }
    if (!wctx_alloc(&m->wctx, m->backend)) { gf_close(&gf); return false; }
    gf_close(&gf);
    fprintf(stderr, "[SA3-DiT] loaded %dL embed=%d\n", m->cfg.depth, m->cfg.embed);
    return true;
}

// AdaLN modulation: x*(1+scale)+shift = x + x*scale + shift  (scale,shift [E,1] broadcast over S)
static struct ggml_tensor * sa3_modulate(struct ggml_context * c, struct ggml_tensor * x,
                                         struct ggml_tensor * scale, struct ggml_tensor * shift) {
    return ggml_add(c, ggml_add(c, x, ggml_mul(c, x, scale)), shift);
}

// x*sigmoid(1-gate)
static struct ggml_tensor * sa3_gate(struct ggml_context * c, struct ggml_tensor * x,
                                     struct ggml_tensor * gate, struct ggml_tensor * one) {
    struct ggml_tensor * s = ggml_sigmoid(c, ggml_add1(c, ggml_neg(c, gate), one));
    return ggml_mul(c, x, s);
}

// forward. x_lat: [io_channels*L] f32 (ggml [C,L]); cross: [cond_dim*257]; global:[cond_dim]; t scalar.
static void sa3dit_forward(SA3DiT * m, const float * x_lat, int L, float t,
                           const float * cross_cond, int cross_T, const float * global_cond, float * vel_out) {
    const SA3DiTConfig & cf = m->cfg;
    int E = cf.embed, D = cf.head_dim, Nh = cf.n_heads, C = cf.io_channels, M = cf.mem_tokens, S = M + L;

    size_t ctx_size = 8192 * ggml_tensor_overhead() + ggml_graph_overhead_custom(32768, false);
    struct ggml_init_params gp = { ctx_size, NULL, true };
    struct ggml_context * ctx = ggml_init(gp);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 32768, false);

    struct ggml_tensor * in_x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, L);        ggml_set_input(in_x);
    struct ggml_tensor * in_cc  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cf.cond_dim, cross_T); ggml_set_input(in_cc);
    struct ggml_tensor * in_gc  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cf.cond_dim, 1);  ggml_set_input(in_gc);
    struct ggml_tensor * in_ts  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cf.ts_feat_dim, 1); ggml_set_input(in_ts);
    struct ggml_tensor * in_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);            ggml_set_input(in_pos);
    struct ggml_tensor * in_one = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);            ggml_set_input(in_one);
    struct ggml_tensor * in_lz  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cf.local_cond_dim, L); ggml_set_input(in_lz);
    struct ggml_tensor * in_mz  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, M);         ggml_set_input(in_mz);

    // conditioning embeds
    struct ggml_tensor * cc = ggml_mul_mat(ctx, m->to_cond0, in_cc);
    cc = ggml_mul_mat(ctx, m->to_cond2, sa3_silu(ctx, cc));                  // [E,257]
    struct ggml_tensor * ge = ggml_mul_mat(ctx, m->to_global0, in_gc);
    ge = ggml_mul_mat(ctx, m->to_global2, sa3_silu(ctx, ge));                // [E,1]
    struct ggml_tensor * te = ggml_add(ctx, ggml_mul_mat(ctx, m->to_ts0_w, in_ts), m->to_ts0_b);
    te = ggml_add(ctx, ggml_mul_mat(ctx, m->to_ts2_w, sa3_silu(ctx, te)), m->to_ts2_b);  // [E,1]
    struct ggml_tensor * gemb = ggml_add(ctx, ge, te);                      // [E,1]
    struct ggml_tensor * gc = ggml_add(ctx, ggml_mul_mat(ctx, m->gce0_w, gemb), m->gce0_b);
    gc = ggml_add(ctx, ggml_mul_mat(ctx, m->gce2_w, sa3_silu(ctx, gc)), m->gce2_b);      // [6E,1]

    // input path
    struct ggml_tensor * x = ggml_add(ctx, sa3_conv1(ctx, m->preprocess_conv, in_x), in_x);  // [C,L]
    x = ggml_mul_mat(ctx, m->project_in, x);                                // [E,L]
    x = ggml_concat(ctx, sa3_f32(ctx, m->mem_tokens), x, 1);                // [E, M+L] (mem may be quantized)

    for (int i = 0; i < cf.depth; i++) {
        SA3DiTLayer * ly = &m->layers[i];
        struct ggml_tensor * ss = ggml_add(ctx, ly->ssg, gc);              // [6E,1]
        struct ggml_tensor * sc_s = sa3_chunk(ctx, ss, E, 1, 0);
        struct ggml_tensor * sh_s = sa3_chunk(ctx, ss, E, 1, E);
        struct ggml_tensor * gt_s = sa3_chunk(ctx, ss, E, 1, 2*E);
        struct ggml_tensor * sc_f = sa3_chunk(ctx, ss, E, 1, 3*E);
        struct ggml_tensor * sh_f = sa3_chunk(ctx, ss, E, 1, 4*E);
        struct ggml_tensor * gt_f = sa3_chunk(ctx, ss, E, 1, 5*E);

        // self-attn (AdaLN, differential, partial NEOX rope)
        struct ggml_tensor * res = x;
        struct ggml_tensor * h = sa3_modulate(ctx, sa3_rms(ctx, x, ly->pre_norm, cf.rms_eps), sc_s, sh_s);
        struct ggml_tensor * qkv = ggml_mul_mat(ctx, ly->sa_qkv, h);       // [5E, S]
        struct ggml_tensor * q  = sa3_heads(ctx, sa3_chunk(ctx,qkv,E,S,0),     D,Nh,S, ly->sa_qn, cf.rms_eps, in_pos, cf.rope_ndims, cf.rope_theta);
        struct ggml_tensor * k  = sa3_heads(ctx, sa3_chunk(ctx,qkv,E,S,E),     D,Nh,S, ly->sa_kn, cf.rms_eps, in_pos, cf.rope_ndims, cf.rope_theta);
        struct ggml_tensor * v  = sa3_heads(ctx, sa3_chunk(ctx,qkv,E,S,2*E),   D,Nh,S, NULL, cf.rms_eps, NULL, 0, 0);
        struct ggml_tensor * qd = sa3_heads(ctx, sa3_chunk(ctx,qkv,E,S,3*E),   D,Nh,S, ly->sa_qn, cf.rms_eps, in_pos, cf.rope_ndims, cf.rope_theta);
        struct ggml_tensor * kd = sa3_heads(ctx, sa3_chunk(ctx,qkv,E,S,4*E),   D,Nh,S, ly->sa_kn, cf.rms_eps, in_pos, cf.rope_ndims, cf.rope_theta);
        struct ggml_tensor * attn = sa3_diff_attn(ctx, q,qd,k,kd,v, NULL, D,Nh,S, cf.attn_scale);
        attn = ggml_mul_mat(ctx, ly->sa_out, attn);
        x = ggml_add(ctx, res, sa3_gate(ctx, attn, gt_s, in_one));

        // cross-attn (no AdaLN, no rope, differential)
        struct ggml_tensor * res2 = x;
        struct ggml_tensor * hc = sa3_rms(ctx, x, ly->ca_norm, cf.rms_eps);
        struct ggml_tensor * cq = ggml_mul_mat(ctx, ly->ca_q, hc);         // [2E,S]
        struct ggml_tensor * ckv = ggml_mul_mat(ctx, ly->ca_kv, cc);       // [3E,257]
        struct ggml_tensor * cqb = sa3_heads(ctx, sa3_chunk(ctx,cq,E,S,0),       D,Nh,S, ly->ca_qn, cf.rms_eps, NULL,0,0);
        struct ggml_tensor * cqd = sa3_heads(ctx, sa3_chunk(ctx,cq,E,S,E),       D,Nh,S, ly->ca_qn, cf.rms_eps, NULL,0,0);
        struct ggml_tensor * ck  = sa3_heads(ctx, sa3_chunk(ctx,ckv,E,cross_T,0),    D,Nh,cross_T, ly->ca_kn, cf.rms_eps, NULL,0,0);
        struct ggml_tensor * ckd = sa3_heads(ctx, sa3_chunk(ctx,ckv,E,cross_T,E),    D,Nh,cross_T, ly->ca_kn, cf.rms_eps, NULL,0,0);
        struct ggml_tensor * cv  = sa3_heads(ctx, sa3_chunk(ctx,ckv,E,cross_T,2*E),  D,Nh,cross_T, NULL, cf.rms_eps, NULL,0,0);
        struct ggml_tensor * cat = sa3_diff_attn(ctx, cqb,cqd,ck,ckd,cv, NULL, D,Nh,S, cf.attn_scale);
        cat = ggml_mul_mat(ctx, ly->ca_out, cat);
        x = ggml_add(ctx, res2, cat);

        // local additive cond: to_local_embed(zeros) -> [E,L], pad memory region with zeros
        struct ggml_tensor * lc = ggml_add(ctx, ggml_mul_mat(ctx, ly->loc0_w, in_lz), ly->loc0_b);
        lc = ggml_add(ctx, ggml_mul_mat(ctx, ly->loc2_w, sa3_silu(ctx, lc)), ly->loc2_b);  // [E,L]
        struct ggml_tensor * lpad = ggml_concat(ctx, in_mz, lc, 1);        // [E, M+L]
        x = ggml_add(ctx, x, lpad);

        // FF (AdaLN, GLU = first * silu(second))
        struct ggml_tensor * res3 = x;
        struct ggml_tensor * hf = sa3_modulate(ctx, sa3_rms(ctx, x, ly->ff_norm, cf.rms_eps), sc_f, sh_f);
        struct ggml_tensor * proj = ggml_add(ctx, ggml_mul_mat(ctx, ly->ff_in_w, hf), ly->ff_in_b);  // [2*inter, S]
        int inter = (int) ly->ff_out_w->ne[0];
        struct ggml_tensor * first  = sa3_chunk(ctx, proj, inter, S, 0);
        struct ggml_tensor * second = sa3_chunk(ctx, proj, inter, S, inter);
        struct ggml_tensor * ff = ggml_mul(ctx, first, sa3_silu(ctx, second));
        ff = ggml_add(ctx, ggml_mul_mat(ctx, ly->ff_out_w, ff), ly->ff_out_b);  // [E,S]
        x = ggml_add(ctx, res3, sa3_gate(ctx, ff, gt_f, in_one));
    }

    // strip memory tokens, project out, postprocess conv
    x = ggml_cont(ctx, ggml_view_2d(ctx, x, E, L, x->nb[1], (size_t) M * x->nb[1]));  // [E,L]
    x = ggml_mul_mat(ctx, m->project_out, x);                              // [C,L]
    x = ggml_add(ctx, sa3_conv1(ctx, m->postprocess_conv, x), x);          // [C,L]
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);

    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) { fprintf(stderr,"[SA3-DiT] alloc fail\n"); exit(1); }
    if (getenv("SA3_PROBE")) {
        fprintf(stderr, "[probe] DiT L=%d S=%d compute-buffer = %.0f MiB\n", L, S,
                ggml_backend_sched_get_buffer_size(m->sched, m->backend) / 1048576.0);
        ggml_backend_sched_reset(m->sched); ggml_free(ctx); return;
    }

    // ExpoFourier timestep features
    std::vector<float> ts(cf.ts_feat_dim);
    int half = cf.ts_feat_dim / 2;
    float lo = logf(cf.ts_min), hi = logf(cf.ts_max);
    for (int j = 0; j < half; j++) {
        float ramp = (float) j / (float) (half - 1);
        float freq = expf(ramp * (hi - lo) + lo);
        float arg = t * freq * 2.0f * (float) M_PI;
        ts[j] = cosf(arg); ts[half + j] = sinf(arg);
    }
    std::vector<int> pos(S); for (int i = 0; i < S; i++) pos[i] = i;
    float one = 1.0f;
    std::vector<float> lz((size_t) cf.local_cond_dim * L, 0.0f), mz((size_t) E * M, 0.0f);

    ggml_backend_tensor_set(in_x,  x_lat,       0, (size_t) C * L * sizeof(float));
    ggml_backend_tensor_set(in_cc, cross_cond,  0, (size_t) cf.cond_dim * cross_T * sizeof(float));
    ggml_backend_tensor_set(in_gc, global_cond, 0, (size_t) cf.cond_dim * sizeof(float));
    ggml_backend_tensor_set(in_ts, ts.data(),   0, ts.size() * sizeof(float));
    ggml_backend_tensor_set(in_pos, pos.data(), 0, pos.size() * sizeof(int));
    ggml_backend_tensor_set(in_one, &one,       0, sizeof(float));
    ggml_backend_tensor_set(in_lz, lz.data(),   0, lz.size() * sizeof(float));
    ggml_backend_tensor_set(in_mz, mz.data(),   0, mz.size() * sizeof(float));

    sa3_imx_attach(m->sched);  // no-op unless g_sa3_imx.enabled (imatrix calibration)
    ggml_backend_sched_graph_compute(m->sched, gf);
    ggml_backend_tensor_get(x, vel_out, 0, (size_t) C * L * sizeof(float));
    ggml_backend_sched_reset(m->sched);
    ggml_free(ctx);
}

static void sa3dit_free(SA3DiT * m) {
    if (m->sched) ggml_backend_sched_free(m->sched);
    backend_release(m->backend, m->cpu_backend);
    wctx_free(&m->wctx);
    *m = {};
}
