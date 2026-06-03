// songgen-septoken.h : SongGeneration from-codes decode driver (host glue).
//
// Turns LM code streams (vocal + bgm) into a VAE latent[64,T] via the CFM reflow ODE,
// reusing the validated CFM estimator graph (songgen-cfm.h). No prompt / single segment:
// every latent position has mask==2, attention all-ones, no incontext freeze.
//
// Source of truth: model_septoken.py inference_codes (589-657) + solve_euler (139-194).
//
//   from_codes(codes[T])         -> mu[1024,T]   (codebook[code] -> out_proj matmul + bias)
//   inputs_embeds[2,T,2200]      = [ mask_emb(2)[24] | zeros[64] | mu_vocal[1024] |
//                                    mu_bgm[1024] | x_next[64] ]   (row0=uncond zeros the
//                                    2048 mu block, row1=cond keeps it)
//   euler ODE (20 steps, t=linspace(0,1,21), dt=1/20, guidance 1.5):
//     v = est(inputs_embeds, t)[...,-64:] ; v = v_un + 1.5*(v_cond - v_un) ; x += dt*v
//   denorm: latent = x*std + mean   (normfeat return_sample, target_std=1, power_std=1)
#pragma once

#include "songgen-cfm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

struct SgRvqStream {
    std::vector<float> codebook;  // [16384*32] row-major (code, dim)
    std::vector<float> in_w;      // [32*1024] row-major (out=32, in=1024)
    std::vector<float> in_b;      // [32]
    std::vector<float> out_w;     // [1024*32] row-major (out, in)
    std::vector<float> out_b;     // [1024]
};

struct SonggenSeptoken {
    SgRvqStream        vocal;
    SgRvqStream        bgm;
    std::vector<float> mask_emb;   // [3*24]
    std::vector<float> zero_cond;  // [1024]
    std::vector<float> nf_mean;    // [64]
    std::vector<float> nf_std;     // [64]
};

static const int SGSEP_CODEBOOK = 16384;
static const int SGSEP_CB_DIM   = 32;
static const int SGSEP_MU_DIM   = 1024;
static const int SGSEP_MASK_DIM = 24;
static const int SGSEP_LAT_DIM  = 64;
static const int SGSEP_H        = 2200;  // 24 + 64 + 1024 + 1024 + 64

// Pull an F32 tensor (already F32 in the aux gguf) straight out of the mmap.
static void sgsep_read_f32(const GGUFModel & gf, const char * name, std::vector<float> & dst, size_t n) {
    int64_t idx = gguf_find_tensor(gf.gguf, name);
    if (idx < 0) {
        fprintf(stderr, "[SEP] FATAL: tensor '%s' not found in aux gguf\n", name);
        exit(1);
    }
    struct ggml_tensor * src = ggml_get_tensor(gf.meta, name);
    if (src->type != GGML_TYPE_F32) {
        fprintf(stderr, "[SEP] FATAL: tensor '%s' is not F32 (type=%d)\n", name, src->type);
        exit(1);
    }
    if ((size_t) ggml_nelements(src) != n) {
        fprintf(stderr, "[SEP] FATAL: tensor '%s' has %lld elems, expected %zu\n", name,
                (long long) ggml_nelements(src), n);
        exit(1);
    }
    dst.resize(n);
    const float * p = (const float *) gf_get_data(gf, name);
    memcpy(dst.data(), p, n * sizeof(float));
}

static bool sgsep_load(SonggenSeptoken * m, const char * gguf_path) {
    *m = {};
    GGUFModel gf;
    if (!gf_load(&gf, gguf_path)) {
        return false;
    }

    sgsep_read_f32(gf, "rvq_vocal.codebook.weight", m->vocal.codebook, (size_t) SGSEP_CODEBOOK * SGSEP_CB_DIM);
    sgsep_read_f32(gf, "rvq_vocal.in_proj.weight", m->vocal.in_w, (size_t) SGSEP_CB_DIM * SGSEP_MU_DIM);
    sgsep_read_f32(gf, "rvq_vocal.in_proj.bias", m->vocal.in_b, SGSEP_CB_DIM);
    sgsep_read_f32(gf, "rvq_vocal.out_proj.weight", m->vocal.out_w, (size_t) SGSEP_MU_DIM * SGSEP_CB_DIM);
    sgsep_read_f32(gf, "rvq_vocal.out_proj.bias", m->vocal.out_b, SGSEP_MU_DIM);
    sgsep_read_f32(gf, "rvq_bgm.codebook.weight", m->bgm.codebook, (size_t) SGSEP_CODEBOOK * SGSEP_CB_DIM);
    sgsep_read_f32(gf, "rvq_bgm.in_proj.weight", m->bgm.in_w, (size_t) SGSEP_CB_DIM * SGSEP_MU_DIM);
    sgsep_read_f32(gf, "rvq_bgm.in_proj.bias", m->bgm.in_b, SGSEP_CB_DIM);
    sgsep_read_f32(gf, "rvq_bgm.out_proj.weight", m->bgm.out_w, (size_t) SGSEP_MU_DIM * SGSEP_CB_DIM);
    sgsep_read_f32(gf, "rvq_bgm.out_proj.bias", m->bgm.out_b, SGSEP_MU_DIM);
    sgsep_read_f32(gf, "mask_emb.weight", m->mask_emb, (size_t) 3 * SGSEP_MASK_DIM);
    sgsep_read_f32(gf, "zero_cond_embedding1", m->zero_cond, SGSEP_MU_DIM);
    sgsep_read_f32(gf, "normfeat.mean", m->nf_mean, SGSEP_LAT_DIM);
    sgsep_read_f32(gf, "normfeat.std", m->nf_std, SGSEP_LAT_DIM);

    fprintf(stderr, "[SEP] aux loaded: codebook %dx%d out_proj %dx%d  nf_mean[0]=%.4f nf_std[0]=%.4f\n",
            SGSEP_CODEBOOK, SGSEP_CB_DIM, SGSEP_MU_DIM, SGSEP_CB_DIM, m->nf_mean[0], m->nf_std[0]);

    gf_close(&gf);
    return true;
}

// Load the single 1rvq codebook (model_1rvq) as an SgRvqStream from the 1rvq-aux gguf.
// Used by the clone-encode path for the pmt stream (rvq_pmt.*).
static bool sgsep_load_1rvq(SgRvqStream * s, const char * gguf_path) {
    *s = {};
    GGUFModel gf;
    if (!gf_load(&gf, gguf_path)) {
        return false;
    }
    sgsep_read_f32(gf, "rvq_pmt.codebook.weight", s->codebook, (size_t) SGSEP_CODEBOOK * SGSEP_CB_DIM);
    sgsep_read_f32(gf, "rvq_pmt.in_proj.weight", s->in_w, (size_t) SGSEP_CB_DIM * SGSEP_MU_DIM);
    sgsep_read_f32(gf, "rvq_pmt.in_proj.bias", s->in_b, SGSEP_CB_DIM);
    sgsep_read_f32(gf, "rvq_pmt.out_proj.weight", s->out_w, (size_t) SGSEP_MU_DIM * SGSEP_CB_DIM);
    sgsep_read_f32(gf, "rvq_pmt.out_proj.bias", s->out_b, SGSEP_MU_DIM);
    fprintf(stderr, "[1RVQ] pmt codebook loaded: %dx%d\n", SGSEP_CODEBOOK, SGSEP_CB_DIM);
    gf_close(&gf);
    return true;
}

// from_codes for one stream: codes[T] (already validated 0..16383) -> mu[T*1024],
// laid out time-major: mu[t*1024 + oc]. mu[oc] = sum_ic out_w[oc,ic]*codebook[code,ic] + bias[oc].
static void sgsep_from_codes(const SgRvqStream & s, const std::vector<int> & codes, std::vector<float> & mu) {
    int T = (int) codes.size();
    mu.resize((size_t) T * SGSEP_MU_DIM);
    for (int t = 0; t < T; t++) {
        const float * z   = &s.codebook[(size_t) codes[t] * SGSEP_CB_DIM];  // [32]
        float *       dst = &mu[(size_t) t * SGSEP_MU_DIM];
        for (int oc = 0; oc < SGSEP_MU_DIM; oc++) {
            const float * w   = &s.out_w[(size_t) oc * SGSEP_CB_DIM];
            float         acc = s.out_b[oc];
            for (int ic = 0; ic < SGSEP_CB_DIM; ic++) {
                acc += w[ic] * z[ic];
            }
            dst[(size_t) oc] = acc;
        }
    }
}

// RVQ encode (mirror of VectorQuantize.forward / decode_latents, eval mode).
//   feats[1024,T] channel-major (npy [1,1024,T]) ->
//     in_proj -> z_e[32,T]; L2-normalize z_e cols AND codebook rows;
//     argmin euclidean (= argmax cosine) over 16384 -> codes[T];
//     out_proj(original codebook row) -> quantized[1024,T].
// codes/z_e/quantized written channel-major. argmax tie-break = first index (matches torch .max).
static void sgsep_rvq_encode(const SgRvqStream & s, const float * feats, int T, std::vector<int> & codes,
                             std::vector<float> & z_e, std::vector<float> & quantized) {
    codes.resize(T);
    z_e.resize((size_t) SGSEP_CB_DIM * T);
    quantized.resize((size_t) SGSEP_MU_DIM * T);

    // Pre-normalize codebook rows once (F.normalize default eps=1e-12).
    std::vector<float> cb_norm((size_t) SGSEP_CODEBOOK * SGSEP_CB_DIM);
    for (int n = 0; n < SGSEP_CODEBOOK; n++) {
        const float * row = &s.codebook[(size_t) n * SGSEP_CB_DIM];
        float         ss  = 0.0f;
        for (int d = 0; d < SGSEP_CB_DIM; d++) ss += row[d] * row[d];
        float inv = 1.0f / std::max(std::sqrt(ss), 1e-12f);
        float *     dst = &cb_norm[(size_t) n * SGSEP_CB_DIM];
        for (int d = 0; d < SGSEP_CB_DIM; d++) dst[d] = row[d] * inv;
    }

    std::vector<float> ze_n(SGSEP_CB_DIM);
    for (int t = 0; t < T; t++) {
        // in_proj: z_e[d] = bias[d] + sum_ic in_w[d,ic] * feats[ic,t].
        float ze_col[SGSEP_CB_DIM];
        for (int d = 0; d < SGSEP_CB_DIM; d++) {
            const float * w   = &s.in_w[(size_t) d * SGSEP_MU_DIM];
            float         acc = s.in_b[d];
            for (int ic = 0; ic < SGSEP_MU_DIM; ic++) acc += w[ic] * feats[(size_t) ic * T + t];
            ze_col[d]                          = acc;
            z_e[(size_t) d * T + t]            = acc;
        }
        // L2-normalize z_e column.
        float ss = 0.0f;
        for (int d = 0; d < SGSEP_CB_DIM; d++) ss += ze_col[d] * ze_col[d];
        float inv = 1.0f / std::max(std::sqrt(ss), 1e-12f);
        for (int d = 0; d < SGSEP_CB_DIM; d++) ze_n[d] = ze_col[d] * inv;

        // argmin euclidean on normalized = argmax cosine = argmax dot.
        // torch computes -dist and takes .max(1)[1] (first index of the max). dist =
        // |e|^2 - 2 e.cb + |cb|^2; |e|^2=1, |cb|^2=1 so argmin dist == argmax dot.
        int   best = 0;
        float bestdot = -1e30f;
        for (int n = 0; n < SGSEP_CODEBOOK; n++) {
            const float * cbn = &cb_norm[(size_t) n * SGSEP_CB_DIM];
            float         dot = 0.0f;
            for (int d = 0; d < SGSEP_CB_DIM; d++) dot += ze_n[d] * cbn[d];
            if (dot > bestdot) {
                bestdot = dot;
                best    = n;
            }
        }
        codes[t] = best;

        // out_proj on ORIGINAL (un-normalized) codebook row.
        const float * z = &s.codebook[(size_t) best * SGSEP_CB_DIM];
        for (int oc = 0; oc < SGSEP_MU_DIM; oc++) {
            const float * w   = &s.out_w[(size_t) oc * SGSEP_CB_DIM];
            float         acc = s.out_b[oc];
            for (int d = 0; d < SGSEP_CB_DIM; d++) acc += w[d] * z[d];
            quantized[(size_t) oc * T + t] = acc;
        }
    }
}

// normfeat.project_sample forward normalize (model_septoken.py:80-88, power_std=1, target_std=1):
//   x_norm = (x - mean) / std.  Input is the raw VAE-encoded prompt latent, channel-major
//   [64*T] (lat[c*T + t]); output is the TIME-MAJOR [T*64] normalized latent that
//   sgsep_decode_seg consumes as incontext_latent (it expects project_sample already applied,
//   matching the previous-chunk-overlap path which feeds normalized latents back). Exact inverse
//   of the return_sample (x*std + mean) at the bottom of sgsep_decode_seg.
static void sgsep_project_sample(const SonggenSeptoken * sep, const std::vector<float> & lat_cm, int T,
                                 std::vector<float> & norm_tm) {
    norm_tm.resize((size_t) T * SGSEP_LAT_DIM);
    for (int c = 0; c < SGSEP_LAT_DIM; c++) {
        float mean = sep->nf_mean[c];
        float istd = 1.0f / std::max(sep->nf_std[c], 1e-12f);
        for (int t = 0; t < T; t++) {
            norm_tm[(size_t) t * SGSEP_LAT_DIM + c] = (lat_cm[(size_t) c * T + t] - mean) * istd;
        }
    }
}

// Segment decode (mirrors inference_codes). codes each [T]. incontext_latent (if non-empty)
// is the previous segment's overlap, ALREADY normalized (project_sample applied), length
// incontext_length frames, laid out time-major [incontext_length*64]. For incontext frames
// the mask row is 1 (start of an other_seg); rest use row 2. x_next for incontext frames is
// blended toward incontext_latent each euler step, and the final latent's incontext frames are
// overwritten with incontext_latent (matching model_septoken latents[:inc]=incontext_latents).
// Output: latent_norm (normalized, time-major [T*64], pre-return_sample) AND latent_out
// (denormalized channel-major [64*T]). latent_norm is what the next chunk feeds as incontext.
static void sgsep_decode_seg(SonggenSeptoken * sep, SonggenCfm * cfm, const std::vector<int> & codes_vocal,
                             const std::vector<int> & codes_bgm, uint64_t seed, int incontext_length,
                             const std::vector<float> & incontext_latent, std::vector<float> & latent_norm,
                             std::vector<float> & latent_out) {
    int T = (int) codes_vocal.size();
    if ((int) codes_bgm.size() != T) {
        fprintf(stderr, "[SEP] FATAL: vocal/bgm length mismatch (%d vs %d)\n", T, (int) codes_bgm.size());
        exit(1);
    }
    if (incontext_length < 0) incontext_length = 0;
    if (incontext_length > T) incontext_length = T;

    std::vector<float> mu_v, mu_b;
    sgsep_from_codes(sep->vocal, codes_vocal, mu_v);  // [T*1024]
    sgsep_from_codes(sep->bgm, codes_bgm, mu_b);      // [T*1024]

    // mask_emb rows: row 1 (incontext), row 2 (generate).
    const float * mrow1 = &sep->mask_emb[(size_t) 1 * SGSEP_MASK_DIM];
    const float * mrow2 = &sep->mask_emb[(size_t) 2 * SGSEP_MASK_DIM];

    // noise = seeded randn[T,64]; x_next starts as a clone.
    std::vector<float> noise((size_t) T * SGSEP_LAT_DIM);
    {
        std::mt19937_64                 rng(seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (size_t i = 0; i < noise.size(); i++) noise[i] = nd(rng);
    }
    std::vector<float> x_next = noise;

    const int   H        = SGSEP_H;
    const int   off_mask = 0;
    const int   off_inc  = SGSEP_MASK_DIM;            // 24
    const int   off_muv  = off_inc + SGSEP_LAT_DIM;   // 88
    const int   off_mub  = off_muv + SGSEP_MU_DIM;    // 1112
    const int   off_xn   = off_mub + SGSEP_MU_DIM;    // 2136
    const int   num_steps = 20;
    const float dt        = 1.0f / (float) num_steps;
    const float guidance  = 1.5f;
    const float sigma_min = 1e-4f;  // CFM default

    std::vector<float> emb((size_t) 2 * T * H, 0.0f);
    auto erow = [&](int b, int t) -> float * { return &emb[((size_t) b * T + t) * H]; };
    for (int t = 0; t < T; t++) {
        const float * mrow = (t < incontext_length) ? mrow1 : mrow2;
        for (int b = 0; b < 2; b++) {
            float * e = erow(b, t);
            for (int j = 0; j < SGSEP_MASK_DIM; j++) e[off_mask + j] = mrow[j];
        }
        // incontext_x block (off_inc..off_inc+64): the normalized prev-segment latent for
        // incontext frames, zero elsewhere. Present in BOTH rows (double(incontext_x)).
        if (t < incontext_length) {
            const float * il = &incontext_latent[(size_t) t * SGSEP_LAT_DIM];
            for (int b = 0; b < 2; b++) {
                float * e = erow(b, t) + off_inc;
                for (int j = 0; j < SGSEP_LAT_DIM; j++) e[j] = il[j];
            }
        }
        // mu blocks: cond row keeps them; uncond row zero. (latent_masks here are all >0.5,
        // so mu is the real codes for every frame — no zero_cond substitution needed.)
        float *       ec = erow(1, t);
        const float * pv = &mu_v[(size_t) t * SGSEP_MU_DIM];
        const float * pb = &mu_b[(size_t) t * SGSEP_MU_DIM];
        for (int j = 0; j < SGSEP_MU_DIM; j++) ec[off_muv + j] = pv[j];
        for (int j = 0; j < SGSEP_MU_DIM; j++) ec[off_mub + j] = pb[j];
    }

    std::vector<float> est_out((size_t) 2 * T * H);

    for (int i = 0; i < num_steps; i++) {
        float ti = (float) i / (float) num_steps;

        // incontext blend then write x_next into both rows.
        for (int t = 0; t < T; t++) {
            float * xn = &x_next[(size_t) t * SGSEP_LAT_DIM];
            if (t < incontext_length) {
                const float * nz = &noise[(size_t) t * SGSEP_LAT_DIM];
                const float * il = &incontext_latent[(size_t) t * SGSEP_LAT_DIM];
                float         a  = 1.0f - (1.0f - sigma_min) * ti;
                for (int j = 0; j < SGSEP_LAT_DIM; j++) xn[j] = a * nz[j] + ti * il[j];
            }
            for (int b = 0; b < 2; b++) {
                float * e = erow(b, t) + off_xn;
                for (int j = 0; j < SGSEP_LAT_DIM; j++) e[j] = xn[j];
            }
        }

        sgcfm_forward(cfm, emb.data(), /*B=*/2, T, ti, est_out.data());

        for (int t = 0; t < T; t++) {
            const float * ou = &est_out[((size_t) 0 * T + t) * H + (H - SGSEP_LAT_DIM)];
            const float * oc = &est_out[((size_t) 1 * T + t) * H + (H - SGSEP_LAT_DIM)];
            float *       xn = &x_next[(size_t) t * SGSEP_LAT_DIM];
            for (int j = 0; j < SGSEP_LAT_DIM; j++) {
                float v = ou[j] + guidance * (oc[j] - ou[j]);
                xn[j] += dt * v;
            }
        }
    }

    // latents[:inc] = incontext_latents (overwrite, normalized space).
    for (int t = 0; t < incontext_length; t++) {
        const float * il = &incontext_latent[(size_t) t * SGSEP_LAT_DIM];
        float *       xn = &x_next[(size_t) t * SGSEP_LAT_DIM];
        for (int j = 0; j < SGSEP_LAT_DIM; j++) xn[j] = il[j];
    }

    latent_norm = x_next;  // time-major [T*64], normalized (pre return_sample)

    // return_sample: latent = x*std + mean. Output channel-major [64,T].
    latent_out.resize((size_t) SGSEP_LAT_DIM * T);
    for (int c = 0; c < SGSEP_LAT_DIM; c++) {
        float mean = sep->nf_mean[c];
        float std  = sep->nf_std[c];
        for (int t = 0; t < T; t++) {
            latent_out[(size_t) c * T + t] = x_next[(size_t) t * SGSEP_LAT_DIM + c] * std + mean;
        }
    }
}

// Decode vocal+bgm code streams (each [T]) -> latent[64*T] channel-major (lat[c*T + t]).
// Mirrors inference_codes for scenario=start_seg, no prompt (single segment).
static void sgsep_decode(SonggenSeptoken * sep, SonggenCfm * cfm, const std::vector<int> & codes_vocal,
                         const std::vector<int> & codes_bgm, uint64_t seed, std::vector<float> & latent_out) {
    std::vector<float> latent_norm;
    sgsep_decode_seg(sep, cfm, codes_vocal, codes_bgm, seed, /*incontext_length=*/0, {}, latent_norm, latent_out);
}
