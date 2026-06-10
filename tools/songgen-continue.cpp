// songgen-continue: audio-prompt CONTINUATION driver ("extend this clip").
//
//   The prompt clip seeds the START of the generation acoustically (CFM in-context freeze on the
//   VAE-encoded prompt latent) AND conditions the LM (prompt codes, same as clone). The model then
//   continues from the prompt's character; the prompt portion is trimmed from the output wav.
//
//   GGML_BACKEND=CPU ./songgen-continue \
//       <lelm.gguf> <cfm.gguf> <vae.gguf> <vae-encoder.gguf> <septoken-aux.gguf> <1rvq-aux.gguf> \
//       <musicfm-sep.gguf> <musicfm-1rvq.gguf> \
//       --vocal-stem v.wav --bgm-stem b.wav [--full-mix m.wav] \
//       --lyric "..." [--description "..."] <out.wav> [seed]
//
// Source of truth: generate_septoken.code2sound (175-285, the prompt_vocal/bgm branch) +
//   model_septoken.inference_codes (589-657, scenario='other_seg' in-context freeze).
//
// vs songgen-clone: clone steers ONLY via LM conditioning (decode skips true_latents). Continuation
//   ADDS the CFM in-context: prompt audio -> VAE.encode(vocal+bgm sum) -> project_sample -> true
//   latents seeded/frozen over the prompt frames of the first decode window, then trimmed out.
//
// PROMPT CLIPPING (code2sound:206-213): a <30s prompt is clipped to its first 10s before both the
//   VAE-encode and sound2code (matches the reference; a 10.24s prompt -> ~250 latent frames + ~256
//   prompt code frames). >=30s prompts use the 20-30s window. MVP: pre-separated stems (no demucs).
#include "audio-io.h"
#include "npy.h"
#include "songgen-cfm.h"
#include "songgen-clone-encode.h"
#include "songgen-decode-post.h"
#include "songgen-lelm-gen.h"
#include "songgen-septoken.h"
#include "songgen-tokenizer.h"
#include "songgen-vae.h"
#include "songgen-vae-enc.h"
#include "wav.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#ifdef SONGGEN_AS_LIB
namespace sgrun_continue {
#endif

static void stats(const char * tag, const float * a, size_t n) {
    double mn = 1e30, mx = -1e30, sum = 0, sq = 0;
    size_t n_bad = 0;
    for (size_t i = 0; i < n; i++) {
        float v = a[i];
        if (!std::isfinite(v)) { n_bad++; continue; }
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v;
        sq += (double) v * v;
    }
    double mean = sum / (double) n, var = sq / (double) n - mean * mean;
    printf("%-10s min=%+.5f max=%+.5f mean=%+.5f std=%.5f  (n=%zu nan/inf=%zu)\n", tag, mn, mx, mean,
           var > 0 ? sqrt(var) : 0.0, n, n_bad);
}

static float * read_stem(const char * path, int * T) {
    int     sr = 0;
    float * p  = audio_read(path, T, &sr);
    if (!p) { fprintf(stderr, "[cont] stem read failed: %s\n", path); exit(1); }
    if (sr != 48000) {
        int     n2 = 0;
        float * r  = audio_resample(p, *T, sr, 48000, 2, &n2);
        free(p);
        p  = r;
        *T = n2;
        fprintf(stderr, "[cont] resampled %s %d->48000 Hz\n", path, sr);
    }
    return p;
}

#ifdef SONGGEN_AS_LIB
int run(int argc, char ** argv) {
#else
int main(int argc, char ** argv) {
#endif
    std::string lyric, description, vocal_path, bgm_path, full_path, gen_type_s = "mixed";
    bool        have_lyric = false, have_desc = false;
    // Sampling/fade defaults live ONLY in SgGenParams (songgen-decode-post.h).
    const SgGenParams sgdef;
    float       duration_sec = -1.0f, arg_temp = sgdef.temp, arg_cfg = sgdef.cfg_coef, arg_fade = sgdef.fade_ms;
    int         frames_arg = -1, arg_topk = sgdef.top_k;
    std::vector<char *> pos;
    for (int i = 0; i < argc; i++) {
        std::string a = i > 0 ? argv[i] : "";
        if (a == "--lyric" && i + 1 < argc) { lyric = argv[++i]; have_lyric = true; }
        else if (a == "--description" && i + 1 < argc) { description = argv[++i]; have_desc = true; }
        else if (a == "--vocal-stem" && i + 1 < argc) { vocal_path = argv[++i]; }
        else if (a == "--bgm-stem" && i + 1 < argc) { bgm_path = argv[++i]; }
        else if (a == "--full-mix" && i + 1 < argc) { full_path = argv[++i]; }
        else if (a == "--duration" && i + 1 < argc) { duration_sec = (float) atof(argv[++i]); }
        else if (a == "--frames" && i + 1 < argc) { frames_arg = atoi(argv[++i]); }
        else if (a == "--temp" && i + 1 < argc) { arg_temp = (float) atof(argv[++i]); }
        else if (a == "--top-k" && i + 1 < argc) { arg_topk = atoi(argv[++i]); }
        else if (a == "--cfg" && i + 1 < argc) { arg_cfg = (float) atof(argv[++i]); }
        else if (a == "--fade" && i + 1 < argc) { arg_fade = (float) atof(argv[++i]); }
        else if (a == "--gen-type" && i + 1 < argc) { gen_type_s = argv[++i]; }
        else pos.push_back(argv[i]);
    }
    argc = (int) pos.size();
    for (int i = 0; i < argc; i++) argv[i] = pos[i];

    if (argc < 10 || vocal_path.empty() || bgm_path.empty() || !have_lyric) {
        fprintf(stderr,
                "usage: %s <lelm.gguf> <cfm.gguf> <vae.gguf> <vae-encoder.gguf> <septoken-aux.gguf> <1rvq-aux.gguf> "
                "<musicfm-sep.gguf> <musicfm-1rvq.gguf> <out.wav> [seed]\n"
                "  --vocal-stem <wav> --bgm-stem <wav> [--full-mix <wav>] --lyric \"...\" [--description \"...\"]\n"
                "  [--duration <sec>] [--frames <N>] [--temp <f>] [--top-k <n>] [--cfg <f>] [--fade <ms>]"
                " [--gen-type mixed|vocal|bgm]\n",
                argv[0]);
        return 1;
    }
    SgGenType gtype = SG_MIXED;
    if (!sgpost_parse_gen_type(gen_type_s, &gtype)) {
        fprintf(stderr, "[cont] bad --gen-type '%s'\n", gen_type_s.c_str());
        return 1;
    }
    const char * lelm_path = argv[1];
    const char * cfm_path  = argv[2];
    const char * vae_path  = argv[3];
    const char * vaeenc_path = argv[4];
    const char * aux_path  = argv[5];
    const char * aux1_path = argv[6];
    const char * mf_sep    = argv[7];
    const char * mf_1rvq   = argv[8];
    const char * out_path  = argv[9];
    uint64_t     seed      = (argc > 10) ? (uint64_t) strtoull(argv[10], NULL, 10) : 1234ull;

    if (!have_desc) description = "";

    // ---- 1. load + read prompt stems (48k stereo planar [2*n]) ----
    SonggenSeptoken sep;
    if (!sgsep_load(&sep, aux_path)) { fprintf(stderr, "[cont] aux load failed\n"); return 1; }
    SonggenCloneEnc enc;
    if (!sgclone_load(&enc, mf_sep, mf_1rvq, aux1_path, &sep)) { fprintf(stderr, "[cont] clone-enc load failed\n"); return 1; }

    int     nv = 0, nb = 0, nf = 0;
    float * vstem = read_stem(vocal_path.c_str(), &nv);
    float * bstem = read_stem(bgm_path.c_str(), &nb);
    float * fmix  = full_path.empty() ? nullptr : read_stem(full_path.c_str(), &nf);

    // ---- prompt clip window (code2sound:206-213): <30s -> first 10s; else 20-30s. Applied to the
    //      stems used for BOTH the VAE-encode and sound2code so latent/codes stay aligned. ----
    const int SR  = 48000;
    int       common = std::min(nv, nb);
    int       clip_beg, clip_len;
    if (common < 30 * SR) {
        clip_beg = 0;
        clip_len = std::min(common, 10 * SR);
    } else {
        clip_beg = 20 * SR;
        clip_len = std::min(common - clip_beg, 10 * SR);
    }
    fprintf(stderr, "[cont] prompt clip: [%.2f, %.2f]s of %.2fs\n", (float) clip_beg / SR,
            (float) (clip_beg + clip_len) / SR, (float) common / SR);

    // clipped planar stems for the encoders.
    std::vector<float> vclip((size_t) 2 * clip_len), bclip((size_t) 2 * clip_len);
    for (int t = 0; t < clip_len; t++) {
        vclip[t]            = vstem[clip_beg + t];
        vclip[clip_len + t] = vstem[nv + clip_beg + t];
        bclip[t]            = bstem[clip_beg + t];
        bclip[clip_len + t] = bstem[nb + clip_beg + t];
    }
    std::vector<float> fclip;
    if (fmix) {
        int fcommon = std::min(nf - clip_beg, clip_len);
        if (fcommon < 0) fcommon = 0;
        fclip.assign((size_t) 2 * fcommon, 0.0f);
        for (int t = 0; t < fcommon; t++) {
            fclip[t]            = fmix[clip_beg + t];
            fclip[fcommon + t]  = fmix[nf + clip_beg + t];
        }
    }
    free(vstem); free(bstem); if (fmix) free(fmix);

    // ---- 2. prompt codes (sound2code) — vocal/bgm streams used for LM conditioning AND prepended
    //      to the gen codes for the in-context decode. pmt = LM conditioner only. ----
    std::vector<int> pmt_codes, vocal_codes, bgm_codes;
    int Tp = sgclone_encode(&enc, vclip.data(), clip_len, bclip.data(), clip_len,
                            fclip.empty() ? nullptr : fclip.data(), fclip.empty() ? 0 : (int) (fclip.size() / 2),
                            SR, pmt_codes, vocal_codes, bgm_codes);
    fprintf(stderr, "[cont] prompt tokens: %d frames/stream\n", Tp);
    sgclone_free(&enc);

    // ---- 3. true_latents: VAE.encode(vocal+bgm SUMMED, 48k stereo) -> project_sample (code2sound:215-217) ----
    SonggenVaeEnc venc;
    if (!sgve_load(&venc, vaeenc_path)) { fprintf(stderr, "[cont] vae-encoder load failed\n"); return 1; }
    std::vector<float> sum_audio((size_t) 2 * clip_len);
    for (int i = 0; i < 2 * clip_len; i++) sum_audio[i] = vclip[i] + bclip[i];  // channel-major [2,clip_len]
    std::vector<float> true_lat_cm;
    int                Tlat = sgve_encode(&venc, sum_audio.data(), clip_len, true_lat_cm);  // [64*Tlat] channel-major
    sgve_free(&venc);
    std::vector<float> prompt_latent_norm;
    sgsep_project_sample(&sep, true_lat_cm, Tlat, prompt_latent_norm);  // [Tlat*64] time-major, normalized
    fprintf(stderr, "[cont] true_latents: %d frames (VAE-encoded prompt sum), project_sample applied\n", Tlat);

    // ---- 4. LeLM generation with prompt conditioning (identical to clone) ----
    SonggenLeLM m;
    if (!sglm_load(&m, lelm_path)) { fprintf(stderr, "[cont] lelm load failed\n"); return 1; }
    const SonggenConfig & c    = m.cfg;
    const int             card = c.card;
    const int             SP   = c.special_token;  // 16385
    const int             EOS  = c.code_size;      // 16384
    const int             K    = c.code_depth;
    const int             audN = c.audio_len - 1;  // 251
    const int             delays[3] = { 0, 250, 250 };

    SgGenParams P;
    if (frames_arg > 0)           P.max_gen_len = frames_arg;
    else if (duration_sec > 0.0f) P.max_gen_len = (int) lroundf(duration_sec * 25.0f);
    else                          P.max_gen_len = SGGEN_MODEL_MAX_FRAMES;  // 270 s ceiling, EOS-terminated (was 375=15s)
    if (P.max_gen_len < 1) P.max_gen_len = 1;
    P.top_k    = arg_topk;
    P.temp     = arg_temp;
    P.cfg_coef = arg_cfg;
    P.fade_ms  = arg_fade;
    P.gen_type = gtype;

    const int   max_gen_len = P.max_gen_len;
    const int   top_k       = P.top_k;
    const float temp        = P.temp;
    const float cfg_coef    = P.cfg_coef;
    const float rep_penalty = P.rep_penalty;
    const int   record_win  = P.record_win;
    fprintf(stderr, "[cont] max_gen_len=%d (%.1fs) top_k=%d temp=%.2f cfg=%.2f fade=%.0fms gen_type=%s\n", max_gen_len,
            (float) max_gen_len / 25.0f, top_k, temp, cfg_coef, P.fade_ms, sgpost_gen_type_name(gtype));

    songgen_tok::Tokenizer tok;
    if (!songgen_tok::load(&tok, lelm_path)) { fprintf(stderr, "[cont] tokenizer load failed\n"); return 1; }
    songgen_tok::ConditioningArrays ca;
    songgen_tok::CondGenType        cgt = gtype == SG_BGM     ? songgen_tok::COND_BGM
                                          : gtype == SG_VOCAL ? songgen_tok::COND_VOCAL
                                                              : songgen_tok::COND_MIXED;
    songgen_tok::build(&tok, lyric, description, &ca, cgt);
    std::vector<int32_t> d_ids_c(ca.desc_ids_cond.begin(), ca.desc_ids_cond.end());
    std::vector<int32_t> d_cov_c(ca.desc_cover_cond.begin(), ca.desc_cover_cond.end());
    std::vector<int32_t> d_ids_u(ca.desc_ids_uncond.begin(), ca.desc_ids_uncond.end());
    std::vector<int32_t> d_cov_u(ca.desc_cover_uncond.begin(), ca.desc_cover_uncond.end());
    std::vector<int32_t> t_ids_c(ca.type_ids_cond.begin(), ca.type_ids_cond.end());
    std::vector<int32_t> t_ids_u(ca.type_ids_uncond.begin(), ca.type_ids_uncond.end());

    std::vector<int32_t> aud_pmt_c, aud_voc_c, aud_bgm_c;
    sgclone_build_audio_ids(pmt_codes, audN, EOS, SP, aud_pmt_c);
    sgclone_build_audio_ids(vocal_codes, audN, EOS, SP, aud_voc_c);
    sgclone_build_audio_ids(bgm_codes, audN, EOS, SP, aud_bgm_c);
    std::vector<int32_t> aud_all_sp(audN, SP);

    std::vector<char> ignore(card, 0);
    for (int v : pmt_codes) if (v >= 0 && v < EOS) ignore[v] = 1;

    int max_delay = 250;
    int S         = 1 + (max_gen_len + max_delay);
    std::vector<std::vector<int>>  gen_seq(K, std::vector<int>(S, SP));
    std::vector<std::vector<char>> valid(K, std::vector<char>(S, 0));
    for (int p = 1; p < S; p++) {
        int t = p - 1;
        for (int q = 0; q < K; q++) {
            int tq = t - delays[q];
            if (tq >= 0 && tq < max_gen_len) { gen_seq[q][p] = -1; valid[q][p] = 1; }
        }
    }
    int start_offset = 1;

    SonggenLeLMGen g;
    sggen_alloc_kv(&g, &m, c.desc_len + c.audio_len + c.type_len + S + 4, 2);

    std::mt19937_64 rng(seed);
    auto            uni = std::uniform_real_distribution<double>(0.0, 1.0);

    std::vector<float> lg_c[3], lg_u[3], comb[3];
    for (int k = 0; k < 3; k++) { lg_c[k].resize(card); lg_u[k].resize(card); comb[k].resize(card); }
    std::vector<std::array<int, 3>> token_pool;

    bool is_end[3] = { false, false, false };
    int  eos_step[3] = { -1, -1, -1 };
    bool all_end = false;
    int  last_offset = start_offset;

    for (int offset = start_offset; offset < S && !all_end; offset++) {
        bool prepend = (offset == start_offset);
        int  prev    = prepend ? 0 : (offset - 1);

        SonggenCondInput in_c, in_u;
        if (prepend) {
            in_c.desc_ids = d_ids_c; in_c.desc_cover = d_cov_c; in_c.type_ids = t_ids_c;
            in_u.desc_ids = d_ids_u; in_u.desc_cover = d_cov_u; in_u.type_ids = t_ids_u;
            in_c.audio_ids_pmt = aud_pmt_c; in_c.audio_ids_vocal = aud_voc_c; in_c.audio_ids_bgm = aud_bgm_c;
            in_u.audio_ids = aud_all_sp;
            for (int q = 0; q < K; q++) {
                std::vector<int32_t> * dc = (q == 0 ? &in_c.seq0 : q == 1 ? &in_c.seq1 : &in_c.seq2);
                std::vector<int32_t> * du = (q == 0 ? &in_u.seq0 : q == 1 ? &in_u.seq1 : &in_u.seq2);
                *dc = { gen_seq[q][0] };
                *du = { gen_seq[q][0] };
            }
        } else {
            in_c.seq0 = { gen_seq[0][prev] }; in_c.seq1 = { gen_seq[1][prev] }; in_c.seq2 = { gen_seq[2][prev] };
            in_u.seq0 = in_c.seq0; in_u.seq1 = in_c.seq1; in_u.seq2 = in_c.seq2;
        }

        static const bool no_batch = getenv("SGLM_NO_BATCH") != NULL;
        if (prepend || no_batch) {
            sggen_step(&g, 0, in_c, prepend, lg_c[0].data(), lg_c[1].data(), lg_c[2].data());
            sggen_step(&g, 1, in_u, prepend, lg_u[0].data(), lg_u[1].data(), lg_u[2].data());
        } else {
            std::vector<SonggenCondInput> ins = { in_c, in_u };
            std::vector<float>            b0(2 * card), b1(2 * card), b2(2 * card);
            sggen_step_batch(&g, ins, b0.data(), b1.data(), b2.data());
            for (int v = 0; v < card; v++) {
                lg_c[0][v] = b0[v];        lg_u[0][v] = b0[card + v];
                lg_c[1][v] = b1[v];        lg_u[1][v] = b1[card + v];
                lg_c[2][v] = b2[v];        lg_u[2][v] = b2[card + v];
            }
        }

        for (int q = 0; q < K; q++)
            for (int v = 0; v < card; v++) comb[q][v] = lg_u[q][v] + (lg_c[q][v] - lg_u[q][v]) * cfg_coef;

        if (!token_pool.empty()) {
            int wn = (int) token_pool.size();
            int lo = std::max(0, wn - record_win);
            int tmp = card - 1;
            for (int q = 0; q < K; q++) {
                std::vector<char> seen(tmp, 0);
                for (int i = lo; i < wn; i++) { int t2 = token_pool[i][q]; if (t2 >= 0 && t2 < tmp) seen[t2] = 1; }
                for (int v = 0; v < tmp; v++) if (seen[v]) comb[q][v] /= rep_penalty;
            }
        }

        for (int v = 0; v < card; v++) if (ignore[v]) comb[0][v] = -INFINITY;

        int next[3];
        {
            std::vector<int> idx(card);
            for (int v = 0; v < card; v++) idx[v] = v;
            std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                              [&](int a, int b) { return comb[0][a] > comb[0][b]; });
            double mx = -1e30;
            for (int j = 0; j < top_k; j++) mx = std::max(mx, (double) comb[0][idx[j]]);
            double sum = 0;
            std::vector<double> pr(top_k);
            for (int j = 0; j < top_k; j++) { pr[j] = exp(((double) comb[0][idx[j]] - mx) / temp); sum += pr[j]; }
            double r = uni(rng) * sum, acc = 0;
            int    pick = idx[top_k - 1];
            for (int j = 0; j < top_k; j++) { acc += pr[j]; if (r <= acc) { pick = idx[j]; break; } }
            next[0] = pick;
        }
        for (int q = 1; q < K; q++) {
            int best = 0; float bv = comb[q][0];
            for (int v = 1; v < card; v++) if (comb[q][v] > bv) { bv = comb[q][v]; best = v; }
            next[q] = best;
        }

        token_pool.push_back({ next[0], next[1], next[2] });

        for (int q = 0; q < K; q++) { if (!valid[q][offset]) next[q] = SP; if (is_end[q]) next[q] = SP; }
        for (int q = 0; q < K; q++) if (!is_end[q] && next[q] == EOS) { is_end[q] = true; eos_step[q] = offset; }
        for (int q = 0; q < K; q++) if (gen_seq[q][offset] == -1) gen_seq[q][offset] = next[q];

        all_end = is_end[0] && is_end[1] && is_end[2];
        last_offset = offset;
    }

    int gen_len = last_offset + 1;
    fprintf(stderr, "[cont] generation finished: offset %d (gen_seq len %d)\n", last_offset, gen_len);

    std::vector<int64_t> out_codes((size_t) K * max_gen_len, SP);
    for (int q = 0; q < K; q++)
        for (int t = 0; t < max_gen_len; t++) {
            int s = t + delays[q] + 1;
            int val = (s < gen_len) ? gen_seq[q][s] : -1;
            out_codes[(size_t) q * max_gen_len + t] = (val == -1) ? EOS : val;
        }

    sggen_free_kv(&g);
    sglm_free(&m);

    // ---- 5. continuation decode: prepend prompt codes, in-context freeze on true_latents, trim ----
    const int64_t * vocal = &out_codes[(size_t) 1 * max_gen_len];
    const int64_t * bgm   = &out_codes[(size_t) 2 * max_gen_len];
    int             Tv    = max_gen_len;
    while (Tv > 0) {
        int64_t cv = vocal[Tv - 1], cb = bgm[Tv - 1];
        if (cv >= 0 && cv < SGSEP_CODEBOOK && cb >= 0 && cb < SGSEP_CODEBOOK) break;
        Tv--;
    }
    std::vector<int> gen_vocal(Tv), gen_bgm(Tv);
    for (int t = 0; t < Tv; t++) {
        int64_t cv = vocal[t], cb = bgm[t];
        if (cv < 0 || cv >= SGSEP_CODEBOOK) cv = cv < 0 ? 0 : SGSEP_CODEBOOK - 1;
        if (cb < 0 || cb >= SGSEP_CODEBOOK) cb = cb < 0 ? 0 : SGSEP_CODEBOOK - 1;
        gen_vocal[t] = (int) cv; gen_bgm[t] = (int) cb;
    }
    fprintf(stderr, "[cont] decode: %d generated frames\n", Tv);
    if (Tv <= 0) { fprintf(stderr, "[cont] no valid generated frames\n"); return 1; }

    SonggenCfm cfm;
    if (!sgcfm_load(&cfm, cfm_path)) { fprintf(stderr, "[cont] cfm load failed\n"); return 1; }
    SonggenVae vae;
    if (!sgvae_load(&vae, vae_path)) { fprintf(stderr, "[cont] vae load failed\n"); return 1; }

    std::vector<float> audio;
    int T_audio = sgpost_decode_continue(&sep, &cfm, &vae, gen_vocal, gen_bgm, vocal_codes, bgm_codes,
                                         prompt_latent_norm, Tlat, seed, P, audio);
    stats("wav", audio.data(), audio.size());

    sgpost_fade(audio.data(), T_audio, SR, P.fade_ms, std::min(10.0f, P.fade_ms));
    if (!write_wav_s16_planar(out_path, audio.data(), T_audio, SR)) {
        fprintf(stderr, "[cont] wav write failed\n");
        return 1;
    }
    double dur = (double) T_audio / SR;
    printf("---- songgen continue ----\n");
    printf("prompt code frames  : %d (prepended, trimmed from output)\n", Tp);
    printf("prompt latent frames: %d (VAE-encoded, in-context freeze)\n", Tlat);
    printf("seed                : %llu\n", (unsigned long long) seed);
    printf("generated frames    : %d\n", Tv);
    printf("samples/channel     : %d (2ch) duration %.3f s @ %d Hz\n", T_audio, dur, SR);
    printf("output              : %s\n", out_path);

    sgvae_free(&vae);
    sgcfm_free(&cfm);
    return 0;
}

#ifdef SONGGEN_AS_LIB
}  // namespace sgrun_continue
#endif
