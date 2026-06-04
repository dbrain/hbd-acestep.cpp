// songgen-generate: full SongGeneration GATE-2 driver.
//   LeLM KV-cache autoregressive generation (lyric+style conditioning) -> gen_tokens[1,3,T]
//   -> separate-tokenizer / CFM ODE / Stable-Audio VAE decode -> wav.
//
//   GGML_BACKEND=CPU ./songgen-generate
//       <lelm.gguf> <cfm.gguf> <vae.gguf> <septoken-aux.gguf> <golden-large-dir> <out.wav> [seed]
//
// Conditioning ids are loaded from <golden-large-dir> (desc_ids/desc_cover/type_ids _{cond,uncond},
// audio_ids = all 16385). Faithful port of lm_levo.LmModel.generate()/_sample_next_token().
#include "npy.h"
#include "songgen-cfm.h"
#include "songgen-decode-post.h"
#include "songgen-lelm-gen.h"
#include "songgen-septoken.h"
#include "songgen-tokenizer.h"
#include "songgen-vae.h"
#include "wav.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

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

int main(int argc, char ** argv) {
    // Optional flags: --lyric "..." --description "..." (tokenize in C++ instead of
    // loading golden npy ids). Strip them out, leaving positional args intact.
    std::string lyric, description, gen_type_s = "mixed";
    bool        have_lyric = false, have_desc = false;
    float       duration_sec = -1.0f;  // <0 -> use frames / default
    int         frames_arg   = -1;
    float       arg_temp = 1.0f, arg_cfg = 1.5f, arg_fade = 200.0f;
    int         arg_topk = 250;
    {
        std::vector<char *> pos;
        for (int i = 0; i < argc; i++) {
            std::string a = i > 0 ? argv[i] : "";
            if (a == "--lyric" && i + 1 < argc) { lyric = argv[++i]; have_lyric = true; }
            else if (a == "--description" && i + 1 < argc) { description = argv[++i]; have_desc = true; }
            else if (a == "--duration" && i + 1 < argc) { duration_sec = (float) atof(argv[++i]); }
            else if (a == "--frames" && i + 1 < argc) { frames_arg = atoi(argv[++i]); }
            else if (a == "--temp" && i + 1 < argc) { arg_temp = (float) atof(argv[++i]); }
            else if (a == "--top-k" && i + 1 < argc) { arg_topk = atoi(argv[++i]); }
            else if (a == "--cfg" && i + 1 < argc) { arg_cfg = (float) atof(argv[++i]); }
            else if (a == "--fade" && i + 1 < argc) { arg_fade = (float) atof(argv[++i]); }
            else if (a == "--gen-type" && i + 1 < argc) { gen_type_s = argv[++i]; }
            else { pos.push_back(argv[i]); }
        }
        argc = (int) pos.size();
        for (int i = 0; i < argc; i++) argv[i] = pos[i];
    }
    if (argc < 7) {
        fprintf(stderr,
                "usage: %s <lelm.gguf> <cfm.gguf> <vae.gguf> <septoken-aux.gguf> <golden-large-dir> <out.wav> "
                "[seed]\n"
                "  [--lyric \"...\"] [--description \"...\"] [--duration <sec>] [--frames <N>]\n"
                "  [--temp <f>] [--top-k <n>] [--cfg <f>] [--fade <ms>] [--gen-type mixed|vocal|bgm]\n"
                "  When --lyric and --description are given, conditioning ids are tokenized in C++.\n"
                "  Otherwise they are loaded from <golden-large-dir>/*.npy.\n",
                argv[0]);
        return 1;
    }
    SgGenType gtype = SG_MIXED;
    if (!sgpost_parse_gen_type(gen_type_s, &gtype)) {
        fprintf(stderr, "[gen] bad --gen-type '%s' (expect mixed|vocal|bgm)\n", gen_type_s.c_str());
        return 1;
    }
    const char * lelm_path = argv[1];
    const char * cfm_path  = argv[2];
    const char * vae_path  = argv[3];
    const char * aux_path  = argv[4];
    std::string  gd        = argv[5];
    const char * out_path  = argv[6];
    uint64_t     seed      = (argc > 7) ? (uint64_t) strtoull(argv[7], NULL, 10) : 1234ull;
    auto         gp        = [&](const char * f) { return gd + "/" + f; };
    const bool   tok_cpp   = have_lyric && have_desc;

    // generation hyperparameters (lm_levo defaults / spec)
    SgGenParams P;
    // No --duration/--frames given → generate to the model's max length (270 s) and let EOS stop
    // the song naturally (upstream behaviour). --duration acts as an explicit hard cap (truncate)
    // when you want to bound length/VRAM; growing KV keeps the footprint proportional to the actual
    // (EOS-determined) length regardless of how high the ceiling is.
    if (frames_arg > 0)           P.max_gen_len = frames_arg;
    else if (duration_sec > 0.0f) P.max_gen_len = (int) lroundf(duration_sec * 25.0f);
    else                          P.max_gen_len = SGGEN_MODEL_MAX_FRAMES;  // 270 s, EOS-terminated
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

    fprintf(stderr,
            "[gen] seed=%llu max_gen_len=%d (%.1fs) top_k=%d temp=%.2f cfg=%.2f rep=%.1f win=%d fade=%.0fms gen_type=%s\n",
            (unsigned long long) seed, max_gen_len, (float) max_gen_len / 25.0f, top_k, temp, cfg_coef, rep_penalty,
            record_win, P.fade_ms, sgpost_gen_type_name(gtype));

    SonggenLeLM m;
    if (!sglm_load(&m, lelm_path)) {
        fprintf(stderr, "[gen] lelm load failed\n");
        return 1;
    }
    const SonggenConfig & c    = m.cfg;
    const int             card = c.card;       // 16385
    const int             SP   = c.special_token;  // 16385
    const int             EOS  = c.code_size;  // 16384
    const int             K    = c.code_depth;
    const int             delays[3] = { 0, 250, 250 };

    // ---- conditioning ids ----
    std::vector<int32_t> d_ids_c, d_cov_c, d_ids_u, d_cov_u, t_ids_c, t_ids_u;
    if (tok_cpp) {
        songgen_tok::Tokenizer tok;
        if (!songgen_tok::load(&tok, lelm_path)) {
            fprintf(stderr, "[gen] tokenizer load failed\n");
            return 1;
        }
        songgen_tok::ConditioningArrays a;
        songgen_tok::CondGenType        cgt = gtype == SG_BGM     ? songgen_tok::COND_BGM
                                              : gtype == SG_VOCAL ? songgen_tok::COND_VOCAL
                                                                  : songgen_tok::COND_MIXED;
        songgen_tok::build(&tok, lyric, description, &a, cgt);
        d_ids_c.assign(a.desc_ids_cond.begin(), a.desc_ids_cond.end());
        d_cov_c.assign(a.desc_cover_cond.begin(), a.desc_cover_cond.end());
        d_ids_u.assign(a.desc_ids_uncond.begin(), a.desc_ids_uncond.end());
        d_cov_u.assign(a.desc_cover_uncond.begin(), a.desc_cover_uncond.end());
        t_ids_c.assign(a.type_ids_cond.begin(), a.type_ids_cond.end());
        t_ids_u.assign(a.type_ids_uncond.begin(), a.type_ids_uncond.end());
        fprintf(stderr, "[gen] conditioning: tokenized in C++ from --lyric/--description\n");
    } else {
        d_ids_c = npy_load(gp("desc_ids_cond.npy").c_str()).i32;
        d_cov_c = npy_load(gp("desc_cover_cond.npy").c_str()).i32;
        d_ids_u = npy_load(gp("desc_ids_uncond.npy").c_str()).i32;
        d_cov_u = npy_load(gp("desc_cover_uncond.npy").c_str()).i32;
        t_ids_c = npy_load(gp("type_ids_cond.npy").c_str()).i32;
        t_ids_u = npy_load(gp("type_ids_uncond.npy").c_str()).i32;
        fprintf(stderr, "[gen] conditioning: loaded golden npy ids from %s\n", gd.c_str());
    }
    std::vector<int32_t> audio_ids(c.audio_len - 1, SP);

    // ---- pattern (DelayedPatternProvider, delays=[0,250,250]) ----
    int max_delay = 250;
    int S         = 1 + (max_gen_len + max_delay);  // layout length = gen_sequence length
    // gen_sequence[K][S]: -1 (unknown) at valid coords, SP at invalid. valid_mask[K][S].
    std::vector<std::vector<int>>  gen_seq(K, std::vector<int>(S, SP));
    std::vector<std::vector<char>> valid(K, std::vector<char>(S, 0));
    for (int p = 1; p < S; p++) {
        int t = p - 1;  // pattern timestep
        for (int q = 0; q < K; q++) {
            int tq = t - delays[q];
            if (tq >= 0 && tq < max_gen_len) {
                gen_seq[q][p] = -1;  // unknown, to be generated
                valid[q][p]   = 1;
            }
        }
    }
    int start_offset = 1;  // first step with timestep 0 (cb0 delay 0 -> pattern pos 1)

    // ---- KV cache: cond=set0, uncond=set1 ----
    // Dynamic/growing KV: cap is the full requested ceiling (prefix + pattern length), but we
    // START small (prefix + conditioning + delay headroom) and grow geometrically only as the
    // song actually generates. A high duration ceiling (for natural EOS / long songs) therefore
    // costs only what's produced, not the ceiling — the binding VRAM term tracks real length.
    SonggenLeLMGen g;
    int cap_seq  = c.desc_len + c.audio_len + c.type_len + S + 4;
    int init_seq = c.desc_len + c.audio_len + c.type_len + max_delay + 1 + 4;  // prefill + delay headroom
    if (getenv("SG_KV_STATIC")) init_seq = cap_seq;  // A/B: force one-shot alloc (no growth)
    sggen_alloc_kv_dyn(&g, &m, init_seq, cap_seq, /*n_sets=*/2);

    // ---- RNG ----
    std::mt19937_64 rng(seed);
    auto            uni = std::uniform_real_distribution<double>(0.0, 1.0);

    std::vector<float> lg_c[3], lg_u[3], comb[3];
    for (int k = 0; k < 3; k++) {
        lg_c[k].resize(card);
        lg_u[k].resize(card);
        comb[k].resize(card);
    }

    // rep-penalty sliding window of sampled [K] vectors (record_win newest entries)
    std::vector<std::array<int, 3>> token_pool;

    bool is_end[3]   = { false, false, false };
    int  eos_step[3] = { -1, -1, -1 };
    bool all_end     = false;
    int  last_offset = start_offset;

    for (int offset = start_offset; offset < S && !all_end; offset++) {
        bool prepend = (offset == start_offset);
        int  prev    = prepend ? 0 : (offset - 1);

        // the token(s) fed this step = gen_seq[:, prev:offset]
        SonggenCondInput in_c, in_u;
        if (prepend) {
            in_c.desc_ids = d_ids_c; in_c.desc_cover = d_cov_c; in_c.type_ids = t_ids_c;
            in_u.desc_ids = d_ids_u; in_u.desc_cover = d_cov_u; in_u.type_ids = t_ids_u;
            in_c.audio_ids = audio_ids;  in_u.audio_ids = audio_ids;
            // code position 0 = gen_seq[:,0] = all special
            for (int q = 0; q < K; q++) {
                std::vector<int32_t> * dst_c = (q == 0 ? &in_c.seq0 : q == 1 ? &in_c.seq1 : &in_c.seq2);
                std::vector<int32_t> * dst_u = (q == 0 ? &in_u.seq0 : q == 1 ? &in_u.seq1 : &in_u.seq2);
                *dst_c = { gen_seq[q][0] };
                *dst_u = { gen_seq[q][0] };
            }
        } else {
            in_c.seq0 = { gen_seq[0][prev] }; in_c.seq1 = { gen_seq[1][prev] }; in_c.seq2 = { gen_seq[2][prev] };
            in_u.seq0 = in_c.seq0; in_u.seq1 = in_c.seq1; in_u.seq2 = in_c.seq2;
        }

        static const bool no_batch = getenv("SGLM_NO_BATCH") != NULL;
        if (prepend || no_batch) {
            // step 0: cond/uncond conditioning differs -> two separate prefill forwards.
            // (SGLM_NO_BATCH forces the legacy two-forward path for baseline measurement.)
            sggen_step(&g, 0, in_c, prepend, lg_c[0].data(), lg_c[1].data(), lg_c[2].data());
            sggen_step(&g, 1, in_u, prepend, lg_u[0].data(), lg_u[1].data(), lg_u[2].data());
        } else {
            // decode: cond+uncond batched into ONE forward (N=2). Logits laid out [card, N] row-major.
            std::vector<SonggenCondInput> ins = { in_c, in_u };
            std::vector<float>            b0(2 * card), b1(2 * card), b2(2 * card);
            sggen_step_batch(&g, ins, b0.data(), b1.data(), b2.data());
            for (int v = 0; v < card; v++) {
                lg_c[0][v] = b0[v];        lg_u[0][v] = b0[card + v];
                lg_c[1][v] = b1[v];        lg_u[1][v] = b1[card + v];
                lg_c[2][v] = b2[v];        lg_u[2][v] = b2[card + v];
            }
        }

        // CFG combine: logits = uncond + (cond-uncond)*cfg
        for (int q = 0; q < K; q++)
            for (int v = 0; v < card; v++) comb[q][v] = lg_u[q][v] + (lg_c[q][v] - lg_u[q][v]) * cfg_coef;

        // rep penalty over sliding window (last record_win sampled vectors)
        if (!token_pool.empty()) {
            int wn  = (int) token_pool.size();
            int lo  = std::max(0, wn - record_win);
            int tmp = std::min(card, card - 1);  // = card-1 (16384), excludes EOS id col
            for (int q = 0; q < K; q++) {
                std::vector<char> seen(tmp, 0);
                for (int i = lo; i < wn; i++) {
                    int tok = token_pool[i][q];
                    if (tok >= 0 && tok < tmp) seen[tok] = 1;  // bincount(unique(window))
                }
                for (int v = 0; v < tmp; v++)
                    if (seen[v]) comb[q][v] /= rep_penalty;  // q_count==1 for present ids -> /1.1
            }
        }

        // SGLM_EOS_DEBUG=1: per-step post-CFG EOS(16384) diagnostics per codebook.
        // Reports softmax prob (over full card, temp=1), rank (1=argmax), and whether EOS
        // is inside top_k for cb0. Additive, env-gated; does not affect sampling.
        static const bool eos_dbg = getenv("SGLM_EOS_DEBUG") != NULL;
        if (eos_dbg) {
            for (int q = 0; q < K; q++) {
                double mx = -1e30;
                for (int v = 0; v < card; v++) mx = std::max(mx, (double) comb[q][v]);
                double sum = 0.0;
                for (int v = 0; v < card; v++) sum += exp((double) comb[q][v] - mx);
                double p_eos = exp((double) comb[q][EOS] - mx) / sum;
                int    rank  = 1;
                for (int v = 0; v < card; v++)
                    if (comb[q][v] > comb[q][EOS]) rank++;
                bool in_topk = (q != 0) ? (rank == 1) : (rank <= top_k);
                fprintf(stderr,
                        "[EOS-DBG] off=%d t=%.2fs cb%d  logit_eos=%+.3f logit_max=%+.3f  p_eos=%.3e  rank=%d/%d  in_top%s=%d  valid=%d\n",
                        offset, (float) (offset - start_offset) / 25.0f, q, comb[q][EOS], (float) mx, p_eos, rank,
                        card, q == 0 ? "k" : "1", (int) in_topk, (int) valid[q][offset]);
            }
        }

        // sample next token per codebook
        int next[3];
        // cb0: top_k(250) on softmax(logits/temp), multinomial
        {
            std::vector<int> idx(card);
            for (int v = 0; v < card; v++) idx[v] = v;
            std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                              [&](int a, int b) { return comb[0][a] > comb[0][b]; });
            // softmax over the top_k (temp=1) then sample
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
        // cb1, cb2: argmax (top_k=1 on softmax == argmax)
        for (int q = 1; q < K; q++) {
            int    best = 0;
            float  bv   = comb[q][0];
            for (int v = 1; v < card; v++)
                if (comb[q][v] > bv) { bv = comb[q][v]; best = v; }
            next[q] = best;
        }

        // record raw sampled tokens (before masking) for rep window
        token_pool.push_back({ next[0], next[1], next[2] });

        // valid_mask: force special where pattern pos invalid for codebook q
        for (int q = 0; q < K; q++) {
            if (!valid[q][offset]) next[q] = SP;
            if (is_end[q]) next[q] = SP;  // already ended -> special
        }
        // update is_end (a codebook ends once it samples EOS)
        for (int q = 0; q < K; q++) {
            if (!is_end[q] && next[q] == EOS) { is_end[q] = true; eos_step[q] = offset; }
        }

        // write only over unknown (-1) positions
        for (int q = 0; q < K; q++)
            if (gen_seq[q][offset] == -1) gen_seq[q][offset] = next[q];

        all_end     = is_end[0] && is_end[1] && is_end[2];
        last_offset = offset;
    }

    int gen_len = last_offset + 1;  // gen_sequence trimmed to [..:last_offset+1]
    fprintf(stderr, "[gen] generation finished: offset reached %d (gen_seq len %d / %d)\n", last_offset, gen_len, S);
    for (int q = 0; q < K; q++)
        fprintf(stderr, "[gen]   cb%d EOS at offset %d (%s)\n", q, eos_step[q],
                eos_step[q] >= 0 ? "early-stop" : "no EOS");

    // ---- revert_pattern_sequence -> out_codes[K][max_gen_len] ----
    // out_codes[q][t] = gen_seq[q][ pattern step s where coord (t,q) lives ] for t in [0,max_gen_len).
    // coord (t,q) lives at pattern step s = t + delays[q] + 1 (pos0 empty). Beyond gen_len -> unknown(-1)->EOS.
    std::vector<int64_t> out_codes((size_t) K * max_gen_len, SP);
    for (int q = 0; q < K; q++) {
        for (int t = 0; t < max_gen_len; t++) {
            int s = t + delays[q] + 1;
            int val;
            if (s < gen_len) {
                val = gen_seq[q][s];
            } else {
                val = -1;  // unknown -> never filled
            }
            out_codes[(size_t) q * max_gen_len + t] = (val == -1) ? EOS : val;
        }
    }

    // ---- save gen_tokens_cpp.npy [1,3,max_gen_len] ----
    std::string tok_out = "/home/dbrain/dev/songgen-port/out/gen_tokens_cpp.npy";
    if (!npy_save_i64(tok_out.c_str(), out_codes.data(), { 1, (int64_t) K, (int64_t) max_gen_len })) {
        fprintf(stderr, "[gen] WARNING: failed to write %s\n", tok_out.c_str());
    } else {
        fprintf(stderr, "[gen] wrote %s [1,%d,%d]\n", tok_out.c_str(), K, max_gen_len);
    }
    // print a few sample values
    printf("[gen] sample codes (cb1=vocal, cb2=bgm):\n");
    for (int q = 0; q < K; q++) {
        printf("  cb%d:", q);
        for (int t = 0; t < 8 && t < max_gen_len; t++) printf(" %lld", (long long) out_codes[(size_t) q * max_gen_len + t]);
        printf(" ...\n");
    }

    sggen_free_kv(&g);
    sglm_free(&m);

    // ---- decode (inline, mirrors songgen-decode) ----
    const int64_t * vocal = &out_codes[(size_t) 1 * max_gen_len];
    const int64_t * bgm   = &out_codes[(size_t) 2 * max_gen_len];
    int             Tv    = max_gen_len;
    while (Tv > 0) {
        int64_t cv = vocal[Tv - 1], cb = bgm[Tv - 1];
        if (cv >= 0 && cv < SGSEP_CODEBOOK && cb >= 0 && cb < SGSEP_CODEBOOK) break;
        Tv--;
    }
    int dropped = max_gen_len - Tv;
    std::vector<int> codes_vocal(Tv), codes_bgm(Tv);
    int              n_clamped = 0;
    for (int t = 0; t < Tv; t++) {
        int64_t cv = vocal[t], cb = bgm[t];
        if (cv < 0 || cv >= SGSEP_CODEBOOK) { cv = cv < 0 ? 0 : SGSEP_CODEBOOK - 1; n_clamped++; }
        if (cb < 0 || cb >= SGSEP_CODEBOOK) { cb = cb < 0 ? 0 : SGSEP_CODEBOOK - 1; n_clamped++; }
        codes_vocal[t] = (int) cv;
        codes_bgm[t]   = (int) cb;
    }
    fprintf(stderr, "[gen] decode: valid frames=%d trailing-dropped=%d interior-clamped=%d\n", Tv, dropped, n_clamped);
    if (Tv <= 0) {
        fprintf(stderr, "[gen] no valid frames; skipping decode\n");
        return 1;
    }

    SonggenSeptoken sep;
    if (!sgsep_load(&sep, aux_path)) { fprintf(stderr, "[gen] aux load failed\n"); return 1; }
    SonggenCfm cfm;
    if (!sgcfm_load(&cfm, cfm_path)) { fprintf(stderr, "[gen] cfm load failed\n"); return 1; }
    SonggenVae vae;
    if (!sgvae_load(&vae, vae_path)) { fprintf(stderr, "[gen] vae load failed\n"); return 1; }

    std::vector<float> audio;
    int                T_audio = sgpost_decode(&sep, &cfm, &vae, codes_vocal, codes_bgm, seed, P, audio);
    stats("wav", audio.data(), audio.size());

    const int sr = 48000;
    sgpost_fade(audio.data(), T_audio, sr, P.fade_ms, std::min(10.0f, P.fade_ms));
    if (!write_wav_s16_planar(out_path, audio.data(), T_audio, sr)) {
        fprintf(stderr, "[gen] wav write failed\n");
        return 1;
    }
    double dur = (double) T_audio / (double) sr;
    printf("---- songgen generate ----\n");
    printf("seed             : %llu\n", (unsigned long long) seed);
    printf("gen_tokens       : [1,%d,%d] -> %s\n", K, max_gen_len, tok_out.c_str());
    printf("valid frames     : %d (dropped %d trailing)\n", Tv, dropped);
    printf("samples / channel: %d (2 ch)  duration %.3f s @ %d Hz\n", T_audio, dur, sr);
    printf("output           : %s\n", out_path);

    sgvae_free(&vae);
    sgcfm_free(&cfm);
    return 0;
}
