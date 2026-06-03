// songgen-lelm-test: golden-replay validation for the LeLM ggml graph (task #7).
//
// Reconstructs the delay-pattern code sequence from gen_tokens_lm.npy, replays the
// first N autoregressive steps via full-prefill for both CFG rows (cond/uncond), and
// cossim-compares per-codebook last-position logits against golden-large/lelm_logits.npy.
//
// Run on CPU for determinism:  GGML_BACKEND=CPU ./build/songgen-lelm-test <gguf> <golden_dir>
#include "npy.h"
#include "songgen-lelm.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static double cossim(const float * a, const float * b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += (double) a[i] * b[i];
        na += (double) a[i] * a[i];
        nb += (double) b[i] * b[i];
    }
    if (na == 0 || nb == 0) return 0.0;
    return dot / (sqrt(na) * sqrt(nb));
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <gguf> <golden_dir>\n", argv[0]);
        return 1;
    }
    std::string gguf = argv[1];
    std::string gd   = argv[2];
    auto        gp   = [&](const char * f) { return gd + "/" + f; };

    SonggenLeLM m;
    if (!sglm_load(&m, gguf.c_str())) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    const SonggenConfig & c    = m.cfg;
    const int             card = c.card;
    const int             SP   = c.special_token;  // 16385

    // --- goldens ---
    NpyArray logits = npy_load(gp("lelm_logits.npy").c_str());    // [N,2,3,card] f32
    NpyArray toks   = npy_load(gp("gen_tokens_lm.npy").c_str());  // [1,3,T] i64
    int      N      = (int) logits.shape[0];
    int      Kc     = (int) logits.shape[2];
    int      gcard  = (int) logits.shape[3];
    int      T      = (int) toks.shape[2];
    fprintf(stderr, "[golden] logits [%d,%d,%d,%d]  tokens T=%d  (model card=%d)\n", N, (int) logits.shape[1], Kc,
            gcard, T, card);
    if (gcard != card) fprintf(stderr, "[warn] golden card %d != model card %d\n", gcard, card);

    auto code = [&](int cb, int t) -> int { return (int) toks.i64[(size_t) cb * T + t]; };
    auto glog = [&](int step, int row, int cb) -> const float * {
        return &logits.f32[((((size_t) step * 2 + row) * 3) + cb) * gcard];
    };

    // --- conditioning ids ---
    NpyArray d_ids_c = npy_load(gp("desc_ids_cond.npy").c_str());
    NpyArray d_cov_c = npy_load(gp("desc_cover_cond.npy").c_str());
    NpyArray d_ids_u = npy_load(gp("desc_ids_uncond.npy").c_str());
    NpyArray d_cov_u = npy_load(gp("desc_cover_uncond.npy").c_str());
    NpyArray t_ids_c = npy_load(gp("type_ids_cond.npy").c_str());
    NpyArray t_ids_u = npy_load(gp("type_ids_uncond.npy").c_str());

    std::vector<int32_t> audio_ids(c.audio_len - 1, SP);  // all special (no prompt audio)

    auto make_row = [&](bool cond, int n_code) {
        SonggenCondInput in;
        in.desc_ids   = cond ? d_ids_c.i32 : d_ids_u.i32;
        in.desc_cover = cond ? d_cov_c.i32 : d_cov_u.i32;
        in.type_ids   = cond ? t_ids_c.i32 : t_ids_u.i32;
        in.audio_ids  = audio_ids;
        in.seq0.resize(n_code);
        in.seq1.resize(n_code);
        in.seq2.resize(n_code);
        // delay pattern [0,250,250]: pos0 = all special; pos p>=1 (t=p-1):
        //   cb0=code[0,t] if t<T else SP; cb1=code[1,t-250]..; cb2=code[2,t-250]..
        for (int p = 0; p < n_code; p++) {
            if (p == 0) {
                in.seq0[p] = SP;
                in.seq1[p] = SP;
                in.seq2[p] = SP;
                continue;
            }
            int t      = p - 1;
            in.seq0[p] = (t < T) ? code(0, t) : SP;
            in.seq1[p] = (t - 250 >= 0 && t - 250 < T) ? code(1, t - 250) : SP;
            in.seq2[p] = (t - 250 >= 0 && t - 250 < T) ? code(2, t - 250) : SP;
        }
        return in;
    };

    std::vector<float> cb_cond[3], cb_unc[3];
    for (int k = 0; k < 3; k++) {
        cb_cond[k].resize(card);
        cb_unc[k].resize(card);
    }
    std::vector<float> comb(card), gcomb(card);

    double min_c[3] = { 1, 1, 1 }, min_u[3] = { 1, 1, 1 }, min_m[3] = { 1, 1, 1 };
    printf("step |   cb0_cond cb0_unc cb0_cfg |   cb1_cond cb1_unc cb1_cfg |   cb2_cond cb2_unc cb2_cfg\n");
    for (int i = 0; i < N; i++) {
        int              n_code = i + 1;
        SonggenCondInput inc    = make_row(true, n_code);
        SonggenCondInput inu    = make_row(false, n_code);
        sglm_forward(&m, inc, cb_cond[0].data(), cb_cond[1].data(), cb_cond[2].data());
        sglm_forward(&m, inu, cb_unc[0].data(), cb_unc[1].data(), cb_unc[2].data());

        double sc[3], su[3], sm[3];
        for (int k = 0; k < 3; k++) {
            const float * gc = glog(i, 0, k);
            const float * gu = glog(i, 1, k);
            sc[k]            = cossim(cb_cond[k].data(), gc, card);
            su[k]            = cossim(cb_unc[k].data(), gu, card);
            for (int v = 0; v < card; v++) {
                comb[v]  = cb_unc[k][v] + (cb_cond[k][v] - cb_unc[k][v]) * 1.5f;
                gcomb[v] = gu[v] + (gc[v] - gu[v]) * 1.5f;
            }
            sm[k] = cossim(comb.data(), gcomb.data(), card);
            if (sc[k] < min_c[k]) min_c[k] = sc[k];
            if (su[k] < min_u[k]) min_u[k] = su[k];
            if (sm[k] < min_m[k]) min_m[k] = sm[k];
        }
        printf("%4d | %9.5f %7.5f %7.5f | %9.5f %7.5f %7.5f | %9.5f %7.5f %7.5f\n", i, sc[0], su[0], sm[0], sc[1],
               su[1], sm[1], sc[2], su[2], sm[2]);
        fflush(stdout);
    }
    printf("---- min cossim across %d steps ----\n", N);
    for (int k = 0; k < 3; k++) printf("cb%d: cond=%.5f uncond=%.5f cfg=%.5f\n", k, min_c[k], min_u[k], min_m[k]);
    bool ok = true;
    for (int k = 0; k < 3; k++)
        if (min_c[k] < 0.99 || min_u[k] < 0.99 || min_m[k] < 0.99) ok = false;
    printf("RESULT: %s\n", ok ? "PASS (all >0.99)" : "FAIL (<0.99)");

    sglm_free(&m);
    return ok ? 0 : 2;
}
