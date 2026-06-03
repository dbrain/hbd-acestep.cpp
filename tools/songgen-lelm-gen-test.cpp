// songgen-lelm-gen-test: GATE-1 validation of the KV-cache decode path vs the full-prefill ORACLE.
//
// For both CFG rows (cond=set0, uncond=set1): reset KV, prefill conditioning + code position 0,
// then feed code positions 1..N-1 one token at a time. At each step compare the KV-path
// cb0/cb1/cb2 last-position logits to sglm_forward's full-prefill logits over the same prefix.
// cossim MUST be > 0.999 (same math, just cached).
//
//   GGML_BACKEND=CPU ./build/songgen-lelm-gen-test <gguf> <golden_dir>
#include "npy.h"
#include "songgen-lelm-gen.h"

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
    const int             SP   = c.special_token;

    NpyArray toks = npy_load(gp("gen_tokens_lm.npy").c_str());  // [1,3,T] i64
    int      T    = (int) toks.shape[2];
    auto     code = [&](int cb, int t) -> int { return (int) toks.i64[(size_t) cb * T + t]; };

    NpyArray d_ids_c = npy_load(gp("desc_ids_cond.npy").c_str());
    NpyArray d_cov_c = npy_load(gp("desc_cover_cond.npy").c_str());
    NpyArray d_ids_u = npy_load(gp("desc_ids_uncond.npy").c_str());
    NpyArray d_cov_u = npy_load(gp("desc_cover_uncond.npy").c_str());
    NpyArray t_ids_c = npy_load(gp("type_ids_cond.npy").c_str());
    NpyArray t_ids_u = npy_load(gp("type_ids_uncond.npy").c_str());

    std::vector<int32_t> audio_ids(c.audio_len - 1, SP);

    // delay-pattern code token at sequence position p (p=0 -> all special)
    auto code_at = [&](int cb, int p) -> int {
        if (p == 0) return SP;
        int t = p - 1;
        if (cb == 0) return (t < T) ? code(0, t) : SP;
        return (t - 250 >= 0 && t - 250 < T) ? code(cb, t - 250) : SP;
    };

    // full conditioning input for the full-prefill oracle (n_code positions 0..n_code-1)
    auto make_full = [&](bool cond, int n_code) {
        SonggenCondInput in;
        in.desc_ids   = cond ? d_ids_c.i32 : d_ids_u.i32;
        in.desc_cover = cond ? d_cov_c.i32 : d_cov_u.i32;
        in.type_ids   = cond ? t_ids_c.i32 : t_ids_u.i32;
        in.audio_ids  = audio_ids;
        in.seq0.resize(n_code);
        in.seq1.resize(n_code);
        in.seq2.resize(n_code);
        for (int p = 0; p < n_code; p++) {
            in.seq0[p] = code_at(0, p);
            in.seq1[p] = code_at(1, p);
            in.seq2[p] = code_at(2, p);
        }
        return in;
    };

    int N = 16;  // number of golden AR steps available

    SonggenLeLMGen g;
    sggen_alloc_kv(&g, &m, /*max_seq=*/c.desc_len + c.audio_len + c.type_len + N + 4, /*n_sets=*/2);

    std::vector<float> kv_cb[3], or_cb[3];
    for (int k = 0; k < 3; k++) {
        kv_cb[k].resize(card);
        or_cb[k].resize(card);
    }

    double min_s[2][3] = { { 1, 1, 1 }, { 1, 1, 1 } };
    printf("row step |   cb0      cb1      cb2\n");
    bool ok = true;
    for (int row = 0; row < 2; row++) {
        bool cond = (row == 0);
        for (int p = 0; p < N; p++) {
            // KV step: at p==0 prefill conditioning + code pos 0; else feed 1 new token.
            SonggenCondInput in;
            if (p == 0) {
                in            = make_full(cond, 1);  // conditioning ids + code pos 0
            } else {
                in.seq0 = { code_at(0, p) };
                in.seq1 = { code_at(1, p) };
                in.seq2 = { code_at(2, p) };
            }
            sggen_step(&g, row, in, /*prepend=*/(p == 0), kv_cb[0].data(), kv_cb[1].data(), kv_cb[2].data());

            // oracle full-prefill over prefix [0..p]
            SonggenCondInput full = make_full(cond, p + 1);
            sglm_forward(&m, full, or_cb[0].data(), or_cb[1].data(), or_cb[2].data());

            double s[3];
            for (int k = 0; k < 3; k++) {
                s[k] = cossim(kv_cb[k].data(), or_cb[k].data(), card);
                if (s[k] < min_s[row][k]) min_s[row][k] = s[k];
                if (s[k] < 0.999) ok = false;
            }
            printf("%3s %4d | %8.6f %8.6f %8.6f\n", cond ? "C" : "U", p, s[0], s[1], s[2]);
            fflush(stdout);
        }
    }

    printf("---- min cossim (KV vs full-prefill) ----\n");
    for (int row = 0; row < 2; row++)
        printf("%s: cb0=%.6f cb1=%.6f cb2=%.6f\n", row == 0 ? "cond  " : "uncond", min_s[row][0], min_s[row][1],
               min_s[row][2]);
    printf("GATE-1: %s\n", ok ? "PASS (all >0.999)" : "FAIL (<0.999)");

    sggen_free_kv(&g);
    sglm_free(&m);
    return ok ? 0 : 2;
}
