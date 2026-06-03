// songgen-cfm-test: golden-replay validation for the CFM reflow estimator (custom GPT2).
//
//   GGML_BACKEND=CPU ./songgen-cfm-test <gguf> <golden_dir>
// Loads cfm_in_*.npy, runs one forward, compares to cfm_out_last_hidden_state.npy
// (full [2,32,2200]) and cfm_out_velocity.npy (= lhs[...,-64:]). PASS if lhs cossim > 0.99.
#include "npy.h"
#include "songgen-cfm.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static double cossim(const float * a, const float * b, size_t n) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) {
        dot += (double) a[i] * b[i];
        na += (double) a[i] * a[i];
        nb += (double) b[i] * b[i];
    }
    if (na == 0 || nb == 0) return 0.0;
    return dot / (sqrt(na) * sqrt(nb));
}

static double rmse(const float * a, const float * b, size_t n) {
    double se = 0;
    for (size_t i = 0; i < n; i++) {
        double d = (double) a[i] - b[i];
        se += d * d;
    }
    return sqrt(se / (double) n);
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <gguf> <golden_dir>\n", argv[0]);
        return 1;
    }
    std::string gguf = argv[1];
    std::string gd   = argv[2];
    auto        gp   = [&](const char * f) { return gd + "/" + f; };

    SonggenCfm m;
    if (!sgcfm_load(&m, gguf.c_str())) {
        fprintf(stderr, "load failed\n");
        return 1;
    }

    NpyArray emb = npy_load(gp("cfm_in_inputs_embeds.npy").c_str());      // [B,T,H]
    NpyArray ts  = npy_load(gp("cfm_in_time_step.npy").c_str());          // [B]
    NpyArray lhs = npy_load(gp("cfm_out_last_hidden_state.npy").c_str()); // [B,T,H]
    NpyArray vel = npy_load(gp("cfm_out_velocity.npy").c_str());          // [B,T,64]

    int B = (int) emb.shape[0];
    int T = (int) emb.shape[1];
    int H = (int) emb.shape[2];
    fprintf(stderr, "[golden] inputs_embeds [%d,%d,%d]  time_step=%.4f\n", B, T, H, ts.f32[0]);

    std::vector<float> out((size_t) B * T * H);
    sgcfm_forward(&m, emb.f32.data(), B, T, ts.f32[0], out.data());

    size_t n_lhs = (size_t) B * T * H;
    double cs_lhs   = cossim(out.data(), lhs.f32.data(), n_lhs);
    double rmse_lhs = rmse(out.data(), lhs.f32.data(), n_lhs);

    // velocity slice = out[..., -64:] per (b,t) row.
    int                VD = (int) vel.shape[2];  // 64
    std::vector<float> ov((size_t) B * T * VD);
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++)
            for (int d = 0; d < VD; d++)
                ov[((size_t) b * T + t) * VD + d] = out[((size_t) b * T + t) * H + (H - VD) + d];
    size_t n_vel   = (size_t) B * T * VD;
    double cs_vel   = cossim(ov.data(), vel.f32.data(), n_vel);
    double rmse_vel = rmse(ov.data(), vel.f32.data(), n_vel);

    double os = 0;
    for (size_t i = 0; i < n_lhs; i++) os += (double) out[i] * out[i];
    double our_std = sqrt(os / (double) n_lhs);

    printf("---- CFM estimator vs golden ----\n");
    printf("last_hidden_state cossim : %.6f\n", cs_lhs);
    printf("last_hidden_state rmse   : %.6e\n", rmse_lhs);
    printf("velocity[-64:]    cossim : %.6f\n", cs_vel);
    printf("velocity[-64:]    rmse   : %.6e\n", rmse_vel);
    printf("std ours/gold            : %.5f / %.5f\n", our_std, 0.14167);

    bool ok = cs_lhs > 0.99;
    printf("RESULT: %s (last_hidden_state cossim %.6f %s 0.99)\n", ok ? "PASS" : "FAIL", cs_lhs, ok ? ">" : "<=");

    sgcfm_free(&m);
    return ok ? 0 : 2;
}
