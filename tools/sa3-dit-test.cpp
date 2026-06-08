// sa3-dit-test: validate the DiT velocity model against a per-step golden.
//   ./sa3-dit-test <sa3-dit.gguf> <golden_dir> [step]
// Loads step{N}_xin.npy [1,256,L], step{N}_t.npy, cross_cond.npy [1,257,768], global_cond.npy [1,768];
// runs DiT; compares to step{N}_vout.npy [1,256,L]. Deterministic (no sampler RNG).
#include "npy.h"
#include "sa3-dit.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static double cossim(const float * a, const float * b, size_t n) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) { dot += (double) a[i]*b[i]; na += (double) a[i]*a[i]; nb += (double) b[i]*b[i]; }
    return (na == 0 || nb == 0) ? 0.0 : dot / (sqrt(na) * sqrt(nb));
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <gguf> <golden_dir> [step]\n", argv[0]); return 1; }
    std::string gguf = argv[1], gd = argv[2];
    int step = argc > 3 ? atoi(argv[3]) : 0;
    auto gp = [&](const std::string & f) { return gd + "/" + f; };
    char sb[32]; snprintf(sb, sizeof(sb), "step%d_", step);
    std::string sp(sb);

    NpyArray xin = npy_load(gp(sp + "xin.npy").c_str());     // [1,256,L]
    NpyArray vout = npy_load(gp(sp + "vout.npy").c_str());   // [1,256,L]
    NpyArray tn  = npy_load(gp(sp + "t.npy").c_str());       // [1]
    NpyArray cc  = npy_load(gp("cross_cond.npy").c_str());   // [1,257,768]
    NpyArray gcv = npy_load(gp("global_cond.npy").c_str());  // [1,768]
    int C = (int) xin.shape[xin.shape.size()-2];
    int L = (int) xin.shape[xin.shape.size()-1];
    int cross_T = (int) cc.shape[cc.shape.size()-2];
    int cond_dim = (int) cc.shape[cc.shape.size()-1];
    float t = tn.f32[0];
    fprintf(stderr, "[golden] step %d  C=%d L=%d cross_T=%d cond_dim=%d t=%.5f\n", step, C, L, cross_T, cond_dim, t);

    // numpy latent [C,L] channel-major (c*L+l) -> ggml [C,L] token-major (l*C+c): transpose
    std::vector<float> x_t((size_t) C * L);
    for (int c = 0; c < C; c++) for (int l = 0; l < L; l++) x_t[(size_t) l*C + c] = xin.f32[(size_t) c*L + l];

    SA3DiT m;
    if (!sa3dit_load(&m, gguf.c_str())) { fprintf(stderr, "load failed\n"); return 1; }

    std::vector<float> out((size_t) C * L);  // ggml [C,L] token-major (l*C+c)
    sa3dit_forward(&m, x_t.data(), L, t, cc.f32.data(), cross_T, gcv.f32.data(), out.data());

    // align golden [C,L] (c*L+l) to compare
    std::vector<float> ours((size_t) C*L), gold((size_t) C*L);
    for (int c = 0; c < C; c++) for (int l = 0; l < L; l++) {
        ours[(size_t) c*L+l] = out[(size_t) l*C+c];
        gold[(size_t) c*L+l] = vout.f32[(size_t) c*L+l];
    }
    size_t n = (size_t) C*L;
    double cs = cossim(ours.data(), gold.data(), n);
    double se = 0, maxabs = 0, gn = 0;
    for (size_t i = 0; i < n; i++) { double d = ours[i]-gold[i]; se += d*d; if (fabs(d) > maxabs) maxabs = fabs(d); gn += (double) gold[i]*gold[i]; }
    printf("---- DiT step %d vs golden ----\n", step);
    printf("cossim   : %.6f\n", cs);
    printf("rmse     : %.6e   (golden rms %.4e)\n", sqrt(se/n), sqrt(gn/n));
    printf("maxabs   : %.6e\n", maxabs);
    bool ok = cs > 0.999;
    printf("RESULT: %s (cossim %.6f %s 0.999)\n", ok ? "PASS" : "FAIL", cs, ok ? ">" : "<=");
    sa3dit_free(&m);
    return ok ? 0 : 2;
}
