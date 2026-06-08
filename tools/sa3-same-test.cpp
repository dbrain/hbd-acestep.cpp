// sa3-same-test: validate the SAME decoder against the golden.
//   ./sa3-same-test <sa3-same.gguf> <golden_dir>
// Loads latent_final.npy [1,256,T], decodes, compares to audio.npy [1,2,4096*T].
#include "npy.h"
#include "sa3-same.h"

#include <cmath>
#include <cstdio>
#include <vector>

static double cossim(const float * a, const float * b, size_t n) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) { dot += (double) a[i]*b[i]; na += (double) a[i]*a[i]; nb += (double) b[i]*b[i]; }
    return (na == 0 || nb == 0) ? 0.0 : dot / (sqrt(na) * sqrt(nb));
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <gguf> <golden_dir>\n", argv[0]); return 1; }
    std::string gguf = argv[1], gd = argv[2];
    auto gp = [&](const char * f) { return gd + "/" + f; };

    NpyArray lat = npy_load(gp("latent_final.npy").c_str());  // [1,256,T] channel-major
    NpyArray ref = npy_load(gp("audio.npy").c_str());          // [1,2,N] channel-major
    int C = (int) lat.shape[lat.shape.size()-2];
    int T = (int) lat.shape[lat.shape.size()-1];
    int RC = (int) ref.shape[ref.shape.size()-2];
    int RN = (int) ref.shape[ref.shape.size()-1];
    fprintf(stderr, "[golden] latent [%d,%d]  audio [%d,%d]\n", C, T, RC, RN);

    // numpy [C,T] (c*T+t) -> ggml [C,T] token-major (t*C+c)
    std::vector<float> z_t((size_t) C * T);
    for (int c = 0; c < C; c++) for (int t = 0; t < T; t++) z_t[(size_t) t*C + c] = lat.f32[(size_t) c*T + t];

    SA3Same m;
    if (!sa3same_load(&m, gguf.c_str())) { fprintf(stderr, "load failed\n"); return 1; }

    std::vector<float> audio;
    int N = sa3same_decode(&m, z_t.data(), T, audio);  // [2, N] channel-major
    fprintf(stderr, "[SA3-SAME] decoded N=%d (golden %d)\n", N, RN);

    int Nc = N < RN ? N : RN;
    std::vector<float> ours(2*(size_t)Nc), gold(2*(size_t)Nc);
    for (int c = 0; c < 2; c++) for (int s = 0; s < Nc; s++) {
        ours[(size_t)c*Nc+s] = audio[(size_t)c*N + s];
        gold[(size_t)c*Nc+s] = ref.f32[(size_t)c*RN + s];
    }
    size_t n = 2*(size_t)Nc;
    double cs = cossim(ours.data(), gold.data(), n);
    double se = 0, maxabs = 0, os = 0, gs = 0;
    for (size_t i = 0; i < n; i++) { double d = ours[i]-gold[i]; se += d*d; if (fabs(d)>maxabs) maxabs=fabs(d); os += (double)ours[i]*ours[i]; gs += (double)gold[i]*gold[i]; }
    printf("---- SAME decode vs golden ----\n");
    printf("samples   : 2 x %d\n", Nc);
    printf("cossim    : %.6f\n", cs);
    printf("rmse      : %.6e\n", sqrt(se/n));
    printf("maxabs    : %.6e\n", maxabs);
    printf("std o/g   : %.5f / %.5f\n", sqrt(os/n), sqrt(gs/n));
    bool ok = cs > 0.99;
    printf("RESULT: %s (cossim %.6f %s 0.99)\n", ok ? "PASS" : "FAIL", cs, ok ? ">" : "<=");
    sa3same_free(&m);
    return ok ? 0 : 2;
}
