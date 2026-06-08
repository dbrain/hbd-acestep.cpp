// sa3-same-enc-test: validate the SAME encoder against the golden.
//   ./sa3-same-enc-test <sa3-same.gguf> <golden_dir>
// Loads enc_audio_in.npy [1,2,N], encodes, compares to enc_latent.npy [1,256,T].
#include "npy.h"
#include "sa3-same-enc.h"

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

    NpyArray aud = npy_load(gp("enc_audio_in.npy").c_str());  // [1,2,N] channel-major
    NpyArray ref = npy_load(gp("enc_latent.npy").c_str());     // [1,256,T] channel-major
    int AC = (int) aud.shape[aud.shape.size()-2];
    int N  = (int) aud.shape[aud.shape.size()-1];
    int C  = (int) ref.shape[ref.shape.size()-2];
    int T  = (int) ref.shape[ref.shape.size()-1];
    fprintf(stderr, "[golden] audio [%d,%d]  latent [%d,%d]\n", AC, N, C, T);

    SA3Enc m;
    if (!sa3enc_load(&m, gguf.c_str())) { fprintf(stderr, "load failed\n"); return 1; }

    std::vector<float> lat;  // [C,T] token-major (t*C+c)
    int Tout = sa3enc_encode(&m, aud.f32.data(), N, lat);
    fprintf(stderr, "[SA3-ENC] encoded T=%d (golden %d)\n", Tout, T);

    // golden is channel-major [C,T] (c*T+t); convert ours token-major -> channel-major for compare
    int Tc = Tout < T ? Tout : T;
    std::vector<float> ours((size_t) C * Tc), gold((size_t) C * Tc);
    for (int c = 0; c < C; c++) for (int t = 0; t < Tc; t++) {
        ours[(size_t) c*Tc + t] = lat[(size_t) t*C + c];
        gold[(size_t) c*Tc + t] = ref.f32[(size_t) c*T + t];
    }
    size_t n = (size_t) C * Tc;
    double cs = cossim(ours.data(), gold.data(), n);
    double se = 0, maxabs = 0, os = 0, gs = 0;
    for (size_t i = 0; i < n; i++) { double d = ours[i]-gold[i]; se += d*d; if (fabs(d)>maxabs) maxabs=fabs(d); os += (double)ours[i]*ours[i]; gs += (double)gold[i]*gold[i]; }
    printf("---- SAME encode vs golden ----\n");
    printf("latent    : %d x %d\n", C, Tc);
    printf("cossim    : %.6f\n", cs);
    printf("rmse      : %.6e\n", sqrt(se/n));
    printf("maxabs    : %.6e\n", maxabs);
    printf("std o/g   : %.5f / %.5f\n", sqrt(os/n), sqrt(gs/n));
    bool ok = cs > 0.999;
    printf("RESULT: %s (cossim %.6f %s 0.999)\n", ok ? "PASS" : "FAIL", cs, ok ? ">" : "<=");
    sa3enc_free(&m);
    return ok ? 0 : 2;
}
