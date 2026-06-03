// songgen-vae-enc-test: golden-replay validation for the Oobleck VAE ENCODER.
//
//   GGML_BACKEND=CPU ./songgen-vae-enc-test <gguf> <golden_dir>
// Loads vaeenc_in_audio48k.npy [1,2,T], encodes, compares mean to vaeenc_out_mean.npy [1,64,T/1920].
// GATE: mean cossim > 0.99.
#include "npy.h"
#include "songgen-vae-enc.h"

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

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <gguf> <golden_dir>\n", argv[0]);
        return 1;
    }
    std::string gguf = argv[1];
    std::string gd   = argv[2];
    auto        gp   = [&](const char * f) { return gd + "/" + f; };

    SonggenVaeEnc m;
    if (!sgve_load(&m, gguf.c_str())) {
        fprintf(stderr, "load failed\n");
        return 1;
    }

    NpyArray in  = npy_load(gp("vaeenc_in_audio48k.npy").c_str());  // [1,2,T]
    NpyArray ref = npy_load(gp("vaeenc_out_mean.npy").c_str());     // [1,64,T/1920]
    int      C   = (int) in.shape[1];
    int      T   = (int) in.shape[2];
    int      Cl  = (int) ref.shape[1];
    int      Tl  = (int) ref.shape[2];
    fprintf(stderr, "[golden] audio [1,%d,%d]  mean [1,%d,%d]\n", C, T, Cl, Tl);
    if (C != 2) {
        fprintf(stderr, "[VAE-ENC] expected 2 audio channels, got %d\n", C);
        return 1;
    }

    std::vector<float> mean;
    int                T_lat = sgve_encode(&m, in.f32.data(), T, mean);
    if (T_lat != Tl) {
        fprintf(stderr, "[VAE-ENC] WARNING: T_latent=%d != golden %d\n", T_lat, Tl);
    }

    int    Tc = T_lat < Tl ? T_lat : Tl;
    size_t n  = (size_t) Cl * Tc;
    // both channel-major [C, Tl]; compare overlapping time per channel.
    std::vector<float> ours(n), gold(n);
    for (int c = 0; c < Cl; c++) {
        for (int t = 0; t < Tc; t++) {
            ours[(size_t) c * Tc + t] = mean[(size_t) c * T_lat + t];
            gold[(size_t) c * Tc + t] = ref.f32[(size_t) c * Tl + t];
        }
    }

    double cs = cossim(ours.data(), gold.data(), n);
    double se = 0, maxabs = 0;
    for (size_t i = 0; i < n; i++) {
        double d = (double) ours[i] - gold[i];
        se += d * d;
        if (fabs(d) > maxabs) maxabs = fabs(d);
    }
    double rmse = sqrt(se / (double) n);

    printf("---- VAE encode (mean) vs golden ----\n");
    printf("latent       : %d x %d\n", Cl, Tc);
    printf("cossim       : %.6f\n", cs);
    printf("rmse         : %.6e\n", rmse);
    printf("max-abs-err  : %.6e\n", maxabs);

    bool ok = cs > 0.99;
    printf("RESULT: %s (cossim %.6f %s 0.99)\n", ok ? "PASS" : "FAIL", cs, ok ? ">" : "<=");

    sgve_free(&m);
    return ok ? 0 : 2;
}
