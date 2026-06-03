// songgen-vae-test: golden-replay validation for the Stable-Audio Oobleck VAE decoder.
//
//   GGML_BACKEND=CPU ./songgen-vae-test <gguf> <golden_dir>
// Loads vae_latent.npy [1,64,T], decodes, compares to vae_wav.npy [1,2,T*1920].
// Reports cossim, per-sample RMSE, max-abs-error, plus a crude spectral check.
#include "npy.h"
#include "songgen-vae.h"

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

// crude spectral check: log-energy of a few coarse bands via box-filtered abs diff.
static double band_energy_rel(const float * a, const float * b, size_t n, int bands) {
    double worst = 0;
    size_t bw    = n / bands;
    for (int k = 0; k < bands; k++) {
        double ea = 0, eb = 0;
        for (size_t i = k * bw; i < (k + 1) * bw && i < n; i++) {
            ea += (double) a[i] * a[i];
            eb += (double) b[i] * b[i];
        }
        double rel = fabs(ea - eb) / (eb + 1e-9);
        if (rel > worst) worst = rel;
    }
    return worst;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <gguf> <golden_dir>\n", argv[0]);
        return 1;
    }
    std::string gguf = argv[1];
    std::string gd   = argv[2];
    auto        gp   = [&](const char * f) { return gd + "/" + f; };

    SonggenVae m;
    if (!sgvae_load(&m, gguf.c_str())) {
        fprintf(stderr, "load failed\n");
        return 1;
    }

    NpyArray lat = npy_load(gp("vae_latent.npy").c_str());  // [1,64,T]
    NpyArray ref = npy_load(gp("vae_wav.npy").c_str());      // [1,2,T*1920]
    int      C   = (int) lat.shape[1];
    int      T   = (int) lat.shape[2];
    int      Tw  = (int) ref.shape[2];
    fprintf(stderr, "[golden] latent [1,%d,%d]  wav [1,%d,%d]\n", C, T, (int) ref.shape[1], Tw);
    if (C != 64) {
        fprintf(stderr, "[VAE] expected 64 latent channels, got %d\n", C);
        return 1;
    }

    std::vector<float> audio;
    int                T_audio = sgvae_decode(&m, lat.f32.data(), T, audio);
    if (T_audio != Tw) {
        fprintf(stderr, "[VAE] WARNING: T_audio=%d != golden %d\n", T_audio, Tw);
    }

    // our audio layout: [ch0 (T_audio), ch1 (T_audio)]; golden [1,2,Tw] same channel-major.
    // compare overlapping samples per channel.
    int    Tc  = T_audio < Tw ? T_audio : Tw;
    std::vector<float> ours(2 * (size_t) Tc), gold(2 * (size_t) Tc);
    for (int t = 0; t < Tc; t++) {
        ours[t]            = audio[t];
        ours[Tc + t]       = audio[T_audio + t];
        gold[t]            = ref.f32[t];
        gold[Tc + t]       = ref.f32[Tw + t];
    }

    double cs = cossim(ours.data(), gold.data(), 2 * (size_t) Tc);

    double se = 0, maxabs = 0;
    for (size_t i = 0; i < 2 * (size_t) Tc; i++) {
        double d = (double) ours[i] - gold[i];
        se += d * d;
        if (fabs(d) > maxabs) maxabs = fabs(d);
    }
    double rmse = sqrt(se / (2.0 * Tc));

    double os = 0, gs = 0;
    for (size_t i = 0; i < 2 * (size_t) Tc; i++) {
        os += (double) ours[i] * ours[i];
        gs += (double) gold[i] * gold[i];
    }
    double our_std = sqrt(os / (2.0 * Tc));
    double gold_std = sqrt(gs / (2.0 * Tc));

    double spec = band_energy_rel(ours.data(), gold.data(), 2 * (size_t) Tc, 16);

    printf("---- VAE decode vs golden ----\n");
    printf("samples      : 2 x %d\n", Tc);
    printf("cossim       : %.6f\n", cs);
    printf("rmse         : %.6e\n", rmse);
    printf("max-abs-err  : %.6e\n", maxabs);
    printf("std ours/gold: %.5f / %.5f\n", our_std, gold_std);
    printf("spectral rel : %.4f (worst of 16 bands)\n", spec);

    bool ok = cs > 0.99;
    printf("RESULT: %s (cossim %.6f %s 0.99)\n", ok ? "PASS" : "FAIL", cs, ok ? ">" : "<=");

    sgvae_free(&m);
    return ok ? 0 : 2;
}
