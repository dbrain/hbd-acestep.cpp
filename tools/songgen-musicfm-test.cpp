// songgen-musicfm-test: golden-replay validation for the MusicFM_95M BESTRQ encoder.
//
//   GGML_BACKEND=CPU ./songgen-musicfm-test <gguf> <golden_dir>
// Loads bestrq_in_audio24k.npy [1,N], runs the encoder, compares:
//   layer7 vs bestrq_out_layer7_vocal.npy [1,1024,T]
//   layer3 vs bestrq_out_layer3_bgm.npy   [1,1024,T]
//   (optionally all 13 vs bestrq_out_all_layers.npy [13,1024,T])
// GATE: layer7 AND layer3 cossim > 0.99.
#include "npy.h"
#include "songgen-musicfm.h"

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

// ggml result is [H,T] (ne0=H, memory t-major: for each t, H floats).
// golden is [1024,T] channel-major: for each channel, T floats. Reorder ours to channel-major.
static std::vector<float> to_chan_major(const std::vector<float> & ours, int H, int T) {
    std::vector<float> r((size_t) H * T);
    for (int t = 0; t < T; t++)
        for (int c = 0; c < H; c++) r[(size_t) c * T + t] = ours[(size_t) t * H + c];
    return r;
}

static void report(const char * tag, const std::vector<float> & ours_cm, const float * gold, int H, int T) {
    size_t n  = (size_t) H * T;
    double cs = cossim(ours_cm.data(), gold, n);
    double se = 0, maxabs = 0;
    for (size_t i = 0; i < n; i++) {
        double d = (double) ours_cm[i] - gold[i];
        se += d * d;
        if (fabs(d) > maxabs) maxabs = fabs(d);
    }
    double rmse = sqrt(se / (double) n);
    printf("%-8s cossim=%.6f  rmse=%.6e  max-abs=%.6e\n", tag, cs, rmse, maxabs);
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <gguf> <golden_dir>\n", argv[0]);
        return 1;
    }
    std::string gguf = argv[1];
    std::string gd   = argv[2];
    auto        gp   = [&](const char * f) { return gd + "/" + f; };

    SonggenMusicFM m;
    if (!sgmf_load(&m, gguf.c_str())) {
        fprintf(stderr, "load failed\n");
        return 1;
    }

    NpyArray in  = npy_load(gp("bestrq_in_audio24k.npy").c_str());      // [1,N]
    NpyArray l7  = npy_load(gp("bestrq_out_layer7_vocal.npy").c_str()); // [1,1024,T]
    NpyArray l3  = npy_load(gp("bestrq_out_layer3_bgm.npy").c_str());   // [1,1024,T]
    int      N   = (int) in.shape[in.shape.size() - 1];
    int      H   = (int) l7.shape[1];
    int      Tg  = (int) l7.shape[2];
    fprintf(stderr, "[golden] audio N=%d  layer7/3 [%d,%d]\n", N, H, Tg);

    std::vector<std::vector<float>> layers;
    int                             T = 0;
    sgmf_encode(&m, in.f32.data(), N, layers, &T);
    fprintf(stderr, "[musicfm] produced %zu layers, T=%d (golden T=%d)\n", layers.size(), T, Tg);

    int Tc = T < Tg ? T : Tg;

    auto slice_cm = [&](const std::vector<float> & ours) {
        std::vector<float> cm = to_chan_major(ours, H, T);
        std::vector<float> r((size_t) H * Tc);
        for (int c = 0; c < H; c++)
            for (int t = 0; t < Tc; t++) r[(size_t) c * Tc + t] = cm[(size_t) c * T + t];
        return r;
    };
    auto slice_gold = [&](const NpyArray & a) {
        std::vector<float> r((size_t) H * Tc);
        int                Ta = (int) a.shape[2];
        for (int c = 0; c < H; c++)
            for (int t = 0; t < Tc; t++) r[(size_t) c * Tc + t] = a.f32[(size_t) c * Ta + t];
        return r;
    };

    printf("---- MusicFM BESTRQ vs golden (T=%d) ----\n", Tc);

    // optional: all 13 layers
    bool have_all = false;
    NpyArray all;
    {
        FILE * fp = fopen(gp("bestrq_out_all_layers.npy").c_str(), "rb");
        if (fp) {
            fclose(fp);
            all      = npy_load(gp("bestrq_out_all_layers.npy").c_str());  // [13,1024,T]
            have_all = (int) all.shape[0] == (int) layers.size();
        }
    }
    if (have_all) {
        int Ta = (int) all.shape[2];
        for (size_t li = 0; li < layers.size(); li++) {
            std::vector<float> ours_cm = slice_cm(layers[li]);
            std::vector<float> g((size_t) H * Tc);
            for (int c = 0; c < H; c++)
                for (int t = 0; t < Tc; t++)
                    g[(size_t) c * Tc + t] = all.f32[((size_t) li * H + c) * Ta + t];
            char tag[32];
            snprintf(tag, sizeof(tag), "layer%d", (int) li);
            report(tag, ours_cm, g.data(), H, Tc);
        }
    }

    std::vector<float> g7 = slice_gold(l7);
    std::vector<float> g3 = slice_gold(l3);
    std::vector<float> o7 = slice_cm(layers[7]);
    std::vector<float> o3 = slice_cm(layers[3]);

    double cs7 = cossim(o7.data(), g7.data(), (size_t) H * Tc);
    double cs3 = cossim(o3.data(), g3.data(), (size_t) H * Tc);
    printf("\nGATE layer7=%.6f layer3=%.6f\n", cs7, cs3);
    bool ok = cs7 > 0.99 && cs3 > 0.99;
    printf("RESULT: %s\n", ok ? "PASS" : "FAIL");

    sgmf_free(&m);
    return ok ? 0 : 2;
}
