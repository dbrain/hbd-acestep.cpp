// songgen-htdemucs-test: block-by-block golden validation for the HTDemucs port.
//
//   GGML_BACKEND=CPU ./songgen-htdemucs-test <gguf> <ref_dir> [stage]
// ref_dir holds the per-block .npy dumped by scripts/dump_htdemucs_ref.py plus input_seg.npy.
// stage: "enc" (default) validates spec/time encoders.
#include "npy.h"
#include "songgen-htdemucs.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static double cossim(const float * a, const float * b, size_t n) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) { dot += (double) a[i] * b[i]; na += (double) a[i] * a[i]; nb += (double) b[i] * b[i]; }
    if (na == 0 || nb == 0) return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

static int report(const char * tag, const std::vector<float> & ours, const float * gold, size_t n, double thr) {
    double cs = cossim(ours.data(), gold, n);
    double se = 0, mx = 0;
    for (size_t i = 0; i < n; i++) { double d = (double) ours[i] - gold[i]; se += d * d; if (std::fabs(d) > mx) mx = std::fabs(d); }
    bool ok = cs > thr;
    printf("%-14s cossim=%.6f rmse=%.3e max=%.3e  %s\n", tag, cs, std::sqrt(se / n), mx, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <gguf> <ref_dir> [stage]\n", argv[0]); return 1; }
    std::string gguf = argv[1], rd = argv[2];
    std::string stage = argc > 3 ? argv[3] : "enc";
    auto rp = [&](const char * f) { return rd + "/" + f; };

    SonggenHTDemucs m;
    if (!htd_load(&m, gguf.c_str())) { fprintf(stderr, "load failed\n"); return 1; }

    NpyArray in = npy_load(rp("input_seg.npy").c_str());  // [1,2,343980]
    int AC = (int) in.shape[1];
    int N = (int) in.shape[2];
    fprintf(stderr, "[test] input AC=%d N=%d\n", AC, N);

    int fails = 0;
    if (stage == "enc") {
        HtdEncOut eo;
        htd_encode_encoders(&m, in.f32.data(), N, AC, &eo);
        printf("---- encoders (T=%d) ----\n", eo.spec_T);
        {
            NpyArray g = npy_load(rp("enc0_pre_shipped.npy").c_str());  // shipped encoder0_out = PRE freq_emb
            fails += report("enc0_pre", eo.spec[0], g.f32.data(), npy_numel(g), 0.99);
        }
        for (int i = 0; i < 4; i++) {
            char f[64]; snprintf(f, sizeof(f), "enc%d.npy", i);
            NpyArray g = npy_load(rp(f).c_str());  // my ref dump (POST freq_emb for enc0)
            const std::vector<float> & ours = (i == 0) ? eo.spec_fe0 : eo.spec[i];
            size_t n = npy_numel(g);
            char tag[32]; snprintf(tag, sizeof(tag), "enc%d", i);
            fails += report(tag, ours, g.f32.data(), n, 0.99);
        }
        for (int i = 0; i < 4; i++) {
            char f[64]; snprintf(f, sizeof(f), "tenc%d.npy", i);
            NpyArray g = npy_load(rp(f).c_str());
            size_t n = npy_numel(g);
            char tag[32]; snprintf(tag, sizeof(tag), "tenc%d", i);
            fails += report(tag, eo.time[i], g.f32.data(), n, 0.99);
        }
    }

    if (stage == "fwd") {
        HtdForwardOut fo;
        htd_forward(&m, in.f32.data(), N, AC, &fo);
        printf("---- transformer layers ----\n");
        for (int i = 0; i < 5; i++) {
            char f[64]; snprintf(f, sizeof(f), "ct_x_layer%d.npy", i);
            NpyArray g = npy_load(rp(f).c_str());
            char tag[32]; snprintf(tag, sizeof(tag), "ct_x_l%d", i);
            fails += report(tag, fo.ct_x_layer[i], g.f32.data(), npy_numel(g), 0.99);
            snprintf(f, sizeof(f), "ct_xt_layer%d.npy", i);
            NpyArray gt = npy_load(rp(f).c_str());
            snprintf(tag, sizeof(tag), "ct_xt_l%d", i);
            fails += report(tag, fo.ct_xt_layer[i], gt.f32.data(), npy_numel(gt), 0.99);
        }
        printf("---- decoders ----\n");
        for (int i = 0; i < 4; i++) {
            char f[64]; snprintf(f, sizeof(f), "dec%d.npy", i);
            NpyArray g = npy_load(rp(f).c_str());
            char tag[32]; snprintf(tag, sizeof(tag), "dec%d", i);
            fails += report(tag, fo.dec[i], g.f32.data(), npy_numel(g), 0.99);
            snprintf(f, sizeof(f), "tdec%d.npy", i);
            NpyArray gt = npy_load(rp(f).c_str());
            snprintf(tag, sizeof(tag), "tdec%d", i);
            fails += report(tag, fo.tdec[i], gt.f32.data(), npy_numel(gt), 0.99);
        }
        printf("---- final stems ----\n");
        NpyArray g = npy_load(rp("final_out.npy").c_str());  // [1,4,2,343980]
        fails += report("final_out", fo.stems, g.f32.data(), npy_numel(g), 0.99);
    }

    printf("RESULT: %s\n", fails == 0 ? "PASS" : "FAIL");
    htd_free(&m);
    return fails == 0 ? 0 : 2;
}
