// songgen-rvq-enc-test: golden-replay validation for the descript RVQ ENCODE path.
//
//   GGML_BACKEND=CPU ./songgen-rvq-enc-test <aux.gguf> <golden_dir>
// For vocal & bgm: feed bestrq_out_layer{7,3}_*.npy [1,1024,T], compute codes/z_e/quantized,
// compare codes EXACT, z_e/quantized cossim. GATE: codes EXACT both streams, quantized cossim > 0.99.
#include "npy.h"
#include "songgen-septoken.h"

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

static bool run_stream(const char * tag, const SgRvqStream & s, const std::string & feats_path,
                       const std::string & codes_path, const std::string & ze_path, const std::string & quant_path) {
    NpyArray feats = npy_load(feats_path.c_str());  // [1,1024,T]
    NpyArray gcode = npy_load(codes_path.c_str());  // [1,1,T] i64
    NpyArray gze   = npy_load(ze_path.c_str());     // [1,32,T]
    NpyArray gquan = npy_load(quant_path.c_str());  // [1,1024,T]
    int      T     = (int) feats.shape[2];
    fprintf(stderr, "[%s] feats [1,%d,%d]\n", tag, (int) feats.shape[1], T);

    std::vector<int>   codes;
    std::vector<float> z_e, quant;
    sgsep_rvq_encode(s, feats.f32.data(), T, codes, z_e, quant);

    int n_mismatch = 0, first_mm = -1;
    for (int t = 0; t < T; t++) {
        int g = (int) gcode.i64[t];
        if (codes[t] != g) {
            if (first_mm < 0) first_mm = t;
            n_mismatch++;
        }
    }

    double cs_ze = cossim(z_e.data(), gze.f32.data(), (size_t) SGSEP_CB_DIM * T);
    double cs_q  = cossim(quant.data(), gquan.f32.data(), (size_t) SGSEP_MU_DIM * T);

    double se = 0;
    for (size_t i = 0; i < (size_t) SGSEP_MU_DIM * T; i++) {
        double d = (double) quant[i] - gquan.f32[i];
        se += d * d;
    }
    double rmse_q = sqrt(se / ((double) SGSEP_MU_DIM * T));

    printf("---- RVQ encode [%s] vs golden ----\n", tag);
    printf("frames       : %d\n", T);
    printf("codes EXACT  : %s (%d mismatch%s)\n", n_mismatch == 0 ? "yes" : "NO",
           n_mismatch, first_mm >= 0 ? (" first@" + std::to_string(first_mm)).c_str() : "");
    printf("z_e cossim   : %.6f\n", cs_ze);
    printf("quant cossim : %.6f\n", cs_q);
    printf("quant rmse   : %.6e\n", rmse_q);

    bool ok = (n_mismatch == 0) && (cs_q > 0.99);
    printf("RESULT [%s]: %s\n", tag, ok ? "PASS" : "FAIL");
    return ok;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <aux.gguf> <golden_dir>\n", argv[0]);
        return 1;
    }
    std::string aux = argv[1];
    std::string gd  = argv[2];
    auto        gp  = [&](const char * f) { return gd + "/" + f; };

    SonggenSeptoken sep;
    if (!sgsep_load(&sep, aux.c_str())) {
        fprintf(stderr, "aux load failed\n");
        return 1;
    }

    bool ok_v = run_stream("vocal", sep.vocal, gp("bestrq_out_layer7_vocal.npy"), gp("rvq_vocal_codes.npy"),
                           gp("rvq_vocal_z_e.npy"), gp("rvq_vocal_quantized.npy"));
    bool ok_b = run_stream("bgm", sep.bgm, gp("bestrq_out_layer3_bgm.npy"), gp("rvq_bgm_codes.npy"),
                           gp("rvq_bgm_z_e.npy"), gp("rvq_bgm_quantized.npy"));

    bool ok = ok_v && ok_b;
    printf("OVERALL: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 2;
}
