// sa3-t5gemma-test: validate the T5Gemma encoder port against the Python golden.
//   ./sa3-t5gemma-test <sa3-t5gemma.gguf> <golden_dir>
// Loads tokens.npy [1,S] i32, attn_mask.npy [1,S] i32, t5gemma_out.npy [1,S,768] f32.
#include "npy.h"
#include "sa3-t5gemma-enc.h"

#include <cmath>
#include <cstdio>
#include <vector>

static double cossim(const float * a, const float * b, size_t n) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) { dot += (double) a[i] * b[i]; na += (double) a[i] * a[i]; nb += (double) b[i] * b[i]; }
    return (na == 0 || nb == 0) ? 0.0 : dot / (sqrt(na) * sqrt(nb));
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <gguf> <golden_dir>\n", argv[0]); return 1; }
    std::string gguf = argv[1], gd = argv[2];
    auto gp = [&](const char * f) { return gd + "/" + f; };

    NpyArray tok = npy_load(gp("tokens.npy").c_str());
    NpyArray am  = npy_load(gp("attn_mask.npy").c_str());
    NpyArray ref = npy_load(gp("t5gemma_out.npy").c_str());
    int S = (int) tok.shape[tok.shape.size() - 1];
    int H = (int) ref.shape[ref.shape.size() - 1];
    fprintf(stderr, "[golden] S=%d H=%d  (ref %zu elems)\n", S, H, npy_numel(ref));

    std::vector<int> ids(S), valid(S);
    for (int i = 0; i < S; i++) ids[i]   = (int) tok.i32[i];
    for (int i = 0; i < S; i++) valid[i] = (int) am.i32[i];
    int nvalid = 0; for (int i = 0; i < S; i++) nvalid += valid[i];
    fprintf(stderr, "[golden] valid tokens: %d/%d\n", nvalid, S);

    SA3T5GModel m;
    if (!sa3t5g_load(&m, gguf.c_str())) { fprintf(stderr, "load failed\n"); return 1; }

    std::vector<float> out((size_t) H * S);
    sa3t5g_encode(&m, ids.data(), valid.data(), S, out.data());

    // compare full [S,H] and valid-only region
    size_t n = (size_t) H * S;
    double cs_all = cossim(out.data(), ref.f32.data(), n);
    double cs_val = cossim(out.data(), ref.f32.data(), (size_t) H * nvalid);
    double maxabs = 0, se = 0;
    for (size_t i = 0; i < (size_t) H * nvalid; i++) { double d = out[i] - ref.f32[i]; se += d * d; if (fabs(d) > maxabs) maxabs = fabs(d); }
    printf("---- T5Gemma encoder vs golden ----\n");
    printf("cossim (all %d tok)   : %.6f\n", S, cs_all);
    printf("cossim (valid %d tok) : %.6f\n", nvalid, cs_val);
    printf("rmse  (valid)         : %.6e\n", sqrt(se / ((size_t) H * nvalid)));
    printf("maxabs(valid)         : %.6e\n", maxabs);
    bool ok = cs_val > 0.999;
    printf("RESULT: %s (valid cossim %.6f %s 0.999)\n", ok ? "PASS" : "FAIL", cs_val, ok ? ">" : "<=");
    sa3t5g_free(&m);
    return ok ? 0 : 2;
}
