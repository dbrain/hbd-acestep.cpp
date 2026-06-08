// sa3-tok-test: validate Gemma BPE tokenizer against golden tokens.npy.
//   ./sa3-tok-test <t5gemma.gguf> <golden_dir> "<prompt>"
#include "npy.h"
#include "sa3-tokenizer.h"
#include "gguf.h"

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <gguf> <golden_dir> <prompt>\n", argv[0]); return 1; }
    std::string gguf = argv[1], gd = argv[2], prompt = argv[3];

    struct ggml_context * meta = nullptr;
    struct gguf_init_params p = { true, &meta };
    struct gguf_context * gf = gguf_init_from_file(gguf.c_str(), p);
    if (!gf) { fprintf(stderr, "cannot open %s\n", gguf.c_str()); return 1; }

    SA3Tokenizer tok;
    if (!sa3tok_load_from_gguf(&tok, gf)) return 1;

    std::vector<int> ids, valid;
    int n = sa3tok_encode_padded(&tok, prompt, 256, ids, valid);
    printf("prompt: \"%s\"  -> %d tokens\n", prompt.c_str(), n);
    printf("ids: "); for (int i = 0; i < n; i++) printf("%d ", ids[i]); printf("\n");

    NpyArray g = npy_load((gd + "/tokens.npy").c_str());
    int S = (int) g.shape[g.shape.size()-1];
    int mism = 0, gvalid = 0;
    NpyArray am = npy_load((gd + "/attn_mask.npy").c_str());
    for (int i = 0; i < S; i++) gvalid += (int) am.i32[i];
    for (int i = 0; i < S; i++) if (ids[i] != (int) g.i32[i]) { if (mism < 8) printf("  mismatch @%d: ours=%d gold=%d\n", i, ids[i], (int) g.i32[i]); mism++; }
    bool ok = (n == gvalid) && (mism == 0);
    printf("golden valid=%d ours=%d  mismatches=%d\n", gvalid, n, mism);
    printf("RESULT: %s\n", ok ? "PASS" : "FAIL");
    gguf_free(gf); ggml_free(meta);
    return ok ? 0 : 2;
}
