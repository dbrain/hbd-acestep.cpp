// songgen-tok-test: GATE for the in-C++ Qwen2 conditioning tokenizer.
//   ./songgen-tok-test <lelm.gguf> <golden-large-dir>
// Tokenizes the golden prompt and asserts the produced cond/uncond arrays
// EXACTLY equal golden-large/*.npy (bit-for-bit token ids).
#include "npy.h"
#include "songgen-tokenizer.h"

#include <cstdio>
#include <string>
#include <vector>

static const char * LYRIC =
    "[verse] walking through the quiet streets at night [verse] city lights are "
    "calling out my name [chorus] and i will find my way back home";
static const char * DESCRIPTION = "warm indie folk, male vocal, acoustic guitar, 92 bpm";

static int cmp_arr(const char * name, const int32_t * got, int n, const NpyArray & gold) {
    if ((int) npy_numel(gold) != n || gold.dtype != 'i') {
        printf("FAIL %-18s gold shape/dtype mismatch (numel=%zu dtype=%c)\n", name, npy_numel(gold), gold.dtype);
        return 1;
    }
    for (int i = 0; i < n; i++) {
        if (got[i] != gold.i32[i]) {
            printf("FAIL %-18s first mismatch at %d: got=%d expected=%d\n", name, i, got[i], gold.i32[i]);
            return 1;
        }
    }
    printf("PASS %-18s (%d ids)\n", name, n);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <lelm.gguf> <golden-large-dir>\n", argv[0]);
        return 2;
    }
    const char * gguf_path = argv[1];
    std::string  gd        = argv[2];
    auto         gp        = [&](const char * f) { return gd + "/" + f; };

    songgen_tok::Tokenizer tok;
    if (!songgen_tok::load(&tok, gguf_path)) {
        fprintf(stderr, "tokenizer load failed\n");
        return 2;
    }

    songgen_tok::ConditioningArrays a;
    songgen_tok::build(&tok, LYRIC, DESCRIPTION, &a);

    int fails = 0;
    fails += cmp_arr("desc_ids_cond", a.desc_ids_cond.data(), songgen_tok::DESC_MAX_LEN,
                     npy_load(gp("desc_ids_cond.npy").c_str()));
    fails += cmp_arr("desc_cover_cond", a.desc_cover_cond.data(), songgen_tok::DESC_MAX_LEN,
                     npy_load(gp("desc_cover_cond.npy").c_str()));
    fails += cmp_arr("desc_ids_uncond", a.desc_ids_uncond.data(), songgen_tok::DESC_MAX_LEN,
                     npy_load(gp("desc_ids_uncond.npy").c_str()));
    fails += cmp_arr("desc_cover_uncond", a.desc_cover_uncond.data(), songgen_tok::DESC_MAX_LEN,
                     npy_load(gp("desc_cover_uncond.npy").c_str()));
    fails += cmp_arr("type_ids_cond", a.type_ids_cond.data(), songgen_tok::TYPE_MAX_LEN,
                     npy_load(gp("type_ids_cond.npy").c_str()));
    fails += cmp_arr("type_ids_uncond", a.type_ids_uncond.data(), songgen_tok::TYPE_MAX_LEN,
                     npy_load(gp("type_ids_uncond.npy").c_str()));

    printf("\n%s (%d/%d arrays matched)\n", fails == 0 ? "ALL PASS" : "FAILED", 6 - fails, 6);
    return fails == 0 ? 0 : 1;
}
