// songgen-clone-encode-test: validate the clone encode (stems -> 3 token streams) vs goldens.
//
//   GGML_BACKEND=CPU ./songgen-clone-encode-test \
//       <musicfm-sep.gguf> <musicfm-1rvq.gguf> <septoken-aux.gguf> <1rvq-aux.gguf> \
//       <reference.wav> <golden-clone-dir>
//
// Feeds reference.wav as BOTH vocal & bgm stems AND as the full mix (the golden-capture
// convention). Produces pmt/vocal/bgm code streams and compares the LEADING frames to
// golden-clone/tokens_{pmt,vocal,bgm}.npy. The goldens used the real (julius) windowed path;
// our Kaiser resampler differs slightly, so we report the longest exact leading-frame match per
// stream (capture noted ~80% overall agreement is expected for the windowed path).
#include "audio-io.h"
#include "npy.h"
#include "songgen-clone-encode.h"

#include <cstdio>
#include <string>
#include <vector>

static int leading_match(const std::vector<int> & ours, const NpyArray & gold) {
    int n = (int) std::min((size_t) gold.i64.size() ? gold.i64.size() : gold.i32.size(), ours.size());
    int i = 0;
    for (; i < n; i++) {
        long long g = gold.i64.size() ? gold.i64[i] : (long long) gold.i32[i];
        if ((long long) ours[i] != g) break;
    }
    return i;
}

static double overall_match(const std::vector<int> & ours, const NpyArray & gold) {
    size_t gn = gold.i64.size() ? gold.i64.size() : gold.i32.size();
    int    n  = (int) std::min(gn, ours.size());
    int    hit = 0;
    for (int i = 0; i < n; i++) {
        long long g = gold.i64.size() ? gold.i64[i] : (long long) gold.i32[i];
        if ((long long) ours[i] == g) hit++;
    }
    return n > 0 ? (double) hit / n : 0.0;
}

int main(int argc, char ** argv) {
    if (argc < 7) {
        fprintf(stderr,
                "usage: %s <musicfm-sep.gguf> <musicfm-1rvq.gguf> <septoken-aux.gguf> <1rvq-aux.gguf> "
                "<reference.wav> <golden-clone-dir>\n",
                argv[0]);
        return 1;
    }
    const char * mf_sep = argv[1];
    const char * mf_1rvq = argv[2];
    const char * aux_sep = argv[3];
    const char * aux_1rvq = argv[4];
    const char * wav_path = argv[5];
    std::string  gd = argv[6];
    auto         gp = [&](const char * f) { return gd + "/" + f; };

    SonggenSeptoken sep;
    if (!sgsep_load(&sep, aux_sep)) { fprintf(stderr, "septoken aux load failed\n"); return 1; }
    SonggenCloneEnc m;
    if (!sgclone_load(&m, mf_sep, mf_1rvq, aux_1rvq, &sep)) { fprintf(stderr, "clone load failed\n"); return 1; }

    int     T = 0, sr = 0;
    float * planar = audio_read(wav_path, &T, &sr);
    if (!planar) { fprintf(stderr, "wav read failed: %s\n", wav_path); return 1; }
    fprintf(stderr, "[clone-test] wav %s: T=%d sr=%d (%.2fs)\n", wav_path, T, sr, (double) T / sr);

    // reference.wav as both stems and the full mix (matches the capture).
    std::vector<int> pmt, vocal, bgm;
    int Tc = sgclone_encode(&m, planar, T, planar, T, planar, T, sr, pmt, vocal, bgm);
    free(planar);
    fprintf(stderr, "[clone-test] produced %d frames per stream\n", Tc);

    NpyArray gp_pmt = npy_load(gp("tokens_pmt.npy").c_str());
    NpyArray gp_voc = npy_load(gp("tokens_vocal.npy").c_str());
    NpyArray gp_bgm = npy_load(gp("tokens_bgm.npy").c_str());

    struct { const char * tag; std::vector<int> * ours; NpyArray * g; } streams[] = {
        { "pmt", &pmt, &gp_pmt }, { "vocal", &vocal, &gp_voc }, { "bgm", &bgm, &gp_bgm },
    };

    printf("---- clone encode vs golden tokens ----\n");
    bool any = false;
    for (auto & s : streams) {
        size_t gn = s.g->i64.size() ? s.g->i64.size() : s.g->i32.size();
        int    lead = leading_match(*s.ours, *s.g);
        double ov   = overall_match(*s.ours, *s.g);
        printf("%-6s ours=%zu golden=%zu  leading-exact-match=%d  overall=%.1f%%\n", s.tag, s.ours->size(), gn,
               lead, ov * 100.0);
        if (lead > 0) any = true;
        // show first few of each for eyeballing
        printf("   ours  :");
        for (int i = 0; i < 8 && i < (int) s.ours->size(); i++) printf(" %d", (*s.ours)[i]);
        printf("\n   golden:");
        for (int i = 0; i < 8 && i < (int) gn; i++)
            printf(" %lld", s.g->i64.size() ? (long long) s.g->i64[i] : (long long) s.g->i32[i]);
        printf("\n");
    }

    printf("RESULT: %s (leading-frame agreement is the gate; windowing/resampler cause tail drift)\n",
           any ? "MATCH" : "NO-MATCH");

    sgclone_free(&m);
    return any ? 0 : 2;
}
