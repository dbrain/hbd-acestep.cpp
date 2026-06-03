// songgen-decode-chunk-test: validate chunked CFM+VAE decode continuity.
//   Loads golden gen_tokens [1,3,T0], tiles vocal+bgm to ~target frames, runs sgpost_decode
//   with a small chunk_frames to force chunking, and reports seam-region sample stats
//   (max |delta| across the expected seam sample) plus a baseline non-seam delta for scale.
//
//   GGML_BACKEND=CPU ./songgen-decode-chunk-test <cfm.gguf> <vae.gguf> <septoken-aux.gguf>
//       <gen_tokens.npy> <out.wav> [target_frames] [chunk_frames]
#include "npy.h"
#include "songgen-cfm.h"
#include "songgen-decode-post.h"
#include "songgen-septoken.h"
#include "songgen-vae.h"
#include "wav.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

static void seam_stats(const float * audio, int T_audio, int up, int hop_f, int n_seams) {
    // seams in concatenated output land near hop_f*up*seg in the time axis (post cross-fade
    // they should be continuous). Report the largest |x[t]-x[t-1]| jump near each seam vs the
    // global mean |delta| for scale.
    const float * L = audio;
    double        gsum = 0;
    int           gn   = 0;
    for (int t = 1; t < T_audio; t++) { gsum += fabs((double) L[t] - L[t - 1]); gn++; }
    double gmean = gn ? gsum / gn : 0;
    printf("[seam] global mean |dL| = %.6e over %d samples\n", gmean, gn);
    for (int s = 1; s <= n_seams; s++) {
        int    center = s * hop_f * up;
        if (center <= 1 || center >= T_audio) continue;
        int    lo = std::max(1, center - 64), hi = std::min(T_audio, center + 64);
        double mx = 0;
        int    at = lo;
        for (int t = lo; t < hi; t++) {
            double d = fabs((double) L[t] - L[t - 1]);
            if (d > mx) { mx = d; at = t; }
        }
        printf("[seam] seam %d near sample %d: max |dL| = %.6e at %d (%.1fx global)\n", s, center, mx, at,
               gmean > 0 ? mx / gmean : 0.0);
    }
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr,
                "usage: %s <cfm.gguf> <vae.gguf> <septoken-aux.gguf> <gen_tokens.npy> <out.wav> "
                "[target_frames=1000] [chunk_frames=500]\n",
                argv[0]);
        return 1;
    }
    const char * cfm_path = argv[1], *vae_path = argv[2], *aux_path = argv[3], *tok_path = argv[4],
               *out_path = argv[5];
    int target = argc > 6 ? atoi(argv[6]) : 1000;
    int chunk  = argc > 7 ? atoi(argv[7]) : 500;

    NpyArray a = npy_load(tok_path);
    if (a.shape.size() != 3 || a.shape[1] < 3) {
        fprintf(stderr, "[chunk-test] expected [1,3,T] gen_tokens, got %zud\n", a.shape.size());
        return 1;
    }
    int T0 = (int) a.shape[2];
    const int64_t * d = a.i64.data();
    auto clampc = [](int64_t v) { return (int) (v < 0 ? 0 : v >= SGSEP_CODEBOOK ? SGSEP_CODEBOOK - 1 : v); };
    std::vector<int> base_v(T0), base_b(T0);
    for (int t = 0; t < T0; t++) { base_v[t] = clampc(d[1 * T0 + t]); base_b[t] = clampc(d[2 * T0 + t]); }

    std::vector<int> cv, cb;
    while ((int) cv.size() < target) {
        cv.insert(cv.end(), base_v.begin(), base_v.end());
        cb.insert(cb.end(), base_b.begin(), base_b.end());
    }
    cv.resize(target);
    cb.resize(target);
    fprintf(stderr, "[chunk-test] tiled to %d frames (%.1fs), chunk_frames=%d\n", target, target / 25.0f, chunk);

    SonggenSeptoken sep;
    if (!sgsep_load(&sep, aux_path)) return 1;
    SonggenCfm cfm;
    if (!sgcfm_load(&cfm, cfm_path)) return 1;
    SonggenVae vae;
    if (!sgvae_load(&vae, vae_path)) return 1;

    SgGenParams P;
    P.max_gen_len  = target;
    P.chunk_frames = chunk;
    P.fade_ms      = 200.0f;

    std::vector<float> audio;
    int                T_audio = sgpost_decode(&sep, &cfm, &vae, cv, cb, 1234, P, audio);

    // finiteness
    size_t n_bad = 0;
    for (float v : audio) if (!std::isfinite(v)) n_bad++;
    printf("[chunk-test] T_audio=%d samples (%.2fs), nan/inf=%zu\n", T_audio, T_audio / 48000.0f, n_bad);

    int hop_f = chunk / 4 * 3;
    seam_stats(audio.data(), T_audio, 1920, hop_f, target / hop_f + 1);

    sgpost_fade(audio.data(), T_audio, 48000, P.fade_ms, 10.0f);
    write_wav_s16_planar(out_path, audio.data(), T_audio, 48000);
    printf("[chunk-test] wrote %s\n", out_path);

    sgvae_free(&vae);
    sgcfm_free(&cfm);
    return 0;
}
