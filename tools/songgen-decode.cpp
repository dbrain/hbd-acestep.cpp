// songgen-decode: SongGeneration audio DECODE driver (codes -> latent -> wav).
//
//   GGML_BACKEND=CPU ./songgen-decode <cfm.gguf> <vae.gguf> <aux.gguf> <gen_tokens.npy> <out.wav>
//
// gen_tokens[1,3,T] (i64) from the LM: codebook 0 = song (unused in sep decode),
// 1 = VOCAL, 2 = BGM. Codes must be 0..16383; the LM may emit EOS=16384 at the tail, so any
// trailing frame whose vocal OR bgm code is out of range (>=16384, i.e. EOS/invalid) is
// dropped, then the kept streams are decoded as a single chunk.
//
//   from_codes -> mu[1024,T] per stream -> inputs_embeds[2,T,2200] -> CFM euler ODE (20 steps,
//   guidance 1.5) -> normfeat denorm -> latent[64,T] -> VAE decode -> wav[2,T*1920] @ 48kHz.
#include "npy.h"
#include "songgen-cfm.h"
#include "songgen-septoken.h"
#include "songgen-vae.h"
#include "wav.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static void stats(const char * tag, const float * a, size_t n) {
    double mn = 1e30, mx = -1e30, sum = 0, sq = 0;
    size_t n_bad = 0;
    for (size_t i = 0; i < n; i++) {
        float v = a[i];
        if (!std::isfinite(v)) { n_bad++; continue; }
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v;
        sq += (double) v * v;
    }
    double mean = sum / (double) n;
    double var  = sq / (double) n - mean * mean;
    double sd   = var > 0 ? sqrt(var) : 0.0;
    printf("%-10s min=%+.5f max=%+.5f mean=%+.5f std=%.5f  (n=%zu nan/inf=%zu)\n", tag, mn, mx, mean, sd, n, n_bad);
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <cfm.gguf> <vae.gguf> <aux.gguf> <gen_tokens.npy> <out.wav>\n", argv[0]);
        return 1;
    }
    const char * cfm_path = argv[1];
    const char * vae_path = argv[2];
    const char * aux_path = argv[3];
    const char * tok_path = argv[4];
    const char * out_path = argv[5];

    NpyArray tok = npy_load(tok_path);  // [1,3,T] i64
    if (tok.shape.size() != 3 || tok.shape[1] < 3) {
        fprintf(stderr, "[decode] expected gen_tokens [1,3,T], got shape with %zu dims\n", tok.shape.size());
        return 1;
    }
    if (tok.dtype != 'q') {
        fprintf(stderr, "[decode] expected i64 gen_tokens, got dtype '%c'\n", tok.dtype);
        return 1;
    }
    int          T_in   = (int) tok.shape[2];
    const int    NCB    = (int) tok.shape[1];
    const int64_t * cb1 = &tok.i64[(size_t) 1 * T_in];  // VOCAL
    const int64_t * cb2 = &tok.i64[(size_t) 2 * T_in];  // BGM
    fprintf(stderr, "[decode] gen_tokens [1,%d,%d]\n", NCB, T_in);

    // Drop trailing EOS/invalid frames: keep the longest valid prefix where both
    // vocal and bgm codes are in 0..16383. EOS = 16384.
    int T = T_in;
    while (T > 0) {
        int64_t cv = cb1[T - 1], cb = cb2[T - 1];
        if (cv >= 0 && cv < SGSEP_CODEBOOK && cb >= 0 && cb < SGSEP_CODEBOOK) break;
        T--;
    }
    int dropped = T_in - T;
    // Any interior out-of-range code is clamped (shouldn't happen with valid LM output).
    int n_clamped = 0;
    std::vector<int> codes_vocal(T), codes_bgm(T);
    for (int t = 0; t < T; t++) {
        int64_t cv = cb1[t], cb = cb2[t];
        if (cv < 0 || cv >= SGSEP_CODEBOOK) { cv = cv < 0 ? 0 : SGSEP_CODEBOOK - 1; n_clamped++; }
        if (cb < 0 || cb >= SGSEP_CODEBOOK) { cb = cb < 0 ? 0 : SGSEP_CODEBOOK - 1; n_clamped++; }
        codes_vocal[t] = (int) cv;
        codes_bgm[t]   = (int) cb;
    }
    fprintf(stderr, "[decode] valid frames=%d  trailing EOS/invalid dropped=%d  interior clamped=%d\n", T, dropped,
            n_clamped);
    if (T <= 0) {
        fprintf(stderr, "[decode] no valid frames after EOS drop\n");
        return 1;
    }

    SonggenSeptoken sep;
    if (!sgsep_load(&sep, aux_path)) {
        fprintf(stderr, "[decode] aux load failed\n");
        return 1;
    }
    SonggenCfm cfm;
    if (!sgcfm_load(&cfm, cfm_path)) {
        fprintf(stderr, "[decode] cfm load failed\n");
        return 1;
    }
    SonggenVae vae;
    if (!sgvae_load(&vae, vae_path)) {
        fprintf(stderr, "[decode] vae load failed\n");
        return 1;
    }

    std::vector<float> latent;  // [64*T] channel-major
    sgsep_decode(&sep, &cfm, codes_vocal, codes_bgm, /*seed=*/0, latent);
    stats("latent", latent.data(), latent.size());

    std::vector<float> audio;  // planar [ch0(T_audio), ch1(T_audio)]
    int                T_audio = sgvae_decode(&vae, latent.data(), T, audio);
    stats("wav", audio.data(), audio.size());

    const int sr = 48000;
    if (!write_wav_s16_planar(out_path, audio.data(), T_audio, sr)) {
        fprintf(stderr, "[decode] wav write failed\n");
        return 1;
    }
    double dur = (double) T_audio / (double) sr;
    printf("---- songgen decode ----\n");
    printf("valid frames     : %d (dropped %d trailing EOS/invalid)\n", T, dropped);
    printf("latent           : [64,%d]\n", T);
    printf("samples / channel: %d  (2 ch)\n", T_audio);
    printf("expected samples : %d (= %d frames * 1920)\n", T * 1920, T);
    printf("duration         : %.3f s @ %d Hz\n", dur, sr);
    printf("output           : %s\n", out_path);

    sgvae_free(&vae);
    sgcfm_free(&cfm);
    return 0;
}
