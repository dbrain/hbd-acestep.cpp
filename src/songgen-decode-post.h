// songgen-decode-post.h : shared post-LM decode orchestration + wav post-processing.
//
//   - SgGenParams: CLI-tunable generation/decode knobs (length, sampling, fade, gen_type, chunk).
//   - sgpost_fade: in-place fade-in/out on a planar-stereo wav to kill boundary clicks.
//   - sgpost_decode: codes(vocal,bgm)[T] -> planar-stereo wav. Single-segment for short T,
//     chunked CFM-ODE+VAE with latent incontext + linear wav cross-fade for long T
//     (port of generate_septoken.code2sound: chunk=duration*25 frames, hop=3/4, overlap=1/4).
#pragma once

#include "songgen-cfm.h"
#include "songgen-septoken.h"
#include "songgen-vae.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

enum SgGenType { SG_MIXED = 0, SG_VOCAL = 1, SG_BGM = 2 };

struct SgGenParams {
    int       max_gen_len = 375;    // frames; round(duration*25)
    int       top_k       = 5000;   // cb0 only (LeVo full-mem generate(); overridden by --top-k)
    float     temp        = 0.8f;   // LeVo full-mem generate() (overridden by --temp)
    float     cfg_coef    = 1.5f;   // LeVo inference_coef (config.yaml)
    float     rep_penalty = 1.1f;   // LeVo logits /= 1.1**count over record_win (lm_levo.py)
    int       record_win  = 50;     // LeVo record_window=50 (was 150); NOT CLI-overridable, so this is the effective value
    float     fade_ms     = 200.0f; // fade-out length; fade-in is min(10ms, fade_ms)
    SgGenType gen_type    = SG_MIXED;
    // chunked decode: enabled when max_gen_len > chunk_frames. Frame = 40ms (25 fps).
    int       chunk_frames = 750;   // 30 s per chunk window (== code2sound duration*25 default)
};

static const char * sgpost_gen_type_name(SgGenType t) {
    return t == SG_VOCAL ? "vocal" : t == SG_BGM ? "bgm" : "mixed";
}

static bool sgpost_parse_gen_type(const std::string & s, SgGenType * out) {
    if (s == "mixed") { *out = SG_MIXED; return true; }
    if (s == "vocal") { *out = SG_VOCAL; return true; }
    if (s == "bgm" || s == "pure-music" || s == "pure_music") { *out = SG_BGM; return true; }
    return false;
}

// gen_type stream override (codeclm.generate_audio): bgm zeroes the vocal code stream to the
// constant 3142; vocal zeroes the bgm stream to 9670. Applied to the LM-produced codes before
// from_codes. Done in code space (clamped 0..16383 already).
static void sgpost_apply_gen_type(SgGenType gt, std::vector<int> & codes_vocal, std::vector<int> & codes_bgm) {
    if (gt == SG_BGM) {
        std::fill(codes_vocal.begin(), codes_vocal.end(), 3142);
    } else if (gt == SG_VOCAL) {
        std::fill(codes_bgm.begin(), codes_bgm.end(), 9670);
    }
}

// In-place fade on planar stereo wav [L:T_audio][R:T_audio] at sr Hz.
// Cosine-squared fade-out over the last fade_out_ms; short linear-ish (cos) fade-in over
// fade_in_ms. Always finite-safe (caller already clamps on write).
static void sgpost_fade(float * audio, int T_audio, int sr, float fade_out_ms, float fade_in_ms) {
    if (T_audio <= 1) return;
    float * L = audio;
    float * R = audio + T_audio;

    int n_out = (int) (fade_out_ms * 0.001f * (float) sr);
    n_out     = std::min(n_out, T_audio);
    for (int i = 0; i < n_out; i++) {
        // i=0 (start of fade window) -> gain 1; last sample -> gain 0.
        float x   = (float) (n_out - 1 - i) / (float) std::max(1, n_out - 1);
        float g   = 0.5f * (1.0f - cosf((float) M_PI * x));  // cos^2-equivalent smoothstep
        int   idx = T_audio - n_out + i;
        L[idx] *= g;
        R[idx] *= g;
    }

    int n_in = (int) (fade_in_ms * 0.001f * (float) sr);
    n_in     = std::min(n_in, T_audio);
    for (int i = 0; i < n_in; i++) {
        float x = (float) i / (float) std::max(1, n_in - 1);
        float g = 0.5f * (1.0f - cosf((float) M_PI * x));
        L[i] *= g;
        R[i] *= g;
    }
}

// Decode one segment's normalized latent (time-major [T*64]) -> planar stereo wav via VAE.
// latent_norm is denormalized to channel-major [64*T] first (= return_sample).
static int sgpost_vae_decode_norm(SonggenVae * vae, SonggenSeptoken * sep, const std::vector<float> & latent_norm,
                                  int T, std::vector<float> & wav_planar) {
    std::vector<float> lat_cm((size_t) SGSEP_LAT_DIM * T);
    for (int c = 0; c < SGSEP_LAT_DIM; c++) {
        float mean = sep->nf_mean[c], std = sep->nf_std[c];
        for (int t = 0; t < T; t++) lat_cm[(size_t) c * T + t] = latent_norm[(size_t) t * SGSEP_LAT_DIM + c] * std + mean;
    }
    return sgvae_decode(vae, lat_cm.data(), T, wav_planar);
}

// Full codes -> planar stereo wav. Chunks the CFM-ODE+VAE decode when T_frames exceeds
// chunk_frames; otherwise single segment. Returns T_audio (samples per channel).
//
// Chunk layout (code2sound): min=chunk_frames, hop=min*3/4, ovlp=min-hop. Each window decodes
// chunk_frames; for windows after the first the first ovlp frames are seeded with the previous
// segment's overlap latent (incontext, mask row 1) so the latent is continuous; the decoded wav
// is then linearly cross-faded over the overlap samples (ovlp_frames * 1920) and concatenated.
static int sgpost_decode(SonggenSeptoken * sep, SonggenCfm * cfm, SonggenVae * vae,
                         std::vector<int> codes_vocal, std::vector<int> codes_bgm, uint64_t seed,
                         const SgGenParams & p, std::vector<float> & wav_out) {
    int T = (int) codes_vocal.size();
    sgpost_apply_gen_type(p.gen_type, codes_vocal, codes_bgm);

    const int up = 1920;  // VAE upsample (48k, 25fps frames)

    // SG_CHUNK_FRAMES env overrides chunk_frames (decode-stage VRAM knob: smaller = lower peak,
    // more crossfaded segments). 0/unset -> the SgGenParams default.
    int chunk_frames = p.chunk_frames;
    if (const char * e = getenv("SG_CHUNK_FRAMES")) {
        int v = atoi(e);
        if (v > 0) chunk_frames = v;
    }

    if (T <= chunk_frames) {
        std::vector<float> latent;
        sgsep_decode(sep, cfm, codes_vocal, codes_bgm, (int) seed, latent);
        int T_audio = sgvae_decode(vae, latent.data(), T, wav_out);
        return T_audio;
    }

    int min_f = chunk_frames;
    int hop_f = min_f / 4 * 3;
    int ovlp_f = min_f - hop_f;
    int target_audio = T * up;

    fprintf(stderr, "[post] chunked decode: T=%d frames, chunk=%d hop=%d ovlp=%d (%.1fs/%.1fs overlap)\n", T, min_f,
            hop_f, ovlp_f, (float) min_f / 25.0f, (float) ovlp_f / 25.0f);

    // pad codes (repeat) so windows tile: len = ceil((T-ovlp)/hop)*hop + ovlp.
    int len_codes = T;
    if ((T - ovlp_f) % hop_f > 0) len_codes = ((T - ovlp_f + hop_f - 1) / hop_f) * hop_f + ovlp_f;
    while ((int) codes_vocal.size() < len_codes) {
        size_t cur = codes_vocal.size();
        codes_vocal.insert(codes_vocal.end(), codes_vocal.begin(), codes_vocal.begin() + (size_t) std::min((size_t) (len_codes - cur), cur));
        codes_bgm.insert(codes_bgm.end(), codes_bgm.begin(), codes_bgm.begin() + (size_t) std::min((size_t) (len_codes - codes_bgm.size()), codes_bgm.size()));
    }
    codes_vocal.resize(len_codes);
    codes_bgm.resize(len_codes);

    std::vector<float> prev_overlap_norm;  // last ovlp_f frames of previous chunk, normalized [ovlp_f*64]
    std::vector<float> outL, outR;         // accumulated planar channels
    int                seg_idx = 0;
    int                ncodes  = (int) codes_vocal.size();

    for (int sinx = 0; sinx < ncodes - hop_f; sinx += hop_f) {
        int beg = sinx;
        int end = std::min(beg + min_f, ncodes);
        int Tc  = end - beg;
        std::vector<int> cv(codes_vocal.begin() + beg, codes_vocal.begin() + end);
        std::vector<int> cb(codes_bgm.begin() + beg, codes_bgm.begin() + end);

        int incontext = (seg_idx == 0) ? 0 : (int) (prev_overlap_norm.size() / SGSEP_LAT_DIM);
        if (incontext > Tc) incontext = Tc;

        std::vector<float> in_lat;
        if (incontext > 0) {
            in_lat.assign(prev_overlap_norm.begin(), prev_overlap_norm.begin() + (size_t) incontext * SGSEP_LAT_DIM);
        }

        std::vector<float> lat_norm, lat_cm;
        // vary seed per segment so repeated tiling does not produce identical noise.
        sgsep_decode_seg(sep, cfm, cv, cb, seed + (uint64_t) seg_idx, incontext, in_lat, lat_norm, lat_cm);

        // stash this chunk's trailing overlap (normalized) for the next incontext.
        prev_overlap_norm.assign(lat_norm.end() - (size_t) ovlp_f * SGSEP_LAT_DIM, lat_norm.end());

        std::vector<float> seg_wav;  // planar [Tc*up *2]
        int                Ta = sgvae_decode(vae, lat_cm.data(), Tc, seg_wav);
        const float *      sL = seg_wav.data();
        const float *      sR = seg_wav.data() + Ta;

        int ovlp_s = ovlp_f * up;
        if (seg_idx == 0) {
            outL.assign(sL, sL + Ta);
            outR.assign(sR, sR + Ta);
        } else {
            // linear cross-fade over the first ovlp_s samples against the tail of out.
            int cur = (int) outL.size();
            int ov  = std::min({ ovlp_s, cur, Ta });
            for (int i = 0; i < ov; i++) {
                float w = (float) i / (float) std::max(1, ovlp_s - 1);  // 0..1 over window
                int   oi = cur - ov + i;
                outL[oi] = outL[oi] * (1.0f - w) + sL[i] * w;
                outR[oi] = outR[oi] * (1.0f - w) + sR[i] * w;
            }
            // append the non-overlapped remainder.
            outL.insert(outL.end(), sL + ov, sL + Ta);
            outR.insert(outR.end(), sR + ov, sR + Ta);
        }
        seg_idx++;
    }

    int T_audio = std::min((int) outL.size(), target_audio);
    wav_out.resize((size_t) 2 * T_audio);
    for (int t = 0; t < T_audio; t++) wav_out[t] = outL[t];
    for (int t = 0; t < T_audio; t++) wav_out[T_audio + t] = outR[t];
    fprintf(stderr, "[post] chunked decode done: %d segments, %d samples (target %d)\n", seg_idx, T_audio, target_audio);
    return T_audio;
}

// CONTINUATION decode (port of generate_septoken.code2sound, the prompt_vocal/bgm branch).
//
//   gen codes are the LM-generated streams. prompt_codes_{vocal,bgm}[Tprompt_codes] are PREPENDED
//   (code2sound:223-224 cat([first_latent_codes, codes])). prompt_latent_norm is the VAE-encoded
//   prompt (vocal+bgm summed -> VAE.encode_audio -> project_sample), TIME-MAJOR [Tlat*64], already
//   normalized; its length Tlat = first_latent_length (code2sound:217-218). The FIRST decode window
//   (sinx==0) runs scenario='other_seg' with incontext_length=first_latent_length, seeding/freezing
//   x_next[:Tlat] toward the prompt latent (code2sound:254-255). After all windows, the prompt
//   region is trimmed: latent_list[0]=latent_list[0][:,first_latent_length:] (code2sound:266) and
//   the wav is clipped to target_len = (codes_len - first_latent_codes_length)*1920 (code2sound:228).
//
// FRAME BOOKKEEPING (everything @ 25 fps, VAE up=1920 @ 48 kHz):
//   in : prompt_latent_norm = Tlat latent frames (true_latent), prompt codes = Tpc code frames.
//        Tlat and Tpc need not match exactly (VAE T = samples/1920 vs sound2code int(s/sr*25)+1);
//        the reference trims the LATENT by Tlat and the WAV by (total_codes - Tpc)*1920 independently.
//   out: trimmed wav = the GENERATED region only (prompt seed removed), n_gen_codes*1920 samples.
static int sgpost_decode_continue(SonggenSeptoken * sep, SonggenCfm * cfm, SonggenVae * vae,
                                  std::vector<int> gen_vocal, std::vector<int> gen_bgm,
                                  const std::vector<int> & prompt_vocal, const std::vector<int> & prompt_bgm,
                                  const std::vector<float> & prompt_latent_norm, int first_latent_length,
                                  uint64_t seed, const SgGenParams & p, std::vector<float> & wav_out) {
    const int up  = 1920;
    int       Tpc = (int) prompt_vocal.size();
    if ((int) prompt_bgm.size() != Tpc) {
        fprintf(stderr, "[post] FATAL: prompt vocal/bgm code length mismatch (%d vs %d)\n", Tpc, (int) prompt_bgm.size());
        exit(1);
    }
    int n_gen = (int) gen_vocal.size();

    // gen_type override applies to the GENERATED stream only (prompt is real audio already).
    sgpost_apply_gen_type(p.gen_type, gen_vocal, gen_bgm);

    // prepend prompt codes (code2sound:223-224).
    std::vector<int> codes_vocal, codes_bgm;
    codes_vocal.reserve((size_t) Tpc + n_gen);
    codes_bgm.reserve((size_t) Tpc + n_gen);
    codes_vocal.insert(codes_vocal.end(), prompt_vocal.begin(), prompt_vocal.end());
    codes_vocal.insert(codes_vocal.end(), gen_vocal.begin(), gen_vocal.end());
    codes_bgm.insert(codes_bgm.end(), prompt_bgm.begin(), prompt_bgm.end());
    codes_bgm.insert(codes_bgm.end(), gen_bgm.begin(), gen_bgm.end());

    int Tlat = first_latent_length;
    if (Tlat > (int) (prompt_latent_norm.size() / SGSEP_LAT_DIM)) Tlat = (int) (prompt_latent_norm.size() / SGSEP_LAT_DIM);
    if (Tlat < 0) Tlat = 0;

    // target_len excludes the prompt codes (code2sound:228).
    int total_codes  = (int) codes_vocal.size();
    int target_audio = (total_codes - Tpc) * up;

    fprintf(stderr, "[post] continuation: prompt codes=%d latent frames=%d, gen codes=%d, total=%d -> target %d samples\n",
            Tpc, Tlat, n_gen, total_codes, target_audio);

    int min_f  = p.chunk_frames;
    if (const char * e = getenv("SG_CHUNK_FRAMES")) {
        int v = atoi(e);
        if (v > 0) min_f = v;
    }
    int hop_f  = min_f / 4 * 3;
    int ovlp_f = min_f - hop_f;

    // code repeat to fill >= one chunk then to tile (code2sound:231-245).
    while ((int) codes_vocal.size() < min_f) {
        size_t cur = codes_vocal.size();
        codes_vocal.insert(codes_vocal.end(), codes_vocal.begin(), codes_vocal.begin() + cur);
        codes_bgm.insert(codes_bgm.end(), codes_bgm.begin(), codes_bgm.begin() + cur);
    }
    codes_vocal.resize(std::max((int) codes_vocal.size(), min_f));
    codes_bgm.resize(codes_vocal.size());
    int len_codes = (int) codes_vocal.size();
    if ((len_codes - ovlp_f) % hop_f > 0) {
        len_codes = ((len_codes - ovlp_f + hop_f - 1) / hop_f) * hop_f + ovlp_f;
        while ((int) codes_vocal.size() < len_codes) {
            size_t cur = codes_vocal.size();
            codes_vocal.insert(codes_vocal.end(), codes_vocal.begin(),
                               codes_vocal.begin() + std::min((size_t) (len_codes - cur), cur));
            codes_bgm.insert(codes_bgm.end(), codes_bgm.begin(),
                             codes_bgm.begin() + std::min((size_t) (len_codes - codes_bgm.size()), codes_bgm.size()));
        }
        codes_vocal.resize(len_codes);
        codes_bgm.resize(len_codes);
    }

    std::vector<float> prev_overlap_norm;
    std::vector<float> outL, outR;
    int                seg_idx = 0;
    int                ncodes  = (int) codes_vocal.size();

    for (int sinx = 0; sinx < ncodes - hop_f; sinx += hop_f) {
        int beg = sinx;
        int end = std::min(beg + min_f, ncodes);
        int Tc  = end - beg;
        std::vector<int> cv(codes_vocal.begin() + beg, codes_vocal.begin() + end);
        std::vector<int> cb(codes_bgm.begin() + beg, codes_bgm.begin() + end);

        int                incontext;
        std::vector<float> in_lat;
        if (seg_idx == 0) {
            // first window: in-context = the VAE-encoded prompt latent (code2sound:254-255).
            incontext = std::min(Tlat, Tc);
            if (incontext > 0) {
                in_lat.assign(prompt_latent_norm.begin(),
                              prompt_latent_norm.begin() + (size_t) incontext * SGSEP_LAT_DIM);
            }
        } else {
            incontext = std::min((int) (prev_overlap_norm.size() / SGSEP_LAT_DIM), Tc);
            if (incontext > 0) {
                in_lat.assign(prev_overlap_norm.begin(), prev_overlap_norm.begin() + (size_t) incontext * SGSEP_LAT_DIM);
            }
        }

        std::vector<float> lat_norm, lat_cm;
        sgsep_decode_seg(sep, cfm, cv, cb, seed + (uint64_t) seg_idx, incontext, in_lat, lat_norm, lat_cm);

        // overlap stash for the NEXT window's incontext (always the trailing ovlp_f frames).
        if ((int) lat_norm.size() / SGSEP_LAT_DIM >= ovlp_f) {
            prev_overlap_norm.assign(lat_norm.end() - (size_t) ovlp_f * SGSEP_LAT_DIM, lat_norm.end());
        }

        std::vector<float> seg_wav;
        int                Ta = sgvae_decode(vae, lat_cm.data(), Tc, seg_wav);
        const float *      sL = seg_wav.data();
        const float *      sR = seg_wav.data() + Ta;

        int ovlp_s = ovlp_f * up;
        if (seg_idx == 0) {
            outL.assign(sL, sL + Ta);
            outR.assign(sR, sR + Ta);
        } else {
            int cur = (int) outL.size();
            int ov  = std::min({ ovlp_s, cur, Ta });
            for (int i = 0; i < ov; i++) {
                float w  = (float) i / (float) std::max(1, ovlp_s - 1);
                int   oi = cur - ov + i;
                outL[oi] = outL[oi] * (1.0f - w) + sL[i] * w;
                outR[oi] = outR[oi] * (1.0f - w) + sR[i] * w;
            }
            outL.insert(outL.end(), sL + ov, sL + Ta);
            outR.insert(outR.end(), sR + ov, sR + Ta);
        }
        seg_idx++;
    }

    // trim the prompt latent region from the FRONT (code2sound:266 latent_list[0][:,Tlat:]) then
    // clip to target_len (code2sound:284 output[:, :target_len]).
    int trim_s   = Tlat * up;
    int avail    = std::max(0, (int) outL.size() - trim_s);
    int T_audio  = std::min(avail, target_audio);
    if (T_audio < 0) T_audio = 0;
    wav_out.resize((size_t) 2 * T_audio);
    for (int t = 0; t < T_audio; t++) wav_out[t] = outL[(size_t) trim_s + t];
    for (int t = 0; t < T_audio; t++) wav_out[T_audio + t] = outR[(size_t) trim_s + t];
    fprintf(stderr, "[post] continuation done: %d segments, trimmed %d prompt samples, %d output samples (target %d)\n",
            seg_idx, trim_s, T_audio, target_audio);
    return T_audio;
}
