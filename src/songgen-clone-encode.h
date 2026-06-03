// songgen-clone-encode.h : audio-prompt CLONE encode (host glue).
//
// Pre-separated stems (48k stereo) -> 3 LeLM prompt-token streams audio_qt_embs[1,3,T]:
//   cb0 (pmt)   : full mix  -> 1rvq-MusicFM(layer 6) -> 1rvq-RVQ      -> pmt codes
//   cb1 (vocal) : vocal stem -> septoken-MusicFM(layer 7) -> rvq_vocal -> vocal codes
//   cb2 (bgm)   : bgm  stem  -> septoken-MusicFM(layer 3) -> rvq_bgm   -> bgm codes
//
// Source of truth (cited):
//   generate_septoken.sound2code (115-168): 48k stem pair -> mean -> 24k -> bestrq layer7/3
//   generate_1rvq.sound2code (96-130) + fetch_codes_batch: single 48k stem -> layer6
//   model_septoken.fetch_codes_batch (542-584): extract_bestrq_embeds + rvq_bestrq_emb/_bgm
//   codeclm.py _prepare_tokens_and_attributes (229): audio_qt_embs = cat([pmt,vocal,bgm], dim=1)
//
// WINDOWING (sound2code): min_samples = 40*sample_rate (= 40s). A short prompt is repeat-tiled
//   to fill >= one 40s segment, segments are encoded independently, codes concatenated and
//   trimmed to output_len = int(orig_len/sr*25)+1. For a <=40s prompt the LEADING output_len
//   frames come from the FIRST (real-audio) segment, so they match a direct bestrq+rvq on the
//   un-tiled clip up to the tiling boundary. We reproduce the exact tiling to be bit-faithful.
//
// MusicFM mel front-end runs @ 24k internally (sgmf), but its sample_rate field is 24000 and the
// windowing min_samples uses the STEM sample rate (48000). The bestrq input is the 24k-resampled
// mean of the two stem channels (julius.resample_frac in the reference; we reuse audio-resample.h
// — a Kaiser polyphase resampler, NOT bitwise-identical to julius, so leading-frame agreement is
// "close" not exact; documented in the test).
#pragma once

#include "audio-resample.h"
#include "songgen-musicfm.h"
#include "songgen-septoken.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

struct SonggenCloneEnc {
    SonggenMusicFM mf_sep;   // septoken bestrq (vocal layer7 / bgm layer3)
    SonggenMusicFM mf_pmt;   // 1rvq bestrq (pmt layer6)
    SgRvqStream    rvq_pmt;  // model_1rvq single codebook
    SonggenSeptoken * sep;   // borrowed: provides rvq_vocal / rvq_bgm
    int            sample_rate;  // stem sample rate (48000)
    int            layer_vocal, layer_bgm, layer_pmt;
};

static bool sgclone_load(SonggenCloneEnc * m, const char * musicfm_sep_gguf, const char * musicfm_1rvq_gguf,
                         const char * aux_1rvq_gguf, SonggenSeptoken * sep) {
    *m              = {};
    m->sep          = sep;
    m->sample_rate  = 48000;
    m->layer_vocal  = 7;
    m->layer_bgm    = 3;
    m->layer_pmt    = 6;
    if (!sgmf_load(&m->mf_sep, musicfm_sep_gguf)) {
        fprintf(stderr, "[CLONE] septoken musicfm load failed\n");
        return false;
    }
    if (!sgmf_load(&m->mf_pmt, musicfm_1rvq_gguf)) {
        fprintf(stderr, "[CLONE] 1rvq musicfm load failed\n");
        return false;
    }
    if (!sgsep_load_1rvq(&m->rvq_pmt, aux_1rvq_gguf)) {
        fprintf(stderr, "[CLONE] 1rvq aux load failed\n");
        return false;
    }
    return true;
}

static void sgclone_free(SonggenCloneEnc * m) {
    sgmf_free(&m->mf_sep);
    sgmf_free(&m->mf_pmt);
    *m = {};
}

// One 48k stereo stem -> 24k mono mean. in is planar [2*n] (ch0 then ch1). Returns 24k mono vector.
static void sgclone_stem_to_24k_mono(const float * stem_planar, int n_in, int sr_in, std::vector<float> & out24k) {
    std::vector<float> mono(n_in);
    const float * ch0 = stem_planar;
    const float * ch1 = stem_planar + n_in;
    for (int i = 0; i < n_in; i++) mono[i] = 0.5f * (ch0[i] + ch1[i]);
    int      n24 = 0;
    float *  r   = audio_resample(mono.data(), n_in, sr_in, 24000, 1, &n24);
    out24k.assign(r, r + n24);
    free(r);
}

// Replicate sound2code windowing on a 24k mono signal: tile to >= one 40s-equivalent segment
// (min24 = 40*24000), produce non-overlapping segments, encode each via the MusicFM layer + RVQ,
// concat codes, trim to output_len = int(orig24_len/24000*25)+1.
//
// We tile at 24k (post-resample) for simplicity; the reference tiles the 48k stem then resamples
// each 40s segment. For a SINGLE <=40s prompt this is equivalent (one segment, leading frames =
// the real clip), which is all the goldens (and the MVP) exercise. Multi-segment >40s prompts
// would differ slightly at segment seams; flagged for future work.
static void sgclone_encode_stream(SonggenMusicFM * mf, const SgRvqStream & rvq, int layer,
                                  const std::vector<float> & audio24, std::vector<int> & codes_out) {
    const int min24 = 40 * 24000;
    int       orig  = (int) audio24.size();
    int       output_len = (int) ((double) orig / 24000.0 * 25.0) + 1;

    std::vector<float> a = audio24;
    while ((int) a.size() < min24) {
        size_t s = a.size();
        a.insert(a.end(), a.begin(), a.begin() + s);  // cat([a,a])
    }
    int int_max_len = (int) a.size() / min24 + 1;
    {
        size_t s = a.size();
        a.insert(a.end(), a.begin(), a.begin() + s);  // cat([a,a]) again
    }
    a.resize((size_t) int_max_len * min24);  // trim to int_max_len whole segments

    int n_seg = int_max_len;
    codes_out.clear();
    for (int seg = 0; seg < n_seg; seg++) {
        const float *                   sa = &a[(size_t) seg * min24];
        std::vector<std::vector<float>> layers;
        int                             T = 0;
        sgmf_encode(mf, sa, min24, layers, &T);  // layers[layer] is [H,T] t-major

        // to channel-major feats[1024, T] for sgsep_rvq_encode (expects [ic*T + t]).
        const std::vector<float> & lr = layers[layer];
        std::vector<float>         feats((size_t) SGSEP_MU_DIM * T);
        for (int t = 0; t < T; t++)
            for (int c = 0; c < SGSEP_MU_DIM; c++) feats[(size_t) c * T + t] = lr[(size_t) t * SGSEP_MU_DIM + c];

        std::vector<int>   codes;
        std::vector<float> z_e, quant;
        sgsep_rvq_encode(rvq, feats.data(), T, codes, z_e, quant);
        codes_out.insert(codes_out.end(), codes.begin(), codes.end());
    }
    if ((int) codes_out.size() > output_len) codes_out.resize(output_len);
}

// Full clone encode: stems (48k stereo planar) -> 3 code streams (each length T = output_len).
//   vocal_stem / bgm_stem : the separated stems. full_mix : the pmt source (if empty -> vocal+bgm sum).
// All buffers planar [2*n] (ch0 then ch1). Returns common T (min over streams, trimmed to align).
static int sgclone_encode(SonggenCloneEnc * m, const float * vocal_stem, int n_vocal, const float * bgm_stem,
                          int n_bgm, const float * full_mix, int n_full, int sr_in, std::vector<int> & pmt_codes,
                          std::vector<int> & vocal_codes, std::vector<int> & bgm_codes) {
    std::vector<float> voc24, bgm24, full24;
    sgclone_stem_to_24k_mono(vocal_stem, n_vocal, sr_in, voc24);
    sgclone_stem_to_24k_mono(bgm_stem, n_bgm, sr_in, bgm24);

    // sound2code(septoken) truncates vocal/bgm to the shorter of the two BEFORE windowing.
    int common = std::min((int) voc24.size(), (int) bgm24.size());
    voc24.resize(common);
    bgm24.resize(common);

    if (full_mix && n_full > 0) {
        sgclone_stem_to_24k_mono(full_mix, n_full, sr_in, full24);
    } else {
        // approximation: pmt source = vocal + bgm (pre-resample sum would be ideal; post-resample
        // mono sum is equivalent up to resampler linearity). Documented in the CLI.
        full24.resize(common);
        for (int i = 0; i < common; i++) full24[i] = voc24[i] + bgm24[i];
    }

    sgclone_encode_stream(&m->mf_pmt, m->rvq_pmt, m->layer_pmt, full24, pmt_codes);
    sgclone_encode_stream(&m->mf_sep, m->sep->vocal, m->layer_vocal, voc24, vocal_codes);
    sgclone_encode_stream(&m->mf_sep, m->sep->bgm, m->layer_bgm, bgm24, bgm_codes);

    int T = (int) std::min({ pmt_codes.size(), vocal_codes.size(), bgm_codes.size() });
    pmt_codes.resize(T);
    vocal_codes.resize(T);
    bgm_codes.resize(T);
    return T;
}

// Build one codebook's prompt_audio id stream from prompt codes, per
// lm_levo.prepare_condition_tensors (222-230) + conditioners.QuantizedEmbeddingConditioner.forward
// (276-279). audN = audio_len - 1 (251). The graph prepends the learned EOT separately, so this
// stream is exactly wav[:,k]:
//   audio_qt_seq = [eos_token_id(16384), codes_0..codes_{n-1}]   (EOS column prepended)
//   pad to audN with special(16385)   (F.pad value = vocab_size+1 = code_size+1)
//   mask: positions where the PMT codebook == 16385 are set to 16385 across ALL codebooks. With real
//   prompt codes (0..16383) nothing is masked, so we only apply padding here.
static void sgclone_build_audio_ids(const std::vector<int> & codes, int audN, int eos_token_id, int special,
                                    std::vector<int32_t> & ids) {
    ids.assign(audN, special);
    ids[0] = eos_token_id;
    int n = std::min((int) codes.size(), audN - 1);
    for (int i = 0; i < n; i++) ids[i + 1] = codes[i];
}
