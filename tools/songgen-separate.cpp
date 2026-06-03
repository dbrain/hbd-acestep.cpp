// songgen-separate: HTDemucs source separation. mix.wav -> vocal_48k.wav + bgm_48k.wav.
//
//   GGML_BACKEND=CPU ./songgen-separate <gguf> <mix.wav> <out_vocal.wav> <out_bgm.wav>
//
// Runs the HTDemucs separator on the (resampled-to-44.1k) stereo mix via overlap-add of
// 7.8s segments (overlap 0.25, triangular weighting, shifts=0 for determinism — demucs default
// shifts=1 adds a random time-shift averaging worth ~0.2 SDR), maps stems -> vocal=vocals,
// bgm = drums+bass+other, then resamples 44.1k -> 48k and writes 16-bit WAVs.
#include "songgen-htdemucs.h"
#include "wav.h"
#include "audio-resample.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// Kaiser polyphase resample [C, N_in]@sr_in -> [C, N_out]@sr_out (planar in/out).
static std::vector<float> resample_kaiser(const std::vector<float> & in, int C, int N_in, int sr_in, int sr_out,
                                          int * N_out) {
    float * o = audio_resample(in.data(), N_in, sr_in, sr_out, C, N_out);
    std::vector<float> out(o, o + (size_t) C * (*N_out));
    free(o);
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <gguf> <mix.wav> <out_vocal.wav> <out_bgm.wav>\n", argv[0]);
        return 1;
    }
    std::string gguf = argv[1], mixp = argv[2], vocp = argv[3], bgmp = argv[4];

    SonggenHTDemucs m;
    if (!htd_load(&m, gguf.c_str())) { fprintf(stderr, "load failed\n"); return 1; }

    // read wav -> planar [L:N][R:N]
    FILE * f = fopen(mixp.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", mixp.c_str()); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf((size_t) sz); fread(buf.data(), 1, sz, f); fclose(f);
    int T_in = 0, sr_in = 0;
    float * planar = read_wav_buf(buf.data(), buf.size(), &T_in, &sr_in);
    if (!planar) { fprintf(stderr, "wav parse failed\n"); return 1; }

    // read_wav_buf returns INTERLEAVED [L,R,L,R,...]; deinterleave to planar [L:T][R:T].
    int AC = 2;
    int N44 = 0;
    std::vector<float> mix_in((size_t) AC * T_in);
    for (int t = 0; t < T_in; t++) { mix_in[t] = planar[t * 2 + 0]; mix_in[(size_t) T_in + t] = planar[t * 2 + 1]; }
    free(planar);
    std::vector<float> mix44;
    if (sr_in != m.sample_rate) {
        mix44 = resample_kaiser(mix_in, AC, T_in, sr_in, m.sample_rate, &N44);
    } else { mix44 = mix_in; N44 = T_in; }
    fprintf(stderr, "[separate] input %d sr=%d -> 44.1k N=%d\n", T_in, sr_in, N44);
    if (const char * dp = getenv("HTD_DUMP_MIX44")) {
        FILE * df = fopen(dp, "wb");
        if (df) { fwrite(mix44.data(), sizeof(float), mix44.size(), df); fclose(df); }
        fprintf(stderr, "[separate] dumped mix44 [%d,%d]\n", AC, N44);
    }

    int seg = m.segment_samples;             // 343980
    double overlap = 0.25;
    int stride = (int) (seg * (1.0 - overlap));
    // output accumulators per stem [4, AC, N44]
    int S = 4;
    std::vector<double> acc((size_t) S * AC * N44, 0.0);
    std::vector<double> wsum((size_t) N44, 0.0);

    // triangular weight over a segment
    std::vector<float> tri(seg);
    for (int i = 0; i < seg; i++) {
        float w = (float) std::min(i + 1, seg - i);
        tri[i] = w;
    }

    int n_segs = 0;
    for (int off = 0; off < N44; off += stride) {
        int len = std::min(seg, N44 - off);
        // build padded segment [AC, seg]
        std::vector<float> sgm((size_t) AC * seg, 0.f);
        for (int c = 0; c < AC; c++)
            for (int i = 0; i < len; i++) sgm[(size_t) c * seg + i] = mix44[(size_t) c * N44 + off + i];
        HtdForwardOut fo;
        htd_forward(&m, sgm.data(), seg, AC, &fo);
        for (int s = 0; s < S; s++)
            for (int c = 0; c < AC; c++)
                for (int i = 0; i < len; i++)
                    acc[((size_t) s * AC + c) * N44 + (off + i)] +=
                        (double) fo.stems[((size_t) s * AC + c) * seg + i] * tri[i];
        for (int i = 0; i < len; i++) wsum[off + i] += tri[i];
        n_segs++;
        if (off + seg >= N44) break;
    }
    for (int i = 0; i < N44; i++) if (wsum[i] < 1e-8) wsum[i] = 1.0;
    fprintf(stderr, "[separate] %d segments, stride=%d\n", n_segs, stride);

    // stems normalized
    std::vector<float> vocal((size_t) AC * N44), bgm((size_t) AC * N44);
    for (int c = 0; c < AC; c++)
        for (int i = 0; i < N44; i++) {
            double w = wsum[i];
            double drums = acc[((size_t) 0 * AC + c) * N44 + i] / w;
            double bass  = acc[((size_t) 1 * AC + c) * N44 + i] / w;
            double other = acc[((size_t) 2 * AC + c) * N44 + i] / w;
            double voc   = acc[((size_t) 3 * AC + c) * N44 + i] / w;
            vocal[(size_t) c * N44 + i] = (float) voc;
            bgm[(size_t) c * N44 + i]   = (float) (drums + bass + other);
        }

    // optional: dump 44.1k stems [4,2,N44] for golden validation
    if (const char * dp = getenv("HTD_DUMP_STEMS")) {
        std::vector<float> st((size_t) S * AC * N44);
        for (int s = 0; s < S; s++) for (int c = 0; c < AC; c++) for (int i = 0; i < N44; i++)
            st[((size_t) s * AC + c) * N44 + i] = (float) (acc[((size_t) s * AC + c) * N44 + i] / wsum[i]);
        FILE * df = fopen(dp, "wb");
        if (df) { fwrite(st.data(), sizeof(float), st.size(), df); fclose(df);
                  fprintf(stderr, "[separate] dumped stems [4,%d,%d] -> %s\n", AC, N44, dp); }
    }

    // resample 44.1k -> 48k
    int N48 = 0;
    std::vector<float> voc48 = resample_kaiser(vocal, AC, N44, m.sample_rate, 48000, &N48);
    std::vector<float> bgm48 = resample_kaiser(bgm, AC, N44, m.sample_rate, 48000, &N48);

    write_wav_s16_planar(vocp.c_str(), voc48.data(), N48, 48000, 0.0f);
    write_wav_s16_planar(bgmp.c_str(), bgm48.data(), N48, 48000, 0.0f);
    fprintf(stderr, "[separate] wrote %s + %s (48k, %d samples)\n", vocp.c_str(), bgmp.c_str(), N48);

    htd_free(&m);
    return 0;
}
