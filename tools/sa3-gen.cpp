// sa3-gen: Stable Audio 3 audio CLI (the full ported pipeline). Supports four modes:
//   t2a  (default)         : --prompt ... [--seconds 10]
//   a2a  (SDEdit edit)     : --input in.wav --strength 0.7   (encode->renoise to sigma=strength->resample w/ new prompt)
//   inpaint                : --input in.wav --mask "s0:e0,s1:e1"  (seconds; masked regions are regenerated)
//   continuation           : --input in.wav --continue --seconds <total>  (keep input as prefix, extend)
//
//   sa3-gen --t5 t5gemma.gguf --dit dit.gguf --same same.gguf --prompt "..." [--seconds 10]
//           [--steps 8] [--seed 0] [--sampler euler|pingpong] [--out out.wav]
//           [--input in.wav] [--strength 0..1] [--mask "a:b,c:d"] [--continue]
//           [--noise step0_xin.npy] [--cmp <golden_dir>]   (last two: validation)
//
// tokenize -> T5Gemma encode -> assemble cross/global cond (seconds NumberConditioner + learned
// padding) -> sampler (euler/pingpong, LogSNRShift rate=0 schedule) over DiT -> SAME decode -> wav.
// a2a/inpaint/continuation gate on the SAME *encoder* (sa3-same-enc.h). Inpaint is conditioning-based
// (the DiT to_local_embed path: [mask(1);masked_latent(256)]=257), NOT sampler x-pinning -- matches the
// reference, whose euler/pingpong samplers do no masking.
#include "npy.h"
#include "gguf-weights.h"
#include "sa3-tokenizer.h"
#include "sa3-t5gemma-enc.h"
#include "sa3-dit.h"
#include "sa3-same.h"
#include "sa3-same-enc.h"
#include "wav.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// monotonic wall-clock seconds, for the [sa3-timing] phase report (parsed by the ear-test page)
static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static const char * argval(int argc, char ** argv, const char * key, const char * def) {
    for (int i = 1; i < argc - 1; i++) if (!strcmp(argv[i], key)) return argv[i + 1];
    return def;
}
static bool argflag(int argc, char ** argv, const char * key) {
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], key)) return true;
    return false;
}
static double cossim(const float * a, const float * b, size_t n) {
    double d=0,na=0,nb=0; for (size_t i=0;i<n;i++){d+=(double)a[i]*b[i];na+=(double)a[i]*a[i];nb+=(double)b[i]*b[i];}
    return (na==0||nb==0)?0:d/(sqrt(na)*sqrt(nb));
}

// Read a 44.1kHz stereo wav into planar audio[2,Nsamp] (c*Nsamp+s), truncated to a multiple of 4096.
// Returns Nsamp (0 on failure).
static int read_wav_planar(const char * path, std::vector<float> & planar) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[sa3-gen] cannot open %s\n", path); return 0; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf((size_t) sz);
    if (fread(buf.data(), 1, (size_t) sz, f) != (size_t) sz) { fclose(f); return 0; }
    fclose(f);
    int Ti = 0, sr = 0;
    float * inter = read_wav_buf(buf.data(), buf.size(), &Ti, &sr);  // interleaved [t*2+c]
    if (!inter) return 0;
    if (sr != 44100) fprintf(stderr, "[sa3-gen] WARNING: input sr=%d != 44100 (no resampler; results will be off-pitch)\n", sr);
    int Nsamp = (Ti / 4096) * 4096;
    planar.assign((size_t) 2 * Nsamp, 0.0f);
    for (int t = 0; t < Nsamp; t++) { planar[t] = inter[t*2+0]; planar[(size_t)Nsamp + t] = inter[t*2+1]; }
    free(inter);
    return Nsamp;
}

int main(int argc, char ** argv) {
    const char * t5p   = argval(argc, argv, "--t5", "models/sa3-t5gemma-f32.gguf");
    const char * ditp  = argval(argc, argv, "--dit", "models/sa3-dit-f32.gguf");
    const char * samep = argval(argc, argv, "--same", "models/sa3-same-f32.gguf");
    const char * prompt= argval(argc, argv, "--prompt", "8-bit retro arcade power-up jingle, chiptune, square wave");
    float seconds      = atof(argval(argc, argv, "--seconds", "10"));
    int   steps        = atoi(argval(argc, argv, "--steps", "8"));
    int   seed         = atoi(argval(argc, argv, "--seed", "0"));
    const char * samp  = argval(argc, argv, "--sampler", "euler");
    const char * outw  = argval(argc, argv, "--out", "sa3_gen.wav");
    const char * noisep= argval(argc, argv, "--noise", "");
    const char * cmp   = argval(argc, argv, "--cmp", "");
    const char * inputp= argval(argc, argv, "--input", "");
    float strength     = atof(argval(argc, argv, "--strength", "0.7"));
    const char * maskspec = argval(argc, argv, "--mask", "");
    bool  do_continue  = argflag(argc, argv, "--continue");
    const int CD = 768, C = 256, MAXTOK = 256, SR = 44100, DS = 4096;

    enum Mode { T2A, A2A, INPAINT, CONTINUE };
    Mode mode = T2A;
    if (inputp[0]) {
        if (do_continue)      mode = CONTINUE;
        else if (maskspec[0]) mode = INPAINT;
        else                  mode = A2A;
    }
    const char * mode_name[] = {"t2a","a2a","inpaint","continuation"};

    // ---- encode the input audio (a2a / inpaint / continuation) gated on the SAME encoder ----
    std::vector<float> enc_lat;   // [C, T_in] token-major (t*C+c)
    int T_in = 0;
    if (mode != T2A) {
        std::vector<float> audio;
        int Nsamp = read_wav_planar(inputp, audio);
        if (Nsamp == 0) { fprintf(stderr, "[sa3-gen] failed to read --input %s\n", inputp); return 1; }
        double in_dur = (double) Nsamp / SR;
        fprintf(stderr, "[sa3-gen] %s: input %.2fs (%d samples)\n", mode_name[mode], in_dur, Nsamp);
        double t_enc_0 = now_s();
        SA3Enc enc; if (!sa3enc_load(&enc, samep)) return 1;
        T_in = sa3enc_encode_chunked(&enc, audio.data(), Nsamp, enc_lat);
        sa3enc_free(&enc);
        fprintf(stderr, "[sa3-gen] encoded input -> latent T_in=%d (%.3fs)\n", T_in, now_s()-t_enc_0);
        // a2a/inpaint canvas length == input length; continuation extends to --seconds.
        if (mode != CONTINUE) seconds = (float) in_dur;
    }

    // ---- conditioner tensors from same gguf (F32) ----
    GGUFModel cg;
    if (!gf_load(&cg, samep)) return 1;
    const float * sec_w = (const float *) gf_get_data(cg, "sec.emb.w");   // [256,768] ggml (o*256+i)
    const float * sec_b = (const float *) gf_get_data(cg, "sec.emb.b");   // [768]
    const float * pad_e = (const float *) gf_get_data(cg, "prompt.pad_embed"); // [768]
    if (!sec_w || !sec_b || !pad_e) { fprintf(stderr, "conditioner tensors not F32/found in %s\n", samep); return 1; }
    std::vector<float> secW(sec_w, sec_w + (size_t)256*768), secB(sec_b, sec_b+768), padE(pad_e, pad_e+768);
    gf_close(&cg);

    // seconds NumberConditioner: clamp(s,0,384)/384 -> ExpoFourier(256,0.5,10000) -> Linear(256->768)
    std::vector<float> sec(CD);
    {
        float norm = (seconds < 0 ? 0 : seconds > 384 ? 384 : seconds) / 384.0f;
        int half = 128; float lo = logf(0.5f), hi = logf(10000.0f);
        std::vector<float> ef(256);
        for (int j = 0; j < half; j++) {
            float f = expf((float)j/(half-1)*(hi-lo)+lo);
            float a = norm * f * 2.0f * (float)M_PI;
            ef[j] = cosf(a); ef[half+j] = sinf(a);
        }
        for (int o = 0; o < CD; o++) { double acc = secB[o]; for (int i = 0; i < 256; i++) acc += (double)secW[(size_t)o*256+i]*ef[i]; sec[o]=(float)acc; }
    }

    // ---- tokenize + T5Gemma encode ----
    struct ggml_context * meta = nullptr; struct gguf_init_params gp = { true, &meta };
    struct gguf_context * tg = gguf_init_from_file(t5p, gp);
    SA3Tokenizer tok; if (!tg || !sa3tok_load_from_gguf(&tok, tg)) return 1;
    std::vector<int> ids, valid; int ntok = sa3tok_encode_padded(&tok, prompt, MAXTOK, ids, valid);
    gguf_free(tg); ggml_free(meta);
    fprintf(stderr, "[sa3-gen] prompt -> %d tokens\n", ntok);

    double t_t5_0 = now_s();
    SA3T5GModel t5; if (!sa3t5g_load(&t5, t5p)) return 1;
    double t_t5_load = now_s() - t_t5_0;
    std::vector<float> t5out((size_t)CD*MAXTOK);
    double t_t5_enc_0 = now_s();
    sa3t5g_encode(&t5, ids.data(), valid.data(), MAXTOK, t5out.data());  // [768,256] (tok*768+d)
    double t_t5_enc = now_s() - t_t5_enc_0;
    sa3t5g_free(&t5);
    double t_t5 = now_s() - t_t5_0;

    // ---- assemble cross_cond[768,257] + global[768] ----
    int cross_T = MAXTOK + 1;
    std::vector<float> cross((size_t)CD*cross_T), glob = sec;
    for (int t = 0; t < MAXTOK; t++)
        for (int d = 0; d < CD; d++)
            cross[(size_t)t*CD+d] = valid[t] ? t5out[(size_t)t*CD+d] : padE[d];
    for (int d = 0; d < CD; d++) cross[(size_t)MAXTOK*CD+d] = sec[d];

    if (cmp[0]) {
        NpyArray gcc = npy_load((std::string(cmp)+"/cross_cond.npy").c_str());
        NpyArray ggl = npy_load((std::string(cmp)+"/global_cond.npy").c_str());
        printf("[cmp] cross_cond cossim %.6f   global cossim %.6f\n",
               cossim(cross.data(), gcc.f32.data(), cross.size()), cossim(glob.data(), ggl.f32.data(), CD));
    }

    // ---- latent length L ----
    int L;
    if (mode == A2A || mode == INPAINT) {
        L = T_in;  // canvas == input length
    } else {  // T2A or CONTINUE: derive from duration (reference _adapt_sample_size)
        int align = 8192;
        long target = (long)ceilf((seconds + 6.0f) * 44100.0f);
        target = ((target + align - 1) / align) * align;
        L = (int)(target / DS);
        if (mode == CONTINUE && T_in >= L) { fprintf(stderr, "[sa3-gen] --continue: input (T_in=%d) >= canvas L=%d; raise --seconds\n", T_in, L); return 1; }
    }
    fprintf(stderr, "[sa3-gen] mode=%s latent L=%d (%.1fs)\n", mode_name[mode], L, seconds);

    // ---- inpaint/continuation: build mask + local cond [257, L] token-major ----
    // local_cond[l*257 + 0] = mask[l] (1=keep, 0=regenerate); [l*257 + 1 + c] = known_latent[c][l]*mask[l].
    std::vector<float> local_cond;
    if (mode == INPAINT || mode == CONTINUE) {
        std::vector<float> mask((size_t) L, 1.0f);
        if (mode == CONTINUE) {
            for (int l = T_in; l < L; l++) mask[l] = 0.0f;   // keep prefix, regenerate the tail
        } else {  // INPAINT: parse "s0:e0,s1:e1" seconds -> regenerate (mask=0) those frames
            std::string spec(maskspec); size_t p = 0;
            while (p < spec.size()) {
                size_t comma = spec.find(',', p);
                std::string seg = spec.substr(p, comma == std::string::npos ? std::string::npos : comma - p);
                size_t colon = seg.find(':');
                if (colon != std::string::npos) {
                    double s0 = atof(seg.substr(0, colon).c_str()), e0 = atof(seg.substr(colon + 1).c_str());
                    int f0 = (int) lroundf((float)(s0 * SR / DS)), f1 = (int) lroundf((float)(e0 * SR / DS));
                    if (f0 < 0) f0 = 0;
                    if (f1 > L) f1 = L;
                    for (int l = f0; l < f1; l++) mask[l] = 0.0f;
                    fprintf(stderr, "[sa3-gen] inpaint region %.2f-%.2fs -> frames [%d,%d)\n", s0, e0, f0, f1);
                }
                if (comma == std::string::npos) break;
                p = comma + 1;
            }
        }
        local_cond.assign((size_t) 257 * L, 0.0f);
        for (int l = 0; l < L; l++) {
            local_cond[(size_t) l * 257 + 0] = mask[l];
            for (int c = 0; c < C; c++) {
                float known = (l < T_in) ? enc_lat[(size_t) l * C + c] : 0.0f;  // token-major t*C+c
                local_cond[(size_t) l * 257 + 1 + c] = known * mask[l];
            }
        }
    }

    // ---- initial latent (token-major [C,L]) + SDEdit init mix ----
    float sigma_max = (mode == A2A) ? strength : 1.0f;
    std::vector<float> x((size_t)C*L);
    if (noisep[0]) {
        NpyArray nz = npy_load(noisep);  // [1,256,L] channel-major
        for (int c=0;c<C;c++) for (int l=0;l<L;l++) x[(size_t)l*C+c] = nz.f32[(size_t)c*L+l];
        fprintf(stderr, "[sa3-gen] loaded noise from %s\n", noisep);
    } else {
        std::mt19937 rng((unsigned)seed); std::normal_distribution<float> nd(0.f,1.f);
        for (auto & v : x) v = nd(rng);
    }
    if (mode == A2A) {
        // x = init_data*(1-sigma_max) + noise*sigma_max  (reference sample_diffusion init mix)
        for (int l = 0; l < L; l++) for (int c = 0; c < C; c++) {
            size_t k = (size_t) l * C + c;
            x[k] = enc_lat[k] * (1.0f - sigma_max) + x[k] * sigma_max;
        }
        fprintf(stderr, "[sa3-gen] a2a init mix at sigma_max=%.3f\n", sigma_max);
    }

    // ---- schedule (LogSNRShift rate=0): sig=sigmoid(t_lin*8.2-2) over linspace(sigma_max,0); sig[0]=sigma_max ----
    std::vector<float> sig(steps+1);
    for (int i=0;i<=steps;i++){
        float tl = sigma_max * (1.f - (float)i/steps);
        sig[i] = (i==steps) ? 0.f : (tl >= 1.f ? 1.f : 1.f/(1.f+expf(-(tl*8.2f-2.f))));
    }
    sig[0] = sigma_max;  // reference build_schedule forces t[0] = sigma_max

    // ---- sampler loop ----
    double t_dit_load_0 = now_s();
    SA3DiT dit; if (!sa3dit_load(&dit, ditp)) return 1;
    double t_dit_load = now_s() - t_dit_load_0;
    std::vector<float> v((size_t)C*L);
    std::mt19937 prng((unsigned)seed+1); std::normal_distribution<float> pnd(0.f,1.f);
    bool pingpong = !strcmp(samp, "pingpong");
    const float * lcp = local_cond.empty() ? nullptr : local_cond.data();
    double t_dit_steps = 0;
    for (int i=0;i<steps;i++){
        double s0 = now_s();
        sa3dit_forward(&dit, x.data(), L, sig[i], cross.data(), cross_T, glob.data(), v.data(), lcp);
        double sd = now_s() - s0; t_dit_steps += sd;
        if (!pingpong) { float dt=sig[i+1]-sig[i]; for (size_t k=0;k<x.size();k++) x[k]+=dt*v[k]; }
        else { float tn=sig[i+1]; for (size_t k=0;k<x.size();k++){ float den=x[k]-sig[i]*v[k]; x[k]=(1.f-tn)*den + tn*pnd(prng); } }
        fprintf(stderr, "[sa3-gen] step %d/%d (t=%.4f) %.3fs\n", i+1, steps, sig[i], sd);
    }
    sa3dit_free(&dit);

    if (cmp[0]) {
        NpyArray el = npy_load((std::string(cmp)+"/euler_latent.npy").c_str());
        std::vector<float> ours((size_t)C*L); for (int c=0;c<C;c++) for (int l=0;l<L;l++) ours[(size_t)c*L+l]=x[(size_t)l*C+c];
        printf("[cmp] final latent cossim %.6f (vs euler golden; only meaningful for --sampler euler)\n",
               cossim(ours.data(), el.f32.data(), (size_t)C*L));
    }

    // ---- decode + wav ----
    double t_same_0 = now_s();
    SA3Same same; if (!sa3same_load(&same, samep)) return 1;
    double t_same_load = now_s() - t_same_0;
    double t_same_dec_0 = now_s();
    std::vector<float> audio; int N = sa3same_decode_chunked(&same, x.data(), L, audio);
    double t_same_dec = now_s() - t_same_dec_0;
    sa3same_free(&same);
    double t_same = now_s() - t_same_0;
    int sr=44100, Nt = (int)(seconds*sr); if (Nt>N) Nt=N;
    std::vector<float> planar(2*(size_t)Nt);
    for (int c=0;c<2;c++) for (int s=0;s<Nt;s++) planar[(size_t)c*Nt+s]=audio[(size_t)c*N + s];
    if (write_wav_s16_planar(outw, planar.data(), Nt, sr)) printf("[sa3-gen] wrote %s (%.2fs)\n", outw, (float)Nt/sr);

    // ---- timing report (parsed by the ear-test page) ----
    double audio_s = (double)Nt / sr;
    double total = t_t5 + t_dit_load + t_dit_steps + t_same;
    double load_total = t_t5_load + t_dit_load + t_same_load;
    double compute_total = t_t5_enc + t_dit_steps + t_same_dec;
    fprintf(stderr, "[sa3-timing] t5gemma %.3fs (load %.3fs + enc %.3fs)\n", t_t5, t_t5_load, t_t5_enc);
    fprintf(stderr, "[sa3-timing] dit_load %.3fs\n", t_dit_load);
    fprintf(stderr, "[sa3-timing] dit_steps %.3fs (%d steps, %.3fs/step)\n", t_dit_steps, steps, steps ? t_dit_steps/steps : 0.0);
    fprintf(stderr, "[sa3-timing] same_decode %.3fs (load %.3fs + dec %.3fs)\n", t_same, t_same_load, t_same_dec);
    fprintf(stderr, "[sa3-timing] loads %.3fs  compute %.3fs\n", load_total, compute_total);
    fprintf(stderr, "[sa3-timing] total %.3fs  audio %.2fs  RTF %.3f\n", total, audio_s, audio_s > 0 ? total/audio_s : 0.0);
    return 0;
}
