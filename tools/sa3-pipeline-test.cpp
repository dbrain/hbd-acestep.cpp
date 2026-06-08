// sa3-pipeline-test: deterministic full-chain validation (euler sampler).
//   ./sa3-pipeline-test <dit.gguf> <same.gguf> <golden_dir> [out.wav]
// Uses golden noise (step0_xin) + cross_cond + global_cond, runs euler 8-step DiT loop with the
// LogSNRShift(rate=0) schedule, compares final latent to euler_latent.npy, then SAME-decodes to wav.
#include "npy.h"
#include "sa3-dit.h"
#include "sa3-same.h"
#include "wav.h"

#include <cmath>
#include <cstdio>
#include <vector>

static double cossim(const float * a, const float * b, size_t n) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) { dot += (double) a[i]*b[i]; na += (double) a[i]*a[i]; nb += (double) b[i]*b[i]; }
    return (na == 0 || nb == 0) ? 0.0 : dot / (sqrt(na) * sqrt(nb));
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <dit.gguf> <same.gguf> <golden_dir> [out.wav]\n", argv[0]); return 1; }
    std::string dit_g = argv[1], same_g = argv[2], gd = argv[3];
    std::string out_wav = argc > 4 ? argv[4] : "sa3_out.wav";
    auto gp = [&](const char * f) { return gd + "/" + f; };
    const int STEPS = 8;

    NpyArray noise = npy_load(gp("step0_xin.npy").c_str());  // [1,256,L] channel-major
    NpyArray cc    = npy_load(gp("cross_cond.npy").c_str()); // [1,257,768]
    NpyArray gcv   = npy_load(gp("global_cond.npy").c_str());// [1,768]
    NpyArray eul   = npy_load(gp("euler_latent.npy").c_str());// [1,256,L] channel-major
    int C = (int) noise.shape[noise.shape.size()-2];
    int L = (int) noise.shape[noise.shape.size()-1];
    int cross_T = (int) cc.shape[cc.shape.size()-2];
    fprintf(stderr, "[pipeline] C=%d L=%d cross_T=%d steps=%d\n", C, L, cross_T, STEPS);

    // schedule: LogSNRShift(rate=0, anchor_logsnr=-6.2, logsnr_end=2.0): sig=sigmoid(t_lin*8.2-2), ends forced
    std::vector<float> sig(STEPS + 1);
    for (int i = 0; i <= STEPS; i++) {
        float t_lin = 1.0f - (float) i / (float) STEPS;
        if (i == 0)      sig[i] = 1.0f;
        else if (i == STEPS) sig[i] = 0.0f;
        else             sig[i] = 1.0f / (1.0f + expf(-(t_lin * 8.2f - 2.0f)));
    }
    fprintf(stderr, "[pipeline] sigmas:"); for (int i=0;i<=STEPS;i++) fprintf(stderr," %.4f", sig[i]); fprintf(stderr, "\n");

    // noise [C,L] channel-major -> token-major
    std::vector<float> x((size_t) C * L);
    for (int c=0;c<C;c++) for (int l=0;l<L;l++) x[(size_t)l*C+c] = noise.f32[(size_t)c*L+l];

    SA3DiT dit;  if (!sa3dit_load(&dit, dit_g.c_str())) return 1;

    std::vector<float> v((size_t) C * L);
    for (int i = 0; i < STEPS; i++) {
        sa3dit_forward(&dit, x.data(), L, sig[i], cc.f32.data(), cross_T, gcv.f32.data(), v.data());
        float dt = sig[i+1] - sig[i];
        for (size_t k = 0; k < x.size(); k++) x[k] += dt * v[k];   // euler
        fprintf(stderr, "[pipeline] step %d t=%.4f done\n", i, sig[i]);
    }
    sa3dit_free(&dit);

    // compare final latent (token-major l*C+c) to golden euler_latent (channel-major c*L+l)
    std::vector<float> ours((size_t)C*L), gold((size_t)C*L);
    for (int c=0;c<C;c++) for (int l=0;l<L;l++) { ours[(size_t)c*L+l]=x[(size_t)l*C+c]; gold[(size_t)c*L+l]=eul.f32[(size_t)c*L+l]; }
    double cs = cossim(ours.data(), gold.data(), (size_t)C*L);
    double se=0,gn=0; for (size_t i=0;i<(size_t)C*L;i++){double d=ours[i]-gold[i]; se+=d*d; gn+=(double)gold[i]*gold[i];}
    printf("---- euler final latent vs golden ----\n");
    printf("cossim : %.6f   rmse %.4e (golden rms %.4e)\n", cs, sqrt(se/((size_t)C*L)), sqrt(gn/((size_t)C*L)));
    bool latent_ok = cs > 0.998;  // 8-step euler free-running accumulation of ~0.9997 per-step DiT error
    printf("latent RESULT: %s\n", latent_ok ? "PASS" : "FAIL");

    // SAME decode our latent (token-major already) -> wav
    SA3Same same; if (!sa3same_load(&same, same_g.c_str())) return 1;
    std::vector<float> audio;
    int N = sa3same_decode(&same, x.data(), L, audio);   // [2,N] planar
    sa3same_free(&same);
    // truncate to 10s (441000) like the reference
    int sr = 44100, Nt = N < 10*sr ? N : 10*sr;
    std::vector<float> planar(2*(size_t)Nt);
    for (int c=0;c<2;c++) for (int s=0;s<Nt;s++) planar[(size_t)c*Nt+s]=audio[(size_t)c*N+s];
    if (write_wav_s16_planar(out_wav.c_str(), planar.data(), Nt, sr)) printf("wrote %s (%d samples, %.2fs)\n", out_wav.c_str(), Nt, (float)Nt/sr);

    return latent_ok ? 0 : 2;
}
