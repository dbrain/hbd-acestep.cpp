// songgen-htdemucs-dump: dump every htd_forward + encoder intermediate as raw float32.
//
//   GGML_BACKEND=CPU  ./songgen-htdemucs-dump <gguf> <input_seg.npy> <outdir>
//   GGML_BACKEND=CUDA0 ./songgen-htdemucs-dump <gguf> <input_seg.npy> <outdir>
// Then diff <outdir_cpu> vs <outdir_cuda> by max-abs error to localize the first
// diverging stage of the CUDA forward (cosine gates hide scale/garbage; use abs).
#include "npy.h"
#include "songgen-htdemucs.h"

#include <cstdio>
#include <string>
#include <vector>

static void dump(const std::string & dir, const char * name, const std::vector<float> & v) {
    std::string path = dir + "/" + name + ".f32";
    FILE *      f    = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); return; }
    fwrite(v.data(), sizeof(float), v.size(), f);
    fclose(f);
    fprintf(stderr, "  dumped %-16s n=%zu\n", name, v.size());
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <gguf> <input_seg.npy> <outdir>\n", argv[0]); return 1; }
    std::string gguf = argv[1], inpath = argv[2], outdir = argv[3];

    SonggenHTDemucs m;
    if (!htd_load(&m, gguf.c_str())) { fprintf(stderr, "load failed\n"); return 1; }

    NpyArray in = npy_load(inpath.c_str());  // [1,2,N]
    int      AC = (int) in.shape[1];
    int      N  = (int) in.shape[2];
    fprintf(stderr, "[dump] input AC=%d N=%d backend=%s\n", AC, N, ggml_backend_name(m.backend));

    // ---- encoders ----
    HtdEncOut eo;
    htd_encode_encoders(&m, in.f32.data(), N, AC, &eo);
    dump(outdir, "enc0_fe", eo.spec_fe0);
    for (int i = 0; i < 4; i++) { char nm[32]; snprintf(nm, sizeof nm, "enc%d", i); dump(outdir, nm, eo.spec[i]); }
    for (int i = 0; i < 4; i++) { char nm[32]; snprintf(nm, sizeof nm, "tenc%d", i); dump(outdir, nm, eo.time[i]); }

    // ---- full forward (transformer + decoders + stems) ----
    HtdForwardOut fo;
    htd_forward(&m, in.f32.data(), N, AC, &fo);
    for (int i = 0; i < 5; i++) {
        char nm[32];
        snprintf(nm, sizeof nm, "ct_x_l%d", i);  dump(outdir, nm, fo.ct_x_layer[i]);
        snprintf(nm, sizeof nm, "ct_xt_l%d", i); dump(outdir, nm, fo.ct_xt_layer[i]);
    }
    for (int i = 0; i < 4; i++) {
        char nm[32];
        snprintf(nm, sizeof nm, "dec%d", i);  dump(outdir, nm, fo.dec[i]);
        snprintf(nm, sizeof nm, "tdec%d", i); dump(outdir, nm, fo.tdec[i]);
    }
    dump(outdir, "stems", fo.stems);

    htd_free(&m);
    return 0;
}
