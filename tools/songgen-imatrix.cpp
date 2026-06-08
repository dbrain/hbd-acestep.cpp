// songgen-imatrix: audio-matched importance-matrix collector for the LeLM.
//
// Runs sglm_forward (full-prefill oracle path) over one or more calibration
// "golden" dirs while an eval callback taps every MUL_MAT's input activation,
// accumulating per-channel sum(act^2). Dumps an .imatrix consumed by quantize.cpp.
// llama-imatrix does the same thing; we keep it in-tree because LeLM is a custom arch.
//
// Calibrate on songs OTHER than the cossim test (golden-large) to avoid overfitting.
//
//   GGML_BACKEND=CUDA ./build/songgen-imatrix <gguf> <out.imatrix> <calib_dir> [<calib_dir2> ...]
#include "npy.h"
#include "sg-imatrix.h"
#include "songgen-lelm.h"

#include <cstdio>
#include <map>
#include <string>
#include <vector>

// ---- accumulator (eval-callback side) ----
static std::map<std::string, std::vector<double>> g_sum;    // per-channel sum(act^2)
static std::map<std::string, uint64_t>            g_count;  // activation vectors seen
static std::vector<float>                         g_buf;

static bool im_eval_cb(struct ggml_tensor * t, bool ask, void * /*ud*/) {
    if (ask) {
        return t->op == GGML_OP_MUL_MAT;  // only observe matmuls
    }
    if (t->op != GGML_OP_MUL_MAT) {
        return true;
    }
    struct ggml_tensor * w = t->src[0];  // weight  [n_per_row, ...]
    struct ggml_tensor * a = t->src[1];  // activation input [n_per_row, n_vec]
    if (!w || !a || w->name[0] == '\0') {
        return true;
    }
    if (a->type != GGML_TYPE_F32 || !ggml_is_contiguous(a)) {
        return true;
    }
    const int64_t nc = a->ne[0];                        // channels (== weight n_per_row)
    const int64_t ne = ggml_nelements(a);
    const int64_t nv = ne / nc;                         // number of activation vectors
    g_buf.resize((size_t) ne);
    ggml_backend_tensor_get(a, g_buf.data(), 0, (size_t) ne * sizeof(float));
    auto & s = g_sum[w->name];
    if ((int64_t) s.size() != nc) {
        s.assign((size_t) nc, 0.0);
    }
    for (int64_t v = 0; v < nv; v++) {
        const float * col = g_buf.data() + v * nc;
        for (int64_t k = 0; k < nc; k++) {
            s[(size_t) k] += (double) col[k] * (double) col[k];
        }
    }
    g_count[w->name] += (uint64_t) nv;
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <gguf> <out.imatrix> <calib_dir> [<calib_dir2> ...]\n", argv[0]);
        return 1;
    }
    const char * gguf_path = argv[1];
    const char * out_path  = argv[2];

    SonggenLeLM m;
    if (!sglm_load(&m, gguf_path)) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    const SonggenConfig & c    = m.cfg;
    const int             card = c.card;
    const int             SP   = c.special_token;

    ggml_backend_sched_set_eval_callback(m.sched, im_eval_cb, nullptr);

    std::vector<float> cb0(card), cb1(card), cb2(card);
    int                total_fwd = 0;

    for (int d = 3; d < argc; d++) {
        std::string gd = argv[d];
        auto        gp = [&](const char * f) { return gd + "/" + f; };

        NpyArray toks    = npy_load(gp("gen_tokens_lm.npy").c_str());  // [1,3,T] i64
        NpyArray d_ids_c = npy_load(gp("desc_ids_cond.npy").c_str());
        NpyArray d_cov_c = npy_load(gp("desc_cover_cond.npy").c_str());
        NpyArray d_ids_u = npy_load(gp("desc_ids_uncond.npy").c_str());
        NpyArray d_cov_u = npy_load(gp("desc_cover_uncond.npy").c_str());
        NpyArray t_ids_c = npy_load(gp("type_ids_cond.npy").c_str());
        NpyArray t_ids_u = npy_load(gp("type_ids_uncond.npy").c_str());
        int      T       = (int) toks.shape[2];
        // calibration depth: walk prefix lengths 1..N, both CFG rows. The 952-position
        // conditioning block (desc 600 + audio 252 + type 100) is present in EVERY forward,
        // so even a short prefix sweep yields ~10^5 position-samples per weight channel.
        // Cap the prefix so S (=952+n_code) stays bounded — full-length prefill would make
        // attention O(S^2) blow VRAM/runtime with no extra importance signal.
        int N      = T + 1 < 64 ? T + 1 : 64;
        int stride = 1;
        fprintf(stderr, "[imatrix] calib dir %s: T=%d, prefix sweep 1..%d x2 CFG (%d forwards)\n", gd.c_str(), T, N,
                N * 2);

        auto code     = [&](int cb, int t) -> int { return (int) toks.i64[(size_t) cb * T + t]; };
        auto make_row = [&](bool cond, int n_code) {
            SonggenCondInput in;
            in.desc_ids   = cond ? d_ids_c.i32 : d_ids_u.i32;
            in.desc_cover = cond ? d_cov_c.i32 : d_cov_u.i32;
            in.type_ids   = cond ? t_ids_c.i32 : t_ids_u.i32;
            in.audio_ids  = std::vector<int32_t>(c.audio_len - 1, SP);
            in.seq0.resize(n_code);
            in.seq1.resize(n_code);
            in.seq2.resize(n_code);
            for (int p = 0; p < n_code; p++) {
                if (p == 0) {
                    in.seq0[p] = in.seq1[p] = in.seq2[p] = SP;
                    continue;
                }
                int t      = p - 1;
                in.seq0[p] = (t < T) ? code(0, t) : SP;
                in.seq1[p] = (t - 250 >= 0 && t - 250 < T) ? code(1, t - 250) : SP;
                in.seq2[p] = (t - 250 >= 0 && t - 250 < T) ? code(2, t - 250) : SP;
            }
            return in;
        };

        for (int i = 0; i < N; i += stride) {
            int n_code = i + 1;
            for (int row = 0; row < 2; row++) {
                SonggenCondInput in = make_row(row == 0, n_code);
                sglm_forward(&m, in, cb0.data(), cb1.data(), cb2.data());
                total_fwd++;
            }
        }
    }

    // sum(act^2) -> mean per channel
    std::map<std::string, std::vector<float>> im;
    for (const auto & kv : g_sum) {
        uint64_t           cnt = g_count[kv.first];
        std::vector<float> v(kv.second.size());
        for (size_t k = 0; k < v.size(); k++) {
            v[k] = cnt ? (float) (kv.second[k] / (double) cnt) : 0.0f;
        }
        im[kv.first] = std::move(v);
    }

    if (!sgim_save(out_path, im)) {
        fprintf(stderr, "[imatrix] FATAL: failed to write %s\n", out_path);
        sglm_free(&m);
        return 1;
    }
    fprintf(stderr, "[imatrix] %zu tensors from %d forwards -> %s\n", im.size(), total_fwd, out_path);

    sglm_free(&m);
    return 0;
}
