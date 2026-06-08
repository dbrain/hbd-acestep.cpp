// sa3-imatrix.h: importance-matrix collection for SA3 quantization (not flat Q).
//
// Hooks ggml_backend_sched eval callback; for every MUL_MAT node it accumulates the per-input-channel
// sum of squares of the activation (src[1]), keyed by the weight tensor name (src[0]). After a
// calibration sweep, imatrix[name][k] = mean_k(x_k^2) — fed to ggml_quantize_chunk to weight quant
// error by activation magnitude. File format: [u32 n] then per tensor [u32 len, name, u32 ne0, f32[ne0]].
#pragma once
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct SA3Imatrix {
    std::map<std::string, std::vector<double>> sums;  // name -> sum x^2 per input channel
    std::map<std::string, int64_t>             cnt;   // name -> #columns summed
    std::vector<float>                         scratch;
    bool enabled = false;
};

static SA3Imatrix g_sa3_imx;  // single global collector (one process at a time)

static bool sa3_imx_cb(struct ggml_tensor * t, bool ask, void * /*ud*/) {
    if (ask) return t->op == GGML_OP_MUL_MAT;  // observe all matmuls
    if (t->op != GGML_OP_MUL_MAT) return true;
    struct ggml_tensor * w = t->src[0];
    struct ggml_tensor * a = t->src[1];
    if (!w || !a || !w->name[0]) return true;
    if (!ggml_is_contiguous(a) || a->type != GGML_TYPE_F32) return true;  // safe linear read only
    int64_t K = a->ne[0];
    int64_t N = a->ne[1] * a->ne[2] * a->ne[3];
    if (K <= 0 || N <= 0) return true;
    auto & s = g_sa3_imx.scratch;
    size_t need = (size_t)(K * N);
    if (s.size() < need) s.resize(need);
    // read activation (F32 expected; matmul src1 is F32 in our graphs)
    if (a->type != GGML_TYPE_F32) return true;
    ggml_backend_tensor_get(a, s.data(), 0, need * sizeof(float));
    auto & acc = g_sa3_imx.sums[w->name];
    if ((int64_t)acc.size() < K) acc.resize(K, 0.0);
    for (int64_t n = 0; n < N; n++) {
        const float * col = s.data() + n * K;
        for (int64_t k = 0; k < K; k++) acc[k] += (double)col[k] * col[k];
    }
    g_sa3_imx.cnt[w->name] += N;
    return true;
}

static void sa3_imx_attach(ggml_backend_sched_t sched) {
    if (g_sa3_imx.enabled) ggml_backend_sched_set_eval_callback(sched, sa3_imx_cb, nullptr);
}

static bool sa3_imx_save(const char * path) {
    FILE * f = fopen(path, "wb");
    if (!f) return false;
    uint32_t n = (uint32_t) g_sa3_imx.sums.size();
    fwrite(&n, 4, 1, f);
    for (auto & kv : g_sa3_imx.sums) {
        const std::string & name = kv.first;
        int64_t c = g_sa3_imx.cnt[name];
        uint32_t len = (uint32_t) name.size();
        uint32_t ne0 = (uint32_t) kv.second.size();
        fwrite(&len, 4, 1, f); fwrite(name.data(), 1, len, f); fwrite(&ne0, 4, 1, f);
        std::vector<float> mean(ne0);
        for (uint32_t k = 0; k < ne0; k++) mean[k] = (float)(kv.second[k] / (c > 0 ? (double)c : 1.0));
        fwrite(mean.data(), 4, ne0, f);
    }
    fclose(f);
    fprintf(stderr, "[imatrix] saved %u tensors -> %s\n", n, path);
    return true;
}

// Loader (for the quantizer): name -> per-input-channel importance vector.
static std::map<std::string, std::vector<float>> sa3_imx_load(const char * path) {
    std::map<std::string, std::vector<float>> out;
    FILE * f = fopen(path, "rb");
    if (!f) return out;
    uint32_t n = 0; if (fread(&n, 4, 1, f) != 1) { fclose(f); return out; }
    for (uint32_t i = 0; i < n; i++) {
        uint32_t len = 0, ne0 = 0;
        if (fread(&len, 4, 1, f) != 1) break;
        std::string name(len, '\0'); if (fread(&name[0], 1, len, f) != len) break;
        if (fread(&ne0, 4, 1, f) != 1) break;
        std::vector<float> v(ne0); if (fread(v.data(), 4, ne0, f) != ne0) break;
        out[name] = std::move(v);
    }
    fclose(f);
    fprintf(stderr, "[imatrix] loaded %zu tensors from %s\n", out.size(), path);
    return out;
}
