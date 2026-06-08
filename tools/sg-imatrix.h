// sg-imatrix.h : tiny importance-matrix file format shared by the songgen imatrix
// collector (songgen-imatrix.cpp) and the requantizer (quantize.cpp).
//
// Stores, per quantizable weight tensor, the mean input-activation energy
// (mean over calibration positions of act^2) for each input channel — i.e. the
// per-column importance ggml_quantize_chunk() consumes. Keyed by GGUF tensor name.
//
// Layout: "SGIM" | u32 version | u32 n_tensors |
//         repeat{ u32 namelen, char[namelen], u32 n_chan, float[n_chan] }
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static inline bool sgim_save(const char * path, const std::map<std::string, std::vector<float>> & im) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        return false;
    }
    uint32_t ver = 1, n = (uint32_t) im.size();
    fwrite("SGIM", 1, 4, f);
    fwrite(&ver, 4, 1, f);
    fwrite(&n, 4, 1, f);
    for (const auto & kv : im) {
        uint32_t nl = (uint32_t) kv.first.size();
        uint32_t cn = (uint32_t) kv.second.size();
        fwrite(&nl, 4, 1, f);
        fwrite(kv.first.data(), 1, nl, f);
        fwrite(&cn, 4, 1, f);
        fwrite(kv.second.data(), sizeof(float), cn, f);
    }
    fclose(f);
    return true;
}

static inline bool sgim_load(const char * path, std::map<std::string, std::vector<float>> & im) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        return false;
    }
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "SGIM", 4) != 0) {
        fclose(f);
        return false;
    }
    uint32_t ver = 0, n = 0;
    if (fread(&ver, 4, 1, f) != 1 || fread(&n, 4, 1, f) != 1) {
        fclose(f);
        return false;
    }
    for (uint32_t i = 0; i < n; i++) {
        uint32_t nl = 0, cn = 0;
        if (fread(&nl, 4, 1, f) != 1) {
            break;
        }
        std::string name(nl, '\0');
        if (fread(&name[0], 1, nl, f) != nl || fread(&cn, 4, 1, f) != 1) {
            break;
        }
        std::vector<float> v(cn);
        if (fread(v.data(), sizeof(float), cn, f) != cn) {
            break;
        }
        im[name] = std::move(v);
    }
    fclose(f);
    return true;
}
