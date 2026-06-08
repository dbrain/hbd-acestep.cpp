// sa3-tokenizer.h: Gemma BPE tokenizer (SentencePiece-style) for the SA3 T5Gemma conditioner.
//
// Matches HF tokenizers "BPE" with: normalizer Replace(" ","▁"), no add_dummy_prefix,
// byte_fallback, ignore_merges, no BOS/EOS. Whole normalized string is one BPE word (Split on
// " " is a no-op after normalization). Loads vocab+merges from GGUF (tokenizer.ggml.tokens/merges).
#pragma once
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

struct SA3Tokenizer {
    std::unordered_map<std::string, int> tok2id;
    std::unordered_map<std::string, int> merge_rank;  // key = A '\0' B
    int byte_id[256];
    int pad_id = 0;
    bool ok = false;
};

static bool sa3tok_load_from_gguf(SA3Tokenizer * t, const struct gguf_context * gf) {
    int64_t tk = gguf_find_key(gf, "tokenizer.ggml.tokens");
    int64_t mk = gguf_find_key(gf, "tokenizer.ggml.merges");
    if (tk < 0 || mk < 0) { fprintf(stderr, "[sa3tok] missing tokens/merges KV\n"); return false; }
    int n_tok = (int) gguf_get_arr_n(gf, tk);
    int n_mrg = (int) gguf_get_arr_n(gf, mk);
    t->tok2id.reserve((size_t) n_tok * 2);
    for (int i = 0; i < n_tok; i++) t->tok2id[gguf_get_arr_str(gf, tk, (size_t) i)] = i;
    t->merge_rank.reserve((size_t) n_mrg * 2);
    for (int i = 0; i < n_mrg; i++) {
        std::string m = gguf_get_arr_str(gf, mk, (size_t) i);
        size_t sp = m.find(' ');
        if (sp == std::string::npos) continue;
        std::string key = m.substr(0, sp);
        key.push_back('\0');
        key += m.substr(sp + 1);
        t->merge_rank.emplace(std::move(key), i);
    }
    for (int b = 0; b < 256; b++) {
        char buf[8]; snprintf(buf, sizeof(buf), "<0x%02X>", b);
        auto it = t->tok2id.find(buf);
        t->byte_id[b] = (it != t->tok2id.end()) ? it->second : -1;
    }
    auto pit = t->tok2id.find("<pad>");
    if (pit != t->tok2id.end()) t->pad_id = pit->second;
    t->ok = true;
    fprintf(stderr, "[sa3tok] loaded %d tokens, %d merges\n", n_tok, n_mrg);
    return true;
}

static inline int sa3tok_utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

// Encode text -> token ids (no BOS/EOS, no padding).
static std::vector<int> sa3tok_encode(const SA3Tokenizer * t, const std::string & text) {
    // 1. normalize: ' ' -> U+2581 (E2 96 81)
    std::string s;
    s.reserve(text.size() * 2);
    for (char c : text) {
        if (c == ' ') s.append("\xe2\x96\x81");
        else s.push_back(c);
    }
    // ignore_merges: whole word directly in vocab?
    std::vector<int> out;
    auto whole = t->tok2id.find(s);
    if (whole != t->tok2id.end()) { out.push_back(whole->second); return out; }

    // 2. split into unicode-char symbols
    std::vector<std::string> sym;
    for (size_t i = 0; i < s.size();) {
        int l = sa3tok_utf8_len((unsigned char) s[i]);
        if (i + (size_t) l > s.size()) l = 1;
        sym.emplace_back(s.substr(i, (size_t) l));
        i += (size_t) l;
    }
    // 3. BPE: repeatedly merge the lowest-rank adjacent pair
    while (sym.size() > 1) {
        int best_rank = INT32_MAX, best_i = -1;
        std::string key;
        for (size_t i = 0; i + 1 < sym.size(); i++) {
            key = sym[i]; key.push_back('\0'); key += sym[i + 1];
            auto it = t->merge_rank.find(key);
            if (it != t->merge_rank.end() && it->second < best_rank) { best_rank = it->second; best_i = (int) i; }
        }
        if (best_i < 0) break;
        sym[best_i] += sym[best_i + 1];
        sym.erase(sym.begin() + best_i + 1);
    }
    // 4. map symbols -> ids, byte_fallback otherwise
    for (auto & p : sym) {
        auto it = t->tok2id.find(p);
        if (it != t->tok2id.end()) out.push_back(it->second);
        else for (unsigned char b : p) if (t->byte_id[b] >= 0) out.push_back(t->byte_id[b]);
    }
    return out;
}

// Encode + pad/truncate to max_len. Fills ids[max_len], valid[max_len] (1=real). Returns n real tokens.
static int sa3tok_encode_padded(const SA3Tokenizer * t, const std::string & text, int max_len,
                                std::vector<int> & ids, std::vector<int> & valid) {
    std::vector<int> e = sa3tok_encode(t, text);
    int n = (int) e.size(); if (n > max_len) n = max_len;
    ids.assign(max_len, t->pad_id);
    valid.assign(max_len, 0);
    for (int i = 0; i < n; i++) { ids[i] = e[i]; valid[i] = 1; }
    return n;
}
