// songgen-tokenizer.h — Qwen2 byte-level BPE conditioning tokenizer for SongGeneration LeLM.
//
// Replicates scripts/dump_cond_inputs.py / codeclm.modules.conditioners:
//   QwTokenizerConditioner (description=LYRICS, max_len=600, +structure cover)
//   QwTextConditioner       (type_info=STYLE,  max_len=100)
//
// The Qwen2 tokenizer (tokens + merges) is embedded in the LeLM gguf
// (tokenizer.ggml.tokens / tokenizer.ggml.merges, model='gpt2'). The structure /
// musicality / <|im_*|> special tokens are NOT in the gguf list — they are
// appended with fixed ids (matching add_tokens(special_tokens=True)).
//
// Reuses bpe.h for byte-level encoding, GPT-2 pre-tokenization and merge ranking;
// adds Qwen2 special-token splitting (whole-unit, span-preserving, no EOS append)
// and the structure-cover computation.
#pragma once
#include "bpe.h"

#include <array>
#include <string>
#include <vector>

namespace songgen_tok {

static constexpr int PAD_ID      = 151643;  // <|endoftext|>
static constexpr int IM_START_ID = 151644;
static constexpr int IM_END_ID   = 151645;
static constexpr int DESC_MAX_LEN = 600;
static constexpr int TYPE_MAX_LEN = 100;

// vocab.yaml order, ids 151646.. (added via add_tokens(special_tokens=True))
static const char * const STRUCT_TOKENS[] = {
    "[verse]",        "[chorus]",      "[bridge]",     "[intro-short]",  "[intro-medium]",
    "[intro-long]",   "[outro-short]", "[outro-medium]", "[outro-long]", "[inst-short]",
    "[inst-medium]",  "[inst-long]",   "[silence]",
};
static constexpr int N_STRUCT = 13;  // ids 151646..151658

static const char * const MUSICALITY_TOKENS[] = {
    "[Musicality-very-high]", "[Musicality-high]", "[Musicality-medium]",
    "[Musicality-low]",       "[Musicality-very-low]", "[Pure-Music]",
};
static constexpr int N_MUSICALITY = 6;  // ids 151646..151651

struct SpecialEntry {
    std::string text;
    int         id;
};

struct Tokenizer {
    BPETokenizer              bpe;
    std::vector<SpecialEntry> desc_specials;  // <|im_start|> + 13 structure tokens
    std::vector<SpecialEntry> type_specials;  // <|im_start|> + 6 musicality tokens
};

static bool load(Tokenizer * t, const char * gguf_path) {
    if (!load_bpe_from_gguf(&t->bpe, gguf_path)) {
        return false;
    }
    // '.' is appended to add_token_list with special_tokens=True (version != 'v1'),
    // so it is split out as a whole unit; it keeps its base-vocab id 13.
    const int dot_id = t->bpe.vocab.count(".") ? t->bpe.vocab.at(".") : 13;

    t->desc_specials.push_back({ "<|im_start|>", IM_START_ID });
    for (int i = 0; i < N_STRUCT; i++) {
        t->desc_specials.push_back({ STRUCT_TOKENS[i], 151646 + i });
    }
    t->desc_specials.push_back({ ".", dot_id });
    t->type_specials.push_back({ "<|im_start|>", IM_START_ID });
    for (int i = 0; i < N_MUSICALITY; i++) {
        t->type_specials.push_back({ MUSICALITY_TOKENS[i], 151646 + i });
    }
    t->type_specials.push_back({ ".", dot_id });
    return true;
}

// Byte-level BPE a literal text span (no special handling, no EOS), mirroring
// the HF tokenizer's behavior for the text BETWEEN added special tokens.
static void encode_span(const BPETokenizer * bpe, const std::string & span, std::vector<int> & out) {
    if (span.empty()) {
        return;
    }
    auto chunks = gpt2_pre_tokenize(span);
    for (const auto & ch : chunks) {
        encode_chunk(bpe, ch, out);
    }
}

// Tokenize `text` splitting out `specials` as whole units (longest-match at each
// position, HF added-token semantics: spans between specials kept verbatim,
// including surrounding whitespace). No EOS appended.
static std::vector<int> encode_with_specials(const Tokenizer * t, const std::string & text,
                                             const std::vector<SpecialEntry> & specials) {
    std::vector<int> ids;
    size_t           pos = 0;
    std::string      pending;  // accumulated literal span awaiting a special boundary
    while (pos < text.size()) {
        const SpecialEntry * hit     = nullptr;
        size_t               hit_len = 0;
        for (const auto & sp : specials) {
            const std::string & s = sp.text;
            if (s.size() > hit_len && text.compare(pos, s.size(), s) == 0) {
                hit     = &sp;
                hit_len = s.size();
            }
        }
        if (hit) {
            encode_span(&t->bpe, pending, ids);
            pending.clear();
            ids.push_back(hit->id);
            pos += hit_len;
        } else {
            pending.push_back(text[pos]);
            pos++;
        }
    }
    encode_span(&t->bpe, pending, ids);
    return ids;
}

static void pad_to(std::vector<int> & v, int max_len, int pad_id) {
    if ((int) v.size() > max_len) {
        v.resize(max_len);
    } else {
        v.resize(max_len, pad_id);
    }
}

struct ConditioningArrays {
    std::array<int32_t, DESC_MAX_LEN> desc_ids_cond{};
    std::array<int32_t, DESC_MAX_LEN> desc_cover_cond{};
    std::array<int32_t, DESC_MAX_LEN> desc_ids_uncond{};
    std::array<int32_t, DESC_MAX_LEN> desc_cover_uncond{};
    std::array<int32_t, TYPE_MAX_LEN> type_ids_cond{};
    std::array<int32_t, TYPE_MAX_LEN> type_ids_uncond{};
};

static bool is_struct_id(int id) { return id >= 151646 && id <= 151646 + N_STRUCT - 1; }

// cover[pos]: 0 until first structure token; from each structure-token position st
// to the next structure token (or unpadded end), cover = id - 151645 (i.e. [verse]->1).
static std::vector<int> compute_cover(const std::vector<int> & ids) {
    std::vector<int> cover(ids.size(), 0);
    std::vector<int> sp;
    for (int i = 0; i < (int) ids.size(); i++) {
        if (is_struct_id(ids[i])) {
            sp.push_back(i);
        }
    }
    sp.push_back((int) ids.size());  // attention_mask covers all real tokens here
    for (int i = 0; i + 1 < (int) sp.size(); i++) {
        int st = sp[i];
        for (int p = st; p < sp[i + 1]; p++) {
            cover[p] = ids[st] - IM_END_ID;
        }
    }
    return cover;
}

static void fill_arr(const std::vector<int> & v, int32_t * dst, int max_len) {
    for (int i = 0; i < max_len; i++) {
        dst[i] = (i < (int) v.size()) ? (int32_t) v[i] : 0;
    }
}

// gen_type for conditioning: 0=mixed, 1=vocal, 2=bgm (pure-music).
// bgm matches levo_inference.forward: description gets a leading "[Pure-Music], " and the
// lyric stream is dropped to '.'. mixed/vocal leave the user lyric/description untouched.
enum CondGenType { COND_MIXED = 0, COND_VOCAL = 1, COND_BGM = 2 };

// Build cond + uncond conditioning arrays for (lyric, description).
// lyric/description are arbitrary user text (already containing [verse]/[chorus]/...).
static void build(const Tokenizer * t, const std::string & lyric_in, const std::string & description_in,
                  ConditioningArrays * out, CondGenType gen_type = COND_MIXED) {
    std::string lyric       = (gen_type == COND_BGM) ? "." : lyric_in;
    std::string description = description_in;
    // ----- description (LYRICS) conditioner, max_len=600 -----
    {
        std::vector<int> ids = encode_with_specials(t, "<|im_start|>" + lyric, t->desc_specials);
        std::vector<int> cov = compute_cover(ids);
        pad_to(ids, DESC_MAX_LEN, PAD_ID);
        pad_to(cov, DESC_MAX_LEN, 0);
        fill_arr(ids, out->desc_ids_cond.data(), DESC_MAX_LEN);
        fill_arr(cov, out->desc_cover_cond.data(), DESC_MAX_LEN);
    }
    {  // UNCOND: lyric=None -> just "<|im_start|>"
        std::vector<int> ids = encode_with_specials(t, "<|im_start|>", t->desc_specials);
        std::vector<int> cov = compute_cover(ids);
        pad_to(ids, DESC_MAX_LEN, PAD_ID);
        pad_to(cov, DESC_MAX_LEN, 0);
        fill_arr(ids, out->desc_ids_uncond.data(), DESC_MAX_LEN);
        fill_arr(cov, out->desc_cover_uncond.data(), DESC_MAX_LEN);
    }

    // ----- type_info (STYLE) conditioner, max_len=100 -----
    // levo lowercases description, prefixes "[Musicality-very-high], "
    std::string desc_lower = description;
    for (char & ch : desc_lower) {
        if (ch >= 'A' && ch <= 'Z') {
            ch = (char) (ch + 32);
        }
    }
    {
        // bgm/pure-music: insert [Pure-Music] after the musicality token (levo_inference).
        const char *     pm   = (gen_type == COND_BGM) ? "[Pure-Music], " : "";
        std::string      text = std::string("<|im_start|>[Musicality-very-high], ") + pm + desc_lower;
        std::vector<int> ids  = encode_with_specials(t, text, t->type_specials);
        pad_to(ids, TYPE_MAX_LEN, PAD_ID);
        fill_arr(ids, out->type_ids_cond.data(), TYPE_MAX_LEN);
    }
    {  // UNCOND: "[Musicality-very-low], ."
        std::string      text = "<|im_start|>[Musicality-very-low], .";
        std::vector<int> ids  = encode_with_specials(t, text, t->type_specials);
        pad_to(ids, TYPE_MAX_LEN, PAD_ID);
        fill_arr(ids, out->type_ids_uncond.data(), TYPE_MAX_LEN);
    }
}

}  // namespace songgen_tok
