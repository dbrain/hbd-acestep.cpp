import os
import numpy as np
import torch
from transformers import Qwen2Tokenizer

TOK_PATH = "/tmp/songgen-src/third_party/Qwen2-7B"
if not os.path.exists(TOK_PATH):
    TOK_PATH = "/home/dbrain/dev/songgen-port/qwen2-tokenizer"

OUT_DIR = "/home/dbrain/dev/songgen-port/golden-large"
os.makedirs(OUT_DIR, exist_ok=True)

PAD_ID = 151643
IM_START = 151644
IM_END = 151645

STRUCT_TOKENS = [
    '[verse]', '[chorus]', '[bridge]',
    '[intro-short]', '[intro-medium]', '[intro-long]',
    '[outro-short]', '[outro-medium]', '[outro-long]',
    '[inst-short]', '[inst-medium]', '[inst-long]',
    '[silence]',
]
MUSICALITY_TOKENS = [
    '[Musicality-very-high]', '[Musicality-high]', '[Musicality-medium]',
    '[Musicality-low]', '[Musicality-very-low]', '[Pure-Music]',
]

LYRIC = ("[verse] walking through the quiet streets at night [verse] city lights are "
         "calling out my name [chorus] and i will find my way back home")
TYPE_COND_TEXT = ("[Musicality-very-high], warm indie folk, male vocal, "
                  "acoustic guitar, 92 bpm")
TYPE_UNCOND_TEXT = "[Musicality-very-low], ."

DESC_MAX_LEN = 600
TYPE_MAX_LEN = 100


def pad_1d(ids, max_len, pad_id):
    ids = list(ids)
    if len(ids) > max_len:
        return ids[:max_len]
    return ids + [pad_id] * (max_len - len(ids))


# ---------------------------------------------------------------------------
# description conditioner (QwTokenizerConditioner, max_len=600)
# separate tokenizer instance from type_info
# ---------------------------------------------------------------------------
desc_tok = Qwen2Tokenizer.from_pretrained(TOK_PATH)
# version != 'v1' appends '.' to add_token_list inside QwTokenizerConditioner.__init__
desc_tok.add_tokens(STRUCT_TOKENS + ['.'], special_tokens=True)

desc_vocab = desc_tok.get_vocab()
struct_token_ids = [desc_vocab[t] for t in STRUCT_TOKENS]


def desc_tokenize(text):
    # mirrors QwTokenizerConditioner.tokenize: prepend <|im_start|>, or just it if None
    s = '<|im_start|>' + text if text is not None else '<|im_start|>'
    enc = desc_tok(s)
    return enc['input_ids'], enc['attention_mask']


def desc_cover(token_ids, attention_mask):
    # replicate forward(): for each structure-token start st, fill cover[st:next] = id-151645
    tokens = torch.tensor(token_ids, dtype=torch.long)
    mask = torch.tensor(attention_mask, dtype=torch.long)
    is_sp = torch.zeros_like(tokens, dtype=torch.bool)
    for sid in struct_token_ids:
        is_sp |= (tokens == sid)
    cover = torch.zeros_like(tokens)
    sp_list = torch.where(is_sp)[0].tolist()
    sp_list.append(int(mask.sum().item()))
    for i, st in enumerate(sp_list[:-1]):
        cover[st: sp_list[i + 1]] = tokens[st] - IM_END  # IM_END == 151645
    return cover.tolist()


# COND
desc_ids_c, desc_mask_c = desc_tokenize(LYRIC)
desc_cover_c = desc_cover(desc_ids_c, desc_mask_c)
desc_ids_cond = pad_1d(desc_ids_c, DESC_MAX_LEN, PAD_ID)
desc_cover_cond = pad_1d(desc_cover_c, DESC_MAX_LEN, 0)

# UNCOND (text=None -> just "<|im_start|>")
desc_ids_u, desc_mask_u = desc_tokenize(None)
desc_cover_u = desc_cover(desc_ids_u, desc_mask_u)
desc_ids_uncond = pad_1d(desc_ids_u, DESC_MAX_LEN, PAD_ID)
desc_cover_uncond = pad_1d(desc_cover_u, DESC_MAX_LEN, 0)

# ---------------------------------------------------------------------------
# type_info conditioner (QwTextConditioner, max_len=100) - SEPARATE tokenizer
# ---------------------------------------------------------------------------
type_tok = Qwen2Tokenizer.from_pretrained(TOK_PATH)
type_tok.add_tokens(MUSICALITY_TOKENS + ['.'], special_tokens=True)
type_vocab = type_tok.get_vocab()
musicality_ids = [type_vocab[t] for t in MUSICALITY_TOKENS]


def type_tokenize(text):
    s = '<|im_start|>' + text if text is not None else '<|im_start|>'
    return type_tok(s)['input_ids']


type_ids_c = type_tokenize(TYPE_COND_TEXT)
type_ids_u = type_tokenize(TYPE_UNCOND_TEXT)
type_ids_cond = pad_1d(type_ids_c, TYPE_MAX_LEN, PAD_ID)
type_ids_uncond = pad_1d(type_ids_u, TYPE_MAX_LEN, PAD_ID)

# ---------------------------------------------------------------------------
# Verification prints
# ---------------------------------------------------------------------------
print("=" * 70)
print("TOKENIZER PATH:", TOK_PATH)
print("=" * 70)

print("\n--- SPECIAL ID CHECKS ---")
base = Qwen2Tokenizer.from_pretrained(TOK_PATH)
got_im_start = base.convert_tokens_to_ids('<|im_start|>')
got_eot = base.convert_tokens_to_ids('<|endoftext|>')
got_im_end = base.convert_tokens_to_ids('<|im_end|>')
print(f"<|im_start|> = {got_im_start} (expect 151644) {'OK' if got_im_start==151644 else '*** MISMATCH ***'}")
print(f"<|endoftext|> = {got_eot} (expect 151643) {'OK' if got_eot==151643 else '*** MISMATCH ***'}")
print(f"<|im_end|>   = {got_im_end} (expect 151645) {'OK' if got_im_end==151645 else '*** MISMATCH ***'}")

print("\n--- DESCRIPTION STRUCTURE TOKEN IDS (expect 151646..151658) ---")
ok_struct = True
for i, (t, sid) in enumerate(zip(STRUCT_TOKENS, struct_token_ids)):
    exp = 151646 + i
    flag = 'OK' if sid == exp else '*** MISMATCH ***'
    if sid != exp:
        ok_struct = False
    print(f"  {t:18s} id={sid} expect={exp} {flag}")
print("STRUCT IDS CONTIGUOUS 151646..151658:", ok_struct)

print("\n--- TYPE_INFO MUSICALITY TOKEN IDS (expect 151646..151651) ---")
ok_mus = True
for i, (t, sid) in enumerate(zip(MUSICALITY_TOKENS, musicality_ids)):
    exp = 151646 + i
    flag = 'OK' if sid == exp else '*** MISMATCH ***'
    if sid != exp:
        ok_mus = False
    print(f"  {t:24s} id={sid} expect={exp} {flag}")
print("MUSICALITY IDS CONTIGUOUS 151646..151651:", ok_mus)
print("type_info '.' id =", type_vocab['.'])

print("\n--- DESCRIPTION COND: unpadded len =", len(desc_ids_c), "---")
print("decoded tokens (first 40):")
print([desc_tok.convert_ids_to_tokens(i) for i in desc_ids_c[:40]])
print("ids (first 40):", desc_ids_c[:40])
print("cover (first 40):", desc_cover_c[:40])
print("cover unique values:", sorted(set(desc_cover_c)))

print("\n--- DESCRIPTION UNCOND: unpadded len =", len(desc_ids_u), "---")
print("decoded:", [desc_tok.convert_ids_to_tokens(i) for i in desc_ids_u])
print("ids:", desc_ids_u, "cover:", desc_cover_u)

print("\n--- TYPE_INFO COND: unpadded len =", len(type_ids_c), "---")
print("decoded tokens (first 40):")
print([type_tok.convert_ids_to_tokens(i) for i in type_ids_c[:40]])
print("ids (first 40):", type_ids_c[:40])

print("\n--- TYPE_INFO UNCOND: unpadded len =", len(type_ids_u), "---")
print("decoded:", [type_tok.convert_ids_to_tokens(i) for i in type_ids_u])
print("ids:", type_ids_u)

# ---------------------------------------------------------------------------
# Save .npy (int32)
# ---------------------------------------------------------------------------
outputs = {
    "desc_ids_cond": (desc_ids_cond, DESC_MAX_LEN),
    "desc_cover_cond": (desc_cover_cond, DESC_MAX_LEN),
    "desc_ids_uncond": (desc_ids_uncond, DESC_MAX_LEN),
    "desc_cover_uncond": (desc_cover_uncond, DESC_MAX_LEN),
    "type_ids_cond": (type_ids_cond, TYPE_MAX_LEN),
    "type_ids_uncond": (type_ids_uncond, TYPE_MAX_LEN),
}
print("\n--- WRITING .npy FILES ---")
for name, (arr, exp_len) in outputs.items():
    a = np.asarray(arr, dtype=np.int32)
    assert a.shape == (exp_len,), f"{name}: shape {a.shape} != ({exp_len},)"
    path = os.path.join(OUT_DIR, name + ".npy")
    np.save(path, a)
    print(f"  {path}  shape={a.shape} dtype={a.dtype}")
print("DONE")
