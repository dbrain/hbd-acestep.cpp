#!/usr/bin/env python3
"""
Slice SongGeneration (LeVo 2) model.pt -> a single BF16 GGUF for the LeLM.
Mirrors acestep.cpp convert.py conventions: BF16 base (Q8 applied downstream via
ggml quantize, selectively on the transformer matmuls). Embeds config + the Qwen2
BPE tokenizer + structure-token map so the GGUF is self-contained for the cpp port.

Audio stack (model_1rvq / model_septoken / vae) is a SEPARATE converter (CFM/VAE port).

  python convert_songgen_lelm.py <ckpt_dir> <qwen2_tok_dir> <out.gguf>
    ckpt_dir     : has config.yaml + model.pt
    qwen2_tok_dir: has vocab.json + merges.txt   (third_party/Qwen2-7B)
"""
import sys, json, os
import numpy as np
import torch
import gguf
from omegaconf import OmegaConf

CKPT, QTOK, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
ARCH = "songgen-lelm"
BF16 = gguf.GGMLQuantizationType.BF16

def to_bf16_u16(t):
    # bf16 source -> exact; fp32 source -> truncated (acestep convention). Q8 downstream.
    a = t.detach().to(torch.float32).contiguous().numpy()
    return (a.view(np.uint32) >> 16).astype(np.uint16)

def main():
    cfg = OmegaConf.load(os.path.join(CKPT, "config.yaml"))
    lm = cfg.lm
    sd = torch.load(os.path.join(CKPT, "model.pt"), map_location="cpu", mmap=True)

    w = gguf.GGUFWriter(OUT, ARCH, use_temp_file=True)
    w.add_name("SongGeneration-v2-LeLM")
    # main transformer dims (sub mirrors dim/ffn/heads, only depth differs)
    w.add_block_count(int(lm.num_layers))
    w.add_embedding_length(int(lm.dim))
    w.add_feed_forward_length(int(lm.intermediate_size))
    w.add_head_count(int(lm.num_heads))
    w.add_head_count_kv(int(lm.num_heads))          # MHA, kv_repeat=1
    w.add_rope_freq_base(float(lm.rope_theta))
    w.add_layer_norm_rms_eps(1e-5)
    # songgen-specific structural metadata the cpp graph needs
    w.add_uint32("songgen.num_layers_sub", int(lm.num_layers_sub))
    w.add_float32("songgen.rope_theta_sub", float(lm.rope_theta_sub))
    w.add_uint32("songgen.code_depth", int(lm.code_depth))
    w.add_uint32("songgen.code_size", int(lm.code_size))            # 16384 (+EOS=16385 head dim)
    w.add_array("songgen.delays", list(cfg.codebooks_pattern.delay.delays))
    w.add_uint32("songgen.frame_rate", int(cfg.audio_tokenizer_frame_rate))
    w.add_float32("songgen.cfg_coef", float(cfg.classifier_free_guidance.inference_coef))
    w.add_uint32("songgen.prompt_len", int(cfg.get("prompt_len", 10)))
    w.add_string("songgen.config_json", json.dumps(OmegaConf.to_container(lm), separators=(",", ":")))

    # structure tokens ([verse] etc.) -> ids, for lyric tokenization in cpp
    vocab_yaml = OmegaConf.load(os.path.join(os.path.dirname(__file__), "conf", "vocab.yaml"))
    w.add_string("songgen.structure_tokens", json.dumps(list(vocab_yaml)))

    # Qwen2 BPE tokenizer (lyrics/description -> token ids -> embedding gather)
    with open(os.path.join(QTOK, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    toks = [""] * len(vocab)
    for s, i in vocab.items():
        if 0 <= i < len(toks):
            toks[i] = s
    merges = []
    with open(os.path.join(QTOK, "merges.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if line and not line.startswith("#version:"):
                merges.append(line)
    w.add_tokenizer_model("gpt2")
    w.add_token_list(toks)
    w.add_token_merges(merges)

    # tensors: keep stable names, drop only the "audiolm." prefix
    n, nbytes = 0, 0
    for k in sorted(sd.keys()):
        if not k.startswith("audiolm."):
            continue
        name = k[len("audiolm."):]
        arr = to_bf16_u16(sd[k])
        w.add_tensor(name, arr, raw_dtype=BF16)
        n += 1; nbytes += arr.nbytes
    print(f"[gguf] {n} tensors, {nbytes/1e9:.2f} GB bf16 -> {OUT}", file=sys.stderr)

    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(progress=True); w.close()
    print(f"[gguf] wrote {os.path.getsize(OUT)/1e9:.2f} GB", file=sys.stderr)

if __name__ == "__main__":
    main()
