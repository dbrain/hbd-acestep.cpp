"""
Clone-path golden #3: RVQ encode (bestrq features -> integer codes + quantized embedding).

fetch_codes_batch: quantized, codes, *_ = self.rvq_bestrq_emb(bestrq_emb)   # bestrq_emb [B,1024,T]
ResidualVectorQuantize(input_dim=1024, n_codebooks=1, codebook_size=16384, codebook_dim=32).
With n_codebooks=1, the single VectorQuantize:
    z_e = in_proj(z)                     # WNConv1d 1024->32, kernel 1
    encodings = L2norm(z_e flattened (b t) d);  codebook = L2norm(codebook.weight)  # [16384,32]
    dist = |e|^2 - 2 e.cb^T + |cb|^2 ;   indices = argmax(-dist) = argmin(dist)     # == cosine NN
    z_q = codebook[indices] (decode_code);  z_q = out_proj(z_q)                      # WNConv1d 32->1024
codes shape [B, 1, T]; quantized [B, 1024, T]. eval() => no stale-code replacement.
Distance is euclidean ON L2-NORMALIZED vectors == cosine similarity NN (ViT-VQGAN factorized codes).

Feeds the layer-7 (vocal) and layer-3 (bgm) bestrq goldens from golden_clone_bestrq.py.

Run from /tmp/songgen-src:
  PYTHONPATH=codeclm/tokenizer:.:codeclm/tokenizer/Flow1dVAE:tools/gradio \
  CUDA_VISIBLE_DEVICES="" /tmp/sg-venv/bin/python \
  /home/dbrain/dev/songgen-port/scripts/golden_clone_rvq.py
"""
import os, sys, json, types
import numpy as np
import torch
import sg_compat
sys.modules.setdefault("k_diffusion", types.ModuleType("k_diffusion"))

from libs.rvq.descript_quantize3 import ResidualVectorQuantize
from safetensors import safe_open

OUT = "/home/dbrain/dev/songgen-port/golden-large/clone"
CKPT = "/home/dbrain/dev/songgen-port/ckpt-audio/model_septoken/model_2.safetensors"
os.makedirs(OUT, exist_ok=True)


def build_rvq(prefix):
    rvq = ResidualVectorQuantize(input_dim=1024, n_codebooks=1, codebook_size=16384,
                                 codebook_dim=32, quantizer_dropout=0.0, stale_tolerance=200).eval()
    sd = {}
    with safe_open(CKPT, "pt") as f:
        for k in f.keys():
            if k.startswith(prefix):
                sd[k[len(prefix):]] = f.get_tensor(k).float()
    missing, unexpected = rvq.load_state_dict(sd, strict=False)
    print(f"[rvq] prefix='{prefix}' loaded={len(sd)} missing={list(missing)} unexpected={list(unexpected)}")
    return rvq


def run(rvq, feat, tag):
    feat_t = torch.from_numpy(feat).float()  # [1,1024,T]
    with torch.no_grad():
        # forward returns: z_q, codes, latents, commit, codebook_loss, usage
        z_q, codes, latents, *_ = rvq(feat_t)
        # intermediate z_e for port debugging (pre-quant projected latents)
        z_e = rvq.quantizers[0].in_proj(feat_t)
    z_q_np = z_q.cpu().numpy().astype(np.float32)        # [1,1024,T]
    codes_np = codes.cpu().numpy().astype(np.int64)      # [1,1,T]
    z_e_np = z_e.cpu().numpy().astype(np.float32)        # [1,32,T]
    np.save(f"{OUT}/rvq_{tag}_codes.npy", codes_np)
    np.save(f"{OUT}/rvq_{tag}_quantized.npy", z_q_np)
    np.save(f"{OUT}/rvq_{tag}_z_e.npy", z_e_np)
    print(f"[rvq] {tag}: feat{feat.shape} -> codes{codes_np.shape} quant{z_q_np.shape} "
          f"code_range[{codes_np.min()},{codes_np.max()}]")
    return dict(codes=list(codes_np.shape), quantized=list(z_q_np.shape), z_e=list(z_e_np.shape),
                code_min=int(codes_np.min()), code_max=int(codes_np.max()),
                quant_std=float(z_q_np.std()), unique_codes=int(np.unique(codes_np).size))


emb7 = np.load(f"{OUT}/bestrq_out_layer7_vocal.npy")
emb3 = np.load(f"{OUT}/bestrq_out_layer3_bgm.npy")

rvq_vocal = build_rvq("rvq_bestrq_emb.")
rvq_bgm = build_rvq("rvq_bestrq_bgm_emb.")
s_vocal = run(rvq_vocal, emb7, "vocal")
s_bgm = run(rvq_bgm, emb3, "bgm")

meta = dict(
    task="clone_rvq_encode",
    module="libs.rvq.descript_quantize3.ResidualVectorQuantize",
    config=dict(input_dim=1024, n_codebooks=1, codebook_size=16384, codebook_dim=32, quantizer_dropout=0.0),
    weights=dict(ckpt=CKPT, vocal_prefix="rvq_bestrq_emb.", bgm_prefix="rvq_bestrq_bgm_emb.",
                 weight_norm="in_proj/out_proj are WNConv1d (weight_g/weight_v); reconstruct W = g * v/||v||"),
    quantize_math=dict(
        in_proj="WNConv1d(1024->32, k=1) -> z_e [B,32,T]",
        normalize="L2-normalize z_e (per time-step, over 32 dims) AND codebook rows (F.normalize, default p=2 dim=1)",
        distance="euclidean on the L2-normalized vectors = |e|^2 - 2 e.cb^T + |cb|^2",
        selection="indices = argmax(-dist) = argmin(dist) == nearest by COSINE similarity",
        decode="z_q_proj = codebook.weight[indices] (NOT renormalized; raw codebook row) -> out_proj WNConv1d(32->1024)",
        note="ViT-VQGAN factorized + L2-normalized codes; eval() => no stale-code replacement; "
             "codebook lookup returns the ORIGINAL (un-normalized) codebook embedding, normalization is only for the argmin",
    ),
    inputs=dict(vocal="bestrq layer-7 features [1,1024,T] from golden_clone_bestrq",
                bgm="bestrq layer-3 features [1,1024,T]"),
    outputs=dict(vocal=s_vocal, bgm=s_bgm),
    saved=["rvq_{vocal,bgm}_codes.npy [1,1,T] int64",
           "rvq_{vocal,bgm}_quantized.npy [1,1024,T] f32 (out_proj output)",
           "rvq_{vocal,bgm}_z_e.npy [1,32,T] f32 (pre-quant projected latents)"],
    note="codes are exactly the per-stem audio_qt tokens the LeLM consumes (fetch_codes_batch returns these). "
         "codebook_size=16384 => 14-bit tokens per frame per stem.",
)
with open(f"{OUT}/rvq_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"[rvq] saved to {OUT}")
