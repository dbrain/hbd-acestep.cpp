#!/usr/bin/env python3
"""
Slice the SongGeneration model_septoken aux tensors needed for the from-codes
decode path -> a single F32 GGUF.

  /tmp/sg-venv/bin/python scripts/convert_septoken_aux.py <model_2.safetensors> <out.gguf>

Contents:
  rvq_{vocal,bgm}.codebook.weight [16384,32]   (descript RVQ codebook embedding)
  rvq_{vocal,bgm}.out_proj.weight [1024,32]    (weight-norm folded 1x1 conv 32->1024)
  rvq_{vocal,bgm}.out_proj.bias   [1024]
  mask_emb.weight [3,24]
  zero_cond_embedding1 [1024]
  normfeat.mean[64] / normfeat.std[64]  (pre-computed from the running sums)

normfeat (Feature1DProcessor): mean = sum_x/counts, std = sqrt(sum_x2/counts - mean^2).
return_sample (denorm) does x*(std/target_std)^power_std + mean, target_std=1, power_std=1,
so std is stored directly and the cpp side multiplies by std then adds mean.

out_proj is a weight_norm'd Conv1d(32->1024,k=1): w = g * v / ||v|| with the L2 norm taken
over the (in,k) dims per output channel (torch weight_norm dim=0). k=1 so it folds to a plain
[1024,32] matmul matrix; stored row-major [out=1024, in=32] -> the cpp graph mul_mat's it.
"""
import sys
import numpy as np
import torch
from safetensors import safe_open
import gguf

CKPT, OUT = sys.argv[1], sys.argv[2]
ARCH = "songgen-septoken"


def fold_wnconv1d(f, pfx):
    # weight_norm Conv1d(k=1) dim=0: w = g * v / ||v|| (L2 over the in dim per out channel).
    g = f.get_tensor(pfx + ".weight_g").float().numpy()  # [out,1,1]
    v = f.get_tensor(pfx + ".weight_v").float().numpy()  # [out,in,1]
    b = f.get_tensor(pfx + ".bias").float().numpy()      # [out]
    out_ch = v.shape[0]
    vf = v.reshape(out_ch, -1)                           # [out,in]
    norm = np.sqrt((vf * vf).sum(axis=1, keepdims=True))  # [out,1]
    w = vf * (g.reshape(out_ch, 1) / norm)               # [out,in]
    return np.ascontiguousarray(w, dtype=np.float32), np.ascontiguousarray(b, dtype=np.float32)


def main():
    f = safe_open(CKPT, "pt")

    w = gguf.GGUFWriter(OUT, ARCH, use_temp_file=True)
    w.add_name("SongGeneration-septoken-aux")

    counts = f.get_tensor("normfeat.counts").float().numpy()      # [1]
    sum_x = f.get_tensor("normfeat.sum_x").float().numpy()        # [64]
    sum_x2 = f.get_tensor("normfeat.sum_x2").float().numpy()      # [64]
    c = float(counts[0])
    if c < 10:
        mean = np.zeros_like(sum_x)
        std = np.ones_like(sum_x)
    else:
        mean = sum_x / c
        std = np.sqrt(np.clip(sum_x2 / c - mean * mean, 0.0, None))
    mean = np.ascontiguousarray(mean, dtype=np.float32)
    std = np.ascontiguousarray(std, dtype=np.float32)

    names = []

    def put(name, arr):
        w.add_tensor(name, np.ascontiguousarray(arr, dtype=np.float32))
        names.append((name, tuple(arr.shape)))

    for tag, pfx in (("vocal", "rvq_bestrq_emb"), ("bgm", "rvq_bestrq_bgm_emb")):
        cb = f.get_tensor(pfx + ".quantizers.0.codebook.weight").float().numpy()  # [16384,32]
        ip_w, ip_b = fold_wnconv1d(f, pfx + ".quantizers.0.in_proj")              # [32,1024],[32]
        op_w, op_b = fold_wnconv1d(f, pfx + ".quantizers.0.out_proj")             # [1024,32],[1024]
        put(f"rvq_{tag}.codebook.weight", cb)
        put(f"rvq_{tag}.in_proj.weight", ip_w)
        put(f"rvq_{tag}.in_proj.bias", ip_b)
        put(f"rvq_{tag}.out_proj.weight", op_w)
        put(f"rvq_{tag}.out_proj.bias", op_b)

    put("mask_emb.weight", f.get_tensor("mask_emb.weight").float().numpy())       # [3,24]
    put("zero_cond_embedding1", f.get_tensor("zero_cond_embedding1").float().numpy())  # [1024]
    put("normfeat.mean", mean)
    put("normfeat.std", std)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()

    print(f"[septoken-aux] {len(names)} tensors -> {OUT}", file=sys.stderr)
    for n, s in names:
        print(f"   {n:34s} {s}", file=sys.stderr)
    print(f"[septoken-aux] normfeat counts={c:.0f} mean[:3]={mean[:3]} std[:3]={std[:3]}", file=sys.stderr)


if __name__ == "__main__":
    main()
