#!/usr/bin/env python3
"""
Slice the model_1rvq single-RVQ codebook (rvq_bestrq_emb) -> F32 GGUF for the clone
pmt-stream encode (src/songgen-clone-encode.h).

  /tmp/sg-venv/bin/python scripts/convert_1rvq_aux.py <model_2_fixed.safetensors> <out.gguf>

Contents (mirror of the septoken aux rvq block, single stream named 'pmt'):
  rvq_pmt.codebook.weight [16384,32]
  rvq_pmt.in_proj.weight  [32,1024]   (weight-norm folded WNConv1d 1024->32, k=1)
  rvq_pmt.in_proj.bias    [32]
  rvq_pmt.out_proj.weight [1024,32]    (weight-norm folded WNConv1d 32->1024, k=1)
  rvq_pmt.out_proj.bias   [1024]
"""
import sys
import numpy as np
from safetensors import safe_open
import gguf

CKPT, OUT = sys.argv[1], sys.argv[2]
ARCH = "songgen-1rvq"


def fold_wnconv1d(f, pfx):
    g = f.get_tensor(pfx + ".weight_g").float().numpy()  # [out,1,1]
    v = f.get_tensor(pfx + ".weight_v").float().numpy()  # [out,in,1]
    b = f.get_tensor(pfx + ".bias").float().numpy()      # [out]
    out_ch = v.shape[0]
    vf = v.reshape(out_ch, -1)
    norm = np.sqrt((vf * vf).sum(axis=1, keepdims=True))
    w = vf * (g.reshape(out_ch, 1) / norm)
    return np.ascontiguousarray(w, dtype=np.float32), np.ascontiguousarray(b, dtype=np.float32)


def main():
    f = safe_open(CKPT, "pt")
    w = gguf.GGUFWriter(OUT, ARCH, use_temp_file=True)
    w.add_name("SongGeneration-1rvq-aux")

    names = []

    def put(name, arr):
        w.add_tensor(name, np.ascontiguousarray(arr, dtype=np.float32))
        names.append((name, tuple(arr.shape)))

    pfx = "rvq_bestrq_emb"
    cb = f.get_tensor(pfx + ".quantizers.0.codebook.weight").float().numpy()  # [16384,32]
    ip_w, ip_b = fold_wnconv1d(f, pfx + ".quantizers.0.in_proj")              # [32,1024],[32]
    op_w, op_b = fold_wnconv1d(f, pfx + ".quantizers.0.out_proj")             # [1024,32],[1024]
    put("rvq_pmt.codebook.weight", cb)
    put("rvq_pmt.in_proj.weight", ip_w)
    put("rvq_pmt.in_proj.bias", ip_b)
    put("rvq_pmt.out_proj.weight", op_w)
    put("rvq_pmt.out_proj.bias", op_b)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()

    print(f"[1rvq-aux] {len(names)} tensors -> {OUT}", file=sys.stderr)
    for n, s in names:
        print(f"   {n:30s} {s}", file=sys.stderr)


if __name__ == "__main__":
    main()
