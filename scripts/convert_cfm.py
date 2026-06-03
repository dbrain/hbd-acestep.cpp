#!/usr/bin/env python3
"""
Slice the SongGeneration CFM reflow ESTIMATOR (custom time-conditioned non-causal
GPT2) -> a single GGUF.

Source: ckpt-audio/model_septoken/model_2.safetensors, keys under
`cfm_wrapper.estimator.` (221 keys). The estimator is the custom GPT2Model in
models_gpt.models.gpt2_rope2_time_new_correct_mask_noncasual_reflow.

GPT2 Conv1D layers (c_attn/c_proj/c_fc/mlp.c_proj) store weight as [in, out].
ggml mul_mat(W, x) with x ne=[in, T] wants W ne=[in, out] i.e. numpy [out, in].
So Conv1D weights are TRANSPOSED here ([in,out] -> [out,in]); plain nn.Linear
weights are already [out, in] and pass through.

  /tmp/sg-venv/bin/python scripts/convert_cfm.py <ckpt> <out.gguf>
"""
import sys
import numpy as np
import gguf
from safetensors import safe_open

CKPT, OUT = sys.argv[1], sys.argv[2]
ARCH = "songgen-cfm"
PFX = "cfm_wrapper.estimator."

N_LAYER = 16
N_HEAD = 20
N_EMBD = 2200
N_INNER = 4400
N_POS = 1000
HEAD_DIM = N_EMBD // N_HEAD  # 110
FLOW_T_SIZE = 512

# n_embd=2200 is not a multiple of 32, so Q8_0 (block=32 along ne[0]) is illegal for
# the matmul weights whose input dim is 2200/4400. Use BF16 (no block constraint).
BF16 = gguf.GGMLQuantizationType.BF16
F32 = gguf.GGMLQuantizationType.F32


def to_bf16_u16(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    return (a.view(np.uint32) >> 16).astype(np.uint16)


def main():
    f = safe_open(CKPT, "pt")
    sd = {}
    for k in f.keys():
        if k.startswith(PFX):
            sd[k[len(PFX):]] = f.get_tensor(k).float().numpy()

    w = gguf.GGUFWriter(OUT, ARCH, use_temp_file=True)
    w.add_name("SongGeneration-CFM-reflow-estimator")
    w.add_uint32("songgen-cfm.n_layer", N_LAYER)
    w.add_uint32("songgen-cfm.n_head", N_HEAD)
    w.add_uint32("songgen-cfm.n_embd", N_EMBD)
    w.add_uint32("songgen-cfm.n_inner", N_INNER)
    w.add_uint32("songgen-cfm.n_positions", N_POS)
    w.add_uint32("songgen-cfm.head_dim", HEAD_DIM)
    w.add_uint32("songgen-cfm.flow_t_size", FLOW_T_SIZE)
    w.add_float32("songgen-cfm.layer_norm_eps", 1e-5)
    w.add_float32("songgen-cfm.rope_theta", 10000.0)

    names = []

    # Q8_0 requires the quantized dim (ne[0], i.e. last numpy axis) be a multiple
    # of 32. 2200/4400/512 are all multiples of 32 -> safe. Quantize the big mats;
    # keep small vectors (biases, norms, tables) F32.
    def put_q8(name, arr):
        w.add_tensor(name, to_bf16_u16(arr), raw_dtype=BF16)
        names.append(name)

    def put_f32(name, arr):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        w.add_tensor(name, arr, raw_dtype=F32)
        names.append(name)

    # Conv1D weight [in,out] -> [out,in] (so ggml ne=[in,out], mul_mat-ready).
    def put_conv1d_w(name, arr):
        assert arr.ndim == 2
        put_q8(name, np.ascontiguousarray(arr.T))

    # ---- position embeddings (wpe). wte unused (inputs_embeds given). ----
    put_f32("wpe.weight", sd["wpe.weight"])  # [1000, 2200]

    # ---- adaln_single time embedder ----
    # timestep_embedder: Linear(512->2200), SiLU, Linear(2200->2200)  (already [out,in])
    put_q8("adaln.te.l1.weight", sd["adaln_single.emb.timestep_embedder.linear_1.weight"])  # [2200,512]
    put_f32("adaln.te.l1.bias", sd["adaln_single.emb.timestep_embedder.linear_1.bias"])
    put_q8("adaln.te.l2.weight", sd["adaln_single.emb.timestep_embedder.linear_2.weight"])  # [2200,2200]
    put_f32("adaln.te.l2.bias", sd["adaln_single.emb.timestep_embedder.linear_2.bias"])
    # adaln_single.linear: Linear(2200 -> 6*2200=13200)
    put_q8("adaln.linear.weight", sd["adaln_single.linear.weight"])  # [13200,2200]
    put_f32("adaln.linear.bias", sd["adaln_single.linear.bias"])

    # ---- global scale_shift_table (final proj modulation) [2,2200] ----
    put_f32("scale_shift_table", sd["scale_shift_table"])

    # ---- per-block ----
    for i in range(N_LAYER):
        sp = f"h.{i}."
        op = f"blk.{i}."
        put_f32(op + "ln_1.weight", sd[sp + "ln_1.weight"])
        put_f32(op + "ln_1.bias", sd[sp + "ln_1.bias"])
        put_f32(op + "ln_2.weight", sd[sp + "ln_2.weight"])
        put_f32(op + "ln_2.bias", sd[sp + "ln_2.bias"])
        # attn c_attn [2200, 6600] -> [6600, 2200]; columns q|k|v each 2200
        put_conv1d_w(op + "attn.c_attn.weight", sd[sp + "attn.c_attn.weight"])
        put_f32(op + "attn.c_attn.bias", sd[sp + "attn.c_attn.bias"])  # [6600]
        put_conv1d_w(op + "attn.c_proj.weight", sd[sp + "attn.c_proj.weight"])  # [2200,2200]->[2200,2200]
        put_f32(op + "attn.c_proj.bias", sd[sp + "attn.c_proj.bias"])
        put_conv1d_w(op + "mlp.c_fc.weight", sd[sp + "mlp.c_fc.weight"])    # [2200,4400]->[4400,2200]
        put_f32(op + "mlp.c_fc.bias", sd[sp + "mlp.c_fc.bias"])             # [4400]
        put_conv1d_w(op + "mlp.c_proj.weight", sd[sp + "mlp.c_proj.weight"])  # [4400,2200]->[2200,4400]
        put_f32(op + "mlp.c_proj.bias", sd[sp + "mlp.c_proj.bias"])           # [2200]
        # per-block scale_shift_table [6,2200]
        put_f32(op + "scale_shift_table", sd[sp + "scale_shift_table"])

    # ---- final ----
    put_f32("ln_f.weight", sd["ln_f.weight"])
    put_f32("ln_f.bias", sd["ln_f.bias"])
    put_q8("proj_out.weight", sd["proj_out.weight"])  # [2200,2200]
    put_f32("proj_out.bias", sd["proj_out.bias"])

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()

    print(f"[cfm-gguf] {len(names)} tensors -> {OUT}", file=sys.stderr)
    for n in ["wpe.weight", "adaln.te.l1.weight", "adaln.linear.weight",
              "blk.0.attn.c_attn.weight", "blk.0.mlp.c_fc.weight",
              "blk.0.mlp.c_proj.weight", "scale_shift_table", "proj_out.weight"]:
        print("  ", n, file=sys.stderr)


if __name__ == "__main__":
    main()
