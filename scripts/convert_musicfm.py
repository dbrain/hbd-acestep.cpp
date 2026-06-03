#!/usr/bin/env python3
"""
Slice the MusicFM_95M (BESTRQ) audio encoder from model_septoken/model_2.safetensors
(prefix `bestrq.`) -> a single GGUF for the C++/ggml port (src/songgen-musicfm.h).

  /tmp/sg-venv/bin/python scripts/convert_musicfm.py <model_2.safetensors> <out.gguf>

features_only path uses: mel front-end -> Conv2dSubsampling -> Wav2Vec2ConformerEncoder
(do_stable_layer_norm). The RandomProjectionQuantizer / self.linear / cls_token are UNUSED.
The encoder's pos_conv_embed is also UNUSED (HF Wav2Vec2ConformerEncoder.forward never calls
it for the rotary path), so it is skipped here.

Tensor dtypes:
  - 2D matmul weights (linear, attn q/k/v/out, ffn, conv stem linear) : BF16
  - conv stem Conv2d weights, conformer conv1d weights                : BF16
  - norms / biases / batchnorm running stats / mel fb+window          : F32
BatchNorm is folded NOT here (kept as running_mean/var/weight/bias F32 so the cpp applies
eval-mode batchnorm = (x-mean)/sqrt(var+eps)*w+b verbatim).
"""
import sys
import numpy as np
import torch
from safetensors import safe_open
import gguf

CKPT, OUT = sys.argv[1], sys.argv[2]
ARCH = "songgen-musicfm"
PREFIX = "bestrq.model."
BF16 = gguf.GGMLQuantizationType.BF16

N_LAYERS = 12
HIDDEN = 1024
HEADS = 16
HEAD_DIM = 64
FFN = 4096
CONV_DIM = 512
N_MELS = 128
N_FFT = 2048
HOP = 240
SR = 24000
BN_EPS = 1e-5
LN_EPS = 1e-5
ROTARY_BASE = 10000
DEPTHWISE_K = 31
MEL_MEAN = 6.768444971712967
MEL_STD = 18.417922652295623


def to_bf16_u16(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    return (a.view(np.uint32) >> 16).astype(np.uint16)


def main():
    f = safe_open(CKPT, "pt")

    def g(name):
        return f.get_tensor(PREFIX + name).float().numpy()

    w = gguf.GGUFWriter(OUT, ARCH, use_temp_file=True)
    w.add_name("SongGeneration-MusicFM_95M-BESTRQ")
    w.add_uint32(ARCH + ".n_layers", N_LAYERS)
    w.add_uint32(ARCH + ".hidden", HIDDEN)
    w.add_uint32(ARCH + ".heads", HEADS)
    w.add_uint32(ARCH + ".head_dim", HEAD_DIM)
    w.add_uint32(ARCH + ".ffn", FFN)
    w.add_uint32(ARCH + ".conv_dim", CONV_DIM)
    w.add_uint32(ARCH + ".n_mels", N_MELS)
    w.add_uint32(ARCH + ".n_fft", N_FFT)
    w.add_uint32(ARCH + ".hop", HOP)
    w.add_uint32(ARCH + ".sample_rate", SR)
    w.add_uint32(ARCH + ".depthwise_kernel", DEPTHWISE_K)
    w.add_uint32(ARCH + ".rotary_base", ROTARY_BASE)
    w.add_float32(ARCH + ".bn_eps", BN_EPS)
    w.add_float32(ARCH + ".ln_eps", LN_EPS)
    w.add_float32(ARCH + ".mel_mean", MEL_MEAN)
    w.add_float32(ARCH + ".mel_std", MEL_STD)

    names = []

    def put_bf16(name, arr):
        w.add_tensor(name, to_bf16_u16(arr), raw_dtype=BF16)
        names.append((name, tuple(arr.shape), "bf16"))

    def put_f32(name, arr):
        w.add_tensor(name, np.ascontiguousarray(arr, dtype=np.float32))
        names.append((name, tuple(arr.shape), "f32"))

    # ---- mel front-end buffers (exact, shipped) ----
    put_f32("mel.fb", g("preprocessor_melspec_2048.mel_stft.mel_scale.fb"))       # [1025,128]
    put_f32("mel.window", g("preprocessor_melspec_2048.mel_stft.spectrogram.window"))  # [2048]

    # ---- conv stem (Conv2dSubsampling: 2x Res2dModule + Linear) ----
    def put_res2d(idx):
        p = f"conv.conv.{idx}"
        o = f"conv.res.{idx}"
        for cv in ("conv1", "conv2", "conv3"):
            put_bf16(f"{o}.{cv}.weight", g(f"{p}.{cv}.weight"))   # [OC,IC,3,3]
            put_f32(f"{o}.{cv}.bias", g(f"{p}.{cv}.bias"))
        for bn in ("bn1", "bn2", "bn3"):
            put_f32(f"{o}.{bn}.weight", g(f"{p}.{bn}.weight"))
            put_f32(f"{o}.{bn}.bias", g(f"{p}.{bn}.bias"))
            put_f32(f"{o}.{bn}.running_mean", g(f"{p}.{bn}.running_mean"))
            put_f32(f"{o}.{bn}.running_var", g(f"{p}.{bn}.running_var"))

    put_res2d(0)
    put_res2d(1)
    put_bf16("conv.linear.weight", g("conv.linear.weight"))   # [1024,16384]
    put_f32("conv.linear.bias", g("conv.linear.bias"))

    # ---- conformer ----
    put_f32("conformer.embed_positions.inv_freq", g("conformer.embed_positions.inv_freq"))  # [32]
    put_f32("conformer.layer_norm.weight", g("conformer.layer_norm.weight"))
    put_f32("conformer.layer_norm.bias", g("conformer.layer_norm.bias"))

    for li in range(N_LAYERS):
        lp = f"conformer.layers.{li}"
        # ffn1 / ffn2
        for ff in ("ffn1", "ffn2"):
            put_f32(f"{lp}.{ff}_layer_norm.weight", g(f"{lp}.{ff}_layer_norm.weight"))
            put_f32(f"{lp}.{ff}_layer_norm.bias", g(f"{lp}.{ff}_layer_norm.bias"))
            put_bf16(f"{lp}.{ff}.intermediate_dense.weight", g(f"{lp}.{ff}.intermediate_dense.weight"))
            put_f32(f"{lp}.{ff}.intermediate_dense.bias", g(f"{lp}.{ff}.intermediate_dense.bias"))
            put_bf16(f"{lp}.{ff}.output_dense.weight", g(f"{lp}.{ff}.output_dense.weight"))
            put_f32(f"{lp}.{ff}.output_dense.bias", g(f"{lp}.{ff}.output_dense.bias"))
        # self-attn
        put_f32(f"{lp}.self_attn_layer_norm.weight", g(f"{lp}.self_attn_layer_norm.weight"))
        put_f32(f"{lp}.self_attn_layer_norm.bias", g(f"{lp}.self_attn_layer_norm.bias"))
        for pr in ("linear_q", "linear_k", "linear_v", "linear_out"):
            put_bf16(f"{lp}.self_attn.{pr}.weight", g(f"{lp}.self_attn.{pr}.weight"))
            put_f32(f"{lp}.self_attn.{pr}.bias", g(f"{lp}.self_attn.{pr}.bias"))
        # conv_module
        cm = f"{lp}.conv_module"
        put_f32(f"{cm}.layer_norm.weight", g(f"{cm}.layer_norm.weight"))
        put_f32(f"{cm}.layer_norm.bias", g(f"{cm}.layer_norm.bias"))
        # pointwise_conv1 [2048,1024,1] -> store as 2d matmul [2048,1024]
        put_bf16(f"{cm}.pointwise_conv1.weight", g(f"{cm}.pointwise_conv1.weight")[..., 0])
        # depthwise_conv [1024,1,31] -> [1024,31]
        put_bf16(f"{cm}.depthwise_conv.weight", g(f"{cm}.depthwise_conv.weight")[:, 0, :])
        put_f32(f"{cm}.batch_norm.weight", g(f"{cm}.batch_norm.weight"))
        put_f32(f"{cm}.batch_norm.bias", g(f"{cm}.batch_norm.bias"))
        put_f32(f"{cm}.batch_norm.running_mean", g(f"{cm}.batch_norm.running_mean"))
        put_f32(f"{cm}.batch_norm.running_var", g(f"{cm}.batch_norm.running_var"))
        put_bf16(f"{cm}.pointwise_conv2.weight", g(f"{cm}.pointwise_conv2.weight")[..., 0])
        # final ln
        put_f32(f"{lp}.final_layer_norm.weight", g(f"{lp}.final_layer_norm.weight"))
        put_f32(f"{lp}.final_layer_norm.bias", g(f"{lp}.final_layer_norm.bias"))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()

    nbf = sum(1 for _, _, d in names if d == "bf16")
    nf32 = sum(1 for _, _, d in names if d == "f32")
    print(f"[musicfm-gguf] {len(names)} tensors ({nbf} bf16 mats, {nf32} f32) -> {OUT}", file=sys.stderr)
    print(f"[musicfm-gguf] groups: mel(2), conv-stem({2*15+2}), conformer.head(3), 12x layer", file=sys.stderr)


if __name__ == "__main__":
    main()
