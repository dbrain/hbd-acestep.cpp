#!/usr/bin/env python3
# Convert HTDemucs (hybrid transformer Demucs source separator) state_dict -> single GGUF
# for the C++/ggml port (src/songgen-htdemucs.h).
#
#   /tmp/demucs-venv/bin/python scripts/convert_htdemucs.py \
#       ckpt-audio/demucs/htdemucs_state_dict.pth gguf/songgen-htdemucs.gguf
#
# Architecture (config): depth=4, channels=48 (growth 2 -> 48/96/192/384), nfft=4096 hop=1024,
# cac=True, bottom_channels=512, sources=[drums,bass,other,vocals], t_layers=5 dim512 8heads.
# dconv_mode=1 -> DConv in ENCODERS only (decoder/tdecoder dconv keys in the .pth are UNUSED).
#
# Tensor dtypes:
#   - 2D transformer matmul weights (in_proj, out_proj, linear1/2, channel up/down samplers): BF16
#   - conv weights (Conv1d/Conv2d/ConvTranspose, dconv convs): BF16
#   - all norms / biases / layerscale / groupnorm / freq_emb: F32
# Conv weights are kept in torch [OC,IC,KH,KW] / [OC,IC,K] order (the cpp re-lays them out).
import sys, json
import numpy as np
import torch
import gguf

CKPT, OUT = sys.argv[1], sys.argv[2]
ARCH = "songgen-htdemucs"
BF16 = gguf.GGMLQuantizationType.BF16

DEPTH = 4
T_LAYERS = 5
T_DIM = 512
T_HEADS = 8
T_FFN = 2048
TRANSFORMER_CH = 384
NFFT = 4096
HOP = 1024
SR = 44100
SEGMENT_SAMPLES = 343980  # int(7.8 * 44100)


def to_bf16_u16(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    return (a.view(np.uint32) >> 16).astype(np.uint16)


def main():
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)

    def g(name):
        return sd[name].float().numpy()

    w = gguf.GGUFWriter(OUT, ARCH, use_temp_file=True)
    w.add_name("HTDemucs-htdemucs")
    w.add_uint32(ARCH + ".depth", DEPTH)
    w.add_uint32(ARCH + ".t_layers", T_LAYERS)
    w.add_uint32(ARCH + ".t_dim", T_DIM)
    w.add_uint32(ARCH + ".t_heads", T_HEADS)
    w.add_uint32(ARCH + ".t_ffn", T_FFN)
    w.add_uint32(ARCH + ".transformer_ch", TRANSFORMER_CH)
    w.add_uint32(ARCH + ".nfft", NFFT)
    w.add_uint32(ARCH + ".hop", HOP)
    w.add_uint32(ARCH + ".sample_rate", SR)
    w.add_uint32(ARCH + ".segment_samples", SEGMENT_SAMPLES)
    w.add_float32(ARCH + ".freq_emb_scale", 0.2)
    w.add_float32(ARCH + ".max_period", 10000.0)

    names = []

    def put_bf16(name, arr):
        w.add_tensor(name, to_bf16_u16(arr), raw_dtype=BF16)
        names.append((name, tuple(arr.shape), "bf16"))

    def put_f32(name, arr):
        w.add_tensor(name, np.ascontiguousarray(arr, dtype=np.float32))
        names.append((name, tuple(arr.shape), "f32"))

    def put_dconv(prefix):
        # DConv with depth=2; each sub-layer: conv1d(.0)+groupnorm(.1)+gelu, conv1d(.3)+groupnorm(.4)+glu, layerscale(.6)
        for s in range(2):
            p = f"{prefix}.dconv.layers.{s}"
            put_bf16(f"{p}.0.weight", g(f"{p}.0.weight"))   # [hidden, C, 3]
            put_f32(f"{p}.0.bias", g(f"{p}.0.bias"))
            put_f32(f"{p}.1.weight", g(f"{p}.1.weight"))    # groupnorm(1,hidden)
            put_f32(f"{p}.1.bias", g(f"{p}.1.bias"))
            put_bf16(f"{p}.3.weight", g(f"{p}.3.weight"))   # [2C, hidden, 1]
            put_f32(f"{p}.3.bias", g(f"{p}.3.bias"))
            put_f32(f"{p}.4.weight", g(f"{p}.4.weight"))    # groupnorm(1,2C)
            put_f32(f"{p}.4.bias", g(f"{p}.4.bias"))
            put_f32(f"{p}.6.scale", g(f"{p}.6.scale"))      # layerscale [C]

    # ---- spec encoders (Conv2d) ----
    for i in range(DEPTH):
        p = f"encoder.{i}"
        put_bf16(f"{p}.conv.weight", g(f"{p}.conv.weight"))   # [OC,IC,K,1]
        put_f32(f"{p}.conv.bias", g(f"{p}.conv.bias"))
        put_bf16(f"{p}.rewrite.weight", g(f"{p}.rewrite.weight"))  # [2OC,OC,1,1]
        put_f32(f"{p}.rewrite.bias", g(f"{p}.rewrite.bias"))
        put_dconv(p)

    # ---- time encoders (Conv1d) ----
    for i in range(DEPTH):
        p = f"tencoder.{i}"
        put_bf16(f"{p}.conv.weight", g(f"{p}.conv.weight"))   # [OC,IC,K]
        put_f32(f"{p}.conv.bias", g(f"{p}.conv.bias"))
        # tencoder.3 is empty (last_freq) -> only conv, no rewrite/dconv
        if f"{p}.rewrite.weight" in sd:
            put_bf16(f"{p}.rewrite.weight", g(f"{p}.rewrite.weight"))
            put_f32(f"{p}.rewrite.bias", g(f"{p}.rewrite.bias"))
            put_dconv(p)

    # ---- freq embedding ----
    put_f32("freq_emb.embedding.weight", g("freq_emb.embedding.weight"))  # [512,48]

    # ---- channel up/down samplers (Conv1d 1x1 == matmul) ----
    for nm in ["channel_upsampler", "channel_downsampler", "channel_upsampler_t", "channel_downsampler_t"]:
        put_bf16(f"{nm}.weight", g(f"{nm}.weight")[..., 0])  # [out,in,1] -> [out,in]
        put_f32(f"{nm}.bias", g(f"{nm}.bias"))

    # ---- crosstransformer ----
    put_f32("crosstransformer.norm_in.weight", g("crosstransformer.norm_in.weight"))
    put_f32("crosstransformer.norm_in.bias", g("crosstransformer.norm_in.bias"))
    put_f32("crosstransformer.norm_in_t.weight", g("crosstransformer.norm_in_t.weight"))
    put_f32("crosstransformer.norm_in_t.bias", g("crosstransformer.norm_in_t.bias"))

    def put_layer(stream, idx, is_cross):
        p = f"crosstransformer.{stream}.{idx}"
        if is_cross:
            put_bf16(f"{p}.cross_attn.in_proj_weight", g(f"{p}.cross_attn.in_proj_weight"))  # [3*512,512]
            put_f32(f"{p}.cross_attn.in_proj_bias", g(f"{p}.cross_attn.in_proj_bias"))
            put_bf16(f"{p}.cross_attn.out_proj.weight", g(f"{p}.cross_attn.out_proj.weight"))
            put_f32(f"{p}.cross_attn.out_proj.bias", g(f"{p}.cross_attn.out_proj.bias"))
            put_f32(f"{p}.norm3.weight", g(f"{p}.norm3.weight"))
            put_f32(f"{p}.norm3.bias", g(f"{p}.norm3.bias"))
        else:
            put_bf16(f"{p}.self_attn.in_proj_weight", g(f"{p}.self_attn.in_proj_weight"))
            put_f32(f"{p}.self_attn.in_proj_bias", g(f"{p}.self_attn.in_proj_bias"))
            put_bf16(f"{p}.self_attn.out_proj.weight", g(f"{p}.self_attn.out_proj.weight"))
            put_f32(f"{p}.self_attn.out_proj.bias", g(f"{p}.self_attn.out_proj.bias"))
        put_bf16(f"{p}.linear1.weight", g(f"{p}.linear1.weight"))   # [2048,512]
        put_f32(f"{p}.linear1.bias", g(f"{p}.linear1.bias"))
        put_bf16(f"{p}.linear2.weight", g(f"{p}.linear2.weight"))   # [512,2048]
        put_f32(f"{p}.linear2.bias", g(f"{p}.linear2.bias"))
        put_f32(f"{p}.norm1.weight", g(f"{p}.norm1.weight"))
        put_f32(f"{p}.norm1.bias", g(f"{p}.norm1.bias"))
        put_f32(f"{p}.norm2.weight", g(f"{p}.norm2.weight"))
        put_f32(f"{p}.norm2.bias", g(f"{p}.norm2.bias"))
        put_f32(f"{p}.norm_out.weight", g(f"{p}.norm_out.weight"))
        put_f32(f"{p}.norm_out.bias", g(f"{p}.norm_out.bias"))
        put_f32(f"{p}.gamma_1.scale", g(f"{p}.gamma_1.scale"))
        put_f32(f"{p}.gamma_2.scale", g(f"{p}.gamma_2.scale"))

    for idx in range(T_LAYERS):
        is_cross = (idx % 2 == 1)  # classic_parity=0 -> even=self, odd=cross
        put_layer("layers", idx, is_cross)
        put_layer("layers_t", idx, is_cross)

    # ---- decoders (Conv2d, no dconv) ----
    for i in range(DEPTH):
        p = f"decoder.{i}"
        put_bf16(f"{p}.conv_tr.weight", g(f"{p}.conv_tr.weight"))  # [IC,OC,K,1] (convtranspose2d)
        put_f32(f"{p}.conv_tr.bias", g(f"{p}.conv_tr.bias"))
        put_bf16(f"{p}.rewrite.weight", g(f"{p}.rewrite.weight"))  # [2IC,IC,3,3]
        put_f32(f"{p}.rewrite.bias", g(f"{p}.rewrite.bias"))

    # ---- time decoders (Conv1d/ConvTranspose1d, no dconv) ----
    for i in range(DEPTH):
        p = f"tdecoder.{i}"
        put_bf16(f"{p}.conv_tr.weight", g(f"{p}.conv_tr.weight"))  # [IC,OC,K]
        put_f32(f"{p}.conv_tr.bias", g(f"{p}.conv_tr.bias"))
        if f"{p}.rewrite.weight" in sd:
            put_bf16(f"{p}.rewrite.weight", g(f"{p}.rewrite.weight"))  # [2IC,IC,3]
            put_f32(f"{p}.rewrite.bias", g(f"{p}.rewrite.bias"))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()

    nbf = sum(1 for _, _, d in names if d == "bf16")
    nf32 = sum(1 for _, _, d in names if d == "f32")
    print(f"[htdemucs-gguf] {len(names)} tensors ({nbf} bf16, {nf32} f32) -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
