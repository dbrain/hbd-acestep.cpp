#!/usr/bin/env python3
"""
Slice the SongGeneration Stable-Audio Oobleck VAE DECODER -> a single BF16 GGUF.

Folds torch weight_norm (weight_g/weight_v) into plain conv weights at convert
time and pre-computes the SnakeBeta gains (alpha = exp(a), beta_inv = 1/exp(b),
alpha_logscale=True) so the cpp graph only does sin/sqr/mul/add. Conv weights are
BF16; snake gains + biases stay F32. Config goes into KV metadata.

  /tmp/sg-venv/bin/python scripts/convert_vae.py <ckpt> <out.gguf>
    ckpt : autoencoder_music_1320k.ckpt  (torch.load(...)["state_dict"], prefix decoder.)
"""
import sys
import numpy as np
import torch
import gguf

CKPT, OUT = sys.argv[1], sys.argv[2]
ARCH = "songgen-vae"
BF16 = gguf.GGMLQuantizationType.BF16

# decoder structure (reverse of encoder): c_mults=[1,2,4,8,16], strides=[2,4,4,6,10]
STRIDES = [10, 6, 4, 4, 2]            # coarse -> fine
IN_CH   = [2048, 1024, 512, 256, 128]
OUT_CH  = [1024, 512, 256, 128, 128]


def to_bf16_u16(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    return (a.view(np.uint32) >> 16).astype(np.uint16)


def fold_wn(sd, pfx):
    # torch.nn.utils.weight_norm dim=0: w = g * v / ||v||, norm over all non-0 dims.
    # Conv1d weight_v [out,in,k] -> norm over (in,k) per out.
    # ConvTranspose1d weight_v [in,out,k] -> dim=0 means norm over (out,k) per IN channel,
    # and weight_g there is [in,1,1]. We replicate torch exactly via per-dim0 L2.
    g = sd[pfx + ".weight_g"].float().numpy()      # [d0,1,1]
    v = sd[pfx + ".weight_v"].float().numpy()      # [d0, *, *]
    d0 = v.shape[0]
    vf = v.reshape(d0, -1)
    norm = np.sqrt((vf * vf).sum(axis=1))          # [d0]
    scale = (g.reshape(d0) / norm).reshape(d0, *([1] * (v.ndim - 1)))
    return v * scale                               # same shape as weight_v


def main():
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    sd = {k[len("decoder."):]: v for k, v in sd.items() if k.startswith("decoder.")}

    w = gguf.GGUFWriter(OUT, ARCH, use_temp_file=True)
    w.add_name("SongGeneration-StableAudio-Oobleck-VAE-decoder")
    w.add_uint32("songgen-vae.out_channels", 2)
    w.add_uint32("songgen-vae.channels", 128)
    w.add_uint32("songgen-vae.latent_dim", 64)
    w.add_array("songgen-vae.strides", STRIDES)
    w.add_array("songgen-vae.in_channels", IN_CH)
    w.add_array("songgen-vae.out_channels_blocks", OUT_CH)
    w.add_uint32("songgen-vae.n_blocks", 5)
    w.add_uint32("songgen-vae.downsampling_ratio", 1920)
    w.add_uint32("songgen-vae.sample_rate", 48000)
    w.add_bool("songgen-vae.use_snake", True)
    w.add_bool("songgen-vae.final_tanh", False)

    names = []

    def put_conv(out_name, src_pfx, has_bias):
        ww = fold_wn(sd, src_pfx)
        w.add_tensor(out_name + ".weight", to_bf16_u16(ww), raw_dtype=BF16)
        names.append(out_name + ".weight")
        if has_bias:
            b = sd[src_pfx + ".bias"].float().numpy().astype(np.float32)
            w.add_tensor(out_name + ".bias", b)
            names.append(out_name + ".bias")

    def put_snake(out_name, src_pfx):
        a = sd[src_pfx + ".alpha"].float().numpy()
        b = sd[src_pfx + ".beta"].float().numpy()
        # alpha_logscale=True: runtime alpha=exp(a), beta=exp(b); fold exp here.
        alpha = np.exp(a).astype(np.float32)
        beta_inv = (1.0 / (np.exp(b) + 1e-9)).astype(np.float32)
        w.add_tensor(out_name + ".alpha", alpha)
        w.add_tensor(out_name + ".beta_inv", beta_inv)
        names.extend([out_name + ".alpha", out_name + ".beta_inv"])

    # layers.0 : input conv 64->2048 k=7
    put_conv("conv_in", "layers.0", has_bias=True)

    # layers.1..5 : DecoderBlock = [0]=snake [1]=convT [2,3,4]=ResUnit(dil 1,3,9)
    for bi in range(5):
        L = bi + 1
        bp = f"layers.{L}"
        op = f"block.{bi}"
        put_snake(op + ".snake", bp + ".layers.0")
        put_conv(op + ".conv_t", bp + ".layers.1", has_bias=True)   # ConvTranspose1d
        for ri in range(3):
            rp = f"{bp}.layers.{ri + 2}.layers"   # ResidualUnit.layers
            rop = f"{op}.res.{ri}"
            put_snake(rop + ".snake1", rp + ".0")
            put_conv(rop + ".conv1", rp + ".1", has_bias=True)   # k=7 dilated
            put_snake(rop + ".snake2", rp + ".2")
            put_conv(rop + ".conv2", rp + ".3", has_bias=True)   # k=1

    # layers.6 : final snake ; layers.7 : output conv 128->2 k=7 (no bias)
    put_snake("snake_out", "layers.6")
    put_conv("conv_out", "layers.7", has_bias=False)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()

    print(f"[vae-gguf] {len(names)} tensors -> {OUT}", file=sys.stderr)
    for n in names:
        print("  ", n, file=sys.stderr)


if __name__ == "__main__":
    main()
