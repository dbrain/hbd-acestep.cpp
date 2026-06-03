#!/usr/bin/env python3
"""
Slice the SongGeneration Stable-Audio Oobleck VAE ENCODER -> a single BF16 GGUF.

Mirror of convert_vae.py (decoder). Folds torch weight_norm (weight_g/weight_v)
into plain conv weights and pre-computes SnakeBeta gains (alpha=exp(a),
beta_inv=1/exp(b), alpha_logscale=True). Conv weights BF16; snake gains + biases F32.

OobleckEncoder (prefix encoder.):
  layers.0          : conv_in WNConv1d(2->128, k7, p3)
  layers.1..5       : EncoderBlock i; res units run at in_dim, strided downsample in->out:
                        layers.0,1,2 : 3 ResidualUnit (snake1, conv1 k7 dil{1,3,9}, snake2, conv2 k1)
                        layers.3     : snake (in_dim)
                        layers.4     : strided WNConv1d(in->out, k=2*stride, p=ceil(stride/2))
  layers.6          : final snake (2048)
  layers.7          : conv_out WNConv1d(2048->128, k3, p1)   (128 = 2*latent_dim)

  /tmp/sg-venv/bin/python scripts/convert_vae_encoder.py <ckpt> <out.gguf>
    ckpt : autoencoder_music_1320k.ckpt  (torch.load(...)["state_dict"], prefix encoder.)
"""
import sys
import numpy as np
import torch
import gguf

CKPT, OUT = sys.argv[1], sys.argv[2]
ARCH = "songgen-vae-enc"
BF16 = gguf.GGMLQuantizationType.BF16

STRIDES = [2, 4, 4, 6, 10]               # fine -> coarse (encoder order)
# block in_dim (res units / pre-downsample) and out_dim (post-downsample)
IN_CH   = [128, 128, 256, 512, 1024]
OUT_CH  = [128, 256, 512, 1024, 2048]


def to_bf16_u16(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    return (a.view(np.uint32) >> 16).astype(np.uint16)


def fold_wn(sd, pfx):
    # weight_norm dim=0: w = g * v / ||v||, L2 norm over all non-0 dims per out channel.
    g = sd[pfx + ".weight_g"].float().numpy()      # [out,1,1]
    v = sd[pfx + ".weight_v"].float().numpy()      # [out,*,*]
    d0 = v.shape[0]
    vf = v.reshape(d0, -1)
    norm = np.sqrt((vf * vf).sum(axis=1))          # [out]
    scale = (g.reshape(d0) / norm).reshape(d0, *([1] * (v.ndim - 1)))
    return v * scale                               # same shape as weight_v


def main():
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}

    w = gguf.GGUFWriter(OUT, ARCH, use_temp_file=True)
    w.add_name("SongGeneration-StableAudio-Oobleck-VAE-encoder")
    w.add_uint32("songgen-vae-enc.in_channels", 2)
    w.add_uint32("songgen-vae-enc.channels", 128)
    w.add_uint32("songgen-vae-enc.latent_dim", 64)
    w.add_array("songgen-vae-enc.strides", STRIDES)
    w.add_array("songgen-vae-enc.in_channels_blocks", IN_CH)
    w.add_array("songgen-vae-enc.out_channels_blocks", OUT_CH)
    w.add_uint32("songgen-vae-enc.n_blocks", 5)
    w.add_uint32("songgen-vae-enc.downsampling_ratio", 1920)
    w.add_uint32("songgen-vae-enc.sample_rate", 48000)
    w.add_bool("songgen-vae-enc.use_snake", True)

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
        alpha = np.exp(a).astype(np.float32)
        beta_inv = (1.0 / (np.exp(b) + 1e-9)).astype(np.float32)
        w.add_tensor(out_name + ".alpha", alpha)
        w.add_tensor(out_name + ".beta_inv", beta_inv)
        names.extend([out_name + ".alpha", out_name + ".beta_inv"])

    # layers.0 : conv_in 2->128 k=7
    put_conv("conv_in", "layers.0", has_bias=True)

    # layers.1..5 : EncoderBlock = [0,1,2]=ResUnit(dil 1,3,9) [3]=snake [4]=strided conv
    for bi in range(5):
        L = bi + 1
        bp = f"layers.{L}"
        op = f"block.{bi}"
        for ri in range(3):
            rp = f"{bp}.layers.{ri}.layers"   # ResidualUnit.layers
            rop = f"{op}.res.{ri}"
            put_snake(rop + ".snake1", rp + ".0")
            put_conv(rop + ".conv1", rp + ".1", has_bias=True)   # k=7 dilated
            put_snake(rop + ".snake2", rp + ".2")
            put_conv(rop + ".conv2", rp + ".3", has_bias=True)   # k=1
        put_snake(op + ".snake", bp + ".layers.3")
        put_conv(op + ".conv_d", bp + ".layers.4", has_bias=True)  # strided downsample

    # layers.6 : final snake ; layers.7 : output conv 2048->128 k=3
    put_snake("snake_out", "layers.6")
    put_conv("conv_out", "layers.7", has_bias=True)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()

    print(f"[vae-enc-gguf] {len(names)} tensors -> {OUT}", file=sys.stderr)
    for n in names:
        print("  ", n, file=sys.stderr)


if __name__ == "__main__":
    main()
