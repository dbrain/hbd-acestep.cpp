#!/usr/bin/env python3
# Reference dumper for the HTDemucs C++/ggml port. Runs the real demucs model on the
# captured input_seg and dumps EVERY sub-block intermediate (with the exact ggml-friendly
# layout) so the C++ port can be validated block-by-block.
#
#   /tmp/demucs-venv/bin/python scripts/dump_htdemucs_ref.py <ckpt_dir> <intermediates_dir> <out_dir>
#
# All dumps are float32 .npy in the natural torch shape; the C++ test reads them and
# compares cossim. Single-segment forward (use_train_segment already pads input_seg to
# 343980 = 7.8*44100).
import sys, json, math
import numpy as np
import torch
import torch.nn.functional as F
from demucs.htdemucs import HTDemucs

CKPT_DIR, INTER_DIR, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
import os
os.makedirs(OUT, exist_ok=True)


def save(name, t):
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    np.save(os.path.join(OUT, name + ".npy"), np.ascontiguousarray(t.astype(np.float32)))
    print(f"  {name} {tuple(t.shape)}")


cfg = json.load(open(os.path.join(CKPT_DIR, "htdemucs_config.json")))
sd = torch.load(os.path.join(CKPT_DIR, "htdemucs_state_dict.pth"), map_location="cpu", weights_only=False)
m = HTDemucs(sources=cfg["sources"], audio_channels=2, channels=cfg["channels"], nfft=cfg["nfft"],
             depth=cfg["depth"], bottom_channels=cfg["bottom_channels"], cac=cfg["cac"],
             samplerate=cfg["samplerate"], segment=7.8)
m.load_state_dict(sd, strict=False)
m.eval()

inp = np.load(os.path.join(INTER_DIR, "input_seg.npy"))  # [1,2,343980]
mix = torch.from_numpy(inp).float()
length = mix.shape[-1]

with torch.no_grad():
    # ---- STFT / CAC ----
    z = m._spec(mix)                      # complex [1,2,2048,336]
    save("z_real", z.real)
    save("z_imag", z.imag)
    mag = m._magnitude(z)                 # [1,4,2048,336]
    save("mag", mag)
    x = mag
    B, C, Fq, T = x.shape
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std = x.std(dim=(1, 2, 3), keepdim=True)
    save("x_norm_stats", torch.cat([mean.flatten(), std.flatten()]))
    x = (x - mean) / (1e-5 + std)
    save("x_norm", x)

    xt = mix
    meant = xt.mean(dim=(1, 2), keepdim=True)
    stdt = xt.std(dim=(1, 2), keepdim=True)
    save("xt_norm_stats", torch.cat([meant.flatten(), stdt.flatten()]))
    xt = (xt - meant) / (1e-5 + stdt)

    # ---- encoders ----
    saved, saved_t, lengths, lengths_t = [], [], [], []
    for idx, encode in enumerate(m.encoder):
        lengths.append(x.shape[-1])
        inject = None
        if idx < len(m.tencoder):
            lengths_t.append(xt.shape[-1])
            tenc = m.tencoder[idx]
            xt = tenc(xt)
            if not tenc.empty:
                saved_t.append(xt)
                save(f"tenc{idx}", xt)
            else:
                inject = xt
                save(f"tenc{idx}_inject", xt)
        x = encode(x, inject)
        if idx == 0 and m.freq_emb is not None:
            frs = torch.arange(x.shape[-2])
            emb = m.freq_emb(frs).t()[None, :, :, None].expand_as(x)
            x = x + m.freq_emb_scale * emb
        saved.append(x)
        save(f"enc{idx}", x)

    # ---- bottleneck transformer ----
    b, c, f, t = x.shape
    from einops import rearrange
    xu = rearrange(x, "b c f t-> b c (f t)")
    xu = m.channel_upsampler(xu)
    xu = rearrange(xu, "b c (f t)-> b c f t", f=f)
    xtu = m.channel_upsampler_t(xt)
    save("ct_in_x", xu)
    save("ct_in_xt", xtu)

    # per-layer dump inside the transformer
    ct = m.crosstransformer
    Bc, Cc, Fr, T1 = xu.shape
    from demucs.transformer import create_2d_sin_embedding
    pe2d = create_2d_sin_embedding(Cc, Fr, T1, xu.device, ct.max_period)
    pe2d = rearrange(pe2d, "b c fr t1 -> b (t1 fr) c")
    xs = rearrange(xu, "b c fr t1 -> b (t1 fr) c")
    xs = ct.norm_in(xs)
    xs = xs + ct.weight_pos_embed * pe2d
    save("ct_x_after_normin", rearrange(xs, "b (t1 fr) c -> b c fr t1", t1=T1))

    Bt, Ct2, T2 = xtu.shape
    xtt = rearrange(xtu, "b c t2 -> b t2 c")
    pe = ct._get_pos_embedding(T2, Bt, Ct2, xtu.device)
    pe = rearrange(pe, "t2 b c -> b t2 c")
    xtt = ct.norm_in_t(xtt)
    xtt = xtt + ct.weight_pos_embed * pe
    save("ct_xt_after_normin", rearrange(xtt, "b t2 c -> b c t2"))

    for idx in range(ct.num_layers):
        if idx % 2 == ct.classic_parity:
            xs = ct.layers[idx](xs)
            xtt = ct.layers_t[idx](xtt)
        else:
            old = xs
            xs = ct.layers[idx](xs, xtt)
            xtt = ct.layers_t[idx](xtt, old)
        save(f"ct_x_layer{idx}", rearrange(xs, "b (t1 fr) c -> b c fr t1", t1=T1))
        save(f"ct_xt_layer{idx}", rearrange(xtt, "b t2 c -> b c t2"))

    xs = rearrange(xs, "b (t1 fr) c -> b c fr t1", t1=T1)
    xtt = rearrange(xtt, "b t2 c -> b c t2")

    # downsample back
    xd = rearrange(xs, "b c f t-> b c (f t)")
    xd = m.channel_downsampler(xd)
    xd = rearrange(xd, "b c (f t)-> b c f t", f=f)
    xtd = m.channel_downsampler_t(xtt)
    save("ct_out_x", xd)
    save("ct_out_xt", xtd)

    x = xd
    xt = xtd
    # ---- decoders ----
    for idx, decode in enumerate(m.decoder):
        skip = saved.pop(-1)
        x, pre = decode(x, skip, lengths.pop(-1))
        save(f"dec{idx}", x)
        offset = m.depth - len(m.tdecoder)
        if idx >= offset:
            tdec = m.tdecoder[idx - offset]
            length_t = lengths_t.pop(-1)
            if tdec.empty:
                pre = pre[:, :, 0]
                xt, _ = tdec(pre, None, length_t)
            else:
                sk = saved_t.pop(-1)
                xt, _ = tdec(xt, sk, length_t)
            save(f"tdec{idx}", xt)

    S = len(m.sources)
    x = x.view(B, S, -1, Fq, T)
    x = x * std[:, None] + mean[:, None]
    save("x_spec_pre_istft", x)
    zout = m._mask(z, x)
    training_length = int(m.segment * m.samplerate)
    xspec = m._ispec(zout, training_length)
    save("x_spec_wave", xspec)
    xt = xt.view(B, S, -1, training_length)
    xt = xt * stdt[:, None] + meant[:, None]
    save("x_time_wave", xt)
    out = xt + xspec
    save("final_out", out)

print("done ->", OUT)
