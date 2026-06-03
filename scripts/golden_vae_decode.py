"""
Task 1: deterministic VAE decode golden for the SongGeneration audio stage.

Loads the Stable Audio Oobleck decoder (stable-audio-tools) + weights from
autoencoder_music_1320k.ckpt, feeds a fixed post-bottleneck latent z[1,64,32]
through DECODER ONLY, and saves the latent + wav as .npy goldens.

Run:
  PYTHONPATH=codeclm/tokenizer:.:codeclm/tokenizer/Flow1dVAE:tools/gradio \
  /tmp/sg-venv/bin/python /home/dbrain/dev/songgen-port/scripts/golden_vae_decode.py
(run from /tmp/songgen-src)
"""
import os, sys, json, types
import numpy as np
import torch
import sg_compat  # torchaudio/transformers env shims (import-time only)

# stable_audio_tools.models.autoencoders -> inference.sampling -> k_diffusion -> torchvision.
# k_diffusion is only used by the sampling helper (unused on the decode path); stub it.
sys.modules.setdefault("k_diffusion", types.ModuleType("k_diffusion"))

from stable_audio_tools.models.autoencoders import OobleckDecoder

OUT = "/home/dbrain/dev/songgen-port/golden-large/audio"
CKPT = "/home/dbrain/dev/songgen-port/ckpt-audio/vae/autoencoder_music_1320k.ckpt"
os.makedirs(OUT, exist_ok=True)

DEC_CFG = dict(out_channels=2, channels=128, latent_dim=64,
               c_mults=[1, 2, 4, 8, 16], strides=[2, 4, 4, 6, 10],
               use_snake=True, final_tanh=False)

dec = OobleckDecoder(**DEC_CFG).eval()

sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
PREFIX = "decoder."
dec_sd = {k[len(PREFIX):]: v for k, v in sd.items() if k.startswith(PREFIX)}
missing, unexpected = dec.load_state_dict(dec_sd, strict=False)
print(f"[vae] state_dict prefix='{PREFIX}'  loaded={len(dec_sd)} keys")
print(f"[vae] missing={list(missing)}  unexpected={list(unexpected)}")
assert not missing and not unexpected, "decoder weight mismatch"

torch.manual_seed(0)
z = torch.randn(1, 64, 32)  # post-bottleneck latent, 64 ch, 32 frames

with torch.no_grad():
    wav = dec(z)  # OobleckDecoder.forward -> [1, 2, T]

z_np = z.cpu().numpy().astype(np.float32)
wav_np = wav.cpu().numpy().astype(np.float32)
np.save(f"{OUT}/vae_latent.npy", z_np)
np.save(f"{OUT}/vae_wav.npy", wav_np)

meta = dict(
    task="vae_decode_golden",
    module="stable_audio_tools.models.autoencoders.OobleckDecoder",
    method="forward (decoder only, no bottleneck/encoder)",
    call="dec(z) where z=torch.randn(1,64,32) under torch.manual_seed(0)",
    weights=dict(ckpt=CKPT, state_dict_prefix=PREFIX,
                 keys_loaded=len(dec_sd), weight_norm="weight_g/weight_v (torch.nn.utils.weight_norm)"),
    decoder_config=DEC_CFG,
    latent_shape=list(z_np.shape), wav_shape=list(wav_np.shape),
    downsampling_ratio=1920, sample_rate=48000, audio_channels=2,
    note="latent is POST-bottleneck (vae bottleneck already sampled to 64ch). "
         "T = 32 * 1920 / (encoder downsample already applied); decoder upsamples by prod(strides)=1920.",
    wav_stats=dict(min=float(wav_np.min()), max=float(wav_np.max()),
                   mean=float(wav_np.mean()), std=float(wav_np.std())),
)
with open(f"{OUT}/vae_decode_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"[vae] z{z_np.shape} -> wav{wav_np.shape}  "
      f"min={wav_np.min():.4f} max={wav_np.max():.4f} std={wav_np.std():.5f}")
print(f"[vae] saved {OUT}/vae_latent.npy, vae_wav.npy, vae_decode_meta.json")
