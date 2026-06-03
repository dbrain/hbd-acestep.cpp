"""
Clone-path golden #2: Stable-Audio Oobleck VAE ENCODE.

generate_septoken.code2sound does: true_latent = self.vae.encode_audio(prompt_vocal+prompt_bgm).permute(0,2,1)
encode_audio (no chunk) -> AudioAutoencoder.encode -> OobleckEncoder (-> 128ch) ->
VAEBottleneck.encode: mean,scale = latents.chunk(2,dim=1) (64 each);
  stdev = softplus(scale)+1e-4; latent = randn_like(mean)*stdev + mean   (STOCHASTIC sample).
DOWNSTREAM USES THE SAMPLE (not the mean) — code2sound feeds encode_audio() output directly.

We capture: pre_bottleneck_latents[1,128,T], mean[1,64,T], scale[1,64,T], stdev[1,64,T],
and the sampled latent[1,64,T] under a fixed seed (reproducible), PLUS the mean (== deterministic
latent if a port chooses mean). The port should reproduce the sampler with the SAME RNG, or use mean.

REFERENCE AUDIO CAVEAT: encode_audio receives (prompt_vocal+prompt_bgm). With no demucs we feed
first.wav (48k stereo) as both -> input = 2*first.wav. Exercises the encoder graph deterministically.

Run from /tmp/songgen-src:
  PYTHONPATH=codeclm/tokenizer:.:codeclm/tokenizer/Flow1dVAE:tools/gradio \
  CUDA_VISIBLE_DEVICES="" /tmp/sg-venv/bin/python \
  /home/dbrain/dev/songgen-port/scripts/golden_clone_vae_encode.py
"""
import os, sys, json, types
import numpy as np
import torch
import soundfile as sf
import sg_compat
sys.modules.setdefault("k_diffusion", types.ModuleType("k_diffusion"))

from stable_audio_tools.models.autoencoders import create_autoencoder_from_config
from stable_audio_tools.models.bottleneck import vae_sample

OUT = "/home/dbrain/dev/songgen-port/golden-large/clone"
CKPT = "/home/dbrain/dev/songgen-port/ckpt-audio/vae/autoencoder_music_1320k.ckpt"
CFG = "/home/dbrain/dev/songgen-port/ckpt-audio/vae/stable_audio_1920_vae.json"
WAV = "/home/dbrain/dev/songgen-port/out/first.wav"
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)

with open(CFG) as f:
    model_config = json.load(f)
vae = create_autoencoder_from_config(model_config).eval()
sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
missing, unexpected = vae.load_state_dict(sd, strict=False)
# unexpected: discriminator / loss / EMA buffers not part of the inference graph
enc_missing = [k for k in missing if k.startswith("encoder.")]
print(f"[vae-enc] loaded; encoder missing={enc_missing}  total_unexpected={len(unexpected)}")
assert not enc_missing, "encoder weights incomplete"

# ---- input: first.wav as both stems -> prompt_vocal+prompt_bgm = 2*first.wav ----
wav, sr = sf.read(WAV, dtype="float32", always_2d=True)
assert sr == 48000, sr
wav = torch.from_numpy(wav.T)  # [C, L]
if wav.shape[0] == 1:
    wav = wav.repeat(2, 1)
# code2sound clips prompt to 10s; mirror that so T is the real clone-path length
wav = wav[:, : int(10 * 48000)]
audio = (wav + wav).unsqueeze(0)  # [1, 2, L]  (vocal+bgm, both = first.wav)
# pad to multiple of downsampling_ratio (1920) as the real path does
ds = int(model_config["model"]["downsampling_ratio"])
L = audio.shape[-1]
pad = (-L) % ds
if pad:
    audio = torch.nn.functional.pad(audio, (0, pad))

with torch.no_grad():
    pre_latents = vae.encoder(audio)  # [1,128,T]
    mean, scale = pre_latents.chunk(2, dim=1)  # 64 each
    stdev = torch.nn.functional.softplus(scale) + 1e-4
    torch.manual_seed(0)  # fix the sample RNG for reproducibility
    sampled, kl = vae_sample(mean, scale)  # randn_like(mean)*stdev + mean
    # cross-check: the high-level encode_audio() path should match `sampled` (same seed)
    torch.manual_seed(0)
    latent_via_encode = vae.encode_audio(audio)  # [1,64,T]

a_np = audio.cpu().numpy().astype(np.float32)
pl_np = pre_latents.cpu().numpy().astype(np.float32)
mean_np = mean.cpu().numpy().astype(np.float32)
scale_np = scale.cpu().numpy().astype(np.float32)
std_np = stdev.cpu().numpy().astype(np.float32)
samp_np = sampled.cpu().numpy().astype(np.float32)
enc_np = latent_via_encode.cpu().numpy().astype(np.float32)
np.save(f"{OUT}/vaeenc_in_audio48k.npy", a_np)
np.save(f"{OUT}/vaeenc_pre_bottleneck_latents.npy", pl_np)
np.save(f"{OUT}/vaeenc_out_mean.npy", mean_np)
np.save(f"{OUT}/vaeenc_out_scale.npy", scale_np)
np.save(f"{OUT}/vaeenc_out_stdev.npy", std_np)
np.save(f"{OUT}/vaeenc_out_sampled.npy", samp_np)
np.save(f"{OUT}/vaeenc_out_encode_audio.npy", enc_np)

meta = dict(
    task="clone_vae_encode",
    module="stable_audio_tools.models.autoencoders.AudioAutoencoder (built via create_autoencoder_from_config)",
    method="vae.encoder(audio) [OobleckEncoder, 128ch] -> VAEBottleneck.encode: chunk to mean/scale(64), "
            "stdev=softplus(scale)+1e-4, sample=randn_like(mean)*stdev+mean",
    weights=dict(ckpt=CKPT, config=CFG, encoder_prefix="encoder.",
                 note="full AudioAutoencoder loaded strict=False; discriminator/loss/EMA tensors are unexpected and unused"),
    encoder_config=model_config["model"]["encoder"]["config"],
    bottleneck=dict(type="vae",
        downstream_uses="SAMPLE (encode_audio() output) — code2sound feeds it directly; "
                        "mean is also saved for a deterministic-port option",
        sampler="latent = randn_like(mean)*(softplus(scale)+1e-4) + mean; RNG fixed via torch.manual_seed(0)",
        encoder_out_channels=128, latent_dim=64),
    reference_audio=dict(path=WAV, sr=48000, channels=2,
        note="encode_audio input = prompt_vocal+prompt_bgm; no demucs so both = first.wav -> input = 2*first.wav, clipped to 10s, padded to multiple of 1920"),
    downsampling_ratio=ds, sample_rate=48000,
    encoder_structure="OobleckEncoder = mirror of OobleckDecoder: conv_in -> N EncoderBlock "
                       "(each: 3 ResidualUnit(snake+WNConv1d dilations 1,3,9) then snake+strided WNConv1d) "
                       "with strides [2,4,4,6,10], c_mults [1,2,4,8,16], then snake+WNConv1d to 2*latent_dim(128). "
                       "Decoder mirrors with transposed convs; same snake/weight_norm. Encoder downsamples by prod(strides)=1920.",
    shapes=dict(in_audio48k=list(a_np.shape), pre_bottleneck=list(pl_np.shape),
                mean=list(mean_np.shape), scale=list(scale_np.shape),
                sampled=list(samp_np.shape), encode_audio=list(enc_np.shape)),
    checks=dict(sampled_vs_encode_audio_maxabsdiff=float(np.abs(samp_np - enc_np).max()),
                kl=float(kl.item())),
    stats=dict(mean_std=float(mean_np.std()), scale_mean=float(scale_np.mean()),
               sampled_std=float(samp_np.std())),
)
with open(f"{OUT}/vaeenc_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"[vae-enc] audio{a_np.shape} -> pre{pl_np.shape} mean{mean_np.shape} sampled{samp_np.shape}")
print(f"[vae-enc] sampled vs encode_audio maxabsdiff={np.abs(samp_np-enc_np).max():.2e}  kl={kl.item():.3f}")
print(f"[vae-enc] saved to {OUT}")
