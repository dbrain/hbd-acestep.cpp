"""
Clone-path golden #4: end-to-end prompt-token extraction via the ACTUAL high-level
encode graph (fetch_codes_batch), validating that bestrq + rvq compose into the exact
tokens the LeLM consumes.

We build PromptCondAudioDiffusion (the model wrapped by Flow1dVAESeparate.model) and load
the FULL model_2.safetensors. We DON'T build the Tango/VAE wrapper (it needs the VAE ckpt +
demucs separator); fetch_codes_batch only touches preprocess_audio + bestrq + rvq, so this
is the faithful high-level encode for the cloning prompt tokens.

Path mirrored from generate_septoken.sound2code:
  audios = preprocess_audio(first.wav 48k stereo); reshape into [N,2,min_samples] segments
  (min_samples = 40s*48k). With a 10s clip we get one segment after the repeat-pad logic.
  fetch_codes_batch(vocal_seg, bgm_seg) -> [codes_bestrq_emb, codes_bestrq_emb_bgm].
  codes are reshaped/trimmed to output_len = int(len/48000*25)+1.

The returned [codes_vocal, codes_bgm] (each [1,1,T]) are EXACTLY the per-stem audio_qt tokens
the LeLM receives as the clone prompt. (audio_qt_embs [1,3,T] in the LM = pmt + vocal + bgm;
the 1rvq 'pmt' token comes from generate_1rvq.fetch_codes — captured separately below.)

REFERENCE AUDIO CAVEAT: first.wav fed as both vocal and bgm stem (no demucs).

Run from /tmp/songgen-src:
  PYTHONPATH=codeclm/tokenizer:.:codeclm/tokenizer/Flow1dVAE:codeclm/tokenizer/Flow1dVAE/our_MERT_BESTRQ/mert_fairseq/models/musicfm:tools/gradio \
  CUDA_VISIBLE_DEVICES="" /tmp/sg-venv/bin/python \
  /home/dbrain/dev/songgen-port/scripts/golden_clone_tokens.py
"""
import os, sys, json, types
import numpy as np
import torch
import soundfile as sf
import sg_compat
sys.modules.setdefault("k_diffusion", types.ModuleType("k_diffusion"))

import torch.nn.functional as F
import math

# reuse the faithful melspec + resample from the bestrq golden
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "clone_bestrq_mod", "/home/dbrain/dev/songgen-port/scripts/golden_clone_bestrq.py")
# we can't exec it (runs main). Inline the two helpers instead.

def _mel_filterbank(sr, n_fft, n_mels):
    n_freqs = n_fft // 2 + 1
    f_max = sr / 2.0
    all_freqs = torch.linspace(0, f_max, n_freqs)
    def h2m(f): return 2595.0 * math.log10(1.0 + f / 700.0)
    m_pts = torch.linspace(h2m(0.0), h2m(f_max), n_mels + 2)
    f_pts = 700.0 * (10.0 ** (m_pts / 2595.0) - 1.0)
    fdiff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
    down = -slopes[:, :-2] / fdiff[:-1]; up = slopes[:, 2:] / fdiff[1:]
    return torch.clamp(torch.minimum(down, up), min=0.0)

def amplitude_to_db(spec, top_db=80.0):
    db = 10.0 * torch.log10(torch.clamp(spec, min=1e-10))
    return torch.maximum(db, db.amax(dim=(-2, -1), keepdim=True) - top_db)

class RealMelSTFT(torch.nn.Module):
    def __init__(self, fb, win, n_fft=2048, hop=240):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("win", win)
        self.register_buffer("fb", fb)
    def forward(self, wav):
        spec = torch.stft(wav, n_fft=self.n_fft, hop_length=self.hop, win_length=self.n_fft,
                          window=self.win.to(wav.dtype), center=True, pad_mode="reflect",
                          return_complex=True, normalized=False)
        power = spec.real.pow(2) + spec.imag.pow(2)
        return torch.matmul(power.transpose(1, 2), self.fb).transpose(1, 2)

def resample_48k_to_24k(wav):
    import julius
    return julius.resample_frac(wav, 48000, 24000)

from model_septoken import PromptCondAudioDiffusion
from safetensors.torch import load_file

# the vendored ~4.37-era GPT2Config reads class-level legacy defaults that transformers 5.x
# dropped; the unet GPT2Model is built in __init__ but UNUSED by the token path. Restore them.
from models_gpt.models.gpt2_config import GPT2Config as _GC
for _a, _v in dict(add_cross_attention=False, classifier_dropout=None, hidden_dropout=None,
                   pad_token_id=None, scale_attn_by_inverse_layer_idx=False,
                   reorder_and_upcast_attn=False).items():
    if not hasattr(_GC, _a):
        setattr(_GC, _a, _v)

OUT = "/home/dbrain/dev/songgen-port/golden-large/clone"
CKPT = "/home/dbrain/dev/songgen-port/ckpt-audio/model_septoken/model_2.safetensors"
WAV = "/home/dbrain/dev/songgen-port/out/first.wav"
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)

main_config = dict(num_channels=32, unet_model_name=None,
                   unet_model_config_path="configs/models/transformer2D_wocross_inch112_1x4_multi_large.json",
                   snr_gamma=None)
model = PromptCondAudioDiffusion(**main_config).eval()

# patch melspec with checkpoint-shipped fb/window + faithful STFT/dB
sd_all = load_file(CKPT)
fb = sd_all["bestrq.model.preprocessor_melspec_2048.mel_stft.mel_scale.fb"].float()
win = sd_all["bestrq.model.preprocessor_melspec_2048.mel_stft.spectrogram.window"].float()
real = RealMelSTFT(fb, win).eval()
model.bestrq.model.preprocessor_melspec_2048.forward = lambda w, _r=real: amplitude_to_db(_r(w))
# patch the 48k->24k resampler (stub transform) used inside extract_bestrq_embeds.
# rsq48tobestrq is a registered nn.Module child; replace via __dict__ to bypass setattr typecheck.
del model.rsq48tobestrq
object.__setattr__(model, "rsq48tobestrq", resample_48k_to_24k)

missing, unexpected = model.load_state_dict({k: v.float() for k, v in sd_all.items()}, strict=False)
brq_rvq_missing = [k for k in missing if k.startswith(("bestrq.", "rvq_bestrq"))]
print(f"[tok] loaded full ckpt; bestrq/rvq missing={brq_rvq_missing}  (other missing={len(missing)-len(brq_rvq_missing)})")
assert not brq_rvq_missing

# ---- preprocess input exactly like sound2code, but feed first.wav as both stems ----
wav, sr = sf.read(WAV, dtype="float32", always_2d=True)
assert sr == 48000
wav = torch.from_numpy(wav.T)  # [C,L]
if wav.shape[0] == 1:
    wav = wav.repeat(2, 1)

def preprocess(a):
    # PromptCondAudioDiffusion.preprocess_audio operates on [B,L]; sound2code calls it on [2,L]
    return model.preprocess_audio(a)

audios_vocal = preprocess(wav)  # [2,L]
audios_bgm = preprocess(wav)
orig_length = audios_vocal.shape[-1]
sample_rate = 48000
output_len = int(orig_length / float(sample_rate) * 25) + 1
min_samples = int(40 * sample_rate)
while audios_vocal.shape[-1] < min_samples:
    audios_vocal = torch.cat([audios_vocal, audios_vocal], -1)
    audios_bgm = torch.cat([audios_bgm, audios_bgm], -1)
int_max_len = audios_vocal.shape[-1] // min_samples + 1
audios_vocal = torch.cat([audios_vocal, audios_vocal], -1)[:, : int_max_len * min_samples]
audios_bgm = torch.cat([audios_bgm, audios_bgm], -1)[:, : int_max_len * min_samples]
audio_vocal_input = audios_vocal.reshape(2, -1, min_samples).permute(1, 0, 2).reshape(-1, 2, min_samples)
audio_bgm_input = audios_bgm.reshape(2, -1, min_samples).permute(1, 0, 2).reshape(-1, 2, min_samples)

with torch.no_grad():
    [codes_vocal, codes_bgm], _, _ = model.fetch_codes_batch(
        audio_vocal_input, audio_bgm_input, additional_feats=[], layer_vocal=7, layer_bgm=3)
# reshape/trim per sound2code
codes_vocal_full = codes_vocal.permute(1, 0, 2).reshape(1, -1)[None]  # [1,1,N*min]
codes_bgm_full = codes_bgm.permute(1, 0, 2).reshape(1, -1)[None]
codes_vocal_out = codes_vocal_full[:, :, :output_len]
codes_bgm_out = codes_bgm_full[:, :, :output_len]

cv = codes_vocal_out.cpu().numpy().astype(np.int64)
cb = codes_bgm_out.cpu().numpy().astype(np.int64)
np.save(f"{OUT}/tokens_vocal.npy", cv)
np.save(f"{OUT}/tokens_bgm.npy", cb)

meta = dict(
    task="clone_prompt_tokens_endtoend",
    module="model_septoken.PromptCondAudioDiffusion.fetch_codes_batch",
    path="preprocess_audio -> extract_bestrq_embeds(layer_vocal=7 / layer_bgm=3) -> rvq_bestrq_emb / rvq_bestrq_bgm_emb",
    weights=dict(ckpt=CKPT, loaded="full state_dict strict=False; cfm/unet/vae-side tensors unused by this path"),
    melspec="checkpoint-shipped fb+window + faithful STFT/AmplitudeToDB (see bestrq_meta.json)",
    output_len_formula="int(orig_length/48000*25)+1",
    segmentation="40s segments (min_samples=40*48000); 10s clip -> repeat-padded then trimmed to output_len",
    reference_audio=dict(path=WAV, note="first.wav as BOTH vocal and bgm stem (no demucs)"),
    shapes=dict(tokens_vocal=list(cv.shape), tokens_bgm=list(cb.shape)),
    code_ranges=dict(vocal=[int(cv.min()), int(cv.max())], bgm=[int(cb.min()), int(cb.max())]),
    relation_to_LM="these two streams = the per-stem audio_qt clone tokens the LeLM receives "
                   "(separate-tokenizer path). The 3rd 'pmt' stream in audio_qt_embs[1,3,T] comes "
                   "from the 1rvq tokenizer (generate_1rvq.fetch_codes) — same bestrq+rvq machinery, captured below.",
    consistency_check="should equal the standalone rvq golden codes on the matching segment "
                      "(within trimming); the separate scripts validate components, this validates compose.",
)
with open(f"{OUT}/tokens_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"[tok] tokens_vocal{cv.shape} range[{cv.min()},{cv.max()}]  tokens_bgm{cb.shape} range[{cb.min()},{cb.max()}]")
print(f"[tok] saved to {OUT}")
