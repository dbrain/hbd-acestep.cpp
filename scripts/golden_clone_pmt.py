"""
Clone-path golden #4b: the 1rvq 'pmt' prompt-token stream (3rd channel of the LM's
audio_qt_embs [1,3,T]). Same bestrq+single-RVQ machinery as the separate tokenizer,
but a single (non-separated) stem and DEFAULT layer 6 (generate_1rvq layer_num=6).

model_1rvq.PromptCondAudioDiffusion.fetch_codes_batch(input_audios[N,2,min], layer=6)
  -> extract_bestrq_embeds(chan0,chan1,layer=6) -> rvq_bestrq_emb -> codes [B,1,T]
Weights: model_1rvq/model_2_fixed.safetensors (bestrq.* + rvq_bestrq_emb.*).

REFERENCE AUDIO CAVEAT: first.wav (48k stereo) is the single stem (no demucs).

Run from /tmp/songgen-src:
  PYTHONPATH=codeclm/tokenizer:.:codeclm/tokenizer/Flow1dVAE:codeclm/tokenizer/Flow1dVAE/our_MERT_BESTRQ/mert_fairseq/models/musicfm:tools/gradio \
  CUDA_VISIBLE_DEVICES="" /tmp/sg-venv/bin/python \
  /home/dbrain/dev/songgen-port/scripts/golden_clone_pmt.py
"""
import os, sys, json, types, math
import numpy as np
import torch
import soundfile as sf
import sg_compat
sys.modules.setdefault("k_diffusion", types.ModuleType("k_diffusion"))


def _mel_fb(sr, n_fft, n_mels):
    n_freqs = n_fft // 2 + 1; f_max = sr / 2.0
    all_freqs = torch.linspace(0, f_max, n_freqs)
    def h2m(f): return 2595.0 * math.log10(1.0 + f / 700.0)
    m_pts = torch.linspace(h2m(0.0), h2m(f_max), n_mels + 2)
    f_pts = 700.0 * (10.0 ** (m_pts / 2595.0) - 1.0)
    fdiff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
    return torch.clamp(torch.minimum(-slopes[:, :-2] / fdiff[:-1], slopes[:, 2:] / fdiff[1:]), min=0.0)

def amplitude_to_db(spec, top_db=80.0):
    db = 10.0 * torch.log10(torch.clamp(spec, min=1e-10))
    return torch.maximum(db, db.amax(dim=(-2, -1), keepdim=True) - top_db)

class RealMelSTFT(torch.nn.Module):
    def __init__(self, fb, win, n_fft=2048, hop=240):
        super().__init__(); self.n_fft, self.hop = n_fft, hop
        self.register_buffer("win", win); self.register_buffer("fb", fb)
    def forward(self, wav):
        spec = torch.stft(wav, n_fft=self.n_fft, hop_length=self.hop, win_length=self.n_fft,
                          window=self.win.to(wav.dtype), center=True, pad_mode="reflect",
                          return_complex=True, normalized=False)
        power = spec.real.pow(2) + spec.imag.pow(2)
        return torch.matmul(power.transpose(1, 2), self.fb).transpose(1, 2)

def resample_48k_to_24k(wav):
    import julius; return julius.resample_frac(wav, 48000, 24000)

from models_gpt.models.gpt2_config import GPT2Config as _GC
for _a, _v in dict(add_cross_attention=False, classifier_dropout=None, hidden_dropout=None,
                   pad_token_id=None, scale_attn_by_inverse_layer_idx=False,
                   reorder_and_upcast_attn=False).items():
    if not hasattr(_GC, _a):
        setattr(_GC, _a, _v)

from model_1rvq import PromptCondAudioDiffusion
from safetensors.torch import load_file

OUT = "/home/dbrain/dev/songgen-port/golden-large/clone"
CKPT = "/home/dbrain/dev/songgen-port/ckpt-audio/model_1rvq/model_2_fixed.safetensors"
WAV = "/home/dbrain/dev/songgen-port/out/first.wav"
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)
LAYER = 6

main_config = dict(num_channels=32, unet_model_name=None,
                   unet_model_config_path="configs/models/transformer2D_wocross_inch112_1x4_multi_large.json",
                   snr_gamma=None)
model = PromptCondAudioDiffusion(**main_config).eval()

sd_all = load_file(CKPT)
fb = sd_all["bestrq.model.preprocessor_melspec_2048.mel_stft.mel_scale.fb"].float()
win = sd_all["bestrq.model.preprocessor_melspec_2048.mel_stft.spectrogram.window"].float()
real = RealMelSTFT(fb, win).eval()
model.bestrq.model.preprocessor_melspec_2048.forward = lambda w, _r=real: amplitude_to_db(_r(w))
del model.rsq48tobestrq
object.__setattr__(model, "rsq48tobestrq", resample_48k_to_24k)

missing, _ = model.load_state_dict({k: v.float() for k, v in sd_all.items()}, strict=False)
brq_rvq_missing = [k for k in missing if k.startswith(("bestrq.", "rvq_bestrq"))]
print(f"[pmt] bestrq/rvq missing={brq_rvq_missing}")
assert not brq_rvq_missing

wav, sr = sf.read(WAV, dtype="float32", always_2d=True); assert sr == 48000
wav = torch.from_numpy(wav.T)
if wav.shape[0] == 1:
    wav = wav.repeat(2, 1)

# sound2code segmentation (single stem)
audios = model.preprocess_audio(wav)  # [2,L]
orig_length = audios.shape[-1]
output_len = int(orig_length / 48000.0 * 25) + 1
min_samples = int(40 * 48000)
while audios.shape[-1] < min_samples:
    audios = torch.cat([audios, audios], -1)
int_max_len = audios.shape[-1] // min_samples + 1
audios = torch.cat([audios, audios], -1)[:, : int_max_len * min_samples]
audio_input = audios.reshape(2, -1, min_samples).permute(1, 0, 2).reshape(-1, 2, min_samples)

with torch.no_grad():
    codes, _, _ = model.fetch_codes_batch(audio_input, additional_feats=[], layer=LAYER)
codes = torch.cat(codes, 1)  # [B,1,T]
codes_full = codes.permute(1, 0, 2).reshape(1, -1)[None]
codes_out = codes_full[:, :, :output_len]

cp = codes_out.cpu().numpy().astype(np.int64)
np.save(f"{OUT}/tokens_pmt.npy", cp)

meta = dict(
    task="clone_pmt_token_1rvq",
    module="model_1rvq.PromptCondAudioDiffusion.fetch_codes_batch",
    path="preprocess_audio -> extract_bestrq_embeds(layer=6, single stem) -> rvq_bestrq_emb -> codes",
    weights=dict(ckpt=CKPT, loaded="full strict=False"),
    layer=LAYER, codebook_size=16384,
    reference_audio=dict(path=WAV, note="single stem = first.wav (no demucs)"),
    shape=list(cp.shape), code_range=[int(cp.min()), int(cp.max())],
    relation_to_LM="this is the 'pmt' stream = channel 0 of the LM clone audio_qt_embs[1,3,T]; "
                   "channels 1,2 are the separate-tokenizer vocal/bgm streams (tokens_vocal/tokens_bgm).",
    note="same bestrq+RVQ math as the separate path; differs only in layer (6) and single-stem input.",
)
with open(f"{OUT}/pmt_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"[pmt] tokens_pmt{cp.shape} range[{cp.min()},{cp.max()}] saved to {OUT}")
