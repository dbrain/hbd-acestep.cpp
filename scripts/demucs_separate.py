#!/usr/bin/env python3
"""Single-file-in front-end for the SongGeneration clone path.

Separates a stereo mix into the two stems the C++ clone expects:
  vocal = htdemucs 'vocals' stem ; bgm = drums+bass+other (the encode-path bgm).
Outputs 48kHz stereo wavs (the clone's encode path resamples 48k->24k internally).

This is an EXTERNAL preprocessing tool (Meta's demucs, MIT) — the core SongGeneration
inference stays pure C++/ggml. A native ggml HTDemucs port is the pure-C++ alternative.

Run with the isolated demucs venv (keeps the ROCm sg-venv untouched):
  /tmp/demucs-venv/bin/python scripts/demucs_separate.py <mix.wav> <out_vocal.wav> <out_bgm.wav>
"""
import sys
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from demucs.pretrained import get_model
from demucs.apply import apply_model

MIX, OUT_VOCAL, OUT_BGM = sys.argv[1], sys.argv[2], sys.argv[3]

# load mix -> [2, T] @ its sr, make stereo
wav, sr = sf.read(MIX, dtype="float32", always_2d=True)  # [T, C]
x = torch.from_numpy(wav.T)                                # [C, T]
if x.shape[0] == 1:
    x = x.repeat(2, 1)
elif x.shape[0] > 2:
    x = x[:2]

model = get_model("htdemucs").cpu().eval()                 # sources [drums,bass,other,vocals] @ 44100
msr = model.samplerate
if sr != msr:
    x = AF.resample(x, sr, msr)

with torch.no_grad():
    # apply_model wants [B, C, T]; ref normalizes by mix mean/std then restores
    ref = x.mean(0)
    xn = (x - ref.mean()) / (ref.std() + 1e-8)
    stems = apply_model(model, xn[None], split=True, overlap=0.25, shifts=1, progress=False)[0]
    stems = stems * ref.std() + ref.mean()                 # [4, 2, T] drums,bass,other,vocals

names = model.sources
vocal = stems[names.index("vocals")]                       # [2, T]
bgm = sum(stems[names.index(n)] for n in ("drums", "bass", "other"))

# -> 48k stereo for the clone encode path
def to48k(t):
    return AF.resample(t, msr, 48000).cpu().numpy().astype(np.float32)

sf.write(OUT_VOCAL, to48k(vocal).T, 48000)
sf.write(OUT_BGM, to48k(bgm).T, 48000)
print(f"[demucs] {MIX} ({sr}Hz) -> {OUT_VOCAL} + {OUT_BGM} (48k stereo); model_sr={msr}")
