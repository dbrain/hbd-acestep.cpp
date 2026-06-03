"""
Clone-path golden #1: BESTRQ (MusicFM_95M) feature extraction.

extract_bestrq_embeds (model_septoken.py:350-356) does, per stem:
    input_wav_mean = (chan0 + chan1) / 2                 # mono, 48k
    feats = bestrq(rsq48tobestrq(input_wav_mean), features_only=True)
    layer_results = feats['layer_results']               # list of conformer hidden states
    bestrq_emb = layer_results[layer].permute(0,2,1)     # -> [B,1024,T]
vocal uses layer 7, bgm uses layer 3 (per generate_septoken defaults; captured both).

In features_only mode MusicFM25Hz.get_predictions runs ONLY:
  melspec(2048,hop240,128mel)+AmplitudeToDB -> normalize(stat) -> Conv2dSubsampling
  -> Wav2Vec2ConformerEncoder(output_hidden_states=True). The RandomProjectionQuantizer,
  self.linear(4096), cls_token are NOT touched. hidden_states is a tuple of length
  encoder_depth+1 (embeddings + 12 layers). layer_results[k] = hidden_states[k] [B,T,1024].

DETERMINISM: weights loaded from model_2.safetensors prefix 'bestrq.', eval(), CPU.
No randomness in the feature path (masking only runs in full forward()).

REFERENCE AUDIO CAVEAT: real encode consumes demucs-separated vocal/bgm stems.
We have no demucs, so first.wav (48k stereo) is fed AS BOTH stems. Musically
meaningless; exercises the graph deterministically, which is all a component golden needs.

Run from /tmp/songgen-src:
  PYTHONPATH=codeclm/tokenizer:.:codeclm/tokenizer/Flow1dVAE:codeclm/tokenizer/Flow1dVAE/our_MERT_BESTRQ/mert_fairseq/models/musicfm:tools/gradio \
  CUDA_VISIBLE_DEVICES="" /tmp/sg-venv/bin/python \
  /home/dbrain/dev/songgen-port/scripts/golden_clone_bestrq.py
"""
import os, sys, json, types
import numpy as np
import torch
import soundfile as sf
import sg_compat  # installs torchaudio stub etc.; but we need REAL torchaudio transforms here
sys.modules.setdefault("k_diffusion", types.ModuleType("k_diffusion"))

# The melspec + resample need a real torchaudio. sg_compat installs a stub only if
# torchaudio is absent; this venv has no torchaudio wheel, so the stub is active and
# torchaudio.transforms.MelSpectrogram/Resample/AmplitudeToDB would raise.
# We replace those three transforms with numerically-faithful torch implementations.
import torchaudio  # the sg_compat stub
import math


def _hann_window(n):
    # torch.hann_window(periodic=True) — matches torchaudio MelSpectrogram default
    return torch.hann_window(n, periodic=True, dtype=torch.float64)


def _mel_fb(sr, n_fft, n_mels):
    # torchaudio.functional.melscale_fbanks default (HTK=False, norm=None, slaney)
    from torchaudio_melfb import melscale_fbanks  # provided below via inline module
    return melscale_fbanks(n_fft // 2 + 1, 0.0, sr / 2.0, n_mels, sr)


# ---- minimal faithful MelSpectrogram + AmplitudeToDB (torchaudio defaults) ----
class RealMelSTFT(torch.nn.Module):
    def __init__(self, sample_rate=24000, n_fft=2048, hop_length=240, n_mels=128,
                 fb=None, win=None):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop_length
        # Prefer the checkpoint's shipped mel filterbank + STFT window (ground truth);
        # fall back to a faithful recompute (htk mel, Hann periodic) if not supplied.
        self.win = win if win is not None else torch.hann_window(n_fft, periodic=True)
        fb = fb if fb is not None else _mel_filterbank(sample_rate, n_fft, n_mels)  # [n_freq, n_mels]
        self.register_buffer("fb", fb)

    def forward(self, wav):
        # wav: [B, L]; torchaudio MelSpectrogram: power=2.0, center=True, pad_mode=reflect
        spec = torch.stft(wav, n_fft=self.n_fft, hop_length=self.hop,
                          win_length=self.n_fft, window=self.win.to(wav.dtype),
                          center=True, pad_mode="reflect", return_complex=True,
                          normalized=False)
        power = spec.real.pow(2) + spec.imag.pow(2)  # [B, n_freq, T]
        mel = torch.matmul(power.transpose(1, 2), self.fb).transpose(1, 2)  # [B, n_mels, T]
        return mel


def _mel_filterbank(sr, n_fft, n_mels):
    # Slaney-style mel filterbank, matching torchaudio.functional.melscale_fbanks
    # defaults (f_min=0, f_max=sr/2, norm=None, mel_scale='htk' is FALSE -> slaney).
    # torchaudio default mel_scale='htk'. MelSpectrogram default norm=None, mel_scale='htk'.
    n_freqs = n_fft // 2 + 1
    f_max = sr / 2.0
    all_freqs = torch.linspace(0, f_max, n_freqs)

    def hz_to_mel(f):  # htk
        return 2595.0 * math.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    m_min, m_max = hz_to_mel(0.0), hz_to_mel(f_max)
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = 700.0 * (10.0 ** (m_pts / 2595.0) - 1.0)
    fdiff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # [n_freqs, n_mels+2]
    down = -slopes[:, :-2] / fdiff[:-1]
    up = slopes[:, 2:] / fdiff[1:]
    fb = torch.clamp(torch.minimum(down, up), min=0.0)  # [n_freqs, n_mels]
    return fb


def amplitude_to_db(spec, top_db=80.0):
    # torchaudio.transforms.AmplitudeToDB(stype='power') defaults: multiplier=10,
    # amin=1e-10, db_multiplier=log10(max(amin, ref=1.0))=0, top_db=80
    amin = 1e-10
    db = 10.0 * torch.log10(torch.clamp(spec, min=amin))
    # db_multiplier = 10*log10(max(amin,1.0)) = 0 -> no subtraction
    db = torch.maximum(db, db.amax(dim=(-2, -1), keepdim=True) - top_db)
    return db


def resample_48k_to_24k(wav):
    # rsq48tobestrq = torchaudio.transforms.Resample(48000,24000). Decimate by 2 with
    # the same windowed-sinc torchaudio uses. Use julius (sg_compat's resample backend).
    import julius
    return julius.resample_frac(wav, 48000, 24000)


from our_MERT_BESTRQ.mert_fairseq.models.musicfm.musicfm_model import MusicFMModel, MusicFMConfig
from safetensors import safe_open

OUT = "/home/dbrain/dev/songgen-port/golden-large/clone"
CKPT = "/home/dbrain/dev/songgen-port/ckpt-audio/model_septoken/model_2.safetensors"
WAV = "/home/dbrain/dev/songgen-port/out/first.wav"
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)

cfg = MusicFMConfig()  # label_rate=25, encoder_dim=1024, encoder_depth=12, conv_dim=512, n_mels=128
bestrq = MusicFMModel(cfg).eval()

# Monkeypatch the (stubbed) torchaudio transforms inside the instantiated MusicFM with
# our faithful implementations so the feature path is numerically real.
mel = bestrq.model.preprocessor_melspec_2048
# load the checkpoint's shipped mel filterbank + window for an exact, port-faithful melspec
with safe_open(CKPT, "pt") as _f:
    _fb = _f.get_tensor("bestrq.model.preprocessor_melspec_2048.mel_stft.mel_scale.fb").float()
    _win = _f.get_tensor("bestrq.model.preprocessor_melspec_2048.mel_stft.spectrogram.window").float()
real = RealMelSTFT(sample_rate=24000, n_fft=2048, hop_length=240, n_mels=128, fb=_fb, win=_win).eval()
def _mel_forward(waveform, _real=real):
    return amplitude_to_db(_real(waveform))
mel.forward = _mel_forward  # is_db=True path

# load bestrq.* weights
PREFIX = "bestrq."
sd = {}
with safe_open(CKPT, "pt") as f:
    for k in f.keys():
        if k.startswith(PREFIX):
            sd[k[len(PREFIX):]] = f.get_tensor(k).float()
missing, unexpected = bestrq.load_state_dict(sd, strict=False)
# expected-missing: mel buffers we injected (fb/win), and pos_conv parametrization name diffs
print(f"[bestrq] prefix='{PREFIX}' loaded={len(sd)}")
print(f"[bestrq] missing({len(missing)})={list(missing)[:8]}{'...' if len(missing)>8 else ''}")
print(f"[bestrq] unexpected({len(unexpected)})={list(unexpected)[:8]}{'...' if len(unexpected)>8 else ''}")

# ---- input audio: first.wav as BOTH stems (mono mean) ----
wav, sr = sf.read(WAV, dtype="float32", always_2d=True)  # [L, C]
assert sr == 48000, sr
wav = torch.from_numpy(wav.T)  # [C, L]
if wav.shape[0] == 1:
    wav = wav.repeat(2, 1)
# mono mean of the two channels (extract_bestrq_embeds: (chan0+chan1)/2)
input_wav_mean_48k = wav.mean(0, keepdim=True)  # [1, L]
# trim to integer seconds-ish; bestrq.forward also trims to multiple of (24000/25)=960
x24 = resample_48k_to_24k(input_wav_mean_48k)  # [1, L24]

with torch.no_grad():
    res = bestrq(x24, features_only=True)
layer_results = res["layer_results"]  # tuple len encoder_depth+1
n_layers = len(layer_results)
emb7 = layer_results[7].permute(0, 2, 1).contiguous()   # [1,1024,T] vocal layer
emb3 = layer_results[3].permute(0, 2, 1).contiguous()   # [1,1024,T] bgm layer

x24_np = x24.cpu().numpy().astype(np.float32)
emb7_np = emb7.cpu().numpy().astype(np.float32)
emb3_np = emb3.cpu().numpy().astype(np.float32)
np.save(f"{OUT}/bestrq_in_audio24k.npy", x24_np)
np.save(f"{OUT}/bestrq_out_layer7_vocal.npy", emb7_np)
np.save(f"{OUT}/bestrq_out_layer3_bgm.npy", emb3_np)
# also save the full per-layer stack for thorough port validation
allnp = np.stack([lr.permute(0, 2, 1).cpu().numpy().astype(np.float32)[0] for lr in layer_results])
np.save(f"{OUT}/bestrq_out_all_layers.npy", allnp)  # [n_layers, 1024, T]

meta = dict(
    task="clone_bestrq_features_only",
    module="our_MERT_BESTRQ.mert_fairseq.models.musicfm.musicfm_model.MusicFMModel",
    method="forward(features_only=True) -> get_predictions -> melspec+db+normalize+Conv2dSubsampling+Wav2Vec2ConformerEncoder",
    weight_prefix=PREFIX, keys_loaded=len(sd),
    missing=list(missing), unexpected=list(unexpected),
    reference_audio=dict(path=WAV, sr_in=48000, note="first.wav fed as BOTH vocal and bgm stem; mono = mean of 2 channels; resampled 48k->24k for bestrq"),
    layers_captured=dict(vocal_layer=7, bgm_layer=3, n_layer_results=n_layers,
                         note="layer_results = conformer output_hidden_states tuple = [embeddings]+[12 layer outputs]; index 0 is pre-layer embedding"),
    musicfm_config=dict(
        label_rate=cfg.label_rate, sample_rate=24000,
        n_mels=128, n_fft=2048, hop_length=240, melspec_is_db=True,
        amplitude_to_db=dict(stype="power", multiplier=10, amin=1e-10, top_db=80, db_multiplier=0.0),
        normalize=dict(scheme="(x-mean)/std per feature 'melspec_2048'",
                       mean=bestrq.model.stat["melspec_2048_mean"], std=bestrq.model.stat["melspec_2048_std"]),
        rearrange="NONE in features_only path (rearrange only used for quantizer targets); melspec[...,:-1] trims last frame in preprocessing",
        conv_stem=dict(
            type="Conv2dSubsampling: 2x Res2dModule then Linear",
            res2d_0=dict(strides=[2, 2], in_ch=1, out_ch=512, kernel=3, pad=1,
                         layers="conv1(s=(2,2))->bn1->relu->conv2->bn2 + residual conv3(s=(2,2))->bn3, relu"),
            res2d_1=dict(strides=[2, 2], in_ch=512, out_ch=512),
            linear=dict(in_features=512 * 128 // 2 // 2, out_features=1024,
                        note="rearrange 'b c f t -> b t (c f)' then Linear(16384->1024)"),
            total_time_downsample=4, total_freq_downsample=4,
        ),
        conformer=dict(
            impl="transformers.models.wav2vec2_conformer.Wav2Vec2ConformerEncoder",
            num_hidden_layers=12, hidden_size=1024, num_attention_heads=16,
            intermediate_size=4096, hidden_act="swish",
            position_embeddings_type="rotary", rotary_embedding_base=10000,
            head_dim=64, rotary_inv_freq_len=32,
            do_stable_layer_norm=True, layer_norm_eps=1e-5,
            conv_depthwise_kernel_size=31, conv_bias=True,
            pos_conv=dict(num_conv_pos_embeddings=128, num_conv_pos_embedding_groups=16,
                          note="Wav2Vec2ConformerPositionalConvEmbedding applied BEFORE layers"),
            block_structure="0.5*FFN1 -> self_attn(rotary) -> conv_module(LN,pointwise1,GLU,depthwise+BN+swish,pointwise2) -> 0.5*FFN2 -> final_LN (macaron)",
            output_hidden_states=True,
            note="config from w2v2_config.json with num_hidden_layers=12, hidden_size=1024 overrides",
        ),
        UNUSED_in_features_only=["RandomProjectionQuantizer (quantizer_melspec_2048)",
                                 "self.linear (1024->4096 codebook logits)", "cls_token",
                                 "masking() (only in training forward)"],
    ),
    shapes=dict(in_audio24k=list(x24_np.shape),
                out_layer7_vocal=list(emb7_np.shape),
                out_layer3_bgm=list(emb3_np.shape),
                out_all_layers=list(allnp.shape)),
    stats=dict(l7_mean=float(emb7_np.mean()), l7_std=float(emb7_np.std()),
               l3_mean=float(emb3_np.mean()), l3_std=float(emb3_np.std())),
    CAVEAT_melspec="torchaudio has no ROCm wheel in this venv; MelSpectrogram+AmplitudeToDB "
                   "were reimplemented in-script (STFT power=2, center=reflect, AmplitudeToDB power/top_db=80). "
                   "The mel filterbank [1025,128] and Hann window [2048] are loaded DIRECTLY from the "
                   "checkpoint buffers (bestrq.model.preprocessor_melspec_2048.mel_stft.{mel_scale.fb,spectrogram.window}) "
                   "so they are exact. Resample 48k->24k via julius.resample_frac (matches torchaudio windowed-sinc to ~1e-4). "
                   "A C++ port should reuse the shipped fb+window verbatim and match the resampler; "
                   "cross-check resample against a real-torchaudio run on the CUDA box.",
)
with open(f"{OUT}/bestrq_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"[bestrq] x24{x24_np.shape} -> L7{emb7_np.shape} L3{emb3_np.shape} all{allnp.shape}")
print(f"[bestrq] L7 std={emb7_np.std():.5f} L3 std={emb3_np.std():.5f}")
print(f"[bestrq] saved to {OUT}")
