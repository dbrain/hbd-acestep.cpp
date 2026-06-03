"""
READ-ONLY clipping diagnosis: run the REFERENCE audio decode on OUR C++ gen_tokens
and report wav + latent stats. Mirrors generate_septoken.Tango.code2sound exactly
(no-prompt path, guidance_scale=1.5), but builds PromptCondAudioDiffusion + the VAE
directly to avoid the demucs/bestrq import graph (bestrq is encode-only, unused here).

Run from /home/dbrain/dev/songgen-port/songgen-src-code:
  PYTHONPATH=codeclm/tokenizer:.:codeclm/tokenizer/Flow1dVAE:tools/gradio \
  CUDA_VISIBLE_DEVICES="" /tmp/sg-venv/bin/python \
  /home/dbrain/dev/songgen-port/scripts/diag_ref_decode.py
"""
import os, sys, json, types, math
import numpy as np
import torch
import soundfile as sf
import sg_compat
sys.modules.setdefault("k_diffusion", types.ModuleType("k_diffusion"))

# third_party.demucs is absent; generate_septoken imports it at module load (Separator only,
# unused on decode). Stub the import target.
for _m in ("third_party.demucs", "third_party.demucs.models",
           "third_party.demucs.models.pretrained"):
    if _m not in sys.modules:
        _mm = types.ModuleType(_m); _mm.__path__ = []
        sys.modules[_m] = _mm
sys.modules["third_party.demucs.models.pretrained"].get_model_from_yaml = lambda *a, **k: None

# vendored GPT2Config legacy class attrs dropped by transformers 5.x (unet path unused).
from models_gpt.models.gpt2_config import GPT2Config as _GC
for _a, _v in dict(add_cross_attention=False, classifier_dropout=None, hidden_dropout=None,
                   pad_token_id=None, scale_attn_by_inverse_layer_idx=False,
                   reorder_and_upcast_attn=False).items():
    if not hasattr(_GC, _a):
        setattr(_GC, _a, _v)

from model_septoken import PromptCondAudioDiffusion
from safetensors.torch import load_file
from tools.get_1dvae_large import get_model

DEV = "cpu"
ROOT = "/home/dbrain/dev/songgen-port"
CKPT = f"{ROOT}/ckpt-audio/model_septoken/model_2.safetensors"
VAE_CFG = f"{ROOT}/ckpt-audio/vae/stable_audio_1920_vae.json"
VAE_CKPT = f"{ROOT}/ckpt-audio/vae/autoencoder_music_1320k.ckpt"
OUT = f"{ROOT}/out"
TOKENS = f"{OUT}/gen_tokens_cpp.npy"

torch.manual_seed(0)

# ---- build the diffusion model (rvq + cfm + normfeat) and load full ckpt ----
main_config = dict(num_channels=32, unet_model_name=None,
                   unet_model_config_path="configs/models/transformer2D_wocross_inch112_1x4_multi_large.json",
                   snr_gamma=None)
model = PromptCondAudioDiffusion(**main_config).to(DEV).eval()
sd = load_file(CKPT)
missing, unexpected = model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
crit = [k for k in missing if k.startswith(("rvq_bestrq", "normfeat", "cfm_wrapper", "mask_emb", "zero_cond"))]
print(f"[load] missing total={len(missing)} unexpected={len(unexpected)} critical_missing={crit}")
assert not crit, crit
model.init_device_dtype(torch.device(DEV), torch.float32)

# transformers 5.x dropped ModuleUtilsMixin.get_head_mask; decode always passes head_mask=None.
_est = model.cfm_wrapper.estimator
if not hasattr(_est, "get_head_mask"):
    def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        return [None] * num_hidden_layers
    type(_est).get_head_mask = _get_head_mask

print(f"[normfeat] counts={float(model.normfeat.counts):.0f} "
      f"std[min/mean/max]={float(model.normfeat.std.min()):.3f}/"
      f"{float(model.normfeat.std.mean()):.3f}/{float(model.normfeat.std.max()):.3f} "
      f"mean[absmax]={float(model.normfeat.mean.abs().max()):.3f}")

# ---- VAE ----
vae = get_model(VAE_CFG, VAE_CKPT).to(DEV).eval()

# ---- tokens: cb1=vocal, cb2=bgm (codeclm.py:291) ----
gen = torch.from_numpy(np.load(TOKENS)).long().to(DEV)  # [1,3,375]
codes_vocal = gen[:, [1], :]   # [1,1,375]
codes_bgm = gen[:, [2], :]
print(f"[tokens] gen{tuple(gen.shape)} vocal{tuple(codes_vocal.shape)} bgm{tuple(codes_bgm.shape)} "
      f"vmax={int(codes_vocal.max())} bmax={int(codes_bgm.max())}")

# ===== replicate generate_septoken.Tango.code2sound (no-prompt path) =====
sample_rate = 48000
duration = 40
guidance_scale = 1.5
num_steps = 20   # task-requested; note the high-level decode() default is 10

min_samples = duration * 25          # frames = 1000
hop_samples = min_samples // 4 * 3   # 750
ovlp_samples = min_samples - hop_samples  # 250
first_latent = torch.randn(codes_vocal.shape[0], min_samples, 64).to(DEV)
first_latent_length = 0
first_latent_codes_length = 0

codes_len = codes_vocal.shape[-1]
target_len = int((codes_len - first_latent_codes_length) / 100 * 4 * sample_rate)

# code repeat to >= min_samples
if codes_len < min_samples:
    while codes_vocal.shape[-1] < min_samples:
        codes_vocal = torch.cat([codes_vocal, codes_vocal], -1)
        codes_bgm = torch.cat([codes_bgm, codes_bgm], -1)
    codes_vocal = codes_vocal[:, :, 0:min_samples]
    codes_bgm = codes_bgm[:, :, 0:min_samples]
codes_len = codes_vocal.shape[-1]
if (codes_len - ovlp_samples) % hop_samples > 0:
    len_codes = math.ceil((codes_len - ovlp_samples) / float(hop_samples)) * hop_samples + ovlp_samples
    while codes_vocal.shape[-1] < len_codes:
        codes_vocal = torch.cat([codes_vocal, codes_vocal], -1)
        codes_bgm = torch.cat([codes_bgm, codes_bgm], -1)
    codes_vocal = codes_vocal[:, :, 0:len_codes]
    codes_bgm = codes_bgm[:, :, 0:len_codes]

latent_length = min_samples
latent_list = []
spk_embeds = torch.zeros([1, 32, 1, 32], device=DEV)

for sinx in range(0, codes_vocal.shape[-1] - hop_samples, hop_samples):
    cv = codes_vocal[:, :, sinx:sinx + min_samples]
    cb = codes_bgm[:, :, sinx:sinx + min_samples]
    if sinx == 0:
        incontext_length = first_latent_length
        lat = model.inference_codes([cv, cb], spk_embeds, first_latent, latent_length,
                                    incontext_length=incontext_length, additional_feats=[],
                                    guidance_scale=guidance_scale, num_steps=num_steps,
                                    disable_progress=True, scenario='other_seg')
        latent_list.append(lat)
    else:
        true_latent = latent_list[-1][:, :, -ovlp_samples:].permute(0, 2, 1)
        len_add = min_samples - true_latent.shape[-2]
        incontext_length = true_latent.shape[-2]
        true_latent = torch.cat([true_latent,
                                 torch.randn(true_latent.shape[0], len_add, true_latent.shape[-1]).to(DEV)], -2)
        lat = model.inference_codes([cv, cb], spk_embeds, true_latent, latent_length,
                                    incontext_length=incontext_length, additional_feats=[],
                                    guidance_scale=guidance_scale, num_steps=num_steps,
                                    disable_progress=True, scenario='other_seg')
        latent_list.append(lat)

latent_list = [l.float() for l in latent_list]
latent_list[0] = latent_list[0][:, :, first_latent_length:]

# report the post-normfeat (denormalized, VAE-native) latent that is fed to the VAE
cat_lat = torch.cat(latent_list, -1)
print(f"[latent->VAE] denorm latent {tuple(cat_lat.shape)} "
      f"std={float(cat_lat.std()):.4f} min={float(cat_lat.min()):.3f} max={float(cat_lat.max()):.3f}")

min_s = int(min_samples * sample_rate // 1000 * 40)
hop_s = int(hop_samples * sample_rate // 1000 * 40)
ovlp_s = min_s - hop_s

output = None
for i in range(len(latent_list)):
    cur = vae.decode_audio(latent_list[i], chunked=False)[0].detach().cpu()
    if output is None:
        output = cur
    else:
        ov = torch.from_numpy(np.linspace(0, 1, ovlp_s)[None, :])
        ov = torch.cat([ov, 1 - ov], -1)
        output[:, -ovlp_s:] = output[:, -ovlp_s:] * ov[:, -ovlp_s:] + cur[:, 0:ovlp_s] * ov[:, 0:ovlp_s]
        output = torch.cat([output, cur[:, ovlp_s:]], -1)
output = output[:, 0:target_len]

wav = output.float().numpy()  # [2, T]
print(f"[REF wav] shape={wav.shape} min={wav.min():.4f} max={wav.max():.4f} "
      f"absmax={np.abs(wav).max():.4f} std={wav.std():.5f} "
      f"frac|>1|={(np.abs(wav) > 1.0).mean():.4%}")

np.save(f"{OUT}/ref_decode.npy", wav.astype(np.float32))
sf.write(f"{OUT}/ref_decode.wav", wav.T, sample_rate)
print(f"[save] {OUT}/ref_decode.npy  {OUT}/ref_decode.wav")

# ===== TASK 2: compare to OUR generated.wav =====
our, sr2 = sf.read(f"{OUT}/generated.wav", dtype="float32", always_2d=True)
our = our.T  # [C, T]
print(f"[OUR wav] generated.wav shape={our.shape} sr={sr2} min={our.min():.4f} "
      f"max={our.max():.4f} absmax={np.abs(our).max():.4f} std={our.std():.5f}")

n = min(wav.shape[1], our.shape[1])
a = wav[:, :n].reshape(-1).astype(np.float64)
b = our[:, :n].reshape(-1).astype(np.float64)
cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
print(f"[compare] aligned T={n} cossim(ref,our)={cos:.5f} "
      f"ref_absmax={np.abs(wav).max():.4f} our_absmax={np.abs(our).max():.4f} "
      f"std_ratio(our/ref)={our.std()/ (wav.std()+1e-12):.4f}")

meta = dict(
    num_steps=num_steps, guidance_scale=guidance_scale,
    normfeat=dict(counts=float(model.normfeat.counts),
                  std_mean=float(model.normfeat.std.mean()),
                  std_min=float(model.normfeat.std.min()),
                  std_max=float(model.normfeat.std.max())),
    denorm_latent=dict(std=float(cat_lat.std()), min=float(cat_lat.min()), max=float(cat_lat.max())),
    ref_wav=dict(min=float(wav.min()), max=float(wav.max()), absmax=float(np.abs(wav).max()),
                 std=float(wav.std()), frac_gt1=float((np.abs(wav) > 1.0).mean())),
    our_wav=dict(min=float(our.min()), max=float(our.max()), absmax=float(np.abs(our).max()),
                 std=float(our.std())),
    cossim=cos,
)
with open(f"{OUT}/diag_ref_decode_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("[done]")
