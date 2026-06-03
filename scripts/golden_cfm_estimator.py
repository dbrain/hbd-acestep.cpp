"""
Task 2: deterministic CFM-estimator single-forward golden.

The CFM velocity-field estimator is
PromptCondAudioDiffusion.cfm_wrapper.estimator, a custom RoPE+time-conditioned
non-causal reflow GPT2:
    models_gpt.models.gpt2_rope2_time_new_correct_mask_noncasual_reflow.GPT2Model
built with GPT2Config(n_positions=1000, n_layer=16, n_head=20, n_embd=2200, n_inner=4400).

We instantiate the estimator ALONE, load its weights from model_2.safetensors
(prefix 'cfm_wrapper.estimator.'), and run ONE forward with fixed-seed inputs of
the exact shapes the decode path (model_septoken.solve_euler, guidance_scale>1)
produces, then save inputs + output.

inputs_embeds layout along feature axis (sums to n_embd=2200), per solve_euler:
    cat([ latent_mask_input(24), incontext_x(64), mu(2048), x_next(64) ], dim=-1)
  where mu = cat([quantized_bestrq_emb(1024), quantized_bestrq_emb_bgm(1024)]).
Under classifier-free guidance the batch is doubled -> [2B, T, 2200].
The estimator is called as:
    estimator(inputs_embeds=model_input, attention_mask=attention_mask, time_step=timestep)
and decode uses out.last_hidden_state[..., -64:] as the velocity. We save the FULL
last_hidden_state [2B, T, 2200].

Run from /tmp/songgen-src:
  PYTHONPATH=codeclm/tokenizer:.:codeclm/tokenizer/Flow1dVAE:tools/gradio \
  CUDA_VISIBLE_DEVICES="" /tmp/sg-venv/bin/python \
  /home/dbrain/dev/songgen-port/scripts/golden_cfm_estimator.py
"""
import os, sys, json, types
import numpy as np
import torch
import sg_compat
sys.modules.setdefault("k_diffusion", types.ModuleType("k_diffusion"))

# the estimator module lives under codeclm/tokenizer/Flow1dVAE; its package
# imports are relative to that dir (already on PYTHONPATH per the run cmd).
from models_gpt.models.gpt2_rope2_time_new_correct_mask_noncasual_reflow import GPT2Model
from models_gpt.models.gpt2_config import GPT2Config
from safetensors import safe_open

OUT = "/home/dbrain/dev/songgen-port/golden-large/audio"
CKPT = "/home/dbrain/dev/songgen-port/ckpt-audio/model_septoken/model_2.safetensors"
os.makedirs(OUT, exist_ok=True)

CFG = dict(n_positions=1000, n_layer=16, n_head=20, n_embd=2200, n_inner=4400)
config = GPT2Config(**CFG)
config._attn_implementation = "eager"
# transformers 5.x PretrainedConfig dropped these legacy class-level defaults that
# the vendored ~4.37-era GPT2 reads. Restore the original GPT2 defaults (none alter
# the active numeric path: cross-attn off, dropouts unused at eval).
for _a, _v in dict(add_cross_attention=False, classifier_dropout=None,
                   hidden_dropout=None, pad_token_id=None,
                   n_layers=CFG["n_layer"]).items():
    if not hasattr(config, _a):
        setattr(config, _a, _v)
est = GPT2Model(config).eval()

# transformers 5.x dropped ModuleUtilsMixin.get_head_mask. With head_mask=None
# (always, on the decode path) the original returns [None] * n_layer.
if not hasattr(est, "get_head_mask"):
    def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        return [None] * num_hidden_layers
    type(est).get_head_mask = _get_head_mask

PREFIX = "cfm_wrapper.estimator."
est_sd = {}
with safe_open(CKPT, "pt") as f:
    for k in f.keys():
        if k.startswith(PREFIX):
            est_sd[k[len(PREFIX):]] = f.get_tensor(k)
est = est.float()
est_sd = {k: v.float() for k, v in est_sd.items()}
missing, unexpected = est.load_state_dict(est_sd, strict=False)
print(f"[cfm] prefix='{PREFIX}' loaded={len(est_sd)} keys")
print(f"[cfm] missing={list(missing)}")
print(f"[cfm] unexpected={list(unexpected)}")

# ---- fixed-seed inputs of the exact decode-path shapes ----
# B=1 -> 2B=2 under classifier-free guidance. T = sequence length (frames).
torch.manual_seed(0)
B = 1
T = 32  # frame count (small, deterministic; real decode uses ~num_frames)
N_EMBD = 2200
SEG = dict(latent_mask=24, incontext=64, mu=2048, x_next=64)  # sums to 2200
assert sum(SEG.values()) == N_EMBD

inputs_embeds = torch.randn(2 * B, T, N_EMBD)        # [2B, T, 2200]
time_step = torch.full((2 * B,), 0.3, dtype=torch.float32)  # ti broadcast in solve_euler

# attention_mask: decode builds [B,1,L,L] from latent_masks>0.5, then double()s the
# batch under guidance -> [2B,1,T,T]. Use all-ones (all valid frames) for the golden.
attention_mask = torch.ones(2 * B, 1, T, T, dtype=torch.float32)

with torch.no_grad():
    out = est(inputs_embeds=inputs_embeds, attention_mask=attention_mask, time_step=time_step)
last_hidden = out.last_hidden_state  # [2B, T, 2200]
velocity = last_hidden[..., -SEG["x_next"]:]  # what decode actually consumes

ie = inputs_embeds.cpu().numpy().astype(np.float32)
ts = time_step.cpu().numpy().astype(np.float32)
am = attention_mask.cpu().numpy().astype(np.float32)
lh = last_hidden.cpu().numpy().astype(np.float32)
vel = velocity.cpu().numpy().astype(np.float32)
np.save(f"{OUT}/cfm_in_inputs_embeds.npy", ie)
np.save(f"{OUT}/cfm_in_time_step.npy", ts)
np.save(f"{OUT}/cfm_in_attention_mask.npy", am)
np.save(f"{OUT}/cfm_out_last_hidden_state.npy", lh)
np.save(f"{OUT}/cfm_out_velocity.npy", vel)

meta = dict(
    task="cfm_estimator_single_forward_golden",
    approach="standalone estimator (option B): instantiate estimator alone, load "
             "weights from septoken safetensors, run ONE forward with fixed-seed "
             "inputs of the exact decode-path shapes. Not wired through the full "
             "separate-tokenizer decode (avoids bestrq/rvq/clap deps), so inputs are "
             "synthetic-but-shaped, fully reproducible in C++.",
    module="models_gpt.models.gpt2_rope2_time_new_correct_mask_noncasual_reflow.GPT2Model",
    config=CFG, attn_implementation="eager",
    weights=dict(ckpt=CKPT, state_dict_prefix=PREFIX, keys_loaded=len(est_sd),
                 missing=list(missing), unexpected=list(unexpected)),
    call="est(inputs_embeds=[2B,T,2200], attention_mask=[2B,1,T,T] ones, time_step=[2B]=0.3)",
    seed="torch.manual_seed(0); inputs_embeds=randn(2,32,2200); time_step=0.3; attn=ones",
    inputs_embeds_feature_layout=SEG,
    note="decode consumes out.last_hidden_state[..., -64:] as the flow velocity; both "
         "the full last_hidden_state and the -64 velocity slice are saved. "
         "adaln_single time-embedding + scale_shift_table + proj_out are part of this "
         "GPT2Model (not vanilla HF GPT2). time_step is a float in [0,1] (continuous "
         "reflow time), NOT an integer diffusion index.",
    shapes=dict(inputs_embeds=list(ie.shape), time_step=list(ts.shape),
                attention_mask=list(am.shape), last_hidden_state=list(lh.shape),
                velocity=list(vel.shape)),
    out_stats=dict(lh_min=float(lh.min()), lh_max=float(lh.max()),
                   lh_mean=float(lh.mean()), lh_std=float(lh.std()),
                   vel_std=float(vel.std())),
)
with open(f"{OUT}/cfm_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"[cfm] inputs_embeds{ie.shape} time_step{ts.shape} -> last_hidden{lh.shape}")
print(f"[cfm] lh min={lh.min():.4f} max={lh.max():.4f} std={lh.std():.5f}  vel_std={vel.std():.5f}")
print(f"[cfm] saved cfm_in_*.npy, cfm_out_*.npy, cfm_meta.json to {OUT}")
