"""
Golden-tensor dump for the SongGeneration (LeVo 2) -> cpp port.
Reuses the shipped LeVoInference orchestration; patches in capture hooks.
Emits per-step LeLM logits, final codes, gen_tokens and decoded wav into GOLD/.
Portable: forces eager attention + plain fp16 load (no flash-attn, no offload).

Run from the SongGeneration repo root after assembling the ckpt dir:
    PYTHONPATH=codeclm/tokenizer:.:codeclm/tokenizer/Flow1dVAE:tools/gradio \
    GOLD=golden N_STEPS=16 DURATION=30 \
    python dump_golden.py <ckpt_dir>
"""
import os, sys, json, types
import numpy as np
import torch
import sg_compat  # applies all py3.14/torch-2.12/transformers-5 env shims

# description-only path never runs the separator; stub it so the (vendored,
# not-cloned) third_party.demucs import in separator.py doesn't break module load.
_sep = types.ModuleType("separator")
class Separator:  # noqa
    def __init__(self, *a, **k):
        raise RuntimeError("separator stub: prompt-audio path disabled for golden dump")
_sep.Separator = Separator
sys.modules["separator"] = _sep

# flashy is imported by codeclm.utils.utils but never called (it drags in a broken
# wandb/protobuf chain). Stub it + its .distrib submodule.
import importlib.machinery as _imach
for _m in ("flashy", "flashy.distrib"):
    _mod = types.ModuleType(_m); _mod.__spec__ = _imach.ModuleSpec(_m, loader=None)
    sys.modules[_m] = _mod
sys.modules["flashy"].distrib = sys.modules["flashy.distrib"]

# torchaudio has no ROCm wheel matching this torch; its real funcs (load/resample)
# are only used in the prompt-audio ENCODE path, which we skip. Stub the import,
# routing the rare real calls to julius/soundfile.
def _install_torchaudio_stub():
    import importlib.util, importlib.machinery
    try:
        if importlib.util.find_spec("torchaudio") is not None:
            return
    except (ValueError, ModuleNotFoundError):
        pass
    def _spec(n): return importlib.machinery.ModuleSpec(n, loader=None)
    import torch.nn as _nn
    ta = types.ModuleType("torchaudio"); ta.__version__ = "0.0-stub"; ta.__spec__ = _spec("torchaudio")
    func = types.ModuleType("torchaudio.functional")
    def resample(wav, orig_freq, new_freq, *a, **k):
        import julius; return julius.resample_frac(wav, int(orig_freq), int(new_freq))
    func.resample = resample
    def _load(path, *a, **k):
        import soundfile as sf
        d, sr = sf.read(path, dtype="float32", always_2d=True)
        return torch.from_numpy(d.T).contiguous(), sr
    def _save(path, wav, sr, *a, **k):
        import soundfile as sf
        w = wav.detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
        sf.write(path, w.T, int(sr))
    ta.load, ta.save, ta.functional = _load, _save, func
    trans = types.ModuleType("torchaudio.transforms")
    def _t_getattr(name):
        if name.startswith("__"):
            raise AttributeError(name)
        class _Stub(_nn.Module):
            def __init__(self, *a, **k): super().__init__()
            def forward(self, *a, **k):
                raise RuntimeError(f"torchaudio.transforms.{name} forwarded (unexpected on decode path)")
        _Stub.__name__ = name
        return _Stub
    trans.__getattr__ = _t_getattr
    ta.transforms = trans
    func.__spec__ = _spec("torchaudio.functional"); trans.__spec__ = _spec("torchaudio.transforms")
    sys.modules.update({"torchaudio": ta, "torchaudio.functional": func, "torchaudio.transforms": trans})
_install_torchaudio_stub()

CKPT = sys.argv[1]
GOLD = os.environ.get("GOLD", "golden")
N_STEPS = int(os.environ.get("N_STEPS", "16"))
DURATION = float(os.environ.get("DURATION", "30"))
os.makedirs(GOLD, exist_ok=True)

# structured lyric + style; keep fixed so cpp can reproduce the exact conditioning
LYRIC = os.environ.get("LYRIC",
    "[verse] walking through the quiet streets at night [verse] "
    "city lights are calling out my name [chorus] and i will find my way back home")
DESCRIPTION = os.environ.get("DESCRIPTION", "warm indie folk, male vocal, acoustic guitar, 92 bpm")

# compat shims: vendored modeling_llama.py is transformers~4.37-era; current
# transformers removed/moved some symbols. Flash is off, so stubs are safe.
def _shim_transformers():
    import transformers
    # levo.py asserts transformers<4.40, but its LmModel/CausalLM are fully
    # self-contained on the vendored llama (no post-4.40 internals). Spoof the
    # version so the guard passes; numerics come from the vendored code.
    transformers.__version__ = "4.39.0"
    import transformers.utils as tu
    if not hasattr(tu, "is_flash_attn_available"):
        tu.is_flash_attn_available = lambda: False
    if not hasattr(tu, "is_flash_attn_2_available"):
        tu.is_flash_attn_2_available = lambda: False
    try:
        import transformers.pytorch_utils as tp
        if not hasattr(tp, "ALL_LAYERNORM_LAYERS"):
            tp.ALL_LAYERNORM_LAYERS = [torch.nn.LayerNorm]
    except Exception:
        pass
    # SequenceSummary removed in transformers v5; the Flow1dVAE GPT2 builds it as a
    # vestigial multiple_choice_head that decode never forwards. No-op stub.
    # GPT2 Conv1D + pruning helpers removed from pytorch_utils in v5. Conv1D is
    # used (real impl, checkpoint-compatible weight/bias layout); pruners are dead.
    try:
        import transformers.pytorch_utils as tp2
        if not hasattr(tp2, "Conv1D"):
            class Conv1D(torch.nn.Module):
                def __init__(self, nf, nx):
                    super().__init__(); self.nf = nf
                    wv = torch.empty(nx, nf); torch.nn.init.normal_(wv, std=0.02)
                    self.weight = torch.nn.Parameter(wv); self.bias = torch.nn.Parameter(torch.zeros(nf))
                def forward(self, x):
                    so = x.size()[:-1] + (self.nf,)
                    return torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight).view(so)
            tp2.Conv1D = Conv1D
        for _n in ("find_pruneable_heads_and_indices", "prune_conv1d_layer", "prune_linear_layer"):
            if not hasattr(tp2, _n):
                setattr(tp2, _n, (lambda *a, **k: (_ for _ in ()).throw(RuntimeError("pruning stub"))))
    except Exception:
        pass
    import transformers.modeling_utils as mu
    if not hasattr(mu, "SequenceSummary"):
        class SequenceSummary(torch.nn.Module):
            def __init__(self, *a, **k): super().__init__()
            def forward(self, *a, **k): raise RuntimeError("SequenceSummary stub forwarded")
        mu.SequenceSummary = SequenceSummary
_shim_transformers()

import codeclm.models.lm_levo as lm_levo
import codeclm.models.codeclm as codeclm_mod

# this model uses plain RoPE (rope_theta only); newer transformers injects a
# rope_scaling dict without the legacy "type" key, tripping vendored _init_rope.
import codeclm.models.llama.modeling_llama as _ml
def _init_rope_plain(self):
    self.rotary_emb = _ml.LlamaRotaryEmbedding(
        self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_theta)
_ml.LlamaAttention._init_rope = _init_rope_plain

# --- capture: next-token logits per autoregressive step (cond+null rows, all codebooks) ---
_caps = []
_orig_fwd = lm_levo.LmModel.forward
def _fwd(self, sequence, condition_tensors):
    out = _orig_fwd(self, sequence, condition_tensors)        # [2B, K, S, card]
    if len(_caps) < N_STEPS:
        _caps.append(out[:, :, -1, :].detach().float().cpu().numpy())  # [2B, K, card]
    return out
lm_levo.LmModel.forward = _fwd

# --- capture: tokens handed to the codec/diffusion stage, and the decoded wav ---
_orig_ga = codeclm_mod.CodecLM.generate_audio
def _ga(self, gen_tokens, *a, **k):
    np.save(f"{GOLD}/gen_tokens.npy", gen_tokens.detach().cpu().numpy())
    wav = _orig_ga(self, gen_tokens, *a, **k)
    np.save(f"{GOLD}/wav.npy", wav.detach().cpu().numpy())
    return wav
codeclm_mod.CodecLM.generate_audio = _ga

# save LM goldens the instant the LM stage finishes, so any later audio-stage
# failure can't discard them (the LeLM port only needs these to start).
_orig_gen = codeclm_mod.CodecLM.generate
def _gen(self, *a, **k):
    toks = _orig_gen(self, *a, **k)
    logits = np.stack(_caps, 0) if _caps else np.zeros((0,))
    np.save(f"{GOLD}/lelm_logits.npy", logits)
    np.save(f"{GOLD}/gen_tokens_lm.npy", toks.detach().cpu().numpy())
    print(f"[golden] LM stage saved: lelm_logits{logits.shape}, tokens{tuple(toks.shape)}", flush=True)
    return toks
codeclm_mod.CodecLM.generate = _gen

# --- best-effort: capture one CFM estimator (velocity field) call for diffusion validation ---
def _try_hook_cfm(sep_tok):
    try:
        est = sep_tok.model.model.cfm_wrapper.estimator
        orig = est.forward
        state = {"n": 0}
        def wrapped(*a, **k):
            r = orig(*a, **k)
            if state["n"] == 0:
                ins = [x.detach().float().cpu().numpy() for x in a if torch.is_tensor(x)]
                for i, t in enumerate(ins):
                    np.save(f"{GOLD}/cfm_in_{i}.npy", t)
                out = r[0] if isinstance(r, tuple) else r
                if torch.is_tensor(out):
                    np.save(f"{GOLD}/cfm_out.npy", out.detach().float().cpu().numpy())
                state["n"] = 1
            return r
        est.forward = wrapped
    except Exception as e:
        print(f"[golden] cfm hook skipped: {e}")

from omegaconf import OmegaConf
from levo_inference_lowmem import LeVoInference

torch.manual_seed(0)
np.random.seed(0)

inf = LeVoInference(CKPT)
OmegaConf.set_struct(inf.cfg, False)
inf.cfg.lm.use_flash_attn_2 = False          # eager attention -> portable + deterministic
if "offload" in inf.cfg:
    del inf.cfg.offload                       # plain .cuda().fp16 load path

os.makedirs(GOLD, exist_ok=True)
with open(f"{GOLD}/meta.json", "w") as f:
    json.dump(dict(lyric=LYRIC, description=DESCRIPTION, duration=DURATION, n_steps=N_STEPS,
        sample_rate=int(inf.cfg.sample_rate_sep), code_depth=int(inf.cfg.lm.code_depth),
        code_size=int(inf.cfg.lm.code_size), dim=int(inf.cfg.lm.dim),
        num_layers=int(inf.cfg.lm.num_layers), num_layers_sub=int(inf.cfg.lm.num_layers_sub),
        delays=list(inf.cfg.codebooks_pattern.delay.delays),
        cfg_coef=float(inf.cfg.classifier_free_guidance.inference_coef)), f, indent=2)

# hook the separate tokenizer's CFM after it is built: wrap generate_audio entry once more
_orig_ga2 = codeclm_mod.CodecLM.generate_audio
def _ga_hooked(self, gen_tokens, *a, **k):
    if self.seperate_tokenizer is not None:
        _try_hook_cfm(self.seperate_tokenizer)
    return _ga(self, gen_tokens, *a, **k)
codeclm_mod.CodecLM.generate_audio = _ga_hooked

if os.environ.get("IMPORT_ONLY"):
    print(f"[golden] IMPORT_ONLY ok: imports + cfg load fine "
          f"(dim={inf.cfg.lm.dim}, layers={inf.cfg.lm.num_layers}+{inf.cfg.lm.num_layers_sub}, "
          f"sr={inf.cfg.sample_rate_sep})")
    sys.exit(0)

wav = inf(lyric=LYRIC, description=DESCRIPTION, gen_type="mixed",
          params={"duration": DURATION})

# --- persist goldens + the exact inputs needed to reproduce in cpp ---
logits = np.stack(_caps, axis=0) if _caps else np.zeros((0,))
np.save(f"{GOLD}/lelm_logits.npy", logits)    # [N_STEPS, 2B, K, card]
try:
    import torchaudio
    torchaudio.save(f"{GOLD}/out.flac", wav.detach().cpu(), inf.cfg.sample_rate_sep)
except Exception as e:
    print(f"[golden] flac save skipped: {e}")

meta = dict(lyric=LYRIC, description=DESCRIPTION, duration=DURATION, n_steps=N_STEPS,
            sample_rate=int(inf.cfg.sample_rate_sep), code_depth=int(inf.cfg.lm.code_depth),
            code_size=int(inf.cfg.lm.code_size), dim=int(inf.cfg.lm.dim),
            num_layers=int(inf.cfg.lm.num_layers), num_layers_sub=int(inf.cfg.lm.num_layers_sub),
            delays=list(inf.cfg.codebooks_pattern.delay.delays), cfg_coef=1.5)
with open(f"{GOLD}/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"[golden] wrote {GOLD}/: lelm_logits{logits.shape}, gen_tokens, wav, meta.json")
