"""
Compat shims to run the SongGeneration (LeVo 2) reference on Python 3.14 /
torch 2.12 ROCm / transformers 5.x. Import this BEFORE importing codeclm.* .
All shims are import-time only; none alter the numeric forward path.
"""
import sys, types, importlib.machinery
import torch

def _spec(n):
    return importlib.machinery.ModuleSpec(n, loader=None)

# 1) separator (vendored third_party.demucs not present; prompt-audio path unused)
_sep = types.ModuleType("separator"); _sep.__spec__ = _spec("separator")
class Separator:  # noqa
    def __init__(self, *a, **k):
        raise RuntimeError("separator stub: prompt-audio path disabled")
_sep.Separator = Separator
sys.modules["separator"] = _sep

# 2) flashy (imported by codeclm.utils.utils, never called; drags broken wandb)
for _m in ("flashy", "flashy.distrib"):
    _mod = types.ModuleType(_m); _mod.__spec__ = _spec(_m)
    sys.modules[_m] = _mod
sys.modules["flashy"].distrib = sys.modules["flashy.distrib"]

# 3) torchaudio (no ROCm wheel; real funcs only used in skipped encode path)
def _install_torchaudio_stub():
    import importlib.util
    try:
        if importlib.util.find_spec("torchaudio") is not None:
            return
    except (ValueError, ModuleNotFoundError):
        pass
    ta = types.ModuleType("torchaudio"); ta.__version__ = "0.0-stub"; ta.__spec__ = _spec("torchaudio")
    func = types.ModuleType("torchaudio.functional")
    def resample(wav, orig_freq, new_freq, *a, **k):
        import julius; return julius.resample_frac(wav, int(orig_freq), int(new_freq))
    func.resample = resample
    def _load(path, *a, **k):
        import soundfile as sf, numpy as np
        d, sr = sf.read(path, dtype="float32", always_2d=True)
        return torch.from_numpy(d.T).contiguous(), sr
    def _save(path, wav, sr, *a, **k):
        import soundfile as sf, numpy as np
        w = wav.detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
        sf.write(path, w.T, int(sr))
    ta.load, ta.save, ta.functional = _load, _save, func
    trans = types.ModuleType("torchaudio.transforms")
    def _t_getattr(name):
        if name.startswith("__"):
            raise AttributeError(name)
        class _Stub(torch.nn.Module):
            def __init__(self, *a, **k): super().__init__()
            def forward(self, *a, **k):
                raise RuntimeError(f"torchaudio.transforms.{name} forwarded")
        _Stub.__name__ = name
        return _Stub
    trans.__getattr__ = _t_getattr
    func.__spec__ = _spec("torchaudio.functional"); trans.__spec__ = _spec("torchaudio.transforms")
    ta.transforms = trans
    sys.modules.update({"torchaudio": ta, "torchaudio.functional": func, "torchaudio.transforms": trans})
_install_torchaudio_stub()

# NOTE: the audio decode (Stable Audio VAE) also pulls k_diffusion -> torchvision, whose
# enum-heavy API resists stubbing and risks regressing the LM path via transformers'
# image utils. Capturing the wav/CFM golden is deferred to task #8 (needs a real
# ABI-matched torchvision or running that stage on the CUDA box). LM golden is complete.

# 3b) stable_audio_tools: repo imports it as third_party.stable_audio_tools.stable_audio_tools;
# alias to the pip-installed package (its `from torchaudio import transforms` hits the stub above).
def _alias_stable_audio_tools():
    try:
        import stable_audio_tools
    except Exception:
        return
    for p in ("third_party", "third_party.stable_audio_tools"):
        if p not in sys.modules:
            m = types.ModuleType(p); m.__path__ = []; m.__spec__ = _spec(p); sys.modules[p] = m
    sys.modules["third_party.stable_audio_tools.stable_audio_tools"] = stable_audio_tools
_alias_stable_audio_tools()

# 4) transformers 5.x vs vendored ~4.37 code
def _shim_transformers():
    import transformers
    transformers.__version__ = "4.39.0"   # levo.py asserts <4.40; code is self-contained
    import transformers.utils as tu
    for n in ("is_flash_attn_available", "is_flash_attn_2_available"):
        if not hasattr(tu, n):
            setattr(tu, n, lambda: False)
    # GPT2 legacy bits removed from pytorch_utils in v5
    try:
        import transformers.pytorch_utils as tp
        if not hasattr(tp, "ALL_LAYERNORM_LAYERS"):
            tp.ALL_LAYERNORM_LAYERS = [torch.nn.LayerNorm]
        if not hasattr(tp, "Conv1D"):
            class Conv1D(torch.nn.Module):
                def __init__(self, nf, nx):
                    super().__init__(); self.nf = nf
                    wv = torch.empty(nx, nf); torch.nn.init.normal_(wv, std=0.02)
                    self.weight = torch.nn.Parameter(wv); self.bias = torch.nn.Parameter(torch.zeros(nf))
                def forward(self, x):
                    so = x.size()[:-1] + (self.nf,)
                    return torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight).view(so)
            tp.Conv1D = Conv1D
        for n in ("find_pruneable_heads_and_indices", "prune_conv1d_layer", "prune_linear_layer"):
            if not hasattr(tp, n):
                setattr(tp, n, lambda *a, **k: (_ for _ in ()).throw(RuntimeError("pruning stub")))
    except Exception:
        pass
    # SequenceSummary removed; built as a vestigial multiple_choice_head, never forwarded
    import transformers.modeling_utils as mu
    if not hasattr(mu, "SequenceSummary"):
        class SequenceSummary(torch.nn.Module):
            def __init__(self, *a, **k): super().__init__()
            def forward(self, *a, **k): raise RuntimeError("SequenceSummary stub forwarded")
        mu.SequenceSummary = SequenceSummary
    # model_parallel_utils removed in v5 (GPT2 parallelism, unused at inference)
    if "transformers.utils.model_parallel_utils" not in sys.modules:
        mpu = types.ModuleType("transformers.utils.model_parallel_utils")
        mpu.__spec__ = _spec("transformers.utils.model_parallel_utils")
        mpu.assert_device_map = lambda *a, **k: None
        mpu.get_device_map = lambda n_layers, devices: {0: list(range(n_layers))}
        sys.modules["transformers.utils.model_parallel_utils"] = mpu
    # transformers.onnx removed in v5 (ONNX export base classes; GPT2Config subclasses
    # OnnxConfigWithPast but it's never used at inference)
    if "transformers.onnx" not in sys.modules:
        onnx = types.ModuleType("transformers.onnx"); onnx.__spec__ = _spec("transformers.onnx")
        class _OnnxStub:
            def __init__(self, *a, **k): pass
        onnx.OnnxConfigWithPast = _OnnxStub
        onnx.PatchingSpec = _OnnxStub
        onnx.OnnxConfig = _OnnxStub
        sys.modules["transformers.onnx"] = onnx
_shim_transformers()


def patch_rope():
    """Call AFTER codeclm import: force plain RoPE (newer rope_scaling lacks 'type')."""
    import codeclm.models.llama.modeling_llama as _ml
    def _init_rope_plain(self):
        self.rotary_emb = _ml.LlamaRotaryEmbedding(
            self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_theta)
    _ml.LlamaAttention._init_rope = _init_rope_plain
