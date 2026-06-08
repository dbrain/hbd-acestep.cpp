# SA3 port — HANDOFF: feature-complete + perf-tune (then prod)

Fresh-context handoff. The text-to-audio (T2A) port is **DONE, validated on GPU, committed + pushed**
(`origin/sa3-port` @ `79d2aaa` on `dbrain/hbd-acestep.cpp`; kobbler dev harness @ `35d83688`).
This phase = **feature-complete the model surface, then performance-tune**. Prod (server / docker /
koblem wiring / master merge) is the phase AFTER this — do NOT start it here.

Read `SA3-PORT-PLAN.md` first (full design, validation, quant ladder, "GPU ENABLEMENT + PERF" section).
This doc is the task list on top of it.

---
## Where things stand (all on GPU, validated vs goldens)
- **T2A pipeline complete:** tokenize → T5Gemma (0.99995) → cond assembly → euler/pingpong sampler over
  DiT (0.9997) → SAME **decode** (0.99998) → wav. `tools/sa3-gen.cpp`. Runs F32 and Q8.
- **Flash attention shipped, default-on for GPU** (`src/sa3-attn.h`, `SA3_FLASH=0` opt-out; materialized
  F32 path is the CPU/oracle). Wall −25/−33/−48% at 10/30/120s; near-lossless; the prod path.
- **Perf characterised:** loads ~0.5s constant (per-invocation GPU upload — killed only by a warm server,
  a prod-phase concern). Compute RTF ~0.04–0.07 for 30–240s clips. 380s = 24–27s, split ~50/50
  DiT-attention (L²) / SAME-decode (chunked, ~linear).

## Build / run / test / ear / perf  (everything is docker; host has no usable nvcc)
Harness: `kobbler/docker/acestep-sa3-dev/` (`REPO_DIR` defaults to this worktree).
```
cd ~/dev/kobbler/docker/acestep-sa3-dev
./iter.sh build                              # CUDA build sa3-gen+tests -> build-cuda/<name>
./iter.sh gen  -- --prompt "..." --seconds 10 --out /src/out/x.wav     # GPU render (timing on stderr)
./iter.sh test dit  -- models/sa3-dit-f32.gguf .sa3ref/goldens 0       # per-component golden check (GPU)
./iter.sh test same -- models/sa3-same-f32.gguf .sa3ref/goldens
SA3_FLASH=0 ./iter.sh test dit -- ...        # force materialized F32 oracle (CPU-equivalent)
./eartest.sh 8103 "a:" "b:SA3_FLASH=0"       # side-by-side audio page (live :8103)
./perf.sh nsys 30                            # kernel-time summary   |   ./perf.sh ncu mul_mat 200 30
```
CPU oracle build (for new-component F32 validation, no GPU drift): in the worktree `./buildcpu.sh`
then `GGML_BACKEND=CPU ./build/sa3-<x>-test ...`.

Reference + goldens: `.sa3ref/` (gitignored) — cloned `stable-audio-3` + `.venv` (torch 2.7.1 cu126).
Golden generators: `.sa3ref/gen_goldens.py`, `gen_euler_golden.py`. HF token in `kobbler/.env`.

---
## PHASE 2 — feature-complete (gated on the SAME encoder)
SA3 supports **T2A (done)**, **audio-to-audio editing**, **inpainting**, **continuation**. All three of the
latter need the **SAME *encoder*** (audio→latent), which was deliberately deferred — only the decoder exists.

### 2a. SAME encoder  →  `src/sa3-same-enc.h`  [the unlock]
Mirror of the decoder already in `src/sa3-same.h`. Architecture (from SA3-PORT-PLAN.md §SAME):
patch(256) → weightnorm conv 512→1536 → **12× DynamicTanh differential-attn transformer** (same blocks as
decode: band mask |i−j|≤17, partial NEOX RoPE, alpha/gamma/beta norm) → proj 1536→256 → **softnorm
bottleneck encode** (`scaling_factor`/`running_std`/`noise_scaling_factor`/`auto_scale`).
- Reference: `.sa3ref/stable-audio-3/stable_audio_3/models/autoencoders.py` (encode path) +
  `models/bottleneck.py`. Clean cross-check: `optimized/mlx/models/defs/same_l_encoder.py`.
- Reuse the decode helpers in `sa3-attn.h`/`sa3-same.h` (`sa3_heads_dyt`, `sa3_diff_attn`, weightnorm conv).
  The encoder transformer is the SAME block stack — most of it is copy-from-decoder with patch/proj swapped.
- **Golden:** add an encode dump to `gen_goldens.py` (feed a real wav → reference `encode()` → latent npy);
  new `tools/sa3-same-enc-test.cpp` cosine vs golden (target ≥0.999 like decode). The bottleneck is the
  fiddly bit — validate `running_std`/softnorm direction against the reference exactly.

### 2b. Audio-to-audio editing  (SDEdit-style)
encode(input wav) → latent → renoise to a chosen σ (strength knob) → run the sampler from that σ with the
NEW prompt → decode. Reference: `inference/sampling.py` (the init-from-latent path) + `models/diffusion.py`.
CLI: extend `sa3-gen` with `--input wav --strength 0..1` (strength picks the start σ on the LogSNRShift
schedule). Validate: deterministic-euler latent cosine vs a reference edit golden.

### 2c. Inpainting + continuation
Reference: `.sa3ref/stable-audio-3/stable_audio_3/models/inpainting.py`. Mechanism: the DiT's
`to_local_embed` (257→1536, currently fed **zeros** for T2A) takes **[256 masked-latent channels ; 1 mask
channel] = 257**; the known region is pinned each step (replace the unmasked latents with the
noised-known values before each denoise). Continuation = inpaint with the mask = "everything after the
prompt-audio tail". CLI: `--input wav --mask <spec>` (and a `--continue` convenience = tail mask).
Wire `to_local_embed` in `sa3-dit.h` to accept the real (latent,mask) instead of the zero constant
(the bias-replicate path is already there). Validate vs an inpaint golden.

**Order:** 2a → (2b ∥ 2c). 2a is ~the decoder's effort and was first-build-correct last time; budget the
bottleneck + goldens as the real cost.

---
## PHASE 3 — performance tune (after features, before prod)
Grounded targets (Q8, flash, RTX 3060):
- **Steps reduction (highest ROI, all lengths):** currently 8 euler steps. Render steps 4/6/8 to the ear
  page (`./eartest.sh 8103 "s8:--steps 8" "s6:--steps 6" "s4:--steps 4"`) and judge by EAR — flow models
  often hold at 6. Linear DiT win if quality survives. (Timing-only A/B is noisy; quality is the gate.)
- **SAME decode (bigger half of 380s = 12s):** never tuned. Sweep chunk size (`sa3same_decode_chunked`
  chunk=128 default) — fewer/larger chunks = less overlap waste; profile the weightnorm convs with
  `./perf.sh nsys`. Watch the long-form mask memory (full-attn OOMs; chunking is load-bearing).
- **DiT L² attention (other half):** flash already fused it; the QK² flops are inherent. No cheap lever
  without windowing (quality risk). 380s is the edge case (game audio = short SFX + loops) — low ROI.
- **Minor:** pre-store K/V as F16 inside `sa3_heads`/`sa3_heads_dyt` to drop the flash F16-cast copies
  (`cpy_scalar` 4.4% + `cpy_perm` 7.3% in the flash profile). ~5–8%, mechanical.
- `mul_mat_q` (Q8 weights) is the 31% floor — at-floor on this GPU (cf. acestep). No lever at Q8.

---
## Gotchas / rules (validated)
- **Flash is default-on for GPU** via `sa3_set_flash(bp.has_gpu)` in both model loads. Keep the
  materialized F32 path as the oracle (flash F16-accumulates → CPU drift). New attention features must
  go through `sa3_diff_attn` so they get flash for free.
- **pingpong sampler is stochastic** → e2e audio NOT bit-matchable; validate per-component on deterministic
  goldens (euler / DiT-step / encode / decode), like the qwen3-tts 2-seed rule.
- **Conditioner tensors stay F32** in quant ggufs (`sec.emb*`/`pad_embed`/`mem_tokens`/`new_tokens`) —
  `quantize.cpp` already keeps them; `sa3-gen` reads them F32 from the SAME gguf.
- **Quant:** all-Q8 ≈2.67GB (lossless-grade) or all-Q6 ≈2.1GB both fit ≤3GB. imatrix REGRESSES this DiT —
  use flat-Q. Regenerate quant ggufs from current F32 if conditioner/tokenizer tensors changed (loop in
  SA3-PORT-PLAN.md "Max length" note).
- Binaries land in `build-cuda/<name>` (NOT `bin/`). GGML_MAX_NAME=64 (short tensor names).
- GPU is single — stop heavy prod containers before a long bench; coordinate (owner sanctioned GPU use,
  will say when needed elsewhere). `:8097` is taken by longcat's nava_eyetest_server — ear page uses `:8103`.

## NOT this phase (prod — comes after features+perf)
sa3-server (async + cancel + warm fork/IPC worker-isolation, mirror songgen-server) · prod Dockerfile ·
koblem light-GPU gate + UI · FF-merge sa3-port → acestep.cpp master + bump kobbler ref.
**Product decision (made):** SA3 folds in **under "music"** beside acestep (NOT a separate engine) — it's a
selectable model/mode within the existing music surface, sharing the light-GPU gate.
