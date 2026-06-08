# Stable Audio 3 Medium → acestep.cpp port

Worktree `acestep-sa3` / branch `sa3-port`. Goal: text-to-audio (game SFX/music) in the
**light GPU group (~3 GB, ideally ≤3)**, CPU-only dev for now, merged **flat** into
acestep.cpp master without disturbing acestep/songgen (mirror the `songgen-*.h` convention →
all new files prefixed `sa3-`).

Reference (gated; token in `kobbler/.env`): `stabilityai/stable-audio-3-medium`.
Downloading to `checkpoints/sa3-medium/` (see `.sa3ref/download.log`). Architecture
already extracted to `.sa3ref/` (`model_config.json`, `st_header.json`, `t5gemma_config.json`).

## What it is (no surprises — confirmed from model_config.json + safetensors header)

Pure latent-diffusion, **no autoregressive LM** (simpler than acestep/songgen). 2.305 B params F32.
Three components, all F32 in the checkpoint:

| Component | Source prefix | Role |
|---|---|---|
| **DiT** (1.4 B) | `model.model.*` | rf_denoiser flow-matching, 8 steps |
| **SAME AE** (taae_v2) | `pretransform.model.*` | 256-dim latent ↔ stereo 44.1 kHz, downsample 4096 |
| **conditioner** (3 tensors) | `conditioner.*` | learned prompt-pad embed + seconds_total fourier |
| **T5Gemma-b-b-ul2** | `t5gemma-b-b-ul2/` (separate) | text encoder → 768-dim, max_length 256 |

### DiT (`model.model`, embed 1536 / depth 24 / heads 24 / hd 64)
- `project_in` 256→1536, `project_out` 1536→256; `preprocess_conv`/`postprocess_conv` [256,256,1].
- `memory_tokens` **[64,1536]** — learned tokens prepended to the sequence (NEW vs acestep).
- Per layer (`...transformer.layers.N`):
  - `self_attn.to_qkv` **[7680,1536]** = Q3072 + K3072 + V1536 → **differential attention** (Q,K doubled). `to_out`[1536,1536]. `q_norm/k_norm.gamma`[64] rms.
  - `cross_attn.to_q`[3072,1536], `to_kv`[4608,1536] (K3072+V1536, differential), `to_out`, q/k_norm. Cross-attends to `[prompt_tokens ; seconds_total_token]`.
  - `ff.ff.0.proj`[12288,1536]=GLU(gate+up, inter 6144) **+bias**; `ff.ff.2`[1536,6144]+bias.
  - `pre_norm/cross_attend_norm/ff_norm.gamma`[1536] rms.
  - `to_scale_shift_gate`[9216]=6×1536 AdaLN table; `to_local_embed` 257→1536→1536 (inpaint; feed zeros for t2a, replicate bias).
- Heads: `to_timestep_embed` 256→1536→1536 (timestep expo-fourier), `to_cond_embed`/`to_global_embed` 768→1536→1536, `global_cond_embedder` 1536→1536→9216, `rotary_pos_emb.inv_freq`[16].
- `global_cond_type: adaLN`, `diffusion_objective: rf_denoiser`, `mask_padding_attention`.

### SAME autoencoder (single stage! c_mults [6], strides [16], depth 12, 256-dim latent)
- **Decoder (t2a path, port first):** `decoder.layers.1` linear 256→1536 → 12× differential-attn transformer (RoPE, `transformers.M`) → `decoder.layers.3.mapping` **weightnorm** conv (weight_g/weight_v) 1536→512 + `new_tokens`[1,1,1536] → unpatch(256)/stride16 → stereo audio.
- Bottleneck **softnorm**: `scaling_factor`/`running_std`/`noise_scaling_factor`, `auto_scale`.
- Encoder mirrors (patch 256 → weightnorm conv 512→1536 → 12× transformer → proj 256). **Defer** — only needed for audio2audio / inpaint / continuation.
- `chunked: True` decode → caps activation VRAM (key for 3 GB budget).
- AE transformer norm is gamma+beta+**alpha** scalar (learnable-scaled), differs from DiT's plain rms gamma — handle both.

### T5Gemma encoder (`t5gemma_config.json`)
Gemma2-style **encoder only** (drop decoder): hidden 768, 12 layers, 12 heads hd64, inter 2048,
`gelu_pytorch_tanh`, RMS eps 1e-6, RoPE θ10000, alternating **sliding(4096)/full** attention,
`attn_logit_softcapping 50`, `query_pre_attn_scalar 64`, vocab 256000, **bidirectional (no causal mask)**.
Gemma SentencePiece tokenizer (256k). Output 768-dim → cross-attn cond. Template: `src/qwen3-enc.h` + `src/bpe.h`.

## Component → host-file mapping (clone, don't edit originals)

| New file | Cloned/templated from | Notes |
|---|---|---|
| `src/sa3-attn.h` | `dit.h` attn section | **shared** differential-attn + memory-token block (DiT & SAME) — the core new work |
| `src/sa3-dit.h` / `sa3-dit-graph.h` | `dit.h`/`dit-graph.h`/`dit-sampler.h` | reuse `solvers/` (euler), rf_denoiser schedule |
| `src/sa3-same.h` (+`-enc.h`) | `vae.h`/`songgen-vae.h` | weightnorm conv, softnorm bottleneck, chunked decode |
| `src/sa3-t5gemma-enc.h` | `qwen3-enc.h`+`bpe.h` | Gemma2 encoder, softcapping, sliding/full |
| `src/sa3-pipeline.{h,cpp}` | `pipeline-synth*` | T5Gemma→DiT→SAME→wav |
| `tools/sa3-gen.cpp` | `tools/songgen-generate.cpp` | CLI; add CMake target, leave others untouched |
| `convert.py` (extend) | existing | add `sa3` arch classifier + tensor remaps + weightnorm fuse |

## VRAM budget — 3 GB is comfortable

Game SFX = short sequences → activations tiny (the 4096-token / 380 s case is the only stressor, and chunked AE decode + flash attn keep it bounded). Weights at Q8 imatrix:
- DiT 1.4 B → ~1.5 GB (Q8) / ~1.15 GB (Q6_K) / ~0.8 GB (Q4_K)
- T5Gemma-b enc ~0.3 GB (Q8); 256k embd table is the swing — keep Q8/Q6.
- SAME decoder ~0.3–0.4 GB (Q8); keep conv/bottleneck high-precision.

**Target ~2.2–2.8 GB at Q8-ish** → fits ≤3 with headroom. The imatrix ladder (below) picks the
exact mix. Reuse songgen `worker-isolation` + idle-unload for true-0 idle VRAM.

## Python goldens (CPU) — the oracle [task #1]
venv with `stable_audio_3` (PyTorch 2.7.1). Medium "requires flash-attn 2" → **patch attn to eager**
for CPU (mathematically equiv to F32 reference; our port does plain attention anyway). Dump npy at every
boundary: tokenizer ids, T5Gemma enc out, seconds fourier, DiT project_in, **per-step latents ×8**, final
latent, SAME decode audio. Drives per-component cosine validation AND is the fidelity target for the ladder.

## imatrix TF% quant ladder [task #8] — NOT flat Q's
Lesson learned: flat Q4_K hurt quality. Plan:
1. **imatrix**: during calibration forwards (diverse SFX+music prompts), accumulate per-matmul input
   activation importance (Σ x²) per tensor. ggml's `ggml_quantize_chunk` takes an optional importance
   matrix — wire it into the `quantize` tool path (currently flat-only).
2. **TF% ladder**: for each config (IQ4_XS / Q4_K / Q5_K / Q6_K / Q8_0, with per-tensor overrides — keep
   norms/bottleneck/embd high), measure **TF% = cosine(final latent, F32 golden)** and **cosine(decoded
   audio mel, F32 golden)**. Tabulate size vs TF%.
3. Pick smallest config that holds quality AND fits ~3 GB. Record as a ladder doc (like the other ports).
   Per-component imatrix (DiT matters most; SAME conv/bottleneck stay high-precision).

## Reference findings — VALIDATED against real model (F32, GPU) [task #1 ✓]
Golden oracle in `.sa3ref/goldens/` (32 npy, run by `.sa3ref/gen_goldens.py`; prompt "8-bit retro
arcade power-up jingle…", 10 s, 8 steps, seed 0). Reference code in `.sa3ref/stable-audio-3/`.

- **Differential attention = plain subtract, NO learned λ:** `out = attn(q,k,v) − attn(q_diff,k_diff,v)`,
  shared V. Self: `to_qkv`→5·dim chunk(q,k,v,q_diff,k_diff). Cross: `to_q`→2·dim, `to_kv`→3·dim_kv. rms qk-norm per head_dim. (transformer.py `Attention`).
- **AdaLN block:** `scale_self,shift_self,gate_self,scale_ff,shift_ff,gate_ff = (to_scale_shift_gate + global_cond).chunk(6)`.
  Modulate: `x = pre_norm(x)*(1+scale)+shift`; **gate is `x*sigmoid(1−gate)`** (not plain gate); residual add. Cross-attn has NO adaLN (just `cross_attend_norm`). `layer_scale`/zero-init scales are **Identity** in medium (no-ops).
- **RoPE includes the 64 memory tokens** (positions computed after prepend). Memory tokens prepended, stripped before `project_out`. DiT RoPE dim = `max(dim_heads//2,32)` ; `inv_freq[16]`.
- **DiT global cond:** `global_cond = to_global_embed(seconds_768) + to_timestep_embed(ExpoFourier(t))`, then `global_cond_embedder`(1536→1536→9216) per layer. timestep ExpoFourier(dim256,0.5,10000). `t` kept **fp32** (logsnr sensitivity). objective `rf_denoiser`: model returns velocity `v`; `denoised = x − t·v`.
- **cross context = [prompt(256) ; seconds(1)] = 257 tokens**, pre-`to_cond_embed`. Padding positions use **learned `padding_embedding`** (padding_mode "learned"), not zeros. `global_cond == cross_cond[:,256]` exactly (shared seconds embed).
- **Sampler default for rf_denoiser = `pingpong` (STOCHASTIC):** `denoised = x − t·v; x = (1−t_next)·denoised + t_next·randn()`. Last step t_next=0 ⇒ deterministic (`latent_final == x7 − t7·v7`, verified err 0.0). Schedule (dist-shift warped, NOT linear): `[1.0,.994,.985,.958,.891,.746,.513,.274]`. Euler (`x += (t_next−t)·v`) available + deterministic.
  → **Validation rule:** end-to-end audio canNOT be bit-matched (pingpong RNG); validate **per-component** (T5Gemma(tokens)→t5_out; DiT(x_in_i,t_i,cond)→v_out_i; SAME.decode(latent_final)→audio) which ARE deterministic. Like the qwen3-tts 2-seed rule for e2e.
- **SAME AE attention uses sliding-window (band, w=[1,1]) via flex/SDPA mask** + RoPE + differential; norm has gamma+beta+**alpha scalar**. Decoder also has `new_tokens` learned token + weightnorm (weight_g/weight_v) conv. Chunked decode confirmed.
- **T5Gemma:** Gemma SentencePiece, pad id 0; prompt → 256 padded ids (mask marks valid). Encoder-only (`T5GemmaEncoderModel`, `is_encoder_decoder=False`), output `last_hidden_state` 768. proj_out is Identity (768==768).

Golden manifest: tokens/attn_mask[1,256], t5gemma_out[1,256,768], cross_cond[1,257,768], global_cond[1,768], step{0..7}_{t,xin,vout} ([1,256,174]), latent_final[1,256,174], audio_final[1,2,441000].

## Quant TF% ladder (flat K-quant, no imatrix yet) — [task #8]
Per-component golden cosine (`.sa3ref/run_ladder.sh`); fix: mem_tokens stays F32 (cast in graph; it's concatenated, not a matmul weight).

| comp | F32 | Q8_0 | Q6_K | Q5_K_S | Q4_K_S |
|---|---|---|---|---|---|
| t5gemma | .99995/1.1G | .99972/302M | .99828/237M | .99571/201M | .98688/168M |
| dit | .99987/5.5G | .99941/1.5G | .99635/1.2G | .99607/986M | .97220/814M |
| same | .99998/3.2G | .99978/870M | .99906/673M | .99773/566M | .99215/464M |

**Weights VRAM totals:** Q8 **2.67GB** (min cos .9994, lossless-grade) · Q6 **2.11GB** (min cos .9964) · Q5 1.75GB · Q4 1.45GB (DiT drops to .972 — most quant-sensitive).
**Recommendation for ≤3GB light group:** all-**Q6 ≈ 2.1GB** (≥.996 everywhere) is the sweet spot; all-Q8 (2.67GB) for max fidelity. Game SFX = short seq → tiny activations, so even Q8 fits with margin.

### imatrix — IMPLEMENTED, but NOT beneficial here (flat already near-lossless)
Built `sa3-imatrix.h` (sched eval-callback collector, per-matmul Σx² per input channel) + `tools/sa3-imatrix-collect` (12-prompt calibration sweep) + `tools/quantize` 4th-arg imatrix → `ggml_quantize_chunk`. Result: imatrix **regressed** DiT (Q5 .996→.980, Q4 .972→.967), reproducible. Diagnosis: flat K-quant already near-lossless on this DiT (unlike LLMs where flat Q4 hurts — the original lesson), so imatrix has no room; plus rough edges — zero-input inpaint `loc.*` tensors get all-zero importance (degenerate quant) and calibration likely over-concentrates precision. **Not needed for the budget.** If revisited: exclude/clamp zero-imatrix tensors, weight calibration toward late (low-t) steps. Files: `models/sa3-dit-*-imx.gguf`, `models/sa3-dit.imatrix`.

### Max length + VRAM-at-length (Q8, measured)
Max audio = **380 s** (sample_size 16,777,216 / 44.1kHz = **4096 latent frames**; dist_shift max_length 4096).
`sa3-gen` loads each model then frees it (sequential), so peak = max phase (weights + ggml compute-buffer).
Compute-buffer is backend-independent → probed on CPU via `SA3_PROBE=1` (`ggml_backend_sched_get_buffer_size`).
DiT compute-buffer scales with L² (full self-attention, materialized scores):

| duration | L | DiT compute-buf | peak Q8 (DiT phase: ~1430MiB wt + buf) |
|---|---|---|---|
| 4 s (SFX) | 108 | 22 MiB | **~1.45 GB** |
| 30 s | 388 | 51 MiB | ~1.48 GB |
| 120 s | 1358 | 290 MiB | ~1.72 GB |
| 380 s (MAX) | 4156 | 1915 MiB | **~3.35 GB** |

(+ ~0.3–0.5 GB CUDA context on GPU. SAME phase is constant ~1.4 GB via chunked decode, chunk=128 → 592 MiB buf.)
**SFX/short = ~1.5 GB Q8 (huge margin). 380s max = ~3.35 GB Q8, slightly over 3** — driven by the DiT's
materialized-scores attention (1.66 GB of the 1.9 GB). **Fix for long-form ≤3GB: switch DiT to `ggml_flash_attn_ext`**
(O(S) memory, no [S,S] scores) → max-length DiT buf drops to ~few-hundred MiB → ~1.7 GB peak even at 380s.
Or Q6 DiT → 380s ≈ 3.05 GB. SAME long-form needs chunked decode (`sa3same_decode_chunked`, added; full-attn OOMs: 19GB mask at N=69632).
Conditioner-on-quant: **FIXED** — `quantize` now keeps `sec.emb*/pad_embed/mem_tokens/new_tokens` F32 (tiny, not matmul weights, ~789KB). Full pipeline runs end-to-end on Q8 weights (verified: coin SFX, healthy audio). NOTE: re-quant ALL levels from the current F32 ggufs (the quant ggufs were made before tokenizer-embed + conditioner-fix; t5gemma Q8 already re-done). Regenerate: `for c in dit same t5gemma; do for q in Q8_0 Q6_K Q5_K_S Q4_K_S; do ./build/quantize models/sa3-$c-f32.gguf models/sa3-$c-$q.gguf $q; done; done`.

---
# ===== GPU ENABLEMENT + PERF (2026-06-08, DONE) =====
Port now **runs on GPU as a real service-style dev loop**, has an **ear-test page**, is **deep-profiled**,
and the headline speed lever (**flash attention**) is **implemented + default-on for GPU + validated**.

### Dev/perf harness — `kobbler/docker/acestep-sa3-dev/`
- `Dockerfile` — CUDA 12.9.1 devel + cmake + ccache (== longcat-avatar-dev; reuse `longcat-avatar-dev:builder` to skip rebuild).
- `iter.sh` — `./iter.sh build` (CUDA build of `sa3-gen`+tests+quantize into `build-cuda/`), `./iter.sh gen -- <args>`,
  `./iter.sh test <dit|same|t5gemma> -- <gguf> <goldens> [step]`, `./iter.sh shell`. Auto-forwards `SA3_*`/`GGML_*` env.
  Binaries land in `build-cuda/<name>` (NOT `bin/`). GPU build = CUDA backend auto-selected (`ggml_backend_init_best`).
- `eartest.sh [PORT] "LABEL:EXTRA" ...` — matrix of {prompt × config} → side-by-side `<audio>` + per-render
  total/RTF/DiT-per-step/peak+net VRAM table + VRAM sparkline (3GB budget line). Configs override the all-Q8 base;
  EXTRA may embed `SA3_*`/`GGML_*` k=v tokens (pulled to `-e`) so env-gated levers A/B in one page
  (e.g. `"flash:" "materialized:SA3_FLASH=0"`). Default prompts: jingle/laser/ambient/drums. Live at :8097.
- `perf.sh nsys [sec]` / `perf.sh ncu <kernel-regex> [skip] [sec]` — nsys kernel-time summary + ncu SoL/Occ/Mem;
  toolchain at `/mnt/hdd/3d/avatar-shootout/toolchain` mounted `/tc`, ncu needs `--cap-add SYS_ADMIN`.

### sa3-gen instrumentation
`tools/sa3-gen.cpp` now prints `[sa3-timing]` lines (per-phase load vs compute, per-step DiT, loads-vs-compute split,
total + RTF=wall/audio) parsed by eartest. `[sa3-timing] step i/N (t=..) <s>` per step.

### GPU baseline (Q8, RTX 3060) — load vs compute split
Loads are a CONSTANT ~0.49s (t5 0.19 + dit 0.21 + same 0.09 = per-invocation GPU weight upload). For SFX (≤4s)
that's ~40% of wall → **the short-clip lever is a persistent/warm process** (deferred server scope), not a kernel.
Compute scales with duration; **SAME decode ≈ or > DiT steps at every length** (10s: 0.60 vs 0.57; 120s: 5.3 vs 4.3).

### nsys profile (30s render) — attention was ~49% of GPU
materialized: `soft_max_f32` **31.9%** + cutlass attn GEMMs 10.2%+7.0% ≈ **49%**; `mul_mat_q` (Q8 weights) 18.4%.
→ flash fuses QK^T/softmax/·V, killing softmax + both attn GEMMs.

### LEVER #1 — flash attention (`ggml_flash_attn_ext`) — SHIPPED, default-on GPU
`src/sa3-attn.h`: `sa3_set_flash(has_gpu)` (called in both model loads) flips `sa3_diff_attn` to a flash path
(2 flash calls q/k + qd/kd sharing F16 V, then subtract → identical `[E,Tq]` layout). Default-ON for GPU,
`SA3_FLASH=0` opt-out; materialized F32 path stays the CPU/oracle (flash F16-accumulates → CPU drift). DiT mask=NULL,
SAME band-mask cast→F16. set_prec(GGML_PREC_F32).
- **Validated vs goldens (GPU):** DiT step0 .99981 / step7 .99969, SAME .99996 (vs materialized .99986/.99969/.99998 —
  4th-decimal deltas, no quality loss). Full euler `--cmp` final-latent **.9921 (flash) vs .9768 (materialized)** —
  flash is CLOSER to the golden because the reference uses flash-attn-2.
- **Speed (Q8 total wall):** 10s 1.78→1.34s (−25%), 30s 3.06→2.06s (−33%), 120s 10.21→5.32s (−48%). SAME decode −52..55%.
- **VRAM compute-buffer @380s:** DiT 1915→**458 MiB**, SAME 592→**209 MiB** → 380s max-length peak ≈ Q8 weights 1.43GB + 458 ≈ **~1.9 GB**, full duration range fits ≤3GB.
- **nsys after flash:** softmax + attn GEMMs GONE; `flash_attn_ext_f16` 15.5%, `mul_mat_q` 31.2% now dominant (Q8 weight
  matmul = at-floor on this GPU, cf. acestep `mul_mat_q`). Remaining time = optimal kernels; no further math lever at Q8.

**NEXT levers (not yet done):** (a) **persistent/warm server** — kills the ~0.49s load tax that dominates SFX-length wall
(deferred server scope, the real short-clip win); (b) head-permute/F16-cast copies (`cpy_perm` 7.3% + `cpy_scalar` 4.4%)
are flash's minor overhead — low value; (c) Q6 DiT for −VRAM if 380s headroom ever matters (already fits at Q8).
Re-quant note from below still applies if regenerating quant ggufs.

---
# ===== HANDOFF: NEXT PHASE (fresh agent) =====
Port is FUNCTIONALLY COMPLETE & validated (T5Gemma .99995 / DiT .9997 / SAME .99998 / tokenizer exact / full `sa3-gen` CLI works on F32 and Q8). Worktree `acestep-sa3` (branch `sa3-port`), UNCOMMITTED. All work is `sa3-*` prefixed, additive CMake targets, acestep/songgen untouched. Remaining goals (user wants GENERAL use, not just SFX):

1. **Docker build + run (iter.sh style)** — host can't build CUDA (nvcc 13.3 toolchain fails compiler-ID; the cpp forks ALL build CUDA in docker). Mirror an existing builder: `kobbler/docker/longcat-avatar-dev/iter.sh` and `kobbler/docker/flux2-dev/` (base `nvidia/cuda:12.9.1`, sm_86). acestep already has a prod docker (songgen) under kobbler/docker — find it. Build sa3-gen + tests in-container, run on GPU.
2. **Ear-test page** — clone the pattern of `kobbler/docker/longcat-avatar-dev/compare.sh` (renders N configs, serves side-by-side HTML w/ players + timing table + VRAM sparkline w/ budget line; owner likes it). Adapt for audio: N prompts × quant levels → `<audio>` players + per-render **TTFA, RTF, peak VRAM** table. Output to a mounted dir; one GPU job at a time.
3. **Conditioner quant** — DONE (kept F32). No action.
4. **Full perf profile (ncu + nsys) + VRAM breakdown + fix levers** — toolchain has ncu/nsys bundled (see `reference_ncu_docker_syadmin`: ncu needs `--cap-add SYS_ADMIN`; sudo on host). Reuse scripts from `kobbler/docker/longcat-avatar-dev/`: `ncu_hot.sh`, `nsys_*.sh`, `breakdown.sh`, `mem_vram_trace.sh`. Known levers already identified:
   - **DiT → `ggml_flash_attn_ext`** (BIGGEST): currently materializes [S,S,Nh] scores (1.66 GB at L=4156) via manual `sa3_attn_once`. Flash = O(S) mem → ~1.7 GB peak at 380s Q8 (vs 3.35). Also faster. Differential = 2 flash calls (q/k and qd/kd, shared V) then subtract. Mirror qwen3-enc.h's flash path (cast K/V→F16, GGML_PREC_F32). Validate vs goldens after.
   - **SAME band attention**: per-chunk full-O(N²) (N=2176 @ chunk128). Could flash + sliding-window; lower priority (already bounded/constant ~1.4GB).
   - mem_tokens in-graph `sa3_f32` cast now redundant (F32 in gguf) — can drop.
   - Probe tool: `SA3_PROBE=1` prints ggml compute-buffer per phase (backend-independent VRAM proxy).
   - VRAM/length table + max (380s/L4096) measured — see above.

Validation harnesses (CPU, fast): `sa3-{t5gemma,dit,same,tok}-test`, `sa3-pipeline-test`, all vs `.sa3ref/goldens`. Build: `cd build && cmake .. && cmake --build . --target X`. Re-gen goldens: `.sa3ref/gen_goldens.py` (+ `gen_euler_golden.py`). Python venv: `.sa3ref/stable-audio-3/.venv` (torch 2.7.1 cu126). HF token in `kobbler/.env`.
Prod wiring (later): koblem HeavyKind gate + worker-isolation like songgen (`project_acestep_worker_isolation`).

### Perf/VRAM measurement note
Host has no `nvcc` on PATH (CUDA builds for the cpp forks go through docker iter.sh; nvcc only in pixi envs) → GPU build/timing deferred to the docker path. VRAM figures above are weights-only (exact); activation overhead is small for short clips (the only large activation is SAME decode's N=17·T band attention, fine for SFX-length T). Wall-time on GPU TBD via docker builder.

## Build/merge discipline
- CPU build only here: `./buildcpu.sh` (cpp forks build fine on server; Rust no-build rule doesn't apply).
- Add CMake targets additively; never modify acestep/songgen targets. Goal = clean `sa3-*` superset that
  FF-merges to master like songgen did.
- ggml already at consolidated `292516d5` — differential attn / softcapping may need new graph code but
  no submodule bump expected; if a kernel is missing, prefer graph-level composition first.

## Status / next
- [x] #1 goldens (32 npy oracle) ; [x] #2 convert-sa3.py → 3 GGUFs (DiT 5.4G / SAME 3.2G / T5Gemma 1.05G f32, shapes verified)
- [x] #4 **T5Gemma encoder PASS** — `tools/sa3-t5gemma-test`, cossim 0.99995 (all)/0.99991 (valid). softcap OFF (SDPA path) correct. Build: plain CPU (`cmake .. && cmake --build build --target X`; BLAS off — host lacks cblas; CUDA via buildcuda.sh when wanted).
- [x] #3 shared diff-attn (`sa3-attn.h`) + [x] #5 **DiT PASS** — `tools/sa3-dit-test`, cossim step0 0.99987 / step4 0.99976 / step7 0.99969 (all >0.999). First build correct: differential attn (plain subtract, shared V), AdaLN `x*(1+s)+sh` + gate `sigmoid(1-g)`, partial NEOX rope n_dims=32, 64 mem tokens, swapped GLU `first*silu(second)`, local-cond `to_local_embed(0)` constant. (cosine dips slightly at low t — fine.)
- [x] #6 **SAME decode PASS** — `tools/sa3-same-test`, cossim 0.99998 (rmse 9e-4), exact sample count. First build correct: DynamicTanh norms, differential attn + band mask |i-j|<=17 over N=17*T, new-tokens upsample (1 real+16 learned per frame, keep last 16), Sin(pi*) gate blocks 5-11 / SiLU 0-4, weightnorm conv, channel-major unpatch, bottleneck = z*running_std only.
- **ALL 3 components validated** (T5Gemma 0.99995, DiT 0.9997, SAME 0.99998). Toolchain fully proven.
- [~] #7 **pipeline core WORKS** — `tools/sa3-pipeline-test`: euler 8-step (schedule `sigmoid(t_lin*8.2-2)`, LogSNRShift rate=0, seq-len independent) → SAME decode → wav. Deterministic final latent vs euler golden cossim 0.9986 (8-step free-running accumulation of ~0.9997/step DiT). Produces valid 10s stereo 44.1k wav (`out/sa3_euler.wav`). Used golden cond+noise. Euler golden via `.sa3ref/gen_euler_golden.py`.
- [x] Gemma BPE tokenizer (`sa3-tokenizer.h`, `tools/sa3-tok-test`) — EXACT match to golden tokens (15/15, 0 mismatch). HF BPE: normalize space->U+2581, whole-string word, merge-by-rank, byte_fallback, ignore_merges, no BOS. vocab+merges embedded in t5gemma gguf (tokenizer.ggml.tokens/merges, 256k/580k).
- [x] **#7 DONE — `tools/sa3-gen` full CLI works**: tokenize→T5Gemma→cond assembly→sampler→SAME→wav. Validated: cross_cond cossim 0.999907, seconds-embedder (global) 1.000000 vs golden. euler full-chain latent 0.9758 (CPU T5Gemma vs GPU golden compounding over 8 steps — not a bug; output is a valid render). Arbitrary prompts + euler/pingpong + seed RNG + duration-driven L all working. Args: --t5/--dit/--same/--prompt/--seconds/--steps/--seed/--sampler/--out/--noise/--cmp.
- [ ] OLD remaining #7 notes: (a) DONE tokenizer (tokenizer.model in `sa3.tokenizer.spm_bytes`); (b) conditioning assembly from scratch (seconds NumberConditioner + learned padding -> cross_cond[257,768]+global[768], validate vs golden cross_cond/global); (c) C++ RNG noise + pingpong sampler; (d) real `sa3-gen` CLI + maybe koblem wiring. Then #8 imatrix ladder. Old detail below:
- [ ] OLD next-notes: #7 pipeline + `sa3-gen` CLI. Sub-pieces: (a) Gemma SPM tokenizer (tokenizer.model embedded in gguf as `sa3.tokenizer.spm_bytes`; need C++ unigram) — or stage with golden tokens first; (b) seconds NumberConditioner (normalize (s-0)/(384), ExpoFourier(256,0.5,10000)? NO — NumberEmbedder dim=256 default, check; Linear `sec.emb` 256->768) for cross token[768] + global[768]; (c) prompt learned-padding (`prompt.pad_embed`) -> cross_cond[257,768]; (d) dist-shift schedule (port distribution_shift.py) + euler & pingpong samplers; (e) noise init + SAME decode + wav. → #8 imatrix ladder.
- Build mechanics learned: header-only test tools = `add_executable + link_ggml_backends`; GGML_MAX_NAME=64 (drove short tensor names); ggml reversed-dim => numpy [out,in,k] conv & [out,in] linear store as-is.
