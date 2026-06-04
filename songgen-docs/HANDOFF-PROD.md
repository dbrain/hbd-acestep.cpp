# songgen.cpp — PROD READINESS HANDOFF (single source of truth)

RTX 3060 (Ampere/CUDA) bring-up of the SongGeneration port. This doc is the running
checklist of what works, what's broken, and what must be resolved before production.
Deep perf/VRAM detail lives in `PERF-3060.md`; this is the prod-gating summary.

Worktree: `~/dev/songgen.cpp` (branch `songgen-port`). Models: `~/dev/songgen-data/{gguf,golden-large}`.
Build/run: `songgen-dev:builder` (CUDA 12.9.1), sm_86. Harness: `~/dev/songgen-data/` (serve.sh,
profile_gen.py, profile_stem.py, wav_metric.py, make_eartest.py). Ear-test page: http://10.0.0.208:8110.

## ✅ WORKING + validated (ear-confirmed)
- **Core text→song** (`/generate`). Optimized config **Q4_K_M + flash-attn + Q8_0 KV + chunk_frames=200**
  = **3931 MiB @30s, RTF 1.61**, ear-confirmed "sounds good", chunk joins inaudible. (Baseline was 9235/1.97.)
- All 7 core graphs re-validated on CUDA (LeLM cb0 1.0, CFM/VAE/RVQ/MusicFM pass).
- Server endpoints `/generate /clone /continue /separate` implemented (multipart upload).

## 🔴 PROD-BLOCKING / must-resolve before shipping clone+continue
### 1. Clone/continue CONDITIONING is wrong + UNVALIDATED  (task #6)
Symptoms (ear, Q4, real chorus prompt): clone = "close bgm, but **voice off (male vs female prompt)** and
**doesn't match the lyric**"; continue = "**like playing a record backwards**".
- Inputs are now CLEAN (see #2 fixed) — these are genuine CONDITIONING-LOGIC failures, not input garbage.
- The clone/continue end-to-end output was **NEVER validated against the Python reference** (handoff calls it
  "fuzzy validation"; only the encoders were checked, and only at 83–88% frame-match).
- The prompt-audio path is shared by clone AND continue → likely the same root cause.
- **Right fix = reference-golden method** (same as every other graph): on a torch box (dev box 10.0.0.151 has
  it) capture the Python reference clone/continue tokens+audio for identical inputs, then bisect where the
  port's `sggen` conditioning diverges (audio_qt_embs cb0=pmt/cb1=vocal/cb2=bgm gather, EOT prepend, 16385
  masking in lm_levo.prepare_condition_tensors:222-230, CFG handling of the audio prompt). Debugging blind
  (no golden) is unreliable — do NOT ship clone/continue on blind guesses.
- USAGE NOTE (real, not a bug): prompt audio must be a **~10s clip around the chorus** (prompt_len=10s,
  audio_len=252). Passing a full song silently uses only its first 10s (often the intro → no vocal to clone).
  The harness now auto-picks the peak-vocal-energy 10s window (`stems/chorus_*.wav`).

### 2. ✅ FIXED — htdemucs separation was feeding garbage (was the "complete junk" cause)
- htdemucs forward is **CUDA-specific broken**: junk output, cossim ~0.001 vs golden, ~60× over-scaled →
  91% clipped. **Correct on CPU** (cossim 0.9957/0.9918). Slipped through because block goldens are
  cosine-validated (scale-invariant) and the CUDA e2e gate never ran (missing fine goldens).
- SHIPPED WORKAROUND: `/separate` forces `GGML_BACKEND=CPU` (19s for a 30s song). Single-file→clone works.
- **Follow-up (task #10):** fix the CUDA htdemucs forward so GPU separation (~4s) returns. Bisect blocks
  CUDA-vs-CPU; suspects = conv_transpose/col2im or the cross-transformer FA at head_dim 64.

## 🟡 PERF/VRAM — IN PROGRESS (task #8, the current focus)
User wants to **stay on Q8** for benchmarking and push VRAM/perf hard. Targets/constraints:
- Two VRAM buckets: **~3.5 GB "light GPU guard"** (may peak slightly higher, ideal 3.5 max) vs **7.5 GB** fallback.
- Want **multi-minute songs (180s)**. Model self-terminates on EOS at a ~250s ceiling — **`--duration` just
  truncates**; expose a high ceiling, not a target length (task #9).
- KV is currently pre-allocated to the duration ceiling → a 250s ceiling = 250s of KV VRAM even if the song
  ends early. **Dynamic/growing KV** is the key enabler for 180s in a tight bucket.
- Open questions to answer: **can we offload weights (sd.cpp streams them)? how much VRAM does it save, at
  what perf cost?** Where does Q8 land in each bucket (current Q8 LM stage ≈ 5.1 GB wts + KV)?
- LM decode floor (ncu): `mul_mat_vec_q` at ~70–75% of compute+memory roofline (weight-load bound, N=2).
  Architectural-only speedups left: smaller weights (Q4) or speculative/multi-token decode.

## ⚪ OTHER FOLLOW-UPS
- **task #7 — q4_0 KV crash**: SG_KV_TYPE=q4_0 → CUDA illegal access under FA. q8_0 works (shipped). q4_0
  would ~halve KV again (→ ideal 3.5 GB + long songs). Quality impact of q8_0→q4_0 KV: untested (q8_0 KV
  passed the logit gate >0.999, ~negligible; q4_0 would be more lossy — worth an ear A/B once it runs).
- **Quant verdict**: Q8 weights (5.1 GB) can't fit 4 GB; Q4_K_M is the only quant under 4 GB and the user
  judged it audibly OK. Q8 is the quality option for the 7.5 GB bucket. (User will A/B Q8 vs Q4 to pick prod.)

## ENV KNOBS (all default-safe; deploy/serve.sh sets the budget config)
- `SGLM_NO_FA=1` — revert LM decode to manual attention (FA is default).
- `SG_KV_TYPE=f16|q8_0|q4_0` — KV cache type (default f16 in code; serve.sh deploy = q8_0).
- `SG_CHUNK_FRAMES=N` — decode chunk size (default 750 in code; serve.sh deploy = 200).
- `SGLM_PROFILE=1` — per-step LM timing. `HTD_DEBUG_RMS=1` — htdemucs stem RMS debug.
