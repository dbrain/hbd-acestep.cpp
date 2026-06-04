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

## 🟢 PERF/VRAM — task #8/#9 LANDED (dynamic KV + model-max ceiling + offload verdict)
Detail in `PERF-3060.md` Levers 4–6. Summary:
- **Dynamic/growing KV (DONE, bit-identical)**: KV starts small (≈1207) and grows on demand up to the
  ceiling, host-bridged (no co-resident old+new buffer → no grow transient spike), geometric growth with
  per-grow increment capped at `SG_KV_GROW_BLOCK` (default 1500 frames). `wav_metric cmp` = corr 1.0000 /
  RMSE 0.0000 vs static. This decouples the ceiling from the footprint.
- **Model-max EOS ceiling (task #9, DONE)**: upstream has no duration target — generates to `max_dur`
  (270 s = 6750 frames) and stops on EOS. `songgen-generate` with no `--duration` now defaults to that
  (`SGGEN_MODEL_MAX_FRAMES`); server `/generate` with no `duration` field does the same. `--duration` is
  now a hard cap (truncate), not a target. **Song length is lyric-driven**: short golden prompt EOSes ~35 s,
  a 6-verse lyric sustains to **156 s** (validated, RTF ~2.0, 0% clip).
- **Weight offload: QUANTIFIED → not for the LM.** PCIe H2D ≈ 12 GB/s (measured). The 48 LM layers (5.1 GB
  Q8) are read once per AR step (1000–4100×) and the decode is weight-load-bound at VRAM BW (~360 GB/s) →
  streaming them is ~30× slower per streamed byte. Prefetch hides only ~0.5 GB (~5 layers) under the
  47 ms/step compute; beyond that ≈ +83 ms/step per GB → reaching 3.5 GB = ~5.5× slower (non-viable).
  Decode-stage weights are streamable but aren't the peak. **Offload can't put Q8 in 3.5 GB.**
- **q4_0 KV (task #7): no longer crashes** (current ggml compiles the Q4_0/Q4_0 FA-vec kernel). −47% KV
  (412.8 vs 779.7 MB @30s). cb0 gate 0.9999 but **cb1/cb2 ≈0.98** (lossier vocal/bgm) → **ear A/B pending**;
  not auto-enabled.

**Bucket placement (Q8 weights 5.1 GB + LM act ≈ 0.3 GB fixed; KV q8_0 ≈0.40 / q4_0 ≈0.21 MB/token):**
| bucket | Q8 verdict |
|---|---|
| **3.5 GB light-guard** | **impossible on Q8** (weights alone 5.1 GB > 3.5). Q4_K_M weights (2.86 GB) + dynamic KV is the only fit. |
| **7.5 GB** | Q8 + dynamic KV. q8_0 KV → ~150 s (grow-transient bound; lower `SG_KV_GROW_BLOCK` to reach further). **q4_0 KV → full 270 s model-max comfortably** (270 s cap = 1678 MB KV → ~7.2 GB peak). |

## 🔭 REMAINING WORK BEFORE PROD (next-session handoff, 2026-06-05)
Perf/VRAM (task #8/#9) + the KV/bucketing/floor work is **DONE & measured** (above + PERF-3060 Levers 4–8).
What's left to ship songgen as a prod service, roughly in priority order:

### A. Correctness / quality (feature-blocking)
1. **#6 clone + continue conditioning — WRONG, unvalidated (the big one).** Symptoms (ear): clone = bgm close
   but **voice off (male vs female prompt) + doesn't match lyric**; continue = "**record backwards**". Inputs
   are clean (htdemucs feed was fixed) → genuine conditioning-logic bugs in the shared prompt-audio path.
   **Fix = reference-golden bisect** (NOT blind): on the torch dev box `10.0.0.151`, capture the Python ref
   clone/continue tokens+audio for identical inputs, then bisect where `sggen` conditioning diverges —
   `audio_qt_embs` cb0=pmt/cb1=vocal/cb2=bgm gather, EOT prepend, the 16385 masking in
   `lm_levo.prepare_condition_tensors:222-230`, CFG handling of the audio prompt. Usage note (not a bug):
   prompt audio must be a ~10 s chorus clip (`prompt_len=10s`/`audio_len=252`); the harness auto-picks the
   peak-vocal 10 s window (`stems/chorus_*.wav`).
2. **#10 htdemucs CUDA forward broken** — `/separate` is force-CPU (19 s vs ~4 s GPU). CUDA output is junk
   (cossim ~0.001, ~60× over-scaled, 91% clipped); correct on CPU. Bisect blocks CUDA-vs-CPU; suspects =
   conv_transpose/col2im or the cross-transformer FA at head_dim 64. Single-file→clone works via the CPU path.

### B. VRAM (clone/continue path)
3. **clone/continue ENCODE stage is an unchunked VRAM hog** — MusicFM ×2 over a 40 s tile → 7437 MiB peak
   (the clone wall). Needs the same chunking treatment the decode got (Lever 1). Only matters once #6 lands.

### C. Server / prod infra
4. **Worker isolation upgrade (the user's ask).** v0 server is **subprocess-exec per request**: it gives
   true-0 idle VRAM (child exits) but **reloads ~7 GB of ggufs on EVERY request** (cold-start tax each gen).
   Upgrade to acestep-style **fork+IPC warm worker** (`acestep.cpp/tools/ace-server.cpp` is the pattern):
   weights stay resident between requests (fast repeats) + idle-unload watchdog → true-0 when idle. Big UX win.
5. **Async job model** — `/generate` is synchronous/blocking (140–280 s walls) behind a single global compute
   mutex (one gen at a time). For prod, async-job+poll or SSE. (No `out/API-DESIGN.md` exists yet — design it.)
6. **Streaming output (optional, future)** — currently batch-only (full gen → full decode → one wav). Feasible
   because the LM is AR + the decode is already chunked, BUT: (a) ~10 s startup-latency floor from the delay
   pattern (cb1/cb2 lag cb0 by 250 frames); (b) needs LM+decode **co-resident** (gives up the peak=max(stages)
   VRAM trick → higher peak); (c) **only works at RTF<1 → Q4_K/q4_0 (0.89) can stream, Q8/q8_0 (1.19) cannot**;
   (d) needs streaming transport (SSE/chunked/ws).

### D. Integration / packaging
7. **koblem/kobbler integration** — songgen is NOT wired into the prod stack (zero refs in `koblem/`/`kobbler/`).
   acestep is the current "music" engine. Decide replace-vs-coexist, then wire UI/`HeavyKind` gate/profile/
   docker like the acestep koblem integration. Needs a **prod Dockerfile** (dev uses `songgen-dev:builder`).

### E. Quality decisions (user's ear calls — not code-blocking)
8. **q4_0 vs q8_0 KV** — ear A/B staged (eartest `kv-q8kv-30s`/`kv-q4kv-30s`, `dynkv-q8-q4kv-175s`). q4_0 is
   the most-perturbed KV but still less-perturbed than the ear-approved Q4_K **weights** (see ladder, Lever 7).
9. **Q8 vs Q4_K weights for prod** — Q8+q8_0 = 7.5 GB bucket, ~150 s, RTF 1.19 (quality option). **Q4_K+q4_0 =
   ~4 GB light-guard, 140–171 s, RTF 0.89 sub-realtime** (and the only config that could stream). Pick by ear.

### F. Minor / documented
- single-step/prefill FA path not 256-bucketed (runs once → negligible; batch decode is bucketed).
- `SG_KV_GROW_BLOCK=250` lets q8_0 reach ~165 s in 7.5 GB (more host-bridge copies, ~10 s aggregate).
- **conditioning-prefix right-size: investigated, DECLINED** — the 600-tok region is the lyrics (scales with
  song), so long songs have no pad to trim; only ~330 constant tokens, trimming risks the trained layout.
- DEPLOY-3060 §8 is STALE ("clone/continue/separate 501 stubs") — they're implemented; clone/continue broken (#6),
  separate works (CPU).

## ENV KNOBS (all default-safe; deploy/serve.sh sets the budget config)
- `SGLM_NO_FA=1` — revert LM decode to manual attention (FA is default).
- `SG_KV_TYPE=f16|q8_0|q4_0` — KV cache type (default f16 in code; serve.sh deploy = q8_0).
- `SG_CHUNK_FRAMES=N` — decode chunk size (default 750 in code; serve.sh deploy = 200).
- `SG_KV_GROW_BLOCK=N` — max per-grow KV increment in frames (default 1500). Lower → tighter footprint
  near the ceiling (less overshoot/transient) at the cost of more host-bridge copies.
- `SG_KV_STATIC=1` — force one-shot KV alloc to the full ceiling (no growth); A/B / debug only.
- `SG_KV_BUCKET=N` — round the FA KV-read up to N (default 256) for the CUDA fast path (~2× long-context
  decode). `SG_KV_BUCKET=1` = ragged/bit-exact (slower). `--duration <s>` = hard cap; omit = model-max (270 s).
- `SGLM_PROFILE=1` — per-step LM timing. `HTD_DEBUG_RMS=1` — htdemucs stem RMS debug.
