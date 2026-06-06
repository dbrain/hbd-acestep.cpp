# songgen.cpp — E2E EXECUTION HANDOFF (htdemucs GPU + perf/VRAM, clone/continue encode chunk, productionising)

Single source of truth for a **fresh session** to finish songgen.cpp to prod. Read this top-to-bottom first.
Companion docs: `HANDOFF-PROD.md` (older prod tracker, partly superseded by this), `PERF-3060.md` (perf detail).
Memory: `project_songgen_clone_continue_validated`, `project_songgen_dynamic_kv`, `project_songgen_ggml_bump_snake_dead`.

Worktree `~/dev/songgen.cpp` (branch `songgen-port`). Harness/models `~/dev/songgen-data/`.

---
## 0. ENVIRONMENT & GOTCHAS — read before running anything
- **Build:** `~/dev/songgen-data/ibuild.sh "<targets>"` (incremental, runs cmake in the `songgen-dev:builder` CUDA image). Full build line if needed: `docker run --rm --gpus all -v /home/dbrain/dev/songgen.cpp:/src -w /src/build songgen-dev:builder bash -lc 'cmake --build . --config Release -j "$(nproc)" --target <T>'`.
- **Binaries are CUDA-linked** → you MUST `docker run --gpus all` even for `GGML_BACKEND=CPU` runs (else `libcuda.so.1` load error).
- **Run pattern:** `docker run --rm --gpus all -v /home/dbrain/dev/songgen.cpp:/src -v /home/dbrain/dev/songgen-data:/data -w /src songgen-dev:builder bash /data/<script>.sh`. Put scripts under `/data` — the container's `/tmp` is NOT the host `/tmp`.
- **Drive heavy iterative work from the MAIN LOOP** (this/your session) with `run_in_background: true` for builds/renders — you'll be notified on completion. Do NOT delegate the loop to an Agent-tool sub-agent: they are not woken on background completion and deadlock. Bounded read-only fan-out is fine.
- **cosine gates are SCALE-INVARIANT.** Every `*-test` golden uses cossim, which hides global gain/scale bugs. Always read the `std ours/gold` line the test prints; for the htdemucs bug compare **absolute** error.
- **GPU:** RTX 3060, 12 GB, sm_86. One GPU — one heavy job at a time. nsys/ncu: mount `/mnt/hdd/3d/avatar-shootout/toolchain:/tc:ro`, nsys at `/tc/nsight-compute/.../nsys`; ncu needs `--cap-add SYS_ADMIN` (see `reference_ncu_docker_syadmin`).
- **git:** commit only when the user asks; plain git OK; **never** add Co-Authored-By/Claude trailers. No pushes unless asked.
- **Models:** `/data/gguf/songgen-*.gguf`. Golden: `/data/golden-large/`. Eartest server: http://10.0.0.208:8110 (curated `index.html` + `cleanAB.html` + `levo.html`; drop new wavs in `/data/eartest/` and link from a page). LeVo reference material in `/data/levo/` (their prompts + outputs + our/ref demucs stems + extracted lyric `1_en_lyric.txt`).
- **Server:** `serve.sh start|stop|logs` (songgen-server on host :8101, container :8097).

## STATE — what's DONE & VERIFIED (do not re-litigate)
- **Clone/continue WORK.** The handoff-#6 "conditioning broken" was a ghost: "garbled/record-backwards" = the broken htdemucs feed (now CPU); "voice male" = bad/borderline test input. On the LeVo official female prompt our clone is female (302 Hz) and matches their output (304 Hz); user confirmed "right match". Conditioning path audited faithful; encode 83–88% vs golden = expected resampler drift.
- **Separation correctness verified:** our ggml htdemucs == real demucs 4.0.1 (vocal cossim 0.9929, bgm 0.9954). Only the CUDA *speed* path is broken (Task 1).
- **No scale bug:** VAE decode std 0.16335/0.16873, CFM std 0.14136/0.14167 — bit-faithful.
- **Loudness fixed — commit `177772a`** (`wav.h`): default is now attenuate-only peak-normalize (`gain = peak>0.985 ? 0.985/peak : 1`), replaces the loudness-maximizing limiter; `SG_OUTPUT_LIMITER=1` reverts. **⚠ committed but UNBUILT → Task 0.**
- **Reference sampling params:** cfg 1.5, temp **0.9**, top_k **50** (songgen-clone defaults are temp 1.0/top_k 250 — pass `--temp 0.9 --top-k 50` to match LeVo). songgen-clone hard-caps at **15s** (`tools/songgen-clone.cpp:156`); pass `--duration` for full songs. continue **trims the prompt by design** (output = continuation only).

---
## TASK 0 — build-verify the peak-norm commit (5 min)
`177772a` changed `src/wav.h` (one-line default flip + computation block). Rebuild and confirm + A/B:
1. `ibuild.sh "songgen-clone songgen-continue songgen-generate"` → expect clean build.
2. Render any clone with default (peaknorm) and `SG_OUTPUT_LIMITER=1` (old); confirm peaknorm log line `[WAV] peak-normalize: global peak X -> uniform gain Y` and no clipping. (A/B already staged on `levo.html` §5.)
Done = builds clean + peaknorm is active by default.

## TASK 1 (#2) — htdemucs CUDA forward: bisect + fix
**Symptom:** `/separate` on CUDA = garbage (cossim ~0.001, ~60× over-scaled, 91% clipped); correct on CPU. Currently force-CPU (19 s; want ~4 s GPU). cossim ~0.001 = *orthogonal/garbage*, NOT merely scaled → a genuinely-wrong CUDA kernel result.
**Already inspected & CLEARED (CUDA kernels correct):** `col2im_1d` (time decoder, `songgen-htdemucs.h:917` — gather/sum kernel correct), F32 `im2col` (htd_conv1d `:475` — CUDA supports F32 dst, same templated kernel), `pad_ext` left-pad (spec decoder `:890-891` — CUDA honors lp params, op_params order matches ggml.c). The handoff's named suspects are NOT it.
**Plan — empirical per-stage bisect (the cosine gates hid this; use ABSOLUTE error):**
1. `htd_forward` (`:939`) exposes every intermediate via `HtdForwardOut`: `enc_pre[4]`, `enc_post[0]`, `tenc[4]`, transformer `ct_x_layer[5]`/`ct_xt_layer[5]`, `dec_out[4]`, `tdec_out[4]`. `songgen-htdemucs-test <gguf> <ref_dir> [stage]` already does staged golden compares (`enc` default; check the source `tools/songgen-htdemucs-test.cpp` for other stages).
2. Add an env-gated path (or extend the test) to run `htd_forward` on **CPU** and **CUDA0** for the same `input_seg.npy`, and diff each intermediate by **max-abs error** (not cosine). Find the FIRST stage that diverges → that localizes the op.
3. Remaining suspects after the clean three: the manual `soft_max_ext` attention (`:754`), `htd_gn1` GroupNorm (`:494`, reshape+cont+norm), an encoder/transformer op, or a `permute`/`view`/`cont` that CUDA mishandles on a specific stride.
4. Fix the kernel/usage, re-validate (`songgen-htdemucs-test` enc + e2e vs CPU absolute), confirm `/separate` GPU == CPU (cossim >0.99 AND matched scale) and timing ~4 s.
**Validate path:** the LeVo prompt is a ready A/B — `/data/levo/1_en_prompt.wav`; ref demucs stems at `/data/levo/ref_{vocal,bgm}.wav`, our CPU stems `/data/levo/ours_{vocal,bgm}.wav` (cossim 0.993). GPU stems must match these.

## TASK 2 (#5) — htdemucs perf/VRAM pass (AFTER Task 1)
Once GPU separation is correct: profile and speed it up (user: even ~4 s is slowish). htdemucs = STFT(host FFT) → spec+time encoders (convs) → 5-layer cross-domain transformer (manual attn, head_dim 64) → spec decoder (conv2d + pad_ext overlap-add) + time decoder (col2im_1d) → iSTFT(host). Use nsys to find hot kernels; check the host-side STFT/iSTFT FFT (radix-2, `:344`) isn't dominating. Standard levers: gallocr reuse for VRAM, kernel occupancy, avoid redundant `cont`. Lock numbers vs the bench harness.

## TASK 3 (#4) — clone/continue MusicFM encode VRAM (7437 MiB)
**Root cause:** `sgclone_encode_stream` (`songgen-clone-encode.h:95-133`) repeat-pads each ~10s stem to 40s AND computes a redundant 2nd 40s segment (`:106-111`, `int_max_len = size/min24 + 1` + a second doubling), then trims codes to `output_len` (~251 frames ≈ 10s). It encodes ~80s to keep ~10s; peak VRAM (7437 MiB) = one 40s MusicFM forward.
**Levers:**
- **A (free, bit-exact):** drop the redundant 2nd segment — only seg0's first `output_len` frames are ever kept. ~2× less compute (peak unchanged).
- **B (the VRAM win, ~bit-exact for a single clip):** stop padding to 40s. The pad is just repeated copies of the same clip and only the first copy's frames are kept → encode ~20s (2 copies) instead of 40s+ → peak VRAM ~2–4× lower. NOT bit-exact (conformer self-attn is global → sees fewer repeats) → **validate with `songgen-clone-encode-test`** (`/data/gguf/...` + `/data/first.wav` + `/data/golden-large/clone`) leading-frame match (~80–88% is the expected pass band; ensure no regression vs current).
- **C (bit-exact, harder):** shrink the 40s forward graph peak via gallocr reuse / F16 acts (global attn blocks naive time-chunking).
**Rec:** A + B; validate leading-frame match unchanged. Re-measure peak VRAM via a clone render with `nvidia-smi` polling or the tool's VRAM log.

## TASK 4 — productionising (from HANDOFF-PROD.md §C/D, still open)
1. **Worker isolation (warm fork+IPC):** current server is subprocess-exec per request → true-0 idle VRAM but reloads ~7 GB ggufs EVERY request (cold-start tax). Port the acestep pattern (`acestep.cpp/tools/ace-server.cpp`): resident weights between requests + idle-unload watchdog → true-0 when idle. See `project_acestep_worker_isolation`.
2. **Async job model:** `/generate` is synchronous (140–280 s walls) behind one global compute mutex. Add async-job+poll or SSE. (No API-DESIGN doc yet — design it.)
3. **koblem/kobbler integration:** songgen has ZERO refs in the prod stack; acestep is the current "music" engine. Decide replace-vs-coexist, then wire UI / `HeavyKind` gate / profile / docker like the acestep koblem integration (`project_acestep_koblem_music`). Needs a **prod Dockerfile** (dev uses `songgen-dev:builder`). Follow kobbler ecosystem rules (CLAUDE.md): koblibs changes commit/push separately; bump refs.
4. **Streaming output (optional/future):** only viable at RTF<1 (Q4_K/q4_0); gives up the peak=max(stages) VRAM trick; needs SSE/chunked transport + handles the ~10s delay-pattern startup latency.

## SUGGESTED ORDER
Task 0 → 1 → 5(perf) → 4(encode VRAM) → productionising (worker-isolation first, it's the biggest UX win and unblocks the rest).
