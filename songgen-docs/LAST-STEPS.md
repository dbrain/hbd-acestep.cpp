# songgen.cpp — LAST STEPS to prod (ggml merge · acestep merge · Dockerfile · koblem wiring)

State as of 2026-06-07: the E2E work (htdemucs GPU fix+perf, clone-encode VRAM levers, async
server + warm fork+IPC worker) is **built & validated, UNCOMMITTED** on branch `songgen-port`
(see `HANDOFF-E2E.md` + `out/API-DESIGN.md`). The repo IS `dbrain/hbd-acestep.cpp`; songgen is a
worktree/branch of it. acestep is PROD — the two reconciliation points below need an acestep
rebuild+smoke before touching master. You do NOT have to merge to master to ship the songgen
image (the Dockerfile can pin the `songgen-port` ref).

Suggested order: **0 commit → 1 ggml → 2 acestep merge → 3 Dockerfile → 4 koblem wiring.**

---
## 0. Commit & push (nothing ships from git until this is done)
- Review `git diff` on `songgen-port` (14 modified + new `src/songgen-cache.h`, `tools/songgen-htdemucs-dump.cpp`, `out/`, `songgen-docs/`).
- Shared-header changes that touch acestep: `src/weight-ctx.h` (`unique_ptr`→`shared_ptr` staging, for warm-cache copyability) + `src/gguf-weights.h`. Behavior-preserving.
- Commit in logical chunks (htdemucs / clone-encode / server-async+worker / cache-plumbing) or one; **no Co-Authored-By / AI trailers** (project rule). Push `songgen-port`.

---
## 1. ggml submodule reconcile
- master pins `ggml @ 4e3ee737`; songgen-port pins `ggml @ f3bc6505`. **They diverged** (not a fast-forward): f3bc6505 adds the snake-fusion review series; 4e3ee737 has commit(s) f3bc6505 lacks.
- Target = the consolidated `dbrain/ggml` line (the ggml-consolidation work targets `292516d5 = 4e3ee737 + concat + madd`). Land both engines on ONE ref:
  1. In `dbrain/ggml`, merge/rebase the snake-fusion series (…f3bc6505) onto the consolidated master so the unified ref contains BOTH lines.
  2. Point the acestep.cpp submodule at that unified ref; `git submodule update`.
- **Gate:** rebuild `ace-server` AND the songgen targets on the unified ggml; run the acestep smoke + the songgen golden gates (`songgen-htdemucs-test`, `songgen-clone-encode-test` → expect 83.5/87.8/85.9, `songgen-decode-chunk-test`). VAE snake fusion must stay bit-exact for both.

---
## 2. Merge `songgen-port` → `master`
- Almost purely additive: +8 committed songgen commits (13,115 insertions / 1 deletion) + the uncommitted server/cache work. **No binary-name conflict** — `songgen-*` vs `ace-*` build side-by-side (already in CMakeLists). The acestep *image* is unaffected (its Dockerfile builds `--target ace-server` only).
- Two things to verify before/at merge (both PROD-touching):
  - **ggml** = §1 (the submodule ref the merge lands on).
  - **shared headers** = the `weight-ctx.h`/`gguf-weights.h` changes feed `dit.h`/`vae.h` → `ace-server`. Rebuild ace-server + smoke (a known song render, idle→true-0 via worker isolation).
- CMakeLists already carries `link_ggml_backends(songgen-server)` (added for the in-process worker) — keep it.
- Decoupling option: if you'd rather not gate songgen's image on the acestep verification, **skip the master merge for now** and build the image from the `songgen-port` ref (Docker clones any ref). Merge to master later as cleanup.

---
## 3. Prod Dockerfile (`kobbler/docker/songgen/Dockerfile`)
Mirror `kobbler/docker/acestep/Dockerfile` (clone-at-ref → build → slim runtime, models mounted not baked):
- **Builder** (`nvidia/cuda:12.9.1-devel`): `git clone --recurse-submodules dbrain/hbd-acestep.cpp`, checkout the pinned `SONGGEN_REF` (full SHA, not a branch), `cmake -DCMAKE_CUDA_ARCHITECTURES=86 …`, then
  `cmake --build build --target songgen-server songgen-separate songgen-generate songgen-clone songgen-continue -j"$(nproc)"`.
  (server execs `songgen-separate` for `/separate`; the other CLIs are the non-isolation fallback. With isolation ON, generate/clone/continue are compiled INTO the server via `-DSONGGEN_AS_LIB`, but ship them anyway for the fallback path.)
- **Runtime** (`nvidia/cuda:12.9.1-runtime`): copy the binaries + ggml `.so`s. `HEALTHCHECK` → `GET /health`.
  `ENTRYPOINT songgen-server --host 0.0.0.0 --port 8097 --gguf /models --golden /models/golden --bindir /usr/local/bin --tmpdir /tmp --max-duration 300 --worker-isolation`
  Env: `SG_WORKER_ISOLATION=1`, `SG_IDLE_UNLOAD_SEC=15` (tune), `GGML_BACKEND=CUDA0`.
- **Models (~11 GB, NOT in repo):** `songgen-lelm-large-Q8_0` (5.4G) + `-Q4_K_M`, `songgen-cfm`, `songgen-vae`, `songgen-vae-encoder`, `songgen-septoken-aux`, `songgen-1rvq-aux`, `songgen-musicfm`, `songgen-musicfm-1rvq`, `songgen-htdemucs`. Mount read-only at `/models` (compose `${SONGGEN_MODELS_DIR}:/models:ro`, like `ACESTEP_MODELS_DIR`). `--golden` just needs an existing dir (server tokenizes lyrics in C++; the golden npy fallback isn't hit) — `/models/golden` empty is fine.
- **compose service** (`kobbler/docker-compose.yml`, copy the `acestep:` block at ~L369): name `songgen`, build `./docker/songgen`, GPU reservation, models volume, env above, reached at `http://songgen:8097`.
- **Cleanups for a clean image:** guard `songgen-generate`'s hardcoded `gen_tokens_cpp.npy` host-path write (logs a harmless warning otherwise). Confirm `--worker-isolation` truly idles to 0 VRAM in-container (validated locally; re-check under compose).

---
## 4. koblem wiring (keep acestep AND add songgen as a selectable music engine)
acestep is wired as the single GPU **"music"** engine; the cleanest "keep both" is an engine
selector under the same `music` HeavyKind so they time-share the one GPU. Mirror the acestep
integration (canonical: `acestep.cpp/HANDOFF-koblem-integration.md`). Touch points:
- **GPU gate** (`kob_gpu_gate`, used in `koblem/api/src/{avatar,flux2}.rs` via `HeavyKind::{Avatar,Flux}`): music already has a heavy kind. songgen renders go through the **same `music` gate** so acestep ⊕ songgen ⊕ avatar/flux all serialize on the GPU and force-unload each other (TTS preempt already SIGKILL-unloads acestep's worker — wire songgen's `/unload` the same way; it returns true-0 identically).
- **Service client**: add `koblem/api/src/songgen.rs` modeled on `koblem/api/src/music.rs` (the ace-server client). songgen's API differs: renders are **async** (`POST /generate|/clone|/continue` → `{job_id}`, poll `GET /job?id=`, `&download=1` for the wav, `&cancel=1`) vs acestep's job model — adapt the client to poll, and map cancel/disconnect → `/job?...&cancel=1` (or `/unload`).
- **Config** (`koblem/api/src/config.rs:25` has the acestep base URL): add a songgen base URL + a per-request/DB setting `music_engine ∈ {acestep, songgen}`. `main.rs:49/188` registers the music service + preempt hooks — register songgen alongside and route by the setting.
- **Routing/storage**: `generations.rs` already maps `"music" → songs_dir`; songgen songs land in the same place. Keep `generation_type="music"`; add an `engine` column/param if you want per-song provenance.
- **UI / config** (`koblem/web` `/music` page): add an engine toggle (acestep = "lighter/faster", songgen = "LeVo2, clone/continue from a prompt"). songgen adds **clone** (prompt stems → cloned style) and **continue** (extend a clip) — surface stem upload + the `--lyric/--description/--duration/--gen-type/--temp/--top-k/--cfg` knobs. Show async progress (poll `/job`) + a cancel button.
- **Ecosystem rules** (kobbler CLAUDE.md): koblibs changes commit/push separately; bump the songgen image ref in compose deliberately; koblem is a Rust workspace → **author code, don't build on this host** (off-server CI builds it).

---
## Validation gates (run before calling it done)
- songgen golden gates green on the unified ggml (htdemucs-test, clone-encode-test 83.5/87.8/85.9, decode-chunk-test).
- acestep ace-server rebuilt + a known song renders correctly on the shared-header + ggml changes; idle → true-0.
- songgen image: `/health` ok; a `/generate` job + poll + download returns a valid wav; `/job?cancel` and `/unload` drop VRAM to true-0 (`nvidia-smi` per-process); warm reuse skips the 5 GB reload across two jobs in one grant.
- koblem: music engine toggle renders via both engines; GPU gate serializes music ⊕ avatar/flux; TTS preempt force-unloads whichever music engine is resident.
