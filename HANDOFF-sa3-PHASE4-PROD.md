# SA3 port — HANDOFF: Phase 4 (PROD) — fresh-context kickoff

The model surface is **DONE**: t2a + a2a + inpaint + continuation, all validated, committed + pushed
(`origin/sa3-port` @ `cd8b636` on `dbrain/hbd-acestep.cpp`). Phases 2 (features) and 3 (perf) are closed —
see `HANDOFF-sa3-FEATURES-PERF.md` (top "STATUS UPDATE" block) and `SA3-PORT-PLAN.md`. **Steps stays at 8**
(SA3 reference default; perf surface exhausted at Q8 — `mul_mat_q` is the floor). This doc = the prod phase.

## What "done" looks like
SA3 running as a **prod GPU service** folded **under "music"** beside acestep (a selectable model/mode in
the existing music surface, sharing the **light-GPU gate** — NOT a separate engine). Idle VRAM true-0 via
warm fork/IPC worker-isolation. Reachable from koblem's `/music` UI.

## The precedent to mirror (do this first: read it)
songgen did this exact journey weeks ago. **Copy its shape, don't reinvent:**
- `project_acestep_worker_isolation` (memory) — fork+IPC ⇒ idle VRAM TRUE-0; the pattern.
- `project_songgen_e2e_shipped` + `reference_songgen_prod_flags` (memory) — async jobs + cancel + warm
  fork/IPC worker (`--worker-isolation`, idle-unload, resident cache), prod server flags, the 12GB-fit math.
- `project_acestep_koblem_music` (memory) — how acestep is wired as the koblem "music" engine (`/music` UI,
  profile `acestep`). SA3 slots in **beside** it under the same surface.
- In the repo: `songgen-server.*` / `tools/songgen-server*` (the async+cancel+isolation server) and
  `kobbler/docker/songgen/` (the prod Dockerfile) are the templates. acestep's own prod docker lives under
  `kobbler/docker/` too — find it.

## Tasks (in order)
1. **sa3-server** — async job queue + cancellation + warm fork/IPC worker-isolation, mirroring
   songgen-server. The whole `sa3-gen` pipeline (encode? t2a/a2a/inpaint/continuation) behind a request
   API. Resident model cache in the warm worker; idle watchdog → unload → VRAM true-0. Light-GPU peer.
2. **prod Dockerfile** — `kobbler/docker/sa3/` (mirror `kobbler/docker/songgen/`). Base `nvidia/cuda:12.9.1`,
   sm_86. Builds `sa3-server` + ships the Q8 ggufs (or Q6 — see budget below). NOT the dev builder.
3. **koblem wiring** — SA3 as a selectable model/mode under the existing **music** surface beside acestep.
   Light-GPU gate (shared with acestep, NOT a new HeavyKind). Expose t2a + the 3 edit modes in the API/UI.
   **Rust ⇒ build OFF-SERVER** (no-build-on-server rule is Rust-only; see `feedback_no_build_on_server`).
4. **koblibs gate/client + kobbler ref bump** — Rust, off-server. Commit/push koblibs separately (Docker
   builds clone from git).
5. **FF-merge `sa3-port` → `acestep.cpp` master** (songgen did exactly this: clean `sa3-*` superset,
   acestep/songgen targets untouched, additive CMake). Then bump the kobbler `ACESTEP_REF`/`SA3_REF`.

## Budget / facts the prod phase needs
- **VRAM:** all-Q8 ≈ 2.67 GB (lossless-grade) or all-Q6 ≈ 2.1 GB (≥.996), both ≤3 GB with margin. Game
  SFX = short seq ⇒ tiny activations; even 380s max fits ≤3 GB (flash). Pick Q8 unless the gate wants the
  extra headroom. ggufs: `models/sa3-{dit,same,t5gemma}-{Q8_0,Q6_K,...}.gguf` (regen from F32 if tensors
  change — loop in `SA3-PORT-PLAN.md` "Max length" note).
- **Models:** 3 ggufs (DiT / SAME enc+dec / T5Gemma). SAME gguf also carries the conditioner (`sec.*`,
  `prompt.pad_embed`) + the encoder (`enc.*`) — one file does t2a AND all edit modes.
- **CLI surface already built** (wrap these in the server): `--prompt --seconds --steps --seed --sampler
  --out` (t2a); `--input --strength` (a2a); `--input --mask "s:e,.."` (inpaint); `--input --continue
  --seconds` (continuation). All in `tools/sa3-gen.cpp`.
- **Flash is default-on for GPU**; warm server kills the ~0.49s per-invocation load tax (the real
  short-clip win, deferred to exactly this phase).

## Loose end to pick up
`kobbler/docker/acestep-sa3-dev/iter.sh` has a 1-line **uncommitted** change (`SA3_TARGETS` +=
`sa3-same-enc-test`) on kobbler `main` — commit it (or fold into the prod-docker work). Needed so
`./iter.sh build` includes the encoder test.

## Build/validate reminders
- C++ forks build fine on server via `kobbler/docker/acestep-sa3-dev/iter.sh build` (CUDA) and
  `./buildcpu.sh` (CPU oracle — **must** configure `-DGGML_BLAS=OFF`, host lacks cblas).
- Validation harnesses (all vs `.sa3ref/goldens*`): `sa3-{t5gemma,dit,same,same-enc,pipeline,tok}-test`.
  Golden gens live in `.sa3ref/` (gitignored): `gen_goldens.py`, `gen_euler_golden.py`, `gen_enc_golden.py`,
  `gen_inpaint_golden.py`. Reference venv: `.sa3ref/stable-audio-3/.venv` (torch 2.7.1 cu126).
- GPU is single + shared — coordinate (owner sanctions; a small TTS job may be co-resident). Ear page:
  `kobbler/docker/acestep-sa3-dev/eartest.sh PORT "label:args" ...` (use :8103, :8097 is longcat's).
- pingpong sampler is stochastic ⇒ validate per-component on deterministic goldens, never e2e bit-match.
