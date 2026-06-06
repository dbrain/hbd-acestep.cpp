# songgen-server — API design & worker-isolation notes

Status: **Stage 1 + Stage 2 SHIPPED & validated live on :8101.**
- Stage 1: async jobs + cancellation + `/unload` (subprocess model).
- Stage 2: warm fork+IPC worker (`--worker-isolation` / `SG_WORKER_ISOLATION=1`) — opt-in model
  cache in the loaders, parent CUDA-free, idle-unload watchdog (`SG_IDLE_UNLOAD_SEC`, default 15s).
  Validated: warm reuse skips the 5 GB LeLM reload (byte-identical output), cancel/idle/`/unload`
  all SIGKILL the child → VRAM true-0, both async and `?wait=1` paths go through the worker and are
  serialized by `g_compute_mtx` (one render at a time; `?wait=1` does NOT bypass the limit).

## HTTP surface

| Method | Path | Notes |
|---|---|---|
| GET  | `/health` | `{status, version}` |
| POST | `/generate` | text→song. **Async by default** → `202 {job_id}`. `?wait=1` blocks and streams `audio/wav` (back-compat). JSON body: `lyric`*, `description`*, `duration`, `seed`, `gen_type` (mixed/vocal/bgm), `temp`, `top_k`, `cfg`, `fade`, `model` (q8/q4). |
| POST | `/clone` | multipart: `vocal_stem`*, `bgm_stem`*, `full_mix`, `lyric`*, + same tunables. Async by default; `?wait=1` to block. |
| POST | `/continue` | as `/clone` (adds VAE-encoder in-context seed). Async by default. |
| POST | `/separate` | multipart `mix`* → JSON `{vocal,bgm}` (base64 wavs). **Synchronous** (htdemucs is ~1.7s on GPU). |
| GET/POST | `/job?id=<id>` | poll → `{status, kind, elapsed_ms, ...}`. `&download=1` → the wav once `done` (streamed once). `&cancel=1` → cancel a pending/running job. |
| POST | `/unload` | GPU-guard preempt: SIGKILL any in-flight render → VRAM true-0. |

`status` ∈ `pending | running | done | failed | cancelled`. GPU work is serialized by one global mutex (one render at a time on the single 3060).

### Why async only for the long endpoints
`/generate`/`/clone`/`/continue` are 140–280 s renders — a synchronous HTTP hold risks proxy/client timeouts and gives no cancel handle. `/separate` is fast, so it stays synchronous (no job bookkeeping needed). This matches the operational note: *"other things aren't async but this is longer running."*

### Cancellation
`/job?id=...&cancel=1` (and `/unload`) set the job's cancel flag and **SIGKILL the running child**. Because the child is a separate process that owns the CUDA context, the kill reclaims **100% of VRAM instantly** (true-0) and the job is finalized as `cancelled`. Validated: render at 6114 MiB → cancel → songgen process gone, VRAM back to idle baseline.

## Worker model — current (Stage 1) vs warm (Stage 2)

### Stage 1 (shipped): subprocess-exec per request
The parent server **never touches CUDA** — it `fork`+`execv`s a `songgen-*` CLI child per job. Properties, which already satisfy the GPU-guard requirements:
- **No VRAM / no CUDA context in the parent, ever** — idle footprint is true-0 by construction.
- **Frees instantly** — child exit (or SIGKILL on cancel/unload) reclaims all VRAM.
- **Cost:** each request cold-loads its ggufs (~12 s measured: 5.4 GB Q8 LeLM disk read + CUDA-context init; ggufs are mmap'd so a warm page cache helps repeats).

For a **per-request GPU guard** (guard grants one render then preempts), this is already optimal: the guard would unload a warm worker between grants anyway, so every grant pays the cold load regardless of architecture. The lever that would help here is **faster loading** (warm page cache / pinned mmap / smaller quant), not a resident worker.

### Stage 2 (SHIPPED): warm fork+IPC worker (mirror of `ace-server.cpp`)
Pays off only when **multiple requests share one GPU grant** (warm-reuse within a grant), while keeping true-0 on idle/preempt. Mirrors the ace-server isolation layer already in this tree:
- Parent stays CUDA-free; forks `songgen-server --worker <data_fd> --control <ctrl_fd>` (re-exec of self).
- **DATA** socket (AF_UNIX, 12-byte framed): `HELLO` / `RUN` / `RESULT`. **CONTROL** socket: `CANCEL`.
- Child holds a **model cache** keyed by gguf path (load-once, reuse across `RUN`s; per-request KV/scratch still freed each job). Because parent+child share the filesystem, `RUN` marshals just the **argv** the parent already builds and the child writes the wav to the shared tmpdir — the parent reads it back (no large-payload IPC needed).
- **Idle watchdog** SIGKILLs the child after N idle seconds; `/unload` SIGKILLs immediately → true-0.
- Cancellation: SIGKILL the child (simplest, instant free) — or, to keep the worker warm after cancel, a cooperative cancel flag threaded into the AR generation + decode loops (the ace `g_child_active_cancel` pattern).

**Implementation sketch** (lowest-risk path): add an *opt-in* cache inside the model loaders (`sglm_load`/`sgcfm_load`/`sgvae_load`/`sgsep_load`/`sgmf_load`/`htd_load`) so a cached path returns a shallow-copy handle and `*_free` becomes a no-op for cached handles (default off → CLI tools & golden tests unchanged). Rename each tool's `main`→`<tool>_entry` (guarded standalone `main` preserved as the regression oracle), compile the tool `.cpp`s into the server, and have `--worker` dispatch `RUN`→`<tool>_entry(argc,argv)` with the cache on. Validate worker output is **byte-identical** to the CLI tool.

Resident footprint with all models warm ≈ 7 GB (generate: LeLM Q8 5.4 + CFM 1.3 + VAE 0.17) to ~9 GB (clone adds MusicFM ×2 + the transient ≤7.4 GB encode peak — see `SG_CLONE_MIN_SEC` lever). Fits the 12 GB 3060; only resident while serving, dropped on idle/unload.
