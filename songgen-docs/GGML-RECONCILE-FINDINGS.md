# ggml reconcile — findings (LAST-STEPS §1) — analysis done 2026-06-07, EXECUTION DEFERRED

**TL;DR:** the f3bc6505↔consolidated divergence is NOT a mechanical merge — the two
lines model `snake` with *architecturally different* designs that collide. Resolving it
is a small **port decision** + bit-exact GPU golden gates on BOTH engines (prod acestep
+ songgen). That validation is GPU-gated, so it's deferred to a GPU window. The songgen
**image ships independently** from `songgen-port`'s own self-consistent ggml `f3bc6505`
(decoupling path, LAST-STEPS §3) — the reconcile is cleanup, not a launch blocker.

## The exact topology (verified)
- dbrain/ggml `origin/master` = **`292516d5`** (= `4e3ee737` + `0bcf0e83` single-launch
  concat + `292516d5` same-shape madd-fuse — the NAVA perf push, 2026-06-06). The acestep
  worktree submodule is checked out at the now-stale `4e3ee737`; `git submodule update` →
  `292516d5` is step 0 of any reconcile.
- songgen-port submodule = ServeurpersoCom/ggml `master` tip = **`f3bc6505`** (the
  snake-fusion review series).
- merge-base(`292516d5`, `f3bc6505`) = `387fa29f`. They genuinely diverged.
- `f3bc6505` is present as an object in the shared `.git/modules/ggml` store (the songgen +
  acestep worktrees share it), so the reconcile can be done locally without re-fetching.

## Why it conflicts (non-destructive `git merge-tree --write-tree 292516d5 f3bc6505`)
Conflicts: `include/ggml.h`, `src/ggml.c`, `src/ggml-cpu/ggml-cpu.c`,
`src/ggml-cuda/ggml-cuda.cu`, `src/ggml-cuda/im2col.cu`, `src/ggml-vulkan/*`, and
**add/add on `src/ggml-cuda/snake.cu` + `snake.cuh`** (both lines created the file).

## The root catch — two incompatible snake designs
- **`292516d5` (dbrain):** explicit first-class op. `include/ggml.h` has `GGML_OP_SNAKE`
  (enum 557) + `ggml_snake(ctx, a, alpha, beta)` / `ggml_snake_inplace`. snake.cu = 182
  lines. **acestep `vae.h` calls `ggml_snake()` directly.**
- **`f3bc6505` (ServeurpersoCom):** **NO `GGML_OP_SNAKE` enum at all** — `a7c63cfb`
  "refactor snake into a graph autofuse pass" fuses the existing mul/sin/sqr/mul/add chain
  at graph-compile via `ggml_can_fuse`; snake.cu = 72 lines (the fused kernel the pass
  dispatches to). **songgen `songgen-vae.h` was built against THIS** — see memory
  `project_songgen_ggml_bump_snake_dead`: "load snake gains as [1,C], drop the in-graph
  reshape that broke `ggml_can_fuse`" → VAE-GPU −20% bit-exact (180 launches → 36 fused).
- Both also carry `GGML_OP_COL2IM_1D`, but the dbrain side already has it
  (`048cba4d` port from ServeurpersoCom), so col2im is reconcilable; **snake is the catch.**

## A/B RESULT (2026-06-07) — RESOLVED: just bump the submodule, no merge/port
Built songgen on **f3bc6505** vs **292516d5** (the only var; same songgen code incl. the
isolation fix) and rendered an identical request (seed 42, 60s, q8/q8_0 KV/chunk 200,
isolation on):

| ref | peak VRAM | wall | rendered wav md5 |
|---|---|---|---|
| f3bc6505 (autofuse series) | 6446 MiB | 77 s | `4de87d86…` |
| **292516d5 (consolidated)** | **6446 MiB** | **75 s** | **`4de87d86…` — IDENTICAL** |

**292516d5 wins / is the "most right":** byte-identical output (its OWN snake autofusion,
upstream `b3af00e0`, fuses songgen-vae.h's mul/sin/sqr/mul/add chain bit-exactly — so the
[1,C] gain trick still matches and NO explicit-op port is needed), identical VRAM, ~2s
faster (concat single-launch + madd fusions). songgen **compiles clean** on 292516d5.

⇒ The reconcile is NOT a merge and NOT a port. It's: **point songgen's ggml submodule at
292516d5** (already done in the working tree) and drop the f3bc6505 line entirely — its
snake-review series is redundant for songgen (292516d5 already fuses the chain) and acestep
keeps its explicit `GGML_OP_SNAKE` op. Both engines then share dbrain/ggml @ 292516d5, which
makes the master merge (LAST-STEPS §2) trivial (no ggml conflict). The Option A/B below are
now MOOT — kept for history.

Gate confirmation on 292516d5 (2026-06-07):
- songgen-clone-encode-test (MusicFM/RVQ path): pmt **83.5%** / vocal **87.8%** / bgm **85.9%**
  → EXACT match to the expected gate. RESULT: MATCH.
- songgen-decode-chunk-test (CFM+VAE chunked): nan/inf=0, seams 1.1x/1.5x global (normal). OK.
- generate render: byte-identical md5 (above) — covers LM+CFM+VAE incl. snake.
- songgen-htdemucs-test: golden `input_seg.npy` not in this dataset, so not run — but htdemucs
  is snake-FREE (convs/STFT/attention, ops identical between the two refs; the only divergent
  op is snake, which is VAE-only and proved bit-identical). No regression expected.

**DEPLOY DETAIL for the bump:** songgen's ggml submodule remote is ServeurpersoCom, but
292516d5 is a dbrain/ggml commit. The local A/B built because the SHA was already in the
shared object store. A fresh clone (Docker build) needs `.gitmodules` ggml `url` repointed to
`https://github.com/dbrain/ggml.git` so `git submodule update` can fetch the pinned 292516d5.
(The working tree's ggml is left checked out at 292516d5 = the staged reconcile.)

## (HISTORICAL) Recommended resolution — pick ONE
**Option A (recommended): migrate songgen's VAE to the explicit `ggml_snake()` op, drop the
autofuse series entirely.** Then there is NO ggml merge — both engines run on dbrain's
`292516d5` (bump the songgen submodule to it). songgen's `songgen-vae.h` /
`songgen-vae-enc.h` stop building the mul/sin/sqr/mul/add chain and call
`ggml_snake(x, alpha, beta)` (same fusion benefit, first-class op). Smallest unified
surface; one ggml line forever. Re-validate the songgen golden gates are bit-exact.

**Option B: actually merge `f3bc6505` into `292516d5`** keeping BOTH the explicit op and the
autofuse pass. Heavier: resolve the add/add snake.cu/.cuh by namespacing one kernel, keep
both dispatch paths, ensure the autofuse pass doesn't double-fire when the explicit op is
present. More code to carry; only worth it if some non-songgen consumer needs the autofuse
pass (none known).

## Validation gate (BOTH required, GPU-gated — that's why this is deferred)
1. Rebuild `ace-server` (PROD) on the unified ref + acestep smoke (a known song renders
   correctly; idle → true-0 via worker isolation). VAE snake must stay bit-exact.
2. Rebuild songgen targets + golden gates: `songgen-htdemucs-test`,
   `songgen-clone-encode-test` → **83.5 / 87.8 / 85.9**, `songgen-decode-chunk-test`.
   VAE snake fusion bit-exact (std ours/gold unchanged).

## Status / next step
- Analysis complete; **no merge executed, nothing pushed** (prod-acestep risk + needs GPU
  golden gates). The acestep ggml working tree was NOT touched (merge-tree is in-memory).
- Unblocked path that needs NONE of this: the songgen prod image (kobbler/docker/songgen,
  pinned `SONGGEN_REF=b7fdd0e` → ggml `f3bc6505`) + koblem wiring. Do the ggml unification
  as a dedicated GPU session (Option A first).
