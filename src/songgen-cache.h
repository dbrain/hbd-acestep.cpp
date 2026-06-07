#pragma once
// songgen-cache.h: warm-worker memory model.
//
// g_sg_cache_on is set true in songgen-server's --worker child. The worker keeps
// the PROCESS warm across requests — its CUDA context stays initialised and the
// MAP_PRIVATE gguf mmaps stay in the OS page cache — but GPU weight BUFFERS are
// still freed per stage, exactly like a standalone run. That's load-bearing on a
// 12 GB card: songgen-generate frees the LeLM (sglm_free) before the CFM/VAE
// decode stage, so the render peak is max(LM-stage, decode-stage) — NOT their
// sum. An earlier version retained every loaded model GPU-resident (no-op free)
// for back-to-back reuse; that made the decode buffer STACK on ~7 GB of resident
// weights (peak ~9-13 GB, growing across requests) for a ~2 s wall saving. The
// right trade is per-stage free + warm host: a reload is an H2D copy from the
// warm page cache (~0.4 s / 5 GB), not a disk read, so the worker stays "as warm
// as a normal run" while the peak stays at the single-stage max (~7.5 GB).
//
// Idle watchdog / /unload SIGKILLs the worker → true-0 (mmaps are reclaimable
// page cache, no mlock). Golden tests / CLI tools run with the flag off and are
// byte-identical either way.

#include <string>

// Set true exactly once, in the worker child, before serving any job. Read here
// only to document the warm-worker path; residency is intentionally NOT kept.
static bool g_sg_cache_on = false;

// Load helper for a model type T loaded by `loader(T*, path)`. Always loads fresh
// (no GPU-buffer residency) so the matching *_free really frees per stage; the
// worker's warmth comes from the persistent process + page cache, not resident
// GPU buffers.
template <typename T, typename Loader>
static bool sgcache_load(const char * path, T * out, Loader loader) {
    (void) g_sg_cache_on;
    return loader(out, path);
}
