#pragma once
// sa3-cache.h: warm-worker RESIDENT model cache for sa3-server.
//
// The SA3 weight set is small enough to keep ENTIRELY resident: all-Q8 ≈ 2.67 GB
// (t5gemma 313M + dit 1.57G + same 912M), inside the 3 GB light-GPU budget. So the
// warm fork/IPC worker (sa3-server --worker-isolation) keeps every loaded model
// GPU-resident across requests, killing the ~0.49 s per-invocation H2D weight upload
// that dominates short-SFX wall time — the whole reason the warm server was deferred
// to the prod phase. Idle-unload / /unload SIGKILLs the worker → VRAM true-0 between
// bursts (the cache dies with the process; no clean free needed).
//
// This DIFFERS from songgen-cache.h ON PURPOSE: songgen runs on a 12 GB card with
// 5-7 GB weights beside a resident 9B LLM, so it frees GPU buffers per stage (residency
// would stack/OOM). SA3 has the headroom, so it keeps weights resident for the latency
// win. The shallow-copy borrow is the contract weight-ctx.h already documents (no
// destructors; staging/pending cleared at wctx_alloc) — copying a loaded struct shares
// its ggml ctx/buffer pointers, and with caching on we never free them per request.
//
// g_sa3_cache_on is set true ONLY in the worker child. With it off (standalone sa3-gen
// CLI + golden tests) load/free are byte-identical to before (fresh load, real free).

#include <string>
#include <unordered_map>

static bool g_sa3_cache_on = false;

// Cache-aware load: when caching is on, the first load of a (type,path) stays
// resident and later loads reuse it (shallow-copy borrow, no H2D); when off it is a
// plain loader call. `loader` is bool(T*, const char*). One static map per model type
// T, so the SAME gguf shared by SA3Same and SA3Enc never collides (distinct T).
template <typename T, typename Loader>
static bool sa3cache_load(T * out, const char * path, Loader loader) {
    if (!g_sa3_cache_on) {
        return loader(out, path);
    }
    static std::unordered_map<std::string, T> cache;
    auto                                      it = cache.find(path);
    if (it == cache.end()) {
        T m{};
        if (!loader(&m, path)) {
            return false;
        }
        it = cache.emplace(path, m).first;  // resident copy (shares the ggml buffers)
    }
    *out = it->second;  // borrow the resident buffers (no per-request free)
    return true;
}

// Cache-aware free: a no-op while caching is on (the cache owns the GPU buffers,
// reclaimed by worker teardown); a real free otherwise. `freer` is void(T*).
template <typename T, typename Freer>
static void sa3cache_free(T * model, Freer freer) {
    if (!g_sa3_cache_on) {
        freer(model);
    }
}
