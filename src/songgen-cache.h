#pragma once
// songgen-cache.h: opt-in resident model cache for the warm worker.
//
// When g_sg_cache_on is true (set only by songgen-server's --worker child), the
// model loaders (sglm_load / sgcfm_load / sgvae_load / sgsep_load / sgmf_load /
// sgvaeenc_load) keep each loaded model resident keyed by gguf path and hand back
// a shallow copy on a repeat load, and the matching *_free becomes a no-op. This
// lets the warm worker serve back-to-back requests within a GPU grant without
// reloading ~7 GB of ggufs. The cache is process-local; SIGKILL on /unload or the
// idle watchdog reclaims everything to true-0 (the loaders mmap MAP_PRIVATE, no
// mlock, so resident host RAM is reclaimable page cache, not pinned).
//
// Default OFF → the standalone CLI tools and golden tests load/free exactly as
// before (the cache code is inert). The model structs hold only raw ggml pointers
// + small std::vector metadata (no owning smart pointers), so the shallow copy
// shares the ggml resources and the no-op free avoids any double-free.

#include <map>
#include <string>

// Set true exactly once, in the worker child, before serving any job.
static bool g_sg_cache_on = false;

// Cache helper for a model type T loaded by `loader(T*, path)`. Returns true and
// fills *out from cache on a hit; on a miss runs `loader`, stores a copy, returns
// its result. With the cache off it just calls the loader (no residency).
template <typename T, typename Loader>
static bool sgcache_load(const char * path, T * out, Loader loader) {
    if (!g_sg_cache_on) {
        return loader(out, path);
    }
    static std::map<std::string, T> cache;
    auto                            it = cache.find(path);
    if (it != cache.end()) {
        *out = it->second;
        return true;
    }
    if (!loader(out, path)) {
        return false;
    }
    cache[path] = *out;  // shallow copy: shares ggml resources, freed only on process exit
    return true;
}
