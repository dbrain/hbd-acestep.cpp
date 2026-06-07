// songgen-server.cpp: HTTP API for the SongGeneration C++/ggml port.
//
// v0 worker model: SUBPROCESS-EXEC per request. Each compute endpoint forks +
// execs the existing songgen-* CLI binary, which loads its ggufs, generates,
// writes a wav, and exits. Model memory is reclaimed to true-0 on child exit
// (same goal as ace-server's fork+IPC isolation, achieved here by process
// teardown instead of a long-lived worker). The upgrade path to integrated
// fork+IPC (a warm worker child that keeps weights resident) is documented in
// out/API-DESIGN.md.
//
// GPU/compute is serialized by a single global mutex: this box does one
// generation at a time (~244s Q8 / ~173s Q4 for 15s). /generate is synchronous
// and blocking in v0 — it returns the finished wav as audio/wav. For long jobs
// the design doc lays out async-job+poll and SSE options (the user's call).
//
// Endpoints:
//   GET  /health   -> {"status":"ok","version":...}
//   POST /generate -> text->song, returns audio/wav (IMPLEMENTED, end-to-end)
//   POST /clone    -> 501 (schema in body)   [wraps songgen-clone]
//   POST /continue -> 501 (schema in body)   [wraps songgen-continue]
//   POST /separate -> 501 (schema in body)   [wraps songgen-separate]
//
// Paths (ggufs, golden dir, binary dir) are resolved at startup from flags with
// sensible defaults for the dev box.

#include "version.h"
#include "yyjson.h"

#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wshadow"
#endif
#include "httplib.h"
#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

// ───────────────────────────────────────────────────────────────────────────
// Config (resolved at startup; defaults target the dev box layout)
// ───────────────────────────────────────────────────────────────────────────
struct Config {
    std::string host    = "127.0.0.1";
    int         port    = 8090;
    std::string gguf    = "/home/dbrain/dev/songgen-port/gguf";
    std::string golden  = "/home/dbrain/dev/songgen-port/golden-large";
    std::string bindir;  // dir holding songgen-generate etc. (default: this exe's dir)
    std::string tmpdir  = "/tmp";
    float       max_duration = 60.0f;  // reject absurd durations (KV/time guard)
};

static Config g_cfg;

// Single global compute lock: one generation at a time on this box.
static std::mutex g_compute_mtx;

static httplib::Server * g_svr = nullptr;

static void on_signal(int) {
    if (g_svr) {
        g_svr->stop();
    }
}

// ───────────────────────────────────────────────────────────────────────────
// gguf path resolution (q8 vs q4 = a lelm gguf-path choice)
// ───────────────────────────────────────────────────────────────────────────
static std::string lelm_path(const std::string & model) {
    // model: "q4" -> Q4_K_M, anything else -> Q8_0 (the safe default)
    const char * f = (model == "q4") ? "songgen-lelm-large-Q4_K_M.gguf" : "songgen-lelm-large-Q8_0.gguf";
    return g_cfg.gguf + "/" + f;
}

static std::string gguf_path(const char * file) {
    return g_cfg.gguf + "/" + file;
}

// ───────────────────────────────────────────────────────────────────────────
// JSON helpers (yyjson)
// ───────────────────────────────────────────────────────────────────────────
static std::string json_get_str(yyjson_val * root, const char * key, const char * dflt) {
    yyjson_val * v = yyjson_obj_get(root, key);
    if (v && yyjson_is_str(v)) {
        return yyjson_get_str(v);
    }
    return dflt;
}

static bool json_has(yyjson_val * root, const char * key) {
    return yyjson_obj_get(root, key) != nullptr;
}

static double json_get_num(yyjson_val * root, const char * key, double dflt) {
    yyjson_val * v = yyjson_obj_get(root, key);
    if (v && yyjson_is_num(v)) {
        return yyjson_get_num(v);
    }
    return dflt;
}

static void json_error(httplib::Response & res, int status, const char * msg) {
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_str(doc, root, "error", msg);
    char * out = yyjson_mut_write(doc, 0, nullptr);
    res.status = status;
    res.set_content(out ? out : "{\"error\":\"unknown\"}", "application/json");
    free(out);
    yyjson_mut_doc_free(doc);
}

// Stub responder: 501 + the planned request schema so the surface is visible.
static void stub_501(httplib::Response & res, const char * endpoint, const char * schema_json) {
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_str(doc, root, "error", "not implemented in v0");
    yyjson_mut_obj_add_str(doc, root, "endpoint", endpoint);
    yyjson_mut_obj_add_strcpy(doc, root, "planned_schema", schema_json);
    yyjson_mut_obj_add_str(doc, root, "see", "out/API-DESIGN.md for the open decisions (upload, async, auth)");
    char * out = yyjson_mut_write(doc, YYJSON_WRITE_PRETTY, nullptr);
    res.status = 501;
    res.set_content(out ? out : "{}", "application/json");
    free(out);
    yyjson_mut_doc_free(doc);
}

// ───────────────────────────────────────────────────────────────────────────
// subprocess exec: run argv, inherit stderr (logs), wait. Returns exit code or
// -1 on spawn failure. No shell — argv is passed straight to execv.
// ───────────────────────────────────────────────────────────────────────────
// Runs argv as a child. If pid_out is non-null, the child pid is published there
// before the blocking wait so a /job cancel (or /unload) can SIGKILL it mid-render
// — the kill makes waitpid report a signal (not WIFEXITED) and we return -1, so the
// caller finalizes the job as CANCELLED/FAILED and VRAM drops to true-0 on teardown.
static int run_subprocess(const std::vector<std::string> & args, const char * force_backend = nullptr,
                          std::atomic<pid_t> * pid_out = nullptr) {
    std::vector<char *> argv;
    argv.reserve(args.size() + 1);
    for (const auto & a : args) {
        argv.push_back(const_cast<char *>(a.c_str()));
    }
    argv.push_back(nullptr);

    pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "[songgen-server] fork failed: %s\n", strerror(errno));
        return -1;
    }
    if (pid == 0) {
        // Optional per-subprocess backend override (kept as a general hook).
        if (force_backend) {
            setenv("GGML_BACKEND", force_backend, 1);
        }
        execv(argv[0], argv.data());
        fprintf(stderr, "[songgen-server] execv %s failed: %s\n", argv[0], strerror(errno));
        _exit(127);
    }
    if (pid_out) {
        pid_out->store(pid);
    }
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
    if (pid_out) {
        pid_out->store(-1);
    }
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    return -1;
}

static bool read_file(const std::string & path, std::string & out);  // defined below

// Worker isolation (Stage 2): when on, renders run in a warm forked child that
// keeps models resident across requests (this parent stays CUDA-free). Defined in
// the worker block below; forward-declared here for job_run / job_cancel.
static bool g_isolation = false;
struct Job;
static int  dispatch_job(std::shared_ptr<Job> job, const std::string & kind, const std::vector<std::string> & args);
static void worker_unload();  // SIGKILL the warm child -> VRAM true-0

// ───────────────────────────────────────────────────────────────────────────
// Async job registry. The long endpoints (/generate, /clone, /continue) return a
// job id immediately and run on a detached thread; the client polls /job. Cancel
// SIGKILLs the running child (true-0 VRAM). GPU work stays serialized by
// g_compute_mtx — at most one render runs at a time on this single GPU.
// ───────────────────────────────────────────────────────────────────────────
enum class JobStatus { PENDING, RUNNING, DONE, FAILED, CANCELLED };

static const char * job_status_name(JobStatus s) {
    switch (s) {
        case JobStatus::PENDING:   return "pending";
        case JobStatus::RUNNING:   return "running";
        case JobStatus::DONE:      return "done";
        case JobStatus::FAILED:    return "failed";
        case JobStatus::CANCELLED: return "cancelled";
    }
    return "unknown";
}

static long long now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

struct Job {
    std::string            id;
    std::string            kind;                       // "generate" | "clone" | "continue"
    std::atomic<JobStatus> status{ JobStatus::PENDING };
    std::atomic<pid_t>     pid{ -1 };                  // running child (for cancel)
    std::atomic<bool>      cancel{ false };
    std::string              out_path;                 // wav written here on success
    std::vector<std::string> cleanup;                  // temp upload files to unlink when done
    std::vector<std::string> pre_args;                 // optional pre-step argv (htdemucs separation) run before the render
    std::string              result;                   // rendered wav bytes, retained in memory for download (once-only)
    std::string              mime = "audio/wav";
    std::string              error;
    long long              created_ms  = 0;
    long long              started_ms  = 0;
    long long              finished_ms = 0;
};

static std::mutex                                            g_jobs_mtx;
static std::unordered_map<std::string, std::shared_ptr<Job>> g_jobs;

static std::string new_job_id() {
    static std::atomic<uint64_t> ctr{ 0 };
    char                         buf[48];
    snprintf(buf, sizeof(buf), "job-%lld-%llu", (long long) now_ms(), (unsigned long long) ctr.fetch_add(1));
    return buf;
}

static std::shared_ptr<Job> job_get(const std::string & id) {
    std::lock_guard<std::mutex> lk(g_jobs_mtx);
    auto                        it = g_jobs.find(id);
    return it == g_jobs.end() ? nullptr : it->second;
}

// Run a job's subprocess to completion (serialized on the GPU mutex), updating the
// job's status. `args` is the full songgen-* argv; `out_path` is the wav target.
static void job_run(std::shared_ptr<Job> job, std::vector<std::string> args) {
    if (job->cancel.load()) {
        job->status.store(JobStatus::CANCELLED);
        return;
    }
    int rc;
    {
        std::lock_guard<std::mutex> lock(g_compute_mtx);
        if (job->cancel.load()) {
            job->status.store(JobStatus::CANCELLED);
            return;
        }
        job->status.store(JobStatus::RUNNING);
        job->started_ms = now_ms();
        rc              = 0;
        if (!job->pre_args.empty()) {
            // Pre-step: htdemucs separation of the full mix into vocal+bgm stems
            // (always its own subprocess, even in worker-isolation mode). The main
            // render argv references the stem paths this writes. SIGKILLable via pid.
            rc = run_subprocess(job->pre_args, nullptr, &job->pid);
            if (rc != 0 && job->error.empty()) {
                job->error = "stem separation failed (rc=" + std::to_string(rc) + ")";
            }
        }
        if (rc == 0 && !job->cancel.load()) {
            rc = dispatch_job(job, job->kind, args);
        }
    }
    job->finished_ms = now_ms();
    std::string wav;
    if (job->cancel.load()) {
        job->status.store(JobStatus::CANCELLED);
    } else if (rc == 0 && read_file(job->out_path, wav)) {
        // Retain the bytes in memory; the temp file is unlinked below. The client
        // polls /job (status only) then fetches /job?download=1 — serving from disk
        // would race the cleanup, so we keep the result and stream it once.
        job->result = std::move(wav);
        job->status.store(JobStatus::DONE);
    } else {
        if (job->error.empty()) {
            job->error = "render failed (rc=" + std::to_string(rc) + ")";
        }
        job->status.store(JobStatus::FAILED);
    }
    unlink(job->out_path.c_str());
    for (const auto & f : job->cleanup) {
        unlink(f.c_str());
    }
}

// Submit an async job: register it and detach the worker thread. Returns the id.
static std::string job_submit(const std::string & kind, std::vector<std::string> args, const std::string & out_path,
                              std::vector<std::string> cleanup = {}, std::vector<std::string> pre_args = {}) {
    auto job        = std::make_shared<Job>();
    job->id         = new_job_id();
    job->kind       = kind;
    job->out_path   = out_path;
    job->cleanup    = std::move(cleanup);
    job->pre_args   = std::move(pre_args);
    job->created_ms = now_ms();
    {
        std::lock_guard<std::mutex> lk(g_jobs_mtx);
        g_jobs[job->id] = job;
    }
    std::thread(job_run, job, std::move(args)).detach();
    return job->id;
}

// Reaper: terminal jobs (and their retained wav bytes) linger in g_jobs so the
// client can poll then download; without this they'd accumulate forever. Drop any
// finished job older than the TTL — far longer than the poll→download gap.
static constexpr long long JOB_TTL_MS = 30 * 60 * 1000;  // 30 minutes

static void jobs_reaper() {
    for (;;) {
        std::this_thread::sleep_for(std::chrono::minutes(5));
        long long now = now_ms();
        std::lock_guard<std::mutex> lk(g_jobs_mtx);
        for (auto it = g_jobs.begin(); it != g_jobs.end();) {
            JobStatus s        = it->second->status.load();
            bool      terminal = s == JobStatus::DONE || s == JobStatus::FAILED || s == JobStatus::CANCELLED;
            long long stamp    = it->second->finished_ms ? it->second->finished_ms : it->second->created_ms;
            if (terminal && now - stamp > JOB_TTL_MS) {
                it = g_jobs.erase(it);  // shared_ptr keeps any in-flight download alive
            } else {
                ++it;
            }
        }
    }
}

// Cancel a job: flag it and SIGKILL the running child (reclaims VRAM immediately).
// Subprocess model -> kill the per-job child; warm-worker model -> kill the worker
// child (the active render dies; the worker re-spawns on the next request).
static void job_cancel(std::shared_ptr<Job> job) {
    job->cancel.store(true);
    // Kill the current child if one is running. run_subprocess resets pid to -1
    // once a child exits, so this never targets a stale/reused pid. This covers
    // the non-isolation render child AND a separation pre-step (which always runs
    // as its own subprocess, even under worker isolation).
    pid_t p = job->pid.load();
    if (p > 0) {
        ::kill(p, SIGKILL);
        fprintf(stderr, "[songgen-server] job %s cancelled -> child pid=%d killed (VRAM true-0)\n", job->id.c_str(),
                (int) p);
    }
    // In worker-isolation mode the render runs in the warm worker (not a tracked
    // child) — tear it down too so VRAM drops to true-0.
    if (g_isolation) {
        worker_unload();
        fprintf(stderr, "[songgen-server] job %s cancelled -> worker killed (VRAM true-0)\n", job->id.c_str());
    }
}

static bool read_file(const std::string & path, std::string & out) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0) {
        fclose(f);
        return false;
    }
    out.resize((size_t) n);
    size_t rd = fread(&out[0], 1, (size_t) n, f);
    fclose(f);
    out.resize(rd);
    return rd > 0;
}

static std::string unique_out_path() {
    static std::atomic<uint64_t> ctr{ 0 };
    char buf[64];
    snprintf(buf, sizeof(buf), "/songgen-%d-%llu.wav", (int) getpid(),
             (unsigned long long) ctr.fetch_add(1));
    return g_cfg.tmpdir + buf;
}

// ───────────────────────────────────────────────────────────────────────────
// POST /generate  (text -> song)  — IMPLEMENTED end-to-end
// JSON body:
//   lyric        string  (required)  song lyrics, e.g. "[verse]\n...\n[chorus]\n..."
//   description  string  (required)  style prompt, e.g. "warm acoustic folk"
//   duration     number  (optional, default 15) seconds
//   seed         number  (optional, default 1234)
//   gen_type     string  (optional, "mixed"|"vocal"|"bgm", default "mixed")
//   temp         number  (optional, default 1.0)
//   top_k        number  (optional, default 250)
//   cfg          number  (optional, default 1.5)
//   fade         number  (optional ms, default 200)
//   model        string  (optional, "q8"|"q4", default "q8")
// returns: audio/wav (the generated song), or JSON error.
// ───────────────────────────────────────────────────────────────────────────
// Stage 2 — warm worker isolation (fork + length-prefixed IPC over a socketpair).
//
// When --worker-isolation is on, renders run in a long-lived forked child that this
// (CUDA-free) parent talks to over a DATA socket. The child sets g_sg_cache_on so the
// model loaders keep weights resident across requests — back-to-back jobs in one GPU
// grant skip the ~12 s reload. Parent+child share the filesystem, so a RUN frame
// marshals only the argv the parent already builds; the child writes the wav to the
// shared tmpdir and the parent reads it back (no big-payload IPC). Cancel / idle /
// /unload SIGKILL the child -> 100% VRAM reclaimed (true-0, no CUDA context here).
// ───────────────────────────────────────────────────────────────────────────
#define SONGGEN_AS_LIB
#include "songgen-cache.h"
#include "songgen-generate.cpp"  // -> sgrun_generate::run
#include "songgen-clone.cpp"     // -> sgrun_clone::run
#include "songgen-continue.cpp"  // -> sgrun_continue::run

static int    g_argc = 0;
static char **g_argv = nullptr;
static bool   g_worker_mode    = false;  // this process is the forked child
static int    g_worker_data_fd = -1;
static int    g_idle_unload_seconds = 15;  // child SIGKILL'd after N idle s (env SG_IDLE_UNLOAD_SEC; 0=off)

enum WJobKind : uint32_t { WK_GEN = 1, WK_CLONE = 2, WK_CONT = 3 };

static uint32_t kind_code(const std::string & k) {
    if (k == "clone")    return WK_CLONE;
    if (k == "continue") return WK_CONT;
    return WK_GEN;
}

enum class WFrame : uint32_t { HELLO = 1, RUN = 0x10, RESULT = 0x11 };
struct WHdr {
    uint32_t type, len, req;
};

static bool io_rw(int fd, void * buf, size_t len, bool write) {
    char * p = (char *) buf;
    size_t n = 0;
    while (n < len) {
        ssize_t r = write ? ::write(fd, p + n, len - n) : ::read(fd, p + n, len - n);
        if (r > 0) { n += (size_t) r; continue; }
        if (r == 0) return false;            // EOF
        if (errno == EINTR) continue;
        return false;
    }
    return true;
}
static bool send_frame(int fd, WFrame t, const std::string & payload) {
    WHdr h{ (uint32_t) t, (uint32_t) payload.size(), 0 };
    if (!io_rw(fd, &h, sizeof(h), true)) return false;
    return payload.empty() || io_rw(fd, (void *) payload.data(), payload.size(), true);
}
static bool recv_frame(int fd, WHdr * h, std::string * payload) {
    if (!io_rw(fd, h, sizeof(*h), false)) return false;
    if (h->len > (1u << 30)) return false;
    payload->resize(h->len);
    return h->len == 0 || io_rw(fd, &(*payload)[0], h->len, false);
}

// ── child side ──────────────────────────────────────────────────────────────
static int worker_run_loop(int data_fd) {
    g_sg_cache_on = true;  // keep loaded models resident across RUNs
    if (!send_frame(data_fd, WFrame::HELLO, "ready")) return 1;
    fprintf(stderr, "[songgen-worker] pid=%d ready (models load lazily + stay resident)\n", (int) getpid());
    for (;;) {
        WHdr        h{};
        std::string pl;
        if (!recv_frame(data_fd, &h, &pl)) break;  // parent gone -> exit (frees CUDA)
        if ((WFrame) h.type != WFrame::RUN) continue;
        // payload: [u32 kind][u32 argc][argc × (u32 len + bytes)]
        size_t off = 0;
        auto   rdu = [&](uint32_t & v) { if (off + 4 > pl.size()) { v = 0; return false; } std::memcpy(&v, pl.data() + off, 4); off += 4; return true; };
        uint32_t kind = 0, argc = 0;
        rdu(kind); rdu(argc);
        std::vector<std::string> args;
        for (uint32_t i = 0; i < argc; i++) {
            uint32_t l = 0;
            if (!rdu(l) || off + l > pl.size()) { argc = 0; break; }
            args.emplace_back(pl.data() + off, l);
            off += l;
        }
        std::vector<char *> cargv;
        for (auto & a : args) cargv.push_back(const_cast<char *>(a.c_str()));
        cargv.push_back(nullptr);
        int rc = 127;
        if (!args.empty()) {
            int n = (int) args.size();
            if (kind == WK_CLONE)      rc = sgrun_clone::run(n, cargv.data());
            else if (kind == WK_CONT)  rc = sgrun_continue::run(n, cargv.data());
            else                       rc = sgrun_generate::run(n, cargv.data());
        }
        std::string out(reinterpret_cast<const char *>(&rc), sizeof(rc));
        if (!send_frame(data_fd, WFrame::RESULT, out)) break;
    }
    return 0;
}

// ── parent side ─────────────────────────────────────────────────────────────
static std::mutex             g_wk_mtx;
static pid_t                  g_wk_pid  = -1;
static int                    g_wk_data = -1;
static std::atomic<bool>      g_wk_busy{ false };
static std::atomic<long long> g_wk_last_activity{ 0 };

static pid_t worker_spawn(int * out_data) {
    int sv[2];
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) return -1;
    pid_t pid = ::fork();
    if (pid < 0) { ::close(sv[0]); ::close(sv[1]); return -1; }
    if (pid == 0) {
        ::close(sv[0]);
        char dbuf[16];
        std::snprintf(dbuf, sizeof(dbuf), "%d", sv[1]);
        char wflag[] = "--worker";
        std::vector<char *> a;
        for (int i = 0; i < g_argc; i++) a.push_back(g_argv[i]);
        a.push_back(wflag);
        a.push_back(dbuf);
        a.push_back(nullptr);
        ::execv(g_argv[0], a.data());
        ::_exit(127);
    }
    ::close(sv[1]);
    *out_data = sv[0];
    return pid;
}

static void kill_worker_locked() {  // caller holds g_wk_mtx
    if (g_wk_pid > 0) {
        ::kill(g_wk_pid, SIGKILL);
        int ws = 0;
        ::waitpid(g_wk_pid, &ws, 0);
        fprintf(stderr, "[songgen-server] worker pid=%d killed -> VRAM true-0\n", (int) g_wk_pid);
    }
    if (g_wk_data >= 0) ::close(g_wk_data);
    g_wk_pid  = -1;
    g_wk_data = -1;
}

static bool ensure_worker_locked() {  // caller holds g_wk_mtx
    if (g_wk_pid > 0) return true;
    int   d   = -1;
    pid_t pid = worker_spawn(&d);
    if (pid < 0) { fprintf(stderr, "[songgen-server] worker spawn failed\n"); return false; }
    WHdr        h{};
    std::string pl;
    if (!recv_frame(d, &h, &pl) || (WFrame) h.type != WFrame::HELLO) {
        ::kill(pid, SIGKILL);
        int ws = 0;
        ::waitpid(pid, &ws, 0);
        ::close(d);
        fprintf(stderr, "[songgen-server] worker failed to start\n");
        return false;
    }
    g_wk_pid  = pid;
    g_wk_data = d;
    fprintf(stderr, "[songgen-server] worker pid=%d spawned (isolation)\n", (int) pid);
    return true;
}

static void worker_unload() {
    if (!g_isolation) return;
    std::lock_guard<std::mutex> lk(g_wk_mtx);
    kill_worker_locked();
}

// Marshal one job to the worker and block for its RESULT rc. The blocking recv runs
// WITHOUT g_wk_mtx so cancel/unload/watchdog can SIGKILL the child mid-render (which
// makes the recv fail -> rc<0 -> job finalized as cancelled/failed).
static int dispatch_remote(const std::string & kind, const std::vector<std::string> & args) {
    int   data_fd;
    pid_t pid;
    {
        std::lock_guard<std::mutex> lk(g_wk_mtx);
        if (!ensure_worker_locked()) return -1;
        data_fd = g_wk_data;
        pid     = g_wk_pid;
        g_wk_busy.store(true);
        std::string p;
        auto        pku = [&](uint32_t v) { p.append((const char *) &v, 4); };
        pku(kind_code(kind));
        pku((uint32_t) args.size());
        for (auto & a : args) { pku((uint32_t) a.size()); p.append(a); }
        if (!send_frame(data_fd, WFrame::RUN, p)) {
            kill_worker_locked();
            g_wk_busy.store(false);
            g_wk_last_activity.store(now_ms());
            return -1;
        }
    }
    WHdr        h{};
    std::string resp;
    bool        got = recv_frame(data_fd, &h, &resp) && (WFrame) h.type == WFrame::RESULT && resp.size() >= sizeof(int);
    {
        std::lock_guard<std::mutex> lk(g_wk_mtx);
        g_wk_busy.store(false);
        g_wk_last_activity.store(now_ms());
        if (!got) {
            if (g_wk_pid == pid) kill_worker_locked();  // worker died (crash or SIGKILL)
            return -1;
        }
    }
    int rc = 0;
    std::memcpy(&rc, resp.data(), sizeof(rc));
    return rc;
}

// Idle watchdog: SIGKILL the worker once it has been idle (no in-flight RUN) for
// g_idle_unload_seconds, dropping VRAM to true-0 between grants.
static void worker_watchdog_main() {
    for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (g_idle_unload_seconds <= 0 || g_wk_busy.load()) continue;
        long long thresh = (long long) g_idle_unload_seconds * 1000;
        if (now_ms() - g_wk_last_activity.load() < thresh) continue;
        std::lock_guard<std::mutex> lk(g_wk_mtx);
        if (g_wk_pid > 0 && !g_wk_busy.load() && now_ms() - g_wk_last_activity.load() >= thresh) {
            kill_worker_locked();
        }
    }
}

// dispatch a job either to the warm worker (isolation) or a fresh subprocess.
static int dispatch_job(std::shared_ptr<Job> job, const std::string & kind, const std::vector<std::string> & args) {
    if (g_isolation) {
        return dispatch_remote(kind, args);
    }
    return run_subprocess(args, nullptr, &job->pid);
}

// Finalize a long compute request: async by default (register a job, return
// {job_id} 202); ?wait=1 (or ?sync=1) blocks and streams the wav for simple
// clients. /separate stays fully synchronous (it's fast) and does not use this.
static void respond_compute(const httplib::Request & req, httplib::Response & res, const std::string & kind,
                            std::vector<std::string> args, const std::string & out_wav,
                            std::vector<std::string> cleanup = {}, std::vector<std::string> pre_args = {}) {
    bool want_sync = req.has_param("wait") || req.has_param("sync");
    if (!want_sync) {
        std::string      id   = job_submit(kind, std::move(args), out_wav, std::move(cleanup), std::move(pre_args));
        yyjson_mut_doc * doc  = yyjson_mut_doc_new(nullptr);
        yyjson_mut_val * root = yyjson_mut_obj(doc);
        yyjson_mut_doc_set_root(doc, root);
        yyjson_mut_obj_add_strcpy(doc, root, "job_id", id.c_str());
        yyjson_mut_obj_add_str(doc, root, "status", "pending");
        yyjson_mut_obj_add_str(doc, root, "poll", "GET /job?id=<job_id> (&download=1 for wav, &cancel=1 to cancel)");
        char * out = yyjson_mut_write(doc, 0, nullptr);
        res.status = 202;
        res.set_content(out ? out : "{}", "application/json");
        free(out);
        yyjson_mut_doc_free(doc);
        fprintf(stderr, "[songgen-server] /%s queued as %s\n", kind.c_str(), id.c_str());
        return;
    }
    int rc = 0;
    {
        std::lock_guard<std::mutex> lock(g_compute_mtx);
        if (!pre_args.empty()) {
            rc = run_subprocess(pre_args);  // htdemucs separation pre-step (mix -> stems)
        }
        if (rc == 0) {
            rc = g_isolation ? dispatch_remote(kind, args) : run_subprocess(args);
        }
    }
    std::string wav;
    bool        ok = (rc == 0) && read_file(out_wav, wav);
    unlink(out_wav.c_str());
    for (const auto & f : cleanup) {
        unlink(f.c_str());
    }
    if (!ok) {
        json_error(res, 500, rc != 0 ? "render failed" : "render produced no output");
        return;
    }
    res.set_content(wav, "audio/wav");
    fprintf(stderr, "[songgen-server] /%s done sync (%zu bytes)\n", kind.c_str(), wav.size());
}

static void handle_generate(const httplib::Request & req, httplib::Response & res) {
    yyjson_doc * doc = yyjson_read(req.body.c_str(), req.body.size(), 0);
    if (!doc) {
        json_error(res, 400, "Invalid JSON");
        return;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    if (!root || !yyjson_is_obj(root)) {
        yyjson_doc_free(doc);
        json_error(res, 400, "Body must be a JSON object");
        return;
    }

    std::string lyric       = json_get_str(root, "lyric", "");
    std::string description = json_get_str(root, "description", "");
    if (lyric.empty() || description.empty()) {
        yyjson_doc_free(doc);
        json_error(res, 400, "Both 'lyric' and 'description' are required");
        return;
    }

    bool        has_duration = json_has(root, "duration");
    double      duration = json_get_num(root, "duration", 15.0);
    double      seed     = json_get_num(root, "seed", 1234.0);
    std::string gen_type = json_get_str(root, "gen_type", "mixed");
    std::string model    = json_get_str(root, "model", "q8");
    bool        has_temp = json_has(root, "temp");
    bool        has_topk = json_has(root, "top_k");
    bool        has_cfg  = json_has(root, "cfg");
    bool        has_fade = json_has(root, "fade");
    double      temp     = json_get_num(root, "temp", 1.0);
    double      top_k    = json_get_num(root, "top_k", 250.0);
    double      cfg      = json_get_num(root, "cfg", 1.5);
    double      fade     = json_get_num(root, "fade", 200.0);
    yyjson_doc_free(doc);

    if (gen_type != "mixed" && gen_type != "vocal" && gen_type != "bgm") {
        json_error(res, 400, "gen_type must be one of: mixed, vocal, bgm");
        return;
    }
    if (model != "q8" && model != "q4") {
        json_error(res, 400, "model must be one of: q8, q4");
        return;
    }
    if (has_duration && (duration <= 0.0 || duration > (double) g_cfg.max_duration)) {
        json_error(res, 400, "duration out of range");
        return;
    }

    std::string out_wav = unique_out_path();

    std::vector<std::string> args = {
        g_cfg.bindir + "/songgen-generate",
        lelm_path(model),
        gguf_path("songgen-cfm.gguf"),
        gguf_path("songgen-vae.gguf"),
        gguf_path("songgen-septoken-aux.gguf"),
        g_cfg.golden,
        out_wav,
        std::to_string((unsigned long long) seed),
        "--lyric", lyric,
        "--description", description,
        "--gen-type", gen_type,
    };
    // No "duration" in request → generate to the model max (270 s) and stop on EOS (upstream
    // behaviour). Growing KV keeps VRAM proportional to the real length. An explicit duration is
    // a hard cap (truncate) for bounding length/VRAM in a given bucket.
    if (has_duration) { args.push_back("--duration"); args.push_back(std::to_string(duration)); }
    if (has_temp) { args.push_back("--temp");  args.push_back(std::to_string(temp)); }
    if (has_topk) { args.push_back("--top-k"); args.push_back(std::to_string((int) top_k)); }
    if (has_cfg)  { args.push_back("--cfg");   args.push_back(std::to_string(cfg)); }
    if (has_fade) { args.push_back("--fade");  args.push_back(std::to_string(fade)); }

    fprintf(stderr, "[songgen-server] /generate model=%s dur=%.1fs seed=%llu type=%s\n",
            model.c_str(), duration, (unsigned long long) seed, gen_type.c_str());

    respond_compute(req, res, "generate", std::move(args), out_wav);
}

// ───────────────────────────────────────────────────────────────────────────
// clone / continue / separate — IMPLEMENTED via multipart/form-data upload.
//
// Audio inputs arrive as multipart file parts; text params as multipart fields
// (or, for backward-friendliness, none required beyond the files + lyric). The
// wrapped CLI runs under the same global compute lock as /generate. Temp wavs
// live in g_cfg.tmpdir and are unlinked after the response is built.
// ───────────────────────────────────────────────────────────────────────────

// base64 encoder (for /separate JSON response carrying two wavs).
static std::string b64_encode(const std::string & in) {
    static const char * T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string         out;
    out.reserve(((in.size() + 2) / 3) * 4);
    size_t i = 0;
    for (; i + 2 < in.size(); i += 3) {
        unsigned n = (unsigned char) in[i] << 16 | (unsigned char) in[i + 1] << 8 | (unsigned char) in[i + 2];
        out += T[(n >> 18) & 63]; out += T[(n >> 12) & 63]; out += T[(n >> 6) & 63]; out += T[n & 63];
    }
    if (i < in.size()) {
        unsigned n = (unsigned char) in[i] << 16;
        if (i + 1 < in.size()) n |= (unsigned char) in[i + 1] << 8;
        out += T[(n >> 18) & 63];
        out += T[(n >> 12) & 63];
        out += (i + 1 < in.size()) ? T[(n >> 6) & 63] : '=';
        out += '=';
    }
    return out;
}

// Write a multipart file part's content to a unique temp .wav. Returns "" if absent.
static std::string save_upload(const httplib::Request & req, const char * field) {
    if (!req.form.has_file(field)) {
        return "";
    }
    const auto   f    = req.form.get_file(field);
    std::string  path = unique_out_path();  // .../songgen-<pid>-<n>.wav
    FILE *       fp   = fopen(path.c_str(), "wb");
    if (!fp) {
        return "";
    }
    fwrite(f.content.data(), 1, f.content.size(), fp);
    fclose(fp);
    return path;
}

// multipart field value (text), else default.
static std::string form_str(const httplib::Request & req, const char * key, const char * dflt) {
    if (req.form.has_field(key)) {  // multipart text field
        return req.form.get_field(key);
    }
    if (req.form.has_file(key)) {   // file part used as a value
        return req.form.get_file(key).content;
    }
    if (req.has_param(key)) {       // urlencoded / query param
        return req.get_param_value(key);
    }
    return dflt;
}
static bool form_has(const httplib::Request & req, const char * key) {
    return req.form.has_field(key) || req.form.has_file(key) || req.has_param(key);
}

// Shared driver for clone/continue: assemble argv, run, return wav.
static void run_stem_job(const httplib::Request & req, httplib::Response & res, bool is_continue) {
    std::string lyric = form_str(req, "lyric", "");
    if (lyric.empty()) {
        json_error(res, 400, "'lyric' is required");
        return;
    }
    std::string vocal = save_upload(req, "vocal_stem");
    std::string bgm   = save_upload(req, "bgm_stem");
    std::string mix   = save_upload(req, "full_mix");  // optional reference / auto-separation source
    if (mix.empty()) {
        mix = save_upload(req, "mix");                 // accept 'mix' as an alias for 'full_mix'
    }

    // Auto-separate: if the caller didn't supply pre-split stems but did give a full
    // mix, run htdemucs ourselves (a pre-step inside this job) to derive vocal+bgm,
    // then clone over them. The original mix is still passed as --full-mix (reference).
    std::vector<std::string> pre_args;
    if ((vocal.empty() || bgm.empty()) && !mix.empty()) {
        if (!vocal.empty()) unlink(vocal.c_str());
        if (!bgm.empty())   unlink(bgm.c_str());
        vocal    = unique_out_path();
        bgm      = unique_out_path();
        pre_args = { g_cfg.bindir + "/songgen-separate", gguf_path("songgen-htdemucs.gguf"),
                     mix, vocal, bgm };
    } else if (vocal.empty() || bgm.empty()) {
        if (!vocal.empty()) unlink(vocal.c_str());
        if (!bgm.empty())   unlink(bgm.c_str());
        if (!mix.empty())   unlink(mix.c_str());
        json_error(res, 400,
                   "provide both 'vocal_stem' and 'bgm_stem', or a 'full_mix'/'mix' to auto-separate");
        return;
    }

    std::string model    = form_str(req, "model", "q8");
    std::string gen_type = form_str(req, "gen_type", "mixed");
    double      duration = atof(form_str(req, "duration", "15").c_str());
    double      seed     = atof(form_str(req, "seed", "1234").c_str());
    if (model != "q8" && model != "q4") model = "q8";
    if (duration <= 0.0 || duration > (double) g_cfg.max_duration) duration = 15.0;

    std::string out_wav = unique_out_path();
    const char * exe    = is_continue ? "/songgen-continue" : "/songgen-clone";

    // both CLIs: <lelm> <cfm> <vae> [<vae-encoder> (continue only)] <septoken-aux>
    //            <1rvq-aux> <musicfm-sep> <musicfm-1rvq> <out.wav> [seed] --vocal-stem ...
    std::vector<std::string> args = { g_cfg.bindir + exe, lelm_path(model), gguf_path("songgen-cfm.gguf"),
                                      gguf_path("songgen-vae.gguf") };
    if (is_continue) {
        args.push_back(gguf_path("songgen-vae-encoder.gguf"));
    }
    args.push_back(gguf_path("songgen-septoken-aux.gguf"));
    args.push_back(gguf_path("songgen-1rvq-aux.gguf"));
    args.push_back(gguf_path("songgen-musicfm.gguf"));
    args.push_back(gguf_path("songgen-musicfm-1rvq.gguf"));
    args.push_back(out_wav);
    args.push_back(std::to_string((unsigned long long) seed));
    args.push_back("--vocal-stem"); args.push_back(vocal);
    args.push_back("--bgm-stem");   args.push_back(bgm);
    if (!mix.empty()) { args.push_back("--full-mix"); args.push_back(mix); }
    args.push_back("--lyric");      args.push_back(lyric);
    if (form_has(req, "description")) { args.push_back("--description"); args.push_back(form_str(req, "description", "")); }
    args.push_back("--duration");   args.push_back(std::to_string(duration));
    args.push_back("--gen-type");   args.push_back(gen_type);
    if (form_has(req, "temp"))  { args.push_back("--temp");  args.push_back(form_str(req, "temp", "1.0")); }
    if (form_has(req, "top_k")) { args.push_back("--top-k"); args.push_back(form_str(req, "top_k", "250")); }
    if (form_has(req, "cfg"))   { args.push_back("--cfg");   args.push_back(form_str(req, "cfg", "1.5")); }
    if (form_has(req, "fade"))  { args.push_back("--fade");  args.push_back(form_str(req, "fade", "200")); }

    fprintf(stderr, "[songgen-server] /%s model=%s dur=%.1fs seed=%llu type=%s\n",
            is_continue ? "continue" : "clone", model.c_str(), duration, (unsigned long long) seed, gen_type.c_str());

    std::vector<std::string> cleanup = { vocal, bgm };
    if (!mix.empty()) {
        cleanup.push_back(mix);
    }
    respond_compute(req, res, is_continue ? "continue" : "clone", std::move(args), out_wav, std::move(cleanup),
                    std::move(pre_args));
}

static void handle_clone(const httplib::Request & req, httplib::Response & res) {
    run_stem_job(req, res, /*is_continue=*/false);
}

static void handle_continue(const httplib::Request & req, httplib::Response & res) {
    run_stem_job(req, res, /*is_continue=*/true);
}

// /separate (htdemucs): multipart 'mix' -> JSON { vocal: <b64 wav>, bgm: <b64 wav> }.
static void handle_separate(const httplib::Request & req, httplib::Response & res) {
    std::string mix = save_upload(req, "mix");
    if (mix.empty()) {
        json_error(res, 400, "multipart 'mix' audio file is required");
        return;
    }
    std::string vocal_out = unique_out_path();
    std::string bgm_out   = unique_out_path();
    std::vector<std::string> args = { g_cfg.bindir + "/songgen-separate", gguf_path("songgen-htdemucs.gguf"),
                                      mix, vocal_out, bgm_out };
    int rc;
    {
        std::lock_guard<std::mutex> lock(g_compute_mtx);
        rc = run_subprocess(args);  // htdemucs CUDA forward verified correct (cos 1.0 vs CPU) — run on GPU
    }
    unlink(mix.c_str());
    if (rc != 0) {
        unlink(vocal_out.c_str());
        unlink(bgm_out.c_str());
        char msg[128];
        snprintf(msg, sizeof(msg), "separation failed (exit %d)", rc);
        json_error(res, 500, msg);
        return;
    }
    std::string vwav, bwav;
    if (!read_file(vocal_out, vwav) || !read_file(bgm_out, bwav)) {
        unlink(vocal_out.c_str());
        unlink(bgm_out.c_str());
        json_error(res, 500, "separation produced no output");
        return;
    }
    unlink(vocal_out.c_str());
    unlink(bgm_out.c_str());

    yyjson_mut_doc * doc  = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_strcpy(doc, root, "vocal", b64_encode(vwav).c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "bgm", b64_encode(bwav).c_str());
    char * out = yyjson_mut_write(doc, 0, nullptr);
    res.set_content(out ? out : "{}", "application/json");
    free(out);
    yyjson_mut_doc_free(doc);
    fprintf(stderr, "[songgen-server] /separate done (vocal %zu + bgm %zu bytes)\n", vwav.size(), bwav.size());
}

// GET /job?id=<id>            -> JSON status
//     /job?id=<id>&download=1 -> the wav (audio/wav) once status==done
//     /job?id=<id>&cancel=1   -> cancel a pending/running job (SIGKILL child -> true-0 VRAM)
static void handle_job(const httplib::Request & req, httplib::Response & res) {
    std::string id = req.has_param("id") ? req.get_param_value("id") : "";
    if (id.empty()) {
        json_error(res, 400, "query param 'id' is required");
        return;
    }
    auto job = job_get(id);
    if (!job) {
        json_error(res, 404, "unknown job id");
        return;
    }
    if (req.has_param("cancel")) {
        JobStatus s = job->status.load();
        if (s == JobStatus::PENDING || s == JobStatus::RUNNING) {
            job_cancel(job);
        }
    }
    JobStatus st = job->status.load();
    if (req.has_param("download")) {
        if (st != JobStatus::DONE) {
            json_error(res, 409, "job not done");
            return;
        }
        std::string wav;
        {
            // Take the retained bytes (once-only). job_run keeps the result in
            // memory and unlinks the temp file; re-download is not supported.
            std::lock_guard<std::mutex> lk(g_jobs_mtx);
            wav.swap(job->result);
        }
        if (wav.empty()) {
            json_error(res, 410, "result no longer available (already retrieved)");
            return;
        }
        res.set_content(std::move(wav), job->mime);
        return;
    }
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_strcpy(doc, root, "job_id", id.c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "kind", job->kind.c_str());
    yyjson_mut_obj_add_str(doc, root, "status", job_status_name(st));
    if (st == JobStatus::DONE) {
        yyjson_mut_obj_add_str(doc, root, "result", "GET /job?id=<id>&download=1");
    }
    if (st == JobStatus::FAILED && !job->error.empty()) {
        yyjson_mut_obj_add_strcpy(doc, root, "error", job->error.c_str());
    }
    if (job->started_ms) {
        long long end = job->finished_ms ? job->finished_ms : now_ms();
        yyjson_mut_obj_add_int(doc, root, "elapsed_ms", end - job->started_ms);
    }
    char * out = yyjson_mut_write(doc, 0, nullptr);
    res.set_content(out ? out : "{}", "application/json");
    free(out);
    yyjson_mut_doc_free(doc);
}

// POST /unload: cancel/kill any in-flight render so VRAM drops to true-0. The
// subprocess worker model already holds no VRAM while idle (no resident weights,
// no CUDA context in this parent), so this just tears down an active child.
static void handle_unload(const httplib::Request &, httplib::Response & res) {
    int killed = 0;
    {
        std::lock_guard<std::mutex> lk(g_jobs_mtx);
        for (auto & kv : g_jobs) {
            JobStatus s = kv.second->status.load();
            if (s == JobStatus::PENDING || s == JobStatus::RUNNING) {
                job_cancel(kv.second);
                killed++;
            }
        }
    }
    worker_unload();  // drop the resident warm worker (if any) -> true-0, even when idle
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_str(doc, root, "status", "ok");
    yyjson_mut_obj_add_int(doc, root, "cancelled", killed);
    yyjson_mut_obj_add_str(doc, root, "vram", "true-0 (no resident weights in the parent)");
    char * out = yyjson_mut_write(doc, 0, nullptr);
    res.set_content(out ? out : "{}", "application/json");
    free(out);
    yyjson_mut_doc_free(doc);
    fprintf(stderr, "[songgen-server] /unload: cancelled %d in-flight job(s)\n", killed);
}

// ───────────────────────────────────────────────────────────────────────────
// main
// ───────────────────────────────────────────────────────────────────────────
static void usage(const char * prog) {
    fprintf(stderr,
            "Usage: %s [options]\n"
            "  --host <addr>     listen address (default %s)\n"
            "  --port <N>        listen port (default %d)\n"
            "  --gguf <dir>      gguf directory (default %s)\n"
            "  --golden <dir>    golden conditioning dir (default %s)\n"
            "  --bindir <dir>    dir holding songgen-* binaries (default: this exe's dir)\n"
            "  --tmpdir <dir>    scratch dir for output wavs (default %s)\n"
            "  --max-duration <s> reject /generate above this (default %.0f)\n",
            prog, g_cfg.host.c_str(), g_cfg.port, g_cfg.gguf.c_str(), g_cfg.golden.c_str(),
            g_cfg.tmpdir.c_str(), g_cfg.max_duration);
}

static std::string exe_dir() {
    char    buf[4096];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) {
        return ".";
    }
    buf[n] = '\0';
    std::string p(buf);
    auto        slash = p.find_last_of('/');
    return slash == std::string::npos ? "." : p.substr(0, slash);
}

int main(int argc, char ** argv) {
    g_cfg.bindir = exe_dir();
    g_argc       = argc;
    g_argv       = argv;
    if (const char * e = getenv("SG_WORKER_ISOLATION")) g_isolation = atoi(e) != 0;
    if (const char * e = getenv("SG_IDLE_UNLOAD_SEC")) g_idle_unload_seconds = atoi(e);

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--worker" && i + 1 < argc) {
            g_worker_mode    = true;
            g_worker_data_fd = atoi(argv[++i]);
        } else if (a == "--worker-isolation") {
            g_isolation = true;
        } else if (a == "--host" && i + 1 < argc) {
            g_cfg.host = argv[++i];
        } else if (a == "--port" && i + 1 < argc) {
            g_cfg.port = atoi(argv[++i]);
        } else if (a == "--gguf" && i + 1 < argc) {
            g_cfg.gguf = argv[++i];
        } else if (a == "--golden" && i + 1 < argc) {
            g_cfg.golden = argv[++i];
        } else if (a == "--bindir" && i + 1 < argc) {
            g_cfg.bindir = argv[++i];
        } else if (a == "--tmpdir" && i + 1 < argc) {
            g_cfg.tmpdir = argv[++i];
        } else if (a == "--max-duration" && i + 1 < argc) {
            g_cfg.max_duration = (float) atof(argv[++i]);
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "[songgen-server] unknown arg: %s\n", a.c_str());
            usage(argv[0]);
            return 1;
        }
    }

    // Forked worker child: serve RUN frames over the inherited socket; never binds HTTP.
    if (g_worker_mode) {
        return worker_run_loop(g_worker_data_fd);
    }

    std::thread(jobs_reaper).detach();  // bound g_jobs growth (terminal jobs + retained wavs)

    httplib::Server svr;
    g_svr = &svr;
    svr.set_read_timeout(600);
    svr.set_write_timeout(600);
    svr.set_payload_max_length(256 * 1024 * 1024);
    svr.set_socket_options([](socket_t sock) {
        int one = 1;
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    });

    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        std::string body = std::string("{\"status\":\"ok\",\"version\":\"") + ACE_VERSION + "\"}";
        res.set_content(body, "application/json");
    });
    svr.Post("/generate", handle_generate);
    svr.Post("/clone", handle_clone);
    svr.Post("/continue", handle_continue);
    svr.Post("/separate", handle_separate);
    svr.Get("/job", handle_job);       // poll/download/cancel an async job
    svr.Post("/job", handle_job);      // (POST accepted too, e.g. for &cancel=1)
    svr.Post("/unload", handle_unload);  // GPU-guard preempt: kill in-flight -> true-0 VRAM

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGCHLD, SIG_DFL);  // ensure waitpid sees child exits

    fprintf(stderr, "[songgen-server] songgen.cpp %s\n", ACE_VERSION);
    fprintf(stderr, "[songgen-server] bindir=%s gguf=%s golden=%s\n", g_cfg.bindir.c_str(),
            g_cfg.gguf.c_str(), g_cfg.golden.c_str());
    fprintf(stderr, "[songgen-server] listening on %s:%d\n", g_cfg.host.c_str(), g_cfg.port);
    if (g_isolation) {
        fprintf(stderr, "[songgen-server] worker isolation ON (warm resident models; idle-unload %ds; /unload -> true-0)\n",
                g_idle_unload_seconds);
        g_wk_last_activity.store(now_ms());
        std::thread(worker_watchdog_main).detach();
    } else {
        fprintf(stderr, "[songgen-server] worker isolation OFF (subprocess per request; true-0 idle by teardown)\n");
    }
    fprintf(stderr, "[songgen-server] endpoints: GET /health, POST /generate|/clone|/continue (async: {job_id}; "
                    "?wait=1 to block), POST /separate (sync), GET /job?id=, POST /unload\n");

    if (!svr.listen(g_cfg.host.c_str(), g_cfg.port)) {
        fprintf(stderr, "[songgen-server] FATAL: cannot bind %s:%d\n", g_cfg.host.c_str(), g_cfg.port);
        return 1;
    }
    fprintf(stderr, "[songgen-server] shutting down\n");
    return 0;
}
