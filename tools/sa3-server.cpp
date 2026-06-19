// sa3-server.cpp: HTTP API for the Stable Audio 3 C++/ggml port (sa3-gen pipeline).
//
// Folds under the koblem "music" GPU engine beside acestep — a light-GPU peer (all-Q8
// weights ≈ 2.67 GB ≤ 3 GB), NOT a separate engine. Exposes the full SA3 surface:
//   POST /generate  text -> audio (t2a)
//   POST /edit      audio + prompt -> audio (a2a SDEdit / inpaint / continuation)
// async by default (returns {job_id}; poll GET /job; ?wait=1 blocks), with cancel and
// idle-unload -> VRAM true-0.
//
// Worker model (mirrors songgen-server): a single global compute mutex serializes the
// GPU (one render at a time). With --worker-isolation (default in prod) renders run in a
// long-lived forked child that this CUDA-free parent talks to over a socketpair; the
// child sets g_sa3_cache_on so the sa3-gen model loaders keep weights RESIDENT across
// requests (sa3-cache.h), killing the ~0.49 s per-invocation H2D weight upload that
// dominates short-SFX wall time. Cancel / idle / /unload SIGKILL the child -> true-0.
// Without isolation, each request execs the sa3-gen CLI fresh (true-0 idle by teardown,
// but pays the full load every time).
//
// Paths: --gguf <dir> holds sa3-{t5gemma,dit,same}-{Q8_0,Q6_K}.gguf (one file each
// carries everything — the SAME gguf also has the conditioner + encoder, the t5gemma
// gguf the tokenizer). model=q8 (default) / q6 picks the quant.

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
// Config (resolved at startup; defaults target the prod /models layout)
// ───────────────────────────────────────────────────────────────────────────
struct Config {
    std::string host         = "127.0.0.1";
    int         port         = 8104;
    std::string gguf         = "/home/dbrain/dev/acestep-sa3/models";
    std::string bindir;  // dir holding sa3-gen (default: this exe's dir)
    std::string tmpdir       = "/tmp";
    float       max_duration = 380.0f;  // SA3 hard max (sample_size 16,777,216 / 44.1kHz)
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
// gguf path resolution. model: "q6" -> Q6_K, anything else -> Q8_0 (the default).
// ───────────────────────────────────────────────────────────────────────────
static std::string quant_suffix(const std::string & model) {
    return (model == "q6") ? "Q6_K" : "Q8_0";
}
static std::string gguf_path(const char * comp, const std::string & model) {
    return g_cfg.gguf + "/sa3-" + comp + "-" + quant_suffix(model) + ".gguf";
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

// ───────────────────────────────────────────────────────────────────────────
// subprocess exec (non-isolation fallback): run argv, inherit stderr (logs), wait.
// Publishes the child pid before the blocking wait so a /job cancel (or /unload) can
// SIGKILL it mid-render -> waitpid reports a signal (not WIFEXITED) -> we return -1
// and the caller finalizes the job CANCELLED/FAILED; VRAM drops to true-0 on teardown.
// ───────────────────────────────────────────────────────────────────────────
static int run_subprocess(const std::vector<std::string> & args, std::atomic<pid_t> * pid_out = nullptr) {
    std::vector<char *> argv;
    argv.reserve(args.size() + 1);
    for (const auto & a : args) {
        argv.push_back(const_cast<char *>(a.c_str()));
    }
    argv.push_back(nullptr);

    pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "[sa3-server] fork failed: %s\n", strerror(errno));
        return -1;
    }
    if (pid == 0) {
        execv(argv[0], argv.data());
        fprintf(stderr, "[sa3-server] execv %s failed: %s\n", argv[0], strerror(errno));
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

// Worker isolation (warm resident worker). Forward-declared for job_run / job_cancel.
static bool g_isolation = false;
struct Job;
static int  dispatch_job(std::shared_ptr<Job> job, const std::vector<std::string> & args);
static void worker_unload();  // SIGKILL the warm child -> VRAM true-0

// ───────────────────────────────────────────────────────────────────────────
// Async job registry. The compute endpoints return a job id immediately and run on a
// detached thread; the client polls /job. Cancel SIGKILLs the running child (true-0
// VRAM). GPU work stays serialized by g_compute_mtx.
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
    std::string              id;
    std::string              kind;  // "generate" | "edit"
    std::atomic<JobStatus>   status{ JobStatus::PENDING };
    std::atomic<pid_t>       pid{ -1 };  // running child (non-isolation, for cancel)
    std::atomic<bool>        cancel{ false };
    std::string              out_path;  // wav written here on success
    std::vector<std::string> cleanup;   // temp upload files to unlink when done
    std::string              result;    // rendered wav bytes, retained for download (once-only)
    std::string              mime = "audio/wav";
    std::string              error;
    long long                created_ms  = 0;
    long long                started_ms  = 0;
    long long                finished_ms = 0;
};

static std::mutex                                           g_jobs_mtx;
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

// Run a job's render to completion (serialized on the GPU mutex), updating status.
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
        rc              = dispatch_job(job, args);
    }
    job->finished_ms = now_ms();
    std::string wav;
    if (job->cancel.load()) {
        job->status.store(JobStatus::CANCELLED);
    } else if (rc == 0 && read_file(job->out_path, wav)) {
        // Retain the bytes; the client polls /job then fetches /job?download=1 (serving
        // from disk would race the cleanup below).
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
                              std::vector<std::string> cleanup = {}) {
    auto job        = std::make_shared<Job>();
    job->id         = new_job_id();
    job->kind       = kind;
    job->out_path   = out_path;
    job->cleanup    = std::move(cleanup);
    job->created_ms = now_ms();
    {
        std::lock_guard<std::mutex> lk(g_jobs_mtx);
        g_jobs[job->id] = job;
    }
    std::thread(job_run, job, std::move(args)).detach();
    return job->id;
}

// Reaper: terminal jobs (and their retained wav bytes) linger so the client can poll
// then download; drop any finished job older than the TTL.
static constexpr long long JOB_TTL_MS = 30 * 60 * 1000;  // 30 minutes

static void jobs_reaper() {
    for (;;) {
        std::this_thread::sleep_for(std::chrono::minutes(5));
        long long                   now = now_ms();
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
static void job_cancel(std::shared_ptr<Job> job) {
    job->cancel.store(true);
    pid_t p = job->pid.load();
    if (p > 0) {
        ::kill(p, SIGKILL);  // non-isolation render child
        fprintf(stderr, "[sa3-server] job %s cancelled -> child pid=%d killed (VRAM true-0)\n", job->id.c_str(),
                (int) p);
    }
    if (g_isolation) {
        worker_unload();  // warm worker holds the resident weights -> tear it down too
        fprintf(stderr, "[sa3-server] job %s cancelled -> worker killed (VRAM true-0)\n", job->id.c_str());
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

static std::string unique_path(const char * ext) {
    static std::atomic<uint64_t> ctr{ 0 };
    char                         buf[64];
    snprintf(buf, sizeof(buf), "/sa3-%d-%llu%s", (int) getpid(), (unsigned long long) ctr.fetch_add(1), ext);
    return g_cfg.tmpdir + buf;
}

// ───────────────────────────────────────────────────────────────────────────
// Warm worker isolation (fork + length-prefixed IPC over a socketpair).
//
// The render body is sa3-gen's run() compiled in-process via -DSA3_GEN_AS_LIB. The
// child sets g_sa3_cache_on (sa3-cache.h) so weights stay RESIDENT across RUNs. Parent
// + child share the filesystem, so a RUN frame marshals only the argv (the parent
// already built it); the child writes the wav to the shared tmpdir and the parent reads
// it back. Cancel / idle / /unload SIGKILL the child -> 100% VRAM reclaimed (true-0).
// ───────────────────────────────────────────────────────────────────────────
#define SA3_GEN_AS_LIB
#include "sa3-gen.cpp"  // -> sa3run_gen::run  (+ defines g_sa3_cache_on via sa3-cache.h)

static int    g_argc                = 0;
static char **g_argv                = nullptr;
static bool   g_worker_mode         = false;  // this process is the forked child
static int    g_worker_data_fd      = -1;
static int    g_idle_unload_seconds = 15;  // child SIGKILL'd after N idle s (SA3_IDLE_UNLOAD_SEC; 0=off)

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
        if (r == 0) return false;  // EOF
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
    g_sa3_cache_on = true;  // keep loaded models resident across RUNs (sa3-cache.h)
    if (!send_frame(data_fd, WFrame::HELLO, "ready")) return 1;
    fprintf(stderr, "[sa3-worker] pid=%d ready (models load on first use + stay resident)\n", (int) getpid());
    for (;;) {
        WHdr        h{};
        std::string pl;
        if (!recv_frame(data_fd, &h, &pl)) break;  // parent gone -> exit (frees CUDA)
        if ((WFrame) h.type != WFrame::RUN) continue;
        // payload: [u32 argc][argc × (u32 len + bytes)]
        size_t off = 0;
        auto   rdu = [&](uint32_t & v) {
            if (off + 4 > pl.size()) { v = 0; return false; }
            std::memcpy(&v, pl.data() + off, 4);
            off += 4;
            return true;
        };
        uint32_t argc = 0;
        rdu(argc);
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
            rc = sa3run_gen::run((int) args.size(), cargv.data());
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
// GPU placement (multi-GPU scheduler): default card (UUID) for un-targeted
// requests, pending per-request target, and the GPU the warm worker is on.
static std::string            g_default_gpu;   // from WORKER_DEFAULT_GPU env
static std::string            g_next_gpu;      // pending per-request override (g_wk_mtx)
static std::string            g_wk_gpu;        // GPU the live worker is pinned to (g_wk_mtx)

static pid_t worker_spawn(int * out_data, const std::string & gpu) {
    int sv[2];
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) return -1;
    pid_t pid = ::fork();
    if (pid < 0) { ::close(sv[0]); ::close(sv[1]); return -1; }
    if (pid == 0) {
        ::close(sv[0]);
        // Pin this worker to a specific GPU before execv (parent is CUDA-free).
        if (!gpu.empty()) ::setenv("CUDA_VISIBLE_DEVICES", gpu.c_str(), 1);
        char dbuf[16];
        std::snprintf(dbuf, sizeof(dbuf), "%d", sv[1]);
        char                wflag[] = "--worker";
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
        fprintf(stderr, "[sa3-server] worker pid=%d killed -> VRAM true-0\n", (int) g_wk_pid);
    }
    if (g_wk_data >= 0) ::close(g_wk_data);
    g_wk_pid  = -1;
    g_wk_data = -1;
}

static bool ensure_worker_locked() {  // caller holds g_wk_mtx
    const std::string want_gpu = g_next_gpu.empty() ? g_default_gpu : g_next_gpu;
    if (g_wk_pid > 0 && g_wk_gpu == want_gpu) return true;
    if (g_wk_pid > 0) {  // alive but wrong card → relocate
        fprintf(stderr, "[sa3-server] relocating worker '%s' -> '%s'\n", g_wk_gpu.c_str(), want_gpu.c_str());
        kill_worker_locked();
    }
    int   d   = -1;
    pid_t pid = worker_spawn(&d, want_gpu);
    if (pid < 0) { fprintf(stderr, "[sa3-server] worker spawn failed\n"); return false; }
    WHdr        h{};
    std::string pl;
    if (!recv_frame(d, &h, &pl) || (WFrame) h.type != WFrame::HELLO) {
        ::kill(pid, SIGKILL);
        int ws = 0;
        ::waitpid(pid, &ws, 0);
        ::close(d);
        fprintf(stderr, "[sa3-server] worker failed to start\n");
        return false;
    }
    g_wk_pid  = pid;
    g_wk_data = d;
    g_wk_gpu  = want_gpu;
    fprintf(stderr, "[sa3-server] worker pid=%d spawned (isolation) gpu='%s'\n", (int) pid, want_gpu.c_str());
    return true;
}

static void worker_unload() {
    if (!g_isolation) return;
    std::lock_guard<std::mutex> lk(g_wk_mtx);
    kill_worker_locked();
}

// Marshal one job to the worker and block for its RESULT rc. The blocking recv runs
// WITHOUT g_wk_mtx so cancel/unload/watchdog can SIGKILL the child mid-render (the recv
// then fails -> rc<0 -> job finalized cancelled/failed).
static int dispatch_remote(const std::vector<std::string> & args) {
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
    bool got = recv_frame(data_fd, &h, &resp) && (WFrame) h.type == WFrame::RESULT && resp.size() >= sizeof(int);
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

// Idle watchdog: SIGKILL the worker once it has been idle for g_idle_unload_seconds,
// dropping the resident weights' VRAM to true-0 between grants.
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

// Dispatch a job either to the warm worker (isolation) or a fresh subprocess.
static int dispatch_job(std::shared_ptr<Job> job, const std::vector<std::string> & args) {
    if (g_isolation) {
        return dispatch_remote(args);
    }
    return run_subprocess(args, &job->pid);
}

// Finalize a long compute request: async by default (register a job, return {job_id}
// 202); ?wait=1 (or ?sync=1) blocks and streams the wav for simple clients.
static void respond_compute(const httplib::Request & req, httplib::Response & res, const std::string & kind,
                            std::vector<std::string> args, const std::string & out_wav,
                            std::vector<std::string> cleanup = {}) {
    bool want_sync = req.has_param("wait") || req.has_param("sync");
    if (!want_sync) {
        std::string      id   = job_submit(kind, std::move(args), out_wav, std::move(cleanup));
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
        fprintf(stderr, "[sa3-server] /%s queued as %s\n", kind.c_str(), id.c_str());
        return;
    }
    int rc;
    {
        std::lock_guard<std::mutex> lock(g_compute_mtx);
        rc = g_isolation ? dispatch_remote(args) : run_subprocess(args);
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
    fprintf(stderr, "[sa3-server] /%s done sync (%zu bytes)\n", kind.c_str(), wav.size());
}

// Common sampler/steps/seed/model tail appended to every sa3-gen argv.
static void push_common(std::vector<std::string> & args, const std::string & sampler, int steps, long long seed) {
    args.push_back("--steps");   args.push_back(std::to_string(steps));
    args.push_back("--seed");    args.push_back(std::to_string(seed));
    args.push_back("--sampler"); args.push_back(sampler);
}

// Base argv: sa3-gen + the three model ggufs for the chosen quant + --out.
static std::vector<std::string> base_args(const std::string & model, const std::string & out_wav) {
    return { g_cfg.bindir + "/sa3-gen",
             "--t5",   gguf_path("t5gemma", model),
             "--dit",  gguf_path("dit", model),
             "--same", gguf_path("same", model),
             "--out",  out_wav };
}

// ───────────────────────────────────────────────────────────────────────────
// POST /generate  (text -> audio, t2a)  — JSON body:
//   prompt   string  (required)  text description, e.g. "8-bit arcade power-up jingle"
//   seconds  number  (optional, default 10)
//   seed     number  (optional, default 0)
//   sampler  string  (optional, "pingpong"|"euler", default "pingpong" — the rf_denoiser default)
//   steps    number  (optional, default 8 — the SA3 reference default; leave it)
//   model    string  (optional, "q8"|"q6", default "q8")
// returns: {job_id} (async) or audio/wav (?wait=1).
// ───────────────────────────────────────────────────────────────────────────
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
    // Per-request GPU target (gate placement); next ensure_worker relocates.
    { std::string g = json_get_str(root, "gpu", "");
      if (!g.empty()) { std::lock_guard<std::mutex> lk(g_wk_mtx); g_next_gpu = g; } }
    std::string prompt = json_get_str(root, "prompt", "");
    if (prompt.empty()) {
        yyjson_doc_free(doc);
        json_error(res, 400, "'prompt' is required");
        return;
    }
    double      seconds = json_get_num(root, "seconds", 10.0);
    long long   seed    = (long long) json_get_num(root, "seed", 0.0);
    int         steps   = (int) json_get_num(root, "steps", 8.0);
    std::string sampler = json_get_str(root, "sampler", "pingpong");
    std::string model   = json_get_str(root, "model", "q8");
    yyjson_doc_free(doc);

    if (model != "q8" && model != "q6") {
        json_error(res, 400, "model must be one of: q8, q6");
        return;
    }
    if (sampler != "pingpong" && sampler != "euler") {
        json_error(res, 400, "sampler must be one of: pingpong, euler");
        return;
    }
    if (seconds <= 0.0 || seconds > (double) g_cfg.max_duration) {
        json_error(res, 400, "seconds out of range");
        return;
    }
    if (steps < 1 || steps > 64) {
        json_error(res, 400, "steps out of range");
        return;
    }

    std::string              out_wav = unique_path(".wav");
    std::vector<std::string> args    = base_args(model, out_wav);
    args.push_back("--prompt");  args.push_back(prompt);
    args.push_back("--seconds"); args.push_back(std::to_string(seconds));
    push_common(args, sampler, steps, seed);

    fprintf(stderr, "[sa3-server] /generate model=%s dur=%.1fs steps=%d seed=%lld sampler=%s\n", model.c_str(),
            seconds, steps, seed, sampler.c_str());
    respond_compute(req, res, "generate", std::move(args), out_wav);
}

// ───────────────────────────────────────────────────────────────────────────
// multipart helpers (audio upload + text fields) for /edit.
// ───────────────────────────────────────────────────────────────────────────
static std::string save_upload(const httplib::Request & req, const char * field) {
    if (!req.form.has_file(field)) {
        return "";
    }
    const auto   f    = req.form.get_file(field);
    std::string  path = unique_path(".wav");
    FILE *       fp   = fopen(path.c_str(), "wb");
    if (!fp) {
        return "";
    }
    fwrite(f.content.data(), 1, f.content.size(), fp);
    fclose(fp);
    return path;
}
static std::string form_str(const httplib::Request & req, const char * key, const char * dflt) {
    if (req.form.has_field(key)) {
        return req.form.get_field(key);
    }
    if (req.has_param(key)) {
        return req.get_param_value(key);
    }
    return dflt;
}
static bool form_has(const httplib::Request & req, const char * key) {
    return req.form.has_field(key) || req.has_param(key);
}

// ───────────────────────────────────────────────────────────────────────────
// POST /edit  (audio + prompt -> audio)  — multipart/form-data:
//   input    file    (required)  44.1 kHz stereo wav to edit
//   mode     field   (required)  "a2a" | "inpaint" | "continuation"
//   prompt   field   (required)  text description for the (re)generated audio
//   model    field   (optional, "q8"|"q6", default "q8")
//   seed     field   (optional, default 0)
//   sampler  field   (optional, "pingpong"|"euler", default "pingpong")
//   steps    field   (optional, default 8)
//   strength field   (a2a, default 0.7)   SDEdit start sigma in [0,1]
//   mask     field   (inpaint, required)  "s0:e0,s1:e1" seconds to regenerate
//   seconds  field   (continuation, required)  total output length (> input length)
// returns: {job_id} (async) or audio/wav (?wait=1).
// ───────────────────────────────────────────────────────────────────────────
static void handle_edit(const httplib::Request & req, httplib::Response & res) {
    std::string mode   = form_str(req, "mode", "");
    std::string prompt = form_str(req, "prompt", "");
    if (mode != "a2a" && mode != "inpaint" && mode != "continuation") {
        json_error(res, 400, "mode must be one of: a2a, inpaint, continuation");
        return;
    }
    if (prompt.empty()) {
        json_error(res, 400, "'prompt' is required");
        return;
    }
    std::string input = save_upload(req, "input");
    if (input.empty()) {
        json_error(res, 400, "multipart 'input' audio file is required");
        return;
    }
    std::string model   = form_str(req, "model", "q8");
    std::string sampler = form_str(req, "sampler", "pingpong");
    long long   seed    = atoll(form_str(req, "seed", "0").c_str());
    int         steps   = atoi(form_str(req, "steps", "8").c_str());
    if (model != "q8" && model != "q6") model = "q8";
    if (sampler != "pingpong" && sampler != "euler") sampler = "pingpong";
    if (steps < 1 || steps > 64) steps = 8;

    std::string              out_wav = unique_path(".wav");
    std::vector<std::string> args    = base_args(model, out_wav);
    args.push_back("--prompt"); args.push_back(prompt);
    args.push_back("--input");  args.push_back(input);

    if (mode == "a2a") {
        args.push_back("--strength");
        args.push_back(form_str(req, "strength", "0.7"));
    } else if (mode == "inpaint") {
        std::string mask = form_str(req, "mask", "");
        if (mask.empty()) {
            unlink(input.c_str());
            json_error(res, 400, "inpaint mode requires 'mask' (\"s0:e0,s1:e1\" seconds)");
            return;
        }
        args.push_back("--mask");
        args.push_back(mask);
    } else {  // continuation
        if (!form_has(req, "seconds")) {
            unlink(input.c_str());
            json_error(res, 400, "continuation mode requires 'seconds' (total output length)");
            return;
        }
        double seconds = atof(form_str(req, "seconds", "0").c_str());
        if (seconds <= 0.0 || seconds > (double) g_cfg.max_duration) {
            unlink(input.c_str());
            json_error(res, 400, "seconds out of range");
            return;
        }
        args.push_back("--continue");
        args.push_back("--seconds");
        args.push_back(std::to_string(seconds));
    }
    push_common(args, sampler, steps, seed);

    fprintf(stderr, "[sa3-server] /edit mode=%s model=%s steps=%d seed=%lld sampler=%s\n", mode.c_str(),
            model.c_str(), steps, seed, sampler.c_str());
    respond_compute(req, res, "edit", std::move(args), out_wav, { input });
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
            std::lock_guard<std::mutex> lk(g_jobs_mtx);
            wav.swap(job->result);  // once-only (the temp file is already unlinked)
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

// POST /unload: cancel/kill any in-flight render and tear down the warm worker so VRAM
// drops to true-0 (the GPU-gate preempt hook; SA3 holds resident weights while a worker
// is alive, so this is what reclaims them between grants).
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
    yyjson_mut_obj_add_str(doc, root, "vram", "true-0 (warm worker torn down)");
    char * out = yyjson_mut_write(doc, 0, nullptr);
    res.set_content(out ? out : "{}", "application/json");
    free(out);
    yyjson_mut_doc_free(doc);
    fprintf(stderr, "[sa3-server] /unload: cancelled %d in-flight job(s)\n", killed);
}

// ───────────────────────────────────────────────────────────────────────────
// main
// ───────────────────────────────────────────────────────────────────────────
static void usage(const char * prog) {
    fprintf(stderr,
            "Usage: %s [options]\n"
            "  --host <addr>      listen address (default %s)\n"
            "  --port <N>         listen port (default %d)\n"
            "  --gguf <dir>       gguf directory holding sa3-{t5gemma,dit,same}-{Q8_0,Q6_K}.gguf (default %s)\n"
            "  --bindir <dir>     dir holding the sa3-gen binary (default: this exe's dir)\n"
            "  --tmpdir <dir>     scratch dir for output/upload wavs (default %s)\n"
            "  --max-duration <s> reject above this many seconds (default %.0f)\n"
            "  --worker-isolation warm resident-model worker (or env SA3_WORKER_ISOLATION=1)\n",
            prog, g_cfg.host.c_str(), g_cfg.port, g_cfg.gguf.c_str(), g_cfg.tmpdir.c_str(), g_cfg.max_duration);
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
    if (const char * e = getenv("SA3_WORKER_ISOLATION")) g_isolation = atoi(e) != 0;
    if (const char * e = getenv("SA3_IDLE_UNLOAD_SEC")) g_idle_unload_seconds = atoi(e);
    if (const char * e = getenv("WORKER_DEFAULT_GPU")) { g_default_gpu = e; fprintf(stderr, "[sa3-server] default GPU = %s\n", e); }

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
            fprintf(stderr, "[sa3-server] unknown arg: %s\n", a.c_str());
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
    svr.Post("/edit", handle_edit);
    svr.Get("/job", handle_job);
    svr.Post("/job", handle_job);
    svr.Post("/unload", handle_unload);

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGCHLD, SIG_DFL);  // ensure waitpid sees child exits

    fprintf(stderr, "[sa3-server] sa3.cpp %s\n", ACE_VERSION);
    fprintf(stderr, "[sa3-server] bindir=%s gguf=%s\n", g_cfg.bindir.c_str(), g_cfg.gguf.c_str());
    fprintf(stderr, "[sa3-server] listening on %s:%d\n", g_cfg.host.c_str(), g_cfg.port);
    if (g_isolation) {
        fprintf(stderr,
                "[sa3-server] worker isolation ON (warm RESIDENT models ~2.67 GB Q8; idle-unload %ds; /unload "
                "-> true-0)\n",
                g_idle_unload_seconds);
        g_wk_last_activity.store(now_ms());
        std::thread(worker_watchdog_main).detach();
    } else {
        fprintf(stderr, "[sa3-server] worker isolation OFF (subprocess per request; true-0 idle by teardown)\n");
    }
    fprintf(stderr, "[sa3-server] endpoints: GET /health, POST /generate (t2a JSON), POST /edit "
                    "(a2a/inpaint/continuation multipart), GET /job?id=, POST /unload\n");

    if (!svr.listen(g_cfg.host.c_str(), g_cfg.port)) {
        fprintf(stderr, "[sa3-server] FATAL: cannot bind %s:%d\n", g_cfg.host.c_str(), g_cfg.port);
        return 1;
    }
    fprintf(stderr, "[sa3-server] shutting down\n");
    return 0;
}
