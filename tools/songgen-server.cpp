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
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
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
static int run_subprocess(const std::vector<std::string> & args) {
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
        execv(argv[0], argv.data());
        fprintf(stderr, "[songgen-server] execv %s failed: %s\n", argv[0], strerror(errno));
        _exit(127);
    }
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    return -1;
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
    if (duration <= 0.0 || duration > (double) g_cfg.max_duration) {
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
        "--duration", std::to_string(duration),
        "--gen-type", gen_type,
    };
    if (has_temp) { args.push_back("--temp");  args.push_back(std::to_string(temp)); }
    if (has_topk) { args.push_back("--top-k"); args.push_back(std::to_string((int) top_k)); }
    if (has_cfg)  { args.push_back("--cfg");   args.push_back(std::to_string(cfg)); }
    if (has_fade) { args.push_back("--fade");  args.push_back(std::to_string(fade)); }

    fprintf(stderr, "[songgen-server] /generate model=%s dur=%.1fs seed=%llu type=%s\n",
            model.c_str(), duration, (unsigned long long) seed, gen_type.c_str());

    int rc;
    {
        std::lock_guard<std::mutex> lock(g_compute_mtx);  // serialize: one gen at a time
        rc = run_subprocess(args);
    }

    if (rc != 0) {
        unlink(out_wav.c_str());
        char msg[128];
        snprintf(msg, sizeof(msg), "generation failed (songgen-generate exit %d)", rc);
        json_error(res, 500, msg);
        return;
    }

    std::string wav;
    if (!read_file(out_wav, wav)) {
        unlink(out_wav.c_str());
        json_error(res, 500, "generation produced no output");
        return;
    }
    unlink(out_wav.c_str());

    res.set_content(wav, "audio/wav");
    fprintf(stderr, "[songgen-server] /generate done (%zu bytes)\n", wav.size());
}

// ───────────────────────────────────────────────────────────────────────────
// Stubbed endpoints (501 + planned schema). The wrapped CLIs already exist;
// the open question is audio upload + async delivery (see out/API-DESIGN.md).
// ───────────────────────────────────────────────────────────────────────────
static void handle_clone(const httplib::Request &, httplib::Response & res) {
    // wraps songgen-clone: separate-then-clone from stems (or a single mix via
    // scripts/songgen-clone-native.sh). Needs vocal+bgm stems (or a full mix to
    // run native separation first) -> open decision: upload transport.
    stub_501(res, "/clone",
             "{ lyric: string (required), description?: string, "
             "vocal_stem: audio (required), bgm_stem: audio (required), "
             "full_mix?: audio, duration?: number, seed?: number, "
             "gen_type?: mixed|vocal|bgm, temp?, top_k?, cfg?, fade?, model?: q8|q4 } "
             "-> audio/wav  [transport for audio inputs is the open question]");
}

static void handle_continue(const httplib::Request &, httplib::Response & res) {
    // wraps songgen-continue: extends a prompt song in-context.
    stub_501(res, "/continue",
             "{ lyric: string (required), description?: string, "
             "vocal_stem: audio (required), bgm_stem: audio (required), "
             "full_mix?: audio, duration?: number, seed?: number, "
             "gen_type?: mixed|vocal|bgm, temp?, top_k?, cfg?, fade?, model?: q8|q4 } "
             "-> audio/wav  [transport for audio inputs is the open question]");
}

static void handle_separate(const httplib::Request &, httplib::Response & res) {
    // wraps songgen-separate (htdemucs): mix -> {vocal, bgm}.
    stub_501(res, "/separate",
             "{ mix: audio (required) } "
             "-> { vocal: audio/wav, bgm: audio/wav }  "
             "[transport for audio in/out is the open question — multipart or URLs]");
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

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--host" && i + 1 < argc) {
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

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGCHLD, SIG_DFL);  // ensure waitpid sees child exits

    fprintf(stderr, "[songgen-server] songgen.cpp %s\n", ACE_VERSION);
    fprintf(stderr, "[songgen-server] bindir=%s gguf=%s golden=%s\n", g_cfg.bindir.c_str(),
            g_cfg.gguf.c_str(), g_cfg.golden.c_str());
    fprintf(stderr, "[songgen-server] listening on %s:%d\n", g_cfg.host.c_str(), g_cfg.port);
    fprintf(stderr, "[songgen-server] endpoints: GET /health, POST /generate, "
                    "POST /clone|/continue|/separate (501 stubs)\n");

    if (!svr.listen(g_cfg.host.c_str(), g_cfg.port)) {
        fprintf(stderr, "[songgen-server] FATAL: cannot bind %s:%d\n", g_cfg.host.c_str(), g_cfg.port);
        return 1;
    }
    fprintf(stderr, "[songgen-server] shutting down\n");
    return 0;
}
