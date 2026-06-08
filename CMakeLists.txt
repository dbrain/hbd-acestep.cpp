cmake_minimum_required(VERSION 3.14)
project(acestep-ggml LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# version.h: embed git commit hash into all binaries.
# runs on every build, only rewrites if the hash changed.
set(VERSION_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/version.h")
add_custom_target(version ALL
    COMMAND "${CMAKE_COMMAND}" "-DSRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}" "-DOUTPUT=${VERSION_OUTPUT}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tools/version.cmake"
    BYPRODUCTS "${VERSION_OUTPUT}"
    COMMENT "Checking git version"
)

# pthread: required explicitly on older glibc (< 2.34) where libpthread
# is not merged into libc. Modern distros link it implicitly but aarch64
# and older x86_64 toolchains need the explicit dependency.
find_package(Threads REQUIRED)

# Suppress MSVC fopen/sprintf deprecation warnings
if(MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

# Put executables and backend .so in the same directory (build root).
# Without this, ggml defaults to bin/ for .so but executables stay in root,
# and ggml_backend_load_all() can't find the backends at runtime.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# DiT tensor names can exceed default GGML_MAX_NAME of 64
add_compile_definitions(GGML_MAX_NAME=128)

# Harden: mark fread/fwrite/etc with warn_unused_result on all platforms
# SYCL excluded: _FORTIFY_SOURCE swaps memcpy for __memcpy_chk, unresolved in device code
if(NOT MSVC AND NOT GGML_SYCL)
    add_compile_definitions(_FORTIFY_SOURCE=2)
endif()

# CUDA architectures: cover Turing to Blackwell for distributed binaries.
# Users can override with -DCMAKE_CUDA_ARCHITECTURES=native for local builds.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
        set(CMAKE_CUDA_ARCHITECTURES "75-virtual;80-virtual;86-real;89-real;120a-real;121a-real")
    else()
        set(CMAKE_CUDA_ARCHITECTURES "75-virtual;80-virtual;86-real;89-real")
    endif()
endif()

# ggml as subdirectory, inherits GGML_CUDA, GGML_METAL, etc. from cmake flags
add_subdirectory(ggml)

# cpp-httplib (HTTP server library, used by ace-server)
add_subdirectory(vendor/cpp-httplib)

# Shared compile options and ggml linkage
macro(link_ggml_backends target)
    target_include_directories(${target} PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}
    )
    target_include_directories(${target} SYSTEM PRIVATE
        ${CMAKE_SOURCE_DIR}/ggml/include
    )
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /wd4100 /wd4505)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wshadow -Wconversion
                              -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion)
    endif()
    target_link_libraries(${target} PRIVATE ggml Threads::Threads)
    if(TARGET ggml-base)
        target_link_libraries(${target} PRIVATE ggml-base)
    endif()
    foreach(backend cpu blas cuda metal vulkan sycl)
        if(TARGET ggml-${backend})
            get_target_property(CURRENT_BACKEND_TYPE ggml-${backend} TYPE)
            if (CURRENT_BACKEND_TYPE STREQUAL "MODULE_LIBRARY")
                # DL mode: backend is loaded at runtime via dlopen,
                # skip all link-time deps.
                continue()
            endif()
            target_link_libraries(${target} PRIVATE ggml-${backend})
        endif()
    endforeach()

    # SYCL links its runtime PRIVATE, consumers need -fsycl to resolve libsycl.so
    if(TARGET ggml-sycl AND GGML_SYCL)
        target_link_options(${target} PRIVATE -fsycl)
    endif()
    add_dependencies(${target} version)
endmacro()

# yyjson (MIT, fast JSON parser/writer)
add_library(yyjson STATIC vendor/yyjson/yyjson.c)
target_include_directories(yyjson PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vendor/yyjson)
if(MSVC)
    target_compile_options(yyjson PRIVATE /W0)
else()
    target_compile_options(yyjson PRIVATE -w)
endif()

# Core library (shared between binaries)
add_library(acestep-core STATIC
    src/request.cpp
    src/model-store.cpp
    src/pipeline-lm.cpp
    src/pipeline-synth.cpp
    src/pipeline-synth-ops.cpp
    src/pipeline-understand.cpp
)
target_link_libraries(acestep-core PUBLIC yyjson)
link_ggml_backends(acestep-core)

# ace-synth: full pipeline (text-enc + cond + dit + vae + wav)
add_executable(ace-synth tools/ace-synth.cpp)
target_link_libraries(ace-synth PRIVATE acestep-core)
link_ggml_backends(ace-synth)

# ace-lm: LLM inference (CoT + audio codes)
add_executable(ace-lm tools/ace-lm.cpp)
target_link_libraries(ace-lm PRIVATE acestep-core)
link_ggml_backends(ace-lm)

# songgen-lelm-test: SongGeneration LeLM bring-up harness (npy I/O, no core deps)
add_executable(songgen-lelm-test tools/songgen-lelm-test.cpp)
link_ggml_backends(songgen-lelm-test)

# songgen-lelm-gen-test: GATE-1 KV-cache decode vs full-prefill oracle (npy I/O, no core deps)
add_executable(songgen-lelm-gen-test tools/songgen-lelm-gen-test.cpp)
link_ggml_backends(songgen-lelm-gen-test)

# songgen-vae-test: SongGeneration Stable-Audio Oobleck VAE decoder harness (npy I/O, no core deps)
add_executable(songgen-vae-test tools/songgen-vae-test.cpp)
link_ggml_backends(songgen-vae-test)

# songgen-cfm-test: SongGeneration CFM reflow estimator (custom GPT2) harness (npy I/O, no core deps)
add_executable(songgen-cfm-test tools/songgen-cfm-test.cpp)
link_ggml_backends(songgen-cfm-test)

# songgen-vae-enc-test: Oobleck VAE ENCODER harness (npy I/O, no core deps)
add_executable(songgen-vae-enc-test tools/songgen-vae-enc-test.cpp)
link_ggml_backends(songgen-vae-enc-test)

# songgen-rvq-enc-test: descript RVQ ENCODE harness (npy I/O, no core deps)
add_executable(songgen-rvq-enc-test tools/songgen-rvq-enc-test.cpp)
link_ggml_backends(songgen-rvq-enc-test)

# songgen-musicfm-test: MusicFM_95M BESTRQ encoder harness (npy I/O, no core deps)
add_executable(songgen-musicfm-test tools/songgen-musicfm-test.cpp)
link_ggml_backends(songgen-musicfm-test)

# songgen-htdemucs-test: HTDemucs source separator block-by-block golden harness (npy I/O)
add_executable(songgen-htdemucs-test tools/songgen-htdemucs-test.cpp)
link_ggml_backends(songgen-htdemucs-test)

# songgen-separate: HTDemucs mix.wav -> vocal_48k.wav + bgm_48k.wav
add_executable(songgen-separate tools/songgen-separate.cpp)
link_ggml_backends(songgen-separate)

# songgen-htdemucs-dump: dump every forward intermediate as raw f32 (CPU-vs-CUDA bisect)
add_executable(songgen-htdemucs-dump tools/songgen-htdemucs-dump.cpp)
link_ggml_backends(songgen-htdemucs-dump)

# songgen-decode: SongGeneration audio decode driver (codes -> CFM ODE -> VAE -> wav)
# header-only: songgen-cfm.h + songgen-vae.h + songgen-septoken.h + npy.h + wav.h
add_executable(songgen-decode tools/songgen-decode.cpp)
link_ggml_backends(songgen-decode)

# songgen-decode-chunk-test: validate chunked CFM+VAE decode continuity (seam stats)
add_executable(songgen-decode-chunk-test tools/songgen-decode-chunk-test.cpp)
link_ggml_backends(songgen-decode-chunk-test)

# songgen-tok-test: GATE for in-C++ Qwen2 conditioning tokenizer vs golden npy
add_executable(songgen-tok-test tools/songgen-tok-test.cpp)
link_ggml_backends(songgen-tok-test)

# songgen-generate: GATE-2 full pipeline (LeLM KV-cache generation -> CFM/VAE decode -> wav)
add_executable(songgen-generate tools/songgen-generate.cpp)
link_ggml_backends(songgen-generate)

# songgen-clone-encode-test: audio-prompt clone encode (stems -> 3 token streams) vs goldens
add_executable(songgen-clone-encode-test tools/songgen-clone-encode-test.cpp)
link_ggml_backends(songgen-clone-encode-test)

# songgen-clone: full clone pipeline (stems -> prompt tokens -> LeLM -> CFM/VAE decode -> wav)
add_executable(songgen-clone tools/songgen-clone.cpp)
link_ggml_backends(songgen-clone)

# songgen-continue: audio-prompt CONTINUATION (prompt seeds CFM in-context + LM, prompt trimmed out)
add_executable(songgen-continue tools/songgen-continue.cpp)
link_ggml_backends(songgen-continue)

# ---- Stable Audio 3 (sa3-*) port: header-only golden harnesses (npy I/O, no core deps) ----
add_executable(sa3-t5gemma-test tools/sa3-t5gemma-test.cpp)
link_ggml_backends(sa3-t5gemma-test)
add_executable(sa3-dit-test tools/sa3-dit-test.cpp)
link_ggml_backends(sa3-dit-test)
add_executable(sa3-same-test tools/sa3-same-test.cpp)
link_ggml_backends(sa3-same-test)
add_executable(sa3-same-enc-test tools/sa3-same-enc-test.cpp)
link_ggml_backends(sa3-same-enc-test)
add_executable(sa3-pipeline-test tools/sa3-pipeline-test.cpp)
link_ggml_backends(sa3-pipeline-test)
add_executable(sa3-tok-test tools/sa3-tok-test.cpp)
link_ggml_backends(sa3-tok-test)
add_executable(sa3-gen tools/sa3-gen.cpp)
link_ggml_backends(sa3-gen)
add_executable(sa3-imatrix-collect tools/sa3-imatrix-collect.cpp)
link_ggml_backends(sa3-imatrix-collect)

# webui: convert tools/webui/public/index.html.gz to a C header for embedding.
# the .gz is committed to git so the C++ build works without npm.
# to update: cd tools/webui && npm install && npm run build, then rebuild ace-server.
set(WEBUI_INPUT "${CMAKE_CURRENT_SOURCE_DIR}/tools/public/index.html.gz")
set(WEBUI_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/index.html.gz.hpp")
add_custom_command(
    OUTPUT "${WEBUI_OUTPUT}"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${WEBUI_INPUT}" "-DOUTPUT=${WEBUI_OUTPUT}" -P "${CMAKE_CURRENT_SOURCE_DIR}/tools/xxd.cmake"
    DEPENDS "${WEBUI_INPUT}"
    COMMENT "Embedding webui into index.html.gz.hpp"
)
set_source_files_properties(${WEBUI_OUTPUT} PROPERTIES GENERATED TRUE)

# ace-server: HTTP server (LM + synth endpoints + embedded webui)
add_executable(ace-server tools/ace-server.cpp ${WEBUI_OUTPUT})
target_include_directories(ace-server PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(ace-server PRIVATE acestep-core httplib)
link_ggml_backends(ace-server)

# songgen-server: HTTP API for the SongGeneration port.
# Subprocess-exec per request by default; with --worker-isolation it compiles the
# songgen-generate/clone/continue bodies in-process (-DSONGGEN_AS_LIB) for a warm
# fork+IPC worker, so it links ggml + the src/ headers like the other tools.
add_executable(songgen-server tools/songgen-server.cpp)
target_include_directories(songgen-server PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(songgen-server PRIVATE httplib yyjson)
link_ggml_backends(songgen-server)

add_executable(ace-understand tools/ace-understand.cpp)
target_link_libraries(ace-understand PRIVATE acestep-core)
link_ggml_backends(ace-understand)

# quantize: GGUF requantizer (BF16 -> K-quants)
add_executable(quantize tools/quantize.cpp)
link_ggml_backends(quantize)

# neural-codec: Oobleck VAE neural audio codec (encode/decode WAV <-> latent)
add_executable(neural-codec tools/neural-codec.cpp)
link_ggml_backends(neural-codec)

# mp3-codec: MP3 encoder/decoder (standalone, no ggml needed)
# The mp3/ headers are header-only and usable by ace-synth too via #include "mp3/mp3enc.h"
add_executable(mp3-codec tools/mp3-codec.cpp)
target_include_directories(mp3-codec PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)
add_dependencies(mp3-codec version)
if(MSVC)
    target_compile_options(mp3-codec PRIVATE /W4 /wd4100 /wd4505)
else()
    target_compile_options(mp3-codec PRIVATE -Wall -Wextra -Wconversion
                          -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion)
    target_link_libraries(mp3-codec PRIVATE m)
endif()
target_link_libraries(mp3-codec PRIVATE Threads::Threads)

add_subdirectory(tests)
