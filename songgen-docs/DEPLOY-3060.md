# SongGeneration ggml port — RTX 3060 (CUDA) deploy + validation handoff

Self-contained guide to build, validate, and run the pure-C++/ggml SongGeneration (LeVo2) port
on the RTX 3060 (Ampere, CUDA) — the intended target. Everything was developed + validated on a
CPU/ROCm box; this is the CUDA bring-up. The port lives **inside an acestep.cpp fork** and is fully
additive (the `ace-*` binaries still build/run unchanged — see "Coexistence").

## 0. What this is
A from-scratch ggml/C++ port of Tencent SongGeneration v2 (LeVo2): lyric+style → song, no Python in
the inference path. Pipeline: LeLM (hierarchical Llama LM) → gen_tokens → RVQ → CFM reflow ODE →
Stable-Audio Oobleck VAE → 48kHz stereo wav. Plus: audio-prompt clone, continuation, single-file-in
(native ggml HTDemucs separation), an HTTP server. All graphs golden-validated (cossim >0.99).

## 1. Build (CUDA)
```
# from the repo root (this is the hbd-acestep.cpp fork)
mkdir -p build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release   # 3060 = Ampere (sm_86), covered by default arch list
cmake --build . -j
```
- The CPU dev box needed `-DBLAS_LIBRARIES="/usr/lib/libcblas.so;/usr/lib/libblas.so"` (a local libblas
  missing cblas_sgemm). On CUDA you likely don't need it; if a BLAS link error appears, add it or `-DGGML_BLAS=OFF`.
- Targets produced: `songgen-{generate,clone,continue,separate,server}` + per-graph tests
  `songgen-{lelm-test,lelm-gen-test,vae-test,vae-enc-test,cfm-test,rvq-enc-test,musicfm-test,htdemucs-test,tok-test}`.

## 2. Models / data (NOT in git — transfer or regenerate)
ggufs (~14GB) live OUTSIDE the repo at `<data>/gguf/`; audio ckpts at `<data>/ckpt-audio/`; goldens at
`<data>/golden-large/`. Either copy them from the dev box, or regenerate ggufs with the converters in
`songgen-docs/../scripts/` (convert_songgen_lelm.py, convert_vae.py, convert_cfm.py, convert_musicfm.py,
convert_septoken_aux.py, convert_1rvq_aux.py, convert_vae_encoder.py, convert_htdemucs.py — each has a
docstring with its source ckpt + run command; use a py3.11/clean torch venv, NOT this repo).
Required ggufs: songgen-lelm-large-Q8_0.gguf (or Q4_K_M), songgen-cfm.gguf, songgen-vae.gguf,
songgen-vae-encoder.gguf, songgen-septoken-aux.gguf, songgen-1rvq-aux.gguf, songgen-musicfm.gguf,
songgen-musicfm-1rvq.gguf, songgen-htdemucs.gguf.

## 3. VALIDATE on CUDA (the important part — re-confirm the gates)
All goldens were captured + matched on CPU. On CUDA the math should match, but TWO things need a real check
because they differ on GPU: **flash-attn** (auto-enabled when a GPU is present) and **RoPE mode**. Run the
gates (each loads a gguf + runs; fast on GPU):
```
./build/songgen-lelm-test     gguf/songgen-lelm-large-Q8_0.gguf golden-large           # cb0~1.0 cb1/cb2>0.998
./build/songgen-cfm-test      gguf/songgen-cfm.gguf  golden-large/audio                # >0.999
./build/songgen-vae-test      gguf/songgen-vae.gguf  golden-large/audio                # 0.9998
./build/songgen-vae-enc-test  gguf/songgen-vae-encoder.gguf golden-large/clone         # 0.9995
./build/songgen-rvq-enc-test  gguf/songgen-septoken-aux.gguf golden-large/clone        # codes EXACT
./build/songgen-musicfm-test  gguf/songgen-musicfm.gguf golden-large/clone             # layers >0.9998
./build/songgen-htdemucs-test gguf/songgen-htdemucs.gguf <htd-ref> fwd                 # blocks >0.999
./build/songgen-lelm-gen-test gguf/songgen-lelm-large-Q8_0.gguf golden-large           # KV vs prefill >0.999
```
**RoPE GOTCHA (must hold on CUDA):** LeLM uses NEOX rope (mode 2, θ=500000); the CFM estimator uses
INTERLEAVED/GPT-J rope (mode 0, θ=10000). If cb-cossim drops on GPU, flash-attn fp16 accumulation is the
usual suspect — there's a `clamp_fp16` path in qwen3-lm.h for sub-Ampere; 3060 is Ampere so it should be fine,
but if logits diverge, force the F32 attention fallback (don't use_flash_attn) and re-check.
Note: a Vulkan-iGPU experiment on the dev box DIVERGED numerically (GPU fp-accumulation flips argmax) AND was
no faster — so per-token bit-identity across backends is NOT guaranteed; validate by cossim, not bit-equality.

## 4. RUN
```
# text -> song  (MIND THE LYRIC FORMAT — see §5; default --duration is short, the model self-terminates on EOS)
GGML_BACKEND=CUDA0 ./build/songgen-generate <lelm> <cfm> <vae> <septoken-aux> <golden-dir> out.wav <seed> \
   --lyric "<formatted lyric>" --description "<comma,separated,tags>" [--duration 120] [--gen-type mixed|vocal|bgm] \
   [--temp 1.0 --top-k 250 --cfg 1.5 --fade 200] [--model q8|q4 via the lelm gguf path]
# single-file-in clone (pure C++):  scripts/songgen-clone-native.sh <mix.wav> <out.wav> "<lyric>" [seed]
# continuation:  ./build/songgen-continue <8 ggufs incl vae-encoder> out.wav <seed> --vocal-stem v --bgm-stem b --lyric "..."
# separation:    ./build/songgen-separate gguf/songgen-htdemucs.gguf mix.wav vocal48.wav bgm48.wav
# API:           ./build/songgen-server --port 8097   ; POST /generate {lyric,description,duration,seed,gen_type,...}
```

## 5. CRITICAL: lyric formatting (this is the #1 footgun — got it wrong initially)
The model REQUIRES the structured format or quality degrades AND it won't self-terminate (EOS won't fire):
- Sections separated by ` ; ` (semicolon).
- Instrumental tags **standalone, NO lyrics**: `[intro-*]`, `[outro-*]`, `[inst-*]`, `[silence]`.
- Lyrical tags `[verse]`/`[chorus]`/`[bridge]` REQUIRE lyrics; separate phrases with `. ` ; each lyrical block's
  final phrase ends with `.` before the ` ;`. English half-width punctuation only.
- Description = comma-separated TAGS, not sentences.
- DURATION: set a generous ceiling (the reference uses 270s) — the model emits EOS at the song's natural end and
  stops itself (verified: a 120s-ceiling gen self-terminated at 112s). Too-short = clean cut (autoregressive, NO
  word-cramming, unlike ACE-Step). `extend_stride` is vestigial (NotImplemented) — single-shot only.
Example: `[intro-short] ; [verse] line one. line two. ; [chorus] hook one. hook two. ; [bridge] ... . ; [outro-short]`

## 6. Perf
CPU dev box: 15s in ~244s (Q8) / ~173s (Q4). The 3060 (GDDR6 + tensor cores) should be far faster — this is the
whole point of the target. CFG batching (2 forwards/step → 1 N=2 forward) is on by default (SGLM_NO_BATCH=1 to A/B).
Q4_K_M is viable (codes valid, ~1.4x faster, 2.9G vs 5.1G) — confirm quality by ear on the 3060.

## 7. Coexistence with acestep
The port adds new files + 3 ADDITIVE edits to shared files: CMakeLists.txt (new songgen targets), src/backend.h
(an opt-in SGLM_THREADS env, default unchanged), src/wav.h (a new `write_wav_s16_planar` writer acestep doesn't
call). The `ace-{lm,synth,server,understand}` binaries build + behave identically. One repo, two model families.

## 8. Known caveats / open items
- Clone end-to-end has no numeric golden (fuzzy frame-agreement + ear). MusicFM/RVQ/VAE-enc each golden-validated.
- single-file clone has TWO paths: native ggml (songgen-clone-native.sh, pure C++, vocal 0.9955/bgm 0.9916) and a
  python demucs front-end (songgen-clone-file.sh, marginally higher SDR via shifts=1, needs a demucs venv).
- HTDemucs port uses deterministic shifts=0 (vs demucs default shifts=1 random averaging).
- songgen-server is a v0 skeleton: /generate works end-to-end; /clone /continue /separate are 501 stubs.
  Design forks (sync/async, upload transport, worker model, queue, auth, delivery) in out/API-DESIGN.md — undecided.
- Full architecture detail: songgen-docs/{SPEC-lelm.md, SPEC-audio.md, HANDOFF.md}.
