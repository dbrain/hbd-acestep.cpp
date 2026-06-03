# SongGeneration (LeVo 2) → ggml/C++ port — HANDOFF

Goal: run Tencent **SongGeneration v2 (LeVo 2)** locally without Python, as a ggml/C++
port modeled on **acestep.cpp**. End device: **RTX 3060 12 GB (Ampere/CUDA)**. This box
(AMD Radeon 890M iGPU, py3.14/ROCm) is dev/correctness only — all artifacts target CUDA.

Status: **FEATURE-COMPLETE (MVP) & parity-audited.** Full pure-C++/ggml pipeline works end-to-end
(text->song ear-tested x3; arbitrary --lyric/--description tokenizer bit-exact; audio-prompt clone;
all 6 neural graphs golden-validated; long-form chunked decode; gen_type variants; sampling controls;
lookahead-limiter clipping fix; CFG-batching perf ~1.4x; Q4_K_M ~1.4x faster + codes valid). Build green.
DOCUMENTED GAPS (not MVP-blocking): (1) HTDemucs separation absent -> clone needs pre-separated stems
(no single-file audio input); (2) clone uses LM-conditioning only, the validated VAE-encoder graph
(songgen-vae-enc.h) is built but deliberately UNWIRED (in-context true_latents = continuation-only);
(3) GPU/3060 CUDA validation pending (all goldens CPU; recheck flash-attn + RoPE NEOX/GPT-J);
(4) clone end-to-end has no numeric golden (fuzzy frame-agreement gate + ear-test). See §7+ / SPEC-*.md.
Speed (this box, CPU): 15s in ~244s Q8 / ~173s Q4. Build needs -DBLAS_LIBRARIES="/usr/lib/libcblas.so;/usr/lib/libblas.so".

---

## 1. What this model is (architecture)

Two stages, sequential, never co-resident (the shipped lowmem path already frees each):

**Stage A — LeLM** (the language model; `model.pt` is ONLY this, 2.83B medium / ~4.4B large):
- Hierarchical: **main transformer** (Llama: MHA, SwiGLU, RMSNorm, RoPE θ=500000) →
  codebook-0 logits + hidden; **bridge MLP** (Linear 2d→d · GELU · Linear d→d);
  **sub transformer** (12L, smaller) → `linears[0,1]` heads → codebook-1,2 logits.
- 3 codebooks (song/vocal/bgm), `code_size=16384` (+EOS=16385). **Delay pattern [0,250,250]**
  (MusicGen-style: cb0 leads, cb1/cb2 lag 250 steps).
- Per autoregressive step: **2× forward (CFG, cond+null, coef=1.5)** through BOTH stacks.
  cb0 = top_k(250) sampling; cb1/cb2 = **greedy argmax**. Dual KV cache. Sliding-window
  repetition penalty (window 50). prompt-audio tokens masked from cb0.
- **Conditioning = pure embedding gather, NO text transformer**: lyrics → Qwen2 BPE token
  ids → `description.output_proj` (151659×dim) gather → prepend; `type_info.output_proj`
  (151652×dim); structure tokens [verse]=151646…[silence]=151658 (conf/vocab.yaml);
  `prompt_audio.emb.{0,1,2}` (16386×dim) for the optional 10s reference.
- `out_norm` exists in weights but is VESTIGIAL (commented out in forward). `transformer2.lm_head` also vestigial.
- Large dims: dim 2048, ffn 11008, 16 heads, 36 main + 12 sub layers. Medium: 1536/8960/12h/28+12.

**Stage B — audio decode** (separate `ckpt/` files, ~4.9G, shared medium/large):
- `model_septoken/model_2.safetensors` (3.6G) = Flow1dVAESeparate: a **GPT2-based reflow CFM**
  (flow-matching) estimator + RVQ.
- `vae/autoencoder_music_1320k.ckpt` (0.64G) + `stable_audio_1920_vae.json` = **Stable Audio
  1D VAE** (Oobleck/Snake) — decodes latents → 48kHz stereo wav.
- `model_1rvq/model_2_fixed.safetensors` (0.63G) = single-RVQ tokenizer (prompt encode path).
- Pipeline: gen_tokens [1,3,T] → seperate_tokenizer.decode([vocal,bgm], chunked=True) → wav.

Cloning: 10s reference audio (`prompt_audio`) + structured lyrics simultaneously → style/timbre
similarity (not identity-locked). Don't combine prompt_audio + text descriptions.

---

## 2. Artifact inventory (all on nvme, /home/dbrain/dev/songgen-port/)

```
gguf/songgen-lelm-large-BF16.gguf    9.6G  base (re-quant source)
gguf/songgen-lelm-large-Q8_0.gguf    5.1G  PRIMARY deliverable (3060)
gguf/songgen-lelm-large-Q4_K_M.gguf  2.9G  trial (may break codes like ACE-Step; untested)
gguf/songgen-lelm-medium-Q8_0.gguf   2.9G  dev scaffolding
golden-large/  lelm_logits.npy [16,2,3,16385] + gen_tokens_lm.npy [1,3,375] + meta.json  (LeLM port oracle, VALIDATED)
golden-medium/ same shape, medium (architecture-identical → also validates port logic)
scripts/  sg_compat.py, convert_songgen_lelm.py, dump_golden.py, inspect_components.py, conf/vocab.yaml
qwen2-tokenizer/  vocab.json+merges.txt+tokenizer.json (Qwen2 BPE, embedded in gguf too)
v2large/model.pt  13G  source LeLM weights
```
GGUF arch = `songgen-lelm`. Metadata embedded: block_count, embedding_length, ffn, head_count,
rope θ, + songgen.{num_layers_sub, rope_theta_sub, code_depth, code_size, delays, frame_rate,
cfg_coef, prompt_len, structure_tokens} + Qwen2 BPE tokenizer. `quantize` tool reads it natively:
`./build/quantize <BF16.gguf> <out.gguf> Q8_0|Q4_K_M|Q5_K_M|Q6_K`.

Audio-stack ckpts (for Stage B port) are NOT yet converted to gguf — still PyTorch at
/tmp/sg-run/ckpt (ephemeral! re-download from tencent/SongGeneration ckpt/ to persist).

---

## 3. Reproduce the Python reference (oracle generator)

On a box where torchvision/torchaudio install cleanly (i.e. NOT this py3.14 one — use the
CUDA box, or a py3.11 venv), it's straightforward. On THIS box the env needs `sg_compat.py`
which stubs ~12 incompatibilities (see that file's docstring). Recipe:
- venv --system-site-packages (inherit ROCm torch 2.12); pip install omegaconf transformers
  diffusers einops safetensors flashy peft lightning gguf julius soundfile stable-audio-tools
  (NO torch/torchaudio/torchvision pins).
- `import sg_compat` BEFORE codeclm.*; call `sg_compat.patch_rope()` after.
- Run: `cd /tmp/songgen-src && GOLD=<dir> N_STEPS=16 DURATION=15 PYTHONPATH=codeclm/tokenizer:.:codeclm/tokenizer/Flow1dVAE:tools/gradio python dump_golden.py <ckpt_dir>`
- LM goldens save mid-pipeline (robust). **Audio goldens (wav/cfm) NOT yet captured** —
  blocked here by stable_audio_tools→k_diffusion→torchvision enum chain. Capture on the CUDA
  box (deps install normally) or resolve torchvision. dump_golden already hooks cfm_in/out + wav.

Source repo cloned at /tmp/songgen-src (github tencent-ailab/SongGeneration, shallow).

---

## 4. The port (remaining work) — modeled on acestep.cpp (/tmp/acestep-cpp, BUILT)

acestep.cpp reuse: engine (backend/gguf/quantize/sampling/philox/offload/server) ~60-70% as-is.
acestep.cpp built clean here: `cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -j`.
Templates: src/qwen3-lm.h (LM), src/dit-graph.h + dit-sampler.h (diffusion), src/vae.h (VAE),
src/gguf-weights.h (loading), convert.py (gguf conventions). Validation pattern already in the
repo: tests/debug-lm-logits.py, debug-dit-cossim.py.

### #7 LeLM graph — DONE ✅ (full architecture spec in SPEC-lelm.md)
Built on hbd-acestep.cpp checkout at /home/dbrain/dev/songgen-port/songgen.cpp.
- src/songgen-lelm.h : graph+load (main 36L + bridge MLP(erf-GELU) + sub 12L, NO QK-norm,
  NEOX RoPE theta=500000, embedding-gather conditioning, no text transformer).
- tools/songgen-lelm-test.cpp : golden-replay validator (full-prefill per AR step, no KV cache).
- src/npy.h : numpy loader. scripts/dump_cond_inputs.py : dumps exact Qwen2-BPE conditioning ids.
- Reference ids in golden-large/{desc,type}_ids_*.npy + desc_cover_*.npy.
RESULT (GGML_BACKEND=CPU ./build/songgen-lelm-test <Q8 gguf> golden-large):
  cb0 cossim 1.00000, cb1 min 0.99848, cb2 min 0.99814 — all 16 steps, cond/uncond/CFG > 0.99.
KEY FACTS (validated): special_token=16385, card=16385, prepend order
  [description(lyrics,600) ++ prompt_audio(252: EOT+251 zeros) ++ type_info(style,100) ++ codes],
  cond1==cond2 for text conds, prompt_audio differs (emb0 vs emb1+emb2). uncond: desc->just
  <|im_start|>, type_info->"[Musicality-very-low], .", audio same all-16385. delay seq: pos0 all
  special; pos p>=1 (t=p-1) cb0=code[0,t], cb1/cb2=code[.,t-250] else special. start_offset=1.
  6 vestigial tensors: layer2_emb.0, out_norm.{w,b}, transformer{,2}.model.embed_tokens, transformer2.lm_head.
TODO for #9: add dual KV cache (main+sub, 2 sets each for CFG) + batched-2 decode for real
  generation (validation used full-prefill); cb0 top_k(250)/cb1,cb2 argmax; rep penalty win 50;
  re-verify the KV-cache path reproduces gen_tokens_lm. Re-check RoPE/attn on CUDA (ROCm here).

### #8 CFM sampler + Stable Audio VAE — IN PROGRESS (full arch in SPEC-audio.md)
Audio goldens CAPTURED on this box (CPU, deterministic): golden-large/audio/{vae_*, cfm_*}.npy.
Audio ckpts persisted: ckpt-audio/. Reference code: songgen-src-code/. This box suffices (no CUDA
box needed — stub k_diffusion in sys.modules, torchvision not needed; scripts/golden_*.py).
- 8c VAE decoder DONE ✅ cossim 0.9998 (src/songgen-vae.h, gguf/songgen-vae.gguf, scripts/convert_vae.py).
  SnakeBeta alpha_logscale, weight_norm folded, conv_transpose via im2col(F32). col2im is oc-major.
- 8d CFM estimator DONE ✅ cossim 0.9995 (src/songgen-cfm.h, gguf/songgen-cfm.gguf, scripts/convert_cfm.py).
  RoPE2 = INTERLEAVED/GPT-J (mode 0, θ=10000, full head_dim 110) — NOT NEOX. adaLN-single per-block
  (scale_shift_table[H,6]+time6 -> shift/scale/gate for msa+mlp), LayerNorm+bias eps1e-5, gelu_new(tanh),
  wpe added, GPT2 Conv1D transposed at convert. n_embd 2200 BF16 (2200/4400 not /32 so no Q8).
See SPEC-audio.md for exact configs/golden shapes. Build: songgen-{lelm,vae,cfm}-test all PASS on CPU.

### #9 status: DECODE DRIVER DONE ✅ — first wav produced (out/first.wav, out/first_norm.wav)
gen_tokens_lm.npy[1,3,375] -> RVQ from_codes(vocal,bgm) -> mu -> CFM euler ODE(20 steps,CFG1.5)
-> normfeat denorm -> VAE decode -> 15s 48k stereo. Files: src/songgen-septoken.h, tools/songgen-decode.cpp,
scripts/convert_septoken_aux.py, gguf/songgen-septoken-aux.gguf, wav.h gained write_wav_s16_planar.
Stats healthy (latent std 1.23, wav std 0.21, no NaN, RMS varies => real structure). Awaiting ear test.
REMAINING in #9 = Part A only: the LeLM autoregressive generation loop (dual KV cache + sampling) to
produce gen_tokens from a fresh lyric+style prompt (currently uses the golden tokens). See §E in SPEC-audio.md.

### #9 COMPLETE ✅ — end-to-end generation works in pure C++/ggml
src/songgen-lelm-gen.h (dual KV cache main+sub × cond/uncond, NEOX RoPE, MHA), tools/songgen-generate.cpp
(CLI: lelm.gguf cfm.gguf vae.gguf septoken-aux.gguf golden-dir out.wav [seed]). GATE-1 (KV vs full-prefill
oracle) PASS cossim>0.999. GATE-2: fresh 15s clip generated (seed 1234, out/generated.wav, all codes valid,
no EOS) and ear-tested. tools/songgen-lelm-gen-test.cpp = gate-1 harness. sglm_forward oracle untouched.
Run: GGML_BACKEND=CPU ./build/songgen-generate <4 ggufs> golden-large out.wav [seed]  (CPU slow ~slow; 3060 fast).
BUILD GOTCHA: a clean `cmake ..`/buildcpu.sh relinks ggml-blas against /usr/lib/libblas.so which lacks
cblas_sgemm -> link fails. Fix: configure with -DBLAS_LIBRARIES="/usr/lib/libcblas.so;/usr/lib/libblas.so"
OR -DGGML_BLAS=OFF. (Not a CMakeLists change; lives in build cache.)
OPTIONAL POLISH / future: chunking+crossfade for >15s tracks (SPEC-audio §D); prompt-audio clone path;
cb0 top_k/temp tuning; Q4 LM trial; verify on RTX 3060 CUDA (flash-attn + RoPE NEOX/GPT-J); perf (KV decode
is CPU-bound here). Decode top end slightly hot (spectral centroid ~5.5k) — consistent across golden+fresh,
so it's a decode characteristic not a gen bug; revisit if ear-test wants it tamed.

### #13 CLONE PATH — DONE ✅ (encoders + wiring + CLI). Ear-test pending; MVP needs separated stems.
Wiring decisions (cited): audio_qt_embs cb0=pmt(1rvq layer6)/cb1=vocal(septoken layer7)/cb2=bgm(layer3);
clone uses lyrics+prompt_audio (NOT description); decode does NOT use prompt true_latents in-context (style
cloning via LM conditioning only — confirmed levo_inference calls generate_audio without prompt wavs for the
sep path). VAE-encoder NOT wired into clone (left for future continuation). Files: src/songgen-clone-encode.h,
tools/songgen-clone.cpp + clone-encode-test.cpp, scripts/convert_1rvq_aux.py, gguf/songgen-musicfm-1rvq.gguf +
songgen-1rvq-aux.gguf; conditioning added ADDITIVELY to songgen-lelm.h/-gen.h (audio_ids_{pmt,vocal,bgm};
empty => text->song bit-identical, REGRESSION CONFIRMED songgen-lelm-test still 0.999+). CLI songgen-clone
--vocal-stem/--bgm-stem [--full-mix] --lyric. Encode leading-frame match vs token goldens pmt83/voc88/bgm86%
(divergence = resampler Kaiser vs julius tie-flips; rvq-encode bit-exact, MusicFM>0.9998). clone.wav sanity:
15s finite but HOT std0.45 + 0.38% clip (vs no-prompt 0.21) — likely the unseparated-double-stem input +
prompt steering; ear-test pending. GAP for single-file audio input: HTDemucs separation (weights public/not
shipped, sizable port) — MVP requires user-supplied vocal+bgm stems.

### CLIPPING FIX — DONE ✅ (decode is inherently hot, not a bug)
Diagnosed via reference decode of our tokens (scripts/diag_ref_decode.py): reference also peaks ~1.44
(std 0.268 vs our 0.286 — distributionally identical; cossim~0 is just unseeded CFM noise). Oobleck VAE
is unbounded (final_tanh=false); normfeat rescale ~0.948 (not the cause). Reference does NO level mgmt —
saves raw float to FLAC which also hard-clips (rarely). FIX: downward-only float peak-limit to 0.985 in
write_wav_s16_planar (wav.h, peak_target param default 0.985, <=0 disables) — universal write choke point,
never boosts quiet content, logs "[WAV] peak-limit:" when it engages. Verified: first_clean.wav peak 0.985,
0.0000% clip. TRADEOFF: a single loud transient sets the ceiling -> can be quiet (folk clip std 0.21->0.136).
A lookahead/soft limiter would preserve loudness; offered to user. #14 fade composes before this.

### (historical) #13 CLONE PATH — encoders DONE ✅, wiring remaining
All clone-encoder graphs ported + golden-validated (goldens golden-large/clone/, specs SPEC-audio §F/§G):
- VAE encoder (src/songgen-vae-enc.h, gguf/songgen-vae-encoder.gguf) cossim 0.9995 vs vaeenc_out_mean.
- RVQ encode (sgsep_rvq_encode in src/songgen-septoken.h, in_proj added to gguf/songgen-septoken-aux.gguf)
  codes BIT-EXACT (vocal+bgm) vs rvq_*_codes.
- MusicFM_95M bestrq (src/songgen-musicfm.h, gguf/songgen-musicfm.gguf 605MB, scripts in songgen.cpp/scripts/
  convert_musicfm.py) all 13 layers >0.9998 vs bestrq_out_*. Host mel(STFT n_fft2048 hop240 + shipped fb/window)
  + Conv2dSubsampling + 12L Wav2Vec2-Conformer (rotary applied to LN'd hidden PRE q/k proj; pos_conv UNUSED in
  rotary path; depthwise k31 via im2col F32). songgen-musicfm-test PASS.
GOTCHAS: (1) convert_musicfm wrote to a stray songgen.cpp/gguf/ — moved to canonical gguf/; convert script is in
songgen.cpp/scripts/ not songgen-port/scripts/. (2) model_1rvq bestrq differs from septoken bestrq in 36/463
tensors + its own rvq codebook -> the pmt stream needs its OWN converted gguf (reuse convert_musicfm.py on
model_1rvq/model_2_fixed.safetensors).
REMAINING #13c WIRING (no end-to-end golden; gate the encode pipeline vs tokens_{vocal,bgm,pmt} goldens, then ear-test):
  encode: stems(48k) -> mono/resample24k -> MusicFM(layer7 vocal / layer3 bgm) -> RVQ encode -> vocal,bgm codes;
    pmt: full mix -> 1rvq-MusicFM -> 1rvq-RVQ -> pmt codes. => audio_qt_embs[1,3,T].
  Validate vs golden-large/clone/tokens_{vocal,bgm,pmt}.npy (feed first.wav as both stems like the capture did).
  Then: feed REAL codes into LeLM prompt_audio conditioner (currently all-16385 zeros; the embed-gather + EOT
    prepend already built in songgen-lelm.h/-gen.h) AND prompt VAE-encode -> true_latents -> CFM in-context
    (latent_masks=2 over prompt frames; the no-op in songgen-septoken.h euler becomes active). CLI: --vocal-stem
    --bgm-stem (+optional --full for pmt; MVP skips demucs). Clone mode pairs prompt_audio + LYRICS (drop type_info
    description per reference). NOTE prepare_condition_tensors prompt_audio branch in lm_levo.py:222-230 for the
    exact audio_qt_seq (EOS prepend + 16385 masking).
STATUS SNAPSHOT: text->song fully works + ear-tested (folk seed1234, synthwave seed42); arbitrary --lyric/--description
works; all 6 neural graphs (LeLM, CFM, VAE dec/enc, MusicFM, RVQ enc/dec) validated. Remaining: #13c wire, #14
long-form/endings/variants, #15 perf (CFG-batch + graph-reuse ~2x), #16 final parity audit.

### #13c CLONE WIRING — DONE ✅ (integration; fuzzy validation)
END-TO-END clone path works in pure C++/ggml. Files:
- src/songgen-clone-encode.h: stems(48k stereo planar) -> mono-mean + resample24k (audio-resample.h Kaiser)
  -> MusicFM -> layer7(vocal)/layer3(bgm) via septoken bestrq; full-mix -> 1rvq-MusicFM(layer6) -> 1rvq-RVQ -> pmt.
  Replicates generate_septoken.sound2code windowing (min_samples=40s tile + trim to output_len=int(len/sr*25)+1).
  sgsep_load_1rvq() added to songgen-septoken.h (additive) loads the single 1rvq codebook as an SgRvqStream.
- tools/songgen-clone-encode-test.cpp: first.wav as both stems + full mix -> 3 streams vs golden-large/clone/
  tokens_{pmt,vocal,bgm}.npy.  RESULT (376 frames each): overall pmt 83.5% / vocal 88.0% / bgm 85.9% agreement
  (capture predicted ~80%). Divergence is ENTIRELY the resampler: sgsep_rvq_encode is bit-exact, MusicFM is >0.9998
  on the golden 24k audio; mismatches are single-frame cosine-NN tie flips between adjacent codes from Kaiser-vs-
  julius resample. No numeric golden for full chain — leading-frame/overall-% is the gate, then ear-test.
- WEIGHTS: convert_musicfm.py on model_1rvq -> gguf/songgen-musicfm-1rvq.gguf (439 tensors; mel norm constants are
  global so identical to septoken). scripts/convert_1rvq_aux.py -> gguf/songgen-1rvq-aux.gguf (5 tensors: rvq_pmt
  codebook[16384,32]+in/out_proj WN-folded). 1rvq layer = 6 (generate_1rvq layer_num), bgm layer = 3, vocal = 7.
- CONDITIONING: SonggenCondInput gained audio_ids_{pmt,vocal,bgm} (per-codebook prompt ids). songgen-lelm.h
  sglm_forward + songgen-lelm-gen.h sggen_step now gather audio_emb.{0,1,2} from these 3 streams instead of a
  single all-16385 audio_ids. Empty -> falls back to audio_ids (no-prompt path UNCHANGED, lelm oracle still passes).
  Each id stream = sgclone_build_audio_ids: [eos_token_id(16384), codes..., special(16385) pad] to audN=251 (per
  lm_levo.prepare_condition_tensors 222-230 + QuantizedEmbeddingConditioner.forward; the learned EOT is prepended
  in-graph). cb0=pmt, cb1=vocal, cb2=bgm (codeclm.py:229 cat order). ignore_tokens: prompt pmt codes <16384 are
  -inf'd out of cb0 logits (lm_levo 525-526). Uncond CFG row drops audio (all-special).
- CLI tools/songgen-clone.cpp: 7 ggufs + --vocal-stem/--bgm-stem [--full-mix] --lyric [--description] out.wav [seed].
  MVP: pre-separated stems required (no demucs). --full-mix omitted -> pmt source = resampled vocal+bgm sum (approx).
STEP-0 DECISIONS (cited):
  (a) audio_qt_embs[1,3,T]: cb0=pmt(1rvq bestrq L6 -> rvq_bestrq_emb), cb1=vocal(septoken bestrq L7 -> rvq_bestrq_emb),
      cb2=bgm(septoken bestrq L3 -> rvq_bestrq_bgm_emb). generate_septoken.py:74-75, generate_1rvq.py:20, codeclm.py:229.
  (b) Clone uses LYRICS + prompt_audio; type_info conditioner stays on its "[Musicality-very-high], ." default (always
      built by lm_levo); user --description is NOT combined with prompt_audio (reference convention).
  (c) Decode does NOT use prompt true_latents in-context. levo_inference_lowmem.py:181 calls generate_audio(tokens)
      WITHOUT prompt wavs for the separate-tokenizer path (melody_is_wav=False), so the latent_masks=2/incontext
      euler path is a continuation-only feature, NOT used for clone. Style cloning is carried purely by the LM
      conditioning. The VAE-encoder graph (songgen-vae-enc.h) is therefore NOT wired into the clone CLI (left for
      future chunked continuation). This is the MVP decision; documented in songgen-clone.cpp header.
VALIDATE: encode-test overall% above; end-to-end out/clone.wav (sanity + ear, no full-chain golden).
CMake: songgen-clone-encode-test + songgen-clone targets added. Build -DBLAS_LIBRARIES="/usr/lib/libcblas.so;/usr/lib/libblas.so".

### (historical) #9 Part A — LeLM generation loop
Needs reverse-engineering of model_septoken.py separate-tokenizer DECODE path (not yet done):
  gen_tokens[1,3,T] -> split vocal/bgm -> RVQ-codebook embed -> mu(2048=bestrq+bgm) + incontext_x +
  x_next assembly -> CFM euler ODE loop (estimator validated) -> latent -> VAE scale -> decode -> wav.
Plus LeLM real generation: dual KV cache (main+sub × {cond,uncond}) + batched-2 CFG decode + sampling
  (cb0 top_k 250, cb1/cb2 argmax, sliding rep-penalty win 50) reproducing gen_tokens_lm. Then CLI + ear test.
The 3 graphs are validated in isolation; #9 is integration glue + the LeLM KV-cache decode path.

### #9 wire LeLM→CFM→VAE + optional prompt-audio path, expose via CLI/server. Verify end-to-end.

---

## 5. Gotchas / decisions
- model.pt is LeLM ONLY (audio stack separate). Conditioning is embeddings, not a transformer.
- Quant ladder (from acestep quantize.sh): LM→Q8_0 (Q4_K_M "broke audio codes" on ACE-Step —
  UNTESTED here, Q4 minted for trial). VAE stays BF16 (ggml runs bf16 at runtime).
- /tmp is tmpfs (47G RAM, ephemeral). Keep deliverables on nvme.
- Weights split: LeLM from lglg666/SongGeneration-v2-large, audio ckpts from tencent/SongGeneration.
- transformers 5.x removed many symbols the 2024-era repo needs; all shimmed in sg_compat (import-time only, no numeric impact). LeLM forward verified correct on transformers 5.9 with shims.

## 6. Open questions
- BEFORE porting: is there already a SongGeneration/LeVo ggml/cpp port online? (research in flight)
- Audio goldens not yet captured (torchvision chain) — easiest on the CUDA box.
- Large LeLM Q8 = 5.1G; fits 3060 with audio stack staged (peak = max stage). Confirm on hardware.

### #22 HTDemucs (single-file audio in) — recon DONE, native port IN PROGRESS
HTDemucs is NOT core — external Meta separator (MIT), only the audio-CLONE preprocessing splits a mix
into vocal/bgm. Weights downloadable (80MB, dl.fbaipublicfiles.com, no auth) -> persisted ckpt-audio/demucs/
{htdemucs.th, htdemucs_state_dict.pth (533 keys), htdemucs_config.json}. ~42M params (NOT 600M). Golden +
per-block oracles captured: golden-large/demucs/{stems_44k, vocal_48k, bgm_sum_48k(=drums+bass+other, the
ENCODE-path bgm), bgm_sub_48k(=mix-vocals, gradio path), full_48k, intermediates/{spec_z, encoder0-3_out,
tencoder0-3_out, crosstransformer_out, final_out, input_seg}}. Arch: depth4, ch48, nfft4096 hop1024, CAC,
2 parallel U-Nets (spec Conv2d + time Conv1d) + CrossTransformerEncoder bottleneck (5 layers [self,cross,
self,cross,self], dim512/8h, LayerScale, sinusoidal pos, 31.5M params = the hard 75%). Stem map: vocal=vocals
stem; bgm=drums+bass+other (generate_septoken.py:64) for the encode path. demucs ran in isolated /tmp/demucs-venv
(py3.12, torch2.12+torchaudio2.11; NOT in /tmp/sg-venv to avoid corrupting the ROCm venv; soundfile IO).
DECISION (user asleep): attempt native ggml port (block-by-block gated on the intermediates). FALLBACK if it
stalls overnight: demucs front-end (python /tmp/demucs-venv separates -> stems -> existing C++ clone), keeping
core inference pure-C++ and demucs as an external preprocessing tool. Either way single-file-in ships by morning.

### #22 SINGLE-FILE-IN — DELIVERED via demucs front-end ✅ (native ggml port = bonus, in progress)
scripts/demucs_separate.py (runs in /tmp/demucs-venv): mix -> vocal_48k + bgm_48k(drums+bass+other),
VERIFIED cossim 0.9995/0.9994 vs recon golden stems. scripts/songgen-clone-file.sh <mix> <out> "<lyric>" [seed]
= separate -> songgen-clone. Core inference stays pure C++; demucs is external preprocessing only.
Native pure-C++ ggml HTDemucs port attempted separately (agent, block-gated on golden-large/demucs/intermediates).

### #22 NATIVE HTDemucs ggml port — COMPLETE ✅ (single-file-in now PURE C++)
src/songgen-htdemucs.h + gguf/songgen-htdemucs.gguf (389t,83MB) + tools/songgen-separate.cpp + convert_htdemucs.py.
Block-by-block all >0.9998 (spec+time U-Nets, 5L cross-transformer, decoders, final_out 0.99996); end-to-end
vs golden vocal 0.9955 / bgm 0.9916 (gap = our deterministic shifts=0 vs demucs shifts=1 + resampler). Bug caught:
freq_emb is ScaledEmbedding(scale=10) so add factor = 0.2*10 = 2.0. dconv_mode=1 -> decoders have no DConv.
CLI: songgen-separate <htdemucs.gguf> <mix.wav> <vocal48.wav> <bgm48.wav>. Native single-file clone:
scripts/songgen-clone-native.sh <mix> <out> "<lyric>" [seed] (pure C++); scripts/songgen-clone-file.sh = python
demucs front-end alternative (shifts=1, marginally higher SDR). EOS FINDING (overnight): model does NOT emit EOS
in a 40s [outro] gen -> duration is the effective length (under-emit, like most music LMs; detect logic matches ref).

### LYRIC FORMATTING RULES (CRITICAL — I missed these initially; README §"Lyrics Input Format")
All prior gens (golden, synthwave, lofi, rock, clones, the first EOS test) used MALFORMED lyrics ->
degraded quality AND likely the no-EOS behavior (model never saw a proper ending). CORRECT format:
  1. Sections separated by " ; " (semicolon). [NOT spaces]
  2. Instrumental tags [intro-*]/[outro-*]/[inst-*]/[silence] = STANDALONE, NO lyrics after them.
  3. Lyrical tags [verse]/[chorus]/[bridge] REQUIRE lyrics; separate phrases with ". " ; final phrase
     in each block ends with "." before the " ;". English half-width punctuation only.
  Example: "[intro-medium] ; [verse] Trails wind through the forest. Trees stand tall. ; [chorus] ... honest. ; [inst-medium] ; [verse] ... ; [outro-medium]"
IMPLICATION: the EOS deep-dive conclusion ("model genuinely under-emits EOS") was based on malformed
lyrics -> SUSPECT. Re-testing with a properly-formatted full song ending in standalone [outro-short]
(out/night_fulllen.wav, SGLM_EOS_DEBUG). If EOS fires now -> the fix was just "format lyrics correctly".
TODO: re-gen the demos with correct formatting; the README also notes structure 'short≈0-10s/medium≈10-20s/long'
hints (advisory). Description format: comma-separated TAGS not sentences (already mostly doing this).

### EOS RESOLVED ✅ — it was the lyric formatting, full stop.
With CORRECTLY-formatted lyrics (proper ; separators, standalone [intro]/[outro], periods), a 120s-ceiling
gen SELF-TERMINATED via EOS at frame 2802 = 112s (out/fulllen_eos_selfterminated.wav). EOS prob rose to
rank 1 (argmax) at the song's natural end. The earlier "model under-emits EOS / duration is the worst part"
was 100% an artifact of malformed lyrics (no proper [outro] -> never reached song-over). NO fix needed:
behaves like the reference (single-shot, 270s ceiling, stops on EOS; extend_stride is vestigial/NotImplemented).
GUIDANCE: format lyrics per the rules + set a generous --duration (or default ~270s), model ends itself.
