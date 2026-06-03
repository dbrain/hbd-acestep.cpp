# Audio stage (#8) — architecture spec + goldens

Captured deterministically on this box (CPU, bit-exact). Goldens in golden-large/audio/.
Reference source (code) persisted at /home/dbrain/dev/songgen-port/songgen-src-code/.
Audio ckpts persisted at /home/dbrain/dev/songgen-port/ckpt-audio/.

Pipeline: gen_tokens[1,3,T] -> seperate_tokenizer.decode([vocal,bgm], chunked) ->
  per chunk: bestrq/rvq encode cond -> CFM reflow ODE (euler) over estimator -> latent[.,64,.]
  -> Stable Audio Oobleck VAE decode -> 48kHz stereo wav.

## A. Stable Audio VAE decoder (Oobleck/Snake)  [task #8c]
module: stable_audio_tools.models.autoencoders.OobleckDecoder (decoder only; bottleneck already
  sampled 128->64 upstream, so the port's input is the 64ch post-bottleneck latent).
weights: ckpt-audio/vae/autoencoder_music_1320k.ckpt -> ["state_dict"], prefix `decoder.` (182 keys).
  weight-norm pairs weight_g/weight_v -> FOLD at convert time: w = g * v / ||v|| (norm over the
  conv's (in,kernel) dims per output channel, i.e. dim=[1,2] for conv1d weight [out,in,k]).
config: out_channels=2, channels=128, latent_dim=64, c_mults=[1,2,4,8,16], strides=[2,4,4,6,10],
  use_snake=true, final_tanh=false. downsampling_ratio=prod(strides)=1920. sr=48000.
Oobleck decoder structure (standard Stable Audio Open):
  in: Conv1d(latent_dim=64 -> channels*c_mults[-1]=128*16=2048, k=7, pad=3)
  then for each (stride, cmult) from coarse->fine (reverse of encoder): DecoderBlock =
    Snake -> ConvTranspose1d(upsample by stride) -> 3x ResidualUnit(dilations 1,3,9).
  ResidualUnit: Snake -> Conv1d(k=7,dilation,pad) -> Snake -> Conv1d(k=1) + residual.
  final: Snake -> Conv1d(-> out_channels=2, k=7, pad=3) [no tanh].
Snake activation: x + (1/alpha) * sin(alpha*x)^2   (alpha,beta params per channel; this VAE
  uses SnakeBeta? confirm in source: alpha & beta -> x + (1/beta)*sin(alpha x)^2). CHECK source.
ggml: compose Snake from ggml_sin/ggml_sqr/ggml_mul/ggml_add/ggml_div. Conv1d/ConvTranspose1d
  via ggml_conv_1d / ggml_conv_transpose_1d (or im2col). Reuse acestep src/vae.h patterns where
  applicable (verify it has Snake/weight-norm; ACE-Step VAE may differ).
GOLDEN: vae_latent.npy [1,64,32] (seed0 randn) -> vae_wav.npy [1,2,61440]. Validate cossim/MSE
  + spectral distance. wav std ~0.169, range ~[-0.86,0.71].

## B. CFM reflow estimator (custom GPT2)  [task #8d]
module: models_gpt.models.gpt2_rope2_time_new_correct_mask_noncasual_reflow.GPT2Model
  (custom: RoPE2 positional, time-conditioned via adaLN-single, NON-causal, reflow). NOT vanilla HF GPT2.
weights: ckpt-audio/model_septoken/model_2.safetensors, prefix `cfm_wrapper.estimator.` (221 keys).
config: GPT2Config(n_positions=1000, n_layer=16, n_head=20, n_embd=2200, n_inner=4400), eager attn.
call: est(inputs_embeds=[2B,T,2200], attention_mask=[2B,1,T,T] ones, time_step=[2B] float in [0,1])
  -> out.last_hidden_state[2B,T,2200]; decode consumes velocity = last_hidden_state[..., -64:].
inputs_embeds feature layout (sums to 2200): latent_mask_emb(24) | incontext_x(64) | mu(2048 =
  bestrq 1024 + bestrq_bgm 1024) | x_next(64).
non-vanilla details: adaln_single = sinusoidal timestep embed -> 2x Linear; a scale_shift_table
  param; final out = proj_out( ln_f(h)*(1+scale) + shift ). RoPE2 custom (read source for theta/mode).
  time_step is continuous reflow time, not integer diffusion step.
ODE: euler integration around the estimator (solve_euler) — trivial arithmetic; see reflow update
  rule in source. Validate single forward vs golden first, then the euler loop.
GOLDEN: cfm_in_inputs_embeds.npy[2,32,2200], cfm_in_time_step.npy[2]=0.3,
  cfm_in_attention_mask.npy[2,1,32,32] ones -> cfm_out_last_hidden_state.npy[2,32,2200] +
  cfm_out_velocity.npy[2,32,64]. lh std ~0.142, vel std ~0.793.

## C. RVQ / separate tokenizer  [task #8e, part of #9 wiring]
model_1rvq/model_2_fixed.safetensors (single-RVQ tokenizer, prompt encode path) +
model_septoken (Flow1dVAESeparate: estimator above + RVQ). The mu conditioning (bestrq 2048) and
incontext come from the encode path — needed for real generation, not for the single-forward golden.
Defer full wiring to #9; estimator+VAE graphs validated independently first.

## Notes
- Import blocker resolved: stub k_diffusion (sys.modules) before importing stable_audio_tools /
  models_gpt; torchvision NOT needed. transformers 5.x GPT2Config/GPT2Model compat shims needed
  (see scripts/golden_cfm_estimator.py). This box suffices — no CUDA box required for goldens.
- VAE stays BF16 at runtime (ggml runs bf16); estimator can Q8.

## D. Decode glue (gen_tokens -> wav) — #9, from-codes no-prompt path (VALIDATED via source read)
Source: model_septoken.py inference_codes (589-657) + solve_euler (139-194) + Feature1DProcessor.
gen_tokens[1,3,T] from LM: codebook 0=song(unused in sep decode), 1=VOCAL, 2=BGM.
For each of vocal/bgm code stream [1,1,T] (codes 0..16383):
  mu_x = rvq.from_codes(codes) -> [B,1024,T] then permute -> [B,T,1024].
  rvq = ResidualVectorQuantize(input_dim=1024, n_codebooks=1, codebook_size=16384, codebook_dim=32).
  from_codes(1 codebook): z = codebook.weight[codes] (32-dim) -> out_proj (32->1024). Weights under
  cfm_wrapper... NO: under rvq_bestrq_emb.* and rvq_bestrq_bgm_emb.* in model_septoken/model_2.safetensors.
  (descript_quantize3: quantizers.0.codebook.weight[16384,32], quantizers.0.out_proj.weight[1024,32]
   (1x1 conv, may be weight_normed) — VERIFY exact key names + weight_norm at convert.)
mu = cat([mu_vocal, mu_bgm], feature) -> [B,T,2048].
NO-PROMPT single segment (scenario=start_seg, latent_length=T=num_frames):
  latent_masks = all 2  => mu kept everywhere (no zero_cond), incontext_latents = 0 (no mask==1),
  incontext_length=0 (freeze no-op), attention_mask all-ones (full bidirectional).
  latent_mask_input = mask_emb(2) broadcast [B,T,24]  (mask_emb = Embedding(3,24)).
inputs_embeds[B,T,2200] = cat([ latent_mask_input(24), incontext_x(64)=zeros, mu(2048), x_next(64) ], dim=2).
  NOTE order matches estimator golden layout 24|64|2048|64. x_next is the LAST 64 (the ODE state).
EULER ODE (solve_euler): t_span=linspace(0,1,num_steps+1), num_steps=20, dt=1/20. x_next init=randn[B,T,64]*1.0.
  guidance_scale=1.5 (CFG): build batch 2 = [uncond, cond] where uncond zeros the mu(2048) block only
  (latent_mask_input, incontext_x, x_next duplicated). timestep=ti for both rows.
  v = estimator(inputs_embeds, attn=ones, time_step)[..., -64:]; v = v_uncond + 1.5*(v_cond - v_uncond);
  x_next += dt*v. (sigma_min=1e-4 only matters with incontext; no-op here.)
LATENT -> VAE: latents[B,T,64] -> permute [B,64,T] -> normfeat.return_sample:
  x = x * (std/target_std)^power_std + mean,  target_std=1, power_std=1.0, mean/std[64] from
  normfeat buffers (sum_x,sum_x2,counts -> mean=sum_x/counts, std=sqrt(sum_x2/counts-mean^2)). VERIFY keys.
  then VAE.decode (songgen-vae.h) -> wav[B,2,T*1920] @ 48kHz.
Chunking/cross-fade: SKIP for first ear test (single chunk, T<= ~ a few hundred frames). Add later for long tracks.
First ear test: feed golden gen_tokens_lm.npy[1,3,375] -> wav. (LM generation loop = Part A, separate.)

## E. LeLM real generation (Part A of #9) — produces gen_tokens (well understood from lm_levo.py)
Add dual KV cache to songgen-lelm.h: main(36L)+sub(12L), each 2 sets (cond,uncond). Batched-2 CFG decode.
Per step: forward 1 new token (both transformers keep KV), CFG combine logits = uncond+(cond-uncond)*1.5.
Sampling: cb0 = top_k(250) softmax(temp=1.0) multinomial; cb1,cb2 = argmax. rep penalty: logits[:,q,:n]/=
  1.1^bincount(unique(window)) over sliding window 50 (record_window 150 actually; win=50 per SPEC, CHECK).
  ignore_tokens (prompt audio codes <16384 masked from cb0) = empty when no prompt. delay [0,250,250]
  unroll; mask non-valid positions to special; EOS=16384 ends a codebook. Validate KV path logits vs the
  full-prefill path (already golden-matched) before trusting generation.

## F. Audio-prompt CLONE path (#13) — feasibility CONFIRMED, all NN weights present
CORRECTION to early recon: MusicFM/BESTRQ weights ARE shipped (inside model_septoken/model_2.safetensors
AND model_1rvq/model_2_fixed.safetensors under prefix `bestrq.` = 463 tensors each). Config =
MusicFM_95M (~95M param BESTRQ conformer; our_MERT_BESTRQ/mert_fairseq/models/musicfm). NOT a blocker.
Reference flow (levo_inference_lowmem.py:62-95, prompt_audio_path branch):
  Separator(demucs).run(wav) -> (full,vocal,bgm)   [HTDemucs ~600M, NOT shipped, public/downloadable, OPTIONAL
                                                     if user supplies pre-separated stems]
  audio_tokenizer.encode(full) -> pmt tokens  (Flow1dVAE1rvq: bestrq+rvq, model_1rvq)
  seperate_tokenizer.encode(vocal,bgm) -> vocal,bgm tokens (Flow1dVAESeparate.encode: bestrq layer7(voc)/
    layer3(bgm) -> rvq_bestrq_emb / rvq_bestrq_bgm_emb -> codes)
  => audio_qt_embs[1,3,T] (pmt,vocal,bgm) -> LeLM prompt_audio conditioner (REAL codes, not the zeros we feed now).
  ALSO: prompt wav -> VAE ENCODER -> true_latents -> CFM in-context (the no-op we currently skip).
Sub-models to port (weights, effort):
  - VAE ENCODER (Oobleck, prefix `encoder.` in autoencoder_music ckpt) — mirror of our decoder. SMALL.
  - RVQ ENCODE (in_proj[1024->32] + nearest-codebook argmin over 16384 + out_proj) — mirror of from_codes. SMALL.
  - bestrq = MusicFM_95M conformer (prefix bestrq., 463 tensors) — NEW arch, but small + weights present. MED.
    resamplers rsq48tobestrq (48k->24k) present (1 tensor each, buffers). bestrq features_only -> layer_results[L].
  - Wire prompt_audio conditioner (already embed-gather ported, feed real codes) + CFM in-context (true_latents).
  - Demucs HTDemucs: download public weights OR require pre-separated vocal/bgm stems (MVP). MED/optional.
Cheaper alt: genre library = ckpt/prompt.pt (3MB, dict[genre]->[1,3,250] precomputed tokens) -> directly to LeLM
  prompt_audio, NO encode/demucs/bestrq needed. Genres: Pop,R&B,Dance,Jazz,Folk,Rock,Chinese*,Metal,Reggae,Opera.
  TRIVIAL to add; gives style steering without the encoder ports. Good stepping stone / standalone feature.
Validation: capture goldens on THIS box (CPU, k_diffusion stub) for bestrq forward, VAE encode, RVQ encode,
  and full prompt-token extraction from a sample wav; cossim-port each like #8.

## G. Clone encoders — exact arch (captured, goldens in golden-large/clone/) [#13]
Reference-audio caveat for goldens: out/first.wav fed as BOTH vocal & bgm (no demucs); musically
meaningless but exercises encode graphs deterministically. MusicFM ref code: songgen-src-code/.../
our_MERT_BESTRQ/mert_fairseq/models/musicfm (persisted). torchaudio mel/resample reimplemented in the
golden scripts (no ROCm wheel) — C++ MUST reuse the checkpoint's mel_scale.fb[1025,128] + window[2048].

### MusicFM_95M (bestrq.* , 463 tensors, features_only=True => quantizer/cls/linear UNUSED) [MED-LARGE]
golden: golden-large/clone/bestrq_{in_audio24k[1,360000], out_layer7_vocal[1,1024,375], out_layer3_bgm,
  out_all_layers[13,1024,375]}. 48k->24k = mean(2ch) then julius.resample_frac (cross-check on CUDA).
1. Mel front-end @24k: MelSpectrogram(n_fft=2048, hop=240, n_mels=128, power=2, center, reflect) using the
   SHIPPED Hann window[2048] + mel fb[1025,128]; AmplitudeToDB(power, top_db=80, amin=1e-10); normalize
   (x-6.7684)/18.4179; drop last mel frame (melspec[...,:-1]).
2. Conv2dSubsampling: 2x Res2dModule (Conv2d3x3+BN+ReLU + strided residual conv), strides (2,2)/(2,2) =>
   4x down in time&freq; rearrange b c f t -> b t (c f); Linear(16384->1024). channels 1->512->512.
3. Wav2Vec2ConformerEncoder: 12 layers, hidden1024, 16 heads(hd64), ffn4096, swish, rotary pos
   (base10000, inv_freq len32) + pos_conv_embed(Conv1d k=128 groups=16 weight-normed) before layers;
   do_stable_layer_norm=True eps1e-5. Macaron block: 0.5*FFN1 -> self_attn(rotary) -> conv_module(LN->
   pointwise1->GLU->depthwise k=31 +BN+swish->pointwise2) -> 0.5*FFN2 -> final_LN. 24k/960 => 25Hz.
   output_hidden_states => 13-elem layer_results (idx0=embeddings, 1..12=layers). USE idx7(vocal)/idx3(bgm).

### VAE encoder (autoencoder_music, prefix encoder.) [SMALL, mirror of decoder]
golden: vaeenc_in_audio48k[1,2,480000] -> pre_bottleneck_latents[1,128,250] -> {mean,scale,stdev,sampled}[1,64,250].
OobleckEncoder: conv_in -> 5 EncoderBlocks (each: 3 ResidualUnit[snake+WNConv1d dil1,3,9] then snake+strided
  WNConv1d), strides[2,4,4,6,10] c_mults[1,2,4,8,16] -> snake+WNConv1d to 128ch. Bottleneck(vae):
  mean,scale=chunk(2) [64 each]; stdev=softplus(scale)+1e-4; sample=randn*stdev+mean. code2sound uses the
  SAMPLE (RNG) — port may use mean as deterministic substitute (both saved; for true_latents in-context).
  WN fold W=g*v/||v|| like decoder.

### RVQ encode (rvq_bestrq_emb / _bgm , mirror of from_codes) [SMALL]
golden: rvq_{vocal,bgm}_{z_e[1,32,375], codes[1,1,375], quantized[1,1024,375]}.
Per frame: in_proj(WNConv1d 1024->32) -> L2-normalize z_e AND codebook rows -> euclidean dist on normalized
  (= cosine NN) -> argmin over 16384 -> look up ORIGINAL (un-normalized) codebook row -> out_proj(WNConv1d
  32->1024). eval => no stale replacement. WN folded.

### Wiring (#13c): prompt audio -> 3 token streams (pmt via 1rvq, vocal/bgm via septoken) = audio_qt_embs[1,3,T]
  -> LeLM prompt_audio conditioner (REAL codes, replaces our all-16385). true_latents (VAE-encode of prompt)
  -> CFM in-context (latent_masks=2 for prompt region; the no-op we currently skip becomes active).
  Demucs absent: MVP requires user to pass pre-separated vocal+bgm stems (1rvq pmt needs no separation),
  OR download facebook/demucs htdemucs. Golden end-to-end token streams: golden-large/clone/tokens_{vocal,bgm,pmt}[1,1,376].
