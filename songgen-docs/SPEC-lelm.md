# LeLM ggml port — authoritative graph spec (task #7)

Derived from /tmp/songgen-src PyTorch reference + gguf inspection. This is ground truth
for the C++/ggml graph. Build on hbd-acestep.cpp checkout at
/home/dbrain/dev/songgen-port/songgen.cpp (origin git@github:dbrain/hbd-acestep.cpp).

## GGUF: songgen-lelm-large-Q8_0.gguf
arch=songgen-lelm. KV: block_count=36, embedding_length=2048, feed_forward_length=11008,
head_count=16, head_count_kv=16, rope.freq_base=500000, rms_eps≈1e-5 (9.999e-6).
songgen.{num_layers_sub=12, rope_theta_sub=500000, code_depth=3, code_size=16384,
delays=[0,250,250], frame_rate=25, cfg_coef=1.5, prompt_len=10}.
Tokenizer (gpt2/Qwen2 BPE) embedded: tokens[151643], merges[151387].

### Tensors (ne order = [in_dim, rows]) — Q8_0 unless noted
Main transformer (36L Llama, NO QK-norm):
  transformer.model.layers.{0..35}.{input_layernorm.weight[2048]F32,
    post_attention_layernorm.weight[2048]F32, self_attn.{q,k,v,o}_proj.weight[2048,2048],
    mlp.{gate,up}_proj.weight[2048,11008], mlp.down_proj.weight[11008,2048]}
  transformer.model.norm.weight[2048]F32            (final RMSNorm)
  transformer.lm_head.weight[2048,16385]            (cb0 logits head; NOT tied)
  transformer.model.embed_tokens.weight[2048,16385] VESTIGIAL (forward uses inputs_embeds)
Sub transformer (12L Llama, identical dims):
  transformer2.model.layers.{0..11}.* (same layout)
  transformer2.model.norm.weight[2048]F32
  transformer2.lm_head.weight[2048,16385]           VESTIGIAL
  transformer2.model.embed_tokens.weight[2048,16385] VESTIGIAL
Code embeddings (input):
  emb.0.weight[2048,16386]            cb0 input embed (gather sequence[:,0])
  layer2_emb.{0,1,2}.weight[2048,16386]  layer2_emb.0 VESTIGIAL; .1,.2 used (gather seq[:,1],[:,2]) summed
Bridge MLP (Linear 4096->2048, GELU, Linear 2048->2048):
  mlp.0.weight[4096,2048] + mlp.0.bias[2048]F32
  mlp.2.weight[2048,2048] + mlp.2.bias[2048]F32
Sub heads:
  linears.0.weight[2048,16385]  cb1 logits
  linears.1.weight[2048,16385]  cb2 logits
out_norm.{weight,bias}[2048]F32  VESTIGIAL (norm_first=False)
Conditioners:
  condition_provider.conditioners.description.output_proj.weight[2048,151659]  (LYRICS text embed)
  condition_provider.conditioners.description.structure_emb.weight[2048,200]   (structure embed)
  condition_provider.conditioners.type_info.output_proj.weight[2048,151652]    (STYLE desc embed)
  condition_provider.conditioners.prompt_audio.emb.{0,1,2}.weight[2048,16386]  (padding_idx=16385 -> row 16385 is ZERO)
  condition_provider.conditioners.prompt_audio.EOT_emb[2048]F32
  condition_provider.conditioners.prompt_audio.layer2_EOT_emb[2048]F32

## Codebook token id semantics (lm_levo)
self.code_size = code_size+1 = 16385 (= card of logits). input_emb_dim = code_size+2 = 16386.
special_token_id = 16385 (EOP/mask, last emb row). eos_token_id = 16384.
Logits card per codebook = 16385.

## Forward (lm_levo.LmModel.forward + levo.CausalLM) — ONE step, per CFG row
emb code tokens (S code positions, K=3):
  input_1[p] = emb.0[ seq[0,p] ]
  input_2[p] = layer2_emb.1[ seq[1,p] ] + layer2_emb.2[ seq[2,p] ]
fuser PREPEND (first step only), order = [description, prompt_audio, type_info] then codes.
  Applied to BOTH input_1 and input_2. Implementation detail: fuser reverses list and
  cats at front; net result order is description ++ prompt_audio ++ type_info ++ codes.
  Text conds: cond1==cond2 (same embed). prompt_audio: cond1=emb1, cond2=emb2 (differ).
  => fused_input1 = [desc, audio_emb1, type, input_1]   (feature-concat along sequence dim)
     fused_input2 = [desc, audio_emb2, type, input_2]
Main transformer: h1 = RMSNorm_final(Llama36(fused_input1))     [H,S]   (post final norm)
  cb0_logits = lm_head @ h1                                     [16385,S]
Bridge: cat = concat([fused_input2, h1], feature)  -> [4096,S]
  bridged = mlp.2 @ gelu(mlp.0 @ cat + mlp.0.bias) + mlp.2.bias [2048,S]
Sub transformer: h2 = RMSNorm_final(Llama12(bridged))           [H,S]
  cb1_logits = linears.0 @ h2 ; cb2_logits = linears.1 @ h2     [16385,S]
Trim: keep last S_code positions (prepend removed). Captured golden = last position only.
RoPE: BOTH transformers θ=500000, mode=NEOX(2), head_dim=128. rms_eps=1e-5.
Positions: continuous 0..S-1 over [prepend ++ codes] (single inputs_embeds, past_kv=0 at step0).
Both transformers process the SAME sequence positions (sub also keeps KV across AR steps).

## Conditioning construction (the golden's exact inputs)
meta: lyric="[verse] walking through the quiet streets at night [verse] city lights are
calling out my name [chorus] and i will find my way back home"
description(style)="warm indie folk, male vocal, acoustic guitar, 92 bpm"
levo_inference prefixes style with "[Musicality-very-high], " and lowercases.

Special ids (Qwen2 base vocab 0..151642):
  151643=<|endoftext|>, 151644=<|im_start|>, 151645=<|im_end|>,
  description added (vocab.yaml, 13 structure tokens): 151646..151658 =
    [verse],[chorus],[bridge],[intro-short],[intro-medium],[intro-long],
    [outro-short],[outro-medium],[outro-long],[inst-short],[inst-medium],[inst-long],[silence]
    (description.output_proj vocab=151659 = 151643+3+13)
  type_info added (6 musicality, '.' already in base): 151646..151651 =
    [Musicality-very-high],[Musicality-high],[Musicality-medium],[Musicality-low],
    [Musicality-very-low],[Pure-Music]  (type_info.output_proj vocab=151652)

description conditioner (QwTokenizerConditioner, max_len=600):
  tokenize: ids = Qwen2BPE("<|im_start|>" + lyric)   [special tokens matched whole]
  structure cover: tp_cover[pos]=0 until first structure token; from each structure token
    position st until next structure token (or attention end), tp_cover = (id[st]-151645).
    i.e. [verse]->1, [chorus]->2, ... value indexes structure_emb (200 rows, padding_idx=0->zero).
  pad ids to 600 with 151643; pad cover to 600 with 0.
  embed = output_proj[ids] + structure_emb[cover]   ; cond1==cond2.
prompt_audio conditioner (qt_embedding, max_len=prompt_len*frame_rate+2=252):
  NO reference audio => audio tokens all 16385 (len 251 after the prep). emb rows at 16385 are ZERO.
  embeds1 = [EOT_emb, 0,0,...,0]  (252 positions)
  embeds2 = [layer2_EOT_emb, 0,0,...,0] (252)
  SAME for cond and uncond (null audio is also all 16385).
type_info conditioner (QwTextConditioner, max_len=100):
  cond: ids=Qwen2BPE("<|im_start|>[Musicality-very-high], warm indie folk, male vocal, acoustic guitar, 92 bpm")
  pad to 100 with 151643. embed=output_proj[ids]. cond1==cond2. NO structure_emb.
Total prepend P = 600 + 252 + 100 = 952 positions.

## CFG (batch of 2): row0=cond, row1=uncond. golden lelm_logits[i] dim1: [cond,uncond].
uncond (null) conditioning:
  description -> None -> tokenize("<|im_start|>") only (1 real token) pad to 600. cover all 0.
  type_info -> "[Musicality-very-low], ." (because cond contained [Musicality-very-high]).
  prompt_audio -> all 16385 (same as cond, = EOT + zeros).
Combine offline for sampling: logits = uncond + (cond-uncond)*1.5.

## Delay pattern (DelayedPatternProvider, delays=[0,250,250], empty_initial=0, flatten_first=0)
get_pattern(T): layout[0]=[] (all special); layout[1+t] for t in 0..T+249:
  codebook q present iff t-delays[q]>=0, value=code[q, t-delays[q]].
build_pattern_sequence fills missing (q,pos) with special_token_id=16385.
=> seq[:,0] = [16385,16385,16385]
   seq[:,p] (p>=1, let t=p-1): cb0=code[0,t] if t<T else 16385;
     cb1=code[1,t-250] if 0<=t-250<T else 16385; cb2=code[2,t-250] same.
start_offset_sequence=1.

## Golden replay (validation, full-prefill per step, NO kv-cache needed)
golden lelm_logits[16,2,3,16385], gen_tokens_lm[1,3,375] (=out_codes T=375).
Step i (i=0..15): code positions fed = seq[:, 0..i] (i+1 tokens). predict position i+1.
  Build full sequence = prepend(row) ++ code_emb[0..i]; causal; take LAST position logits.
  Compare cb0/cb1/cb2 to golden[i, row]. Target cossim>0.99 per codebook (Q8 vs fp16 ref).
Validate cond (row0) and uncond (row1) separately, plus CFG-combined.

## Build/run
Reuse: backend.h, gguf-weights.h, weight-ctx.h, qwen3-enc.h (SwiGLU MLP qwen3_build_mlp;
write songgen attn WITHOUT QK-norm). Force CPU f32 for determinism: GGML_BACKEND=CPU.
New: src/songgen-lelm.h (graph+load), tools/songgen-lelm-test.cpp (golden replay), CMake target.
Reference-input ids dumped to golden-large/cond_inputs.npz by scripts/dump_cond_inputs.py
(uses /tmp/sg-venv transformers + qwen2-tokenizer + conf/vocab.yaml).
