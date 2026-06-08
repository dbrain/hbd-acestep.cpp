#!/usr/bin/env python3
# convert-sa3.py: Stable Audio 3 Medium safetensors -> GGUF (separate from acestep convert.py).
# Emits 3 self-contained GGUFs into models/:
#   sa3-dit-<dtype>.gguf      diffusion transformer (model.model.*)
#   sa3-same-<dtype>.gguf     SAME autoencoder enc+dec (pretransform.model.*) + conditioner.* (seconds embedder)
#   sa3-t5gemma-<dtype>.gguf  T5Gemma encoder only (t5gemma-b-b-ul2/model.encoder.*) + tokenizer.model bytes
#
# Tensor names are remapped to short (<64 char) ggml names; full config is embedded as JSON KV.
# Usage: ./convert-sa3.py [--ckpt checkpoints/sa3-medium] [--dtype f32|f16] [--only dit,same,t5gemma]
import os, sys, json, struct, argparse
import numpy as np
import gguf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

F32 = gguf.GGMLQuantizationType.F32
F16 = gguf.GGMLQuantizationType.F16

def log(m): print("[sa3-conv] " + m, file=sys.stderr, flush=True)

# ---------- safetensors mmap reader ----------
class ST:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.meta = json.loads(f.read(n))
        self.meta.pop("__metadata__", None)
        self.base = 8 + n
        self._mm = np.memmap(path, dtype=np.uint8, mode="r")
    def names(self): return list(self.meta.keys())
    def get(self, name):
        info = self.meta[name]
        o0, o1 = info["data_offsets"]
        raw = self._mm[self.base + o0 : self.base + o1]
        dt = info["dtype"]
        if dt == "F32":   arr = raw.view(np.float32)
        elif dt == "F16": arr = raw.view(np.float16)
        elif dt == "BF16":
            u = raw.view(np.uint16).astype(np.uint32) << 16
            arr = u.view(np.float32)
        else: raise ValueError("dtype %s for %s" % (dt, name))
        return np.ascontiguousarray(arr.reshape(info["shape"]))

def fuse_weight_norm(g, v):
    # PyTorch weight_norm default dim=0: norm of v over all dims except 0.
    out = v.shape[0]
    vf = v.reshape(out, -1).astype(np.float64)
    norm = np.sqrt((vf * vf).sum(axis=1, keepdims=True))
    w = (g.reshape(out, 1).astype(np.float64) * vf / norm).astype(np.float32)
    return w.reshape(v.shape)

# ---------- writer helper ----------
class Out:
    def __init__(self, path, arch, dtype):
        self.w = gguf.GGUFWriter(path, arch, use_temp_file=True)
        self.dtype = dtype
        self.n = 0; self.bytes = 0; self.path = path
    def add(self, name, arr, force_f32=False):
        assert len(name) < 64, "name too long (%d): %s" % (len(name), name)
        arr = np.ascontiguousarray(arr)
        if arr.dtype == np.float32 and self.dtype == F16 and not force_f32:
            self.w.add_tensor(name, arr.astype(np.float16))
        else:
            self.w.add_tensor(name, arr.astype(np.float32) if arr.dtype != np.float32 else arr)
        self.n += 1; self.bytes += arr.nbytes
    def finish(self, cfg):
        self.w.add_string("sa3.config_json", json.dumps(cfg, separators=(",", ":")))
        self.w.write_header_to_file(); self.w.write_kv_data_to_file()
        self.w.write_tensors_to_file(progress=True); self.w.close()
        log("wrote %s : %d tensors, %.2f GB" % (os.path.basename(self.path), self.n, self.bytes/(1<<30)))

# ===================== DiT =====================
def conv_dit(st, cfg, out):
    P = "model.model."
    direct = {
        P+"preprocess_conv.weight": "preprocess_conv",
        P+"postprocess_conv.weight": "postprocess_conv",
        P+"to_cond_embed.0.weight": "to_cond.0", P+"to_cond_embed.2.weight": "to_cond.2",
        P+"to_global_embed.0.weight": "to_global.0", P+"to_global_embed.2.weight": "to_global.2",
        P+"to_timestep_embed.0.weight": "to_tstep.0.w", P+"to_timestep_embed.0.bias": "to_tstep.0.b",
        P+"to_timestep_embed.2.weight": "to_tstep.2.w", P+"to_timestep_embed.2.bias": "to_tstep.2.b",
        P+"transformer.global_cond_embedder.0.weight": "gce.0.w", P+"transformer.global_cond_embedder.0.bias": "gce.0.b",
        P+"transformer.global_cond_embedder.2.weight": "gce.2.w", P+"transformer.global_cond_embedder.2.bias": "gce.2.b",
        P+"transformer.memory_tokens": "mem_tokens",
        P+"transformer.project_in.weight": "project_in",
        P+"transformer.project_out.weight": "project_out",
        P+"transformer.rotary_pos_emb.inv_freq": "rope_inv_freq",
    }
    for src, dst in direct.items():
        out.add(dst, st.get(src), force_f32=("rope_inv_freq" in dst or "norm" in dst))
    L = P + "transformer.layers."
    nl = cfg["model"]["diffusion"]["config"]["depth"]
    lmap = {
        "pre_norm.gamma":"pre_norm", "ff_norm.gamma":"ff_norm", "cross_attend_norm.gamma":"ca_norm",
        "self_attn.to_qkv.weight":"sa_qkv", "self_attn.to_out.weight":"sa_out",
        "self_attn.q_norm.gamma":"sa_qn", "self_attn.k_norm.gamma":"sa_kn",
        "cross_attn.to_q.weight":"ca_q", "cross_attn.to_kv.weight":"ca_kv", "cross_attn.to_out.weight":"ca_out",
        "cross_attn.q_norm.gamma":"ca_qn", "cross_attn.k_norm.gamma":"ca_kn",
        "ff.ff.0.proj.weight":"ff_in.w", "ff.ff.0.proj.bias":"ff_in.b",
        "ff.ff.2.weight":"ff_out.w", "ff.ff.2.bias":"ff_out.b",
        "to_scale_shift_gate":"ssg",
        "to_local_embed.0.weight":"loc.0.w", "to_local_embed.0.bias":"loc.0.b",
        "to_local_embed.2.weight":"loc.2.w", "to_local_embed.2.bias":"loc.2.b",
    }
    for i in range(nl):
        for src, dst in lmap.items():
            out.add("blk.%d.%s" % (i, dst), st.get("%s%d.%s" % (L, i, src)),
                    force_f32=("_norm" in dst or "_qn" in dst or "_kn" in dst or dst=="ssg"))

# ===================== SAME (taae_v2) =====================
def conv_same_stack(st, prefix_blk, out_prefix, out, depth):
    # transformers.M.* differential-attn block (DynamicTanh norms)
    nmap_w = {
        "self_attn.to_qkv.weight":"sa_qkv", "self_attn.to_out.weight":"sa_out",
        "ff.ff.0.proj.weight":"ff_in.w", "ff.ff.0.proj.bias":"ff_in.b",
        "ff.ff.2.weight":"ff_out.w", "ff.ff.2.bias":"ff_out.b",
    }
    nmap_dyt = {  # DynamicTanh: alpha[1],beta[1536],gamma[1536]; qk: alpha[1],beta[64],gamma[64]
        "pre_norm":"pre_norm", "ff_norm":"ff_norm",
        "self_attn.q_norm":"sa_qn", "self_attn.k_norm":"sa_kn",
    }
    for m in range(depth):
        b = "%s%d." % (prefix_blk, m)
        op = "%sblk.%d." % (out_prefix, m)
        for src, dst in nmap_w.items():
            out.add(op+dst, st.get(b+src))
        for src, dst in nmap_dyt.items():
            for p in ("alpha","beta","gamma"):
                out.add(op+dst+"."+p[0], st.get(b+src+"."+p), force_f32=True)
        out.add(op+"rope", st.get(b+"rope.inv_freq"), force_f32=True)

def conv_same(st, cfg, out):
    aecfg = cfg["model"]["pretransform"]["config"]
    depth = aecfg["decoder"]["config"]["transformer_depths"][0]
    PB = "pretransform.model.bottleneck."
    out.add("bn.bias", st.get(PB+"bias").reshape(-1), force_f32=True)
    out.add("bn.scale", st.get(PB+"scaling_factor").reshape(-1), force_f32=True)
    out.add("bn.running_std", st.get(PB+"running_std").reshape(-1), force_f32=True)
    # ----- decoder -----
    PD = "pretransform.model.decoder.layers."
    out.add("dec.in.w", st.get(PD+"1.weight")); out.add("dec.in.b", st.get(PD+"1.bias"), force_f32=True)
    conv_same_stack(st, PD+"3.transformers.", "dec.", out, depth)
    wmap = fuse_weight_norm(st.get(PD+"3.mapping.weight_g"), st.get(PD+"3.mapping.weight_v"))
    out.add("dec.map.w", wmap); out.add("dec.map.b", st.get(PD+"3.mapping.bias"), force_f32=True)
    out.add("dec.new_tokens", st.get(PD+"3.new_tokens").reshape(1,-1), force_f32=True)
    # ----- encoder (for audio2audio; convert but t2a unused) -----
    PE = "pretransform.model.encoder.layers."
    wmape = fuse_weight_norm(st.get(PE+"0.mapping.weight_g"), st.get(PE+"0.mapping.weight_v"))
    out.add("enc.map.w", wmape); out.add("enc.map.b", st.get(PE+"0.mapping.bias"), force_f32=True)
    out.add("enc.new_tokens", st.get(PE+"0.new_tokens").reshape(1,-1), force_f32=True)
    conv_same_stack(st, PE+"0.transformers.", "enc.", out, depth)
    out.add("enc.out.w", st.get(PE+"2.weight")); out.add("enc.out.b", st.get(PE+"2.bias"), force_f32=True)
    # ----- conditioner: seconds_total expo-fourier embedder (Linear 256->768) + prompt pad embed -----
    out.add("sec.emb.w", st.get("conditioner.conditioners.seconds_total.embedder.embedding.1.weight"))
    out.add("sec.emb.b", st.get("conditioner.conditioners.seconds_total.embedder.embedding.1.bias"), force_f32=True)
    out.add("prompt.pad_embed", st.get("conditioner.conditioners.prompt.padding_embedding"), force_f32=True)

# ===================== T5Gemma encoder =====================
def conv_t5gemma(st, cfg, out, tok_model_path):
    P = "model.encoder."
    out.add("tok_embd", st.get(P+"embed_tokens.weight"))
    out.add("output_norm", st.get(P+"norm.weight"), force_f32=True)
    nl = cfg["encoder"]["num_hidden_layers"]
    lmap_w = {
        "self_attn.q_proj.weight":"attn_q", "self_attn.k_proj.weight":"attn_k",
        "self_attn.v_proj.weight":"attn_v", "self_attn.o_proj.weight":"attn_o",
        "mlp.gate_proj.weight":"ffn_gate", "mlp.up_proj.weight":"ffn_up", "mlp.down_proj.weight":"ffn_down",
    }
    lmap_n = {
        "pre_self_attn_layernorm.weight":"attn_norm", "post_self_attn_layernorm.weight":"post_attn_norm",
        "pre_feedforward_layernorm.weight":"ffn_norm", "post_feedforward_layernorm.weight":"post_ffn_norm",
    }
    for i in range(nl):
        for src, dst in lmap_w.items():
            out.add("blk.%d.%s" % (i, dst), st.get("%slayers.%d.%s" % (P, i, src)))
        for src, dst in lmap_n.items():
            out.add("blk.%d.%s" % (i, dst), st.get("%slayers.%d.%s" % (P, i, src)), force_f32=True)
    # Embed Gemma BPE tokenizer (vocab + merges) from tokenizer.json for in-C++ tokenization.
    tj_path = os.path.join(os.path.dirname(tok_model_path), "tokenizer.json") if tok_model_path else None
    if tj_path and os.path.exists(tj_path):
        tj = json.load(open(tj_path))
        vocab = tj["model"]["vocab"]
        toks = [""] * len(vocab)
        for t, i in vocab.items():
            if 0 <= i < len(toks): toks[i] = t
        merges = [a + " " + b for a, b in tj["model"]["merges"]]
        out.w.add_tokenizer_model("gpt2")
        out.w.add_token_list(toks)
        out.w.add_token_merges(merges)
        log("embedded Gemma BPE: %d vocab, %d merges" % (len(toks), len(merges)))

# ===================== main =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(SCRIPT_DIR, "checkpoints", "sa3-medium"))
    ap.add_argument("--dtype", default="f32", choices=["f32", "f16"])
    ap.add_argument("--only", default="dit,same,t5gemma")
    ap.add_argument("--outdir", default=os.path.join(SCRIPT_DIR, "models"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    dt = F32 if args.dtype == "f32" else F16
    want = set(args.only.split(","))

    cfg = json.load(open(os.path.join(args.ckpt, "model_config.json")))
    main_st = os.path.join(args.ckpt, "model.safetensors")

    if "dit" in want or "same" in want:
        st = ST(main_st)
        if "dit" in want:
            o = Out(os.path.join(args.outdir, "sa3-dit-%s.gguf" % args.dtype), "sa3-dit", dt)
            conv_dit(st, cfg, o); o.finish(cfg["model"]["diffusion"])
        if "same" in want:
            o = Out(os.path.join(args.outdir, "sa3-same-%s.gguf" % args.dtype), "sa3-same", dt)
            conv_same(st, cfg, o); o.finish(cfg["model"]["pretransform"]["config"])
    if "t5gemma" in want:
        sub = os.path.join(args.ckpt, "t5gemma-b-b-ul2")
        st = ST(os.path.join(sub, "model.safetensors"))
        tcfg = json.load(open(os.path.join(sub, "config.json")))
        o = Out(os.path.join(args.outdir, "sa3-t5gemma-%s.gguf" % args.dtype), "sa3-t5gemma", dt)
        conv_t5gemma(st, tcfg, o, os.path.join(sub, "tokenizer.model")); o.finish(tcfg)
    log("done")

if __name__ == "__main__":
    main()
