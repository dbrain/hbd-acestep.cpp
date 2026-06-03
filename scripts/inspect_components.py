import argparse, re, collections
import torch

BUCKETS = [
    ("lm.main",   re.compile(r"audiolm.*transformer\.")),
    ("lm.sub",    re.compile(r"audiolm.*transformer2\.")),
    ("lm.embed",  re.compile(r"audiolm.*(emb|linear|out|head|norm)")),
    ("lm.other",  re.compile(r"^audiolm\.")),
    ("cfm",       re.compile(r"cfm|estimator|flow")),
    ("vae",       re.compile(r"vae|autoencoder")),
    ("vocoder",   re.compile(r"hifigan|bigvgan|vocoder|generator")),
    ("hubert",    re.compile(r"hubert|mert|bestrq|semantic")),
    ("separator", re.compile(r"separat|demucs")),
]

Q4K_BYTES_PER_PARAM = 4.5 / 8  # Q4_K_M effective

def bucket_of(k):
    for name, rx in BUCKETS:
        if rx.search(k):
            return name
    return "UNMATCHED"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pt", help="path to model.pt")
    ap.add_argument("--show-unmatched", action="store_true")
    args = ap.parse_args()

    sd = torch.load(args.pt, map_location="cpu")
    if not isinstance(sd, dict):
        sd = sd.state_dict()

    params = collections.Counter()
    fp16_bytes = collections.Counter()
    dtypes = collections.defaultdict(collections.Counter)
    unmatched = []

    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        b = bucket_of(k)
        n = v.numel()
        params[b] += n
        fp16_bytes[b] += n * 2
        dtypes[b][str(v.dtype)] += n
        if b == "UNMATCHED":
            unmatched.append((k, tuple(v.shape), str(v.dtype)))

    print(f"{'bucket':12} {'params':>14} {'fp16 GB':>9} {'dtypes'}")
    tot_p = tot_fp16 = 0
    for b, _ in BUCKETS + [("UNMATCHED", None)]:
        if params[b] == 0:
            continue
        dt = ", ".join(f"{d.split('.')[-1]}:{n/1e6:.0f}M" for d, n in dtypes[b].most_common())
        print(f"{b:12} {params[b]:>14,} {fp16_bytes[b]/1e9:>9.2f} {dt}")
        tot_p += params[b]; tot_fp16 += fp16_bytes[b]
    print(f"{'TOTAL':12} {tot_p:>14,} {tot_fp16/1e9:>9.2f}")

    lm_p = sum(params[b] for b in ("lm.main", "lm.sub"))
    audio_p = tot_p - lm_p - params["lm.embed"] - params["lm.other"]
    keep_fp16 = (params["lm.embed"] + params["lm.other"]) * 2 + audio_p * 2
    q4 = lm_p * Q4K_BYTES_PER_PARAM
    print(f"\nProjection: Q4_K on lm.main+lm.sub, fp16 elsewhere")
    print(f"  LM (Q4_K)        : {q4/1e9:6.2f} GB  ({lm_p/1e9:.2f}B params)")
    print(f"  everything fp16  : {keep_fp16/1e9:6.2f} GB")
    print(f"  disk total       : {(q4+keep_fp16)/1e9:6.2f} GB")
    print(f"  ~peak resident   : max(LM stage, codec stage), NOT the sum -> staged exec")

    if args.show_unmatched and unmatched:
        print(f"\n{len(unmatched)} UNMATCHED tensors (refine BUCKETS):")
        for k, shp, dt in unmatched[:60]:
            print(f"  {k}  {shp}  {dt}")

if __name__ == "__main__":
    main()
