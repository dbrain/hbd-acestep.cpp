#!/usr/bin/env bash
# Overnight serial gen batch for the morning ear-test. Runs one at a time (respect the box),
# logs timing + EOS, never aborts the batch on a single failure. Outputs -> out/night_*.wav.
ROOT=/home/dbrain/dev/songgen-port
G=$ROOT/gguf; B=$ROOT/songgen.cpp/build; OUT=$ROOT/out
LELM=$G/songgen-lelm-large-Q8_0.gguf
LOG=/tmp/overnight.log
: > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

gen(){ # name seed duration lyric desc
  local name=$1 seed=$2 dur=$3 lyric=$4 desc=$5 t0=$(date +%s)
  say "GEN $name (seed $seed, ${dur}s) ..."
  GGML_BACKEND=CPU "$B/songgen-generate" "$LELM" "$G/songgen-cfm.gguf" "$G/songgen-vae.gguf" \
    "$G/songgen-septoken-aux.gguf" "$ROOT/golden-large" "$OUT/night_$name.wav" "$seed" \
    --lyric "$lyric" --description "$desc" --duration "$dur" >>"$LOG" 2>&1 || say "  !! $name FAILED (continuing)"
  local eos=$(grep -iE "EOS at offset|no EOS" "$LOG" | tail -1)
  say "  $name done in $(($(date +%s)-t0))s. ${eos}"
}

say "=== overnight batch start ==="

# 1) EOS / natural-ending test: short resolving lyric + [outro-short], generous 40s budget.
gen eos_outro 7 40 \
  "[verse] the sun is going down slow [chorus] hold me close before we have to go [outro-short] goodnight my love goodnight" \
  "gentle acoustic ballad, female vocal, soft piano, 70 bpm"

# 2) fresh genre A
gen lofi 11 15 \
  "[verse] coffee in the morning light [chorus] everything is gonna be alright [verse] dancing in the kitchen slow" \
  "warm lofi soul, male vocal, rhodes piano, 85 bpm"

# 3) fresh genre B
gen rock 23 15 \
  "[verse] thunder in the distance calling [chorus] we will rise above it all tonight [bridge] no surrender and no goodbye" \
  "anthemic rock, male vocal, electric guitar and drums, 140 bpm"

# 4) continuation demo (reuse the verified continue.wav)
cp -f "$OUT/continue.wav" "$OUT/night_continue.wav" 2>/dev/null && say "continuation demo -> night_continue.wav (8s, prompt-trimmed)"

# 5) single-file-in clone demo: use a prior generated song as the 'reference mix' -> demucs -> clone
say "GEN clonefile (single-file-in: separate out/q8.wav -> clone) ..."
t5=$(date +%s)
bash "$ROOT/scripts/songgen-clone-file.sh" "$OUT/q8.wav" "$OUT/night_clonefile.wav" \
  "[verse] city lights are fading out [chorus] carry me across the night" 42 >>"$LOG" 2>&1 \
  && say "  clonefile done in $(($(date +%s)-t5))s" || say "  !! clonefile FAILED (continuing)"

say "=== overnight batch DONE ==="
ls -la "$OUT"/night_*.wav 2>/dev/null | awk '{print $NF, $5}' | tee -a "$LOG"
