#!/usr/bin/env bash
# Single-file-in clone, FULLY PURE C++/ggml (native HTDemucs separator — no python/demucs venv).
# Usage: songgen-clone-native.sh <mix.wav> <out.wav> "<lyric>" [seed]
# (For the marginally-higher-SDR demucs shifts=1 path, use songgen-clone-file.sh instead.)
set -euo pipefail
ROOT=/home/dbrain/dev/songgen-port
G=$ROOT/gguf; B=$ROOT/songgen.cpp/build
MIX=${1:?mix.wav}; OUT=${2:?out.wav}; LYRIC=${3:?lyric}; SEED=${4:-42}
TMP=$(mktemp -d); VOC=$TMP/vocal.wav; BGM=$TMP/bgm.wav
trap 'rm -rf "$TMP"' EXIT

echo "[1/2] separating $MIX (native ggml HTDemucs)..."
GGML_BACKEND=CPU "$B/songgen-separate" "$G/songgen-htdemucs.gguf" "$MIX" "$VOC" "$BGM"

echo "[2/2] cloning -> $OUT ..."
GGML_BACKEND=CPU "$B/songgen-clone" \
  "$G/songgen-lelm-large-Q8_0.gguf" "$G/songgen-cfm.gguf" "$G/songgen-vae.gguf" \
  "$G/songgen-septoken-aux.gguf" "$G/songgen-1rvq-aux.gguf" \
  "$G/songgen-musicfm.gguf" "$G/songgen-musicfm-1rvq.gguf" \
  "$OUT" "$SEED" --vocal-stem "$VOC" --bgm-stem "$BGM" --full-mix "$MIX" --lyric "$LYRIC"
echo "done -> $OUT"
