#!/usr/bin/env bash
# Single-file-in clone: a stereo mix + lyric -> cloned song, in the reference's style/timbre.
# Front-end = Meta demucs (isolated venv) splits the mix into vocal/bgm stems; the core
# SongGeneration inference (separation aside) stays pure C++/ggml via songgen-clone.
#
# Usage: songgen-clone-file.sh <mix.wav> <out.wav> "<lyric>" [seed]
set -euo pipefail
ROOT=/home/dbrain/dev/songgen-port
GGUF=$ROOT/gguf
MIX=${1:?mix.wav}; OUT=${2:?out.wav}; LYRIC=${3:?lyric}; SEED=${4:-42}
TMP=$(mktemp -d)
VOC=$TMP/vocal.wav; BGM=$TMP/bgm.wav
trap 'rm -rf "$TMP"' EXIT

echo "[1/2] separating $MIX (demucs htdemucs)..."
/tmp/demucs-venv/bin/python "$ROOT/scripts/demucs_separate.py" "$MIX" "$VOC" "$BGM"

echo "[2/2] cloning -> $OUT ..."
GGML_BACKEND=CPU "$ROOT/songgen.cpp/build/songgen-clone" \
  "$GGUF/songgen-lelm-large-Q8_0.gguf" "$GGUF/songgen-cfm.gguf" "$GGUF/songgen-vae.gguf" \
  "$GGUF/songgen-septoken-aux.gguf" "$GGUF/songgen-1rvq-aux.gguf" \
  "$GGUF/songgen-musicfm.gguf" "$GGUF/songgen-musicfm-1rvq.gguf" \
  "$OUT" "$SEED" --vocal-stem "$VOC" --bgm-stem "$BGM" --full-mix "$MIX" --lyric "$LYRIC"
echo "done -> $OUT"
