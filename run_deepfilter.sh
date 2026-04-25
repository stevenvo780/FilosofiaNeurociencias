#!/usr/bin/env bash
# Process audio chunks through DeepFilterNet in sequence
# Limited to 70% CPU cores to leave room for Video2X GPU workers
set -euo pipefail

AUDIO_DIR="/datos/Neuro/audio_chunks"
DF_OUT="/datos/Neuro/audio_df_chunks"
FINAL_DF="/datos/Neuro/audio_df_enhanced.wav"
CORES=22  # 70% of 32

DEEPFILTER="/home/stev/.local/share/pipx/venvs/deepfilternet/bin/deepFilter"

mkdir -p "$DF_OUT"

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "DeepFilterNet — processing $(ls "$AUDIO_DIR"/audio_*.wav | wc -l) chunks with $CORES threads"

export OMP_NUM_THREADS=$CORES
export MKL_NUM_THREADS=$CORES
export NUMEXPR_NUM_THREADS=$CORES
export OPENBLAS_NUM_THREADS=$CORES
export CUDA_VISIBLE_DEVICES=""
export DF_DEVICE="cpu"

for wav in "$AUDIO_DIR"/audio_*.wav; do
    name="$(basename "$wav")"
    out="$DF_OUT/$name"
    if [[ -f "$out" && "$(stat -c%s "$out" 2>/dev/null || echo 0)" -gt 1000 ]]; then
        log "  ✓ $name already processed, skipping"
        continue
    fi
    log "  → Processing $name ($(du -h "$wav" | cut -f1))..."
    "$DEEPFILTER" --pf --no-suffix -o "$DF_OUT" "$wav" 2>&1 | tail -3
    log "  ✓ $name done"
done

# Concatenate DF chunks back into one WAV
log "Concatenating DF chunks..."
CONCAT="/datos/Neuro/audio_df_concat.txt"
> "$CONCAT"
for f in $(ls "$DF_OUT"/audio_*.wav | sort); do
    echo "file '$f'" >> "$CONCAT"
done

ffmpeg -y -f concat -safe 0 -i "$CONCAT" -c:a copy "$FINAL_DF" -loglevel warning
rm -f "$CONCAT"

log "✓ DeepFilterNet done: $FINAL_DF ($(du -h "$FINAL_DF" | cut -f1))"
