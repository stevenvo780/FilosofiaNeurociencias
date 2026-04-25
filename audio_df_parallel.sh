#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  audio_df_parallel.sh — DeepFilterNet audio enhancement via chunked parallel
#
#  Splits audio into small segments, processes each with DeepFilterNet
#  in parallel (avoiding the RAM explosion of processing the full file),
#  then concatenates the results.
#
#  Usage: ./audio_df_parallel.sh
# ──────────────────────────────────────────────────────────────
set -euo pipefail

# ── Config ───────────────────────────────────────────────────
INPUT="/home/stev/Descargas/FilosofiaNeurociencias/videos/GMT20260320-130023_Recording_2240x1260.mp4"
OUTDIR="/datos/Neuro"
WORKDIR="$OUTDIR/audio_work"
CHUNK_SEC=300          # 5 min per chunk
PARALLEL_JOBS=16       # ~50% of 32 cores, leaves room for Video2X CPU work
DF_BIN="$HOME/.local/bin/deepFilter"

AUDIO_WAV="$WORKDIR/audio_full.wav"
CHUNKS_DIR="$WORKDIR/chunks"
DF_DIR="$WORKDIR/df_out"
FINAL_WAV="$OUTDIR/audio_df_enhanced.wav"

mkdir -p "$CHUNKS_DIR" "$DF_DIR"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ── Step 1: Extract audio ────────────────────────────────────
log "=== Step 1: Extract audio as WAV ==="
if [[ ! -f "$AUDIO_WAV" ]]; then
    ffmpeg -y -i "$INPUT" -vn -c:a pcm_s16le -ar 48000 -ac 2 "$AUDIO_WAV" -loglevel warning
    log "Extracted: $(du -h "$AUDIO_WAV" | cut -f1)"
else
    log "WAV exists ($(du -h "$AUDIO_WAV" | cut -f1)), skipping"
fi

# ── Step 2: Split into chunks ────────────────────────────────
log "=== Step 2: Split into ${CHUNK_SEC}s chunks ==="
EXISTING=$(find "$CHUNKS_DIR" -name 'seg_*.wav' 2>/dev/null | wc -l)
if [[ "$EXISTING" -gt 0 ]]; then
    log "Chunks already split ($EXISTING found), skipping"
else
    ffmpeg -y -i "$AUDIO_WAV" \
        -f segment -segment_time "$CHUNK_SEC" \
        -c:a pcm_s16le -ar 48000 -ac 2 \
        "$CHUNKS_DIR/seg_%04d.wav" \
        -loglevel warning
    EXISTING=$(find "$CHUNKS_DIR" -name 'seg_*.wav' | wc -l)
    log "Split into $EXISTING segments"
fi

# ── Step 3: Process with DeepFilterNet in parallel ───────────
log "=== Step 3: DeepFilterNet × $PARALLEL_JOBS parallel ==="

process_one() {
    local seg="$1"
    local df_bin="$2"
    local outdir="$3"
    local name
    name="$(basename "$seg")"
    local outfile="$outdir/$name"

    if [[ -f "$outfile" && "$(stat -c%s "$outfile" 2>/dev/null || echo 0)" -gt 1000 ]]; then
        return 0
    fi

    # Force CPU-only to avoid fighting Video2X for GPU
    CUDA_VISIBLE_DEVICES="" "$df_bin" --pf -o "$outdir" "$seg" 2>/dev/null
}
export -f process_one

PENDING=$(find "$CHUNKS_DIR" -name 'seg_*.wav' -print | sort | while read seg; do
    name="$(basename "$seg")"
    outfile="$DF_DIR/$name"
    if [[ ! -f "$outfile" || "$(stat -c%s "$outfile" 2>/dev/null || echo 0)" -le 1000 ]]; then
        echo "$seg"
    fi
done | wc -l)

log "$PENDING segments pending, launching $PARALLEL_JOBS parallel workers..."

find "$CHUNKS_DIR" -name 'seg_*.wav' -print | sort | while read seg; do
    name="$(basename "$seg")"
    outfile="$DF_DIR/$name"
    if [[ ! -f "$outfile" || "$(stat -c%s "$outfile" 2>/dev/null || echo 0)" -le 1000 ]]; then
        echo "$seg"
    fi
done | parallel -j "$PARALLEL_JOBS" --bar process_one {} "$DF_BIN" "$DF_DIR"

DONE=$(find "$DF_DIR" -name 'seg_*.wav' -size +1k | wc -l)
log "DeepFilterNet done: $DONE/$EXISTING segments processed"

# ── Step 4: Concatenate enhanced segments ────────────────────
log "=== Step 4: Concatenate ==="
CONCAT="$WORKDIR/concat.txt"
> "$CONCAT"
for f in $(ls "$DF_DIR"/seg_*.wav 2>/dev/null | sort); do
    echo "file '$f'" >> "$CONCAT"
done

ffmpeg -y -f concat -safe 0 -i "$CONCAT" \
    -c:a pcm_s16le -ar 48000 -ac 2 \
    "$FINAL_WAV" \
    -loglevel warning

log "=== DONE: $FINAL_WAV ($(du -h "$FINAL_WAV" | cut -f1)) ==="
log "Next: mux into final video with:"
log "  ffmpeg -i VIDEO.mkv -i $FINAL_WAV -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k OUTPUT.mkv"
