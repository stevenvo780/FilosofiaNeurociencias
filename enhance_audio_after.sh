#!/usr/bin/env bash
# Wait for enhance.sh to finish, then run DeepFilterNet on the audio
set -euo pipefail

INPUT="videos/GMT20260320-130023_Recording_2240x1260.mp4"
AUDIO_WAV="/datos/Neuro/audio_original.wav"
AUDIO_DF="/datos/Neuro/audio_df_enhanced.wav"
FINAL_OUTPUT="/datos/Neuro/GMT20260320-130023_Recording_2240x1260_4K50.mkv"
FINAL_DF="/datos/Neuro/GMT20260320-130023_Recording_2240x1260_4K50_DFaudio.mkv"

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "Waiting for enhance.sh to finish (monitoring $FINAL_OUTPUT)..."

# Wait for the main enhance.sh process to exit
while pgrep -f "enhance.sh.*Recording" >/dev/null 2>&1; do
    sleep 30
done

log "enhance.sh finished."

# Check output exists
if [[ ! -f "$FINAL_OUTPUT" ]]; then
    log "ERROR: $FINAL_OUTPUT not found. enhance.sh may have failed."
    exit 1
fi

log "=== Step 1: Extract audio as WAV ==="
if [[ ! -f "$AUDIO_WAV" ]]; then
    ffmpeg -y -i "$INPUT" -vn -acodec pcm_s16le -ar 48000 -ac 2 "$AUDIO_WAV" -loglevel warning
    log "Extracted $(du -h "$AUDIO_WAV" | cut -f1)"
else
    log "WAV already exists, skipping"
fi

log "=== Step 2: DeepFilterNet enhancement ==="
if [[ ! -f "$AUDIO_DF" ]]; then
    deep-filter --post-filter -o "$(dirname "$AUDIO_DF")" "$AUDIO_WAV"
    # DF outputs with same name in output dir
    df_out="$(dirname "$AUDIO_DF")/$(basename "$AUDIO_WAV")"
    if [[ -f "$df_out" && "$df_out" != "$AUDIO_DF" ]]; then
        mv "$df_out" "$AUDIO_DF"
    fi
    log "DeepFilterNet done: $(du -h "$AUDIO_DF" | cut -f1)"
else
    log "DF audio already exists, skipping"
fi

log "=== Step 3: Mux DF audio into final video ==="
ffmpeg -y -i "$FINAL_OUTPUT" -i "$AUDIO_DF" \
    -map 0:v:0 -map 1:a:0 \
    -c:v copy -c:a aac -b:a 192k \
    "$FINAL_DF" \
    -loglevel warning

log "✓ Final output with DF audio: $FINAL_DF ($(du -h "$FINAL_DF" | cut -f1))"
