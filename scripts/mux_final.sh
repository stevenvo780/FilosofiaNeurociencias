#!/bin/bash
# Mux final: creates two videos
# 1) Original audio
# 2) Resemble-enhanced audio
set -euo pipefail

VIDEO="/datos/Neuro/GMT20260320-130023_Recording_2240x1260_4K50.mkv"
ENHANCED_AUDIO="/datos/Neuro/audio_resemble/enhanced_full.wav"
OUT_ORIG="/datos/Neuro/GMT20260320-130023_Recording_4K50_original_audio.mkv"
OUT_ENHANCED="/datos/Neuro/GMT20260320-130023_Recording_4K50_enhanced_audio.mkv"

echo "=== Muxing video with original audio ==="
# The existing _4K50.mkv already has original audio, just copy/rename
if [ ! -f "$OUT_ORIG" ]; then
    cp -v "$VIDEO" "$OUT_ORIG"
fi

echo "=== Muxing video with enhanced audio ==="
ffmpeg -y -i "$VIDEO" -i "$ENHANCED_AUDIO" \
    -map 0:v -map 1:a \
    -c:v copy -c:a aac -b:a 256k \
    -shortest \
    "$OUT_ENHANCED"

echo "=== Done ==="
echo "Original audio: $OUT_ORIG"
echo "Enhanced audio: $OUT_ENHANCED"

# Verify
for f in "$OUT_ORIG" "$OUT_ENHANCED"; do
    echo ""
    echo "--- $f ---"
    ffprobe -hide_banner -i "$f" 2>&1 | grep -E "Duration|Stream"
    ls -lh "$f"
done
