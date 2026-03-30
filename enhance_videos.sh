#!/bin/bash
# Video & Audio Enhancement Script - RTX 5070 Ti GPU Accelerated
# Phase 1: Quality enhancement (denoise, sharpen, better encoding, audio cleanup)
# Phase 2: Frame interpolation 25fps → 50fps with RIFE

set -e

WORKDIR="/home/stev/Descargas/FilosofiaNeurociencias"
OUTDIR="${WORKDIR}/enhanced"
RIFE_BIN="/tmp/rife-ncnn/rife-ncnn-vulkan-20221029-ubuntu/rife-ncnn-vulkan"
RIFE_MODEL="/tmp/rife-ncnn/rife-ncnn-vulkan-20221029-ubuntu/rife-v4.6"

mkdir -p "$OUTDIR"

# GPU device for NVENC (0 = RTX 5070 Ti)
GPU=0

# ============================================================
# PHASE 1: Video quality + Audio enhancement
# ============================================================
enhance_video() {
    local input="$1"
    local basename=$(basename "$input" .mp4)
    local output="${OUTDIR}/${basename}_enhanced.mp4"

    if [ -f "$output" ]; then
        echo "[SKIP] $output already exists"
        return
    fi

    echo "============================================"
    echo "[PHASE 1] Enhancing: $basename"
    echo "============================================"

    # Video filters:
    #   - hqdn3d: High quality 3D denoising (removes compression artifacts)
    #   - unsharp: Sharpening to recover edge detail
    # Audio filters:
    #   - afftdn: FFT-based noise reduction (nr=20 = moderate noise reduction)
    #   - loudnorm: EBU R128 loudness normalization
    #   - acompressor: Gentle compression for speech clarity
    # Encoding:
    #   - HEVC NVENC on GPU 0 (5070 Ti) at high quality
    #   - CQ 22 (visually lossless for this type of content)
    #   - AAC at 192kbps (up from 75kbps)

    ffmpeg -hwaccel cuda -hwaccel_device $GPU \
        -i "$input" \
        -vf "hqdn3d=4:3:6:4.5,unsharp=5:5:0.8:5:5:0.3" \
        -c:v hevc_nvenc -gpu $GPU \
        -preset p6 -tune hq \
        -rc vbr -cq 22 -b:v 8M -maxrate 12M -bufsize 16M \
        -profile:v main10 \
        -af "afftdn=nf=-20:nt=w:om=o,acompressor=threshold=-20dB:ratio=3:attack=5:release=50,loudnorm=I=-16:TP=-1.5:LRA=11" \
        -c:a aac -b:a 192k -ar 48000 \
        -movflags +faststart \
        -y "$output"

    echo "[DONE] Phase 1 complete: $output"
    echo "Original size: $(du -h "$input" | cut -f1)"
    echo "Enhanced size: $(du -h "$output" | cut -f1)"
}

# ============================================================
# PHASE 1b: Audio-only enhancement (for the .m4a file)
# ============================================================
enhance_audio() {
    local input="$1"
    local basename=$(basename "$input" .m4a)
    local output="${OUTDIR}/${basename}_enhanced.m4a"

    if [ -f "$output" ]; then
        echo "[SKIP] $output already exists"
        return
    fi

    echo "============================================"
    echo "[PHASE 1b] Enhancing audio: $basename"
    echo "============================================"

    ffmpeg -i "$input" \
        -af "afftdn=nf=-20:nt=w:om=o,acompressor=threshold=-20dB:ratio=3:attack=5:release=50,loudnorm=I=-16:TP=-1.5:LRA=11" \
        -c:a aac -b:a 192k -ar 48000 \
        -movflags +faststart \
        -y "$output"

    echo "[DONE] Audio enhancement complete: $output"
}

# ============================================================
# PHASE 2: Frame Interpolation 25fps → 50fps with RIFE
# ============================================================
interpolate_video() {
    local input="$1"
    local basename=$(basename "$input" .mp4)
    local phase1="${OUTDIR}/${basename}_enhanced.mp4"
    local output="${OUTDIR}/${basename}_enhanced_50fps.mp4"

    # Use phase 1 output if available, otherwise original
    if [ -f "$phase1" ]; then
        input="$phase1"
    fi

    if [ -f "$output" ]; then
        echo "[SKIP] $output already exists"
        return
    fi

    echo "============================================"
    echo "[PHASE 2] Frame interpolation: $basename (25→50fps)"
    echo "  This will take several hours for 7h+ videos"
    echo "============================================"

    local TMPDIR=$(mktemp -d)
    local FRAMES_IN="${TMPDIR}/frames_in"
    local FRAMES_OUT="${TMPDIR}/frames_out"
    mkdir -p "$FRAMES_IN" "$FRAMES_OUT"

    # Process in 5-minute chunks to manage disk space
    local duration=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$input" | cut -d. -f1)
    local chunk_duration=300  # 5 minutes
    local chunk=0
    local concat_list="${TMPDIR}/concat.txt"
    > "$concat_list"

    echo "[INFO] Total duration: ${duration}s, processing in ${chunk_duration}s chunks"

    for (( start=0; start<duration; start+=chunk_duration )); do
        chunk=$((chunk + 1))
        local remaining=$((duration - start))
        local this_chunk=$((remaining < chunk_duration ? remaining : chunk_duration))

        echo "[CHUNK $chunk] Processing ${start}s - $((start + this_chunk))s / ${duration}s"

        # Clean frame directories
        rm -f "${FRAMES_IN}"/* "${FRAMES_OUT}"/*

        # Extract frames for this chunk
        ffmpeg -ss "$start" -i "$input" -t "$this_chunk" \
            -vsync 0 -frame_pts 1 \
            "${FRAMES_IN}/%08d.png" -y -loglevel warning 2>&1

        # Run RIFE interpolation (doubles frame count)
        "$RIFE_BIN" -i "$FRAMES_IN" -o "$FRAMES_OUT" \
            -m "$RIFE_MODEL" -g $GPU -j 4:4:4 -f "%08d.png" 2>&1

        # Encode interpolated frames for this chunk
        local chunk_file="${TMPDIR}/chunk_$(printf '%04d' $chunk).mp4"
        ffmpeg -framerate 50 -i "${FRAMES_OUT}/%08d.png" \
            -c:v hevc_nvenc -gpu $GPU \
            -preset p6 -tune hq \
            -rc vbr -cq 22 -b:v 8M -maxrate 12M -bufsize 16M \
            -profile:v main10 -pix_fmt yuv420p10le \
            -y "$chunk_file" -loglevel warning 2>&1

        echo "file '$chunk_file'" >> "$concat_list"

        # Clean up frames to save disk space
        rm -f "${FRAMES_IN}"/* "${FRAMES_OUT}"/*
    done

    # Concatenate all chunks + add enhanced audio
    echo "[MERGE] Concatenating chunks and adding audio..."
    ffmpeg -f concat -safe 0 -i "$concat_list" \
        -i "$input" \
        -map 0:v -map 1:a \
        -c:v copy -c:a copy \
        -movflags +faststart \
        -y "$output"

    # Cleanup
    rm -rf "$TMPDIR"

    echo "[DONE] Phase 2 complete: $output"
    echo "Enhanced size: $(du -h "$output" | cut -f1)"
}

# ============================================================
# MAIN
# ============================================================
echo "======================================================="
echo "  VIDEO ENHANCEMENT - RTX 5070 Ti GPU Accelerated"
echo "======================================================="
echo ""

# Phase 1: Enhance all files
for f in "$WORKDIR"/*.mp4; do
    [ -f "$f" ] && enhance_video "$f"
done

for f in "$WORKDIR"/*.m4a; do
    [ -f "$f" ] && enhance_audio "$f"
done

echo ""
echo "======================================================="
echo "  PHASE 1 COMPLETE - Enhanced files in: $OUTDIR"
echo "======================================================="
echo ""

# Ask before Phase 2 (very slow)
read -p "Start Phase 2 (Frame Interpolation 25→50fps)? This takes several hours. [y/N]: " answer
if [[ "$answer" =~ ^[Yy] ]]; then
    for f in "$WORKDIR"/*.mp4; do
        [ -f "$f" ] && interpolate_video "$f"
    done
    echo ""
    echo "======================================================="
    echo "  PHASE 2 COMPLETE - 50fps files in: $OUTDIR"
    echo "======================================================="
fi

echo ""
echo "All done! Enhanced files are in: $OUTDIR"
