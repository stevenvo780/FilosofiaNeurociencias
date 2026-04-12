#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  enhance.sh — Video enhancement pipeline
#
#  Uses Video2X for upscale (Real-ESRGAN) + frame interpolation (RIFE)
#  and ffmpeg for audio enhancement.
#
#  Video2X runs ONE processor per invocation, so upscale and interpolation
#  are two separate passes per chunk:
#    1) video2x -p realesrgan -s N   (upscale)
#    2) video2x -p rife -m N         (frame interpolation)
#
#  Result: resolution ×2, FPS ×2, cleaned audio.
#
#  Usage:
#    ./enhance.sh input.mp4
#    ./enhance.sh input.mp4 output.mp4
#    ./enhance.sh input.mp4 output.mp4 audio_sidecar.m4a
# ──────────────────────────────────────────────────────────────
set -euo pipefail

# ── Config (override via env) ────────────────────────────────
V2X_BIN="${V2X_BIN:-video2x}"                    # video2x binary or AppImage path
V2X_UPSCALE_FACTOR="${V2X_UPSCALE_FACTOR:-2}"    # resolution multiplier
V2X_UPSCALE_MODEL="${V2X_UPSCALE_MODEL:-realesr-animevideov3}"  # --realesrgan-model
V2X_INTERP_FACTOR="${V2X_INTERP_FACTOR:-2}"      # FPS multiplier (frame rate mul)
V2X_INTERP_MODEL="${V2X_INTERP_MODEL:-rife-v4.6}"              # --rife-model
V2X_GPU="${V2X_GPU:-0}"                           # Vulkan device index
V2X_GPU_WORKERS="${V2X_GPU_WORKERS:-4}"           # parallel chunk workers
V2X_EXTRA_ARGS="${V2X_EXTRA_ARGS:-}"             # any extra video2x args

AUDIO_FILTER="${AUDIO_FILTER:-highpass=f=80,anlmdn=s=7:p=0.002:m=15,loudnorm=I=-16:TP=-1.5:LRA=11,alimiter=level_in=1:level_out=1:limit=0.95:release=50}"
AUDIO_CODEC="${AUDIO_CODEC:-aac}"
AUDIO_BITRATE="${AUDIO_BITRATE:-256k}"
AUDIO_SAMPLE_RATE="${AUDIO_SAMPLE_RATE:-48000}"

CHUNK_MINUTES="${CHUNK_MINUTES:-15}"
WORKDIR="${WORKDIR:-}"                            # default: next to input

# ── Args ─────────────────────────────────────────────────────
INPUT="${1:?Usage: $0 input.mp4 [output.mp4] [audio_sidecar.m4a]}"
INPUT="$(realpath "$INPUT")"
BASENAME="$(basename "${INPUT%.*}")"

OUTPUT="${2:-$(dirname "$INPUT")/${BASENAME}_enhanced.mp4}"
OUTPUT="$(realpath "$OUTPUT")"

AUDIO_SIDECAR="${3:-}"

# ── Derived paths ────────────────────────────────────────────
WORK="${WORKDIR:-$(dirname "$OUTPUT")/work_${BASENAME}}"
CHUNKS_DIR="$WORK/chunks"
UPSCALED_DIR="$WORK/upscaled"
INTERP_DIR="$WORK/interpolated"
AUDIO_OUT="$WORK/${BASENAME}_audio_enhanced.m4a"
mkdir -p "$CHUNKS_DIR" "$UPSCALED_DIR" "$INTERP_DIR"

# ── Helpers ──────────────────────────────────────────────────
log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { log "ERROR: $*" >&2; exit 1; }

check_deps() {
    command -v ffmpeg  >/dev/null || die "ffmpeg not found"
    command -v ffprobe >/dev/null || die "ffprobe not found"
    if ! command -v "$V2X_BIN" >/dev/null 2>&1 && [[ ! -x "$V2X_BIN" ]]; then
        die "video2x not found. Install it or set V2X_BIN=/path/to/Video2X-x86_64.AppImage"
    fi
}

get_duration() {
    ffprobe -v error -show_entries format=duration -of csv=p=0 "$1" | cut -d. -f1
}

get_video_info() {
    ffprobe -v error -select_streams v:0 \
        -show_entries stream=width,height,r_frame_rate \
        -of csv=p=0 "$1"
}

find_audio_source() {
    # Priority: explicit sidecar > .m4a next to video > embedded audio
    if [[ -n "$AUDIO_SIDECAR" && -f "$AUDIO_SIDECAR" ]]; then
        echo "$AUDIO_SIDECAR"; return
    fi
    local dir stem m4a
    dir="$(dirname "$INPUT")"
    stem="$(basename "${INPUT%.*}")"
    # Try exact stem match (e.g. Recording.m4a for Recording_2240x1260.mp4)
    for m4a in "$dir/${stem}.m4a" "$dir/$(echo "$stem" | sed 's/_[0-9]*x[0-9]*$//').m4a"; do
        [[ -f "$m4a" ]] && { echo "$m4a"; return; }
    done
    # Fall back to embedded audio
    if ffprobe -v error -select_streams a -show_entries stream=index -of csv=p=0 "$INPUT" | grep -q .; then
        echo "$INPUT"; return
    fi
    echo ""
}

# ── Step 1: Split into chunks ────────────────────────────────
split_chunks() {
    local dur chunk_s n_chunks
    dur="$(get_duration "$INPUT")"
    chunk_s=$((CHUNK_MINUTES * 60))
    n_chunks=$(( (dur + chunk_s - 1) / chunk_s ))

    log "Splitting $INPUT ($((dur/3600))h$((dur%3600/60))m) into $n_chunks chunks of ${CHUNK_MINUTES}min"

    local existing
    existing=$(find "$CHUNKS_DIR" -name 'chunk_*.mp4' 2>/dev/null | wc -l)
    if [[ "$existing" -ge "$n_chunks" ]]; then
        log "Chunks already split ($existing found), skipping"
        return
    fi

    ffmpeg -y -i "$INPUT" \
        -c copy -f segment \
        -segment_time "$chunk_s" \
        -reset_timestamps 1 \
        "$CHUNKS_DIR/chunk_%04d.mp4" \
        -loglevel warning

    log "Split done: $(ls "$CHUNKS_DIR"/chunk_*.mp4 | wc -l) chunks"
}

# ── Step 2: Upscale + interpolate via Video2X ────────────────
process_chunk() {
    local chunk="$1"
    local name out_up out_interp done_flag
    name="$(basename "${chunk%.*}")"
    out_up="$UPSCALED_DIR/${name}_upscaled.mp4"
    out_interp="$INTERP_DIR/${name}_enhanced.mp4"
    done_flag="$INTERP_DIR/${name}.done"

    if [[ -f "$done_flag" ]]; then
        return 0
    fi

    log "  Processing $name ..."

    # Pass 1: Upscale with Real-ESRGAN
    if [[ ! -f "$out_up" || "$(stat -c%s "$out_up" 2>/dev/null || echo 0)" -le 1000 ]]; then
        log "    ↑ Upscaling ×${V2X_UPSCALE_FACTOR} (${V2X_UPSCALE_MODEL})"
        "$V2X_BIN" -i "$chunk" -o "$out_up" \
            -p realesrgan \
            -s "${V2X_UPSCALE_FACTOR}" \
            --realesrgan-model "${V2X_UPSCALE_MODEL}" \
            -d "${V2X_GPU}" \
            $V2X_EXTRA_ARGS \
            2>&1 | tail -1
    fi

    if [[ ! -f "$out_up" || "$(stat -c%s "$out_up" 2>/dev/null || echo 0)" -le 1000 ]]; then
        log "  ✗ $name upscale FAILED"
        return 1
    fi

    # Pass 2: Frame interpolation with RIFE (skip if factor is 1)
    if [[ "$V2X_INTERP_FACTOR" -le 1 ]]; then
        cp "$out_up" "$out_interp"
    else
        log "    ↗ Interpolating ×${V2X_INTERP_FACTOR} FPS (${V2X_INTERP_MODEL})"
        "$V2X_BIN" -i "$out_up" -o "$out_interp" \
            -p rife \
            -m "${V2X_INTERP_FACTOR}" \
            --rife-model "${V2X_INTERP_MODEL}" \
            -d "${V2X_GPU}" \
            $V2X_EXTRA_ARGS \
            2>&1 | tail -1
    fi

    if [[ -f "$out_interp" && "$(stat -c%s "$out_interp" 2>/dev/null || echo 0)" -gt 1000 ]]; then
        touch "$done_flag"
        # Clean up intermediate upscale file to save disk
        rm -f "$out_up"
        log "  ✓ $name done"
    else
        log "  ✗ $name interpolation FAILED"
        return 1
    fi
}

run_video2x() {
    local chunks=("$CHUNKS_DIR"/chunk_*.mp4)
    local total=${#chunks[@]}
    local done_count
    done_count=$(find "$INTERP_DIR" -name '*.done' 2>/dev/null | wc -l)

    log "Video2X: $total chunks, $done_count already done, $((total - done_count)) remaining"
    log "Workers: $V2X_GPU_WORKERS parallel | GPU: $V2X_GPU"

    local pids=() running=0 fails=0
    for chunk in "${chunks[@]}"; do
        local name done_flag
        name="$(basename "${chunk%.*}")"
        done_flag="$INTERP_DIR/${name}.done"
        [[ -f "$done_flag" ]] && continue

        # Throttle: wait if max workers reached
        while (( running >= V2X_GPU_WORKERS )); do
            wait -n -p wpid 2>/dev/null || true
            running=$((running - 1))
        done

        process_chunk "$chunk" &
        pids+=($!)
        running=$((running + 1))
    done

    # Wait for all remaining
    for pid in "${pids[@]}"; do
        wait "$pid" || fails=$((fails + 1))
    done

    done_count=$(find "$INTERP_DIR" -name '*.done' 2>/dev/null | wc -l)
    log "Video2X complete: $done_count/$total done, $fails fails"
    [[ "$fails" -eq 0 ]] || die "$fails chunks failed"
}

# ── Step 3: Audio enhancement ────────────────────────────────
enhance_audio() {
    local audio_src
    audio_src="$(find_audio_source)"

    if [[ -z "$audio_src" ]]; then
        log "No audio source found, skipping audio"
        AUDIO_OUT=""
        return
    fi

    if [[ -f "$AUDIO_OUT" && "$(stat -c%s "$AUDIO_OUT" 2>/dev/null || echo 0)" -gt 1024 ]]; then
        log "Audio already enhanced, skipping"
        return
    fi

    log "Enhancing audio from $(basename "$audio_src")"
    ffmpeg -y -i "$audio_src" \
        -vn -map 0:a:0 \
        -af "$AUDIO_FILTER" \
        -c:a "$AUDIO_CODEC" \
        -b:a "$AUDIO_BITRATE" \
        -ar "$AUDIO_SAMPLE_RATE" \
        -movflags +faststart \
        "$AUDIO_OUT" \
        -loglevel warning

    log "Audio done: $(du -h "$AUDIO_OUT" | cut -f1)"
}

# ── Step 4: Concatenate + mux ────────────────────────────────
concat_and_mux() {
    local concat_file="$WORK/concat.txt"

    # Build concat list (sorted)
    > "$concat_file"
    for f in $(ls "$INTERP_DIR"/chunk_*_enhanced.mp4 2>/dev/null | sort); do
        echo "file '$f'" >> "$concat_file"
    done

    local n_files
    n_files=$(wc -l < "$concat_file")
    [[ "$n_files" -gt 0 ]] || die "No enhanced chunks found in $INTERP_DIR"

    log "Concatenating $n_files chunks + muxing audio → $OUTPUT"

    local cmd=(ffmpeg -y -f concat -safe 0 -i "$concat_file")
    if [[ -n "$AUDIO_OUT" && -f "$AUDIO_OUT" ]]; then
        cmd+=(-i "$AUDIO_OUT" -map 0:v -map 1:a:0 -shortest -c:a copy)
    else
        cmd+=(-map 0:v)
    fi
    cmd+=(-c:v copy -movflags +faststart "$OUTPUT" -loglevel warning)

    "${cmd[@]}"

    local size
    size="$(du -h "$OUTPUT" | cut -f1)"
    log "✓ DONE: $OUTPUT ($size)"
}

# ── Main ─────────────────────────────────────────────────────
main() {
    check_deps

    local info dur
    info="$(get_video_info "$INPUT")"
    dur="$(get_duration "$INPUT")"

    log "═══════════════════════════════════════════"
    log "  VIDEO ENHANCEMENT"
    log "═══════════════════════════════════════════"
    log "  Input:   $(basename "$INPUT")"
    log "  Info:    ${info} | ${dur}s ($((dur/3600))h$((dur%3600/60))m)"
    log "  Output:  $(basename "$OUTPUT")"
    log "  Upscale: ${V2X_UPSCALE_MODEL} ×${V2X_UPSCALE_FACTOR}"
    log "  Interp:  ${V2X_INTERP_MODEL} ×${V2X_INTERP_FACTOR}"
    log "  Workers: ${V2X_GPU_WORKERS}"
    log "  Audio:   ${AUDIO_CODEC} ${AUDIO_BITRATE}"
    log "═══════════════════════════════════════════"

    split_chunks
    enhance_audio &
    local audio_pid=$!

    run_video2x

    wait "$audio_pid" 2>/dev/null || log "Audio process finished"

    concat_and_mux

    log "═══════════════════════════════════════════"
    log "  ✓ ALL DONE"
    log "═══════════════════════════════════════════"
}

main
