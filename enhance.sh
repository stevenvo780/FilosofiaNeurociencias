#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  enhance.sh — Dual-GPU video enhancement pipeline
#
#  Uses Video2X for upscale (Real-ESRGAN) + frame interpolation (RIFE)
#  Distributes chunks across two GPUs with proportional workers.
#
#  Video2X runs ONE processor per invocation:
#    1) video2x -p realesrgan -s N   (upscale)
#    2) video2x -p rife -m N         (frame interpolation)
#
#  Result: resolution ×2, FPS ×2, original audio preserved.
#
#  Usage:
#    ./enhance.sh input.mp4
#    ./enhance.sh input.mp4 output.mkv
#    ./enhance.sh input.mp4 output.mkv audio_sidecar.m4a
# ──────────────────────────────────────────────────────────────
set -euo pipefail

# ── Config (override via env) ────────────────────────────────
V2X_BIN="${V2X_BIN:-$(dirname "$(realpath "$0")")/Video2X-x86_64.AppImage}"
V2X_UPSCALE_FACTOR="${V2X_UPSCALE_FACTOR:-2}"
V2X_UPSCALE_MODEL="${V2X_UPSCALE_MODEL:-realesr-animevideov3}"
V2X_INTERP_FACTOR="${V2X_INTERP_FACTOR:-2}"
V2X_INTERP_MODEL="${V2X_INTERP_MODEL:-rife-v4.6}"
V2X_EXTRA_ARGS="${V2X_EXTRA_ARGS:-}"

# Dual GPU config — workers per GPU
# GPU0: RTX 5070 Ti (fast), GPU1: RTX 2060 (slower)
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU0_WORKERS="${GPU0_WORKERS:-3}"
GPU1_WORKERS="${GPU1_WORKERS:-1}"

CHUNK_MINUTES="${CHUNK_MINUTES:-15}"
WORKDIR="${WORKDIR:-}"

# ── Args ─────────────────────────────────────────────────────
INPUT="${1:?Usage: $0 input.mp4 [output.mkv]}"
INPUT="$(realpath "$INPUT")"
BASENAME="$(basename "${INPUT%.*}")"

OUTPUT="${2:-$(dirname "$INPUT")/${BASENAME}_enhanced.mkv}"
OUTPUT="$(realpath "$OUTPUT")"

# ── Derived paths ────────────────────────────────────────────
WORK="${WORKDIR:-$(dirname "$OUTPUT")/work_${BASENAME}}"
CHUNKS_DIR="$WORK/chunks"
UPSCALED_DIR="$WORK/upscaled"
INTERP_DIR="$WORK/interpolated"
mkdir -p "$CHUNKS_DIR" "$UPSCALED_DIR" "$INTERP_DIR"

# ── Helpers ──────────────────────────────────────────────────
log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { log "ERROR: $*" >&2; exit 1; }

check_deps() {
    command -v ffmpeg  >/dev/null || die "ffmpeg not found"
    command -v ffprobe >/dev/null || die "ffprobe not found"
    [[ -x "$V2X_BIN" ]] || die "Video2X not found at $V2X_BIN"
}

get_duration() {
    ffprobe -v error -show_entries format=duration -of csv=p=0 "$1" | cut -d. -f1
}

get_video_info() {
    ffprobe -v error -select_streams v:0 \
        -show_entries stream=width,height,r_frame_rate \
        -of csv=p=0 "$1"
}

# ── Step 1: Split into chunks ────────────────────────────────
split_chunks() {
    local dur chunk_s n_chunks
    dur="$(get_duration "$INPUT")"
    chunk_s=$((CHUNK_MINUTES * 60))
    n_chunks=$(( (dur + chunk_s - 1) / chunk_s ))

    log "Splitting $INPUT ($((dur/3600))h$((dur%3600/60))m) into ~$n_chunks chunks of ${CHUNK_MINUTES}min"

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

# ── Step 2: Process a single chunk (upscale + interpolate) ───
process_chunk() {
    local chunk="$1"
    local gpu="$2"
    local name out_up out_interp done_flag
    name="$(basename "${chunk%.*}")"
    out_up="$UPSCALED_DIR/${name}_upscaled.mp4"
    out_interp="$INTERP_DIR/${name}_enhanced.mp4"
    done_flag="$INTERP_DIR/${name}.done"

    [[ -f "$done_flag" ]] && return 0

    log "  [GPU$gpu] Processing $name ..."

    # Pass 1: Upscale with Real-ESRGAN
    if [[ ! -f "$out_up" || "$(stat -c%s "$out_up" 2>/dev/null || echo 0)" -le 1000 ]]; then
        log "    [GPU$gpu] ↑ Upscaling ×${V2X_UPSCALE_FACTOR} ($name)"
        "$V2X_BIN" -i "$chunk" -o "$out_up" \
            -p realesrgan \
            -s "${V2X_UPSCALE_FACTOR}" \
            --realesrgan-model "${V2X_UPSCALE_MODEL}" \
            -d "${gpu}" \
            $V2X_EXTRA_ARGS \
            2>&1 | tail -1 || true
    fi

    if [[ ! -f "$out_up" || "$(stat -c%s "$out_up" 2>/dev/null || echo 0)" -le 1000 ]]; then
        log "  ✗ [GPU$gpu] $name upscale FAILED"
        return 1
    fi

    # Pass 2: Frame interpolation with RIFE
    if [[ "$V2X_INTERP_FACTOR" -le 1 ]]; then
        cp "$out_up" "$out_interp"
    else
        log "    [GPU$gpu] ↗ Interpolating ×${V2X_INTERP_FACTOR} FPS ($name)"
        "$V2X_BIN" -i "$out_up" -o "$out_interp" \
            -p rife \
            -m "${V2X_INTERP_FACTOR}" \
            --rife-model "${V2X_INTERP_MODEL}" \
            -d "${gpu}" \
            $V2X_EXTRA_ARGS \
            2>&1 | tail -1 || true
    fi

    if [[ -f "$out_interp" && "$(stat -c%s "$out_interp" 2>/dev/null || echo 0)" -gt 1000 ]]; then
        touch "$done_flag"
        rm -f "$out_up"
        log "  ✓ [GPU$gpu] $name done"
    else
        log "  ✗ [GPU$gpu] $name interpolation FAILED"
        return 1
    fi
}

# ── Step 3: Dual-GPU worker dispatcher ───────────────────────
run_video2x() {
    local chunks=("$CHUNKS_DIR"/chunk_*.mp4)
    local total=${#chunks[@]}
    local done_count
    done_count=$(find "$INTERP_DIR" -name '*.done' 2>/dev/null | wc -l)
    local remaining=$((total - done_count))

    log "Video2X: $total chunks, $done_count done, $remaining remaining"
    log "GPU0 (device $GPU0): $GPU0_WORKERS workers"
    log "GPU1 (device $GPU1): $GPU1_WORKERS workers"

    if [[ "$remaining" -le 0 ]]; then
        log "All chunks already processed, skipping"
        return
    fi

    # Build list of pending chunks
    local pending=()
    for chunk in "${chunks[@]}"; do
        local name done_flag
        name="$(basename "${chunk%.*}")"
        done_flag="$INTERP_DIR/${name}.done"
        [[ -f "$done_flag" ]] || pending+=("$chunk")
    done

    # Semaphore-style dispatch with per-GPU slot tracking
    local gpu0_running=0 gpu1_running=0
    local -a pids_gpu0=() pids_gpu1=()
    local fails=0
    local dispatched=0

    for chunk in "${pending[@]}"; do
        # Wait for a free slot on either GPU
        while (( gpu0_running >= GPU0_WORKERS && gpu1_running >= GPU1_WORKERS )); do
            sleep 2
            local new0=0 new1=0
            for p in "${pids_gpu0[@]}"; do kill -0 "$p" 2>/dev/null && new0=$((new0+1)); done
            for p in "${pids_gpu1[@]}"; do kill -0 "$p" 2>/dev/null && new1=$((new1+1)); done
            gpu0_running=$new0
            gpu1_running=$new1
        done

        # Assign to GPU with free slots (prefer GPU0 — faster)
        local gpu
        if (( gpu0_running < GPU0_WORKERS )); then
            gpu="$GPU0"
            process_chunk "$chunk" "$gpu" &
            pids_gpu0+=($!)
            gpu0_running=$((gpu0_running + 1))
        else
            gpu="$GPU1"
            process_chunk "$chunk" "$gpu" &
            pids_gpu1+=($!)
            gpu1_running=$((gpu1_running + 1))
        fi
        dispatched=$((dispatched + 1))
        log "Dispatched $dispatched/${#pending[@]} → GPU$gpu (GPU0: $gpu0_running/$GPU0_WORKERS, GPU1: $gpu1_running/$GPU1_WORKERS)"
    done

    # Wait for all remaining
    local all_pids=("${pids_gpu0[@]}" "${pids_gpu1[@]}")
    for pid in "${all_pids[@]}"; do
        wait "$pid" || fails=$((fails + 1))
    done

    done_count=$(find "$INTERP_DIR" -name '*.done' 2>/dev/null | wc -l)
    log "Video2X complete: $done_count/$total done, $fails fails"
    [[ "$fails" -eq 0 ]] || die "$fails chunks failed"
}

# ── Step 4: Concatenate + mux with ORIGINAL audio ────────────
concat_and_mux() {
    local concat_file="$WORK/concat.txt"

    > "$concat_file"
    for f in $(ls "$INTERP_DIR"/chunk_*_enhanced.mp4 2>/dev/null | sort); do
        echo "file '$f'" >> "$concat_file"
    done

    local n_files
    n_files=$(wc -l < "$concat_file")
    [[ "$n_files" -gt 0 ]] || die "No enhanced chunks found in $INTERP_DIR"

    log "Concatenating $n_files chunks → $OUTPUT"

    # Mux with ORIGINAL audio from input (copy, no re-encoding)
    # Preserves perfect A/V sync by using the original audio stream
    ffmpeg -y \
        -f concat -safe 0 -i "$concat_file" \
        -i "$INPUT" \
        -map 0:v:0 -map 1:a:0 \
        -c:v copy -c:a copy \
        -shortest \
        "$OUTPUT" \
        -loglevel warning

    local size
    size="$(du -h "$OUTPUT" | cut -f1)"
    log "✓ OUTPUT: $OUTPUT ($size)"
}

# ── Main ─────────────────────────────────────────────────────
main() {
    check_deps

    local info dur
    info="$(get_video_info "$INPUT")"
    dur="$(get_duration "$INPUT")"
    local total_workers=$((GPU0_WORKERS + GPU1_WORKERS))

    log "═══════════════════════════════════════════"
    log "  VIDEO ENHANCEMENT — DUAL GPU"
    log "═══════════════════════════════════════════"
    log "  Input:    $(basename "$INPUT")"
    log "  Info:     ${info} | ${dur}s ($((dur/3600))h$((dur%3600/60))m)"
    log "  Output:   $(basename "$OUTPUT")"
    log "  Upscale:  ${V2X_UPSCALE_MODEL} ×${V2X_UPSCALE_FACTOR}"
    log "  Interp:   ${V2X_INTERP_MODEL} ×${V2X_INTERP_FACTOR}"
    log "  GPU0:     device $GPU0 — $GPU0_WORKERS workers"
    log "  GPU1:     device $GPU1 — $GPU1_WORKERS workers"
    log "  Total:    $total_workers parallel workers"
    log "  Chunks:   ${CHUNK_MINUTES}min each"
    log "  Audio:    original (copied as-is)"
    log "  Work:     $WORK"
    log "═══════════════════════════════════════════"

    split_chunks
    run_video2x
    concat_and_mux

    log "═══════════════════════════════════════════"
    log "  ✓ ALL DONE"
    log "═══════════════════════════════════════════"
}

main