#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="$ROOT/enhanced"
LOGDIR="$OUTDIR/logs"
mkdir -p "$LOGDIR"

export ENHANCE_TORCH_COMPILE="${ENHANCE_TORCH_COMPILE:-0}"
export ENHANCE_NVENC_GPUS="${ENHANCE_NVENC_GPUS:-0,1}"
export ENHANCE_RIFE_GPU="${ENHANCE_RIFE_GPU:-1}"
export ENHANCE_RIFE_THREADS="${ENHANCE_RIFE_THREADS:-1:8:4}"
export ENHANCE_RIFE_STREAM_WINDOW="${ENHANCE_RIFE_STREAM_WINDOW:-192}"
export ENHANCE_RIFE_MIN_WINDOW="${ENHANCE_RIFE_MIN_WINDOW:-64}"
export ENHANCE_RIFE_POLL_SECONDS="${ENHANCE_RIFE_POLL_SECONDS:-0.05}"
export ENHANCE_RIFE_FILE_SETTLE_SECONDS="${ENHANCE_RIFE_FILE_SETTLE_SECONDS:-0.05}"
export ENHANCE_NVENC_STREAM_BUFFER="${ENHANCE_NVENC_STREAM_BUFFER:-8}"
export ENHANCE_MAX_ESRGAN_READY_FRAMES="${ENHANCE_MAX_ESRGAN_READY_FRAMES:-192}"
export ENHANCE_MAX_NVENC_BUFFERED_FRAMES="${ENHANCE_MAX_NVENC_BUFFERED_FRAMES:-8}"
export ENHANCE_MAX_EXTRACT_BYTES_IN_FLIGHT="${ENHANCE_MAX_EXTRACT_BYTES_IN_FLIGHT:-6442450944}"
export ENHANCE_MAX_RIFE_READY_BYTES="${ENHANCE_MAX_RIFE_READY_BYTES:-3221225472}"
export ENHANCE_ENABLE_JSONL_METRICS="${ENHANCE_ENABLE_JSONL_METRICS:-1}"
export ENHANCE_ESRGAN_PINNED_STAGING="${ENHANCE_ESRGAN_PINNED_STAGING:-1}"

# ── Profile selection (quality + throughput defaults) ────────
export ENHANCE_VISUAL_PROFILE="${ENHANCE_VISUAL_PROFILE:-quality}"
export ENHANCE_AUDIO_PROFILE="${ENHANCE_AUDIO_PROFILE:-natural}"
export ENHANCE_SCHEDULER_PROFILE="${ENHANCE_SCHEDULER_PROFILE:-production}"
export ENHANCE_RIFE_BACKEND="${ENHANCE_RIFE_BACKEND:-baseline}"
# Experimental: enabling this makes ESRGAN share GPU1 with RIFE. It raises
# occupancy, but quality/real_x2plus currently OOMs on the RTX 2060.
export ENHANCE_SHARE_RIFE_GPU="${ENHANCE_SHARE_RIFE_GPU:-0}"
export ENHANCE_CHUNK_SECONDS="${ENHANCE_CHUNK_SECONDS:-30}"

# real_x2plus at full input resolution is not safe with batch=16 on the 5070 Ti.
# Keep a conservative default for the real-world quality profiles while retaining
# the larger batch for lighter anime/downscaled profiles.
case "${ENHANCE_VISUAL_PROFILE}" in
  quality|face_preserve|production|real_x2|real_x2plus)
    DEFAULT_ESRGAN_GPUS=0
    DEFAULT_GPU0_BATCH=4
    DEFAULT_GPU1_BATCH=1
    ;;
  *)
    DEFAULT_ESRGAN_GPUS=0,1
    DEFAULT_GPU0_BATCH=16
    DEFAULT_GPU1_BATCH=4
    ;;
esac
export ENHANCE_ESRGAN_GPUS="${ENHANCE_ESRGAN_GPUS:-$DEFAULT_ESRGAN_GPUS}"
export ENHANCE_GPU0_BATCH="${ENHANCE_GPU0_BATCH:-$DEFAULT_GPU0_BATCH}"
export ENHANCE_GPU1_BATCH="${ENHANCE_GPU1_BATCH:-$DEFAULT_GPU1_BATCH}"

run_video() {
  local input="$1"
  shift || true
  local name
  name="$(basename "$input")"
  local logfile="$LOGDIR/${name%.*}_production.log"

  echo "[$(date --iso-8601=seconds)] START $name" | tee -a "$logfile"
  stdbuf -oL -eL python3 "$ROOT/scripts/run.py" "$input" --outdir "$OUTDIR" \
    --chunk "$ENHANCE_CHUNK_SECONDS" \
    "$@" \
    2>&1 | tee -a "$logfile"
  echo "[$(date --iso-8601=seconds)] END $name" | tee -a "$logfile"
}

run_video "$ROOT/videos/GMT20260320-130023_Recording_2240x1260.mp4" \
  --visual-profile "$ENHANCE_VISUAL_PROFILE" \
  --audio-profile "$ENHANCE_AUDIO_PROFILE" \
  --scheduler-profile "$ENHANCE_SCHEDULER_PROFILE" \
  --rife-backend "$ENHANCE_RIFE_BACKEND"
run_video "$ROOT/videos/GMT20260320-130023_Recording_gallery_2240x1260.mp4" --clean \
  --visual-profile "$ENHANCE_VISUAL_PROFILE" \
  --audio-profile "$ENHANCE_AUDIO_PROFILE" \
  --scheduler-profile "$ENHANCE_SCHEDULER_PROFILE" \
  --rife-backend "$ENHANCE_RIFE_BACKEND"
