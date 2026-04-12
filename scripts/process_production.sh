#!/usr/bin/env bash
# Production run — enhance video + audio for the recording.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="$ROOT/enhanced"
LOGDIR="$OUTDIR/logs"
mkdir -p "$LOGDIR"

# ── GPU / pipeline tuning ────────────────────────────────────
export ENHANCE_ESRGAN_GPUS="${ENHANCE_ESRGAN_GPUS:-0}"
export ENHANCE_GPU0_BATCH="${ENHANCE_GPU0_BATCH:-4}"
export ENHANCE_GPU1_BATCH="${ENHANCE_GPU1_BATCH:-1}"
export ENHANCE_RIFE_GPU="${ENHANCE_RIFE_GPU:-1}"
export ENHANCE_CHUNK_SECONDS="${ENHANCE_CHUNK_SECONDS:-30}"
export ENHANCE_NVENC_GPUS="${ENHANCE_NVENC_GPUS:-0,1}"
export ENHANCE_ESRGAN_PINNED_STAGING="${ENHANCE_ESRGAN_PINNED_STAGING:-1}"
export ENHANCE_ENABLE_JSONL_METRICS="${ENHANCE_ENABLE_JSONL_METRICS:-1}"

# ── Profiles ─────────────────────────────────────────────────
VISUAL="${ENHANCE_VISUAL_PROFILE:-quality}"
AUDIO="${ENHANCE_AUDIO_PROFILE:-natural}"
SCHED="${ENHANCE_SCHEDULER_PROFILE:-production}"
RIFE="${ENHANCE_RIFE_BACKEND:-baseline}"

# ── Run ──────────────────────────────────────────────────────
INPUT="$ROOT/videos/GMT20260320-130023_Recording_2240x1260.mp4"
LOGFILE="$LOGDIR/production_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date --iso-8601=seconds)] START $(basename "$INPUT")" | tee -a "$LOGFILE"

stdbuf -oL -eL python3 "$ROOT/scripts/run.py" "$INPUT" \
  --outdir "$OUTDIR" \
  --chunk "$ENHANCE_CHUNK_SECONDS" \
  --visual-profile "$VISUAL" \
  --audio-profile "$AUDIO" \
  --scheduler-profile "$SCHED" \
  --rife-backend "$RIFE" \
  2>&1 | tee -a "$LOGFILE"

echo "[$(date --iso-8601=seconds)] END" | tee -a "$LOGFILE"
