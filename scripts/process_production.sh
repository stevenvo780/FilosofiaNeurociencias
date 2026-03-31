#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="$ROOT/enhanced"
LOGDIR="$OUTDIR/logs"
mkdir -p "$LOGDIR"

export ENHANCE_TORCH_COMPILE="${ENHANCE_TORCH_COMPILE:-0}"

run_video() {
  local input="$1"
  local name
  name="$(basename "$input")"
  local logfile="$LOGDIR/${name%.*}_production.log"

  echo "[$(date --iso-8601=seconds)] START $name" | tee -a "$logfile"
  stdbuf -oL -eL python3 "$ROOT/scripts/run.py" "$input" --outdir "$OUTDIR" \
    2>&1 | tee -a "$logfile"
  echo "[$(date --iso-8601=seconds)] END $name" | tee -a "$logfile"
}

run_video "$ROOT/videos/GMT20260320-130023_Recording_2240x1260.mp4"
run_video "$ROOT/videos/GMT20260320-130023_Recording_gallery_2240x1260.mp4"
