#!/usr/bin/env bash
set -euo pipefail

SEGMENT="${1:?Uso: $0 segmento.wav outdir [deepfilter_bin]}"
OUTPUT_DIR="${2:?Uso: $0 segmento.wav outdir [deepfilter_bin]}"
DEEPFILTER_BIN="${3:-$(command -v deepFilter || true)}"
DEEPFILTER_EXTRA_ARGS="${DEEPFILTER_EXTRA_ARGS:-}"

[[ -n "$DEEPFILTER_BIN" && -x "$DEEPFILTER_BIN" ]] || {
  echo "deepFilter no está disponible" >&2
  exit 1
}

mkdir -p "$OUTPUT_DIR"

OUTPUT_FILE="$OUTPUT_DIR/$(basename "$SEGMENT")"
if [[ -s "$OUTPUT_FILE" ]]; then
  exit 0
fi

CUDA_VISIBLE_DEVICES="" \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
"$DEEPFILTER_BIN" $DEEPFILTER_EXTRA_ARGS --no-suffix -o "$OUTPUT_DIR" "$SEGMENT" >/dev/null 2>&1
