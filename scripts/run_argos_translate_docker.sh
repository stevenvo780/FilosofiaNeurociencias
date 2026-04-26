#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SITE_DIR="$ROOT_DIR/work/argos-site"
HOME_DIR="$ROOT_DIR/work/argos-home"
TMP_DIR="$ROOT_DIR/work/argos-tmp"

INPUT_SRT="${1:?Uso: $0 input.srt output.srt from_lang to_lang}"
OUTPUT_SRT="${2:?Uso: $0 input.srt output.srt from_lang to_lang}"
FROM_LANG="${3:?Uso: $0 input.srt output.srt from_lang to_lang}"
TO_LANG="${4:?Uso: $0 input.srt output.srt from_lang to_lang}"

mkdir -p "$SITE_DIR" "$HOME_DIR" "$TMP_DIR" "$(dirname "$OUTPUT_SRT")"

TMP_INPUT="$TMP_DIR/$(basename "${INPUT_SRT%.*}")_$(date +%s)_$$.srt"
TMP_OUTPUT="$TMP_DIR/$(basename "${OUTPUT_SRT%.*}")_$(date +%s)_$$.srt"

cp -f "$INPUT_SRT" "$TMP_INPUT"

docker run --rm \
  --runtime=runc \
  -e HOME=/workspace/work/argos-home \
  -e PYTHONPATH=/workspace/work/argos-site \
  -v "$ROOT_DIR":/workspace \
  -w /workspace \
  python:3.11-slim \
  bash -lc 'set -euo pipefail \
    && if [[ ! -f /workspace/work/argos-site/argostranslate/__init__.py ]]; then python -m pip install -q --target /workspace/work/argos-site argostranslate srt; fi \
    && python /workspace/scripts/translate_srt_argos.py "$1" "$2" "$3" "$4"' \
  -- \
  "/workspace/${TMP_INPUT#${ROOT_DIR}/}" \
  "/workspace/${TMP_OUTPUT#${ROOT_DIR}/}" \
  "$FROM_LANG" \
  "$TO_LANG"

cp -f "$TMP_OUTPUT" "$OUTPUT_SRT"
rm -f "$TMP_INPUT" "$TMP_OUTPUT"
