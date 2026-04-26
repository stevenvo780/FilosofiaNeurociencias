#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_MEDIA="${1:?Uso: $0 input_video_o_wav output_wav [work_dir]}"
OUTPUT_WAV="${2:?Uso: $0 input_video_o_wav output_wav [work_dir]}"
WORK_DIR="${3:-$ROOT_DIR/work/deepfilter_cuda/$(basename "${OUTPUT_WAV%.*}")}" 

GPU_DEVICE="${GPU_DEVICE:-0}"
DEEPFILTER_IMAGE="${DEEPFILTER_IMAGE:-filosofia-deepfilter:cu128}"
AUDIO_CHUNK_SEC="${AUDIO_CHUNK_SEC:-300}"
DEEPFILTER_EXTRA_ARGS="${DEEPFILTER_EXTRA_ARGS:-}"
FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"
DOCKER_SHM_SIZE="${DOCKER_SHM_SIZE:-2g}"
DEEPFILTER_POST_FILTER="${DEEPFILTER_POST_FILTER:-highpass=f=70,lowpass=f=12000,loudnorm=I=-24:LRA=11:TP=-3:linear=true}"

log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "[$(date +%H:%M:%S)] ERROR: $*" >&2; exit 1; }

container_path() {
  local abs
  abs="$(realpath "$1")"
  if [[ "$abs" == "$ROOT_DIR" ]]; then
    echo "/workspace"
  else
    echo "/workspace/${abs#${ROOT_DIR}/}"
  fi
}

[[ -s "$INPUT_MEDIA" ]] || die "No existe input: $INPUT_MEDIA"
mkdir -p "$(dirname "$OUTPUT_WAV")" "$WORK_DIR/chunks" "$WORK_DIR/df"

RAW_WAV="$WORK_DIR/raw.wav"
MERGED_WAV="$WORK_DIR/deepfilter_merged.wav"
CONCAT_FILE="$WORK_DIR/concat.txt"
TMP_OUT="$WORK_DIR/output.tmp.wav"

if [[ ! -s "$RAW_WAV" ]]; then
  log "Extrayendo audio 48 kHz para DeepFilterNet"
  "$FFMPEG_BIN" -nostdin -y -i "$INPUT_MEDIA" -vn -c:a pcm_s16le -ar 48000 -ac 2 "$RAW_WAV"
fi

if [[ $(find "$WORK_DIR/chunks" -maxdepth 1 -name 'seg_*.wav' | wc -l) -eq 0 ]]; then
  log "Dividiendo audio en segmentos de ${AUDIO_CHUNK_SEC}s"
  "$FFMPEG_BIN" -nostdin -y -i "$RAW_WAV" \
    -f segment -segment_time "$AUDIO_CHUNK_SEC" \
    -c:a pcm_s16le -ar 48000 -ac 2 \
    "$WORK_DIR/chunks/seg_%04d.wav"
fi

log "Procesando segmentos con DeepFilterNet en Docker/CUDA"
docker run --rm \
  --runtime=runc \
  --device="nvidia.com/gpu=${GPU_DEVICE}" \
  --shm-size="$DOCKER_SHM_SIZE" \
  -e NVIDIA_VISIBLE_DEVICES="$GPU_DEVICE" \
  -e DF_DEVICE=cuda \
  -e DEEPFILTER_EXTRA_ARGS="$DEEPFILTER_EXTRA_ARGS" \
  -v "$ROOT_DIR":/workspace \
  -w /workspace \
  "$DEEPFILTER_IMAGE" \
  bash -lc 'set -euo pipefail
    chunk_dir="$1"
    out_dir="$2"
    mkdir -p "$out_dir"
    for wav in "$chunk_dir"/seg_*.wav; do
      out="$out_dir/$(basename "$wav")"
      if [[ -s "$out" ]]; then
        echo "skip $(basename "$wav")"
        continue
      fi
      deepFilter $DEEPFILTER_EXTRA_ARGS --no-suffix -o "$out_dir" "$wav"
    done
  ' bash "$(container_path "$WORK_DIR/chunks")" "$(container_path "$WORK_DIR/df")"

: > "$CONCAT_FILE"
while IFS= read -r wav; do
  echo "file '$(realpath "$wav")'" >> "$CONCAT_FILE"
done < <(find "$WORK_DIR/df" -maxdepth 1 -name 'seg_*.wav' | sort)

[[ -s "$CONCAT_FILE" ]] || die "DeepFilterNet no produjo segmentos"

log "Uniendo segmentos DeepFilterNet"
"$FFMPEG_BIN" -nostdin -y -f concat -safe 0 -i "$CONCAT_FILE" \
  -c:a pcm_s16le -ar 48000 -ac 2 \
  "$MERGED_WAV"

log "Normalizando salida IA de forma conservadora"
"$FFMPEG_BIN" -nostdin -y -i "$MERGED_WAV" \
  -af "$DEEPFILTER_POST_FILTER" \
  -c:a pcm_s16le -ar 48000 -ac 2 \
  "$TMP_OUT"

mv -f "$TMP_OUT" "$OUTPUT_WAV"
log "Listo: $OUTPUT_WAV"