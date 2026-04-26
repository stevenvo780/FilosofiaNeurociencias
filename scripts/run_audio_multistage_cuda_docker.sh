#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_MEDIA="${1:?Uso: $0 input_video_o_wav output_wav [work_dir]}"
OUTPUT_WAV="${2:?Uso: $0 input_video_o_wav output_wav [work_dir]}"
WORK_DIR="${3:-$ROOT_DIR/work/audio_multistage/$(basename "${OUTPUT_WAV%.*}")}" 

GPU_DEVICE="${GPU_DEVICE:-0}"
DEEPFILTER_IMAGE="${DEEPFILTER_IMAGE:-filosofia-deepfilter:cu128}"
AUDIO_CHUNK_SEC="${AUDIO_CHUNK_SEC:-300}"
AUDIO_MAX_DURATION="${AUDIO_MAX_DURATION:-}"
FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"
DOCKER_SHM_SIZE="${DOCKER_SHM_SIZE:-2g}"
GPU_LOCK_FILE="${GPU_LOCK_FILE:-$ROOT_DIR/work/deepfilter_cuda_gpu${GPU_DEVICE}.lock}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PRE_CLEAN_PROFILE="${PRE_CLEAN_PROFILE:-electronic}"
PRE_CLEAN_FILTER="${PRE_CLEAN_FILTER:-}"
HUM_FREQ="${HUM_FREQ:-60}"
DUAL_HUM="${DUAL_HUM:-0}"
HUM_HARMONICS="${HUM_HARMONICS:-10}"
HUM_MAX_FREQ="${HUM_MAX_FREQ:-900}"
HUM_NOTCH_WIDTH="${HUM_NOTCH_WIDTH:-8}"
PRE_LOW_PASS="${PRE_LOW_PASS:-11500}"
AFFTDN_NR="${AFFTDN_NR:-8}"
AFFTDN_NF="${AFFTDN_NF:--38}"
DEEPFILTER_EXTRA_ARGS="${DEEPFILTER_EXTRA_ARGS:-}"
DEEPFILTER_BATCH_SIZE="${DEEPFILTER_BATCH_SIZE:-200}"
POST_FILTER="${POST_FILTER:-highpass=f=70,lowpass=f=12000,loudnorm=I=-24:LRA=11:TP=-3:linear=true}"
KEEP_TMP="${KEEP_TMP:-1}"

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

build_hum_notches() {
  local base="$1"
  local filter=""
  local harmonic freq width
  for (( harmonic=1; harmonic<=HUM_HARMONICS; harmonic++ )); do
    freq=$(( base * harmonic ))
    (( freq <= HUM_MAX_FREQ )) || break
    width=$(( HUM_NOTCH_WIDTH + harmonic - 1 ))
    filter+=",bandreject=f=${freq}:w=${width}"
  done
  printf '%s' "$filter"
}

build_pre_clean_filter() {
  if [[ -n "$PRE_CLEAN_FILTER" ]]; then
    printf '%s' "$PRE_CLEAN_FILTER"
    return 0
  fi

  case "$PRE_CLEAN_PROFILE" in
    none)
      printf '%s' "anull"
      ;;
    gentle)
      printf '%s' "highpass=f=70,lowpass=f=12000,afftdn=nr=5:nf=-40"
      ;;
    hum|electronic)
      local filter="highpass=f=75,lowpass=f=${PRE_LOW_PASS}"
      filter+="$(build_hum_notches "$HUM_FREQ")"
      if [[ "$DUAL_HUM" == "1" ]]; then
        if [[ "$HUM_FREQ" == "60" ]]; then
          filter+="$(build_hum_notches 50)"
        else
          filter+="$(build_hum_notches 60)"
        fi
      fi
      filter+=",afftdn=nr=${AFFTDN_NR}:nf=${AFFTDN_NF}"
      printf '%s' "$filter"
      ;;
    *)
      die "PRE_CLEAN_PROFILE inválido: $PRE_CLEAN_PROFILE (none, gentle, hum, electronic)"
      ;;
  esac
}

[[ -s "$INPUT_MEDIA" ]] || die "No existe input: $INPUT_MEDIA"
command -v "$FFMPEG_BIN" >/dev/null || die "No encuentro ffmpeg: $FFMPEG_BIN"

mkdir -p "$(dirname "$OUTPUT_WAV")" "$WORK_DIR/chunks" "$WORK_DIR/df"

RAW_WAV="$WORK_DIR/raw_48k.wav"
PRECLEAN_WAV="$WORK_DIR/preclean_no_ai.wav"
MERGED_WAV="$WORK_DIR/deepfilter_merged.wav"
CONCAT_FILE="$WORK_DIR/concat.txt"
TMP_OUT="$WORK_DIR/output.tmp.wav"

if [[ ! -s "$RAW_WAV" ]]; then
  log "Extrayendo audio PCM 48 kHz"
  duration_args=()
  if [[ -n "$AUDIO_MAX_DURATION" ]]; then
    duration_args=(-t "$AUDIO_MAX_DURATION")
  fi
  "$FFMPEG_BIN" -nostdin -y -i "$INPUT_MEDIA" -vn "${duration_args[@]}" \
    -c:a pcm_s16le -ar 48000 -ac 2 \
    "$RAW_WAV"
fi

if [[ ! -s "$PRECLEAN_WAV" ]]; then
  PRE_FILTER="$(build_pre_clean_filter)"
  log "Pre-limpieza sin IA (${PRE_CLEAN_PROFILE}): $PRE_FILTER"
  "$FFMPEG_BIN" -nostdin -y -i "$RAW_WAV" \
    -af "$PRE_FILTER" \
    -c:a pcm_s16le -ar 48000 -ac 2 \
    "$PRECLEAN_WAV"
fi

if [[ $(find "$WORK_DIR/chunks" -maxdepth 1 -name 'seg_*.wav' | wc -l) -eq 0 ]]; then
  log "Dividiendo audio pre-limpio en segmentos de ${AUDIO_CHUNK_SEC}s"
  "$FFMPEG_BIN" -nostdin -y -i "$PRECLEAN_WAV" \
    -f segment -segment_time "$AUDIO_CHUNK_SEC" \
    -c:a pcm_s16le -ar 48000 -ac 2 \
    "$WORK_DIR/chunks/seg_%04d.wav"
fi

mkdir -p "$(dirname "$GPU_LOCK_FILE")"
log "Esperando turno exclusivo de GPU para DeepFilterNet: $GPU_LOCK_FILE"
(
  flock 200
  log "Procesando segmentos con DeepFilterNet en Docker/CUDA (GPU exclusiva; una carga de modelo por lote)"
  docker run --rm \
    --runtime=runc \
    --device="nvidia.com/gpu=${GPU_DEVICE}" \
    --shm-size="$DOCKER_SHM_SIZE" \
    -e NVIDIA_VISIBLE_DEVICES="$GPU_DEVICE" \
    -e DF_DEVICE=cuda \
    -e PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    -e DEEPFILTER_EXTRA_ARGS="$DEEPFILTER_EXTRA_ARGS" \
    -e DEEPFILTER_BATCH_SIZE="$DEEPFILTER_BATCH_SIZE" \
    -v "$ROOT_DIR":/workspace \
    -w /workspace \
    "$DEEPFILTER_IMAGE" \
    bash -lc 'set -euo pipefail
      chunk_dir="$1"
      out_dir="$2"
      mkdir -p "$out_dir"

      mapfile -t all_chunks < <(find "$chunk_dir" -maxdepth 1 -name "seg_*.wav" | sort)
      pending=()
      for wav in "${all_chunks[@]}"; do
        out="$out_dir/$(basename "$wav")"
        if [[ -s "$out" ]]; then
          echo "skip $(basename "$wav")"
        else
          pending+=("$wav")
        fi
      done

      if (( ${#pending[@]} == 0 )); then
        echo "DeepFilterNet: todos los segmentos ya existen"
        exit 0
      fi

      batch_size="${DEEPFILTER_BATCH_SIZE:-200}"
      (( batch_size > 0 )) || batch_size=200
      for (( i=0; i<${#pending[@]}; i+=batch_size )); do
        batch=("${pending[@]:i:batch_size}")
        echo "DeepFilterNet lote: $((i+1))-$((i+${#batch[@]})) / ${#pending[@]}"
        # shellcheck disable=SC2086
        deepFilter $DEEPFILTER_EXTRA_ARGS --no-suffix -o "$out_dir" "${batch[@]}"
      done
    ' bash "$(container_path "$WORK_DIR/chunks")" "$(container_path "$WORK_DIR/df")"
) 200>"$GPU_LOCK_FILE"

: > "$CONCAT_FILE"
while IFS= read -r wav; do
  echo "file '$(realpath "$wav")'" >> "$CONCAT_FILE"
done < <(find "$WORK_DIR/df" -maxdepth 1 -name 'seg_*.wav' | sort)

[[ -s "$CONCAT_FILE" ]] || die "DeepFilterNet no produjo segmentos"

log "Uniendo segmentos IA"
"$FFMPEG_BIN" -nostdin -y -f concat -safe 0 -i "$CONCAT_FILE" \
  -c:a pcm_s16le -ar 48000 -ac 2 \
  "$MERGED_WAV"

log "Normalización final conservadora: $POST_FILTER"
"$FFMPEG_BIN" -nostdin -y -i "$MERGED_WAV" \
  -af "$POST_FILTER" \
  -c:a pcm_s16le -ar 48000 -ac 2 \
  "$TMP_OUT"

mv -f "$TMP_OUT" "$OUTPUT_WAV"

if [[ "$KEEP_TMP" == "0" ]]; then
  rm -f "$RAW_WAV" "$PRECLEAN_WAV" "$MERGED_WAV"
  rm -rf "$WORK_DIR/chunks" "$WORK_DIR/df"
fi

log "Listo: $OUTPUT_WAV"