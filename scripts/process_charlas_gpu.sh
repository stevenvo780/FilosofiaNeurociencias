#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_PATTERN="${INPUT_PATTERN:-$ROOT_DIR/output/charla*.mp4}"
DELIVER_DIR="${DELIVER_DIR:-$ROOT_DIR/output}"
WORK_ROOT="${WORK_ROOT:-$ROOT_DIR/work/charlas_batch}"
GPU_DEVICE="${GPU_DEVICE:-0}"
CUDA_IMAGE="${CUDA_IMAGE:-nvidia/cuda:12.8.1-devel-ubuntu22.04}"
WHISPER_MODEL="${WHISPER_MODEL:-$ROOT_DIR/work/whispercpp-models/ggml-large-v3-turbo-q5_0.bin}"
VAD_MODEL="${VAD_MODEL:-$ROOT_DIR/work/whispercpp-models/ggml-silero-v6.2.0.bin}"
WHISPER_BIN="${WHISPER_BIN:-$ROOT_DIR/work/whispercpp-src/build/bin/whisper-cli}"
DEEPFILTER_BIN="${DEEPFILTER_BIN:-$(command -v deepFilter || true)}"
AUDIO_MODE="${AUDIO_MODE:-safe}"
AUDIO_CHUNK_SEC="${AUDIO_CHUNK_SEC:-300}"
AUDIO_PARALLEL_JOBS="${AUDIO_PARALLEL_JOBS:-8}"
AUDIO_SAFE_FILTER="${AUDIO_SAFE_FILTER:-highpass=f=70,lowpass=f=12000,afftdn=nr=6:nf=-35,loudnorm=I=-24:LRA=11:TP=-3:linear=true}"
AUDIO_DEEPFILTER_POST_FILTER="${AUDIO_DEEPFILTER_POST_FILTER:-highpass=f=70,lowpass=f=12000,loudnorm=I=-24:LRA=11:TP=-3:linear=true}"
FORCE_REGEN_AUDIO="${FORCE_REGEN_AUDIO:-0}"
FORCE_REGEN_ASR="${FORCE_REGEN_ASR:-0}"
FORCE_REMUX_FINAL="${FORCE_REMUX_FINAL:-0}"
FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"

log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "[$(date +%H:%M:%S)] ERROR: $*" >&2; exit 1; }

ffmpeg_cmd() {
  "$FFMPEG_BIN" -nostdin -y "$@"
}

container_path() {
  local abs
  abs="$(realpath "$1")"
  if [[ "$abs" == "$ROOT_DIR" ]]; then
    echo "/workspace"
  else
    echo "/workspace/${abs#${ROOT_DIR}/}"
  fi
}

parse_detected_language() {
  local json_file="$1"
  awk '
    /"result"[[:space:]]*:[[:space:]]*\{/ { in_result=1; next }
    in_result && /"language"[[:space:]]*:/ {
      if (match($0, /"language"[[:space:]]*:[[:space:]]*"([^"]+)"/, m)) {
        print m[1]
        exit
      }
    }
  ' "$json_file"
}

run_whisper() {
  local audio_file="$1"
  local out_base="$2"
  shift 2

  docker run --rm \
    --runtime=runc \
    --device="nvidia.com/gpu=${GPU_DEVICE}" \
    -v "$ROOT_DIR":/workspace \
    -w /workspace \
    "$CUDA_IMAGE" \
    "$(container_path "$WHISPER_BIN")" \
    -m "$(container_path "$WHISPER_MODEL")" \
    -vm "$(container_path "$VAD_MODEL")" \
    --vad \
    -of "$(container_path "$out_base")" \
    -f "$(container_path "$audio_file")" \
    "$@"
}

enhance_audio() {
  local input_video="$1"
  local output_wav="$2"
  local audio_work_dir="$3"
  local raw_wav="$audio_work_dir/raw.wav"
  local chunk_dir="$audio_work_dir/chunks"
  local df_dir="$audio_work_dir/deepfilter"
  local merged_wav="$audio_work_dir/merged.wav"
  local concat_file="$audio_work_dir/concat.txt"

  mkdir -p "$audio_work_dir" "$chunk_dir" "$df_dir"

  if [[ "$AUDIO_MODE" == "safe" ]]; then
    local safe_tmp="$audio_work_dir/$(basename "${output_wav%.wav}").safe.tmp.wav"
    rm -f "$safe_tmp"
    ffmpeg_cmd -i "$input_video" -vn \
      -af "$AUDIO_SAFE_FILTER" \
      -c:a pcm_s16le -ar 48000 -ac 2 \
      "$safe_tmp"
    mv -f "$safe_tmp" "$output_wav"
    return 0
  fi

  [[ "$AUDIO_MODE" == "deepfilter" ]] || die "AUDIO_MODE inválido: $AUDIO_MODE (usa safe o deepfilter)"

  if [[ ! -s "$raw_wav" ]]; then
    ffmpeg_cmd -i "$input_video" -vn \
      -c:a pcm_s16le -ar 48000 -ac 2 \
      "$raw_wav"
  fi

  if [[ $(find "$chunk_dir" -maxdepth 1 -name 'seg_*.wav' | wc -l) -eq 0 ]]; then
    ffmpeg_cmd -i "$raw_wav" \
      -f segment -segment_time "$AUDIO_CHUNK_SEC" \
      -c:a pcm_s16le -ar 48000 -ac 2 \
      "$chunk_dir/seg_%04d.wav"
  fi

  find "$chunk_dir" -maxdepth 1 -name 'seg_*.wav' -print | sort | parallel -j "$AUDIO_PARALLEL_JOBS" "$ROOT_DIR/scripts/deepfilter_chunk.sh" {} "$df_dir" "$DEEPFILTER_BIN"

  : > "$concat_file"
  while IFS= read -r f; do
    echo "file '$f'" >> "$concat_file"
  done < <(find "$df_dir" -maxdepth 1 -name 'seg_*.wav' | sort)

  ffmpeg_cmd -f concat -safe 0 -i "$concat_file" \
    -c:a pcm_s16le -ar 48000 -ac 2 \
    "$merged_wav"

  ffmpeg_cmd -i "$merged_wav" \
    -af "$AUDIO_DEEPFILTER_POST_FILTER" \
    -c:a pcm_s16le -ar 48000 -ac 2 \
    "$output_wav"
}

prepare_asr_audio() {
  local input_wav="$1"
  local output_wav="$2"
  ffmpeg_cmd -i "$input_wav" -ac 1 -ar 16000 "$output_wav"
}

mux_final() {
  local input_video="$1"
  local enhanced_audio="$2"
  local srt_es="$3"
  local srt_en="$4"
  local output_video="$5"

  ffmpeg_cmd \
    -i "$input_video" \
    -i "$enhanced_audio" \
    -i "$srt_es" \
    -i "$srt_en" \
    -map 0:v:0 -map 1:a:0 -map 2:0 -map 3:0 \
    -c:v copy \
    -c:a aac -b:a 192k \
    -c:s srt \
    -metadata:s:s:0 language=spa \
    -metadata:s:s:0 title="Español" \
    -metadata:s:s:1 language=eng \
    -metadata:s:s:1 title="English" \
    -shortest \
    "$output_video"
}

main() {
  command -v ffmpeg >/dev/null || die "ffmpeg no está instalado"
  if [[ "$AUDIO_MODE" == "deepfilter" ]]; then
    command -v parallel >/dev/null || die "GNU parallel no está instalado"
    [[ -n "$DEEPFILTER_BIN" && -x "$DEEPFILTER_BIN" ]] || die "No encuentro DeepFilterNet (deepFilter) en PATH"
  fi

  "$ROOT_DIR/scripts/bootstrap_whispercpp_cuda.sh"

  mkdir -p "$DELIVER_DIR" "$WORK_ROOT"
  local inputs=( $INPUT_PATTERN )
  [[ ${#inputs[@]} -gt 0 ]] || die "No se encontraron charlas con el patrón: $INPUT_PATTERN"

  log "Procesando ${#inputs[@]} charla(s)"

  for input_video in "${inputs[@]}"; do
    input_video="$(realpath "$input_video")"
    local base
    base="$(basename "${input_video%.*}")"
    local talk_work="$WORK_ROOT/$base"
    local talk_out="$DELIVER_DIR/$base"
    local enhanced_audio="$talk_out/${base}_audio_mejorado.wav"
    local audio_work_dir="$talk_work/audio"
    local asr_audio="$talk_work/${base}_asr.wav"
    local source_base="$talk_work/${base}_source"
    local source_srt="${source_base}.srt"
    local source_json="${source_base}.json"
    local en_base="$talk_work/${base}_en"
    local en_srt="$talk_out/${base}.en.srt"
    local es_srt="$talk_out/${base}.es.srt"
    local final_video="$talk_out/${base}_final.mkv"
    local detected_lang

    mkdir -p "$talk_work" "$talk_out"

    log "=== $base ==="

    if [[ "$FORCE_REGEN_AUDIO" == "1" || ! -s "$enhanced_audio" ]]; then
      rm -f "$enhanced_audio"
      log "Mejorando audio (modo $AUDIO_MODE)"
      enhance_audio "$input_video" "$enhanced_audio" "$audio_work_dir"
    else
      log "Audio mejorado ya existe"
    fi

    if [[ "$FORCE_REGEN_AUDIO" == "1" || "$FORCE_REGEN_ASR" == "1" ]]; then
      rm -f "$asr_audio"
    fi

    if [[ ! -s "$asr_audio" ]]; then
      log "Preparando WAV mono 16 kHz para ASR"
      prepare_asr_audio "$enhanced_audio" "$asr_audio"
    fi

    if [[ ! -s "$source_srt" || ! -s "$source_json" ]]; then
      log "Generando transcripción base con whisper.cpp + CUDA"
      run_whisper "$asr_audio" "$source_base" -l auto -oj -osrt
    else
      log "Transcripción base ya existe"
    fi

    detected_lang="$(parse_detected_language "$source_json")"
    detected_lang="${detected_lang:-unknown}"
    log "Idioma detectado: $detected_lang"

    case "$detected_lang" in
      en)
        if [[ ! -s "$en_srt" ]]; then
          cp -f "$source_srt" "$en_srt"
        fi
        if [[ ! -s "$es_srt" ]]; then
          "$ROOT_DIR/scripts/run_argos_translate_docker.sh" "$source_srt" "$es_srt" en es
        fi
        ;;
      es)
        if [[ ! -s "$es_srt" ]]; then
          cp -f "$source_srt" "$es_srt"
        fi
        if [[ ! -s "${en_base}.srt" ]]; then
          log "Generando subtítulos en inglés desde whisper.cpp"
          run_whisper "$asr_audio" "$en_base" -l auto -tr -osrt
        fi
        if [[ ! -s "$en_srt" ]]; then
          cp -f "${en_base}.srt" "$en_srt"
        fi
        ;;
      *)
        if [[ ! -s "${en_base}.srt" ]]; then
          log "Generando subtítulos en inglés desde whisper.cpp"
          run_whisper "$asr_audio" "$en_base" -l auto -tr -osrt
        fi
        if [[ ! -s "$en_srt" ]]; then
          cp -f "${en_base}.srt" "$en_srt"
        fi
        if [[ ! -s "$es_srt" ]]; then
          "$ROOT_DIR/scripts/run_argos_translate_docker.sh" "$en_srt" "$es_srt" en es
        fi
        ;;
    esac

    if [[ "$FORCE_REMUX_FINAL" == "1" || ! -s "$final_video" ]]; then
      rm -f "$final_video"
      log "Mux final con audio mejorado + subtítulos ES/EN"
      mux_final "$input_video" "$enhanced_audio" "$es_srt" "$en_srt" "$final_video"
    else
      log "Video final ya existe"
    fi

    log "Listo: $final_video"
  done
}

main "$@"
