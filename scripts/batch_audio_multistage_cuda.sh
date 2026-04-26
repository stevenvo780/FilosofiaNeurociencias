#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_PATTERN="${INPUT_PATTERN:-$ROOT_DIR/output/charla*.mp4}"
DELIVER_DIR="${DELIVER_DIR:-$ROOT_DIR/output}"
WORK_ROOT="${WORK_ROOT:-$ROOT_DIR/work/audio_multistage_batch}"
BATCH_JOBS="${BATCH_JOBS:-2}"
OUTPUT_TAG="${OUTPUT_TAG:-audio_ia_multietapa}"
FORCE_REGEN_AUDIO="${FORCE_REGEN_AUDIO:-0}"

log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "[$(date +%H:%M:%S)] ERROR: $*" >&2; exit 1; }

process_one() {
  local input_video="$1"
  input_video="$(realpath "$input_video")"
  local base
  base="$(basename "${input_video%.*}")"
  local talk_out="$DELIVER_DIR/$base"
  local output_wav="$talk_out/${base}_${OUTPUT_TAG}.wav"
  local work_dir="$WORK_ROOT/$base"

  mkdir -p "$talk_out" "$work_dir"

  if [[ "$FORCE_REGEN_AUDIO" != "1" && -s "$output_wav" ]]; then
    log "skip $base: ya existe $output_wav"
    return 0
  fi

  rm -f "$output_wav"
  log "=== $base: multi-etapa IA ==="
  "$ROOT_DIR/scripts/run_audio_multistage_cuda_docker.sh" "$input_video" "$output_wav" "$work_dir"
}

main() {
  (( BATCH_JOBS > 0 )) || die "BATCH_JOBS debe ser mayor que 0"
  mkdir -p "$WORK_ROOT" "$DELIVER_DIR"

  local inputs=( $INPUT_PATTERN )
  [[ ${#inputs[@]} -gt 0 ]] || die "No se encontraron entradas: $INPUT_PATTERN"

  log "Procesando ${#inputs[@]} charla(s) con BATCH_JOBS=$BATCH_JOBS"
  log "Nota: BATCH_JOBS paraleliza extracción/pre-limpieza; DeepFilterNet/CUDA queda serializado con lock para evitar OOM"
  local active=0
  local failures=0

  for input_video in "${inputs[@]}"; do
    (
      process_one "$input_video"
    ) &
    active=$((active + 1))

    if (( active >= BATCH_JOBS )); then
      if ! wait -n; then
        ((failures+=1))
      fi
      active=$((active - 1))
    fi
  done

  while (( active > 0 )); do
    if ! wait -n; then
      ((failures+=1))
    fi
    active=$((active - 1))
  done

  (( failures == 0 )) || die "$failures trabajo(s) fallaron"
  log "Batch multi-etapa listo"
}

main "$@"