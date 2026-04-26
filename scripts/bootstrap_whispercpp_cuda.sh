#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="$ROOT_DIR/work"
SRC_DIR="$WORK_DIR/whispercpp-src"
MODEL_DIR="$WORK_DIR/whispercpp-models"
CUDA_IMAGE="${CUDA_IMAGE:-nvidia/cuda:12.8.1-devel-ubuntu22.04}"
CUDA_ARCH="${CUDA_ARCH:-120}"
WHISPER_MODEL_NAME="${WHISPER_MODEL_NAME:-large-v3-turbo-q5_0}"
VAD_MODEL_NAME="${VAD_MODEL_NAME:-silero-v6.2.0}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

mkdir -p "$WORK_DIR" "$MODEL_DIR"

if [[ ! -d "$SRC_DIR/.git" ]]; then
  log "Clonando whisper.cpp en $SRC_DIR"
  git clone --depth 1 https://github.com/ggerganov/whisper.cpp "$SRC_DIR"
fi

if [[ ! -x "$SRC_DIR/build/bin/whisper-cli" ]]; then
  log "Compilando whisper.cpp con CUDA (sm_${CUDA_ARCH}) dentro de Docker"
  docker run --rm \
    --runtime=runc \
    -v "$ROOT_DIR":/workspace \
    -w /workspace \
    "$CUDA_IMAGE" \
    bash -lc "set -euo pipefail \
      && export DEBIAN_FRONTEND=noninteractive \
      && apt-get update >/dev/null \
      && apt-get install -y --no-install-recommends git build-essential cmake ffmpeg curl >/dev/null \
      && cmake -S /workspace/work/whispercpp-src -B /workspace/work/whispercpp-src/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} >/dev/null \
      && cmake --build /workspace/work/whispercpp-src/build -j\$(nproc) >/dev/null"
fi

if [[ ! -f "$MODEL_DIR/ggml-${WHISPER_MODEL_NAME}.bin" ]]; then
  log "Descargando modelo Whisper ${WHISPER_MODEL_NAME}"
  bash "$SRC_DIR/models/download-ggml-model.sh" "$WHISPER_MODEL_NAME" "$MODEL_DIR"
fi

if [[ ! -f "$MODEL_DIR/ggml-${VAD_MODEL_NAME}.bin" ]]; then
  log "Descargando modelo VAD ${VAD_MODEL_NAME}"
  bash "$SRC_DIR/models/download-vad-model.sh" "$VAD_MODEL_NAME" "$MODEL_DIR"
fi

log "Whisper.cpp listo"
log "  Binario: $SRC_DIR/build/bin/whisper-cli"
log "  Modelo : $MODEL_DIR/ggml-${WHISPER_MODEL_NAME}.bin"
log "  VAD    : $MODEL_DIR/ggml-${VAD_MODEL_NAME}.bin"
