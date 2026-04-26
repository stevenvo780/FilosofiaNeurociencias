FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    XDG_CACHE_HOME=/workspace/.cache \
    HF_HOME=/workspace/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    git \
    build-essential \
    ca-certificates \
    libsndfile1 \
    sox \
    libsox-dev \
    libsox-fmt-all && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

COPY docker/cuda-speech/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio && \
    python3 -m pip install -r /tmp/requirements.txt

WORKDIR /workspace

CMD ["python3", "--version"]
