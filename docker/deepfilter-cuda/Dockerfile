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
    ffmpeg \
    ca-certificates \
    git \
    libsndfile1 \
    sox \
    libsox-fmt-all && \
    rm -rf /var/lib/apt/lists/*

COPY docker/deepfilter-cuda/patch_df_io.py /tmp/patch_df_io.py

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio && \
    python3 -m pip install deepfilternet soundfile && \
    python3 -m pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu128 torch torchaudio && \
    python3 -m pip install --force-reinstall "packaging==23.2" "wheel<0.45" && \
    python3 /tmp/patch_df_io.py && \
    rm -f /tmp/patch_df_io.py && \
    python3 -m pip check && \
    deepFilter --help >/tmp/deepfilter-help.txt

WORKDIR /workspace

CMD ["deepFilter", "--help"]