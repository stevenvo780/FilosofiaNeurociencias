"""Centralized configuration — all tunables from environment variables.

Every constant here is consumed by the pipeline modules via
``from enhance import config as C``.  Override any value by setting
the corresponding ``ENHANCE_*`` environment variable before launch.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_env = os.environ.get


def _int(key: str, default: int) -> int:
    return int(_env(key, str(default)))


def _float(key: str, default: float) -> float:
    return float(_env(key, str(default)))


def _bool(key: str, default: bool) -> bool:
    return _env(key, str(int(default))) not in ("0", "false", "False", "")


def _list_int(key: str, default: str) -> list[int]:
    return [int(x) for x in _env(key, default).split(",") if x.strip()]


# ── GPU assignment ───────────────────────────────────────────

CUDA_VISIBLE_DEVICES = _env("CUDA_VISIBLE_DEVICES", "0,1")
ESRGAN_GPUS = _list_int("ENHANCE_ESRGAN_GPUS", "0")
NVENC_GPUS = _list_int("ENHANCE_NVENC_GPUS", "0")
RIFE_GPU = _int("ENHANCE_RIFE_GPU", 1)

# ── Batch sizes ──────────────────────────────────────────────

GPU0_BATCH = _int("ENHANCE_GPU0_BATCH", 4)
GPU1_BATCH = _int("ENHANCE_GPU1_BATCH", 1)
CPU_SHARE = _float("ENHANCE_CPU_SHARE", 0.0)

# ── ESRGAN model ─────────────────────────────────────────────

ESRGAN_MODEL = _env(
    "ENHANCE_ESRGAN_MODEL",
    str(_ROOT / "enhanced" / "models" / "RealESRGAN_x2plus.pth"),
)
MODELS_DIR = _env("ENHANCE_MODELS_DIR", str(_ROOT / "enhanced" / "models"))

def resolve_esrgan_model(key: str | None = None) -> str:
    """Return model path — profile key overrides env default."""
    if key and key != "anime_baseline":
        try:
            from .models import ModelRegistry
            return str(ModelRegistry().get_path(key))
        except Exception:
            pass
    return ESRGAN_MODEL

# ── RIFE — ncnn Vulkan backend ───────────────────────────────

RIFE_BIN = _env("ENHANCE_RIFE_BIN", "rife-ncnn-vulkan")
RIFE_MODEL_DIR = _env("ENHANCE_RIFE_MODEL_DIR", "")
RIFE_THREADS = _env("ENHANCE_RIFE_THREADS", "1:4:4")
RIFE_WORKERS = _int("ENHANCE_RIFE_WORKERS", 1)
RIFE_CPU_THREADS_PER_WORKER = _int("ENHANCE_RIFE_CPU_THREADS_PER_WORKER", 4)

# ── RIFE — streaming window ─────────────────────────────────

RIFE_STREAM_WINDOW = _int("ENHANCE_RIFE_STREAM_WINDOW", 192)
RIFE_MIN_WINDOW = _int("ENHANCE_RIFE_MIN_WINDOW", 64)
RIFE_POLL_SECONDS = _float("ENHANCE_RIFE_POLL_SECONDS", 0.05)
RIFE_FILE_SETTLE_SECONDS = _float("ENHANCE_RIFE_FILE_SETTLE_SECONDS", 0.05)
RIFE_CLEANUP_MODE = _env("ENHANCE_RIFE_CLEANUP_MODE", "inline")

# ── RIFE — torch backend ────────────────────────────────────

RIFE_TORCH_MODEL_NAME = _env("ENHANCE_RIFE_TORCH_MODEL_NAME", "paper_v6")
RIFE_TORCH_MODEL_FILE = _env("ENHANCE_RIFE_TORCH_MODEL_FILE", "")
RIFE_TORCH_MODEL_DIR = _env("ENHANCE_RIFE_TORCH_MODEL_DIR", "")
RIFE_TORCH_THREADS = _int("ENHANCE_RIFE_TORCH_THREADS", 4)
RIFE_TORCH_BATCH = _int("ENHANCE_RIFE_TORCH_BATCH", 4)

# ── Pipeline ─────────────────────────────────────────────────

CHUNK_SECONDS = _int("ENHANCE_CHUNK_SECONDS", 15)
PIPELINE_DEPTH = _int("ENHANCE_PIPELINE_DEPTH", 2)
TMPFS_WORK = _env("ENHANCE_TMPFS_WORK", "/dev/shm/enhance_work")

# ── Back-pressure limits ─────────────────────────────────────

MAX_EXTRACT_BYTES_IN_FLIGHT = _int("ENHANCE_MAX_EXTRACT_BYTES", 3 * 1024**3)
MAX_RIFE_READY_BYTES = _int("ENHANCE_MAX_RIFE_READY_BYTES", 6 * 1024**3)
MAX_ESRGAN_READY_FRAMES = _int("ENHANCE_MAX_ESRGAN_READY_FRAMES", 256)
MAX_NVENC_BUFFERED_FRAMES = _int("ENHANCE_MAX_NVENC_BUFFERED_FRAMES", 64)
NVENC_STREAM_BUFFER = _int("ENHANCE_NVENC_STREAM_BUFFER", 128)

# ── Thread pools ─────────────────────────────────────────────

READ_WORKERS = _int("ENHANCE_READ_WORKERS", 4)
WRITE_WORKERS = _int("ENHANCE_WRITE_WORKERS", 4)
EXTRACT_THREADS = _int("ENHANCE_EXTRACT_THREADS", 4)
ENCODE_THREADS = _int("ENHANCE_ENCODE_THREADS", 4)
ENABLE_NVDEC = _bool("ENHANCE_ENABLE_NVDEC", True)

# ── NVENC encoding ───────────────────────────────────────────

NVENC_PRESET = _env("ENHANCE_NVENC_PRESET", "p7")
NVENC_CQ = _env("ENHANCE_NVENC_CQ", "20")
NVENC_BITRATE = _env("ENHANCE_NVENC_BITRATE", "40M")
NVENC_MAXRATE = _env("ENHANCE_NVENC_MAXRATE", "60M")
NVENC_BUFSIZE = _env("ENHANCE_NVENC_BUFSIZE", "80M")

# ── Audio ────────────────────────────────────────────────────

AUDIO_THREADS = _int("ENHANCE_AUDIO_THREADS", 4)
AUDIO_CODEC = _env("ENHANCE_AUDIO_CODEC", "aac")
AUDIO_BITRATE = _env("ENHANCE_AUDIO_BITRATE", "256k")
AUDIO_FILTER = _env("ENHANCE_AUDIO_FILTER", "")
AUDIO_CPUSET = _env("ENHANCE_AUDIO_CPUSET", "")

# ── Face detection (visual_eval) ─────────────────────────────

FACE_DETECT_MIN_SIZE_FRAC = _float("ENHANCE_FACE_DETECT_MIN_SIZE_FRAC", 0.07)
FACE_DETECT_SCALE_FACTOR = _float("ENHANCE_FACE_DETECT_SCALE_FACTOR", 1.15)
FACE_DETECT_MIN_NEIGHBORS = _int("ENHANCE_FACE_DETECT_MIN_NEIGHBORS", 5)

# ── Metrics ──────────────────────────────────────────────────

ENABLE_JSONL_METRICS = _bool("ENHANCE_ENABLE_JSONL_METRICS", True)

# ── ESRGAN staging experiments (disabled by default) ─────────

ESRGAN_EXPERIMENTAL_PINNED_STAGING = _bool("ENHANCE_ESRGAN_PINNED_STAGING", False)
ESRGAN_D2H_DOUBLE_BUFFER = _bool("ENHANCE_ESRGAN_D2H_DOUBLE_BUF", False)

# ── Torch tuning (applied at ESRGAN init) ────────────────────

TORCH_MATMUL_PRECISION = _env("ENHANCE_TORCH_MATMUL_PRECISION", "high")
CUDNN_BENCHMARK = _bool("ENHANCE_CUDNN_BENCHMARK", True)
CUDA_MATMUL_ALLOW_TF32 = _bool("ENHANCE_CUDA_MATMUL_ALLOW_TF32", True)
CUDNN_ALLOW_TF32 = _bool("ENHANCE_CUDNN_ALLOW_TF32", True)
CUDNN_BENCHMARK_LIMIT = _int("ENHANCE_CUDNN_BENCHMARK_LIMIT", 0)

# ── torch.compile (disabled — no perf gain on this pipeline) ─

ENABLE_TORCH_COMPILE = _bool("ENHANCE_TORCH_COMPILE", False)
TORCH_COMPILE_MODE = _env("ENHANCE_TORCH_COMPILE_MODE", "reduce-overhead")
TORCH_COMPILE_FULLGRAPH = _bool("ENHANCE_TORCH_COMPILE_FULLGRAPH", False)
TORCH_COMPILE_DISABLE_CUDAGRAPHS = _bool("ENHANCE_TORCH_COMPILE_NO_CUDAGRAPHS", False)

# ── GPU sharing (disabled — causes OOM on RTX 2060 PCIe x4) ─

SHARE_RIFE_GPU = _bool("ENHANCE_SHARE_RIFE_GPU", False)
RIFE_SHARED_ESRGAN_TILE = _int("ENHANCE_RIFE_SHARED_ESRGAN_TILE", 0)
RIFE_SHARED_ESRGAN_PAD = _int("ENHANCE_RIFE_SHARED_ESRGAN_PAD", 10)
RIFE_SHARED_ESRGAN_TILE_MAX_GPU_MEM_MIB = _int(
    "ENHANCE_RIFE_SHARED_ESRGAN_TILE_MAX_MEM_MIB", 6144,
)

# ── Profile names (overridden by run.py / env) ───────────────

VISUAL_PROFILE_NAME = _env("ENHANCE_VISUAL_PROFILE", "baseline")
AUDIO_PROFILE_NAME = _env("ENHANCE_AUDIO_PROFILE", "baseline")
SCHEDULER_PROFILE_NAME = _env("ENHANCE_SCHEDULER_PROFILE", "baseline")
RIFE_BACKEND_NAME = _env("ENHANCE_RIFE_BACKEND", "baseline")

# ── Graceful shutdown ────────────────────────────────────────

shutdown = threading.Event()


def _handle_signal(sig, _frame):
    print(f"\n[config] Signal {sig} received — shutting down gracefully …")
    shutdown.set()


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)
