"""Shared config and constants."""
import os
import signal
import threading

# ── ENV (must be before torch import) ───────────────────────
# Force both GPUs visible regardless of parent shell config
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ.setdefault("TORCH_COMPILE_THREADS", "4")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── PATHS ───────────────────────────────────────────────────
_ESRGAN_MODEL_FALLBACK = "/tmp/realesr-animevideov3.pth"
ESRGAN_MODEL = _ESRGAN_MODEL_FALLBACK  # backward-compat alias
RIFE_BIN = "/tmp/rife-ncnn/rife-ncnn-vulkan-20221029-ubuntu/rife-ncnn-vulkan"
RIFE_MODEL_DIR = "/tmp/rife-ncnn/rife-ncnn-vulkan-20221029-ubuntu/rife-v4.6"
TMPFS_WORK = "/tmp/enhance_work"  # tmpfs ramdisk for intermediate frames

# ── TUNING ──────────────────────────────────────────────────
CHUNK_SECONDS = int(os.getenv("ENHANCE_CHUNK_SECONDS", "15"))

# Benchmarked locally on this machine. Larger batches increased VRAM usage
# without increasing effective FPS, so the tuned defaults stay conservative.
GPU0_BATCH = int(os.getenv("ENHANCE_GPU0_BATCH", "8"))
GPU1_BATCH = int(os.getenv("ENHANCE_GPU1_BATCH", "4"))

GPU0_SHARE = 0.65  # RTX 5070 Ti
GPU1_SHARE = 0.20  # RTX 2060
CPU_SHARE = 0.0    # Disabled: CPU worker hurts GPU throughput.

READ_WORKERS = int(os.getenv("ENHANCE_READ_WORKERS", "16"))
WRITE_WORKERS = int(os.getenv("ENHANCE_WRITE_WORKERS", "16"))
EXTRACT_THREADS = int(os.getenv("ENHANCE_EXTRACT_THREADS", "4"))
ENCODE_THREADS = int(os.getenv("ENHANCE_ENCODE_THREADS", "4"))

ENABLE_TORCH_COMPILE = os.getenv("ENHANCE_TORCH_COMPILE", "0") == "1"
TORCH_COMPILE_MODE = os.getenv("ENHANCE_TORCH_COMPILE_MODE", "reduce-overhead")
TORCH_COMPILE_FULLGRAPH = os.getenv("ENHANCE_TORCH_COMPILE_FULLGRAPH", "0") == "1"
TORCH_COMPILE_DISABLE_CUDAGRAPHS = (
    os.getenv("ENHANCE_TORCH_COMPILE_DISABLE_CUDAGRAPHS", "1") == "1"
)

TORCH_MATMUL_PRECISION = os.getenv("ENHANCE_TORCH_MATMUL_PRECISION", "highest")
CUDNN_BENCHMARK = os.getenv("ENHANCE_CUDNN_BENCHMARK", "0") == "1"
CUDA_MATMUL_ALLOW_TF32 = os.getenv("ENHANCE_CUDA_MATMUL_ALLOW_TF32", "0") == "1"
CUDNN_ALLOW_TF32 = os.getenv("ENHANCE_CUDNN_ALLOW_TF32", "1") == "1"
CUDNN_BENCHMARK_LIMIT = int(os.getenv("ENHANCE_CUDNN_BENCHMARK_LIMIT", "10"))

ENABLE_NVDEC = os.getenv("ENHANCE_NVDEC", "0") == "1"
NVENC_GPUS = tuple(
    int(token.strip())
    for token in os.getenv("ENHANCE_NVENC_GPUS", "0").split(",")
    if token.strip()
)
ESRGAN_GPUS = tuple(
    int(token.strip())
    for token in os.getenv("ENHANCE_ESRGAN_GPUS", "0").split(",")
    if token.strip()
)
RIFE_GPU = int(os.getenv("ENHANCE_RIFE_GPU", "1"))
RIFE_THREADS = os.getenv("ENHANCE_RIFE_THREADS", "1:8:4")
RIFE_STREAM_WINDOW = max(int(os.getenv("ENHANCE_RIFE_STREAM_WINDOW", "192")), 16)
RIFE_MIN_WINDOW = max(int(os.getenv("ENHANCE_RIFE_MIN_WINDOW", "64")), 1)
RIFE_POLL_SECONDS = float(os.getenv("ENHANCE_RIFE_POLL_SECONDS", "0.05"))
RIFE_FILE_SETTLE_SECONDS = float(os.getenv("ENHANCE_RIFE_FILE_SETTLE_SECONDS", "0.05"))
PIPELINE_DEPTH = max(int(os.getenv("ENHANCE_PIPELINE_DEPTH", "2")), 1)
MAX_EXTRACT_BYTES_IN_FLIGHT = max(
    int(os.getenv("ENHANCE_MAX_EXTRACT_BYTES_IN_FLIGHT", str(6 * 1024**3))), 1
)
MAX_RIFE_READY_BYTES = max(
    int(os.getenv("ENHANCE_MAX_RIFE_READY_BYTES", str(3 * 1024**3))), 1
)
MAX_ESRGAN_READY_FRAMES = max(
    int(os.getenv("ENHANCE_MAX_ESRGAN_READY_FRAMES", os.getenv("ENHANCE_RIFE_STREAM_WINDOW", "192"))),
    1,
)
MAX_NVENC_BUFFERED_FRAMES = max(
    int(os.getenv("ENHANCE_MAX_NVENC_BUFFERED_FRAMES", os.getenv("ENHANCE_NVENC_STREAM_BUFFER", "8"))),
    2,
)
NVENC_STREAM_BUFFER = MAX_NVENC_BUFFERED_FRAMES
ENABLE_JSONL_METRICS = os.getenv("ENHANCE_ENABLE_JSONL_METRICS", "1") == "1"
ESRGAN_EXPERIMENTAL_PINNED_STAGING = (
    os.getenv("ENHANCE_ESRGAN_PINNED_STAGING", "0") == "1"
)

NVENC_PRESET = os.getenv("ENHANCE_NVENC_PRESET", "p1")
NVENC_CQ = os.getenv("ENHANCE_NVENC_CQ", "19")
NVENC_BITRATE = os.getenv("ENHANCE_NVENC_BITRATE", "20M")
NVENC_MAXRATE = os.getenv("ENHANCE_NVENC_MAXRATE", "32M")
NVENC_BUFSIZE = os.getenv("ENHANCE_NVENC_BUFSIZE", "64M")

AUDIO_THREADS = str(min(os.cpu_count() or 8, 24))
AUDIO_CODEC = os.getenv("ENHANCE_AUDIO_CODEC", "aac")
AUDIO_BITRATE = os.getenv("ENHANCE_AUDIO_BITRATE", "256k")
AUDIO_FILTER = os.getenv(
    "ENHANCE_AUDIO_FILTER",
    ",".join([
        "afftdn=nf=-20:nt=w:om=o",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
        "dynaudnorm=f=250:g=31:p=0.95:m=8.0",
    ]),
)

# ── PROFILE-AWARE OVERLAY ───────────────────────────────────
# These variables integrate with the new profile system (profiles.py,
# audio_profiles.py, scheduler.py, rife_backend.py).  When set via env
# vars the corresponding profile is loaded; otherwise the low-level
# constants above are used directly (full backward compatibility).

VISUAL_PROFILE_NAME = os.getenv("ENHANCE_VISUAL_PROFILE", None)
AUDIO_PROFILE_NAME = os.getenv("ENHANCE_AUDIO_PROFILE", None)
SCHEDULER_PROFILE_NAME = os.getenv("ENHANCE_SCHEDULER_PROFILE", None)
RIFE_BACKEND_NAME = os.getenv("ENHANCE_RIFE_BACKEND", None)

MODELS_DIR = os.getenv("ENHANCE_MODELS_DIR", None)  # e.g. "/data/models"
AUDIO_CPUSET = os.getenv("ENHANCE_AUDIO_CPUSET", "")  # e.g. "0-3"
RIFE_CLEANUP_MODE = os.getenv("ENHANCE_RIFE_CLEANUP_MODE", "inline")  # inline | deferred | none


def resolve_esrgan_model(profile_model_filename: str | None = None) -> str:
    """Return the absolute path to the ESRGAN model weights.

    Resolution order:
      1. *profile_model_filename* under ``MODELS_DIR`` (if both are set and the
         file exists).
      2. ``ESRGAN_MODEL`` env-var / hardcoded fallback.

    This keeps the old ``ESRGAN_MODEL`` constant working for every caller that
    doesn't know about profiles.
    """
    if MODELS_DIR and profile_model_filename:
        candidate = os.path.join(MODELS_DIR, profile_model_filename)
        if os.path.isfile(candidate):
            return candidate
    return ESRGAN_MODEL


# ── GRACEFUL SHUTDOWN ───────────────────────────────────────
shutdown = threading.Event()

def _on_signal(sig, frame):
    print("\n[!] Interrupt — saving progress…")
    shutdown.set()

signal.signal(signal.SIGINT,  _on_signal)
signal.signal(signal.SIGTERM, _on_signal)
