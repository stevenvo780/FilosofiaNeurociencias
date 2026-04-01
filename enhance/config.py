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
ESRGAN_MODEL = "/tmp/realesr-animevideov3.pth"
RIFE_BIN = "/tmp/rife-ncnn/rife-ncnn-vulkan-20221029-ubuntu/rife-ncnn-vulkan"
RIFE_MODEL_DIR = "/tmp/rife-ncnn/rife-ncnn-vulkan-20221029-ubuntu/rife-v4.6"
TMPFS_WORK = "/tmp/enhance_work"  # tmpfs ramdisk for intermediate frames

# ── TUNING ──────────────────────────────────────────────────
CHUNK_SECONDS = int(os.getenv("ENHANCE_CHUNK_SECONDS", "15"))

GPU0_BATCH = int(os.getenv("ENHANCE_GPU0_BATCH", "8"))
GPU1_BATCH = int(os.getenv("ENHANCE_GPU1_BATCH", "4"))

CPU_SHARE = float(os.getenv("ENHANCE_CPU_SHARE", "0.0"))

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
    for token in os.getenv("ENHANCE_ESRGAN_GPUS", "0,1").split(",")
    if token.strip()
)
RIFE_GPU = os.getenv("ENHANCE_RIFE_GPU", "0")
RIFE_THREADS = os.getenv("ENHANCE_RIFE_THREADS", "1:4:4")
RIFE_WORKERS = max(int(os.getenv("ENHANCE_RIFE_WORKERS", "2")), 1)
# Auto-divide CPU threads among concurrent RIFE workers to avoid over-subscription.
# Default: cpu_count // RIFE_WORKERS.  Override: ENHANCE_RIFE_CPU_THREADS_PER_WORKER.
RIFE_CPU_THREADS_PER_WORKER = max(
    int(os.getenv("ENHANCE_RIFE_CPU_THREADS_PER_WORKER",
                   str(max((os.cpu_count() or 16) // RIFE_WORKERS, 1)))),
    1,
)
RIFE_STREAM_WINDOW = max(int(os.getenv("ENHANCE_RIFE_STREAM_WINDOW", "192")), 16)
RIFE_MIN_WINDOW = max(int(os.getenv("ENHANCE_RIFE_MIN_WINDOW", "64")), 1)
RIFE_POLL_SECONDS = float(os.getenv("ENHANCE_RIFE_POLL_SECONDS", "0.05"))
RIFE_FILE_SETTLE_SECONDS = float(os.getenv("ENHANCE_RIFE_FILE_SETTLE_SECONDS", "0.05"))
RIFE_TORCH_MODEL_NAME = os.getenv("ENHANCE_RIFE_TORCH_MODEL_NAME", "paper_v6")
RIFE_TORCH_MODEL_FILE = os.getenv("ENHANCE_RIFE_TORCH_MODEL_FILE", "")
RIFE_TORCH_MODEL_DIR = os.getenv("ENHANCE_RIFE_TORCH_MODEL_DIR", "")
RIFE_TORCH_THREADS = max(int(os.getenv("ENHANCE_RIFE_TORCH_THREADS", "0")), 0)
RIFE_TORCH_BATCH = max(int(os.getenv("ENHANCE_RIFE_TORCH_BATCH", "2")), 1)
# Only relevant when RIFE runs on GPU (ncnn/torch_gpu backends).
SHARE_RIFE_GPU = os.getenv("ENHANCE_SHARE_RIFE_GPU", "0") == "1"
RIFE_SHARED_ESRGAN_TILE = max(int(os.getenv("ENHANCE_RIFE_SHARED_ESRGAN_TILE", "256")), 0)
RIFE_SHARED_ESRGAN_PAD = max(int(os.getenv("ENHANCE_RIFE_SHARED_ESRGAN_PAD", "16")), 0)
RIFE_SHARED_ESRGAN_TILE_MAX_GPU_MEM_MIB = max(
    int(os.getenv("ENHANCE_RIFE_SHARED_ESRGAN_TILE_MAX_GPU_MEM_MIB", "8192")), 0
)
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
    os.getenv("ENHANCE_ESRGAN_PINNED_STAGING", "1") == "1"
)
ESRGAN_D2H_DOUBLE_BUFFER = (
    os.getenv("ENHANCE_ESRGAN_D2H_DOUBLE_BUFFER", "1") == "1"
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
        "highpass=f=80",
        "anlmdn=s=7:p=0.002:m=15",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
    ]),
)

# ── PROFILE-AWARE OVERLAY ───────────────────────────────────

VISUAL_PROFILE_NAME = os.getenv("ENHANCE_VISUAL_PROFILE", None)
AUDIO_PROFILE_NAME = os.getenv("ENHANCE_AUDIO_PROFILE", None)
SCHEDULER_PROFILE_NAME = os.getenv("ENHANCE_SCHEDULER_PROFILE", None)
RIFE_BACKEND_NAME = os.getenv("ENHANCE_RIFE_BACKEND", None)

MODELS_DIR = os.getenv("ENHANCE_MODELS_DIR", None)  # e.g. "/data/models"
AUDIO_CPUSET = os.getenv("ENHANCE_AUDIO_CPUSET", "")  # e.g. "0-3"
RIFE_CLEANUP_MODE = os.getenv("ENHANCE_RIFE_CLEANUP_MODE", "inline")  # inline | deferred | none

# ── FACE DETECTION TUNING ───────────────────────────────────
FACE_DETECT_SCALE_FACTOR = float(os.getenv("ENHANCE_FACE_DETECT_SCALE", "1.1"))
FACE_DETECT_MIN_NEIGHBORS = int(os.getenv("ENHANCE_FACE_DETECT_MIN_NEIGHBORS", "5"))
FACE_DETECT_MIN_SIZE_FRAC = float(os.getenv("ENHANCE_FACE_DETECT_MIN_SIZE_FRAC", "0.05"))


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
