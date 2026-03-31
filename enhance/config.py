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

ENABLE_TORCH_COMPILE = os.getenv("ENHANCE_TORCH_COMPILE", "1") == "1"
TORCH_COMPILE_MODE = os.getenv("ENHANCE_TORCH_COMPILE_MODE", "reduce-overhead")
TORCH_COMPILE_FULLGRAPH = os.getenv("ENHANCE_TORCH_COMPILE_FULLGRAPH", "0") == "1"
TORCH_COMPILE_DISABLE_CUDAGRAPHS = (
    os.getenv("ENHANCE_TORCH_COMPILE_DISABLE_CUDAGRAPHS", "1") == "1"
)

ENABLE_NVDEC = os.getenv("ENHANCE_NVDEC", "0") == "1"
NVENC_GPUS = tuple(
    int(token.strip())
    for token in os.getenv("ENHANCE_NVENC_GPUS", "0,1").split(",")
    if token.strip()
)
RIFE_GPU = int(os.getenv("ENHANCE_RIFE_GPU", "1"))
RIFE_THREADS = os.getenv("ENHANCE_RIFE_THREADS", "1:4:2")
PIPELINE_DEPTH = max(int(os.getenv("ENHANCE_PIPELINE_DEPTH", "2")), 1)

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

# ── GRACEFUL SHUTDOWN ───────────────────────────────────────
shutdown = threading.Event()

def _on_signal(sig, frame):
    print("\n[!] Interrupt — saving progress…")
    shutdown.set()

signal.signal(signal.SIGINT,  _on_signal)
signal.signal(signal.SIGTERM, _on_signal)
