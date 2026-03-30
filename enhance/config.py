"""Shared config and constants."""
import os, signal, threading

# ── ENV (must be before torch import) ───────────────────────
# Force both GPUs visible regardless of parent shell config
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ.setdefault("TORCH_COMPILE_THREADS", "4")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")

# ── PATHS ───────────────────────────────────────────────────
ESRGAN_MODEL    = "/tmp/realesr-animevideov3.pth"
RIFE_BIN        = "/tmp/rife-ncnn/rife-ncnn-vulkan-20221029-ubuntu/rife-ncnn-vulkan"
RIFE_MODEL_DIR  = "/tmp/rife-ncnn/rife-ncnn-vulkan-20221029-ubuntu/rife-v4.6"
TMPFS_WORK      = "/tmp/enhance_work"   # tmpfs ramdisk for intermediate frames

# ── TUNING ──────────────────────────────────────────────────
CHUNK_SECONDS   = 30
GPU0_BATCH      = 8
GPU1_BATCH      = 4
GPU0_SHARE      = 0.65      # RTX 5070
GPU1_SHARE      = 0.20      # RTX 2060
CPU_SHARE       = 0.15      # AMD Ryzen 9 9950X3D CPU
READ_WORKERS    = 16        # CPU threads for PNG reads
WRITE_WORKERS   = 16        # CPU threads for PNG writes
EXTRACT_THREADS = 4         # <=4 to avoid NVDEC "too many surfaces" error
ENCODE_THREADS  = 4
NVENC_GPU       = 0
PIPELINE_DEPTH  = 3

# ── GRACEFUL SHUTDOWN ───────────────────────────────────────
shutdown = threading.Event()

def _on_signal(sig, frame):
    print("\n[!] Interrupt — saving progress…")
    shutdown.set()

signal.signal(signal.SIGINT,  _on_signal)
signal.signal(signal.SIGTERM, _on_signal)
