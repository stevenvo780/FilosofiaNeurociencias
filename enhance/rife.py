"""RIFE frame interpolation via ncnn-Vulkan (uses Vulkan compute, not CUDA)."""
import subprocess
from pathlib import Path
from . import config as C


def interpolate(in_dir: Path, out_dir: Path) -> int:
    """25fps → 50fps frame interpolation using both GPUs via Vulkan."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_in = len(list(in_dir.glob("*.png")))
    n_out = len(list(out_dir.glob("*.png")))
    if n_out >= n_in * 2 - 1 and n_in > 0:
        return n_out
    cmd = [
        C.RIFE_BIN,
        "-i", str(in_dir), "-o", str(out_dir),
        "-m", C.RIFE_MODEL_DIR,
        "-g", "0,1", "-j", "4:4,4:4",
        "-f", "%08d.png",
    ]
    subprocess.run(cmd, check=True)
    return len(list(out_dir.glob("*.png")))
