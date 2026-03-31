"""RIFE frame interpolation via ncnn-Vulkan (uses Vulkan compute, not CUDA)."""
import subprocess
from pathlib import Path
from . import config as C


def expected_output_frames(n_in: int) -> int:
    return max(n_in, 0) * 2


def _build_command(in_dir: Path, out_dir: Path) -> list[str]:
    return [
        C.RIFE_BIN,
        "-i", str(in_dir), "-o", str(out_dir),
        "-m", C.RIFE_MODEL_DIR,
        "-g", str(C.RIFE_GPU),
        "-j", C.RIFE_THREADS,
        "-f", "%08d.png",
    ]


def start_interpolate(in_dir: Path, out_dir: Path) -> subprocess.Popen:
    """Start RIFE asynchronously so downstream stages can stream outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(_build_command(in_dir, out_dir))


def interpolate(in_dir: Path, out_dir: Path) -> int:
    """25fps → 50fps frame interpolation via Vulkan."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_in = len(list(in_dir.glob("*.png")))
    n_out = len(list(out_dir.glob("*.png")))
    if n_out >= expected_output_frames(n_in) and n_in > 0:
        return n_out
    proc = start_interpolate(in_dir, out_dir)
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, proc.args)
    return len(list(out_dir.glob("*.png")))
