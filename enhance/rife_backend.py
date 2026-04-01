"""Abstract RIFE backend with ncnn and torch implementations.

Provides a pluggable backend interface so the pipeline can switch between
the proven ncnn-Vulkan binary and the official IFNet torch implementation
without touching the rest of the codebase.
"""
from __future__ import annotations

import os
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TYPE_CHECKING

from . import config as C
from .rife_torch_model import OfficialRIFEInterpolator, ensure_torch_rife_checkpoint

if TYPE_CHECKING:
    from .profiles import RIFEBackendProfile


# ── Abstract base class ─────────────────────────────────────────────────────

class RIFEBackend(ABC):
    """Uniform interface for RIFE frame-interpolation backends."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable backend identifier."""
        ...

    @abstractmethod
    def expected_output_frames(self, n_input: int) -> int:
        """How many frames will be produced from *n_input* input frames."""
        ...

    @abstractmethod
    def start_interpolate(self, in_dir: Path, out_dir: Path) -> Any:
        """Start async interpolation.  Return a handle (Popen for ncnn, thread for torch)."""
        ...

    @abstractmethod
    def interpolate_sync(self, in_dir: Path, out_dir: Path) -> int:
        """Synchronous interpolation.  Return number of output frames."""
        ...

    @abstractmethod
    def poll(self, handle: Any) -> int | None:
        """Check if interpolation is done.  Return None if running, returncode if done."""
        ...

    @abstractmethod
    def terminate(self, handle: Any) -> None:
        """Request graceful termination of a running interpolation handle."""
        ...

    @abstractmethod
    def wait(self, handle: Any, timeout: float | None = None) -> int:
        """Block until interpolation finishes.  Return exit code / 0 on success."""
        ...

    def get_metrics(self) -> dict[str, float]:
        """Return timing breakdown: spawn, compute, drain, cleanup seconds."""
        return {}

    def uses_dedicated_gpu(self) -> bool:
        """Whether this backend currently occupies `C.RIFE_GPU` for compute."""
        return False


# ── ncnn-Vulkan backend (wraps current rife.py logic) ───────────────────────

class NCNNBackend(RIFEBackend):
    """RIFE interpolation through the rife-ncnn-vulkan binary."""

    def __init__(
        self,
        gpu: int = C.RIFE_GPU,
        threads: str = C.RIFE_THREADS,
        rife_bin: str | None = None,
        model_dir: str | None = None,
    ) -> None:
        self._gpu = gpu
        self._threads = threads
        self._rife_bin = rife_bin or C.RIFE_BIN
        self._model_dir = model_dir or C.RIFE_MODEL_DIR

        # Internal timing metrics (seconds)
        self._spawn_t: float = 0.0
        self._compute_t: float = 0.0
        self._drain_t: float = 0.0
        self._cleanup_t: float = 0.0
        self._running_t0: float | None = None

    # -- helpers ---------------------------------------------------------------

    def _build_command(self, in_dir: Path, out_dir: Path) -> list[str]:
        return [
            self._rife_bin,
            "-i", str(in_dir),
            "-o", str(out_dir),
            "-m", self._model_dir,
            "-g", str(self._gpu),
            "-j", self._threads,
            "-f", "%08d.png",
        ]

    def _reset_metrics(self) -> None:
        self._spawn_t = 0.0
        self._compute_t = 0.0
        self._drain_t = 0.0
        self._cleanup_t = 0.0
        self._running_t0 = None

    # -- ABC implementation ----------------------------------------------------

    def name(self) -> str:
        return "ncnn"

    def expected_output_frames(self, n_input: int) -> int:
        # Benchmarked locally: 375 input PNGs -> 750 outputs (2x).
        return max(n_input, 0) * 2

    def start_interpolate(self, in_dir: Path, out_dir: Path) -> subprocess.Popen:
        """Launch the ncnn binary asynchronously and return the Popen handle."""
        out_dir.mkdir(parents=True, exist_ok=True)
        self._reset_metrics()
        t0 = time.monotonic()
        proc = subprocess.Popen(self._build_command(in_dir, out_dir))
        self._spawn_t = time.monotonic() - t0
        self._running_t0 = time.monotonic()
        return proc

    def interpolate_sync(self, in_dir: Path, out_dir: Path) -> int:
        """Run interpolation to completion and return the output frame count."""
        out_dir.mkdir(parents=True, exist_ok=True)
        n_in = len(list(in_dir.glob("*.png")))
        n_out = len(list(out_dir.glob("*.png")))

        # Skip if already interpolated
        if n_out >= self.expected_output_frames(n_in) and n_in > 0:
            return n_out

        self._reset_metrics()
        t0 = time.monotonic()
        proc = self.start_interpolate(in_dir, out_dir)
        t1 = time.monotonic()
        rc = proc.wait()
        t2 = time.monotonic()

        self._compute_t = t2 - t1
        self._cleanup_t = 0.0  # nothing extra for sync path

        if rc != 0:
            raise subprocess.CalledProcessError(rc, proc.args)
        return len(list(out_dir.glob("*.png")))

    def poll(self, handle: Any) -> int | None:
        """Return None while the ncnn process is still running."""
        return handle.poll()

    def terminate(self, handle: Any) -> None:
        handle.terminate()

    def wait(self, handle: Any, timeout: float | None = None) -> int:
        rc = handle.wait(timeout=timeout)
        if self._running_t0 is not None and self._compute_t == 0.0:
            self._compute_t = time.monotonic() - self._running_t0
            self._running_t0 = None
        return rc

    def get_metrics(self) -> dict[str, float]:
        return {
            "spawn": self._spawn_t,
            "compute": self._compute_t,
            "drain": self._drain_t,
            "cleanup": self._cleanup_t,
        }

    def uses_dedicated_gpu(self) -> bool:
        return True


# ── Torch backend (stub — future pure-Python RIFE) ──────────────────────────

class TorchBackend(RIFEBackend):
    """Official IFNet-based torch backend for CPU/GPU RIFE inference."""

    def __init__(
        self,
        gpu: int = 0,
        model_name: str = "paper_v6",
        device: str | None = None,
        model_file: str | None = None,
        model_dir: str | None = None,
        cpu_threads: int = 0,
    ) -> None:
        self._gpu = gpu
        self._model_name = model_name
        self._model: Any = None
        requested = (device or f"cuda:{gpu}").strip().lower()
        self._device: str = "cpu" if requested == "cpu" else f"cuda:{gpu}"
        self._model_file = model_file
        self._model_dir = model_dir
        self._cpu_threads = cpu_threads
        self._batch_size = C.RIFE_TORCH_BATCH

        # Timing metrics
        self._spawn_t: float = 0.0
        self._compute_t: float = 0.0
        self._drain_t: float = 0.0
        self._cleanup_t: float = 0.0
        self._running_t0: float | None = None

    def _ensure_model(self) -> None:
        """Load real IFNet weights, with opt-in blend fallback for experiments."""
        if self._model is not None:
            return

        try:
            checkpoint = ensure_torch_rife_checkpoint(
                model_name=self._model_name,
                model_file=self._model_file or C.RIFE_TORCH_MODEL_FILE or None,
                model_dir=self._model_dir or C.RIFE_TORCH_MODEL_DIR or None,
            )
            self._model = OfficialRIFEInterpolator(
                checkpoint=checkpoint,
                device=self._device,
                cpu_threads=self._cpu_threads or C.RIFE_TORCH_THREADS,
            )
        except Exception:
            if not C.RIFE_ALLOW_BLEND_FALLBACK:
                raise
            self._model = _BlendInterpolator(self._device)

    # -- ABC implementation ------------------------------------------------

    def name(self) -> str:
        return "torch"

    def expected_output_frames(self, n_input: int) -> int:
        return max(n_input, 0) * 2

    def start_interpolate(self, in_dir: Path, out_dir: Path) -> Any:
        """Synchronously interpolate in-memory, write results as PNGs.

        Returns a _TorchHandle that mimics Popen for compatibility with
        the polling-based pipeline.
        """
        import threading as _th
        self._ensure_model()
        self._reset_metrics()

        handle = _TorchHandle()
        thread = _th.Thread(
            target=self._interpolate_worker,
            args=(in_dir, out_dir, handle),
            daemon=True,
            name="torch-rife-worker",
        )
        thread.start()
        handle._thread = thread
        self._running_t0 = time.monotonic()
        return handle

    def _interpolate_worker(self, in_dir: Path, out_dir: Path,
                            handle: '_TorchHandle') -> None:
        """Background worker: read PNGs → interpolate → write PNGs."""
        import torch
        import cv2

        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()

        png_paths = sorted(in_dir.glob("*.png"))
        if not png_paths:
            handle._returncode = 0
            return

        def _read_rgb(path: Path):
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

        prev = _read_rgb(png_paths[0])
        if prev is None:
            handle._returncode = 1
            return

        if len(png_paths) == 1:
            for i in range(2):
                dst = out_dir / f"{i + 1:08d}.png"
                cv2.imwrite(str(dst), cv2.cvtColor(prev, cv2.COLOR_RGB2BGR))
            handle._returncode = 0
            return

        self._spawn_t = time.monotonic() - t0
        t1 = time.monotonic()

        out_idx = 1
        next_pos = 1
        with torch.inference_mode():
            while next_pos < len(png_paths):
                if C.shutdown.is_set():
                    handle._returncode = -1
                    return

                batch_frames = [prev]
                while len(batch_frames) < self._batch_size + 1 and next_pos < len(png_paths):
                    curr = _read_rgb(png_paths[next_pos])
                    next_pos += 1
                    if curr is None:
                        handle._returncode = 1
                        return
                    batch_frames.append(curr)

                mids = self._model.interpolate_many(batch_frames[:-1], batch_frames[1:])
                for idx, mid in enumerate(mids):
                    dst = out_dir / f"{out_idx:08d}.png"
                    cv2.imwrite(str(dst), cv2.cvtColor(batch_frames[idx], cv2.COLOR_RGB2BGR))
                    out_idx += 1

                    dst = out_dir / f"{out_idx:08d}.png"
                    cv2.imwrite(str(dst), cv2.cvtColor(mid, cv2.COLOR_RGB2BGR))
                    out_idx += 1

                prev = batch_frames[-1]

            # Match ncnn's exact 2x frame count by duplicating the final frame.
            for _ in range(2):
                dst = out_dir / f"{out_idx:08d}.png"
                cv2.imwrite(str(dst), cv2.cvtColor(prev, cv2.COLOR_RGB2BGR))
                out_idx += 1

        self._compute_t = time.monotonic() - t1
        handle._returncode = 0

    def interpolate_sync(self, in_dir: Path, out_dir: Path) -> int:
        handle = self.start_interpolate(in_dir, out_dir)
        self.wait(handle)
        if handle._returncode != 0:
            raise RuntimeError(f"Torch RIFE interpolation failed (rc={handle._returncode})")
        n_out = 0
        while (out_dir / f"{n_out + 1:08d}.png").exists():
            n_out += 1
        return n_out

    def poll(self, handle: Any) -> int | None:
        return handle.poll()

    def terminate(self, handle: Any) -> None:
        C.shutdown.set()

    def wait(self, handle: Any, timeout: float | None = None) -> int:
        rc = handle.wait(timeout=timeout)
        if self._running_t0 is not None and self._compute_t == 0.0:
            self._compute_t = time.monotonic() - self._running_t0
            self._running_t0 = None
        return rc

    def get_metrics(self) -> dict[str, float]:
        return {
            "spawn": self._spawn_t,
            "compute": self._compute_t,
            "drain": self._drain_t,
            "cleanup": self._cleanup_t,
        }

    def uses_dedicated_gpu(self) -> bool:
        return self._device.startswith("cuda:")

    def _reset_metrics(self) -> None:
        self._spawn_t = 0.0
        self._compute_t = 0.0
        self._drain_t = 0.0
        self._cleanup_t = 0.0
        self._running_t0 = None


class _TorchHandle:
    """Mimics subprocess.Popen for TorchBackend compatibility."""
    __slots__ = ("_returncode", "_thread")

    def __init__(self):
        self._returncode: int | None = None
        self._thread: Any = None

    def poll(self) -> int | None:
        if self._thread is not None and self._thread.is_alive():
            return None
        return self._returncode

    def wait(self, timeout: float | None = None) -> int:
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        return self._returncode if self._returncode is not None else 0

    def terminate(self):
        C.shutdown.set()


class _BlendInterpolator:
    """Debug-only frame blending fallback.

    This is useful to stress the scheduler, but it is not a valid substitute
    for real RIFE quality and must never be treated as production output.
    """

    def __init__(self, device: str):
        self._device = device

    def interpolate(self, f0, f1):
        """Return the mid-frame between f0 and f1 via alpha blending."""
        import numpy as _np
        return (f0.astype(_np.uint16) + f1.astype(_np.uint16) + 1) // 2


# ── Factory ──────────────────────────────────────────────────────────────────

def create_backend(profile: RIFEBackendProfile | None = None) -> RIFEBackend:
    """Instantiate the appropriate backend from a profile (or defaults).

    Parameters
    ----------
    profile:
        An optional ``RIFEBackendProfile`` (or similar object with a
        ``.backend`` string attribute).  When *None* the ncnn backend is
        used with values from ``config``.
    """
    backend_name = getattr(profile, "backend", "ncnn") if profile is not None else "ncnn"

    effective_gpu = int(os.getenv("ENHANCE_RIFE_GPU", str(getattr(profile, "gpu", C.RIFE_GPU) if profile is not None else C.RIFE_GPU)))

    if backend_name == "torch":
        gpu = effective_gpu
        model = getattr(profile, "model_name", C.RIFE_TORCH_MODEL_NAME)
        device = os.getenv("ENHANCE_RIFE_DEVICE", getattr(profile, "device", "cuda"))
        model_file = os.getenv("ENHANCE_RIFE_TORCH_MODEL_FILE", "")
        model_dir = os.getenv("ENHANCE_RIFE_TORCH_MODEL_DIR", "")
        cpu_threads = int(os.getenv("ENHANCE_RIFE_TORCH_THREADS", str(C.RIFE_TORCH_THREADS)))
        return TorchBackend(
            gpu=gpu,
            model_name=model,
            device=device,
            model_file=model_file or None,
            model_dir=model_dir or None,
            cpu_threads=cpu_threads,
        )

    # Default: ncnn
    gpu = effective_gpu
    threads = getattr(profile, "threads", C.RIFE_THREADS) if profile is not None else C.RIFE_THREADS
    rife_bin = getattr(profile, "rife_bin", None) if profile is not None else None
    model_dir = getattr(profile, "model_dir", None) if profile is not None else None
    return NCNNBackend(gpu=gpu, threads=threads, rife_bin=rife_bin, model_dir=model_dir)


# ── Module-level backward-compatible helpers ─────────────────────────────────
# These mirror the original rife.py public API so existing callers keep working.

_default_backend: NCNNBackend | None = None


def _get_default() -> NCNNBackend:
    global _default_backend
    if _default_backend is None:
        _default_backend = NCNNBackend()
    return _default_backend


def expected_output_frames(n_in: int) -> int:
    """Backward-compatible wrapper — delegates to the default NCNNBackend."""
    return _get_default().expected_output_frames(n_in)


def start_interpolate(in_dir: Path, out_dir: Path) -> subprocess.Popen:
    """Backward-compatible wrapper — delegates to the default NCNNBackend."""
    return _get_default().start_interpolate(in_dir, out_dir)


def interpolate(in_dir: Path, out_dir: Path) -> int:
    """Backward-compatible wrapper — delegates to the default NCNNBackend."""
    return _get_default().interpolate_sync(in_dir, out_dir)
