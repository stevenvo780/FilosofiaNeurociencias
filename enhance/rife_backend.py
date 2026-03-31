"""Abstract RIFE backend with ncnn and torch implementations.

Provides a pluggable backend interface so the pipeline can switch between
the proven ncnn-Vulkan binary and a future pure-torch implementation
without touching the rest of the codebase.
"""
from __future__ import annotations

import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TYPE_CHECKING

from . import config as C

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
        return handle.wait(timeout=timeout)

    def get_metrics(self) -> dict[str, float]:
        return {
            "spawn": self._spawn_t,
            "compute": self._compute_t,
            "drain": self._drain_t,
            "cleanup": self._cleanup_t,
        }


# ── Torch backend (stub — future pure-Python RIFE) ──────────────────────────

class TorchBackend(RIFEBackend):
    """Pure-torch RIFE backend.  Currently a stub; all methods raise
    ``NotImplementedError`` until the IFNet weights are integrated.
    """

    _NOT_READY = "Torch RIFE backend not yet available — use ncnn"

    def __init__(self, gpu: int = 0, model_name: str = "rife46") -> None:
        self._gpu = gpu
        self._model_name = model_name
        self._model: Any = None

    def _ensure_model(self) -> None:
        """Download and load IFNet weights from the official RIFE repo
        (ECCV2022-RIFE).  Currently a stub.
        """
        raise NotImplementedError(self._NOT_READY)

    # -- ABC implementation (all stubs) ----------------------------------------

    def name(self) -> str:
        return "torch (stub)"

    def expected_output_frames(self, n_input: int) -> int:
        # Same 2x ratio as ncnn once implemented
        return max(n_input, 0) * 2

    def start_interpolate(self, in_dir: Path, out_dir: Path) -> Any:
        raise NotImplementedError(self._NOT_READY)

    def interpolate_sync(self, in_dir: Path, out_dir: Path) -> int:
        raise NotImplementedError(self._NOT_READY)

    def poll(self, handle: Any) -> int | None:
        raise NotImplementedError(self._NOT_READY)

    def terminate(self, handle: Any) -> None:
        raise NotImplementedError(self._NOT_READY)

    def wait(self, handle: Any, timeout: float | None = None) -> int:
        raise NotImplementedError(self._NOT_READY)


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

    if backend_name == "torch":
        gpu = getattr(profile, "gpu", 0)
        model = getattr(profile, "model_name", "rife46")
        return TorchBackend(gpu=gpu, model_name=model)

    # Default: ncnn
    gpu = getattr(profile, "gpu", C.RIFE_GPU) if profile is not None else C.RIFE_GPU
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
