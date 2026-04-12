"""Abstract RIFE backend with ncnn and torch implementations.

Provides a pluggable backend interface so the pipeline can switch between
the proven ncnn-Vulkan binary and the official IFNet torch implementation
without touching the rest of the codebase.
"""
from __future__ import annotations

import os
import subprocess
import threading
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

    def _reset_metrics(self) -> None:
        self._spawn_t = 0.0
        self._compute_t = 0.0
        self._drain_t = 0.0
        self._cleanup_t = 0.0
        self._running_t0 = None


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
        self._model_lock = threading.Lock()
        requested = (device or f"cuda:{gpu}").strip().lower()
        self._device: str = "cpu" if requested == "cpu" else f"cuda:{gpu}"
        self._model_file = model_file
        self._model_dir = model_dir
        self._cpu_threads = cpu_threads
        self._batch_size = C.RIFE_TORCH_BATCH

        # Timing metrics (per-call, not thread-safe — only for last call)
        self._spawn_t: float = 0.0
        self._compute_t: float = 0.0
        self._drain_t: float = 0.0
        self._cleanup_t: float = 0.0
        self._running_t0: float | None = None

    def _ensure_model(self) -> None:
        """Load real IFNet weights (thread-safe, loads once)."""
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is not None:
                return
            effective_threads = self._cpu_threads or (os.cpu_count() or 16)
            try:
                checkpoint = ensure_torch_rife_checkpoint(
                    model_name=self._model_name,
                    model_file=self._model_file or C.RIFE_TORCH_MODEL_FILE or None,
                    model_dir=self._model_dir or C.RIFE_TORCH_MODEL_DIR or None,
                )
                self._model = OfficialRIFEInterpolator(
                    checkpoint=checkpoint,
                    device=self._device,
                    cpu_threads=effective_threads,
                )
                print(
                    f"  [RIFE torch] Model loaded on {self._device} "
                    f"(threads_per_worker={effective_threads})",
                    flush=True,
                )
            except Exception:
                raise

    # -- ABC implementation ------------------------------------------------

    def name(self) -> str:
        return "torch"

    def expected_output_frames(self, n_input: int) -> int:
        return max(n_input, 0) * 2

    def start_interpolate(self, in_dir: Path, out_dir: Path) -> Any:
        """Launch interpolation in a background thread.

        Thread-safe: multiple concurrent calls are supported. The model
        is loaded once and shared across all worker threads (read-only
        inference is safe under torch.inference_mode).

        Returns a _TorchHandle that mimics Popen for compatibility with
        the polling-based pipeline.
        """
        self._ensure_model()
        self._reset_metrics()

        handle = _TorchHandle()
        thread = threading.Thread(
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
        """Background worker: read PNGs → interpolate → write PNGs.

        Uses concurrent segment workers to saturate all CPU cores:
        - Splits input frames into N segments (N = number of workers)
        - Each segment runs interpolation concurrently
        - PyTorch releases the GIL during compute, enabling true parallelism
        - PNG I/O is parallelized across segments
        """
        import torch
        import cv2
        from concurrent.futures import ThreadPoolExecutor, as_completed

        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()

        png_paths = sorted(in_dir.glob("*.png"))
        if not png_paths:
            handle._returncode = 0
            return

        def _read_rgb(path: Path):
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

        # Pre-read all frames in parallel (tmpfs so I/O is fast)
        with ThreadPoolExecutor(max_workers=min(16, len(png_paths))) as pool:
            all_frames = list(pool.map(_read_rgb, png_paths))

        if all_frames[0] is None:
            handle._returncode = 1
            return

        if len(all_frames) == 1:
            for i in range(2):
                dst = out_dir / f"{i + 1:08d}.png"
                cv2.imwrite(str(dst), cv2.cvtColor(all_frames[0], cv2.COLOR_RGB2BGR))
            handle._returncode = 0
            return

        self._spawn_t = time.monotonic() - t0
        t1 = time.monotonic()

        n_frames = len(all_frames)
        # Use fewer torch threads per call, run multiple concurrent workers
        n_workers = max(1, min(C.RIFE_WORKERS, n_frames - 1))
        threads_per_worker = max(1, (os.cpu_count() or 16) // n_workers)
        torch.set_num_threads(threads_per_worker)

        # Split frames into overlapping segments for concurrent processing
        # Each segment needs [start..end] frames, overlapping by 1 at boundaries
        segment_size = max(2, (n_frames + n_workers - 1) // n_workers)
        segments = []
        for i in range(0, n_frames - 1, segment_size - 1):
            seg_end = min(i + segment_size, n_frames)
            segments.append((i, seg_end))
            if seg_end >= n_frames:
                break

        # Each segment produces 2*(seg_len-1) + 2 output frames (last seg gets +2 for tail)
        # Calculate output index offsets
        seg_out_offsets = []
        out_offset = 1
        for seg_idx, (seg_start, seg_end) in enumerate(segments):
            seg_out_offsets.append(out_offset)
            n_pairs = seg_end - seg_start - 1
            out_offset += n_pairs * 2

        # Add 2 for the duplicated final frame
        total_out = out_offset + 1

        write_lock = threading.Lock()
        errors = []

        def _process_segment(seg_idx: int, seg_start: int, seg_end: int, out_base: int):
            """Process a segment of consecutive frames."""
            try:
                frames = all_frames[seg_start:seg_end]
                local_out_idx = out_base
                batch_size = self._batch_size

                pos = 0
                with torch.inference_mode():
                    while pos < len(frames) - 1:
                        if C.shutdown.is_set():
                            return

                        end = min(pos + batch_size + 1, len(frames))
                        batch = frames[pos:end]

                        mids = self._model.interpolate_many(batch[:-1], batch[1:])
                        for idx, mid in enumerate(mids):
                            dst = out_dir / f"{local_out_idx:08d}.png"
                            cv2.imwrite(str(dst), cv2.cvtColor(batch[idx], cv2.COLOR_RGB2BGR))
                            local_out_idx += 1

                            dst = out_dir / f"{local_out_idx:08d}.png"
                            cv2.imwrite(str(dst), cv2.cvtColor(mid, cv2.COLOR_RGB2BGR))
                            local_out_idx += 1

                        pos = end - 1

                # Last segment: duplicate final frame (match ncnn 2x count)
                if seg_idx == len(segments) - 1:
                    last_frame = frames[-1]
                    for _ in range(2):
                        dst = out_dir / f"{local_out_idx:08d}.png"
                        cv2.imwrite(str(dst), cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR))
                        local_out_idx += 1
            except Exception as exc:
                errors.append(exc)

        if len(segments) == 1:
            # Single segment: run directly
            _process_segment(0, segments[0][0], segments[0][1], seg_out_offsets[0])
        else:
            # Multiple segments: run concurrently
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = []
                for seg_idx, ((seg_start, seg_end), out_base) in enumerate(
                    zip(segments, seg_out_offsets)
                ):
                    futures.append(
                        pool.submit(_process_segment, seg_idx, seg_start, seg_end, out_base)
                    )
                for f in as_completed(futures):
                    f.result()

        # Restore full thread count
        torch.set_num_threads(os.cpu_count() or 16)

        self._compute_t = time.monotonic() - t1
        if errors:
            handle._returncode = 1
        else:
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

    effective_gpu = os.getenv("ENHANCE_RIFE_GPU", str(getattr(profile, "gpu", C.RIFE_GPU) if profile is not None else C.RIFE_GPU))

    if backend_name == "torch":
        gpu = effective_gpu
        model = getattr(profile, "model_name", C.RIFE_TORCH_MODEL_NAME)
        device = os.getenv("ENHANCE_RIFE_DEVICE", getattr(profile, "device", "cuda"))
        model_file = os.getenv("ENHANCE_RIFE_TORCH_MODEL_FILE", "")
        model_dir = os.getenv("ENHANCE_RIFE_TORCH_MODEL_DIR", "")
        explicit_threads = int(os.getenv("ENHANCE_RIFE_TORCH_THREADS", str(C.RIFE_TORCH_THREADS)))
        # When running on CPU, use all available cores unless explicitly overridden
        if explicit_threads == 0 and device.strip().lower() == "cpu":
            cpu_threads = os.cpu_count() or 16
        else:
            cpu_threads = explicit_threads
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

