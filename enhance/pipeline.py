"""
4-stage pipeline: extract → RIFE → ESRGAN → NVENC.

Chunk queues overlap CPU decode, Vulkan interpolation, dual-GPU ESRGAN and
hardware NVENC encode across chunk boundaries.
"""
import gc
import json
import queue
import shutil
import subprocess
import threading
import time
import traceback
from contextlib import suppress
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from .progress import Progress
from .ffmpeg_utils import extract_frames, extract_frames_to_ram
from .esrgan import ESRGANEngine
from .rife_backend import RIFEBackend, create_backend


def _frame_bytes(w: int, h: int) -> int:
    return w * h * 3


class _BudgetController:
    """Track in-flight work in real units instead of queue length."""

    __slots__ = ("cond", "_extract", "_rife_ready")

    def __init__(self):
        self.cond = threading.Condition()
        self._extract: dict[int, int] = {}
        self._rife_ready: dict[int, int] = {}

    def extract_total(self) -> int:
        return sum(self._extract.values())

    def rife_ready_total(self) -> int:
        return sum(self._rife_ready.values())

    def reserve_extract(self, cid: int, nbytes: int) -> tuple[int, int]:
        with self.cond:
            while (
                self.extract_total() + nbytes > C.MAX_EXTRACT_BYTES_IN_FLIGHT
                or self.rife_ready_total() > C.MAX_RIFE_READY_BYTES
            ) and not C.shutdown.is_set():
                self.cond.wait(timeout=0.1)
            self._extract[cid] = nbytes
            return self.extract_total(), self.rife_ready_total()

    def update_extract(self, cid: int, nbytes: int) -> tuple[int, int]:
        with self.cond:
            self._extract[cid] = nbytes
            self.cond.notify_all()
            return self.extract_total(), self.rife_ready_total()

    def release_extract(self, cid: int):
        with self.cond:
            self._extract.pop(cid, None)
            self.cond.notify_all()

    def set_rife_ready(self, cid: int, nbytes: int) -> tuple[int, int]:
        with self.cond:
            self._rife_ready[cid] = nbytes
            self.cond.notify_all()
            return self.extract_total(), self.rife_ready_total()

    def clear_rife_ready(self, cid: int):
        with self.cond:
            self._rife_ready.pop(cid, None)
            self.cond.notify_all()

    def wait_for_rife_room(self, extra_bytes: int = 0) -> tuple[int, int]:
        with self.cond:
            while (
                self.rife_ready_total() + extra_bytes > C.MAX_RIFE_READY_BYTES
            ) and not C.shutdown.is_set():
                self.cond.wait(timeout=C.RIFE_POLL_SECONDS)
            return self.extract_total(), self.rife_ready_total()

    def snapshot(self) -> tuple[int, int]:
        with self.cond:
            return self.extract_total(), self.rife_ready_total()


class _MetricsStore:
    __slots__ = ("enabled", "lock", "path", "records")

    def __init__(self, work: Path):
        self.enabled = C.ENABLE_JSONL_METRICS
        self.lock = threading.Lock()
        self.path = work / "chunk_metrics.jsonl"
        self.records: dict[int, dict] = {}

    def update(self, cid: int, **fields):
        with self.lock:
            record = self.records.setdefault(cid, {"chunk": cid})
            record.update(fields)

    def emit(self, cid: int):
        if not self.enabled:
            return
        with self.lock:
            record = self.records.pop(cid, None)
        if not record:
            return
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    def snapshot(self, cid: int) -> dict:
        with self.lock:
            return dict(self.records.get(cid, {}))


def _tmpfs_chunk(cid: int) -> Path:
    """Return tmpfs path for a chunk's intermediate frames."""
    return Path(C.TMPFS_WORK) / f"chunk_{cid:04d}"


class _ReorderWriter:
    """Reorder buffer that writes frames to an NVENC pipe in sequential order.

    Dual-GPU produces frames out of order (GPU0: 0..split, GPU1: split..N).
    on_frame() does a fast numpy copy and enqueues it.
    A background writer thread drains the queue sequentially to the pipe,
    fully decoupling GPU inference from pipe I/O blocking.

    Bounded buffer: if > MAX_BUFFERED frames accumulate (NVENC slower than
    ESRGAN), on_frame blocks until the writer thread drains, providing
    natural backpressure to GPU workers without OOM risk.
    """
    __slots__ = ("pipe", "lock", "cond", "buf", "next_idx", "total",
                 "written", "_finished", "_writer_thread", "_error", "_sem",
                 "_buffered", "max_buffered", "buffer_capacity")

    @staticmethod
    def _capacity() -> int:
        return max(
            C.NVENC_STREAM_BUFFER,
            C.GPU0_BATCH + C.GPU1_BATCH,
            C.GPU0_BATCH,
            C.GPU1_BATCH,
            2,
        )

    def __init__(self, pipe_stdin, total: int):
        self.pipe = pipe_stdin
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.buf: dict[int, np.ndarray] = {}
        self.next_idx = 0
        self.total = total
        self.written = 0
        self._finished = False
        self._error: Exception | None = None
        self.buffer_capacity = self._capacity()
        self._sem = threading.Semaphore(self.buffer_capacity)
        self._buffered = 0
        self.max_buffered = 0
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="nvenc-writer")
        self._writer_thread.start()

    def on_frame(self, idx: int, frame):
        """Callback from ESRGAN GPU workers — thread-safe, minimal blocking.

        *frame* can be a numpy array or a torch tensor (pinned memory).
        T2: both support the buffer protocol, so pipe.write(memoryview(arr))
        works transparently.  When a torch tensor is received we keep it as-is
        to avoid an unnecessary numpy copy on the hot path.
        """
        self._sem.acquire()
        # Accept numpy ndarray or torch tensor — both support buffer protocol
        if hasattr(frame, 'flags'):
            # numpy path
            arr = frame if frame.flags.c_contiguous else np.ascontiguousarray(frame)
        elif hasattr(frame, 'is_contiguous'):
            # torch tensor path (T2): use a zero-copy numpy view backed by the
            # pinned CPU tensor, because memoryview(torch.Tensor) is not stable
            # enough across Python/torch builds for the ffmpeg pipe hot path.
            tensor = frame if frame.is_contiguous() else frame.contiguous()
            arr = tensor.numpy()
        else:
            arr = np.ascontiguousarray(frame)
        with self.cond:
            self.buf[idx] = arr
            self._buffered += 1
            if self._buffered > self.max_buffered:
                self.max_buffered = self._buffered
            if idx == self.next_idx:
                self.cond.notify()

    def _writer_loop(self):
        """Background thread: drain buffer in order → pipe."""
        try:
            self.lock.acquire()
            while True:
                while self.next_idx not in self.buf and not self._finished:
                    self.cond.wait()
                batch: list[np.ndarray] = []
                while self.next_idx in self.buf:
                    batch.append(self.buf.pop(self.next_idx))
                    self.next_idx += 1
                    self._buffered -= 1
                done = self._finished and self.next_idx not in self.buf
                self.lock.release()
                for arr in batch:
                    try:
                        self.pipe.write(memoryview(arr))
                        self.written += 1
                    except BrokenPipeError:
                        return
                    finally:
                        self._sem.release()
                if done:
                    return
                self.lock.acquire()
        except Exception as e:
            self._error = e
        finally:
            try:
                self.lock.release()
            except RuntimeError:
                pass

    def flush_remaining(self):
        """Signal writer thread to drain remaining frames and join."""
        with self.cond:
            self._finished = True
            self.cond.notify()
        self._writer_thread.join(timeout=120)
        with self.lock:
            orphaned = len(self.buf)
        if self._error:
            raise RuntimeError(f"ReorderWriter error: {self._error}") from self._error
        if orphaned:
            raise RuntimeError(
                f"ReorderWriter left {orphaned} orphan frames waiting for missing indices"
            )

    def buffered_frames(self) -> int:
        with self.lock:
            return self._buffered


def _open_nvenc_pipe(out_file: Path, w: int, h: int, fps: float,
                     gpu: int = 0) -> subprocess.Popen:
    """Open an ffmpeg NVENC subprocess that accepts raw RGB24 on stdin."""
    import fcntl
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "hevc_nvenc", "-gpu", str(gpu),
        "-preset", C.NVENC_PRESET,
        "-rc", "vbr", "-cq", C.NVENC_CQ,
        "-b:v", C.NVENC_BITRATE,
        "-maxrate", C.NVENC_MAXRATE,
        "-bufsize", C.NVENC_BUFSIZE,
        "-profile:v", "main10", "-pix_fmt", "p010le",
        str(out_file), "-loglevel", "warning",
    ]
    frame_bytes = w * h * 3
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            bufsize=frame_bytes * 16)
    try:
        F_SETPIPE_SZ = 1031
        fcntl.fcntl(proc.stdin.fileno(), F_SETPIPE_SZ, 1 << 24)  # 16 MB pipe
    except OSError:
        pass
    return proc


def _encode_from_numpy(frames: list[np.ndarray], out_file: Path,
                        fps: float, gpu: int = 0):
    """Pipe numpy RGB frames directly to NVENC — zero PNG I/O."""
    if out_file.exists() and out_file.stat().st_size > 1000:
        return
    if not frames:
        return
    h, w = frames[0].shape[:2]
    proc = _open_nvenc_pipe(out_file, w, h, fps, gpu)
    try:
        for f in frames:
            proc.stdin.write(memoryview(np.ascontiguousarray(f)))
        proc.stdin.close()
    except BrokenPipeError:
        pass
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"NVENC encode failed (rc={proc.returncode})")


def _try_gpu_resident_encode(tensors: list, out_file: Path,
                              fps: float, gpu: int = 0) -> bool:
    """T16: Attempt GPU-resident ESRGAN→NVENC encoding via CUDA shared surfaces.

    Uses pynvvideocodec (PyNvVideoCodec) for direct GPU tensor → NVENC path
    when available.  Returns True if successful, False if the library is not
    installed or the encode fails (caller should fall back to pipe-based path).

    Parameters
    ----------
    tensors : list of torch.Tensor
        NHWC uint8 tensors on the target CUDA device.
    out_file : Path
        Output .mp4 path.
    fps : float
        Target framerate.
    gpu : int
        CUDA device ordinal for NVENC.
    """
    if not C.ESRGAN_GPU_RESIDENT:
        return False

    try:
        # Check for PyNvVideoCodec or similar CUDA video encoder bindings
        import PyNvVideoCodec as nvc  # type: ignore[import-not-found]
    except ImportError:
        return False

    try:
        import torch
        if not tensors or not isinstance(tensors[0], torch.Tensor):
            return False

        h, w = tensors[0].shape[0], tensors[0].shape[1]
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # Create NVENC encoder with CUDA interop
        encoder = nvc.CreateEncoder(
            w, h, "hevc",
            gpu_id=gpu,
            preset="P1",
            rate_control="VBR",
            bitrate=int(C.NVENC_BITRATE.rstrip("M")) * 1_000_000,
            max_bitrate=int(C.NVENC_MAXRATE.rstrip("M")) * 1_000_000,
            fps=int(fps),
        )

        with open(str(out_file), "wb") as fout:
            for tensor in tensors:
                # Convert RGB → NV12 on GPU and encode
                packet = encoder.encode_frame(tensor)
                if packet:
                    fout.write(packet)
            # Flush encoder
            while True:
                packet = encoder.flush()
                if not packet:
                    break
                fout.write(packet)

        return True
    except Exception:
        # GPU-resident encode failed; caller falls back to pipe path
        return False


def _stream_esrgan_to_nvenc(esr: ESRGANEngine, frames: list[np.ndarray],
                            out_file: Path, fps: float, gpu: int) -> tuple[int, float]:
    """Upscale and encode a chunk without materializing 4K frames in RAM."""
    if out_file.exists() and out_file.stat().st_size > 1000:
        return 0, 0.0
    if not frames:
        return 0, 0.0

    h, w = frames[0].shape[:2]
    proc = _open_nvenc_pipe(out_file, w * esr.output_scale, h * esr.output_scale, fps, gpu)
    writer = _ReorderWriter(proc.stdin, len(frames))
    t0 = time.time()

    try:
        produced = esr.process_streaming(frames, writer.on_frame)
        writer.flush_remaining()

        with suppress(BrokenPipeError, OSError, ValueError):
            proc.stdin.close()
        proc.wait()
    except Exception:
        with suppress(Exception):
            writer.flush_remaining()
        with suppress(Exception):
            proc.kill()
        with suppress(Exception):
            proc.wait(timeout=5)
        raise

    if proc.returncode != 0 and not C.shutdown.is_set():
        raise RuntimeError(f"NVENC streaming encode failed (rc={proc.returncode})")

    return produced, time.time() - t0


def _read_png_rgb(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None


def _load_png_window(paths: list[Path]) -> list[np.ndarray]:
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=C.READ_WORKERS) as pool:
        frames = list(pool.map(_read_png_rgb, paths))

    ready = []
    for frame in frames:
        if frame is None:
            break
        ready.append(frame)
    return ready


def _count_numbered_pngs(root: Path) -> int:
    total = 0
    while (root / f"{total + 1:08d}.png").exists():
        total += 1
    return total


def _apply_rife_backend_metrics(
    metrics: _MetricsStore | None,
    cid: int,
    backend: RIFEBackend,
) -> dict[str, float]:
    backend_stats = backend.get_metrics()
    mapped = {
        f"rife_{name}_seconds": float(value)
        for name, value in backend_stats.items()
    }
    if metrics is not None and mapped:
        metrics.update(cid, **mapped)
    return mapped


def _cleanup_chunk_dir(path: Path):
    if C.RIFE_CLEANUP_MODE == "none":
        return
    shutil.rmtree(path, ignore_errors=True)


def _stream_rife_esrgan_to_nvenc(esr: ESRGANEngine, rife_backend: RIFEBackend,
                                 rife_in: Path, out_file: Path,
                                 fps: float, gpu: int, cid: int,
                                 w: int, h: int,
                                 budget: _BudgetController | None = None,
                                 metrics: _MetricsStore | None = None,
                                 prefetched: dict | None = None) -> dict[str, float]:
    """Run RIFE asynchronously and feed ready windows into ESRGAN/NVENC."""
    if out_file.exists() and out_file.stat().st_size > 1000:
        prefetched_handle = prefetched.get("handle") if prefetched else None
        if prefetched_handle is not None:
            with suppress(Exception):
                rife_backend.terminate(prefetched_handle)
            with suppress(Exception):
                rife_backend.wait(prefetched_handle, timeout=5)
        if budget is not None:
            budget.clear_rife_ready(cid)
            budget.release_extract(cid)
        _cleanup_chunk_dir(rife_in.parent)
        return {
            "produced": 0,
            "total_seconds": 0.0,
            "rife_seconds": 0.0,
            "readback_seconds": 0.0,
            "encode_tail_seconds": 0.0,
            "window_count": 0,
            "window_avg_frames": 0.0,
            "window_max_frames": 0,
            "rife_ready_peak_bytes": 0,
            "extract_peak_bytes": 0,
            "nvenc_peak_frames": 0,
        }

    expected = rife_backend.expected_output_frames(_count_numbered_pngs(rife_in))
    if expected == 0:
        if budget is not None:
            budget.clear_rife_ready(cid)
            budget.release_extract(cid)
        _cleanup_chunk_dir(rife_in.parent)
        return {
            "produced": 0,
            "total_seconds": 0.0,
            "rife_seconds": 0.0,
            "readback_seconds": 0.0,
            "encode_tail_seconds": 0.0,
            "window_count": 0,
            "window_avg_frames": 0.0,
            "window_max_frames": 0,
            "rife_ready_peak_bytes": 0,
            "extract_peak_bytes": 0,
            "nvenc_peak_frames": 0,
        }

    prefetched_handle = prefetched.get("handle") if prefetched else None
    prefetched_out = prefetched.get("out_dir") if prefetched else None
    rife_out = prefetched_out or (rife_in.parent / "rife")
    if prefetched_out is None:
        shutil.rmtree(rife_out, ignore_errors=True)
    frame_nbytes = _frame_bytes(w, h)
    window_cap = max(1, min(C.RIFE_STREAM_WINDOW, C.MAX_ESRGAN_READY_FRAMES))

    nvenc = _open_nvenc_pipe(out_file, w * esr.output_scale, h * esr.output_scale, fps, gpu)
    writer = _ReorderWriter(nvenc.stdin, expected)
    rife_handle = prefetched_handle or rife_backend.start_interpolate(rife_in, rife_out)

    t0 = time.time()
    t_rife = float(prefetched.get("started_at", t0)) if prefetched else time.time()
    read_dt = 0.0
    encode_tail_dt = 0.0
    next_idx = 0
    window_count = 0
    window_total = 0
    window_max = 0
    rife_ready_peak = 0
    extract_peak = 0
    esr_stats: dict[str, float] = {}

    try:
        while next_idx < expected:
            if C.shutdown.is_set():
                with suppress(Exception):
                    rife_backend.terminate(rife_handle)
                break

            limit = min(expected, next_idx + window_cap)
            now = time.time()
            ready_paths: list[Path] = []
            probe = next_idx
            # T14: use pre-calculated paths instead of glob scanning
            while probe < limit:
                path = rife_out / f"{probe + 1:08d}.png"
                try:
                    st = path.stat()
                except OSError:
                    break
                if st.st_size <= 0 or (now - st.st_mtime) < C.RIFE_FILE_SETTLE_SECONDS:
                    break
                ready_paths.append(path)
                probe += 1

            ready_bytes = len(ready_paths) * frame_nbytes
            if budget is not None:
                extract_total, rife_ready_total = budget.set_rife_ready(cid, ready_bytes)
                extract_peak = max(extract_peak, extract_total)
                rife_ready_peak = max(rife_ready_peak, rife_ready_total)

            ready_enough = len(ready_paths) >= C.RIFE_MIN_WINDOW or (
                rife_backend.poll(rife_handle) is not None and len(ready_paths) > 0
            )
            if ready_enough:
                if budget is not None:
                    extract_total, rife_ready_total = budget.wait_for_rife_room()
                    extract_peak = max(extract_peak, extract_total)
                    rife_ready_peak = max(rife_ready_peak, rife_ready_total)
                while (
                    writer.buffered_frames() >= C.MAX_NVENC_BUFFERED_FRAMES
                    and not C.shutdown.is_set()
                ):
                    time.sleep(C.RIFE_POLL_SECONDS)
                t_read = time.time()
                frames = _load_png_window(ready_paths)
                read_dt += time.time() - t_read
                if not frames:
                    time.sleep(C.RIFE_POLL_SECONDS)
                    continue

                ready_count = len(frames)
                offset = next_idx
                produced = esr.process_streaming(
                    frames,
                    lambda idx, frame, base=offset: writer.on_frame(base + idx, frame),
                    log_progress=False,
                    telemetry=esr_stats,
                )
                if produced != ready_count:
                    raise RuntimeError(
                        f"Streaming ESRGAN produced {produced} frames, expected {ready_count}"
                    )

                next_idx += ready_count
                window_count += 1
                window_total += ready_count
                window_max = max(window_max, ready_count)
                # T14: batch unlink all consumed paths at once
                for path in ready_paths[:ready_count]:
                    try:
                        path.unlink()
                    except OSError:
                        pass
                if budget is not None:
                    extract_total, _ = budget.set_rife_ready(
                        cid, max(len(ready_paths) - ready_count, 0) * frame_nbytes
                    )
                    extract_peak = max(extract_peak, extract_total)
                continue

            rc = rife_backend.poll(rife_handle)
            if rc is not None:
                if rc != 0 and not C.shutdown.is_set():
                    raise RuntimeError(f"RIFE failed for chunk {cid:04d} (rc={rc})")
                if next_idx >= expected:
                    break
            time.sleep(C.RIFE_POLL_SECONDS)

        rc = rife_backend.wait(rife_handle)
        rife_dt = time.time() - t_rife
        if rc != 0 and not C.shutdown.is_set():
            raise RuntimeError(f"RIFE failed for chunk {cid:04d} (rc={rc})")

        t_tail = time.time()
        writer.flush_remaining()
        with suppress(BrokenPipeError, OSError, ValueError):
            nvenc.stdin.close()
        nvenc.wait()
        encode_tail_dt = time.time() - t_tail
    except Exception:
        with suppress(Exception):
            rife_backend.terminate(rife_handle)
        with suppress(Exception):
            rife_backend.wait(rife_handle, timeout=5)
        with suppress(Exception):
            writer.flush_remaining()
        with suppress(Exception):
            nvenc.kill()
        with suppress(Exception):
            nvenc.wait(timeout=5)
        raise
    finally:
        if budget is not None:
            budget.clear_rife_ready(cid)
            budget.release_extract(cid)
        _cleanup_chunk_dir(rife_in.parent)

    if nvenc.returncode != 0 and not C.shutdown.is_set():
        raise RuntimeError(f"NVENC streaming encode failed (rc={nvenc.returncode})")

    total_dt = time.time() - t0
    stats = {
        "produced": writer.written,
        "total_seconds": total_dt,
        "rife_seconds": rife_dt,
        "readback_seconds": read_dt,
        "encode_tail_seconds": encode_tail_dt,
        "window_count": window_count,
        "window_avg_frames": (window_total / window_count) if window_count else 0.0,
        "window_max_frames": window_max,
        "rife_ready_peak_bytes": rife_ready_peak,
        "extract_peak_bytes": extract_peak,
        "nvenc_peak_frames": writer.max_buffered,
    }
    stats.update(_apply_rife_backend_metrics(metrics, cid, rife_backend))
    for key, value in esr_stats.items():
        stats[f"esrgan_{key}_seconds"] = value
    if metrics is not None:
        metrics.update(cid, **stats)
    return stats


def _write_pngs(frames: list[np.ndarray], out_dir: Path):
    """Write numpy frames as PNGs to tmpfs (for RIFE)."""
    from concurrent.futures import ThreadPoolExecutor
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = ThreadPoolExecutor(max_workers=C.WRITE_WORKERS)
    futs = []
    for i, img in enumerate(frames):
        dst = out_dir / f"{i+1:08d}.png"
        futs.append(pool.submit(
            cv2.imwrite, str(dst),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR)))
    for f in futs:
        f.result()
    pool.shutdown(wait=False)


def extract_worker(chunks, src: Path, w: int, h: int, fps: float,
                   do_rife: bool, prog: Progress, out_q: queue.Queue,
                   budget: _BudgetController | None,
                   metrics: _MetricsStore | None):
    """Stage 1: extract directly to the format the next stage needs."""
    frame_nbytes = _frame_bytes(w, h)
    for cid, start, dur in chunks:
        if C.shutdown.is_set():
            break

        print(f"\n  == Chunk {cid:04d}  [{start:.0f}s–{start+dur:.0f}s] ==", flush=True)

        if prog.done(cid, "encode"):
            out_q.put((cid, None, None))
            continue

        t0 = time.time()
        if do_rife:
            raw_dir = _tmpfs_chunk(cid) / "raw"
            shutil.rmtree(raw_dir, ignore_errors=True)
            reserved = max(1, int(round(dur * fps))) * frame_nbytes
            extract_peak = 0
            try:
                if budget is not None:
                    extract_total, _ = budget.reserve_extract(cid, reserved)
                    extract_peak = max(extract_peak, extract_total)
                n = extract_frames(src, start, dur, raw_dir, fps)
                actual_bytes = n * frame_nbytes
                if budget is not None:
                    extract_total, _ = budget.update_extract(cid, actual_bytes)
                    extract_peak = max(extract_peak, extract_total)
            except Exception:
                if budget is not None:
                    budget.release_extract(cid)
                raise
            dt = time.time() - t0
            if metrics is not None:
                metrics.update(
                    cid,
                    extract_seconds=dt,
                    extract_frames=n,
                    extract_fps=(n / dt) if dt > 0 else 0.0,
                    extract_bytes=actual_bytes,
                    extract_peak_bytes=extract_peak,
                )
            print(
                f"  [Extract→RIFE PNG] chunk {cid:04d}: "
                f"{n} frames ({dt:.1f}s, {n/dt:.0f}fps)",
                flush=True,
            )
            prog.mark(cid, "extract")
            out_q.put((cid, None, raw_dir))
            continue

        frames = extract_frames_to_ram(src, start, dur, w, h)
        dt = time.time() - t0
        if frames:
            n = len(frames)
            if metrics is not None:
                metrics.update(
                    cid,
                    extract_seconds=dt,
                    extract_frames=n,
                    extract_fps=(n / dt) if dt > 0 else 0.0,
                    extract_bytes=n * frame_nbytes,
                )
            print(f"  [Extract] chunk {cid:04d}: {n} frames ({dt:.1f}s, {n/dt:.0f}fps)", flush=True)
        prog.mark(cid, "extract")

        out_q.put((cid, frames, None))
    out_q.put(None)


def rife_first_worker(do_rife: bool, fps: float, prog: Progress,
                      in_q: queue.Queue, out_q: queue.Queue,
                      rife_backend_profile,
                      budget: _BudgetController | None,
                      metrics: _MetricsStore | None,
                      w: int, h: int):
    """Stage 2 (NEW ORDER): RIFE interpolation at ORIGINAL resolution (1260p, 3.8x faster than 4K).
    
    Receives raw extracted frames, writes them as PNGs, runs RIFE Vulkan, 
    reads interpolated frames back into RAM, and passes downstream to ESRGAN.
    """
    while True:
        item = in_q.get()
        if item is None:
            out_q.put(None)
            break
        if C.shutdown.is_set():
            out_q.put(None)
            break
            
        cid, frames, raw_dir = item
        if frames is None and raw_dir is None:
            out_q.put((cid, None))
            continue

        if not do_rife:
            # Pass through — no interpolation
            out_q.put((cid, frames))
            continue

        rife_backend = create_backend(rife_backend_profile)
        rife_in = raw_dir if raw_dir is not None else (_tmpfs_chunk(cid) / "raw")
        rife_out = _tmpfs_chunk(cid) / "rife"
        shutil.rmtree(rife_out, ignore_errors=True)
        if frames is not None:
            frame_count = len(frames)
            tw = time.time()
            _write_pngs(frames, rife_in)
            dt_write = time.time() - tw
            del frames
            gc.collect()
            print(
                f"  | RIFE input write: {frame_count} frames "
                f"({dt_write:.1f}s, {frame_count/dt_write:.1f}fps) chunk {cid:04d}",
                flush=True,
            )

        tr = time.time()
        n = rife_backend.interpolate_sync(rife_in, rife_out)
        prog.mark(cid, "rife")
        dt = time.time() - tr
        if metrics is not None:
            metrics.update(
                cid,
                rife_seconds=dt,
                rife_frames=n,
                rife_fps=(n / dt) if dt > 0 else 0.0,
            )
        _apply_rife_backend_metrics(metrics, cid, rife_backend)
        print(f"  | RIFE (1260p): {n} frames  ({dt:.1f}s, {n/dt:.1f}fps) chunk {cid:04d}", flush=True)
        if C.RIFE_CLEANUP_MODE != "none":
            shutil.rmtree(rife_in, ignore_errors=True)

        # Read interpolated frames back into RAM for ESRGAN
        from concurrent.futures import ThreadPoolExecutor
        png_files = [rife_out / f"{i:08d}.png" for i in range(1, _count_numbered_pngs(rife_out) + 1)]
        
        def _read_png(p):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
        
        tread = time.time()
        with ThreadPoolExecutor(max_workers=C.READ_WORKERS) as pool:
            interp_frames = list(pool.map(_read_png, png_files))
        dt_read = time.time() - tread
        interp_frames = [f for f in interp_frames if f is not None]
        if interp_frames:
            if metrics is not None:
                metrics.update(
                    cid,
                    readback_seconds=dt_read,
                    readback_frames=len(interp_frames),
                    readback_fps=(len(interp_frames) / dt_read) if dt_read > 0 else 0.0,
                )
            print(
                f"  | RIFE readback: {len(interp_frames)} frames "
                f"({dt_read:.1f}s, {len(interp_frames)/dt_read:.1f}fps) chunk {cid:04d}",
                flush=True,
            )
        
        # Cleanup RIFE tmpfs
        if budget is not None:
            budget.release_extract(cid)
        _cleanup_chunk_dir(_tmpfs_chunk(cid))
        
        out_q.put((cid, interp_frames))


def _report_progress(total: int, done_n: int, counter: list[int],
                     lock: threading.Lock, t_start: float):
    with lock:
        counter[0] += 1
        completed = done_n + counter[0]
        wall = time.time() - t_start
        avg = wall / max(counter[0], 1)
        rem = (total - completed) * avg
    print(f"  | Progress: {completed}/{total} ({100*completed/total:.1f}%)  ETA {rem/3600:.1f}h\n", flush=True)


def esrgan_worker(esr: ESRGANEngine | None, do_esr: bool, do_rife: bool,
                  fps: float, work: Path, prog: Progress,
                  in_q: queue.Queue, out_q: queue.Queue,
                  total: int, done_n: int, t_start: float,
                  counter: list[int], lock: threading.Lock,
                  w: int, h: int,
                  rife_backend_profile,
                  budget: _BudgetController | None,
                  metrics: _MetricsStore | None):
    """Stage 3: upscale frames and queue them for encode.

    T3 optimisation: when direct_rife_stream is in effect (do_rife + do_esr),
    the worker pre-starts RIFE for the *next* chunk while ESRGAN processes the
    current chunk.  Since RIFE runs on GPU1 via Vulkan and ESRGAN uses GPU0
    via CUDA, they execute truly in parallel on different GPUs.
    """
    carried_item = None
    prefetched: dict | None = None

    def _start_rife_prefetch(cid_pf: int, raw_dir_pf: Path) -> dict:
        """Start RIFE for the next chunk before the current ESRGAN begins."""
        backend = create_backend(rife_backend_profile)
        rife_out_pf = raw_dir_pf.parent / "rife"
        shutil.rmtree(rife_out_pf, ignore_errors=True)
        handle = backend.start_interpolate(raw_dir_pf, rife_out_pf)
        return {
            "backend": backend,
            "cid": cid_pf,
            "handle": handle,
            "out_dir": rife_out_pf,
            "started_at": time.time(),
        }

    while True:
        if carried_item is not None:
            item = carried_item
            carried_item = None
        else:
            item = in_q.get()
        if item is None:
            if prefetched is not None:
                backend = prefetched["backend"]
                with suppress(Exception):
                    backend.terminate(prefetched["handle"])
                with suppress(Exception):
                    backend.wait(prefetched["handle"], timeout=5)
            for _ in C.NVENC_GPUS:
                out_q.put(None)
            break
        if C.shutdown.is_set():
            if prefetched is not None:
                backend = prefetched["backend"]
                with suppress(Exception):
                    backend.terminate(prefetched["handle"])
                with suppress(Exception):
                    backend.wait(prefetched["handle"], timeout=5)
            for _ in C.NVENC_GPUS:
                out_q.put(None)
            break

        if len(item) == 3:
            cid, frames, raw_dir = item
        else:
            cid, frames = item
            raw_dir = None

        if prog.done(cid, "encode"):
            if frames is not None:
                del frames
                gc.collect()
            continue

        disk_dir = work / f"chunk_{cid:04d}"
        vid = disk_dir / "output.mp4"
        disk_dir.mkdir(parents=True, exist_ok=True)
        out_fps = fps * 2 if do_rife else fps
        n = len(frames) if frames is not None else 0

        if do_esr and esr and do_rife and raw_dir is not None:
            gpu = C.NVENC_GPUS[cid % len(C.NVENC_GPUS)]
            expected = create_backend(rife_backend_profile).expected_output_frames(
                _count_numbered_pngs(raw_dir)
            )

            current_prefetch = None
            if prefetched is not None:
                if prefetched["cid"] == cid:
                    current_prefetch = prefetched
                else:
                    backend = prefetched["backend"]
                    with suppress(Exception):
                        backend.terminate(prefetched["handle"])
                    with suppress(Exception):
                        backend.wait(prefetched["handle"], timeout=5)
                prefetched = None

            if carried_item is None:
                try:
                    carried_item = in_q.get_nowait()
                except queue.Empty:
                    carried_item = None

            if carried_item is not None:
                if len(carried_item) == 3:
                    next_cid, _, next_raw = carried_item
                else:
                    next_cid, next_raw = carried_item[0], None
                if (
                    next_raw is not None
                    and next_cid is not None
                    and not prog.done(next_cid, "encode")
                ):
                    prefetched = _start_rife_prefetch(next_cid, next_raw)

            stats = _stream_rife_esrgan_to_nvenc(
                esr,
                current_prefetch["backend"] if current_prefetch is not None else create_backend(rife_backend_profile),
                raw_dir,
                vid,
                out_fps,
                gpu,
                cid,
                w,
                h,
                budget,
                metrics,
                prefetched=current_prefetch,
            )
            prog.mark(cid, "rife")
            prog.mark(cid, "esrgan")
            prog.mark(cid, "encode")
            prog.mark(cid, "clean")
            produced = int(stats["produced"])
            dt_total = stats["total_seconds"]
            dt_rife = stats["rife_seconds"]
            dt_read = stats["readback_seconds"]
            if produced != expected and not C.shutdown.is_set():
                raise RuntimeError(
                    f"Streaming RIFE/ESRGAN/NVENC produced {produced} frames, expected {expected}"
                )
            if metrics is not None:
                metrics.update(
                    cid,
                    rife_seconds=dt_rife,
                    rife_frames=expected,
                    rife_fps=(expected / dt_rife) if dt_rife > 0 else 0.0,
                    readback_seconds=dt_read,
                    readback_frames=expected,
                    readback_fps=(expected / dt_read) if dt_read > 0 else 0.0,
                    esrgan_seconds=sum(
                        float(stats.get(key, 0.0))
                        for key in (
                            "esrgan_fill_seconds",
                            "esrgan_h2d_seconds",
                            "esrgan_downscale_seconds",
                            "esrgan_infer_seconds",
                            "esrgan_d2h_seconds",
                        )
                    ),
                    encode_seconds=float(stats.get("esrgan_writer_wait_seconds", 0.0))
                    + float(stats.get("encode_tail_seconds", 0.0)),
                    effective_fps=(produced / dt_total) if dt_total > 0 else 0.0,
                )
            record = metrics.snapshot(cid) if metrics is not None else {}
            print(
                f"  [Chunk {cid:04d} summary] "
                f"extract={record.get('extract_seconds', 0.0):.1f}s "
                f"rife={dt_rife:.1f}s ({record.get('rife_fps', 0.0):.1f}fps) "
                f"read={dt_read:.1f}s ({record.get('readback_fps', 0.0):.1f}fps) "
                f"esr={record.get('esrgan_seconds', 0.0):.1f}s "
                f"encode={record.get('encode_seconds', 0.0):.1f}s "
                f"total={dt_total:.1f}s "
                f"win_avg={stats['window_avg_frames']:.1f} "
                f"win_max={int(stats['window_max_frames'])} "
                f"backlog_extract={stats['extract_peak_bytes'] / 1024**3:.2f}GiB "
                f"backlog_rife={stats['rife_ready_peak_bytes'] / 1024**3:.2f}GiB "
                f"nvenc_buf={int(stats['nvenc_peak_frames'])}",
                flush=True,
            )
            print(
                f"  [RIFE→ESRGAN→NVENC@GPU{gpu}] chunk {cid:04d}: "
                f"{produced} frames ({dt_total:.1f}s, {produced/dt_total:.1f}fps)",
                flush=True,
            )
            if metrics is not None:
                metrics.emit(cid)
            _report_progress(total, done_n, counter, lock, t_start)
            continue

        if frames is None:
            continue

        if do_esr and esr:
            gpu = C.NVENC_GPUS[cid % len(C.NVENC_GPUS)]
            produced, dt_esr = _stream_esrgan_to_nvenc(esr, frames, vid, out_fps, gpu)
            prog.mark(cid, "esrgan")
            del frames
            gc.collect()
            if produced != n:
                raise RuntimeError(
                    f"Streaming ESRGAN/NVENC produced {produced} frames, expected {n}"
                )
            prog.mark(cid, "encode")
            prog.mark(cid, "clean")
            if metrics is not None:
                metrics.update(
                    cid,
                    esrgan_seconds=dt_esr,
                    encode_seconds=dt_esr,
                    effective_fps=(n / dt_esr) if dt_esr > 0 else 0.0,
                )
                metrics.emit(cid)
            print(
                f"  [ESRGAN→NVENC@GPU{gpu}] chunk {cid:04d}: "
                f"{n} frames ({dt_esr:.1f}s, {n/dt_esr:.1f}fps)",
                flush=True,
            )
            _report_progress(total, done_n, counter, lock, t_start)
            continue
        else:
            enc_frames = frames

        out_q.put((cid, enc_frames, vid, out_fps))
        gc.collect()


def encode_worker(worker_idx: int, gpu: int, prog: Progress,
                  in_q: queue.Queue, total: int, done_n: int,
                  t_start: float, counter: list[int], lock: threading.Lock,
                  metrics: _MetricsStore | None):
    """Stage 4: encode queued numpy frames with a dedicated NVENC device."""
    label = f"NVENC{worker_idx}@GPU{gpu}"
    while True:
        item = in_q.get()
        if item is None:
            break
        if C.shutdown.is_set():
            break

        cid, frames, vid, out_fps = item
        if not frames:
            _report_progress(total, done_n, counter, lock, t_start)
            continue
        frames = [frame for frame in frames if frame is not None]
        if not frames:
            raise RuntimeError(f"{label} received an empty frame set for chunk {cid:04d}")

        h, w = frames[0].shape[:2]
        n = len(frames)
        proc = _open_nvenc_pipe(vid, w, h, out_fps, gpu)
        t0 = time.time()
        try:
            for frame in frames:
                if frame is not None:
                    proc.stdin.write(memoryview(np.ascontiguousarray(frame)))
            proc.stdin.close()
        except BrokenPipeError:
            pass
        except Exception:
            try:
                proc.stdin.close()
            except Exception:
                pass
            raise

        proc.wait()
        dt = time.time() - t0
        del frames
        gc.collect()

        if proc.returncode != 0:
            raise RuntimeError(f"{label} failed for chunk {cid:04d} (rc={proc.returncode})")

        prog.mark(cid, "encode")
        prog.mark(cid, "clean")
        if metrics is not None:
            metrics.update(
                cid,
                encode_seconds=dt,
                encode_frames=n,
                encode_fps=(n / dt) if dt > 0 else 0.0,
                effective_fps=(n / dt) if dt > 0 else 0.0,
            )
            metrics.emit(cid)
        print(f"  [{label}] chunk {cid:04d}: {n} frames ({dt:.1f}s, {n/dt:.1f}fps)", flush=True)
        _report_progress(total, done_n, counter, lock, t_start)


def run(chunks, src: Path, work: Path, prog: Progress,
        do_esr: bool, do_rife: bool, fps: float,
        esr: ESRGANEngine | None, w: int = 2240, h: int = 1260,
        visual_profile=None, audio_profile=None,
        scheduler_profile=None, rife_backend_profile=None):
    """Pipeline: Extract → RIFE(1260p) → ESRGAN → NVENC."""
    total = len(chunks)
    done_n = sum(1 for c in chunks if prog.done(c[0], "encode"))
    pending = [c for c in chunks if not prog.done(c[0], "encode")]
    
    if not pending:
        print(f"  All {total} chunks already processed!")
        return done_n

    Path(C.TMPFS_WORK).mkdir(parents=True, exist_ok=True)
    budget = _BudgetController()
    metrics = _MetricsStore(work)
    backend_name = create_backend(rife_backend_profile).name()

    # Record profile identifiers in metrics store for all chunks
    profile_tags = {}
    if visual_profile:
        profile_tags["visual_profile"] = visual_profile.name
    if audio_profile:
        profile_tags["audio_profile"] = audio_profile.name
    if scheduler_profile:
        profile_tags["scheduler_profile"] = scheduler_profile.name
    profile_tags["rife_backend"] = backend_name

    # Inject profile tags into metrics for every pending chunk
    for cid, _, _ in pending:
        if profile_tags:
            metrics.update(cid, **profile_tags)

    q_extract = queue.Queue(maxsize=C.PIPELINE_DEPTH)
    q_rife = queue.Queue(maxsize=C.PIPELINE_DEPTH)
    q_encode = queue.Queue(maxsize=max(C.PIPELINE_DEPTH, 1))
    direct_rife_stream = do_rife and do_esr

    t_start = time.time()
    encode_counter = [0]
    encode_lock = threading.Lock()
    errors: list[BaseException] = []

    def spawn(name: str, target, *args) -> threading.Thread:
        def runner():
            try:
                target(*args)
            except BaseException as exc:
                print(f"  [!] {name} failed: {exc}", flush=True)
                traceback.print_exc()
                errors.append(exc)
                C.shutdown.set()
        return threading.Thread(target=runner, name=name)

    t_ext = spawn(
        "Extractor Thread",
        extract_worker,
        pending,
        src,
        w,
        h,
        fps,
        do_rife,
        prog,
        q_extract,
        budget,
        metrics,
    )
    t_rife = None if direct_rife_stream else spawn(
        "RIFE Interpolator (1260p)",
        rife_first_worker,
        do_rife,
        fps,
        prog,
        q_extract,
        q_rife,
        rife_backend_profile,
        budget,
        metrics,
        w,
        h,
    )
    t_esr = spawn(
        "ESRGAN Upscaler",
        esrgan_worker,
        esr,
        do_esr,
        do_rife,
        fps,
        work,
        prog,
        q_extract if direct_rife_stream else q_rife,
        q_encode,
        total,
        done_n,
        t_start,
        encode_counter,
        encode_lock,
        w,
        h,
        rife_backend_profile,
        budget,
        metrics,
    )
    encode_threads = [
        spawn(
            f"NVENC Worker {idx}",
            encode_worker,
            idx, gpu, prog, q_encode, total, done_n, t_start,
            encode_counter, encode_lock, metrics,
        )
        for idx, gpu in enumerate(C.NVENC_GPUS)
    ]

    t_ext.start()
    if t_rife is not None:
        t_rife.start()
    t_esr.start()
    for thread in encode_threads:
        thread.start()

    all_threads = [t_ext, t_esr, *encode_threads]
    if t_rife is not None:
        all_threads.insert(1, t_rife)
    sentinels_pushed = False
    while any(thread.is_alive() for thread in all_threads):
        if errors and not sentinels_pushed:
            sentinels_pushed = True
            with suppress(queue.Full):
                q_extract.put_nowait(None)
            if t_rife is not None:
                with suppress(queue.Full):
                    q_rife.put_nowait(None)
            for _ in encode_threads:
                with suppress(queue.Full):
                    q_encode.put_nowait(None)
        for thread in all_threads:
            thread.join(timeout=0.2)

    if errors:
        raise RuntimeError(f"Pipeline worker failed: {errors[0]}") from errors[0]

    # Recalculate done chunks post-execution
    done_n = sum(1 for c in chunks if prog.done(c[0], "encode"))
    return done_n
