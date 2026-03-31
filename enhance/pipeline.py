"""
4-stage pipeline: extract → RIFE → ESRGAN → NVENC.

Chunk queues overlap CPU decode, Vulkan interpolation, dual-GPU ESRGAN and
hardware NVENC encode across chunk boundaries.
"""
import gc
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from .progress import Progress
from .ffmpeg_utils import extract_frames_to_ram
from .esrgan import ESRGANEngine
from .rife import interpolate as rife_interpolate


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
    MAX_BUFFERED = 48  

    __slots__ = ("pipe", "lock", "cond", "buf", "next_idx", "total",
                 "written", "_finished", "_writer_thread", "_error", "_sem")

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
        self._sem = threading.Semaphore(self.MAX_BUFFERED)
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="nvenc-writer")
        self._writer_thread.start()

    def on_frame(self, idx: int, frame: np.ndarray):
        """Callback from ESRGAN GPU workers — thread-safe, minimal blocking."""
        self._sem.acquire()  
        arr = np.ascontiguousarray(frame).copy()  
        with self.cond:
            self.buf[idx] = arr
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
        if self._error:
            print(f"    [!] ReorderWriter error: {self._error}", flush=True)
        with self.lock:
            if self.buf:
                print(f"    [!] ReorderWriter: {len(self.buf)} orphan frames "
                      f"(missing indices before them)", flush=True)


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


def extract_worker(chunks, src: Path, w: int, h: int, do_esr: bool, prog: Progress, out_q: queue.Queue):
    """Stage 1: Extracts frame arrays sequentially into dynamic memory via ffmpeg decoding."""
    for cid, start, dur in chunks:
        if C.shutdown.is_set(): break

        print(f"\n  == Chunk {cid:04d}  [{start:.0f}s–{start+dur:.0f}s] ==", flush=True)

        if prog.done(cid, "encode"):
            out_q.put((cid, None))
            continue

        t0 = time.time()
        frames = extract_frames_to_ram(src, start, dur, w, h)
        dt = time.time() - t0
        if frames:
            n = len(frames)
            print(f"  [Extract] chunk {cid:04d}: {n} frames ({dt:.1f}s, {n/dt:.0f}fps)", flush=True)
        prog.mark(cid, "extract")
        
        out_q.put((cid, frames))
    out_q.put(None)


def rife_first_worker(do_rife: bool, fps: float, prog: Progress,
                      in_q: queue.Queue, out_q: queue.Queue):
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
            continue
            
        cid, frames = item
        if frames is None or prog.done(cid, "encode"):
            out_q.put((cid, None))
            continue

        if not do_rife:
            # Pass through — no interpolation
            out_q.put((cid, frames))
            continue

        rife_in = _tmpfs_chunk(cid) / "raw"
        rife_out = _tmpfs_chunk(cid) / "rife"

        # RIFE intermediates live on tmpfs, so this stage must rerun unless
        # the chunk is already encoded.
        _write_pngs(frames, rife_in)
        del frames
        gc.collect()
        
        tr = time.time()
        n = rife_interpolate(rife_in, rife_out)
        prog.mark(cid, "rife")
        dt = time.time() - tr
        print(f"  | RIFE (1260p): {n} frames  ({dt:.1f}s, {n/dt:.1f}fps) chunk {cid:04d}", flush=True)
        shutil.rmtree(rife_in, ignore_errors=True)

        # Read interpolated frames back into RAM for ESRGAN
        from concurrent.futures import ThreadPoolExecutor
        png_files = sorted(rife_out.glob("*.png"))
        
        def _read_png(p):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
        
        with ThreadPoolExecutor(max_workers=C.READ_WORKERS) as pool:
            interp_frames = list(pool.map(_read_png, png_files))
        interp_frames = [f for f in interp_frames if f is not None]
        
        # Cleanup RIFE tmpfs
        shutil.rmtree(_tmpfs_chunk(cid), ignore_errors=True)
        
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
                  in_q: queue.Queue, out_q: queue.Queue):
    """Stage 3: upscale frames and queue them for encode."""
    while True:
        item = in_q.get()
        if item is None:
            for _ in C.NVENC_GPUS:
                out_q.put(None)
            break
        if C.shutdown.is_set():
            continue

        cid, frames = item
        if frames is None or prog.done(cid, "encode"):
            if frames is not None:
                del frames
                gc.collect()
            continue

        disk_dir = work / f"chunk_{cid:04d}"
        vid = disk_dir / "output.mp4"
        disk_dir.mkdir(parents=True, exist_ok=True)
        out_fps = fps * 2 if do_rife else fps
        n = len(frames)

        if do_esr and esr:
            t0 = time.time()
            enc_frames = esr.process_frames(frames, out_dir=None)
            dt_esr = time.time() - t0
            prog.mark(cid, "esrgan")
            print(f"  [ESRGAN] chunk {cid:04d}: {n} frames ({dt_esr:.1f}s, {n/dt_esr:.1f}fps)", flush=True)
            del frames
        else:
            enc_frames = frames

        out_q.put((cid, enc_frames, vid, out_fps))
        gc.collect()


def encode_worker(worker_idx: int, gpu: int, prog: Progress,
                  in_q: queue.Queue, total: int, done_n: int,
                  t_start: float, counter: list[int], lock: threading.Lock):
    """Stage 4: encode queued numpy frames with a dedicated NVENC device."""
    label = f"NVENC{worker_idx}@GPU{gpu}"
    while True:
        item = in_q.get()
        if item is None:
            break
        if C.shutdown.is_set():
            continue

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
        print(f"  [{label}] chunk {cid:04d}: {n} frames ({dt:.1f}s, {n/dt:.1f}fps)", flush=True)
        _report_progress(total, done_n, counter, lock, t_start)


def run(chunks, src: Path, work: Path, prog: Progress,
        do_esr: bool, do_rife: bool, fps: float,
        esr: ESRGANEngine | None, w: int = 2240, h: int = 1260):
    """Pipeline: Extract → RIFE(1260p) → ESRGAN → NVENC."""
    total = len(chunks)
    done_n = sum(1 for c in chunks if prog.done(c[0], "encode"))
    pending = [c for c in chunks if not prog.done(c[0], "encode")]
    
    if not pending:
        print(f"  All {total} chunks already processed!")
        return done_n

    Path(C.TMPFS_WORK).mkdir(parents=True, exist_ok=True)
    q_extract = queue.Queue(maxsize=C.PIPELINE_DEPTH)
    q_rife = queue.Queue(maxsize=C.PIPELINE_DEPTH)
    q_encode = queue.Queue(maxsize=1)

    t_start = time.time()
    encode_counter = [0]
    encode_lock = threading.Lock()
    errors: list[BaseException] = []

    def spawn(name: str, target, *args) -> threading.Thread:
        def runner():
            try:
                target(*args)
            except BaseException as exc:
                errors.append(exc)
                C.shutdown.set()
        return threading.Thread(target=runner, name=name)

    t_ext = spawn("Extractor Thread", extract_worker, pending, src, w, h, do_esr, prog, q_extract)
    t_rife = spawn("RIFE Interpolator (1260p)", rife_first_worker, do_rife, fps, prog, q_extract, q_rife)
    t_esr = spawn("ESRGAN Upscaler", esrgan_worker, esr, do_esr, do_rife, fps, work, prog, q_rife, q_encode)
    encode_threads = [
        spawn(
            f"NVENC Worker {idx}",
            encode_worker,
            idx, gpu, prog, q_encode, total, done_n, t_start,
            encode_counter, encode_lock,
        )
        for idx, gpu in enumerate(C.NVENC_GPUS)
    ]

    t_ext.start()
    t_rife.start()
    t_esr.start()
    for thread in encode_threads:
        thread.start()

    t_ext.join()
    t_rife.join()
    t_esr.join()
    for thread in encode_threads:
        thread.join()

    if errors:
        raise RuntimeError(f"Pipeline worker failed: {errors[0]}") from errors[0]

    # Recalculate done chunks post-execution
    done_n = sum(1 for c in chunks if prog.done(c[0], "encode"))
    return done_n
