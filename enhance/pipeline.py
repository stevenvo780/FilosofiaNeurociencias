"""
4-stage pipeline: extract → ESRGAN → RIFE → NVENC
Each stage on different silicon, overlapping across chunks via Queues.

Streaming path (no RIFE):
  ffmpeg pipe → numpy RAM → GPU ESRGAN → [reorder buffer] → ffmpeg NVENC pipe

RIFE path (needs files):
  ffmpeg pipe → numpy RAM → GPU ESRGAN → collect all → PNGs on tmpfs → RIFE → NVENC

Queue architecture overlaps Extract (CPU), ESRGAN (GPU0/1 PyTorch), RIFE (GPU0/1 Vulkan), and NVENC (Hardware ASIC).
"""
import time, shutil, threading, subprocess, gc, queue
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from .progress import Progress
from .ffmpeg_utils import (extract_frames, extract_frames_to_ram,
                           probe, nvenc_encode)
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
        "-preset", "p6", "-tune", "hq",
        "-rc", "vbr", "-cq", "20",
        "-b:v", "12M", "-maxrate", "18M", "-bufsize", "24M",
        "-profile:v", "main10", "-pix_fmt", "p010le",
        str(out_file), "-loglevel", "warning",
    ]
    frame_bytes = w * h * 3
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            bufsize=frame_bytes * 8)
    try:
        F_SETPIPE_SZ = 1031
        fcntl.fcntl(proc.stdin.fileno(), F_SETPIPE_SZ, 1 << 20)  # 1 MB
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

        frames = None
        if not prog.done(cid, "extract") or (do_esr and not prog.done(cid, "esrgan")):
            t0 = time.time()
            frames = extract_frames_to_ram(src, start, dur, w, h)
            dt = time.time() - t0
            if frames:
                n = len(frames)
                print(f"  [Extract] chunk {cid:04d}: {n} frames ({dt:.1f}s, {n/dt:.0f}fps)", flush=True)
            prog.mark(cid, "extract")
        
        out_q.put((cid, frames))
    out_q.put(None)


def esrgan_worker(esr: ESRGANEngine | None, do_esr: bool, do_rife: bool, fps: float, 
                  work: Path, prog: Progress, out_w: int, out_h: int, 
                  in_q: queue.Queue, out_q: queue.Queue):
    """Stage 2: Enhances structures asynchronously utilizing 2x GPUs via local streaming pipes or memory blocks."""
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

        if do_rife:
            esr_frames = None
            if do_esr and esr and not prog.done(cid, "esrgan"):
                esr_frames = esr.process_frames(frames, out_dir=None)
                prog.mark(cid, "esrgan")
            
            src_frames = esr_frames if esr_frames is not None else frames
            out_q.put((cid, src_frames))
            del src_frames, esr_frames, frames
            gc.collect()

        else:
            # Streaming path
            disk_dir = work / f"chunk_{cid:04d}"
            vid = disk_dir / "output.mp4"
            if do_esr and esr and not prog.done(cid, "esrgan"):
                n = len(frames)
                proc = _open_nvenc_pipe(vid, out_w, out_h, fps, C.NVENC_GPU)
                writer = _ReorderWriter(proc.stdin, n)
                t0 = time.time()
                esr.process_streaming(frames, writer.on_frame)
                writer.flush_remaining()
                try: proc.stdin.close()
                except: pass
                proc.wait()
                dt = time.time() - t0
                if proc.returncode == 0:
                    prog.mark(cid, "esrgan")
                    prog.mark(cid, "encode")
                    prog.mark(cid, "clean")
                    print(f"  [Stream] chunk {cid:04d}: {writer.written}/{n} frames → NVENC  ({dt:.1f}s, {n/dt:.1f}fps)", flush=True)
            elif not do_esr:
                n = len(frames)
                proc = _open_nvenc_pipe(vid, out_w, out_h, fps, C.NVENC_GPU)
                t0 = time.time()
                for f in frames:
                    proc.stdin.write(memoryview(np.ascontiguousarray(f)))
                try: proc.stdin.close()
                except: pass
                proc.wait()
                dt = time.time() - t0
                if proc.returncode == 0:
                    prog.mark(cid, "encode")
                    prog.mark(cid, "clean")
                    print(f"  [Encode] chunk {cid:04d}: {n} frames ({dt:.1f}s)", flush=True)
            
            del frames
            gc.collect()
            out_q.put((cid, None))


def rife_worker(do_rife: bool, fps: float, work: Path, prog: Progress,
               in_q: queue.Queue, total: int, done_n: int, t_start: float):
    """Stage 3: Manages local file dumps, frame interpolations internally mapping Vulkan allocations, and hard MP4 encodings."""
    completed = done_n
    while True:
        item = in_q.get()
        if item is None:
            break
        if C.shutdown.is_set():
            continue
            
        cid, frames = item
        if frames is None or prog.done(cid, "encode") or not do_rife:
            if frames is not None:
                del frames
                gc.collect()
            completed += 1
            wall = time.time() - t_start
            avg = wall / max(completed - done_n, 1)
            rem = (total - completed) * avg
            print(f"  | Progress: {completed}/{total} ({100*completed/total:.1f}%)  ETA {rem/3600:.1f}h\n", flush=True)
            continue

        rife_in = _tmpfs_chunk(cid) / "esrgan"
        rife_out = _tmpfs_chunk(cid) / "rife"
        disk_dir = work / f"chunk_{cid:04d}"
        vid = disk_dir / "output.mp4"

        if not prog.done(cid, "rife"):
            _write_pngs(frames, rife_in)
            del frames
            gc.collect()
            
            tr = time.time()
            n = rife_interpolate(rife_in, rife_out)
            prog.mark(cid, "rife")
            print(f"  | RIFE:   {n} frames  ({time.time()-tr:.1f}s) chunk {cid:04d}", flush=True)
            shutil.rmtree(rife_in, ignore_errors=True)
        else:
            if frames is not None:
                del frames
                gc.collect()

        out_fps = fps * 2
        if not prog.done(cid, "encode"):
            disk_dir.mkdir(parents=True, exist_ok=True)
            try:
                t0 = time.time()
                nvenc_encode(rife_out, vid, out_fps)
                prog.mark(cid, "encode")
                print(f"  [NVENC] chunk {cid:04d} encoded ({time.time()-t0:.1f}s)", flush=True)
            except Exception as e:
                print(f"  [!] Encode chunk {cid:04d} failed: {e}")

        if vid.exists() and vid.stat().st_size > 1000:
            shutil.rmtree(_tmpfs_chunk(cid), ignore_errors=True)
            prog.mark(cid, "clean")

        completed += 1
        wall = time.time() - t_start
        avg = wall / max(completed - done_n, 1)
        rem = (total - completed) * avg
        print(f"  | Progress: {completed}/{total} ({100*completed/total:.1f}%)  ETA {rem/3600:.1f}h\n", flush=True)


def run(chunks, src: Path, work: Path, prog: Progress,
        do_esr: bool, do_rife: bool, fps: float,
        esr: ESRGANEngine | None, w: int = 2240, h: int = 1260):
    """Spins up concurrent overlapping pipeline stages interconnected by capacity-capped Queues."""
    total = len(chunks)
    done_n = sum(1 for c in chunks if prog.done(c[0], "encode"))
    pending = [c for c in chunks if not prog.done(c[0], "encode")]
    
    if not pending:
        print(f"  All {total} chunks already processed!")
        return done_n

    Path(C.TMPFS_WORK).mkdir(parents=True, exist_ok=True)

    out_w = w * 2 if do_esr else w
    out_h = h * 2 if do_esr else h
    
    # maxsize=1 pushes active backpressure: Threads wait if upstream finishes too quickly
    q_extract = queue.Queue(maxsize=1)
    q_esrgan  = queue.Queue(maxsize=1)

    t_start = time.time()

    t_ext = threading.Thread(
        target=extract_worker,
        args=(pending, src, w, h, do_esr, prog, q_extract),
        name="Extractor Thread"
    )
    t_esr = threading.Thread(
        target=esrgan_worker,
        args=(esr, do_esr, do_rife, fps, work, prog, out_w, out_h, q_extract, q_esrgan),
        name="ESRGAN Engine Thread"
    )
    t_rife = threading.Thread(
        target=rife_worker,
        args=(do_rife, fps, work, prog, q_esrgan, total, done_n, t_start),
        name="RIFE & Encode Subprocessor"
    )

    t_ext.start()
    t_esr.start()
    t_rife.start()

    t_ext.join()
    t_esr.join()
    t_rife.join()

    # Recalculate done chunks post-execution
    done_n = sum(1 for c in chunks if prog.done(c[0], "encode"))
    return done_n

