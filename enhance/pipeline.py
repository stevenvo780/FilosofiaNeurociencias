"""
4-stage pipeline: extract → ESRGAN → RIFE → NVENC
Each stage on different silicon, overlapping across chunks.

Streaming path (no RIFE):
  ffmpeg pipe → numpy RAM → GPU ESRGAN → [reorder buffer] → ffmpeg NVENC pipe
  Peak RAM = input_frames + 1 ESRGAN batch + reorder buffer (~12 frames)
  No PNG touches disk. No enc_q. No output accumulation.

RIFE path (needs files):
  ffmpeg pipe → numpy RAM → GPU ESRGAN → collect all → PNGs on tmpfs → RIFE → NVENC
"""
import time, shutil, threading, subprocess, gc
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
    This collects out-of-order frames in a tiny dict buffer and flushes them
    to the pipe as soon as the next expected index is available.

    Peak buffer: at most ~(GPU0_BATCH + GPU1_BATCH) frames × 33.9 MB ≈ 407 MB.
    """
    __slots__ = ("pipe", "lock", "buf", "next_idx", "total", "written")

    def __init__(self, pipe_stdin, total: int):
        self.pipe = pipe_stdin
        self.lock = threading.Lock()
        self.buf: dict[int, np.ndarray] = {}
        self.next_idx = 0
        self.total = total
        self.written = 0

    def on_frame(self, idx: int, frame: np.ndarray):
        """Callback from ESRGAN GPU workers — thread-safe."""
        with self.lock:
            if idx == self.next_idx:
                # Fast path: write immediately
                self._write(frame)
                self.next_idx += 1
                # Flush any buffered sequential frames
                while self.next_idx in self.buf:
                    self._write(self.buf.pop(self.next_idx))
                    self.next_idx += 1
            else:
                # Out of order — buffer it
                self.buf[idx] = frame

    def _write(self, frame: np.ndarray):
        """Write one frame to NVENC pipe. Caller holds lock."""
        try:
            self.pipe.write(memoryview(np.ascontiguousarray(frame)))
            self.written += 1
        except BrokenPipeError:
            pass

    def flush_remaining(self):
        """Flush any leftover buffered frames (shouldn't happen normally)."""
        with self.lock:
            while self.next_idx in self.buf:
                self._write(self.buf.pop(self.next_idx))
                self.next_idx += 1
            if self.buf:
                print(f"    [!] ReorderWriter: {len(self.buf)} orphan frames "
                      f"(missing indices before them)", flush=True)


def _open_nvenc_pipe(out_file: Path, w: int, h: int, fps: float,
                     gpu: int = 0) -> subprocess.Popen:
    """Open an ffmpeg NVENC subprocess that accepts raw RGB24 on stdin."""
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
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            bufsize=w * h * 3 * 8)


def _encode_from_numpy(frames: list[np.ndarray], out_file: Path,
                        fps: float, gpu: int = 0):
    """Pipe numpy RGB frames directly to NVENC — zero PNG I/O.
    Used by tests and RIFE path. For streaming, use _open_nvenc_pipe."""
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


def run(chunks, src: Path, work: Path, prog: Progress,
        do_esr: bool, do_rife: bool, fps: float,
        esr: ESRGANEngine | None, w: int = 2240, h: int = 1260):
    """Process all chunks.

    Streaming (no RIFE): ESRGAN → reorder buffer → NVENC pipe. Zero accumulation.
    RIFE path: collect all ESRGAN output → PNGs → RIFE → NVENC from dir.
    """
    total = len(chunks)
    done_n = sum(1 for c in chunks if prog.done(c[0], "encode"))
    pending = [c for c in chunks if not prog.done(c[0], "encode")]
    if not pending:
        print(f"  All {total} chunks already processed!")
        return done_n

    Path(C.TMPFS_WORK).mkdir(parents=True, exist_ok=True)

    # Output dimensions after ESRGAN (4x upscale on 0.5x downscaled input = 2x)
    out_w = w * 2 if do_esr else w
    out_h = h * 2 if do_esr else h

    t_start = time.time()
    completed = done_n

    for cid, start, dur in pending:
        if C.shutdown.is_set():
            break

        tc = time.time()
        print(f"\n  == Chunk {cid:04d}  [{start:.0f}s–{start+dur:.0f}s] ==",
              flush=True)

        # ── Stage 1: Extract → numpy arrays in RAM ──
        frames = None
        if not prog.done(cid, "extract"):
            t0 = time.time()
            frames = extract_frames_to_ram(src, start, dur, w, h)
            dt = time.time() - t0
            n = len(frames)
            print(f"  [Extract] chunk {cid:04d}: {n} frames ({dt:.1f}s, "
                  f"{n/dt:.0f}fps)", flush=True)
            prog.mark(cid, "extract")
        elif do_esr and not prog.done(cid, "esrgan"):
            frames = extract_frames_to_ram(src, start, dur, w, h)

        if do_rife:
            # ── RIFE PATH: collect all → PNGs → RIFE → NVENC from dir ──
            _process_chunk_rife(
                cid, frames, esr, do_esr, fps, work, prog, w, h)
        else:
            # ── STREAMING PATH: ESRGAN → reorder → NVENC pipe ──
            _process_chunk_streaming(
                cid, frames, esr, do_esr, fps, work, prog,
                out_w, out_h)

        # Free input frames explicitly
        del frames
        gc.collect()

        completed += 1
        wall = time.time() - t_start
        avg = wall / max(completed - done_n, 1)
        rem = (total - completed) * avg
        print(f"  | Progress: {completed}/{total} ({100*completed/total:.1f}%)  "
              f"ETA {rem/3600:.1f}h", flush=True)

    return completed


def _process_chunk_streaming(cid: int, frames: list | None,
                             esr: ESRGANEngine | None, do_esr: bool,
                             fps: float, work: Path, prog: Progress,
                             out_w: int, out_h: int):
    """Streaming: ESRGAN batches → reorder buffer → NVENC pipe.

    Peak RAM: input frames + ~12 reorder frames + pipe buffer.
    No output accumulation.
    """
    disk_dir = work / f"chunk_{cid:04d}"
    vid = disk_dir / "output.mp4"

    if prog.done(cid, "encode"):
        return

    if do_esr and esr and not prog.done(cid, "esrgan"):
        if frames is None:
            return

        n = len(frames)
        # Open NVENC pipe BEFORE ESRGAN starts
        proc = _open_nvenc_pipe(vid, out_w, out_h, fps, C.NVENC_GPU)
        writer = _ReorderWriter(proc.stdin, n)

        t0 = time.time()
        esr.process_streaming(frames, writer.on_frame)
        writer.flush_remaining()

        # Close pipe and wait for NVENC to finish
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass
        proc.wait()

        dt = time.time() - t0
        if proc.returncode != 0:
            print(f"  [!] NVENC chunk {cid:04d} failed (rc={proc.returncode})",
                  flush=True)
        else:
            prog.mark(cid, "esrgan")
            prog.mark(cid, "encode")
            prog.mark(cid, "clean")
            print(f"  [Stream] chunk {cid:04d}: {writer.written}/{n} frames "
                  f"→ NVENC  ({dt:.1f}s, {n/dt:.1f}fps)", flush=True)

    elif not do_esr and frames is not None:
        # No ESRGAN — pipe raw frames directly
        n = len(frames)
        proc = _open_nvenc_pipe(vid, out_w, out_h, fps, C.NVENC_GPU)
        t0 = time.time()
        try:
            for f in frames:
                proc.stdin.write(memoryview(np.ascontiguousarray(f)))
            proc.stdin.close()
        except BrokenPipeError:
            pass
        proc.wait()
        dt = time.time() - t0
        if proc.returncode == 0:
            prog.mark(cid, "encode")
            prog.mark(cid, "clean")
            print(f"  [Encode] chunk {cid:04d}: {n} frames ({dt:.1f}s)",
                  flush=True)


def _process_chunk_rife(cid: int, frames: list | None,
                        esr: ESRGANEngine | None, do_esr: bool,
                        fps: float, work: Path, prog: Progress,
                        w: int, h: int):
    """RIFE path: collect all ESRGAN output → PNGs → RIFE → NVENC from dir."""
    rife_in = _tmpfs_chunk(cid) / "esrgan"
    rife_out = _tmpfs_chunk(cid) / "rife"
    disk_dir = work / f"chunk_{cid:04d}"
    vid = disk_dir / "output.mp4"

    # ESRGAN (collect all — RIFE needs PNGs)
    esr_frames = None
    if do_esr and esr and not prog.done(cid, "esrgan"):
        if frames is None:
            return
        esr_frames = esr.process_frames(frames, out_dir=None)
        if C.shutdown.is_set():
            return
        prog.mark(cid, "esrgan")
        # Free raw input immediately
        # (caller will also del frames, but clear our ref)

    # RIFE
    out_fps = fps
    if not prog.done(cid, "rife"):
        src_frames = esr_frames if esr_frames is not None else frames
        if src_frames is not None:
            _write_pngs(src_frames, rife_in)
            del src_frames, esr_frames
            esr_frames = None
            gc.collect()
        tr = time.time()
        n = rife_interpolate(rife_in, rife_out)
        prog.mark(cid, "rife")
        print(f"  | RIFE:   {n} frames  ({time.time()-tr:.1f}s)", flush=True)
        shutil.rmtree(rife_in, ignore_errors=True)
    out_fps = fps * 2

    # Encode from RIFE output PNGs
    if not prog.done(cid, "encode"):
        disk_dir.mkdir(parents=True, exist_ok=True)
        try:
            t0 = time.time()
            nvenc_encode(rife_out, vid, out_fps)
            prog.mark(cid, "encode")
            print(f"  [NVENC] chunk {cid:04d} encoded ({time.time()-t0:.1f}s)",
                  flush=True)
        except Exception as e:
            print(f"  [!] Encode chunk {cid:04d} failed: {e}")

    # Cleanup tmpfs
    if vid.exists() and vid.stat().st_size > 1000:
        shutil.rmtree(_tmpfs_chunk(cid), ignore_errors=True)
        prog.mark(cid, "clean")
