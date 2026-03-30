"""
4-stage pipeline: extract → ESRGAN → RIFE → NVENC
Each stage on different silicon, overlapping across chunks.

Zero-PNG path (no RIFE):
  ffmpeg pipe → numpy RAM → GPU ESRGAN → numpy RAM → ffmpeg pipe encode
  PNG never touches disk or tmpfs.

RIFE path (needs files):
  ffmpeg pipe → numpy RAM → GPU ESRGAN → PNGs on tmpfs → RIFE → NVENC
"""
import time, shutil, threading, queue, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

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


def _encode_from_numpy(frames: list[np.ndarray], out_file: Path,
                        fps: float, gpu: int = 0):
    """Pipe numpy RGB frames directly to NVENC — zero PNG I/O."""
    if out_file.exists() and out_file.stat().st_size > 1000:
        return
    if not frames:
        return
    h, w = frames[0].shape[:2]
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
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"NVENC encode failed (rc={proc.returncode})")


def _write_pngs(frames: list[np.ndarray], out_dir: Path):
    """Write numpy frames as PNGs to tmpfs (for RIFE)."""
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
    """Process all chunks — zero-PNG when possible, tmpfs when RIFE needs files."""
    total = len(chunks)
    done_n = sum(1 for c in chunks if prog.done(c[0], "encode"))
    pending = [c for c in chunks if not prog.done(c[0], "encode")]
    if not pending:
        print(f"  All {total} chunks already processed!")
        return done_n

    Path(C.TMPFS_WORK).mkdir(parents=True, exist_ok=True)

    # Background encoder queue (used only when RIFE produces PNGs)
    enc_q: queue.Queue = queue.Queue()

    def _encoder():
        while True:
            item = enc_q.get()
            if item is None:
                break
            if len(item) == 3:
                # PNG dir-based (from RIFE)
                cid, frames_dir, out_fps = item
                disk_dir = work / f"chunk_{cid:04d}"
                disk_dir.mkdir(parents=True, exist_ok=True)
                vid = disk_dir / "output.mp4"
                if not prog.done(cid, "encode"):
                    try:
                        t0 = time.time()
                        nvenc_encode(frames_dir, vid, out_fps)
                        prog.mark(cid, "encode")
                        print(f"  [NVENC] chunk {cid:04d} encoded ({time.time()-t0:.1f}s)",
                              flush=True)
                    except Exception as e:
                        print(f"  [!] Encode chunk {cid:04d} failed: {e}")
                tmp_chunk = _tmpfs_chunk(cid)
                if vid.exists() and vid.stat().st_size > 1000:
                    shutil.rmtree(tmp_chunk, ignore_errors=True)
                    prog.mark(cid, "clean")
            else:
                # Numpy pipe-based (zero PNG)
                cid, np_frames, out_fps, vid = item
                vid.parent.mkdir(parents=True, exist_ok=True)
                if not prog.done(cid, "encode"):
                    try:
                        t0 = time.time()
                        _encode_from_numpy(np_frames, vid, out_fps, C.NVENC_GPU)
                        prog.mark(cid, "encode")
                        prog.mark(cid, "clean")
                        print(f"  [NVENC] chunk {cid:04d} pipe-encoded ({time.time()-t0:.1f}s)",
                              flush=True)
                    except Exception as e:
                        print(f"  [!] Encode chunk {cid:04d} failed: {e}")
                del np_frames
            enc_q.task_done()

    enc_thread = threading.Thread(target=_encoder, daemon=True, name="encoder")
    enc_thread.start()

    t_start = time.time()
    completed = done_n

    for cid, start, dur in pending:
        if C.shutdown.is_set():
            break

        tc = time.time()
        print(f"\n  == Chunk {cid:04d}  [{start:.0f}s–{start+dur:.0f}s] ==",
              flush=True)

        # ── Stage 1: Extract → numpy arrays in RAM (zero disk) ──
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
            # Need frames for ESRGAN but extract was done — re-extract to RAM
            frames = extract_frames_to_ram(src, start, dur, w, h)

        # ── Stage 2: ESRGAN (Tensor Cores FP16) — numpy → numpy ──
        esr_frames = None
        if do_esr and esr:
            if not prog.done(cid, "esrgan"):
                if frames is None:
                    frames = extract_frames_to_ram(src, start, dur, w, h)
                esr_frames = esr.process_frames(frames, out_dir=None)
                if C.shutdown.is_set():
                    break
                prog.mark(cid, "esrgan")
                n = len(esr_frames)
                print(f"  | ESRGAN: {n} frames  ({time.time()-tc:.1f}s)",
                      flush=True)
                # Free raw frames immediately
                del frames
                frames = None

        # ── Stage 3: RIFE (Vulkan compute) — needs PNGs on tmpfs ──
        out_fps = fps
        if do_rife:
            # RIFE binary needs PNG files — write to tmpfs
            rife_in = _tmpfs_chunk(cid) / "esrgan"
            rife_out = _tmpfs_chunk(cid) / "rife"
            if not prog.done(cid, "rife"):
                src_frames = esr_frames if esr_frames is not None else frames
                if src_frames is not None:
                    _write_pngs(src_frames, rife_in)
                    del src_frames, esr_frames, frames
                    esr_frames = frames = None
                tr = time.time()
                n = rife_interpolate(rife_in, rife_out)
                prog.mark(cid, "rife")
                print(f"  | RIFE:   {n} frames  ({time.time()-tr:.1f}s)",
                      flush=True)
                # Free ESRGAN PNGs
                shutil.rmtree(rife_in, ignore_errors=True)
            out_fps = fps * 2
            # Encode from PNG dir (RIFE output)
            enc_q.put((cid, rife_out, out_fps))
        else:
            # ── Stage 4: NVENC — pipe numpy in background thread ──
            disk_dir = work / f"chunk_{cid:04d}"
            vid = disk_dir / "output.mp4"
            if not prog.done(cid, "encode"):
                src_frames = esr_frames if esr_frames is not None else frames
                if src_frames is not None:
                    # Queue encode to run in background while next chunk processes
                    enc_q.put((cid, src_frames, fps, vid))
            # Free references (data owned by enc_q now)
            del esr_frames, frames

        completed += 1
        wall = time.time() - t_start
        avg = wall / max(completed - done_n, 1)
        rem = (total - completed) * avg
        print(f"  | Progress: {completed}/{total} ({100*completed/total:.1f}%)  "
              f"ETA {rem/3600:.1f}h", flush=True)

    # Drain encoder queue
    enc_q.join()
    enc_q.put(None)
    return completed
