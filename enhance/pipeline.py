"""
4-stage pipeline: extract → ESRGAN → RIFE → NVENC
Each stage on different silicon, overlapping across chunks.
"""
import time, shutil, threading, queue
from pathlib import Path

from . import config as C
from .progress import Progress
from .ffmpeg_utils import extract_frames, nvenc_encode
from .esrgan import ESRGANEngine
from .rife import interpolate as rife_interpolate


def run(chunks, src: Path, work: Path, prog: Progress,
        do_esr: bool, do_rife: bool, fps: float, esr: ESRGANEngine | None):
    """Process all chunks with pipelined overlap."""
    total = len(chunks)
    done_n = sum(1 for c in chunks if prog.done(c[0], "encode"))
    pending = [c for c in chunks if not prog.done(c[0], "encode")]
    if not pending:
        print(f"  All {total} chunks already processed!")
        return done_n

    # ── Background threads for extract + encode ─────────────
    ext_q = queue.Queue(maxsize=C.PIPELINE_DEPTH)
    enc_q = queue.Queue()

    def _extractor():
        for cid, start, dur in pending:
            if C.shutdown.is_set():
                break
            raw = work / f"chunk_{cid:04d}" / "raw"
            if not prog.done(cid, "extract"):
                t0 = time.time()
                n = extract_frames(src, start, dur, raw, fps)
                dt = time.time() - t0
                print(f"  [NVDEC] chunk {cid:04d}: {n} frames ({dt:.1f}s)",
                      flush=True)
                prog.mark(cid, "extract")
            ext_q.put(cid)
        ext_q.put(None)

    def _encoder():
        while True:
            item = enc_q.get()
            if item is None:
                break
            cid, frames_dir, out_fps = item
            vid = work / f"chunk_{cid:04d}" / "output.mp4"
            if not prog.done(cid, "encode"):
                try:
                    t0 = time.time()
                    nvenc_encode(frames_dir, vid, out_fps)
                    prog.mark(cid, "encode")
                    print(f"  [NVENC] chunk {cid:04d} encoded ({time.time()-t0:.1f}s)",
                          flush=True)
                except Exception as e:
                    print(f"  [!] Encode chunk {cid:04d} failed: {e}")
            # Cleanup intermediates to free disk
            if vid.exists() and vid.stat().st_size > 1000:
                for d in ["raw", "esrgan", "rife"]:
                    p = work / f"chunk_{cid:04d}" / d
                    if p.exists():
                        shutil.rmtree(p, ignore_errors=True)
                prog.mark(cid, "clean")
            enc_q.task_done()

    threading.Thread(target=_extractor, daemon=True, name="extractor").start()
    threading.Thread(target=_encoder,   daemon=True, name="encoder").start()

    t_start = time.time()
    completed = done_n

    for cid, start, dur in pending:
        if C.shutdown.is_set():
            break
        ready = ext_q.get()
        if ready is None:
            break

        tc = time.time()
        print(f"\n  == Chunk {cid:04d}  [{start:.0f}s–{start+dur:.0f}s] ==",
              flush=True)

        cur = work / f"chunk_{cid:04d}" / "raw"

        # ESRGAN (Tensor Cores + CUDA Cores)
        if do_esr and esr:
            esr_out = work / f"chunk_{cid:04d}" / "esrgan"
            if not prog.done(cid, "esrgan"):
                n = esr.process_directory(cur, esr_out)
                if C.shutdown.is_set():
                    break
                prog.mark(cid, "esrgan")
                print(f"  | ESRGAN: {n} frames  ({time.time()-tc:.1f}s)",
                      flush=True)
            cur = esr_out

        # RIFE (Vulkan compute — separate from CUDA)
        out_fps = fps
        if do_rife:
            rife_out = work / f"chunk_{cid:04d}" / "rife"
            if not prog.done(cid, "rife"):
                tr = time.time()
                n = rife_interpolate(cur, rife_out)
                prog.mark(cid, "rife")
                print(f"  | RIFE:   {n} frames  ({time.time()-tr:.1f}s)",
                      flush=True)
            cur = rife_out
            out_fps = fps * 2

        # Queue for NVENC (dedicated ASIC, runs while next chunk processes)
        enc_q.put((cid, cur, out_fps))
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
