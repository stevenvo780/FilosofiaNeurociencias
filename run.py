#!/usr/bin/env python3
"""
AI Video Enhancement — entry point.

Usage:
  python3 run.py <input.mp4> [options]
  python3 run.py GMT20260320-130023_Recording_2240x1260.mp4 --skip-rife
  python3 run.py GMT20260320-130023_Recording_2240x1260.mp4 --skip-rife --clean
"""
import sys, time, shutil, argparse
from pathlib import Path

from enhance import config as C
from enhance.progress import Progress
from enhance.ffmpeg_utils import probe, merge_chunks
from enhance.pipeline import run as run_pipeline


def main():
    ap = argparse.ArgumentParser(description="AI Video Enhancement")
    ap.add_argument("input", help="Input video file")
    ap.add_argument("--skip-esrgan", action="store_true")
    ap.add_argument("--skip-rife",   action="store_true")
    ap.add_argument("--chunk", type=int, default=C.CHUNK_SECONDS,
                    help=f"Chunk duration seconds (default {C.CHUNK_SECONDS})")
    ap.add_argument("--clean", action="store_true",
                    help="Clean work directory and restart")
    ap.add_argument("--outdir", type=str, default=None,
                    help="Output directory (default: <input_dir>/enhanced)")
    args = ap.parse_args()

    src = Path(args.input).resolve()
    if not src.exists():
        print(f"[!] Not found: {src}"); sys.exit(1)

    if args.outdir:
        out_dir = Path(args.outdir).resolve()
    else:
        out_dir = src.parent / "enhanced"
    out_dir.mkdir(exist_ok=True)
    work = out_dir / f"work_{src.stem}"

    if args.clean and work.exists():
        shutil.rmtree(work)
    work.mkdir(exist_ok=True)

    do_esr  = not args.skip_esrgan
    do_rife = not args.skip_rife

    dur, fps, w, h = probe(src)
    scale = 2 if do_esr else 1
    out_fps = fps * 2 if do_rife else fps
    total_frames = int(dur * fps)

    suf = "_ai" + (f"_{int(out_fps)}fps" if do_rife else "")
    dst = out_dir / f"{src.stem}{suf}.mp4"

    print("=" * 60)
    print("  AI VIDEO ENHANCEMENT")
    print("=" * 60)
    print(f"  Input:  {src.name}  ({dur/3600:.1f}h, {w}x{h} @ {fps}fps)")
    print(f"  Output: {dst.name}  ({w*scale}x{h*scale} @ {out_fps}fps)")
    print(f"  Frames: ~{total_frames:,}  |  Chunks: {args.chunk}s each")
    print(f"  ESRGAN: {'ON' if do_esr else 'SKIP'}  |  RIFE: {'ON' if do_rife else 'SKIP'}")
    print(f"  Tmpfs:  {C.TMPFS_WORK}  (frames in RAM, zero disk I/O)")
    print("=" * 60)

    prog = Progress(work)
    chunks = []
    s = 0.0
    i = 0
    while s < dur:
        cd = min(args.chunk, dur - s)
        if cd <= 0:
            break
        chunks.append((i, s, cd))
        s += args.chunk
        i += 1

    n_chunks = len(chunks)
    n_done = sum(1 for c in chunks if prog.done(c[0], "encode"))
    print(f"\n  {n_chunks} chunks total, {n_done} done, "
          f"{n_chunks - n_done} remaining\n")

    t_go = time.time()
    if n_done < n_chunks:
        esr = None
        if do_esr:
            from enhance.esrgan import ESRGANEngine
            esr = ESRGANEngine()
        completed = run_pipeline(
            chunks, src, work, prog, do_esr, do_rife, fps, esr,
            w=w, h=h)
        wall = time.time() - t_go
        if wall > 0:
            print(f"\n  Processing: {wall/3600:.1f}h  "
                  f"({dur/wall:.2f}x realtime)")

    if not C.shutdown.is_set():
        merge_chunks(work, src, dst, n_chunks)
    else:
        print("\n[!] Interrupted. Run again to resume.")


if __name__ == "__main__":
    main()
