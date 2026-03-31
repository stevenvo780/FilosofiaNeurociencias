#!/usr/bin/env python3
"""AI Video Enhancement — entry point."""
import argparse
import shutil
import sys
import threading
import time
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from enhance import config as C
from enhance.progress import Progress
from enhance.ffmpeg_utils import (
    enhance_audio,
    merge_chunks,
    probe,
    resolve_audio_source,
)
from enhance.pipeline import run as run_pipeline


def main():
    C.shutdown.clear()
    ap = argparse.ArgumentParser(description="AI Video Enhancement")
    ap.add_argument("input", help="Input video file")
    ap.add_argument("--skip-esrgan", action="store_true")
    ap.add_argument("--skip-rife", action="store_true")
    ap.add_argument("--skip-audio", action="store_true",
                    help="Skip audio enhancement and mux original audio when available")
    ap.add_argument("--audio-input", type=str, default=None,
                    help="Optional external audio source to enhance/mux")
    ap.add_argument("--chunk", type=int, default=C.CHUNK_SECONDS,
                    help=f"Chunk duration seconds (default {C.CHUNK_SECONDS})")
    ap.add_argument("--start", type=float, default=0.0,
                    help="Start offset in seconds for test slices")
    ap.add_argument("--duration", type=float, default=None,
                    help="Optional processing duration in seconds")
    ap.add_argument("--clean", action="store_true",
                    help="Clean work directory and restart")
    ap.add_argument("--outdir", type=str, default=None,
                    help="Output directory (default: <input_dir>/enhanced)")
    ap.add_argument("--visual-profile", type=str, default=None,
                    help="Visual quality profile (baseline, real_x2, hybrid_detail, face_adaptive)")
    ap.add_argument("--audio-profile", type=str, default=None,
                    help="Audio processing profile (baseline, conservative, voice, natural)")
    ap.add_argument("--scheduler-profile", type=str, default=None,
                    help="CPU scheduler profile (baseline, split_l3_a, split_l3_b)")
    ap.add_argument("--rife-backend", type=str, default=None,
                    help="RIFE backend (baseline, torch)")
    ap.add_argument("--models-dir", type=str, default=None,
                    help="Directory for model weights")
    ap.add_argument("--benchmark-tag", type=str, default=None,
                    help="Tag for benchmark identification")
    ap.add_argument("--quality-only", action="store_true",
                    help="Only run quality evaluation")
    ap.add_argument("--throughput-only", action="store_true",
                    help="Skip quality evaluation")
    ap.add_argument("--nsys", action="store_true",
                    help="Wrap with NVIDIA Nsight Systems profiler")
    args = ap.parse_args()

    # ── Load profiles ──
    from enhance.profiles import get_profiles
    from enhance.scheduler import apply_scheduler_profile
    vp, aup, sp, rp = get_profiles(
        visual=args.visual_profile,
        audio=args.audio_profile,
        scheduler=args.scheduler_profile,
        rife_backend=args.rife_backend,
    )
    apply_scheduler_profile(sp)

    # Override config from profiles where applicable
    if args.models_dir:
        os.environ["ENHANCE_MODELS_DIR"] = args.models_dir

    src = Path(args.input).resolve()
    if not src.exists():
        print(f"[!] Not found: {src}"); sys.exit(1)

    if args.outdir:
        out_dir = Path(args.outdir).resolve()
    else:
        out_dir = src.parent / "enhanced"
    out_dir.mkdir(exist_ok=True)

    do_esr  = not args.skip_esrgan
    do_rife = not args.skip_rife

    full_dur, fps, w, h = probe(src)
    start_at = max(args.start, 0.0)
    remaining = max(full_dur - start_at, 0.0)
    if remaining <= 0:
        print(f"[!] Start {start_at:.2f}s fuera del rango del video"); sys.exit(1)

    process_dur = min(args.duration, remaining) if args.duration is not None else remaining
    if process_dur <= 0:
        print(f"[!] Duration inválida: {process_dur}"); sys.exit(1)

    slice_tag = ""
    if start_at > 0 or args.duration is not None:
        slice_tag = f"_s{int(start_at)}_d{int(process_dur)}"

    work = out_dir / f"work_{src.stem}_{args.chunk}s{slice_tag}"
    if args.clean and work.exists():
        shutil.rmtree(work)
    work.mkdir(exist_ok=True)

    scale = 2 if do_esr else 1
    out_fps = fps * 2 if do_rife else fps
    total_frames = int(process_dur * fps)

    suf = "_ai" + (f"_{int(out_fps)}fps" if do_rife else "") + slice_tag
    dst = out_dir / f"{src.stem}{suf}.mp4"
    explicit_audio = Path(args.audio_input).resolve() if args.audio_input else None
    audio_src = resolve_audio_source(src, explicit_audio)
    audio_out = None
    audio_thread = None
    audio_error = {}

    if not args.skip_audio and audio_src is not None:
        audio_out = out_dir / f"{audio_src.stem}{slice_tag}_enhanced.m4a"
        if args.clean:
            audio_out.unlink(missing_ok=True)

        if not audio_out.exists():
            def _run_audio():
                try:
                    enhance_audio(audio_src, audio_out, start=start_at, duration=process_dur, audio_profile=aup)
                except Exception as exc:
                    audio_error["error"] = exc
            audio_thread = threading.Thread(target=_run_audio, name="Audio Enhance", daemon=True)
            audio_thread.start()

    print("=" * 60)
    print("  AI VIDEO ENHANCEMENT")
    print("=" * 60)
    print(f"  Input:  {src.name}  ({full_dur/3600:.1f}h total, {w}x{h} @ {fps}fps)")
    print(f"  Output: {dst.name}  ({w*scale}x{h*scale} @ {out_fps}fps)")
    print(f"  Slice:  {start_at:.1f}s → {start_at + process_dur:.1f}s  ({process_dur/60:.1f} min)")
    print(f"  Frames: ~{total_frames:,}  |  Chunks: {args.chunk}s each")
    print(f"  ESRGAN: {'ON' if do_esr else 'SKIP'}  |  RIFE: {'ON' if do_rife else 'SKIP'}")
    print(f"  Audio:  {'ENHANCE' if (audio_src and not args.skip_audio) else ('COPY' if audio_src else 'NONE')}")
    print(f"  Tmpfs:  {C.TMPFS_WORK}  (PNG intermedios en tmpfs + ventanas en RAM)")
    print(f"  Profiles: visual={vp.name} audio={aup.name} sched={sp.name} rife={rp.name}")
    print("=" * 60)

    prog = Progress(work)
    chunks = []
    s = start_at
    i = 0
    end_at = start_at + process_dur
    min_chunk = 0.5 / max(fps, 1.0)
    while s < end_at - 1e-6:
        cd = min(args.chunk, end_at - s)
        if cd <= 0 or cd < min_chunk:
            break
        chunks.append((i, s, cd))
        s += cd
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
            esr = ESRGANEngine(visual_profile=vp)
        run_pipeline(
            chunks, src, work, prog, do_esr, do_rife, fps, esr,
            w=w, h=h,
            visual_profile=vp, audio_profile=aup,
            scheduler_profile=sp, rife_backend_profile=rp)
        wall = time.time() - t_go
        if wall > 0:
            print(f"\n  Processing: {wall/3600:.1f}h  "
                  f"({process_dur/wall:.2f}x realtime)")

    if audio_thread is not None:
        audio_thread.join()
        if "error" in audio_error:
            raise audio_error["error"]

    if not C.shutdown.is_set():
        final_audio = audio_out if (audio_out and audio_out.exists()) else audio_src
        merge_chunks(work, dst, n_chunks, audio_src=final_audio)
    else:
        print("\n[!] Interrupted. Run again to resume.")


if __name__ == "__main__":
    main()
