#!/usr/bin/env python3
"""
AI Video Enhancement Pipeline - Optimized for:
  - Ryzen 9 9950X3D (32 threads)
  - RTX 5070 Ti (16GB) + RTX 2060 (6GB)
  - 128GB RAM
  - NVMe RAID 6.7GB/s

Processes video in resumable chunks:
  1. Real-ESRGAN (AI upscale/enhancement)
  2. RIFE (frame interpolation 25→50fps)
  3. HEVC NVENC encoding

Usage:
  python3 ai_enhance.py <input.mp4> [--resume] [--skip-esrgan] [--skip-rife]
"""

import os
import sys
import subprocess
import time
import signal
import json
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading

# ============================================================
# CONFIG
# ============================================================
CHUNK_DURATION = 120  # seconds per chunk (2 min = manageable)
ESRGAN_SCALE = 2
ESRGAN_BIN = "/tmp/realesrgan-ncnn/realesrgan-ncnn-vulkan"
ESRGAN_MODEL_DIR = "/tmp/realesrgan-ncnn/models"
ESRGAN_MODEL = "realesr-animevideov3"
RIFE_BIN = "/tmp/rife-ncnn/rife-ncnn-vulkan-20221029-ubuntu/rife-ncnn-vulkan"
RIFE_MODEL = "/tmp/rife-ncnn/rife-ncnn-vulkan-20221029-ubuntu/rife-v4.6"

# GPU assignments - run ESRGAN on both GPUs processing different chunks
GPU_ESRGAN = "0,1"  # both GPUs for ESRGAN
GPU_RIFE = "0,1"    # both GPUs for RIFE
NVENC_GPU = 0       # 5070 Ti for final encoding

# Thread counts: load:proc:save (per GPU)
ESRGAN_THREADS = "8:8,4:8"  # more proc threads on GPU0 (5070Ti), less on GPU1 (2060)
RIFE_THREADS = "8:8,4:8"

# FFmpeg threads for extraction/encoding
FFMPEG_THREADS = 16  # half of 32 threads, leave room for other work

# ============================================================
# PROGRESS TRACKING (resumable)
# ============================================================
class ProgressTracker:
    def __init__(self, work_dir):
        self.file = work_dir / "progress.json"
        self.lock = threading.Lock()
        if self.file.exists():
            with open(self.file) as f:
                self.data = json.load(f)
        else:
            self.data = {"chunks": {}, "phase": "init"}

    def save(self):
        with self.lock:
            with open(self.file, "w") as f:
                json.dump(self.data, f, indent=2)

    def chunk_done(self, chunk_id, phase):
        with self.lock:
            key = str(chunk_id)
            if key not in self.data["chunks"]:
                self.data["chunks"][key] = {}
            self.data["chunks"][key][phase] = True
        self.save()

    def is_done(self, chunk_id, phase):
        key = str(chunk_id)
        return self.data.get("chunks", {}).get(key, {}).get(phase, False)

    def set_phase(self, phase):
        self.data["phase"] = phase
        self.save()


# ============================================================
# VIDEO INFO
# ============================================================
def get_video_info(input_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(input_path)
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(r.stdout)
    duration = float(info["format"]["duration"])
    for s in info["streams"]:
        if s["codec_type"] == "video":
            fps = eval(s["r_frame_rate"])
            w, h = int(s["width"]), int(s["height"])
            return duration, fps, w, h
    raise ValueError("No video stream found")


# ============================================================
# CHUNK EXTRACTION (multi-threaded, uses all CPU cores)
# ============================================================
def extract_chunk_frames(input_path, chunk_id, start_time, duration, output_dir, threads=FFMPEG_THREADS):
    """Extract frames from a video chunk using ffmpeg."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already has frames
    existing = list(output_dir.glob("*.png"))
    expected_approx = int(duration * 25)  # approximate
    if len(existing) >= expected_approx - 2:
        return len(existing)

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", str(input_path),
        "-t", str(duration),
        "-threads", str(threads),
        "-pix_fmt", "rgb24",
        str(output_dir / "%08d.png"),
        "-loglevel", "warning"
    ]
    subprocess.run(cmd, check=True)
    return len(list(output_dir.glob("*.png")))


# ============================================================
# AI ENHANCEMENT: Real-ESRGAN
# ============================================================
def run_esrgan_chunk(input_dir, output_dir, gpu_id="0,1", threads=ESRGAN_THREADS):
    """Run Real-ESRGAN on a directory of frames."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already processed
    in_count = len(list(input_dir.glob("*.png")))
    out_count = len(list(output_dir.glob("*.png")))
    if out_count >= in_count and in_count > 0:
        return out_count

    cmd = [
        ESRGAN_BIN,
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-s", str(ESRGAN_SCALE),
        "-n", ESRGAN_MODEL,
        "-g", gpu_id,
        "-j", threads,
        "-f", "png",
        "-m", ESRGAN_MODEL_DIR
    ]
    subprocess.run(cmd, check=True)
    return len(list(output_dir.glob("*.png")))


# ============================================================
# AI FRAME INTERPOLATION: RIFE
# ============================================================
def run_rife_chunk(input_dir, output_dir, gpu_id="0,1", threads=RIFE_THREADS):
    """Run RIFE frame interpolation (doubles frame count)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    in_count = len(list(input_dir.glob("*.png")))
    out_count = len(list(output_dir.glob("*.png")))
    expected = in_count * 2 - 1
    if out_count >= expected and in_count > 0:
        return out_count

    cmd = [
        RIFE_BIN,
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-m", RIFE_MODEL,
        "-g", gpu_id,
        "-j", threads,
        "-f", "%08d.png"
    ]
    subprocess.run(cmd, check=True)
    return len(list(output_dir.glob("*.png")))


# ============================================================
# ENCODE CHUNK TO VIDEO
# ============================================================
def encode_chunk(frames_dir, output_file, fps=50, gpu=NVENC_GPU, threads=FFMPEG_THREADS):
    """Encode frames to HEVC using NVENC."""
    if output_file.exists() and output_file.stat().st_size > 1000:
        return

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "%08d.png"),
        "-c:v", "hevc_nvenc",
        "-gpu", str(gpu),
        "-preset", "p6", "-tune", "hq",
        "-rc", "vbr", "-cq", "20",
        "-b:v", "12M", "-maxrate", "18M", "-bufsize", "24M",
        "-profile:v", "main10",
        "-pix_fmt", "yuv420p10le",
        "-threads", str(threads),
        str(output_file),
        "-loglevel", "warning"
    ]
    subprocess.run(cmd, check=True)


# ============================================================
# PARALLEL CHUNK PROCESSOR
# ============================================================
def process_chunk(args):
    """Process a single chunk through the full pipeline."""
    chunk_id, input_path, start, duration, work_dir, progress, do_esrgan, do_rife, orig_fps = args

    chunk_dir = work_dir / f"chunk_{chunk_id:04d}"
    raw_dir = chunk_dir / "raw"
    esrgan_dir = chunk_dir / "esrgan"
    rife_dir = chunk_dir / "rife"
    chunk_video = chunk_dir / "output.mp4"

    t0 = time.time()
    print(f"  [Chunk {chunk_id:04d}] Start: {start:.0f}s - {start+duration:.0f}s")

    # Step 1: Extract frames
    if not progress.is_done(chunk_id, "extract"):
        n = extract_chunk_frames(input_path, chunk_id, start, duration, raw_dir)
        progress.chunk_done(chunk_id, "extract")
        print(f"  [Chunk {chunk_id:04d}] Extracted {n} frames ({time.time()-t0:.1f}s)")

    # Step 2: Real-ESRGAN enhancement
    current_frames_dir = raw_dir
    if do_esrgan:
        if not progress.is_done(chunk_id, "esrgan"):
            n = run_esrgan_chunk(raw_dir, esrgan_dir)
            progress.chunk_done(chunk_id, "esrgan")
            print(f"  [Chunk {chunk_id:04d}] ESRGAN done: {n} frames ({time.time()-t0:.1f}s)")
        current_frames_dir = esrgan_dir

    # Step 3: RIFE frame interpolation
    output_fps = orig_fps
    if do_rife:
        if not progress.is_done(chunk_id, "rife"):
            n = run_rife_chunk(current_frames_dir, rife_dir)
            progress.chunk_done(chunk_id, "rife")
            print(f"  [Chunk {chunk_id:04d}] RIFE done: {n} frames ({time.time()-t0:.1f}s)")
        current_frames_dir = rife_dir
        output_fps = orig_fps * 2

    # Step 4: Encode to video
    if not progress.is_done(chunk_id, "encode"):
        encode_chunk(current_frames_dir, chunk_video, fps=output_fps)
        progress.chunk_done(chunk_id, "encode")
        print(f"  [Chunk {chunk_id:04d}] Encoded ({time.time()-t0:.1f}s)")

    # Step 5: Cleanup intermediate frames to save disk space
    if chunk_video.exists() and chunk_video.stat().st_size > 1000:
        if not progress.is_done(chunk_id, "cleanup"):
            shutil.rmtree(raw_dir, ignore_errors=True)
            if do_esrgan:
                shutil.rmtree(esrgan_dir, ignore_errors=True)
            if do_rife:
                shutil.rmtree(rife_dir, ignore_errors=True)
            progress.chunk_done(chunk_id, "cleanup")

    elapsed = time.time() - t0
    print(f"  [Chunk {chunk_id:04d}] COMPLETE in {elapsed:.1f}s")
    return chunk_id, elapsed


# ============================================================
# CONCATENATE ALL CHUNKS + ADD ENHANCED AUDIO
# ============================================================
def concatenate_and_finalize(work_dir, input_path, output_path, num_chunks):
    """Merge all chunk videos and add enhanced audio."""
    concat_file = work_dir / "concat.txt"
    with open(concat_file, "w") as f:
        for i in range(num_chunks):
            chunk_video = work_dir / f"chunk_{i:04d}" / "output.mp4"
            if chunk_video.exists():
                f.write(f"file '{chunk_video}'\n")

    # Check if enhanced audio exists
    enhanced_audio = output_path.parent / (input_path.stem + "_enhanced.m4a")
    audio_src = str(enhanced_audio) if enhanced_audio.exists() else str(input_path)

    print(f"\n[FINAL] Concatenating {num_chunks} chunks + audio...")
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-i", audio_src,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy",
        "-af", "afftdn=nf=-20:nt=w:om=o,acompressor=threshold=-20dB:ratio=3:attack=5:release=50,loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-threads", "16",
        str(output_path),
        "-loglevel", "warning"
    ]
    subprocess.run(cmd, check=True)
    print(f"[DONE] Output: {output_path} ({output_path.stat().st_size / 1e9:.2f} GB)")


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="AI Video Enhancement Pipeline")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--skip-esrgan", action="store_true", help="Skip Real-ESRGAN step")
    parser.add_argument("--skip-rife", action="store_true", help="Skip RIFE interpolation step")
    parser.add_argument("--chunk-duration", type=int, default=CHUNK_DURATION, help="Chunk duration in seconds")
    parser.add_argument("--parallel-chunks", type=int, default=2,
                        help="Number of chunks to process in parallel (default 2 for dual GPU)")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    # Setup directories
    output_dir = input_path.parent / "enhanced"
    output_dir.mkdir(exist_ok=True)
    work_dir = output_dir / f"work_{input_path.stem}"
    work_dir.mkdir(exist_ok=True)

    suffix = "_ai_enhanced"
    if not args.skip_rife:
        suffix += "_50fps"
    output_path = output_dir / f"{input_path.stem}{suffix}.mp4"

    # Get video info
    duration, fps, w, h = get_video_info(input_path)
    print(f"=" * 60)
    print(f"AI VIDEO ENHANCEMENT PIPELINE")
    print(f"=" * 60)
    print(f"Input:      {input_path.name}")
    print(f"Resolution: {w}x{h} @ {fps}fps")
    print(f"Duration:   {duration:.0f}s ({duration/3600:.1f}h)")
    print(f"ESRGAN:     {'SKIP' if args.skip_esrgan else f'x{ESRGAN_SCALE} ({w}x{h} → {w*ESRGAN_SCALE}x{h*ESRGAN_SCALE})'}")
    print(f"RIFE:       {'SKIP' if args.skip_rife else f'{fps}fps → {fps*2}fps'}")
    print(f"Chunks:     {args.chunk_duration}s each, {args.parallel_chunks} parallel")
    print(f"Output:     {output_path.name}")
    print(f"=" * 60)

    # Progress tracker for resume
    progress = ProgressTracker(work_dir)

    # Calculate chunks
    num_chunks = int(duration / args.chunk_duration) + 1
    chunks = []
    for i in range(num_chunks):
        start = i * args.chunk_duration
        chunk_dur = min(args.chunk_duration, duration - start)
        if chunk_dur <= 0:
            break
        chunks.append((i, input_path, start, chunk_dur, work_dir, progress,
                       not args.skip_esrgan, not args.skip_rife, fps))

    actual_chunks = len(chunks)
    done_chunks = sum(1 for i in range(actual_chunks)
                      if progress.is_done(i, "encode"))

    print(f"\nTotal chunks: {actual_chunks}, Already done: {done_chunks}")
    print(f"Processing {actual_chunks - done_chunks} remaining chunks...\n")

    # Process chunks - GPU is the bottleneck so we process sequentially
    # but each chunk uses BOTH GPUs internally
    t_start = time.time()
    completed = done_chunks

    for chunk_args in chunks:
        chunk_id = chunk_args[0]
        if progress.is_done(chunk_id, "encode"):
            continue

        _, elapsed = process_chunk(chunk_args)
        completed += 1

        # ETA calculation
        avg_time = (time.time() - t_start) / (completed - done_chunks)
        remaining = (actual_chunks - completed) * avg_time
        print(f"\n  Progress: {completed}/{actual_chunks} chunks "
              f"({100*completed/actual_chunks:.1f}%) "
              f"ETA: {remaining/3600:.1f}h\n")

    # Final concatenation
    concatenate_and_finalize(work_dir, input_path, output_path, actual_chunks)

    total_time = time.time() - t_start
    print(f"\nTotal processing time: {total_time/3600:.1f}h")
    print(f"Speedup vs realtime: {duration/total_time:.2f}x")


if __name__ == "__main__":
    main()
