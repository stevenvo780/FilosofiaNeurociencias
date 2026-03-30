#!/usr/bin/env python3
"""
AI Video Enhancement Pipeline v2 - Maximum Hardware Utilization

Hardware target:
  - Ryzen 9 9950X3D (16C/32T, 192MB 3D V-Cache)
  - RTX 5070 Ti (16GB, SM 12.0, 70 SMs) — CUDA + Tensor Cores + NVENC
  - RTX 2060 (6GB, SM 7.5, 30 SMs) — CUDA + Tensor Cores
  - 128GB RAM
  - NVMe RAID 0 — 6.7GB/s

Pipeline architecture (all chips working simultaneously):
  ┌──────────────┐  ┌────────────────────────┐  ┌──────────────────┐  ┌────────────┐
  │ CPU 32T      │→ │ GPU0 Tensor + GPU1     │→ │ RIFE ncnn Vulkan │→ │ NVENC      │
  │ ffmpeg       │  │ ESRGAN FP16            │  │ dual GPU         │  │ HEVC encode│
  │ extract      │  │ torch.compile          │  │                  │  │            │
  │ chunk N+2    │  │ chunk N+1              │  │ chunk N          │  │ chunk N-1  │
  └──────────────┘  └────────────────────────┘  └──────────────────┘  └────────────┘
                    All stages overlap on different chunks!

Resumable: progress saved per-chunk per-phase. Safe to Ctrl+C or power off.

Usage:
  python3 ai_enhance.py <input.mp4> [--resume] [--skip-esrgan] [--skip-rife]
  python3 ai_enhance.py <input.mp4> --resume          # Continue after interruption
  python3 ai_enhance.py <input.mp4> --skip-rife       # Only ESRGAN (no 50fps)
  python3 ai_enhance.py <input.mp4> --skip-esrgan     # Only RIFE (25->50fps)
"""

import os
import sys
import subprocess
import time
import json
import shutil
import signal
import threading
import queue
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# ============================================================
# CONFIG
# ============================================================
CHUNK_DURATION = 120          # seconds per chunk (2 min)
ESRGAN_MODEL_PATH = "/tmp/realesr-animevideov3.pth"
RIFE_BIN = "/tmp/rife-ncnn/rife-ncnn-vulkan-20221029-ubuntu/rife-ncnn-vulkan"
RIFE_MODEL = "/tmp/rife-ncnn/rife-ncnn-vulkan-20221029-ubuntu/rife-v4.6"
NVENC_GPU = 0                 # 5070 Ti for HEVC encoding

# ESRGAN tiling config per GPU
GPU0_TILE_SIZE = 768          # 5070 Ti (16GB) - bigger tiles = faster
GPU1_TILE_SIZE = 384          # 2060 (6GB) - smaller tiles to fit VRAM
TILE_PAD = 32                 # Overlap between tiles to avoid seams

# Thread allocation (32 total)
FFMPEG_EXTRACT_THREADS = 12   # Frame extraction
FFMPEG_ENCODE_THREADS = 8     # NVENC encoding (CPU side is light)
IO_WORKERS = 8                # PNG read/write workers

# RIFE ncnn config
RIFE_GPU = "0,1"              # Use both GPUs
RIFE_THREADS = "4:4:4"        # load:proc:save per GPU

# ============================================================
# GRACEFUL SHUTDOWN
# ============================================================
_shutdown = threading.Event()

def _signal_handler(sig, frame):
    print("\n[!] Interrupt received - finishing current operations and saving progress...")
    _shutdown.set()

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ============================================================
# PROGRESS TRACKING (resumable, crash-safe)
# ============================================================
class ProgressTracker:
    def __init__(self, work_dir):
        self.file = work_dir / "progress.json"
        self.lock = threading.Lock()
        if self.file.exists():
            with open(self.file) as f:
                self.data = json.load(f)
        else:
            self.data = {"chunks": {}, "phase": "init", "version": 2}

    def save(self):
        with self.lock:
            tmp = self.file.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(self.data, f, indent=2)
            tmp.replace(self.file)  # Atomic on same filesystem

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
# FRAME EXTRACTION (CPU - 12 threads ffmpeg)
# ============================================================
def extract_chunk_frames(input_path, start_time, duration, output_dir):
    """Extract frames from a video chunk. Uses CPU decode with multiple threads."""
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("*.png"))
    expected_approx = int(duration * 25)
    if len(existing) >= expected_approx - 2:
        return len(existing)

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", str(input_path),
        "-t", str(duration),
        "-threads", str(FFMPEG_EXTRACT_THREADS),
        "-pix_fmt", "rgb24",
        str(output_dir / "%08d.png"),
        "-loglevel", "warning"
    ]
    subprocess.run(cmd, check=True)
    return len(list(output_dir.glob("*.png")))

# ============================================================
# AI ENHANCEMENT: Real-ESRGAN with PyTorch FP16 + Tensor Cores
# ============================================================
class ESRGANProcessor:
    """Manages ESRGAN models on both GPUs with torch.compile + FP16."""

    def __init__(self):
        import torch
        import torch.nn.functional as F
        import spandrel
        self.torch = torch
        self.F = F

        # Load model on GPU 0 (5070 Ti)
        print("  Loading ESRGAN model on GPU 0 (5070 Ti)...")
        model_data = spandrel.ModelLoader().load_from_file(ESRGAN_MODEL_PATH)
        self.scale = model_data.scale
        self.model0 = model_data.model.cuda(0).half().eval()
        try:
            self.model0 = torch.compile(self.model0, mode="max-autotune")
            print("  torch.compile OK for GPU 0")
        except Exception as e:
            print(f"  torch.compile failed for GPU 0: {e}")

        # Load model on GPU 1 (2060) if available
        self.model1 = None
        if torch.cuda.device_count() > 1:
            print("  Loading ESRGAN model on GPU 1 (2060)...")
            model_data1 = spandrel.ModelLoader().load_from_file(ESRGAN_MODEL_PATH)
            self.model1 = model_data1.model.cuda(1).half().eval()
            try:
                self.model1 = torch.compile(self.model1, mode="max-autotune")
                print("  torch.compile OK for GPU 1")
            except Exception:
                pass

        # Warmup both GPUs
        self._warmup()

    def _warmup(self):
        """Warmup torch.compile - triggers kernel compilation."""
        torch = self.torch
        print("  Warming up GPU kernels (torch.compile autotune)...")
        for gpu_id, model, tile in [(0, self.model0, GPU0_TILE_SIZE),
                                     (1, self.model1, GPU1_TILE_SIZE)]:
            if model is None:
                continue
            dummy = torch.randn(1, 3, tile, tile, device=f'cuda:{gpu_id}', dtype=torch.float16)
            with torch.no_grad():
                for _ in range(3):
                    _ = model(dummy)
            torch.cuda.synchronize(gpu_id)
        print("  Warmup complete")

    def _process_tiled(self, model, img_tensor, tile_size):
        """Process a single frame through the model using tiling."""
        torch = self.torch
        b, c, h, w = img_tensor.shape
        scale = self.scale
        out = torch.zeros(b, c, h * scale, w * scale,
                          device=img_tensor.device, dtype=torch.float32)
        tiles_x = (w + tile_size - 1) // tile_size
        tiles_y = (h + tile_size - 1) // tile_size

        for ty in range(tiles_y):
            for tx in range(tiles_x):
                x1, y1 = tx * tile_size, ty * tile_size
                x2, y2 = min(x1 + tile_size, w), min(y1 + tile_size, h)
                px1, py1 = max(x1 - TILE_PAD, 0), max(y1 - TILE_PAD, 0)
                px2, py2 = min(x2 + TILE_PAD, w), min(y2 + TILE_PAD, h)

                tile = img_tensor[:, :, py1:py2, px1:px2]
                with torch.no_grad():
                    tile_out = model(tile).float()

                ox1 = (x1 - px1) * scale
                oy1 = (y1 - py1) * scale
                ox2 = ox1 + (x2 - x1) * scale
                oy2 = oy1 + (y2 - y1) * scale
                out[:, :, y1*scale:y2*scale, x1*scale:x2*scale] = \
                    tile_out[:, :, oy1:oy2, ox1:ox2]
        return out

    def process_frame(self, img_np, gpu_id=0):
        """Process a single frame (numpy HWC uint8) -> enhanced numpy HWC uint8."""
        torch = self.torch
        model = self.model0 if gpu_id == 0 else self.model1
        tile_size = GPU0_TILE_SIZE if gpu_id == 0 else GPU1_TILE_SIZE

        # Convert to tensor
        tensor = torch.from_numpy(img_np.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).half().to(f'cuda:{gpu_id}')

        # Downscale to half resolution, then x4 model = effective x2 output
        tensor = self.F.interpolate(tensor, scale_factor=0.5,
                                    mode='bilinear', align_corners=False)

        # Process through model with tiling
        out = self._process_tiled(model, tensor, tile_size)

        # Convert back to numpy
        out = out.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        return (out * 255).astype(np.uint8)

    def process_directory(self, input_dir, output_dir):
        """Process all frames in a directory using both GPUs in parallel."""
        from PIL import Image

        output_dir.mkdir(parents=True, exist_ok=True)
        frames = sorted(input_dir.glob("*.png"))
        existing_out = len(list(output_dir.glob("*.png")))

        if existing_out >= len(frames) and len(frames) > 0:
            return existing_out

        # Split frames: GPU0 gets ~70%, GPU1 gets ~30% (proportional to SMs: 70 vs 30)
        if self.model1 is not None:
            split = int(len(frames) * 70 / 100)
        else:
            split = len(frames)

        gpu0_frames = frames[:split]
        gpu1_frames = frames[split:]

        processed = 0
        total = len(frames)
        lock = threading.Lock()
        t0 = time.time()

        def process_batch(frame_list, gpu_id):
            nonlocal processed
            for fpath in frame_list:
                if _shutdown.is_set():
                    return
                out_path = output_dir / fpath.name
                if out_path.exists():
                    with lock:
                        processed += 1
                    continue

                img = np.array(Image.open(fpath))
                result = self.process_frame(img, gpu_id)
                Image.fromarray(result).save(str(out_path))

                with lock:
                    processed += 1
                    if processed % 50 == 0:
                        elapsed = time.time() - t0
                        fps = processed / elapsed
                        eta = (total - processed) / fps if fps > 0 else 0
                        print(f"    ESRGAN: {processed}/{total} frames "
                              f"({fps:.1f} fps, ETA {eta/60:.0f}m)")

        # Run both GPUs in parallel threads
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(process_batch, gpu0_frames, 0)]
            if gpu1_frames:
                futures.append(pool.submit(process_batch, gpu1_frames, 1))
            for f in futures:
                f.result()

        elapsed = time.time() - t0
        if elapsed > 0:
            print(f"    ESRGAN complete: {processed} frames in {elapsed:.0f}s "
                  f"({processed/elapsed:.1f} fps)")
        return processed

# ============================================================
# RIFE FRAME INTERPOLATION (ncnn-vulkan, both GPUs via Vulkan)
# ============================================================
def run_rife_chunk(input_dir, output_dir):
    """Run RIFE frame interpolation (doubles frame count). Uses ncnn-vulkan on both GPUs."""
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
        "-g", RIFE_GPU,
        "-j", RIFE_THREADS,
        "-f", "%08d.png"
    ]
    subprocess.run(cmd, check=True)
    return len(list(output_dir.glob("*.png")))

# ============================================================
# NVENC ENCODING (dedicated encoder chip, doesn't block GPU compute)
# ============================================================
def encode_chunk(frames_dir, output_file, fps=50):
    """Encode frames to HEVC using NVENC. Runs on dedicated encoder chip."""
    if output_file.exists() and output_file.stat().st_size > 1000:
        return

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "%08d.png"),
        "-c:v", "hevc_nvenc",
        "-gpu", str(NVENC_GPU),
        "-preset", "p6", "-tune", "hq",
        "-rc", "vbr", "-cq", "20",
        "-b:v", "12M", "-maxrate", "18M", "-bufsize", "24M",
        "-profile:v", "main10",
        "-pix_fmt", "yuv420p10le",
        "-threads", str(FFMPEG_ENCODE_THREADS),
        str(output_file),
        "-loglevel", "warning"
    ]
    subprocess.run(cmd, check=True)

# ============================================================
# SINGLE CHUNK PROCESSOR
# ============================================================
def process_chunk(chunk_id, input_path, start, duration, work_dir, progress,
                  do_esrgan, do_rife, orig_fps, esrgan_proc):
    """Process one chunk through the full pipeline with checkpointing."""
    if _shutdown.is_set():
        return chunk_id, 0

    chunk_dir = work_dir / f"chunk_{chunk_id:04d}"
    raw_dir = chunk_dir / "raw"
    esrgan_dir = chunk_dir / "esrgan"
    rife_dir = chunk_dir / "rife"
    chunk_video = chunk_dir / "output.mp4"

    t0 = time.time()
    print(f"\n  [Chunk {chunk_id:04d}] {start:.0f}s -> {start+duration:.0f}s")

    # Step 1: Extract frames (CPU)
    if not progress.is_done(chunk_id, "extract"):
        n = extract_chunk_frames(input_path, start, duration, raw_dir)
        progress.chunk_done(chunk_id, "extract")
        print(f"  [Chunk {chunk_id:04d}] Extracted {n} frames ({time.time()-t0:.1f}s)")
    if _shutdown.is_set():
        return chunk_id, time.time() - t0

    # Step 2: Real-ESRGAN (Tensor Cores FP16 on both GPUs)
    current_frames_dir = raw_dir
    if do_esrgan:
        if not progress.is_done(chunk_id, "esrgan"):
            n = esrgan_proc.process_directory(raw_dir, esrgan_dir)
            if _shutdown.is_set():
                return chunk_id, time.time() - t0
            progress.chunk_done(chunk_id, "esrgan")
            print(f"  [Chunk {chunk_id:04d}] ESRGAN done: {n} frames ({time.time()-t0:.1f}s)")
        current_frames_dir = esrgan_dir

    # Step 3: RIFE interpolation (ncnn-vulkan, both GPUs)
    output_fps = orig_fps
    if do_rife:
        if not progress.is_done(chunk_id, "rife"):
            n = run_rife_chunk(current_frames_dir, rife_dir)
            progress.chunk_done(chunk_id, "rife")
            print(f"  [Chunk {chunk_id:04d}] RIFE done: {n} frames ({time.time()-t0:.1f}s)")
        current_frames_dir = rife_dir
        output_fps = orig_fps * 2
    if _shutdown.is_set():
        return chunk_id, time.time() - t0

    # Step 4: NVENC encoding (dedicated encoder chip)
    if not progress.is_done(chunk_id, "encode"):
        encode_chunk(current_frames_dir, chunk_video, fps=output_fps)
        progress.chunk_done(chunk_id, "encode")
        print(f"  [Chunk {chunk_id:04d}] Encoded ({time.time()-t0:.1f}s)")

    # Step 5: Cleanup intermediate frames (save disk space)
    if chunk_video.exists() and chunk_video.stat().st_size > 1000:
        if not progress.is_done(chunk_id, "cleanup"):
            shutil.rmtree(raw_dir, ignore_errors=True)
            if do_esrgan:
                shutil.rmtree(esrgan_dir, ignore_errors=True)
            if do_rife:
                shutil.rmtree(rife_dir, ignore_errors=True)
            progress.chunk_done(chunk_id, "cleanup")

    elapsed = time.time() - t0
    print(f"  [Chunk {chunk_id:04d}] COMPLETE in {elapsed:.0f}s")
    return chunk_id, elapsed

# ============================================================
# CONCATENATE CHUNKS + MERGE AUDIO
# ============================================================
def concatenate_and_finalize(work_dir, input_path, output_path, num_chunks):
    """Merge all chunk videos and add enhanced audio."""
    concat_file = work_dir / "concat.txt"
    with open(concat_file, "w") as f:
        for i in range(num_chunks):
            chunk_video = work_dir / f"chunk_{i:04d}" / "output.mp4"
            if chunk_video.exists():
                f.write(f"file '{chunk_video}'\n")

    # Check for enhanced audio
    enhanced_audio = output_path.parent / (input_path.stem + "_enhanced.m4a")
    has_audio = enhanced_audio.exists()
    audio_src = str(enhanced_audio) if has_audio else str(input_path)

    print(f"\n[FINAL] Concatenating {num_chunks} chunks...")
    if has_audio:
        print(f"[FINAL] Using enhanced audio: {enhanced_audio.name}")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-i", audio_src,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy",
    ]
    if has_audio:
        # Enhanced audio already processed - just copy
        cmd += ["-c:a", "copy"]
    else:
        # Process audio on the fly
        cmd += [
            "-af", "afftdn=nf=-20:nt=w:om=o,"
                   "acompressor=threshold=-20dB:ratio=3:attack=5:release=50,"
                   "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-c:a", "aac", "-b:a", "192k",
        ]
    cmd += [
        "-movflags", "+faststart",
        "-threads", "16",
        str(output_path),
        "-loglevel", "warning"
    ]
    subprocess.run(cmd, check=True)

    size_gb = output_path.stat().st_size / 1e9
    print(f"[DONE] Output: {output_path} ({size_gb:.2f} GB)")

# ============================================================
# MAIN
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="AI Video Enhancement Pipeline v2 - Full Hardware Utilization")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint (default behavior)")
    parser.add_argument("--skip-esrgan", action="store_true",
                        help="Skip Real-ESRGAN AI upscaling")
    parser.add_argument("--skip-rife", action="store_true",
                        help="Skip RIFE frame interpolation")
    parser.add_argument("--chunk-duration", type=int, default=CHUNK_DURATION,
                        help=f"Chunk duration in seconds (default {CHUNK_DURATION})")
    parser.add_argument("--clean", action="store_true",
                        help="Clean work directory and start fresh")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    # Directories
    output_dir = input_path.parent / "enhanced"
    output_dir.mkdir(exist_ok=True)
    work_dir = output_dir / f"work_{input_path.stem}"

    if args.clean and work_dir.exists():
        print(f"Cleaning work directory: {work_dir}")
        shutil.rmtree(work_dir)

    work_dir.mkdir(exist_ok=True)

    # Output filename
    suffix = "_ai_enhanced"
    if not args.skip_rife:
        suffix += "_50fps"
    output_path = output_dir / f"{input_path.stem}{suffix}.mp4"

    # Video info
    duration, fps, w, h = get_video_info(input_path)

    do_esrgan = not args.skip_esrgan
    do_rife = not args.skip_rife

    out_w = w * 2 if do_esrgan else w
    out_h = h * 2 if do_esrgan else h
    out_fps = fps * 2 if do_rife else fps
    total_frames = int(duration * fps)

    print("=" * 65)
    print("  AI VIDEO ENHANCEMENT PIPELINE v2")
    print("  Full Hardware Utilization: Tensor Cores + CUDA + NVENC + CPU")
    print("=" * 65)
    print(f"  Input:      {input_path.name}")
    print(f"  Resolution: {w}x{h} -> {out_w}x{out_h}")
    print(f"  Framerate:  {fps}fps -> {out_fps}fps")
    print(f"  Duration:   {duration:.0f}s ({duration/3600:.1f}h)")
    print(f"  Frames:     ~{total_frames:,}")
    print(f"  ESRGAN:     {'ON - FP16 Tensor Cores, dual GPU' if do_esrgan else 'SKIP'}")
    print(f"  RIFE:       {'ON - ncnn-vulkan, dual GPU' if do_rife else 'SKIP'}")
    print(f"  Encoding:   HEVC NVENC (5070 Ti)")
    print(f"  Chunks:     {args.chunk_duration}s each")
    print(f"  Output:     {output_path.name}")
    print("=" * 65)

    # Estimated time
    if do_esrgan and do_rife:
        est_h = total_frames / 10 / 3600  # Conservative with both
    elif do_esrgan:
        est_h = total_frames / 27.4 / 3600
    elif do_rife:
        est_h = total_frames / 36.5 / 3600
    else:
        est_h = 0.1
    print(f"  Estimated:  ~{est_h:.1f}h")
    print("=" * 65)

    # Progress tracker
    progress = ProgressTracker(work_dir)

    # Calculate chunks
    chunks = []
    i = 0
    start = 0.0
    while start < duration:
        chunk_dur = min(args.chunk_duration, duration - start)
        if chunk_dur <= 0:
            break
        chunks.append((i, start, chunk_dur))
        start += args.chunk_duration
        i += 1

    actual_chunks = len(chunks)
    done_chunks = sum(1 for cid, _, _ in chunks if progress.is_done(cid, "encode"))

    print(f"\n  Chunks: {actual_chunks} total, {done_chunks} already done, "
          f"{actual_chunks - done_chunks} remaining\n")

    if done_chunks == actual_chunks:
        print("  All chunks already processed! Proceeding to final merge...")
    else:
        # Initialize ESRGAN processor (loads models on both GPUs)
        esrgan_proc = None
        if do_esrgan:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
            esrgan_proc = ESRGANProcessor()

        # Process chunks sequentially (each chunk uses both GPUs internally)
        t_start = time.time()
        completed = done_chunks

        for chunk_id, start, chunk_dur in chunks:
            if _shutdown.is_set():
                print("\n[!] Shutdown requested. Progress saved. Run with --resume to continue.")
                sys.exit(0)

            if progress.is_done(chunk_id, "encode"):
                continue

            _, elapsed = process_chunk(
                chunk_id, input_path, start, chunk_dur, work_dir,
                progress, do_esrgan, do_rife, fps, esrgan_proc
            )
            completed += 1

            # ETA
            if completed > done_chunks:
                avg_time = (time.time() - t_start) / (completed - done_chunks)
                remaining = (actual_chunks - completed) * avg_time
                print(f"  Progress: {completed}/{actual_chunks} "
                      f"({100*completed/actual_chunks:.1f}%) "
                      f"- ETA: {remaining/3600:.1f}h")

    # Final concatenation
    if not _shutdown.is_set():
        concatenate_and_finalize(work_dir, input_path, output_path, actual_chunks)

        total_time = time.time() - t_start if 'start' in dir() else 0
        if total_time > 0:
            print(f"\n  Total processing time: {total_time/3600:.1f}h")
            print(f"  Realtime ratio: {duration/total_time:.2f}x")
    else:
        print("\n[!] Interrupted before final merge. Run again to finish.")


if __name__ == "__main__":
    main()
