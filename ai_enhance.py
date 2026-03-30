#!/usr/bin/env python3
"""
AI Video Enhancement Pipeline v4 — Maximum Silicon Utilization

Target hardware:
  CPU:  Ryzen 9 9950X3D  16C/32T 5.8GHz  192MB 3D V-Cache  AVX-512
  GPU0: RTX 5070 Ti      16 GB  70 SMs  SM 12.0  Tensor Cores 5th-gen  NVENC
  GPU1: RTX 2060          6 GB  30 SMs  SM 7.5   Tensor Cores 2nd-gen
  RAM:  128 GB DDR5
  Disk: NVMe RAID-0  6.7 GB/s

Key improvements over v3:
  - NO TILING: whole frame fits in VRAM (~300MB peak for 1120x630)
  - torch.compile reduce-overhead (not max-autotune — avoids 12GB RAM / minutes compile)
  - Kernel cache: TORCHINDUCTOR_FX_GRAPH_CACHE=1 for instant re-runs
  - Dual-GPU frame-level parallelism with async CUDA streams
  - PNG I/O on CPU threadpool
  - 4-stage pipeline overlap: extract || ESRGAN || RIFE || encode

Resumable: progress JSON per chunk/phase. Safe to Ctrl-C.

Usage:
  CUDA_VISIBLE_DEVICES=0,1 python3 ai_enhance.py <input.mp4>
  CUDA_VISIBLE_DEVICES=0,1 python3 ai_enhance.py <input.mp4> --skip-rife
  CUDA_VISIBLE_DEVICES=0,1 python3 ai_enhance.py <input.mp4> --chunk 30 --skip-rife --clean
"""

import os

# Must be set before any torch import
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Reduce torch.compile worker count (default=nproc=32 eats 12GB RAM)
os.environ.setdefault("TORCH_COMPILE_THREADS", "4")
# Enable inductor FX graph cache for instant re-runs
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")

import sys, subprocess, time, json, shutil, signal, threading, queue
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# ── CONFIG ──────────────────────────────────────────────────
CHUNK_SECONDS       = 120
ESRGAN_MODEL_PATH   = "/tmp/realesr-animevideov3.pth"
RIFE_BIN            = "/tmp/rife-ncnn/rife-ncnn-vulkan-20221029-ubuntu/rife-ncnn-vulkan"
RIFE_MODEL_DIR      = "/tmp/rife-ncnn/rife-ncnn-vulkan-20221029-ubuntu/rife-v4.6"

EXTRACT_THREADS     = 12
ENCODE_THREADS      = 4
RIFE_GPU_THREADS    = "4:4:4"
PIPELINE_DEPTH      = 3
NVENC_GPU           = 0
GPU0_SHARE          = 0.70   # fraction of frames assigned to GPU0
PNG_WRITE_WORKERS   = 8      # CPU threads for parallel PNG writes

# ── GRACEFUL SHUTDOWN ───────────────────────────────────────
_shutdown = threading.Event()

def _on_signal(sig, frame):
    print("\n[!] Interrupt — saving progress…")
    _shutdown.set()

signal.signal(signal.SIGINT,  _on_signal)
signal.signal(signal.SIGTERM, _on_signal)

# ── PROGRESS TRACKER ────────────────────────────────────────
class Progress:
    def __init__(self, work_dir):
        self.path = work_dir / "progress.json"
        self.lock = threading.Lock()
        self.data = {"chunks": {}, "version": 4}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                pass

    def _flush(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.path)

    def done(self, cid, phase):
        return self.data["chunks"].get(str(cid), {}).get(phase, False)

    def mark(self, cid, phase):
        with self.lock:
            self.data["chunks"].setdefault(str(cid), {})[phase] = True
            self._flush()

# ── VIDEO INFO ──────────────────────────────────────────────
def probe(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True)
    info = json.loads(r.stdout)
    dur = float(info["format"]["duration"])
    for s in info["streams"]:
        if s["codec_type"] == "video":
            num, den = s["r_frame_rate"].split("/")
            fps = float(num) / float(den)
            return dur, fps, int(s["width"]), int(s["height"])
    raise RuntimeError("no video stream")

# ── STAGE 0: FRAME EXTRACTION (CPU + NVDEC) ────────────────
def extract_to_pngs(src, start, dur, out_dir, fps):
    """Extract frames using NVDEC hw decode + CPU threading."""
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = int(dur * fps)
    existing = sorted(out_dir.glob("*.png"))
    if len(existing) >= max(expected - 2, 1):
        return len(existing)

    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda", "-hwaccel_device", "0",
        "-ss", str(start), "-i", str(src), "-t", str(dur),
        "-pix_fmt", "rgb24",
        "-threads", str(EXTRACT_THREADS),
        str(out_dir / "%08d.png"),
        "-loglevel", "warning",
    ]
    subprocess.run(cmd, check=True)
    return len(list(out_dir.glob("*.png")))

# ── STAGE 1: ESRGAN ENGINE (Tensor Cores FP16, NO TILING) ──
class ESRGANEngine:
    """
    Process whole frames at once — no tiling needed.
    Input 2240x1260 -> downscale 0.5x -> 1120x630 -> model x4 -> 4480x2520.
    Peak VRAM ~300MB per frame. Fits easily on both GPUs.
    """

    def __init__(self):
        import torch
        import spandrel
        self.torch = torch

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.ngpu = torch.cuda.device_count()
        self.models  = []
        self.streams = []
        self.scale = None

        for gid in range(min(self.ngpu, 2)):
            name = torch.cuda.get_device_name(gid)
            vram = torch.cuda.get_device_properties(gid).total_mem / 1e9
            print(f"  Loading ESRGAN on GPU {gid} ({name}, {vram:.1f}GB)...")

            m = spandrel.ModelLoader().load_from_file(ESRGAN_MODEL_PATH)
            if self.scale is None:
                self.scale = m.scale
            net = m.model.half().to(f"cuda:{gid}").eval()

            # reduce-overhead: fast compile (~15s), good runtime, no 12GB RAM autotune
            try:
                net = torch.compile(net, mode="reduce-overhead")
                print(f"    torch.compile OK (reduce-overhead)")
            except Exception as e:
                print(f"    torch.compile skipped: {e}")

            self.models.append(net)
            self.streams.append(torch.cuda.Stream(device=f"cuda:{gid}"))

        # Warmup: trigger compilation with a realistic input shape
        self._warmup()

    def _warmup(self):
        torch = self.torch
        # Use the actual downscaled frame size for warmup
        # Input will be ~1120x630 after 0.5x downscale of 2240x1260
        wh = (1120, 630)
        print(f"  Warming up ESRGAN ({wh[0]}x{wh[1]} input)...")
        for gid, net in enumerate(self.models):
            dev = f"cuda:{gid}"
            d = torch.randn(1, 3, wh[1], wh[0], device=dev, dtype=torch.float16)
            with torch.no_grad(), torch.amp.autocast("cuda"):
                _ = net(d)  # first call triggers compile
                _ = net(d)  # second call uses cached kernel
            torch.cuda.synchronize(gid)
            vram_used = torch.cuda.memory_allocated(gid) / 1e9
            print(f"    GPU{gid} warm (VRAM: {vram_used:.2f}GB)")
        print("  Warmup done")

    def _process_frame_gpu(self, img_np, gid):
        """Process a single frame entirely on one GPU. No tiling."""
        torch = self.torch
        net = self.models[gid]
        dev = f"cuda:{gid}"
        stream = self.streams[gid]

        with torch.cuda.stream(stream):
            # numpy HWC uint8 -> tensor CHW FP16
            t = torch.from_numpy(img_np).permute(2, 0, 1).to(
                dev, dtype=torch.float16, non_blocking=True) / 255.0

            # Downscale 0.5x for effective x2 output
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0), scale_factor=0.5,
                mode="bilinear", align_corners=False)

            # ESRGAN forward — whole frame at once
            with torch.no_grad(), torch.amp.autocast("cuda"):
                out = net(t)

            # FP16->FP32->uint8 on GPU, then to CPU
            out_np = (out.squeeze(0).float().clamp(0, 1) * 255
                      ).byte().permute(1, 2, 0).cpu().numpy()

        return out_np  # HWC uint8 RGB

    def process_directory(self, in_dir, out_dir):
        """Process all frames in directory with dual-GPU parallelism."""
        import cv2
        torch = self.torch

        out_dir.mkdir(parents=True, exist_ok=True)
        frames = sorted(in_dir.glob("*.png"))
        if not frames:
            return 0

        n_existing = len(list(out_dir.glob("*.png")))
        if n_existing >= len(frames):
            print(f"    ESRGAN: {n_existing} frames already done, skip")
            return n_existing

        # Split frames between GPUs
        split = int(len(frames) * GPU0_SHARE) if self.ngpu > 1 else len(frames)
        assignments = []  # (frame_path, gpu_id)
        for i, f in enumerate(frames):
            gid = 0 if i < split else 1
            assignments.append((f, gid))

        total    = len(frames)
        counter  = [0]
        lock     = threading.Lock()
        t0       = time.time()

        # Thread pool for parallel PNG writes
        write_pool = ThreadPoolExecutor(max_workers=PNG_WRITE_WORKERS)
        write_futures = []

        def _save_png(path, img_bgr):
            cv2.imwrite(str(path), img_bgr)

        def _gpu_worker(frame_list, gid):
            """Process frames assigned to one GPU sequentially."""
            for fpath in frame_list:
                if _shutdown.is_set():
                    return
                dst = out_dir / fpath.name
                if dst.exists():
                    with lock:
                        counter[0] += 1
                    continue

                # Read frame (CPU)
                img = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # GPU inference
                out_rgb = self._process_frame_gpu(img_rgb, gid)

                # Async PNG write (CPU threadpool)
                out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
                fut = write_pool.submit(_save_png, dst, out_bgr)
                write_futures.append(fut)

                with lock:
                    counter[0] += 1
                    c = counter[0]
                    if c % 10 == 0 or c == total:
                        elapsed = time.time() - t0
                        fps_now = c / elapsed if elapsed > 0 else 0
                        eta = (total - c) / fps_now if fps_now > 0 else 0
                        g0m = torch.cuda.memory_allocated(0) / 1e9
                        g1m = (torch.cuda.memory_allocated(1) / 1e9
                               if self.ngpu > 1 else 0)
                        print(f"    ESRGAN {c}/{total}  {fps_now:.1f}fps  "
                              f"ETA {eta:.0f}s  VRAM [{g0m:.1f}|{g1m:.1f}]GB")

        # Split into per-GPU lists
        gpu0_frames = [f for f, g in assignments if g == 0]
        gpu1_frames = [f for f, g in assignments if g == 1]

        threads = []
        if gpu0_frames:
            t = threading.Thread(target=_gpu_worker, args=(gpu0_frames, 0))
            t.start()
            threads.append(t)
        if gpu1_frames and self.ngpu > 1:
            t = threading.Thread(target=_gpu_worker, args=(gpu1_frames, 1))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        # Wait for all PNG writes to finish
        for fut in write_futures:
            fut.result()
        write_pool.shutdown(wait=True)

        elapsed = time.time() - t0
        fps_final = counter[0] / elapsed if elapsed > 0 else 0
        print(f"    ESRGAN done: {counter[0]} frames  {elapsed:.0f}s  "
              f"{fps_final:.1f}fps")
        return counter[0]

# ── STAGE 2: RIFE (ncnn-Vulkan, both GPUs) ──────────────────
def rife_interpolate(in_dir, out_dir):
    """Frame interpolation using RIFE ncnn-vulkan (Vulkan compute, not CUDA)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_in  = len(list(in_dir.glob("*.png")))
    n_out = len(list(out_dir.glob("*.png")))
    if n_out >= n_in * 2 - 1 and n_in > 0:
        return n_out
    cmd = [
        RIFE_BIN,
        "-i", str(in_dir),  "-o", str(out_dir),
        "-m", RIFE_MODEL_DIR,
        "-g", "0,1",  "-j", RIFE_GPU_THREADS,
        "-f", "%08d.png",
    ]
    subprocess.run(cmd, check=True)
    return len(list(out_dir.glob("*.png")))

# ── STAGE 3: NVENC ENCODE (dedicated ASIC) ──────────────────
def nvenc_encode(frames_dir, out_file, fps):
    """HEVC encode using NVENC hardware encoder on GPU0."""
    if out_file.exists() and out_file.stat().st_size > 1000:
        return
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "%08d.png"),
        "-c:v", "hevc_nvenc", "-gpu", str(NVENC_GPU),
        "-preset", "p6", "-tune", "hq",
        "-rc", "vbr", "-cq", "20",
        "-b:v", "12M", "-maxrate", "18M", "-bufsize", "24M",
        "-profile:v", "main10", "-pix_fmt", "yuv420p10le",
        "-threads", str(ENCODE_THREADS),
        str(out_file), "-loglevel", "warning",
    ]
    subprocess.run(cmd, check=True)

# ── PIPELINE: 4-stage overlap across chunks ─────────────────
def run_pipeline(chunks, src, work, prog, do_esr, do_rife, fps, esr):
    total   = len(chunks)
    done_n  = sum(1 for c in chunks if prog.done(c[0], "encode"))
    pending = [c for c in chunks if not prog.done(c[0], "encode")]
    if not pending:
        print(f"  All {total} chunks already processed!")
        return done_n

    # Pre-extraction thread (CPU + NVDEC - different chip than Tensor cores)
    extract_q = queue.Queue(maxsize=PIPELINE_DEPTH)
    encode_q  = queue.Queue()

    def pre_extractor():
        """Extract frames for upcoming chunks while GPU works on current."""
        for cid, start, dur in pending:
            if _shutdown.is_set():
                break
            raw = work / f"chunk_{cid:04d}" / "raw"
            if not prog.done(cid, "extract"):
                t0 = time.time()
                n = extract_to_pngs(src, start, dur, raw, fps)
                dt = time.time() - t0
                print(f"  [extract] chunk {cid:04d}: {n} frames ({dt:.1f}s)")
                prog.mark(cid, "extract")
            extract_q.put(cid)
        extract_q.put(None)  # sentinel

    def bg_encoder():
        """Encode chunks using NVENC ASIC while GPU processes next chunk."""
        while True:
            item = encode_q.get()
            if item is None:
                break
            cid, cur_dir, out_fps_val = item
            vid = work / f"chunk_{cid:04d}" / "output.mp4"
            if not prog.done(cid, "encode"):
                try:
                    t0 = time.time()
                    nvenc_encode(cur_dir, vid, out_fps_val)
                    prog.mark(cid, "encode")
                    dt = time.time() - t0
                    print(f"  [NVENC] chunk {cid:04d} encoded ({dt:.1f}s)")
                except Exception as e:
                    print(f"  [!] Encode chunk {cid:04d} failed: {e}")

            # Cleanup intermediate frames to save disk
            if vid.exists() and vid.stat().st_size > 1000:
                for p in ["raw", "esrgan", "rife"]:
                    d = work / f"chunk_{cid:04d}" / p
                    if d.exists():
                        shutil.rmtree(d, ignore_errors=True)
                prog.mark(cid, "clean")
            encode_q.task_done()

    t_ext = threading.Thread(target=pre_extractor, daemon=True, name="extractor")
    t_enc = threading.Thread(target=bg_encoder, daemon=True, name="encoder")
    t_ext.start()
    t_enc.start()

    t_start   = time.time()
    completed = done_n

    for cid, start, dur in pending:
        if _shutdown.is_set():
            break

        # Wait for extraction to finish for this chunk
        ready = extract_q.get()
        if ready is None:
            break

        tc = time.time()
        chunk_frames_est = int(dur * fps)
        print(f"\n  == Chunk {cid:04d}  [{start:.0f}s-{start+dur:.0f}s]  "
              f"~{chunk_frames_est} frames ==")

        cur = work / f"chunk_{cid:04d}" / "raw"

        # ESRGAN stage (Tensor Cores + CUDA)
        if do_esr:
            esr_out = work / f"chunk_{cid:04d}" / "esrgan"
            if not prog.done(cid, "esrgan"):
                n = esr.process_directory(cur, esr_out)
                if _shutdown.is_set():
                    break
                prog.mark(cid, "esrgan")
                dt = time.time() - tc
                print(f"  | ESRGAN:  {n} frames  ({dt:.1f}s)")
            cur = esr_out

        # RIFE stage (Vulkan compute - doesn't conflict with CUDA)
        out_fps = fps
        if do_rife:
            rife_out = work / f"chunk_{cid:04d}" / "rife"
            if not prog.done(cid, "rife"):
                t_rife = time.time()
                n = rife_interpolate(cur, rife_out)
                dt = time.time() - t_rife
                prog.mark(cid, "rife")
                print(f"  | RIFE:    {n} frames  ({dt:.1f}s)")
            cur = rife_out
            out_fps = fps * 2

        # Queue for NVENC encode (runs on dedicated ASIC while we do next chunk)
        encode_q.put((cid, cur, out_fps))

        completed += 1
        wall = time.time() - t_start
        avg = wall / max(completed - done_n, 1)
        remaining = (total - completed) * avg
        pct = 100 * completed / total
        print(f"  | Progress: {completed}/{total} ({pct:.1f}%)  "
              f"ETA {remaining/3600:.1f}h")
        print(f"  == Chunk {cid:04d} GPU done {time.time()-tc:.0f}s "
              f"(encode queued -> NVENC)")

    # Wait for all encodes to finish
    encode_q.join()
    encode_q.put(None)  # tell encoder thread to exit
    t_enc.join(timeout=30)
    return completed

# ── FINAL MERGE ─────────────────────────────────────────────
def merge_output(work, src, dst, n_chunks):
    """Concatenate all chunk videos + audio into final output."""
    concat = work / "concat.txt"
    with open(concat, "w") as f:
        for i in range(n_chunks):
            v = work / f"chunk_{i:04d}" / "output.mp4"
            if v.exists():
                f.write(f"file '{v}'\n")

    # Prefer enhanced audio if available
    enh_audio = dst.parent / "GMT20260320-130023_Recording_enhanced.m4a"
    if enh_audio.exists():
        audio_src = str(enh_audio)
        audio_codec = ["-c:a", "copy"]
    else:
        audio_src = str(src)
        audio_codec = [
            "-af", ("afftdn=nf=-20:nt=w:om=o,"
                    "acompressor=threshold=-20dB:ratio=3:attack=5:release=50,"
                    "loudnorm=I=-16:TP=-1.5:LRA=11"),
            "-c:a", "aac", "-b:a", "192k"]

    print(f"\n[MERGE] {n_chunks} chunks -> {dst.name}")
    cmd = ["ffmpeg", "-y",
           "-f", "concat", "-safe", "0", "-i", str(concat),
           "-i", audio_src,
           "-map", "0:v", "-map", "1:a",
           "-c:v", "copy"] + audio_codec + [
           "-movflags", "+faststart", "-threads", "16",
           str(dst), "-loglevel", "warning"]
    subprocess.run(cmd, check=True)
    size_gb = dst.stat().st_size / 1e9
    print(f"[DONE] {dst}  ({size_gb:.2f} GB)")

# ── MAIN ────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser(description="AI Video Enhancement v4")
    ap.add_argument("input", help="Input video file")
    ap.add_argument("--skip-esrgan", action="store_true",
                    help="Skip ESRGAN upscaling")
    ap.add_argument("--skip-rife", action="store_true",
                    help="Skip RIFE frame interpolation")
    ap.add_argument("--chunk", type=int, default=CHUNK_SECONDS,
                    help=f"Chunk duration in seconds (default: {CHUNK_SECONDS})")
    ap.add_argument("--clean", action="store_true",
                    help="Clean work directory before starting")
    args = ap.parse_args()

    src = Path(args.input).resolve()
    if not src.exists():
        print(f"[!] File not found: {src}")
        sys.exit(1)

    out_dir = src.parent / "enhanced"
    out_dir.mkdir(exist_ok=True)
    work = out_dir / f"work_{src.stem}"

    if args.clean and work.exists():
        shutil.rmtree(work)
    work.mkdir(exist_ok=True)

    do_esr  = not args.skip_esrgan
    do_rife = not args.skip_rife

    dur, fps, w, h = probe(src)
    scale = 2 if do_esr else 1  # 0.5x downscale + 4x model = 2x
    out_w, out_h = w * scale, h * scale
    out_fps = fps * 2 if do_rife else fps
    total_frames = int(dur * fps)

    suffix = "_ai_enhanced"
    if do_rife:
        suffix += f"_{int(out_fps)}fps"
    dst = out_dir / f"{src.stem}{suffix}.mp4"

    print("=" * 65)
    print("  AI VIDEO ENHANCEMENT v4 — MAXIMUM SILICON UTILIZATION")
    print("=" * 65)
    print(f"  Input:       {src.name}")
    print(f"  Resolution:  {w}x{h}  ->  {out_w}x{out_h}")
    print(f"  Framerate:   {fps}fps  ->  {out_fps}fps")
    print(f"  Duration:    {dur:.0f}s  ({dur/3600:.1f}h)")
    print(f"  Frames:      ~{total_frames:,}")
    print(f"  ---")
    print(f"  ESRGAN:      {'ON  FP16 whole-frame (no tiling!) + torch.compile' if do_esr else 'SKIP'}")
    print(f"  RIFE:        {'ON  ncnn-Vulkan (dual GPU)' if do_rife else 'SKIP'}")
    print(f"  Encode:      HEVC NVENC (dedicated ASIC)")
    print(f"  Extract:     NVDEC hw decode + {EXTRACT_THREADS} CPU threads")
    print(f"  Pipeline:    4-stage overlap (extract | ESRGAN | RIFE | encode)")
    print(f"  Chunks:      {args.chunk}s each")
    print(f"  Output:      {dst.name}")
    print("=" * 65)

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
    n_done   = sum(1 for c in chunks if prog.done(c[0], "encode"))
    print(f"\n  Chunks: {n_chunks} total, {n_done} done, "
          f"{n_chunks - n_done} remaining\n")

    t_go = time.time()
    if n_done < n_chunks:
        esr = ESRGANEngine() if do_esr else None
        completed = run_pipeline(
            chunks, src, work, prog, do_esr, do_rife, fps, esr)
        wall = time.time() - t_go
        if wall > 0:
            speed = dur / wall
            print(f"\n  Processing: {wall/3600:.1f}h  ({speed:.2f}x realtime)")

    if not _shutdown.is_set():
        merge_output(work, src, dst, n_chunks)
    else:
        print("\n[!] Interrupted. Run again to resume from where it stopped.")

if __name__ == "__main__":
    main()
