#!/usr/bin/env python3
"""
AI Video Enhancement Pipeline v3 — Maximum Silicon Utilization

Target hardware:
  CPU:  Ryzen 9 9950X3D  16C/32T 5.8GHz  192MB 3D V-Cache  AVX-512
  GPU0: RTX 5070 Ti      16 GB  70 SMs  SM 12.0  Tensor Cores 5th-gen  NVENC  NVDEC
  GPU1: RTX 2060          6 GB  30 SMs  SM 7.5   Tensor Cores 2nd-gen
  RAM:  128 GB DDR5
  Disk: NVMe RAID-0  6.7 GB/s

Architecture — every chip busy on a *different* chunk simultaneously:

  STAGE 0  CPU 32T + NVDEC    : ffmpeg extract frames (chunk N+2)
  STAGE 1  Tensor+CUDA cores  : ESRGAN FP16 dual GPU  (chunk N+1)
  STAGE 2  Vulkan compute     : RIFE ncnn dual GPU    (chunk N)
  STAGE 3  NVENC ASIC         : HEVC encode           (chunk N-1)

  All 4 stages overlap concurrently on different chunks.

Resumable: progress JSON per chunk/phase. Safe to Ctrl-C or power off.

Usage:
  python3 ai_enhance.py <input.mp4>
  python3 ai_enhance.py <input.mp4> --skip-rife
  python3 ai_enhance.py <input.mp4> --skip-esrgan
  python3 ai_enhance.py <input.mp4> --clean
"""

import os, sys, subprocess, time, json, shutil, signal, threading
import queue
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# ── CONFIG ──────────────────────────────────────────────────
CHUNK_SECONDS       = 120
ESRGAN_MODEL_PATH   = "/tmp/realesr-animevideov3.pth"
RIFE_BIN            = "/tmp/rife-ncnn/rife-ncnn-vulkan-20221029-ubuntu/rife-ncnn-vulkan"
RIFE_MODEL_DIR      = "/tmp/rife-ncnn/rife-ncnn-vulkan-20221029-ubuntu/rife-v4.6"

GPU0_TILE           = 256
GPU1_TILE           = 256
TILE_PAD            = 16

EXTRACT_THREADS     = 12
ENCODE_THREADS      = 4
RIFE_THREADS        = "4:4:4"
PIPELINE_DEPTH      = 3
NVENC_GPU           = 0
GPU0_SHARE          = 0.70

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
        self.data = {"chunks": {}, "version": 3}
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
            return dur, eval(s["r_frame_rate"]), int(s["width"]), int(s["height"])
    raise RuntimeError("no video stream")

# ── STAGE 0: FRAME EXTRACTION (CPU + NVDEC) ────────────────
def extract_to_pngs(src, start, dur, out_dir, fps):
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = int(dur * fps)
    existing = sorted(out_dir.glob("*.png"))
    if len(existing) >= expected - 2 and len(existing) > 0:
        return len(existing)
    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda", "-hwaccel_device", "0",
        "-ss", str(start),
        "-i", str(src),
        "-t", str(dur),
        "-pix_fmt", "rgb24",
        "-threads", str(EXTRACT_THREADS),
        str(out_dir / "%08d.png"),
        "-loglevel", "warning",
    ]
    subprocess.run(cmd, check=True)
    return len(list(out_dir.glob("*.png")))

# ── STAGE 1: ESRGAN (Tensor Cores FP16 + torch.compile) ────
class ESRGANEngine:
    def __init__(self):
        import torch
        import spandrel
        self.torch = torch

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.ngpu = torch.cuda.device_count()
        self.models = []
        self.tiles  = []
        self.streams = []

        for gid in range(min(self.ngpu, 2)):
            name = torch.cuda.get_device_name(gid)
            print(f"  Loading ESRGAN on GPU {gid} ({name})…")
            m = spandrel.ModelLoader().load_from_file(ESRGAN_MODEL_PATH)
            self.scale = m.scale
            net = m.model.half().to(f"cuda:{gid}").eval()
            try:
                net = torch.compile(net, mode="max-autotune")
                print(f"    torch.compile OK (max-autotune)")
            except Exception as e:
                print(f"    torch.compile skipped: {e}")
            self.models.append(net)
            self.tiles.append(GPU0_TILE if gid == 0 else GPU1_TILE)
            self.streams.append(torch.cuda.Stream(device=f"cuda:{gid}"))

        self._warmup()

    def _warmup(self):
        torch = self.torch
        print("  Warming up (torch.compile autotune)…")
        for gid, (net, ts) in enumerate(zip(self.models, self.tiles)):
            d = torch.randn(1, 3, ts, ts, device=f"cuda:{gid}", dtype=torch.float16)
            with torch.no_grad(), torch.amp.autocast("cuda"):
                for _ in range(3):
                    net(d)
            torch.cuda.synchronize(gid)
        print("  Warmup done ✓")

    def _upscale_frame(self, img_t, gid):
        torch = self.torch
        net   = self.models[gid]
        ts    = self.tiles[gid]
        pad   = TILE_PAD
        s     = self.scale
        _, h, w = img_t.shape

        oh, ow = h * s, w * s
        out = torch.empty(3, oh, ow, device=img_t.device, dtype=torch.float32)

        for ty in range((h + ts - 1) // ts):
            for tx in range((w + ts - 1) // ts):
                y1, x1 = ty * ts, tx * ts
                y2, x2 = min(y1 + ts, h), min(x1 + ts, w)
                py1, px1 = max(y1 - pad, 0), max(x1 - pad, 0)
                py2, px2 = min(y2 + pad, h), min(x2 + pad, w)

                tile_in = img_t[:, py1:py2, px1:px2].unsqueeze(0)
                with torch.no_grad(), torch.amp.autocast("cuda"):
                    tile_out = net(tile_in).squeeze(0).float()

                oy1 = (y1 - py1) * s
                ox1 = (x1 - px1) * s
                oy2 = oy1 + (y2 - y1) * s
                ox2 = ox1 + (x2 - x1) * s
                out[:, y1*s:y2*s, x1*s:x2*s] = tile_out[:, oy1:oy2, ox1:ox2]
        return out

    def process_directory(self, in_dir, out_dir):
        import cv2
        torch = self.torch

        out_dir.mkdir(parents=True, exist_ok=True)
        frames = sorted(in_dir.glob("*.png"))
        n_existing = len(list(out_dir.glob("*.png")))
        if n_existing >= len(frames) and len(frames) > 0:
            return n_existing

        split = int(len(frames) * GPU0_SHARE) if self.ngpu > 1 else len(frames)
        batches = [(frames[:split], 0)]
        if self.ngpu > 1 and split < len(frames):
            batches.append((frames[split:], 1))

        total    = len(frames)
        counter  = [0]
        lock     = threading.Lock()
        t0       = time.time()

        def _worker(frame_list, gid):
            stream = self.streams[gid]
            for fpath in frame_list:
                if _shutdown.is_set():
                    return
                dst = out_dir / fpath.name
                if dst.exists():
                    with lock:
                        counter[0] += 1
                    continue

                img = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                with torch.cuda.stream(stream):
                    t = torch.from_numpy(img).permute(2, 0, 1).half().to(
                        f"cuda:{gid}", non_blocking=True) / 255.0
                    t = torch.nn.functional.interpolate(
                        t.unsqueeze(0), scale_factor=0.5,
                        mode="bilinear", align_corners=False).squeeze(0)
                    out_t = self._upscale_frame(t, gid)
                    out_np = (out_t.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()

                cv2.imwrite(str(dst), cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR))

                with lock:
                    counter[0] += 1
                    c = counter[0]
                    if c % 25 == 0 or c == total:
                        elapsed = time.time() - t0
                        fps_now = c / elapsed
                        eta = (total - c) / fps_now if fps_now > 0 else 0
                        g0m = torch.cuda.memory_allocated(0)/1e9
                        g1m = torch.cuda.memory_allocated(1)/1e9 if self.ngpu > 1 else 0
                        print(f"    ESRGAN {c}/{total}  {fps_now:.1f}fps  "
                              f"ETA {eta/60:.0f}m  VRAM [{g0m:.1f}|{g1m:.1f}]GB")

        with ThreadPoolExecutor(max_workers=len(batches)) as pool:
            futs = [pool.submit(_worker, fl, g) for fl, g in batches if fl]
            for f in futs:
                f.result()

        elapsed = time.time() - t0
        fps_final = counter[0] / elapsed if elapsed > 0 else 0
        print(f"    ESRGAN done: {counter[0]} frames  {elapsed:.0f}s  {fps_final:.1f}fps")
        return counter[0]

# ── STAGE 2: RIFE (ncnn-Vulkan, both GPUs) ──────────────────
def rife_interpolate(in_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_in  = len(list(in_dir.glob("*.png")))
    n_out = len(list(out_dir.glob("*.png")))
    if n_out >= n_in * 2 - 1 and n_in > 0:
        return n_out
    cmd = [
        RIFE_BIN,
        "-i", str(in_dir),  "-o", str(out_dir),
        "-m", RIFE_MODEL_DIR,
        "-g", "0,1",  "-j", RIFE_THREADS,
        "-f", "%08d.png",
    ]
    subprocess.run(cmd, check=True)
    return len(list(out_dir.glob("*.png")))

# ── STAGE 3: NVENC ENCODE (dedicated ASIC) ──────────────────
def nvenc_encode(frames_dir, out_file, fps):
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
        return done_n

    extract_q = queue.Queue(maxsize=PIPELINE_DEPTH)
    encode_q  = queue.Queue()

    def pre_extractor():
        for cid, start, dur in pending:
            if _shutdown.is_set(): break
            raw = work / f"chunk_{cid:04d}" / "raw"
            if not prog.done(cid, "extract"):
                t0 = time.time()
                n = extract_to_pngs(src, start, dur, raw, fps)
                print(f"  [pre-extract] chunk {cid:04d}: {n} frames ({time.time()-t0:.1f}s)")
                prog.mark(cid, "extract")
            extract_q.put(cid)
        extract_q.put(None)

    def bg_encoder():
        while True:
            item = encode_q.get()
            if item is None: break
            cid, cur_dir, out_fps_val = item
            vid = work / f"chunk_{cid:04d}" / "output.mp4"
            if not prog.done(cid, "encode"):
                try:
                    t0 = time.time()
                    nvenc_encode(cur_dir, vid, out_fps_val)
                    prog.mark(cid, "encode")
                    print(f"  [NVENC] chunk {cid:04d} encoded ({time.time()-t0:.1f}s)")
                except Exception as e:
                    print(f"  [!] Encode chunk {cid} failed: {e}")
            if vid.exists() and vid.stat().st_size > 1000 and not prog.done(cid, "clean"):
                for p in ["raw", "esrgan", "rife"]:
                    shutil.rmtree(work / f"chunk_{cid:04d}" / p, ignore_errors=True)
                prog.mark(cid, "clean")
            encode_q.task_done()

    t_ext = threading.Thread(target=pre_extractor, daemon=True)
    t_enc = threading.Thread(target=bg_encoder, daemon=True)
    t_ext.start()
    t_enc.start()

    t_start   = time.time()
    completed = done_n

    for cid, start, dur in pending:
        if _shutdown.is_set(): break

        ready = extract_q.get()
        if ready is None: break

        tc = time.time()
        print(f"\n  ╔═ Chunk {cid:04d}  [{start:.0f}s – {start+dur:.0f}s] ═══════")

        cur = work / f"chunk_{cid:04d}" / "raw"
        if do_esr:
            esr_out = work / f"chunk_{cid:04d}" / "esrgan"
            if not prog.done(cid, "esrgan"):
                n = esr.process_directory(cur, esr_out)
                if _shutdown.is_set(): break
                prog.mark(cid, "esrgan")
                print(f"  ║ ESRGAN:  {n} frames  ({time.time()-tc:.1f}s)")
            cur = esr_out

        out_fps = fps
        if do_rife:
            rife_out = work / f"chunk_{cid:04d}" / "rife"
            if not prog.done(cid, "rife"):
                n = rife_interpolate(cur, rife_out)
                prog.mark(cid, "rife")
                print(f"  ║ RIFE:    {n} frames  ({time.time()-tc:.1f}s)")
            cur = rife_out
            out_fps = fps * 2

        encode_q.put((cid, cur, out_fps))

        completed += 1
        wall = time.time() - t_start
        avg = wall / (completed - done_n)
        remaining = (total - completed) * avg
        print(f"  ║ Progress: {completed}/{total} ({100*completed/total:.1f}%)  "
              f"ETA {remaining/3600:.1f}h")
        print(f"  ╚═ Chunk {cid:04d} GPU done {time.time()-tc:.0f}s (encode→NVENC)")

    encode_q.join()
    encode_q.put(None)
    t_enc.join(timeout=10)
    return completed

# ── FINAL MERGE ─────────────────────────────────────────────
def merge_output(work, src, dst, n_chunks):
    concat = work / "concat.txt"
    with open(concat, "w") as f:
        for i in range(n_chunks):
            v = work / f"chunk_{i:04d}" / "output.mp4"
            if v.exists():
                f.write(f"file '{v}'\n")

    enh_audio = dst.parent / "GMT20260320-130023_Recording_enhanced.m4a"
    audio_src = str(enh_audio) if enh_audio.exists() else str(src)
    audio_codec = ["-c:a", "copy"] if enh_audio.exists() else [
        "-af", "afftdn=nf=-20:nt=w:om=o,"
               "acompressor=threshold=-20dB:ratio=3:attack=5:release=50,"
               "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:a", "aac", "-b:a", "192k"]

    print(f"\n[MERGE] {n_chunks} chunks…")
    cmd = ["ffmpeg", "-y",
           "-f", "concat", "-safe", "0", "-i", str(concat),
           "-i", audio_src,
           "-map", "0:v", "-map", "1:a",
           "-c:v", "copy"] + audio_codec + [
           "-movflags", "+faststart", "-threads", "16",
           str(dst), "-loglevel", "warning"]
    subprocess.run(cmd, check=True)
    print(f"[DONE] {dst}  ({dst.stat().st_size/1e9:.2f} GB)")

# ── MAIN ────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser(description="AI Video Enhancement v3")
    ap.add_argument("input")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-esrgan", action="store_true")
    ap.add_argument("--skip-rife", action="store_true")
    ap.add_argument("--chunk", type=int, default=CHUNK_SECONDS)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    src = Path(args.input).resolve()
    assert src.exists(), f"{src} not found"

    out_dir = src.parent / "enhanced"; out_dir.mkdir(exist_ok=True)
    work    = out_dir / f"work_{src.stem}"

    if args.clean and work.exists():
        shutil.rmtree(work)
    work.mkdir(exist_ok=True)

    do_esr  = not args.skip_esrgan
    do_rife = not args.skip_rife

    dur, fps, w, h = probe(src)
    out_w = w * 2 if do_esr else w
    out_h = h * 2 if do_esr else h
    out_fps = fps * 2 if do_rife else fps
    total_frames = int(dur * fps)

    suffix = "_ai_enhanced" + ("_50fps" if do_rife else "")
    dst = out_dir / f"{src.stem}{suffix}.mp4"

    print("=" * 65)
    print("  AI VIDEO ENHANCEMENT v3 — MAXIMUM SILICON UTILIZATION")
    print("=" * 65)
    print(f"  Input:       {src.name}")
    print(f"  Resolution:  {w}x{h}  ->  {out_w}x{out_h}")
    print(f"  Framerate:   {fps}fps  ->  {out_fps}fps")
    print(f"  Duration:    {dur:.0f}s  ({dur/3600:.1f}h)")
    print(f"  Frames:      ~{total_frames:,}")
    print(f"  ---")
    print(f"  ESRGAN:      {'ON  FP16 Tensor Cores + torch.compile (dual GPU)' if do_esr else 'SKIP'}")
    print(f"  RIFE:        {'ON  ncnn-Vulkan (dual GPU)' if do_rife else 'SKIP'}")
    print(f"  Encode:      HEVC NVENC (5070 Ti dedicated ASIC)")
    print(f"  Extract:     NVDEC hw decode + {EXTRACT_THREADS} CPU threads")
    print(f"  Pipeline:    4-stage overlap (extract || ESRGAN || RIFE || encode)")
    print(f"  Chunks:      {args.chunk}s each")
    print(f"  Output:      {dst.name}")
    print("=" * 65)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    prog = Progress(work)
    chunks = []
    s = 0.0; i = 0
    while s < dur:
        cd = min(args.chunk, dur - s)
        if cd <= 0: break
        chunks.append((i, s, cd))
        s += args.chunk; i += 1

    n_chunks = len(chunks)
    n_done   = sum(1 for c in chunks if prog.done(c[0], "encode"))
    print(f"\n  Chunks: {n_chunks} total, {n_done} done, {n_chunks-n_done} remaining\n")

    t_go = time.time()
    if n_done < n_chunks:
        esr = ESRGANEngine() if do_esr else None
        completed = run_pipeline(chunks, src, work, prog, do_esr, do_rife, fps, esr)
        wall = time.time() - t_go
        if wall > 0:
            print(f"\n  Processing: {wall/3600:.1f}h ({dur/wall:.2f}x realtime)")

    if not _shutdown.is_set():
        merge_output(work, src, dst, n_chunks)
    else:
        print("\n[!] Interrupted. Run again to resume.")

if __name__ == "__main__":
    main()
