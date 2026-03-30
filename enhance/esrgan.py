"""
ESRGAN engine — dual-GPU batched inference with Tensor Cores FP16.

Key fixes vs v5:
  - Loads model separately per GPU (avoids device mismatch in torch.compile)
  - Parallel PNG read via ThreadPool BEFORE GPU needs data (prefetch)
  - Proper pinned→device copy with .copy_() not .to()
  - GPU downscale (F.interpolate) instead of CPU cv2.resize
  - Double-buffered batches: while GPU runs batch N, CPU prepares batch N+1
"""
import time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import config as C


class ESRGANEngine:
    def __init__(self):
        import torch
        import spandrel
        self.torch = torch

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.ngpu = min(torch.cuda.device_count(), 2)
        self.models = []
        self.batches = ([C.GPU0_BATCH, C.GPU1_BATCH]
                        if self.ngpu > 1 else [C.GPU0_BATCH])
        self.streams = []  # one stream per GPU (compute)

        for gid in range(self.ngpu):
            dev = f"cuda:{gid}"
            name = torch.cuda.get_device_name(gid)
            bs = self.batches[gid]
            print(f"  [ESRGAN] GPU{gid} {name}  batch={bs}")

            # Load fresh copy for each GPU to avoid device mismatches
            m = spandrel.ModelLoader().load_from_file(C.ESRGAN_MODEL)
            self.scale = m.scale
            net = m.model.half().to(dev).eval()
            self.models.append(net)
            self.streams.append(torch.cuda.Stream(device=dev))

            # Warmup — trigger cuDNN autotuning & JIT caches
            dummy = torch.randn(bs, 3, 315, 560, device=dev, dtype=torch.float16)
            with torch.inference_mode():
                _ = net(dummy)
                _ = net(dummy)
            torch.cuda.synchronize(gid)
            vram = torch.cuda.memory_allocated(gid) / 1e9
            print(f"    warm  VRAM={vram:.2f}GB")

    # ────────────────────────────────────────────────────────
    def process_directory(self, in_dir: Path, out_dir: Path) -> int:
        """Process all PNGs with dual-GPU batched inference."""
        torch = self.torch
        out_dir.mkdir(parents=True, exist_ok=True)
        frames = sorted(in_dir.glob("*.png"))
        if not frames:
            return 0
        done = len(list(out_dir.glob("*.png")))
        if done >= len(frames):
            return done

        total = len(frames)
        counter = [0]
        lock = threading.Lock()
        t0 = time.time()

        # ── Split work between GPUs ─────────────────────────
        split = int(total * C.GPU0_SHARE) if self.ngpu > 1 else total
        gpu_work = [frames[:split], frames[split:]] if self.ngpu > 1 else [frames]

        # ── I/O thread pools ────────────────────────────────
        read_pool  = ThreadPoolExecutor(max_workers=C.READ_WORKERS)
        write_pool = ThreadPoolExecutor(max_workers=C.WRITE_WORKERS)
        write_futs = []

        def _read_png(fpath):
            """Read single PNG → RGB uint8 numpy (CPU threaded)."""
            img = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        def _write_png(rgb_np, dst):
            """Write RGB numpy → PNG (CPU threaded)."""
            cv2.imwrite(str(dst), cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

        def _gpu_worker(gid, work_frames):
            """Process one GPU's share with batched inference + prefetch."""
            net = self.models[gid]
            dev = f"cuda:{gid}"
            bs = self.batches[gid]
            stream = self.streams[gid]

            # Group into batches
            batches = [work_frames[i:i+bs]
                       for i in range(0, len(work_frames), bs)]

            # Prefetch first batch of images
            prefetch = None

            for ib, batch in enumerate(batches):
                if C.shutdown.is_set():
                    return

                real_bs = len(batch)

                # Skip already-processed frames
                todo = [(f, out_dir / f.name) for f in batch
                        if not (out_dir / f.name).exists()]
                if not todo:
                    with lock:
                        counter[0] += real_bs
                    continue

                # ── Read: use prefetched data or read now ───
                if prefetch is not None:
                    imgs = prefetch
                else:
                    imgs = list(read_pool.map(
                        _read_png, [f for f, _ in todo]))

                # ── Prefetch NEXT batch while GPU works ─────
                if ib + 1 < len(batches):
                    next_batch = batches[ib + 1]
                    next_todo = [f for f in next_batch
                                 if not (out_dir / f.name).exists()]
                    if next_todo:
                        prefetch_futs = [read_pool.submit(_read_png, f)
                                         for f in next_todo]
                    else:
                        prefetch_futs = None
                else:
                    prefetch_futs = None

                # ── H2D + downscale + inference on GPU ──────
                with torch.cuda.stream(stream):
                    # Stack numpy → tensor on CPU, then move to GPU
                    # HWC uint8 → NCHW float16
                    np_stack = np.stack(imgs, axis=0)  # (N, H, W, 3)
                    t_cpu = torch.from_numpy(np_stack).permute(0, 3, 1, 2)
                    t_gpu = t_cpu.to(dev, dtype=torch.float16,
                                     non_blocking=True) / 255.0

                    # Downscale 0.5x on GPU (uses CUDA cores, fast)
                    t_small = torch.nn.functional.interpolate(
                        t_gpu, scale_factor=0.5,
                        mode="bilinear", align_corners=False)

                    # ESRGAN forward — Tensor Cores FP16
                    with torch.inference_mode():
                        out = net(t_small)

                    # D2H: clamp + to uint8 on GPU, then to CPU
                    out_u8 = (out.float().clamp(0, 1) * 255).byte()
                    result_np = out_u8.permute(0, 2, 3, 1).cpu().numpy()

                # ── Async writes (CPU threads) ──────────────
                for i, (fpath, dst) in enumerate(todo):
                    if i < len(result_np):
                        fut = write_pool.submit(
                            _write_png, result_np[i].copy(), dst)
                        write_futs.append(fut)

                # ── Collect prefetch ────────────────────────
                if prefetch_futs is not None:
                    prefetch = [f.result() for f in prefetch_futs]
                else:
                    prefetch = None

                # ── Report progress ─────────────────────────
                with lock:
                    counter[0] += real_bs
                    c = counter[0]
                    if c % 50 < bs or c >= total:
                        elapsed = time.time() - t0
                        fps_now = c / elapsed
                        eta = (total - c) / fps_now if fps_now > 0 else 0
                        print(f"    ESRGAN {c}/{total}  "
                              f"{fps_now:.1f}fps  ETA {eta:.0f}s",
                              flush=True)

        # ── Launch GPU threads ──────────────────────────────
        threads = []
        for gid in range(self.ngpu):
            if gid < len(gpu_work) and gpu_work[gid]:
                t = threading.Thread(target=_gpu_worker,
                                     args=(gid, gpu_work[gid]),
                                     name=f"esrgan-gpu{gid}")
                t.start()
                threads.append(t)

        for t in threads:
            t.join()

        # Wait for all writes
        for f in write_futs:
            f.result()
        read_pool.shutdown(wait=False)
        write_pool.shutdown(wait=False)

        elapsed = time.time() - t0
        print(f"    ESRGAN done: {counter[0]} frames  "
              f"{elapsed:.0f}s  {counter[0]/elapsed:.1f}fps", flush=True)
        return counter[0]
