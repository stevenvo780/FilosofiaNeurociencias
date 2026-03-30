"""
ESRGAN engine — dual-GPU batched inference with Tensor Cores FP16.

Architecture (zero-PNG for extract→ESRGAN):
  ffmpeg pipe → numpy arrays in RAM → GPU batch inference → numpy → PNG write

The bottleneck was cv2.imread (PNG decode at 75fps serial).
Now frames arrive as raw numpy — no decode needed. GPU gets fed at full speed.
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
        self.streams = []

        for gid in range(self.ngpu):
            dev = f"cuda:{gid}"
            name = torch.cuda.get_device_name(gid)
            bs = self.batches[gid]
            print(f"  [ESRGAN] GPU{gid} {name}  batch={bs}")

            m = spandrel.ModelLoader().load_from_file(C.ESRGAN_MODEL)
            self.scale = m.scale
            net = m.model.half().to(dev).eval()
            self.models.append(net)
            self.streams.append(torch.cuda.Stream(device=dev))

            # Warmup
            dummy = torch.randn(bs, 3, 315, 560, device=dev, dtype=torch.float16)
            with torch.inference_mode():
                _ = net(dummy)
                _ = net(dummy)
            torch.cuda.synchronize(gid)
            vram = torch.cuda.memory_allocated(gid) / 1e9
            print(f"    warm  VRAM={vram:.2f}GB")

    def process_frames(self, frames: list[np.ndarray],
                       out_dir: Path | None = None) -> list[np.ndarray]:
        """Process numpy RGB frames with dual-GPU batched ESRGAN.

        Args:
            frames: list of HWC uint8 RGB numpy arrays
            out_dir: if given, write output PNGs here (for RIFE/NVENC)

        Returns:
            list of enhanced HWC uint8 RGB numpy arrays
        """
        torch = self.torch
        total = len(frames)
        if total == 0:
            return []

        results = [None] * total
        counter = [0]
        lock = threading.Lock()
        t0 = time.time()

        # Split work between GPUs
        split = int(total * C.GPU0_SHARE) if self.ngpu > 1 else total

        def _gpu_worker(gid, start_idx, end_idx):
            net = self.models[gid]
            dev = f"cuda:{gid}"
            bs = self.batches[gid]
            stream = self.streams[gid]

            # Pre-allocate pinned buffer for 8.2x faster H2D transfer
            sample_h, sample_w = frames[0].shape[:2]
            pinned_buf = torch.empty(
                bs, sample_h, sample_w, 3,
                dtype=torch.uint8, pin_memory=True)

            for bi in range(start_idx, end_idx, bs):
                if C.shutdown.is_set():
                    return
                batch_end = min(bi + bs, end_idx)
                batch_frames = frames[bi:batch_end]
                real_bs = len(batch_frames)

                with torch.cuda.stream(stream):
                    # Copy numpy → pinned buffer (CPU memcpy, fast)
                    for i, f in enumerate(batch_frames):
                        pinned_buf[i, :f.shape[0], :f.shape[1], :].copy_(
                            torch.from_numpy(f))

                    # Pinned → GPU (DMA, 8.2x faster than pageable)
                    t_gpu = (pinned_buf[:real_bs]
                             .permute(0, 3, 1, 2)
                             .to(dev, dtype=torch.float16,
                                 non_blocking=True) / 255.0)

                    # Downscale 0.5x on GPU (CUDA cores)
                    t_small = torch.nn.functional.interpolate(
                        t_gpu, scale_factor=0.5,
                        mode="bilinear", align_corners=False)

                    # ESRGAN forward (Tensor Cores FP16)
                    with torch.inference_mode():
                        out = net(t_small)

                    # D2H
                    out_np = (out.float().clamp(0, 1) * 255
                              ).byte().permute(0, 2, 3, 1).cpu().numpy()

                for i in range(real_bs):
                    results[bi + i] = out_np[i]

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

        # Launch GPU threads
        threads = []
        if split > 0:
            t = threading.Thread(target=_gpu_worker, args=(0, 0, split))
            t.start()
            threads.append(t)
        if self.ngpu > 1 and split < total:
            t = threading.Thread(target=_gpu_worker, args=(1, split, total))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        # Write PNGs if output dir given (for RIFE or NVENC)
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            write_pool = ThreadPoolExecutor(max_workers=C.WRITE_WORKERS)
            futs = []
            for i, img in enumerate(results):
                if img is not None:
                    dst = out_dir / f"{i+1:08d}.png"
                    futs.append(write_pool.submit(
                        cv2.imwrite, str(dst),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR)))
            for f in futs:
                f.result()
            write_pool.shutdown(wait=False)

        elapsed = time.time() - t0
        print(f"    ESRGAN done: {counter[0]} frames  "
              f"{elapsed:.0f}s  {counter[0]/elapsed:.1f}fps", flush=True)
        return results

    # Keep backward compat for directory-based processing
    def process_directory(self, in_dir: Path, out_dir: Path) -> int:
        """Process PNGs from directory (fallback)."""
        frames_paths = sorted(in_dir.glob("*.png"))
        if not frames_paths:
            return 0
        done = len(list(out_dir.glob("*.png")))
        if done >= len(frames_paths):
            return done

        pool = ThreadPoolExecutor(max_workers=C.READ_WORKERS)
        def _read(f):
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames = list(pool.map(_read, frames_paths))
        pool.shutdown(wait=False)

        self.process_frames(frames, out_dir)
        return len(frames)
