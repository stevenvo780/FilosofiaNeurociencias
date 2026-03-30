"""
ESRGAN engine — dual-GPU batched inference with Tensor Cores FP16.

Architecture (zero-PNG for extract→ESRGAN):
  ffmpeg pipe → numpy arrays in RAM → GPU batch inference → numpy → encode

Key optimizations:
  - Double-buffered pinned memory: CPU preps batch N+1 while GPU runs batch N
  - CPU prep OUTSIDE cuda stream so GPU never waits for memcpy
  - .copy() on result slices to free batch tensor memory immediately
  - Explicit del of GPU intermediates after each batch
"""
import time, threading, gc
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

            # Warmup — trigger cuDNN autotuning & JIT caches
            dummy = torch.randn(bs, 3, 315, 560, device=dev, dtype=torch.float16)
            with torch.inference_mode():
                _ = net(dummy)
                _ = net(dummy)
            torch.cuda.synchronize(gid)
            del dummy
            torch.cuda.empty_cache()
            vram = torch.cuda.memory_allocated(gid) / 1e9
            print(f"    warm  VRAM={vram:.2f}GB")

    def process_frames(self, frames: list[np.ndarray],
                       out_dir: Path | None = None) -> list[np.ndarray]:
        """Process numpy RGB frames with dual-GPU batched ESRGAN.

        Args:
            frames: list of HWC uint8 RGB numpy arrays (writable)
            out_dir: if given, write output PNGs here (for RIFE)

        Returns:
            list of enhanced HWC uint8 RGB numpy arrays (independent copies)
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
            sample_h, sample_w = frames[0].shape[:2]

            # Double-buffered pinned memory for H2D overlap
            pinned = [
                torch.empty(bs, sample_h, sample_w, 3,
                            dtype=torch.uint8, pin_memory=True),
                torch.empty(bs, sample_h, sample_w, 3,
                            dtype=torch.uint8, pin_memory=True),
            ]
            buf_idx = 0

            # Pre-fill first batch into pinned[0] (CPU, no stream)
            first_end = min(start_idx + bs, end_idx)
            first_frames = frames[start_idx:first_end]
            for i, f in enumerate(first_frames):
                pinned[0][i].copy_(torch.from_numpy(f))
            first_bs = len(first_frames)

            bi = start_idx
            while bi < end_idx:
                if C.shutdown.is_set():
                    return

                cur_buf = buf_idx
                cur_bs = min(bs, end_idx - bi)
                next_bi = bi + bs
                next_buf = 1 - buf_idx
                has_next = next_bi < end_idx

                # Launch GPU work for current batch from pinned[cur_buf]
                with torch.cuda.stream(stream):
                    t_gpu = (pinned[cur_buf][:cur_bs]
                             .permute(0, 3, 1, 2)
                             .to(dev, dtype=torch.float16,
                                 non_blocking=True) / 255.0)

                    t_small = torch.nn.functional.interpolate(
                        t_gpu, scale_factor=0.5,
                        mode="bilinear", align_corners=False)

                    with torch.inference_mode():
                        out = net(t_small)

                    out_u8 = (out.clamp(0, 1) * 255).byte()
                    out_cpu = out_u8.permute(0, 2, 3, 1).cpu()

                # While GPU runs (or just finished), prep NEXT batch
                # into pinned[next_buf] on CPU — this is the double-buffer
                if has_next:
                    next_end = min(next_bi + bs, end_idx)
                    next_frames = frames[next_bi:next_end]
                    for i, f in enumerate(next_frames):
                        pinned[next_buf][i].copy_(torch.from_numpy(f))

                # Wait for GPU to finish current batch
                torch.cuda.synchronize(gid)

                # Store results as independent copies (free batch tensor)
                out_np = out_cpu.numpy()
                for i in range(cur_bs):
                    results[bi + i] = out_np[i].copy()

                # Explicitly free GPU intermediates
                del t_gpu, t_small, out, out_u8, out_cpu, out_np

                buf_idx = next_buf
                with lock:
                    counter[0] += cur_bs
                    c = counter[0]
                    if c % 50 < bs or c >= total:
                        elapsed = time.time() - t0
                        fps_now = c / elapsed
                        eta = (total - c) / fps_now if fps_now > 0 else 0
                        print(f"    ESRGAN {c}/{total}  "
                              f"{fps_now:.1f}fps  ETA {eta:.0f}s",
                              flush=True)
                bi = next_bi if has_next else end_idx

            # Free pinned buffers
            del pinned

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

        # Free input frames now — caller should not use them after this
        # (pipeline.py already does `del frames` after calling us)

        # Write PNGs if output dir given (for RIFE)
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            pool = ThreadPoolExecutor(max_workers=C.WRITE_WORKERS)
            futs = []
            for i, img in enumerate(results):
                if img is not None:
                    dst = out_dir / f"{i+1:08d}.png"
                    futs.append(pool.submit(
                        cv2.imwrite, str(dst),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR)))
            for f in futs:
                f.result()
            pool.shutdown(wait=False)

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
