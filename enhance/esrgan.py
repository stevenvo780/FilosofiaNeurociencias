"""
ESRGAN engine — dual-GPU batched inference with Tensor Cores FP16.

Two modes:
  process_frames()    — collect all results (needed for RIFE path)
  process_streaming() — call on_frame(idx, np_array) per frame, zero accumulation

Key optimizations:
  - Double-buffered pinned memory: CPU preps batch N+1 while GPU runs batch N
  - CPU prep OUTSIDE cuda stream so GPU never waits for memcpy
  - Streaming mode: each batch piped to NVENC immediately, RAM = input + 1 batch
  - No closure capture of frames list — passed as arg to prevent reference leaks
"""
import time, threading
from pathlib import Path
from typing import Callable
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

    # ─────────────────────────────────────────────────────────
    def _gpu_worker(self, gid: int, work_frames: list[np.ndarray],
                    base_idx: int, store: list | None,
                    on_frame: Callable | None,
                    counter: list, lock: threading.Lock,
                    total: int, t0: float):
        """Process a slice of frames on one GPU.

        Args:
            gid:         GPU index
            work_frames: the numpy frame slice this GPU owns (NOT a closure)
            base_idx:    global index of first frame in work_frames
            store:       if not None, collect results[global_idx] = frame
            on_frame:    if not None, call on_frame(global_idx, np_array) per frame
            counter:     shared [count] for progress
            lock:        shared lock for counter
            total:       total frames across all GPUs
            t0:          start time
        """
        torch = self.torch
        net = self.models[gid]
        dev = f"cuda:{gid}"
        bs = self.batches[gid]
        stream = self.streams[gid]
        n = len(work_frames)
        if n == 0:
            return

        sample_h, sample_w = work_frames[0].shape[:2]

        # Double-buffered pinned memory for H2D overlap
        pinned = [
            torch.empty(bs, sample_h, sample_w, 3,
                        dtype=torch.uint8, pin_memory=True),
            torch.empty(bs, sample_h, sample_w, 3,
                        dtype=torch.uint8, pin_memory=True),
        ]

        # Pre-fill first batch into pinned[0]
        first_n = min(bs, n)
        for i in range(first_n):
            pinned[0][i].copy_(torch.from_numpy(work_frames[i]))

        buf_idx = 0
        pos = 0

        while pos < n:
            if C.shutdown.is_set():
                return

            cur_buf = buf_idx
            cur_bs = min(bs, n - pos)
            next_pos = pos + cur_bs
            next_buf = 1 - buf_idx
            has_next = next_pos < n

            # GPU inference on current batch
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

            # CPU: prep next batch while GPU might still finish
            if has_next:
                next_n = min(bs, n - next_pos)
                for i in range(next_n):
                    pinned[next_buf][i].copy_(
                        torch.from_numpy(work_frames[next_pos + i]))

            torch.cuda.synchronize(gid)

            # Deliver results — store and/or stream
            out_np = out_cpu.numpy()
            for i in range(cur_bs):
                gidx = base_idx + pos + i
                frame_np = out_np[i].copy()
                if store is not None:
                    store[gidx] = frame_np
                if on_frame is not None:
                    on_frame(gidx, frame_np)

            del t_gpu, t_small, out, out_u8, out_cpu, out_np

            buf_idx = next_buf
            pos = next_pos

            with lock:
                counter[0] += cur_bs
                c = counter[0]
                if c % 50 < bs or c >= total:
                    elapsed = time.time() - t0
                    fps_now = c / elapsed if elapsed > 0 else 0
                    eta = (total - c) / fps_now if fps_now > 0 else 0
                    print(f"    ESRGAN {c}/{total}  "
                          f"{fps_now:.1f}fps  ETA {eta:.0f}s",
                          flush=True)

        del pinned

    # ─────────────────────────────────────────────────────────
    def _run_dual(self, frames: list[np.ndarray],
                  store: list | None,
                  on_frame: Callable | None) -> int:
        """Dispatch frames to dual-GPU workers. Returns processed count."""
        total = len(frames)
        if total == 0:
            return 0

        split = int(total * C.GPU0_SHARE) if self.ngpu > 1 else total
        counter = [0]
        lock = threading.Lock()
        t0 = time.time()

        # Pass frame slices as explicit args — no closure capture
        gpu0_frames = frames[:split]
        gpu1_frames = frames[split:] if self.ngpu > 1 else []

        threads = []
        if gpu0_frames:
            t = threading.Thread(
                target=self._gpu_worker,
                args=(0, gpu0_frames, 0, store, on_frame,
                      counter, lock, total, t0))
            t.start()
            threads.append(t)
        if gpu1_frames:
            t = threading.Thread(
                target=self._gpu_worker,
                args=(1, gpu1_frames, split, store, on_frame,
                      counter, lock, total, t0))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        elapsed = time.time() - t0
        c = counter[0]
        print(f"    ESRGAN done: {c} frames  "
              f"{elapsed:.0f}s  {c/elapsed:.1f}fps", flush=True)
        return c

    # ─────────────────────────────────────────────────────────
    def process_frames(self, frames: list[np.ndarray],
                       out_dir: Path | None = None) -> list[np.ndarray]:
        """Collect all results in RAM. Use for RIFE path (needs all frames).

        Returns:
            list of enhanced HWC uint8 RGB numpy arrays
        """
        total = len(frames)
        results = [None] * total
        self._run_dual(frames, store=results, on_frame=None)

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

        return results

    # ─────────────────────────────────────────────────────────
    def process_streaming(self, frames: list[np.ndarray],
                          on_frame: Callable[[int, np.ndarray], None]) -> int:
        """Stream results via callback — zero accumulation in RAM.

        on_frame(global_idx, rgb_uint8_hwc) is called per frame as produced.
        Frames arrive out of order (GPU0 and GPU1 run in parallel).
        Caller must handle ordering if sequential output is needed.

        Returns:
            number of frames processed
        """
        return self._run_dual(frames, store=None, on_frame=on_frame)
