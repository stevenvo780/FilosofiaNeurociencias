"""
ESRGAN engine — dual-GPU batched inference + CPU Multi-Thread inference.

Key optimizations:
  - Double-buffered pinned memory: CPU preps batch N+1 while GPU runs batch N
  - CPU worker offloads chunks explicitly processing via `torch.set_num_threads`
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

            # Warmup
            dummy = torch.randn(bs, 3, 315, 560, device=dev, dtype=torch.float16)
            with torch.inference_mode():
                _ = net(dummy)
                _ = net(dummy)
            torch.cuda.synchronize(gid)
            del dummy
            torch.cuda.empty_cache()
            vram = torch.cuda.memory_allocated(gid) / 1e9
            print(f"    warm  VRAM={vram:.2f}GB")

        # Configure CPU inference
        self.cpu_enabled = C.CPU_SHARE > 0.0
        if self.cpu_enabled:
            print(f"  [ESRGAN] CPU Worker (AMD Threads=16)  batch=1")
            torch.set_num_threads(16)
            m = spandrel.ModelLoader().load_from_file(C.ESRGAN_MODEL)
            self.cpu_model = m.model.to("cpu").eval()
            self.scale = m.scale
            
            dummy = torch.randn(1, 3, 315, 560, device="cpu", dtype=torch.float32)
            with torch.inference_mode():
                _ = self.cpu_model(dummy)
            del dummy

    def _gpu_worker(self, gid: int, work_frames: list[np.ndarray],
                    base_idx: int, store: list | None,
                    on_frame: Callable | None,
                    counter: list, lock: threading.Lock,
                    total: int, t0: float):
        torch = self.torch
        net = self.models[gid]
        dev = f"cuda:{gid}"
        bs = self.batches[gid]
        stream = self.streams[gid]
        n = len(work_frames)
        if n == 0:
            return

        sample_h, sample_w = work_frames[0].shape[:2]

        pinned = [
            torch.empty(bs, sample_h, sample_w, 3,
                        dtype=torch.uint8, pin_memory=True),
            torch.empty(bs, sample_h, sample_w, 3,
                        dtype=torch.uint8, pin_memory=True),
        ]

        first_n = min(bs, n)
        for i in range(first_n):
            pinned[0][i].copy_(torch.from_numpy(work_frames[i]))

        buf_idx = 0
        pos = 0
        prev_out_cpu = None
        prev_pos = 0
        prev_bs = 0

        while pos < n:
            if C.shutdown.is_set():
                return

            cur_buf = buf_idx
            cur_bs = min(bs, n - pos)
            next_pos = pos + cur_bs
            next_buf = 1 - buf_idx
            has_next = next_pos < n

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

            if prev_out_cpu is not None:
                out_np = prev_out_cpu.numpy()
                for i in range(prev_bs):
                    gidx = base_idx + prev_pos + i
                    if store is not None:
                        store[gidx] = out_np[i].copy()
                    if on_frame is not None:
                        on_frame(gidx, out_np[i])
                del prev_out_cpu
                prev_out_cpu = None

                with lock:
                    counter[0] += prev_bs
                    c = counter[0]
                    if c % 50 < bs or c >= total:
                        elapsed = time.time() - t0
                        fps_now = c / elapsed if elapsed > 0 else 0
                        eta = (total - c) / fps_now if fps_now > 0 else 0
                        print(f"    ESRGAN {c}/{total}  "
                              f"{fps_now:.1f}fps  ETA {eta:.0f}s",
                              flush=True)

            if has_next:
                next_n = min(bs, n - next_pos)
                for i in range(next_n):
                    pinned[next_buf][i].copy_(
                        torch.from_numpy(work_frames[next_pos + i]))

            torch.cuda.synchronize(gid)

            prev_out_cpu = out_cpu
            prev_pos = pos
            prev_bs = cur_bs

            del t_gpu, t_small, out, out_u8

            buf_idx = next_buf
            pos = next_pos

        if prev_out_cpu is not None:
            out_np = prev_out_cpu.numpy()
            for i in range(prev_bs):
                gidx = base_idx + prev_pos + i
                if store is not None:
                    store[gidx] = out_np[i].copy()
                if on_frame is not None:
                    on_frame(gidx, out_np[i])
            del prev_out_cpu

            with lock:
                counter[0] += prev_bs
                c = counter[0]
                elapsed = time.time() - t0
                fps_now = c / elapsed if elapsed > 0 else 0
                eta = (total - c) / fps_now if fps_now > 0 else 0
                print(f"    ESRGAN {c}/{total}  "
                      f"{fps_now:.1f}fps  ETA {eta:.0f}s",
                      flush=True)

        del pinned


    def _cpu_worker(self, work_frames: list[np.ndarray],
                    base_idx: int, store: list | None,
                    on_frame: Callable | None,
                    counter: list, lock: threading.Lock,
                    total: int, t0: float):
        """Process a slice of frames on CPU entirely, overlapping with GPU computation."""
        torch = self.torch
        net = self.cpu_model
        n = len(work_frames)
        if n == 0:
            return

        for pos in range(n):
            if C.shutdown.is_set():
                return
            
            frame = work_frames[pos]
            t_cpu = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to("cpu", dtype=torch.float32) / 255.0

            t_small = torch.nn.functional.interpolate(
                t_cpu, scale_factor=0.5,
                mode="bilinear", align_corners=False)

            with torch.inference_mode():
                out = net(t_small)

            out_u8 = (out.clamp(0, 1) * 255).byte()
            out_cpu_frame = out_u8[0].permute(1, 2, 0).numpy()

            gidx = base_idx + pos
            if store is not None:
                store[gidx] = out_cpu_frame.copy()
            if on_frame is not None:
                on_frame(gidx, out_cpu_frame)

            with lock:
                counter[0] += 1
                c = counter[0]
                if c % 50 == 0 or c >= total:
                    elapsed = time.time() - t0
                    fps_now = c / elapsed if elapsed > 0 else 0
                    eta = (total - c) / fps_now if fps_now > 0 else 0
                    print(f"    ESRGAN {c}/{total}  "
                          f"{fps_now:.1f}fps  ETA {eta:.0f}s",
                          flush=True)
            
            del t_cpu, t_small, out, out_u8

    def _run_parallel(self, frames: list[np.ndarray],
                      store: list | None,
                      on_frame: Callable | None) -> int:
        """Dispatch frames to dual-GPU and CPU multi-thread workers."""
        total = len(frames)
        if total == 0:
            return 0

        # Adjust indices according to user HW architecture and settings Configs
        split0 = int(total * C.GPU0_SHARE)
        split1 = split0 + int(total * C.GPU1_SHARE) if self.ngpu > 1 else split0

        gpu0_frames = frames[:split0]
        gpu1_frames = frames[split0:split1] if self.ngpu > 1 else []
        cpu_frames = frames[split1:] if self.cpu_enabled else (frames[split0:] if self.ngpu == 1 else [])

        counter = [0]
        lock = threading.Lock()
        t0 = time.time()

        threads = []
        if gpu0_frames:
            t = threading.Thread(
                target=self._gpu_worker,
                args=(0, gpu0_frames, 0, store, on_frame, counter, lock, total, t0))
            t.start()
            threads.append(t)
            
        if gpu1_frames:
            t = threading.Thread(
                target=self._gpu_worker,
                args=(1, gpu1_frames, split0, store, on_frame, counter, lock, total, t0))
            t.start()
            threads.append(t)
            
        if self.cpu_enabled and cpu_frames:
            t = threading.Thread(
                target=self._cpu_worker,
                args=(cpu_frames, split1 if self.ngpu > 1 else split0, store, on_frame, counter, lock, total, t0))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        elapsed = time.time() - t0
        c = counter[0]
        print(f"    ESRGAN done: {c} frames  "
              f"{elapsed:.0f}s  {c/elapsed:.1f}fps", flush=True)
        return c

    def process_frames(self, frames: list[np.ndarray],
                       out_dir: Path | None = None) -> list[np.ndarray]:
        total = len(frames)
        results = [None] * total
        self._run_parallel(frames, store=results, on_frame=None)

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

    def process_streaming(self, frames: list[np.ndarray],
                          on_frame: Callable[[int, np.ndarray], None]) -> int:
        return self._run_parallel(frames, store=None, on_frame=on_frame)
