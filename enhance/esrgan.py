"""
ESRGAN engine — dual-GPU batched inference + CPU Multi-Thread inference.

Key optimizations:
  - Dynamic Task Allocation: GPUs and CPU independently pull frame indexes avoiding static blocks.
  - Double-buffered pinned memory: Preps batch N+1 while GPU runs batch N dynamically.
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

        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "benchmark_limit"):
            torch.backends.cudnn.benchmark_limit = 0

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
            net = self._maybe_compile(net, gid, bs, dev)
            self.models.append(net)
            self.streams.append(torch.cuda.Stream(device=dev))

            dummy = torch.randn(bs, 3, 315, 560, device=dev, dtype=torch.float16)
            with torch.inference_mode():
                _ = net(dummy)
                _ = net(dummy)
            torch.cuda.synchronize(gid)
            del dummy
            torch.cuda.empty_cache()
            vram = torch.cuda.memory_allocated(gid) / 1e9
            print(f"    warm  VRAM={vram:.2f}GB")

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

    def _maybe_compile(self, net, gid: int, bs: int, dev: str):
        """Compile the model when available, but fall back cleanly."""
        torch = self.torch
        if not C.ENABLE_TORCH_COMPILE:
            return net

        try:
            compile_kwargs = {
                "fullgraph": C.TORCH_COMPILE_FULLGRAPH,
                "dynamic": False,
            }
            if C.TORCH_COMPILE_DISABLE_CUDAGRAPHS:
                compile_kwargs["options"] = {"triton.cudagraphs": False}
            else:
                compile_kwargs["mode"] = C.TORCH_COMPILE_MODE

            compiled = torch.compile(net, **compile_kwargs)
            dummy = torch.randn(bs, 3, 315, 560, device=dev, dtype=torch.float16)
            with torch.inference_mode():
                _ = compiled(dummy)
                _ = compiled(dummy)
            torch.cuda.synchronize(gid)
            del dummy
            msg = "cudagraphs=off" if C.TORCH_COMPILE_DISABLE_CUDAGRAPHS else f"mode={C.TORCH_COMPILE_MODE}"
            print(f"    compile=on  {msg}", flush=True)
            return compiled
        except Exception as exc:
            print(f"    compile=off  reason={exc}", flush=True)
            return net

    def _gpu_worker(self, gid: int, frames: list[np.ndarray], get_batch: Callable,
                    store: list | None, on_frame: Callable | None,
                    counter: list, lock: threading.Lock, total: int, t0: float):
        torch = self.torch
        net = self.models[gid]
        dev = f"cuda:{gid}"
        bs = self.batches[gid]
        stream = self.streams[gid]

        start, end = get_batch(bs)
        if start is None:
            return

        sample_h, sample_w = frames[start].shape[:2]

        pinned = [
            torch.empty(bs, sample_h, sample_w, 3,
                        dtype=torch.uint8, pin_memory=True),
            torch.empty(bs, sample_h, sample_w, 3,
                        dtype=torch.uint8, pin_memory=True),
        ]

        cur_bs = end - start
        # Bulk copy: stack all frames at once, then single copy_ into pinned buffer
        pinned[0][:cur_bs].copy_(torch.from_numpy(
            np.stack([frames[start + i] for i in range(cur_bs)])))
            
        buf_idx = 0
        prev_out_cpu = None
        prev_start = 0
        prev_bs = 0

        while cur_bs > 0:
            if C.shutdown.is_set():
                return
            
            cur_buf = buf_idx
            next_start, next_end = get_batch(bs)
            next_bs = next_end - next_start if next_start is not None else 0
            next_buf = 1 - buf_idx

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
                # .contiguous() on GPU is ~5x faster than .copy() on CPU
                out_cpu = out_u8.permute(0, 2, 3, 1).contiguous().cpu()

            if prev_out_cpu is not None:
                out_np = prev_out_cpu.numpy()
                for i in range(prev_bs):
                    gidx = prev_start + i
                    if store is not None:
                        store[gidx] = out_np[i]  # already contiguous from GPU
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

            # Bulk copy next batch while GPU works (overlapped via CUDA stream)
            if next_bs > 0:
                pinned[next_buf][:next_bs].copy_(torch.from_numpy(
                    np.stack([frames[next_start + i] for i in range(next_bs)])))

            torch.cuda.synchronize(gid)

            prev_out_cpu = out_cpu
            prev_start = start
            prev_bs = cur_bs

            del t_gpu, t_small, out, out_u8

            buf_idx = next_buf
            start = next_start
            cur_bs = next_bs

        if prev_out_cpu is not None:
            out_np = prev_out_cpu.numpy()
            for i in range(prev_bs):
                gidx = prev_start + i
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


    def _cpu_worker(self, frames: list[np.ndarray], get_batch: Callable,
                    store: list | None, on_frame: Callable | None,
                    counter: list, lock: threading.Lock, total: int, t0: float):
        torch = self.torch
        net = self.cpu_model

        while True:
            start, _ = get_batch(1)
            if start is None:
                break
            if C.shutdown.is_set():
                return
            
            frame = frames[start]
            t_cpu = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to("cpu", dtype=torch.float32) / 255.0

            t_small = torch.nn.functional.interpolate(
                t_cpu, scale_factor=0.5,
                mode="bilinear", align_corners=False)

            with torch.inference_mode():
                out = net(t_small)

            out_u8 = (out.clamp(0, 1) * 255).byte()
            out_cpu_frame = out_u8[0].permute(1, 2, 0).numpy()

            gidx = start
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
        """Dispatch dynamically fetching threads."""
        total = len(frames)
        if total == 0:
            return 0

        pos = [0]
        lock = threading.Lock()
        
        def get_batch(batch_size):
            with lock:
                if pos[0] >= total:
                    return None, 0
                start = pos[0]
                end = min(total, start + batch_size)
                pos[0] = end
                return start, end

        counter = [0]
        t0 = time.time()

        threads = []
        errors: list[BaseException] = []

        def launch(target, *args):
            def runner():
                try:
                    target(*args)
                except BaseException as exc:
                    errors.append(exc)
                    C.shutdown.set()
            thread = threading.Thread(target=runner)
            thread.start()
            threads.append(thread)

        # Deploy fetching worker for GPU0 (Fastest: Pulls 8 frames automatically)
        launch(self._gpu_worker, 0, frames, get_batch, store, on_frame, counter, lock, total, t0)
        
        # Deploy fetching worker for GPU1 (Slightly Slower: Pulls 4 frames automatically)
        if self.ngpu > 1:
            launch(self._gpu_worker, 1, frames, get_batch, store, on_frame, counter, lock, total, t0)
            
        # Deploy fetching worker for CPU (R9 Processor: Pulls 1 frame slowly behind them)
        if self.cpu_enabled:
            launch(self._cpu_worker, frames, get_batch, store, on_frame, counter, lock, total, t0)

        for t in threads:
            t.join()

        if errors:
            raise RuntimeError(f"ESRGAN worker failed: {errors[0]}") from errors[0]

        elapsed = time.time() - t0
        c = counter[0]
        print(f"    ESRGAN done: {c} frames  "
              f"{elapsed:.0f}s  {c/elapsed:.1f}fps", flush=True)
        return c

    def process_frames(self, frames: list[np.ndarray],
                       out_dir: Path | None = None,
                       store: list | None = None) -> list[np.ndarray]:
        total = len(frames)
        if store is None:
            store = [None] * total
        self._run_parallel(frames, store=store, on_frame=None)

        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            pool = ThreadPoolExecutor(max_workers=C.WRITE_WORKERS)
            futs = []
            for i, img in enumerate(store):
                if img is not None:
                    dst = out_dir / f"{i+1:08d}.png"
                    futs.append(pool.submit(
                        cv2.imwrite, str(dst),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR)))
            for f in futs:
                f.result()
            pool.shutdown(wait=False)

        return store

    def process_streaming(self, frames: list[np.ndarray],
                          on_frame: Callable[[int, np.ndarray], None]) -> int:
        return self._run_parallel(frames, store=None, on_frame=on_frame)
