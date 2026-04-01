"""
ESRGAN engine — dual-GPU batched inference + CPU Multi-Thread inference.

Key optimizations:
  - Dynamic task allocation across workers.
  - Reusable CPU batch buffers to avoid per-batch np.stack allocations.
  - Pinned CPU staging + split CUDA streams for H2D and compute telemetry.
  - Async D2H with double-buffered pinned output buffers: overlaps GPU→CPU
    transfer of batch N with compute of batch N+1 (see ESRGAN_D2H_DOUBLE_BUFFER).
"""
import time, threading
from pathlib import Path
from typing import Callable
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import config as C


class ESRGANEngine:
    def __init__(self, visual_profile=None):
        import torch
        import spandrel
        self.torch = torch
        self.visual_profile = visual_profile
        self._downscale_factor = visual_profile.downscale_factor if visual_profile else 0.5
        self._hybrid_weight = visual_profile.hybrid_detail_weight if visual_profile else 0.0
        self._face_adaptive = visual_profile.face_adaptive if visual_profile else False
        self._face_roi = visual_profile.face_roi if visual_profile else (0.5, 0.0, 1.0, 0.5)

        torch.set_float32_matmul_precision(C.TORCH_MATMUL_PRECISION)
        torch.backends.cudnn.benchmark = C.CUDNN_BENCHMARK
        torch.backends.cuda.matmul.allow_tf32 = C.CUDA_MATMUL_ALLOW_TF32
        torch.backends.cudnn.allow_tf32 = C.CUDNN_ALLOW_TF32
        if hasattr(torch.backends.cudnn, "benchmark_limit"):
            torch.backends.cudnn.benchmark_limit = C.CUDNN_BENCHMARK_LIMIT

        visible_gpus = torch.cuda.device_count()
        configured = [gid for gid in C.ESRGAN_GPUS if gid < visible_gpus]
        self.gpu_ids = configured[:2] or list(range(min(visible_gpus, 2)))
        self.ngpu = len(self.gpu_ids)
        self.models = []
        self.gpu_total_mem_mib = {}
        self.batches = [
            C.GPU0_BATCH if gid == 0 else C.GPU1_BATCH
            for gid in self.gpu_ids
        ]

        for gid in self.gpu_ids:
            try:
                props = torch.cuda.get_device_properties(gid)
                self.gpu_total_mem_mib[gid] = int(props.total_memory / 1024**2)
            except Exception:
                self.gpu_total_mem_mib[gid] = 0

        for idx, gid in enumerate(self.gpu_ids):
            dev = f"cuda:{gid}"
            name = torch.cuda.get_device_name(gid)
            bs = self.batches[idx]
            print(f"  [ESRGAN] GPU{gid} {name}  batch={bs}")
            if C.SHARE_RIFE_GPU and gid == C.RIFE_GPU:
                shared_mode = "tiled" if self._use_tiled_shared_rife(gid) else "direct"
                print(
                    f"    shared-with-RIFE {shared_mode}  "
                    f"mem={self.gpu_total_mem_mib.get(gid, 0)}MiB  "
                    f"tile={C.RIFE_SHARED_ESRGAN_TILE} pad={C.RIFE_SHARED_ESRGAN_PAD}"
                )

            model_path = self._resolve_model_path()
            m = spandrel.ModelLoader().load_from_file(model_path)
            self.scale = m.scale
            self.output_scale = max(1, int(round(self.scale * self._downscale_factor)))
            net = m.model.half().to(dev).eval()
            net = self._maybe_compile(net, gid, bs, dev)
            self.models.append(net)

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
            model_path = self._resolve_model_path()
            m = spandrel.ModelLoader().load_from_file(model_path)
            self.cpu_model = m.model.to("cpu").eval()
            self.scale = m.scale
            
            dummy = torch.randn(1, 3, 315, 560, device="cpu", dtype=torch.float32)
            with torch.inference_mode():
                _ = self.cpu_model(dummy)
            del dummy

    def _resolve_model_path(self) -> str:
        """Resolve ESRGAN model path from visual profile or config fallback."""
        if self.visual_profile and self.visual_profile.model_key != "anime_baseline":
            try:
                from .models import ModelRegistry
                registry = ModelRegistry()
                return str(registry.get_path(self.visual_profile.model_key))
            except Exception as exc:
                print(f"  [ESRGAN] Model registry fallback: {exc}")
        return C.ESRGAN_MODEL

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

    def _use_tiled_shared_rife(self, dev_id: int) -> bool:
        return (
            C.SHARE_RIFE_GPU
            and dev_id == C.RIFE_GPU
            and C.RIFE_SHARED_ESRGAN_TILE > 0
            and self.gpu_total_mem_mib.get(dev_id, 0)
            <= C.RIFE_SHARED_ESRGAN_TILE_MAX_GPU_MEM_MIB
        )

    def _infer_tiled_u8(self, net, t_small, tile_size: int, tile_pad: int):
        """Run tiled inference to fit ESRGAN on low-VRAM shared GPUs."""
        torch = self.torch
        n, c, h, w = t_small.shape
        scale = self.scale
        out_u8 = torch.empty(
            (n, c, h * scale, w * scale),
            dtype=torch.uint8,
            device=t_small.device,
        )

        for bi in range(n):
            img = t_small[bi : bi + 1]
            for y in range(0, h, tile_size):
                y_end = min(y + tile_size, h)
                y0 = max(y - tile_pad, 0)
                y1 = min(y_end + tile_pad, h)
                for x in range(0, w, tile_size):
                    x_end = min(x + tile_size, w)
                    x0 = max(x - tile_pad, 0)
                    x1 = min(x_end + tile_pad, w)

                    tile = img[:, :, y0:y1, x0:x1]
                    with torch.inference_mode():
                        out_tile = net(tile)
                    out_tile_u8 = (out_tile.clamp(0, 1) * 255).byte()

                    crop_top = (y - y0) * scale
                    crop_bottom = crop_top + (y_end - y) * scale
                    crop_left = (x - x0) * scale
                    crop_right = crop_left + (x_end - x) * scale
                    out_u8[
                        bi : bi + 1,
                        :,
                        y * scale : y_end * scale,
                        x * scale : x_end * scale,
                    ] = out_tile_u8[:, :, crop_top:crop_bottom, crop_left:crop_right]
                    del tile, out_tile, out_tile_u8

        return out_u8

    def _run_model_to_u8(self, net, t_small, dev_id: int):
        if self._use_tiled_shared_rife(dev_id):
            return self._infer_tiled_u8(
                net,
                t_small,
                C.RIFE_SHARED_ESRGAN_TILE,
                C.RIFE_SHARED_ESRGAN_PAD,
            )
        with self.torch.inference_mode():
            out = net(t_small)
        out_u8 = (out.clamp(0, 1) * 255).byte()
        del out
        return out_u8

    def _gpu_worker(self, worker_idx: int, dev_id: int,
                    frames: list[np.ndarray], get_batch: Callable,
                    store: list | None, on_frame: Callable | None,
                    counter: list, lock: threading.Lock, total: int, t0: float,
                    log_progress: bool,
                    telemetry: dict[str, float] | None,
                    telemetry_lock: threading.Lock):
        torch = self.torch
        net = self.models[worker_idx]
        dev = f"cuda:{dev_id}"
        bs = self.batches[worker_idx]
        use_pinned = C.ESRGAN_EXPERIMENTAL_PINNED_STAGING
        use_double_buf = C.ESRGAN_D2H_DOUBLE_BUFFER

        # -- CUDA streams ------------------------------------------------
        copy_stream = torch.cuda.Stream(device=dev_id) if use_pinned else None
        compute_stream = torch.cuda.Stream(device=dev_id) if use_pinned else None
        d2h_stream = torch.cuda.Stream(device=dev_id)  # always: async D2H

        # -- H2D pinned staging buffers (pinned path only) ----------------
        batch_buf = None   # numpy (bs, H, W, 3) uint8
        cpu_stage = None   # pinned torch (bs, 3, H, W) float16

        # -- D2H output buffers (lazy-allocated on first inference) -------
        d2h_bufs = [None, None]  # pinned torch (bs, out_h, out_w, 3) uint8
        buf_idx = 0              # toggles 0/1 for double-buffering

        # -- Double-buffer deferred-processing state ----------------------
        prev = None  # dict with previous iteration's metadata

        # -----------------------------------------------------------------
        # Helper: consume a completed D2H buffer (visual post-proc + callbacks)
        # -----------------------------------------------------------------
        def _consume_output(buf_tensor, iter_start, iter_bs):
            """Post-process and deliver frames from a completed D2H buffer.

            Returns writer_wait_dt (seconds spent inside on_frame callback).
            T2 optimisation: when no visual post-processing is needed, frames
            are delivered as pinned-memory torch tensors (zero-copy buffer
            protocol) instead of converting to numpy first.  The ReorderWriter
            and pipe.write() both accept objects that implement the buffer
            protocol, so the numpy round-trip is eliminated on the hot path.
            When store is requested the tensor slice is copied once via
            .numpy().copy().
            """
            needs_postproc = self._hybrid_weight > 0.0 or self._face_adaptive

            if needs_postproc:
                # Fall back to numpy for visual post-processing
                out_np = buf_tensor[:iter_bs].numpy()
                from .visual_eval import apply_hybrid_detail, apply_face_adaptive
                for i in range(iter_bs):
                    gidx = iter_start + i
                    original = frames[gidx]
                    if self._hybrid_weight > 0.0:
                        out_np[i] = apply_hybrid_detail(
                            out_np[i], original, self._hybrid_weight)
                    if self._face_adaptive:
                        out_np[i] = apply_face_adaptive(
                            out_np[i], original, self._face_roi)

                writer_wait_dt = 0.0
                for i in range(iter_bs):
                    gidx = iter_start + i
                    if store is not None:
                        store[gidx] = out_np[i].copy()
                    if on_frame is not None:
                        write_t0 = time.time()
                        on_frame(gidx, out_np[i])
                        writer_wait_dt += time.time() - write_t0
                return writer_wait_dt

            # T2: zero-copy path — pass pinned tensor slices directly
            writer_wait_dt = 0.0
            for i in range(iter_bs):
                gidx = iter_start + i
                frame_tensor = buf_tensor[i]
                if store is not None:
                    store[gidx] = frame_tensor.numpy().copy()
                if on_frame is not None:
                    write_t0 = time.time()
                    on_frame(gidx, frame_tensor)
                    writer_wait_dt += time.time() - write_t0
            return writer_wait_dt

        # -----------------------------------------------------------------
        # Helper: flush deferred prev state (sync + consume + telemetry)
        # -----------------------------------------------------------------
        def _flush_prev():
            nonlocal prev
            if prev is None:
                return
            prev["d2h_event"].synchronize()
            ww = _consume_output(prev["buf"], prev["start"], prev["cur_bs"])

            # Resolve deferred d2h timing from CUDA events if not yet computed
            d2h_dt_resolved = prev["d2h_dt"]
            if d2h_dt_resolved == 0.0 and "d2h_ev0" in prev:
                d2h_dt_resolved = prev["d2h_ev0"].elapsed_time(prev["d2h_ev1"]) / 1000.0

            if telemetry is not None:
                p = prev
                with telemetry_lock:
                    telemetry["fill"] = telemetry.get("fill", 0.0) + p["fill_dt"]
                    telemetry["h2d"] = telemetry.get("h2d", 0.0) + p["h2d_dt"]
                    telemetry["downscale"] = telemetry.get("downscale", 0.0) + p["downscale_dt"]
                    telemetry["infer"] = telemetry.get("infer", 0.0) + p["infer_dt"]
                    telemetry["d2h"] = telemetry.get("d2h", 0.0) + d2h_dt_resolved
                    telemetry["writer_wait"] = telemetry.get("writer_wait", 0.0) + ww

            with lock:
                counter[0] += prev["cur_bs"]
                c = counter[0]
                if log_progress and (c % 50 < bs or c >= total):
                    elapsed = time.time() - t0
                    fps_now = c / elapsed if elapsed > 0 else 0
                    eta = (total - c) / fps_now if fps_now > 0 else 0
                    print(f"    ESRGAN {c}/{total}  "
                          f"{fps_now:.1f}fps  ETA {eta:.0f}s",
                          flush=True)
            prev = None

        # =================================================================
        #  Main loop
        # =================================================================
        while True:
            if C.shutdown.is_set():
                _flush_prev()
                return

            start, end = get_batch(bs)
            if start is None:
                break
            cur_bs = end - start

            # ----- Pinned-staging path (default) -------------------------
            if use_pinned:
                frame0 = frames[start]
                if (
                    batch_buf is None
                    or batch_buf.shape[0] != bs
                    or batch_buf.shape[1:] != frame0.shape
                ):
                    h, w = frame0.shape[:2]
                    batch_buf = np.empty((bs, h, w, 3), dtype=np.uint8)
                    cpu_stage = torch.empty(
                        (bs, 3, h, w), dtype=torch.float16, pin_memory=True
                    )

                fill_t0 = time.time()
                for i in range(cur_bs):
                    batch_buf[i, ...] = frames[start + i]
                fill_dt = time.time() - fill_t0

                # Telemetry CUDA events
                h2d_ev0 = torch.cuda.Event(enable_timing=True)
                h2d_ev1 = torch.cuda.Event(enable_timing=True)
                ds_ev0 = torch.cuda.Event(enable_timing=True)
                ds_ev1 = torch.cuda.Event(enable_timing=True)
                inf_ev0 = torch.cuda.Event(enable_timing=True)
                inf_ev1 = torch.cuda.Event(enable_timing=True)
                d2h_ev0 = torch.cuda.Event(enable_timing=True)
                d2h_ev1 = torch.cuda.Event(enable_timing=True)

                # H2D: pinned → GPU
                cpu_batch = torch.from_numpy(batch_buf[:cur_bs]).permute(0, 3, 1, 2)
                cpu_stage[:cur_bs].copy_(cpu_batch)

                with torch.cuda.stream(copy_stream):
                    h2d_ev0.record()
                    t_gpu = cpu_stage[:cur_bs].to(dev, non_blocking=True)
                    t_gpu = t_gpu.mul_(1.0 / 255.0)
                    h2d_ev1.record()

                # Compute: downscale + inference
                with torch.cuda.stream(compute_stream):
                    compute_stream.wait_stream(copy_stream)
                    ds_ev0.record()
                    if self._downscale_factor < 1.0:
                        t_small = torch.nn.functional.interpolate(
                            t_gpu, scale_factor=self._downscale_factor,
                            mode="bilinear", align_corners=False)
                    else:
                        t_small = t_gpu
                    ds_ev1.record()

                    inf_ev0.record()
                    out_u8 = self._run_model_to_u8(net, t_small, dev_id)
                    inf_ev1.record()

                    # Permute to NHWC and make contiguous on GPU (fast)
                    out_gpu_nhwc = out_u8.permute(0, 2, 3, 1).contiguous()

                # Lazy-allocate D2H pinned output buffers on first inference
                if d2h_bufs[0] is None:
                    out_h, out_w = out_gpu_nhwc.shape[1], out_gpu_nhwc.shape[2]
                    d2h_bufs[0] = torch.empty(
                        (bs, out_h, out_w, 3), dtype=torch.uint8, pin_memory=True
                    )
                    if use_double_buf:
                        d2h_bufs[1] = torch.empty(
                            (bs, out_h, out_w, 3), dtype=torch.uint8, pin_memory=True
                        )
                    else:
                        d2h_bufs[1] = d2h_bufs[0]  # alias: no real double-buffering

                cur_d2h_buf = d2h_bufs[buf_idx]

                # Async D2H copy on dedicated stream
                compute_done = torch.cuda.Event()
                compute_done.record(compute_stream)
                with torch.cuda.stream(d2h_stream):
                    d2h_stream.wait_event(compute_done)
                    d2h_ev0.record()
                    cur_d2h_buf[:cur_bs].copy_(out_gpu_nhwc, non_blocking=True)
                    d2h_ev1.record()

                d2h_done = torch.cuda.Event()
                d2h_done.record(d2h_stream)

                # ---- Process PREVIOUS iteration while D2H runs ----------
                if use_double_buf:
                    _flush_prev()
                else:
                    # Without double-buffering we must wait for THIS D2H
                    _flush_prev()
                    d2h_done.synchronize()

                # Resolve telemetry for current iteration (need compute sync)
                compute_stream.synchronize()
                h2d_dt = h2d_ev0.elapsed_time(h2d_ev1) / 1000.0
                downscale_dt = ds_ev0.elapsed_time(ds_ev1) / 1000.0
                infer_dt = inf_ev0.elapsed_time(inf_ev1) / 1000.0
                # d2h timing resolved when prev is flushed next iteration
                d2h_dt_val = d2h_ev0.elapsed_time(d2h_ev1) / 1000.0 if not use_double_buf else 0.0

                if use_double_buf:
                    # Defer output processing to next iteration
                    prev = {
                        "d2h_event": d2h_done,
                        "buf": cur_d2h_buf,
                        "start": start,
                        "cur_bs": cur_bs,
                        "fill_dt": fill_dt,
                        "h2d_dt": h2d_dt,
                        "downscale_dt": downscale_dt,
                        "infer_dt": infer_dt,
                        "d2h_dt": 0.0,  # placeholder, resolved at flush
                        "d2h_ev0": d2h_ev0,
                        "d2h_ev1": d2h_ev1,
                    }
                    buf_idx = 1 - buf_idx
                else:
                    # Already synced above; process immediately
                    ww = _consume_output(cur_d2h_buf, start, cur_bs)
                    if telemetry is not None:
                        with telemetry_lock:
                            telemetry["fill"] = telemetry.get("fill", 0.0) + fill_dt
                            telemetry["h2d"] = telemetry.get("h2d", 0.0) + h2d_dt
                            telemetry["downscale"] = telemetry.get("downscale", 0.0) + downscale_dt
                            telemetry["infer"] = telemetry.get("infer", 0.0) + infer_dt
                            telemetry["d2h"] = telemetry.get("d2h", 0.0) + d2h_dt_val
                            telemetry["writer_wait"] = telemetry.get("writer_wait", 0.0) + ww
                    with lock:
                        counter[0] += cur_bs
                        c = counter[0]
                        if log_progress and (c % 50 < bs or c >= total):
                            elapsed = time.time() - t0
                            fps_now = c / elapsed if elapsed > 0 else 0
                            eta = (total - c) / fps_now if fps_now > 0 else 0
                            print(f"    ESRGAN {c}/{total}  "
                                  f"{fps_now:.1f}fps  ETA {eta:.0f}s",
                                  flush=True)

                # Cleanup GPU tensors (pinned D2H buffers are reused)
                del cpu_batch, t_gpu, t_small, out_u8, out_gpu_nhwc

            # ----- Default (non-pinned) fallback path --------------------
            else:
                fill_t0 = time.time()
                batch_np = np.stack([frames[start + i] for i in range(cur_bs)])
                fill_dt = time.time() - fill_t0

                h2d_t0 = time.time()
                t_gpu = (
                    torch.from_numpy(batch_np)
                    .permute(0, 3, 1, 2)
                    .to(dev, dtype=torch.float16, non_blocking=True)
                    / 255.0
                )
                h2d_dt = time.time() - h2d_t0

                downscale_t0 = time.time()
                if self._downscale_factor < 1.0:
                    t_small = torch.nn.functional.interpolate(
                        t_gpu, scale_factor=self._downscale_factor,
                        mode="bilinear", align_corners=False)
                else:
                    t_small = t_gpu
                downscale_dt = time.time() - downscale_t0

                infer_t0 = time.time()
                out_u8 = self._run_model_to_u8(net, t_small, dev_id)
                infer_dt = time.time() - infer_t0

                d2h_t0 = time.time()
                # Permute+contiguous on GPU, then async D2H to pinned buffer
                out_gpu_nhwc = out_u8.permute(0, 2, 3, 1).contiguous()

                # Lazy-allocate single pinned D2H buffer
                if d2h_bufs[0] is None:
                    out_h, out_w = out_gpu_nhwc.shape[1], out_gpu_nhwc.shape[2]
                    d2h_bufs[0] = torch.empty(
                        (bs, out_h, out_w, 3), dtype=torch.uint8, pin_memory=True
                    )

                with torch.cuda.stream(d2h_stream):
                    d2h_bufs[0][:cur_bs].copy_(out_gpu_nhwc, non_blocking=True)
                d2h_stream.synchronize()
                d2h_dt = time.time() - d2h_t0

                out_np = d2h_bufs[0][:cur_bs].numpy()

                # Apply visual post-processing from profile
                if self._hybrid_weight > 0.0 or self._face_adaptive:
                    from .visual_eval import apply_hybrid_detail, apply_face_adaptive
                    for i in range(cur_bs):
                        gidx = start + i
                        original = frames[gidx]
                        if self._hybrid_weight > 0.0:
                            out_np[i] = apply_hybrid_detail(
                                out_np[i], original, self._hybrid_weight)
                        if self._face_adaptive:
                            out_np[i] = apply_face_adaptive(
                                out_np[i], original, self._face_roi)

                writer_wait_dt = 0.0
                for i in range(cur_bs):
                    gidx = start + i
                    if store is not None:
                        store[gidx] = out_np[i].copy()
                    if on_frame is not None:
                        write_t0 = time.time()
                        on_frame(gidx, out_np[i])
                        writer_wait_dt += time.time() - write_t0

                if telemetry is not None:
                    with telemetry_lock:
                        telemetry["fill"] = telemetry.get("fill", 0.0) + fill_dt
                        telemetry["h2d"] = telemetry.get("h2d", 0.0) + h2d_dt
                        telemetry["downscale"] = telemetry.get("downscale", 0.0) + downscale_dt
                        telemetry["infer"] = telemetry.get("infer", 0.0) + infer_dt
                        telemetry["d2h"] = telemetry.get("d2h", 0.0) + d2h_dt
                        telemetry["writer_wait"] = telemetry.get("writer_wait", 0.0) + writer_wait_dt

                del batch_np, t_gpu, t_small, out_u8, out_gpu_nhwc

                with lock:
                    counter[0] += cur_bs
                    c = counter[0]
                    if log_progress and (c % 50 < bs or c >= total):
                        elapsed = time.time() - t0
                        fps_now = c / elapsed if elapsed > 0 else 0
                        eta = (total - c) / fps_now if fps_now > 0 else 0
                        print(f"    ESRGAN {c}/{total}  "
                              f"{fps_now:.1f}fps  ETA {eta:.0f}s",
                              flush=True)

        # =================================================================
        #  Post-loop: flush last deferred double-buffer iteration
        # =================================================================
        if prev is not None:
            _flush_prev()


    def _cpu_worker(self, frames: list[np.ndarray], get_batch: Callable,
                    store: list | None, on_frame: Callable | None,
                    counter: list, lock: threading.Lock, total: int, t0: float,
                    log_progress: bool,
                    telemetry: dict[str, float] | None,
                    telemetry_lock: threading.Lock):
        torch = self.torch
        net = self.cpu_model

        while True:
            start, _ = get_batch(1)
            if start is None:
                break
            if C.shutdown.is_set():
                return
            
            frame = frames[start]
            batch_t0 = time.time()
            t_cpu = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to("cpu", dtype=torch.float32) / 255.0

            if self._downscale_factor < 1.0:
                t_small = torch.nn.functional.interpolate(
                    t_cpu, scale_factor=self._downscale_factor,
                    mode="bilinear", align_corners=False)
            else:
                t_small = t_cpu

            with torch.inference_mode():
                out = net(t_small)

            out_u8 = (out.clamp(0, 1) * 255).byte()
            out_cpu_frame = out_u8[0].permute(1, 2, 0).numpy()

            gidx = start
            if store is not None:
                store[gidx] = out_cpu_frame.copy()
            if on_frame is not None:
                on_frame(gidx, out_cpu_frame)

            if telemetry is not None:
                with telemetry_lock:
                    telemetry["cpu"] = telemetry.get("cpu", 0.0) + (time.time() - batch_t0)

            with lock:
                counter[0] += 1
                c = counter[0]
                if log_progress and (c % 50 == 0 or c >= total):
                    elapsed = time.time() - t0
                    fps_now = c / elapsed if elapsed > 0 else 0
                    eta = (total - c) / fps_now if fps_now > 0 else 0
                    print(f"    ESRGAN {c}/{total}  "
                          f"{fps_now:.1f}fps  ETA {eta:.0f}s",
                          flush=True)
            
            del t_cpu, t_small, out, out_u8

    def _run_parallel(self, frames: list[np.ndarray],
                      store: list | None,
                      on_frame: Callable | None,
                      log_progress: bool = True,
                      telemetry: dict[str, float] | None = None,
                      active_gpu_ids: list[int] | None = None) -> int:
        """Dispatch dynamically fetching threads."""
        total = len(frames)
        if total == 0:
            return 0

        pos = [0]
        lock = threading.Lock()
        telemetry_lock = threading.Lock()

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

        enabled_gpu_ids = set(active_gpu_ids or self.gpu_ids)

        # Deploy fetching workers only on GPUs enabled for this pass.
        for worker_idx, gpu_id in enumerate(self.gpu_ids):
            if gpu_id not in enabled_gpu_ids:
                continue
            launch(
                self._gpu_worker,
                worker_idx,
                gpu_id,
                frames,
                get_batch,
                store,
                on_frame,
                counter,
                lock,
                total,
                t0,
                log_progress,
                telemetry,
                telemetry_lock,
            )
            
        # Deploy fetching worker for CPU (R9 Processor: Pulls 1 frame slowly behind them)
        if self.cpu_enabled:
            launch(
                self._cpu_worker,
                frames,
                get_batch,
                store,
                on_frame,
                counter,
                lock,
                total,
                t0,
                log_progress,
                telemetry,
                telemetry_lock,
            )

        for t in threads:
            t.join()

        if errors:
            raise RuntimeError(f"ESRGAN worker failed: {errors[0]}") from errors[0]

        elapsed = time.time() - t0
        c = counter[0]
        if log_progress:
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
                          on_frame: Callable[[int, np.ndarray], None],
                          log_progress: bool = True,
                          telemetry: dict[str, float] | None = None,
                          active_gpu_ids: list[int] | None = None) -> int:
        return self._run_parallel(
            frames,
            store=None,
            on_frame=on_frame,
            log_progress=log_progress,
            telemetry=telemetry,
            active_gpu_ids=active_gpu_ids,
        )
