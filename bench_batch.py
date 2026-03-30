#!/usr/bin/env python3
"""Benchmark optimal batch sizes for both GPUs."""
import os, time, gc
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import numpy as np
import torch
import spandrel

MODEL = "/tmp/realesr-animevideov3.pth"
W, H = 2240, 1260
# After 0.5x downscale: 1120x630
INW, INH = W // 2, H // 2

def bench_gpu(gid, batch_sizes):
    dev = f"cuda:{gid}"
    name = torch.cuda.get_device_name(gid)
    print(f"\n{'='*60}")
    print(f"  GPU{gid}: {name}  ({torch.cuda.get_device_properties(gid).total_memory/1e9:.1f}GB)")
    print(f"{'='*60}")

    m = spandrel.ModelLoader().load_from_file(MODEL)
    net = m.model.half().to(dev).eval()

    for bs in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(gid)

        # Create input batch (already at 0.5x size, as the pipeline does)
        try:
            inp = torch.randn(bs, 3, INH, INW, device=dev, dtype=torch.float16)

            # Warmup
            with torch.inference_mode():
                _ = net(inp)
            torch.cuda.synchronize(gid)

            # Benchmark 5 iterations
            t0 = time.time()
            iters = 5
            for _ in range(iters):
                with torch.inference_mode():
                    out = net(inp)
                torch.cuda.synchronize(gid)
            dt = time.time() - t0

            total_frames = bs * iters
            fps = total_frames / dt
            vram_peak = torch.cuda.max_memory_allocated(gid) / 1e9
            vram_now = torch.cuda.memory_allocated(gid) / 1e9

            print(f"  batch={bs:3d}  {fps:6.1f} fps  "
                  f"VRAM peak={vram_peak:.2f}GB  current={vram_now:.2f}GB  "
                  f"({dt/iters:.3f}s/batch)")

            del inp, out
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"  batch={bs:3d}  ❌ OOM!")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"  batch={bs:3d}  ❌ {e}")
            torch.cuda.empty_cache()
            break

    del net
    torch.cuda.empty_cache()
    gc.collect()

# GPU0: RTX 5070 Ti 16GB
bench_gpu(0, [8, 12, 16, 20, 24, 28, 32])

# GPU1: RTX 2060 6GB
bench_gpu(1, [4, 6, 8, 10, 12])

print("\n" + "="*60)
print("  Done! Pick batch sizes with best fps before OOM.")
print("="*60)
