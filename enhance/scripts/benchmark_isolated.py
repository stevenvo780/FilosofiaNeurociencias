"""
Benchmark aislado: prueba ESRGAN en cada dispositivo por separado
para identificar cuellos de botella reales.

Test 1: Solo GPU0 (RTX 5070 Ti)
Test 2: Solo GPU1 (RTX 2060)
Test 3: GPU0 + GPU1
Test 4: GPU0 + GPU1 + CPU
"""
import os, sys, time, threading, subprocess, json

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import torch
import numpy as np

# Generate 200 fake frames at input resolution
N_FRAMES = 200
H, W = 1260, 2240
print(f"Generating {N_FRAMES} random frames ({W}x{H})...")
frames = [np.random.randint(0, 255, (H, W, 3), dtype=np.uint8) for _ in range(N_FRAMES)]
print("Done.\n")


def gpu_utilization():
    """Sample GPU utilization once."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"], text=True)
        lines = out.strip().split('\n')
        result = {}
        for line in lines:
            parts = [x.strip() for x in line.split(',')]
            result[int(parts[0])] = {'util': int(parts[1]), 'mem_mb': int(parts[2])}
        return result
    except:
        return {}


def monitor_gpu(stop_event, samples, interval=0.5):
    """Background thread that samples GPU utilization."""
    while not stop_event.is_set():
        s = gpu_utilization()
        if s:
            samples.append(s)
        time.sleep(interval)


def run_esrgan_test(label, gpu_ids, use_cpu, frames):
    """Run ESRGAN inference on specified devices and measure."""
    import spandrel
    from enhance.config import ESRGAN_MODEL

    print(f"{'='*60}")
    print(f"  TEST: {label}")
    print(f"  GPUs: {gpu_ids}  CPU: {use_cpu}")
    print(f"{'='*60}")

    # Load models for requested GPUs
    models = {}
    streams = {}
    batches = {0: 8, 1: 4}

    for gid in gpu_ids:
        dev = f"cuda:{gid}"
        m = spandrel.ModelLoader().load_from_file(ESRGAN_MODEL)
        net = m.model.half().to(dev).eval()
        models[gid] = net
        streams[gid] = torch.cuda.Stream(device=dev)
        # warmup
        dummy = torch.randn(batches[gid], 3, 315, 560, device=dev, dtype=torch.float16)
        with torch.inference_mode():
            _ = net(dummy); _ = net(dummy)
        torch.cuda.synchronize(gid)
        del dummy
        torch.cuda.empty_cache()
        print(f"  GPU{gid} ready, VRAM={torch.cuda.memory_allocated(gid)/1e9:.2f}GB")

    cpu_model = None
    if use_cpu:
        torch.set_num_threads(16)
        m = spandrel.ModelLoader().load_from_file(ESRGAN_MODEL)
        cpu_model = m.model.to("cpu").eval()
        dummy = torch.randn(1, 3, 315, 560, dtype=torch.float32)
        with torch.inference_mode():
            _ = cpu_model(dummy)
        del dummy
        print(f"  CPU ready (16 threads)")

    # Dynamic queue
    total = len(frames)
    pos = [0]
    counter = [0]
    lock = threading.Lock()

    def get_batch(bs):
        with lock:
            if pos[0] >= total:
                return None, 0
            s = pos[0]
            e = min(total, s + bs)
            pos[0] = e
            return s, e

    def gpu_worker(gid):
        net = models[gid]
        dev = f"cuda:{gid}"
        bs = batches[gid]
        stream = streams[gid]

        while True:
            start, end = get_batch(bs)
            if start is None:
                break

            cur_bs = end - start
            # Simple: build tensor directly (no pinned buffer to isolate GPU perf)
            batch_np = np.stack([frames[start + i] for i in range(cur_bs)])
            t_in = torch.from_numpy(batch_np).permute(0, 3, 1, 2).to(dev, dtype=torch.float16, non_blocking=True) / 255.0

            t_small = torch.nn.functional.interpolate(
                t_in, scale_factor=0.5, mode="bilinear", align_corners=False)

            with torch.inference_mode():
                out = net(t_small)

            # Force sync to measure real GPU time
            torch.cuda.synchronize(gid)

            with lock:
                counter[0] += cur_bs

            del batch_np, t_in, t_small, out

    def cpu_worker_fn():
        net = cpu_model
        while True:
            start, end = get_batch(1)
            if start is None:
                break
            frame = frames[start]
            t_cpu = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            t_small = torch.nn.functional.interpolate(
                t_cpu, scale_factor=0.5, mode="bilinear", align_corners=False)
            with torch.inference_mode():
                out = net(t_small)
            with lock:
                counter[0] += 1
            del t_cpu, t_small, out

    # Monitor GPU
    stop_mon = threading.Event()
    samples = []
    mon_t = threading.Thread(target=monitor_gpu, args=(stop_mon, samples, 0.3))
    mon_t.start()

    t0 = time.time()

    threads = []
    for gid in gpu_ids:
        t = threading.Thread(target=gpu_worker, args=(gid,))
        t.start()
        threads.append(t)
    if use_cpu and cpu_model is not None:
        t = threading.Thread(target=cpu_worker_fn)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    elapsed = time.time() - t0
    stop_mon.set()
    mon_t.join()

    fps = counter[0] / elapsed
    print(f"\n  RESULT: {counter[0]} frames in {elapsed:.1f}s = {fps:.1f} fps")

    # Analyze GPU samples
    for gid in sorted(set(gpu_ids)):
        utils = [s[gid]['util'] for s in samples if gid in s]
        if utils:
            avg = sum(utils) / len(utils)
            peak = max(utils)
            print(f"  GPU{gid} avg={avg:.0f}%  peak={peak}%  samples={len(utils)}")
    
    print()

    # Cleanup
    for gid in gpu_ids:
        del models[gid]
        torch.cuda.empty_cache()
    if cpu_model is not None:
        del cpu_model
    torch.cuda.synchronize()

    return fps


# ── Run tests ──
results = {}

print("\n" + "="*60)
print("  ISOLATED BENCHMARK — ESRGAN on 200 frames (2240x1260)")
print("="*60 + "\n")

results['gpu0_only'] = run_esrgan_test("GPU0 sola (RTX 5070 Ti)", [0], False, frames)
time.sleep(2)

results['gpu1_only'] = run_esrgan_test("GPU1 sola (RTX 2060)", [1], False, frames)
time.sleep(2)

results['gpu0_gpu1'] = run_esrgan_test("GPU0 + GPU1 (sin CPU)", [0, 1], False, frames)
time.sleep(2)

results['all'] = run_esrgan_test("GPU0 + GPU1 + CPU", [0, 1], True, frames)

print("\n" + "="*60)
print("  RESUMEN FINAL")
print("="*60)
for k, v in results.items():
    print(f"  {k:20s}: {v:.1f} fps")
print()
