#!/usr/bin/env python3
"""
Component tests — validate each piece independently.
Run:  python3 test_components.py
"""
import sys, os, time, gc, subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import numpy as np
from pathlib import Path

SRC = Path("GMT20260320-130023_Recording_2240x1260.mp4").resolve()
W, H, FPS = 2240, 1260, 25.0
CHUNK_DUR = 10.0  # 10 seconds = 250 frames
START = 60.0      # skip first minute (might be black)

def mem_mb():
    """RSS in MB."""
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

def gpu_mem():
    """VRAM allocated per GPU in MB."""
    try:
        import torch
        return {i: torch.cuda.memory_allocated(i) / 1e6
                for i in range(torch.cuda.device_count())}
    except:
        return {}

def separator(name):
    print(f"\n{'='*60}")
    print(f"  TEST: {name}")
    print(f"{'='*60}")

# ─────────────────────────────────────────────────────────────
# TEST 1: extract_frames_to_ram
# ─────────────────────────────────────────────────────────────
def test_extract():
    separator("extract_frames_to_ram")
    from enhance.ffmpeg_utils import extract_frames_to_ram

    ram0 = mem_mb()
    t0 = time.time()
    frames = extract_frames_to_ram(SRC, START, CHUNK_DUR, W, H)
    dt = time.time() - t0
    n = len(frames)
    ram1 = mem_mb()

    expected_n = int(CHUNK_DUR * FPS)
    frame_mb = frames[0].nbytes / 1e6 if frames else 0
    total_mb = n * frame_mb

    print(f"  Frames:   {n}  (expected ~{expected_n})")
    print(f"  Shape:    {frames[0].shape}  dtype={frames[0].dtype}")
    print(f"  Writable: {frames[0].flags.writeable}")
    print(f"  Per frame: {frame_mb:.1f}MB  Total: {total_mb:.0f}MB")
    print(f"  Time:     {dt:.1f}s  ({n/dt:.0f}fps)")
    print(f"  RSS:      {ram0:.0f} → {ram1:.0f}MB  (+{ram1-ram0:.0f}MB)")

    ok = True
    if abs(n - expected_n) > 3:
        print(f"  ❌ FAIL: frame count off by {abs(n-expected_n)}")
        ok = False
    if not frames[0].flags.writeable:
        print(f"  ❌ FAIL: frames not writable (will cause torch warning)")
        ok = False
    if frames[0].shape != (H, W, 3):
        print(f"  ❌ FAIL: wrong shape")
        ok = False

    # Cleanup
    del frames
    gc.collect()
    ram2 = mem_mb()
    print(f"  After del: RSS={ram2:.0f}MB (freed {ram1-ram2:.0f}MB)")

    if ok:
        print(f"  ✅ PASS")
    return ok

# ─────────────────────────────────────────────────────────────
# TEST 2: ESRGAN process_frames
# ─────────────────────────────────────────────────────────────
def test_esrgan():
    separator("ESRGAN process_frames (dual-GPU)")
    from enhance.ffmpeg_utils import extract_frames_to_ram
    from enhance.esrgan import ESRGANEngine
    import torch

    # Extract small set
    frames = extract_frames_to_ram(SRC, START, CHUNK_DUR, W, H)
    n = len(frames)
    raw_mb = sum(f.nbytes for f in frames) / 1e6
    print(f"  Input: {n} frames ({raw_mb:.0f}MB)")

    gpu0_before = torch.cuda.memory_allocated(0) / 1e6
    gpu1_before = torch.cuda.memory_allocated(1) / 1e6 if torch.cuda.device_count() > 1 else 0

    esr = ESRGANEngine()
    ram0 = mem_mb()
    t0 = time.time()
    results = esr.process_frames(frames, out_dir=None)
    dt = time.time() - t0

    # Free input immediately
    del frames
    gc.collect()

    ok = True
    nr = len(results)
    if nr != n:
        print(f"  ❌ FAIL: got {nr} results, expected {n}")
        ok = False

    if results[0] is None:
        print(f"  ❌ FAIL: first result is None")
        ok = False
    else:
        oh, ow = results[0].shape[:2]
        out_mb = sum(r.nbytes for r in results if r is not None) / 1e6
        print(f"  Output: {nr} frames  shape={results[0].shape}  dtype={results[0].dtype}")
        print(f"  Output size: {out_mb:.0f}MB")
        # Check scale: input 2240x1260 → downscale 0.5x → ESRGAN 4x → 4480x2520
        if oh != H * 2 or ow != W * 2:
            print(f"  ❌ FAIL: expected {W*2}x{H*2}, got {ow}x{oh}")
            ok = False
        # Check each result is an independent copy (not a view)
        print(f"  result[0] owns data: {results[0].flags.owndata}")
        print(f"  result[0] c_contiguous: {results[0].flags.c_contiguous}")

    print(f"  Time:  {dt:.1f}s  ({n/dt:.1f}fps)")
    print(f"  GPU0 VRAM: {gpu0_before:.0f} → {torch.cuda.memory_allocated(0)/1e6:.0f}MB")
    if torch.cuda.device_count() > 1:
        print(f"  GPU1 VRAM: {gpu1_before:.0f} → {torch.cuda.memory_allocated(1)/1e6:.0f}MB")

    # Free results and check RAM drops
    ram1 = mem_mb()
    del results
    gc.collect()
    ram2 = mem_mb()
    print(f"  RSS: before_del={ram1:.0f}  after_del={ram2:.0f}  freed={ram1-ram2:.0f}MB")

    # Free ESRGAN model
    del esr
    torch.cuda.empty_cache()
    gc.collect()

    if ok:
        print(f"  ✅ PASS")
    return ok

# ─────────────────────────────────────────────────────────────
# TEST 3: _encode_from_numpy
# ─────────────────────────────────────────────────────────────
def test_encode():
    separator("_encode_from_numpy (NVENC pipe)")

    out_file = Path("/tmp/test_encode.mp4")
    out_file.unlink(missing_ok=True)

    # Create 100 fake 4480x2520 frames (gradient pattern)
    n = 100
    print(f"  Generating {n} test frames 4480x2520...")
    frames = []
    for i in range(n):
        # Simple gradient so we can verify the encode works
        f = np.zeros((2520, 4480, 3), dtype=np.uint8)
        f[:, :, 0] = (i * 2) % 256  # R varies per frame
        f[:, :, 1] = 128            # G constant
        f[:, :, 2] = 64             # B constant
        frames.append(f)

    total_mb = sum(f.nbytes for f in frames) / 1e6
    print(f"  Total data: {total_mb:.0f}MB")

    from enhance.pipeline import _encode_from_numpy
    t0 = time.time()
    _encode_from_numpy(frames, out_file, FPS, gpu=0)
    dt = time.time() - t0

    ok = True
    if not out_file.exists():
        print(f"  ❌ FAIL: output file not created")
        ok = False
    else:
        sz = out_file.stat().st_size
        print(f"  Output: {sz/1e6:.1f}MB")
        print(f"  Time:  {dt:.1f}s  ({n/dt:.1f}fps)")

        # Verify with ffprobe
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=width,height,nb_frames,codec_name",
             "-print_format", "json", str(out_file)],
            capture_output=True, text=True)
        import json
        info = json.loads(r.stdout)
        stream = info["streams"][0]
        print(f"  Codec: {stream.get('codec_name')}  "
              f"{stream.get('width')}x{stream.get('height')}  "
              f"frames={stream.get('nb_frames')}")

        if stream.get("codec_name") != "hevc":
            print(f"  ❌ FAIL: expected hevc codec")
            ok = False

    del frames
    out_file.unlink(missing_ok=True)

    if ok:
        print(f"  ✅ PASS")
    return ok

# ─────────────────────────────────────────────────────────────
# TEST 4: Full pipeline 1 chunk (extract→ESRGAN→encode)
# ─────────────────────────────────────────────────────────────
def test_pipeline_1chunk():
    separator("Pipeline 1 chunk (extract→ESRGAN→NVENC)")
    import shutil
    from enhance.ffmpeg_utils import extract_frames_to_ram
    from enhance.esrgan import ESRGANEngine
    from enhance.pipeline import _encode_from_numpy
    import torch

    out_file = Path("/tmp/test_pipeline.mp4")
    out_file.unlink(missing_ok=True)

    ram0 = mem_mb()
    print(f"  RSS start: {ram0:.0f}MB")

    # Stage 1: Extract
    t0 = time.time()
    frames = extract_frames_to_ram(SRC, START, CHUNK_DUR, W, H)
    t_ext = time.time() - t0
    n = len(frames)
    raw_mb = sum(f.nbytes for f in frames) / 1e6
    ram1 = mem_mb()
    print(f"  [Extract] {n} frames  {t_ext:.1f}s  {n/t_ext:.0f}fps  "
          f"RSS={ram1:.0f}MB (+{ram1-ram0:.0f}MB, data={raw_mb:.0f}MB)")

    # Stage 2: ESRGAN
    esr = ESRGANEngine()
    t1 = time.time()
    results = esr.process_frames(frames, out_dir=None)
    t_esr = time.time() - t1
    del frames  # free raw
    gc.collect()
    ram2 = mem_mb()
    out_mb = sum(r.nbytes for r in results if r is not None) / 1e6
    print(f"  [ESRGAN]  {len(results)} frames  {t_esr:.1f}s  {n/t_esr:.1f}fps  "
          f"RSS={ram2:.0f}MB (raw freed, out={out_mb:.0f}MB)")

    # Stage 3: Encode
    t2 = time.time()
    _encode_from_numpy(results, out_file, FPS, gpu=0)
    t_enc = time.time() - t2
    del results  # free ESRGAN output
    gc.collect()
    ram3 = mem_mb()
    print(f"  [NVENC]   {t_enc:.1f}s  ({n/t_enc:.1f}fps)  RSS={ram3:.0f}MB")

    # Cleanup
    del esr
    torch.cuda.empty_cache()
    gc.collect()
    ram4 = mem_mb()
    print(f"  Final RSS: {ram4:.0f}MB  (started at {ram0:.0f}MB)")

    total = time.time() - t0
    print(f"  Total time: {total:.1f}s for {n} frames  ({n/total:.1f}fps overall)")

    ok = out_file.exists() and out_file.stat().st_size > 1000
    if ok:
        sz = out_file.stat().st_size
        print(f"  Output: {sz/1e6:.1f}MB")
        print(f"  ✅ PASS")
    else:
        print(f"  ❌ FAIL: output missing or too small")
    out_file.unlink(missing_ok=True)
    return ok


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not SRC.exists():
        print(f"[!] Video not found: {SRC}")
        sys.exit(1)

    tests = [
        ("extract", test_extract),
        ("esrgan", test_esrgan),
        ("encode", test_encode),
        ("pipeline", test_pipeline_1chunk),
    ]

    # Allow running specific test: python3 test_components.py esrgan
    if len(sys.argv) > 1:
        names = sys.argv[1:]
        tests = [(n, f) for n, f in tests if n in names]

    results = {}
    for name, func in tests:
        try:
            results[name] = func()
        except Exception as e:
            print(f"  ❌ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for name, ok in results.items():
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {name:20s}  {status}")
    print(f"{'='*60}")

    sys.exit(0 if all(results.values()) else 1)
