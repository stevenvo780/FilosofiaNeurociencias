#!/usr/bin/env python3
"""
Component tests — validate each piece independently.
Run:  python3 test_components.py
      python3 test_components.py streaming   # run only streaming test
"""
import sys, os, time, gc, subprocess, json, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import numpy as np

SRC = (ROOT / "videos" / "GMT20260320-130023_Recording_2240x1260.mp4").resolve()
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
# TEST 4: ESRGAN process_streaming → NVENC pipe (zero accumulation)
# ─────────────────────────────────────────────────────────────
def test_streaming():
    separator("ESRGAN streaming → NVENC pipe (zero accumulation)")
    from enhance.ffmpeg_utils import extract_frames_to_ram
    from enhance.esrgan import ESRGANEngine
    from enhance.pipeline import _open_nvenc_pipe, _ReorderWriter
    import torch

    out_file = Path("/tmp/test_streaming.mp4")
    out_file.unlink(missing_ok=True)

    # Extract frames
    frames = extract_frames_to_ram(SRC, START, CHUNK_DUR, W, H)
    n = len(frames)
    raw_mb = sum(f.nbytes for f in frames) / 1e6
    print(f"  Input: {n} frames ({raw_mb:.0f}MB)")

    esr = ESRGANEngine()

    # Output dimensions: 2x (4x ESRGAN on 0.5x downscale)
    out_w, out_h = W * 2, H * 2

    # Open NVENC pipe
    proc = _open_nvenc_pipe(out_file, out_w, out_h, FPS, gpu=0)
    writer = _ReorderWriter(proc.stdin, n)

    ram0 = mem_mb()
    t0 = time.time()
    count = esr.process_streaming(frames, writer.on_frame)
    writer.flush_remaining()

    try:
        proc.stdin.close()
    except BrokenPipeError:
        pass
    proc.wait()
    dt = time.time() - t0

    # Free input + engine
    del frames
    del esr
    torch.cuda.empty_cache()
    gc.collect()
    ram1 = mem_mb()

    ok = True
    print(f"  Processed: {count} frames")
    print(f"  Written to pipe: {writer.written} frames")
    print(f"  Time:  {dt:.1f}s  ({n/dt:.1f}fps end-to-end)")
    print(f"  RSS:   {ram0:.0f} → {ram1:.0f}MB")

    if writer.written != n:
        print(f"  ❌ FAIL: wrote {writer.written}/{n}")
        ok = False

    if proc.returncode != 0:
        print(f"  ❌ FAIL: NVENC rc={proc.returncode}")
        ok = False

    if out_file.exists():
        sz = out_file.stat().st_size
        print(f"  Output: {sz/1e6:.1f}MB")
        # Verify with ffprobe
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=width,height,nb_frames,codec_name",
             "-print_format", "json", str(out_file)],
            capture_output=True, text=True)
        info = json.loads(r.stdout)
        stream = info["streams"][0]
        print(f"  Codec: {stream.get('codec_name')}  "
              f"{stream.get('width')}x{stream.get('height')}  "
              f"frames={stream.get('nb_frames')}")
        if stream.get("codec_name") != "hevc":
            print(f"  ❌ FAIL: expected hevc")
            ok = False
        probe_w = int(stream.get("width", 0))
        probe_h = int(stream.get("height", 0))
        if probe_w != out_w or probe_h != out_h:
            print(f"  ❌ FAIL: expected {out_w}x{out_h}, got {probe_w}x{probe_h}")
            ok = False
    else:
        print(f"  ❌ FAIL: output file not created")
        ok = False

    # Check reorder buffer is empty
    if writer.buf:
        print(f"  ❌ FAIL: {len(writer.buf)} orphan frames in reorder buffer")
        ok = False

    out_file.unlink(missing_ok=True)
    if ok:
        print(f"  ✅ PASS")
    return ok


# ─────────────────────────────────────────────────────────────
# TEST 5: Full pipeline 1 chunk (extract→ESRGAN→encode)
# ─────────────────────────────────────────────────────────────
def test_pipeline_1chunk():
    separator("Pipeline 1 chunk (streaming: extract→ESRGAN→NVENC pipe)")
    import shutil
    from enhance.ffmpeg_utils import extract_frames_to_ram
    from enhance.esrgan import ESRGANEngine
    from enhance.pipeline import _open_nvenc_pipe, _ReorderWriter
    import torch

    out_file = Path("/tmp/test_pipeline.mp4")
    out_file.unlink(missing_ok=True)

    out_w, out_h = W * 2, H * 2
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

    # Stage 2: ESRGAN → NVENC streaming
    esr = ESRGANEngine()
    proc = _open_nvenc_pipe(out_file, out_w, out_h, FPS, gpu=0)
    writer = _ReorderWriter(proc.stdin, n)

    t1 = time.time()
    esr.process_streaming(frames, writer.on_frame)
    writer.flush_remaining()
    try:
        proc.stdin.close()
    except BrokenPipeError:
        pass
    proc.wait()
    t_esr_enc = time.time() - t1

    del frames  # free raw
    gc.collect()
    ram2 = mem_mb()
    print(f"  [Stream] {writer.written} frames  {t_esr_enc:.1f}s  "
          f"{n/t_esr_enc:.1f}fps  RSS={ram2:.0f}MB (raw freed)")

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
# TEST 6: Profiles load
# ─────────────────────────────────────────────────────────────
def test_profiles_load():
    """Smoke test: load all profile types and verify defaults."""
    from enhance.profiles import get_profiles, VISUAL_PROFILES, AUDIO_PROFILES
    
    # Default profiles
    vp, aup, sp, rp = get_profiles()
    assert vp.name == "baseline", f"Expected baseline visual, got {vp.name}"
    assert aup.name == "baseline", f"Expected baseline audio, got {aup.name}"
    assert sp.name == "baseline", f"Expected baseline scheduler, got {sp.name}"
    assert rp.name == "baseline", f"Expected baseline rife backend, got {rp.name}"
    
    # All registered profiles should load
    for name in VISUAL_PROFILES:
        vp, _, _, _ = get_profiles(visual=name)
        assert vp.name == name
    for name in AUDIO_PROFILES:
        _, aup, _, _ = get_profiles(audio=name)
        assert aup.name == name
    
    print("  [OK] All profiles loaded successfully")


# ─────────────────────────────────────────────────────────────
# TEST 7: Model registry
# ─────────────────────────────────────────────────────────────
def test_model_registry():
    """Smoke test: model registry lists known models."""
    from enhance.models import ModelRegistry
    
    registry = ModelRegistry()
    models = registry.list_models()
    assert len(models) >= 2, f"Expected >= 2 models, got {len(models)}"
    
    keys = [m.key for m in models]
    assert "anime_baseline" in keys, "Missing anime_baseline model"
    assert "real_x2" in keys, "Missing real_x2 model"
    
    print("  [OK] Model registry: {len(models)} models registered")


# ─────────────────────────────────────────────────────────────
# TEST 8: Audio filter syntax
# ─────────────────────────────────────────────────────────────
def test_audio_filter_syntax():
    """Smoke test: all audio profile filter chains are valid ffmpeg syntax."""
    from enhance.profiles import AUDIO_PROFILES
    import subprocess
    
    for name, profile in AUDIO_PROFILES.items():
        # Use ffmpeg's filter validation: -af <chain> with null input
        cmd = [
            "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
            "-t", "0.1", "-af", profile.filter_chain,
            "-f", "null", "-",
            "-loglevel", "error",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [FAIL] Audio profile '{name}': {result.stderr.strip()}")
            raise RuntimeError(f"Invalid audio filter for profile '{name}'")
        print(f"  [OK] Audio profile '{name}' filter chain valid")


# ─────────────────────────────────────────────────────────────
# TEST 9: Scheduler wrapping
# ─────────────────────────────────────────────────────────────
def test_scheduler_wrapping():
    """Smoke test: scheduler wrap_subprocess produces valid command prefixes."""
    from enhance.profiles import SchedulerProfile
    from enhance.scheduler import apply_scheduler_profile, wrap_subprocess
    
    # Baseline — no wrapping
    apply_scheduler_profile(SchedulerProfile(name="baseline"))
    cmd = wrap_subprocess(["echo", "test"], role="ffmpeg")
    assert cmd == ["echo", "test"], f"Baseline should not wrap: {cmd}"
    
    # split_l3_a — should add taskset
    apply_scheduler_profile(SchedulerProfile(
        name="test_l3",
        cpuset_ffmpeg="0-7,16-23",
        cpuset_audio="0-7,16-23",
        cpuset_python="8-15,24-31",
    ))
    cmd = wrap_subprocess(["echo", "test"], role="ffmpeg")
    assert "taskset" in cmd, f"Expected taskset in wrapped cmd: {cmd}"
    assert "0-7,16-23" in cmd, f"Expected cpuset in cmd: {cmd}"
    
    # Reset to baseline
    apply_scheduler_profile(SchedulerProfile(name="baseline"))
    print("  [OK] Scheduler wrapping works correctly")


# ─────────────────────────────────────────────────────────────
# TEST 10: RIFE backend creation
# ─────────────────────────────────────────────────────────────
def test_rife_backend_creation():
    """Smoke test: RIFE backend factory creates correct backend type."""
    from enhance.rife_backend import create_backend, NCNNBackend, TorchBackend
    from enhance.profiles import RIFEBackendProfile
    
    # Default — ncnn
    backend = create_backend()
    assert isinstance(backend, NCNNBackend), f"Expected NCNNBackend, got {type(backend)}"
    assert backend.name() == "ncnn"
    
    # Explicit ncnn
    backend = create_backend(RIFEBackendProfile(name="test", backend="ncnn"))
    assert isinstance(backend, NCNNBackend)
    
    # Torch — should create but methods raise NotImplementedError
    backend = create_backend(RIFEBackendProfile(name="test", backend="torch"))
    assert isinstance(backend, TorchBackend)
    try:
        backend.interpolate_sync(None, None)
        raise AssertionError("TorchBackend should raise NotImplementedError")
    except NotImplementedError:
        pass
    
    print("  [OK] RIFE backend factory works correctly")


# ─────────────────────────────────────────────────────────────
# TEST 11: Metrics new fields
# ─────────────────────────────────────────────────────────────
def test_metrics_new_fields():
    """Smoke test: new metric fields are present in chunk_metrics schema."""
    from enhance.pipeline import _MetricsStore
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        ms = _MetricsStore(Path(tmpdir))
        ms.update(0, 
                  rife_spawn_seconds=0.1,
                  rife_compute_seconds=1.0,
                  rife_drain_seconds=0.2,
                  rife_cleanup_seconds=0.05,
                  visual_profile="baseline",
                  audio_profile="baseline",
                  scheduler_profile="baseline",
                  rife_backend="ncnn")
        snap = ms.snapshot(0)
        assert snap["visual_profile"] == "baseline"
        assert snap["rife_spawn_seconds"] == 0.1
        ms.emit(0)
        metrics_file = Path(tmpdir) / "chunk_metrics.jsonl"
        assert metrics_file.exists(), "chunk_metrics.jsonl not created"
        import json
        data = json.loads(metrics_file.read_text().strip())
        assert data["rife_backend"] == "ncnn"
    
    print("  [OK] New metric fields work correctly")


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not SRC.exists():
        print(f"[!] Video not found: {SRC}")
        sys.exit(1)

    tests = [
        ("extract", test_extract),
        ("esrgan", test_esrgan),
        ("encode", test_encode),
        ("streaming", test_streaming),
        ("pipeline", test_pipeline_1chunk),
        ("profiles", test_profiles_load),
        ("models", test_model_registry),
        ("audio_filter", test_audio_filter_syntax),
        ("scheduler", test_scheduler_wrapping),
        ("rife_backend", test_rife_backend_creation),
        ("metrics", test_metrics_new_fields),
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
