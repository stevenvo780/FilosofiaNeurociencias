#!/usr/bin/env python3
"""Benchmark runner — wraps run.py with full system instrumentation."""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _run(cmd, **kw):
    """Run a command and return CompletedProcess (never raises)."""
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


def _kill_orphans():
    """Kill leftover pipeline processes."""
    for pattern in ("scripts/run.py", "rife-ncnn-vulkan", "ffmpeg.*enhance"):
        _run(f"pkill -9 -f '{pattern}'")
    time.sleep(0.5)


def _clean_tmpfs():
    """Wipe the tmpfs work directory."""
    tmpfs = Path("/tmp/enhance_work")
    if tmpfs.exists():
        shutil.rmtree(tmpfs, ignore_errors=True)
    tmpfs.mkdir(parents=True, exist_ok=True)


# ── Hardware capture ─────────────────────────────────────────────────────────

def _cpu_info() -> dict:
    """Parse /proc/cpuinfo for model and core count."""
    info = {"model": "unknown", "cores": 0}
    try:
        text = Path("/proc/cpuinfo").read_text()
        models = re.findall(r"model name\s*:\s*(.+)", text)
        if models:
            info["model"] = models[0].strip()
        info["cores"] = len(models)
    except Exception:
        pass
    return info


def _mem_info() -> dict:
    """Memory info from free -b."""
    info = {}
    r = _run("free -b")
    if r.returncode == 0:
        for line in r.stdout.splitlines():
            if line.startswith("Mem:"):
                parts = line.split()
                info["total_bytes"] = int(parts[1])
                info["used_bytes"] = int(parts[2])
                info["available_bytes"] = int(parts[6]) if len(parts) > 6 else 0
            elif line.startswith("Swap:"):
                parts = line.split()
                info["swap_total_bytes"] = int(parts[1])
                info["swap_used_bytes"] = int(parts[2])
    return info


def _gpu_info() -> list:
    """Query nvidia-smi for GPU details."""
    fields = (
        "name,memory.total,pcie.link.gen.max,pcie.link.width.max,"
        "pcie.link.gen.current,pcie.link.width.current"
    )
    r = _run(f"nvidia-smi --query-gpu={fields} --format=csv,noheader")
    gpus = []
    if r.returncode == 0:
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            gpus.append({
                "name": parts[0] if len(parts) > 0 else "unknown",
                "memory_total": parts[1] if len(parts) > 1 else "unknown",
                "pcie_gen_max": parts[2] if len(parts) > 2 else "?",
                "pcie_width_max": parts[3] if len(parts) > 3 else "?",
                "pcie_gen_current": parts[4] if len(parts) > 4 else "?",
                "pcie_width_current": parts[5] if len(parts) > 5 else "?",
            })
    return gpus


def _check_pcie_blocked(gpus: list) -> bool:
    """Return True if GPU1 has PCIe width <= 4 (bottleneck indicator)."""
    if len(gpus) < 2:
        return False
    try:
        width = int(gpus[1].get("pcie_width_current", "16"))
        return width <= 4
    except (ValueError, TypeError):
        return False


def capture_hardware(bench_dir: Path) -> dict:
    """Collect and persist hardware snapshot."""
    gpus = _gpu_info()
    hw = {
        "timestamp": datetime.now().isoformat(),
        "cpu": _cpu_info(),
        "memory": _mem_info(),
        "gpus": gpus,
        "pcie_blocked": _check_pcie_blocked(gpus),
    }
    (bench_dir / "hardware.json").write_text(json.dumps(hw, indent=2))
    return hw


# ── Background monitors ─────────────────────────────────────────────────────

class MonitorSet:
    """Manages background system monitor subprocesses."""

    def __init__(self, bench_dir: Path):
        self._procs: list[subprocess.Popen] = []
        self._dir = bench_dir

    def start(self):
        """Launch all monitors."""
        # mpstat
        self._procs.append(subprocess.Popen(
            "mpstat -P ALL 1",
            shell=True,
            stdout=open(self._dir / "mpstat.log", "w"),
            stderr=subprocess.DEVNULL,
        ))
        # iostat
        self._procs.append(subprocess.Popen(
            "iostat -xz 1",
            shell=True,
            stdout=open(self._dir / "iostat.log", "w"),
            stderr=subprocess.DEVNULL,
        ))
        # memory loop
        self._procs.append(subprocess.Popen(
            "while true; do echo \"--- $(date +%H:%M:%S) ---\"; free -h; sleep 2; done",
            shell=True,
            stdout=open(self._dir / "memory.log", "w"),
            stderr=subprocess.DEVNULL,
        ))
        # GPU CSV loop
        gpu_header = "timestamp,index,gpu_util,mem_util,mem_used,mem_total,temp,power,pcie_tx,pcie_rx"
        gpu_query = (
            "timestamp,index,utilization.gpu,utilization.memory,"
            "memory.used,memory.total,temperature.gpu,power.draw,"
            "pcie.link.gen.current,pcie.link.width.current"
        )
        with open(self._dir / "gpu.csv", "w") as f:
            f.write(gpu_header + "\n")
        self._procs.append(subprocess.Popen(
            f"while true; do nvidia-smi --query-gpu={gpu_query} "
            f"--format=csv,noheader,nounits >> {self._dir / 'gpu.csv'}; sleep 1; done",
            shell=True,
            stderr=subprocess.DEVNULL,
        ))

    def stop(self):
        """Terminate all monitors."""
        for p in self._procs:
            try:
                p.send_signal(signal.SIGTERM)
            except Exception:
                pass
        for p in self._procs:
            try:
                p.wait(timeout=3)
            except Exception:
                p.kill()
        self._procs.clear()


# ── Main ─────────────────────────────────────────────────────────────────────

def build_run_cmd(args) -> list[str]:
    """Build the scripts/run.py command list."""
    cmd = [
        sys.executable, str(ROOT / "scripts" / "run.py"),
        str(args.input),
        "--start", str(args.start),
        "--duration", str(args.duration),
        "--chunk", "15",
    ]
    if args.outdir:
        cmd += ["--outdir", args.outdir]
    # Forward profile arguments to run.py
    cmd += [
        "--visual-profile", args.visual_profile,
        "--audio-profile", args.audio_profile,
        "--scheduler-profile", args.scheduler_profile,
        "--rife-backend", args.rife_backend,
    ]
    if args.models_dir:
        cmd += ["--models-dir", args.models_dir]
    return cmd


def parse_args():
    ap = argparse.ArgumentParser(
        description="Benchmark runner — wraps run.py with full system instrumentation.",
    )
    ap.add_argument("input", help="Input video file")
    ap.add_argument("--start", type=float, default=60.0,
                    help="Start offset in seconds (default: 60)")
    ap.add_argument("--duration", type=float, default=60.0,
                    help="Duration in seconds (default: 60)")
    ap.add_argument("--tag", type=str, default="",
                    help="Benchmark tag for identification")
    ap.add_argument("--outdir", type=str, default="/tmp",
                    help="Base output directory (default: /tmp)")
    ap.add_argument("--visual-profile", type=str, default="baseline",
                    help="Visual enhancement profile")
    ap.add_argument("--audio-profile", type=str, default="baseline",
                    help="Audio enhancement profile")
    ap.add_argument("--scheduler-profile", type=str, default="baseline",
                    help="Scheduler profile")
    ap.add_argument("--rife-backend", type=str, default="baseline",
                    help="RIFE interpolation backend")
    ap.add_argument("--quality-only", action="store_true",
                    help="Only run quality evaluation, skip throughput")
    ap.add_argument("--throughput-only", action="store_true",
                    help="Skip quality evaluation")
    ap.add_argument("--nsys", action="store_true",
                    help="Wrap pipeline with nsys profile")
    ap.add_argument("--models-dir", type=str, default=None,
                    help="Path to models directory")
    return ap.parse_args()


def _find_work_dir(outdir: str) -> Path | None:
    """Find the most recent work directory under outdir."""
    out = Path(outdir)
    candidates = sorted(out.glob("work_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _try_perf_stat(pid: int, bench_dir: Path):
    """Attempt to capture perf stat for the pipeline process."""
    if shutil.which("perf") is None:
        return
    try:
        subprocess.run(
            f"perf stat -p {pid} -o {bench_dir / 'perf.stat'} sleep 10",
            shell=True, timeout=15,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def main():
    args = parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag or "run"
    bench_dir = ROOT / "enhanced" / "logs" / f"bench_{tag}_{stamp}"
    bench_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 64}")
    print(f"  BENCHMARK RUNNER — {tag}")
    print(f"  Output: {bench_dir}")
    print(f"{'=' * 64}\n")

    # Step 1 — Kill orphans
    print("[bench] Killing orphan processes …")
    _kill_orphans()

    # Step 2 — Clean tmpfs
    print("[bench] Cleaning /tmp/enhance_work …")
    _clean_tmpfs()

    # Step 3 — Hardware snapshot
    print("[bench] Capturing hardware info …")
    hw = capture_hardware(bench_dir)
    for i, g in enumerate(hw.get("gpus", [])):
        pcie_note = " ⚠ PCIe BLOCKED" if (i == 1 and hw.get("pcie_blocked")) else ""
        print(f"  GPU{i}: {g['name']}  mem={g['memory_total']}  "
              f"PCIe gen{g['pcie_gen_current']} x{g['pcie_width_current']}{pcie_note}")

    # Step 4 — Set environment from profiles
    env_vars = {
        "ENHANCE_VISUAL_PROFILE": args.visual_profile,
        "ENHANCE_AUDIO_PROFILE": args.audio_profile,
        "ENHANCE_SCHEDULER_PROFILE": args.scheduler_profile,
        "ENHANCE_RIFE_BACKEND": args.rife_backend,
        "ENHANCE_ENABLE_JSONL_METRICS": "1",
    }
    if args.models_dir:
        env_vars["ENHANCE_MODELS_DIR"] = args.models_dir
    for k, v in env_vars.items():
        os.environ[k] = v
    print(f"[bench] Environment: {json.dumps(env_vars, indent=2)}")

    # Step 5 — Build command
    cmd = build_run_cmd(args)
    if args.nsys:
        nsys_out = str(bench_dir / "nsys_report")
        cmd = ["nsys", "profile", "--output", nsys_out, "--force-overwrite", "true"] + cmd

    cmd_str = " ".join(str(c) for c in cmd)
    print(f"[bench] Command: {cmd_str}\n")

    # Step 6 — Start monitors
    monitors = MonitorSet(bench_dir)
    monitors.start()
    print("[bench] Monitors started (mpstat, iostat, memory, gpu)")

    # Step 7 — Run pipeline
    t0 = time.time()
    pipeline_log = bench_dir / "pipeline.log"
    with open(pipeline_log, "w") as log_f:
        proc = subprocess.Popen(
            cmd_str, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )

        # Optionally attach perf stat in background
        perf_thread = None
        if not args.throughput_only and proc.pid:
            import threading
            perf_thread = threading.Thread(
                target=_try_perf_stat, args=(proc.pid, bench_dir), daemon=True
            )
            perf_thread.start()

        # Stream output to both stdout and log file
        for line in proc.stdout:
            sys.stdout.write(line)
            log_f.write(line)
        proc.wait()

    wall_time = time.time() - t0
    exit_code = proc.returncode

    # Step 8 — Stop monitors
    monitors.stop()
    print(f"\n[bench] Pipeline finished in {wall_time:.1f}s  (exit={exit_code})")

    # Step 9 — Collect chunk_metrics.jsonl
    work_dir = _find_work_dir(args.outdir)
    metrics_copied = False
    if work_dir:
        jsonl_src = work_dir / "chunk_metrics.jsonl"
        if jsonl_src.exists():
            shutil.copy2(jsonl_src, bench_dir / "chunk_metrics.jsonl")
            metrics_copied = True
            print(f"[bench] Copied chunk_metrics.jsonl ({jsonl_src.stat().st_size} bytes)")

    # Step 10 — Summary
    print(f"\n{'=' * 64}")
    print(f"  BENCHMARK SUMMARY — {tag}")
    print(f"{'=' * 64}")
    print(f"  Wall time:       {wall_time:.1f}s  ({wall_time / 60:.2f} min)")

    process_dur = args.duration
    throughput_ratio = process_dur / wall_time if wall_time > 0 else 0.0
    print(f"  Throughput:      {throughput_ratio:.3f}x realtime")

    # Estimate full video duration if we have it
    try:
        r = _run(f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {args.input}")
        full_dur = float(r.stdout.strip())
        eta_seconds = full_dur / throughput_ratio if throughput_ratio > 0 else float("inf")
        print(f"  Full video:      {full_dur:.0f}s  ({full_dur / 3600:.1f}h)")
        print(f"  ETA full:        {eta_seconds / 3600:.1f}h")
    except Exception:
        pass

    print(f"  Metrics:         {'OK' if metrics_copied else 'NOT FOUND'}")
    print(f"  PCIe blocked:    {hw.get('pcie_blocked', False)}")
    print(f"  Logs:            {bench_dir}")
    print(f"{'=' * 64}\n")

    # Save summary JSON
    summary = {
        "tag": tag,
        "stamp": stamp,
        "wall_time_s": round(wall_time, 2),
        "process_duration_s": process_dur,
        "throughput_ratio": round(throughput_ratio, 4),
        "exit_code": exit_code,
        "pcie_blocked": hw.get("pcie_blocked", False),
        "metrics_collected": metrics_copied,
        "profiles": {
            "visual": args.visual_profile,
            "audio": args.audio_profile,
            "scheduler": args.scheduler_profile,
            "rife_backend": args.rife_backend,
        },
    }
    (bench_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
