#!/usr/bin/env python3
"""Acceptance gate — validates pipeline output meets quality and throughput criteria."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# ── Thresholds ───────────────────────────────────────────────────────────────

MIN_EFFECTIVE_FPS = 20.0
MAX_CHUNK_SECONDS = 37.5
MIN_THROUGHPUT_RATIO = 0.40
MAX_SWAP_GROWTH_BYTES = 2 * 1024**3  # 2 GB
EXPECTED_RESOLUTION = (4480, 2520)
EXPECTED_FPS_RANGE = (48.0, 52.0)  # ~50 fps tolerance


# ── Helpers ──────────────────────────────────────────────────────────────────

def _run(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def _load_jsonl(path: Path) -> list[dict]:
    """Load newline-delimited JSON."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


# ── Criterion checkers ───────────────────────────────────────────────────────

class Result:
    """Single acceptance criterion result."""

    def __init__(self, name: str, passed: bool, value, threshold, warning: bool = False, detail: str = ""):
        self.name = name
        self.passed = passed
        self.value = value
        self.threshold = threshold
        self.warning = warning  # True = non-fatal unless --strict
        self.detail = detail

    @property
    def status(self) -> str:
        if self.passed:
            return "PASS"
        if self.warning:
            return "WARN"
        return "FAIL"


def check_effective_fps(metrics: list[dict]) -> Result:
    """Average effective_fps across chunks must meet minimum."""
    fps_values = [m["effective_fps"] for m in metrics if "effective_fps" in m]
    avg = sum(fps_values) / len(fps_values) if fps_values else 0.0
    return Result(
        name="effective_fps",
        passed=avg >= MIN_EFFECTIVE_FPS,
        value=round(avg, 2),
        threshold=f">= {MIN_EFFECTIVE_FPS}",
    )


def check_chunk_time(metrics: list[dict]) -> Result:
    """Average total_seconds per chunk must be under threshold."""
    times = [m["total_seconds"] for m in metrics if "total_seconds" in m]
    avg = sum(times) / len(times) if times else float("inf")
    return Result(
        name="avg_chunk_seconds",
        passed=avg <= MAX_CHUNK_SECONDS,
        value=round(avg, 2),
        threshold=f"<= {MAX_CHUNK_SECONDS}",
    )


def check_throughput(metrics: list[dict], wall_time: float | None = None) -> Result:
    """Throughput ratio = (chunk_seconds * n_chunks) / wall_time."""
    chunk_secs = [m.get("chunk_seconds", m.get("duration", 0)) for m in metrics]
    total_content = sum(chunk_secs)
    n = len(metrics)

    # Try to get wall time from summary if not given
    if wall_time is None:
        total_proc = sum(m.get("total_seconds", 0) for m in metrics)
        wall_time = total_proc if total_proc > 0 else 1.0

    ratio = total_content / wall_time if wall_time > 0 else 0.0
    return Result(
        name="throughput_ratio",
        passed=ratio >= MIN_THROUGHPUT_RATIO,
        value=round(ratio, 4),
        threshold=f">= {MIN_THROUGHPUT_RATIO}",
        detail=f"{total_content:.1f}s content / {wall_time:.1f}s wall ({n} chunks)",
    )


def check_swap_stability(bench_dir: Path) -> Result:
    """Swap usage should not grow more than 2 GB across the run."""
    mem_log = bench_dir / "memory.log"
    if not mem_log.exists():
        return Result("swap_stability", True, "N/A", "log missing", warning=True)

    swap_values = []
    text = mem_log.read_text()
    for line in text.splitlines():
        if line.startswith("Swap:"):
            parts = line.split()
            if len(parts) >= 3:
                swap_values.append(_parse_human_bytes(parts[2]))

    if len(swap_values) < 2:
        return Result("swap_stability", True, "insufficient data", "<= 2GB growth", warning=True)

    # Check growth between first and last samples
    growth = swap_values[-1] - swap_values[0]
    growth_gb = growth / 1024**3
    return Result(
        name="swap_stability",
        passed=growth <= MAX_SWAP_GROWTH_BYTES,
        value=f"{growth_gb:.2f} GB growth",
        threshold="<= 2.0 GB growth",
    )


def _parse_human_bytes(s: str) -> int:
    """Parse human-readable byte strings like '1.5Gi', '512Mi', '0B'."""
    s = s.strip()
    if s in ("0B", "0"):
        return 0
    multipliers = {
        "B": 1, "Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "Ti": 1024**4,
        "K": 1000, "M": 1000**2, "G": 1000**3, "T": 1000**4,
    }
    for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if s.endswith(suffix):
            try:
                return int(float(s[:-len(suffix)]) * mult)
            except ValueError:
                return 0
    try:
        return int(s)
    except ValueError:
        return 0


def check_zombie_processes() -> Result:
    """No orphan rife-ncnn or ffmpeg enhance processes should be running."""
    r = _run("pgrep -f 'rife-ncnn|ffmpeg.*enhance'")
    pids = [p.strip() for p in r.stdout.strip().splitlines() if p.strip()]
    return Result(
        name="zombie_processes",
        passed=len(pids) == 0,
        value=f"{len(pids)} found" if pids else "none",
        threshold="0 orphans",
        detail=f"PIDs: {', '.join(pids)}" if pids else "",
    )


def check_video(video_path: str, expected_duration: float | None = None) -> list[Result]:
    """Validate output video with ffprobe."""
    results = []
    vp = Path(video_path)

    # File exists and size > 0
    if not vp.exists() or vp.stat().st_size == 0:
        results.append(Result("video_exists", False, "missing or empty", "> 0 bytes"))
        return results
    results.append(Result("video_exists", True, f"{vp.stat().st_size / 1024**2:.1f} MB", "> 0 bytes"))

    # Probe video stream
    probe_cmd = (
        f"ffprobe -v error -select_streams v:0 "
        f"-show_entries stream=width,height,r_frame_rate,duration "
        f"-of json {video_path}"
    )
    r = _run(probe_cmd)
    if r.returncode != 0:
        results.append(Result("ffprobe", False, "probe failed", "success", detail=r.stderr))
        return results

    try:
        info = json.loads(r.stdout)
        stream = info["streams"][0]
    except (json.JSONDecodeError, KeyError, IndexError):
        results.append(Result("ffprobe", False, "parse failed", "valid JSON"))
        return results

    # Resolution
    w = int(stream.get("width", 0))
    h = int(stream.get("height", 0))
    exp_w, exp_h = EXPECTED_RESOLUTION
    results.append(Result(
        name="resolution",
        passed=(w == exp_w and h == exp_h),
        value=f"{w}x{h}",
        threshold=f"{exp_w}x{exp_h}",
    ))

    # FPS
    fps_str = stream.get("r_frame_rate", "0/1")
    try:
        num, den = fps_str.split("/")
        fps = float(num) / float(den)
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    lo, hi = EXPECTED_FPS_RANGE
    results.append(Result(
        name="fps",
        passed=(lo <= fps <= hi),
        value=round(fps, 2),
        threshold=f"{lo}-{hi}",
    ))

    # Duration
    dur = float(stream.get("duration", 0))
    if dur == 0:
        # Try container-level duration
        r2 = _run(
            f"ffprobe -v error -show_entries format=duration "
            f"-of default=noprint_wrappers=1:nokey=1 {video_path}"
        )
        try:
            dur = float(r2.stdout.strip())
        except ValueError:
            dur = 0.0

    if expected_duration is not None and expected_duration > 0:
        tolerance = max(expected_duration * 0.05, 1.0)  # 5% or 1s
        results.append(Result(
            name="duration",
            passed=abs(dur - expected_duration) <= tolerance,
            value=f"{dur:.1f}s",
            threshold=f"{expected_duration:.1f}s ± {tolerance:.1f}s",
        ))
    else:
        results.append(Result(
            name="duration",
            passed=dur > 0,
            value=f"{dur:.1f}s",
            threshold="> 0s",
            warning=True,
            detail="No expected duration provided for strict check",
        ))

    return results


def check_pcie(bench_dir: Path) -> Result:
    """Check hardware.json for PCIe bottleneck warning."""
    hw_path = bench_dir / "hardware.json"
    if not hw_path.exists():
        return Result("pcie_check", True, "no hardware.json", "no bottleneck", warning=True)

    try:
        hw = json.loads(hw_path.read_text())
    except json.JSONDecodeError:
        return Result("pcie_check", True, "invalid JSON", "no bottleneck", warning=True)

    blocked = hw.get("pcie_blocked", False)
    detail = ""
    if blocked and len(hw.get("gpus", [])) >= 2:
        gpu1 = hw["gpus"][1]
        detail = (
            f"GPU1 ({gpu1.get('name', '?')}) PCIe gen{gpu1.get('pcie_gen_current', '?')} "
            f"x{gpu1.get('pcie_width_current', '?')}"
        )
    return Result(
        name="pcie_check",
        passed=not blocked,
        value="blocked" if blocked else "OK",
        threshold="no bottleneck",
        warning=True,  # PCIe is a warning, not a hard failure
        detail=detail,
    )


# ── Gate runner ──────────────────────────────────────────────────────────────

def run_gate(bench_dir: Path, video: str | None, strict: bool) -> int:
    """Execute all acceptance checks and return exit code."""
    results: list[Result] = []

    # Load metrics
    jsonl_path = bench_dir / "chunk_metrics.jsonl"
    if jsonl_path.exists():
        metrics = _load_jsonl(jsonl_path)
    else:
        metrics = []
        results.append(Result("metrics_file", False, "not found", "exists"))

    # Load summary for wall time
    wall_time = None
    summary_path = bench_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            wall_time = summary.get("wall_time_s")
        except Exception:
            pass

    # Core performance checks
    if metrics:
        results.append(check_effective_fps(metrics))
        results.append(check_chunk_time(metrics))
        results.append(check_throughput(metrics, wall_time))

    # System health
    results.append(check_swap_stability(bench_dir))
    results.append(check_zombie_processes())

    # Video validation
    if video:
        results.extend(check_video(video))

    # PCIe check
    results.append(check_pcie(bench_dir))

    # ── Print results table ──────────────────────────────────────────────
    col_w = max(len(r.name) for r in results) + 2
    print(f"\n{'=' * 72}")
    print(f"  ACCEPTANCE GATE — {bench_dir.name}")
    print(f"{'=' * 72}")
    print(f"  {'Criterion':<{col_w}} {'Status':<8} {'Value':<24} {'Threshold'}")
    print(f"  {'-' * col_w} {'-' * 7} {'-' * 23} {'-' * 20}")

    for r in results:
        icon = "✅" if r.passed else ("⚠️ " if r.warning else "❌")
        print(f"  {r.name:<{col_w}} {icon} {r.status:<5} {str(r.value):<24} {r.threshold}")
        if r.detail:
            print(f"  {'':<{col_w}}        ↳ {r.detail}")

    # ── Determine overall result ─────────────────────────────────────────
    hard_failures = [r for r in results if not r.passed and not r.warning]
    warnings = [r for r in results if not r.passed and r.warning]

    print(f"\n  Hard failures: {len(hard_failures)}  |  Warnings: {len(warnings)}")

    if strict:
        all_pass = len(hard_failures) == 0 and len(warnings) == 0
    else:
        all_pass = len(hard_failures) == 0

    if all_pass:
        print(f"  ✅ GATE PASSED {'(strict)' if strict else ''}")
    else:
        print(f"  ❌ GATE FAILED {'(strict)' if strict else ''}")
        if hard_failures:
            for r in hard_failures:
                print(f"     • {r.name}: {r.value} (need {r.threshold})")
        if strict and warnings:
            for r in warnings:
                print(f"     • {r.name}: {r.value} (need {r.threshold}) [warning]")

    print(f"{'=' * 72}\n")

    return 0 if all_pass else 1


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Acceptance gate — validates pipeline output meets quality and throughput criteria.",
    )
    ap.add_argument("bench_dir", help="Path to benchmark log directory")
    ap.add_argument("--video", type=str, default=None,
                    help="Path to output video for validation")
    ap.add_argument("--strict", action="store_true",
                    help="Fail on any warning (not just hard failures)")
    return ap.parse_args()


def main():
    args = parse_args()
    bench_dir = Path(args.bench_dir).resolve()

    if not bench_dir.exists():
        print(f"[!] Benchmark directory not found: {bench_dir}")
        sys.exit(1)

    exit_code = run_gate(bench_dir, args.video, args.strict)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
