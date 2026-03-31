#!/usr/bin/env python3
"""Audio A/B benchmark — compare audio profiles on same slice."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is importable when running as a standalone script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from enhance.audio_profiles import compare_audio_files, render_audio_ab


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the same audio slice through multiple profiles and compare.",
    )
    parser.add_argument(
        "input",
        help="Path to input video or audio file.",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=60.0,
        help="Seek position in seconds (default: 60.0).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Slice length in seconds (default: 60.0).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="enhanced/audio_bench",
        help="Output directory (default: enhanced/audio_bench).",
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default="baseline,conservative,voice,natural",
        help="Comma-separated audio profile names.",
    )
    parser.add_argument(
        "--threads",
        type=str,
        default="8,16,24",
        help="Comma-separated thread counts to sweep.",
    )
    return parser.parse_args(argv)


def _print_summary(render_results: dict, compare_results: dict) -> None:
    """Pretty-print a summary table to stdout."""
    results = render_results.get("results", [])
    files_info = {f["path"]: f for f in compare_results.get("files", [])}

    # Header
    header = f"{'Profile':<16} {'Threads':>7} {'Wall(s)':>8} {'Size(KB)':>10} {'Duration':>10} {'Bitrate':>10} {'Hash(8)':>10}"
    print()
    print("=" * len(header))
    print("  Audio A/B Benchmark Results")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        finfo = files_info.get(r["output"], {})
        size_kb = finfo.get("size_bytes", 0) / 1024
        dur = finfo.get("duration_s", 0.0)
        br = finfo.get("bitrate_kbps", 0.0)
        short_hash = r["hash"][:8]

        print(
            f"{r['profile']:<16} {r['threads']:>7} {r['wall_seconds']:>8.2f} "
            f"{size_kb:>10.1f} {dur:>10.2f} {br:>10.1f} {short_hash:>10}"
        )

    print("=" * len(header))
    print()


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    src = Path(args.input).resolve()
    if not src.exists():
        print(f"Error: input file not found: {src}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.outdir)
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    thread_sweep = [int(t.strip()) for t in args.threads.split(",") if t.strip()]

    print(f"Input:    {src}")
    print(f"Output:   {output_dir}")
    print(f"Profiles: {profiles}")
    print(f"Threads:  {thread_sweep}")
    print(f"Slice:    {args.start}s + {args.duration}s")
    print()

    # Step 1: Render all profile × thread combinations.
    render_results = render_audio_ab(
        src=src,
        output_dir=output_dir,
        start=args.start,
        duration=args.duration,
        profiles=profiles,
        thread_sweep=thread_sweep,
    )

    # Step 2: Compare the resulting files.
    output_files = [Path(r["output"]) for r in render_results["results"]]
    compare_results = compare_audio_files(output_files)

    # Step 3: Print summary.
    _print_summary(render_results, compare_results)

    # Step 4: Save combined results as JSON.
    combined = {
        "render": render_results,
        "compare": compare_results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "audio_ab_results.json"
    with open(json_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Results saved to {json_path}")


if __name__ == "__main__":
    main()
