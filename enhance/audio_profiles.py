"""Audio profile utilities for A/B benchmarking.

Provides helpers to render the same audio slice through multiple ffmpeg
filter-chain profiles and compare the resulting files.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

from enhance.profiles import AUDIO_PROFILES


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_audio_filter(profile_name: str) -> str:
    """Return the ffmpeg ``-af`` filter chain for *profile_name*.

    Raises ``KeyError`` if the profile is not registered in
    ``enhance.profiles.AUDIO_PROFILES``.
    """
    try:
        profile = AUDIO_PROFILES[profile_name]
    except KeyError:
        raise KeyError(
            f"Unknown audio profile '{profile_name}'. "
            f"Available: {sorted(AUDIO_PROFILES)}"
        )
    return profile.filter_chain


def render_audio_ab(
    src: Path,
    output_dir: Path,
    start: float,
    duration: float,
    profiles: list[str] | None = None,
    thread_sweep: list[int] | None = None,
) -> dict:
    """Render the same audio slice with each profile × thread count.

    Parameters
    ----------
    src:
        Input video or audio file.
    output_dir:
        Directory where rendered files are written.
    start:
        Seek position in seconds.
    duration:
        Slice length in seconds.
    profiles:
        List of audio profile names.  Defaults to the four built-in ones.
    thread_sweep:
        List of thread counts to benchmark.  Defaults to ``[8, 16, 24]``.

    Returns
    -------
    dict
        ``{"results": [{"profile", "threads", "wall_seconds", "hash",
        "output"}, …]}``
    """
    if profiles is None:
        profiles = ["baseline", "conservative", "voice", "natural"]
    if thread_sweep is None:
        thread_sweep = [8, 16, 24]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    for profile_name in profiles:
        filter_chain = get_audio_filter(profile_name)
        ap = AUDIO_PROFILES[profile_name]

        for threads in thread_sweep:
            out_path = output_dir / f"audio_{profile_name}_{threads}t.m4a"

            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i", str(src),
                "-t", str(duration),
                "-vn",
                "-map", "0:a:0",
                "-af", filter_chain,
                "-c:a", ap.codec,
                "-b:a", ap.bitrate,
                "-ar", str(ap.sample_rate),
                "-threads", str(threads),
                "-movflags", "+faststart",
                str(out_path),
                "-loglevel", "warning",
            ]

            t0 = time.monotonic()
            subprocess.run(cmd, check=True)
            wall = time.monotonic() - t0

            file_hash = _hash_head(out_path)

            results.append({
                "profile": profile_name,
                "threads": threads,
                "wall_seconds": round(wall, 3),
                "hash": file_hash,
                "output": str(out_path),
            })

    return {"results": results}


def compare_audio_files(files: list[Path]) -> dict:
    """Basic comparison of audio files: size, duration, bitrate.

    Parameters
    ----------
    files:
        Paths to audio/video files to compare.

    Returns
    -------
    dict
        ``{"files": [{"path", "size_bytes", "duration_s", "bitrate_kbps"}, …]}``
    """
    info_list: List[Dict[str, Any]] = []

    for fpath in files:
        fpath = Path(fpath)
        size = fpath.stat().st_size if fpath.exists() else 0
        duration, bitrate = _probe_audio(fpath)
        info_list.append({
            "path": str(fpath),
            "size_bytes": size,
            "duration_s": duration,
            "bitrate_kbps": bitrate,
        })

    return {"files": info_list}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hash_head(path: Path, head_bytes: int = 65536) -> str:
    """Return SHA-256 hex digest of the first *head_bytes* of *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(head_bytes))
    return h.hexdigest()


def _probe_audio(path: Path) -> tuple[float, float]:
    """Return ``(duration_seconds, bitrate_kbps)`` via ffprobe."""
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=duration,bit_rate",
                "-show_entries", "format=duration,bit_rate",
                "-print_format", "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        info = json.loads(r.stdout)

        # Try stream-level first, fall back to format-level.
        dur = 0.0
        br = 0.0
        for stream in info.get("streams", []):
            if stream.get("duration"):
                dur = float(stream["duration"])
            if stream.get("bit_rate"):
                br = float(stream["bit_rate"]) / 1000.0
        if dur == 0.0 and info.get("format", {}).get("duration"):
            dur = float(info["format"]["duration"])
        if br == 0.0 and info.get("format", {}).get("bit_rate"):
            br = float(info["format"]["bit_rate"]) / 1000.0

        return round(dur, 3), round(br, 1)
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
        return 0.0, 0.0
