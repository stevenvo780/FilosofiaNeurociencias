"""ffmpeg helpers: probe, extract frames, audio enhancement and merge."""
import json
import re
import subprocess
import numpy as np
from pathlib import Path
from . import config as C


def probe(path: Path):
    """Return (duration_s, fps, width, height)."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True)
    info = json.loads(r.stdout)
    dur = float(info["format"]["duration"])
    for s in info["streams"]:
        if s["codec_type"] == "video":
            num, den = s["r_frame_rate"].split("/")
            return dur, float(num) / float(den), int(s["width"]), int(s["height"])
    raise RuntimeError("no video stream found")


def has_audio_stream(path: Path) -> bool:
    """Return whether a media file contains at least one audio stream."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True)
    return bool(r.stdout.strip())


def resolve_audio_source(src: Path, explicit: Path | None = None) -> Path | None:
    """Return the best audio source for a video, preferring sidecar m4a files."""
    if explicit is not None:
        return explicit if explicit.exists() else None

    stem = src.stem
    parent = src.parent
    candidates = [
        parent / f"{stem}.m4a",
        parent / f"{re.sub(r'_(\d+)x(\d+)$', '', stem)}.m4a",
        parent / f"{re.sub(r'_gallery(_(\d+)x(\d+))?$', '', stem)}.m4a",
    ]
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate

    return src if has_audio_stream(src) else None


def enhance_audio(src: Path, dst: Path,
                  start: float = 0.0,
                  duration: float | None = None,
                  audio_profile=None):
    """Enhance audio quality using configurable filter chain from profile."""
    if dst.exists() and dst.stat().st_size > 1024:
        return

    # Resolve audio parameters from profile or config fallback
    if audio_profile is not None:
        af = audio_profile.filter_chain
        codec = audio_profile.codec
        bitrate = audio_profile.bitrate
        sample_rate = str(audio_profile.sample_rate)
        threads = str(audio_profile.threads)
    else:
        af = C.AUDIO_FILTER
        codec = C.AUDIO_CODEC
        bitrate = C.AUDIO_BITRATE
        sample_rate = "48000"
        threads = C.AUDIO_THREADS

    dst.parent.mkdir(parents=True, exist_ok=True)

    # Optionally wrap command with scheduler affinity
    base_cmd = ["ffmpeg", "-y"]
    try:
        from .scheduler import wrap_subprocess
        base_cmd = wrap_subprocess(base_cmd, role="audio")
    except ImportError:
        pass

    cmd = list(base_cmd)
    if start > 0:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(src)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += [
        "-vn",
        "-map", "0:a:0",
        "-af", af,
        "-c:a", codec,
        "-b:a", bitrate,
        "-ar", sample_rate,
        "-threads", threads,
        "-movflags", "+faststart",
        str(dst),
        "-loglevel", "warning",
    ]
    subprocess.run(cmd, check=True)


def extract_frames_to_ram(src: Path, start: float, dur: float,
                          w: int, h: int) -> list[np.ndarray]:
    """Extract frames via ffmpeg pipe → list of numpy arrays in RAM.
    No disk I/O at all. CPU decode remains the default because it measured
    faster than NVDEC for this rawvideo path on this machine."""
    cmd = ["ffmpeg"]
    if C.ENABLE_NVDEC:
        cmd += ["-hwaccel", "cuda", "-c:v", "h264_cuvid"]
    cmd += [
        "-ss", str(start), "-i", str(src), "-t", str(dur),
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-threads", str(C.EXTRACT_THREADS),
        "-v", "error", "-"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            bufsize=w * h * 3 * 4)
    frame_bytes = w * h * 3
    frames = []
    while True:
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy())
    proc.wait()
    return frames


def extract_frames(src: Path, start: float, dur: float,
                   out_dir: Path, fps: float) -> int:
    """Extract frames to PNGs (fallback for RIFE which needs files)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = int(dur * fps)
    existing = len(list(out_dir.glob("*.png")))
    if existing >= max(expected - 2, 1):
        return existing

    cmd = ["ffmpeg", "-y"]
    try:
        from .scheduler import wrap_subprocess
        cmd = wrap_subprocess(cmd, role="ffmpeg")
    except ImportError:
        pass
    if C.ENABLE_NVDEC:
        cmd += ["-hwaccel", "cuda", "-c:v", "h264_cuvid"]
    cmd += [
        "-ss", str(start), "-i", str(src), "-t", str(dur),
        "-pix_fmt", "rgb24", "-threads", str(C.EXTRACT_THREADS),
        str(out_dir / "%08d.png"), "-loglevel", "warning",
    ]
    subprocess.run(cmd, check=True)
    return len(list(out_dir.glob("*.png")))


def nvenc_encode(frames_dir: Path, out_file: Path, fps: float):
    """HEVC encode with NVENC ASIC."""
    if out_file.exists() and out_file.stat().st_size > 1000:
        return
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "%08d.png"),
        "-c:v", "hevc_nvenc", "-gpu", str(C.NVENC_GPUS[0]),
        "-preset", C.NVENC_PRESET,
        "-rc", "vbr", "-cq", C.NVENC_CQ,
        "-b:v", C.NVENC_BITRATE,
        "-maxrate", C.NVENC_MAXRATE,
        "-bufsize", C.NVENC_BUFSIZE,
        "-profile:v", "main10", "-pix_fmt", "p010le",
        "-threads", str(C.ENCODE_THREADS),
        str(out_file), "-loglevel", "warning",
    ]
    subprocess.run(cmd, check=True)


def merge_chunks(work: Path, dst: Path, n_chunks: int,
                 audio_src: Path | None = None):
    """Concatenate chunk videos and optionally mux audio."""
    concat = work / "concat.txt"
    with open(concat, "w") as f:
        for i in range(n_chunks):
            v = work / f"chunk_{i:04d}" / "output.mp4"
            if v.exists():
                f.write(f"file '{v}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat),
    ]
    if audio_src is not None:
        cmd += [
            "-i", str(audio_src),
            "-map", "0:v", "-map", "1:a:0",
            "-shortest",
            "-c:a", "copy",
        ]
    else:
        cmd += ["-map", "0:v"]
    cmd += [
        "-c:v", "copy",
        "-movflags", "+faststart",
        "-threads", C.AUDIO_THREADS,
        str(dst),
        "-loglevel", "warning",
    ]
    subprocess.run(cmd, check=True)
    print(f"\n[DONE] {dst}  ({dst.stat().st_size / 1e9:.2f} GB)")
