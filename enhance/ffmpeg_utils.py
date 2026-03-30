"""ffmpeg helpers: probe, extract frames, NVENC encode, merge."""
import json, subprocess
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


def extract_frames_to_ram(src: Path, start: float, dur: float,
                          w: int, h: int) -> list[np.ndarray]:
    """Extract frames via ffmpeg pipe → list of numpy arrays in RAM.
    No disk I/O at all. Uses software decode (CPU) which is plenty fast."""
    cmd = [
        "ffmpeg",
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
        frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3))
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

    cmd = [
        "ffmpeg", "-y",
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
        "-c:v", "hevc_nvenc", "-gpu", str(C.NVENC_GPU),
        "-preset", "p6", "-tune", "hq",
        "-rc", "vbr", "-cq", "20",
        "-b:v", "12M", "-maxrate", "18M", "-bufsize", "24M",
        "-profile:v", "main10", "-pix_fmt", "yuv420p10le",
        "-threads", str(C.ENCODE_THREADS),
        str(out_file), "-loglevel", "warning",
    ]
    subprocess.run(cmd, check=True)


def merge_chunks(work: Path, src: Path, dst: Path, n_chunks: int):
    """Concatenate chunk videos + best audio into final file."""
    concat = work / "concat.txt"
    with open(concat, "w") as f:
        for i in range(n_chunks):
            v = work / f"chunk_{i:04d}" / "output.mp4"
            if v.exists():
                f.write(f"file '{v}'\n")

    # Prefer the pre-enhanced m4a
    enh_audio = dst.parent / "GMT20260320-130023_Recording_enhanced.m4a"
    if enh_audio.exists():
        a_src, a_codec = str(enh_audio), ["-c:a", "copy"]
    else:
        a_src = str(src)
        a_codec = [
            "-af",
            ("afftdn=nf=-20:nt=w:om=o,"
             "acompressor=threshold=-20dB:ratio=3:attack=5:release=50,"
             "loudnorm=I=-16:TP=-1.5:LRA=11"),
            "-c:a", "aac", "-b:a", "192k",
        ]

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat),
        "-i", a_src,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy",
    ] + a_codec + [
        "-movflags", "+faststart", "-threads", "16",
        str(dst), "-loglevel", "warning",
    ]
    subprocess.run(cmd, check=True)
    print(f"\n[DONE] {dst}  ({dst.stat().st_size / 1e9:.2f} GB)")
