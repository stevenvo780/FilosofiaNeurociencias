#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import torch
import torchaudio
import whisper
from tqdm import tqdm
from resemble_enhance.denoiser.inference import denoise

AUDIO_SR = 48_000
AUDIO_CHANNELS = 1
AUDIO_FILTER = "highpass=f=70,loudnorm=I=-16:LRA=11:TP=-1.5,alimiter=limit=0.95"


@dataclass
class VideoResult:
    input_video: str
    enhanced_audio_wav: str
    subtitles_es: str
    subtitles_en: str
    output_mkv: str
    duration_seconds: float
    elapsed_seconds: float
    status: str
    error: str = ""


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    printable = " ".join(shlex.quote(part) for part in cmd)
    log(f"$ {printable}")
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def ffprobe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    return float(result.stdout.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enhance audio and generate Spanish/English subtitles for a batch of talks."
    )
    parser.add_argument("--input-glob", default="output/charla*.mp4")
    parser.add_argument("--work-dir", default="work/audio_subs")
    parser.add_argument("--deliver-dir", default="deliverables")
    parser.add_argument("--chunk-seconds", type=int, default=300)
    parser.add_argument("--audio-device", default="cuda:0")
    parser.add_argument("--whisper-device", default="cuda")
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--keep-intermediates", action="store_true")
    return parser.parse_args()


def discover_inputs(pattern: str) -> list[Path]:
    matches = sorted(Path().glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No se encontraron videos con el patrón: {pattern}")
    return matches


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_audio(input_video: Path, wav_path: Path) -> None:
    if wav_path.exists() and wav_path.stat().st_size > 10_000:
        log(f"Audio extraído ya existe: {wav_path}")
        return
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(AUDIO_SR),
            "-ac",
            str(AUDIO_CHANNELS),
            str(wav_path),
        ]
    )


def split_audio(input_wav: Path, chunks_dir: Path, chunk_seconds: int) -> list[Path]:
    existing = sorted(chunks_dir.glob("chunk_*.wav"))
    if existing:
        log(f"Chunks ya existen: {len(existing)}")
        return existing

    ensure_dir(chunks_dir)
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_wav),
            "-f",
            "segment",
            "-segment_time",
            str(chunk_seconds),
            "-c:a",
            "pcm_s16le",
            "-ar",
            str(AUDIO_SR),
            "-ac",
            str(AUDIO_CHANNELS),
            str(chunks_dir / "chunk_%04d.wav"),
        ]
    )

    chunks = sorted(chunks_dir.glob("chunk_*.wav"))
    if not chunks:
        raise RuntimeError(f"No se generaron chunks en {chunks_dir}")
    return chunks


def denoise_chunk(chunk_path: Path, output_path: Path, device: str) -> None:
    if output_path.exists() and output_path.stat().st_size > 10_000:
        return

    wav, sr = torchaudio.load(str(chunk_path))
    mono = wav.mean(dim=0)

    with torch.inference_mode():
        denoised, new_sr = denoise(mono.to(device), sr, device=device, run_dir=None)

    denoised = denoised.detach().cpu()
    if new_sr != AUDIO_SR:
        denoised = torchaudio.transforms.Resample(new_sr, AUDIO_SR)(denoised)

    torchaudio.save(str(output_path), denoised.unsqueeze(0), AUDIO_SR)

    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def enhance_audio(input_video: Path, work_dir: Path, device: str, chunk_seconds: int, keep_intermediates: bool) -> Path:
    raw_wav = work_dir / "audio_raw.wav"
    chunks_dir = work_dir / "chunks"
    done_dir = work_dir / "chunks_done"
    merged_wav = work_dir / "audio_denoised.wav"
    normalized_wav = work_dir / "audio_final.wav"

    extract_audio(input_video, raw_wav)
    chunks = split_audio(raw_wav, chunks_dir, chunk_seconds)
    ensure_dir(done_dir)

    for chunk in tqdm(chunks, desc=f"Denoise {input_video.stem}"):
        out_chunk = done_dir / chunk.name
        denoise_chunk(chunk, out_chunk, device)

    concat_file = work_dir / "concat_audio.txt"
    concat_file.write_text(
        "\n".join(f"file '{chunk.as_posix()}'" for chunk in sorted(done_dir.glob("chunk_*.wav"))) + "\n",
        encoding="utf-8",
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c:a",
            "pcm_s16le",
            "-ar",
            str(AUDIO_SR),
            "-ac",
            str(AUDIO_CHANNELS),
            str(merged_wav),
        ]
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(merged_wav),
            "-af",
            AUDIO_FILTER,
            "-c:a",
            "pcm_s16le",
            "-ar",
            str(AUDIO_SR),
            "-ac",
            str(AUDIO_CHANNELS),
            str(normalized_wav),
        ]
    )

    if not keep_intermediates:
        concat_file.unlink(missing_ok=True)

    return normalized_wav


def format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def write_srt(segments: Iterable[dict], output_path: Path) -> None:
    lines: list[str] = []
    for idx, segment in enumerate(segments, start=1):
        text = " ".join(str(segment["text"]).strip().split())
        if not text:
            continue
        lines.extend(
            [
                str(idx),
                f"{format_srt_timestamp(segment['start'])} --> {format_srt_timestamp(segment['end'])}",
                text,
                "",
            ]
        )
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def transcribe_to_srt(model: whisper.Whisper, audio_path: Path, output_path: Path, task: str) -> None:
    if output_path.exists() and output_path.stat().st_size > 1_000:
        log(f"Subtítulo ya existe: {output_path}")
        return

    result = model.transcribe(
        str(audio_path),
        language="es",
        task=task,
        verbose=False,
        fp16=torch.cuda.is_available(),
        temperature=0.0,
        condition_on_previous_text=True,
        compression_ratio_threshold=2.4,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
    )
    write_srt(result["segments"], output_path)


def mux_final_video(input_video: Path, cleaned_audio: Path, srt_es: Path, srt_en: Path, output_video: Path) -> None:
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-i",
            str(cleaned_audio),
            "-i",
            str(srt_es),
            "-i",
            str(srt_en),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-map",
            "2:0",
            "-map",
            "3:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-c:s",
            "srt",
            "-metadata:s:s:0",
            "language=spa",
            "-metadata:s:s:0",
            "title=Español",
            "-metadata:s:s:1",
            "language=eng",
            "-metadata:s:s:1",
            "title=English",
            "-shortest",
            str(output_video),
        ]
    )


def process_video(input_video: Path, model: whisper.Whisper, args: argparse.Namespace) -> VideoResult:
    started = time.time()
    talk_work = ensure_dir(Path(args.work_dir) / input_video.stem)
    talk_out = ensure_dir(Path(args.deliver_dir) / input_video.stem)

    enhanced_audio = talk_out / f"{input_video.stem}_audio_mejorado.wav"
    subtitles_es = talk_out / f"{input_video.stem}.es.srt"
    subtitles_en = talk_out / f"{input_video.stem}.en.srt"
    output_video = talk_out / f"{input_video.stem}_final.mkv"

    try:
        audio_final = enhance_audio(
            input_video=input_video,
            work_dir=talk_work,
            device=args.audio_device,
            chunk_seconds=args.chunk_seconds,
            keep_intermediates=args.keep_intermediates,
        )
        if audio_final != enhanced_audio:
            shutil.copy2(audio_final, enhanced_audio)

        transcribe_to_srt(model, enhanced_audio, subtitles_es, task="transcribe")
        transcribe_to_srt(model, enhanced_audio, subtitles_en, task="translate")
        mux_final_video(input_video, enhanced_audio, subtitles_es, subtitles_en, output_video)

        return VideoResult(
            input_video=str(input_video),
            enhanced_audio_wav=str(enhanced_audio),
            subtitles_es=str(subtitles_es),
            subtitles_en=str(subtitles_en),
            output_mkv=str(output_video),
            duration_seconds=ffprobe_duration(input_video),
            elapsed_seconds=time.time() - started,
            status="ok",
        )
    except Exception as exc:  # noqa: BLE001
        return VideoResult(
            input_video=str(input_video),
            enhanced_audio_wav=str(enhanced_audio),
            subtitles_es=str(subtitles_es),
            subtitles_en=str(subtitles_en),
            output_mkv=str(output_video),
            duration_seconds=ffprobe_duration(input_video),
            elapsed_seconds=time.time() - started,
            status="error",
            error=str(exc),
        )


def main() -> int:
    args = parse_args()

    if args.audio_device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA no está disponible dentro del contenedor.")

    if args.audio_device.startswith("cuda"):
        gpu_name = torch.cuda.get_device_name(0)
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        log(f"GPU activa: {gpu_name}")
        log(f"VRAM libre: {free_mem / 1024**3:.1f} / {total_mem / 1024**3:.1f} GB")

    inputs = discover_inputs(args.input_glob)
    log(f"Videos detectados: {len(inputs)}")
    for video in inputs:
        log(f"  - {video}")

    log(f"Cargando Whisper ({args.whisper_model}) en {args.whisper_device}...")
    whisper_model = whisper.load_model(args.whisper_model, device=args.whisper_device)

    results: list[VideoResult] = []
    for input_video in inputs:
        log(f"=== Procesando {input_video.name} ===")
        result = process_video(input_video, whisper_model, args)
        results.append(result)
        if result.status == "ok":
            log(f"✓ Listo: {result.output_mkv} ({result.elapsed_seconds / 60:.1f} min)")
        else:
            log(f"✗ Error en {input_video.name}: {result.error}")

    summary_path = ensure_dir(Path(args.deliver_dir)) / "batch_summary.json"
    summary_path.write_text(
        json.dumps([asdict(result) for result in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log(f"Resumen guardado en {summary_path}")

    failures = [result for result in results if result.status != "ok"]
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
