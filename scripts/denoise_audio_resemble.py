#!/usr/bin/env python3
"""
Audio denoise-only using Resemble-Enhance denoiser (GPU).
NO generative enhancement - just noise removal.
Processes as mono (Zoom conference = same audio both channels).
"""
import os
import sys
import time
import subprocess
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

INPUT_WAV = "/datos/Neuro/audio_resemble/original_full.wav"
OUTPUT_DIR = "/datos/Neuro/audio_resemble/chunks_denoise"
OUTPUT_DONE = "/datos/Neuro/audio_resemble/chunks_denoise_done"
OUTPUT_WAV = "/datos/Neuro/audio_resemble/denoised_full.wav"
CHUNK_SECONDS = 300  # 5 min
DEVICE = "cuda:0"


def get_duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True)
    return float(r.stdout.strip())


def split_audio(input_path, output_dir, chunk_sec):
    os.makedirs(output_dir, exist_ok=True)
    duration = get_duration(input_path)
    print(f"Total duration: {duration:.1f}s ({duration/3600:.2f}h)")

    chunks = []
    start = 0
    idx = 0
    while start < duration:
        chunk_path = os.path.join(output_dir, f"chunk_{idx:04d}.wav")
        length = min(chunk_sec, duration - start)
        if not os.path.exists(chunk_path):
            subprocess.run([
                "ffmpeg", "-y", "-ss", str(start), "-t", str(length),
                "-i", input_path, "-acodec", "pcm_s16le", "-ar", "48000",
                "-ac", "1",  # mono!
                chunk_path
            ], capture_output=True)
        chunks.append({"idx": idx, "path": chunk_path, "start": start, "duration": length})
        start += chunk_sec
        idx += 1
    print(f"Split into {len(chunks)} mono chunks")
    return chunks


def denoise_chunk(chunk_path, output_path, device):
    from resemble_enhance.denoiser.inference import denoise
    wav, sr = torchaudio.load(chunk_path)
    mono = wav[0]  # already mono from split

    denoised, new_sr = denoise(mono.to(device), sr, device=device, run_dir=None)
    denoised = denoised.cpu()
    if new_sr != 48000:
        denoised = torchaudio.transforms.Resample(new_sr, 48000)(denoised)

    torchaudio.save(output_path, denoised.unsqueeze(0), 48000)


def concat_chunks(chunks, output_path, sr=48000):
    """Simple concat (no overlap needed for denoise-only)."""
    parts = []
    for c in tqdm(chunks, desc="Concatenating"):
        wav, _ = torchaudio.load(c["done_path"])
        parts.append(wav)
    full = torch.cat(parts, dim=1)
    torchaudio.save(output_path, full, sr)
    print(f"Final: {output_path} ({full.shape[1]/sr:.1f}s)")


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Mode: DENOISE ONLY (no generative enhancement)")
    print()

    # Split
    chunks = split_audio(INPUT_WAV, OUTPUT_DIR, CHUNK_SECONDS)

    # Denoise each chunk
    os.makedirs(OUTPUT_DONE, exist_ok=True)
    t0 = time.time()
    for i, c in enumerate(chunks):
        done_path = os.path.join(OUTPUT_DONE, f"denoised_{c['idx']:04d}.wav")
        c["done_path"] = done_path
        if os.path.exists(done_path) and os.path.getsize(done_path) > 10000:
            print(f"[{i+1}/{len(chunks)}] Skip (done)")
            continue
        print(f"[{i+1}/{len(chunks)}] Denoising chunk {c['idx']}...")
        ct = time.time()
        denoise_chunk(c["path"], done_path, DEVICE)
        el = time.time() - ct
        done_count = i + 1
        avg = (time.time() - t0) / done_count
        eta = avg * (len(chunks) - done_count)
        print(f"  {el:.1f}s | ETA {eta/60:.0f}min")

    # Concat
    concat_chunks(chunks, OUTPUT_WAV)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
