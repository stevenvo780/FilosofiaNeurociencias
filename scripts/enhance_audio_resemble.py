#!/usr/bin/env python3
"""
Audio enhancement using Resemble-Enhance (GPU-accelerated).
Processes audio in chunks to avoid OOM, then concatenates.
"""
import os
import sys
import time
import subprocess
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

# Config
INPUT_WAV = "/datos/Neuro/audio_resemble/original_full.wav"
OUTPUT_DIR = "/datos/Neuro/audio_resemble/chunks_enhanced"
OUTPUT_WAV = "/datos/Neuro/audio_resemble/enhanced_full.wav"
CHUNK_SECONDS = 300  # 5 minutes
OVERLAP_SECONDS = 2  # crossfade overlap
DEVICE = "cuda:0"
# Resemble-enhance params
DENOISE_CFG_RATE = 0.7  # 0=less denoise, 1=max denoise
ENHANCE_CFG_RATE = 0.3  # balance: 0=natural, 1=max enhance (too high = artifacts)

def split_audio(input_path, output_dir, chunk_sec, overlap_sec):
    """Split WAV into overlapping chunks using ffmpeg."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get duration
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", input_path],
        capture_output=True, text=True
    )
    duration = float(result.stdout.strip())
    print(f"Total duration: {duration:.1f}s ({duration/3600:.2f}h)")
    
    chunks = []
    start = 0
    idx = 0
    while start < duration:
        chunk_path = os.path.join(output_dir, f"chunk_{idx:04d}.wav")
        # Include overlap at start (except first chunk)
        actual_start = max(0, start - overlap_sec) if idx > 0 else 0
        actual_duration = chunk_sec + (overlap_sec if idx > 0 else 0)
        actual_duration = min(actual_duration, duration - actual_start)
        
        if not os.path.exists(chunk_path):
            subprocess.run([
                "ffmpeg", "-y", "-ss", str(actual_start), "-t", str(actual_duration),
                "-i", input_path, "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "2",
                chunk_path
            ], capture_output=True)
        
        chunks.append({
            "idx": idx,
            "path": chunk_path,
            "start": actual_start,
            "duration": actual_duration,
            "has_overlap_start": idx > 0,
        })
        start += chunk_sec
        idx += 1
    
    print(f"Split into {len(chunks)} chunks")
    return chunks


def enhance_chunk(chunk_path, output_path, device):
    """Enhance a single chunk using resemble-enhance."""
    from resemble_enhance.enhancer.inference import enhance as resemble_enhance_fn
    
    wav, sr = torchaudio.load(chunk_path)
    enhanced_channels = []
    for ch in range(wav.shape[0]):
        mono = wav[ch, :]  # (samples,)
        
        # Run enhancement on GPU
        enhanced, new_sr = resemble_enhance_fn(
            mono.to(device),
            sr,
            device=device,
            nfe=64,
            solver="midpoint",
            lambd=ENHANCE_CFG_RATE,
            tau=DENOISE_CFG_RATE,
        )
        
        # Resample back to 48000 if needed
        enhanced = enhanced.cpu()
        if new_sr != 48000:
            enhanced = torchaudio.transforms.Resample(new_sr, 48000)(enhanced)
        
        enhanced_channels.append(enhanced)
    
    # Stack channels back to stereo
    enhanced_stereo = torch.stack(enhanced_channels, dim=0)
    torchaudio.save(output_path, enhanced_stereo, 48000)
    return output_path


def crossfade_concat(chunks, output_path, sr=48000, overlap_sec=2):
    """Concatenate chunks with crossfade on overlapping regions."""
    overlap_samples = int(overlap_sec * sr)
    
    result = None
    for i, chunk_info in enumerate(tqdm(chunks, desc="Concatenating")):
        enhanced_path = chunk_info["enhanced_path"]
        wav, _ = torchaudio.load(enhanced_path)
        
        if result is None:
            result = wav
            continue
        
        if chunk_info["has_overlap_start"] and overlap_samples > 0:
            # Crossfade
            fade_out = torch.linspace(1, 0, overlap_samples)
            fade_in = torch.linspace(0, 1, overlap_samples)
            
            # Apply crossfade to overlap region
            end_region = result[:, -overlap_samples:]
            start_region = wav[:, :overlap_samples]
            
            crossfaded = end_region * fade_out + start_region * fade_in
            result = torch.cat([result[:, :-overlap_samples], crossfaded, wav[:, overlap_samples:]], dim=1)
        else:
            result = torch.cat([result, wav], dim=1)
    
    torchaudio.save(output_path, result, sr)
    print(f"Final enhanced audio: {output_path} ({result.shape[1]/sr:.1f}s)")
    return output_path


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM free: {torch.cuda.mem_get_info(0)[0]/1024**3:.1f} GB")
    
    # Step 1: Split
    print("\n=== Splitting audio ===")
    chunks = split_audio(INPUT_WAV, OUTPUT_DIR, CHUNK_SECONDS, OVERLAP_SECONDS)
    
    # Step 2: Enhance each chunk
    print("\n=== Enhancing with Resemble-Enhance (GPU) ===")
    enhanced_dir = OUTPUT_DIR + "_done"
    os.makedirs(enhanced_dir, exist_ok=True)
    
    t0 = time.time()
    for i, chunk in enumerate(chunks):
        enhanced_path = os.path.join(enhanced_dir, f"enhanced_{chunk['idx']:04d}.wav")
        chunk["enhanced_path"] = enhanced_path
        
        if os.path.exists(enhanced_path):
            sz = os.path.getsize(enhanced_path)
            if sz > 10000:
                print(f"[{i+1}/{len(chunks)}] Already done: {enhanced_path}")
                continue
        
        print(f"\n[{i+1}/{len(chunks)}] Enhancing chunk {chunk['idx']}...")
        ct = time.time()
        enhance_chunk(chunk["path"], enhanced_path, DEVICE)
        elapsed = time.time() - ct
        print(f"  Done in {elapsed:.1f}s")
        
        # ETA
        done = i + 1
        total_elapsed = time.time() - t0
        avg = total_elapsed / done
        remaining = avg * (len(chunks) - done)
        print(f"  ETA: {remaining/60:.0f} min remaining")
    
    # Step 3: Concatenate with crossfade
    print("\n=== Concatenating ===")
    crossfade_concat(chunks, OUTPUT_WAV, sr=48000, overlap_sec=OVERLAP_SECONDS)
    
    total = time.time() - t0
    print(f"\n=== Done! Total: {total/60:.1f} min ===")
    print(f"Output: {OUTPUT_WAV}")


if __name__ == "__main__":
    main()
