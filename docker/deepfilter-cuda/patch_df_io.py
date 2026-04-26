from __future__ import annotations

from pathlib import Path

DF_IO = Path('/usr/local/lib/python3.10/dist-packages/df/io.py')
DF_IO.write_text(
    r'''import os
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from loguru import logger
from numpy import ndarray
from torch import Tensor

from df.logger import warn_once
from df.utils import download_file, get_cache_dir, get_git_root

AudioMetaData = Any


def load_audio(
    file: str, sr: Optional[int] = None, verbose=True, **kwargs
) -> Tuple[Tensor, AudioMetaData]:
    """Loads an audio file using soundfile; returns channels-first torch audio."""
    method = kwargs.pop("method", "sinc_fast")
    frame_offset = int(kwargs.pop("frame_offset", kwargs.pop("offset", 0)) or 0)
    num_frames = kwargs.pop("num_frames", -1)
    frames = -1 if num_frames is None else int(num_frames)

    data, orig_sr = sf.read(
        file,
        dtype="float32",
        always_2d=True,
        start=frame_offset,
        frames=frames,
    )
    info = SimpleNamespace(
        sample_rate=int(orig_sr),
        num_frames=int(data.shape[0]),
        num_channels=int(data.shape[1]),
        bits_per_sample=0,
        encoding="UNKNOWN",
    )
    audio = torch.from_numpy(np.ascontiguousarray(data.T))
    if sr is not None and orig_sr != sr:
        if verbose:
            warn_once(
                f"Audio sampling rate does not match model sampling rate ({orig_sr}, {sr}). "
                "Resampling..."
            )
        audio = resample(audio, int(orig_sr), int(sr), method=method)
    return audio.contiguous(), info


def save_audio(
    file: str,
    audio: Union[Tensor, ndarray],
    sr: int,
    output_dir: Optional[str] = None,
    suffix: Optional[str] = None,
    log: bool = False,
    dtype=torch.int16,
):
    outpath = file
    if suffix is not None:
        file, ext = os.path.splitext(file)
        outpath = file + f"_{suffix}" + ext
    if output_dir is not None:
        outpath = os.path.join(output_dir, os.path.basename(outpath))
    if log:
        logger.info(f"Saving audio file '{outpath}'")

    audio = torch.as_tensor(audio).detach().cpu()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.ndim != 2:
        raise ValueError(f"Expected audio with 1 or 2 dims, got {audio.shape}")

    arr = audio.float().clamp(-1.0, 1.0).numpy()
    if arr.shape[0] <= 16 and arr.shape[1] >= arr.shape[0]:
        arr = arr.T
    subtype = "FLOAT" if dtype == torch.float32 else "PCM_16"
    sf.write(outpath, arr, int(sr), subtype=subtype)


def get_resample_params(method: str) -> Dict[str, Any]:
    params = {
        "sinc_fast": {},
        "sinc_best": {},
        "kaiser_fast": {},
        "kaiser_best": {},
    }
    assert method in params.keys(), f"method must be one of {list(params.keys())}"
    return params[method]


def resample(audio: Tensor, orig_sr: int, new_sr: int, method="sinc_fast"):
    if orig_sr == new_sr:
        return audio
    audio = torch.as_tensor(audio)
    squeeze = audio.ndim == 1
    if squeeze:
        audio = audio.unsqueeze(0)
    if audio.ndim != 2:
        raise ValueError(f"Expected [C, T] audio, got {audio.shape}")
    new_len = max(1, int(round(audio.shape[-1] * float(new_sr) / float(orig_sr))))
    x = audio.unsqueeze(0).float()
    y = F.interpolate(x, size=new_len, mode="linear", align_corners=False).squeeze(0)
    return y.squeeze(0) if squeeze else y


def get_test_sample(sr: int = 48000) -> Tensor:
    dir = get_git_root()
    file_path = os.path.join("assets", "clean_freesound_33711.wav")
    if dir is None:
        url = "https://github.com/Rikorose/DeepFilterNet/raw/main/" + file_path
        save_dir = get_cache_dir()
        path = download_file(url, save_dir)
    else:
        path = os.path.join(dir, file_path)
    sample, _ = load_audio(path, sr=sr)
    return sample
'''
)
print(f"Patched {DF_IO}")
