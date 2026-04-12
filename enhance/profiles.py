"""Named profiles for the video enhancement pipeline.

Each profile category (visual, audio, scheduler, RIFE backend) is a frozen
dataclass.  ``get_profiles()`` resolves the active profile by checking:
  1. An explicit *name* argument,
  2. The corresponding ``ENHANCE_*_PROFILE`` environment variable,
  3. The ``"baseline"`` default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class VisualProfile:
    name: str = "baseline"
    model_key: str = "anime_baseline"
    downscale_factor: float = 0.5
    hybrid_detail_weight: float = 0.0
    face_adaptive: bool = False
    face_roi: Tuple[float, float, float, float] = (0.5, 0.0, 1.0, 0.5)


@dataclass
class AudioProfile:
    name: str = "baseline"
    filter_chain: str = (
        "afftdn=nf=-20:nt=w:om=o,"
        "loudnorm=I=-16:TP=-1.5:LRA=11,"
        "dynaudnorm=f=250:g=31:p=0.95:m=8.0"
    )
    codec: str = "aac"
    bitrate: str = "256k"
    sample_rate: int = 48000
    threads: int = field(default_factory=lambda: min(os.cpu_count() or 8, 24))


@dataclass
class SchedulerProfile:
    name: str = "baseline"
    cpuset_ffmpeg: str = ""
    cpuset_audio: str = ""
    cpuset_python: str = ""
    ionice_class: int = 0
    ionice_level: int = 0
    use_chrt: bool = False
    rife_threads: str = "1:4:4"
    chunk_seconds: int = 15


@dataclass
class RIFEBackendProfile:
    name: str = "baseline"
    backend: str = "ncnn"
    device: str = "cuda"
    model_name: str = "paper_v6"
    gpu: int = 1
    stream_window: int = 192
    min_window: int = 64
    poll_seconds: float = 0.05
    file_settle_seconds: float = 0.05
    cleanup_mode: str = "inline"


# ── Visual profiles ─────────────────────────────────────────

VISUAL_PROFILES: Dict[str, VisualProfile] = {
    "baseline": VisualProfile(
        name="baseline",
        model_key="anime_baseline",
        downscale_factor=0.5,
    ),
    "quality": VisualProfile(
        name="quality",
        model_key="real_x2plus",
        downscale_factor=1.0,
        hybrid_detail_weight=0.15,
        face_adaptive=True,
        face_roi=(0.5, 0.0, 1.0, 0.5),
    ),
    "production": VisualProfile(
        name="production",
        model_key="real_x2plus",
        downscale_factor=1.0,
        hybrid_detail_weight=0.15,
        face_adaptive=True,
        face_roi=(0.5, 0.0, 1.0, 0.5),
    ),
}

# ── Audio profiles ───────────────────────────────────────────

AUDIO_PROFILES: Dict[str, AudioProfile] = {
    "baseline": AudioProfile(
        name="baseline",
    ),
    "natural": AudioProfile(
        name="natural",
        filter_chain=(
            "highpass=f=80,"
            "anlmdn=s=7:p=0.002:m=15,"
            "loudnorm=I=-16:TP=-1.5:LRA=11,"
            "alimiter=level_in=1:level_out=1:limit=0.95:release=50"
        ),
    ),
    "production": AudioProfile(
        name="production",
        filter_chain=(
            "highpass=f=80,"
            "anlmdn=s=7:p=0.002:m=15,"
            "dialoguenhance,"
            "loudnorm=I=-16:TP=-1.5:LRA=11,"
            "alimiter=level_in=1:level_out=1:limit=0.95:release=50"
        ),
    ),
}

# ── Scheduler profiles ───────────────────────────────────────

SCHEDULER_PROFILES: Dict[str, SchedulerProfile] = {
    "baseline": SchedulerProfile(),
    "production": SchedulerProfile(
        name="production",
        cpuset_ffmpeg="0-7,16-23",
        cpuset_audio="0-7,16-23",
        cpuset_python="8-15,24-31",
        ionice_class=2,
        ionice_level=0,
        use_chrt=True,
        chunk_seconds=30,
        rife_threads="1:4:4",
    ),
}

# ── RIFE backend profiles ───────────────────────────────────

RIFE_BACKEND_PROFILES: Dict[str, RIFEBackendProfile] = {
    "baseline": RIFEBackendProfile(name="baseline", backend="ncnn"),
    "torch_cpu": RIFEBackendProfile(name="torch_cpu", backend="torch", device="cpu"),
    "torch_gpu": RIFEBackendProfile(name="torch_gpu", backend="torch", device="cuda"),
}

# ── Resolver ─────────────────────────────────────────────────

_REGISTRIES = {
    "visual": VISUAL_PROFILES,
    "audio": AUDIO_PROFILES,
    "scheduler": SCHEDULER_PROFILES,
    "rife_backend": RIFE_BACKEND_PROFILES,
}


def get_profiles(
    visual: str | None = None,
    audio: str | None = None,
    scheduler: str | None = None,
    rife_backend: str | None = None,
) -> tuple[VisualProfile, AudioProfile, SchedulerProfile, RIFEBackendProfile]:
    """Resolve the active profile for each category."""

    def _resolve(explicit, env_var, registry, category):
        name = explicit or os.environ.get(env_var) or "baseline"
        if name not in registry:
            available = ", ".join(sorted(registry.keys()))
            raise KeyError(f"Unknown {category} profile {name!r}. Available: {available}")
        return registry[name]

    return (
        _resolve(visual, "ENHANCE_VISUAL_PROFILE", VISUAL_PROFILES, "visual"),
        _resolve(audio, "ENHANCE_AUDIO_PROFILE", AUDIO_PROFILES, "audio"),
        _resolve(scheduler, "ENHANCE_SCHEDULER_PROFILE", SCHEDULER_PROFILES, "scheduler"),
        _resolve(rife_backend, "ENHANCE_RIFE_BACKEND", RIFE_BACKEND_PROFILES, "rife_backend"),
    )
