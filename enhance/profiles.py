"""Named profiles for the video enhancement pipeline.

Each profile category (visual, audio, scheduler, RIFE backend) is a frozen
dataclass.  The ``get_profiles()`` helper resolves the active profile for
every category by checking:
  1. An explicit *name* argument,
  2. The corresponding ``ENHANCE_*_PROFILE`` environment variable,
  3. The ``"baseline"`` default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Profile dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VisualProfile:
    """Controls ESRGAN model selection, pre-downscale and detail mixing."""
    name: str = "baseline"
    model_key: str = "anime_baseline"
    downscale_factor: float = 0.5
    hybrid_detail_weight: float = 0.0
    face_adaptive: bool = False
    face_roi: Tuple[float, float, float, float] = (0.5, 0.0, 1.0, 0.5)


@dataclass
class AudioProfile:
    """Controls the ffmpeg audio filter chain, codec and bitrate."""
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
    """CPU-pinning, ionice and chunk sizing knobs."""
    name: str = "baseline"
    cpuset_ffmpeg: str = ""
    cpuset_audio: str = ""
    cpuset_python: str = ""
    ionice_class: int = 0
    ionice_level: int = 0
    use_chrt: bool = False
    rife_threads: str = "1:8:4"
    chunk_seconds: int = 15


@dataclass
class RIFEBackendProfile:
    """RIFE interpolation backend configuration."""
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


# ---------------------------------------------------------------------------
# Built-in profile registries
# ---------------------------------------------------------------------------

VISUAL_PROFILES: Dict[str, VisualProfile] = {
    "baseline": VisualProfile(
        name="baseline",
        model_key="anime_baseline",
        downscale_factor=0.5,
    ),
    "fast": VisualProfile(
        name="fast",
        model_key="anime_baseline",
        downscale_factor=0.5,
    ),
    "real_x2": VisualProfile(
        name="real_x2",
        model_key="real_x2",
        downscale_factor=1.0,
    ),
    "real_x2plus": VisualProfile(
        name="real_x2plus",
        model_key="real_x2plus",
        downscale_factor=1.0,
    ),
    "real_x4plus": VisualProfile(
        name="real_x4plus",
        model_key="real_x4plus",
        downscale_factor=0.5,
        hybrid_detail_weight=0.08,
    ),
    "general_light": VisualProfile(
        name="general_light",
        model_key="general_x4v3",
        downscale_factor=0.5,
        hybrid_detail_weight=0.10,
        face_adaptive=True,
        face_roi=(0.5, 0.0, 1.0, 0.5),
    ),
    "general_light_wdn": VisualProfile(
        name="general_light_wdn",
        model_key="general_wdn_x4v3",
        downscale_factor=0.5,
        hybrid_detail_weight=0.08,
        face_adaptive=True,
        face_roi=(0.5, 0.0, 1.0, 0.5),
    ),
    "hybrid_detail": VisualProfile(
        name="hybrid_detail",
        model_key="real_x4plus",
        downscale_factor=0.5,
        hybrid_detail_weight=0.12,
    ),
    "face_adaptive": VisualProfile(
        name="face_adaptive",
        model_key="real_x2plus",
        downscale_factor=1.0,
        hybrid_detail_weight=0.2,
        face_adaptive=True,
    ),
    "quality": VisualProfile(
        name="quality",
        model_key="real_x2plus",
        downscale_factor=1.0,
        hybrid_detail_weight=0.15,
        face_adaptive=True,
        face_roi=(0.5, 0.0, 1.0, 0.5),  # top-right quadrant (speaker tile in Zoom)
    ),
    "face_preserve": VisualProfile(
        name="face_preserve",
        model_key="real_x2plus",
        downscale_factor=1.0,
        hybrid_detail_weight=0.25,
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

AUDIO_PROFILES: Dict[str, AudioProfile] = {
    "baseline": AudioProfile(
        name="baseline",
        filter_chain=(
            "afftdn=nf=-20:nt=w:om=o,"
            "loudnorm=I=-16:TP=-1.5:LRA=11,"
            "dynaudnorm=f=250:g=31:p=0.95:m=8.0"
        ),
    ),
    "conservative": AudioProfile(
        name="conservative",
        filter_chain=(
            "highpass=f=80,"
            "anlmdn=s=7:p=0.002:m=15,"
            "loudnorm=I=-16:TP=-1.5:LRA=11,"
            "alimiter=level_in=1:level_out=1:limit=0.95"
        ),
    ),
    "voice": AudioProfile(
        name="voice",
        filter_chain=(
            "highpass=f=80,"
            "anlmdn=s=7:p=0.002:m=15,"
            "dialoguenhance,"
            "speechnorm=e=12.5:r=0.0001:l=1,"
            "acompressor=threshold=0.05:ratio=3:attack=5:release=50,"
            "loudnorm=I=-16:TP=-1.5:LRA=11,"
            "alimiter=level_in=1:level_out=1:limit=0.95"
        ),
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
    "voice_natural": AudioProfile(
        name="voice_natural",
        filter_chain=(
            "highpass=f=80,"
            "anlmdn=s=7:p=0.002:m=15,"
            "dialoguenhance,"
            "loudnorm=I=-16:TP=-1.5:LRA=11,"
            "alimiter=level_in=1:level_out=1:limit=0.95:release=50"
        ),
    ),
    "lecture_natural": AudioProfile(
        name="lecture_natural",
        filter_chain=(
            "highpass=f=80,"
            "anlmdn=s=7:p=0.002:m=15,"
            "dialoguenhance,"
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

SCHEDULER_PROFILES: Dict[str, SchedulerProfile] = {
    "baseline": SchedulerProfile(
        name="baseline",
    ),
    # T7: CCD-aware pinning with ionice + chrt for Ryzen 9 9950X3D dual-CCD.
    # CCD0 cores 0-7 (threads 0-7,16-23) handle I/O-heavy ffmpeg/audio;
    # CCD1 cores 8-15 (threads 8-15,24-31) handle Python/ESRGAN work.
    "split_l3_a": SchedulerProfile(
        name="split_l3_a",
        cpuset_ffmpeg="0-7,16-23",
        cpuset_audio="0-7,16-23",
        cpuset_python="8-15,24-31",
        ionice_class=2,
        ionice_level=0,
        use_chrt=True,
    ),
    "split_l3_b": SchedulerProfile(
        name="split_l3_b",
        cpuset_ffmpeg="8-15,24-31",
        cpuset_audio="8-15,24-31",
        cpuset_python="0-7,16-23",
        rife_threads="2:8:4",
    ),
    # T9: Production profile — CCD split + ionice/chrt + tuned chunk size.
    # Use with ENHANCE_NVENC_GPUS=0,1 for dual-NVENC encoding.
    "production": SchedulerProfile(
        name="production",
        cpuset_ffmpeg="0-7,16-23",
        cpuset_audio="0-7,16-23",
        cpuset_python="8-15,24-31",
        ionice_class=2,
        ionice_level=0,
        use_chrt=True,
        chunk_seconds=30,
        rife_threads="1:8:4",
    ),
}

RIFE_BACKEND_PROFILES: Dict[str, RIFEBackendProfile] = {
    "baseline": RIFEBackendProfile(
        name="baseline",
    ),
    "torch": RIFEBackendProfile(
        name="torch",
        backend="torch",
        model_name="paper_v6",
    ),
    "torch_cpu": RIFEBackendProfile(
        name="torch_cpu",
        backend="torch",
        device="cpu",
        model_name="paper_v6",
    ),
}

# ---------------------------------------------------------------------------
# Registry look-up map (category → dict)
# ---------------------------------------------------------------------------

_REGISTRIES = {
    "visual": VISUAL_PROFILES,
    "audio": AUDIO_PROFILES,
    "scheduler": SCHEDULER_PROFILES,
    "rife_backend": RIFE_BACKEND_PROFILES,
}

# ---------------------------------------------------------------------------
# Public resolver
# ---------------------------------------------------------------------------

def get_profiles(
    visual: str | None = None,
    audio: str | None = None,
    scheduler: str | None = None,
    rife_backend: str | None = None,
) -> tuple[VisualProfile, AudioProfile, SchedulerProfile, RIFEBackendProfile]:
    """Resolve the active profile for each category.

    Resolution order for every category:
      1. The explicit keyword argument (if not ``None``).
      2. The ``ENHANCE_<CATEGORY>_PROFILE`` environment variable.
      3. ``"baseline"``.

    Raises ``KeyError`` if a requested profile name is not registered.
    """

    def _resolve(
        explicit: str | None,
        env_var: str,
        registry: dict,
        category: str,
    ):
        name = explicit or os.environ.get(env_var) or "baseline"
        if name not in registry:
            available = ", ".join(sorted(registry.keys()))
            raise KeyError(
                f"Unknown {category} profile {name!r}. "
                f"Available: {available}"
            )
        return registry[name]

    v = _resolve(visual, "ENHANCE_VISUAL_PROFILE", VISUAL_PROFILES, "visual")
    a = _resolve(audio, "ENHANCE_AUDIO_PROFILE", AUDIO_PROFILES, "audio")
    s = _resolve(scheduler, "ENHANCE_SCHEDULER_PROFILE", SCHEDULER_PROFILES, "scheduler")
    r = _resolve(rife_backend, "ENHANCE_RIFE_BACKEND", RIFE_BACKEND_PROFILES, "rife_backend")

    return v, a, s, r
