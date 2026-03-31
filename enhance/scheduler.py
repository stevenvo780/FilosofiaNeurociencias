"""CPU affinity and scheduling helpers.

Wraps subprocess commands with ``taskset``, ``ionice`` and ``chrt``
according to the active :class:`~enhance.profiles.SchedulerProfile`.
"""

from __future__ import annotations

from typing import List, Optional

from enhance.profiles import SchedulerProfile

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_active_profile: Optional[SchedulerProfile] = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_scheduler_profile(profile: SchedulerProfile) -> None:
    """Activate *profile* for all subsequent :func:`wrap_subprocess` calls."""
    global _active_profile
    _active_profile = profile


def get_active_scheduler() -> Optional[SchedulerProfile]:
    """Return the currently active scheduler profile, or ``None``."""
    return _active_profile


def wrap_subprocess(cmd: list[str], role: str = "default") -> list[str]:
    """Prepend CPU-pinning / scheduling prefixes to *cmd*.

    Parameters
    ----------
    cmd:
        The original command list (e.g. ``["ffmpeg", "-i", …]``).
    role:
        One of ``"ffmpeg"``, ``"audio"``, ``"rife"``, ``"python"`` or
        ``"default"``.  Determines which cpuset from the active profile
        is used.

    Returns
    -------
    list[str]
        The command with ``taskset``, ``ionice`` and/or ``chrt`` prepended
        as appropriate.  If no scheduler profile is active or the role has
        no cpuset configured, *cmd* is returned unchanged.

    Role → cpuset mapping
    ---------------------
    * ``"ffmpeg"``  → ``cpuset_ffmpeg``
    * ``"audio"``   → ``cpuset_audio``
    * ``"rife"``    → ``cpuset_ffmpeg``  (shares with ffmpeg)
    * ``"python"``  → ``cpuset_python``
    * ``"default"`` → no pinning
    """
    if _active_profile is None:
        return list(cmd)

    prefix: List[str] = []

    # --- chrt (batch scheduling class) ---
    if _active_profile.use_chrt:
        prefix += ["chrt", "-b", "0"]

    # --- ionice ---
    if _active_profile.ionice_class > 0:
        prefix += [
            "ionice",
            "-c", str(_active_profile.ionice_class),
            "-n", str(_active_profile.ionice_level),
        ]

    # --- taskset (CPU affinity) ---
    cpuset = _resolve_cpuset(_active_profile, role)
    if cpuset:
        prefix += ["taskset", "-c", cpuset]

    if not prefix:
        return list(cmd)

    return prefix + list(cmd)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_ROLE_MAP = {
    "ffmpeg": "cpuset_ffmpeg",
    "audio": "cpuset_audio",
    "rife": "cpuset_ffmpeg",  # RIFE shares the ffmpeg cpuset
    "python": "cpuset_python",
}


def _resolve_cpuset(profile: SchedulerProfile, role: str) -> str:
    """Return the cpuset string for *role*, or ``""`` if none configured."""
    attr = _ROLE_MAP.get(role, "")
    if not attr:
        return ""
    return getattr(profile, attr, "") or ""
