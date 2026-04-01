"""Model registry with auto-download capability for video enhancement models."""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ModelEntry:
    """Descriptor for a downloadable model."""

    key: str
    filename: str
    url: str
    sha256: str
    scale: int
    description: str


# ---------------------------------------------------------------------------
# Pre-registered models
# ---------------------------------------------------------------------------

KNOWN_MODELS = [
    ModelEntry(
        key="anime_baseline",
        filename="realesr-animevideov3.pth",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        sha256="c82afd38612a80f57e7b24e012e76ef66bfbe65a39f3e7c3a942db1e454dd485",
        scale=4,
        description="Real-ESRGAN anime video model v3 (x4 scale)",
    ),
    ModelEntry(
        key="real_x2",
        filename="RealESRGAN_x2plus.pth",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        sha256="49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb",
        scale=2,
        description="Real-ESRGAN x2plus real-world model (x2 native scale)",
    ),
    ModelEntry(
        key="real_x2plus",
        filename="RealESRGAN_x2plus.pth",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        sha256="49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb",
        scale=2,
        description="Alias for Real-ESRGAN x2plus real-world model",
    ),
    ModelEntry(
        key="real_x4plus",
        filename="RealESRGAN_x4plus.pth",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        sha256="4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
        scale=4,
        description="Real-ESRGAN x4plus general model for real-world scenes",
    ),
]

# ---------------------------------------------------------------------------
# Project root helper
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _default_models_dir() -> Path:
    return _PROJECT_ROOT / "enhanced" / "models"


# ---------------------------------------------------------------------------
# Download progress callback
# ---------------------------------------------------------------------------

def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
    """Simple progress callback for urlretrieve."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100.0)
        bar_len = 40
        filled = int(bar_len * pct / 100.0)
        bar = "=" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%  ({downloaded}/{total_size} bytes)")
    else:
        sys.stdout.write(f"\r  downloaded {downloaded} bytes")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# SHA-256 verification
# ---------------------------------------------------------------------------

def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Central registry that resolves model keys to local file paths,
    downloading and verifying weights automatically when needed."""

    def __init__(self, models_dir: Path | None = None) -> None:
        self._models_dir = models_dir or _default_models_dir()
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, ModelEntry] = {}

        # Pre-register known models
        for entry in KNOWN_MODELS:
            self.register(entry)

    # -- public API ---------------------------------------------------------

    def register(self, entry: ModelEntry) -> None:
        """Add (or replace) a model entry in the registry."""
        self._entries[entry.key] = entry

    def get_path(self, key: str) -> Path:
        """Return the local path for *key*, downloading the model first if it
        is not already present on disk."""
        if key not in self._entries:
            raise KeyError(f"Unknown model key: {key!r}")
        entry = self._entries[key]
        local = self._models_dir / entry.filename
        if not local.exists():
            return self._download(entry)
        return local

    def ensure_model(self, key: str) -> Path:
        """Alias for :meth:`get_path`."""
        return self.get_path(key)

    def list_models(self) -> list[ModelEntry]:
        """Return a list of all registered model entries."""
        return list(self._entries.values())

    # -- internal -----------------------------------------------------------

    def _download(self, entry: ModelEntry) -> Path:
        """Download *entry*, verify its SHA-256 checksum and update the
        on-disk manifest."""
        dest = self._models_dir / entry.filename
        tmp = dest.with_suffix(dest.suffix + ".tmp")

        print(f"Downloading model '{entry.key}' from {entry.url} ...")
        try:
            urllib.request.urlretrieve(entry.url, str(tmp), reporthook=_reporthook)
            print()  # newline after progress bar
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to download model '{entry.key}': {exc}"
            ) from exc

        # Verify SHA-256
        actual = _sha256_file(tmp)
        if actual != entry.sha256:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA-256 mismatch for '{entry.key}':\n"
                f"  expected: {entry.sha256}\n"
                f"  got:      {actual}"
            )

        tmp.rename(dest)
        print(f"Model '{entry.key}' saved to {dest}")

        self._update_manifest(entry)
        return dest

    def _update_manifest(self, entry: ModelEntry) -> None:
        """Write / update ``manifest.json`` with metadata for the just-downloaded
        model."""
        manifest_path = self._models_dir / "manifest.json"

        manifest: dict = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                manifest = {}

        manifest[entry.key] = {
            "filename": entry.filename,
            "sha256": entry.sha256,
            "url": entry.url,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
        }

        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
