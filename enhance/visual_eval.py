"""Visual evaluation module with ROI-based comparison and detail blending."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ROI definitions — each value is (x0, y0, x1, y1) in normalised [0..1] coords
# ---------------------------------------------------------------------------

ROIS: dict[str, tuple[float, float, float, float]] = {
    "speaker_tile": (0.50, 0.00, 1.00, 0.50),  # top-right quadrant
    "poster_face":  (0.35, 0.20, 0.65, 0.55),  # central head area
    "poster_text":  (0.25, 0.55, 0.75, 0.85),  # lower-center text block
    "full_frame":   (0.00, 0.00, 1.00, 1.00),
}

# Image extensions considered when scanning directories
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_gray(img: np.ndarray) -> np.ndarray:
    """Convert to single-channel grayscale if needed."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _ssim_simple(ref: np.ndarray, enh: np.ndarray) -> float:
    """Compute a simplified SSIM between two same-sized images.

    Falls back to a Gaussian-blur based approximation when *skimage* is not
    available.
    """
    # Try skimage first — it gives the canonical SSIM value
    try:
        from skimage.metrics import structural_similarity  # type: ignore[import-untyped]

        gray_ref = _to_gray(ref)
        gray_enh = _to_gray(enh)
        return float(structural_similarity(gray_ref, gray_enh))
    except ImportError:
        pass

    # Fallback: simple SSIM via Gaussian statistics
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    g_ref = _to_gray(ref).astype(np.float64)
    g_enh = _to_gray(enh).astype(np.float64)

    mu1 = cv2.GaussianBlur(g_ref, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(g_enh, (11, 11), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(g_ref * g_ref, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(g_enh * g_enh, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(g_ref * g_enh, (11, 11), 1.5) - mu1_mu2

    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = num / den
    return float(ssim_map.mean())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_roi(
    frame: np.ndarray,
    roi: tuple[float, float, float, float],
) -> np.ndarray:
    """Crop a normalised ROI ``(x0, y0, x1, y1)`` from *frame*.

    Coordinates are fractions in [0, 1] relative to the frame dimensions.
    """
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = roi
    px0, py0 = int(round(x0 * w)), int(round(y0 * h))
    px1, py1 = int(round(x1 * w)), int(round(y1 * h))
    # Clamp
    px0, py0 = max(0, px0), max(0, py0)
    px1, py1 = min(w, px1), min(h, py1)
    return frame[py0:py1, px0:px1].copy()


def compute_metrics(
    ref: np.ndarray,
    enhanced: np.ndarray,
) -> dict[str, float]:
    """Compute PSNR and SSIM between *ref* and *enhanced*.

    Both arrays must have the same spatial dimensions.
    """
    if ref.shape != enhanced.shape:
        # Attempt to resize enhanced to ref size for robustness
        enhanced = cv2.resize(enhanced, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_CUBIC)

    psnr = float(cv2.PSNR(ref, enhanced))
    ssim = _ssim_simple(ref, enhanced)
    return {"psnr": psnr, "ssim": ssim}


def evaluate_visual_quality(
    original_dir: Path,
    enhanced_dir: Path,
    output_dir: Path,
    rois: dict[str, tuple[float, float, float, float]] | None = None,
) -> dict[str, Any]:
    """Evaluate visual quality across ROIs for matching frame pairs.

    For every ROI the function:
    * loads pairs of frames (matched by filename),
    * computes PSNR/SSIM,
    * saves a side-by-side comparison PNG into *output_dir*, and
    * returns a summary dict with per-ROI average metrics.
    """
    if rois is None:
        rois = ROIS

    original_dir = Path(original_dir)
    enhanced_dir = Path(enhanced_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover matching frame pairs
    orig_files = sorted(
        f for f in original_dir.iterdir() if f.suffix.lower() in _IMG_EXTS
    )
    pairs: list[tuple[Path, Path]] = []
    for orig_path in orig_files:
        enh_path = enhanced_dir / orig_path.name
        if enh_path.exists():
            pairs.append((orig_path, enh_path))

    if not pairs:
        logger.warning("No matching frame pairs found between %s and %s", original_dir, enhanced_dir)
        return {"pairs_evaluated": 0, "rois": {}}

    # Accumulate metrics per ROI
    roi_accum: dict[str, list[dict[str, float]]] = {name: [] for name in rois}

    for orig_path, enh_path in pairs:
        orig_frame = cv2.imread(str(orig_path), cv2.IMREAD_COLOR)
        enh_frame = cv2.imread(str(enh_path), cv2.IMREAD_COLOR)
        if orig_frame is None or enh_frame is None:
            logger.warning("Skipping unreadable pair: %s", orig_path.name)
            continue

        for roi_name, roi_coords in rois.items():
            orig_roi = extract_roi(orig_frame, roi_coords)
            enh_roi = extract_roi(enh_frame, roi_coords)

            # Resize enhanced ROI to match original ROI dims for metric computation
            if orig_roi.shape[:2] != enh_roi.shape[:2]:
                enh_roi = cv2.resize(
                    enh_roi,
                    (orig_roi.shape[1], orig_roi.shape[0]),
                    interpolation=cv2.INTER_CUBIC,
                )

            metrics = compute_metrics(orig_roi, enh_roi)
            roi_accum[roi_name].append(metrics)

            # Save side-by-side comparison
            sbs = np.hstack([orig_roi, enh_roi])
            comp_name = f"{orig_path.stem}_{roi_name}_compare.png"
            cv2.imwrite(str(output_dir / comp_name), sbs)

    # Build summary
    summary: dict[str, Any] = {"pairs_evaluated": len(pairs), "rois": {}}
    for roi_name, metric_list in roi_accum.items():
        if not metric_list:
            summary["rois"][roi_name] = {"psnr_avg": 0.0, "ssim_avg": 0.0, "count": 0}
            continue
        avg_psnr = sum(m["psnr"] for m in metric_list) / len(metric_list)
        avg_ssim = sum(m["ssim"] for m in metric_list) / len(metric_list)
        summary["rois"][roi_name] = {
            "psnr_avg": round(avg_psnr, 4),
            "ssim_avg": round(avg_ssim, 6),
            "count": len(metric_list),
        }

    return summary


# ---------------------------------------------------------------------------
# Detail / hybrid blending
# ---------------------------------------------------------------------------


def apply_hybrid_detail(
    sr_frame: np.ndarray,
    original_frame: np.ndarray,
    weight: float,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Blend SR output with detail/luma from the original frame.

    Steps:
    1. Resize *original_frame* to match *sr_frame* (bicubic).
    2. Convert both to LAB.
    3. Blend L channels: ``result_L = sr_L * (1 - weight) + orig_L * weight``.
    4. Reconstruct and convert back to BGR.

    Parameters
    ----------
    sr_frame:
        Super-resolved frame (BGR, uint8).
    original_frame:
        Source frame (BGR, uint8).
    weight:
        Blending weight for the original's L channel (0 = pure SR, 1 = pure original luma).
    target_size:
        Optional ``(width, height)`` to resize the result.
    """
    weight = float(np.clip(weight, 0.0, 1.0))

    # Resize original to SR dimensions
    sr_h, sr_w = sr_frame.shape[:2]
    orig_resized = cv2.resize(original_frame, (sr_w, sr_h), interpolation=cv2.INTER_CUBIC)

    # Convert to LAB
    sr_lab = cv2.cvtColor(sr_frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    orig_lab = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Blend L channel
    sr_lab[:, :, 0] = sr_lab[:, :, 0] * (1.0 - weight) + orig_lab[:, :, 0] * weight

    # Convert back to BGR
    result = cv2.cvtColor(sr_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    if target_size is not None:
        result = cv2.resize(result, target_size, interpolation=cv2.INTER_CUBIC)

    return result


# ---------------------------------------------------------------------------
# Face-adaptive blending
# ---------------------------------------------------------------------------

# Lazy-loaded Haar cascade path
_HAAR_CASCADE: cv2.CascadeClassifier | None = None


def _get_face_cascade() -> cv2.CascadeClassifier:
    """Return a shared Haar cascade classifier for frontal faces."""
    global _HAAR_CASCADE
    if _HAAR_CASCADE is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
        _HAAR_CASCADE = cv2.CascadeClassifier(cascade_path)
        if _HAAR_CASCADE.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")
    return _HAAR_CASCADE


def _gaussian_feather_mask(h: int, w: int, sigma_frac: float = 0.3) -> np.ndarray:
    """Create a 2-D Gaussian feathered mask of shape (h, w) in [0, 1]."""
    sigma_x = w * sigma_frac
    sigma_y = h * sigma_frac
    cx, cy = w / 2.0, h / 2.0

    y = np.arange(h, dtype=np.float32)
    x = np.arange(w, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)

    mask = np.exp(-(((xv - cx) ** 2) / (2 * sigma_x ** 2) + ((yv - cy) ** 2) / (2 * sigma_y ** 2)))
    return mask


def apply_face_adaptive(
    sr_frame: np.ndarray,
    original_frame: np.ndarray,
    roi: tuple[float, float, float, float],
    blend_strength: float = 0.3,
) -> np.ndarray:
    """Detect faces within *roi* and blend original detail into the SR frame
    using a Gaussian-feathered mask.

    Parameters
    ----------
    sr_frame:
        Super-resolved frame (BGR, uint8).
    original_frame:
        Source frame (BGR, uint8).
    roi:
        Normalised ROI ``(x0, y0, x1, y1)`` to search for faces.
    blend_strength:
        Strength of original detail blending (0 = no change, 1 = fully original).
    """
    blend_strength = float(np.clip(blend_strength, 0.0, 1.0))
    result = sr_frame.copy()
    sr_h, sr_w = sr_frame.shape[:2]

    # Resize original to match SR dimensions
    orig_resized = cv2.resize(original_frame, (sr_w, sr_h), interpolation=cv2.INTER_CUBIC)

    # Extract ROI region for face detection
    x0, y0, x1, y1 = roi
    rx0 = int(round(x0 * sr_w))
    ry0 = int(round(y0 * sr_h))
    rx1 = int(round(x1 * sr_w))
    ry1 = int(round(y1 * sr_h))
    rx0, ry0 = max(0, rx0), max(0, ry0)
    rx1, ry1 = min(sr_w, rx1), min(sr_h, ry1)

    roi_crop = result[ry0:ry1, rx0:rx1]
    if roi_crop.size == 0:
        return result

    # Detect faces in the ROI
    gray_roi = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2GRAY)
    cascade = _get_face_cascade()
    faces = cascade.detectMultiScale(
        gray_roi,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    if len(faces) == 0:
        return result

    for fx, fy, fw, fh in faces:
        # Coordinates in full-frame space
        abs_x = rx0 + fx
        abs_y = ry0 + fy

        # Clamp to frame boundaries
        abs_x2 = min(sr_w, abs_x + fw)
        abs_y2 = min(sr_h, abs_y + fh)
        abs_x = max(0, abs_x)
        abs_y = max(0, abs_y)
        actual_w = abs_x2 - abs_x
        actual_h = abs_y2 - abs_y
        if actual_w <= 0 or actual_h <= 0:
            continue

        # Extract face sub-regions
        sr_face = result[abs_y:abs_y2, abs_x:abs_x2].astype(np.float32)
        orig_face = orig_resized[abs_y:abs_y2, abs_x:abs_x2].astype(np.float32)

        # Create Gaussian feathered mask (single channel -> broadcast to 3)
        mask = _gaussian_feather_mask(actual_h, actual_w)
        mask_3ch = (mask * blend_strength)[:, :, np.newaxis]

        # Blend: keep SR base, pull in original detail where mask is strong
        blended = sr_face * (1.0 - mask_3ch) + orig_face * mask_3ch
        result[abs_y:abs_y2, abs_x:abs_x2] = np.clip(blended, 0, 255).astype(np.uint8)

    return result
