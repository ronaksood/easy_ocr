"""Dynamic cropping around gauge keypoints.

Provides utilities to compute a gauge-radius-relative crop region around
a detected keypoint (min or max label location), safely clamped to image
boundaries.
"""

from __future__ import annotations

import cv2
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ── Data structures ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Point:
    """A 2-D point in pixel coordinates."""

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class CropResult:
    """Result of a keypoint crop operation.

    Attributes:
        image: The cropped image as a numpy array (H×W×C).
        offset_x: X-offset of the crop origin in the original image.
        offset_y: Y-offset of the crop origin in the original image.
        x1: Left boundary of the crop in original image coordinates.
        y1: Top boundary of the crop in original image coordinates.
        x2: Right boundary of the crop in original image coordinates.
        y2: Bottom boundary of the crop in original image coordinates.
    """

    image: np.ndarray
    offset_x: int
    offset_y: int
    x1: int
    y1: int
    x2: int
    y2: int


# ── Public functions ────────────────────────────────────────────────────────


def compute_gauge_radius(center: Point, keypoint: Point) -> float:
    """Compute the Euclidean distance between the gauge centre and a keypoint.

    This distance serves as the gauge radius for scaling crop dimensions.

    Args:
        center: Centre point of the gauge dial.
        keypoint: A keypoint (min or max label location).

    Returns:
        The Euclidean distance in pixels.
    """
    return math.hypot(keypoint.x - center.x, keypoint.y - center.y)


def crop_around_keypoint(
    image: np.ndarray,
    keypoint: Point,
    center: Point,
    crop_ratio: float,
) -> CropResult:
    """Generate a dynamic square crop around a keypoint.

    The crop half-size is ``gauge_radius * crop_ratio``, where the radius
    is ``distance(center, keypoint)``.  Coordinates are clamped to image
    boundaries so the returned crop is always valid.

    Args:
        image: Source image as a numpy array (H×W×C or H×W).
        keypoint: The target keypoint to crop around.
        center: The gauge centre point (used to compute radius).
        crop_ratio: Fraction of the gauge radius used as crop half-size.

    Returns:
        A :class:`CropResult` containing the cropped image and metadata.

    Raises:
        ValueError: If the resulting crop has zero area (degenerate case).
    """
    img_h, img_w = image.shape[:2]
    radius = compute_gauge_radius(center, keypoint)
    crop_half = int(math.ceil(radius * crop_ratio))

    # Compute raw crop boundaries centred on the keypoint.
    raw_x1 = int(math.floor(keypoint.x)) - crop_half
    raw_y1 = int(math.floor(keypoint.y)) - crop_half
    raw_x2 = int(math.floor(keypoint.x)) + crop_half
    raw_y2 = int(math.floor(keypoint.y)) + crop_half

    # Clamp to image boundaries.
    x1 = max(0, raw_x1)
    y1 = max(0, raw_y1)
    x2 = min(img_w, raw_x2)
    y2 = min(img_h, raw_y2)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Degenerate crop region [{x1}:{x2}, {y1}:{y2}] for "
            f"keypoint=({keypoint.x}, {keypoint.y}), "
            f"image size=({img_w}×{img_h})"
        )

    cropped = image[y1:y2, x1:x2].copy()

    logger.info(
        "Crop around keypoint (%.1f, %.1f): radius=%.1f, "
        "crop_half=%d, region=[%d:%d, %d:%d], crop_size=%dx%d",
        keypoint.x,
        keypoint.y,
        radius,
        crop_half,
        x1,
        x2,
        y1,
        y2,
        cropped.shape[1],
        cropped.shape[0],
    )

    return CropResult(
        image=cropped,
        offset_x=x1,
        offset_y=y1,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )


def preprocess_crop(
    crop_img: np.ndarray,
    upscale_factor: float = 3.0,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Preprocess a cropped gauge region to improve OCR accuracy.

    Pipeline:
    1. Grayscale conversion.
    2. CLAHE (Contrast Limited Adaptive Histogram Equalization).
    3. N-x Upscaling using cubic interpolation.

    Args:
        crop_img: BGR crop image numpy array.
        upscale_factor: Factor to upscale the crop.
        clip_limit: Threshold for contrast limiting in CLAHE.
        tile_grid_size: Size of grid for histogram equalization.

    Returns:
        Preprocessed (grayscale, contrast-enhanced, upscaled) image.
    """
    # 1. Grayscale
    if len(crop_img.shape) == 3 and crop_img.shape[2] == 3:
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop_img.copy()

    # 2. CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    equalized = clahe.apply(gray)

    # 3. Upscale (using Cubic interpolation)
    if upscale_factor != 1.0:
        height, width = equalized.shape[:2]
        new_w = int(round(width * upscale_factor))
        new_h = int(round(height * upscale_factor))
        upscaled = cv2.resize(
            equalized, (new_w, new_h), interpolation=cv2.INTER_CUBIC
        )
        logger.info(
            "Preprocessed crop: shape %s -> %s (upscale %.1fx)",
            crop_img.shape,
            upscaled.shape,
            upscale_factor,
        )
        return upscaled

    logger.info("Preprocessed crop: shape %s -> %s", crop_img.shape, equalized.shape)
    return equalized
