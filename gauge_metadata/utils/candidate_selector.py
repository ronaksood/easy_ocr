"""Candidate selection for keypoint-based gauge OCR.

After running OCR on a cropped region around a keypoint, this module:
1. Filters detections to keep only numeric candidates.
2. Selects the candidate whose bounding-box centre is nearest to the
   target keypoint (in original image coordinates).
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gauge_metadata.schemas.ocr_detection import OcrDetection
from gauge_metadata.utils.cropper import Point

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────

_NUMBER_PATTERN: re.Pattern[str] = re.compile(r"^-?\d+\.?\d*$")
"""Pattern for a standalone numeric value (integer or decimal)."""

_LEADING_ARTIFACTS: set[str] = set("'\"~,;:!?_})]>#@")
_TRAILING_ARTIFACTS: set[str] = set("'\"~,;:!?_})]>#@-")
_ALLOWED_NUMERIC_CHARS: set[str] = set("0123456789.- ")

# Gauge scale values almost never exceed 4 digits.
_MAX_GAUGE_VALUE: float = 10_000


# ── Data structures ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class NumericCandidate:
    """A parsed numeric candidate with spatial metadata.

    Attributes:
        value: The parsed float value.
        text: The original OCR text.
        confidence: OCR confidence score.
        center_x: Bounding-box centre X in original image coordinates.
        center_y: Bounding-box centre Y in original image coordinates.
        distance: Euclidean distance to the target keypoint.
    """

    value: float
    text: str
    confidence: float
    center_x: float
    center_y: float
    distance: float


# ── Internal helpers ────────────────────────────────────────────────────────


def _clean_text(text: str) -> str:
    """Strip common OCR artifacts from edges of text."""
    stripped = text.strip()
    while stripped and stripped[0] in _LEADING_ARTIFACTS:
        stripped = stripped[1:]
    while stripped and stripped[-1] in _TRAILING_ARTIFACTS:
        stripped = stripped[:-1]
    return stripped.strip()


def _is_numeric_text(text: str) -> bool:
    """Check if OCR text is a clean numeric entry with no embedded letters.

    Rejects entries like ``'EN 13190'``, ``'62544HO'``, ``'CL 1.0'``.
    """
    cleaned = _clean_text(text)
    if not cleaned:
        return False
    return all(c in _ALLOWED_NUMERIC_CHARS for c in cleaned)


def _parse_numeric_value(text: str) -> float | None:
    """Attempt to parse a single numeric value from cleaned text.

    Returns:
        The float value if parsing succeeds and the magnitude is within
        the plausible gauge range, otherwise ``None``.
    """
    cleaned = _clean_text(text)
    if not cleaned:
        return None

    # Try direct parse first (handles "100", "150.5", "-10").
    try:
        value = float(cleaned)
        if abs(value) < _MAX_GAUGE_VALUE:
            return value
    except ValueError:
        pass

    # Fallback: extract the first numeric token (handles "100 PSI" edge cases
    # that somehow pass the _is_numeric_text filter due to spaces).
    tokens = cleaned.split()
    for token in tokens:
        if _NUMBER_PATTERN.match(token):
            try:
                value = float(token)
                if abs(value) < _MAX_GAUGE_VALUE:
                    return value
            except ValueError:
                continue

    return None


# ── Public functions ────────────────────────────────────────────────────────


def compute_bbox_center(bbox: list[list[float]]) -> tuple[float, float]:
    """Compute the geometric centre of an OCR bounding polygon.

    Args:
        bbox: List of ``[x, y]`` coordinate pairs forming the polygon.

    Returns:
        Tuple of ``(center_x, center_y)``.
    """
    if not bbox:
        return 0.0, 0.0
    xs = [pt[0] for pt in bbox]
    ys = [pt[1] for pt in bbox]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def filter_numeric_candidates(
    detections: list[OcrDetection],
) -> list[OcrDetection]:
    """Filter OCR detections to keep only numeric candidates.

    Rejects unit labels (PSI, BAR, MPa), model strings, and other
    non-numeric text.

    Args:
        detections: Raw OCR detections from the cropped region.

    Returns:
        Subset of detections whose text parses as a valid numeric value.
    """
    candidates: list[OcrDetection] = []
    for det in detections:
        if _is_numeric_text(det.text) and _parse_numeric_value(det.text) is not None:
            candidates.append(det)

    logger.info(
        "Numeric filtering: %d detections → %d candidates",
        len(detections),
        len(candidates),
    )
    return candidates


def select_nearest_candidate(
    candidates: list[OcrDetection],
    target: Point,
    offset_x: int = 0,
    offset_y: int = 0,
) -> NumericCandidate | None:
    """Select the numeric candidate nearest to the target keypoint.

    Bounding-box centres are transformed from crop-local coordinates to
    original image coordinates using the provided offset before computing
    distance.

    Args:
        candidates: Numeric-only OCR detections.
        target: The target keypoint in original image coordinates.
        offset_x: X-offset of the crop origin in the original image.
        offset_y: Y-offset of the crop origin in the original image.

    Returns:
        The :class:`NumericCandidate` closest to the target, or ``None``
        if no valid candidates exist.
    """
    if not candidates:
        logger.warning(
            "No numeric candidates found near keypoint (%.1f, %.1f)",
            target.x,
            target.y,
        )
        return None

    best: NumericCandidate | None = None

    for det in candidates:
        value = _parse_numeric_value(det.text)
        if value is None:
            continue

        # Transform bbox centre from crop coords to original image coords.
        crop_cx, crop_cy = compute_bbox_center(det.bbox)
        orig_cx = crop_cx + offset_x
        orig_cy = crop_cy + offset_y

        dist = math.hypot(orig_cx - target.x, orig_cy - target.y)

        candidate = NumericCandidate(
            value=value,
            text=det.text,
            confidence=det.confidence,
            center_x=orig_cx,
            center_y=orig_cy,
            distance=dist,
        )

        if best is None or dist < best.distance:
            best = candidate

    if best is not None:
        logger.info(
            "Selected candidate: text='%s', value=%.2f, distance=%.1f px, "
            "confidence=%.3f, center=(%.1f, %.1f)",
            best.text,
            best.value,
            best.distance,
            best.confidence,
            best.center_x,
            best.center_y,
        )
    else:
        logger.warning(
            "All candidates near keypoint (%.1f, %.1f) failed numeric parsing",
            target.x,
            target.y,
        )

    return best


def extract_value_from_candidates(
    detections: list[OcrDetection],
    target: Point,
    offset_x: int = 0,
    offset_y: int = 0,
) -> float | None:
    """Full candidate-selection pipeline: filter → select nearest → return value.

    Convenience function combining :func:`filter_numeric_candidates` and
    :func:`select_nearest_candidate`.

    Args:
        detections: Raw OCR detections from a cropped region.
        target: Target keypoint in original image coordinates.
        offset_x: X-offset of the crop origin.
        offset_y: Y-offset of the crop origin.

    Returns:
        The numeric value of the nearest valid candidate, or ``None``.
    """
    numeric = filter_numeric_candidates(detections)
    best = select_nearest_candidate(numeric, target, offset_x, offset_y)
    return best.value if best is not None else None
