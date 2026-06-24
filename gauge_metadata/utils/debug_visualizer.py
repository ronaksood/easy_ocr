"""Debug visualization for keypoint-based gauge OCR pipeline.

Generates annotated images and saves cropped regions to help visually
diagnose whether failures originate from crop generation, OCR, or
candidate selection.

Controlled by the ``ENABLE_OCR_DEBUG_VISUALIZATION`` flag — when
disabled, all functions are no-ops and produce zero I/O.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from gauge_metadata.config import (
    DEBUG_OUTPUT_DIR,
    ENABLE_OCR_DEBUG_VISUALIZATION,
)
from gauge_metadata.schemas.ocr_detection import OcrDetection
from gauge_metadata.utils.candidate_selector import (
    NumericCandidate,
    compute_bbox_center,
    filter_numeric_candidates,
    select_nearest_candidate,
)
from gauge_metadata.utils.cropper import CropResult, Point

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ── Colour palette (BGR for OpenCV) ─────────────────────────────────────────

_COLOR_CENTER: tuple[int, int, int] = (0, 255, 255)    # Yellow
_COLOR_MIN_PT: tuple[int, int, int] = (0, 255, 0)      # Green
_COLOR_MAX_PT: tuple[int, int, int] = (0, 0, 255)       # Red
_COLOR_MIN_CROP: tuple[int, int, int] = (0, 255, 0)     # Green
_COLOR_MAX_CROP: tuple[int, int, int] = (0, 0, 255)     # Red
_COLOR_OCR_BBOX: tuple[int, int, int] = (255, 255, 0)   # Cyan
_COLOR_SELECTED: tuple[int, int, int] = (0, 165, 255)   # Orange
_COLOR_REJECTED: tuple[int, int, int] = (128, 128, 128) # Grey
_COLOR_TEXT_BG: tuple[int, int, int] = (0, 0, 0)        # Black
_COLOR_TEXT_FG: tuple[int, int, int] = (255, 255, 255)   # White

_POINT_RADIUS: int = 6
_CROP_RECT_THICKNESS: int = 2
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE: float = 0.4
_FONT_THICKNESS: int = 1


# ── Data structures ─────────────────────────────────────────────────────────


@dataclass
class KeypointDebugInfo:
    """Debug data collected for a single keypoint (min or max).

    Attributes:
        label: ``"min"`` or ``"max"``.
        keypoint: Target keypoint in original image coordinates.
        crop_result: The crop result (coordinates, offset, image).
        all_detections: Every OCR detection from the crop.
        numeric_candidates: Detections that passed numeric filtering.
        selected_candidate: The candidate chosen by nearest-distance.
        ground_truth: The expected value (if available).
    """

    label: str
    keypoint: Point
    crop_result: CropResult | None = None
    all_detections: list[OcrDetection] = field(default_factory=list)
    numeric_candidates: list[OcrDetection] = field(default_factory=list)
    selected_candidate: NumericCandidate | None = None
    ground_truth: float | None = None


@dataclass
class ImageDebugInfo:
    """Aggregated debug data for a single gauge image."""

    image_name: str
    center: Point
    min_debug: KeypointDebugInfo | None = None
    max_debug: KeypointDebugInfo | None = None
    gauge_radius_min: float = 0.0
    gauge_radius_max: float = 0.0


# ── Directory management ───────────────────────────────────────────────────


def _ensure_debug_dirs(base_dir: str) -> dict[str, Path]:
    """Create the debug output directory tree.

    Structure::

        <base_dir>/
        ├── debug_visualizations/
        ├── debug_crops/
        │   ├── min/
        │   └── max/
        └── debug_logs/

    Returns:
        Dict mapping logical names to resolved ``Path`` objects.
    """
    base = Path(base_dir)
    dirs = {
        "visualizations": base / "debug_visualizations",
        "crops_min": base / "debug_crops" / "min",
        "crops_max": base / "debug_crops" / "max",
        "logs": base / "debug_logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ── Annotation helpers ─────────────────────────────────────────────────────


def _draw_point(
    img: np.ndarray,
    point: Point,
    color: tuple[int, int, int],
    label: str,
) -> None:
    """Draw a labelled point on the image."""
    cx, cy = int(round(point.x)), int(round(point.y))
    cv2.circle(img, (cx, cy), _POINT_RADIUS, color, -1)
    cv2.circle(img, (cx, cy), _POINT_RADIUS + 1, (0, 0, 0), 1)
    _put_label(img, label, cx + _POINT_RADIUS + 4, cy - 4, color)


def _draw_crop_rect(
    img: np.ndarray,
    crop: CropResult,
    color: tuple[int, int, int],
    label: str,
) -> None:
    """Draw a labelled crop rectangle on the image."""
    cv2.rectangle(
        img, (crop.x1, crop.y1), (crop.x2, crop.y2),
        color, _CROP_RECT_THICKNESS,
    )
    _put_label(img, label, crop.x1, crop.y1 - 6, color)


def _put_label(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    """Draw text with a dark background for readability."""
    (tw, th), _ = cv2.getTextSize(text, _FONT, _FONT_SCALE, _FONT_THICKNESS)
    # Clamp position inside image.
    x = max(0, min(x, img.shape[1] - tw - 2))
    y = max(th + 2, min(y, img.shape[0] - 2))
    cv2.rectangle(img, (x - 1, y - th - 2), (x + tw + 1, y + 2), _COLOR_TEXT_BG, -1)
    cv2.putText(img, text, (x, y), _FONT, _FONT_SCALE, color, _FONT_THICKNESS)


def _draw_ocr_detections_on_crop(
    crop_img: np.ndarray,
    all_detections: list[OcrDetection],
    numeric_candidates: list[OcrDetection],
    selected: NumericCandidate | None,
    offset_x: int,
    offset_y: int,
) -> np.ndarray:
    """Annotate a crop image with OCR bounding boxes.

    - Grey bbox + text for rejected (non-numeric) detections.
    - Cyan bbox + text for numeric candidates that were not selected.
    - Orange bbox + text for the selected candidate.
    """
    annotated = crop_img.copy()
    candidate_texts = {d.text for d in numeric_candidates}
    selected_text = selected.text if selected else None

    for det in all_detections:
        if len(det.bbox) < 2:
            continue

        pts = np.array(det.bbox, dtype=np.int32)

        if det.text == selected_text:
            color = _COLOR_SELECTED
            label = f"SELECTED: {det.text} ({det.confidence:.2f})"
        elif det.text in candidate_texts:
            color = _COLOR_OCR_BBOX
            label = f"{det.text} ({det.confidence:.2f})"
        else:
            color = _COLOR_REJECTED
            label = f"[X] {det.text} ({det.confidence:.2f})"

        cv2.polylines(annotated, [pts], True, color, 1)
        cx, cy = compute_bbox_center(det.bbox)
        _put_label(annotated, label, int(cx), int(cy) - 8, color)

    return annotated


# ── Core public API ─────────────────────────────────────────────────────────


def collect_keypoint_debug(
    label: str,
    keypoint: Point,
    crop_result: CropResult | None,
    all_detections: list[OcrDetection],
    ground_truth: float | None = None,
    offset_x: int = 0,
    offset_y: int = 0,
) -> KeypointDebugInfo:
    """Collect debug data for a single keypoint extraction.

    Runs the same filtering/selection logic used in the main pipeline
    so the debug output exactly mirrors production behaviour.

    Args:
        label: ``"min"`` or ``"max"``.
        keypoint: Target keypoint in original image coordinates.
        crop_result: Output of ``crop_around_keypoint()``.
        all_detections: Raw OCR detections from the crop.
        ground_truth: Expected value for comparison.
        offset_x: Crop X offset in original image coords.
        offset_y: Crop Y offset in original image coords.

    Returns:
        Populated :class:`KeypointDebugInfo`.
    """
    numeric = filter_numeric_candidates(all_detections)
    selected = select_nearest_candidate(numeric, keypoint, offset_x, offset_y)

    return KeypointDebugInfo(
        label=label,
        keypoint=keypoint,
        crop_result=crop_result,
        all_detections=list(all_detections),
        numeric_candidates=list(numeric),
        selected_candidate=selected,
        ground_truth=ground_truth,
    )


def save_debug_artifacts(
    image: np.ndarray,
    debug_info: ImageDebugInfo,
    base_dir: str = DEBUG_OUTPUT_DIR,
) -> None:
    """Generate and save all debug artifacts for a single image.

    Creates:
    - Annotated visualization image with keypoints and crop rects.
    - Raw cropped images for min and max.
    - Annotated crop images with OCR bounding boxes.
    - JSON log with all detection metadata.

    Args:
        image: Original full gauge image (BGR numpy array).
        debug_info: Collected debug data for this image.
        base_dir: Root debug output directory.
    """
    if not ENABLE_OCR_DEBUG_VISUALIZATION:
        return

    dirs = _ensure_debug_dirs(base_dir)
    stem = Path(debug_info.image_name).stem  # e.g. "img1"

    # ── 1. Annotated visualization of the full image ────────────────────
    vis = image.copy()

    # Draw centre.
    _draw_point(vis, debug_info.center, _COLOR_CENTER, "CENTER")

    # Draw min keypoint + crop rect.
    if debug_info.min_debug:
        md = debug_info.min_debug
        _draw_point(vis, md.keypoint, _COLOR_MIN_PT, "MIN")
        if md.crop_result:
            _draw_crop_rect(vis, md.crop_result, _COLOR_MIN_CROP, "min_crop")

    # Draw max keypoint + crop rect.
    if debug_info.max_debug:
        xd = debug_info.max_debug
        _draw_point(vis, xd.keypoint, _COLOR_MAX_PT, "MAX")
        if xd.crop_result:
            _draw_crop_rect(vis, xd.crop_result, _COLOR_MAX_CROP, "max_crop")

    # Add summary text in top-left corner.
    lines = [f"Image: {debug_info.image_name}"]
    for kd in [debug_info.min_debug, debug_info.max_debug]:
        if kd is None:
            continue
        sel_val = f"{kd.selected_candidate.value}" if kd.selected_candidate else "None"
        gt_val = f"{kd.ground_truth}" if kd.ground_truth is not None else "?"
        match = "✓" if (
            kd.selected_candidate
            and kd.ground_truth is not None
            and abs(kd.selected_candidate.value - kd.ground_truth) < 0.01
        ) else "✗"
        lines.append(
            f"{kd.label.upper()}: pred={sel_val} gt={gt_val} {match} "
            f"| OCR={len(kd.all_detections)} num={len(kd.numeric_candidates)}"
        )

    for i, line in enumerate(lines):
        _put_label(vis, line, 4, 14 + i * 16, _COLOR_TEXT_FG)

    vis_path = dirs["visualizations"] / f"{stem}_visualization.jpg"
    cv2.imwrite(str(vis_path), vis)
    logger.info("Saved visualization: %s", vis_path)

    # ── 2. Save crop images ─────────────────────────────────────────────
    for kd, crop_dir_key in [
        (debug_info.min_debug, "crops_min"),
        (debug_info.max_debug, "crops_max"),
    ]:
        if kd is None or kd.crop_result is None:
            continue

        # Raw crop.
        raw_path = dirs[crop_dir_key] / f"{stem}.jpg"
        cv2.imwrite(str(raw_path), kd.crop_result.image)

        # Annotated crop with OCR boxes.
        annotated_crop = _draw_ocr_detections_on_crop(
            crop_img=kd.crop_result.image,
            all_detections=kd.all_detections,
            numeric_candidates=kd.numeric_candidates,
            selected=kd.selected_candidate,
            offset_x=kd.crop_result.offset_x,
            offset_y=kd.crop_result.offset_y,
        )
        ann_path = dirs[crop_dir_key] / f"{stem}_annotated.jpg"
        cv2.imwrite(str(ann_path), annotated_crop)

    # ── 3. JSON log ─────────────────────────────────────────────────────
    log_data = _build_log_entry(debug_info)
    log_path = dirs["logs"] / f"{stem}_debug.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, default=str)


def _build_log_entry(info: ImageDebugInfo) -> dict:
    """Serialize debug info to a JSON-friendly dict."""
    entry: dict = {
        "image_name": info.image_name,
        "center": {"x": info.center.x, "y": info.center.y},
        "gauge_radius_min": info.gauge_radius_min,
        "gauge_radius_max": info.gauge_radius_max,
    }

    for kd in [info.min_debug, info.max_debug]:
        if kd is None:
            continue
        key = kd.label
        kd_entry: dict = {
            "keypoint": {"x": kd.keypoint.x, "y": kd.keypoint.y},
            "ground_truth": kd.ground_truth,
            "crop": None,
            "ocr_detections_count": len(kd.all_detections),
            "numeric_candidates_count": len(kd.numeric_candidates),
            "all_detections": [],
            "numeric_candidates": [],
            "selected_candidate": None,
        }

        if kd.crop_result:
            kd_entry["crop"] = {
                "x1": kd.crop_result.x1,
                "y1": kd.crop_result.y1,
                "x2": kd.crop_result.x2,
                "y2": kd.crop_result.y2,
                "width": kd.crop_result.x2 - kd.crop_result.x1,
                "height": kd.crop_result.y2 - kd.crop_result.y1,
                "offset_x": kd.crop_result.offset_x,
                "offset_y": kd.crop_result.offset_y,
            }

        for det in kd.all_detections:
            kd_entry["all_detections"].append({
                "text": det.text,
                "confidence": det.confidence,
                "bbox": det.bbox,
                "bbox_center": list(compute_bbox_center(det.bbox)),
            })

        for det in kd.numeric_candidates:
            kd_entry["numeric_candidates"].append({
                "text": det.text,
                "confidence": det.confidence,
                "bbox_center": list(compute_bbox_center(det.bbox)),
            })

        if kd.selected_candidate:
            sc = kd.selected_candidate
            kd_entry["selected_candidate"] = {
                "text": sc.text,
                "value": sc.value,
                "confidence": sc.confidence,
                "center_x": sc.center_x,
                "center_y": sc.center_y,
                "distance_to_keypoint": sc.distance,
            }

        entry[key] = kd_entry

    return entry
