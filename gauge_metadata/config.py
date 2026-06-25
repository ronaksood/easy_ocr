"""Centralized configuration for the gauge metadata OCR pipeline.

All configurable parameters are defined here to avoid magic numbers
throughout the codebase. Values can be overridden via environment variables.
"""

import os


# ---------------------------------------------------------------------------
# Keypoint-based cropping
# ---------------------------------------------------------------------------

CROP_RATIO: float = float(os.getenv("GAUGE_CROP_RATIO", "0.45"))
"""Ratio of gauge radius used as the half-size of the crop window.

A value of 0.45 means the crop extends 45% of the gauge radius in each
direction from the keypoint, yielding a square region whose side length
equals 90% of the gauge radius.
"""

# ---------------------------------------------------------------------------
# Numeric filtering
# ---------------------------------------------------------------------------

MAX_GAUGE_VALUE: float = float(os.getenv("GAUGE_MAX_VALUE", "10000"))
"""Maximum plausible gauge scale value.

OCR detections above this threshold are rejected as serial numbers,
model numbers, or other non-scale text.
"""

# ---------------------------------------------------------------------------
# Benchmark paths
# ---------------------------------------------------------------------------

BENCHMARK_DIR: str = os.getenv("GAUGE_BENCHMARK_DIR", "data/benchmark")
"""Directory containing benchmark gauge images."""

PIXEL_LABELS_PATH: str = os.getenv(
    "GAUGE_PIXEL_LABELS_PATH", "data/pixel_labels.json"
)
"""Path to the JSON file with keypoint annotations (center, min, max, tip)."""

GROUND_TRUTH_PATH: str = os.getenv(
    "GAUGE_GT_PATH", "data/min_max_reading.json"
)
"""Path to the JSON file with ground-truth min/max/reading values."""


# ---------------------------------------------------------------------------
# Debug visualization
# ---------------------------------------------------------------------------

ENABLE_OCR_DEBUG_VISUALIZATION: bool = (
    os.getenv("ENABLE_OCR_DEBUG_VISUALIZATION", "false").lower() == "true"
)
"""Master switch for all debug visualization output.

When False, every public function in debug_visualizer is a silent no-op.
"""

DEBUG_OUTPUT_DIR: str = os.getenv("OCR_DEBUG_OUTPUT_DIR", "debug_output")
"""Root directory for all debug artifacts."""


# ---------------------------------------------------------------------------
# Crop preprocessing
# ---------------------------------------------------------------------------

OCR_UPSCALE_FACTOR: float = float(os.getenv("GAUGE_OCR_UPSCALE_FACTOR", "3.0"))
"""Factor by which the cropped region is upscaled before OCR."""

CLAHE_CLIP_LIMIT: float = float(os.getenv("GAUGE_CLAHE_CLIP_LIMIT", "2.0"))
"""Contrast limit for CLAHE preprocessing."""

CLAHE_TILE_GRID_SIZE: tuple[int, int] = (8, 8)
"""Size of grid for CLAHE local equalization."""


