"""Centralized configuration for the gauge metadata OCR pipeline.

All configurable parameters are defined here to avoid magic numbers
throughout the codebase. Values can be overridden via environment variables.
"""

import os


# ---------------------------------------------------------------------------
# Keypoint-based cropping
# ---------------------------------------------------------------------------

CROP_RATIO: float = float(os.getenv("GAUGE_CROP_RATIO", "0.25"))
"""Ratio of gauge radius used as the half-size of the crop window.

A value of 0.25 means the crop extends 25% of the gauge radius in each
direction from the keypoint, yielding a square region whose side length
equals 50% of the gauge radius.
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
