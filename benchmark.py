"""Benchmark runner for keypoint-based gauge OCR pipeline.

Evaluates min/max extraction accuracy by comparing predictions against
ground-truth values across all benchmark images.

Usage (on DGX VM):
    python benchmark.py --engine easy_ocr
    python benchmark.py --engine paddle_ocr
    python benchmark.py --engine easy_ocr --crop-ratio 0.30
    python benchmark.py --engine easy_ocr --output results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2

from gauge_metadata.config import (
    BENCHMARK_DIR,
    CROP_RATIO,
    GROUND_TRUTH_PATH,
    PIXEL_LABELS_PATH,
)
from gauge_metadata.services.ocr_service import OcrService

# ── Logging setup ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────

_HASH_PREFIX_PATTERN: re.Pattern[str] = re.compile(r"^[0-9a-f]{8}-")
"""Pattern matching the 8-character hex prefix in pixel_labels image names."""

_TOLERANCE: float = 0.01
"""Relative tolerance for floating-point comparison of gauge values."""


# ── Data structures ─────────────────────────────────────────────────────────


@dataclass
class ImageResult:
    """Result of processing a single benchmark image."""

    image_name: str
    gt_min: float | None = None
    gt_max: float | None = None
    pred_min: float | None = None
    pred_max: float | None = None
    min_correct: bool = False
    max_correct: bool = False
    error: str | None = None
    time_seconds: float = 0.0


@dataclass
class BenchmarkSummary:
    """Aggregate benchmark results for an OCR engine."""

    engine: str
    crop_ratio: float
    total_images: int = 0
    min_correct: int = 0
    max_correct: int = 0
    both_correct: int = 0
    min_accuracy: float = 0.0
    max_accuracy: float = 0.0
    both_accuracy: float = 0.0
    total_time_seconds: float = 0.0
    results: list[ImageResult] = field(default_factory=list)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _strip_hash_prefix(name: str) -> str:
    """Strip the ``<8-hex-chars>-`` prefix from a pixel_labels image name.

    Example:
        ``'9f2b6c65-img1.jpg'`` → ``'img1.jpg'``
    """
    return _HASH_PREFIX_PATTERN.sub("", name)


def _values_match(
    predicted: float | None,
    ground_truth: float | None,
    tolerance: float = _TOLERANCE,
) -> bool:
    """Compare predicted and ground-truth values with relative tolerance.

    Args:
        predicted: The predicted gauge value (may be None).
        ground_truth: The ground-truth gauge value (may be None).
        tolerance: Relative tolerance for comparison.

    Returns:
        True if values match within tolerance.
    """
    if predicted is None or ground_truth is None:
        return predicted is None and ground_truth is None

    if ground_truth == 0.0:
        return abs(predicted) <= tolerance

    return abs(predicted - ground_truth) / abs(ground_truth) <= tolerance


def _load_pixel_labels(path: str) -> dict[str, dict]:
    """Load pixel_labels.json and index by stripped image name.

    Returns:
        Dict mapping ``'img1.jpg'`` → label entry with center/min/max/tip.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    index: dict[str, dict] = {}
    for entry in data:
        clean_name = _strip_hash_prefix(entry["image"])
        index[clean_name] = entry

    logger.info(
        "Loaded %d pixel label entries from %s", len(index), path
    )
    return index


def _load_ground_truth(path: str) -> dict[str, dict]:
    """Load min_max_reading.json ground-truth data.

    Returns:
        Dict mapping ``'img1.jpg'`` → ``{min_value, max_value, gt_reading}``.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info("Loaded %d ground truth entries from %s", len(data), path)
    return data


# ── Main benchmark logic ────────────────────────────────────────────────────


def run_benchmark(
    engine: str,
    crop_ratio: float = CROP_RATIO,
    benchmark_dir: str = BENCHMARK_DIR,
    pixel_labels_path: str = PIXEL_LABELS_PATH,
    gt_path: str = GROUND_TRUTH_PATH,
) -> BenchmarkSummary:
    """Run the keypoint-based OCR benchmark across all images.

    Args:
        engine: OCR engine key (``"easy_ocr"``, ``"paddle_ocr"``, etc.).
        crop_ratio: Fraction of gauge radius used as crop half-size.
        benchmark_dir: Directory containing benchmark gauge images.
        pixel_labels_path: Path to keypoint annotations JSON.
        gt_path: Path to ground-truth min/max/reading JSON.

    Returns:
        :class:`BenchmarkSummary` with per-image results and accuracy.
    """
    pixel_labels = _load_pixel_labels(pixel_labels_path)
    ground_truth = _load_ground_truth(gt_path)
    benchmark_path = Path(benchmark_dir)

    ocr_service = OcrService()
    summary = BenchmarkSummary(engine=engine, crop_ratio=crop_ratio)

    # Process each ground-truth image.
    for img_name, gt in sorted(ground_truth.items()):
        result = ImageResult(
            image_name=img_name,
            gt_min=gt.get("min_value"),
            gt_max=gt.get("max_value"),
        )

        # Check if we have keypoint labels for this image.
        if img_name not in pixel_labels:
            result.error = "No pixel labels found"
            logger.warning("Skipping %s: no pixel labels", img_name)
            summary.results.append(result)
            summary.total_images += 1
            continue

        # Check if the image file exists.
        img_path = benchmark_path / img_name
        if not img_path.exists():
            result.error = "Image file not found"
            logger.warning("Skipping %s: file not found", img_name)
            summary.results.append(result)
            summary.total_images += 1
            continue

        labels = pixel_labels[img_name]
        center = (labels["center"]["x"], labels["center"]["y"])
        min_point = (labels["min"]["x"], labels["min"]["y"])
        max_point = (labels["max"]["x"], labels["max"]["y"])

        # Load image.
        image = cv2.imread(str(img_path))
        if image is None:
            result.error = "Failed to decode image"
            logger.error("Failed to decode: %s", img_name)
            summary.results.append(result)
            summary.total_images += 1
            continue

        # Run keypoint-based OCR.
        start_time = time.monotonic()
        try:
            response = ocr_service.process_image_with_keypoints(
                engine=engine,
                image=image,
                center=center,
                min_point=min_point,
                max_point=max_point,
                crop_ratio=crop_ratio,
                image_name=img_name,
                gt_min=result.gt_min,
                gt_max=result.gt_max,
            )
            result.pred_min = response.min_value
            result.pred_max = response.max_value
        except Exception as e:
            result.error = str(e)
            logger.error("Error processing %s: %s", img_name, e)
        result.time_seconds = time.monotonic() - start_time

        # Evaluate accuracy.
        result.min_correct = _values_match(result.pred_min, result.gt_min)
        result.max_correct = _values_match(result.pred_max, result.gt_max)

        summary.total_images += 1
        if result.min_correct:
            summary.min_correct += 1
        if result.max_correct:
            summary.max_correct += 1
        if result.min_correct and result.max_correct:
            summary.both_correct += 1
        summary.total_time_seconds += result.time_seconds

        status = "✓" if result.min_correct and result.max_correct else "✗"
        logger.info(
            "%s %s: pred_min=%s gt_min=%s | pred_max=%s gt_max=%s (%.2fs)",
            status,
            img_name,
            result.pred_min,
            result.gt_min,
            result.pred_max,
            result.gt_max,
            result.time_seconds,
        )

        summary.results.append(result)

    # Compute overall accuracy.
    if summary.total_images > 0:
        summary.min_accuracy = summary.min_correct / summary.total_images
        summary.max_accuracy = summary.max_correct / summary.total_images
        summary.both_accuracy = summary.both_correct / summary.total_images

    return summary


# ── CLI entrypoint ──────────────────────────────────────────────────────────


def main() -> None:
    """CLI entrypoint for the benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Benchmark keypoint-based gauge OCR pipeline"
    )
    parser.add_argument(
        "--engine",
        type=str,
        required=True,
        choices=["easy_ocr", "paddle_ocr", "rapid_ocr", "tesseract_ocr"],
        help="OCR engine to benchmark",
    )
    parser.add_argument(
        "--crop-ratio",
        type=float,
        default=CROP_RATIO,
        help=f"Crop ratio (default: {CROP_RATIO})",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default=BENCHMARK_DIR,
        help=f"Benchmark images directory (default: {BENCHMARK_DIR})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results (default: stdout summary only)",
    )

    args = parser.parse_args()

    logger.info(
        "Starting benchmark: engine=%s, crop_ratio=%.3f",
        args.engine,
        args.crop_ratio,
    )

    summary = run_benchmark(
        engine=args.engine,
        crop_ratio=args.crop_ratio,
        benchmark_dir=args.benchmark_dir,
    )

    # Print summary.
    print("\n" + "=" * 60)
    print(f"  Benchmark Results: {args.engine}")
    print("=" * 60)
    print(f"  Total images:   {summary.total_images}")
    print(f"  Min accuracy:   {summary.min_correct}/{summary.total_images} "
          f"({summary.min_accuracy:.1%})")
    print(f"  Max accuracy:   {summary.max_correct}/{summary.total_images} "
          f"({summary.max_accuracy:.1%})")
    print(f"  Both accuracy:  {summary.both_correct}/{summary.total_images} "
          f"({summary.both_accuracy:.1%})")
    print(f"  Total time:     {summary.total_time_seconds:.1f}s")
    print(f"  Crop ratio:     {summary.crop_ratio}")
    print("=" * 60)

    # Save detailed results if requested.
    if args.output:
        output_data = asdict(summary)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, default=str)
        logger.info("Results saved to %s", args.output)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
