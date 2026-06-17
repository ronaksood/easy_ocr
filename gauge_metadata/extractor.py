import logging

from .models import GaugeMetadata
from .numbers import extract_numbers
from .ocr import OcrReader
from .units import match_unit

logger = logging.getLogger(__name__)


def _infer_zero(numbers: list[float]) -> list[float]:
    """Infer a missing 0 starting value for gauge scales.

    Gauge scales are arithmetic progressions (0, 10, 20, ...).
    OCR commonly misses the '0' marking near the dial edge.
    If the smallest detected value equals the step size of the
    sequence, 0 was very likely missed.
    """
    if len(numbers) < 3 or 0.0 in numbers or numbers[0] <= 0:
        return numbers

    diffs = [numbers[i + 1] - numbers[i] for i in range(len(numbers) - 1)]
    positive_diffs = sorted(d for d in diffs if d > 0)
    if not positive_diffs:
        return numbers

    step = positive_diffs[len(positive_diffs) // 2]  # median

    if step > 0 and abs(numbers[0] - step) / step <= 0.15:
        return [0.0] + numbers

    return numbers


def extract_gauge_metadata(image_path: str, ocr_reader: OcrReader) -> GaugeMetadata:
    """Extract metadata from a gauge image using OCR.

    Pipeline:
        1. OCR text extraction
        2. Unit matching
        3. Numeric extraction (filtered + sorted ascending)
        4. Zero inference for gauge scales
        5. Min/max assignment (requires >= 2 numbers)
        6. Structured result
    """
    texts = ocr_reader.read_image(image_path)
    unit = match_unit(texts)
    numbers = extract_numbers(texts)
    numbers = _infer_zero(numbers)

    if not unit:
        logger.warning("No engineering unit detected in %s", image_path)

    min_value: float | None = None
    max_value: float | None = None

    if len(numbers) >= 2:
        min_value = numbers[0]
        max_value = numbers[-1]
    else:
        logger.warning(
            "Fewer than 2 numeric values detected in %s (found %d), "
            "min/max will be null",
            image_path,
            len(numbers),
        )

    return GaugeMetadata(
        unit=unit,
        min_value=min_value,
        max_value=max_value,
        all_detected_text=texts,
        all_detected_numbers=numbers,
    )
