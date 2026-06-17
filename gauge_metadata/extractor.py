import logging

from .models import GaugeMetadata
from .numbers import extract_numbers
from .ocr import OcrReader
from .units import match_unit

logger = logging.getLogger(__name__)


def extract_gauge_metadata(image_path: str, ocr_reader: OcrReader) -> GaugeMetadata:
    """Extract metadata from a gauge image using OCR.

    Pipeline:
        1. OCR text extraction
        2. Unit matching
        3. Numeric extraction (sorted ascending)
        4. Min/max assignment (requires ≥ 2 numbers)
        5. Structured result
    """
    texts = ocr_reader.read_image(image_path)
    unit = match_unit(texts)
    numbers = extract_numbers(texts)

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
