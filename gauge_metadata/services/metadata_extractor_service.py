import logging

from gauge_metadata.schemas.metadata import GaugeMetadata
from gauge_metadata.services.ocr_service import OcrService
from gauge_metadata.services.paddle_service import PaddleService
from gauge_metadata.services.tesseract_service import TesseractService
# from gauge_metadata.services.rapid_service import RapidOcrService

from gauge_metadata.utils.numbers import extract_numbers, infer_zero
from gauge_metadata.utils.units import match_unit

logger = logging.getLogger(__name__)


class MetadataExtractorService:
    """Service that orchestrates the extraction of gauge metadata from images."""

    # def __init__(self, ocr_service: OcrService) -> None:
    #     self.ocr_service = ocr_service

    #  def __init__(self, paddle_service: PaddleService) -> None:
    #     self.paddle_service = paddle_service

    def __init__(self, tesseract_service: TesseractService) -> None:
        self.tesseract_service = tesseract_service

    # def __init__(self, rapid_service: RapidOcrService) -> None:
    #     self.rapid_service = rapid_service

    def extract_metadata(self, image_path: str) -> GaugeMetadata:
        """Execute the metadata extraction pipeline on the specified image.

        Pipeline stages:
          1. OCR service text extraction
          2. Unit matching
          3. Number extraction
          4. Zero value inference
          5. Min / max range assignment
          6. Build and return GaugeMetadata
        """
        # texts = self.ocr_service.read_image(image_path)
        # texts = self.paddle_service.read_image(image_path)
        texts = self.tesseract_service.read_image(image_path)

        # texts = self.rapid_service.read_image(image_path)
        unit = match_unit(texts)
        numbers = extract_numbers(texts)
        numbers = infer_zero(numbers)

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
