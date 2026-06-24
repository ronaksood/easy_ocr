import logging

import cv2
import numpy as np

from gauge_metadata.config import CROP_RATIO
from gauge_metadata.schemas.metadata import GaugeMetadataResponse
from gauge_metadata.schemas.ocr_detection import OcrDetection
from gauge_metadata.services.easy_ocr_service import EasyOcrService
from gauge_metadata.services.paddle_ocr_service import PaddleOcrService
from gauge_metadata.services.rapid_ocr_service import RapidOcrService
from gauge_metadata.services.tesseract_ocr_service import TesseractOcrService
from gauge_metadata.utils.candidate_selector import extract_value_from_candidates
from gauge_metadata.utils.cropper import CropResult, Point, crop_around_keypoint
from gauge_metadata.utils.numbers import extract_numbers, infer_zero
from gauge_metadata.utils.units import match_unit

logger = logging.getLogger(__name__)


class OcrService:
    """
    Main OCR service layer that dispatches image extraction requests
    to the specified engine 
    and runs the common metadata post-processing logic.
    """

    def __init__(self) -> None:
        self._engines = {
            "easy_ocr": EasyOcrService(),
            "paddle_ocr": PaddleOcrService(),
            "rapid_ocr": RapidOcrService(),
            "tesseract_ocr": TesseractOcrService(),
        }

    def process_image(
        self,
        engine: str,
        image: str | bytes,
    ) -> GaugeMetadataResponse:
        """
        Process an image using the selected OCR engine and
        return extracted gauge metadata.
        """

        texts = self._engines[engine].read_image(image)

        unit = match_unit(texts)

        numbers = extract_numbers(texts)
        numbers = infer_zero(numbers)

        min_value: float | None = None
        max_value: float | None = None

        if len(numbers) >= 2:
            min_value = numbers[0]
            max_value = numbers[-1]
        else:
            logger.warning(
                "Less than 2 numeric values detected. "
                "min_value and max_value will be null."
            )

        return GaugeMetadataResponse(
            unit=unit,
            min_value=min_value,
            max_value=max_value,
        )

    # ── Keypoint-based pipeline ─────────────────────────────────────────

    def _extract_value_at_keypoint(
        self,
        engine_name: str,
        image: np.ndarray,
        keypoint: Point,
        center: Point,
        crop_ratio: float,
        label: str,
    ) -> float | None:
        """Run the crop → OCR → select pipeline for a single keypoint.

        Args:
            engine_name: OCR engine key (e.g. ``"easy_ocr"``).
            image: Full gauge image as a numpy array.
            keypoint: The target keypoint (min or max label location).
            center: The gauge centre point.
            crop_ratio: Fraction of gauge radius used as crop half-size.
            label: Human-readable label for logging (``"min"`` or ``"max"``).

        Returns:
            The extracted numeric value, or ``None`` if extraction fails.
        """
        logger.info(
            "Extracting %s value: keypoint=(%.1f, %.1f), center=(%.1f, %.1f)",
            label,
            keypoint.x,
            keypoint.y,
            center.x,
            center.y,
        )

        try:
            crop_result: CropResult = crop_around_keypoint(
                image, keypoint, center, crop_ratio
            )
        except ValueError:
            logger.error(
                "Failed to crop around %s keypoint (%.1f, %.1f)",
                label,
                keypoint.x,
                keypoint.y,
            )
            return None

        engine = self._engines[engine_name]
        detections: list[OcrDetection] = engine.read_image_detailed(
            crop_result.image
        )

        logger.info(
            "%s crop OCR (%s): %d detections",
            label.capitalize(),
            engine_name,
            len(detections),
        )

        value = extract_value_from_candidates(
            detections=detections,
            target=keypoint,
            offset_x=crop_result.offset_x,
            offset_y=crop_result.offset_y,
        )

        if value is None:
            logger.warning(
                "No valid numeric candidate found for %s keypoint", label
            )
        else:
            logger.info("%s value extracted: %.4f", label.capitalize(), value)

        return value

    def process_image_with_keypoints(
        self,
        engine: str,
        image: str | bytes | np.ndarray,
        center: tuple[float, float],
        min_point: tuple[float, float],
        max_point: tuple[float, float],
        crop_ratio: float = CROP_RATIO,
    ) -> GaugeMetadataResponse:
        """Process a gauge image using keypoint-based dynamic cropping.

        Instead of running OCR on the full image, this method:
        1. Crops a dynamic region around each keypoint (min and max).
        2. Runs OCR only on the cropped region.
        3. Filters to numeric candidates.
        4. Selects the candidate nearest to the keypoint.

        Args:
            engine: OCR engine key (``"easy_ocr"``, ``"paddle_ocr"``, etc.).
            image: Gauge image as file path, raw bytes, or numpy array.
            center: ``(x, y)`` centre of the gauge dial.
            min_point: ``(x, y)`` location of the minimum scale label.
            max_point: ``(x, y)`` location of the maximum scale label.
            crop_ratio: Fraction of gauge radius used as crop half-size.

        Returns:
            :class:`GaugeMetadataResponse` with extracted min/max values.
            Unit is set to ``None`` (skipped in this pipeline).
        """
        # Decode image to numpy array if needed.
        if isinstance(image, bytes):
            nparr = np.frombuffer(image, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                logger.error("Failed to decode image bytes")
                return GaugeMetadataResponse(
                    unit=None, min_value=None, max_value=None
                )
        elif isinstance(image, np.ndarray):
            img = image
        else:
            img = cv2.imread(image)
            if img is None:
                logger.error("Failed to read image from path: %s", image)
                return GaugeMetadataResponse(
                    unit=None, min_value=None, max_value=None
                )

        center_pt = Point(x=center[0], y=center[1])
        min_pt = Point(x=min_point[0], y=min_point[1])
        max_pt = Point(x=max_point[0], y=max_point[1])

        logger.info(
            "Keypoint pipeline: engine=%s, crop_ratio=%.3f, "
            "image_size=%dx%d",
            engine,
            crop_ratio,
            img.shape[1],
            img.shape[0],
        )

        min_value = self._extract_value_at_keypoint(
            engine_name=engine,
            image=img,
            keypoint=min_pt,
            center=center_pt,
            crop_ratio=crop_ratio,
            label="min",
        )

        max_value = self._extract_value_at_keypoint(
            engine_name=engine,
            image=img,
            keypoint=max_pt,
            center=center_pt,
            crop_ratio=crop_ratio,
            label="max",
        )

        logger.info(
            "Keypoint pipeline result: min_value=%s, max_value=%s",
            min_value,
            max_value,
        )

        return GaugeMetadataResponse(
            unit=None,
            min_value=min_value,
            max_value=max_value,
        )