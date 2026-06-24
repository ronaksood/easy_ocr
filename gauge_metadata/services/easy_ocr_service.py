import logging

import easyocr
import numpy as np

from gauge_metadata.schemas.ocr_detection import OcrDetection

logger = logging.getLogger(__name__)


class EasyOcrService:
    """Service wrapper around EasyOCR for gauge image text extraction."""

    def __init__(self, languages: list[str] | None = None) -> None:
        langs = languages or ["en"]
        logger.info("Initializing EasyOCR reader with languages: %s", langs)
        self._reader = easyocr.Reader(langs, verbose=False)

    def read_image(self, image: str | bytes) -> list[str]:
        """Run OCR on an image (path or bytes) and return detected text strings."""
        results = self._reader.readtext(image)
        texts = [entry[1] for entry in results if entry[1]]
        logger.debug("EasyOCR detected %d text regions", len(texts))
        return texts

    def read_image_detailed(
        self, image: str | bytes | np.ndarray
    ) -> list[OcrDetection]:
        """Run OCR and return detailed detections with bounding boxes.

        EasyOCR ``readtext`` returns a list of
        ``(bbox, text, confidence)`` tuples where bbox is
        ``[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]``.

        Args:
            image: File path, raw bytes, or numpy array.

        Returns:
            List of :class:`OcrDetection` with text, confidence,
            and four-corner bounding polygon.
        """
        results = self._reader.readtext(image)
        detections: list[OcrDetection] = []
        for bbox, text, confidence in results:
            if not text:
                continue
            detections.append(
                OcrDetection(
                    text=text,
                    confidence=float(confidence),
                    bbox=[[float(c) for c in pt] for pt in bbox],
                )
            )
        logger.debug("EasyOCR detailed: %d detections", len(detections))
        return detections
