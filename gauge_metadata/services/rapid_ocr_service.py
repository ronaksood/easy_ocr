import logging
import cv2
import numpy as np

from rapidocr_onnxruntime import RapidOCR

from gauge_metadata.schemas.ocr_detection import OcrDetection

logger = logging.getLogger(__name__)

class RapidOcrService:
    """Service wrapper around RapidOCR."""

    def __init__(self) -> None:
        logger.info("Initializing RapidOCR")
        self._ocr = RapidOCR()

    def _decode_image(self, image: str | bytes | np.ndarray) -> np.ndarray:
        """Decode image input to a numpy array suitable for RapidOCR.

        Args:
            image: File path, raw bytes, or numpy array.

        Returns:
            Image as a numpy array.

        Raises:
            ValueError: If bytes cannot be decoded.
        """
        if isinstance(image, np.ndarray):
            return image
        if isinstance(image, bytes):
            nparr = np.frombuffer(image, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image bytes for RapidOCR")
            return img
        return image  # type: ignore[return-value]  # file path

    def read_image(self, image: str | bytes) -> list[str]:
        """Run OCR on an image (path or bytes) and return detected text strings."""
        img = self._decode_image(image)

        result, _ = self._ocr(img)
        texts: list[str] = []

        if result:
            for item in result:
                # Structure: [bbox, text, confidence]
                text = item[1]
                if text:
                    texts.append(text)

        logger.debug("RapidOCR detected %d text regions", len(texts))
        return texts

    def read_image_detailed(
        self, image: str | bytes | np.ndarray
    ) -> list[OcrDetection]:
        """Run OCR and return detailed detections with bounding boxes.

        RapidOCR returns results as ``[bbox, text, confidence]`` where
        bbox is ``[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]``.

        Args:
            image: File path, raw bytes, or numpy array.

        Returns:
            List of :class:`OcrDetection` with text, confidence,
            and four-corner bounding polygon.
        """
        img = self._decode_image(image)
        result, _ = self._ocr(img)
        detections: list[OcrDetection] = []

        if result:
            for item in result:
                bbox_raw = item[0]
                text = item[1]
                confidence = item[2]
                if not text:
                    continue
                detections.append(
                    OcrDetection(
                        text=text,
                        confidence=float(confidence),
                        bbox=[[float(c) for c in pt] for pt in bbox_raw],
                    )
                )

        logger.debug("RapidOCR detailed: %d detections", len(detections))
        return detections
