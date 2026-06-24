import logging
import cv2
import numpy as np
from paddleocr import PaddleOCR

from gauge_metadata.schemas.ocr_detection import OcrDetection

logger = logging.getLogger(__name__)


class PaddleOcrService:
    """Service wrapper around PaddleOCR for gauge image text extraction."""

    def __init__(self) -> None:
        logger.info("Initializing PaddleOCR")
        self._ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="en",
            use_gpu=True,  # leverage the DGX server GPUs if available
        )

    def _decode_image(self, image: str | bytes | np.ndarray) -> np.ndarray:
        """Decode image input to a numpy array suitable for PaddleOCR.

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
                raise ValueError("Failed to decode image bytes for PaddleOCR")
            return img
        return image  # type: ignore[return-value]  # file path

    def read_image(self, image: str | bytes) -> list[str]:
        """Run OCR on an image (path or bytes) and return detected text strings."""
        img = self._decode_image(image)

        result = self._ocr.ocr(img)
        texts: list[str] = []

        # PaddleOCR returns a list of pages
        if result:
            for page in result:
                if page is None:
                    continue
                for line in page:
                    # Line structure: [[[x1,y1], ...], ('Text', confidence)]
                    detected_text = line[1][0]
                    if detected_text:
                        texts.append(detected_text)

        logger.debug("PaddleOCR detected %d text regions", len(texts))
        return texts

    def read_image_detailed(
        self, image: str | bytes | np.ndarray
    ) -> list[OcrDetection]:
        """Run OCR and return detailed detections with bounding boxes.

        PaddleOCR returns per-line results as
        ``[[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ('text', confidence)]``.

        Args:
            image: File path, raw bytes, or numpy array.

        Returns:
            List of :class:`OcrDetection` with text, confidence,
            and four-corner bounding polygon.
        """
        img = self._decode_image(image)
        result = self._ocr.ocr(img)
        detections: list[OcrDetection] = []

        if result:
            for page in result:
                if page is None:
                    continue
                for line in page:
                    bbox_raw = line[0]
                    text = line[1][0]
                    confidence = line[1][1]
                    if not text:
                        continue
                    detections.append(
                        OcrDetection(
                            text=text,
                            confidence=float(confidence),
                            bbox=[[float(c) for c in pt] for pt in bbox_raw],
                        )
                    )

        logger.debug("PaddleOCR detailed: %d detections", len(detections))
        return detections
