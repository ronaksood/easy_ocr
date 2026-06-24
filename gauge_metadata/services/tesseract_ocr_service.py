import io
import logging

import cv2
import numpy as np
import pytesseract
from PIL import Image

from gauge_metadata.schemas.ocr_detection import OcrDetection

logger = logging.getLogger(__name__)


class TesseractOcrService:
    """Service wrapper around Tesseract for gauge image text extraction."""

    def read_image(self, image: str | bytes) -> list[str]:
        """Run OCR on an image (path or bytes) and return detected text lines."""
        if isinstance(image, bytes):
            img = Image.open(io.BytesIO(image))
        else:
            img = Image.open(image)

        text = pytesseract.image_to_string(img)
        texts = [line.strip() for line in text.splitlines() if line.strip()]
        logger.debug("Tesseract OCR detected %d text regions", len(texts))
        return texts

    def read_image_detailed(
        self, image: str | bytes | np.ndarray
    ) -> list[OcrDetection]:
        """Run OCR and return detailed detections with bounding boxes.

        Uses ``pytesseract.image_to_data`` to obtain per-word bounding
        boxes.  The ``(x, y, w, h)`` rectangles are converted to
        four-corner polygons for consistency with other OCR engines.

        Args:
            image: File path, raw bytes, or numpy array.

        Returns:
            List of :class:`OcrDetection` with text, confidence,
            and four-corner bounding polygon.
        """
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif isinstance(image, bytes):
            pil_img = Image.fromarray(
                cv2.cvtColor(
                    cv2.imdecode(
                        np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR
                    ),
                    cv2.COLOR_BGR2RGB,
                )
            )
        else:
            pil_img = Image.open(image)

        data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
        detections: list[OcrDetection] = []

        n_boxes = len(data["text"])
        for i in range(n_boxes):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])

            # Tesseract returns -1 confidence for empty/invalid entries.
            if not text or conf < 0:
                continue

            x = float(data["left"][i])
            y = float(data["top"][i])
            w = float(data["width"][i])
            h = float(data["height"][i])

            # Convert (x, y, w, h) to four-corner polygon.
            bbox = [
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h],
            ]

            detections.append(
                OcrDetection(
                    text=text,
                    confidence=conf / 100.0,  # normalise to 0.0–1.0
                    bbox=bbox,
                )
            )

        logger.debug("Tesseract detailed: %d detections", len(detections))
        return detections
