import logging

import easyocr

logger = logging.getLogger(__name__)


class OcrReader:
    """Thin wrapper around EasyOCR for gauge image text extraction."""

    def __init__(self, languages: list[str] | None = None):
        langs = languages or ["en"]
        logger.info("Initializing EasyOCR reader with languages: %s", langs)
        self._reader = easyocr.Reader(langs, verbose=False)

    def read_image(self, image_path: str) -> list[str]:
        """Run OCR on an image and return the list of detected text strings."""
        results = self._reader.readtext(image_path)
        texts = [entry[1] for entry in results if entry[1]]
        logger.debug("OCR detected %d text regions in %s", len(texts), image_path)
        return texts
