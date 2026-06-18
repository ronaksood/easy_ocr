import logging

# from rapidocr_onnxruntime import RapidOCR

logger = logging.getLogger(__name__)


class RapidOcrService:
    """Service wrapper around RapidOCR."""

    def __init__(self) -> None:
        logger.info("Initializing RapidOCR")
        self._ocr = RapidOCR()

    def read_image(self, image_path: str) -> list[str]:
        """Run OCR and return detected text strings."""

        result, _ = self._ocr(image_path)

        texts: list[str] = []

        if result:
            for item in result:
                text = item[1]
                if text:
                    texts.append(text)

        logger.debug(
            "OCR detected %d text regions in %s",
            len(texts),
            image_path,
        )

        return texts