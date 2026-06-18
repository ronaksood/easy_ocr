import logging

from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)


class PaddleService:
    """Service wrapper around PaddleOCR for gauge image text extraction."""

    def __init__(self) -> None:
        logger.info("Initializing PaddleOCR")

        self._ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="en",
            use_gpu=True,  # Added to leverage the DGX server GPUs
        )

    def read_image(self, image_path: str) -> list[str]:
        """Run OCR on an image and return detected text strings."""
        result = self._ocr.ocr(image_path)

        texts: list[str] = []

        # PaddleOCR returns a list of pages.
        for page in result:
            if page is None:
                continue
            
            # Each page contains a list of lines.
            for line in page:
                # Line structure: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], ('Text', confidence)]
                # line[1] accesses the ('Text', confidence) tuple.
                # line[1][0] accesses the actual text string.
                detected_text = line[1][0]
                
                if detected_text:
                    texts.append(detected_text)

        logger.debug(
            "OCR detected %d text regions in %s",
            len(texts),
            image_path,
        )

        return texts