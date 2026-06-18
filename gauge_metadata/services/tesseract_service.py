import logging

import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)


class TesseractService:

    def read_image(self, image_path: str) -> list[str]:

        image = Image.open(image_path)

        text = pytesseract.image_to_string(image)

        texts = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        return texts