"""Batch-run gauge metadata extraction on all images in data/clean/."""

import json
import logging
import sys
from pathlib import Path

from gauge_metadata import extract_gauge_metadata
from gauge_metadata.ocr import OcrReader

IMAGE_DIR = Path(__file__).parent / "data" / "clean"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    if not IMAGE_DIR.is_dir():
        logger.error("Image directory not found: %s", IMAGE_DIR)
        sys.exit(1)

    image_files = sorted(
        f for f in IMAGE_DIR.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        logger.error("No images found in %s", IMAGE_DIR)
        sys.exit(1)

    logger.info("Found %d images in %s", len(image_files), IMAGE_DIR)

    ocr_reader = OcrReader()

    results: list[dict] = []
    for image_path in image_files:
        logger.info("Processing: %s", image_path.name)
        try:
            metadata = extract_gauge_metadata(str(image_path), ocr_reader)
            result = {"file": image_path.name, **metadata.to_dict()}
            results.append(result)
        except Exception:
            logger.exception("Failed to process %s", image_path.name)
            results.append({"file": image_path.name, "error": "processing_failed"})

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
