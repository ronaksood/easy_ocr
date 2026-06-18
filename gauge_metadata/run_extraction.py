"""Batch-run gauge metadata extraction on all images in data/clean/."""

import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path if run directly as a script
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# from gauge_metadata.services.rapid_service import RapidOcrService

from gauge_metadata.services.paddle_service import PaddleService
from gauge_metadata.services.metadata_extractor_service import MetadataExtractorService
from gauge_metadata.services.ocr_service import OcrService

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_ROOT / "data" / "clean"
OUTPUT_FILE = PROJECT_ROOT / "results.json"
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
        f for f in IMAGE_DIR.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        logger.error("No images found in %s", IMAGE_DIR)
        sys.exit(1)

    logger.info("Found %d images in %s", len(image_files), IMAGE_DIR)


    # ocr_service = OcrService()
    # extractor_service = MetadataExtractorService(ocr_service=ocr_service)

    paddle_service = PaddleService()
    extractor_service = MetadataExtractorService(paddle_service=paddle_service)

    # rapid_service = RapidOcrService()
    # extractor_service = MetadataExtractorService(rapid_service=rapid_service)

    results: list[dict] = []
    for image_path in image_files:
        logger.info("Processing: %s", image_path.name)
        try:
            metadata = extractor_service.extract_metadata(str(image_path))
            result = {"file": image_path.name, **metadata.to_dict()}
            results.append(result)
        except Exception:
            logger.exception("Failed to process %s", image_path.name)
            results.append(
                {"file": image_path.name, "error": "processing_failed"}
            )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Results saved to %s", OUTPUT_FILE)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
