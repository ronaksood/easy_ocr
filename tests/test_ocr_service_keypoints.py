"""Integration tests for OcrService.process_image_with_keypoints.

Uses mocked OCR engines to verify the end-to-end flow without
requiring actual OCR libraries to be installed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gauge_metadata.schemas.ocr_detection import OcrDetection
from gauge_metadata.services.ocr_service import OcrService


def _make_detection(
    text: str,
    cx: float,
    cy: float,
    size: float = 20.0,
    confidence: float = 0.9,
) -> OcrDetection:
    """Create an OcrDetection with a bbox centred at (cx, cy)."""
    half = size / 2
    bbox = [
        [cx - half, cy - half],
        [cx + half, cy - half],
        [cx + half, cy + half],
        [cx - half, cy + half],
    ]
    return OcrDetection(text=text, confidence=confidence, bbox=bbox)


class TestProcessImageWithKeypoints:
    """Integration tests for the keypoint-based OCR pipeline."""

    def _make_image(self, h: int = 320, w: int = 320) -> np.ndarray:
        return np.zeros((h, w, 3), dtype=np.uint8)

    @patch.object(OcrService, "__init__", lambda self: None)
    def test_correct_min_max_extraction(self) -> None:
        """Verify correct min/max when mock OCR returns known values."""
        service = OcrService()

        # Create a mock engine.
        mock_engine = MagicMock()
        service._engines = {"easy_ocr": mock_engine}

        # For the min crop region (around keypoint ~92, 225):
        # The crop will be roughly [47:137, 180:270] with radius~80, ratio=0.25
        # OCR returns "0" near the min keypoint and "PSI" as noise.
        min_detections = [
            _make_detection("0", cx=20.0, cy=20.0),
            _make_detection("PSI", cx=40.0, cy=10.0),
            _make_detection("50", cx=60.0, cy=60.0),
        ]

        # For the max crop region (around keypoint ~226, 230):
        max_detections = [
            _make_detection("400", cx=15.0, cy=15.0),
            _make_detection("BAR", cx=50.0, cy=30.0),
            _make_detection("300", cx=60.0, cy=60.0),
        ]

        # read_image_detailed is called twice: once for min crop, once for max crop.
        mock_engine.read_image_detailed.side_effect = [
            min_detections,
            max_detections,
        ]

        result = service.process_image_with_keypoints(
            engine="easy_ocr",
            image=self._make_image(),
            center=(159.06, 162.2),
            min_point=(92.42, 225.07),
            max_point=(225.7, 230.1),
            crop_ratio=0.25,
        )

        assert result.min_value is not None
        assert result.max_value is not None
        # The nearest numeric to the min keypoint should be selected.
        # The nearest numeric to the max keypoint should be selected.
        assert result.unit is None  # Unit extraction is skipped.

    @patch.object(OcrService, "__init__", lambda self: None)
    def test_no_detections_returns_none(self) -> None:
        """No OCR detections should produce None for both values."""
        service = OcrService()

        mock_engine = MagicMock()
        mock_engine.read_image_detailed.return_value = []
        service._engines = {"easy_ocr": mock_engine}

        result = service.process_image_with_keypoints(
            engine="easy_ocr",
            image=self._make_image(),
            center=(160.0, 160.0),
            min_point=(80.0, 200.0),
            max_point=(240.0, 200.0),
        )

        assert result.min_value is None
        assert result.max_value is None

    @patch.object(OcrService, "__init__", lambda self: None)
    def test_only_non_numeric_returns_none(self) -> None:
        """Only unit/label detections should produce None values."""
        service = OcrService()

        mock_engine = MagicMock()
        non_numeric = [
            _make_detection("PSI", cx=10.0, cy=10.0),
            _make_detection("Gauge", cx=30.0, cy=30.0),
        ]
        mock_engine.read_image_detailed.return_value = non_numeric
        service._engines = {"easy_ocr": mock_engine}

        result = service.process_image_with_keypoints(
            engine="easy_ocr",
            image=self._make_image(),
            center=(160.0, 160.0),
            min_point=(80.0, 200.0),
            max_point=(240.0, 200.0),
        )

        assert result.min_value is None
        assert result.max_value is None

    @patch.object(OcrService, "__init__", lambda self: None)
    def test_bytes_input(self) -> None:
        """Verify the pipeline handles bytes input correctly."""
        import cv2

        service = OcrService()

        mock_engine = MagicMock()
        mock_engine.read_image_detailed.return_value = [
            _make_detection("10", cx=10.0, cy=10.0),
        ]
        service._engines = {"easy_ocr": mock_engine}

        # Encode a test image to bytes.
        img = self._make_image()
        _, buf = cv2.imencode(".jpg", img)
        img_bytes = buf.tobytes()

        result = service.process_image_with_keypoints(
            engine="easy_ocr",
            image=img_bytes,
            center=(160.0, 160.0),
            min_point=(80.0, 200.0),
            max_point=(240.0, 200.0),
        )

        # Should not crash, and should have valid results.
        assert mock_engine.read_image_detailed.call_count == 2
