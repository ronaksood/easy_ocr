"""Unit tests for gauge_metadata.utils.candidate_selector."""

from __future__ import annotations

import pytest

from gauge_metadata.schemas.ocr_detection import OcrDetection
from gauge_metadata.utils.candidate_selector import (
    NumericCandidate,
    compute_bbox_center,
    extract_value_from_candidates,
    filter_numeric_candidates,
    select_nearest_candidate,
)
from gauge_metadata.utils.cropper import Point


# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_detection(
    text: str,
    cx: float = 0.0,
    cy: float = 0.0,
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


# ── Tests ───────────────────────────────────────────────────────────────────


class TestComputeBboxCenter:
    """Tests for compute_bbox_center."""

    def test_square_bbox(self) -> None:
        bbox = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
        cx, cy = compute_bbox_center(bbox)
        assert cx == pytest.approx(5.0)
        assert cy == pytest.approx(5.0)

    def test_single_point(self) -> None:
        bbox = [[42.0, 17.0]]
        cx, cy = compute_bbox_center(bbox)
        assert cx == pytest.approx(42.0)
        assert cy == pytest.approx(17.0)

    def test_empty_bbox(self) -> None:
        cx, cy = compute_bbox_center([])
        assert cx == 0.0
        assert cy == 0.0


class TestFilterNumericCandidates:
    """Tests for filter_numeric_candidates."""

    def test_filters_units(self) -> None:
        """Unit labels like PSI, BAR, MPa should be rejected."""
        detections = [
            _make_detection("PSI"),
            _make_detection("BAR"),
            _make_detection("MPa"),
            _make_detection("100"),
            _make_detection("0"),
        ]
        result = filter_numeric_candidates(detections)
        texts = [d.text for d in result]
        assert "PSI" not in texts
        assert "BAR" not in texts
        assert "MPa" not in texts
        assert "100" in texts
        assert "0" in texts

    def test_keeps_decimals(self) -> None:
        """Decimal values like 150.5 should be kept."""
        detections = [_make_detection("150.5")]
        result = filter_numeric_candidates(detections)
        assert len(result) == 1
        assert result[0].text == "150.5"

    def test_keeps_negative_numbers(self) -> None:
        """Negative gauge values (e.g., vacuum gauges) should be kept."""
        detections = [_make_detection("-10")]
        result = filter_numeric_candidates(detections)
        assert len(result) == 1

    def test_rejects_mixed_alphanumeric(self) -> None:
        """Mixed text like 'EN 13190' or 'CL 1.0' should be rejected."""
        detections = [
            _make_detection("EN 13190"),
            _make_detection("CL 1.0"),
            _make_detection("62544HO"),
        ]
        result = filter_numeric_candidates(detections)
        assert len(result) == 0

    def test_empty_input(self) -> None:
        result = filter_numeric_candidates([])
        assert result == []

    def test_rejects_huge_numbers(self) -> None:
        """Serial/model numbers exceeding MAX_GAUGE_VALUE should be rejected."""
        detections = [_make_detection("99999")]
        result = filter_numeric_candidates(detections)
        assert len(result) == 0


class TestSelectNearestCandidate:
    """Tests for select_nearest_candidate."""

    def test_selects_nearest(self) -> None:
        """Should select the candidate closest to the target keypoint."""
        candidates = [
            _make_detection("100", cx=50.0, cy=50.0),
            _make_detection("200", cx=90.0, cy=90.0),
            _make_detection("0", cx=10.0, cy=10.0),
        ]
        target = Point(x=12.0, y=12.0)

        result = select_nearest_candidate(candidates, target)

        assert result is not None
        assert result.value == pytest.approx(0.0)
        assert result.text == "0"

    def test_with_offset(self) -> None:
        """Offset should shift candidate coords to original image space."""
        # Candidate at (10, 10) in crop space → (110, 210) in original space.
        candidates = [_make_detection("50", cx=10.0, cy=10.0)]
        target = Point(x=110.0, y=210.0)

        result = select_nearest_candidate(
            candidates, target, offset_x=100, offset_y=200
        )

        assert result is not None
        assert result.value == pytest.approx(50.0)
        assert result.center_x == pytest.approx(110.0)
        assert result.center_y == pytest.approx(210.0)
        assert result.distance == pytest.approx(0.0, abs=1e-6)

    def test_empty_candidates(self) -> None:
        """No candidates should return None."""
        target = Point(x=100.0, y=100.0)
        result = select_nearest_candidate([], target)
        assert result is None

    def test_single_candidate(self) -> None:
        """Single candidate should always be returned."""
        candidates = [_make_detection("42", cx=30.0, cy=30.0)]
        target = Point(x=100.0, y=100.0)

        result = select_nearest_candidate(candidates, target)

        assert result is not None
        assert result.value == pytest.approx(42.0)


class TestExtractValueFromCandidates:
    """Tests for the convenience function extract_value_from_candidates."""

    def test_full_pipeline(self) -> None:
        """End-to-end: raw detections → filter → select → value."""
        detections = [
            _make_detection("PSI", cx=5.0, cy=5.0),
            _make_detection("100", cx=50.0, cy=50.0),
            _make_detection("0", cx=10.0, cy=10.0),
            _make_detection("BAR", cx=80.0, cy=80.0),
        ]
        target = Point(x=12.0, y=12.0)

        value = extract_value_from_candidates(detections, target)

        assert value == pytest.approx(0.0)

    def test_no_numeric_returns_none(self) -> None:
        """All non-numeric detections should return None."""
        detections = [
            _make_detection("PSI"),
            _make_detection("Gauge"),
        ]
        target = Point(x=100.0, y=100.0)

        value = extract_value_from_candidates(detections, target)

        assert value is None

    def test_empty_detections(self) -> None:
        target = Point(x=100.0, y=100.0)
        value = extract_value_from_candidates([], target)
        assert value is None
