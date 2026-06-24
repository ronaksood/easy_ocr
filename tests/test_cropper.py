"""Unit tests for gauge_metadata.utils.cropper."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gauge_metadata.utils.cropper import (
    CropResult,
    Point,
    compute_gauge_radius,
    crop_around_keypoint,
)


class TestComputeGaugeRadius:
    """Tests for compute_gauge_radius."""

    def test_known_distance(self) -> None:
        """3-4-5 right triangle yields distance of 5."""
        center = Point(x=0.0, y=0.0)
        keypoint = Point(x=3.0, y=4.0)
        assert compute_gauge_radius(center, keypoint) == pytest.approx(5.0)

    def test_same_point(self) -> None:
        """Distance from a point to itself is zero."""
        pt = Point(x=100.0, y=200.0)
        assert compute_gauge_radius(pt, pt) == pytest.approx(0.0)

    def test_horizontal_distance(self) -> None:
        """Purely horizontal displacement."""
        center = Point(x=10.0, y=50.0)
        keypoint = Point(x=110.0, y=50.0)
        assert compute_gauge_radius(center, keypoint) == pytest.approx(100.0)

    def test_vertical_distance(self) -> None:
        """Purely vertical displacement."""
        center = Point(x=50.0, y=10.0)
        keypoint = Point(x=50.0, y=90.0)
        assert compute_gauge_radius(center, keypoint) == pytest.approx(80.0)

    def test_fractional_coordinates(self) -> None:
        """Non-integer coordinates matching pixel_labels.json format."""
        center = Point(x=159.06, y=162.2)
        keypoint = Point(x=92.42, y=225.07)
        expected = math.hypot(92.42 - 159.06, 225.07 - 162.2)
        assert compute_gauge_radius(center, keypoint) == pytest.approx(expected)


class TestCropAroundKeypoint:
    """Tests for crop_around_keypoint."""

    def _make_image(self, height: int = 320, width: int = 320) -> np.ndarray:
        """Create a dummy test image."""
        return np.zeros((height, width, 3), dtype=np.uint8)

    def test_normal_crop(self) -> None:
        """Crop fully within image bounds."""
        image = self._make_image(320, 320)
        keypoint = Point(x=160.0, y=160.0)
        center = Point(x=160.0, y=80.0)  # radius = 80

        result = crop_around_keypoint(image, keypoint, center, crop_ratio=0.25)

        # crop_half = ceil(80 * 0.25) = 20
        assert isinstance(result, CropResult)
        assert result.image.shape[0] == 40  # 2 * crop_half
        assert result.image.shape[1] == 40
        assert result.offset_x == 140
        assert result.offset_y == 140

    def test_crop_at_top_left_edge(self) -> None:
        """Keypoint near top-left corner — coordinates clamped to 0."""
        image = self._make_image(320, 320)
        keypoint = Point(x=5.0, y=5.0)
        center = Point(x=160.0, y=160.0)  # radius ≈ 219

        result = crop_around_keypoint(image, keypoint, center, crop_ratio=0.25)

        # Crop should be clamped — offset at 0, not negative.
        assert result.offset_x == 0
        assert result.offset_y == 0
        assert result.x1 == 0
        assert result.y1 == 0

    def test_crop_at_bottom_right_edge(self) -> None:
        """Keypoint near bottom-right corner — coordinates clamped to image size."""
        image = self._make_image(320, 320)
        keypoint = Point(x=315.0, y=315.0)
        center = Point(x=160.0, y=160.0)  # radius ≈ 219

        result = crop_around_keypoint(image, keypoint, center, crop_ratio=0.25)

        # Right/bottom edge should not exceed image dimensions.
        assert result.x2 <= 320
        assert result.y2 <= 320

    def test_crop_preserves_content(self) -> None:
        """Verify the cropped region contains correct pixel values."""
        image = self._make_image(100, 100)
        # Paint a known region.
        image[40:60, 40:60] = [255, 0, 0]  # blue square

        keypoint = Point(x=50.0, y=50.0)
        center = Point(x=50.0, y=20.0)  # radius = 30, crop_half = 8

        result = crop_around_keypoint(image, keypoint, center, crop_ratio=0.25)

        # The crop centre should contain the blue pixel.
        crop_h, crop_w = result.image.shape[:2]
        center_y = crop_h // 2
        center_x = crop_w // 2
        assert list(result.image[center_y, center_x]) == [255, 0, 0]

    def test_degenerate_crop_raises(self) -> None:
        """Zero-area crop raises ValueError."""
        image = self._make_image(10, 10)
        keypoint = Point(x=5.0, y=5.0)
        center = Point(x=5.0, y=5.0)  # radius = 0

        with pytest.raises(ValueError, match="Degenerate crop"):
            crop_around_keypoint(image, keypoint, center, crop_ratio=0.25)

    def test_crop_ratio_affects_size(self) -> None:
        """Larger crop_ratio yields a larger crop."""
        image = self._make_image(400, 400)
        keypoint = Point(x=200.0, y=200.0)
        center = Point(x=200.0, y=100.0)  # radius = 100

        small = crop_around_keypoint(image, keypoint, center, crop_ratio=0.1)
        large = crop_around_keypoint(image, keypoint, center, crop_ratio=0.4)

        assert large.image.shape[0] > small.image.shape[0]
        assert large.image.shape[1] > small.image.shape[1]
