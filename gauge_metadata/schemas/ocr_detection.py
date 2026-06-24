"""Data structure for detailed OCR detection results.

Used by the keypoint-based cropping pipeline to carry bounding-box
information alongside detected text, enabling spatial candidate selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class OcrDetection:
    """A single OCR detection with spatial and confidence metadata.

    Attributes:
        text: The raw detected text string.
        confidence: OCR engine confidence score (0.0 – 1.0).
        bbox: Bounding polygon as a list of [x, y] coordinate pairs.
              Most engines return four corner points
              ``[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]``.
              Tesseract returns ``[[x, y, w, h]]`` which is normalised
              to four corners during construction in the service layer.
    """

    text: str
    confidence: float
    bbox: list[list[float]] = field(default_factory=list)
