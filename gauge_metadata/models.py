from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GaugeMetadata:
    unit: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    all_detected_text: list[str] = field(default_factory=list)
    all_detected_numbers: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "unit": self.unit,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "all_detected_text": self.all_detected_text,
            "all_detected_numbers": self.all_detected_numbers,
        }
