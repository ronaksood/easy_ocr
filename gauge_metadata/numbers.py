import re

_NUMBER_PATTERN = re.compile(r"-?\d+\.?\d*")


def extract_numbers(texts: list[str]) -> list[float]:
    """Extract all numeric values from OCR text strings.

    Supports integers, decimals, and negative numbers.
    Returns a sorted (ascending) list of unique floats.
    """
    numbers: list[float] = []
    for text in texts:
        matches = _NUMBER_PATTERN.findall(text)
        for match in matches:
            try:
                numbers.append(float(match))
            except ValueError:
                continue
    return sorted(set(numbers))
