from typing import Optional

# Ordered longest-first to prevent substring false positives.
# e.g. "mbar" must be checked before "bar", "kPa" before "Pa".
UNIT_VOCABULARY: tuple[str, ...] = (
    "kgf/cm²",
    "kgf/cm2",
    "kg/cm²",
    "kg/cm2",
    "mmh2o",
    "inh2o",
    "l/min",
    "mbar",
    "mmwc",
    "mmhg",
    "inhg",
    "khz",
    "lpm",
    "gpm",
    "rpm",
    "kpa",
    "mpa",
    "psi",
    "vac",
    "bar",
    "pa",
    "°c",
    "°f",
    "ma",
    "hz",
    "%",
    "v",
    "a",
)


def match_unit(texts: list[str]) -> Optional[str]:
    """Match OCR-detected text against the engineering unit vocabulary.

    Returns the first matched unit (in its canonical lowercase form) or None.
    """
    for text in texts:
        normalized = text.strip().lower()
        for unit in UNIT_VOCABULARY:
            if unit in normalized:
                return unit
    return None
