# Set-based vocabulary for O(1) exact word matching.
UNIT_VOCABULARY: frozenset[str] = frozenset({
    "bar",
    "mbar",
    "psi",
    "kpa",
    "mpa",
    "pa",
    "kg/cm2",
    "kg/cm²",
    "kgf/cm2",
    "kgf/cm²",
    "mmwc",
    "mmh2o",
    "inh2o",
    "mmhg",
    "inhg",
    "vac",
    "%",
    "°c",
    "°f",
    "v",
    "ma",
    "a",
    "hz",
    "khz",
    "rpm",
    "lpm",
    "l/min",
    "gpm",
})

# Only strip punctuation that realistically appears as OCR artifact
# around a genuine unit label. Deliberately excludes #, @, _, etc.
# to avoid stripping garbage chars and creating false matches like "#a" → "a".
_STRIP_CHARS = ",.;:!?\""

# Map common OCR variations/artifacts of temperature units to canonical forms
_NORMALIZE_MAP: dict[str, str] = {
    # Celsius variations -> canonical "°c"
    "°c": "°c",
    "℃": "°c",
    "oc": "°c",
    "o°c": "°c",
    "celsius": "°c",
    "c": "°c",
    
    # Fahrenheit variations -> canonical "°f"
    "°f": "°f",
    "℉": "°f",
    "of": "°f",
    "o°f": "°f",
    "fahrenheit": "°f",
    "f": "°f",
}


def match_unit(texts: list[str]) -> str | None:
    """Match OCR-detected text against the engineering unit vocabulary.

    Splits each text into word tokens and checks for exact matches,
    preventing false positives from substring matching.
    """
    for text in texts:
        tokens = text.strip().lower().split()
        for token in tokens:
            cleaned = token.strip(_STRIP_CHARS)
            normalized = _NORMALIZE_MAP.get(cleaned, cleaned)
            if normalized in UNIT_VOCABULARY:
                return normalized
    return None

