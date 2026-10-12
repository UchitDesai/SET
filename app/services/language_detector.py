from langdetect import detect_langs, LangDetectException
import re


# ---------------- SCRIPT DETECTORS ---------------- #
# Script-priority detection is more reliable than langdetect for short texts
# and is language-model-free — no translation dependency.

def contains_devanagari(text: str) -> bool:
    """Devanagari script — covers Hindi, Marathi, Nepali."""
    return bool(re.search(r'[\u0900-\u097F]', text))


def contains_arabic(text: str) -> bool:
    """Arabic script — covers Arabic, Urdu, Farsi, Pashto."""
    return bool(re.search(r'[\u0600-\u06FF\u0750-\u077F]', text))


def contains_cyrillic(text: str) -> bool:
    """Cyrillic script — covers Russian, Ukrainian, Bulgarian, Serbian."""
    return bool(re.search(r'[\u0400-\u04FF]', text))


def contains_cjk(text: str) -> bool:
    """CJK unified ideographs — covers Chinese, Japanese kanji, Korean hanja."""
    return bool(re.search(r'[\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]', text))


def contains_thai(text: str) -> bool:
    """Thai script."""
    return bool(re.search(r'[\u0E00-\u0E7F]', text))


# Ordered priority list: (detector_fn, iso_code)
# Evaluated top-down; first match wins. Latin/langdetect is the fallback.
_SCRIPT_PRIORITY = [
    (contains_devanagari, "hi"),
    (contains_arabic,     "ar"),
    (contains_cyrillic,   "ru"),
    (contains_cjk,        "zh"),
    (contains_thai,       "th"),
]


def detect_language(text: str) -> str:
    """
    Multilingual script-priority language detection.

    Priority order:
      1. Script-level Unicode range check — reliable for non-Latin scripts,
         CPU-free, no model dependency, robust on very short texts.
      2. langdetect probabilistic model — used only for Latin-script inputs
         where script alone cannot distinguish language.

    FIX: Arabic, Cyrillic, CJK, and Thai script detection added alongside
    the existing Devanagari check, completing coverage for major phishing
    target languages globally.

    Returns an ISO 639-1 language code or "unknown".
    """
    if not text or len(text.strip()) < 10:
        return "unknown"

    # Step 1: script-priority checks (order matters — most distinctive first)
    for detector, lang_code in _SCRIPT_PRIORITY:
        if detector(text):
            return lang_code

    # Step 2: langdetect for Latin-script languages
    try:
        langs = detect_langs(text)
        if not langs:
            return "unknown"

        best = langs[0]

        # Low-confidence Latin detection defaults to English rather than unknown
        # — better UX, and English phishing patterns fire correctly regardless.
        if best.prob < 0.65:
            return "en"

        return best.lang

    except LangDetectException:
        return "unknown"