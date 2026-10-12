from typing import Dict, Any, Optional
from app.services.phishing_features import generate_feature_vector


# ISO codes that map to non-Latin script families.
# Mirrors NON_LATIN_SCRIPTS in adaptive_fusion_engine.py — kept here too
# so modality_processor has no import dependency on fusion logic.
_NON_LATIN_ISO = {"ar", "hi", "ru", "zh", "th", "ja", "ko", "uk", "bg"}


def process_modalities(
    raw_text: str = None,
    url: str = None,
    email_subject: str = None,
    email_sender: str = None,
    file_text: str = None
) -> Dict[str, Any]:
    """
    Process each modality separately and attach confidence scores.
    """

    modality_data = {}

    # -------- TEXT -------- #
    if raw_text:
        modality_data["text"] = {
            "content": raw_text,
            "features": generate_feature_vector(raw_text),
            "confidence": 1.0
        }

    # -------- URL -------- #
    if url:
        modality_data["url"] = {
            "content": url,
            "features": generate_feature_vector(url),
            "confidence": 1.0
        }

    # -------- EMAIL -------- #
    if email_subject or email_sender:
        email_text = f"{email_subject or ''} {email_sender or ''}"
        modality_data["email"] = {
            "content": email_text,
            "features": generate_feature_vector(email_text),
            "confidence": 0.9
        }

    # -------- OCR -------- #
    if file_text:
        length = len(file_text.strip())
        confidence = min(1.0, max(0.3, length / 500))

        modality_data["ocr"] = {
            "content": file_text,
            "features": generate_feature_vector(file_text),
            "confidence": confidence
        }

    return modality_data


def merge_features(modality_data: Dict) -> Dict:
    """
    Merge features across modalities with max pooling strategy.
    """

    merged = {}

    for modality, data in modality_data.items():
        features = data["features"]

        for key, value in features.items():
            if isinstance(value, bool) or isinstance(value, int):
                merged[key] = max(merged.get(key, 0), int(value))
            else:
                merged[key] = max(merged.get(key, 0), value)

    return merged


def compute_modality_context(modality_data: Dict, language: Optional[str] = None) -> Dict:
    """
    Extract modality-aware context for adaptive fusion.

    Gap 1 — language-agnostic loop:
      The `language` parameter (ISO 639-1 code from language_detector.py)
      is now forwarded into the context dict as `script_family`.
      adaptive_fusion_engine.compute_dynamic_weights() reads this field
      to apply script-family weight adjustments, closing the loop:

        language_detector → detect_language()
        → endpoints.py passes language to compute_modality_context()
        → context["script_family"]
        → compute_dynamic_weights() Signal 5
        → α/β adjusted for non-Latin inputs

      If language is None or "unknown", script_family defaults to "en"
      (neutral, no weight adjustment applied).
    """

    script_family = language if language and language != "unknown" else "en"
    is_non_latin  = script_family in _NON_LATIN_ISO

    context = {
        "num_modalities":  len(modality_data),
        "has_url":         "url" in modality_data,
        "has_ocr":         "ocr" in modality_data,
        "ocr_confidence":  modality_data.get("ocr", {}).get("confidence", 0),
        "avg_confidence":  (
            sum(m["confidence"] for m in modality_data.values()) / len(modality_data)
            if modality_data else 0
        ),
        # Gap 1: script family forwarded to fusion engine
        "script_family":   script_family,
        "is_non_latin":    is_non_latin,
    }

    return context