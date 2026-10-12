from typing import Dict, Tuple, List


def compute_risk_score(features: Dict) -> Tuple[int, str, List[str]]:
    """
    Compute phishing risk score based on feature vector.
    Returns:
        score (0–100),
        verdict,
        triggered_indicators
    """

    score = 0
    triggered = []

    # URL presence
    if features.get("num_urls", 0) > 0:
        score += 20
        triggered.append("Contains URL")

    # IP-based URL (very suspicious)
    if features.get("has_ip_url"):
        score += 30
        triggered.append("IP-based URL detected")

    # Suspicious keywords
    if features.get("has_suspicious_keywords"):
        score += 20
        triggered.append("Suspicious keywords detected")

    # Credential harvesting attempt
    if features.get("has_credential_request"):
        score += 25
        triggered.append("Credential request detected")

    # Urgency tone
    if features.get("has_urgency"):
        score += 15
        triggered.append("Urgency language detected")

    # Cap score at 100
    score = min(score, 100)

    # Verdict logic
    if score >= 75:
        verdict = "Likely Phishing"
    elif score >= 40:
        verdict = "Suspicious"
    else:
        verdict = "Likely Safe"

    return score, verdict, triggered