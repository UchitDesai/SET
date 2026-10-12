from typing import Dict, List, Tuple


def compute_feature_confidence(feature: str, modality_data: Dict) -> float:
    """
    Estimate confidence of a feature based on modality reliability.
    """
    confidences = []

    for mod, data in modality_data.items():
        if feature in data["features"] and data["features"][feature]:
            confidences.append(data.get("confidence", 0.5))

    if not confidences:
        return 0.3  # low confidence if inferred but not directly observed

    return sum(confidences) / len(confidences)


def build_reasoning_graph(features: Dict, modality_data: Dict) -> Tuple[int, Dict]:
    """
    Builds a hierarchical reasoning graph:
    Feature → Indicator → Category → Final Risk
    """

    graph = {
        "features": [],
        "indicators": [],
        "categories": {},
        "final_score": 0
    }

    category_scores = {}

    def add_node(feature, indicator, category, base_contribution):
        value = float(features.get(feature, 0))

        if value <= 0:
            return

        confidence = compute_feature_confidence(feature, modality_data)

        adjusted_contribution = base_contribution * value * confidence

        # Feature layer
        graph["features"].append({
            "feature": feature,
            "value": value,
            "confidence": round(confidence, 2)
        })

        # Indicator layer
        graph["indicators"].append({
            "indicator": indicator,
            "feature": feature,
            "category": category,
            "base_contribution": base_contribution,
            "adjusted_contribution": round(adjusted_contribution, 2)
        })

        # Category aggregation
        category_scores[category] = category_scores.get(category, 0) + adjusted_contribution

    # -------- STRUCTURAL -------- #
    if features.get("num_urls", 0) > 0:
        add_node("num_urls", "Contains URL", "Structural Risk", 20)

    add_node("has_ip_url", "IP-based URL", "Structural Risk", 30)
    add_node("suspicious_tld", "Suspicious TLD", "Structural Risk", 15)
    add_node("suspicious_host", "Suspicious Host", "Structural Risk", 20)

    # -------- BEHAVIORAL -------- #
    add_node("intent_urgency", "Urgency Intent", "Behavioral Risk", 15)
    add_node("intent_threat", "Threat Intent", "Behavioral Risk", 15)

    # -------- CREDENTIAL -------- #
    add_node("intent_credential_request", "Credential Request", "Credential Risk", 25)

    # -------- FINANCIAL -------- #
    add_node("intent_financial", "Financial Intent", "Financial Risk", 10)

    # -------- CATEGORY LAYER -------- #
    for category, score in category_scores.items():
        graph["categories"][category] = round(score, 2)

    final_score = min(int(sum(category_scores.values())), 100)

    graph["final_score"] = final_score

    return final_score, graph