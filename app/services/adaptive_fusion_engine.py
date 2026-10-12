import math
from typing import Dict, Tuple
from app.services.ml_model import ml_model


# ------------------------------------------------------------------ #
# Script families that reduce NLP-pattern reliability.                #
# For these scripts, structural graph features are more trustworthy   #
# than ML probability (which was trained predominantly on Latin text). #
# Used by compute_dynamic_weights() to implement Gap 1 of the         #
# language-agnostic fusion loop.                                       #
# ------------------------------------------------------------------ #
NON_LATIN_SCRIPTS = {"ar", "hi", "ru", "zh", "th", "ja", "ko", "uk", "bg"}

# Alpha boost applied when detected script is non-Latin.
# Rationale: NLP-based ML classifiers trained on Latin corpora have
# lower calibration quality on non-Latin inputs; structural rule
# features (URL entropy, IP detection, TLD checks) are script-agnostic.
SCRIPT_ALPHA_BOOST    = 0.08
SCRIPT_BETA_SUPPRESS  = 0.05


# ---------------- SIGMOID CALIBRATION ---------------- #
def sigmoid_calibrate(x: float, k: float = 6.0) -> float:
    """
    Maps any score in [0, 1] through a sigmoid centred at 0.5.

    Purpose: normalise heterogeneous signal types (graph, ML, context) onto
    the same effective scale before fusion, without requiring session state.

    k controls steepness:
      k=6  → moderate stretch; values near 0.5 spread apart, extremes compressed.

    Properties:
      sigmoid_calibrate(0.0) ≈ 0.047
      sigmoid_calibrate(0.5) = 0.5      (fixed point — neutral stays neutral)
      sigmoid_calibrate(1.0) ≈ 0.953
    """
    return 1.0 / (1.0 + math.exp(-k * (x - 0.5)))


# ---------------- UNCERTAINTY ---------------- #
def compute_uncertainty(context: Dict) -> float:
    """
    Higher uncertainty → more reliance on rules.
    uncertainty = 1 − avg_confidence
    """
    confidence = context.get("avg_confidence", 0.5)
    return 1 - confidence


# ---------------- DYNAMIC WEIGHTS ---------------- #
def compute_dynamic_weights(
    context: Dict, ml_prob_raw: float, raw_g: float, ml_available: bool = True
) -> Tuple[float, float, float]:
    """
    Continuously adaptive weighting based on uncertainty, ML confidence,
    graph-ML agreement/disagreement coupling, and detected script family.

    NOVELTY — ADAPTIVE WEIGHT COMPUTATION (Patent Claim 2):
      Weights α, β, γ are recomputed fresh for every input instance.
      Five independent signals drive the adaptation:

        1. Input uncertainty (1 − avg_confidence):
           Low-confidence inputs (e.g. short text, low OCR quality) push α
           up so structural rules dominate when evidence quality is poor.

        2. ML confidence (|ml_prob − 0.5| × 2):
           A well-calibrated ML output (far from 0.5) earns a higher β;
           an uncertain ML output stays at its base contribution.

        3. Graph–ML disagreement coupling:
           |graph − ML| > 0.35 → conflict mode: α rises, β falls.
           |graph − ML| < 0.15 → agreement mode: reinforce dominant signal.

        4. ML availability gate:
           If ml_model.is_dummy (no trained weights on disk), β is forced
           to 0.0 and its share redistributed to α. Prevents random-init
           model noise from biasing the final score.

        5. Script-family adjustment (Gap 1 — language-agnostic loop):
           When context["script_family"] is a non-Latin ISO code (Arabic,
           Devanagari, CJK, Cyrillic, Thai), α is boosted by
           SCRIPT_ALPHA_BOOST (0.08) and β is reduced by
           SCRIPT_BETA_SUPPRESS (0.05).

           Rationale: ML classifiers trained on Latin corpora have lower
           calibration quality on non-Latin text. URL structural features
           and regex-based intent signals are script-agnostic and remain
           reliable across all languages. This closes the language-agnostic
           loop: language_detector → context["script_family"] →
           compute_dynamic_weights() → α/β shift → fuse_scores().

    Invariants: α, β, γ always positive and sum to 1.0. γ floored at 0.05.
    """
    uncertainty   = compute_uncertainty(context)
    ml_confidence = abs(ml_prob_raw - 0.5) * 2

    # Signal 4: ML availability gate
    if not ml_available:
        alpha = min(0.75, 0.55 + (uncertainty * 0.4))
        beta  = 0.0
        gamma = max(0.05, 1.0 - alpha)
        total = alpha + beta + gamma
        return alpha / total, 0.0, gamma / total

    # Signals 1 + 2: base weights
    alpha = min(0.6,  0.4 + (uncertainty   * 0.4))
    beta  = min(0.35, 0.3 + (ml_confidence * 0.3))

    # Signal 3: graph-ML coupling
    disagreement = abs(raw_g - ml_prob_raw)

    if disagreement > 0.35:
        alpha = min(0.65, alpha + 0.10)
        beta  = max(0.05, beta  - 0.10)
    elif disagreement < 0.15:
        if raw_g > 0.5:
            alpha = min(0.65, alpha + 0.05)
        else:
            beta = min(0.40, beta + 0.05)

    # Signal 5: script-family adjustment (Gap 1)
    script = context.get("script_family", "en")
    if script in NON_LATIN_SCRIPTS:
        alpha = min(0.70, alpha + SCRIPT_ALPHA_BOOST)
        beta  = max(0.0,  beta  - SCRIPT_BETA_SUPPRESS)

    gamma = max(0.05, 1.0 - alpha - beta)

    total = alpha + beta + gamma
    return alpha / total, beta / total, gamma / total


# ---------------- CATEGORY ENTROPY ---------------- #
def category_entropy(category_scores: Dict) -> float:
    """
    Shannon entropy over category risk distribution.
    High entropy = risk spread across many categories (coordinated multi-vector attack).
    Low entropy  = risk concentrated in one category (single-vector signal).
    """
    total = sum(category_scores.values())
    if total == 0:
        return 0.0

    probs = [v / total for v in category_scores.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


# ---------------- GRAPH SCORE ---------------- #
def graph_score(graph: Dict) -> float:
    """
    Derives a calibrated 0-1 score from the reasoning graph's category layer.

    NOVELTY — ENTROPY-WEIGHTED MULTI-VECTOR AMPLIFICATION (Gap 3 / Patent Claim 3):

      CLAIM: Coordinated multi-category phishing attacks — where risk signals
      appear simultaneously across Structural, Behavioral, Credential, and
      Financial risk categories — are MORE dangerous than single-category attacks
      of equal raw magnitude, because coordinated attacks are harder to block
      with single-signal defenses. This function amplifies the risk score
      proportionally to the Shannon entropy of the category distribution,
      implementing a structural multi-vector detection advantage.

      FORMULA:
        base      = sum(category_scores) / 100
                    [normalised raw risk; 100 = theoretical max raw sum]
        entropy_f = H(categories) / log2(n)
                    [normalised Shannon entropy; 0 = all risk in one category,
                     1 = risk perfectly spread across all n categories]
        weighted  = base × (1 + 0.2 × entropy_f)
                    [multi-vector boost: up to +20% for fully spread attacks]
        final     = sigmoid_calibrate(min(1.0, weighted))
                    [scale normalisation for consistent fusion input]

      PROPERTIES:
        - Single-indicator attack (entropy = 0): scores exactly `base` before
          calibration — no amplification for focused single-vector signals.
        - Four-category coordinated attack (entropy = 1.0): scores up to 20%
          higher than a single-category attack with the same raw sum.
        - Entropy boost is GATED: if base < 0.3, entropy_f is halved to prevent
          weak diffuse noise from masquerading as coordination.
        - Negative category values from cross-modal contradiction edges are
          floored at 0.0 before computation to prevent sigmoid underflow.

      CONTRAST WITH PRIOR ART:
        Simple rule-based scorers (e.g. risk_engine.py) sum indicator scores
        without considering their distribution. This function distinguishes
        between a 60-point attack concentrated in one category and a 60-point
        attack spread across all four categories — the latter scores higher
        because coordinated multi-vector attacks have a higher success rate.
    """
    category_scores = graph.get("categories", {})
    if not category_scores:
        return 0.0

    positive_scores = {k: max(v, 0.0) for k, v in category_scores.items()}
    base = sum(positive_scores.values()) / 100

    n = len(positive_scores)
    if n > 1:
        max_entropy = math.log2(n)
        entropy     = category_entropy(positive_scores)
        entropy_f   = entropy / max_entropy

        if base < 0.3:
            entropy_f *= 0.5
    else:
        entropy_f = 0.0

    weighted = base * (1 + 0.2 * entropy_f)

    return sigmoid_calibrate(min(1.0, weighted))


# ---------------- CONTEXT SCORE ---------------- #
def context_score(context: Dict) -> float:
    """
    Reflects average modality confidence as a proxy for input quality.
    Sigmoid calibrated for scale consistency.
    """
    raw = context.get("avg_confidence", 0.5)
    return sigmoid_calibrate(raw)


# ---------------- MEMORY INFLUENCE ---------------- #
def memory_influence(vs, graph: Dict, context: Dict) -> float:
    """
    Uses FAISS session memory to adjust score based on similarity to past cases.

    Query includes both category scores AND indicator names so that cases with
    identical category scores but different indicators (e.g. "IP URL" vs
    "Suspicious TLD") are correctly distinguished at retrieval time.

    Properties:
      - Distance-weighted similarity: 1 / (1 + distance)
      - Bidirectional: safe past cases reduce score
      - Intensity-aware: scaled by normalised past score magnitude
      - Confidence-scaled: low-confidence inputs trust memory less

    Returns a value in [-0.15, 0.25] after confidence scaling.
    """
    if vs is None:
        return 0.0

    categories = graph.get("categories", {})
    indicators = [i["indicator"] for i in graph.get("indicators", [])]

    query = (
        "CATEGORIES: " +
        " | ".join(f"{k}:{round(v, 2)}" for k, v in sorted(categories.items())) +
        " || INDICATORS: " +
        ", ".join(sorted(indicators))
    )

    similar = vs.search(query, top_k=3)
    if not similar:
        return 0.0

    similarity_strength = 0.0
    for item in similar:
        distance = item.get("distance", 1.0)
        similarity_strength += 1 / (1 + distance)
    similarity_strength = min(1.0, similarity_strength / len(similar))

    risk_alignment = 0.0
    for item in similar:
        meta         = item.get("metadata", {})
        verdict_meta = meta.get("verdict", "")
        score_meta   = meta.get("score", 50)
        norm_score   = score_meta / 100

        if verdict_meta == "Likely Phishing":
            risk_alignment += 0.10 * norm_score
        elif verdict_meta == "Suspicious":
            risk_alignment += 0.05 * norm_score
        elif verdict_meta == "Likely Safe":
            risk_alignment -= 0.05 * (1 - norm_score)

    raw_influence = max(-0.15, min(0.25, similarity_strength * 0.1 + risk_alignment))

    context_conf     = context.get("avg_confidence", 0.5)
    confidence_scale = 0.5 + 0.5 * context_conf

    return raw_influence * confidence_scale


# ---------------- MAIN FUSION ---------------- #
def fuse_scores(graph: Dict, features: Dict, context: Dict, vs=None):
    """
    Final risk score formula:

        raw = α·graph_score + β·ml_prob_calibrated + γ·context_score
              + memory_nudge
              − uncertainty_penalty

    All three primary signals are sigmoid-calibrated onto a shared scale.

    The language-agnostic fusion loop is now complete (Gap 1):
        language_detector → context["script_family"]
        → compute_dynamic_weights() → α/β script adjustment
        → fuse_scores() → final_score

    Debug output includes script_family and script_adjusted so the
    language-driven weight shift is observable and auditable in every response.
    """
    ml_available = not ml_model.is_dummy
    ml_prob_raw  = ml_model.predict_proba(features)

    ml_prob = sigmoid_calibrate(ml_prob_raw)
    g_score = graph_score(graph)
    c_score = context_score(context)
    m_score = memory_influence(vs, graph, context)

    raw_g = graph.get("final_score", 0) / 100

    alpha, beta, gamma = compute_dynamic_weights(
        context, ml_prob_raw, raw_g, ml_available=ml_available
    )

    disagreement = abs(raw_g - ml_prob_raw)

    uncertainty_penalty = 0.0
    if ml_available and disagreement > 0.5:
        uncertainty_penalty = 0.05 * (disagreement ** 1.5)

    raw_score = (
        alpha * g_score   +
        beta  * ml_prob   +
        gamma * c_score   +
        m_score           -
        uncertainty_penalty
    )

    final_score = int(min(max(raw_score * 100, 0), 100))

    if final_score >= 75:
        verdict = "Likely Phishing"
    elif final_score >= 40:
        verdict = "Suspicious"
    else:
        verdict = "Likely Safe"

    script_family = context.get("script_family", "en")

    debug = {
        "weights":             {"alpha": round(alpha, 3), "beta": round(beta, 3), "gamma": round(gamma, 3)},
        "graph_score":         round(g_score, 4),
        "ml_score_raw":        round(ml_prob_raw, 4),
        "ml_score_calibrated": round(ml_prob, 4),
        "ml_available":        ml_available,
        "context_score":       round(c_score, 4),
        "memory_score":        round(m_score, 4),
        "raw_g_coupling":      round(raw_g, 4),
        "disagreement":        round(disagreement, 4),
        "uncertainty_penalty": round(uncertainty_penalty, 4),
        "conflict_flag":       disagreement > 0.35,
        # Gap 1 audit trail — visible in every API response
        "script_family":       script_family,
        "script_adjusted":     script_family in NON_LATIN_SCRIPTS,
    }

    return final_score, verdict, debug