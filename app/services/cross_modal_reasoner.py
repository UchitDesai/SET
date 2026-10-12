from typing import Dict, List, Callable, Optional


# ====================================================================== #
# CROSS-MODAL RULE REGISTRY                                               #
#                                                                          #
# NOVELTY — DECLARATIVE CROSS-MODAL RULE ENGINE (Patent Claim 3 / Gap 2): #
#                                                                          #
# Each rule is a self-contained dict describing a cross-modal relationship  #
# between modalities. Rules fire when their `condition` callable returns    #
# True, then apply a signed `impact` to a named `category` in the          #
# reasoning graph.                                                          #
#                                                                          #
# CLAIM: Unlike hard-coded conditional logic, a declarative rule registry  #
# allows the cross-modal reasoning layer to be:                            #
#   - Auditable: every rule has an ID and human-readable description.       #
#   - Configurable: rules can be added, disabled, or tuned at runtime       #
#     without touching the inference engine.                               #
#   - Traceable: each fired rule is recorded in graph["edges"] with its     #
#     ID, type, impact, and affected category for full audit trail.         #
#   - Extensible: new modality pairs (e.g. video + URL, QR code + text)     #
#     are added by appending to CROSS_MODAL_RULES, not by modifying         #
#     apply_cross_modal_reasoning().                                        #
#                                                                          #
# Rule schema:                                                             #
#   id          (str)  : unique identifier, e.g. "CMR-001"                 #
#   type        (str)  : "contradiction" | "reinforcement"                 #
#   source      (str)  : logical source node label (for graph edge)        #
#   target      (str)  : logical target node label (for graph edge)        #
#   relation    (str)  : edge relation label                               #
#   impact      (float): signed score delta applied to `category`          #
#                        negative = suppression, positive = amplification  #
#   category    (str)  : reasoning graph category to adjust                #
#   condition   (Callable[[set, dict], bool]):                             #
#                        receives (indicator_names, modality_data)         #
#                        returns True if rule should fire                  #
#   description (str)  : human-readable rationale (used in audit trail)    #
#   enabled     (bool) : set False to disable without removing the rule    #
# ====================================================================== #

CROSS_MODAL_RULES: List[Dict] = [

    # ------------------------------------------------------------------ #
    # CONTRADICTION RULES — negative edges, risk suppression             #
    # ------------------------------------------------------------------ #

    {
        "id":       "CMR-001",
        "type":     "contradiction",
        "source":   "safe_text",
        "target":   "malicious_url",
        "relation": "contradiction",
        "impact":   -10,
        "category": "Behavioral Risk",
        "enabled":  True,
        "description": (
            "Benign-text tone + malicious URL structure, no behavioural signal. "
            "A purely structural URL risk without social-engineering language is "
            "less likely to be a targeted phishing attempt."
        ),
        "condition": lambda ind, mod: (
            "Contains URL" in ind and
            ("Suspicious TLD" in ind or "IP-based URL" in ind) and
            "Financial Intent"   not in ind and
            "Credential Request" not in ind and
            "Urgency Intent"     not in ind
        ),
    },

    {
        "id":       "CMR-002",
        "type":     "contradiction",
        "source":   "low_behavior_signal",
        "target":   "high_structural_signal",
        "relation": "suppression",
        "impact":   -8,
        "category": "Behavioral Risk",
        "enabled":  True,
        "description": (
            "No behavioural language + IP-based URL. "
            "IP URL is suspicious structurally but without urgency/threat language "
            "the attack surface is narrow — apply mild suppression."
        ),
        "condition": lambda ind, mod: (
            "Urgency Intent" not in ind and
            "Threat Intent"  not in ind and
            "IP-based URL"   in  ind
        ),
    },

    # ------------------------------------------------------------------ #
    # REINFORCEMENT RULES — positive edges, cross-modal amplification    #
    # ------------------------------------------------------------------ #

    {
        "id":       "CMR-003",
        "type":     "reinforcement",
        "source":   "ocr_signal",
        "target":   "url_signal",
        "relation": "cross_modal_reinforcement",
        "impact":   +12,
        "category": "Credential Risk",
        "enabled":  True,
        "description": (
            "OCR credential terms + phishing URL (multilingual). "
            "Fires when an image/PDF contains credential-harvesting language AND "
            "the accompanying URL is structurally suspicious. "
            "Covers Latin, Devanagari, Arabic, CJK, Cyrillic credential terms."
        ),
        "condition": lambda ind, mod: (
            "ocr" in mod and "url" in mod and
            _ocr_has_credential(mod["ocr"]["content"]) and
            ("IP-based URL" in ind or "Suspicious TLD" in ind)
        ),
    },

    {
        "id":       "CMR-004",
        "type":     "reinforcement",
        "source":   "email_trust",
        "target":   "url_mismatch",
        "relation": "conflict_amplification",
        "impact":   +10,
        "category": "Structural Risk",
        "enabled":  True,
        "description": (
            "Trusted-brand email sender + IP-based URL mismatch. "
            "Checks only the sender portion of the email content (not the body) "
            "to avoid false positives from body text containing brand names."
        ),
        "condition": lambda ind, mod: (
            "email" in mod and "url" in mod and
            _email_spoofs_brand(mod["email"]["content"]) and
            "IP-based URL" in ind
        ),
    },

    {
        "id":       "CMR-005",
        "type":     "reinforcement",
        "source":   "text_credential",
        "target":   "ocr_credential",
        "relation": "multi_modal_convergence",
        "impact":   +8,
        "category": "Credential Risk",
        "enabled":  True,
        "description": (
            "Multi-modality credential convergence. "
            "Both OCR and raw text independently show credential-request signals "
            "alongside a suspicious URL — coordinated multi-vector attack."
        ),
        "condition": lambda ind, mod: (
            "ocr" in mod and "text" in mod and
            "Credential Request" in ind and
            ("IP-based URL" in ind or "Suspicious TLD" in ind)
        ),
    },

]


# ====================================================================== #
# PRIVATE HELPERS — shared across multiple rule conditions               #
# ====================================================================== #

# Multilingual credential signal keywords (lowercase after .lower())
_OCR_CREDENTIAL_SIGNALS = [
    # Latin
    "otp", "password", "pin", "cvv", "passcode", "verification code",
    # Devanagari (Hindi)
    "पासवर्ड", "ओटीपी",
    # Arabic
    "كلمة المرور", "رمز التحقق",
    # CJK (Chinese / Japanese)
    "密码", "验证码", "口令", "パスワード", "認証コード",
    # Cyrillic (Russian)
    "пароль", "код подтверждения",
]

_SPOOFED_BRAND_SIGNALS = [
    "bank", "support", "security", "alert", "noreply",
    "paypal", "amazon", "apple", "microsoft", "google",
]


def _ocr_has_credential(ocr_content: str) -> bool:
    text = ocr_content.lower()
    return any(sig in text for sig in _OCR_CREDENTIAL_SIGNALS)


def _email_spoofs_brand(email_content: str) -> bool:
    """
    Checks email sender/subject content for trusted-brand spoofing signals.
    Operates on the combined subject+sender string provided by modality_processor.
    """
    text = email_content.lower()
    return any(kw in text for kw in _SPOOFED_BRAND_SIGNALS)


# ====================================================================== #
# RULE ENGINE                                                             #
# ====================================================================== #

def apply_cross_modal_reasoning(graph: Dict, modality_data: Dict) -> Dict:
    """
    Applies the CROSS_MODAL_RULES registry to the reasoning graph.

    Called AFTER build_reasoning_graph() and BEFORE fuse_scores().

    Modifies:
      graph["edges"]       — one entry per fired rule, with rule ID + metadata
      graph["categories"]  — signed score adjustments, floored at 0.0
      graph["final_score"] — recalculated from updated category totals

    NOVELTY — DECLARATIVE CROSS-MODAL RULE ENGINE (Patent Claim 3):
      Each fired rule is recorded in graph["edges"] with its ID, type,
      impact, and category — providing a complete, auditable trace of every
      cross-modal interaction that influenced the final score. The LLM
      explanation layer reads these edges to describe precisely which
      modality pairs interacted and whether they contradicted or reinforced
      each other, making the explanation grounded and non-speculative.

    Rule evaluation order: rules are evaluated in CROSS_MODAL_RULES list order.
    Category floors: contradiction edges cannot drive a category below 0.
    Final score: recalculated as min(sum(max(v,0) for v in categories), 100).
    """
    if "edges" not in graph:
        graph["edges"] = []

    categories      = graph.get("categories", {})
    indicators      = graph.get("indicators", [])
    indicator_names = {i["indicator"] for i in indicators}

    fired_rule_ids = []

    for rule in CROSS_MODAL_RULES:
        if not rule.get("enabled", True):
            continue

        try:
            fires = rule["condition"](indicator_names, modality_data)
        except Exception:
            # Guard: a malformed condition lambda should never crash inference
            fires = False

        if not fires:
            continue

        # Record the edge in the graph with full audit metadata
        graph["edges"].append({
            "rule_id":     rule["id"],
            "from":        rule["source"],
            "to":          rule["target"],
            "type":        rule["type"],
            "relation":    rule["relation"],
            "impact":      round(rule["impact"], 2),
            "category":    rule.get("category"),
            "description": rule["description"],
        })

        # Apply impact to the named category, floor at 0.0
        cat = rule.get("category")
        if cat and cat in categories:
            categories[cat] = max(0.0, categories[cat] + rule["impact"])

        fired_rule_ids.append(rule["id"])

    # Record which rules fired at the graph level for LLM and debug access
    graph["fired_cross_modal_rules"] = fired_rule_ids

    # Recalculate final score from updated category values
    total_score = sum(max(v, 0.0) for v in categories.values())
    graph["final_score"] = min(int(total_score), 100)

    return graph