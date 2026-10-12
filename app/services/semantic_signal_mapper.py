import re
from typing import Dict


# ---------------- SIGNAL MAP ---------------- #
# Organised by intent, then by language/script family.
# Each sub-list is a separate language group so scoring is normalised
# PER LANGUAGE GROUP — adding more languages does not dilute scores for
# existing ones.
#
# FIX (score dilution): the original code computed score = matches / len(patterns)
# across the flat list. Adding Arabic patterns would push len() up and reduce
# the score per Hindi or English match. The new structure normalises within
# each language group independently, then takes the maximum across groups.
# A single strong hit in one language is never diluted by the size of another.

SIGNAL_MAP = {
    "credential_request": {
        "latin":     [r"\botp\b", r"\bpassword\b", r"\bpin\b", r"\bcvv\b",
                      r"\bsecret\b", r"\bpasscode\b", r"\bverification code\b",
                      r"senha", r"codigo", r"contraseña", r"mot de passe"],
        "devanagari":[r"पासवर्ड", r"ओटीपी", r"पिन", r"सत्यापन कोड"],
        "arabic":    [r"كلمة المرور", r"رمز التحقق", r"رقم سري", r"\bOTP\b"],
        "cyrillic":  [r"пароль", r"код подтверждения", r"пин"],
        "cjk":       [r"密码", r"验证码", r"口令", r"パスワード", r"認証コード"],
    },
    "urgency": {
        "latin":     [r"urgent", r"immediately", r"act now", r"right away",
                      r"expire[sd]?", r"within \d+ hour", r"last chance",
                      r"urgente", r"inmediato", r"dès maintenant"],
        "devanagari":[r"तुरंत", r"अभी", r"जल्दी", r"अंतिम अवसर"],
        "arabic":    [r"عاجل", r"فوراً", r"على الفور", r"في أقرب وقت"],
        "cyrillic":  [r"срочно", r"немедленно", r"прямо сейчас"],
        "cjk":       [r"紧急", r"立即", r"马上", r"すぐに", r"至急"],
    },
    "threat": {
        "latin":     [r"suspend", r"block", r"limit", r"terminat", r"clos",
                      r"restrict", r"unauthori[sz]ed", r"deactivat",
                      r"bloqueado", r"limitado", r"suspendu"],
        "devanagari":[r"निलंबित", r"ब्लॉक", r"सीमित", r"बंद"],
        "arabic":    [r"تعليق", r"حظر", r"محدود", r"مغلق"],
        "cyrillic":  [r"заблокирован", r"приостановлен", r"ограничен"],
        "cjk":       [r"封号", r"冻结", r"暂停", r"停用", r"ブロック"],
    },
    "financial": {
        "latin":     [r"bank", r"account", r"payment", r"verify", r"credit",
                      r"debit", r"transaction", r"invoice", r"refund",
                      r"banco", r"cuenta", r"pago", r"banque", r"compte"],
        "devanagari":[r"बैंक", r"खाता", r"भुगतान", r"लेनदेन"],
        "arabic":    [r"بنك", r"حساب", r"دفع", r"معاملة", r"تحويل"],
        "cyrillic":  [r"банк", r"счёт", r"платёж", r"перевод"],
        "cjk":       [r"银行", r"账户", r"付款", r"转账", r"振込", r"口座"],
    },
}


def _score_group(text: str, patterns: list) -> float:
    """
    Count regex hits within a single language group and normalise to [0, 1].
    Uses a soft cap: hitting half the patterns in a group already scores ~1.0
    via a square-root curve so a single-language attack isn't under-penalised
    for having fewer patterns than a multi-language group.
    """
    if not patterns:
        return 0.0
    hits = sum(1 for p in patterns if re.search(p, text))
    # Square-root normalisation: 1 hit out of 10 = 0.32, 4/10 = 0.63, 10/10 = 1.0
    return min(1.0, (hits / len(patterns)) ** 0.5)


def detect_intent_signals(text: str) -> Dict:
    """
    Detect phishing intent signals across all supported script families.

    Returns one float per intent in [0, 1], where the score is the MAXIMUM
    across all language groups — a strong hit in any one script is sufficient.

    FIX: per-group normalisation replaces the flat list so adding language
    groups never dilutes the score for existing languages.

    Supported scripts: Latin (en/es/pt/fr), Devanagari (hi), Arabic, Cyrillic, CJK.
    """
    text_lower = text.lower()
    signals = {}

    for intent, groups in SIGNAL_MAP.items():
        group_scores = [
            _score_group(text_lower, patterns)
            for patterns in groups.values()
        ]
        # Maximum across groups: any language's signal is sufficient
        signals[f"intent_{intent}"] = round(max(group_scores), 2)

    return signals