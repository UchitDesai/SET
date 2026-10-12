import re
import math
from urllib.parse import urlparse
from typing import Dict, List

from app.services.semantic_signal_mapper import detect_intent_signals


SUSPICIOUS_TLDS = [".xyz", ".top", ".ru", ".tk", ".ml", ".cf"]

SUSPICIOUS_HOST_KEYWORDS = [
    "weebly", "vercel", "github.io", "blogspot",
    "netlify", "workers.dev", "godaddysites"
]


# ---------------- URL EXTRACTION ---------------- #
def extract_urls(text: str) -> List[str]:
    return re.findall(r'https?://[^\s]+', text)


def contains_ip_based_url(urls: List[str]) -> bool:
    for url in urls:
        if re.search(r'https?://\d+\.\d+\.\d+\.\d+', url):
            return True
    return False


# ---------------- URL FEATURES ---------------- #
def url_length(url: str) -> int:
    return len(url)


def count_dots(url: str) -> int:
    return url.count(".")


def has_at_symbol(url: str) -> bool:
    return "@" in url


def suspicious_tld(url: str) -> bool:
    return any(url.endswith(tld) for tld in SUSPICIOUS_TLDS)


def url_has_https(url: str) -> bool:
    return url.startswith("https")


def digit_ratio(url: str) -> float:
    digits = sum(c.isdigit() for c in url)
    return digits / len(url) if len(url) > 0 else 0


def shannon_entropy(url: str) -> float:
    prob = [float(url.count(c)) / len(url) for c in dict.fromkeys(list(url))]
    return -sum([p * math.log2(p) for p in prob])


def is_suspicious_host(url: str) -> bool:
    try:
        domain = urlparse(url).netloc.lower()
        return any(k in domain for k in SUSPICIOUS_HOST_KEYWORDS)
    except:
        return False


# ---------------- MAIN FEATURE FUNCTION ---------------- #
def generate_feature_vector(text: str) -> Dict:
    urls = extract_urls(text)

    # -------- LANGUAGE-AGNOSTIC INTENTS -------- #
    intent_features = detect_intent_signals(text)

    features = {
        "num_urls": len(urls),
        "has_ip_url": contains_ip_based_url(urls),

        # Intent-based features (NEW CORE)
        **intent_features
    }

    # -------- URL STRUCTURAL FEATURES -------- #
    if urls:
        url = urls[0]

        features.update({
            "url_length": url_length(url),
            "num_dots": count_dots(url),
            "has_at_symbol": has_at_symbol(url),
            "suspicious_tld": suspicious_tld(url),
            "uses_https": url_has_https(url),
            "digit_ratio": digit_ratio(url),
            "url_entropy": shannon_entropy(url),
            "suspicious_host": is_suspicious_host(url),
        })
    else:
        features.update({
            "url_length": 0,
            "num_dots": 0,
            "has_at_symbol": False,
            "suspicious_tld": False,
            "uses_https": False,
            "digit_ratio": 0,
            "url_entropy": 0,
            "suspicious_host": False,
        })

    return features