import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Dict, Tuple

MODEL_PATH = "app/models/phishing_model.pkl"

# FIX: canonical fixed feature key order — must match exactly what
# generate_feature_vector() produces so vector length is always stable,
# regardless of whether URLs are present in the input or not.
#
# generate_feature_vector() always fills ALL of these keys (URL-absent
# keys are zero-filled in the else branch), so this list is exhaustive
# and the vector dimension never changes at prediction time.
FEATURE_KEYS = sorted([
    # URL presence / structural
    "num_urls",
    "has_ip_url",
    "url_length",
    "num_dots",
    "has_at_symbol",
    "suspicious_tld",
    "uses_https",
    "digit_ratio",
    "url_entropy",
    "suspicious_host",
    # Intent signals (from semantic_signal_mapper)
    "intent_credential_request",
    "intent_urgency",
    "intent_threat",
    "intent_financial",
])

N_FEATURES = len(FEATURE_KEYS)   # 14 — fixed at import time


class PhishingMLModel:
    """
    Logistic Regression phishing classifier.

    Fixes applied:
    1. _feature_to_vector() now uses FEATURE_KEYS — a canonical sorted list
       defined at module level — so the output vector is always N_FEATURES
       long regardless of URL presence. The old `sorted(features.keys())`
       approach produced vectors of length 10 or 18 depending on whether
       the URL branch ran, causing dimension-mismatch crashes.

    2. is_dummy property: fusion engine reads this to detect when the model
       has not been trained and collapses beta weight to zero, preventing
       noise from contaminating the adaptive score.

    3. train() helper: call once with a real phishing dataset to persist a
       trained model at MODEL_PATH. After that, _load_or_init picks it up
       automatically on every subsequent server start.
    """

    def __init__(self):
        self.model = None
        self._is_dummy = False
        self._load_or_init()

    def _load_or_init(self):
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            self._is_dummy = False
        else:
            # Dummy init: train on a zero vector labelled 0.
            # The model will always predict ~0.5 — beta weight is zeroed
            # in fusion so this has no effect on final scores.
            self.model = LogisticRegression(max_iter=1000)
            self.model.fit(
                [np.zeros(N_FEATURES), np.ones(N_FEATURES)],
                [0, 1]
            )
            self._is_dummy = True

    @property
    def is_dummy(self) -> bool:
        """True when no trained model is present. Fusion uses this to skip beta."""
        return self._is_dummy

    def predict_proba(self, features: Dict) -> float:
        vec = self._feature_to_vector(features)
        prob = self.model.predict_proba([vec])[0][1]
        return float(prob)

    def _feature_to_vector(self, features: Dict) -> np.ndarray:
        """
        Produce a fixed-length float vector using the canonical FEATURE_KEYS order.
        Missing keys default to 0.0 (safe — they are always present from
        generate_feature_vector but this guards against future schema changes).
        """
        return np.array(
            [float(features.get(k, 0.0)) for k in FEATURE_KEYS],
            dtype=np.float32
        )

    def train(self, X: list, y: list):
        """
        Train on a real dataset and persist to MODEL_PATH.

        X: list of feature dicts (as returned by generate_feature_vector)
        y: list of labels (1 = phishing, 0 = legitimate)

        Example usage:
            from app.services.ml_model import ml_model
            ml_model.train(feature_dicts, labels)
        """
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        vectors = np.array([self._feature_to_vector(f) for f in X])
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.model.fit(vectors, y)
        self._is_dummy = False
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)


# Global instance
ml_model = PhishingMLModel()