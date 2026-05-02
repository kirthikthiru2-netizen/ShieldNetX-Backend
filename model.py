import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

class ThreatScoringModel:

    FEATURE_NAMES = [
        "is_shortener",
        "is_known_malicious",
        "has_phishing_keyword",
        "has_gambling_keyword",
        "is_trusted_domain",
        "url_length",
        "has_ip_address",
        "has_at_symbol",
        "has_double_slash",
        "subdomain_count",
        "has_https",
        "has_suspicious_tld",
        "unknown_sender",
        "from_messaging_app",
        "has_password_field",
        "has_hidden_fields",
        "has_suspicious_js",
        "external_links_ratio",
        "has_favicon_mismatch",
        "page_load_failed",
        "domain_age_score",
    ]

    WEIGHTS = {
        "is_known_malicious":     1.0,
        "has_phishing_keyword":   1.8,
        "is_shortener":           2.0,
        "has_suspicious_js":      0.85,
        "has_password_field":     0.8,
        "has_suspicious_tld":     0.75,
        "has_ip_address":         0.75,
        "has_gambling_keyword":   2.0,
        "unknown_sender":         1.0,
        "has_at_symbol":          0.65,
        "has_double_slash":       0.6,
        "has_hidden_fields":      0.55,
        "external_links_ratio":   0.5,
        "from_messaging_app":     0.4,
        "domain_age_score":       0.5,
        "url_length":             0.3,
        "subdomain_count":        0.4,
        "page_load_failed":       0.3,
        "has_favicon_mismatch":   0.3,
        "has_https":             -0.3,
        "is_trusted_domain":     -1.0,
    }

    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self._train_with_synthetic_data()

    def _train_with_synthetic_data(self):
        """
        Train using synthetic labeled data.
        In production this would use real phishing datasets
        like PhishTank or OpenPhish.
        """
        np.random.seed(42)
        n_samples = 1000

        # Generate phishing samples (label=1)
        phishing = np.zeros((n_samples // 2, len(self.FEATURE_NAMES)))
        for i, name in enumerate(self.FEATURE_NAMES):
            if name in ["is_known_malicious", "has_phishing_keyword",
                        "is_shortener", "has_suspicious_js",
                        "has_password_field"]:
                phishing[:, i] = np.random.choice(
                    [0, 1], size=n_samples // 2, p=[0.2, 0.8])
            elif name in ["is_trusted_domain", "has_https"]:
                phishing[:, i] = np.random.choice(
                    [0, 1], size=n_samples // 2, p=[0.9, 0.1])
            elif name == "domain_age_score":
                phishing[:, i] = np.random.uniform(0.7, 1.0, n_samples // 2)
            else:
                phishing[:, i] = np.random.uniform(0, 0.5, n_samples // 2)

        # Generate safe samples (label=0)
        safe = np.zeros((n_samples // 2, len(self.FEATURE_NAMES)))
        for i, name in enumerate(self.FEATURE_NAMES):
            if name == "is_trusted_domain":
                safe[:, i] = np.random.choice(
                    [0, 1], size=n_samples // 2, p=[0.3, 0.7])
            elif name == "has_https":
                safe[:, i] = np.random.choice(
                    [0, 1], size=n_samples // 2, p=[0.1, 0.9])
            elif name == "domain_age_score":
                safe[:, i] = np.random.uniform(0.0, 0.3, n_samples // 2)
            elif name in ["is_known_malicious", "has_phishing_keyword",
                          "is_shortener", "has_suspicious_js"]:
                safe[:, i] = np.random.choice(
                    [0, 1], size=n_samples // 2, p=[0.95, 0.05])
            else:
                safe[:, i] = np.random.uniform(0, 0.3, n_samples // 2)

        X = np.vstack([phishing, safe])
        y = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2))

        X_scaled = self.scaler.fit_transform(X)

        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_scaled, y)
        print("[MODEL] Trained on synthetic data successfully")

    def predict(self, features: dict) -> tuple:
        vector = np.array([
            features.get(name, 0)
            for name in self.FEATURE_NAMES
        ]).reshape(1, -1)

        vector_scaled = self.scaler.transform(vector)
        ml_prob = self.model.predict_proba(vector_scaled)[0][1]

        rule_score = 0.0
        signal_breakdown = {}
        for name in self.FEATURE_NAMES:
            val = features.get(name, 0)
            weight = self.WEIGHTS.get(name, 0)
            contribution = val * weight
            rule_score += contribution
            signal_breakdown[name] = round(contribution, 3)

        max_possible = sum(w for w in self.WEIGHTS.values() if w > 0)
        rule_score_norm = max(0, min(1, rule_score / (max_possible * 0.25)))

        final_score = (0.3 * ml_prob) + (0.7 * rule_score_norm)

        if features.get("is_known_malicious"):
            final_score = 1.0
        if features.get("is_trusted_domain") and not features.get("has_phishing_keyword"):
            final_score = min(final_score, 0.15)

        threat_score = int(final_score * 100)
        return threat_score, signal_breakdown

    def get_reason(self, features: dict, score: int) -> str:
        reasons = []
        if features.get("is_known_malicious"):
            reasons.append("Known malicious domain")
        if features.get("has_phishing_keyword"):
            reasons.append("Phishing keyword detected")
        if features.get("is_shortener"):
            reasons.append("URL shortener used")
        if features.get("has_suspicious_js"):
            reasons.append("Suspicious JavaScript found")
        if features.get("has_password_field"):
            reasons.append("Fake login form detected")
        if features.get("has_suspicious_tld"):
            reasons.append("Suspicious domain extension")
        if features.get("has_ip_address"):
            reasons.append("IP address used instead of domain")
        if features.get("has_gambling_keyword"):
            reasons.append("Gambling/scam content detected")
        if features.get("unknown_sender"):
            reasons.append("Link from unknown sender")
        if features.get("domain_age_score", 0) > 0.7:
            reasons.append("Very new domain (high risk)")
        if not reasons:
            if score >= 70:
                return "Multiple suspicious signals detected"
            return "No significant threat signals detected"
        return " + ".join(reasons)