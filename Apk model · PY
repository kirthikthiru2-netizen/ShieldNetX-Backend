import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler


class ApkThreatScoringModel:
    FEATURE_NAMES = [
        "is_unparseable",
        "is_signed",
        "has_modern_signature",
        "v1_only_signature",
        "is_debuggable",
        "allows_backup",
        "min_sdk_low",
        "dangerous_permission_count",
        "has_sms_permissions",
        "has_install_packages_permission",
        "has_overlay_permission",
        "has_accessibility_permission",
        "has_device_admin_permission",
        "has_contacts_permission",
        "has_trojan_permission_combo",
        "total_permission_count",
        "mimics_known_app_name",
        "component_count",
        "multidex",
        "suspicious_code_string_hits",
        "uses_dynamic_code_loading",
        "uses_native_libs",
        "has_launcher_icon",
    ]

    # Positive weight = pushes toward "malicious". Negative = pushes
    # toward "safe". Same rule-score approach as the URL model so the
    # two endpoints are consistent in how they reason about risk.
    WEIGHTS = {
        "is_unparseable": 2.5,
        "has_trojan_permission_combo": 2.2,
        "is_debuggable": 0.9,
        "v1_only_signature": 0.8,
        "min_sdk_low": 0.5,
        "dangerous_permission_count": 0.35,   # multiplied by count
        "has_sms_permissions": 1.1,
        "has_install_packages_permission": 1.0,
        "has_overlay_permission": 1.0,
        "has_accessibility_permission": 1.2,
        "has_device_admin_permission": 1.1,
        "has_contacts_permission": 0.5,
        "mimics_known_app_name": 1.4,
        "multidex": 0.15,
        "suspicious_code_string_hits": 0.4,    # multiplied by count
        "uses_dynamic_code_loading": 0.9,
        "uses_native_libs": 0.15,
        "allows_backup": 0.1,
        "is_signed": -1.2,
        "has_modern_signature": -0.6,
        "has_launcher_icon": -0.2,
    }

    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self._train_with_synthetic_data()

    def _train_with_synthetic_data(self):
        """
        Synthetic-data bootstrap, same approach as the URL model.
        Swap this for a real labeled dataset (e.g. AndroZoo / Drebin
        features + labels) when you have one — the predict() interface
        won't need to change.
        """
        np.random.seed(7)
        n = 1000
        names = self.FEATURE_NAMES

        malicious = np.zeros((n // 2, len(names)))
        for i, name in enumerate(names):
            if name == "is_signed":
                malicious[:, i] = np.random.choice([0, 1], n // 2, p=[0.55, 0.45])
            elif name == "has_modern_signature":
                malicious[:, i] = np.random.choice([0, 1], n // 2, p=[0.8, 0.2])
            elif name in ("is_debuggable", "is_unparseable", "v1_only_signature",
                          "has_trojan_permission_combo", "mimics_known_app_name",
                          "uses_dynamic_code_loading", "has_accessibility_permission",
                          "has_device_admin_permission"):
                malicious[:, i] = np.random.choice([0, 1], n // 2, p=[0.35, 0.65])
            elif name in ("has_sms_permissions", "has_install_packages_permission",
                          "has_overlay_permission", "min_sdk_low"):
                malicious[:, i] = np.random.choice([0, 1], n // 2, p=[0.3, 0.7])
            elif name == "dangerous_permission_count":
                malicious[:, i] = np.random.randint(3, 12, n // 2)
            elif name == "total_permission_count":
                malicious[:, i] = np.random.randint(8, 30, n // 2)
            elif name == "suspicious_code_string_hits":
                malicious[:, i] = np.random.randint(0, 5, n // 2)
            elif name == "has_launcher_icon":
                malicious[:, i] = np.random.choice([0, 1], n // 2, p=[0.4, 0.6])
            else:
                malicious[:, i] = np.random.choice([0, 1], n // 2, p=[0.5, 0.5])

        safe = np.zeros((n // 2, len(names)))
        for i, name in enumerate(names):
            if name in ("is_signed", "has_modern_signature", "has_launcher_icon"):
                safe[:, i] = np.random.choice([0, 1], n // 2, p=[0.05, 0.95])
            elif name in ("is_debuggable", "is_unparseable", "v1_only_signature",
                          "has_trojan_permission_combo", "mimics_known_app_name",
                          "uses_dynamic_code_loading", "has_accessibility_permission",
                          "has_device_admin_permission", "has_sms_permissions",
                          "has_install_packages_permission", "has_overlay_permission",
                          "min_sdk_low"):
                safe[:, i] = np.random.choice([0, 1], n // 2, p=[0.92, 0.08])
            elif name == "dangerous_permission_count":
                safe[:, i] = np.random.randint(0, 3, n // 2)
            elif name == "total_permission_count":
                safe[:, i] = np.random.randint(1, 12, n // 2)
            elif name == "suspicious_code_string_hits":
                safe[:, i] = np.random.randint(0, 1, n // 2)
            else:
                safe[:, i] = np.random.choice([0, 1], n // 2, p=[0.85, 0.15])

        X = np.vstack([malicious, safe])
        y = np.array([1] * (n // 2) + [0] * (n // 2))
        X_scaled = self.scaler.fit_transform(X)

        self.model = RandomForestClassifier(
            n_estimators=150, max_depth=12, random_state=7
        )
        self.model.fit(X_scaled, y)
        print("[APK MODEL] Trained on synthetic data successfully")

    def predict(self, features: dict) -> tuple:
        vector = np.array([
            features.get(name, 0) for name in self.FEATURE_NAMES
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
        rule_score_norm = max(0, min(1, rule_score / (max_possible * 0.35)))

        final_score = (0.3 * ml_prob) + (0.7 * rule_score_norm)

        # Hard overrides — these are strong enough signals on their own
        # that blending them into a probability would under-weight them.
        if features.get("is_unparseable"):
            final_score = max(final_score, 0.9)
        if not features.get("is_signed"):
            final_score = max(final_score, 0.75)
        if features.get("has_trojan_permission_combo"):
            final_score = max(final_score, 0.8)

        threat_score = int(round(final_score * 100))
        threat_score = max(0, min(100, threat_score))
        return threat_score, signal_breakdown

    def get_reason(self, features: dict, score: int) -> str:
        reasons = []
        if features.get("is_unparseable"):
            reasons.append("APK could not be parsed (malformed or corrupted package)")
        if not features.get("is_signed"):
            reasons.append("APK is not signed")
        if features.get("v1_only_signature"):
            reasons.append("Only weak/legacy (v1) signature present")
        if features.get("is_debuggable"):
            reasons.append("App is debuggable (should not ship in production)")
        if features.get("has_trojan_permission_combo"):
            reasons.append("Requests SMS + overlay/install permissions (banking-trojan pattern)")
        if features.get("has_accessibility_permission"):
            reasons.append("Requests Accessibility Service access (commonly abused for spyware/overlays)")
        if features.get("has_device_admin_permission"):
            reasons.append("Requests Device Admin access")
        if features.get("mimics_known_app_name"):
            reasons.append("Package name mimics a well-known app")
        if features.get("uses_dynamic_code_loading"):
            reasons.append("Loads code dynamically at runtime (evades static scanning)")
        if features.get("dangerous_permission_count", 0) >= 5:
            reasons.append("Requests an unusually high number of sensitive permissions")

        if not reasons:
            if score >= 60:
                return "Multiple suspicious signals detected"
            return "No significant threat signals detected"
        return " + ".join(reasons)
