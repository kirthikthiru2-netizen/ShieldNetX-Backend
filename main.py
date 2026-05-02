from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from feature_extractor import FeatureExtractor
from model import ThreatScoringModel

app = FastAPI(title="ShieldNetX ML Backend", version="2.0-ml")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and extractor once at startup
print("[STARTUP] Loading ML model...")
extractor = FeatureExtractor()
model = ThreatScoringModel()
print("[STARTUP] Ready!")

# ── Request / Response Models ─────────────────────────────────────────────────

class ScanRequest(BaseModel):
    url: str
    source_app: Optional[str] = "unknown"
    unknown_sender: Optional[bool] = False

class ScanResponse(BaseModel):
    threat_score: int
    reason: str
    signals: dict
    ml_version: str = "2.0-ml"

class GuardianAlertRequest(BaseModel):
    blocked_url: str
    threat_score: int
    guardian_phone: str

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "ShieldNetX ML Backend Running",
        "version": "2.0-ml",
        "signals": [
            "URL Structure Analysis",
            "Sender Trust",
            "HTML Analysis",
            "JavaScript Analysis",
            "Domain Age",
            "ML Random Forest"
        ]
    }

@app.post("/scan", response_model=ScanResponse)
def scan_url(req: ScanRequest):
    print(f"[SCAN] {req.url}")

    # Step 1: Extract all 6 signals
    features = extractor.extract(
        url=req.url,
        source_app=req.source_app,
        unknown_sender=req.unknown_sender
    )

    # Step 2: ML model scores the features
    threat_score, signal_breakdown = model.predict(features)

    # Step 3: Generate human-readable reason
    reason = model.get_reason(features, threat_score)

    print(f"[RESULT] score={threat_score} reason={reason}")

    return ScanResponse(
        threat_score=threat_score,
        reason=reason,
        signals={
            "url_structure": {
                "is_shortener": features.get("is_shortener"),
                "is_known_malicious": features.get("is_known_malicious"),
                "has_phishing_keyword": features.get("has_phishing_keyword"),
                "has_suspicious_tld": features.get("has_suspicious_tld"),
                "has_ip_address": features.get("has_ip_address"),
                "has_https": features.get("has_https"),
            },
            "sender_trust": {
                "unknown_sender": features.get("unknown_sender"),
                "from_messaging_app": features.get("from_messaging_app"),
            },
            "html_analysis": {
                "has_password_field": features.get("has_password_field"),
                "has_hidden_fields": features.get("has_hidden_fields"),
                "has_suspicious_js": features.get("has_suspicious_js"),
                "external_links_ratio": features.get("external_links_ratio"),
            },
            "domain_age_score": features.get("domain_age_score"),
            "ml_breakdown": signal_breakdown,
        }
    )

@app.post("/guardian-alert")
def guardian_alert(req: GuardianAlertRequest):
    print(f"[GUARDIAN] Phone={req.guardian_phone} "
          f"URL={req.blocked_url} Score={req.threat_score}")
    return {
        "status": "alert_sent",
        "message": f"Guardian alerted about threat score {req.threat_score}",
        "phone": req.guardian_phone,
    }

@app.get("/test")
def test_scan():
    test_urls = [
        ("https://testsafebrowsing.appspot.com/s/phishing.html", True),
        ("https://bit.ly/pay-now", True),
        ("https://verify-your-bank-account.com/login", True),
        ("https://betway.com/poker/win", True),
        ("https://github.com/shieldnetx", False),
        ("https://google.com", False),
    ]
    results = []
    for url, unknown in test_urls:
        features = extractor.extract(url, "test", unknown)
        score, _ = model.predict(features)
        reason = model.get_reason(features, score)
        results.append({
            "url": url,
            "score": score,
            "reason": reason
        })
    return results