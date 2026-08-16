import os
import tempfile

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from feature_extractor import FeatureExtractor
from model import ThreatScoringModel
from apk_feature_extractor import APKFeatureExtractor
from apk_model import ApkThreatScoringModel

app = FastAPI(title="ShieldNetX ML Backend", version="2.1-ml")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and extractors once at startup
print("[STARTUP] Loading ML models...")
extractor = FeatureExtractor()
model = ThreatScoringModel()
apk_extractor = APKFeatureExtractor()
apk_model = ApkThreatScoringModel()
print("[STARTUP] Ready!")

MAX_APK_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB safety cap

# ── Request / Response Models ─────────────────────────────────────────────────

class ScanRequest(BaseModel):
    url: str
    source_app: Optional[str] = "unknown"
    unknown_sender: Optional[bool] = False


class ScanResponse(BaseModel):
    threat_score: int
    reason: str
    signals: dict
    ml_version: str = "2.1-ml"


class GuardianAlertRequest(BaseModel):
    blocked_url: str
    threat_score: int
    guardian_phone: str


class ApkScanResponse(BaseModel):
    threat_score: int
    reason: str
    is_malicious: bool
    signals: dict
    ml_version: str = "2.1-ml"

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "ShieldNetX ML Backend Running",
        "version": "2.1-ml",
        "signals": [
            "URL Structure Analysis",
            "Sender Trust",
            "HTML Analysis",
            "JavaScript Analysis",
            "Domain Age",
            "APK Static Analysis",
            "ML Random Forest",
        ],
    }


@app.post("/scan", response_model=ScanResponse)
def scan_url(req: ScanRequest):
    print(f"[SCAN] {req.url}")

    features = extractor.extract(
        url=req.url,
        source_app=req.source_app,
        unknown_sender=req.unknown_sender,
    )

    threat_score, signal_breakdown = model.predict(features)
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
        },
    )


@app.post("/analyze-apk", response_model=ApkScanResponse)
async def analyze_apk(file: UploadFile = File(...)):
    print(f"[APK-SCAN] Received file: {file.filename}")

    # Basic filename / extension sanity check
    if file.filename and not file.filename.lower().endswith(".apk"):
        return ApkScanResponse(
            threat_score=70,
            reason="File does not have a .apk extension",
            is_malicious=True,
            signals={"error": "invalid_extension"},
        )

    tmp_path = None
    try:
        contents = await file.read()

        if len(contents) == 0:
            return ApkScanResponse(
                threat_score=80,
                reason="Empty file received",
                is_malicious=True,
                signals={"error": "empty_file"},
            )

        if len(contents) > MAX_APK_SIZE_BYTES:
            return ApkScanResponse(
                threat_score=60,
                reason="File exceeds maximum allowed size for analysis",
                is_malicious=True,
                signals={"error": "file_too_large", "size_bytes": len(contents)},
            )

        # Write to a temp file — androguard needs a file path, not raw bytes
        with tempfile.NamedTemporaryFile(suffix=".apk", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        features = apk_extractor.extract(tmp_path)
        threat_score, signal_breakdown = apk_model.predict(features)
        reason = apk_model.get_reason(features, threat_score)

        print(f"[APK-RESULT] file={file.filename} score={threat_score} reason={reason}")

        return ApkScanResponse(
            threat_score=threat_score,
            reason=reason,
            is_malicious=threat_score >= 60,
            signals={
                "integrity": {
                    "is_unparseable": features.get("is_unparseable"),
                    "is_signed": features.get("is_signed"),
                    "has_modern_signature": features.get("has_modern_signature"),
                    "v1_only_signature": features.get("v1_only_signature"),
                    "is_debuggable": features.get("is_debuggable"),
                },
                "permissions": {
                    "dangerous_permission_count": features.get("dangerous_permission_count"),
                    "total_permission_count": features.get("total_permission_count"),
                    "has_sms_permissions": features.get("has_sms_permissions"),
                    "has_install_packages_permission": features.get("has_install_packages_permission"),
                    "has_overlay_permission": features.get("has_overlay_permission"),
                    "has_accessibility_permission": features.get("has_accessibility_permission"),
                    "has_device_admin_permission": features.get("has_device_admin_permission"),
                    "has_trojan_permission_combo": features.get("has_trojan_permission_combo"),
                },
                "identity": {
                    "mimics_known_app_name": features.get("mimics_known_app_name"),
                    "has_launcher_icon": features.get("has_launcher_icon"),
                },
                "code_analysis": {
                    "multidex": features.get("multidex"),
                    "uses_native_libs": features.get("uses_native_libs"),
                    "uses_dynamic_code_loading": features.get("uses_dynamic_code_loading"),
                    "suspicious_code_string_hits": features.get("suspicious_code_string_hits"),
                },
                "ml_breakdown": signal_breakdown,
            },
        )

    except Exception as e:
        print(f"[APK-ERROR] {e}")
        # Fail closed: if we can't analyze it, treat it as suspicious
        # rather than silently letting it through.
        return ApkScanResponse(
            threat_score=65,
            reason=f"Analysis failed — treating as suspicious ({str(e)[:100]})",
            is_malicious=True,
            signals={"error": str(e)[:200]},
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


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
            "reason": reason,
        })
    return results
