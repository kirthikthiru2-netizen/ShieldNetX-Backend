from androguard.core.apk import APK


DANGEROUS_PERMISSIONS = {
    "android.permission.READ_SMS",
    "android.permission.SEND_SMS",
    "android.permission.RECEIVE_SMS",
    "android.permission.READ_CONTACTS",
    "android.permission.WRITE_CONTACTS",
    "android.permission.CALL_PHONE",
    "android.permission.READ_CALL_LOG",
    "android.permission.CAMERA",
    "android.permission.RECORD_AUDIO",
    "android.permission.ACCESS_FINE_LOCATION",
    "android.permission.ACCESS_COARSE_LOCATION",
    "android.permission.SYSTEM_ALERT_WINDOW",
    "android.permission.REQUEST_INSTALL_PACKAGES",
    "android.permission.BIND_ACCESSIBILITY_SERVICE",
    "android.permission.BIND_DEVICE_ADMIN",
    "android.permission.WRITE_EXTERNAL_STORAGE",
    "android.permission.READ_EXTERNAL_STORAGE",
    "android.permission.READ_PHONE_STATE",
}

KNOWN_APP_NAMES = {
    "whatsapp", "facebook", "instagram", "gmail", "google play",
    "chrome", "youtube", "paytm", "phonepe", "google pay",
    "amazon", "flipkart", "sbi", "hdfc", "icici bank",
}

SUSPICIOUS_CODE_STRINGS = [
    "DexClassLoader", "PathClassLoader", "loadClass",
    "Runtime.exec", "ProcessBuilder", "reflect.Method.invoke",
    "getSystemService(\"device_policy\")", "su\n", "/system/bin/su",
]


class APKFeatureExtractor:

    def extract(self, apk_path: str) -> dict:
        features = {name: 0 for name in [
            "is_unparseable", "is_signed", "has_modern_signature",
            "v1_only_signature", "is_debuggable", "allows_backup",
            "min_sdk_low", "dangerous_permission_count",
            "has_sms_permissions", "has_install_packages_permission",
            "has_overlay_permission", "has_accessibility_permission",
            "has_device_admin_permission", "has_contacts_permission",
            "has_trojan_permission_combo", "total_permission_count",
            "mimics_known_app_name", "component_count", "multidex",
            "suspicious_code_string_hits", "uses_dynamic_code_loading",
            "uses_native_libs", "has_launcher_icon",
        ]}

        try:
            apk = APK(apk_path)
        except Exception:
            features["is_unparseable"] = 1
            return features

        # ── Signing ──────────────────────────────────────────────
        try:
            v1 = apk.is_signed_v1()
            v2 = apk.is_signed_v2()
            v3 = apk.is_signed_v3()
            features["is_signed"] = int(v1 or v2 or v3)
            features["has_modern_signature"] = int(v2 or v3)
            features["v1_only_signature"] = int(v1 and not (v2 or v3))
        except Exception:
            pass

        # ── Manifest flags ───────────────────────────────────────
        try:
            features["is_debuggable"] = int(bool(apk.get_attribute_value(
                "application", "debuggable")))
        except Exception:
            pass

        try:
            allow_backup = apk.get_attribute_value("application", "allowBackup")
            # AndroidManifest defaults allowBackup to true when unset
            features["allows_backup"] = int(allow_backup != "false")
        except Exception:
            features["allows_backup"] = 1

        try:
            min_sdk = int(apk.get_min_sdk_version() or 21)
            features["min_sdk_low"] = int(min_sdk < 21)
        except Exception:
            pass

        # ── Permissions ──────────────────────────────────────────
        try:
            perms = set(apk.get_permissions())
            features["total_permission_count"] = len(perms)
            dangerous = perms & DANGEROUS_PERMISSIONS
            features["dangerous_permission_count"] = len(dangerous)

            has_sms = any("SMS" in p for p in perms)
            has_install = "android.permission.REQUEST_INSTALL_PACKAGES" in perms
            has_overlay = "android.permission.SYSTEM_ALERT_WINDOW" in perms
            has_accessibility = "android.permission.BIND_ACCESSIBILITY_SERVICE" in perms
            has_device_admin = "android.permission.BIND_DEVICE_ADMIN" in perms
            has_contacts = any("CONTACTS" in p for p in perms)

            features["has_sms_permissions"] = int(has_sms)
            features["has_install_packages_permission"] = int(has_install)
            features["has_overlay_permission"] = int(has_overlay)
            features["has_accessibility_permission"] = int(has_accessibility)
            features["has_device_admin_permission"] = int(has_device_admin)
            features["has_contacts_permission"] = int(has_contacts)

            # Classic banking-trojan pattern: SMS + (overlay or install)
            features["has_trojan_permission_combo"] = int(
                has_sms and (has_overlay or has_install))
        except Exception:
            pass

        # ── Identity / mimicry ───────────────────────────────────
        try:
            app_name = (apk.get_app_name() or "").strip().lower()
            features["mimics_known_app_name"] = int(
                any(known in app_name for known in KNOWN_APP_NAMES)
            )
        except Exception:
            pass

        # ── Components ───────────────────────────────────────────
        try:
            activities = apk.get_activities() or []
            services = apk.get_services() or []
            receivers = apk.get_receivers() or []
            providers = apk.get_providers() or []
            features["component_count"] = (
                len(activities) + len(services) + len(receivers) + len(providers)
            )
        except Exception:
            pass

        try:
            main_activity = apk.get_main_activity()
            features["has_launcher_icon"] = int(bool(main_activity))
        except Exception:
            pass

        # ── Code structure ───────────────────────────────────────
        try:
            dex_files = [f for f in apk.get_files() if f.endswith(".dex")]
            features["multidex"] = int(len(dex_files) > 1)
        except Exception:
            pass

        try:
            libs = apk.get_libraries() if hasattr(apk, "get_libraries") else []
            native_libs = [f for f in apk.get_files() if f.startswith("lib/")]
            features["uses_native_libs"] = int(
                bool(libs) or bool(native_libs))
        except Exception:
            pass

        try:
            all_strings = " ".join(apk.get_files())
            hits = sum(
                1 for s in SUSPICIOUS_CODE_STRINGS if s in all_strings)
            features["suspicious_code_string_hits"] = hits
            features["uses_dynamic_code_loading"] = int(
                "DexClassLoader" in all_strings or "PathClassLoader" in all_strings)
        except Exception:
            pass

        return features
