"""Inferensi + SHAP eksak memakai model GLOBAL terbaru dari koordinator FL.

Server-side inference ini untuk demo / pembanding. Pada rancangan skripsi,
inferensi dan SHAP idealnya dijalankan DI PERANGKAT (data klinis tidak keluar HP);
aplikasi cukup mengunduh model + baseline dari GET /fl/global-model.
"""

from __future__ import annotations

import pandas as pd

from flsys import data as D
from flsys import model as M
from flsys.shap_exact import exact_shap

from .fl_service import get_coordinator

FEATURE_LABELS_ID = {
    "gender": "Jenis kelamin",
    "age": "Usia",
    "hypertension": "Hipertensi",
    "heart_disease": "Penyakit jantung",
    "smoking_history": "Riwayat merokok",
    "bmi": "Indeks massa tubuh (BMI)",
    "HbA1c_level": "Kadar HbA1c",
    "blood_glucose_level": "Kadar glukosa darah",
}

# Saran umum (bukan diagnosis). Hanya ditampilkan untuk fitur yang MENAIKKAN risiko.
RECOMMENDATIONS_ID = {
    "HbA1c_level": "Periksakan ulang HbA1c di fasilitas kesehatan (ADA: 5,7-6,4% prediabetes, >=6,5% ambang diagnosis).",
    "blood_glucose_level": "Kurangi gula sederhana/minuman manis dan periksakan glukosa darah puasa.",
    "bmi": "Turunkan berat badan bertahap; targetkan aktivitas fisik sedang +/-150 menit per minggu.",
    "hypertension": "Kontrol tekanan darah secara rutin dan batasi asupan garam.",
    "heart_disease": "Lanjutkan kontrol rutin penyakit jantung dan diskusikan risiko diabetes dengan dokter.",
    "smoking_history": "Pertimbangkan berhenti merokok; merokok meningkatkan risiko komplikasi metabolik.",
    "age": "Usia tidak dapat diubah; lakukan skrining gula darah secara berkala.",
}

DISCLAIMER = "Hasil ini skrining berbasis model, bukan diagnosis medis. Konsultasikan dengan tenaga kesehatan."


def _category(p: float, low: float, high: float) -> str:
    return "Rendah" if p < low else ("Sedang" if p < high else "Tinggi")


def predict(features: dict) -> dict:
    coord = get_coordinator()
    w, version, h = coord.model_snapshot()
    df = pd.DataFrame([features], columns=D.RAW_FEATURES)
    x = D.preprocess(df)[0]
    p = float(M.predict_proba(w, x[None])[0])
    base_p = float(M.predict_proba(w, coord.baseline[None])[0])
    phi = exact_shap(w, x, coord.baseline)

    items = [
        {
            "feature": f,
            "label": FEATURE_LABELS_ID[f],
            "value": features[f],
            "contribution": float(phi[i]),
            "direction": "menaikkan" if phi[i] > 0 else "menurunkan",
        }
        for i, f in enumerate(D.RAW_FEATURES)
    ]
    items.sort(key=lambda it: abs(it["contribution"]), reverse=True)

    recs = [
        RECOMMENDATIONS_ID[it["feature"]]
        for it in items[:3]
        if it["contribution"] > 0 and it["feature"] in RECOMMENDATIONS_ID
    ]
    recs.append(DISCLAIMER)

    return {
        "probability": p,
        "risk_score": round(p * 100, 2),
        "category": _category(p, coord.s.risk_low, coord.s.risk_high),
        "base_value": base_p,  # f(baseline); sum(contribution) = probability - base_value
        "shap_values": {it["feature"]: it["contribution"] for it in items},
        "shap_breakdown": items,
        "recommendations": recs,
        "model_version": version,
        "model_hash": h,
        "data_source": coord.data_source,
    }
