from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from app.schemas import FEATURE_LABELS_ID

FEATURE_ORDER = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree",
    "age",
]


class DiabetesRiskModel:
    """Model regresi logistik Pima-like + penjelasan kontribusi fitur (SHAP linear)."""

    def __init__(self) -> None:
        rng = np.random.default_rng(42)
        n = 800
        x = np.column_stack(
            [
                rng.integers(0, 8, n),
                rng.normal(121, 32, n).clip(44, 199),
                rng.normal(72, 12, n).clip(40, 122),
                rng.normal(21, 10, n).clip(0, 60),
                rng.normal(80, 70, n).clip(0, 400),
                rng.normal(32, 7, n).clip(18, 55),
                rng.normal(0.47, 0.3, n).clip(0.08, 2.2),
                rng.integers(21, 70, n),
            ]
        )
        logits = (
            0.04 * x[:, 0]
            + 0.035 * (x[:, 1] - 100)
            + 0.01 * (x[:, 2] - 80)
            + 0.008 * x[:, 3]
            + 0.004 * (x[:, 4] - 80)
            + 0.09 * (x[:, 5] - 25)
            + 1.4 * (x[:, 6] - 0.4)
            + 0.03 * (x[:, 7] - 30)
        )
        p = 1 / (1 + np.exp(-logits))
        y = (rng.random(n) < p).astype(int)
        self.model = LogisticRegression(max_iter=400)
        self.model.fit(x, y)
        self.feature_mean = x.mean(axis=0)
        self._explainer = None
        try:
            import shap

            self._explainer = shap.LinearExplainer(self.model, x)
        except Exception:
            self._explainer = None

    def vector(self, payload: dict) -> np.ndarray:
        return np.array([[payload[k] for k in FEATURE_ORDER]], dtype=float)

    def predict(self, payload: dict) -> Tuple[float, str, Dict[str, float], List[dict]]:
        x = self.vector(payload)
        proba = float(self.model.predict_proba(x)[0, 1])
        risk_score = round(proba * 100, 1)
        if proba < 0.34:
            category = "Rendah"
        elif proba < 0.67:
            category = "Sedang"
        else:
            category = "Tinggi"

        shap_raw = self._shap_values(x)
        shap_map = {FEATURE_ORDER[i]: float(shap_raw[i]) for i in range(len(FEATURE_ORDER))}
        breakdown = []
        for i, key in enumerate(FEATURE_ORDER):
            contrib = float(shap_raw[i])
            direction = "menaikkan" if contrib >= 0 else "menurunkan"
            label = FEATURE_LABELS_ID[key]
            breakdown.append(
                {
                    "feature": key,
                    "label": label,
                    "value": float(x[0, i]),
                    "contribution": round(contrib, 4),
                    "direction": direction,
                    "explanation": (
                        f"{label} kamu ({self._pretty(key, x[0, i])}) "
                        f"paling berperan {direction} skor risiko ini."
                        if abs(contrib) == max(abs(v) for v in shap_raw)
                        else f"{label} kamu ({self._pretty(key, x[0, i])}) {direction} skor risiko."
                    ),
                }
            )
        breakdown.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        return risk_score, category, shap_map, breakdown

    def _shap_values(self, x: np.ndarray) -> np.ndarray:
        if self._explainer is not None:
            values = self._explainer.shap_values(x)
            if isinstance(values, list):
                values = values[1]
            return np.array(values).reshape(-1)
        coef = self.model.coef_.reshape(-1)
        return coef * (x.reshape(-1) - self.feature_mean)

    @staticmethod
    def _pretty(key: str, value: float) -> str:
        units = {
            "glucose": " mg/dL",
            "blood_pressure": " mmHg",
            "bmi": "",
            "age": " tahun",
            "insulin": " µU/mL",
        }
        if key in {"pregnancies", "age"}:
            return f"{int(value)}{units.get(key, '')}"
        return f"{value:.1f}{units.get(key, '')}"

    def recommendations(self, category: str, breakdown: List[dict]) -> List[str]:
        tips = [
            "Data klinis untuk prediksi ini tidak dipakai sebagai data pelatihan bersama.",
            "Jalan kaki 20–30 menit setelah makan membantu menjaga gula darah.",
            "Utamakan makanan berserat dan kurangi minuman manis.",
        ]
        if category == "Tinggi":
            tips.insert(
                0,
                "Skor ini bukan diagnosis. Diskusikan hasilnya dengan tenaga kesehatan.",
            )
        elif category == "Rendah":
            tips.insert(0, "Pertahankan pola makan dan aktivitas yang sudah baik.")
        top = breakdown[0]["label"] if breakdown else "Glukosa"
        tips.append(f"Fokus perbaikan paling relevan saat ini: {top.lower()}.")
        return tips


diabetes_model = DiabetesRiskModel()
