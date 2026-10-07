"""Data, prapemrosesan, dan partisi klien.

Prinsip: normalisasi Min-Max memakai BATAS KLINIS TETAP (konstanta bersama),
bukan min/max dari data gabungan, supaya tidak ada statistik data klien yang
bocor ke server. Konstanta yang sama harus dipakai di aplikasi mobile.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# 8 fitur asli (urutan tetap; dipakai juga oleh SHAP dan aplikasi)
RAW_FEATURES = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]
GENDER_CATS = ["Female", "Male", "Other"]
SMOKING_CATS = ["No Info", "never", "former", "current", "not current", "ever"]

# Batas klinis tetap untuk Min-Max. VERIFIKASI terhadap CSV asli (min/max tiap
# kolom) lalu sesuaikan jika perlu; nilai di luar batas di-clip.
FEATURE_BOUNDS = {
    "age": (0.0, 100.0),
    "bmi": (10.0, 100.0),
    "HbA1c_level": (3.5, 9.0),
    "blood_glucose_level": (80.0, 300.0),
}
NUMERIC = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
BINARY = ["hypertension", "heart_disease"]

# Kolom encoded: 4 numerik + 2 biner + 3 gender + 6 smoking = 15
ENCODED_COLUMNS = (
    NUMERIC
    + BINARY
    + [f"gender={c}" for c in GENDER_CATS]
    + [f"smoking_history={c}" for c in SMOKING_CATS]
)
N_FEATURES = len(ENCODED_COLUMNS)  # 15

# Pemetaan fitur asli -> indeks kolom encoded (untuk SHAP berkelompok)
FEATURE_GROUPS: dict[str, list[int]] = {}
for _f in RAW_FEATURES:
    FEATURE_GROUPS[_f] = [
        i for i, c in enumerate(ENCODED_COLUMNS) if c == _f or c.startswith(_f + "=")
    ]


def preprocess(df: pd.DataFrame) -> np.ndarray:
    """DataFrame (8 fitur asli) -> matriks float32 (n, 15)."""
    n = len(df)
    X = np.zeros((n, N_FEATURES), dtype=np.float32)
    for f in NUMERIC:
        lo, hi = FEATURE_BOUNDS[f]
        v = np.clip(df[f].to_numpy(dtype=np.float64), lo, hi)
        X[:, ENCODED_COLUMNS.index(f)] = (v - lo) / (hi - lo)
    for f in BINARY:
        X[:, ENCODED_COLUMNS.index(f)] = df[f].to_numpy(dtype=np.float32)
    for c in GENDER_CATS:
        X[:, ENCODED_COLUMNS.index(f"gender={c}")] = (df["gender"] == c).to_numpy()
    for c in SMOKING_CATS:
        X[:, ENCODED_COLUMNS.index(f"smoking_history={c}")] = (
            df["smoking_history"] == c
        ).to_numpy()
    return X


def make_synthetic_diabetes(n: int = 20000, seed: int = 0) -> pd.DataFrame:
    """Data SINTETIS berskema sama dengan Diabetes Prediction Dataset.

    HANYA untuk uji pipeline (smoke test). Hasil di atas data ini TIDAK boleh
    masuk naskah. Prevalensi positif dibuat ~8-9% agar mirip kondisi nyata.
    """
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(42, 22, n), 0.08, 80).round(2)
    gender = rng.choice(GENDER_CATS, n, p=[0.58, 0.41, 0.01])
    smoking = rng.choice(SMOKING_CATS, n, p=[0.35, 0.35, 0.09, 0.09, 0.06, 0.06])
    bmi = np.clip(rng.normal(27.3, 6.5, n), 10.01, 95.69).round(2)
    hba1c = rng.choice([3.5, 4.0, 4.5, 5.0, 5.7, 6.0, 6.2, 6.5, 6.6, 6.8, 7.0, 7.5, 8.2, 9.0], n)
    glucose = rng.choice(
        [80, 85, 90, 100, 126, 130, 140, 145, 155, 158, 160, 200, 240, 260, 280, 300], n
    )
    hyp = (rng.random(n) < 0.04 + 0.12 * (age > 50)).astype(int)
    heart = (rng.random(n) < 0.02 + 0.08 * (age > 60)).astype(int)
    z = (
        -7.6
        + 1.1 * (hba1c - 5.7)
        + 0.035 * (glucose - 140) / 1.0 * 0.3
        + 0.03 * (bmi - 27)
        + 0.02 * (age - 40)
        + 0.5 * hyp
        + 0.6 * heart
    )
    p = 1 / (1 + np.exp(-(z + 4.0)))
    y = (rng.random(n) < p).astype(int)
    return pd.DataFrame(
        {
            "gender": gender,
            "age": age,
            "hypertension": hyp,
            "heart_disease": heart,
            "smoking_history": smoking,
            "bmi": bmi,
            "HbA1c_level": hba1c,
            "blood_glucose_level": glucose,
            "diabetes": y,
        }
    )


def load_dataset(csv_path: str | None, seed: int = 0, n_synth: int = 20000):
    """Kembalikan (X, y, data_source). csv_path=None -> sintetis."""
    if csv_path is None:
        df = make_synthetic_diabetes(n_synth, seed)
        source = "synthetic"
    else:
        df = pd.read_csv(csv_path)
        missing = [c for c in RAW_FEATURES + ["diabetes"] if c not in df.columns]
        if missing:
            raise ValueError(f"Kolom tidak ditemukan di CSV: {missing}")
        df = df.dropna(subset=RAW_FEATURES + ["diabetes"]).reset_index(drop=True)
        source = "real"
    return preprocess(df), df["diabetes"].to_numpy(dtype=np.float32), source


@dataclass
class Splits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def stratified_split(X, y, seed=0, val_frac=0.15, test_frac=0.15) -> Splits:
    rng = np.random.default_rng(seed)
    idx_tr, idx_va, idx_te = [], [], []
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_te = int(len(idx) * test_frac)
        n_va = int(len(idx) * val_frac)
        idx_te.append(idx[:n_te])
        idx_va.append(idx[n_te : n_te + n_va])
        idx_tr.append(idx[n_te + n_va :])
    tr, va, te = (np.concatenate(i) for i in (idx_tr, idx_va, idx_te))
    for a in (tr, va, te):
        rng.shuffle(a)
    return Splits(X[tr], y[tr], X[va], y[va], X[te], y[te])


def partition_iid(n: int, k: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [a for a in np.array_split(idx, k)]


def partition_dirichlet(
    y: np.ndarray, k: int, alpha: float, seed: int = 0, min_size: int = 20
) -> list[np.ndarray]:
    """Non-IID label-skew: proporsi tiap kelas ke klien ~ Dirichlet(alpha).
    alpha kecil -> makin heterogen. Diulang sampai tiap klien >= min_size."""
    rng = np.random.default_rng(seed)
    for _ in range(200):
        parts = [[] for _ in range(k)]
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            rng.shuffle(idx)
            props = rng.dirichlet([alpha] * k)
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for j, chunk in enumerate(np.split(idx, cuts)):
                parts[j].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.array(sorted(p)) for p in parts]
    raise RuntimeError(
        "Gagal membuat partisi Dirichlet dengan ukuran minimum; naikkan alpha/kurangi k."
    )
