import csv
import random
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/dataset", tags=["Dataset (For Demo/Simulation)"])

DATASET_PATH = Path("data/diabetes_prediction_dataset.csv")
_dataset_cache = []


def load_dataset():
    global _dataset_cache
    if not _dataset_cache:
        if not DATASET_PATH.exists():
            return []

        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            _dataset_cache = [row for row in reader]

    return _dataset_cache


@router.get("/random")
def get_random_patient():
    """
    [DEMO PURPOSE ONLY]
    Mengambil 1 baris acak dari Kaggle dataset untuk mengisi UI Flutter.
    Dalam skenario dunia nyata, data ini diisi manual oleh pasien di HP.
    """
    data = load_dataset()
    if not data:
        raise HTTPException(status_code=404, detail="Dataset CSV tidak ditemukan di server.")

    # Ambil 1 pasien acak
    patient = random.choice(data)

    # Konversi tipe data agar sesuai dengan model
    try:
        return {
            "gender": patient["gender"],
            "age": float(patient["age"]),
            "hypertension": int(patient["hypertension"]),
            "heart_disease": int(patient["heart_disease"]),
            "smoking_history": patient["smoking_history"],
            "bmi": float(patient["bmi"]),
            "HbA1c_level": float(patient["HbA1c_level"]),
            "blood_glucose_level": float(patient["blood_glucose_level"]),
            "diabetes_actual": int(patient["diabetes"]),  # Sebagai ground-truth untuk uji coba
        }
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Format dataset tidak sesuai: {e}")
