import time
import uuid

from fastapi import APIRouter

from app.schemas import PredictRequest, PredictResponse
from app.services.predict_service import predict as run_predict
from app.store import store

router = APIRouter(tags=["predict"])

TONE = {
    "Rendah": "Hasil ini menunjukkan risiko yang masih terkelola. Tetap jaga kebiasaan sehat.",
    "Sedang": "Ada beberapa tanda yang perlu diwaspadai. Ini bukan diagnosis, tapi layak dipantau.",
    "Tinggi": "Skor lebih tinggi dari biasanya. Segera konsultasikan ke tenaga kesehatan, jangan panik.",
}


@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    payload = body.model_dump()
    user_id = payload.pop("user_id") or "demo-user"
    out = run_predict(payload)
    store.add_history(
        {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "timestamp": time.time(),
            "category": out["category"],
            "risk_score": out["risk_score"],
            "probability": out["probability"],
            "top_feature": out["shap_breakdown"][0]["label"],
            "model_version": out["model_version"],
        }
    )
    return PredictResponse(message=TONE[out["category"]], **out)
