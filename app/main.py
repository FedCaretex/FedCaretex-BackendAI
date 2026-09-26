from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import shap
import json
import numpy as np

app = FastAPI(title="SmartWealth AI Backend", description="FastAPI server for Diabetes Prediction, FL & SHAP")

class PredictionRequest(BaseModel):
    glucose: float
    bmi: float
    age: int
    heart_rate: float

@app.get("/")
def read_root():
    return {"status": "ok", "message": "SmartWealth Backend API is running."}

@app.post("/predict")
def predict_risk(data: PredictionRequest):
    # DUMMY LOGIC FOR MVP
    # Dalam skenario nyata, load model TFLite global disini untuk SHAP explainer
    
    # Kalkulasi risk (Dummy)
    risk_score = (data.glucose * 0.4) + (data.bmi * 0.3) + (data.age * 0.2)
    category = "Tinggi" if risk_score > 100 else "Sedang" if risk_score > 70 else "Rendah"
    
    # DUMMY SHAP VALUES (Harusnya dihasilkan dari shap.Explainer)
    shap_values = {
        "glucose": 0.45,
        "bmi": 0.35,
        "age": 0.15,
        "heart_rate": 0.05
    }
    
    return {
        "risk_score": risk_score,
        "category": category,
        "shap_values": shap_values,
        "message": "Prediksi berhasil dilakukan dengan analitik SHAP."
    }

@app.post("/fl/submit-update")
def submit_fl_update(client_id: str, round_id: str):
    # Endpoint simulasi client mengupload bobot lokal
    # Di integrasikan dengan Blockchain untuk Audit Trail
    tx_hash = "0x" + "a" * 64 # Simulasi TX Hash Smart Contract
    return {
        "status": "success",
        "round_id": round_id,
        "blockchain_tx_hash": tx_hash,
        "message": "Model weights submitted and hashed to blockchain."
    }
