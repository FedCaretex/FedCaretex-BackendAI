from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- auth
class RegisterRequest(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    email: str = Field(min_length=5)
    password: str = Field(min_length=6)


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    name: str
    email: str


# ---------------------------------------------------------------- predict
# 8 fitur asli Diabetes Prediction Dataset (urutan & kategori = flsys.data)
class ClinicalFeatures(BaseModel):
    gender: Literal["Female", "Male", "Other"]
    age: float = Field(ge=0, le=120)
    hypertension: int = Field(ge=0, le=1)
    heart_disease: int = Field(ge=0, le=1)
    smoking_history: Literal["No Info", "never", "former", "current", "not current", "ever"]
    bmi: float = Field(ge=10, le=100)
    HbA1c_level: float = Field(ge=3.0, le=20.0)
    blood_glucose_level: float = Field(ge=40, le=500)


class PredictRequest(ClinicalFeatures):
    user_id: Optional[str] = None


class ShapItem(BaseModel):
    feature: str
    label: str
    value: float | int | str
    contribution: float
    direction: str


class PredictResponse(BaseModel):
    probability: float
    risk_score: float
    category: str
    message: str
    base_value: float
    shap_values: Dict[str, float]
    shap_breakdown: List[ShapItem]
    recommendations: List[str]
    model_version: int
    model_hash: str
    data_source: str


# ---------------------------------------------------------------- federated learning
class RegisterClientRequest(BaseModel):
    public_key: str = Field(description="Ed25519 public key raw 32 byte, hex (64 karakter)")


class RegisterClientResponse(BaseModel):
    client_id: str = Field(description="SHA-256(public key), hex")


class TrainingInfo(BaseModel):
    algo: str
    prox_mu: float
    local_epochs: int
    lr: float
    batch_size: int


class DPInfo(BaseModel):
    enabled: bool
    clip_norm: float
    noise_multiplier: float
    delta: float
    accountant: str
    epsilon_per_round: Optional[float]


class GlobalModelResponse(BaseModel):
    round_id: int
    model_version: int
    hash: str
    ledger_verified: bool
    params_b64: str
    n_params: int
    layers: List[int]
    training: TrainingInfo
    dp: DPInfo
    shap_baseline: List[float]
    data_source: str


class FlSubmitRequest(BaseModel):
    client_id: str
    round_id: int
    num_examples: int
    payload_b64: str = Field(
        description="Serialisasi kanonik FLM1 dari DELTA (w_lokal - w_global) setelah DP"
    )
    hash: str = Field(description="SHA-256 hex dari byte payload")
    signature: str = Field(description="Ed25519 atas uint64_be(round_id) || hash_bytes, hex")


class FlSubmitResponse(BaseModel):
    accepted: bool
    round_id: int
    hash: str
    ledger: dict
    pending_in_round: int
    min_clients: int
    aggregated: bool
    client_epsilon: Optional[float]
    aggregation: Optional[dict] = None


# ---------------------------------------------------------------- history
class HistoryItem(BaseModel):
    id: str
    user_id: str
    timestamp: float
    category: str
    risk_score: float
    probability: float
    top_feature: str
    model_version: int
