from fastapi import APIRouter, HTTPException

from app.schemas import (
    FlSubmitRequest,
    FlSubmitResponse,
    GlobalModelResponse,
    RegisterClientRequest,
    RegisterClientResponse,
)
from app.services.fl_service import FLError, get_coordinator

router = APIRouter(prefix="/fl", tags=["federated-learning"])


def _http(e: FLError) -> HTTPException:
    return HTTPException(status_code=e.status, detail={"code": e.code, "message": e.detail})


@router.post("/register-client", response_model=RegisterClientResponse)
def register_client(body: RegisterClientRequest):
    try:
        return RegisterClientResponse(client_id=get_coordinator().register_client(body.public_key))
    except FLError as e:
        raise _http(e) from e


@router.get("/global-model", response_model=GlobalModelResponse)
def global_model():
    """Model global terbaru + konfigurasi training/DP + baseline SHAP.
    Klien WAJIB mencocokkan SHA-256(params) dengan `hash` sebelum melatih."""
    return get_coordinator().global_model()


@router.post("/submit-update", response_model=FlSubmitResponse)
def submit_update(body: FlSubmitRequest):
    try:
        return get_coordinator().submit_update(
            client_id_hex=body.client_id,
            round_id=body.round_id,
            num_examples=body.num_examples,
            payload_b64=body.payload_b64,
            claimed_hash=body.hash,
            signature_hex=body.signature,
        )
    except FLError as e:
        raise _http(e) from e


@router.post("/aggregate")
def aggregate_now():
    """Paksa agregasi ronde berjalan walau update < FL_MIN_CLIENTS (untuk demo)."""
    try:
        return get_coordinator().aggregate()
    except FLError as e:
        raise _http(e) from e


@router.get("/status")
def status():
    return get_coordinator().status()


@router.get("/history")
def history():
    return get_coordinator().history


@router.get("/anomalies")
def anomalies():
    """Update yang ditolak (integritas gagal, ronde basi, duplikat, dll.)."""
    return get_coordinator().anomalies
