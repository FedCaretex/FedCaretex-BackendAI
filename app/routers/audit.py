from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.services.fl_service import get_coordinator

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("/verify")
def verify(round_id: int, hash: str, client_id: Optional[str] = Query(default=None)):
    """Tanpa client_id -> cek hash model global ronde tsb; dengan client_id -> cek hash update klien."""
    try:
        ok = get_coordinator().verify(round_id, hash.lower(), client_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="client_id/hash harus hex") from e
    return {"round_id": round_id, "client_id": client_id, "hash": hash, "verified": ok}


@router.get("/{round_id}")
def get_audit(round_id: int):
    """Jejak audit satu ronde langsung dari ledger (simulated hash-chain atau kontrak AuditTrail)."""
    out = get_coordinator().audit(round_id)
    if not out["updates"] and out["global_model"] is None:
        raise HTTPException(status_code=404, detail="Ronde belum memiliki jejak audit")
    return out
