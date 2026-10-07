"""Tes end-to-end gateway HTTP (FastAPI TestClient) di atas flsys."""

import base64
import os

import numpy as np
import pytest

os.environ.update(
    {
        "FL_BOOTSTRAP_ROUNDS": "2",
        "FL_MIN_CLIENTS": "2",
        "LEDGER": "simulated",
        "DP_ENABLED": "1",
        "DP_NOISE_MULTIPLIER": "0.5",
    }
)

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.services.fl_service import reset_coordinator  # noqa: E402
from flsys import data as D  # noqa: E402
from flsys.client import build_update, decode_global  # noqa: E402
from flsys.identity import ClientIdentity  # noqa: E402
from flsys.serialize import sha256_hex  # noqa: E402

PATIENT = {
    "gender": "Male",
    "age": 55,
    "hypertension": 1,
    "heart_disease": 0,
    "smoking_history": "former",
    "bmi": 31.5,
    "HbA1c_level": 7.0,
    "blood_glucose_level": 180,
}


@pytest.fixture()
def client():
    reset_coordinator()
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def local_data():
    X, y, _ = D.load_dataset(None, seed=7, n_synth=3000)
    parts = D.partition_iid(len(y), 3, 7)
    return [(X[i], y[i]) for i in parts]


def _register(c, n):
    ids = [ClientIdentity(seed=f"t-{i}".encode()) for i in range(n)]
    for ident in ids:
        r = c.post("/fl/register-client", json={"public_key": ident.public_key.hex()})
        assert r.status_code == 200 and r.json()["client_id"] == ident.client_id.hex()
    return ids


def _update(g, ident, Xy, seed=0):
    return build_update(
        decode_global(g),
        *Xy,
        round_id=g["round_id"],
        identity=ident,
        training=g["training"],
        dp=g["dp"],
        seed=seed,
    )


def test_health_and_bootstrap(client):
    j = client.get("/").json()
    assert j["status"] == "ok" and j["round_id"] == 3 and j["data_source"] == "synthetic"
    assert len(client.get("/fl/history").json()) == 2


def test_predict_shap_efficiency(client):
    r = client.post("/predict", json=PATIENT)
    assert r.status_code == 200, r.text
    j = r.json()
    assert 0.0 <= j["probability"] <= 1.0
    assert abs(sum(j["shap_values"].values()) - (j["probability"] - j["base_value"])) < 1e-5
    assert len(j["shap_breakdown"]) == 8 and j["category"] in ("Rendah", "Sedang", "Tinggi")


def test_predict_rejects_unknown_category(client):
    assert client.post("/predict", json={**PATIENT, "smoking_history": "kadang"}).status_code == 422


def test_global_model_is_verifiable(client):
    g = client.get("/fl/global-model").json()
    assert g["ledger_verified"] is True
    assert sha256_hex(base64.b64decode(g["params_b64"])) == g["hash"]
    assert len(decode_global(g)) == g["n_params"] == 1057
    assert len(g["shap_baseline"]) == 15


def test_full_round_aggregates_and_is_audited(client, local_data):
    ids = _register(client, 2)
    g = client.get("/fl/global-model").json()
    r1 = client.post(
        "/fl/submit-update", json=_update(g, ids[0], local_data[0], 1).to_request(ids[0])
    )
    assert r1.status_code == 200 and r1.json()["aggregated"] is False
    r2 = client.post(
        "/fl/submit-update", json=_update(g, ids[1], local_data[1], 2).to_request(ids[1])
    )
    j = r2.json()
    assert j["aggregated"] is True and j["aggregation"]["new_round"] == g["round_id"] + 1

    g2 = client.get("/fl/global-model").json()
    assert g2["round_id"] == g["round_id"] + 1 and g2["ledger_verified"] and g2["hash"] != g["hash"]

    audit = client.get(f"/audit/{g['round_id']}").json()
    assert len(audit["updates"]) == 2 and audit["chain_valid"] is True
    assert audit["global_model"]["hash"] == g2["hash"]
    v = client.get(
        "/audit/verify",
        params={
            "round_id": g["round_id"],
            "hash": r1.json()["hash"],
            "client_id": ids[0].client_id.hex(),
        },
    ).json()
    assert v["verified"] is True


def test_tampered_payload_rejected_and_not_recorded(client, local_data):
    ids = _register(client, 1)
    g = client.get("/fl/global-model").json()
    body = _update(g, ids[0], local_data[0]).to_request(ids[0])
    raw = bytearray(base64.b64decode(body["payload_b64"]))
    raw[100] ^= 1
    body["payload_b64"] = base64.b64encode(bytes(raw)).decode()
    r = client.post("/fl/submit-update", json=body)
    assert r.status_code == 422 and r.json()["detail"]["code"] == "integrity_failed"
    assert client.get("/fl/status").json()["pending_in_round"] == 0
    assert client.get("/fl/anomalies").json()[-1]["type"] == "integrity_failed"


def test_forged_signature_rejected(client, local_data):
    ids = _register(client, 1)
    attacker = ClientIdentity(seed=b"attacker")
    g = client.get("/fl/global-model").json()
    upd = _update(g, ids[0], local_data[0])
    body = upd.to_request(ids[0])
    body["signature"] = attacker.sign(g["round_id"], upd.hash_hex).hex()
    r = client.post("/fl/submit-update", json=body)
    assert r.status_code == 422 and r.json()["detail"]["code"] == "integrity_failed"


def test_unregistered_stale_duplicate(client, local_data):
    stranger = ClientIdentity(seed=b"stranger")
    g = client.get("/fl/global-model").json()
    r = client.post(
        "/fl/submit-update", json=_update(g, stranger, local_data[0]).to_request(stranger)
    )
    assert r.status_code == 403

    ids = _register(client, 1)
    stale = dict(g, round_id=g["round_id"] - 1)
    r = client.post(
        "/fl/submit-update", json=_update(stale, ids[0], local_data[0]).to_request(ids[0])
    )
    assert r.status_code == 409 and r.json()["detail"]["code"] == "stale_round"

    body = _update(g, ids[0], local_data[0]).to_request(ids[0])
    assert client.post("/fl/submit-update", json=body).status_code == 200
    r = client.post("/fl/submit-update", json=body)
    assert r.status_code == 409 and r.json()["detail"]["code"] == "duplicate_update"


def test_non_finite_payload_rejected(client):
    from flsys.serialize import to_canonical_bytes

    ids = _register(client, 1)
    g = client.get("/fl/global-model").json()
    bad = np.full(1057, np.nan, dtype=np.float32)
    payload = to_canonical_bytes(bad)
    h = sha256_hex(payload)
    body = {
        "client_id": ids[0].client_id.hex(),
        "round_id": g["round_id"],
        "num_examples": 10,
        "payload_b64": base64.b64encode(payload).decode(),
        "hash": h,
        "signature": ids[0].sign(g["round_id"], h).hex(),
    }
    r = client.post("/fl/submit-update", json=body)
    assert r.status_code == 422 and r.json()["detail"]["code"] == "bad_payload"
