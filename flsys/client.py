"""Implementasi REFERENSI sisi klien untuk protokol HTTP gateway.

Aplikasi Flutter harus meniru fungsi ini 1:1 (latih -> delta -> DP -> serialisasi
kanonik -> SHA-256 -> tanda tangan Ed25519). Dipakai oleh
`scripts/simulate_clients.py` dan `tests/test_api.py`.

Payload yang dikirim adalah DELTA (w_lokal - w_global) SETELAH DP, bukan bobot penuh.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import numpy as np

from . import dp as dpm
from . import model as M
from .identity import ClientIdentity
from .serialize import from_canonical_bytes, sha256_hex, to_canonical_bytes


@dataclass
class LocalUpdate:
    round_id: int
    payload: bytes
    hash_hex: str
    signature: bytes
    num_examples: int

    def to_request(self, identity: ClientIdentity) -> dict:
        """Body JSON untuk POST /fl/submit-update."""
        return {
            "client_id": identity.client_id.hex(),
            "round_id": self.round_id,
            "num_examples": self.num_examples,
            "payload_b64": base64.b64encode(self.payload).decode(),
            "hash": self.hash_hex,
            "signature": self.signature.hex(),
        }


def decode_global(resp: dict) -> np.ndarray:
    """Dekode & VERIFIKASI respons GET /fl/global-model. Gagal -> ValueError."""
    raw = base64.b64decode(resp["params_b64"])
    if sha256_hex(raw) != resp["hash"]:
        raise ValueError("Hash model global tidak cocok dengan byte yang diterima")
    if not resp.get("ledger_verified", False):
        raise ValueError("Hash model global tidak terverifikasi di ledger")
    return from_canonical_bytes(raw)


def build_update(
    global_params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    round_id: int,
    identity: ClientIdentity,
    training: dict,
    dp: dict,
    seed: int = 0,
) -> LocalUpdate:
    """Latih lokal lalu bungkus update sesuai protokol.

    `training` dan `dp` = isi field dengan nama sama dari GET /fl/global-model.
    """
    prox = float(training["prox_mu"]) if training["algo"] == "fedprox" else 0.0
    w_loc = M.train_local(
        global_params,
        X,
        y,
        epochs=int(training["local_epochs"]),
        lr=float(training["lr"]),
        batch_size=int(training["batch_size"]),
        seed=seed,
        prox_mu=prox,
        w_global=global_params,
    )
    delta = (w_loc - global_params).astype(np.float32)
    if dp["enabled"]:
        delta, _ = dpm.clip_update(delta, float(dp["clip_norm"]))
        delta = dpm.add_gaussian_noise(
            delta,
            float(dp["clip_norm"]),
            float(dp["noise_multiplier"]),
            np.random.default_rng(seed),
        )
    payload = to_canonical_bytes(delta)
    h = sha256_hex(payload)
    return LocalUpdate(round_id, payload, h, identity.sign(round_id, h), int(len(y)))
