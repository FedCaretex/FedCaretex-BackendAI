"""Orkestrasi FL (FedAvg / FedProx) + DP + hash + tanda tangan + ledger.

Alur satu ronde (sesuai rancangan):
  1. klien memverifikasi hash model global terhadap ledger
  2. latih lokal (FedProx: + suku proksimal) -> delta = w_lokal - w_global
  3. DP di klien: clip L2 + noise Gaussian
  4. serialisasi kanonik -> SHA-256 -> tanda tangan Ed25519 (round || hash)
  5. server: hitung ulang hash dari byte yang DITERIMA, cek tanda tangan; gagal -> ditolak
  6. lolos -> catat ke ledger; agregasi berbobot n_k
  7. hash model global dicatat ke ledger
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from . import dp as dpm
from . import model as M
from .identity import ClientIdentity, verify_signature
from .ledger import Ledger
from .metrics import classification_metrics
from .serialize import from_canonical_bytes, sha256_hex, to_canonical_bytes


@dataclass
class FLConfig:
    algo: str = "fedavg"  # "fedavg" | "fedprox"
    rounds: int = 30
    local_epochs: int = 2
    lr: float = 0.05
    batch_size: int = 64
    prox_mu: float = 0.0
    use_dp: bool = False
    clip_norm: float = 1.0
    noise_multiplier: float = 0.0
    dp_delta: float = 1e-5
    seed: int = 0

    def validate(self):
        if self.algo not in ("fedavg", "fedprox"):
            raise ValueError("algo harus 'fedavg' atau 'fedprox'")
        if self.algo == "fedprox" and self.prox_mu <= 0:
            raise ValueError("fedprox membutuhkan prox_mu > 0")


# tamper(client_index, round_id, payload_bytes) -> payload_bytes (mungkin dimodifikasi)
TamperFn = Callable[[int, int, bytes], bytes]


def evaluate(params, X, y) -> dict:
    return classification_metrics(y, M.predict_proba(params, X))


def run_federated(
    cfg: FLConfig,
    client_data: list[tuple[np.ndarray, np.ndarray]],
    X_val,
    y_val,
    X_test,
    y_test,
    ledger: Ledger | None = None,
    tamper: TamperFn | None = None,
    init_params: np.ndarray | None = None,
) -> dict:
    cfg.validate()
    k = len(client_data)
    ids = [ClientIdentity(seed=f"client-{cfg.seed}-{i}".encode()) for i in range(k)]
    registry = {ident.client_id: ident.public_key for ident in ids}  # didaftarkan di gateway
    w = (init_params if init_params is not None else M.init_params(cfg.seed)).copy()

    history, anomalies = [], []
    if ledger is not None:
        ledger.submit_global_model(0, 0, sha256_hex(to_canonical_bytes(w)))

    for r in range(1, cfg.rounds + 1):
        rec = {
            "round": r,
            "accepted": 0,
            "rejected": 0,
            "payload_bytes": 0,
            "t_train_s": 0.0,
            "t_hash_s": 0.0,
            "t_ledger_s": 0.0,
            "gas": 0,
            "ledger_fail": 0,
        }
        w_hash = sha256_hex(to_canonical_bytes(w))
        accepted: list[tuple[np.ndarray, int]] = []

        for ci, (Xc, yc) in enumerate(client_data):
            ident = ids[ci]
            # 1) verifikasi model global terhadap ledger
            if ledger is not None and not ledger.verify_global(r - 1, w_hash):
                anomalies.append({"round": r, "client": ci, "type": "global_hash_mismatch"})
                rec["rejected"] += 1
                continue
            # 2) latih lokal
            t0 = time.perf_counter()
            w_loc = M.train_local(
                w,
                Xc,
                yc,
                epochs=cfg.local_epochs,
                lr=cfg.lr,
                batch_size=cfg.batch_size,
                seed=cfg.seed * 100003 + r * 1009 + ci,
                prox_mu=cfg.prox_mu if cfg.algo == "fedprox" else 0.0,
                w_global=w,
            )
            delta = w_loc - w
            # 3) DP di klien
            if cfg.use_dp:
                delta, _ = dpm.clip_update(delta, cfg.clip_norm)
                rng = np.random.default_rng(cfg.seed * 7919 + r * 104729 + ci)
                delta = dpm.add_gaussian_noise(delta, cfg.clip_norm, cfg.noise_multiplier, rng)
            rec["t_train_s"] += time.perf_counter() - t0
            # 4) serialisasi, hash, tanda tangan
            t0 = time.perf_counter()
            payload = to_canonical_bytes(delta)
            h = sha256_hex(payload)
            sig = ident.sign(r, h)
            rec["t_hash_s"] += time.perf_counter() - t0
            # transport (bisa disusupi pada pengujian)
            received = tamper(ci, r, payload) if tamper else payload
            rec["payload_bytes"] += len(received)
            # 5) server: hash ulang + tanda tangan
            if sha256_hex(received) != h or not verify_signature(ident.public_key, r, h, sig):
                anomalies.append({"round": r, "client": ci, "type": "integrity_failed"})
                rec["rejected"] += 1
                continue
            # 6) ledger
            if ledger is not None:
                res = ledger.submit_model_update(r, ident.client_id, r, h)
                rec["t_ledger_s"] += res.latency_s
                rec["gas"] += res.gas or 0
                if not res.ok:
                    rec["ledger_fail"] += 1
                    anomalies.append(
                        {"round": r, "client": ci, "type": "ledger_rejected", "reason": res.reason}
                    )
                    rec["rejected"] += 1
                    continue
            accepted.append((from_canonical_bytes(received), len(yc)))
            rec["accepted"] += 1

        # agregasi berbobot n_k
        t0 = time.perf_counter()
        if accepted:
            tot = float(sum(n for _, n in accepted))
            agg = sum(d * (n / tot) for d, n in accepted)
            w = (w + agg).astype(np.float32)
        rec["t_agg_s"] = time.perf_counter() - t0
        # 7) hash model global
        if ledger is not None:
            res = ledger.submit_global_model(r, r, sha256_hex(to_canonical_bytes(w)))
            rec["t_ledger_s"] += res.latency_s
            rec["gas"] += res.gas or 0
        v, t = evaluate(w, X_val, y_val), evaluate(w, X_test, y_test)
        rec.update({f"val_{a}": v[a] for a in ("f1", "recall", "auc", "accuracy")})
        rec.update({f"test_{a}": t[a] for a in ("f1", "recall", "auc", "accuracy")})
        history.append(rec)

    out = {
        "params": w,
        "history": history,
        "anomalies": anomalies,
        "final_val": evaluate(w, X_val, y_val),
        "final_test": evaluate(w, X_test, y_test),
        "config": asdict(cfg),
        "n_clients": k,
    }
    if cfg.use_dp:
        out["epsilon_rdp"] = dpm.rdp_epsilon(cfg.noise_multiplier, cfg.rounds, cfg.dp_delta)
    return out


def run_centralized(
    X_tr,
    y_tr,
    X_val,
    y_val,
    X_test,
    y_test,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int = 0,
) -> dict:
    w0 = M.init_params(seed)
    w = M.train_local(w0, X_tr, y_tr, epochs=epochs, lr=lr, batch_size=batch_size, seed=seed)
    return {
        "params": w,
        "final_val": evaluate(w, X_val, y_val),
        "final_test": evaluate(w, X_test, y_test),
    }
