"""Koordinator FL di sisi server (gateway HTTP di atas paket `flsys`).

Aturan verifikasi SAMA dengan `flsys.fedlearn.run_federated`, hanya saja klien
sekarang datang lewat HTTP, bukan loop simulasi:
  1. klien terdaftar (public key Ed25519 -> client_id = SHA-256(pk))
  2. round_id == ronde berjalan, klien belum mengirim di ronde ini
  3. SHA-256 dihitung ulang dari byte yang DITERIMA dan harus == hash yang diklaim
  4. tanda tangan Ed25519 atas (round_id || hash) valid
  5. payload kanonik valid, panjang == jumlah parameter model, semua nilai finite
  6. hash dicatat ke ledger (gagal -> ditolak)
  7. setelah >= FL_MIN_CLIENTS update: agregasi berbobot n_k, hash global dicatat ke ledger

Batasan yang perlu disebut di naskah: server TIDAK bisa membuktikan klien benar-benar
menerapkan DP; hash/tanda tangan mendeteksi modifikasi, bukan poisoning yang di-hash benar.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import threading
import time
from dataclasses import asdict

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from app.config import Settings, load_settings
from flsys import data as D
from flsys import dp as dpm
from flsys import model as M
from flsys.fedlearn import FLConfig, evaluate, run_federated
from flsys.identity import verify_signature
from flsys.ledger import EvmLedger, Ledger, SimulatedLedger
from flsys.serialize import from_canonical_bytes, sha256_hex, to_canonical_bytes
from flsys.shap_exact import baseline_from


class FLError(Exception):
    """Penolakan yang dikembalikan ke klien (status HTTP + kode alasan)."""

    def __init__(self, status: int, code: str, detail: str = ""):
        super().__init__(detail or code)
        self.status, self.code, self.detail = status, code, detail or code


def make_ledger(s: Settings) -> Ledger:
    if s.ledger == "evm":
        return EvmLedger(rpc_url=s.ledger_rpc_url)
    if s.ledger == "simulated":
        return SimulatedLedger()
    raise ValueError("LEDGER harus 'simulated' atau 'evm'")


def _metrics_brief(m: dict) -> dict:
    return {k: float(m[k]) for k in ("accuracy", "precision", "recall", "f1", "auc")}


class FLCoordinator:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or load_settings()
        if self.s.algo not in ("fedavg", "fedprox"):
            raise ValueError("FL_ALGO harus 'fedavg' atau 'fedprox'")
        self.lock = threading.RLock()

        X, y, self.data_source = D.load_dataset(self.s.dataset_csv, seed=self.s.seed)
        self.splits = D.stratified_split(X, y, self.s.seed)
        self.baseline = baseline_from(self.splits.X_val)
        self.ledger = make_ledger(self.s)

        self.registry: dict[bytes, bytes] = {}  # client_id -> public key
        self.participation: dict[bytes, int] = {}  # client_id -> jumlah ronde diterima
        self.pending: dict[bytes, tuple[np.ndarray, int, str]] = {}
        self.history: list[dict] = []
        self.anomalies: list[dict] = []
        self._bootstrap()

    # ------------------------------------------------------------------ setup
    def _bootstrap(self) -> None:
        """Model awal. bootstrap_rounds>0 -> FL SIMULASI (flsys) atas partisi data server,
        sehingga /predict langsung bermakna. Catatan ledger bootstrap ikut tersimpan."""
        s, sp = self.s, self.splits
        if s.bootstrap_rounds <= 0:
            self.w = M.init_params(s.seed)
            self.ledger.submit_global_model(0, 0, sha256_hex(to_canonical_bytes(self.w)))
            self.round = 1
            return
        parts = D.partition_iid(len(sp.y_train), s.bootstrap_clients, s.seed)
        clients = [(sp.X_train[i], sp.y_train[i]) for i in parts]
        cfg = FLConfig(
            algo=s.algo,
            rounds=s.bootstrap_rounds,
            local_epochs=s.local_epochs,
            lr=s.lr,
            batch_size=s.batch_size,
            prox_mu=s.prox_mu if s.algo == "fedprox" else 0.0,
            use_dp=s.dp_enabled,
            clip_norm=s.clip_norm,
            noise_multiplier=s.noise_multiplier,
            dp_delta=s.dp_delta,
            seed=s.seed,
        )
        out = run_federated(
            cfg, clients, sp.X_val, sp.y_val, sp.X_test, sp.y_test, ledger=self.ledger
        )
        self.w = out["params"].astype(np.float32)
        self.round = s.bootstrap_rounds + 1
        for rec in out["history"]:
            self.history.append(
                {
                    "round": rec["round"],
                    "source": "bootstrap-simulation",
                    "accepted": rec["accepted"],
                    "rejected": rec["rejected"],
                    "val_auc": rec["val_auc"],
                    "val_f1": rec["val_f1"],
                    "test_auc": rec["test_auc"],
                    "test_f1": rec["test_f1"],
                }
            )
        self.anomalies.extend({"source": "bootstrap-simulation", **a} for a in out["anomalies"])

    # ---------------------------------------------------------------- helpers
    @property
    def global_hash(self) -> str:
        return sha256_hex(to_canonical_bytes(self.w))

    def dp_info(self) -> dict:
        s = self.s
        return {
            "enabled": s.dp_enabled,
            "clip_norm": s.clip_norm,
            "noise_multiplier": s.noise_multiplier,
            "delta": s.dp_delta,
            "accountant": "RDP Gaussian, sensitivitas=C, tanpa amplifikasi subsampling",
            "epsilon_per_round": (
                dpm.rdp_epsilon(s.noise_multiplier, 1, s.dp_delta) if s.dp_enabled else None
            ),
        }

    def training_info(self) -> dict:
        s = self.s
        return {
            "algo": s.algo,
            "prox_mu": s.prox_mu if s.algo == "fedprox" else 0.0,
            "local_epochs": s.local_epochs,
            "lr": s.lr,
            "batch_size": s.batch_size,
        }

    def model_snapshot(self) -> tuple[np.ndarray, int, str]:
        with self.lock:
            return self.w.copy(), self.round - 1, self.global_hash

    def _reject(self, code: str, status: int, detail: str, **ctx) -> None:
        self.anomalies.append(
            {
                "source": "http",
                "round": self.round,
                "type": code,
                "detail": detail,
                "timestamp": time.time(),
                **ctx,
            }
        )
        raise FLError(status, code, detail)

    # ---------------------------------------------------------------- API ops
    def register_client(self, public_key_hex: str) -> str:
        try:
            pk = bytes.fromhex(public_key_hex)
            Ed25519PublicKey.from_public_bytes(pk)
        except ValueError as e:
            raise FLError(
                400, "invalid_public_key", "Public key harus Ed25519 raw 32 byte (hex)"
            ) from e
        cid = hashlib.sha256(pk).digest()
        with self.lock:
            self.registry[cid] = pk
        return cid.hex()

    def global_model(self) -> dict:
        with self.lock:
            payload = to_canonical_bytes(self.w)
            version = self.round - 1
            h = sha256_hex(payload)
            return {
                "round_id": self.round,  # ronde yang harus dilatih klien
                "model_version": version,  # ronde tempat model ini dicatat di ledger
                "hash": h,
                "ledger_verified": bool(self.ledger.verify_global(version, h)),
                "params_b64": base64.b64encode(payload).decode(),
                "n_params": M.n_params(),
                "layers": list(M.LAYERS),
                "training": self.training_info(),
                "dp": self.dp_info(),
                "shap_baseline": self.baseline.astype(float).tolist(),
                "data_source": self.data_source,
            }

    def submit_update(
        self,
        *,
        client_id_hex: str,
        round_id: int,
        num_examples: int,
        payload_b64: str,
        claimed_hash: str,
        signature_hex: str,
    ) -> dict:
        with self.lock:
            try:
                cid = bytes.fromhex(client_id_hex)
            except ValueError:
                self._reject("bad_client_id", 400, "client_id bukan hex")
            pk = self.registry.get(cid)
            ctx = {"client_id": client_id_hex}
            if pk is None:
                self._reject("unregistered_client", 403, "Klien belum terdaftar", **ctx)
            if round_id != self.round:
                self._reject("stale_round", 409, f"Ronde berjalan = {self.round}", **ctx)
            if cid in self.pending:
                self._reject("duplicate_update", 409, "Klien sudah mengirim di ronde ini", **ctx)
            if num_examples <= 0:
                self._reject("bad_num_examples", 400, "num_examples harus > 0", **ctx)
            try:
                received = base64.b64decode(payload_b64, validate=True)
                sig = bytes.fromhex(signature_hex)
            except (binascii.Error, ValueError):
                self._reject("bad_encoding", 400, "payload_b64 / signature tidak valid", **ctx)

            h = sha256_hex(received)
            if h != claimed_hash.lower() or not verify_signature(pk, round_id, h, sig):
                self._reject(
                    "integrity_failed",
                    422,
                    "Hash byte yang diterima / tanda tangan tidak cocok",
                    **ctx,
                )
            try:
                delta = from_canonical_bytes(received)
            except ValueError as e:
                self._reject("bad_payload", 422, str(e), **ctx)
            if delta.size != M.n_params() or not np.all(np.isfinite(delta)):
                self._reject(
                    "bad_payload", 422, "Ukuran parameter salah / nilai tidak finite", **ctx
                )

            res = self.ledger.submit_model_update(round_id, cid, round_id, h)
            if not res.ok:
                self._reject("ledger_rejected", 409, res.reason, **ctx)

            self.pending[cid] = (delta, int(num_examples), h)
            self.participation[cid] = self.participation.get(cid, 0) + 1
            out = {
                "accepted": True,
                "round_id": round_id,
                "hash": h,
                "ledger": {"kind": self.ledger.kind, "gas": res.gas, "latency_s": res.latency_s},
                "pending_in_round": len(self.pending),
                "min_clients": self.s.min_clients,
                "aggregated": False,
                "client_epsilon": self.client_epsilon(cid),
            }
            if len(self.pending) >= self.s.min_clients:
                out["aggregated"] = True
                out["aggregation"] = self.aggregate()
            return out

    def aggregate(self) -> dict:
        """FedAvg berbobot n_k atas delta yang lolos verifikasi (sama dengan flsys.fedlearn)."""
        with self.lock:
            if not self.pending:
                raise FLError(409, "no_updates", "Belum ada update yang diterima di ronde ini")
            r = self.round
            tot = float(sum(n for _, n, _ in self.pending.values()))
            agg = sum(d * (n / tot) for d, n, _ in self.pending.values())
            self.w = (self.w + agg).astype(np.float32)
            gres = self.ledger.submit_global_model(r, r, self.global_hash)
            sp = self.splits
            v, t = evaluate(self.w, sp.X_val, sp.y_val), evaluate(self.w, sp.X_test, sp.y_test)
            rec = {
                "round": r,
                "source": "http",
                "accepted": len(self.pending),
                "rejected": 0,
                "val_auc": v["auc"],
                "val_f1": v["f1"],
                "test_auc": t["auc"],
                "test_f1": t["f1"],
                "global_hash": self.global_hash,
                "ledger_ok": gres.ok,
            }
            self.history.append(rec)
            self.pending.clear()
            self.round = r + 1
            return {
                "round": r,
                "new_round": self.round,
                "global_hash": rec["global_hash"],
                "ledger_ok": gres.ok,
                "val": _metrics_brief(v),
                "test": _metrics_brief(t),
            }

    def client_epsilon(self, cid: bytes) -> float | None:
        if not self.s.dp_enabled:
            return None
        return dpm.rdp_epsilon(
            self.s.noise_multiplier, self.participation.get(cid, 0), self.s.dp_delta
        )

    # ------------------------------------------------------------------ audit
    def global_record(self, round_id: int) -> dict | None:
        if isinstance(self.ledger, SimulatedLedger):
            g = self.ledger._globals.get(round_id)
            return dict(g) if g else None
        h, ver, ts, exists = self.ledger.c.functions.getGlobalModel(round_id).call()
        return {"hash": h.hex(), "model_version": ver, "timestamp": ts} if exists else None

    def audit(self, round_id: int) -> dict:
        with self.lock:
            trail = self.ledger.get_audit_trail(round_id)
            g = self.global_record(round_id)
            out = {
                "round_id": round_id,
                "ledger_kind": self.ledger.kind,
                "updates": trail,
                "global_model": g,
            }
            if isinstance(self.ledger, SimulatedLedger):
                out["chain_valid"] = self.ledger.chain_is_valid()
                out["chain_length"] = len(self.ledger.chain)
            return out

    def verify(self, round_id: int, hash_hex: str, client_id_hex: str | None) -> bool:
        with self.lock:
            if client_id_hex is None:
                return bool(self.ledger.verify_global(round_id, hash_hex))
            return bool(
                self.ledger.verify_integrity(round_id, bytes.fromhex(client_id_hex), hash_hex)
            )

    def status(self) -> dict:
        with self.lock:
            return {
                "round_id": self.round,
                "model_version": self.round - 1,
                "global_hash": self.global_hash,
                "pending_in_round": len(self.pending),
                "min_clients": self.s.min_clients,
                "registered_clients": len(self.registry),
                "ledger_kind": self.ledger.kind,
                "data_source": self.data_source,
                "training": self.training_info(),
                "dp": self.dp_info(),
                "last_round": self.history[-1] if self.history else None,
                "anomalies": len(self.anomalies),
                "config": asdict(self.s),
            }


_coord: FLCoordinator | None = None
_coord_lock = threading.Lock()


def get_coordinator() -> FLCoordinator:
    global _coord
    with _coord_lock:
        if _coord is None:
            _coord = FLCoordinator()
        return _coord


def reset_coordinator(settings: Settings | None = None) -> FLCoordinator:
    """Untuk tes: buat ulang koordinator dengan settings baru."""
    global _coord
    with _coord_lock:
        _coord = FLCoordinator(settings)
        return _coord
