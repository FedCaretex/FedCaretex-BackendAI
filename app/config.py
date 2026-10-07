"""Konfigurasi backend dari environment variable.

Dibaca lewat `load_settings()` (bukan saat import) supaya tes / skrip bisa
menimpa nilai env sebelum koordinator FL dibuat.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    # Data evaluasi server + data bootstrap (None -> data SINTETIS)
    dataset_csv: str | None
    seed: int
    # Ledger: "simulated" (hash-chain in-memory) | "evm" (AuditTrail.sol via web3)
    ledger: str
    ledger_rpc_url: str | None
    # Protokol FL
    min_clients: int  # update diterima per ronde sebelum agregasi otomatis
    bootstrap_rounds: int  # ronde FL simulasi saat start (0 = model acak)
    bootstrap_clients: int
    algo: str  # "fedavg" | "fedprox" (dikirim ke klien)
    prox_mu: float
    local_epochs: int
    lr: float
    batch_size: int
    # Differential Privacy (diterapkan DI KLIEN; server hanya mengumumkan parameternya)
    dp_enabled: bool
    clip_norm: float
    noise_multiplier: float
    dp_delta: float
    # Kategori risiko (BELUM divalidasi klinis; pilih threshold di data validasi)
    risk_low: float
    risk_high: float


def load_settings() -> Settings:
    csv = _env("DATASET_CSV", "").strip() or None
    rpc = _env("LEDGER_RPC_URL", "").strip() or None
    return Settings(
        dataset_csv=csv,
        seed=int(_env("FL_SEED", "0")),
        ledger=_env("LEDGER", "simulated").strip().lower(),
        ledger_rpc_url=rpc,
        min_clients=int(_env("FL_MIN_CLIENTS", "3")),
        bootstrap_rounds=int(_env("FL_BOOTSTRAP_ROUNDS", "10")),
        bootstrap_clients=int(_env("FL_BOOTSTRAP_CLIENTS", "5")),
        algo=_env("FL_ALGO", "fedavg").strip().lower(),
        prox_mu=float(_env("FL_PROX_MU", "0.01")),
        local_epochs=int(_env("FL_LOCAL_EPOCHS", "2")),
        lr=float(_env("FL_LR", "0.05")),
        batch_size=int(_env("FL_BATCH_SIZE", "64")),
        dp_enabled=_bool("DP_ENABLED", True),
        clip_norm=float(_env("DP_CLIP_NORM", "1.0")),
        noise_multiplier=float(_env("DP_NOISE_MULTIPLIER", "1.0")),
        dp_delta=float(_env("DP_DELTA", "1e-5")),
        risk_low=float(_env("RISK_LOW", "0.3")),
        risk_high=float(_env("RISK_HIGH", "0.6")),
    )
