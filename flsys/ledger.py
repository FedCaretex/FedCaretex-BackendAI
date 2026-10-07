"""Abstraksi ledger audit-trail.

Dua implementasi dengan aturan yang SAMA dengan kontrak AuditTrail.sol:
  - SimulatedLedger : hash-chain in-memory (untuk E1-E5 / unit test / fallback)
  - EvmLedger       : kontrak asli via web3.py. Provider: EVM in-process (py-evm,
                      untuk tes) atau RPC URL (Hardhat node / Besu / Ganache).
Aturan: satu catatan per (round, client); duplikat ditolak; hash nol ditolak;
verifikasi = hash yang diklaim harus sama dengan yang tercatat.
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LedgerResult:
    ok: bool
    latency_s: float = 0.0
    gas: int | None = None
    reason: str = ""


class Ledger(ABC):
    kind: str = "abstract"

    @abstractmethod
    def submit_model_update(
        self, round_id: int, client_id: bytes, model_version: int, hash_hex: str
    ) -> LedgerResult: ...

    @abstractmethod
    def submit_global_model(
        self, round_id: int, model_version: int, hash_hex: str
    ) -> LedgerResult: ...

    @abstractmethod
    def verify_integrity(self, round_id: int, client_id: bytes, hash_hex: str) -> bool: ...

    @abstractmethod
    def verify_global(self, round_id: int, hash_hex: str) -> bool: ...

    @abstractmethod
    def get_audit_trail(self, round_id: int) -> list[dict]: ...


class SimulatedLedger(Ledger):
    """Hash-chain in-memory. Tiap blok memuat hash blok sebelumnya (tamper-evident)."""

    kind = "simulated-hashchain"

    def __init__(self):
        self._updates: dict[tuple[int, bytes], dict] = {}
        self._order: dict[int, list[bytes]] = {}
        self._globals: dict[int, dict] = {}
        self.chain: list[dict] = []

    def _append(self, payload: dict):
        prev = self.chain[-1]["block_hash"] if self.chain else "0" * 64
        body = json.dumps(payload, sort_keys=True)
        self.chain.append(
            {
                "payload": payload,
                "prev": prev,
                "block_hash": hashlib.sha256((prev + body).encode()).hexdigest(),
            }
        )

    def chain_is_valid(self) -> bool:
        prev = "0" * 64
        for blk in self.chain:
            body = json.dumps(blk["payload"], sort_keys=True)
            if (
                blk["prev"] != prev
                or blk["block_hash"] != hashlib.sha256((prev + body).encode()).hexdigest()
            ):
                return False
            prev = blk["block_hash"]
        return True

    def submit_model_update(self, round_id, client_id, model_version, hash_hex):
        t0 = time.perf_counter()
        if set(hash_hex) == {"0"}:
            return LedgerResult(False, 0, None, "ZeroHash")
        if (round_id, client_id) in self._updates:
            return LedgerResult(False, 0, None, "AlreadyRecorded")
        rec = {"hash": hash_hex, "model_version": model_version, "timestamp": int(time.time())}
        self._updates[(round_id, client_id)] = rec
        self._order.setdefault(round_id, []).append(client_id)
        self._append({"type": "update", "round": round_id, "client": client_id.hex(), **rec})
        return LedgerResult(True, time.perf_counter() - t0)

    def submit_global_model(self, round_id, model_version, hash_hex):
        t0 = time.perf_counter()
        if set(hash_hex) == {"0"}:
            return LedgerResult(False, 0, None, "ZeroHash")
        if round_id in self._globals:
            return LedgerResult(False, 0, None, "AlreadyRecorded")
        rec = {"hash": hash_hex, "model_version": model_version, "timestamp": int(time.time())}
        self._globals[round_id] = rec
        self._append({"type": "global", "round": round_id, **rec})
        return LedgerResult(True, time.perf_counter() - t0)

    def verify_integrity(self, round_id, client_id, hash_hex):
        r = self._updates.get((round_id, client_id))
        return r is not None and r["hash"] == hash_hex

    def verify_global(self, round_id, hash_hex):
        g = self._globals.get(round_id)
        return g is not None and g["hash"] == hash_hex

    def get_audit_trail(self, round_id):
        return [
            {"client_id": c.hex(), **self._updates[(round_id, c)]}
            for c in self._order.get(round_id, [])
        ]


class EvmLedger(Ledger):
    """Kontrak AuditTrail.sol via web3.py."""

    kind = "evm"

    def __init__(
        self,
        rpc_url: str | None = None,
        artifact: str | Path = Path(__file__).resolve().parents[1]
        / "chain"
        / "build"
        / "AuditTrail.json",
    ):
        from web3 import Web3

        if rpc_url is None:
            from web3 import EthereumTesterProvider

            self.w3 = Web3(EthereumTesterProvider())
            self.kind = "evm-inprocess(py-evm)"
        else:
            self.w3 = Web3(Web3.HTTPProvider(rpc_url))
            self.kind = f"evm-rpc({rpc_url})"
        art = json.loads(Path(artifact).read_text())
        self.gateway = self.w3.eth.accounts[0]
        factory = self.w3.eth.contract(abi=art["abi"], bytecode=art["bytecode"])
        tx = factory.constructor(self.gateway).transact({"from": self.gateway})
        rc = self.w3.eth.wait_for_transaction_receipt(tx)
        self.deploy_gas = rc.gasUsed
        self.c = self.w3.eth.contract(address=rc.contractAddress, abi=art["abi"])

    @staticmethod
    def _b32(hex_str: str) -> bytes:
        return bytes.fromhex(hex_str)

    def _send(self, fn) -> LedgerResult:
        t0 = time.perf_counter()
        try:
            tx = fn.transact({"from": self.gateway})
            rc = self.w3.eth.wait_for_transaction_receipt(tx)
            if rc.status != 1:
                return LedgerResult(False, time.perf_counter() - t0, rc.gasUsed, "reverted")
            return LedgerResult(True, time.perf_counter() - t0, rc.gasUsed)
        except Exception as e:  # revert saat estimasi gas pada provider tester
            return LedgerResult(
                False, time.perf_counter() - t0, None, type(e).__name__ + ": " + str(e)[:120]
            )

    def submit_model_update(self, round_id, client_id, model_version, hash_hex):
        return self._send(
            self.c.functions.submitModelUpdate(
                round_id, client_id, model_version, self._b32(hash_hex)
            )
        )

    def submit_global_model(self, round_id, model_version, hash_hex):
        return self._send(
            self.c.functions.submitGlobalModel(round_id, model_version, self._b32(hash_hex))
        )

    def verify_integrity(self, round_id, client_id, hash_hex):
        return bool(
            self.c.functions.verifyIntegrity(round_id, client_id, self._b32(hash_hex)).call()
        )

    def verify_global(self, round_id, hash_hex):
        return bool(self.c.functions.verifyGlobal(round_id, self._b32(hash_hex)).call())

    def get_audit_trail(self, round_id):
        ids, hashes, vers, ts = self.c.functions.getAuditTrail(round_id).call()
        return [
            {"client_id": i.hex(), "hash": h.hex(), "model_version": v, "timestamp": t}
            for i, h, v, t in zip(ids, hashes, vers, ts)
        ]
