"""Simulasi beberapa klien (pengganti sementara aplikasi Flutter) yang berbicara ke
gateway lewat HTTP sungguhan.

Tiap klien: buat keypair Ed25519 -> register -> unduh model global & VERIFIKASI hash
-> latih lokal -> DP -> hash -> tanda tangan -> submit. Data tiap klien adalah partisi
dataset yang dipegang skrip ini (BUKAN data server).

Contoh:
  uvicorn app.main:app --port 8000                       # terminal 1
  python scripts/simulate_clients.py --clients 3 --rounds 2  # terminal 2
  python scripts/simulate_clients.py --tamper 1          # klien #1 merusak payload -> ditolak
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flsys import data as D  # noqa: E402
from flsys.client import build_update, decode_global  # noqa: E402
from flsys.identity import ClientIdentity  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description="Simulasi klien FL via HTTP")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--csv", default=None, help="CSV dataset untuk data klien (kosong = sintetis)")
    p.add_argument("--clients", type=int, default=3)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.0, help=">0 -> partisi non-IID Dirichlet")
    p.add_argument("--tamper", type=int, default=-1, help="indeks klien yang merusak payload")
    p.add_argument("--seed", type=int, default=123)
    a = p.parse_args(argv)

    X, y, src = D.load_dataset(a.csv, seed=a.seed, n_synth=6000)
    parts = (
        D.partition_dirichlet(y, a.clients, a.alpha, a.seed)
        if a.alpha > 0
        else D.partition_iid(len(y), a.clients, a.seed)
    )
    ids = [ClientIdentity(seed=f"sim-client-{a.seed}-{i}".encode()) for i in range(a.clients)]
    print(f"data klien: {src}, ukuran partisi: {[len(i) for i in parts]}")

    with httpx.Client(base_url=a.url, timeout=120) as http:
        for ident in ids:
            http.post(
                "/fl/register-client", json={"public_key": ident.public_key.hex()}
            ).raise_for_status()
        for _ in range(a.rounds):
            g = http.get("/fl/global-model").json()
            w = decode_global(g)
            print(
                f"\n== ronde {g['round_id']} | model v{g['model_version']} | hash {g['hash'][:16]}... "
                f"(ledger_verified={g['ledger_verified']})"
            )
            for ci, (ident, idx) in enumerate(zip(ids, parts)):
                upd = build_update(
                    w,
                    X[idx],
                    y[idx],
                    round_id=g["round_id"],
                    identity=ident,
                    training=g["training"],
                    dp=g["dp"],
                    seed=a.seed * 1000 + ci,
                )
                body = upd.to_request(ident)
                if ci == a.tamper:
                    raw = bytearray(upd.payload)
                    raw[100] ^= 1
                    body["payload_b64"] = base64.b64encode(bytes(raw)).decode()
                r = http.post("/fl/submit-update", json=body)
                if r.status_code == 200:
                    j = r.json()
                    msg = f"diterima (eps kumulatif={j['client_epsilon']})"
                    if j["aggregated"]:
                        ag = j["aggregation"]
                        msg += (
                            f" -> AGREGASI ronde {ag['round']}: val AUC={ag['val']['auc']:.4f} "
                            f"F1={ag['val']['f1']:.4f}"
                        )
                else:
                    msg = f"DITOLAK {r.status_code}: {r.json()['detail']}"
                print(f"  klien {ci} ({ident.client_id.hex()[:10]}...): {msg}")
            st = http.get("/fl/status").json()
            if st["pending_in_round"]:
                print(
                    f"  {st['pending_in_round']} update menunggu (< min_clients={st['min_clients']}); "
                    "panggil POST /fl/aggregate untuk memaksa agregasi"
                )


if __name__ == "__main__":
    main()
