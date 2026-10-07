"""Runner eksperimen E1-E6 (sesuai Tabel 5 naskah).

E1 Centralized | E2 FedAvg IID | E3 FedAvg Non-IID | E4 FedProx Non-IID
E5 FedProx + DP (grid sigma) | E6 = E5(sigma terpilih) + Blockchain (EVM) + SHAP

Aturan metodologis yang ditegakkan kode:
 - mu FedProx dipilih pada DATA VALIDASI, bukan data uji.
 - metrik dilaporkan sebagai mean +- SD atas beberapa seed.
 - hasil dari data SINTETIS diberi awalan SYNTHETIC_ dan peringatan; TIDAK boleh masuk naskah.
 - E6 memverifikasi bahwa blockchain tidak mengubah bobot model (identik dengan E5).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import data as D
from . import model as M
from .fedlearn import FLConfig, run_centralized, run_federated
from .ledger import EvmLedger
from .serialize import hash_params
from .shap_exact import baseline_from, exact_shap

METRICS = ("accuracy", "precision", "recall", "f1", "auc")


def _clients(sp, parts):
    return [(sp.X_train[p], sp.y_train[p]) for p in parts]


def run_seed(seed: int, X, y, a) -> dict:
    sp = D.stratified_split(X, y, seed)
    iid = D.partition_iid(len(sp.y_train), a.clients, seed)
    noniid = D.partition_dirichlet(sp.y_train, a.clients, a.alpha, seed)
    cd_iid, cd_non = _clients(sp, iid), _clients(sp, noniid)
    ev = (sp.X_val, sp.y_val, sp.X_test, sp.y_test)
    base = dict(
        rounds=a.rounds, local_epochs=a.local_epochs, lr=a.lr, batch_size=a.batch_size, seed=seed
    )
    res: dict = {"seed": seed}

    res["E1"] = run_centralized(
        sp.X_train,
        sp.y_train,
        *ev[:2],
        *ev[2:],
        epochs=a.rounds * a.local_epochs,
        lr=a.lr,
        batch_size=a.batch_size,
        seed=seed,
    )["final_test"]
    res["E2"] = run_federated(FLConfig(algo="fedavg", **base), cd_iid, *ev)["final_test"]
    res["E3"] = run_federated(FLConfig(algo="fedavg", **base), cd_non, *ev)["final_test"]

    # E4: pilih mu di VALIDASI
    best_mu, best_val, best = None, -1.0, None
    mu_log = {}
    for mu in a.mu_grid:
        o = run_federated(FLConfig(algo="fedprox", prox_mu=mu, **base), cd_non, *ev)
        mu_log[str(mu)] = o["final_val"]["auc"]
        if o["final_val"]["auc"] > best_val:
            best_mu, best_val, best = mu, o["final_val"]["auc"], o
    res["E4"] = best["final_test"]
    res["E4_mu"] = best_mu
    res["E4_mu_val_auc"] = mu_log

    # E5: grid sigma
    res["E5"] = {}
    e5_out = {}
    for sg in a.sigma_grid:
        o = run_federated(
            FLConfig(
                algo="fedprox",
                prox_mu=best_mu,
                use_dp=True,
                clip_norm=a.clip,
                noise_multiplier=sg,
                **base,
            ),
            cd_non,
            *ev,
        )
        res["E5"][str(sg)] = {**o["final_test"], "epsilon_rdp": o["epsilon_rdp"], "delta": 1e-5}
        e5_out[sg] = o

    # E6: konfigurasi E5(sigma_e6) + ledger EVM + SHAP
    sg6 = a.e6_sigma
    ledger = EvmLedger(a.rpc_url)
    cfg6 = FLConfig(
        algo="fedprox", prox_mu=best_mu, use_dp=True, clip_norm=a.clip, noise_multiplier=sg6, **base
    )
    o6 = run_federated(cfg6, cd_non, *ev, ledger=ledger)
    h = o6["history"]
    res["E6"] = {
        **o6["final_test"],
        "epsilon_rdp": o6["epsilon_rdp"],
        "ledger": ledger.kind,
        "same_model_as_without_ledger": (
            bool(np.array_equal(o6["params"], e5_out[sg6]["params"])) if sg6 in e5_out else None
        ),
        "rejected_total": int(sum(r["rejected"] for r in h)),
        "overhead": {
            "payload_bytes_per_client": float(
                np.mean([r["payload_bytes"] / max(r["accepted"] + r["rejected"], 1) for r in h])
            ),
            "t_hash_ms_per_client": 1000 * float(np.mean([r["t_hash_s"] / a.clients for r in h])),
            "t_ledger_ms_per_client": 1000
            * float(np.mean([r["t_ledger_s"] / a.clients for r in h])),
            "gas_per_round": float(np.mean([r["gas"] for r in h])),
            "t_train_s_per_round": float(np.mean([r["t_train_s"] for r in h])),
        },
    }
    # SHAP eksak pada sampel uji
    n_s = min(a.shap_samples, len(sp.y_test))
    base_x = baseline_from(sp.X_val)
    t0 = time.perf_counter()
    phis, eff = [], []
    for i in range(n_s):
        x = sp.X_test[i]
        phi = exact_shap(o6["params"], x, base_x)
        phis.append(phi)
        eff.append(
            abs(
                phi.sum()
                - (
                    M.predict_proba(o6["params"], x[None])[0]
                    - M.predict_proba(o6["params"], base_x[None])[0]
                )
            )
        )
    res["E6"]["shap"] = {
        "samples": n_s,
        "ms_per_sample": 1000 * (time.perf_counter() - t0) / max(n_s, 1),
        "max_efficiency_error": float(max(eff)),
        "mean_abs_phi": dict(zip(D.RAW_FEATURES, np.abs(np.stack(phis)).mean(0).round(5).tolist())),
    }
    # tamper demo pada ronde terakhir: hash model akhir harus cocok dgn ledger
    res["E6"]["final_global_hash_verified"] = bool(
        ledger.verify_global(a.rounds, hash_params(o6["params"]))
    )
    return res


def _ms(vals):
    v = np.asarray(vals, dtype=float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0}


def summarize(runs: list[dict], a) -> dict:
    S = {}
    for e in ("E1", "E2", "E3", "E4"):
        S[e] = {m: _ms([r[e][m] for r in runs]) for m in METRICS}
    S["E4"]["mu_chosen_per_seed"] = [r["E4_mu"] for r in runs]
    S["E5"] = {}
    for sg in a.sigma_grid:
        k = str(sg)
        S["E5"][k] = {m: _ms([r["E5"][k][m] for r in runs]) for m in METRICS}
        S["E5"][k]["epsilon_rdp"] = runs[0]["E5"][k]["epsilon_rdp"]
    S["E6"] = {m: _ms([r["E6"][m] for r in runs]) for m in METRICS}
    S["E6"]["epsilon_rdp"] = runs[0]["E6"]["epsilon_rdp"]
    S["E6"]["ledger"] = runs[0]["E6"]["ledger"]
    S["E6"]["same_model_as_without_ledger_all_seeds"] = all(
        bool(r["E6"]["same_model_as_without_ledger"]) for r in runs
    )
    S["E6"]["overhead_mean"] = {
        k: float(np.mean([r["E6"]["overhead"][k] for r in runs])) for k in runs[0]["E6"]["overhead"]
    }
    S["E6"]["shap_ms_per_sample"] = float(np.mean([r["E6"]["shap"]["ms_per_sample"] for r in runs]))
    S["E6"]["shap_max_eff_err"] = float(max(r["E6"]["shap"]["max_efficiency_error"] for r in runs))
    S["E6"]["mean_abs_phi"] = {
        f: float(np.mean([r["E6"]["shap"]["mean_abs_phi"][f] for r in runs]))
        for f in D.RAW_FEATURES
    }
    return S


def markdown_table(S: dict, a, source: str) -> str:
    fm = lambda d: f"{d['mean']:.4f} ± {d['sd']:.4f}"
    head = (
        "| Skenario | "
        + " | ".join(m.capitalize() for m in METRICS)
        + " |\n|---|"
        + "---|" * len(METRICS)
        + "\n"
    )
    rows = []
    for e, name in (
        ("E1", "Centralized"),
        ("E2", "FedAvg IID"),
        ("E3", "FedAvg Non-IID"),
        ("E4", "FedProx Non-IID"),
    ):
        rows.append(f"| {e} {name} | " + " | ".join(fm(S[e][m]) for m in METRICS) + " |")
    for sg, v in S["E5"].items():
        rows.append(
            f"| E5 FedProx+DP (σ={sg}, ε≈{v['epsilon_rdp']:.1f}) | "
            + " | ".join(fm(v[m]) for m in METRICS)
            + " |"
        )
    rows.append(
        f"| E6 Terintegrasi (σ={a.e6_sigma}) | "
        + " | ".join(fm(S["E6"][m]) for m in METRICS)
        + " |"
    )
    warn = (
        "> ⚠️ DATA SINTETIS — hanya uji pipeline. JANGAN masukkan angka ini ke naskah.\n\n"
        if source == "synthetic"
        else ""
    )
    return warn + head + "\n".join(rows) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description="Jalankan eksperimen E1-E6")
    p.add_argument(
        "--csv", default=None, help="Path diabetes_prediction_dataset.csv (kosong = data sintetis)"
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--local-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--alpha", type=float, default=0.5, help="Dirichlet alpha (kecil = makin non-IID)"
    )
    p.add_argument("--mu-grid", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    p.add_argument("--sigma-grid", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--e6-sigma", type=float, default=1.0)
    p.add_argument("--shap-samples", type=int, default=50)
    p.add_argument("--rpc-url", default=None, help="RPC Hardhat/Besu; kosong = EVM in-process")
    p.add_argument("--n-synth", type=int, default=20000)
    p.add_argument("--out", default="results")
    a = p.parse_args(argv)
    if a.e6_sigma not in a.sigma_grid:
        p.error("--e6-sigma harus ada di --sigma-grid")

    X, y, source = D.load_dataset(a.csv, seed=0, n_synth=a.n_synth)
    if source == "synthetic":
        print("⚠️  DATA SINTETIS: hasil hanya untuk uji pipeline, bukan untuk naskah.")
    print(f"data={source} n={len(y)} positif={y.mean():.3%}")
    runs = []
    for s in a.seeds:
        t0 = time.time()
        print(f"== seed {s} ==", flush=True)
        runs.append(run_seed(s, X, y, a))
        print(f"   selesai {time.time()-t0:.0f}s", flush=True)
    S = summarize(runs, a)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    prefix = "SYNTHETIC_" if source == "synthetic" else ""
    (out / f"{prefix}summary.json").write_text(
        json.dumps(
            {"data_source": source, "args": vars(a), "summary": S, "runs": runs},
            indent=2,
            default=float,
        )
    )
    md = markdown_table(S, a, source)
    (out / f"{prefix}summary.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
