import numpy as np
import pytest

from flsys import data
from flsys import model as M
from flsys.fedlearn import FLConfig, run_federated
from flsys.ledger import EvmLedger, SimulatedLedger
from flsys.metrics import auc_roc


@pytest.fixture(scope="module")
def world():
    X, y, _ = data.load_dataset(None, seed=0, n_synth=6000)
    sp = data.stratified_split(X, y, 0)
    parts = data.partition_iid(len(sp.y_train), 5, 0)
    cd = [(sp.X_train[p], sp.y_train[p]) for p in parts]
    return sp, cd


def _run(world, **kw):
    sp, cd = world
    cfg = FLConfig(**{"rounds": 6, "local_epochs": 1, **kw})
    led = kw.pop("ledger", None)
    return cfg, sp, cd


def test_fedavg_learns(world):
    sp, cd = world
    w0 = M.init_params(0)
    out = run_federated(
        FLConfig(rounds=8, local_epochs=2), cd, sp.X_val, sp.y_val, sp.X_test, sp.y_test
    )
    assert out["final_val"]["auc"] > auc_roc(sp.y_val, M.predict_proba(w0, sp.X_val)) + 0.1
    assert out["history"][-1]["val_auc"] > 0.8


def test_fedprox_requires_mu():
    with pytest.raises(ValueError):
        FLConfig(algo="fedprox", prox_mu=0).validate()


def test_deterministic_given_seed(world):
    sp, cd = world
    a = run_federated(
        FLConfig(rounds=3, use_dp=True, noise_multiplier=0.5),
        cd,
        sp.X_val,
        sp.y_val,
        sp.X_test,
        sp.y_test,
    )
    b = run_federated(
        FLConfig(rounds=3, use_dp=True, noise_multiplier=0.5),
        cd,
        sp.X_val,
        sp.y_val,
        sp.X_test,
        sp.y_test,
    )
    assert np.array_equal(a["params"], b["params"])


def test_dp_reports_epsilon_and_hurts_with_big_noise(world):
    sp, cd = world
    clean = run_federated(
        FLConfig(rounds=8, local_epochs=2), cd, sp.X_val, sp.y_val, sp.X_test, sp.y_test
    )
    noisy = run_federated(
        FLConfig(rounds=8, local_epochs=2, use_dp=True, clip_norm=0.5, noise_multiplier=5.0),
        cd,
        sp.X_val,
        sp.y_val,
        sp.X_test,
        sp.y_test,
    )
    assert np.isfinite(noisy["epsilon_rdp"]) and "epsilon_rdp" not in clean
    assert noisy["final_val"]["auc"] < clean["final_val"]["auc"]


@pytest.mark.parametrize("make", [SimulatedLedger, EvmLedger])
def test_tamper_is_rejected_and_not_recorded(world, make):
    sp, cd = world
    L = make()

    def tamper(ci, r, payload):  # klien 2 diserang pada ronde 2: balik 1 byte
        if ci == 2 and r == 2:
            b = bytearray(payload)
            b[50] ^= 0x01
            return bytes(b)
        return payload

    out = run_federated(
        FLConfig(rounds=3, local_epochs=1),
        cd,
        sp.X_val,
        sp.y_val,
        sp.X_test,
        sp.y_test,
        ledger=L,
        tamper=tamper,
    )
    assert out["anomalies"] == [{"round": 2, "client": 2, "type": "integrity_failed"}]
    assert [h["accepted"] for h in out["history"]] == [5, 4, 5]
    assert len(L.get_audit_trail(1)) == 5 and len(L.get_audit_trail(2)) == 4
    # model global akhir sesuai catatan on-chain
    from flsys.serialize import hash_params

    assert L.verify_global(3, hash_params(out["params"]))


def test_ledger_matches_what_clients_sent(world):
    sp, cd = world
    L = EvmLedger()
    out = run_federated(
        FLConfig(rounds=2, local_epochs=1), cd, sp.X_val, sp.y_val, sp.X_test, sp.y_test, ledger=L
    )
    assert all(len(L.get_audit_trail(r)) == 5 for r in (1, 2))
    assert out["history"][0]["gas"] > 0 and out["history"][0]["t_ledger_s"] > 0


def test_blockchain_does_not_change_model(world):
    """Blockchain hanya mencatat; bobot model harus identik dengan/tanpa ledger."""
    sp, cd = world
    a = run_federated(
        FLConfig(rounds=3, local_epochs=1), cd, sp.X_val, sp.y_val, sp.X_test, sp.y_test
    )
    b = run_federated(
        FLConfig(rounds=3, local_epochs=1),
        cd,
        sp.X_val,
        sp.y_val,
        sp.X_test,
        sp.y_test,
        ledger=EvmLedger(),
    )
    assert np.array_equal(a["params"], b["params"])
