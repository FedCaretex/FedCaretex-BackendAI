import numpy as np
import pytest

from flsys import data
from flsys import model as M
from flsys.shap_exact import N_G, _mix, baseline_from, exact_shap


@pytest.fixture(scope="module")
def trained():
    X, y, _ = data.load_dataset(None, seed=0, n_synth=5000)
    sp = data.stratified_split(X, y, 0)
    w = M.train_local(
        M.init_params(0), sp.X_train, sp.y_train, epochs=5, lr=0.05, batch_size=64, seed=0
    )
    return w, sp, baseline_from(sp.X_val)


def test_efficiency_property(trained):
    w, sp, base = trained
    for i in range(5):
        x = sp.X_test[i]
        phi = exact_shap(w, x, base)
        fx = M.predict_proba(w, x[None])[0]
        fb = M.predict_proba(w, base[None])[0]
        assert abs(phi.sum() - (fx - fb)) < 1e-5


def test_matches_shap_library_exact(trained):
    """Validasi silang dengan shap.KernelExplainer (enumerasi penuh 2^8)."""
    shap = pytest.importorskip("shap")
    w, sp, base = trained
    x = sp.X_test[3]

    def f(Z):  # Z: (n, 8) indikator koalisi
        Z = np.atleast_2d(Z)
        batch = np.stack([_mix(x, base, z) for z in Z]).astype(np.float32)
        return M.predict_proba(w, batch)

    ex = shap.KernelExplainer(f, np.zeros((1, N_G)))
    ref = np.asarray(ex.shap_values(np.ones((1, N_G)), nsamples=2**N_G, silent=True)).reshape(-1)
    mine = exact_shap(w, x, base)
    assert np.max(np.abs(ref - mine)) < 1e-4, (ref, mine)


def test_dummy_feature_gets_zero(trained):
    w, sp, base = trained
    x = sp.X_test[0].copy()
    x2 = x.copy()
    # fitur 'gender' diganti ke baseline -> kontribusinya harus 0
    from flsys.data import FEATURE_GROUPS

    x2[FEATURE_GROUPS["gender"]] = base[FEATURE_GROUPS["gender"]]
    phi = exact_shap(w, x2, base)
    assert abs(phi[0]) < 1e-6
