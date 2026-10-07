import numpy as np

from flsys import data
from flsys import model as M
from flsys.metrics import auc_roc


def test_dimensions():
    assert data.N_FEATURES == 15 and M.n_params() == 1057
    assert sorted(i for v in data.FEATURE_GROUPS.values() for i in v) == list(range(15))


def test_preprocess_fixed_bounds_and_range():
    df = data.make_synthetic_diabetes(500, 1)
    X = data.preprocess(df)
    assert X.shape == (500, 15) and X.min() >= 0 and X.max() <= 1
    # one-hot: tepat satu aktif per grup kategorikal
    for g in ("gender", "smoking_history"):
        assert (X[:, data.FEATURE_GROUPS[g]].sum(1) == 1).all()


def test_preprocess_independent_of_other_rows():
    """Normalisasi tidak boleh bergantung pada data lain (tidak ada kebocoran statistik)."""
    df = data.make_synthetic_diabetes(200, 2)
    a = data.preprocess(df.iloc[:1])
    b = data.preprocess(df)[:1]
    assert np.array_equal(a, b)


def test_gradient_matches_finite_difference():
    rng = np.random.default_rng(0)
    w = M.init_params(0).astype(np.float64).astype(np.float32)
    X = rng.random((16, 15)).astype(np.float32)
    y = (rng.random(16) > 0.6).astype(np.float32)
    _, g = M._loss_grad(w, X, y, 3.0)
    for idx in rng.choice(len(w), 12, replace=False):
        e = 1e-3
        wp, wm = w.copy(), w.copy()
        wp[idx] += e
        wm[idx] -= e
        num = (M._loss_grad(wp, X, y, 3.0)[0] - M._loss_grad(wm, X, y, 3.0)[0]) / (2 * e)
        assert abs(num - g[idx]) < 5e-3, (idx, num, g[idx])


def test_training_improves_auc_on_synthetic():
    X, y, src = data.load_dataset(None, seed=0, n_synth=6000)
    sp = data.stratified_split(X, y, 0)
    w0 = M.init_params(0)
    w = M.train_local(w0, sp.X_train, sp.y_train, epochs=8, lr=0.05, batch_size=64, seed=0)
    a0 = auc_roc(sp.y_val, M.predict_proba(w0, sp.X_val))
    a1 = auc_roc(sp.y_val, M.predict_proba(w, sp.X_val))
    assert a1 > a0 and a1 > 0.8


def test_partitions():
    y = (np.random.default_rng(0).random(5000) < 0.09).astype(int)
    iid = data.partition_iid(len(y), 10)
    assert sum(len(p) for p in iid) == len(y)
    nid = data.partition_dirichlet(y, 10, 0.3, seed=1)
    assert sorted(np.concatenate(nid).tolist()) == list(range(len(y)))  # tidak ada duplikat/hilang
    rates = [y[p].mean() for p in nid]
    assert max(rates) - min(rates) > 0.03  # benar-benar heterogen
