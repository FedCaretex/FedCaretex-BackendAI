from __future__ import annotations

import numpy as np


def auc_roc(y, p) -> float:
    y = np.asarray(y)
    p = np.asarray(p)
    pos, neg = p[y > 0.5], p[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    allv = np.concatenate([pos, neg])
    # tangani ties dengan rank rata-rata
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def classification_metrics(y, p, threshold: float = 0.5) -> dict:
    y = np.asarray(y) > 0.5
    pred = np.asarray(p) >= threshold
    tp = int((pred & y).sum())
    tn = int((~pred & ~y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    acc = (tp + tn) / max(len(y), 1)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc_roc(y, p),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
