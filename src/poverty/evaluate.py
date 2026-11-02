"""Survey-weighted evaluation, threshold selection, targeting errors and a
PSU-cluster bootstrap for confidence intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, fbeta_score,
                             log_loss, precision_score, recall_score, roc_auc_score)


def best_threshold(y, s, w=None, beta=1.0):
    """Threshold maximising weighted F-beta on (out-of-fold) scores."""
    grid = np.unique(np.quantile(s, np.linspace(0.50, 0.99, 197)))
    vals = [fbeta_score(y, (s >= t).astype(int), beta=beta, sample_weight=w, zero_division=0)
            for t in grid]
    return float(grid[int(np.argmax(vals))])


def targeting_errors(y, s, w, coverage):
    """Select the highest-scoring households until the weighted share of
    households selected equals ``coverage``. Returns (exclusion, inclusion) error:
      exclusion = share of poor households NOT selected (under-coverage)
      inclusion = share of selected households that are NOT poor (leakage)."""
    order = np.argsort(-s)
    cw = np.cumsum(w[order]) / w.sum()
    sel = np.zeros(len(s), bool)
    sel[order[cw <= coverage]] = True
    return selection_errors(y, sel, w)


def selection_errors(y, sel, w):
    poor = y == 1
    excl = w[poor & ~sel].sum() / w[poor].sum()
    incl = w[sel & ~poor].sum() / max(w[sel].sum(), 1e-12)
    return float(excl), float(incl)


def metrics(y, s, w, threshold, is_prob=True, poverty_rate=None):
    pred = (s >= threshold).astype(int)
    out = {
        "ROC_AUC": roc_auc_score(y, s, sample_weight=w),
        "PR_AUC": average_precision_score(y, s, sample_weight=w),
        "Recall": recall_score(y, pred, sample_weight=w, zero_division=0),
        "Precision": precision_score(y, pred, sample_weight=w, zero_division=0),
        "F1": f1_score(y, pred, sample_weight=w, zero_division=0),
        "F2": fbeta_score(y, pred, beta=2, sample_weight=w, zero_division=0),
        "Brier": brier_score_loss(y, s, sample_weight=w) if is_prob else np.nan,
        "LogLoss": log_loss(y, np.clip(s, 1e-6, 1 - 1e-6), sample_weight=w) if is_prob else np.nan,
    }
    if poverty_rate is not None:
        e, i = targeting_errors(y, s, w, poverty_rate)
        out["Exclusion_err@PR"], out["Inclusion_err@PR"] = e, i
    return out


def cluster_bootstrap(y, scores: dict, w, groups, n_boot=1000, seed=0, metric="PR_AUC"):
    """Resample PSUs with replacement; returns DataFrame of metric draws per model."""
    rng = np.random.default_rng(seed)
    g = pd.Series(groups).reset_index(drop=True)
    idx_by_g = g.groupby(g).indices
    keys = np.array(list(idx_by_g))
    fn = average_precision_score if metric == "PR_AUC" else roc_auc_score
    rows = []
    for _ in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([idx_by_g[k] for k in pick])
        if y[idx].sum() == 0:
            continue
        rows.append({m: fn(y[idx], s[idx], sample_weight=w[idx]) for m, s in scores.items()})
    return pd.DataFrame(rows)


def ci(series, alpha=0.05):
    return float(series.quantile(alpha / 2)), float(series.quantile(1 - alpha / 2))
