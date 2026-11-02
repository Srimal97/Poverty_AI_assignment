"""Model-agnostic SHAP for the stacked ensemble.

The stack mixes tree ensembles, a GAM and regression-based learners, so no
single fast TreeSHAP applies. We compute *exact-in-expectation* Shapley values
of the whole pipeline with the permutation algorithm on the stack's log-odds
output, using an independent background sample. Categorical features are
ordinal-encoded only for the masker; the model always receives the original
labelled DataFrame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class StackExplainer:
    def __init__(self, model, feats, cats, category_levels, background: pd.DataFrame):
        import shap
        self.model, self.feats, self.cats = model, list(feats), list(cats)
        self.levels = {c: list(category_levels[c]) for c in self.cats}
        bg = self.encode(background[self.feats])
        self.explainer = shap.explainers.Permutation(
            self._f, shap.maskers.Independent(bg, max_samples=len(bg)), feature_names=self.feats)

    # categorical <-> integer code (unknown level -> -1)
    def encode(self, X: pd.DataFrame) -> np.ndarray:
        Z = X[self.feats].copy()
        for c in self.cats:
            m = {v: i for i, v in enumerate(self.levels[c])}
            Z[c] = Z[c].map(m).fillna(-1)
        return Z.astype(float).to_numpy()

    def decode(self, A: np.ndarray) -> pd.DataFrame:
        Z = pd.DataFrame(A, columns=self.feats)
        for c in self.cats:
            lv = np.array(self.levels[c] + ["Unknown"], dtype=object)
            codes = Z[c].fillna(-1).astype(int).to_numpy()
            Z[c] = lv[np.where(codes < 0, len(lv) - 1, codes)]
        return Z

    def _f(self, A):
        p = self.model.predict_proba(self.decode(A))[:, 1]
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def shap_values(self, X: pd.DataFrame):
        A = self.encode(X)
        sv = self.explainer(A, max_evals=2 * len(self.feats) + 1, silent=True)
        sv.display_data = X[self.feats].to_numpy(dtype=object)
        return sv

    def top_reasons(self, X_one: pd.DataFrame, k=8):
        """Plain-language top drivers for one household (for the app)."""
        sv = self.shap_values(X_one)
        s = pd.Series(sv.values[0], index=self.feats)
        vals = X_one.iloc[0]
        out = []
        for f in s.abs().sort_values(ascending=False).index[:k]:
            out.append({"feature": f, "value": vals[f], "contribution": float(s[f]),
                        "direction": "raises poverty risk" if s[f] > 0 else "lowers poverty risk"})
        return pd.DataFrame(out), float(sv.base_values[0])
