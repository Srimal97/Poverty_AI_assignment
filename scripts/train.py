"""Train, cross-validate and test every model, build the proposed stacked
ensemble from the out-of-fold predictions, and export the deployable pickle.

Outputs
  models/poverty_model.pkl           deployable artefact (stack + metadata)
  models/oof_scores.parquet          out-of-fold scores (train) for every model
  models/test_scores.parquet         test-set scores for every model
  reports/tables/cv_results.csv      5-fold PSU-grouped CV mean +/- sd
  reports/tables/test_results.csv    held-out test metrics
All runs are logged to MLflow (experiment ``hies-poverty-models``).
"""
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import joblib
import mlflow
import numpy as np
import pandas as pd

from poverty.config import MODELS_DIR, POVERTY_LINE, ROOT, TAB_DIR
from poverty.data import cv_folds, load, split, xyw
from poverty.evaluate import best_threshold, metrics
from poverty.models import (COVERED, NEW, SurveyWeightedStack, fit_any, make_any, score)

STACK_BASE = ["CatBoost", "LightGBM", "XGBoost", "EBM", "Extra Trees",
              "LightGBM PMT regressor", "OLS PMT (existing practice)"]
NEW_ALL = NEW + ["LightGBM PMT regressor"]
log = lambda *a: print(*a, flush=True)


def main():
    t_all = time.time()
    df = load()
    smoke = bool(__import__("os").environ.get("SMOKE"))
    if smoke:
        df = df.sample(2500, random_state=0).reset_index(drop=True)
    tr, te = split(df)
    dtr, dte = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
    Xtr, ytr, wtr, feats, nums, cats = xyw(dtr)
    Xte, yte, wte, *_ = xyw(dte)
    folds = cv_folds(dtr)
    poverty_rate = float(np.average(ytr, weights=wtr))
    bp = MODELS_DIR / "best_params.json"
    best = json.loads(bp.read_text()) if bp.exists() else {}
    params = {k: v["params"] for k, v in best.items()}

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("hies-poverty-models")

    oof, test, fitted, cv_rows, timing = {}, {}, {}, [], {}
    names = ["OLS PMT (existing practice)"] + COVERED + NEW_ALL
    yreg = np.log(dtr.pce.to_numpy())
    if smoke:
        names = ["Logistic Regression"] + STACK_BASE
        for n in STACK_BASE:
            params[n] = {**(params.get(n) or {}), **({"iterations": 50} if n == "CatBoost" else {})}
    cache_dir = MODELS_DIR / "cache"
    cache_dir.mkdir(exist_ok=True)
    for name in names:
        cpath = cache_dir / f"{name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib"
        if cpath.exists() and not smoke:          # resumable: each model is cached once trained
            o, ts, m, fold_m, secs = joblib.load(cpath)
            oof[name], test[name], fitted[name], timing[name] = o, ts, m, secs
            fm = pd.DataFrame(fold_m)
        else:
            t0 = time.time()
            o = np.zeros(len(ytr))
            fold_m = []
            for k, (a, b) in enumerate(folds):
                m = fit_any(make_any(name, nums, cats, params.get(name), POVERTY_LINE),
                            Xtr.iloc[a], ytr[a], wtr[a], name, yreg[a])
                o[b] = score(m, Xtr.iloc[b])
                fold_m.append(metrics(ytr[b], o[b], wtr[b], 0.5, is_prob=name != "SVM (RBF)"))
            m = fit_any(make_any(name, nums, cats, params.get(name), POVERTY_LINE), Xtr, ytr, wtr, name, yreg)
            oof[name], test[name], fitted[name] = o, score(m, Xte), m
            timing[name] = time.time() - t0
            if not smoke:
                joblib.dump((o, test[name], m, fold_m, timing[name]), cpath)
            fm = pd.DataFrame(fold_m)
        row = {"model": name, "group": "Existing practice" if name.startswith("OLS") else
               ("Covered in lectures" if name in COVERED else "New (not covered)")}
        for c in ["ROC_AUC", "PR_AUC", "Brier"]:
            row[f"{c}_mean"], row[f"{c}_sd"] = fm[c].mean(), fm[c].std()
        cv_rows.append(row)
        log(f"{name:30s} CV AUC={row['ROC_AUC_mean']:.4f}±{row['ROC_AUC_sd']:.4f} "
            f"AP={row['PR_AUC_mean']:.4f}±{row['PR_AUC_sd']:.4f}  ({timing[name]:.0f}s)")

    # ------------------------------------------------ proposed stacked ensemble
    t0 = time.time()
    stack = SurveyWeightedStack({n: params.get(n) for n in STACK_BASE}, nums, cats)
    oof_base = np.column_stack([oof[n] for n in STACK_BASE])
    stack.oof_ = oof_base
    from sklearn.linear_model import LogisticRegression
    stack.meta_ = LogisticRegression(C=1.0, max_iter=2000).fit(stack._logit(oof_base), ytr, sample_weight=wtr)
    stack.base_models_ = {n: fitted[n] for n in STACK_BASE}
    stack.classes_ = np.array([0, 1])
    # honest OOF for the stack itself: refit the meta learner within each fold
    so = np.zeros(len(ytr))
    fold_m = []
    for a, b in folds:
        meta = LogisticRegression(C=1.0, max_iter=2000).fit(stack._logit(oof_base[a]), ytr[a], sample_weight=wtr[a])
        so[b] = meta.predict_proba(stack._logit(oof_base[b]))[:, 1]
        fold_m.append(metrics(ytr[b], so[b], wtr[b], 0.5))
    SN = "Stacked Ensemble (proposed)"
    oof[SN], test[SN], fitted[SN] = so, stack.predict_proba(Xte)[:, 1], stack
    fm = pd.DataFrame(fold_m)
    row = {"model": SN, "group": "Proposed"}
    for c in ["ROC_AUC", "PR_AUC", "Brier"]:
        row[f"{c}_mean"], row[f"{c}_sd"] = fm[c].mean(), fm[c].std()
    cv_rows.append(row)
    timing[SN] = sum(timing[n] for n in STACK_BASE) + time.time() - t0
    log(f"{SN:30s} CV AUC={row['ROC_AUC_mean']:.4f} AP={row['PR_AUC_mean']:.4f}  meta weights:\n{stack.meta_weights.round(3)}")

    # ablation: stack of ALL models (meta only, for comparison)
    allnames = [n for n in names]
    oof_all = np.column_stack([oof[n] for n in allnames])
    so2 = np.zeros(len(ytr))
    for a, b in folds:
        meta = LogisticRegression(C=1.0, max_iter=3000).fit(stack._logit(oof_all[a]), ytr[a], sample_weight=wtr[a])
        so2[b] = meta.predict_proba(stack._logit(oof_all[b]))[:, 1]
    meta_all = LogisticRegression(C=1.0, max_iter=3000).fit(stack._logit(oof_all), ytr, sample_weight=wtr)
    test["Stack of all models (ablation)"] = meta_all.predict_proba(
        stack._logit(np.column_stack([test[n] for n in allnames])))[:, 1]
    oof["Stack of all models (ablation)"] = so2

    # ------------------------------------------------ thresholds & test metrics
    rows = []
    thresholds = {}
    for name in oof:
        is_prob = name != "SVM (RBF)"
        t_f1 = best_threshold(ytr, oof[name], wtr, beta=1)
        t_f2 = best_threshold(ytr, oof[name], wtr, beta=2)
        thresholds[name] = {"F1": t_f1, "F2": t_f2}
        mt = metrics(yte, test[name], wte, t_f1, is_prob, poverty_rate)
        mt2 = metrics(yte, test[name], wte, t_f2, is_prob)
        group = next((r["group"] for r in cv_rows if r["model"] == name), "Ablation")
        rows.append({"model": name, "group": group, **mt,
                     "Recall@F2thr": mt2["Recall"], "Precision@F2thr": mt2["Precision"],
                     "train_seconds": timing.get(name, np.nan)})
    res = pd.DataFrame(rows).sort_values("PR_AUC", ascending=False)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(TAB_DIR / "test_results.csv", index=False)
    pd.DataFrame(cv_rows).sort_values("PR_AUC_mean", ascending=False).to_csv(TAB_DIR / "cv_results.csv", index=False)
    log(res[["model", "ROC_AUC", "PR_AUC", "Recall", "Precision", "F1", "Brier",
             "Exclusion_err@PR", "Inclusion_err@PR"]].round(4).to_string(index=False))

    pd.DataFrame(oof).assign(psu=dtr.psu.values, poor=ytr, w=wtr).to_parquet(MODELS_DIR / "oof_scores.parquet")
    pd.DataFrame(test).assign(psu=dte.psu.values, poor=yte, w=wte).to_parquet(MODELS_DIR / "test_scores.parquet")
    stack.meta_weights.rename("meta_coef").to_csv(TAB_DIR / "stack_meta_weights.csv")

    # ------------------------------------------------ export deployable artefact
    artefact = {
        "model": stack, "features": feats, "numeric": nums, "categorical": cats,
        "threshold_f1": thresholds[SN]["F1"], "threshold_f2": thresholds[SN]["F2"],
        "poverty_line": POVERTY_LINE, "train_poverty_rate": poverty_rate,
        "test_metrics": res.set_index("model").loc[SN].drop("group").to_dict(),
        "category_levels": {c: sorted(dtr[c].dropna().unique().tolist()) for c in cats},
        "numeric_defaults": dtr[nums].median().to_dict(),
        "categorical_defaults": {c: dtr[c].mode().iloc[0] for c in cats},
        "numeric_ranges": {c: [float(dtr[c].min()), float(dtr[c].max())] for c in nums},
        "version": "1.0.0", "trained_on": "DCS HIES 2019 (train split, 80%)",
    }
    if smoke:
        log("smoke OK"); return
    joblib.dump(artefact, MODELS_DIR / "poverty_model.pkl", compress=3)
    joblib.dump({n: fitted[n] for n in ["CatBoost", "LightGBM", "EBM", "Random Forest", "XGBoost"]},
                MODELS_DIR / "benchmarks.pkl", compress=3)
    json.dump(thresholds, open(MODELS_DIR / "thresholds.json", "w"), indent=2)

    # ------------------------------------------------ MLflow logging
    for _, r in res.iterrows():
        with mlflow.start_run(run_name=r["model"]):
            mlflow.set_tag("group", r["group"])
            p = params.get(r["model"]) or {}
            if p:
                mlflow.log_params(p)
            mlflow.log_metrics({f"test_{k.replace('@', '_at_')}": float(v) for k, v in r.items()
                                if k not in ("model", "group") and pd.notna(v)})
            cvr = next((c for c in cv_rows if c["model"] == r["model"]), None)
            if cvr:
                mlflow.log_metrics({f"cv_{k}": float(v) for k, v in cvr.items() if k not in ("model", "group")})
            if r["model"] == SN:
                mlflow.log_artifact(str(MODELS_DIR / "poverty_model.pkl"))
    log(f"done in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
