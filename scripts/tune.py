"""Bayesian hyperparameter search (Optuna TPE) for the main boosting/ensemble
learners, using PSU-grouped stratified 3-fold CV on the training set only and
the survey-weighted average precision (PR-AUC) as the objective.

Every trial is logged to MLflow (experiment ``hies-poverty-tuning``).
Output: models/best_params.json
"""
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import mlflow
import numpy as np
import optuna
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from poverty.config import MODELS_DIR, RANDOM_STATE, ROOT
from poverty.data import load, split, xyw
from poverty.models import fit_any, make_any, score

optuna.logging.set_verbosity(optuna.logging.WARNING)

SPACES = {
    "LightGBM": (40, lambda t: dict(
        n_estimators=t.suggest_int("n_estimators", 300, 1500, step=100),
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.1, log=True),
        num_leaves=t.suggest_int("num_leaves", 8, 96, log=True),
        min_child_samples=t.suggest_int("min_child_samples", 10, 150, log=True),
        subsample=t.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
        reg_lambda=t.suggest_float("reg_lambda", 1e-3, 20, log=True),
        reg_alpha=t.suggest_float("reg_alpha", 1e-3, 5, log=True),
        scale_pos_weight=t.suggest_float("scale_pos_weight", 1.0, 7.0))),
    "XGBoost": (15, lambda t: dict(
        n_estimators=t.suggest_int("n_estimators", 300, 1500, step=100),
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.1, log=True),
        max_depth=t.suggest_int("max_depth", 3, 9),
        min_child_weight=t.suggest_float("min_child_weight", 1, 30, log=True),
        subsample=t.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
        reg_lambda=t.suggest_float("reg_lambda", 1e-2, 30, log=True),
        gamma=t.suggest_float("gamma", 0.0, 5.0),
        scale_pos_weight=t.suggest_float("scale_pos_weight", 1.0, 7.0))),
    "CatBoost": (8, lambda t: dict(
        iterations=t.suggest_int("iterations", 400, 1200, step=100),
        learning_rate=t.suggest_float("learning_rate", 0.02, 0.15, log=True),
        depth=t.suggest_int("depth", 4, 8),
        l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1, 30, log=True),
        random_strength=t.suggest_float("random_strength", 0.1, 5, log=True),
        bagging_temperature=t.suggest_float("bagging_temperature", 0.0, 1.0))),
    "EBM": (5, lambda t: dict(
        interactions=t.suggest_categorical("interactions", [0, 5, 10, 20]),
        learning_rate=t.suggest_categorical("learning_rate", [0.01, 0.02, 0.05]),
        max_leaves=t.suggest_categorical("max_leaves", [2, 3]),
        min_samples_leaf=t.suggest_categorical("min_samples_leaf", [10, 20, 50]))),
    "LightGBM PMT regressor": (12, lambda t: dict(
        n_estimators=t.suggest_int("n_estimators", 400, 2000, step=100),
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.08, log=True),
        num_leaves=t.suggest_int("num_leaves", 8, 96, log=True),
        min_child_samples=t.suggest_int("min_child_samples", 10, 150, log=True),
        subsample=t.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
        reg_lambda=t.suggest_float("reg_lambda", 1e-3, 20, log=True))),
    "Extra Trees": (6, lambda t: dict(
        n_estimators=t.suggest_int("n_estimators", 300, 800, step=100),
        max_features=t.suggest_float("max_features", 0.15, 0.8),
        min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 25, log=True))),
}


def main(names):
    df = load()
    tr, _ = split(df)
    dtr = df.iloc[tr].reset_index(drop=True)
    X, y, w, feats, nums, cats = xyw(dtr)
    yreg = np.log(dtr.pce.to_numpy())
    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=RANDOM_STATE)
                 .split(dtr, y, groups=dtr.psu))
    out_path = MODELS_DIR / "best_params.json"
    best = json.loads(out_path.read_text()) if out_path.exists() else {}

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("hies-poverty-tuning")
    for name in names:
        n_trials, space = SPACES[name]
        t0 = time.time()
        with mlflow.start_run(run_name=f"tune-{name}"):
            def objective(trial):
                params = space(trial)
                aps = []
                for k, (a, b) in enumerate(folds):
                    m = make_any(name, nums, cats, params)
                    fit_any(m, X.iloc[a], y[a], w[a], name, yreg[a])
                    aps.append(average_precision_score(y[b], score(m, X.iloc[b]), sample_weight=w[b]))
                    trial.report(np.mean(aps), k)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                with mlflow.start_run(run_name=f"{name}-trial-{trial.number}", nested=True):
                    mlflow.log_params(params)
                    mlflow.log_metric("cv_wAP", float(np.mean(aps)))
                print(f"  {name} trial {trial.number}: AP={np.mean(aps):.4f}", flush=True)
                return float(np.mean(aps))

            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
                                        pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
            study.optimize(objective, n_trials=n_trials)
            mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
            mlflow.log_metric("best_cv_wAP", study.best_value)
            best[name] = {"params": study.best_params, "cv_wAP": study.best_value,
                          "n_trials": n_trials, "minutes": round((time.time() - t0) / 60, 1)}
            study.trials_dataframe().to_csv(ROOT / "reports" / "tables" / f"optuna_{name.replace(' ', '_')}.csv",
                                            index=False)
        out_path.write_text(json.dumps(best, indent=2))
        print(f"{name}: best wAP={study.best_value:.4f} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(SPACES))
