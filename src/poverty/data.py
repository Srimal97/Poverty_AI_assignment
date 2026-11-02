"""Load the processed table and create a reproducible, cluster-aware split."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import PROCESSED_DIR, RANDOM_STATE, N_FOLDS
from .features import build_household_table, feature_lists


def load(rebuild: bool = False) -> pd.DataFrame:
    path = PROCESSED_DIR / "households.parquet"
    if rebuild or not path.exists():
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df = build_household_table()
        df.to_parquet(path)
    return pd.read_parquet(path)


def split(df: pd.DataFrame):
    """80/20 train/test split stratified on the target and grouped by PSU, so that
    households from the same sampling cluster never sit on both sides."""
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    tr, te = next(sgk.split(df, df.poor, groups=df.psu))
    return np.sort(tr), np.sort(te)


def cv_folds(df_train: pd.DataFrame):
    sgk = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return list(sgk.split(df_train, df_train.poor, groups=df_train.psu))


def weights(df: pd.DataFrame) -> np.ndarray:
    """DCS household design weight, normalised to mean 1 (keeps learning rates sane)."""
    w = df.finalweight.to_numpy(dtype=float)
    return w / w.mean()


def xyw(df):
    feats, nums, cats = feature_lists(df)
    return df[feats], df.poor.to_numpy(), weights(df), feats, nums, cats
