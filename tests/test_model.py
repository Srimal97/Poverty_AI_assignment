"""Smoke and contract tests for the deployable model (run in CI without raw data)."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from poverty.features import DERIVED, add_derived  # noqa: E402


@pytest.fixture(scope="module")
def art():
    return joblib.load(ROOT / "models" / "poverty_model.pkl")


@pytest.fixture(scope="module")
def sample():
    return json.loads((ROOT / "app" / "sample_household.json").read_text())


def test_artefact_contract(art):
    for k in ["model", "features", "numeric", "categorical", "threshold_f1", "threshold_f2",
              "category_levels", "numeric_defaults", "categorical_defaults"]:
        assert k in art
    assert set(art["numeric"]) | set(art["categorical"]) == set(art["features"])
    assert 0 < art["threshold_f2"] <= art["threshold_f1"] < 1


def test_no_leakage_or_protected_features(art):
    banned = {"pce", "hhexppm", "hhincomepm", "poor", "audit_samurdhi", "audit_head_ethnicity", "finalweight"}
    assert not banned & set(art["features"])
    assert not any("ethnic" in f or "religion" in f for f in art["features"])


def test_single_prediction(art, sample):
    X = pd.DataFrame([sample])[art["features"]]
    p = art["model"].predict_proba(X)[:, 1]
    assert p.shape == (1,) and 0.0 <= p[0] <= 1.0


def test_derived_features_match_training(art, sample):
    X = pd.DataFrame([sample])
    recomputed = add_derived(X.drop(columns=DERIVED))
    for f in DERIVED:
        assert np.isclose(recomputed[f].iloc[0], X[f].iloc[0], equal_nan=True), f


def test_monotone_sanity(art, sample):
    """More assets and a better-educated head should not increase poverty risk."""
    X = pd.DataFrame([sample])[art["features"]]
    rich = X.copy()
    for f in ["n_fridge", "n_cooker", "n_washing_machine", "n_motorcycle", "n_tv"]:
        rich[f] = 1
    rich["head_edu_yrs"] = 13
    rich = add_derived(rich)[art["features"]]
    m = art["model"]
    assert m.predict_proba(rich)[:, 1][0] <= m.predict_proba(X)[:, 1][0]


def test_unknown_category_is_handled(art, sample):
    X = pd.DataFrame([sample])[art["features"]]
    X["roof_material"] = "Something new"
    p = art["model"].predict_proba(X)[:, 1]
    assert np.isfinite(p).all()
