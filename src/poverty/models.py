"""Model zoo: lecture-covered benchmarks, new (not-covered) algorithms and the
proposed survey-weighted stacked ensemble.

Every model is a scikit-learn Pipeline whose final step is named ``clf`` so
that survey weights can be routed as ``clf__sample_weight``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder, OrdinalEncoder,
                                   PowerTransformer, StandardScaler)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE

RS = RANDOM_STATE


# ------------------------------------------------------------- preprocessing
def tree_preprocessor(nums, cats):
    """Median imputation + ordinal codes (tree models are invariant to scaling)."""
    return ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), nums),
         ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                encoded_missing_value=-1, dtype=np.int64), cats)],
        verbose_feature_names_out=False).set_output(transform="pandas")


def linear_preprocessor(nums, cats):
    """Median imputation + Yeo-Johnson (skewed counts/distances) + scaling;
    one-hot for nominal variables (rare levels < 20 households pooled)."""
    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("yeojohnson", PowerTransformer(method="yeo-johnson", standardize=True))])
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                        sparse_output=False)
    return ColumnTransformer([("num", num, nums), ("cat", cat, cats)],
                             verbose_feature_names_out=False)


class CastCats(BaseEstimator):
    """Cast ordinal-encoded categorical columns to int (CatBoost requires it).
    A module-level class (not a closure) so the pipeline is picklable."""

    def __init__(self, cats=None):
        self.cats = cats

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def __sklearn_is_fitted__(self):
        return True

    def transform(self, X):
        X = X.copy()
        X[list(self.cats)] = X[list(self.cats)].astype(int)
        return X


def _cast_cats(cats):
    return CastCats(list(cats))


# ------------------------------------------------------------- model factories
def make_model(name: str, nums, cats, params: dict | None = None):
    """Return an unfitted pipeline for ``name`` with the (tuned) ``params``."""
    p = dict(params or {})
    tp = lambda: tree_preprocessor(nums, cats)
    lp = lambda: linear_preprocessor(nums, cats)

    if name == "Logistic Regression":
        clf = LogisticRegression(C=p.get("C", 0.3), max_iter=3000, class_weight="balanced")
        return Pipeline([("prep", lp()), ("clf", clf)])
    if name == "Decision Tree":
        clf = DecisionTreeClassifier(max_depth=8, min_samples_leaf=40, class_weight="balanced",
                                     ccp_alpha=1e-4, random_state=RS)
        return Pipeline([("prep", tp()), ("clf", clf)])
    if name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=51, weights="distance", n_jobs=-1)
        return Pipeline([("prep", lp()), ("clf", clf)])
    if name == "SVM (RBF)":
        clf = SVC(C=1.0, gamma="scale", kernel="rbf", class_weight="balanced", cache_size=1000)
        return Pipeline([("prep", lp()), ("clf", clf)])
    if name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=5,
                                     class_weight="balanced_subsample", n_jobs=-1, random_state=RS)
        return Pipeline([("prep", tp()), ("clf", clf)])
    if name == "AdaBoost":
        clf = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), n_estimators=400,
                                 learning_rate=0.3, random_state=RS)
        return Pipeline([("prep", tp()), ("clf", clf)])
    if name == "Gradient Boosting":
        clf = GradientBoostingClassifier(n_estimators=400, learning_rate=0.05, max_depth=3,
                                         subsample=0.8, min_samples_leaf=30, random_state=RS)
        return Pipeline([("prep", tp()), ("clf", clf)])
    if name == "XGBoost":
        from xgboost import XGBClassifier
        base = dict(n_estimators=600, learning_rate=0.03, max_depth=5, min_child_weight=5,
                    subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0, gamma=0.0)
        base.update(p)
        clf = XGBClassifier(**base, tree_method="hist", eval_metric="aucpr", n_jobs=2,
                            random_state=RS)
        return Pipeline([("prep", tp()), ("clf", clf)])
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        base = dict(n_estimators=700, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0)
        base.update(p)
        clf = LGBMClassifier(**base, n_jobs=2, random_state=RS, verbose=-1)
        return Pipeline([("prep", tp()), ("clf", clf)])
    if name == "CatBoost":
        from catboost import CatBoostClassifier
        base = dict(iterations=800, learning_rate=0.05, depth=6, l2_leaf_reg=5.0)
        base.update(p)
        clf = CatBoostClassifier(**base, cat_features=list(cats), thread_count=2,
                                 random_seed=RS, verbose=0, allow_writing_files=False)
        return Pipeline([("prep", tp()), ("cast", _cast_cats(cats)), ("clf", clf)])

    # ---------------- NEW: not covered in lectures
    if name == "Extra Trees":
        base = dict(n_estimators=600, max_features=0.4, min_samples_leaf=8)
        base.update(p)
        clf = ExtraTreesClassifier(**base, class_weight="balanced_subsample", n_jobs=-1,
                                   random_state=RS)
        return Pipeline([("prep", tp()), ("clf", clf)])
    if name == "Balanced Random Forest":
        from imblearn.ensemble import BalancedRandomForestClassifier
        clf = BalancedRandomForestClassifier(n_estimators=600, max_features="sqrt",
                                             min_samples_leaf=3, sampling_strategy="all",
                                             replacement=True, bootstrap=False, n_jobs=-1,
                                             random_state=RS)
        return Pipeline([("prep", tp()), ("clf", clf)])
    if name == "NGBoost":
        from ngboost import NGBClassifier
        from ngboost.distns import Bernoulli
        clf = NGBClassifier(Dist=Bernoulli, n_estimators=500, learning_rate=0.03,
                            minibatch_frac=0.7, col_sample=0.6,
                            Base=DecisionTreeRegressor_(max_depth=4, min_samples_leaf=30),
                            verbose=False, random_state=RS)
        return Pipeline([("prep", tp()), ("clf", clf)])
    if name == "EBM":
        from interpret.glassbox import ExplainableBoostingClassifier
        base = dict(max_bins=256, interactions=10, learning_rate=0.02, max_leaves=3,
                    min_samples_leaf=20, outer_bags=8)
        base.update(p)
        clf = ExplainableBoostingClassifier(**base, n_jobs=2, random_state=RS)
        return Pipeline([("clf", clf)])          # EBM handles raw categoricals & NaNs itself
    raise ValueError(name)


def DecisionTreeRegressor_(**kw):
    from sklearn.tree import DecisionTreeRegressor
    return DecisionTreeRegressor(**kw)


COVERED = ["Logistic Regression", "Decision Tree", "KNN", "SVM (RBF)", "Random Forest",
           "AdaBoost", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost"]
NEW = ["Extra Trees", "Balanced Random Forest", "NGBoost", "EBM"]
SUPPORTS_WEIGHT = set(COVERED + NEW) - {"KNN"}


def fit_model(model, X, y, w=None, name=""):
    if w is not None and name in SUPPORTS_WEIGHT:
        return model.fit(X, y, clf__sample_weight=np.asarray(w))
    return model.fit(X, y)


def name_has(obj, attr):
    try:
        return callable(getattr(obj, attr))
    except AttributeError:
        return False


def score(model, X) -> np.ndarray:
    """Positive-class score in [0,1] (SVM decision values squashed with a logistic)."""
    if isinstance(model, Pipeline):
        Xt = model[:-1].transform(X) if len(model) > 1 else X
        est = model[-1]
    else:
        Xt, est = X, model
    if name_has(est, "predict_proba") and not isinstance(est, SVC):
        return np.asarray(est.predict_proba(Xt))[:, 1]
    d = est.decision_function(Xt)
    return 1 / (1 + np.exp(-d))


# ------------------------------------------------------- existing practice: OLS PMT
class OLSPMT(BaseEstimator, ClassifierMixin):
    """Traditional proxy-means test: OLS of log per-capita consumption on observable
    characteristics; households are ranked by predicted consumption (lower = poorer).
    ``predict_proba`` returns a monotone transform so it can be ranked like the others."""

    def __init__(self, nums=None, cats=None, poverty_line=6966):
        self.nums, self.cats, self.poverty_line = nums, cats, poverty_line

    def fit(self, X, y_logpce, sample_weight=None):
        self.pipe_ = Pipeline([("prep", linear_preprocessor(self.nums, self.cats)),
                               ("ols", LinearRegression())])
        self.pipe_.fit(X, y_logpce, ols__sample_weight=sample_weight)
        resid = y_logpce - self.pipe_.predict(X)
        self.sigma_ = float(np.sqrt(np.average(resid ** 2, weights=sample_weight)))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        from scipy.stats import norm
        mu = self.pipe_.predict(X)
        p = norm.cdf((np.log(self.poverty_line) - mu) / self.sigma_)  # P(log pce < log z)
        return np.c_[1 - p, p]


class GBMPMT(BaseEstimator, ClassifierMixin):
    """ML proxy-means test (hybrid regression->classification learner).

    A LightGBM *regressor* learns log per-capita consumption (using the whole
    welfare distribution, not just a 0/1 label). The predictive distribution is
    taken as Normal(mu(x), sigma^2), with sigma estimated on an inner 15 %
    hold-out (in-sample residuals of boosted trees are optimistically small).
    P(poor | x) = Phi((log z - mu(x)) / sigma)."""

    def __init__(self, nums=None, cats=None, poverty_line=6966, params=None):
        self.nums, self.cats, self.poverty_line, self.params = nums, cats, poverty_line, params

    def _pipe(self):
        from lightgbm import LGBMRegressor
        base = dict(n_estimators=1200, learning_rate=0.02, num_leaves=31, min_child_samples=40,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0)
        base.update(self.params or {})
        return Pipeline([("prep", tree_preprocessor(self.nums, self.cats)),
                         ("reg", LGBMRegressor(**base, n_jobs=2, random_state=RS, verbose=-1))])

    def fit(self, X, y_logpce, sample_weight=None):
        y_logpce = np.asarray(y_logpce)
        w = np.ones(len(y_logpce)) if sample_weight is None else np.asarray(sample_weight)
        rng = np.random.default_rng(RS)
        hold = rng.random(len(y_logpce)) < 0.15
        p = self._pipe().fit(X[~hold], y_logpce[~hold], reg__sample_weight=w[~hold])
        r = y_logpce[hold] - p.predict(X[hold])
        self.sigma_ = float(np.sqrt(np.average(r ** 2, weights=w[hold])))
        self.pipe_ = self._pipe().fit(X, y_logpce, reg__sample_weight=w)
        self.classes_ = np.array([0, 1])
        return self

    def predict_logpce(self, X):
        return self.pipe_.predict(X)

    def predict_proba(self, X):
        from scipy.stats import norm
        p = norm.cdf((np.log(self.poverty_line) - self.pipe_.predict(X)) / self.sigma_)
        return np.c_[1 - p, p]


REGRESSION_MODELS = {"OLS PMT (existing practice)", "LightGBM PMT regressor"}


def make_any(name, nums, cats, params=None, poverty_line=6966):
    if name == "OLS PMT (existing practice)":
        return OLSPMT(nums, cats, poverty_line)
    if name == "LightGBM PMT regressor":
        return GBMPMT(nums, cats, poverty_line, params)
    return make_model(name, nums, cats, params)


def fit_any(model, X, y, w, name, y_reg=None):
    if name in REGRESSION_MODELS:
        return model.fit(X, y_reg, sample_weight=w)
    return fit_model(model, X, y, w, name)


# ------------------------------------------------------------- proposed model
class SurveyWeightedStack(BaseEstimator, ClassifierMixin):
    """Heterogeneous stacked ensemble.

    Level 0: diverse learners (ordered boosting, leaf-wise boosting, depth-wise
    boosting, a glass-box GAM and randomised trees). Their out-of-fold
    probabilities are produced with PSU-grouped K-fold CV so the meta learner
    never sees predictions made on households from the same sampling cluster.
    Level 1: survey-weighted logistic regression on the logits of the base
    probabilities (a calibrated, constrained, interpretable blender).
    """

    def __init__(self, base_specs: dict, nums, cats, n_folds=5, meta_C=1.0):
        self.base_specs, self.nums, self.cats = base_specs, nums, cats
        self.n_folds, self.meta_C = n_folds, meta_C

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-5, 1 - 1e-5)
        return np.log(p / (1 - p))

    def fit(self, X, y, sample_weight=None, groups=None, y_reg=None, oof=None, log=print):
        """``y_reg`` (log per-capita consumption) is required when a regression
        learner (PMT) is among the base specs."""
        y = np.asarray(y)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight)
        yr = None if y_reg is None else np.asarray(y_reg)
        names = list(self.base_specs)
        if oof is None:
            oof = np.zeros((len(y), len(names)))
            for k, (tr, va) in enumerate(GroupKFold(self.n_folds).split(X, y, groups)):
                for j, n in enumerate(names):
                    m = make_any(n, self.nums, self.cats, self.base_specs[n])
                    fit_any(m, X.iloc[tr], y[tr], w[tr], n, None if yr is None else yr[tr])
                    oof[va, j] = score(m, X.iloc[va])
                log(f"  stack fold {k + 1}/{self.n_folds} done")
        self.oof_ = np.asarray(oof)
        self.meta_ = LogisticRegression(C=self.meta_C, max_iter=2000)
        self.meta_.fit(self._logit(self.oof_), y, sample_weight=w)
        self.base_models_ = {}
        for n in names:
            m = make_any(n, self.nums, self.cats, self.base_specs[n])
            self.base_models_[n] = fit_any(m, X, y, w, n, yr)
        self.classes_ = np.array([0, 1])
        return self

    def base_scores(self, X):
        return np.column_stack([score(self.base_models_[n], X) for n in self.base_specs])

    def predict_proba(self, X):
        p = self.meta_.predict_proba(self._logit(self.base_scores(X)))[:, 1]
        return np.c_[1 - p, p]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    @property
    def meta_weights(self):
        return pd.Series(self.meta_.coef_[0], index=list(self.base_specs))
