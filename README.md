# Spotting Poor Households for Welfare Targeting in Sri Lanka

**Applied Machine Learning assignment — MSc in AI, University of Moratuwa**

A proxy-means-test (PMT) model that predicts whether a Sri Lankan household lives below the official
poverty line using only characteristics a field officer can observe and verify: location, demographics,
education, housing, durable assets, land and livestock, and access to services. It is trained on the
Department of Census and Statistics (DCS) **Household Income and Expenditure Survey (HIES) 2019**
(19,911 households across all 25 districts).

Welfare programmes such as Samurdhi and Aswesuma cannot verify household income. In 2019, the actual
Samurdhi programme **missed 59% of poor households**. Selecting the same number of households, this model
misses **23%**.

## Results (held-out test set: 20% of PSUs, never used in training or tuning; survey-weighted)

| Model | Group | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|
| **Survey-weighted hybrid stacked ensemble** | **Proposed** | **0.908** | **0.588** | **0.068** |
| LightGBM PMT regressor (probabilistic) | New | 0.905 | 0.580 | 0.070 |
| LightGBM | Covered | 0.901 | 0.569 | 0.071 |
| CatBoost | Covered | 0.899 | 0.569 | 0.070 |
| OLS PMT (log-consumption regression) | Existing practice | 0.900 | 0.562 | 0.072 |
| EBM (Explainable Boosting Machine) | New | 0.895 | 0.552 | 0.072 |
| Random Forest | Covered | 0.889 | 0.526 | 0.085 |
| Decision Tree | Covered | 0.843 | 0.439 | 0.156 |

The full comparison of 17 models is in `reports/tables/test_results.csv`. PSU-cluster bootstrap 95% CIs are in
`reports/tables/bootstrap_ci.csv`. The stack beats the OLS PMT baseline by +0.027 PR-AUC (CI 0.005 to 0.055).

## What is in the repository

```
src/poverty/        reusable package
  features.py       HIES tables -> one row per household (leakage-safe feature set)
  data.py           PSU-grouped, stratified split and survey weights
  models.py         model zoo (10 covered, 5 new), OLS/GBM PMT, SurveyWeightedStack
  evaluate.py       weighted metrics, thresholds, targeting errors, cluster bootstrap
  explain.py        permutation SHAP for the whole stack
scripts/            eda.py -> tune.py -> train.py -> explain.py -> make_notebook.py
notebooks/          AML_Poverty_Targeting.ipynb (executed walk-through)
app/app.py          Streamlit screener (single household, batch CSV, model card)
models/             poverty_model.pkl (deployable), best_params.json, thresholds.json
reports/            figures/ and tables/ produced by the scripts
tests/              model contract and smoke tests (run in CI)
Dockerfile, .github/workflows/ci.yml, requirements*.txt
```

## Reproduce

```bash
pip install -r requirements-dev.txt
# put the DCS HIES 2019 CSVs in data/raw/   (see data/README.md - raw data is not redistributed)
python scripts/eda.py
python scripts/tune.py          # Optuna, ~30 min on 2 CPUs, logged to MLflow
python scripts/train.py         # 5-fold PSU-grouped CV + test + stack + export (~25 min)
python scripts/explain.py       # SHAP, EBM, PDP/ICE, fairness, Samurdhi comparison
python scripts/make_notebook.py && jupyter nbconvert --execute --inplace notebooks/AML_Poverty_Targeting.ipynb
mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse experiments
```

## Run the app

```bash
pip install -r requirements.txt
streamlit run app/app.py
# or
docker build -t hies-poverty-screener . && docker run -p 8501:8501 hies-poverty-screener
```

To deploy on **Streamlit Community Cloud**, connect this GitHub repo and set the main file to `app/app.py`.
The app needs only `models/poverty_model.pkl`, not the raw data.

## Method in one paragraph

The target is `poor = 1` if household expenditure per person is below Rs 6,966 per month (the DCS official
poverty line, 2019 base). Applying this line reproduces a 13.9% headcount, against 14.3% published by DCS.

Features are restricted to verifiable characteristics. Income, expenditure, debt, Samurdhi receipt,
ethnicity and religion are excluded; the last three are kept for auditing only.

All splits are grouped by PSU (the survey's sampling cluster), and DCS design weights are used in both
training and evaluation. Hyperparameters were designed with Optuna TPE using PSU-grouped CV, with
weighted PR-AUC as the objective.

The proposed model stacks seven diverse learners: CatBoost, LightGBM, XGBoost, EBM, Extra Trees, a
probabilistic LightGBM log-consumption regressor, and the OLS PMT. A survey-weighted logistic meta-learner
blends their out-of-fold logits. This combines classification learners with *regression* learners that see
the whole welfare distribution.

## Data and ethics

Raw microdata are not included; see `data/README.md`. The model is decision support: it ranks households
for field verification and must not be used for automated eligibility decisions. Before any operational
use it must be recalibrated on post-2022 data. See the report's critical discussion for the fairness audit
and its limitations.
