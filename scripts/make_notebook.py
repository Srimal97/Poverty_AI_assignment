"""Generate notebooks/AML_Poverty_Targeting.ipynb (then execute it with nbconvert)."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# Spotting Poor Households for Welfare Targeting in Sri Lanka
### Applied Machine Learning assignment - HIES 2019 proxy-means-test model

**Problem.** Sri Lanka's poverty-targeted cash transfers (Samurdhi, now Aswesuma) must decide which households
are poor without being able to verify income. This notebook builds a model that predicts whether a household's
per-capita expenditure falls below the official poverty line using only characteristics a field officer can
observe - location, demographics, education, housing, assets, land and access to services.

**Data.** Department of Census and Statistics (DCS), Household Income and Expenditure Survey 2019,
19,911 households, all 25 districts.

**Pipeline.** `src/poverty/` holds the reusable code; the heavy steps run as scripts
(`scripts/eda.py -> tune.py -> train.py -> explain.py`) and this notebook walks through the results.""")
code("""import sys, json, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT / 'src'))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, joblib
from IPython.display import Image, display
pd.set_option('display.max_columns', 30); pd.set_option('display.width', 160)
from poverty.data import load, split, xyw
from poverty.features import feature_lists
FIG, TAB = ROOT/'reports'/'figures', ROOT/'reports'/'tables'
show = lambda f, w=820: display(Image(filename=str(FIG/f), width=w))""")

md("## 1. Know the dataset")
code("""df = load()
feats, nums, cats = feature_lists(df)
print(df.shape, '| features:', len(feats), f'({len(nums)} numeric, {len(cats)} categorical)')
pd.read_csv(TAB/'dataset_summary.csv', index_col=0).round(2)""")
code("""df[['district_name','sector_name','n_members','head_edu_yrs','house_type','cooking_fuel',
    'asset_types_owned','pce','poor','finalweight']].head()""")
md("""**Data-quality checks performed in `features.py`**
* All household tables join 1:1 on `psu + snumber + hhno` (no duplicates, no unmatched rows).
* The person file lists 5,311 *non-usual* residents; keeping usual residents reproduces DCS `hhsize` for 100% of households.
* Unknown/refused answers use codes 8/9/99 - mapped to labelled levels, never treated as numbers.
* Heavy skew (PCE skewness 12, income 15) -> log scale for analysis; Yeo-Johnson for linear/KNN/SVM models.
* Missingness is tiny (max 1.5%, distance to government dispensary) -> median imputation inside the pipeline.""")
code("pd.read_csv(TAB/'missing_values.csv', index_col=0).T")
code("show('fig01_pce_distribution.png')")
code("show('fig02_poverty_by_district.png', 700)")
code("pd.read_csv(TAB/'poverty_by_sector.csv')")
code("show('fig03_poverty_correlates.png')")
code("show('fig04_numeric_correlations.png', 700)")

md("""## 2. Problem framing and leakage control
* **Target:** `poor = 1` if monthly expenditure per person < Rs 6,966 (DCS official poverty line, 2019 base).
  Survey-weighted headcount reproduced: **13.9%** of persons (DCS published 14.3% with spatial price deflators).
* **Excluded features:** income, expenditure (they define the target), debt (self-reported), Samurdhi receipt
  (the programme being evaluated), ethnicity and religion (non-discrimination - audit only).
* **Split:** 80/20, *stratified and grouped by PSU* so neighbours from the same sampling cluster never appear in
  both train and test. All CV inside the training set is also PSU-grouped.
* **Survey weights** (`finalweight`) are used as `sample_weight` in training and in every metric.""")
code("""tr, te = split(df)
print('train', len(tr), 'test', len(te), '| shared PSUs:', len(set(df.psu.iloc[tr]) & set(df.psu.iloc[te])))
print('poverty rate train/test:', df.poor.iloc[tr].mean().round(4), df.poor.iloc[te].mean().round(4))""")

md("""## 3. Models
| Group | Models |
|---|---|
| Existing practice | **OLS proxy-means test** - linear regression of log consumption (World Bank-style PMT used for welfare targeting) |
| Covered in lectures | Logistic Regression, Decision Tree, KNN, SVM (RBF), Random Forest, AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost |
| **New (not covered)** | **Explainable Boosting Machine (EBM)**, **Extra Trees**, **Balanced Random Forest**, **NGBoost**, **LightGBM PMT regressor** (probabilistic regression->classification hybrid) |
| **Proposed** | **Survey-weighted hybrid stacked ensemble**: CatBoost + LightGBM + XGBoost + EBM + Extra Trees + LightGBM-PMT + OLS-PMT, blended by a weighted logistic meta-learner on out-of-fold logits |""")
md("### 3.1 Hyperparameter design (Optuna TPE, PSU-grouped CV, objective = weighted PR-AUC)")
code("""bp = json.load(open(ROOT/'models'/'best_params.json'))
pd.DataFrame({k: {**v['params'], 'cv_wAP': round(v['cv_wAP'],4), 'trials': v['n_trials']} for k, v in bp.items()}).T""")

md("## 4. Training and evaluation")
code("""cv = pd.read_csv(TAB/'cv_results.csv')
cv[['model','group','ROC_AUC_mean','ROC_AUC_sd','PR_AUC_mean','PR_AUC_sd','Brier_mean']].round(4)""")
code("show('fig05_cv_pr_auc.png', 760)")
code("""res = pd.read_csv(TAB/'test_results.csv')
res[['model','group','ROC_AUC','PR_AUC','Recall','Precision','F1','Brier','Exclusion_err@PR','Inclusion_err@PR','train_seconds']].round(4)""")
code("show('fig06_roc_pr_test.png')")
code("show('fig07_calibration.png', 520)")
md("### 4.1 Is the improvement real? PSU-cluster bootstrap (1,000 resamples of the test set)")
code("pd.read_csv(TAB/'bootstrap_ci.csv')")
code("pd.read_csv(TAB/'stack_meta_weights.csv')")

md("## 5. Single-sample inference with the exported pickle")
code("""art = joblib.load(ROOT/'models'/'poverty_model.pkl')     # the deployable artefact
sample = json.load(open(ROOT/'app'/'sample_household.json'))  # a held-out test household
X1 = pd.DataFrame([sample])[art['features']]
p = art['model'].predict_proba(X1)[:, 1][0]
print(f"P(poor) = {p:.3f}  | policy threshold (F2) = {art['threshold_f2']:.3f}  ->",
      'FLAG for support' if p >= art['threshold_f2'] else 'not flagged')
json.load(open(TAB/'single_sample_inference.json'))""")
code("show('fig12_single_household_waterfall.png', 760)")

md("## 6. Explainability")
code("show('fig08_shap_beeswarm.png', 760)")
code("show('fig09_shap_bar.png', 700)")
code("""print(json.load(open(TAB/'importance_agreement.json')))
pd.read_csv(TAB/'importance_comparison.csv', index_col=0).head(15)""")
code("show('fig10_ebm_shapes.png')")
code("show('fig11_pdp_ice.png')")

md("## 7. Critical discussion: fairness and real-world targeting")
code("pd.read_csv(TAB/'fairness_audit.csv').round(3)")
code("show('fig13_fairness.png')")
code("pd.read_csv(TAB/'samurdhi_comparison.csv').round(3)")
code("show('fig14_samurdhi_exclusion_curve.png', 720)")
md("### 7.1 How short can the field form be?")
code("pd.read_csv(TAB/'compact_form_ablation.csv')")
md("""## 8. Deployment
* `app/app.py` - Streamlit screener (form -> probability, decision, SHAP reasons; batch CSV scoring; model card).
* `Dockerfile`, pinned `requirements.txt`, GitHub Actions CI (`tests/` + Docker build), MLflow tracking (`mlflow ui --backend-store-uri sqlite:///mlflow.db`).

See the report for the full discussion.""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "notebooks" / "AML_Poverty_Targeting.ipynb"
nbf.write(nb, out)
print("wrote", out)
