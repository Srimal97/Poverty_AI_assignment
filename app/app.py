"""Streamlit front end: Household Welfare Eligibility Screener (HIES 2019 model).

Run:  streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from poverty.features import DERIVED, add_derived

st.set_page_config(page_title="Welfare Eligibility Screener", page_icon="🏠", layout="wide")

# ------------------------------------------------------------------ resources
@st.cache_resource
def load_artefact():
    return joblib.load(ROOT / "models" / "poverty_model.pkl")


@st.cache_resource
def load_explainer(_art):
    from poverty.explain import StackExplainer
    typical = pd.DataFrame([{**_art["numeric_defaults"], **_art["categorical_defaults"]}])[_art["features"]]
    return StackExplainer(_art["model"], _art["features"], _art["categorical"], _art["category_levels"], typical)


art = load_artefact()
FEATS, NUMS, CATS = art["features"], art["numeric"], art["categorical"]
LEVELS, RANGES = art["category_levels"], art["numeric_ranges"]
SAMPLE = json.loads((ROOT / "app" / "sample_household.json").read_text())

LABELS = {
    "district_name": "District", "sector_name": "Sector", "n_members": "Household members",
    "n_children": "Children (under 15)", "n_elderly": "Elderly (65+)", "n_working_age": "Working age (15-64)",
    "n_female": "Female members", "n_employed": "Members employed", "n_unemployed": "Members seeking work",
    "n_disabled_or_old": "Members unable to work (disabled/too old)", "n_students": "Students",
    "n_pensioners": "Pensioners", "n_govt_employees": "Government/semi-govt employees",
    "n_own_account_family": "Own-account / family workers", "n_agri_workers": "Members working in agriculture",
    "n_elementary_occ": "Members in elementary occupations", "max_edu_yrs": "Highest education in household (years)",
    "mean_age": "Average age of members", "mean_adult_edu_yrs": "Average adult education (years)",
    "n_adults_al_plus": "Adults with A/L or higher", "head_female": "Head of household is female",
    "head_age": "Head's age", "head_edu_yrs": "Head's education (years)", "head_in_agriculture": "Head works in agriculture",
    "head_marital": "Head's marital status", "head_employment": "Head's employment status",
    "head_occupation": "Head's occupation group", "house_type": "Type of house", "bedrooms": "Bedrooms",
    "floor_area_band": "Floor area band (1:<100 ... 8:3000+ sq ft)", "wall_material": "Walls", "floor_material": "Floor",
    "roof_material": "Roof", "tenure": "Tenure", "drinking_water": "Drinking water source",
    "water_in_premises": "Water source within premises", "water_insufficient_last_month": "Water shortage last month",
    "toilet_access": "Toilet access", "toilet_type": "Toilet type", "handwash_with_soap": "Hand-washing place with soap",
    "garbage_disposal": "Garbage disposal", "lighting": "Lighting", "cooking_fuel": "Main cooking fuel",
    "collects_firewood": "Collects firewood", "disaster_last_12m": "Affected by disaster (12 months)",
    "power_line_nearby": "Electricity main line nearby", "pipe_water_line_nearby": "Pipe-borne water line nearby",
    "land_perches": "Land owned (perches)", "n_land_plots": "Number of land plots", "owns_land": "Owns agricultural land",
    "n_cattle": "Cattle/buffaloes", "n_goats": "Goats/sheep", "n_poultry": "Poultry",
}
GROUPS = {
    "📍 Location": ["district_name", "sector_name"],
    "👪 Household members": ["n_members", "n_children", "n_elderly", "n_working_age", "n_female", "mean_age",
                            "n_employed", "n_unemployed", "n_disabled_or_old", "n_students", "n_pensioners",
                            "n_govt_employees", "n_own_account_family", "n_agri_workers", "n_elementary_occ",
                            "max_edu_yrs", "mean_adult_edu_yrs", "n_adults_al_plus"],
    "🧑 Head of household": ["head_female", "head_age", "head_edu_yrs", "head_marital", "head_employment",
                            "head_occupation", "head_in_agriculture"],
    "🏠 Housing": ["house_type", "bedrooms", "floor_area_band", "wall_material", "floor_material", "roof_material",
                  "tenure", "drinking_water", "water_in_premises", "water_insufficient_last_month", "toilet_access",
                  "toilet_type", "handwash_with_soap", "garbage_disposal", "lighting", "cooking_fuel",
                  "collects_firewood", "disaster_last_12m"],
    "📺 Durable assets (number owned)": [f for f in NUMS if f.startswith("n_") and f not in
                                       ["n_members", "n_children", "n_elderly", "n_working_age", "n_female",
                                        "n_employed", "n_unemployed", "n_disabled_or_old", "n_students",
                                        "n_pensioners", "n_govt_employees", "n_own_account_family",
                                        "n_agri_workers", "n_elementary_occ", "n_adults_al_plus",
                                        "n_land_plots", "n_cattle", "n_goats", "n_poultry"]],
    "🌾 Land & livestock": ["owns_land", "land_perches", "n_land_plots", "n_cattle", "n_goats", "n_poultry"],
    "🚌 Distance to facilities (km)": [f for f in NUMS if f.startswith("km_to_")] + ["power_line_nearby",
                                                                                   "pipe_water_line_nearby"],
}
BINARY = {f for f in NUMS if set(np.unique([RANGES[f][0], RANGES[f][1]])) <= {0.0, 1.0}}
FLOATS = {"mean_age", "mean_adult_edu_yrs", "max_edu_yrs", "head_edu_yrs"} | {f for f in NUMS if f.startswith("km_to_")}


def label(f):
    return LABELS.get(f, f.replace("n_", "").replace("_", " ").capitalize())


def predict(X: pd.DataFrame) -> np.ndarray:
    X = add_derived(X)[FEATS]
    return art["model"].predict_proba(X)[:, 1]


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🏠 Welfare Eligibility Screener")
    st.caption("Proxy-means test model trained on the Department of Census & Statistics "
               "Household Income and Expenditure Survey 2019.")
    policy = st.radio("Decision policy", ["Minimise exclusion of the poor (recall-weighted, F2)",
                                          "Balanced precision/recall (F1)", "Custom threshold"])
    if policy.startswith("Minimise"):
        thr = art["threshold_f2"]
    elif policy.startswith("Balanced"):
        thr = art["threshold_f1"]
    else:
        thr = st.slider("Threshold on P(poor)", 0.05, 0.95, float(art["threshold_f2"]), 0.01)
    st.metric("Decision threshold", f"{thr:.2f}")
    st.divider()
    st.caption(f"Model v{art['version']} · stacked ensemble of {len(art['model'].base_specs)} learners · "
               f"poverty line Rs {art['poverty_line']:,}/person/month")
    st.caption("Decision-support only: a field officer verifies every case before any eligibility decision.")

tab1, tab2, tab3 = st.tabs(["Assess a household", "Batch scoring", "Model card"])

# ------------------------------------------------------------------ tab 1
with tab1:
    c1, c2 = st.columns([5, 4])
    with c1:
        st.subheader("Household details")
        if "src" not in st.session_state:
            st.session_state.src = "example"
        b1, b2 = st.columns(2)
        if b1.button("Load example (test household)"):
            st.session_state.src = "example"
            st.session_state.pop("vals", None)
        if b2.button("Reset to a typical household"):
            st.session_state.src = "typical"
            st.session_state.pop("vals", None)
        base = SAMPLE if st.session_state.src == "example" else {**art["numeric_defaults"], **art["categorical_defaults"]}

        vals = {}
        with st.form("hh"):
            for gname, fs in GROUPS.items():
                with st.expander(gname, expanded=gname.startswith("📍") or gname.startswith("🏠")):
                    cols = st.columns(3)
                    for i, f in enumerate(fs):
                        col = cols[i % 3]
                        key = f"{st.session_state.src}_{f}"
                        if f in CATS:
                            opts = LEVELS[f]
                            v = base.get(f, opts[0])
                            vals[f] = col.selectbox(label(f), opts, index=opts.index(v) if v in opts else 0, key=key)
                        elif f in BINARY:
                            vals[f] = int(col.checkbox(label(f), value=bool(base.get(f) or 0), key=key))
                        else:
                            lo, hi = RANGES[f]
                            v = base.get(f)
                            v = art["numeric_defaults"][f] if v is None or pd.isna(v) else v
                            if f in FLOATS:
                                vals[f] = col.number_input(label(f), float(lo), float(hi), float(v), 0.5, key=key)
                            else:
                                vals[f] = col.number_input(label(f), int(lo), int(hi), int(round(v)), 1, key=key)
            submitted = st.form_submit_button("Assess eligibility", type="primary", use_container_width=True)

    with c2:
        st.subheader("Assessment")
        X = pd.DataFrame([vals])
        for f in DERIVED:
            X[f] = np.nan
        X = add_derived(X)[FEATS]
        issues = []
        if X.n_children[0] + X.n_elderly[0] + X.n_working_age[0] != X.n_members[0]:
            issues.append("Children + elderly + working-age should equal household members.")
        if X.n_female[0] > X.n_members[0] or X.n_employed[0] > X.n_members[0]:
            issues.append("Female/employed members cannot exceed household size.")
        for m in issues:
            st.warning(m)
        p = float(art["model"].predict_proba(X)[:, 1][0])
        eligible = p >= thr
        band = "High" if p >= max(thr, 0.5) else ("Moderate" if p >= thr * 0.6 else "Low")
        m1, m2 = st.columns(2)
        m1.metric("Probability household is poor", f"{p:.0%}")
        m2.metric("Poverty-risk band", band)
        if eligible:
            st.error("**Flag for welfare support** - predicted below the poverty line. Verify in the field.", icon="🚩")
        else:
            st.success("**Not flagged** - predicted above the poverty line at this threshold.", icon="✅")
        st.progress(min(p, 1.0))

        with st.spinner("Explaining the prediction (SHAP)..."):
            reasons, base_val = load_explainer(art).top_reasons(X, k=10)
        st.markdown("**Why? Top drivers vs a typical household** (SHAP, log-odds)")
        r = reasons.copy()
        r["Feature"] = r.feature.map(label)
        r["Value"] = [f"{v:.2f}".rstrip("0").rstrip(".") if isinstance(v, (float, np.floating)) else str(v)
                      for v in r.value]
        r["Effect"] = np.where(r.contribution > 0, "▲ raises risk", "▼ lowers risk")
        st.dataframe(r[["Feature", "Value", "Effect", "contribution"]].rename(columns={"contribution": "SHAP"}),
                     hide_index=True, use_container_width=True,
                     column_config={"SHAP": st.column_config.NumberColumn(format="%.3f")})
        st.caption("Positive SHAP pushes the household towards 'poor'; values are relative to a typical "
                   "household (training medians/modes).")

        with st.expander("Base-learner probabilities"):
            bs = art["model"].base_scores(X)[0]
            st.dataframe(pd.DataFrame({"Learner": list(art["model"].base_specs), "P(poor)": bs,
                                       "Meta weight": art["model"].meta_weights.values}).round(3),
                         hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ tab 2
with tab2:
    st.subheader("Score a list of households")
    st.write("Upload a CSV with one row per household and the columns below (derived ratios are computed "
             "automatically). Download the template to start.")
    tmpl = pd.DataFrame([SAMPLE])[[f for f in FEATS if f not in DERIVED]]
    st.download_button("Download CSV template", tmpl.to_csv(index=False), "household_template.csv", "text/csv")
    up = st.file_uploader("Household CSV", type="csv")
    if up is not None:
        B = pd.read_csv(up)
        missing = [f for f in FEATS if f not in DERIVED and f not in B.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            B["P_poor"] = predict(B)
            B["flag"] = np.where(B.P_poor >= thr, "Flag for support", "Not flagged")
            B = B.sort_values("P_poor", ascending=False)
            st.dataframe(B[["P_poor", "flag"] + [c for c in B.columns if c not in ("P_poor", "flag")]],
                         use_container_width=True)
            st.download_button("Download scored file", B.to_csv(index=False), "scored_households.csv", "text/csv")

# ------------------------------------------------------------------ tab 3
with tab3:
    st.subheader("Model card")
    tm = art["test_metrics"]
    k = st.columns(5)
    for col, (name, key) in zip(k, [("ROC-AUC", "ROC_AUC"), ("PR-AUC", "PR_AUC"), ("Recall", "Recall"),
                                    ("Precision", "Precision"), ("Brier", "Brier")]):
        col.metric(name, f"{tm[key]:.3f}")
    st.caption("Held-out test set (20% of PSUs, never seen in training or tuning); survey-weighted; recall/precision at the F1 threshold.")
    st.markdown("""
**Intended use.** Ranking and pre-screening households for poverty-targeted welfare (e.g. Samurdhi / Aswesuma)
using characteristics a field officer can observe and verify. Not for automated final decisions.

**Training data.** DCS Household Income and Expenditure Survey 2019 (19,911 households, all 25 districts),
target = per-capita monthly expenditure below the official poverty line (Rs 6,966).

**Excluded on purpose.** Income, expenditure, debt (unverifiable / leakage), Samurdhi receipt (circular),
ethnicity and religion (non-discrimination; used only to audit fairness).

**Known limitations.** Pre-2022-crisis prices and living conditions; nominal (not spatially deflated)
expenditure; model scores must be recalibrated on fresh survey data before operational use.
""")
    for name in ["fairness_audit.csv", "samurdhi_comparison.csv"]:
        p_ = ROOT / "reports" / "tables" / name
        if p_.exists():
            st.markdown(f"**{name.replace('_', ' ').replace('.csv', '').title()}**")
            st.dataframe(pd.read_csv(p_).round(3), hide_index=True, use_container_width=True)
