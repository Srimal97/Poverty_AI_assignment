"""Evaluation graphics, explainability (SHAP / EBM / PDP-ICE), single-sample
inference, fairness audit, Samurdhi targeting comparison and a compact-form
ablation. Run after scripts/train.py."""
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, precision_recall_curve, roc_auc_score,
                             roc_curve)

from poverty.config import FIG_DIR, MODELS_DIR, ROOT, TAB_DIR
from poverty.data import load, split, xyw
from poverty.evaluate import ci, cluster_bootstrap, selection_errors, targeting_errors
from poverty.explain import StackExplainer
from poverty.plotting import INK2, MUTED, NEG, POS, SERIES, save, setup

SN = "Stacked Ensemble (proposed)"
log = lambda *a: print(*a, flush=True)


def main():
    setup()
    art = joblib.load(MODELS_DIR / "poverty_model.pkl")
    stack, feats, nums, cats = art["model"], art["features"], art["numeric"], art["categorical"]
    thr = art["threshold_f2"]
    df = load()
    tr, te = split(df)
    dtr, dte = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
    Xtr, ytr, wtr, *_ = xyw(dtr)
    Xte, yte, wte, *_ = xyw(dte)
    T = pd.read_parquet(MODELS_DIR / "test_scores.parquet")
    res = pd.read_csv(TAB_DIR / "test_results.csv")
    cvres = pd.read_csv(TAB_DIR / "cv_results.csv")
    prate = art["train_poverty_rate"]

    # ================================================= 1. comparison graphics
    order = cvres.sort_values("PR_AUC_mean")
    colors = {"Covered in lectures": SERIES[0], "New (not covered)": SERIES[1],
              "Existing practice": MUTED, "Proposed": SERIES[2]}
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for y, (m, s, g) in enumerate(zip(order.PR_AUC_mean, order.PR_AUC_sd, order.group)):
        ax.plot([m - s, m + s], [y, y], color=colors[g], lw=2, alpha=0.45, solid_capstyle="round")
        ax.plot(m, y, "o", ms=8, color=colors[g], markeredgecolor="white", markeredgewidth=1.5)
        ax.text(m + s + 0.004, y, f"{m:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(order)), order.model)
    ax.set_xlim(min(order.PR_AUC_mean - order.PR_AUC_sd) - 0.01, max(order.PR_AUC_mean + order.PR_AUC_sd) + 0.03)
    ax.set_xlabel("PR-AUC (survey-weighted), 5-fold PSU-grouped CV: mean (dot) ± 1 sd (line)")
    ax.set_title("Cross-validated PR-AUC by model")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=8, color=c, label=g) for g, c in colors.items()],
              loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=4, fontsize=8)
    ax.grid(axis="y", visible=False)
    save(fig, FIG_DIR / "fig05_cv_pr_auc.png")

    show = [SN, "CatBoost", "LightGBM", "EBM", "OLS PMT (existing practice)", "Random Forest", "Logistic Regression"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for i, m in enumerate(show):
        s = T[m].values
        fpr, tpr, _ = roc_curve(yte, s, sample_weight=wte)
        p, r, _ = precision_recall_curve(yte, s, sample_weight=wte)
        lw = 2.4 if m == SN else 1.4
        axs[0].plot(fpr, tpr, lw=lw, color=SERIES[i], label=f"{m} ({roc_auc_score(yte, s, sample_weight=wte):.3f})")
        axs[1].plot(r, p, lw=lw, color=SERIES[i], label=f"{m} ({average_precision_score(yte, s, sample_weight=wte):.3f})")
    axs[0].plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":")
    axs[1].axhline(np.average(yte, weights=wte), color=MUTED, lw=1, ls=":")
    axs[0].set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC curve (test)")
    axs[1].set(xlabel="Recall (share of poor households found)", ylabel="Precision", title="Precision-recall curve (test)")
    axs[0].legend(fontsize=7, loc="lower right"); axs[1].legend(fontsize=7, loc="upper right")
    save(fig, FIG_DIR / "fig06_roc_pr_test.png")

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for i, m in enumerate([SN, "CatBoost", "OLS PMT (existing practice)", "Random Forest"]):
        pt, pp = calibration_curve(yte, np.clip(T[m].values, 0, 1), n_bins=10, strategy="quantile")
        ax.plot(pp, pt, marker="o", ms=4, lw=1.6, color=SERIES[i], label=m)
    ax.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1)
    ax.set(xlabel="Predicted probability of being poor", ylabel="Observed share poor",
           title="Calibration (test, deciles)")
    ax.legend(fontsize=7)
    save(fig, FIG_DIR / "fig07_calibration.png")

    # ---------------- cluster bootstrap CIs and paired differences
    best_single = res[~res.model.isin([SN, "Stack of all models (ablation)"])].iloc[0].model
    cand = {SN: T[SN].values, best_single: T[best_single].values,
            "OLS PMT (existing practice)": T["OLS PMT (existing practice)"].values,
            "Logistic Regression": T["Logistic Regression"].values}
    boot_rows = []
    for metric in ["PR_AUC", "ROC_AUC"]:
        B = cluster_bootstrap(yte, cand, wte, dte.psu.values, n_boot=1000, metric=metric)
        for m in cand:
            lo, hi = ci(B[m])
            boot_rows.append({"metric": metric, "model": m, "estimate": B[m].mean(), "ci_low": lo, "ci_high": hi})
        for m in cand:
            if m == SN:
                continue
            d = B[SN] - B[m]
            lo, hi = ci(d)
            boot_rows.append({"metric": metric, "model": f"Δ Stack − {m}", "estimate": d.mean(),
                              "ci_low": lo, "ci_high": hi, "p_le_0": float((d <= 0).mean())})
    boot = pd.DataFrame(boot_rows)
    boot.round(4).to_csv(TAB_DIR / "bootstrap_ci.csv", index=False)
    log(boot.round(4).to_string(index=False))

    # ================================================= 2. SHAP for the final stack
    ex = StackExplainer(stack, feats, cats, art["category_levels"], Xtr.sample(30, random_state=0))
    rng = np.random.default_rng(1)
    idx = rng.choice(len(Xte), 300, replace=False)
    # chunked + cached: permutation SHAP of a 7-model stack is ~1.5 s per household
    cache = MODELS_DIR / "shap_cache"
    cache.mkdir(exist_ok=True)
    parts = []
    for k in range(0, len(idx), 50):
        f_ = cache / f"chunk_{k:04d}.joblib"
        if not f_.exists():
            t0 = time.time()
            joblib.dump(ex.shap_values(Xte.iloc[idx[k:k + 50]]), f_)
            log(f"  SHAP chunk {k}: {time.time() - t0:.0f}s")
        parts.append(joblib.load(f_))
    import shap
    sv = shap.Explanation(values=np.vstack([p.values for p in parts]),
                          base_values=np.concatenate([p.base_values for p in parts]),
                          data=np.vstack([p.data for p in parts]),
                          display_data=np.vstack([p.display_data for p in parts]),
                          feature_names=feats)
    np.save(MODELS_DIR / "shap_test_values.npy", sv.values)
    imp = pd.Series(np.abs(sv.values).mean(0), index=feats).sort_values(ascending=False)
    imp.rename("mean_abs_shap_logodds").to_csv(TAB_DIR / "shap_importance_stack.csv")

    import shap
    plt.figure()
    shap.plots.beeswarm(sv, max_display=20, show=False, plot_size=(8, 7))
    plt.title("SHAP summary - proposed stacked ensemble (log-odds of being poor)", loc="left", fontweight="bold", fontsize=11)
    plt.gcf().savefig(FIG_DIR / "fig08_shap_beeswarm.png", bbox_inches="tight", dpi=160); plt.close()

    top = imp.head(20)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.barh(top.index, top.values, color=SERIES[0], height=0.7)
    ax.set_xlabel("Mean |SHAP| (log-odds)"); ax.set_title("Global feature importance - stacked ensemble")
    ax.grid(axis="y", visible=False)
    save(fig, FIG_DIR / "fig09_shap_bar.png")

    # TreeSHAP of the CatBoost base learner on the full test set + rank agreement
    from catboost import Pool
    cb = stack.base_models_["CatBoost"]
    Xcb = cb[:-1].transform(Xte)
    cb_sv = cb[-1].get_feature_importance(Pool(Xcb, cat_features=cats), type="ShapValues")[:, :-1]
    cb_imp = pd.Series(np.abs(cb_sv).mean(0), index=Xcb.columns).reindex(feats)
    from scipy.stats import spearmanr
    rho = spearmanr(imp.reindex(feats), cb_imp).correlation
    native = pd.Series(cb[-1].get_feature_importance(), index=Xcb.columns).reindex(feats)
    cmpimp = pd.DataFrame({"stack_perm_SHAP": imp, "catboost_treeSHAP": cb_imp, "catboost_native_importance": native})
    cmpimp["rank_stack"] = cmpimp.stack_perm_SHAP.rank(ascending=False)
    cmpimp["rank_catboost_shap"] = cmpimp.catboost_treeSHAP.rank(ascending=False)
    cmpimp["rank_native"] = cmpimp.catboost_native_importance.rank(ascending=False)
    cmpimp.sort_values("rank_stack").round(4).to_csv(TAB_DIR / "importance_comparison.csv")
    json.dump({"spearman_stack_vs_catboost_treeshap": rho,
               "spearman_stack_vs_native": spearmanr(imp.reindex(feats), native).correlation},
              open(TAB_DIR / "importance_agreement.json", "w"), indent=2)

    # ================================================= 3. EBM glass-box view
    ebm = stack.base_models_["EBM"][-1]
    g = ebm.explain_global()
    gd = g.data()
    ebm_imp = pd.Series(gd["scores"], index=gd["names"]).sort_values(ascending=False)
    ebm_imp.rename("ebm_importance").to_csv(TAB_DIR / "ebm_term_importance.csv")
    shape_feats = [f for f in ["asset_types_owned", "head_edu_yrs", "n_members", "dependency_ratio",
                               "persons_per_bedroom", "km_to_ds_office"] if f in ebm.term_names_]
    fig, axs = plt.subplots(2, 3, figsize=(11, 6))
    for ax, f in zip(axs.flat, shape_feats):
        k = ebm.term_names_.index(f)
        d = g.data(k)
        x = np.asarray(d["names"], dtype=float)
        y = np.asarray(d["scores"], dtype=float)
        ax.step(x[:-1], y, where="post", color=SERIES[0], lw=1.8)
        if "lower_bounds" in d:
            ax.fill_between(x[:-1], d["lower_bounds"], d["upper_bounds"], step="post", color=SERIES[0], alpha=0.15, lw=0)
        ax.axhline(0, color=MUTED, lw=0.8)
        lo, hi = np.nanpercentile(dtr[f], [1, 99])
        ax.set_xlim(lo, hi if hi > lo else lo + 1)
        ax.set_title(f, fontsize=10); ax.set_ylabel("log-odds contribution")
    fig.suptitle("EBM shape functions: exact contribution of each feature (glass-box base learner)", x=0.01,
                 ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, FIG_DIR / "fig10_ebm_shapes.png")

    # ================================================= 4. PDP / ICE on the stack
    from sklearn.inspection import partial_dependence
    pdp_feats = [f for f in imp.index if f in nums][:4]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.4))
    sub = Xte.sample(250, random_state=2)
    for ax, f in zip(axs, pdp_feats):
        grid = np.unique(np.nanpercentile(dtr[f], np.linspace(2, 98, 15)))
        ice = []
        for v in grid:
            Z = sub.copy(); Z[f] = v
            ice.append(stack.predict_proba(Z)[:, 1])
        ice = np.array(ice)
        ax.plot(grid, ice[:, :60], color=SERIES[0], alpha=0.12, lw=0.8)
        ax.plot(grid, ice.mean(1), color=SERIES[1], lw=2.4, label="PDP (average)")
        ax.set_title(f, fontsize=10); ax.set_ylabel("P(poor)")
    axs[0].legend(fontsize=8)
    fig.suptitle("Partial dependence (orange) and ICE curves (blue) - stacked ensemble", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, FIG_DIR / "fig11_pdp_ice.png")

    # ================================================= 5. single-sample inference
    cand_idx = np.where((yte == 1) & (T[SN].values > 0.5) & (T[SN].values < 0.85))[0]
    i = int(cand_idx[0]) if len(cand_idx) else int(np.argmax(T[SN].values))
    one = Xte.iloc[[i]]
    sample = one.iloc[0].to_dict()
    sample = {k: (None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v))
              for k, v in sample.items()}
    json.dump(sample, open(ROOT / "app" / "sample_household.json", "w"), indent=2)
    loaded = joblib.load(MODELS_DIR / "poverty_model.pkl")
    p1 = float(loaded["model"].predict_proba(one)[:, 1][0])
    sv1 = ex.shap_values(one)
    plt.figure()
    shap.plots.waterfall(sv1[0], max_display=14, show=False)
    plt.title(f"Single held-out household: P(poor) = {p1:.2f}, actual = {'poor' if yte[i] else 'not poor'}",
              loc="left", fontweight="bold", fontsize=10)
    plt.gcf().savefig(FIG_DIR / "fig12_single_household_waterfall.png", bbox_inches="tight", dpi=160); plt.close()
    contrib = pd.Series(sv1.values[0], index=feats).sort_values(key=np.abs, ascending=False).head(10)
    json.dump({"row_in_test": i, "psu": int(dte.psu[i]), "predicted_probability": p1,
               "decision_threshold": thr, "decision": "Eligible (poor)" if p1 >= thr else "Not eligible",
               "actual_poor": int(yte[i]), "actual_pce": float(dte.pce[i]),
               "base_learner_probabilities": dict(zip(stack.base_specs, map(float, stack.base_scores(one)[0]))),
               "top_contributions_logodds": contrib.round(4).to_dict()},
              open(TAB_DIR / "single_sample_inference.json", "w"), indent=2)

    # ================================================= 6. fairness audit
    s = T[SN].values
    sel = s >= thr
    dte["head_sex"] = np.where(dte.head_female == 1, "Female head", "Male head")
    rows = []
    for gcol in ["sector_name", "province", "audit_head_ethnicity", "head_sex"]:
        for gval, d in dte.groupby(gcol):
            ix = d.index.values
            if len(ix) < 60 or yte[ix].sum() < 5:
                continue
            excl, incl = selection_errors(yte[ix], sel[ix], wte[ix])
            rows.append({"attribute": gcol.replace("audit_", ""), "group": gval, "n_households": len(ix),
                         "poverty_rate": np.average(yte[ix], weights=wte[ix]),
                         "selection_rate": np.average(sel[ix], weights=wte[ix]),
                         "recall": 1 - excl, "exclusion_error": excl, "inclusion_error": incl,
                         "ROC_AUC": roc_auc_score(yte[ix], s[ix], sample_weight=wte[ix]),
                         "mean_pred": np.average(s[ix], weights=wte[ix])})
    fair = pd.DataFrame(rows)
    fair.round(4).to_csv(TAB_DIR / "fairness_audit.csv", index=False)
    log(fair.round(3).to_string(index=False))

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    fs = fair[fair.attribute.isin(["sector_name", "audit_head_ethnicity", "head_ethnicity", "head_sex"])].copy()
    fs["label"] = fs.group
    ypos = np.arange(len(fs))
    axs[0].barh(ypos, fs.poverty_rate * 100, color=MUTED, height=0.38, label="Actual poverty rate")
    axs[0].barh(ypos + 0.38, fs.selection_rate * 100, color=SERIES[0], height=0.38, label="Share selected by model")
    axs[0].set_yticks(ypos + 0.19, fs.label); axs[0].set_xlabel("% of households"); axs[0].legend(fontsize=8)
    axs[0].set_title("Selection vs need")
    axs[1].barh(ypos + 0.19, fs.recall * 100, color=SERIES[2], height=0.5)
    for y, v in zip(ypos, fs.recall * 100):
        axs[1].text(v + 1, y + 0.19, f"{v:.0f}%", va="center", fontsize=8, color=INK2)
    axs[1].set_xlabel("Recall: % of poor households reached"); axs[1].set_title("Coverage of the poor")
    for a in axs:
        a.grid(axis="y", visible=False)
    fig.suptitle("Fairness audit at the policy threshold (test set)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, FIG_DIR / "fig13_fairness.png")

    # ================================================= 7. Samurdhi comparison
    sam = dte.audit_samurdhi.values.astype(bool)
    cov = float(np.average(sam, weights=wte))
    oi = pd.read_csv(ROOT / "data" / "raw" / "SEC_5_5_1_OTHER_INCOME.csv", low_memory=False)
    samamt = oi.groupby(["psu", "snumber", "hhno"]).samurdhi.sum().rename("sam_amt").reset_index()
    d2 = dte.merge(samamt, on=["psu", "snumber", "hhno"], how="left").fillna({"sam_amt": 0})
    y_pre = (((d2.hhexppm - d2.sam_amt) / d2.hhsize) < art["poverty_line"]).astype(int).values
    rows = []
    for label, yy in [("Post-transfer PCE (as measured)", yte), ("Pre-transfer PCE (Samurdhi removed)", y_pre)]:
        e, i_ = selection_errors(yy, sam, wte)
        rows.append({"poverty definition": label, "rule": "Actual Samurdhi receipt (2019)", "coverage": cov,
                     "exclusion_error": e, "inclusion_error": i_})
        for m in [SN, "OLS PMT (existing practice)", best_single]:
            e, i_ = targeting_errors(yy, T[m].values, wte, cov)
            rows.append({"poverty definition": label, "rule": f"{m} at same coverage", "coverage": cov,
                         "exclusion_error": e, "inclusion_error": i_})
    samtab = pd.DataFrame(rows)
    samtab.round(4).to_csv(TAB_DIR / "samurdhi_comparison.csv", index=False)
    log(samtab.round(3).to_string(index=False))

    covs = np.linspace(0.05, 0.5, 46)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i_, m in enumerate([SN, "OLS PMT (existing practice)"]):
        ex_ = [targeting_errors(yte, T[m].values, wte, c)[0] * 100 for c in covs]
        ax.plot(covs * 100, ex_, color=SERIES[i_] if i_ == 0 else MUTED, lw=2.2 if i_ == 0 else 1.6, label=m)
    e0, _ = selection_errors(yte, sam, wte)
    ax.scatter([cov * 100], [e0 * 100], s=70, color=SERIES[7], zorder=5, label="Actual Samurdhi (2019)")
    ax.annotate(f"Samurdhi: {e0 * 100:.0f}% of poor\nhouseholds excluded", (cov * 100, e0 * 100),
                xytext=(cov * 100 + 4, e0 * 100 - 4), fontsize=8, color=INK2)
    ax.axvline(prate * 100, color=MUTED, lw=0.8, ls=":")
    ax.set(xlabel="Programme coverage, % of households selected", ylabel="Exclusion error, % of poor missed",
           title="Exclusion error vs programme coverage (test set)")
    ax.legend(fontsize=8)
    save(fig, FIG_DIR / "fig14_samurdhi_exclusion_curve.png")

    # ================================================= 8. compact field-form ablation
    from poverty.models import fit_model, make_model
    params = json.loads((MODELS_DIR / "best_params.json").read_text())["CatBoost"]["params"]
    rows = []
    for k in [10, 20, 30, 45, len(feats)]:
        fk = list(imp.index[:k])
        nk, ck = [f for f in fk if f in nums], [f for f in fk if f in cats]
        m = fit_model(make_model("CatBoost", nk, ck, params), Xtr[fk], ytr, wtr, "CatBoost")
        s_ = m.predict_proba(Xte[fk])[:, 1]
        rows.append({"n_features": k, "PR_AUC": average_precision_score(yte, s_, sample_weight=wte),
                     "ROC_AUC": roc_auc_score(yte, s_, sample_weight=wte)})
        log(rows[-1])
    pd.DataFrame(rows).round(4).to_csv(TAB_DIR / "compact_form_ablation.csv", index=False)


if __name__ == "__main__":
    main()
