"""Exploratory data analysis: survey-weighted poverty profile and feature signal.
Outputs figures to reports/figures and tables to reports/tables."""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from poverty.config import FIG_DIR, POVERTY_LINE, TAB_DIR
from poverty.data import load
from poverty.features import feature_lists
from poverty.plotting import INK2, MUTED, SERIES, save, setup


def wrate(d, col="poor", w="finalweight"):
    return np.average(d[col], weights=d[w])


def main():
    setup()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    feats, nums, cats = feature_lists(df)
    df["popw"] = df.finalweight * df.hhsize

    # ---------------- summary table
    summ = pd.DataFrame({
        "Households (sample)": [len(df)], "Persons (sample)": [int(df.hhsize.sum())],
        "Districts": [df.district.nunique()], "PSUs (clusters)": [df.psu.nunique()],
        "Features used": [len(feats)], "Numeric": [len(nums)], "Categorical": [len(cats)],
        "Poor households, unweighted %": [100 * df.poor.mean()],
        "Poor households, weighted %": [100 * wrate(df)],
        "Poor persons (headcount), weighted %": [100 * wrate(df, w="popw")],
        "Median monthly PCE (Rs)": [df.pce.median()],
    }).T.rename(columns={0: "value"})
    summ.to_csv(TAB_DIR / "dataset_summary.csv")
    print(summ.round(2))

    # ---------------- profiles
    for col, name in [("sector_name", "sector"), ("province", "province"), ("district_name", "district")]:
        t = df.groupby(col).apply(lambda d: pd.Series({
            "households": len(d), "poverty_rate_hh_w": 100 * wrate(d),
            "headcount_w": 100 * wrate(d, w="popw"),
            "samurdhi_coverage_w": 100 * wrate(d, "audit_samurdhi")})).sort_values("headcount_w", ascending=False)
        t.round(2).to_csv(TAB_DIR / f"poverty_by_{name}.csv")

    # Fig 1: welfare distribution
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    lp = np.log10(df.pce)
    ax.hist(lp, bins=80, weights=df.popw / df.popw.sum() * 100, color=SERIES[0], edgecolor="white", linewidth=0.4)
    ax.axvline(np.log10(POVERTY_LINE), color=INK2, lw=1.5, ls="--")
    ax.text(np.log10(POVERTY_LINE) - 0.02, ax.get_ylim()[1] * 0.92, f"Poverty line\nRs {POVERTY_LINE:,}/person/month",
            ha="right", va="top", fontsize=9, color=INK2)
    ticks = [2000, 5000, 10000, 20000, 50000, 100000]
    ax.set_xticks(np.log10(ticks), [f"{t:,}" for t in ticks])
    ax.set_xlabel("Monthly expenditure per person, Rs (log scale)")
    ax.set_ylabel("% of population")
    ax.set_title("Per-capita expenditure is log-normal; 14% of people fall below the line")
    save(fig, FIG_DIR / "fig01_pce_distribution.png")

    # Fig 2: poverty by district
    t = pd.read_csv(TAB_DIR / "poverty_by_district.csv").sort_values("headcount_w")
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.barh(t.district_name, t.headcount_w, color=SERIES[0], height=0.7)
    for y, v in enumerate(t.headcount_w):
        ax.text(v + 0.4, y, f"{v:.1f}", va="center", fontsize=8, color=INK2)
    nat = 100 * wrate(df, w="popw")
    ax.axvline(nat, color=MUTED, lw=1, ls="--")
    ax.text(nat + 0.3, -0.9, f"National {nat:.1f}%", fontsize=8, color=INK2)
    ax.set_xlabel("Poverty headcount, % of persons (survey-weighted)")
    ax.set_title("Poverty is highest in Northern, Uva and Sabaragamuwa districts")
    ax.grid(axis="y", visible=False)
    save(fig, FIG_DIR / "fig02_poverty_by_district.png")

    # Fig 3: small multiples of strong observable correlates
    fig, axs = plt.subplots(2, 2, figsize=(9, 6))
    def rate_by(col, order=None, ax=None, title="", xlabel=""):
        g = df.groupby(col).apply(lambda d: pd.Series({"r": 100 * wrate(d), "n": len(d)}))
        g = g[g.n >= 50]
        if order is not None:
            g = g.reindex([o for o in order if o in g.index])
        ax.bar(range(len(g)), g.r, color=SERIES[0], width=0.7)
        ax.set_xticks(range(len(g)), [str(i) for i in g.index], rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("% poor households")
        ax.grid(axis="x", visible=False)
    rate_by("sector_name", ["Urban", "Rural", "Estate"], axs[0, 0], "By sector")
    rate_by("cooking_fuel", ["LP gas", "Kerosene", "Firewood"], axs[0, 1], "By main cooking fuel")
    df["edu_band"] = pd.cut(df.head_edu_yrs, [-1, 0, 5, 10, 11, 13, 25],
                            labels=["None", "Gr 1-5", "Gr 6-10", "O/L", "A/L", "Degree+"])
    rate_by("edu_band", ["None", "Gr 1-5", "Gr 6-10", "O/L", "A/L", "Degree+"], axs[1, 0], "By head's education")
    df["assets_band"] = pd.cut(df.asset_types_owned, [-1, 2, 4, 6, 8, 10, 30],
                               labels=["0-2", "3-4", "5-6", "7-8", "9-10", "11+"])
    rate_by("assets_band", ["0-2", "3-4", "5-6", "7-8", "9-10", "11+"], axs[1, 1], "By number of asset types owned")
    fig.suptitle("Observable household characteristics carry strong poverty signal", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, FIG_DIR / "fig03_poverty_correlates.png")

    # Fig 4: point-biserial correlation of numeric features with poverty
    corr = df[nums].corrwith(df.poor).dropna().sort_values()
    top = pd.concat([corr.head(10), corr.tail(10)])
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    ax.barh(top.index, top.values, color=[SERIES[0] if v < 0 else SERIES[7] for v in top.values], height=0.7)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Correlation with being poor")
    ax.set_title("Top 20 numeric correlates (red = more poverty, blue = less)")
    ax.grid(axis="y", visible=False)
    save(fig, FIG_DIR / "fig04_numeric_correlations.png")
    corr.to_csv(TAB_DIR / "numeric_correlations.csv")

    # data-quality table
    na = df[feats].isna().mean().mul(100)
    dq = pd.DataFrame({"missing_%": na[na > 0].round(3)})
    dq.to_csv(TAB_DIR / "missing_values.csv")
    skew = df[["hhincomepm", "hhexppm", "pce", "land_perches", "n_poultry", "km_to_hospital"]].skew().round(2)
    skew.to_csv(TAB_DIR / "skewness.csv")
    print("missing:\n", dq, "\nskew:\n", skew)


if __name__ == "__main__":
    main()
