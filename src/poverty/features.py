"""Build the household-level modelling table from HIES 2019 microdata.

Design principle (proxy means test, PMT): the model may only use characteristics
an enumerator can observe or verify at the doorstep -- location, demographics,
education, housing quality, durable assets, land/livestock and access to
facilities. Income, expenditure, debt and welfare receipt are deliberately
excluded:
  * income/expenditure define the target (direct leakage) and are what welfare
    targeting cannot verify in the first place;
  * debt is self-reported and easy to misstate;
  * Samurdhi receipt is the programme being evaluated (circular).
Ethnicity and religion are excluded as features (non-discrimination) but kept
as audit-only columns for the fairness analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DISTRICTS, KEYS, POVERTY_LINE, PROVINCES, RAW_DIR, SECTORS, ETHNICITY

# ---------------------------------------------------------------- label maps
STRUCTURE = {1: "Single house 1 floor", 2: "Single house 2 floors", 3: "Single house 3+ floors",
             4: "Attached house/annex", 5: "Flat", 6: "Condominium", 7: "Twin house",
             8: "Line room/row house", 9: "Slum/shanty"}
WALLS = {1: "Brick", 2: "Cement block", 3: "Stone", 4: "Cabook", 5: "Pressed soil block",
         6: "Cadjan/palmyrah", 7: "Wood/takaran/asbestos", 8: "Metal sheet"}
FLOOR = {1: "Cement", 2: "Terrazzo/tile/granite", 3: "Mud", 4: "Wood", 5: "Sand", 6: "Concrete"}
ROOF = {1: "Tile", 2: "Asbestos", 3: "Concrete", 4: "Metal sheet", 5: "Takaran", 6: "Cadjan/straw"}
TENURE = {1: "Built/purchased", 2: "Inherited", 3: "Gift", 4: "Compensated", 5: "Rent free",
          6: "Relief payment", 7: "Rented", 8: "Leased", 9: "Encroached"}
WATER = {1: "Protected well", 2: "Unprotected well", 3: "Tube well", 4: "Tap (NWSDB)",
         5: "Tap (community)", 6: "Tap (local govt)", 7: "Tap (private)", 8: "River/tank/stream",
         9: "Rain water", 10: "Bottled water", 11: "Bowser", 12: "Filter water"}
TOILET_USE = {1: "Exclusive", 2: "Shared with household", 3: "Shared with other unit",
              4: "Public toilet", 5: "No toilet"}
TOILET_TYPE = {1: "Water seal - septic tank", 2: "Water seal - pit", 3: "Water seal - sewer",
               4: "Water seal - drain/river", 5: "Pit latrine with deck", 6: "Open pit",
               7: "No facility"}
LIGHT = {1: "Electricity (grid)", 2: "Rural hydro", 3: "Kerosene", 4: "Solar", 5: "Bio gas"}
FUEL = {1: "Firewood", 2: "Kerosene", 3: "LP gas", 4: "Electricity", 5: "Sawdust/paddy husk",
        6: "Bio gas"}
GARBAGE = {1: "Collected", 2: "Burned", 3: "Dumped in premises", 4: "Composted",
           5: "Dumped outside"}
MARITAL = {1: "Never married", 2: "Married (registered)", 3: "Married (customary)",
           4: "Widowed", 5: "Divorced", 6: "Separated (legal)", 7: "Separated"}
EMP_STATUS = {1: "Government employee", 2: "Semi-government employee", 3: "Private employee",
              4: "Employer", 5: "Own-account worker", 6: "Contributing family worker"}
ISCO_MAJOR = {0: "Armed forces", 1: "Managers", 2: "Professionals", 3: "Technicians",
              4: "Clerical", 5: "Service & sales", 6: "Skilled agriculture",
              7: "Craft & trades", 8: "Machine operators", 9: "Elementary occupations"}

ASSETS = ["radio", "tv", "vcd", "sewingmechine", "washing_mechine", "fridge", "cookers",
          "electric_fans", "telephone", "telephone_mobile", "computers", "camera", "s6a_aircon",
          "bicycle", "motor_bicycle", "three_wheeler", "motor_car_van", "bus_lorry",
          "tractor_2_wheel", "tractor_4_wheel", "pesticider", "threshers", "waterpumps",
          "mechine", "boats", "fishing_nets"]
ASSET_NAMES = {"sewingmechine": "sewing_machine", "washing_mechine": "washing_machine",
               "s6a_aircon": "air_conditioner", "telephone": "landline_phone",
               "telephone_mobile": "mobile_phone", "pesticider": "sprayer",
               "mechine": "combine_harvester", "motor_bicycle": "motorcycle",
               "motor_car_van": "car_van", "cookers": "cooker", "electric_fans": "electric_fan",
               "computers": "computer", "vcd": "vcd_dvd"}

FACILITIES = {"bus_halt": "bus_halt", "primery_school": "primary_school",
              "secondery_school": "secondary_school", "hospital": "hospital",
              "gov_dispensarz": "govt_dispensary", "ds_office": "ds_office",
              "bank": "bank", "post_office": "post_office", "agri_office": "agrarian_office"}


def _edu_years(code: pd.Series) -> pd.Series:
    """Map the DCS 'level of education' code (Section 1, col 13) to approx. years."""
    c = code.astype("float")
    yrs = c.copy()
    yrs[c == 0] = 0.5            # studying grade 1
    yrs[c == 13] = 13            # G.C.E. A/L
    yrs[c == 14] = 14            # GAQ/GSQ
    yrs[c == 15] = 16            # degree
    yrs[c == 16] = 17            # postgraduate
    yrs[c == 17] = 20            # PhD
    yrs[c == 18] = np.nan        # special education - not comparable
    yrs[c == 19] = 0             # no schooling
    return yrs


def _lab(s: pd.Series, mapping: dict, other: str = "Other") -> pd.Series:
    out = s.map(mapping)
    out[s.notna() & out.isna()] = other
    return out.fillna("Unknown")


def _read(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / name, low_memory=False, **kw)


def build_household_table(raw_dir=None) -> pd.DataFrame:
    """Return one row per household: ids, target, audit columns and features."""
    global RAW_DIR
    if raw_dir is not None:
        from pathlib import Path
        RAW_DIR = Path(raw_dir)

    hh = _read("HH_expenditure_hh_Income.csv")
    out = hh[KEYS + ["district", "sector", "finalweight", "hhsize", "hhexppm", "hhincomepm"]].copy()

    # ---------------- target
    out["pce"] = out.hhexppm / out.hhsize
    out["poor"] = (out.pce < POVERTY_LINE).astype(int)

    # ---------------- location
    out["district_name"] = out.district.map(DISTRICTS)
    out["province"] = (out.district // 10).map(PROVINCES)
    out["sector_name"] = out.sector.map(SECTORS)

    # ---------------- demographics (person file)
    p = _read("SEC_1_DEMOGRAPHIC.csv")
    # The person file also lists 5,311 non-usual residents; keeping usual residents
    # (residence == 1) reproduces the official hhsize for 100% of households.
    p = p[p.residence == 1].copy()
    p["edu_yrs"] = _edu_years(p.education)
    p["is_child"] = p.age < 15
    p["is_elder"] = p.age >= 65
    p["is_wa"] = (p.age >= 15) & (p.age < 65)
    p["is_female"] = p.sex == 2
    p["is_employed"] = p.main_activity.isin([1, 2])
    p["is_unemployed"] = p.main_activity == 3
    p["is_disabled_old"] = p.main_activity == 9
    p["is_student"] = p.main_activity == 8
    p["is_pensioner"] = p.main_activity.isin([2, 4])
    p["is_govt"] = p.employment_status.isin([1, 2])
    p["is_casual_own"] = p.employment_status.isin([5, 6])
    p["in_agri"] = p.industry.notna() & (p.industry < 4000)        # ISIC divisions 01-03
    p["elementary_occ"] = p.main_occupation.notna() & (p.main_occupation // 1000 == 9)
    adults = p[p.age >= 18]

    g = p.groupby(KEYS).agg(
        n_members=("age", "size"), n_children=("is_child", "sum"), n_elderly=("is_elder", "sum"),
        n_working_age=("is_wa", "sum"), n_female=("is_female", "sum"),
        n_employed=("is_employed", "sum"), n_unemployed=("is_unemployed", "sum"),
        n_disabled_or_old=("is_disabled_old", "sum"), n_students=("is_student", "sum"),
        n_pensioners=("is_pensioner", "sum"), n_govt_employees=("is_govt", "sum"),
        n_own_account_family=("is_casual_own", "sum"), n_agri_workers=("in_agri", "sum"),
        n_elementary_occ=("elementary_occ", "sum"), max_edu_yrs=("edu_yrs", "max"),
        mean_age=("age", "mean"),
    ).reset_index()
    ga = adults.groupby(KEYS).agg(mean_adult_edu_yrs=("edu_yrs", "mean"),
                                  n_adults_al_plus=("edu_yrs", lambda s: (s >= 13).sum())).reset_index()

    head = p[p.relationship == 1].copy()
    head = head.drop_duplicates(KEYS)
    head_feats = pd.DataFrame({
        **{k: head[k] for k in KEYS},
        "head_female": (head.sex == 2).astype(int),
        "head_age": head.age,
        "head_edu_yrs": head.edu_yrs,
        "head_marital": _lab(head.marital_status, MARITAL),
        "head_employment": _lab(head.employment_status, EMP_STATUS, "Other").where(
            head.employment_status.notna(), "Not employed"),
        "head_occupation": _lab(head.main_occupation // 1000, ISCO_MAJOR).where(
            head.main_occupation.notna(), "Not employed"),
        "head_in_agriculture": head.in_agri.astype(int),
        # audit-only (never used as model inputs)
        "audit_head_ethnicity": _lab(head.ethnicity, ETHNICITY),
    })

    # ---------------- housing (Section 8)
    h = _read("SEC_8_HOUSING.csv")
    housing = pd.DataFrame({
        **{k: h[k] for k in KEYS},
        "house_type": _lab(h.structure, STRUCTURE),
        "bedrooms": h.bed_rooms,
        "floor_area_band": h.area.replace({9: 8}),              # ordinal 1..8
        "wall_material": _lab(h.walls, WALLS),
        "floor_material": _lab(h.floor, FLOOR),
        "roof_material": _lab(h.roof, ROOF),
        "tenure": _lab(h.ownership, TENURE),
        "drinking_water": _lab(h.drinking_water, WATER),
        "water_in_premises": (h.s8_6b1_inside_outside == 1).astype(int),
        "water_insufficient_last_month": (h.s8_6d_water_sufficency == 1).astype(int),
        "toilet_access": _lab(h.tioilet_use, TOILET_USE),
        "toilet_type": _lab(h.toilet_type, TOILET_TYPE),
        "handwash_with_soap": (h.s8_7c23_hand_wash_soap == 1).astype(int),
        "garbage_disposal": _lab(h.garbage_dumping, GARBAGE),
        "lighting": _lab(h.lite_source, LIGHT),
        "cooking_fuel": _lab(h.cooking_fuel, FUEL),
        "collects_firewood": (h.is_collect_firewood == 1).astype(int),
        "disaster_last_12m": (h.natural_calamity == 1).astype(int),
    })

    # ---------------- durable assets (Section 6A): number owned
    d = _read("SEC_6A_DURABLE_GOODS.csv")
    assets = d[KEYS].copy()
    for a in ASSETS:
        has = (d[a] == 1)
        qty = d[f"{a}_q"].where(has, 0)
        qty = qty.fillna(has.astype(float))   # owns but count missing -> 1
        assets[f"n_{ASSET_NAMES.get(a, a)}"] = qty.clip(upper=10)

    # ---------------- facilities (Section 7): distance in km
    f = _read("SEC_7_BASIC_FACILITIES.csv")
    fac = f[KEYS].copy()
    for src, name in FACILITIES.items():
        fac[f"km_to_{name}"] = f[src]
    fac["power_line_nearby"] = (f.is_power_lines_near == 1).astype(int)
    fac["pipe_water_line_nearby"] = (f.is_water_service_near == 1).astype(int)

    # ---------------- land & livestock (Section 9)
    land = _read("SEC_9_LAND_ANIMAL.csv")
    land_g = land.groupby(KEYS).agg(land_perches=("land_own_perch", "sum"),
                                    n_land_plots=("landnumber", "count")).reset_index()
    isl = _read("SECTION_9_IS_A_LAND.csv")[KEYS + ["is_agriland_owner"]]
    lv = _read("SECTION_9_2_OWNED_LIVESTOCKS.csv")
    live = pd.DataFrame({
        **{k: lv[k] for k in KEYS},
        "owns_land": (isl.set_index(KEYS).reindex(lv.set_index(KEYS).index).is_agriland_owner.values == 1).astype(int),
        "n_cattle": lv.cows_count.where(lv.s9_cattle_buffaloes == 1, 0).fillna(0).clip(upper=100),
        "n_goats": lv.goat_count.where(lv.goats_sheeps == 1, 0).fillna(0).clip(upper=100),
        "n_poultry": lv.chicken_count.where(lv.chickens == 1, 0).fillna(0).clip(upper=500),
    })

    # ---------------- welfare receipt (audit-only)
    oi = _read("SEC_5_5_1_OTHER_INCOME.csv")
    sam = oi.groupby(KEYS).samurdhi.sum(min_count=1).gt(0).astype(int).rename("audit_samurdhi").reset_index()

    for t in [g, ga, head_feats, housing, assets, fac, land_g, live, sam]:
        out = out.merge(t, on=KEYS, how="left")

    # fill structural zeros
    zero_cols = [c for c in out.columns if c.startswith("n_") or c in ("land_perches",)]
    out[zero_cols] = out[zero_cols].fillna(0)
    out["audit_samurdhi"] = out.audit_samurdhi.fillna(0).astype(int)
    out["land_perches"] = out.land_perches.clip(upper=2000)

    return add_derived(out)


ASSET_COLS = [f"n_{ASSET_NAMES.get(a, a)}" for a in ASSETS]
DERIVED = ["asset_types_owned", "dependency_ratio", "employed_share", "persons_per_bedroom",
           "female_share"]


def add_derived(out: pd.DataFrame) -> pd.DataFrame:
    """Engineered ratios. Shared by training and the web app so that both compute
    them identically (prevents training/serving skew)."""
    out = out.copy()
    out["asset_types_owned"] = (out[ASSET_COLS].fillna(0) > 0).sum(axis=1)
    out["dependency_ratio"] = (out.n_children + out.n_elderly) / out.n_working_age.clip(lower=1)
    out["employed_share"] = out.n_employed / out.n_members.clip(lower=1)
    out["persons_per_bedroom"] = out.n_members / out.bedrooms.clip(lower=1)
    out["female_share"] = out.n_female / out.n_members.clip(lower=1)
    return out


# ----------------------------------------------------------------- feature lists
ID_COLS = KEYS + ["district", "sector"]
TARGET = "poor"
WEIGHT = "finalweight"
NON_FEATURES = ID_COLS + [TARGET, WEIGHT, "hhexppm", "hhincomepm", "pce", "hhsize"]
AUDIT_COLS = ["audit_head_ethnicity", "audit_samurdhi", "province"]

CAT_FEATURES = ["district_name", "sector_name", "head_marital", "head_employment",
                "head_occupation", "house_type", "wall_material", "floor_material",
                "roof_material", "tenure", "drinking_water", "toilet_access", "toilet_type",
                "garbage_disposal", "lighting", "cooking_fuel"]


def feature_lists(df: pd.DataFrame):
    feats = [c for c in df.columns if c not in NON_FEATURES + AUDIT_COLS]
    cats = [c for c in CAT_FEATURES if c in feats]
    nums = [c for c in feats if c not in cats]
    return feats, nums, cats
