# 09_cea_bounds.py
# Purpose: Bounded cost-effectiveness analysis (Aim 3). Turns the mortality
#          estimates into deaths averted per hospital-year, then asks
#            (1) the most a hospital could justify spending per year, given the
#                largest mortality benefit the confidence interval allows, and
#            (2) the mortality reduction a given annual cost would need to buy
#                to be cost-effective, compared with the confidence interval.
# Author: Caprice Culkin
# Date: November 2026
#
# Inputs: output/att_simple.csv (03_analysis.R), data/processed/analysis_panel.csv,
#         data/processed/cms_complications_panel.csv (patient counts)
# Output: output/tables/cea_bounds.csv, output/tables/cea_threshold.csv
#
# Assumptions are a sensitivity grid, not point values. LIFE_YEARS and
# QALY_WEIGHT are placeholders until replaced with cited estimates for
# Medicare patients surviving each condition. Deaths averted only counts the
# five measured conditions, so it understates any hospital-wide effect.

import itertools
import os

import numpy as np
import pandas as pd

ARM, SPEC = "epic", "main"
MORT = {"mort_ami": "MORT_30_AMI", "mort_hf": "MORT_30_HF", "mort_pn": "MORT_30_PN",
        "mort_copd": "MORT_30_COPD", "mort_cabg": "MORT_30_CABG"}
LIFE_YEARS = [1, 2, 5, 10]          # life-years gained per death averted (to cite)
QALY_WEIGHT = [0.6, 0.8]            # quality weight for those years (to cite)
WTP = [50_000, 100_000, 150_000]    # willingness to pay per QALY
ANNUAL_COST = [1e6, 5e6, 10e6, 20e6]  # annual cost per hospital to test (threshold)

os.makedirs("output/tables", exist_ok=True)
att = pd.read_csv("output/att_simple.csv")
att = att[(att["spec"] == SPEC) & (att["arm"] == ARM) & att["outcome"].isin(MORT)]

# Patients per year at treated hospitals: CMS denominators cover multi-year
# windows, so divide by the window length.
panel = pd.read_csv("data/processed/analysis_panel.csv", dtype={"ccn": str}, low_memory=False)
treated_ccn = panel.loc[(panel["arm"] == ARM) & (panel["gvar"] > 0), "ccn"].dropna().unique()
c = pd.read_csv("data/processed/cms_complications_panel.csv", dtype=str, low_memory=False)
c = c[c["measure_id"].isin(MORT.values()) & c["ccn"].isin(treated_ccn)].copy()
c["n"] = pd.to_numeric(c["denominator"], errors="coerce")
years = (pd.to_datetime(c["end_date"]) - pd.to_datetime(c["start_date"])).dt.days / 365.25
c["per_year"] = c["n"] / years.round()
pts = c.groupby("measure_id")["per_year"].median()

rows = []
for _, r in att.iterrows():
    p = pts.get(MORT[r["outcome"]], np.nan)
    rows.append(dict(outcome=r["outcome"], att_pp=r["att"], ci_low=r["ci_low"], ci_high=r["ci_high"],
                     patients_per_hosp_year=p,
                     deaths_averted_point=-r["att"] / 100 * p,
                     deaths_averted_best_case=max(0.0, -r["ci_low"]) / 100 * p))
eff = pd.DataFrame(rows)
best = eff["deaths_averted_best_case"].sum()
point = eff["deaths_averted_point"].sum()
print(eff.round(3).to_string(index=False))
print(f"\nDeaths averted per treated hospital-year: point {point:.3f}, "
      f"best case the CIs allow {best:.3f}")

# (1) maximum justifiable annual spending per hospital
bounds = pd.DataFrame([dict(life_years=ly, qaly_weight=q, wtp=w,
                            max_justifiable_cost=best * ly * q * w)
                       for ly, q, w in itertools.product(LIFE_YEARS, QALY_WEIGHT, WTP)])
bounds.to_csv("output/tables/cea_bounds.csv", index=False)
print("\nMost a hospital could justify spending per year (best case), $:")
print(bounds.pivot_table(index=["life_years", "qaly_weight"], columns="wtp",
                         values="max_justifiable_cost").round(-3).to_string())

# (2) threshold: deaths that a given annual cost must avert, and the pooled
# mortality reduction (pp across the five conditions) that implies
total_pts = eff["patients_per_hosp_year"].sum()
thr = pd.DataFrame([dict(annual_cost=k, life_years=ly, qaly_weight=q, wtp=w,
                         deaths_needed=k / (ly * q * w),
                         pp_reduction_needed=100 * k / (ly * q * w) / total_pts)
                    for k, ly, q, w in itertools.product(ANNUAL_COST, LIFE_YEARS, QALY_WEIGHT, WTP)])
thr.to_csv("output/tables/cea_threshold.csv", index=False)
pooled_best = 100 * best / total_pts
print(f"\nPooled best-case mortality reduction the CIs allow: {pooled_best:.2f} pp "
      f"over {total_pts:.0f} patients per hospital-year")
print("Mortality reduction needed (pp) at $100k/QALY, quality weight 0.8:")
print(thr[(thr["wtp"] == 100_000) & (thr["qaly_weight"] == 0.8)]
      .pivot_table(index="annual_cost", columns="life_years", values="pp_reduction_needed")
      .round(2).to_string())
print("\nSaved output/tables/cea_bounds.csv and cea_threshold.csv")
