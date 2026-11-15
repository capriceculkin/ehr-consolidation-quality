# 00_make_fake_panel.py
# Makes a fake hospital-year panel with the same columns as the real
# data/processed/analysis_panel.csv, so the descriptives and the estimator can
# run without the licensed AHA data.
#
# Nothing in here is real. Hospitals, systems, IDs and outcomes are all drawn
# at random. Two effects are built in on purpose so you can check that the
# estimator finds them:
#   - Epic arm:   SEP-1 drops 3 points in the year of acquisition and the year
#                 after, then goes back to normal
#   - Oracle arm: SEP-1 drops 2 points over the same two years
#   - Mortality has no built-in effect for either arm
#
# Run from the repo root:
#     python code/00_make_fake_panel.py
# Output:
#     data/processed/analysis_panel.csv

import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(20261113)  # fixed seed so every run makes the same panel

N_HOSP = 1800
YEARS = np.arange(2008, 2025)
EPIC_SEP1_EFFECT = -3.0
ORACLE_SEP1_EFFECT = -2.0


# Hospitals
# One row per hospital: its arm, how it joined a system, when, and fixed traits.

arm = rng.choice(["never_acquired", "epic", "oracle", "mixed", "unclassified"],
                 size=N_HOSP, p=[0.72, 0.13, 0.07, 0.05, 0.03])
acquired = arm != "never_acquired"

event_type = np.where(acquired,
                      rng.choice(["independent_joined", "hospital_moved", "whole_system_acquired", "new_system"],
                                 size=N_HOSP, p=[0.55, 0.25, 0.12, 0.08]),
                      None)
acq_year = np.where(acquired, rng.integers(2012, 2025, size=N_HOSP), np.nan)

# about 40% of hospitals are in the data from 2008, the rest join in 2014
first_year = np.where(rng.random(N_HOSP) < 0.4, 2008, 2014)

hosp = pd.DataFrame({
    "id": [f"F{i:05d}" for i in range(1, N_HOSP + 1)],  # F marks a fake ID
    "ccn": [f"99{i:04d}" for i in range(1, N_HOSP + 1)],
    "arm": arm,
    "event_type": event_type,
    "acquisition_year": acq_year,
    "first_year": first_year,
    # acquired hospitals run a little larger, like the real data
    "bdtot": np.round(np.exp(rng.normal(4.4 + 0.3 * acquired, 0.8))).clip(10, 1500),
    "cntrl": rng.choice([13, 15, 21, 23, 33, 45], size=N_HOSP, p=[0.12, 0.08, 0.38, 0.14, 0.24, 0.04]),
    "cbsatype": rng.choice(["Metro", "Micro", "Rural"], size=N_HOSP, p=[0.62, 0.18, 0.20]),
    "medicare_share": rng.beta(5, 6, size=N_HOSP).round(3),
    "medicaid_share": rng.beta(2, 9, size=N_HOSP).round(3),
    "sep1_base": rng.normal(56, 9, size=N_HOSP),       # each hospital's usual SEP-1 level
    "mort_ami_base": rng.normal(13.5, 1.2, size=N_HOSP),
    "mort_hf_base": rng.normal(11.5, 1.3, size=N_HOSP),
    "mort_pn_base": rng.normal(16.0, 1.6, size=N_HOSP),
    "start_vendor": rng.choice(["epic", "oracle", "other"], size=N_HOSP, p=[0.25, 0.25, 0.50]),
})

# never-acquired hospitals: most are already in some system, the rest independent
in_sys = (((hosp["arm"] == "never_acquired") & (rng.random(N_HOSP) < 0.6))
          | (hosp["event_type"] == "hospital_moved"))  # hospitals that move were already in a system
hosp["base_sysid"] = np.where(in_sys, [f"S{rng.integers(1, 300):03d}" for _ in range(N_HOSP)], None)


# Hospital-years
# Expand to one row per hospital per year it is observed.

panel = hosp.merge(pd.DataFrame({"year": YEARS}), how="cross")
panel = panel[panel["year"] >= panel["first_year"]].copy()


# Treatment timing
# gvar is the acquisition year for Epic and Oracle hospitals that joined in
# one of the main event types, 0 for never acquired, and missing for the rest.
main = panel["event_type"].isin(["independent_joined", "hospital_moved"])
panel["gvar"] = np.where(panel["arm"].isin(["epic", "oracle"]) & main, panel["acquisition_year"],
                np.where(panel["arm"] == "never_acquired", 0, np.nan))
# a hospital acquired in its first observed year never counts as treated
panel.loc[panel["acquisition_year"] <= panel["first_year"], "gvar"] = np.nan
panel["hosp_num"] = panel["id"].astype("category").cat.codes + 1
event_time = panel["year"] - panel["acquisition_year"]
after = panel["acquisition_year"].notna() & (event_time >= 0)

# system ID changes when the hospital is acquired
panel["sysid"] = np.where(after, "A" + panel["hosp_num"].astype(str).str.zfill(4),
                          panel["base_sysid"])


# Vendor
# Acquired Epic and Oracle hospitals move to their system's vendor the year
# after acquisition. A few never-acquired hospitals switch to Epic on their own.
# The IT Supplement has no 2021 survey, so vendor is missing that year.

vendor = panel["start_vendor"].copy()
vendor[(panel["arm"] == "epic") & (event_time >= 1)] = "epic"
vendor[(panel["arm"] == "oracle") & (event_time >= 1)] = "oracle"
own_switch_year = pd.Series(rng.integers(2012, 2040, size=N_HOSP), index=hosp["id"])  # many never switch
own = (panel["arm"] == "never_acquired") & (panel["year"] >= panel["id"].map(own_switch_year))
vendor[own] = "epic"
vendor[panel["year"] == 2021] = np.nan
panel["vendor"] = vendor


# Outcomes
# SEP-1 starts in 2017 and rises over time. The built-in effects above are
# added in event years 0 and 1. Mortality is reported every year except 2020
# and has no built-in effect.

hit = after & (event_time <= 1)
sep1 = panel["sep1_base"] + 1.5 * (panel["year"] - 2017) + rng.normal(0, 6, len(panel))
sep1 = sep1 + np.where(hit & (panel["arm"] == "epic"), EPIC_SEP1_EFFECT, 0)
sep1 = sep1 + np.where(hit & (panel["arm"] == "oracle"), ORACLE_SEP1_EFFECT, 0)
panel["sep1"] = np.where(panel["year"] >= 2017, sep1.clip(0, 100).round(1), np.nan)

for m in ["mort_ami", "mort_hf", "mort_pn"]:
    val = panel[f"{m}_base"] - 0.08 * (panel["year"] - 2008) + rng.normal(0, 0.7, len(panel))
    panel[m] = np.where(panel["year"] == 2020, np.nan, val.round(1))

# Ownership, CBSA type and system ID are missing before 2014, as in the real panel
panel.loc[panel["year"] < 2014, ["cntrl", "cbsatype"]] = np.nan


# Save

cols = ["id", "hosp_num", "ccn", "year", "sysid", "arm", "event_type", "acquisition_year", "gvar",
        "vendor", "medicare_share", "medicaid_share", "bdtot", "cntrl", "cbsatype",
        "sep1", "mort_ami", "mort_hf", "mort_pn"]
os.makedirs("data/processed", exist_ok=True)
panel[cols].to_csv("data/processed/analysis_panel.csv", index=False)

n_treated = panel.loc[panel["gvar"] > 0].groupby("arm")["id"].nunique()
print(f"Fake panel: {len(panel):,} hospital-years, {panel['id'].nunique():,} hospitals, "
      f"{YEARS.min()} to {YEARS.max()}")
print("Treated hospitals by arm:\n" + n_treated.to_string())
print(f"Built-in SEP-1 effect, event years 0 and 1: Epic {EPIC_SEP1_EFFECT}, Oracle {ORACLE_SEP1_EFFECT}")
