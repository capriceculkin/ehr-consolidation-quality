# 06_build_analysis_panel.py
# Purpose: Attach CMS quality outcomes to the hospital-year panel
# Author: Caprice Culkin
# Date: October 2026
#
# Run order: see README.md
#
# Inputs:
#   data/processed/panel_v2.csv                (02_merge_v2.py)
#   data/processed/cms_timely_panel.csv        (05_stack_cms.py)
#   data/processed/cms_complications_panel.csv (05_stack_cms.py)
#   data/processed/cms_unplanned_panel.csv     (05_stack_cms.py)
#   data/processed/cost_report_panel.csv       (05b_build_cost_reports.py, optional)
#   data/processed/hvbp_panel.csv              (05c_stack_hvbp.py, optional)
#
# Output:
#   data/processed/analysis_panel.csv, one row per hospital-year with
#   treatment timing (gvar), arm, outcomes, and covariates
#
# Year alignment: a CMS row is assigned to the calendar year of its End
# Date (measure_year) and joined to the AHA survey year of the same number.
# Mortality and readmission use multi-year windows, so this is a modeling
# choice to confirm with the advisor.

import sys
import pandas as pd
import numpy as np

from config import MAIN_EVENT_TYPES

panel = pd.read_csv("data/processed/panel_v2.csv", dtype={"id": str, "ccn_it": str,
                                                          "ccn_annual": str})

# -------------------------
# One CCN per hospital-year
# -------------------------
# The CCN comes from the Annual Survey where present, else the IT Supplement.
# Hospital-years where the two sources disagree are listed for review.

ccn_a = panel.get("ccn_annual", pd.Series(pd.NA, index=panel.index))
ccn_i = panel.get("ccn_it", pd.Series(pd.NA, index=panel.index))
disagree = ccn_a.notna() & ccn_i.notna() & (ccn_a != ccn_i)
if disagree.any():
    panel[disagree].to_csv("data/processed/ccn_disagreements.csv", index=False)
    print(f"NOTE: {disagree.sum()} hospital-years where the two AHA files give "
          f"different CCNs. Saved to ccn_disagreements.csv; Annual Survey CCN used.")
panel["ccn"] = ccn_a.fillna(ccn_i)

dup_ccn = panel[panel["ccn"].notna()].duplicated(["ccn", "year"], keep=False)
if dup_ccn.any():
    panel.loc[dup_ccn[dup_ccn].index].sort_values(["ccn", "year"]).to_csv(
        "data/processed/ccn_shared_by_multiple_ids.csv", index=False)
    print(f"NOTE: {dup_ccn.sum()} hospital-years share a CCN with another AHA id "
          f"(multi-campus hospitals). These rows get no CMS outcomes, since CMS "
          f"reports one value per CCN. List saved to ccn_shared_by_multiple_ids.csv")

# -------------------------
# Outcomes: long CMS files -> one column per measure
# -------------------------

OUTCOMES = {
    "SEP_1": "sep1",                 # documentation-dependent process measure
    "SEP_SH_3HR": "sep_shock_3hr",   # SEP-1 components: 3-hour bundles are mostly
    "SEV_SEP_3HR": "sep_sev_3hr",    #   orders and timestamps; 6-hour bundles add
    "SEP_SH_6HR": "sep_shock_6hr",   #   documented reassessment (most charting-
    "SEV_SEP_6HR": "sep_sev_6hr",    #   dependent element)
    "OP_18b": "ed_time",             # ED median time (timestamp workflow)
    "OP_22": "ed_lwbs",              # left before being seen
    "IMM_3": "staff_flu_vax",        # placebo: staff vaccination, not from the EHR
    "SAFE_USE_OF_OPIOIDS": "opioid_ecqm",  # eCQM computed from EHR data
    "OP_18c": "ed_time_psych",       # ED time, psychiatric patients
    "OP_23": "stroke_ct_45min",      # head CT read within 45 min (timestamp)
    "MORT_30_COPD": "mort_copd", "MORT_30_CABG": "mort_cabg",
    "READM_30_COPD": "readm_copd", "READM_30_CABG": "readm_cabg",
    "READM_30_HIP_KNEE": "readm_hipknee", "COMP_HIP_KNEE": "comp_hipknee",
    "EDAC_30_HF": "edac_hf", "EDAC_30_PN": "edac_pn",
    # Patient Safety Indicators: computed from billing codes, so they depend on
    # how complications and present-on-admission status are documented and coded
    "PSI_90": "psi90", "PSI_03": "psi03_ulcer", "PSI_06": "psi06_pneumothorax",
    "PSI_08": "psi08_fall_fracture", "PSI_09": "psi09_hemorrhage",
    "PSI_11": "psi11_resp_failure", "PSI_12": "psi12_pe_dvt",
    "PSI_13": "psi13_sepsis", "PSI_15": "psi15_puncture",
    "VTE_1": "vte1", "VTE_2": "vte2",
    "STK_02": "stk02", "STK_03": "stk03", "STK_05": "stk05",
    "MORT_30_AMI": "mort_ami", "MORT_30_HF": "mort_hf",
    "MORT_30_PN": "mort_pn", "MORT_30_STK": "mort_stk",
    "READM_30_AMI": "readm_ami", "READM_30_HF": "readm_hf",
    "READM_30_PN": "readm_pn", "Hybrid_HWR": "readm_hw",
}

frames = []
for f in ["cms_timely_panel", "cms_complications_panel", "cms_unplanned_panel"]:
    df = pd.read_csv(f"data/processed/{f}.csv", dtype=str)
    frames.append(df[["ccn", "measure_id", "measure_year", "end_date", "score"]])
cms = pd.concat(frames, ignore_index=True)
cms = cms[cms["measure_id"].isin(OUTCOMES)]
cms["measure_year"] = pd.to_numeric(cms["measure_year"], errors="coerce")
cms["value"] = pd.to_numeric(cms["score"], errors="coerce")   # "Not Available" -> NaN

print("\nShare of CMS rows with a numeric score, by measure:")
print(cms.groupby("measure_id")["value"].apply(lambda v: round(v.notna().mean(), 3)).to_string())

# If two releases give different windows ending in the same calendar year,
# keep the window that ends latest.
cms = (cms.dropna(subset=["measure_year"])
          .sort_values("end_date")
          .drop_duplicates(["ccn", "measure_id", "measure_year"], keep="last"))

wide = (cms.pivot(index=["ccn", "measure_year"], columns="measure_id", values="value")
           .rename(columns=OUTCOMES).reset_index()
           .rename(columns={"measure_year": "year"}))
wide["year"] = wide["year"].astype(int)

# -------------------------
# Join outcomes to the panel
# -------------------------

rows_before = len(panel)
shared = (panel["ccn"].isin(panel.loc[dup_ccn[dup_ccn].index, "ccn"]) if dup_ccn.any()
          else pd.Series(False, index=panel.index))
panel["ccn_for_join"] = panel["ccn"].where(~shared)
panel = panel.merge(wide.rename(columns={"ccn": "ccn_for_join"}),
                    on=["ccn_for_join", "year"], how="left", validate="many_to_one")
assert len(panel) == rows_before, "Row count changed when joining outcomes"

# -------------------------
# Payment, cost and VBP outcomes (Aim 3)
# -------------------------
# Built by 05b_build_cost_reports.py and 05c_stack_hvbp.py. Joined on CCN and
# year like the CMS measures; multi-campus shared CCNs again get no values.
FINANCE = ["drg_per_mcr_dc", "log_drg_per_mcr_dc", "op_margin", "total_margin",
           "cost_per_dc", "log_total_costs", "mcr_dc_share", "log_charges_per_dc",
           "log_markup", "log_npr_per_dc", "discount_share", "fte_per_bed",
           "log_total_dc", "log_cost_per_dc", "outlier_per_mcr_dc", "outlier_share",
           "bad_debt_share", "charity_share", "dsh_pct"]
try:
    fin = pd.read_csv("data/processed/cost_report_panel.csv", dtype={"ccn": str})
    panel = panel.merge(fin[["ccn", "year"] + FINANCE].rename(columns={"ccn": "ccn_for_join"}),
                        on=["ccn_for_join", "year"], how="left", validate="many_to_one")
    print(f"Cost report outcomes joined: {panel['drg_per_mcr_dc'].notna().sum():,} hospital-years")
except FileNotFoundError:
    FINANCE = []
    print("NOTE: no cost_report_panel.csv; run 05b_build_cost_reports.py for Aim 3 outcomes")
try:
    vbp = pd.read_csv("data/processed/hvbp_panel.csv", dtype={"ccn": str})
    vbp = vbp.dropna(subset=["tps"]).rename(columns={"ccn": "ccn_for_join", "perf_year": "year",
                                                     "tps": "vbp_tps"})
    panel = panel.merge(vbp[["ccn_for_join", "year", "vbp_tps"]], on=["ccn_for_join", "year"],
                        how="left", validate="many_to_one")
    FINANCE.append("vbp_tps")
    print(f"VBP scores joined: {panel['vbp_tps'].notna().sum():,} hospital-years")
except FileNotFoundError:
    print("NOTE: no hvbp_panel.csv; run 05c_stack_hvbp.py for the VBP outcome")
assert len(panel) == rows_before, "Row count changed when joining payment outcomes"

# -------------------------
# Treatment timing for Callaway-Sant'Anna
# -------------------------
# gvar = first treated year for Epic and Oracle arms, 0 for never acquired.
# Mixed and unclassified acquisitions are kept in the file but get gvar NA
# and are excluded in 03_analysis.R.

# Only clean acquisitions (MAIN_EVENT_TYPES in config.py) are treated in the
# main analysis; mergers and new system IDs get gvar NA and are excluded.
treated = panel["arm"].isin(["epic", "oracle"]) & panel["event_type"].isin(MAIN_EVENT_TYPES)
panel["gvar"] = np.where(treated, panel["acquisition_year"],
                np.where(panel["arm"] == "never_acquired", 0, np.nan))
panel["hosp_num"] = panel["id"].astype("category").cat.codes + 1

COVARIATES = ["sysname", "mhsmemb", "bdtot", "hospbd", "admtot", "ipdtot", "mcripd", "mcdipd",
              "ftemd", "ftern", "serv", "cntrl", "mapp3", "mapp5", "mapp8", "mapp18",
              "cbsatype", "hrrcode", "fstcd", "mstate", "commty", "resp", "dtbeg", "dtend"]
for c in ["bdtot", "hospbd", "admtot", "ipdtot", "mcripd", "mcdipd", "ftemd", "ftern"]:
    if c in panel.columns:
        panel[c] = pd.to_numeric(panel[c], errors="coerce")
if {"mcripd", "ipdtot"} <= set(panel.columns):
    panel["medicare_share"] = panel["mcripd"] / panel["ipdtot"]
if {"mcdipd", "ipdtot"} <= set(panel.columns):
    panel["medicaid_share"] = panel["mcdipd"] / panel["ipdtot"]

keep = (["id", "hosp_num", "ccn", "year", "sysid", "arm", "event_type", "acquisition_year", "gvar",
         "vendor", "medicare_share", "medicaid_share"] + COVARIATES +
        [c for c in OUTCOMES.values() if c in panel.columns] + FINANCE)
panel = panel[[c for c in keep if c in panel.columns]]
assert not panel.duplicated(["id", "year"]).any(), "Duplicate hospital-years"

panel.to_csv("data/processed/analysis_panel.csv", index=False)

# -------------------------
# Coverage report: what the estimator will actually have
# -------------------------

print("\nHospital-years with each outcome, by year:")
present = [c for c in OUTCOMES.values() if c in panel.columns]
print(panel.groupby("year")[present].count().to_string())

print("\nHospitals by arm with at least one SEP-1 value:")
has_sep = panel.groupby("id")["sep1"].apply(lambda s: s.notna().any()) if "sep1" in panel else None
if has_sep is not None:
    arms = panel.drop_duplicates("id").set_index("id")["arm"]
    print(arms[has_sep[has_sep].index].value_counts().to_string())

print(f"\nSaved analysis_panel.csv: {len(panel):,} hospital-years, "
      f"{panel['id'].nunique():,} hospitals")
