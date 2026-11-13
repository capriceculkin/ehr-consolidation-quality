# 05b_build_cost_reports.py
# Purpose: Build hospital-year payment, cost and margin outcomes from the CMS
#          Hospital Provider Cost Report files (HCRIS extract), 2011-2023
# Author: Caprice Culkin
# Date: November 2026
#
# Input:  datasets/HPCR/Hospital Provider Cost Report.zip
#         (one CostReport_<year>_Final.csv per year; <year> is the cost
#          report fiscal year, which for most hospitals is the year the
#          fiscal year begins)
# Output: data/processed/cost_report_panel.csv, one row per CCN-year
#
# Outcomes (all from Medicare cost reports):
#   drg_per_mcr_dc      Medicare DRG payments (excluding outliers) per Medicare
#                       discharge. DRG payment = base rate x wage index x DRG
#                       weight, so within a hospital over time this moves with
#                       case mix (coding intensity). Proxy for the case mix index.
#   log_drg_per_mcr_dc  log of the above (percent change interpretation)
#   op_margin           net income from patient care / net patient revenue
#   total_margin        net income / (net patient revenue + other income)
#   log_total_costs     log total costs
#   cost_per_dc         total costs / total discharges
#   mcr_dc_share        Medicare share of discharges
#   log_charges_per_dc  log total (inpatient + outpatient) charges per discharge
#   log_markup          log charges / total costs (inverse cost-to-charge ratio)
#   log_npr_per_dc      log net patient revenue per discharge
#   discount_share      contractual allowances / gross patient revenue
#   fte_per_bed         employee FTEs per bed (staffing, documentation burden)
#
# Pre-specified prediction (H4, written before estimation): an Epic switch
# raises charges, markup and net revenue per discharge, but not Medicare DRG
# payment per discharge, because Medicare pays a fixed amount per DRG.
#
# Choices to confirm with the adviser:
#   - DRG payments are reported in one column through 2012 and split into
#     "before October 1" and "after October 1" from 2013 on. They are summed.
#   - When a hospital files more than one report in a year (fiscal year change,
#     ownership change), flows are summed before ratios are computed.
#   - Ratios are winsorized at the 1st and 99th percentile within each year.
#   - Medicare discharges are the hospital's adults and pediatrics Title XVIII
#     count (excludes subproviders), which matches the units DRG payments cover.

import glob
import os
import zipfile

import numpy as np
import pandas as pd

ZIP = "datasets/HPCR/Hospital Provider Cost Report.zip"
OUT = "data/processed/cost_report_panel.csv"

COLS = {
    "Provider CCN": "ccn",
    "CCN Facility Type": "facility_type",
    "Fiscal Year Begin Date": "fy_begin",
    "Fiscal Year End Date": "fy_end",
    "DRG Amounts Other Than Outlier Payments": "drg_single",
    "DRG Amounts Before October 1": "drg_before",
    "DRG Amounts After October 1": "drg_after",
    "Hospital Total Discharges Title XVIII For Adults & Peds": "mcr_dc",
    "Hospital Total Discharges (V + XVIII + XIX + Unknown) For Adults & Peds": "total_dc",
    "Total Costs": "total_costs",
    "Net Patient Revenue": "net_patient_rev",
    "Net Income from Service to Patients": "net_income_patients",
    "Total Other Income": "other_income",
    "Net Income": "net_income",
    "Combined Outpatient + Inpatient Total Charges": "total_charges",
    "Total Patient Revenue": "gross_patient_rev",
    "Less Contractual Allowance and Discounts on Patients' Accounts": "contractual_allow",
    "FTE - Employees on Payroll": "fte",
    "Number of Beds": "beds",
    "Outlier Payments For Discharges": "outlier",
    "Total Bad Debt Expense": "bad_debt",
    "Cost of Charity Care": "charity_cost",
    "Allowable DSH Percentage": "dsh_pct",
}
FLOWS = ["drg", "mcr_dc", "total_dc", "total_costs", "net_patient_rev",
         "net_income_patients", "other_income", "net_income", "total_charges",
         "gross_patient_rev", "contractual_allow", "outlier", "bad_debt", "charity_cost"]
STOCKS = ["fte", "beds", "dsh_pct"]   # not summed across reports in a year; the larger value is kept

frames = []
with zipfile.ZipFile(ZIP) as z:
    names = sorted(n for n in z.namelist() if n.endswith("_Final.csv"))
    for name in names:
        year = int(os.path.basename(name).split("_")[1])
        with z.open(name) as fh:
            d = pd.read_csv(fh, dtype=str, low_memory=False)
        missing = [c for c in COLS if c not in d.columns]
        if missing:
            raise SystemExit(f"STOP: {name} is missing columns {missing}")
        d = d[list(COLS)].rename(columns=COLS)
        for c in d.columns:
            if c not in ("ccn", "facility_type", "fy_begin", "fy_end"):
                d[c] = pd.to_numeric(d[c], errors="coerce")
        # DRG payments: one column through 2012, split column from 2013
        split = d[["drg_before", "drg_after"]]
        d["drg"] = d["drg_single"].where(d["drg_single"].notna(),
                                         split.sum(axis=1, min_count=1))
        d["year"] = year
        frames.append(d)
        print(f"{year}: {len(d):,} reports, {d['drg'].notna().sum():,} with DRG payments")

cr = pd.concat(frames, ignore_index=True)
cr["ccn"] = cr["ccn"].str.strip().str.replace(r"\.0+$", "", regex=True)
cr["ccn"] = cr["ccn"].where(~cr["ccn"].str.fullmatch(r"\d+"), cr["ccn"].str.zfill(6))

# Sum flows across multiple reports in the same CCN-year
n_reports = cr.groupby(["ccn", "year"]).size().rename("n_reports")
agg = cr.groupby(["ccn", "year"]).agg(
    {**{c: lambda s: s.sum(min_count=1) for c in FLOWS},
     **{c: "max" for c in STOCKS}, "facility_type": "last"})
agg = agg.join(n_reports).reset_index()
print(f"\nCCN-years with more than one report (flows summed): "
      f"{(agg['n_reports'] > 1).sum():,}")


def ratio(num, den):
    return np.where((den > 0) & num.notna(), num / den.where(den > 0), np.nan)


agg["drg_per_mcr_dc"] = ratio(agg["drg"], agg["mcr_dc"])
agg.loc[agg["drg_per_mcr_dc"] <= 0, "drg_per_mcr_dc"] = np.nan
agg["op_margin"] = ratio(agg["net_income_patients"], agg["net_patient_rev"])
agg["total_margin"] = ratio(agg["net_income"],
                            agg["net_patient_rev"] + agg["other_income"].fillna(0))
agg["cost_per_dc"] = ratio(agg["total_costs"], agg["total_dc"])
agg["mcr_dc_share"] = ratio(agg["mcr_dc"], agg["total_dc"])
# Revenue cycle (H4): charge capture shows up in charges and in revenue from
# payers that pay on charges, not in Medicare DRG payments
agg["charges_per_dc"] = ratio(agg["total_charges"], agg["total_dc"])
agg["markup"] = ratio(agg["total_charges"], agg["total_costs"])
agg["npr_per_dc"] = ratio(agg["net_patient_rev"], agg["total_dc"])
agg["discount_share"] = ratio(agg["contractual_allow"], agg["gross_patient_rev"])
agg["fte_per_bed"] = ratio(agg["fte"], agg["beds"])

RATIOS = ["drg_per_mcr_dc", "op_margin", "total_margin", "cost_per_dc", "mcr_dc_share",
          "charges_per_dc", "markup", "npr_per_dc", "discount_share", "fte_per_bed"]
RATIOS2 = ["outlier_per_mcr_dc", "outlier_share", "bad_debt_share", "charity_share"]
for c in RATIOS:
    lo = agg.groupby("year")[c].transform(lambda s: s.quantile(0.01))
    hi = agg.groupby("year")[c].transform(lambda s: s.quantile(0.99))
    agg[c] = agg[c].clip(lo, hi)
agg["log_drg_per_mcr_dc"] = np.log(agg["drg_per_mcr_dc"])
agg["log_total_costs"] = np.log(agg["total_costs"].where(agg["total_costs"] > 0))
for c in ["charges_per_dc", "markup", "npr_per_dc"]:
    agg[f"log_{c}"] = np.log(agg[c].where(agg[c] > 0))
# H5 / H6 (written before estimation): an Epic switch raises Medicare outlier
# payments per Medicare discharge (outlier status is judged on charges times a
# lagged cost-to-charge ratio, so faster charge growth inflates it), and lowers
# bad debt; charity care may fall as collections tighten.
# A blank outlier amount means zero outlier cases, but only in years where the
# extract reports the field at all (it is empty for every hospital after 2018).
reported = agg.groupby("year")["outlier"].transform(lambda s: s.notna().any())
out = agg["outlier"].fillna(0).where(agg["drg"].notna() & reported)
agg["outlier_per_mcr_dc"] = ratio(out, agg["mcr_dc"])
agg["outlier_share"] = ratio(out, agg["drg"])
agg["bad_debt_share"] = ratio(agg["bad_debt"], agg["net_patient_rev"])
agg["charity_share"] = ratio(agg["charity_cost"], agg["total_costs"])
for c in RATIOS2:
    lo = agg.groupby("year")[c].transform(lambda s: s.quantile(0.01))
    hi = agg.groupby("year")[c].transform(lambda s: s.quantile(0.99))
    agg[c] = agg[c].clip(lo, hi)
# volume-adjusted cost (logs avoid the skew of the dollar ratio) and volume
agg["log_total_dc"] = np.log(agg["total_dc"].where(agg["total_dc"] > 0))
agg["log_cost_per_dc"] = agg["log_total_costs"] - agg["log_total_dc"]

assert not agg.duplicated(["ccn", "year"]).any(), "Duplicate CCN-years"
keep = ["ccn", "year", "facility_type", "n_reports", "drg", "mcr_dc", "total_dc",
        "total_costs", "net_patient_rev", "dsh_pct"] + RATIOS + ["log_drg_per_mcr_dc", "log_total_costs",
                                   "log_charges_per_dc", "log_markup", "log_npr_per_dc",
                                   "log_total_dc", "log_cost_per_dc"] + RATIOS2
agg[keep].to_csv(OUT, index=False)

print("\nNon-missing values by year:")
print(agg.groupby("year")[RATIOS + ["log_drg_per_mcr_dc"]].count().to_string())
print("\nMedian by year (short-term hospitals):")
sth = agg[agg["facility_type"] == "STH"]
print(sth.groupby("year")[["drg_per_mcr_dc", "op_margin", "cost_per_dc"]].median()
      .round(3).to_string())
print(f"\nSaved {OUT}: {len(agg):,} CCN-years, {agg['ccn'].nunique():,} hospitals")
