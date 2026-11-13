# 07_extensions.py
# Purpose: Build the analysis files for four extensions that look at Epic from
#          different angles. Each file is ready for 08_extensions.R.
# Author: Caprice Culkin
# Date: October 2026
#
# Run after 06_build_analysis_panel.py (and 01c_pull_it_exchange.py for ext 3).
#
#   EXT 1  Software vs. owner. Separates the EHR switch from the acquisition:
#            switch_only     switched to Epic, never acquired       (gvar = switch year)
#            acq_and_switch  acquired, switched to Epic within the window (gvar = acquisition year)
#            acq_no_switch   acquired, never switched to Epic       (gvar = acquisition year)
#            control         never acquired, never on Epic          (gvar = 0)
#   EXT 2  Homogenization. Among acquired hospitals only, the distance between a
#          hospital's score and the mean of its (eventual) acquiring system's
#          other members. Convergence = the distance shrinks after acquisition.
#          Uses not-yet-acquired hospitals as the comparison.
#   EXT 3  Data exchange and lock-in. AHA IT Supplement exchange variables as
#          outcomes, same treatment as the main analysis.
#   EXT 4  Reporting. Whether SEP-1 is reported with a score, and how many cases
#          were abstracted, same treatment as the main analysis.
#
# Outputs: data/processed/ext1_<group>.csv, ext2_converge.csv,
#          ext3_exchange.csv, ext4_reporting.csv, extensions_summary.txt

import os
import numpy as np
import pandas as pd

from config import MAIN_EVENT_TYPES, EXCHANGE_BINARY, EXCHANGE_SCALES

SWITCH_WINDOW = (-1, 3)   # Epic switch counted as "with acquisition" if within these years of it
MIN_SYSTEM_PEERS = 2      # other system members with a score needed to compute a system mean

P = "data/processed"
panel = pd.read_csv(f"{P}/analysis_panel.csv", dtype={"id": str, "ccn": str, "sysid": str},
                    low_memory=False)
events = pd.read_csv(f"{P}/acquisition_events.csv", dtype={"id": str, "acquiring_sysid": str})
panel["sysid"] = panel["sysid"].str.replace(r"\.0$", "", regex=True)
events["acquiring_sysid"] = events["acquiring_sysid"].str.replace(r"\.0$", "", regex=True)
summary = []


def log(msg):
    print(msg)
    summary.append(msg)


hosp = panel.drop_duplicates("id").set_index("id")
acquired_main = hosp["event_type"].isin(MAIN_EVENT_TYPES)
never_acquired = hosp["arm"].eq("never_acquired")

# =====================================================================
# EXT 1: separate the Epic switch from the acquisition
# =====================================================================
v = panel.dropna(subset=["vendor"]).sort_values(["id", "year"])[["id", "year", "vendor"]]
v["prev"] = v.groupby("id")["vendor"].shift(1)
v["first"] = v.groupby("id").cumcount() == 0
switch = (v[(v["vendor"] == "epic") & v["prev"].notna() & (v["prev"] != "epic") & ~v["first"]]
          .groupby("id")["year"].min().rename("epic_switch_year"))
first_vendor = v.groupby("id")["vendor"].first().rename("first_vendor")
n_vendor_obs = v.groupby("id").size().rename("n_vendor_obs")
ever_epic = v[v["vendor"] == "epic"].groupby("id").size().rename("n_epic_years")

h = (hosp[["hosp_num", "arm", "event_type", "acquisition_year"]]
     .join([switch, first_vendor, n_vendor_obs, ever_epic]))
h = h[h["n_vendor_obs"] >= 2]            # need a vendor history to classify
h["n_epic_years"] = h["n_epic_years"].fillna(0)
gap = h["epic_switch_year"] - h["acquisition_year"]

h["ext1_group"] = np.select(
    [never_acquired.reindex(h.index) & h["epic_switch_year"].notna(),
     acquired_main.reindex(h.index) & gap.between(*SWITCH_WINDOW) & (h["first_vendor"] != "epic"),
     acquired_main.reindex(h.index) & (h["n_epic_years"] == 0),
     never_acquired.reindex(h.index) & (h["n_epic_years"] == 0)],
    ["switch_only", "acq_and_switch", "acq_no_switch", "control"], default="excluded")
h["ext1_gvar"] = np.select(
    [h["ext1_group"] == "switch_only",
     h["ext1_group"].isin(["acq_and_switch", "acq_no_switch"]),
     h["ext1_group"] == "control"],
    [h["epic_switch_year"], h["acquisition_year"], 0], default=np.nan)

log("EXT 1  hospitals by group (vendor history of 2+ years required):")
log(h["ext1_group"].value_counts().to_string())
log("EXT 1  treated hospitals by first-treated year:")
log(h[h["ext1_gvar"] > 0].groupby(["ext1_gvar", "ext1_group"]).size()
    .unstack(fill_value=0).to_string())

outcomes = [c for c in ["sep1", "sep_shock_3hr", "sep_sev_3hr", "sep_shock_6hr", "sep_sev_6hr",
                        "ed_time", "ed_lwbs", "staff_flu_vax", "opioid_ecqm",
                        "ed_time_psych", "stroke_ct_45min", "psi90", "psi03_ulcer", "psi13_sepsis", "psi11_resp_failure", "psi12_pe_dvt", "mort_copd", "mort_cabg", "readm_copd", "readm_cabg", "readm_hipknee", "comp_hipknee", "edac_hf", "edac_pn",
                        "mort_ami", "mort_hf", "mort_pn",
                        "readm_ami", "readm_hf", "readm_pn",
                        # Aim 3: payment, cost and margin (cost reports), VBP score
                        "log_drg_per_mcr_dc", "drg_per_mcr_dc", "op_margin", "total_margin", "log_total_costs", "cost_per_dc", "mcr_dc_share", "vbp_tps", "log_charges_per_dc", "log_markup", "log_npr_per_dc", "discount_share", "fte_per_bed", "log_total_dc", "log_cost_per_dc", "outlier_per_mcr_dc", "outlier_share", "bad_debt_share", "charity_share"] if c in panel.columns]
base_cols = ["id", "hosp_num", "year", "bdtot"] if "bdtot" in panel.columns else ["id", "hosp_num", "year"]
p1 = panel[base_cols + outcomes].merge(h[["ext1_group", "ext1_gvar"]].rename(columns={"ext1_gvar": "gvar"}),
                                       left_on="id", right_index=True, how="inner")
for g in ["switch_only", "acq_and_switch", "acq_no_switch"]:
    d = p1[p1["ext1_group"].isin([g, "control"])]
    d.to_csv(f"{P}/ext1_{g}.csv", index=False)
    log(f"  saved ext1_{g}.csv: {d.loc[d['gvar'] > 0, 'id'].nunique()} treated, "
        f"{d.loc[d['gvar'] == 0, 'id'].nunique()} control hospitals")

# =====================================================================
# EXT 2: homogenization (distance to the acquiring system's other members)
# =====================================================================
acq = events[events["event_type"].isin(MAIN_EVENT_TYPES)][["id", "acquisition_year", "acquiring_sysid"]]
rows = []
for outcome in [o for o in ["sep1", "mort_hf", "readm_hf"] if o in panel.columns]:
    sc = panel[["id", "year", "sysid", outcome]].dropna(subset=[outcome])
    # system means by year, with sum and count so each hospital can be left out
    g = sc.dropna(subset=["sysid"]).groupby(["sysid", "year"])[outcome].agg(["sum", "count"]).reset_index()
    d = (sc[sc["id"].isin(acq["id"])].merge(acq, on="id")
         .merge(g, left_on=["acquiring_sysid", "year"], right_on=["sysid", "year"],
                how="left", suffixes=("", "_sys")))
    own_in_sys = d["sysid"].eq(d["acquiring_sysid"])     # hospital already counted in the system mean
    peer_sum = d["sum"] - np.where(own_in_sys, d[outcome], 0)
    peer_n = d["count"] - own_in_sys.astype(int)
    d["peer_mean"] = np.where(peer_n >= MIN_SYSTEM_PEERS, peer_sum / peer_n.replace(0, np.nan), np.nan)
    d[f"dist_{outcome}"] = (d[outcome] - d["peer_mean"]).abs()
    rows.append(d[["id", "year", f"dist_{outcome}"]])

if rows:
    p2 = rows[0]
    for r in rows[1:]:
        p2 = p2.merge(r, on=["id", "year"], how="outer")
    p2 = p2.merge(panel[["id", "year", "hosp_num"]], on=["id", "year"], how="left") \
           .merge(acq[["id", "acquisition_year"]].rename(columns={"acquisition_year": "gvar"}), on="id")
    p2.to_csv(f"{P}/ext2_converge.csv", index=False)
    log("EXT 2  acquired hospitals with a distance-to-acquirer value, by outcome:")
    for c in [c for c in p2.columns if c.startswith("dist_")]:
        log(f"  {c}: {p2.dropna(subset=[c])['id'].nunique()} hospitals, "
            f"mean distance {p2[c].mean():.2f}")

# =====================================================================
# EXT 3: data exchange (needs data/raw/aha_it_exchange.csv from 01c)
# =====================================================================
ex_path = "data/raw/aha_it_exchange.csv"
if os.path.exists(ex_path):
    ex = pd.read_csv(ex_path, dtype=str)
    ex.columns = ex.columns.str.lower()
    ex["id"] = ex["id"].str.replace(r"\.0$", "", regex=True)
    ex["year"] = pd.to_numeric(ex["year"]).astype(int)
    raw_vars = [c for c in ex.columns if c not in ("id", "year")]
    for c in raw_vars:
        ex[c] = pd.to_numeric(ex[c], errors="coerce")
    ex_vars = []
    for c in EXCHANGE_BINARY:
        if c in ex.columns:
            ex[c] = ex[c].where(ex[c].isin([0, 1]))
            ex_vars.append(c)
    for c, m in EXCHANGE_SCALES.items():
        if c in ex.columns:
            yes, no = ex[c].isin(m.get("yes", [])), ex[c].isin(m.get("no", []))
            ex[f"{c}_yes"] = np.where(yes, 1, np.where(no, 0, np.nan))
            ex_vars.append(f"{c}_yes")
    skipped = [c for c in raw_vars if c not in EXCHANGE_BINARY and c not in EXCHANGE_SCALES]
    if skipped:
        log(f"  scale variables left out until coded in config.EXCHANGE_SCALES: {skipped}")
    ex = ex[["id", "year"] + ex_vars]
    p3 = panel[["id", "hosp_num", "year", "arm", "event_type", "gvar"]].merge(
        ex, on=["id", "year"], how="inner")
    p3.to_csv(f"{P}/ext3_exchange.csv", index=False)
    log(f"EXT 3  exchange file: {len(p3):,} hospital-years, variables: {ex_vars}")
    log("  NOTE: outcomes are used as coded by AHA. Confirm in the codebook which "
        "values mean yes before interpreting signs.")
else:
    log("EXT 3  skipped: run python code/01c_pull_it_exchange.py first")

# =====================================================================
# EXT 4: SEP-1 reporting (score reported? number of cases abstracted)
# =====================================================================
t = pd.read_csv(f"{P}/cms_timely_panel.csv", dtype=str)
t = t[t["measure_id"] == "SEP_1"].copy()
t["measure_year"] = pd.to_numeric(t["measure_year"]).astype(int)
t = t.sort_values("end_date").drop_duplicates(["ccn", "measure_year"], keep="last")
t["sep1_reported"] = pd.to_numeric(t["score"], errors="coerce").notna().astype(int)
case_col = "sample" if "sample" in t.columns else ("denominator" if "denominator" in t.columns else None)
if case_col:
    t["sep1_cases"] = pd.to_numeric(t[case_col], errors="coerce")
    t["sep1_log_cases"] = np.log1p(t["sep1_cases"])
keep4 = ["ccn", "measure_year", "sep1_reported"] + (["sep1_log_cases"] if case_col else [])
p4 = panel[["id", "hosp_num", "ccn", "year", "arm", "event_type", "gvar"]].merge(
    t[keep4].rename(columns={"measure_year": "year"}), on=["ccn", "year"], how="inner")
p4.to_csv(f"{P}/ext4_reporting.csv", index=False)
log(f"EXT 4  reporting file: {len(p4):,} hospital-years; share reported with a score: "
    f"{p4['sep1_reported'].mean():.3f}; case-count column: {case_col or 'not found'}")

with open(f"{P}/extensions_summary.txt", "w") as f:
    f.write("\n".join(summary) + "\n")
print(f"\nSaved {P}/extensions_summary.txt")
