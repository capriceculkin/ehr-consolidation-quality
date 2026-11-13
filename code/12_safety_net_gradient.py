# 12_safety_net_gradient.py
# Purpose: Dose-response check for the safety-net result in 10_cea_psa.py
#          (--subset safety_net). Splits Epic-switch and comparison hospitals
#          into quartiles of a pre-treatment safety-net measure and estimates
#          mortality effects in each. A real effect should grow with the
#          measure, not appear only above one cutoff. Two definitions, from
#          different data, are used as independent checks:
#            medicaid_share  Medicaid share of inpatient days (AHA survey)
#            dsh_pct         Medicare DSH percentage (Medicare cost reports)
# Author: Caprice Culkin
# Date: November 2026

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from check_cs_estimates import cs_att   # noqa: E402

panel = pd.read_csv("data/processed/analysis_panel.csv", dtype={"id": str}, low_memory=False)
d = pd.read_csv("data/processed/ext1_switch_only.csv", dtype={"id": str}, low_memory=False)
rows = []
for measure in ["medicaid_share", "dsh_pct"]:
    ms = d[["id", "year", "gvar"]].merge(panel[["id", "year", measure]], on=["id", "year"], how="left")
    ms = ms[(ms["gvar"] == 0) | (ms["year"] < ms["gvar"])]
    h = ms.groupby("id")[measure].mean().dropna()
    q = pd.qcut(h.rank(method="first"), 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"])
    print(f"\n{measure}: quartile cutoffs {[round(x, 3) for x in h.quantile([.25, .5, .75]).tolist()]}")
    for y in ["mort_pn", "mort_hf", "mort_ami", "mort_copd"]:
        out = []
        for lab in q.cat.categories:
            r = cs_att(d[d["id"].isin(q.index[q == lab])], y, reps=300)
            rows.append(dict(measure=measure, outcome=y, quartile=lab, att=r["att"], se=r["se"],
                             max_abs_pre_t=r["max_abs_pre_t"], n_treated=r["n_treated"]))
            out.append(f"{lab}: {r['att']:+.2f} ({r['se']:.2f}) pre-t {r['max_abs_pre_t']:.1f}")
        print(f"  {y:9s} | " + " | ".join(out))
os.makedirs("output/tables", exist_ok=True)
pd.DataFrame(rows).to_csv("output/tables/safety_net_gradient.csv", index=False)
