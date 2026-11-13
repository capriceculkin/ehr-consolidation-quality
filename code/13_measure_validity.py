# 13_measure_validity.py
# Purpose: Does the measure Medicare scores (SEP-1) track patient outcomes
#          within a hospital? Year-to-year changes in SEP-1 are compared with
#          changes in 30-day pneumonia mortality (the pneumonia cohort includes
#          patients with sepsis since 2016) and heart failure mortality as a
#          comparison condition. Reported for never-acquired hospitals, and for
#          acquired hospitals before and after acquisition, to see whether the
#          link between measure and outcome weakens after an ownership change.
# Author: Caprice Culkin
# Date: November 2026
# Output: output/tables/measure_validity.csv

import os
import numpy as np
import pandas as pd

p = pd.read_csv("data/processed/analysis_panel.csv", dtype={"id": str}, low_memory=False)
p = p[p["arm"].isin(["epic", "oracle", "mixed", "never_acquired"])].sort_values(["id", "year"])
for c in ["sep1", "mort_pn", "mort_hf"]:
    p[f"d_{c}"] = p.groupby("id")[c].diff()
p["gap"] = p.groupby("id")["year"].diff()
p = p[p["gap"] == 1]
acq = p["acquisition_year"].notna() & p["event_type"].isin(["independent_joined", "hospital_moved"])
p["group"] = np.select([p["arm"].eq("never_acquired"),
                        acq & (p["year"] < p["acquisition_year"]),
                        acq & (p["year"] >= p["acquisition_year"])],
                       ["never acquired", "acquired, before", "acquired, after"], default="other")

rng = np.random.default_rng(20261101)
rows = []
for grp in ["never acquired", "acquired, before", "acquired, after"]:
    g = p[p["group"] == grp]
    for out in ["d_mort_pn", "d_mort_hf"]:
        x = g[["id", "d_sep1", out]].dropna()
        if len(x) < 30:
            continue
        # slope of mortality change on SEP-1 change (pp mortality per 10 pp SEP-1)
        def slope(df):
            a, b = df["d_sep1"].to_numpy(), df[out].to_numpy()
            a = a - a.mean()
            return 10 * (a * (b - b.mean())).sum() / (a ** 2).sum()
        est = slope(x)
        ids = x["id"].unique()
        grp_rows = {k: v for k, v in x.groupby("id")}
        boots = []
        for _ in range(500):   # hospital-clustered bootstrap
            pick = rng.choice(ids, len(ids))
            boots.append(slope(pd.concat([grp_rows[i] for i in pick])))
        rows.append(dict(group=grp, outcome=out.replace("d_", ""), n_hospital_years=len(x),
                         n_hospitals=len(ids), corr=x["d_sep1"].corr(x[out]),
                         mort_change_per_10pt_sep1=est, se=np.std(boots)))
r = pd.DataFrame(rows)
os.makedirs("output/tables", exist_ok=True)
r.to_csv("output/tables/measure_validity.csv", index=False)
print(r.round(4).to_string(index=False))
