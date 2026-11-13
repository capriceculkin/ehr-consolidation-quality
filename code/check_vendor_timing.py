# check_vendor_timing.py
# Purpose: Check AHA vendor codes and Epic switch years against an independent
#          source: CMS Meaningful Use attestations (vendor of the certified EHR
#          each hospital attested with, program years 2011-2016).
# Author: Caprice Culkin
# Date: November 2026
# Input: datasets/ONC/MU_REPORT.csv, data/processed/analysis_panel.csv

import pandas as pd

cols = ["CCN", "Provider_Type", "Program_Year", "Vendor_Name"]
parts = []
for ch in pd.read_csv("datasets/ONC/MU_REPORT.csv", dtype=str, usecols=cols,
                      encoding="latin1", chunksize=400_000):
    parts.append(ch[ch["Provider_Type"].str.contains("Hospital", na=False)])
mu = pd.concat(parts)
mu["ccn"] = mu["CCN"].str.strip().str.zfill(6)
mu["year"] = pd.to_numeric(mu["Program_Year"], errors="coerce")
mu["mu_epic"] = mu["Vendor_Name"].str.contains("Epic", case=False, na=False)
g = mu.groupby(["ccn", "year"])["mu_epic"].any().reset_index()

p = pd.read_csv("data/processed/analysis_panel.csv", dtype={"ccn": str}, low_memory=False)
p = p.dropna(subset=["vendor", "ccn"])[["ccn", "year", "vendor"]]
p["aha_epic"] = p["vendor"].eq("epic")
m = p.merge(g, on=["ccn", "year"])
print(f"Matched hospital-years 2011-2016: {len(m):,}")
print(f"AHA says Epic -> MU agrees: {m.loc[m['aha_epic'], 'mu_epic'].mean():.1%}")
print(f"MU says Epic -> AHA agrees: {m.loc[m['mu_epic'], 'aha_epic'].mean():.1%}")


def first_switch(df, flag):
    out = {}
    for k, x in df.sort_values("year").groupby("ccn"):
        f, y = x[flag].to_numpy(), x["year"].to_numpy()
        for i in range(1, len(f)):
            if f[i] and not f[i - 1]:
                out[k] = y[i]
                break
    return pd.Series(out, dtype=float)


both = pd.concat([first_switch(p, "aha_epic").rename("aha"),
                  first_switch(g, "mu_epic").rename("mu")], axis=1).dropna()
both = both[both["aha"].between(2012, 2016)]
gap = both["aha"] - both["mu"]
print(f"\nHospitals switching to Epic in both sources (2012-2016): {len(both)}")
print("AHA switch year minus MU switch year:", gap.value_counts().sort_index().to_dict())
print(f"Same year: {(gap == 0).mean():.0%}; within one year: {(gap.abs() <= 1).mean():.0%}")
