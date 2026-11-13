# 05c_stack_hvbp.py
# Purpose: Stack Hospital Value-Based Purchasing Total Performance Score (TPS)
#          files into one hospital x program-year panel
# Author: Caprice Culkin
# Date: November 2026
#
# Input:  every hvbp_tps*.csv under datasets/CMS/ (duplicates are fine; files
#         with identical content are read once)
# Output: data/processed/hvbp_panel.csv with ccn, fiscal_year, perf_year, tps
#
# Program year: taken from the file's fiscal year column when present. Older
# files have none, so the year comes from the release date in the file name:
# a file released in late year Y (e.g. hvbp_tps_11_10_2016) reports the
# FY Y+1 program. Checked for the November 2016 file: its clinical process
# domain carries a 5% weight (weighted / unweighted score), which is the
# FY2017 program's weight. An earlier panel labeled these files one year
# early and counted the 2019 release twice; this script replaces it.
#
# perf_year = fiscal_year - 2, the year most of the performance period falls
# in. TPS is joined to the analysis panel on perf_year. Mortality domains use
# multi-year windows, so this is an approximation to note in the thesis.
# CMS did not score hospitals for FY2022 and FY2023 (COVID), so those years
# have no TPS.

import glob
import hashlib
import os
import re

import pandas as pd

files = sorted(f for f in glob.glob("datasets/CMS/**/hvbp_tps*.csv", recursive=True)
               if "api" not in os.path.basename(f).lower())
seen, rows = set(), []
for f in files:
    digest = hashlib.md5(open(f, "rb").read()).hexdigest()
    if digest in seen:
        continue
    seen.add(digest)
    d = pd.read_csv(f, dtype=str, low_memory=False)
    d.columns = [re.sub(r"[^a-z]", "", c.lower()) for c in d.columns]
    if "fiscalyear" in d.columns:
        fy = pd.to_numeric(d["fiscalyear"], errors="coerce")
        source = "column"
    else:
        m = re.search(r"(\d{2})_(\d{2})_(\d{4})", os.path.basename(f))
        if not m:
            print(f"  SKIPPED {f}: no fiscal year column and no release date in name")
            continue
        fy = pd.Series(int(m.group(3)) + 1, index=d.index)
        source = "file name"
    id_col = next(c for c in ["facilityid", "providernumber"] if c in d.columns)
    ccn = d[id_col].str.strip().str.replace(r"\.0+$", "", regex=True)
    ccn = ccn.where(~ccn.str.fullmatch(r"\d+"), ccn.str.zfill(6))
    tps = pd.to_numeric(d.get("totalperformancescore"), errors="coerce")
    print(f"{os.path.basename(f):34s} FY {sorted(fy.dropna().unique().astype(int))} "
          f"({source}), {tps.notna().sum():,} scored")
    rows.append(pd.DataFrame({"ccn": ccn, "fiscal_year": fy, "tps": tps, "file": f}))

h = pd.concat(rows, ignore_index=True).dropna(subset=["fiscal_year"])
h["fiscal_year"] = h["fiscal_year"].astype(int)
# Same hospital-year in two releases: they should agree; report and keep one
g = h.dropna(subset=["tps"]).groupby(["ccn", "fiscal_year"])["tps"]
conflicts = (g.max() - g.min()).gt(0.01).sum()
print(f"\nHospital-years scored in more than one file with different TPS: {conflicts}")
h = (h.sort_values(["ccn", "fiscal_year", "tps"], na_position="last")
      .drop_duplicates(["ccn", "fiscal_year"]))
h["perf_year"] = h["fiscal_year"] - 2
h[["ccn", "fiscal_year", "perf_year", "tps"]].to_csv("data/processed/hvbp_panel.csv", index=False)
print(h.groupby("fiscal_year")["tps"].agg(["size", "count", "mean"]).round(2).to_string())
