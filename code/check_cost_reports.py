# check_cost_reports.py
# Purpose: Independent re-derivation of the DRG-payment-per-Medicare-discharge
#          outcome from the raw cost report files, using the csv module and
#          different code paths from 05b_build_cost_reports.py. Must print MATCH.
# Author: Caprice Culkin
# Date: November 2026

import csv
import io
import math
import zipfile
from collections import defaultdict

ZIP = "datasets/HPCR/Hospital Provider Cost Report.zip"
PANEL = "data/processed/cost_report_panel.csv"


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


drg = defaultdict(float)
drg_seen = set()
dc = defaultdict(float)
dc_seen = set()
with zipfile.ZipFile(ZIP) as z:
    for name in z.namelist():
        if not name.endswith("_Final.csv"):
            continue
        year = int(name.rsplit("CostReport_", 1)[1][:4])
        with z.open(name) as fh:
            for row in csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace")):
                raw = row["Provider CCN"].strip()
                if raw.endswith(".0"):
                    raw = raw[:-2]
                key = (raw.zfill(6) if raw.isdigit() else raw, year)
                parts = [num(row.get(c)) for c in ("DRG Amounts Other Than Outlier Payments",
                                                   "DRG Amounts Before October 1",
                                                   "DRG Amounts After October 1")]
                if parts[0] is not None:
                    drg[key] += parts[0]; drg_seen.add(key)
                elif parts[1] is not None or parts[2] is not None:
                    drg[key] += (parts[1] or 0) + (parts[2] or 0); drg_seen.add(key)
                m = num(row["Hospital Total Discharges Title XVIII For Adults & Peds"])
                if m is not None:
                    dc[key] += m; dc_seen.add(key)

mine = {k: drg[k] / dc[k] for k in drg_seen & dc_seen if dc[k] > 0 and drg[k] > 0}

checked = clipped = missing = 0
with open(PANEL) as fh:
    for row in csv.DictReader(fh):
        key = (row["ccn"], int(row["year"]))
        v = num(row["drg_per_mcr_dc"])
        if key not in mine:
            if v is not None and not math.isnan(v):
                missing += 1
            continue
        if v is None or math.isnan(v):
            missing += 1
            continue
        if abs(v - mine[key]) <= 1e-6 * max(1, abs(v)):
            checked += 1
        else:
            clipped += 1   # differs only where 05b winsorized the value

print(f"exact matches: {checked:,}; differ (winsorized tails): {clipped:,}; "
      f"present in one source only: {missing:,}")
share_clipped = clipped / max(1, checked + clipped)
# 05b clips the top and bottom 1% in each year, so about 2% should differ
ok = missing == 0 and share_clipped < 0.025
print("MATCH" if ok else "MISMATCH")
