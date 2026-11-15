# 09_descriptives.py
# Purpose: Descriptive tables and plots for the prospectus section
#          "Descriptive Analysis and Visualization". Everything here is
#          unadjusted; no estimates.
# Author: Caprice Culkin
# Date: November 2026
#
# Run after 06_build_analysis_panel.py, from the repo root:
#     python code/09_descriptives.py
#
# Outputs (output/descriptives/):
#   1_timing_counts.csv, 1_timing.png        acquisitions by arm and Epic switches, by year
#   2_attrition.csv, 2_attrition.png         hospitals observed, entering, leaving, with SEP-1
#   2_treated_followup.csv                   years observed after acquisition, treated arms
#   3_characteristics.csv                    baseline characteristics by arm
#   4_trends_calendar.csv, .png              unadjusted outcome means by calendar year
#   4_trends_event.csv, .png                 unadjusted outcome means by years since acquisition

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import MAIN_EVENT_TYPES

P = "data/processed"
OUT = "output/descriptives"
os.makedirs(OUT, exist_ok=True)

TREATED_ARMS = ["epic", "oracle"]
OTHER = "Other (mixed, unclassified, mergers)"
OUTCOMES = {"sep1": "SEP-1 compliance (%)",
            "mort_ami": "30-day mortality, AMI (%)",
            "mort_hf": "30-day mortality, heart failure (%)",
            "mort_pn": "30-day mortality, pneumonia (%)"}

panel = pd.read_csv(f"{P}/analysis_panel.csv", dtype={"id": str, "ccn": str, "sysid": str},
                    low_memory=False)
panel["sysid"] = panel["sysid"].str.replace(r"\.0$", "", regex=True)
print(f"Panel: {len(panel):,} hospital-years, {panel['id'].nunique():,} hospitals, "
      f"{panel['year'].min()} to {panel['year'].max()}")
print("Rows by arm:\n" + panel["arm"].value_counts(dropna=False).to_string())

# One row per hospital with its arm and treatment year
hosp = panel.drop_duplicates("id").set_index("id")[["arm", "event_type", "acquisition_year", "gvar"]]
main_acq = hosp["arm"].isin(TREATED_ARMS) & hosp["event_type"].isin(MAIN_EVENT_TYPES)
hosp["group"] = np.select(
    [main_acq, hosp["arm"] == "never_acquired"],
    ["Acquired: " + hosp["arm"].astype(str), "Never acquired"], default=OTHER)
panel["group"] = panel["id"].map(hosp["group"])


# 1. Timing: acquisitions by year and arm, and Epic switches by year
acq = hosp[hosp["arm"].isin(TREATED_ARMS + ["mixed"]) & hosp["event_type"].isin(MAIN_EVENT_TYPES)
           & hosp["acquisition_year"].notna()]
acq_counts = (acq.groupby(["acquisition_year", "arm"]).size().rename("n").reset_index()
              .rename(columns={"acquisition_year": "year"}))
acq_counts["series"] = "Acquisition: " + acq_counts["arm"]


# Epic switch: first year the vendor is Epic after a non-Epic year, never in a
# hospital's first observed year (same rule as 07_extensions.py, EXT 1)
v = panel.dropna(subset=["vendor"]).sort_values(["id", "year"])[["id", "year", "vendor"]]
v["prev"] = v.groupby("id")["vendor"].shift(1)
sw = (v[(v["vendor"] == "epic") & v["prev"].notna() & (v["prev"] != "epic")]
      .groupby("id")["year"].min().rename("year").reset_index())
sw["series"] = np.where(sw["id"].map(hosp["group"]) == "Never acquired",
                        "Epic switch: never acquired", "Epic switch: acquired hospital")
sw_counts = sw.groupby(["year", "series"]).size().rename("n").reset_index()

timing = pd.concat([acq_counts[["year", "series", "n"]], sw_counts], ignore_index=True)
timing["year"] = timing["year"].astype(int)
timing = timing.sort_values(["series", "year"])
timing.to_csv(f"{OUT}/1_timing_counts.csv", index=False)

wide = timing.pivot(index="year", columns="series", values="n").fillna(0)
ax = wide.plot(kind="bar", figsize=(10, 5), width=0.85)
ax.set(xlabel="Year", ylabel="Hospitals", title="Acquisitions by arm and switches to Epic, by year")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
plt.tight_layout(); plt.savefig(f"{OUT}/1_timing.png", dpi=200); plt.close()


# 2. Attrition: hospitals observed, entering, and leaving each year
span = panel.groupby("id")["year"].agg(first="min", last="max")
yrs = sorted(panel["year"].unique())
rows = []
for y in yrs:
    rows.append({"year": y,
                 "observed": panel.loc[panel["year"] == y, "id"].nunique(),
                 "entering": int(((span["first"] == y) & (y > yrs[0])).sum()),
                 "leaving_after": int(((span["last"] == y) & (y < yrs[-1])).sum()),
                 "with_sep1": panel.loc[(panel["year"] == y) & panel["sep1"].notna(), "id"].nunique()
                 if "sep1" in panel else np.nan})
attr = pd.DataFrame(rows)
attr.to_csv(f"{OUT}/2_attrition.csv", index=False)
tr = span.join(hosp[["group", "gvar"]])
tr = tr[tr["group"].str.startswith("Acquired")].copy()
tr["years_after"] = tr["last"] - tr["gvar"]


# Hospitals acquired in the last two panel years cannot be followed for 2 years,
# so they are left out of the drop-out share (otherwise it just counts late acquisitions)
tr["can_follow_2y"] = tr["gvar"] <= yrs[-1] - 2
follow = tr.groupby("group").agg(hospitals=("years_after", "size"),
                                 median_years_after=("years_after", "median"),
                                 hospitals_followable_2y=("can_follow_2y", "sum"))
follow["share_gone_within_2y"] = (tr[tr["can_follow_2y"]].groupby("group")["years_after"]
                                  .apply(lambda s: round((s < 2).mean(), 3)))
follow.to_csv(f"{OUT}/2_treated_followup.csv")
print("\nTreated hospitals, years observed after acquisition:\n" + follow.to_string())

ax = attr.set_index("year").plot(marker="o", markersize=3, figsize=(10, 5))
ax.set(xlabel="Year", ylabel="Hospitals", title="Panel size, entry, exit, and SEP-1 coverage by year")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)
plt.tight_layout(); plt.savefig(f"{OUT}/2_attrition.png", dpi=200); plt.close()


# 3. Baseline characteristics by arm
#    Acquired: the year before acquisition. Comparison: first observed year.
print("\nRaw codes, check these before trusting the recodes below:")
for c in ["cbsatype", "cntrl"]:
    if c in panel:
        print(f"{c}:\n" + panel[c].value_counts(dropna=False).sort_index().to_string())
b = panel[panel["group"] != OTHER].copy()


# Acquired hospitals: the year before acquisition.
# Never-acquired hospitals: the year closest to the treated hospitals' median
# baseline year in which the AHA characteristics are reported. Using their first
# observed year instead (often 2008) pulls in years where ownership, CBSA type
# and system ID are missing, which made the comparison column look wrong.
acq_rows = b["group"].str.startswith("Acquired")
b.loc[acq_rows, "ref_year"] = b.loc[acq_rows, "gvar"] - 1
target = int(b.loc[acq_rows, "ref_year"].median())
has_chars = b["cntrl"].notna() if "cntrl" in b else pd.Series(True, index=b.index)
cand = b[~acq_rows & has_chars].copy()
cand["dist"] = (cand["year"] - target).abs()
pick = cand.sort_values(["id", "dist", "year"]).drop_duplicates("id").set_index("id")["year"]
b.loc[~acq_rows, "ref_year"] = b.loc[~acq_rows, "id"].map(pick)
print(f"\nBaseline year: acquired = year before acquisition; never acquired = closest year to {target} "
      f"with AHA characteristics ({pick.size:,} of {b.loc[~acq_rows, 'id'].nunique():,} hospitals have one)")
base = b[b["year"] == b["ref_year"]].copy()

# AHA cbsatype is "Metro" / "Micro" / "Rural" in most files; adjust if your codes differ
if "cbsatype" in base:
    base["rural"] = (base["cbsatype"].astype(str).str.strip().str.lower() == "rural").astype(float)
    base.loc[base["cbsatype"].isna(), "rural"] = np.nan
# AHA control codes: 12-16 nonfederal government, 21-23 nonprofit, 31-33 for-profit, 40s federal
if "cntrl" in base:
    c = pd.to_numeric(base["cntrl"], errors="coerce")
    base["ownership"] = np.select(
        [c.between(12, 16), c.between(21, 23), c.between(31, 33), c >= 40],
        ["Government (nonfederal)", "Nonprofit", "For-profit", "Federal"], default="Missing")
base["in_system"] = base["sysid"].notna() & (base["sysid"].astype(str).str.strip() != "")

def summarize(g):
    out = {"hospitals": len(g),
           "beds_mean": round(g["bdtot"].mean(), 1) if "bdtot" in g else np.nan,
           "beds_median": g["bdtot"].median() if "bdtot" in g else np.nan,
           "rural_pct": round(100 * g["rural"].mean(), 1) if "rural" in g else np.nan,
           "in_system_pct": round(100 * g["in_system"].mean(), 1),
           "medicare_share": round(g["medicare_share"].mean(), 3) if "medicare_share" in g else np.nan,
           "medicaid_share": round(g["medicaid_share"].mean(), 3) if "medicaid_share" in g else np.nan,
           "sep1_mean": round(g["sep1"].mean(), 1) if "sep1" in g else np.nan}
    if "ownership" in g:
        for k, val in (100 * g["ownership"].value_counts(normalize=True)).round(1).items():
            out[f"own_pct_{k}"] = val
    return pd.Series(out)

chars = pd.DataFrame({grp: summarize(g) for grp, g in base.groupby("group")}).T
own_cols = sorted(c for c in chars if c.startswith("own_pct_"))
chars[own_cols] = chars[own_cols].fillna(0)   # an ownership type absent from a group is 0%, not missing
first_cols = ["hospitals", "beds_mean", "beds_median", "rural_pct", "in_system_pct",
              "medicare_share", "medicaid_share", "sep1_mean"]
chars = chars[[c for c in first_cols if c in chars] + own_cols]
chars.index.name = "group"
# Acquired hospitals first seen in their acquisition year have no year-before row and drop out here
dropped = (b.groupby("group")["id"].nunique() - base.groupby("group")["id"].nunique()).fillna(0)
print("\nHospitals without a baseline row (first observed at or after acquisition):\n" + dropped.to_string())
chars.to_csv(f"{OUT}/3_characteristics.csv")
print("\nBaseline characteristics by arm:\n" + chars.to_string())


# 4. Unadjusted outcome trends
#    (a) by calendar year, acquired arms vs never acquired
#    (b) by years since acquisition, acquired arms only
outs = [o for o in OUTCOMES if o in panel]
long = (panel[panel["group"] != OTHER]
        .melt(id_vars=["id", "year", "group", "gvar"], value_vars=outs,
              var_name="outcome", value_name="value").dropna(subset=["value"]))
long["outcome"] = long["outcome"].map(OUTCOMES)
def facet_plot(df, x, title, xlabel, path, vline=None):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False)
    for ax, (name, d) in zip(axes.flat, df.groupby("outcome")):
        for grp, dd in d.groupby("group"):
            ax.plot(dd[x], dd["mean"], marker="o", markersize=3, label=grp)
        if vline is not None:
            ax.axvline(vline, ls="--", c="grey", lw=0.8)
        ax.set_title(name, fontsize=10); ax.set_xlabel(xlabel)
    for ax in axes.flat[len(df["outcome"].unique()):]:
        ax.axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(title)
    plt.tight_layout(rect=(0, 0.05, 1, 0.96)); plt.savefig(path, dpi=200); plt.close()
cal = (long.groupby(["group", "outcome", "year"])["value"]
       .agg(mean="mean", hospitals="size").reset_index())
cal.to_csv(f"{OUT}/4_trends_calendar.csv", index=False)
facet_plot(cal, "year", "Outcomes by calendar year: acquired hospitals vs never acquired (unadjusted)",
           "Year", f"{OUT}/4_trends_calendar.png")
ev = long[long["group"].str.startswith("Acquired") & (long["gvar"] > 0)].copy()
ev["event_time"] = ev["year"] - ev["gvar"]
ev = ev[ev["event_time"].between(-5, 5)]
ev = (ev.groupby(["group", "outcome", "event_time"])["value"]
      .agg(mean="mean", hospitals="size").reset_index())
ev.to_csv(f"{OUT}/4_trends_event.csv", index=False)
facet_plot(ev, "event_time", "Outcomes around acquisition, acquired hospitals only (unadjusted)",
           "Years since acquisition", f"{OUT}/4_trends_event.png", vline=-0.5)
print(f"\nDone. Files in {OUT}/")
