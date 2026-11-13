# 05_stack_cms.py
# Purpose: Stack every CMS Hospital Compare / Care Compare release into one long
#          file per source, labeled by measurement year
# Author: Caprice Culkin
# Date: October 2026
#
# Put each yearly archive (unzipped) anywhere under datasets/CMS/, for example
#   datasets/CMS/archive/2018/Timely and Effective Care - Hospital.csv
#   datasets/CMS/archive/2023/Timely_and_Effective_Care-Hospital.csv
# The script searches all subfolders and recognizes old and new file names:
#   Timely and Effective Care - Hospital       (SEP-1, VTE, stroke)
#   Readmissions and Deaths - Hospital         (mortality + readmission, older releases)
#   Complications and Deaths - Hospital        (mortality, newer releases)
#   Unplanned Hospital Visits - Hospital       (readmission, newer releases)
# and old and new column names (Provider ID / Facility ID, Measure Start Date /
# Start Date, and so on).

import glob
import os
import re
import sys
import pandas as pd

ROOT = "datasets/CMS"

FILE_TYPES = {
    "timely": "timelyandeffectivecarehospital",
    "readm_deaths": "readmissionsanddeathshospital",
    "comp_deaths": "complicationsanddeathshospital",
    "unplanned": "unplannedhospitalvisitshospital",
}

COLUMN_ALIASES = {
    "provider_id": "facility_id",
    "providerid": "facility_id",
    "facilityid": "facility_id",
    "measureid": "measure_id",
    "measurename": "measure_name",
    "measure_start_date": "start_date",
    "measure_end_date": "end_date",
    "startdate": "start_date",
    "enddate": "end_date",
}

process_measures = ["SEP_1", "SEP_SH_3HR", "SEP_SH_6HR", "SEV_SEP_3HR", "SEV_SEP_6HR",
                    "VTE_1", "VTE_2", "STK_02", "STK_03", "STK_05",
                    "OP_18b", "OP_18c", "OP_22", "OP_23", "IMM_3", "SAFE_USE_OF_OPIOIDS"]
outcome_measures = ["MORT_30_AMI", "MORT_30_HF", "MORT_30_PN", "MORT_30_STK",
                    "READM_30_AMI", "READM_30_HF", "READM_30_PN", "Hybrid_HWR",
                    "READM_30_HOSP_WIDE",
                    "MORT_30_COPD", "MORT_30_CABG", "READM_30_COPD", "READM_30_CABG",
                    "READM_30_HIP_KNEE", "COMP_HIP_KNEE", "EDAC_30_HF", "EDAC_30_PN",
                    "PSI_90", "PSI_03", "PSI_06", "PSI_08", "PSI_09", "PSI_11", "PSI_12",
                    "PSI_13", "PSI_15"]


def file_type(path):
    key = re.sub(r"[^a-z]", "", os.path.basename(path).lower().replace(".csv", ""))
    for kind, pattern in FILE_TYPES.items():
        if key.startswith(pattern):
            return kind
    return None


def read_release(path, keep):
    df = pd.read_csv(path, dtype=str, encoding="latin1", low_memory=False)
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_"))
    df = df.rename(columns=COLUMN_ALIASES)
    missing = [c for c in ["facility_id", "measure_id", "score", "end_date"] if c not in df.columns]
    if missing:
        print(f"  SKIPPED {path}: missing columns {missing}. Columns in file: {list(df.columns)[:12]}")
        return None
    df = df[df["measure_id"].isin(keep)].copy()
    # Parse dates file by file: releases use different formats (12/31/2019 vs
    # 2019-12-31), and parsing a mixed column at once blanks one of them.
    for c in ["start_date", "end_date"]:
        if c in df.columns:
            raw = df[c]
            parsed = pd.to_datetime(raw, format="%m/%d/%Y", errors="coerce")
            if parsed.notna().mean() < 0.5:
                parsed = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce")
            if parsed.notna().mean() < 0.5:
                parsed = pd.to_datetime(raw, errors="coerce")
            if raw.notna().any() and parsed.notna().mean() < 0.9:
                print(f"  WARNING {path}: only {parsed.notna().mean():.0%} of {c} parsed "
                      f"(example value: {raw.dropna().iloc[0]!r})")
            df[c] = parsed
    df["source_file"] = path
    return df


files = sorted(glob.glob(f"{ROOT}/**/*.csv", recursive=True))
by_kind = {k: [] for k in FILE_TYPES}
for f in files:
    k = file_type(f)
    if k:
        by_kind[k].append(f)

print("Releases found:")
for k, fs in by_kind.items():
    print(f"  {k}: {len(fs)} file(s)")
if not any(by_kind.values()):
    sys.exit(f"STOP: no Care Compare files found under {ROOT}/")


def stack(kinds, keep, label):
    frames = []
    for k in kinds:
        for f in by_kind[k]:
            df = read_release(f, keep)
            if df is not None:
                frames.append(df)
    if not frames:
        print(f"{label}: no usable files")
        return None
    out = pd.concat(frames, ignore_index=True)
    # Some releases were saved with the ID as a number ("10001.0"), which drops
    # the leading zero and adds a decimal. Strip the ".0" before padding.
    ccn = out["facility_id"].astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
    numeric = ccn.str.fullmatch(r"\d+")
    out["ccn"] = ccn.where(~numeric, ccn.str.zfill(6))
    blank = out["ccn"].isin(["", "nan", "None", "<NA>"])
    if blank.any():
        print(f"  Dropping {blank.sum():,} rows with no facility ID")
        out = out[~blank]
    bad = ~out["ccn"].str.fullmatch(r"[0-9A-Z]{6}")
    if bad.any():
        print(f"  NOTE: {bad.sum():,} rows have a CCN that is not 6 characters, e.g. {out.loc[bad, 'ccn'].iloc[0]!r}")
    no_date = out["end_date"].isna()
    if no_date.any():
        print(f"  NOTE: {no_date.sum():,} rows have no usable end date and are dropped")
        out = out[~no_date]
    out["measure_year"] = out["end_date"].dt.year.astype(int)
    before = len(out)
    out = out.drop_duplicates(subset=["ccn", "measure_id", "end_date", "score"])
    print(f"\n{label}: {len(out):,} rows ({before - len(out):,} duplicates across releases removed)")
    print(out.groupby(["measure_id", "measure_year"]).size().unstack(fill_value=0).to_string())
    return out


os.makedirs("data/processed", exist_ok=True)
timely = stack(["timely"], process_measures, "Timely and Effective Care")
outcomes = stack(["readm_deaths", "comp_deaths", "unplanned"], outcome_measures,
                 "Mortality and readmission")

keep_cols = ["ccn", "measure_id", "measure_year", "start_date", "end_date", "score",
             "sample", "denominator", "footnote", "source_file"]
if timely is not None:
    timely[[c for c in keep_cols if c in timely.columns]].to_csv(
        "data/processed/cms_timely_panel.csv", index=False)
if outcomes is not None:
    o = outcomes[[c for c in keep_cols if c in outcomes.columns]]
    o.to_csv("data/processed/cms_complications_panel.csv", index=False)
    o.iloc[:0].to_csv("data/processed/cms_unplanned_panel.csv", index=False)
print("\nDone.")
