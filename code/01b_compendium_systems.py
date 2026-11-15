# 01b_compendium_systems.py
# Purpose: Build hospital-year system membership from the AHRQ Compendium of
#          U.S. Health Systems, in the same format 02_merge_v2.py expects
#          from the AHA Annual Survey
# Author: Caprice Culkin
# Date: October 2026
#
# Why: Yale's WRDS subscription includes the AHA IT Supplement but not the
# AHA Annual Survey (product aha_annual_survey_recent), which holds SYSID.
# The Compendium hospital linkage files give each hospital's CCN and its
# health system (health_sys_id) for 2016, 2018, 2020, 2021, 2022, 2023.
#
# Inputs:
#   data/raw/aha_it_survey.csv      (01_data_pull.py; gives the CCN -> AHA id crosswalk)
#   datasets/AHRQ/                  Compendium hospital linkage files, either
#                                   as .csv or inside the yearly .zip downloads
# Output:
#   data/raw/aha_annual_survey.csv  columns: id, mcrnum, year, sysid, sysname, bdtot
#
# Years with no Compendium file (2017, 2019) are simply absent. 02_merge_v2.py
# compares each year to the last observed year and records the gap, so an
# acquisition first seen in 2018 may have happened in 2017.

import glob
import os
import re
import sys
import zipfile
import pandas as pd

AHRQ_DIR = "datasets/AHRQ"
IT_PATH = "data/raw/aha_it_survey.csv"
OUT_PATH = "data/raw/aha_annual_survey.csv"


def year_from_name(name):
    m = re.search(r"(20\d\d)", name)
    return int(m.group(1)) if m else None


def is_hospital_linkage(name):
    n = name.lower()
    return n.endswith(".csv") and "hospital" in n and "linkage" in n and "group" not in n


# Find hospital linkage files (loose CSVs and inside zips)
frames = []
sources = []
for path in sorted(glob.glob(f"{AHRQ_DIR}/*")):
    if path.lower().endswith(".zip"):
        zyear = year_from_name(path.split("/")[-1])
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if is_hospital_linkage(name.split("/")[-1]):
                    df = pd.read_csv(z.open(name), dtype=str, encoding="latin1")
                    year = year_from_name(name.split("/")[-1]) or zyear
                    frames.append((year, df)); sources.append(f"{path}:{name}")
    elif is_hospital_linkage(path.split("/")[-1]):
        df = pd.read_csv(path, dtype=str, encoding="latin1")
        frames.append((year_from_name(path.split("/")[-1]), df)); sources.append(path)
if not frames:
    sys.exit("STOP: no Compendium hospital linkage files found in datasets/AHRQ/")
stacked = []
for (year, df), src in zip(frames, sources):
    df.columns = df.columns.str.lower().str.strip()
    if year is None or "ccn" not in df.columns or "health_sys_id" not in df.columns:
        sys.exit(f"STOP: cannot read year/ccn/health_sys_id from {src}")
    keep = df[["ccn", "health_sys_id"]].copy()
    keep["sysname"] = df.get("health_sys_name")
    keep["bdtot"] = df.get("hos_beds")
    keep["acutehosp_flag"] = df.get("acutehosp_flag")
    keep["year"] = year
    stacked.append(keep)
    print(f"  {year}: {len(keep):,} hospitals from {src}")
comp = pd.concat(stacked, ignore_index=True)
comp["ccn"] = comp["ccn"].str.strip().str.replace(r"\.0$", "", regex=True)
comp["ccn"] = comp["ccn"].where(~comp["ccn"].str.fullmatch(r"\d+").fillna(False),
                                comp["ccn"].str.zfill(6))
comp["health_sys_id"] = comp["health_sys_id"].str.strip().replace({"": pd.NA})
no_ccn = comp["ccn"].isna() | comp["ccn"].isin(["", "nan"])
print(f"Dropping {no_ccn.sum():,} Compendium rows with no CCN (they cannot be linked to CMS or the IT Supplement)")
comp = comp[~no_ccn]
if comp.duplicated(["ccn", "year"]).any():
    n = comp.duplicated(["ccn", "year"], keep=False).sum()
    print(f"NOTE: {n} rows share a CCN within a year. Keeping the first.")
    comp = comp.drop_duplicates(["ccn", "year"], keep="first")
years_found = sorted(comp["year"].unique())
print(f"\nCompendium years found: {years_found}")
for want in (2016, 2018, 2020, 2021, 2022, 2023):
    if want not in years_found:
        print(f"  MISSING {want}: download the {want} hospital linkage file from AHRQ "
              f"into datasets/AHRQ/")


# CCN -> AHA id crosswalk from the IT Supplement
it = pd.read_csv(IT_PATH, dtype=str)
it.columns = it.columns.str.lower()
it["id"] = it["id"].str.strip().str.replace(r"\.0$", "", regex=True)
it["ccn"] = it["mcrnum"].str.strip().str.replace(r"\.0$", "", regex=True)
it["ccn"] = it["ccn"].where(~it["ccn"].str.fullmatch(r"\d+").fillna(False), it["ccn"].str.zfill(6))
xw = it.dropna(subset=["id", "ccn"])[["ccn", "id"]].drop_duplicates()
multi_id = xw.groupby("ccn")["id"].nunique()
multi_ccn = xw.groupby("id")["ccn"].nunique()
print(f"\nCrosswalk: {len(xw):,} CCN-id pairs, "
      f"{(multi_id > 1).sum()} CCNs with more than one AHA id, "
      f"{(multi_ccn > 1).sum()} AHA ids with more than one CCN over time")


# A CCN shared by several AHA ids (multi-campus) cannot be assigned to one
# hospital, so those CCNs are dropped and listed.
ambiguous = multi_id[multi_id > 1].index
os.makedirs("data/processed", exist_ok=True)
xw[xw["ccn"].isin(ambiguous)].to_csv("data/processed/compendium_ambiguous_ccn.csv", index=False)
xw = xw[~xw["ccn"].isin(ambiguous)]

out = comp.merge(xw, on="ccn", how="inner", validate="many_to_many")
# An AHA id mapped to two CCNs in the same year: keep the CCN with a system
out = (out.sort_values(["id", "year", "health_sys_id"], na_position="last")
          .drop_duplicates(["id", "year"], keep="first"))

print(f"Compendium hospitals matched to an AHA id: "
      f"{out['ccn'].nunique():,} of {comp['ccn'].nunique():,} CCNs")
print("\nMatched hospitals per year (and share in a system):")
summary = out.groupby("year").agg(hospitals=("id", "nunique"),
                                  in_system=("health_sys_id", lambda s: round(s.notna().mean(), 3)))
print(summary.to_string())

out = out.rename(columns={"health_sys_id": "sysid", "ccn": "mcrnum"})
out[["id", "mcrnum", "year", "sysid", "sysname", "bdtot", "acutehosp_flag"]].to_csv(OUT_PATH, index=False)
print(f"\nSaved {OUT_PATH} ({len(out):,} hospital-years). "
      f"Next: python code/02_merge_v2.py")
