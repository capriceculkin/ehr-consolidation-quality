# 02_merge_v2.py
# Purpose: Build the hospital-year panel and the acquisition treatment variable
# Author: Caprice Culkin
# Date: October 2026
#
# Replaces 02_merge.py. Changes from v1:
#   1. Merges AHA IT Supplement and AHA Annual Survey on AHA id + year
#      (v1 merged on mcrnum, where missing values matched each other)
#   2. Never converts missing Medicare numbers into a fake CCN
#   3. Fails loudly on unreadable rows instead of skipping them
#   4. Defines acquisition as a CHANGE in system affiliation (sysid),
#      not the first year a hospital appears in any system
#   5. Classifies each acquiring system as Epic-standardized or
#      Oracle-standardized from its member hospitals' vendors
#
# Inputs (from 01_data_pull.py, with id added to the annual pull):
#   data/raw/aha_it_survey.csv      columns: id, mcrnum, year, piemr, piemro
#   data/raw/aha_annual_survey.csv  columns: id, mcrnum, year, sysid, bdtot, mstate, mserv
#
# Outputs:
#   data/processed/panel_v2.csv           one row per hospital-year
#   data/processed/acquisition_events.csv one row per hospital
#   data/processed/cohort_counts.csv      counts by arm and acquisition year
#
# Run check_cohorts.py afterwards. It rebuilds the cohort counts with
# separate code and must agree with cohort_counts.csv exactly.

import sys
import pandas as pd
import numpy as np


# Settings live in config.py (shared with check_cohorts.py)
from config import (EPIC_CODES, ORACLE_CODES, STANDARDIZED_SHARE,
                    MIN_MEMBERS, IT_PATH, ANNUAL_PATH, MERGER_SHARE)
def clean_id(series):
    """Keep identifiers as strings. Drop a trailing .0 left by float reads."""
    s = series.astype("string").str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    return s.replace({"": pd.NA, "nan": pd.NA, "<NA>": pd.NA})
def clean_ccn(series):
    """Medicare number to 6-character CCN. Missing stays missing."""
    s = clean_id(series)
    numeric = s.str.fullmatch(r"\d+").fillna(False)
    s = s.where(~numeric, s.str.zfill(6))
    return s


# Load
it = pd.read_csv(IT_PATH, dtype=str, on_bad_lines="error")
annual = pd.read_csv(ANNUAL_PATH, dtype=str, encoding="latin1", on_bad_lines="error")
it.columns = it.columns.str.lower().str.strip()
annual.columns = annual.columns.str.lower().str.strip()
for name, df, needed in [("IT", it, ["id", "year", "piemr"]),
                         ("Annual", annual, ["id", "year", "sysid"])]:
    missing = [c for c in needed if c not in df.columns]
    if missing:
        sys.exit(f"{name} file is missing columns {missing}. "
                 f"Add them to the WRDS pull in 01_data_pull.py.")
for df in (it, annual):
    df["id"] = clean_id(df["id"])
    df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
    if "mcrnum" in df.columns:
        df["ccn"] = clean_ccn(df["mcrnum"])
annual["sysid"] = clean_id(annual["sysid"])
it["piemr"] = clean_id(it["piemr"])
print(f"IT rows: {len(it):,}   Annual rows: {len(annual):,}")


# Vendor codes present (fill EPIC_CODES / ORACLE_CODES from these)
print("\npiemr codes found, with example text from piemro:")
code_table = (it.groupby("piemr", dropna=False)
                .agg(n=("id", "size"),
                     example=("piemro", lambda x: x.dropna().iloc[0] if x.notna().any() else ""))
                if "piemro" in it.columns else it.groupby("piemr", dropna=False).size())
print(code_table)
if not EPIC_CODES or not ORACLE_CODES:
    sys.exit("\nSTOP: fill in EPIC_CODES and ORACLE_CODES at the top of this "
             "script from the AHA codebook, then rerun.")


# Uniqueness checks on the merge key
for name, df in (("IT", it), ("Annual", annual)):
    n_missing = df["id"].isna().sum()
    dups = df[df["id"].notna()].duplicated(["id", "year"], keep=False)
    print(f"{name}: {n_missing} rows with no AHA id, "
          f"{dups.sum()} rows in duplicated id-year pairs")
    if dups.any():
        df[dups].sort_values(["id", "year"]).to_csv(
            f"data/processed/duplicates_{name.lower()}.csv", index=False)
        sys.exit(f"STOP: duplicated id-year pairs in {name}. "
                 f"See data/processed/duplicates_{name.lower()}.csv")
it = it[it["id"].notna()]
annual = annual[annual["id"].notna()]


# Merge on AHA id + year


# Keep every Annual Survey column (covariates included). Columns that also
# exist in the IT Supplement get an _annual suffix in the merge.
annual_cols = [c for c in annual.columns if c != "mcrnum"]
panel = pd.merge(it, annual[annual_cols], on=["id", "year"], how="outer",
                 validate="one_to_one", indicator=True, suffixes=("_it", "_annual"))

print("\nMerge result (rows):")
print(panel["_merge"].value_counts().to_string())
rows_before = len(panel)

panel["vendor"] = np.select(
    [panel["piemr"].isin(EPIC_CODES), panel["piemr"].isin(ORACLE_CODES), panel["piemr"].notna()],
    ["epic", "oracle", "other"], default=pd.NA)


# Acquisition events from changes in sysid


# Rules, applied to each hospital's years in order, using Annual Survey
# years only (sysid comes from there):
#   - The first observed year is never an acquisition (left censoring).
#   - An acquisition happens in year t when the hospital has a sysid in t
#     and its last observed sysid before t was missing or different.
#   - Years with a missing survey are skipped; the comparison is to the
#     last observed year, and the size of the gap is recorded.
#   - Leaving a system (sysid to missing) is recorded but not counted.

sys_obs = (panel[panel["_merge"] != "left_only"][["id", "year", "sysid"]]
           .sort_values(["id", "year"]).copy())
sys_obs["prev_sysid"] = sys_obs.groupby("id")["sysid"].shift(1)
sys_obs["prev_year"] = sys_obs.groupby("id")["year"].shift(1)
sys_obs["is_first"] = sys_obs["prev_year"].isna()

changed = sys_obs["sysid"].notna() & (
    sys_obs["prev_sysid"].isna() | (sys_obs["sysid"] != sys_obs["prev_sysid"]))
sys_obs["acquired"] = changed & ~sys_obs["is_first"]
sys_obs["gap"] = sys_obs["year"] - sys_obs["prev_year"]

events = sys_obs[sys_obs["acquired"]].sort_values(["id", "year"])
# Take the FIRST event row as a whole. (groupby().first() would skip blank
# values column by column and could mix the previous system of a later event
# into the first event.)
first_event = (events.groupby("id", as_index=False).head(1)
               [["id", "year", "sysid", "prev_sysid", "gap"]]
               .rename(columns={"year": "acquisition_year", "sysid": "acquiring_sysid",
                                "prev_sysid": "previous_sysid", "gap": "gap_before_event"}))
first_event = first_event.merge(events.groupby("id").size().rename("n_acquisitions").reset_index(),
                                on="id", how="left", validate="one_to_one")


# Event type: acquisition, or a system-level merger / new system ID?


# Compares system membership in the year before the event (t-1) with year t:
#   new_system            the new sysid had no other members in t-1
#                         (system formed by merger, spin-off, or re-coded ID)
#   whole_system_acquired the hospital was in a system in t-1, and at least
#                         MERGER_SHARE of that system's other members moved to
#                         the same new sysid in t (one system absorbed another)
#   hospital_moved        the hospital moved from one system to an existing one
#   independent_joined    an independent hospital joined an existing system

members_by = {}
for hid, yr, sid in sys_obs[["id", "year", "sysid"]].itertuples(index=False):
    if pd.notna(sid):
        members_by.setdefault((sid, yr), set()).add(hid)


def event_type(row):
    t, new, old, hid = row["acquisition_year"], row["acquiring_sysid"], row["previous_sysid"], row["id"]
    if not (members_by.get((new, t - 1), set()) - {hid}):
        return "new_system"
    if pd.notna(old):
        old_mates = members_by.get((old, t - 1), set()) - {hid}
        if old_mates:
            moved = old_mates & members_by.get((new, t), set())
            if len(moved) / len(old_mates) >= MERGER_SHARE:
                return "whole_system_acquired"
        return "hospital_moved"
    return "independent_joined"


first_event["event_type"] = first_event.apply(event_type, axis=1)


# Classify the acquiring system by its OTHER members' vendors


# Uses the members already in the system the year BEFORE the acquisition,
# which reflects the system's established EHR standard and is not diluted
# by other hospitals acquired at the same time. Falls back to the
# acquisition year if the prior year has too few members with vendor data.

members = panel[panel["sysid"].notna() & panel["vendor"].notna()][["id", "year", "sysid", "vendor"]]


def classify(row, year):
    m = members[(members["sysid"] == row["acquiring_sysid"]) &
                (members["year"] == year) &
                (members["id"] != row["id"])]
    if len(m) < MIN_MEMBERS:
        return None
    share = m["vendor"].value_counts(normalize=True)
    if share.get("epic", 0) >= STANDARDIZED_SHARE:
        return "epic"
    if share.get("oracle", 0) >= STANDARDIZED_SHARE:
        return "oracle"
    return "mixed"


arms = []
for _, row in first_event.iterrows():
    arm = classify(row, row["acquisition_year"] - 1)
    if arm is None:
        arm = classify(row, row["acquisition_year"])
    arms.append(arm if arm is not None else "unclassified")
first_event["arm"] = arms


# Hospital-level file, including never-acquired hospitals
hospitals = (sys_obs.groupby("id")
             .agg(first_year=("year", "min"), last_year=("year", "max"),
                  in_system_at_first_obs=("sysid", lambda s: s.iloc[0] is not pd.NA and pd.notna(s.iloc[0])))
             .reset_index())
hospitals = hospitals.merge(first_event, on="id", how="left", validate="one_to_one")
hospitals["arm"] = hospitals["arm"].fillna("never_acquired")
hospitals["event_type"] = hospitals["event_type"].fillna("none")
hospitals.to_csv("data/processed/acquisition_events.csv", index=False)
panel = panel.merge(hospitals[["id", "acquisition_year", "arm", "event_type"]], on="id",
                    how="left", validate="many_to_one")
assert len(panel) == rows_before, "Row count changed after attaching treatment"
assert not panel.duplicated(["id", "year"]).any(), "Duplicate hospital-years in final panel"
panel.drop(columns="_merge").to_csv("data/processed/panel_v2.csv", index=False)
counts = (hospitals.groupby(["arm", "acquisition_year"], dropna=False)
          .size().rename("hospitals").reset_index())
counts.to_csv("data/processed/cohort_counts.csv", index=False)
print("\nHospitals by arm:")
print(hospitals["arm"].value_counts().to_string())
print("\nAcquired hospitals by event type and arm:")
print(pd.crosstab(hospitals.loc[hospitals["arm"] != "never_acquired", "event_type"],
                  hospitals.loc[hospitals["arm"] != "never_acquired", "arm"], margins=True).to_string())
print("\nNever-acquired hospitals already in a system at first observation:",
      int(hospitals.loc[hospitals["arm"] == "never_acquired", "in_system_at_first_obs"].sum()))
print("Hospitals with more than one acquisition:",
      int((hospitals["n_acquisitions"] > 1).sum()))
print("Acquisitions that came after a survey gap of 2+ years:",
      int((hospitals["gap_before_event"] >= 2).sum()))
print(f"\nFinal panel: {len(panel):,} hospital-years, {panel['id'].nunique():,} hospitals")
print("Saved panel_v2.csv, acquisition_events.csv, cohort_counts.csv")
