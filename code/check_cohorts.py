# check_cohorts.py
# Purpose: Independent check of the acquisition cohorts built by 02_merge_v2.py
# Author: Caprice Culkin
# Date: October 2026
#
# This script rebuilds the acquisition events and treatment arms from the
# raw AHA files with completely separate code: the csv module and plain
# dictionaries, no pandas. It then compares its results hospital by
# hospital against data/processed/acquisition_events.csv.
#
# If both methods agree, the cohort counts are confirmed two ways.
# Any disagreement is printed and saved to data/processed/check_mismatches.csv.
#
# Run from the repo root after 02_merge_v2.py:
#   python code/check_cohorts.py

import csv
import sys
from collections import defaultdict, Counter

from config import (EPIC_CODES, ORACLE_CODES, STANDARDIZED_SHARE,
                    MIN_MEMBERS, IT_PATH, ANNUAL_PATH, MERGER_SHARE)


def norm(value):
    """Strip spaces and a trailing .0; empty means missing (None)."""
    if value is None:
        return None
    v = value.strip()
    if v.endswith(".0"):
        v = v[:-2]
    return v if v and v.lower() not in ("nan", "na", "<na>") else None


def read_rows(path, encoding):
    with open(path, newline="", encoding=encoding) as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [h.strip().lower() for h in reader.fieldnames]
        for row in reader:
            yield row


# -------------------------
# Raw system affiliation: sysid_by[id][year]
# -------------------------

sysid_by = defaultdict(dict)
for row in read_rows(ANNUAL_PATH, "latin1"):
    hid = norm(row["id"])
    if hid is None:
        continue
    year = int(float(row["year"]))
    if year in sysid_by[hid]:
        sys.exit(f"Duplicate annual survey row for id {hid}, year {year}")
    sysid_by[hid][year] = norm(row["sysid"])

# -------------------------
# Raw vendor: vendor_by[id][year]
# -------------------------

vendor_by = defaultdict(dict)
for row in read_rows(IT_PATH, "utf-8"):
    hid = norm(row["id"])
    if hid is None:
        continue
    year = int(float(row["year"]))
    code = norm(row["piemr"])
    if code is None:
        continue
    if code in EPIC_CODES:
        vendor_by[hid][year] = "epic"
    elif code in ORACLE_CODES:
        vendor_by[hid][year] = "oracle"
    else:
        vendor_by[hid][year] = "other"

# -------------------------
# Acquisition year: walk each hospital's observed years in order
# -------------------------

acq = {}
for hid, by_year in sysid_by.items():
    years = sorted(by_year)
    last_sysid = by_year[years[0]]
    found = None
    for y in years[1:]:
        current = by_year[y]
        if current is not None and current != last_sysid:
            found = (y, current, last_sysid)
            break
        last_sysid = current
    acq[hid] = found

# -------------------------
# Arm: vendor mix of the acquiring system's other members
# -------------------------

members_in = defaultdict(list)   # (sysid, year) -> list of hospital ids
for hid, by_year in sysid_by.items():
    for y, s in by_year.items():
        if s is not None:
            members_in[(s, y)].append(hid)


def system_vendor(sysid, year, exclude):
    vendors = [vendor_by[h][year] for h in members_in.get((sysid, year), [])
               if h != exclude and year in vendor_by[h]]
    if len(vendors) < MIN_MEMBERS:
        return None
    n = len(vendors)
    if vendors.count("epic") / n >= STANDARDIZED_SHARE:
        return "epic"
    if vendors.count("oracle") / n >= STANDARDIZED_SHARE:
        return "oracle"
    return "mixed"


def kind_of_event(hid, year, new, old):
    """Same rules as 02_merge_v2.py, written separately."""
    before_new = [h for h in members_in.get((new, year - 1), []) if h != hid]
    if len(before_new) == 0:
        return "new_system"
    if old is None:
        return "independent_joined"
    old_mates = [h for h in members_in.get((old, year - 1), []) if h != hid]
    if len(old_mates) > 0:
        now_in_new = set(members_in.get((new, year), []))
        moved = sum(1 for h in old_mates if h in now_in_new)
        if moved / len(old_mates) >= MERGER_SHARE:
            return "whole_system_acquired"
    return "hospital_moved"


arm_check = {}
for hid, event in acq.items():
    if event is None:
        arm_check[hid] = ("never_acquired", None, "none")
        continue
    year, sysid, old = event
    arm = system_vendor(sysid, year - 1, hid) or system_vendor(sysid, year, hid) or "unclassified"
    arm_check[hid] = (arm, year, kind_of_event(hid, year, sysid, old))

# -------------------------
# Compare with 02_merge_v2.py output
# -------------------------

main = {}
for row in read_rows("data/processed/acquisition_events.csv", "utf-8"):
    year = norm(row.get("acquisition_year"))
    main[norm(row["id"])] = (row["arm"], int(float(year)) if year else None, row["event_type"])

mismatches = []
for hid in sorted(set(main) | set(arm_check)):
    a, b = main.get(hid), arm_check.get(hid)
    if a != b:
        mismatches.append((hid, a, b))

print("Arms (independent check):")
for arm, n in sorted(Counter(v[0] for v in arm_check.values()).items()):
    print(f"  {arm}: {n}")
print("\nEvent types (independent check):")
for k, n in sorted(Counter(v[2] for v in arm_check.values()).items()):
    print(f"  {k}: {n}")
print("\nArms (02_merge_v2.py):")
for arm, n in sorted(Counter(v[0] for v in main.values()).items()):
    print(f"  {arm}: {n}")

if mismatches:
    with open("data/processed/check_mismatches.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "main_arm_year_type", "check_arm_year_type"])
        w.writerows(mismatches)
    print(f"\nMISMATCH: {len(mismatches)} hospitals differ. "
          f"See data/processed/check_mismatches.csv")
    sys.exit(1)

print(f"\nMATCH: all {len(main):,} hospitals have the same arm, acquisition year, "
      f"and event type in both methods.")
