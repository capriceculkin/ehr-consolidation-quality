# 01_data_pull.py
# Purpose: Pull raw AHA data from WRDS for thesis analysis
# Author: Caprice Culkin
# Date: September 2026
#
# This script connects to the WRDS PostgreSQL server and pulls two tables:
#   1. AHA IT Supplement (it_survey): hospital-level EHR vendor identity by year
#   2. AHA Annual Survey: hospital characteristics and system affiliation by year
#
# Both tables are saved as CSVs in data/raw/
# They are merged in 02_merge_v2.py on AHA id and year
#
# Requirements: pip install sqlalchemy psycopg2-binary pandas


# Load required libraries
import os
import pandas as pd
import sqlalchemy as sa
from getpass import getpass  # prompts for password without showing it on screen


# Connect to WRDS


# WRDS uses a PostgreSQL server
# Your WRDS username is read from the WRDS_USER environment variable, or asked for
# Password is entered interactively so it is never stored in this file

wrds_user = os.environ.get("WRDS_USER") or input("WRDS username: ")
wrds_pass = getpass("Enter WRDS password: ")

engine = sa.create_engine(
    f"postgresql+psycopg2://{wrds_user}:{wrds_pass}"
    f"@wrds-pgdata.wharton.upenn.edu:9737/wrds",
    connect_args={"sslmode": "require"}
)


# Pull AHA IT Supplement


# Variables pulled:
#   mcrnum  = Medicare Provider Number (primary linkage key to CMS data)
#   id      = AHA Hospital ID (linkage key to AHRQ Compendium)
#   year    = survey year
#   piemr   = primary inpatient EHR vendor code (our treatment variable)
#   piemro  = primary inpatient EHR vendor text for "other" category
#   mcntrl  = control/ownership from membership

it_query = """
    SELECT mcrnum, id, year, piemr, piemro, mcntrl
    FROM aha_it_survey_recent.it_survey
    WHERE year BETWEEN 2008 AND 2024
"""

print("Pulling AHA IT Supplement...")
it_survey = pd.read_sql(it_query, engine)
print(f"IT Survey rows: {len(it_survey)}")
os.makedirs("data/raw", exist_ok=True)
it_survey.to_csv("data/raw/aha_it_survey.csv", index=False)
print("Saved data/raw/aha_it_survey.csv")


# Pull AHA Annual Survey


# Variables pulled:
#   id      = AHA Hospital ID (join key to the IT Supplement in 02_merge_v2.py)
#   mcrnum  = Medicare Provider Number (CCN, used to link CMS outcomes)
#   year    = survey year
#   sysid   = health system ID (identifies acquisition into a system)
#   bdtot   = total staffed beds (covariate for hospital size)
#   mstate  = state (covariate)
#   mserv   = primary service type (covariate for hospital type)

# The WRDS table name for the Annual Survey differs by subscription, so
# instead of hard-coding it, search WRDS for AHA tables that have the
# columns we need. To force a specific table, set ANNUAL_TABLE below,
# for example ANNUAL_TABLE = "schema_name.table_name".
ANNUAL_TABLE = None

NEEDED = ["id", "year", "sysid"]
WANTED = ["id", "mcrnum", "year", "sysid", "bdtot", "mstate", "mserv"]

cols = pd.read_sql("""
    SELECT table_schema, table_name, lower(column_name) AS col
    FROM information_schema.columns
    WHERE position('aha' in lower(table_schema)) > 0
""", engine)

tables = cols.groupby(["table_schema", "table_name"])["col"].apply(set).reset_index()
tables["full"] = tables["table_schema"] + "." + tables["table_name"]
tables["has_needed"] = tables["col"].apply(lambda c: all(n in c for n in NEEDED))
tables["n_wanted"] = tables["col"].apply(lambda c: sum(w in c for w in WANTED))

print("\nAHA tables on WRDS with a sysid column:")
print(tables[tables["col"].apply(lambda c: "sysid" in c)][["full", "has_needed", "n_wanted"]]
      .to_string(index=False))

if ANNUAL_TABLE is None:
    candidates = tables[tables["has_needed"]].sort_values("n_wanted", ascending=False)
    if candidates.empty:
        raise SystemExit("\nSTOP: no AHA table on WRDS has id, year, and sysid. "
                         "Paste this output to check which table to use, or "
                         "set ANNUAL_TABLE by hand.")
    best = candidates["n_wanted"].max()
    top = candidates[candidates["n_wanted"] == best]
    # Compare the candidates' contents. Identical tables under different
    # schema names are mirrors, and either one can be used.
    summaries = []
    for full in top["full"]:
        q = (f"SELECT count(*) AS n_rows, count(DISTINCT id) AS n_hospitals, "
             f"min(year) AS first_year, max(year) AS last_year, "
             f"count(sysid) AS rows_with_sysid FROM {full}")
        row = pd.read_sql(q, engine).iloc[0].to_dict()
        row["table"] = full
        summaries.append(row)
    summary = pd.DataFrame(summaries).set_index("table")
    print("\nCandidate tables:")
    print(summary.to_string())
    if len(top) > 1 and len(summary.drop_duplicates()) > 1:
        raise SystemExit("\nSTOP: the candidate tables differ. Paste this output "
                         "and set ANNUAL_TABLE to the right one.")
    ANNUAL_TABLE = top["full"].iloc[0]
    if len(top) > 1:
        print("The candidates are identical, so they are the same data under "
              f"different names. Using {ANNUAL_TABLE}.")
    if "samp" in ANNUAL_TABLE.lower():
        print("\nWARNING: this table's schema name says 'sample'. Compare the "
              "hospital count above with the IT Supplement before relying on it.")

available = tables.loc[tables["full"] == ANNUAL_TABLE, "col"].iloc[0]
select_cols = [c for c in WANTED if c in available]
missing = [c for c in WANTED if c not in available]
print(f"\nUsing {ANNUAL_TABLE}")
if missing:
    print(f"  Columns not in this table (skipped): {missing}")

annual_query = f"""
    SELECT {", ".join(select_cols)}
    FROM {ANNUAL_TABLE}
    WHERE year BETWEEN 2008 AND 2024
"""

print("Pulling AHA Annual Survey...")
annual_survey = pd.read_sql(annual_query, engine)
print(f"Annual Survey rows: {len(annual_survey)}")
print("\nHospitals per year, Annual Survey vs IT Supplement:")
cover = pd.DataFrame({
    "annual_hospitals": annual_survey.groupby("year")["id"].nunique(),
    "annual_with_sysid": annual_survey.dropna(subset=["sysid"]).groupby("year")["id"].nunique(),
    "it_hospitals": it_survey.groupby("year")["id"].nunique(),
})
cover["it_ids_found_in_annual"] = [
    len(set(it_survey.loc[it_survey["year"] == y, "id"]) &
        set(annual_survey.loc[annual_survey["year"] == y, "id"]))
    for y in cover.index]
print(cover.fillna(0).astype(int).to_string())


# Save raw files locally


# Saving as CSV — raw unmodified pulls
# Do not edit these files directly
# All cleaning happens in 02_merge_v2.py

annual_survey.to_csv("data/raw/aha_annual_survey.csv", index=False)

print("Saved to data/raw/")


# Dispose connection
engine.dispose()
print("Done.")
