# 01c_pull_it_exchange.py
# Purpose: Pull AHA IT Supplement data-exchange variables for extension 3
#          (does joining an Epic system shift exchange toward the vendor network?)
# Author: Caprice Culkin
# Date: October 2026
#
# Output: data/raw/aha_it_exchange.csv  (id, year, and every variable below that exists)
# Variable years (AHA master file): vendor network and community HIE send/receive
# 2017-2023; HIE participation level 2014-2023; outside information use 2015-2023.

import os
import pandas as pd
import sqlalchemy as sa
from getpass import getpass

VARS = [
    "sehrsnd", "sehrrcv",      # single EHR vendor network (e.g. Care Everywhere): send / receive
    "emovbn", "vennet",        # vendor-based network for record location / exchange
    "hiosnd", "hiorcv",        # community HIE organization: send / receive
    "emoreg",                  # regional/state/local HIE for record location
    "rhio_lvl1",               # level of HIE participation
    "cwha", "spcareq",         # CommonWell, Carequality (national networks)
    "intconsnd", "intconrcv",  # HL7 interfaces across EHR systems
    "difvend",                 # barrier: exchanging across vendor platforms
    "ciaout", "phiout", "eqphios",  # outside information available / used / queryable
    "satisehr",                # satisfaction with primary inpatient EHR
]
TABLE = "aha_it_survey_recent.it_survey"

engine = sa.create_engine(
    f"postgresql+psycopg2://{os.environ.get('WRDS_USER') or input('WRDS username: ')}:{getpass('Enter WRDS password: ')}"
    f"@wrds-pgdata.wharton.upenn.edu:9737/wrds",
    connect_args={"sslmode": "require"})

schema, table = TABLE.split(".")
cols = pd.read_sql(
    f"SELECT lower(column_name) AS col FROM information_schema.columns "
    f"WHERE table_schema = '{schema}' AND table_name = '{table}'", engine)["col"].tolist()
present = [v for v in VARS if v in cols]
missing = [v for v in VARS if v not in cols]
print(f"Found {len(present)} of {len(VARS)} variables. Not in table: {missing}")

df = pd.read_sql(f"SELECT id, year, {', '.join(present)} FROM {TABLE} "
                 f"WHERE year BETWEEN 2008 AND 2024", engine)
os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/aha_it_exchange.csv", index=False)
print(f"Saved data/raw/aha_it_exchange.csv: {len(df):,} rows")
print("\nValues by variable (check coding against the AHA codebook):")
for v in present:
    print(f"  {v}: {df[v].value_counts(dropna=False).head(6).to_dict()}")
engine.dispose()
