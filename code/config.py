# config.py
# Settings shared by 02_merge_v2.py and check_cohorts.py.
# Fill in the vendor codes from the AHA IT Supplement codebook.

EPIC_CODES = {"5"}      # piemr code(s) for Epic, as strings, e.g. {"1"}
ORACLE_CODES = {"2"}    # piemr code(s) for Cerner / Oracle Health, as strings

STANDARDIZED_SHARE = 0.75   # share of other members on one vendor
MIN_MEMBERS = 2             # minimum other members with vendor data

# Event types (see 02_merge_v2.py). A system ID change counts as a whole-system
# move when at least MERGER_SHARE of the hospital's previous co-members moved
# to the same new system in the same year.
MERGER_SHARE = 0.5
# Event types used as treated hospitals in the main analysis
MAIN_EVENT_TYPES = {"independent_joined", "hospital_moved"}

IT_PATH = "data/raw/aha_it_survey.csv"
ANNUAL_PATH = "data/raw/aha_annual_survey.csv"

# Extension 3: how to read the IT Supplement exchange variables.
# Binary 0/1 variables are used as they are. For 1-5 scale variables, fill in
# the codebook meaning before they are used, e.g. for a frequency question:
#   "sehrsnd": {"yes": [1, 2], "no": [3, 4], "missing": [5]}
# Scale variables not listed here are left out of the analysis.
EXCHANGE_BINARY = ["vennet", "cwha", "spcareq"]
EXCHANGE_SCALES = {}
