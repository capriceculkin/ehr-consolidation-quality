# code/03_download_hvbp.py
# Purpose: Download hvbp_tps CSV files from NBER Hospital Compare archive
# Source: https://data.nber.org/data/cms/compare/hospital/

import urllib.request
import os

os.makedirs("datasets/CMS/archive", exist_ok=True)

year_month = {
    2013: "201310",
    2014: "201412",
    2015: "201512",
    2016: "201610",
    2017: "201710",
    2018: "201810",
    2019: "201910",
    2020: "202010",
    2021: "202110",
    2022: "202207",
    2023: "202310",
    2024: "202410",
    2025: "202511",
    2026: "202608",
}

for year, ym in year_month.items():
    url = f"https://data.nber.org/data/cms/compare/hospital/{ym}/hvbp_tps.csv"
    outfile = f"datasets/CMS/archive/hvbp_tps_{year}.csv"
    print(f"Downloading {year}...")
    try:
        urllib.request.urlretrieve(url, outfile)
        print(f"  Saved: {outfile}")
    except Exception as e:
        print(f"  Failed: {e}")
