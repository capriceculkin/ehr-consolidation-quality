# 11_contagion.py
# Purpose: Network contagion in EHR choice. Among hospitals not yet on Epic,
#          does the chance of switching to Epic this year rise with the share
#          of OTHER hospitals in the same market (hospital referral region)
#          already on Epic last year, separately from the share of the
#          hospital's own health system on Epic?
# Author: Caprice Culkin
# Date: November 2026
#
# Model: linear probability model of switching (discrete-time hazard),
#   switch_it = b1 * market_epic_share_{t-1} + b2 * system_epic_share_{t-1}
#               + b3 * in_system + HRR fixed effects + year fixed effects + e
# Market share excludes the hospital itself and members of its own system.
# HRR fixed effects compare the same market over time, so b1 is not just
# "Epic markets are different". Standard errors are clustered by HRR.
# Caveat: common shocks within a market (a regional HIE push, a dominant
# academic center) can still look like contagion; b1 is an association.
#
# Output: output/tables/contagion.csv

import os
import numpy as np
import pandas as pd

p = pd.read_csv("data/processed/analysis_panel.csv", dtype={"id": str, "sysid": str},
                low_memory=False)[["id", "year", "vendor", "hrrcode", "sysid"]]
p = p.dropna(subset=["vendor", "hrrcode"]).copy()
p["epic"] = (p["vendor"] == "epic").astype(float)
p["sysid"] = p["sysid"].str.replace(r"\.0$", "", regex=True)

# Epic share among other hospitals in the market, excluding own system
mk = (p.groupby(["hrrcode", "year"])["epic"].agg(["sum", "count"])
        .rename(columns={"sum": "m_sum", "count": "m_n"}))
sy = (p.dropna(subset=["sysid"]).groupby(["hrrcode", "sysid", "year"])["epic"]
        .agg(["sum", "count"]).rename(columns={"sum": "s_sum", "count": "s_n"}))
q = p.join(mk, on=["hrrcode", "year"]).join(sy, on=["hrrcode", "sysid", "year"])
q[["s_sum", "s_n"]] = q[["s_sum", "s_n"]].fillna(0)
own_sum = np.where(q["sysid"].notna(), q["s_sum"], q["epic"])
own_n = np.where(q["sysid"].notna(), q["s_n"], 1)
denom = (q["m_n"] - own_n).astype(float)
q["market_share"] = np.where(denom >= 2, (q["m_sum"] - own_sum) / denom.where(denom > 0), np.nan)
# Epic share among the other members of the hospital's own system (any market)
ss = (p.dropna(subset=["sysid"]).groupby(["sysid", "year"])["epic"].agg(["sum", "count"])
        .rename(columns={"sum": "ss", "count": "sn"}))
q = q.join(ss, on=["sysid", "year"])
q["system_share"] = np.where(q["sn"] > 1, (q["ss"] - q["epic"]) / (q["sn"] - 1), 0.0)
q["in_system"] = q["sysid"].notna().astype(float)

q = q.sort_values(["id", "year"])
for c in ["epic", "market_share", "system_share", "in_system"]:
    q[f"{c}_lag"] = q.groupby("id")[c].shift(1)
q["lag_year"] = q.groupby("id")["year"].shift(1)
risk = q[(q["epic_lag"] == 0) & (q["lag_year"] == q["year"] - 1)].dropna(
    subset=["market_share_lag", "system_share_lag"]).copy()
risk["switch"] = risk["epic"]
print(f"Risk set: {len(risk):,} hospital-years, {risk['id'].nunique():,} hospitals, "
      f"{int(risk['switch'].sum()):,} switches to Epic")


def lpm(df, xcols, fe=("hrrcode", "year")):
    """OLS with fixed effects (absorbed by alternating demeaning), standard
    errors clustered by HRR."""
    gh = df["hrrcode"].to_numpy()
    groups = [df[f].to_numpy() for f in fe]
    Z = df[["switch"] + xcols].to_numpy(float)
    for _ in range(200):
        Z0 = Z.copy()
        for g in groups:
            Z = Z - pd.DataFrame(Z).groupby(g).transform("mean").to_numpy()
        if np.abs(Z - Z0).max() < 1e-12:
            break
    yd, Xd = Z[:, 0], Z[:, 1:]
    b = np.linalg.lstsq(Xd, yd, rcond=None)[0]
    e = yd - Xd @ b
    A = np.linalg.inv(Xd.T @ Xd)
    meat = np.zeros((len(b), len(b)))
    for g in np.unique(gh):
        s = Xd[gh == g].T @ e[gh == g]
        meat += np.outer(s, s)
    G = len(np.unique(gh))
    return b, np.sqrt(np.diag(A @ meat @ A * G / (G - 1)))


# Within-market (HRR fixed effects) estimates in a hazard are biased toward
# a negative market effect: as share rises, the hospitals most inclined to
# switch have already left the risk set, leaving holdouts. So the model is
# also run with year effects only and with state + year effects, which
# compare across markets instead.
st = pd.read_csv("data/processed/analysis_panel.csv", dtype={"id": str}, usecols=["id", "year", "mstate"])
risk = risk.merge(st, on=["id", "year"], how="left")
risk["mstate"] = risk["mstate"].fillna("NA")
rows = []
base = risk["switch"].mean()
XS = ["market_share_lag", "system_share_lag", "in_system_lag"]
for label, fe in [("HRR + year FE", ("hrrcode", "year")),
                  ("state + year FE", ("mstate", "year")),
                  ("year FE only", ("year",))]:
    b, se = lpm(risk, XS, fe)
    for x, bb, s in zip(XS, b, se):
        rows.append(dict(model=label, term=x, coef=bb, se=s, ci_low=bb - 1.96 * s, ci_high=bb + 1.96 * s))
out = pd.DataFrame(rows)
os.makedirs("output/tables", exist_ok=True)
out.to_csv("output/tables/contagion.csv", index=False)
print(f"Baseline annual probability of switching to Epic: {base:.4f}")
print(out.round(4).to_string(index=False))
for model in out["model"].unique():
    o = out[out["model"] == model].set_index("term")["coef"]
    print(f"{model:16s}: +10 pts rivals' Epic share -> {o['market_share_lag'] * 0.1:+.4f}/yr; "
          f"+10 pts own system's Epic share -> {o['system_share_lag'] * 0.1:+.4f}/yr "
          f"(baseline {base:.4f})")
