# 10_cea_psa.py
# Purpose: Cost-effectiveness analysis of (a) switching to Epic and (b) being
#          acquired into an Epic-standardized system, from three perspectives
#          (society, Medicare, hospital), with a probabilistic sensitivity
#          analysis built directly on the quasi-experimental estimates.
# Author: Caprice Culkin
# Date: November 2026
#
# What is new here: every input effect (costs, Medicare payments, VBP score,
# mortality, readmission, SEP-1) is a Callaway-Sant'Anna estimate, and all of
# them are re-estimated on the SAME bootstrap resamples of hospitals. Each
# draw is a full, internally consistent ledger, so the correlation between
# cost and health effects (same hospitals, same years) carries into the PSA.
#
# The ledger, per treated hospital-year:
#   resource cost      (exp(ATT log cost per discharge) - 1) x baseline total costs
#                      (sensitivity: log total costs, which includes volume growth)
#   Medicare DRG       (exp(ATT log DRG per Medicare discharge) - 1) x baseline DRG payments
#   Medicare VBP       ATT VBP score x (2% withhold x slope / 100) x baseline DRG payments
#   deaths averted     -sum over 5 conditions of ATT mortality (pp) / 100 x patients per year
#   readmissions       -sum over 5 conditions of ATT readmission (pp) / 100 x patients per year
#   QALYs              deaths averted x discounted life-years x quality weight
#   hospital net       ATT operating margin x baseline net patient revenue
# Perspectives:
#   society   cost = resource cost - readmission costs avoided; effect = QALYs
#   Medicare  cost = DRG + VBP payments - readmission payments avoided; effect = QALYs
#   hospital  net financial gain (reported, not a cost-effectiveness ratio)
#   measured quality: society cost per SEP-1 point, the unit Medicare scores
#
# Parameters are ranges sampled uniformly in the PSA; sources are next to each. Deaths and readmissions count only the
# five conditions CMS measures, so health effects are understated hospital-wide.
#
# Usage: python code/10_cea_psa.py [--design epic_switch|epic_acquisition] [--detrend]
#        [--subset safety_net|other]
# Output: output/tables/cea_<design>[_detrend]_summary.csv, _draws.csv,
#         output/figures/cea_<design>[_detrend]_plane.png, _ceac.png

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from check_cs_estimates import cs_att, detrend   # noqa: E402

REPS = 1000
SEED = 20261101
# ---- parameters (uniform ranges for the PSA) ----
# Life-years per death averted: upper end near remaining life expectancy at
# age 80 (8.1 men, 9.5 women; SSA 2022 period life table, 2025 Trustees
# Report); lower end reflects 32-41% one-year mortality after AMI, heart
# failure or pneumonia admission in Medicare (Prescott & Chang, BMC Geriatr 2018).
LIFE_YEARS = (2, 8)
# Quality weight: EQ-5D index scores for chronic conditions in U.S. adults
# (Sullivan & Ghushchyan, Med Decis Making 2006). Range to be narrowed with
# the condition-specific values from that catalogue.
QALY_WEIGHT = (0.6, 0.85)
# Discount rate 3% (Second Panel on Cost-Effectiveness; Sanders et al., JAMA 2016)
DISCOUNT = 0.03
# Cost per readmission: $14,400 average 30-day readmission cost, 2016
# (HCUP Statistical Brief 248, Bailey et al. 2019); range covers payer mix and inflation.
READM_COST = (10_000, 20_000)
# VBP exchange function slope: incentive % = 2% x (TPS / 100) x slope (FY2025
# Hospital VBP PPSR help guide). The program is budget neutral, so the slope
# is about 100 / mean TPS; from hvbp_panel.csv that is 2.6 to 4.4 across FY2017-FY2026.
VBP_SLOPE = (2.6, 4.4)
WTP = np.arange(0, 300_001, 5_000)   # thresholds $100k-150k per QALY (Neumann et al., NEJM 2014)

MORT = ["mort_ami", "mort_hf", "mort_pn", "mort_copd", "mort_cabg"]
READM = ["readm_ami", "readm_hf", "readm_pn", "readm_copd", "readm_cabg"]
CMS_ID = {o: o.upper().replace("MORT_", "MORT_30_").replace("READM_", "READM_30_") for o in MORT + READM}
FIN = ["log_cost_per_dc", "log_total_costs", "log_drg_per_mcr_dc", "vbp_tps", "op_margin"]
QUAL = ["sep1"]

ap = argparse.ArgumentParser()
ap.add_argument("--design", default="epic_switch", choices=["epic_switch", "epic_acquisition"])
ap.add_argument("--detrend", action="store_true", help="use trend-adjusted effects")
ap.add_argument("--subset", choices=["safety_net", "other"], default=None,
                help="distributional analysis: safety-net hospitals (top quarter of Medicaid "
                     "share of inpatient days, measured before treatment) or all others")
args = ap.parse_args()
tag = args.design + ("_detrend" if args.detrend else "") + (f"_{args.subset}" if args.subset else "")
P = "data/processed"

panel = pd.read_csv(f"{P}/analysis_panel.csv", dtype={"ccn": str, "id": str}, low_memory=False)
if args.design == "epic_switch":
    d = pd.read_csv(f"{P}/ext1_switch_only.csv", dtype={"id": str}, low_memory=False)
    d = d.merge(panel[["id", "year", "ccn"]], on=["id", "year"], how="left")
else:
    d = panel[panel["arm"].isin(["epic", "never_acquired"]) & panel["gvar"].notna()].copy()

# ---- distributional subset: safety-net status fixed before treatment ----
if args.subset:
    ms = d[["id", "year", "gvar"]].merge(panel[["id", "year", "medicaid_share"]], on=["id", "year"], how="left")
    ms = ms[(ms["gvar"] == 0) | (ms["year"] < ms["gvar"])]
    h = ms.groupby("id")["medicaid_share"].mean().dropna()
    cut = h.quantile(0.75)
    sn = set(h.index[h >= cut])
    keep = d["id"].isin(sn) if args.subset == "safety_net" else d["id"].isin(set(h.index) - sn)
    d = d[keep].copy()
    print(f"Subset {args.subset}: Medicaid share cutoff {cut:.3f}; "
          f"{d.loc[d['gvar'] > 0, 'id'].nunique()} treated, {d.loc[d['gvar'] == 0, 'id'].nunique()} comparison hospitals")

# ---- joint bootstrap weights: one Poisson(1) weight per hospital per draw ----
rng = np.random.default_rng(SEED)
units = np.sort(d["hosp_num"].unique())
UW = pd.DataFrame(rng.poisson(1.0, size=(len(units), REPS)), index=units)

eff, info = {}, []
for y in MORT + READM + FIN + QUAL:
    if y not in d.columns:
        print(f"  {y}: not in data, set to 0"); eff[y] = np.zeros(REPS + 1); continue
    r = cs_att(d, y, unit_weights=UW)
    if r is None:
        print(f"  {y}: not estimable, set to 0"); eff[y] = np.zeros(REPS + 1); continue
    if args.detrend:
        dt = detrend(r)
        if dt is not None:
            eff[y] = np.concatenate([[dt[0]], dt[3]])
            info.append((y, dt[0], dt[1], r["n_treated"], "trend-adjusted")); continue
    eff[y] = np.concatenate([[r["att"]], r["draws"]])
    info.append((y, r["att"], r["se"], r["n_treated"], "ATT"))
est = pd.DataFrame(info, columns=["outcome", "estimate", "se", "n_treated", "type"])
print(est.round(4).to_string(index=False))

# ---- baselines for treated hospitals: mean of the 3 years before treatment ----
cr = pd.read_csv(f"{P}/cost_report_panel.csv", dtype={"ccn": str})
tr = d[d["gvar"] > 0][["id", "ccn", "year", "gvar"]].dropna(subset=["ccn"])
pre = tr[(tr["year"] >= tr["gvar"] - 3) & (tr["year"] <= tr["gvar"] - 1)]
pre = pre.merge(cr[["ccn", "year", "total_costs", "drg", "net_patient_rev"]], on=["ccn", "year"])
base = pre.groupby("id")[["total_costs", "drg", "net_patient_rev"]].mean().mean()
c = pd.read_csv(f"{P}/cms_complications_panel.csv", dtype=str, low_memory=False)
c = c[c["ccn"].isin(tr["ccn"].unique()) & c["measure_id"].isin(CMS_ID.values())].copy()
yrs = ((pd.to_datetime(c["end_date"]) - pd.to_datetime(c["start_date"])).dt.days / 365.25).round()
c["per_year"] = pd.to_numeric(c["denominator"], errors="coerce") / yrs
pts = c.groupby("measure_id")["per_year"].median()
print("\nBaseline per treated hospital-year: total costs ${:,.0f}; Medicare DRG ${:,.0f}; "
      "net patient revenue ${:,.0f}".format(base["total_costs"], base["drg"], base["net_patient_rev"]))
print("Measured patients per year:", {k: round(v, 1) for k, v in pts.items()})

# ---- PSA draws (draw 0 = point estimates with parameter midpoints) ----
n = REPS + 1
def u(lo_hi):
    v = rng.uniform(*lo_hi, size=n); v[0] = np.mean(lo_hi); return v
ly, qw, rc, slope = u(LIFE_YEARS), u(QALY_WEIGHT), u(READM_COST), u(VBP_SLOPE)
dly = (1 - (1 + DISCOUNT) ** -ly) / DISCOUNT

deaths = -sum(eff[m] / 100 * pts.get(CMS_ID[m], 0) for m in MORT)
readm = -sum(eff[m] / 100 * pts.get(CMS_ID[m], 0) for m in READM)
qalys = deaths * dly * qw
res_cost = (np.exp(eff["log_cost_per_dc"]) - 1) * base["total_costs"]
res_cost_total = (np.exp(eff["log_total_costs"]) - 1) * base["total_costs"]
mcr_drg = (np.exp(eff["log_drg_per_mcr_dc"]) - 1) * base["drg"]
mcr_vbp = eff["vbp_tps"] * 0.02 * slope / 100 * base["drg"]
readm_saved = readm * rc
hosp_net = eff["op_margin"] * base["net_patient_rev"]

draws = pd.DataFrame(dict(deaths_averted=deaths, readmissions_averted=readm, qalys=qalys,
                          society_cost=res_cost - readm_saved,
                          society_cost_total=res_cost_total - readm_saved,
                          medicare_cost=mcr_drg + mcr_vbp - readm_saved,
                          medicare_vbp=mcr_vbp, hospital_net_gain=hosp_net,
                          sep1_points=eff["sep1"]))
os.makedirs("output/tables", exist_ok=True); os.makedirs("output/figures", exist_ok=True)
draws.to_csv(f"output/tables/cea_{tag}_draws.csv", index=False)

rows = []
for persp, col in [("society", "society_cost"), ("society, total-cost version", "society_cost_total"),
                   ("Medicare", "medicare_cost")]:
    C, E = draws[col].to_numpy(), draws["qalys"].to_numpy()
    pt_c, pt_e = C[0], E[0]
    nmb = lambda w: (w * E[1:] - C[1:])
    rows.append(dict(perspective=persp,
                     cost_point=pt_c, cost_lo=np.percentile(C[1:], 2.5), cost_hi=np.percentile(C[1:], 97.5),
                     qaly_point=pt_e, qaly_lo=np.percentile(E[1:], 2.5), qaly_hi=np.percentile(E[1:], 97.5),
                     icer=(pt_c / pt_e) if pt_e > 0 and pt_c > 0 else np.nan,
                     p_dominated=np.mean((C[1:] > 0) & (E[1:] <= 0)),
                     p_dominant=np.mean((C[1:] <= 0) & (E[1:] > 0)),
                     p_ce_50k=np.mean(nmb(50_000) > 0), p_ce_100k=np.mean(nmb(100_000) > 0),
                     p_ce_150k=np.mean(nmb(150_000) > 0)))
S = pd.DataFrame(rows)
sep = draws["sep1_points"].to_numpy()
extra = pd.DataFrame([
    dict(perspective="hospital net financial gain", cost_point=hosp_net[0],
         cost_lo=np.percentile(hosp_net[1:], 2.5), cost_hi=np.percentile(hosp_net[1:], 97.5)),
    dict(perspective="Medicare VBP payment change", cost_point=mcr_vbp[0],
         cost_lo=np.percentile(mcr_vbp[1:], 2.5), cost_hi=np.percentile(mcr_vbp[1:], 97.5)),
    dict(perspective="SEP-1 points (measured quality)", cost_point=sep[0],
         cost_lo=np.percentile(sep[1:], 2.5), cost_hi=np.percentile(sep[1:], 97.5)),
    dict(perspective="deaths averted (5 conditions)", cost_point=deaths[0],
         cost_lo=np.percentile(deaths[1:], 2.5), cost_hi=np.percentile(deaths[1:], 97.5))])
S = pd.concat([S, extra], ignore_index=True)
S.to_csv(f"output/tables/cea_{tag}_summary.csv", index=False)
pd.set_option("display.width", 250)
print(f"\n=== {tag}: per treated hospital-year, 95% intervals from {REPS} joint draws ===")
print(S.round(3).to_string(index=False))

# ---- figures: cost-effectiveness plane and acceptability curve ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 5))
for col, lab, colr in [("society_cost", "Society", "#0f6e6e"), ("medicare_cost", "Medicare", "#c2571a")]:
    ax.scatter(draws["qalys"][1:], draws[col][1:] / 1e6, s=5, alpha=0.25, color=colr, label=lab)
    ax.scatter(draws["qalys"][0], draws[col][0] / 1e6, s=60, color=colr, edgecolor="black", zorder=3)
x = np.linspace(*ax.get_xlim(), 50)
for w, ls in [(100_000, "--"), (150_000, ":")]:
    ax.plot(x, w * x / 1e6, ls, color="gray", lw=1, label=f"${w // 1000}k per QALY")
ax.axhline(0, color="black", lw=0.6); ax.axvline(0, color="black", lw=0.6)
ax.set_xlabel("QALYs gained per hospital-year (5 measured conditions)")
ax.set_ylabel("Incremental cost per hospital-year ($ millions)")
ax.set_title(f"Cost-effectiveness plane: {tag.replace('_', ' ')}")
ax.legend(frameon=False, fontsize=8); fig.tight_layout()
fig.savefig(f"output/figures/cea_{tag}_plane.png", dpi=200); plt.close(fig)
fig, ax = plt.subplots(figsize=(7, 4))
for col, lab, colr in [("society_cost", "Society", "#0f6e6e"), ("medicare_cost", "Medicare", "#c2571a")]:
    C, E = draws[col].to_numpy()[1:], draws["qalys"].to_numpy()[1:]
    ax.plot(WTP / 1000, [np.mean(w * E - C > 0) for w in WTP], color=colr, label=lab)
ax.set_ylim(0, 1); ax.set_xlabel("Willingness to pay per QALY ($ thousands)")
ax.set_ylabel("Probability cost-effective"); ax.set_title(f"Acceptability curve: {tag.replace('_', ' ')}")
ax.legend(frameon=False); fig.tight_layout()
fig.savefig(f"output/figures/cea_{tag}_ceac.png", dpi=200); plt.close(fig)
print(f"\nSaved output/tables/cea_{tag}_*.csv and output/figures/cea_{tag}_*.png")
