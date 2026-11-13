# check_cs_estimates.py
# Purpose: Independent re-implementation of the Callaway-Sant'Anna estimator
#          in Python, used (1) to cross-check the R `did` results in
#          output/att_simple.csv and (2) to get first-pass estimates for new
#          outcomes on a machine without R.
# Author: Caprice Culkin
# Date: November 2026
#
# Matches the main R specification: never-treated comparison, no covariates,
# unbalanced panel treated as repeated cross-sections, base period g-1
# (or the last observed year before g when g-1 has no data).
# With no covariates the doubly robust estimator reduces to a difference in
# mean changes, so each group-time effect is
#   ATT(g,t) = [mean Y(g,t) - mean Y(g,b)] - [mean Y(C,t) - mean Y(C,b)],  b = base
# The overall ATT averages post-treatment cells (t >= g) weighting each by
# its group's number of hospitals. On the October 28 build this reproduces
# every main-specification point estimate in output/att_simple.csv exactly. Standard errors come from a hospital-level
# bootstrap (Poisson weights), not R's multiplier bootstrap, so they will be
# close to, not identical with, the R values.
#
# Usage:
#   python code/check_cs_estimates.py                 cross-check vs R output
#   python code/check_cs_estimates.py out1 out2 ...   estimate listed outcomes
#   add --file data/processed/ext1_switch_only.csv to use an extension file
# For listed outcomes it also prints a trend-adjusted effect (see detrend()),
# a sensitivity check when pre-period points drift.

import sys

import numpy as np
import pandas as pd

REPS = 500
rng = np.random.default_rng(20261029)


def wavg(pairs):
    """Weighted mean of cell estimates. Cells missing in the point estimate
    are dropped; a cell missing in one bootstrap draw (no hospitals drawn)
    is dropped from that draw only."""
    pairs = [(w, a) for w, a in pairs if not np.isnan(a[0])]
    if not pairs:
        return None
    W = np.array([w for w, _ in pairs], float)[:, None]
    A = np.vstack([a for _, a in pairs])
    M = ~np.isnan(A)
    return np.nansum(W * A, axis=0) / (W * M).sum(axis=0)


def cs_att(d, y, reps=REPS, min_e=-5, max_e=5, unit_weights=None):
    """unit_weights: optional DataFrame of bootstrap weights indexed by
    hosp_num (columns = draws). Passing the same weights for several outcomes
    gives jointly bootstrapped estimates, which keeps their correlation."""
    d = d[d[y].notna() & d["gvar"].notna()][["hosp_num", "year", "gvar", y]]
    years = np.sort(d["year"].unique())
    # As in R's did: hospitals first treated after the last observed year are
    # never treated within this outcome's window, so they join the comparison
    # group; hospitals treated in or before the first year are dropped below.
    d = d.assign(gvar=np.where(d["gvar"] > years.max(), 0, d["gvar"]))
    # base period: the last observed year before g (g-1 when it exists). For
    # outcomes with gap years R uses the latest earlier year, and so does this.
    base_of = {g: years[years < g].max() for g in d.loc[d["gvar"] > 0, "gvar"].unique()
               if (years < g).any()}
    groups = sorted(base_of)
    d = d[(d["gvar"] == 0) | d["gvar"].isin(groups)]
    units = d["hosp_num"].unique()
    uidx = pd.Series(np.arange(len(units)), index=units)
    # bootstrap weights: column 0 is the point estimate (all weights 1)
    if unit_weights is not None:
        W = np.hstack([np.ones((len(units), 1)), unit_weights.loc[units].to_numpy()])
        reps = W.shape[1] - 1
    else:
        W = np.ones((len(units), reps + 1))
        W[:, 1:] = rng.poisson(1.0, size=(len(units), reps))
    w = W[uidx[d["hosp_num"]].to_numpy()]               # row weights, rows x (reps+1)
    yv = d[y].to_numpy()[:, None]
    cell = d["gvar"].astype(int).astype(str) + "_" + d["year"].astype(int).astype(str)
    codes, labels = pd.factorize(cell)
    sums = np.zeros((len(labels), reps + 1)); cnts = np.zeros_like(sums)
    np.add.at(sums, codes, w * yv); np.add.at(cnts, codes, w)
    lab = {l: i for i, l in enumerate(labels)}

    def mean(g, t):
        i = lab.get(f"{int(g)}_{int(t)}")
        if i is None:
            return None
        with np.errstate(invalid="ignore", divide="ignore"):
            return sums[i] / cnts[i]

    # group weights: number of hospitals in each group (matches R's aggte())
    gsize = d[d["gvar"] > 0].drop_duplicates("hosp_num").groupby("gvar").size()
    cells = []
    for g in groups:
        for t in years:
            base = base_of[g]
            m = [mean(g, t), mean(g, base), mean(0, t), mean(0, base)]
            if any(v is None for v in m) or t == base:
                continue
            cells.append((g, t, t - g, (m[0] - m[1]) - (m[2] - m[3])))
    post = [(g, a) for g, t, e, a in cells if e >= 0]
    if not post or all(np.isnan(a[0]) for _, a in post):
        return None
    overall = wavg([(gsize[g], a) for g, a in post])
    if overall is None:
        return None
    es = {}
    for e in range(min_e, max_e + 1):
        sel = [(gsize[g], a) for g, t, ee, a in cells if ee == e]
        v = wavg(sel) if sel else None
        if v is not None:
            es[e] = v
    pre_t = [abs(v[0]) / np.nanstd(v[1:]) for e, v in es.items() if e <= -2 and np.nanstd(v[1:]) > 0]
    return dict(att=overall[0], se=np.nanstd(overall[1:]), draws=overall[1:],
                n_treated=int(d.loc[d["gvar"] > 0, "hosp_num"].nunique()),
                n_hospitals=int(len(units)),
                max_abs_pre_t=max(pre_t) if pre_t else np.nan,
                es={e: (v[0], np.nanstd(v[1:])) for e, v in es.items()},
                es_draws=es)


def detrend(r, post=range(0, 6)):
    """Sensitivity check for a pre-existing trend: fit a line through the
    pre-period event-study points (base year -1 fixed at 0), extrapolate it
    into the post period, and average post-period gaps from that line.
    Done within each bootstrap draw, so the SE carries the trend's uncertainty."""
    es = r["es_draws"]
    pre = sorted(e for e in es if e <= -2)
    post = [e for e in post if e in es]
    if len(pre) < 2 or not post:
        return None
    x = np.array(pre + [-1], float)
    Y = np.vstack([es[e] for e in pre] + [np.zeros_like(es[pre[0]])])
    xc = x - x.mean()
    slope = (xc[:, None] * (Y - Y.mean(0))).sum(0) / (xc ** 2).sum()
    icpt = Y.mean(0) - slope * x.mean()
    gaps = np.vstack([es[e] - (icpt + slope * e) for e in post]).mean(0)
    return gaps[0], np.nanstd(gaps[1:]), slope[0], gaps[1:]


def arm_data(panel, arm):
    return panel[panel["arm"].isin([arm, "never_acquired"]) & panel["gvar"].notna()]


def row(label, r):
    lo, hi = r["att"] - 1.96 * r["se"], r["att"] + 1.96 * r["se"]
    return (f"{label:42s} ATT {r['att']:9.4f}  SE {r['se']:7.4f}  CI [{lo:8.4f}, {hi:8.4f}]  "
            f"treated {r['n_treated']:4d}  max|pre t| {r['max_abs_pre_t']:.2f}")


if __name__ == "__main__":
    args = sys.argv[1:]
    path = "data/processed/analysis_panel.csv"
    if "--file" in args:
        i = args.index("--file"); path = args[i + 1]; args = args[:i] + args[i + 2:]
    panel = pd.read_csv(path, low_memory=False)
    if not args:
        # Cross-check against the R estimates for the main specification
        r_out = pd.read_csv("output/att_simple.csv")
        r_out = r_out[r_out["spec"] == "main"]
        worst = 0
        for _, rr in r_out.iterrows():
            if rr["outcome"] not in panel.columns:
                continue
            p = cs_att(arm_data(panel, rr["arm"]), rr["outcome"], reps=200)
            if p is None:
                continue
            gap = abs(p["att"] - rr["att"]) / max(rr["se"], 1e-9)
            worst = max(worst, gap)
            print(f"{rr['arm']:6s} {rr['outcome']:20s} R {rr['att']:9.4f} (se {rr['se']:.4f})"
                  f"   Python {p['att']:9.4f} (se {p['se']:.4f})   gap/SE {gap:.3f}")
        print(f"\nLargest point-estimate gap: {worst:.3f} R standard errors")
        print("MATCH" if worst < 0.05 else "CHECK: estimates differ by more than 0.05 SE")
    else:
        arms = ["epic", "oracle"] if "arm" in panel.columns else [None]
        for y in args:
            for a in arms:
                d = arm_data(panel, a) if a else panel[panel["gvar"].notna()]
                r = cs_att(d, y)
                if r is None:
                    print(f"{(a or 'design')}/{y}: no estimable cells"); continue
                print(row(f"{a or path.split('/')[-1]} | {y}", r))
                print("    event study:", "  ".join(f"{e}:{v[0]:.3f}({v[1]:.3f})"
                                                   for e, v in sorted(r["es"].items())))
                dt = detrend(r)
                if dt:
                    print(f"    trend-adjusted post effect (years 0-5): {dt[0]:.4f} "
                          f"(SE {dt[1]:.4f}; pre-trend slope {dt[2]:.4f} per year)")
