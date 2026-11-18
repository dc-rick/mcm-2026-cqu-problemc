"""T022-M16: direction of the capture-recapture bias in the fleet estimate.

Target population: the bicycles in service during the first window
(2017-05-10 .. 2017-05-19, 5/17 absent).  Capture occasions:
  window 1 = the nine recorded days of 10-19 May,
  window 2 = 2017-05-20 .. 2017-05-24 (truncated sample, 54.5% of the volume).

Deliverables:
  1. reproduce the pooled Chapman estimate 495,540 from data/trips.parquet;
  2. stratified Chapman by first-window usage frequency (5 strata, >=4 required),
     reporting n1, n2, m per stratum and showing the within-stratum degeneracy;
  3. Chao (1987) lower bound / two-occasion Mth-type lower bound for a
     heterogeneous population;
  4. bias-direction derivation for retirement and new deployment relative to
     the first-window fleet;
  5. recompute 4.33 rides per bicycle per day under the new bound.

Reads data/ only; writes data/runs/T022/.  Python must be run with -I.
"""
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, "runs", "T022")
os.makedirs(OUT, exist_ok=True)

# --- input (same load and bbox filter as lib.load_trips, but memory-lean) -----
t = pq.read_table(os.path.join(DATA, "trips.parquet"),
                  columns=["bikeid", "starttime", "start_lat", "start_lon"])
bike = np.asarray(t.column("bikeid")).astype(np.int64)
st = np.asarray(t.column("starttime")).astype("datetime64[s]")
lat = np.asarray(t.column("start_lat"))
lon = np.asarray(t.column("start_lon"))
ok = (lat > 39.4) & (lat < 40.4) & (lon > 115.8) & (lon < 117.0)
bike = bike[ok]
day = (st[ok].astype("datetime64[D]") - np.datetime64("2017-05-10")).astype(int)
del t, st, lat, lon, ok

W1 = day <= 9        # 10-19 May
W2 = day >= 10       # 20-24 May

u1, c1 = np.unique(bike[W1], return_counts=True)      # trips per bike, window 1
u2, c2 = np.unique(bike[W2], return_counts=True)      # trips per bike, window 2
in2 = np.isin(u1, u2, assume_unique=True)             # memory-lean membership
n1, n2 = len(u1), len(u2)
m = int(in2.sum())
union = n1 + n2 - m

# ---------------------------------------------------------------- reproduction
chap = (n1 + 1) * (n2 + 1) / (m + 1) - 1
var = ((n1 + 1) * (n2 + 1) * (n1 - m) * (n2 - m)
       / ((m + 1) ** 2 * (m + 2)))
se = float(np.sqrt(var))
ci = [float(chap - 1.96 * se), float(chap + 1.96 * se)]

# ---------------------------------------------------------------- strata
# Strata by number of sampled trips in window 1.  The cut points follow the
# observed quantiles (P25=3, P50=5, P75=7, P90=10) and isolate the top decile;
# the 21+ group has only 90 bicycles, so it is merged into 11+.
BINS = [(1, 1), (2, 3), (4, 6), (7, 10), (11, 10 ** 9)]
LABELS = ["1 trip", "2-3 trips", "4-6 trips", "7-10 trips", "11+ trips"]
strata = []
for (lo, hi), lab in zip(BINS, LABELS):
    sel = (c1 >= lo) & (c1 <= hi)
    ids = u1[sel]
    rec = np.isin(ids, u2, assume_unique=True)
    ns, rs = int(sel.sum()), int(rec.sum())
    rate = rs / ns
    strata.append(dict(
        label=lab, n1=ns, n2=rs, m=rs, recapture_rate=float(rate),
        rate_se=float(np.sqrt(rate * (1 - rate) / ns)),
        chapman_within=float(ns),          # (n1+1)(m+1)/(m+1)-1 = n1
        share_of_n1=float(ns / n1),
        note=("m equals n2 because stratum membership requires a window-1 "
              "capture; the within-stratum Chapman collapses to n1, so the "
              "strata cannot identify unseen bicycles.")))

# trend test on the 5x2 table (recaptured / not, by stratum)
tab = np.array([[s["n2"], s["n1"] - s["n2"]] for s in strata], dtype=float)
chi2, p, dof, _ = stats.chi2_contingency(tab)
scores = np.array([1, 2, 3, 4, 5], dtype=float)
# Cochran-Armitage trend on the same table
row_tot = tab.sum(1)
col_tot = tab.sum(0)
Ntot = tab.sum()
pbar = col_tot[0] / Ntot
t_num = np.sum(scores * (tab[:, 0] - row_tot * pbar))
t_den = np.sqrt(pbar * (1 - pbar) * (
    np.sum(row_tot * scores ** 2) - np.sum(row_tot * scores) ** 2 / Ntot))
z_trend = float(t_num / t_den)

# zero stratum: first seen in window 2 (no window-1 trips)
zero = dict(n1=0, n2=n2 - m, m=0,
            chapman_within=float(n2 - m),
            note=("bikes first seen in window 2; they cannot be recaptured, "
                  "so the two-occasion design gives no unseen estimate for "
                  "this stratum either."))
w2only = ~np.isin(u2, u1, assume_unique=True)
c2_only = c2[w2only]
c2_both = c2[~w2only]
zero["w2_trips_mean"] = float(c2_only.mean())
zero["w2_trips_median"] = float(np.median(c2_only))
zero["w2_trips_share1"] = float((c2_only == 1).mean())
zero["w2_trips_share_ge4"] = float((c2_only >= 4).mean())
zero["recaptured_w2_trips_mean"] = float(c2_both.mean())
zero["recaptured_w2_trips_share1"] = float((c2_both == 1).mean())
zero["recaptured_w2_trips_share_ge4"] = float((c2_both >= 4).mean())

strat_sum = sum(s["chapman_within"] for s in strata) + zero["chapman_within"]

# ---------------------------------------------------------------- Chao 1987
f1 = n1 + n2 - 2 * m          # seen in exactly one window
f2 = m                        # seen in both
f0_hat = f1 ** 2 / (2 * f2)
N_chao = union + f0_hat
var_c = f2 * (0.5 * (f1 / f2) ** 2 + (f1 / f2) ** 3
              + 0.25 * (f1 / f2) ** 4)
se_log = float(np.sqrt(np.log(1 + var_c / f0_hat ** 2)))
chao_lo = union + f0_hat * np.exp(-1.96 * se_log)
chao_hi = union + f0_hat * np.exp(1.96 * se_log)

# ---------------------------------------------------------------- openness
# Marginal effects on the Chapman estimate, evaluated at the observed counts.
dN_new_captured = (n1 + 1) / (m + 1)                       # n2 +1, m fixed
dN_retire_recaptured = (n1 + 1) * (n2 / m - (n2 + 1) / (m + 1))
dN_retire_unused = 0.0                                     # n1 already observed

# usage of the W1-only group (all possible retirees are in here)
w1only_mask = ~in2
c1_only = c1[w1only_mask]
c1_both = c1[in2]

# ---------------------------------------------------------------- rides/day
RIDES_DAY = 274357.4285714286 / 0.128      # Q1 model scaled daily demand
old_F = 495540.0
old_rate = RIDES_DAY / old_F

out = dict(
    task="T022-M16",
    date="2026-11-18",
    windows=dict(
        w1=["2017-05-10", "2017-05-19"],
        w2=["2017-05-20", "2017-05-24"],
        w1_note="nine recorded days; 2017-05-17 has no records",
        w2_note="five days, truncated sample at 54.5% of the normal volume"),
    counts=dict(n1=int(n1), n2=int(n2), m=int(m), union=int(union),
                w1_only=int(n1 - m), w2_only=int(n2 - m)),
    chapman_pooled=dict(
        estimate=float(chap), se=se, ci95=ci,
        paper_value=old_F, reproduced=bool(abs(chap - old_F) < 1.0),
        formula="(n1+1)(n2+1)/(m+1)-1"),
    strata=strata,
    strata_rule=("five strata by sampled trips in window 1: 1, 2-3, 4-6, "
                 "7-10, 11+; cut points follow the observed quantiles "
                 "(P25=3, P50=5, P75=7, P90=10), and the 21+ group (90 "
                 "bicycles) is merged into 11+"),
    zero_stratum=zero,
    stratified_chapman=dict(
        sum_over_strata=float(strat_sum),
        equals_union=bool(abs(strat_sum - union) < 1e-9),
        note=("Summing the within-stratum Chapman estimates returns the "
              "observed union exactly.  All 21,861 first-seen-in-window-2 "
              "bicycles and all 152,932 window-1-only bicycles stay outside "
              "any unseen term.  The pooled estimator's 10,786 unseen "
              "bicycles are produced entirely by borrowing the low-stratum "
              "recapture rate across the whole fleet, i.e. by the "
              "homogeneity assumption that the monotone rates reject.")),
    heterogeneity=dict(
        chi2_5x2=float(chi2), dof=int(dof), p_value=float(p),
        cochran_armitage_z=z_trend,
        rate_min=float(min(s["recapture_rate"] for s in strata)),
        rate_max=float(max(s["recapture_rate"] for s in strata)),
        rate_ratio=float(max(s["recapture_rate"] for s in strata)
                         / min(s["recapture_rate"] for s in strata)),
        mean_recapture=float(m / n1),
        conclusion=("Recapture probability rises monotonically with "
                    "first-window usage frequency; positive correlation "
                    "between the two windows' detection propensities makes "
                    "m larger than the independence model predicts, so the "
                    "pooled Chapman estimate is biased DOWN.  The paper's "
                    "'heterogeneity biases it up' is reversed.")),
    chao_1987=dict(
        f1=int(f1), f2=int(f2), f0_hat=float(f0_hat), N_hat=float(N_chao),
        ci95=[float(chao_lo), float(chao_hi)],
        var=float(var_c), se_log=float(se_log),
        status=("lower bound for a heterogeneous population under closure "
                "(two-occasion Mth / sample-coverage bound).  The bound lies "
                "above the pooled Chapman point, so the point estimate is "
                "inconsistent with a heterogeneous fleet."),
        caveat=("the formula assumes a common per-bicycle detection "
                "propensity in the two windows; the second window has 30.3% "
                "of the first window's exposure (5 days x 0.545 volume over "
                "9 days), so the bound is indicative, not exact.")),
    openness=dict(
        new_deployment_captured=dict(
            dN_per_bike=float(dN_new_captured),
            direction="up",
            derivation=("a bicycle deployed after window 1 that is used in "
                        "window 2 enters n2 but cannot enter m, so each such "
                        "bicycle adds (n1+1)/(m+1) bicycles to the estimate")),
        retirement_of_recapturable=dict(
            dN_per_bike=float(dN_retire_recaptured),
            direction="up",
            derivation=("a retired bicycle is already counted in n1; it "
                        "changes the estimate only if it would have been "
                        "recaptured, in which case n2 and m both fall by one "
                        "and the ratio (n2+1)/(m+1) rises because n2>m")),
        retirement_of_unused=dict(
            dN_per_bike=0.0,
            direction="none",
            derivation=("a bicycle that would not have been used in window 2 "
                        "contributes to none of n1, n2, m after retirement; "
                        "its removal leaves the estimate unchanged")),
        consequence=("retirement can only bias the estimate up or leave it "
                     "unchanged, never down; new deployment likewise.  The "
                     "paper's 'retired between them biases the estimate "
                     "down' is wrong for this data definition."),
        w2_only_usage=dict(
            mean=zero["w2_trips_mean"],
            recaptured_mean=zero["recaptured_w2_trips_mean"],
            share_ge4=zero["w2_trips_share_ge4"],
            recaptured_share_ge4=zero["recaptured_w2_trips_share_ge4"],
            note=("first-seen-in-window-2 bicycles are only 13% less used in "
                  "window 2 than recaptured bicycles; a pure zero-window-1 "
                  "tail would be far less active, so part of the 21,861 are "
                  "new deployments, which bias the estimate up")),
        w1_only_usage=dict(
            mean=float(c1_only.mean()), recaptured_mean=float(c1_both.mean()),
            note="all possible retirees are inside the window-1-only group")),
    interval=dict(
        old=[484754.0, 506635.0],
        new_lower=float(N_chao),
        note=("the old upper reference 506,635 lies below the heterogeneity "
              "lower bound 534,039, so the interval [484,754, 506,635] is "
              "too narrow and its upper end is invalid; the defensible "
              "statement is a heterogeneity-corrected lower bound, with the "
              "observed union 484,754 as the hard floor")),
    rides_per_bike_day=dict(
        numerator=RIDES_DAY,
        numerator_source="q1_params.json daily_trips / 0.128",
        values={
            "F_495540": float(old_rate),
            "F_534039": float(RIDES_DAY / N_chao),
            "F_chao_ci_low": float(RIDES_DAY / chao_lo),
            "F_chao_ci_high": float(RIDES_DAY / chao_hi)},
        conclusion=("4.33 corresponds to the downward-biased Chapman point; "
                    "with F >= 534,039 the rate is at most about 4.01, so "
                    "4.33 must be reported as an upper estimate, not a "
                    "central one")),
)

with open(os.path.join(OUT, "fleet_estimates.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)

print(json.dumps(dict(
    counts=out["counts"], chapman=out["chapman_pooled"]["estimate"],
    strata=[(s["label"], s["n1"], s["n2"], round(s["recapture_rate"], 4))
            for s in strata],
    zero=(zero["n2"], zero["w2_trips_mean"], zero["recaptured_w2_trips_mean"]),
    trend=(round(z_trend, 1), f"{p:.3g}"),
    chao=(round(f0_hat, 1), round(N_chao, 1),
          [round(chao_lo, 1), round(chao_hi, 1)]),
    dN=(round(dN_new_captured, 3), round(dN_retire_recaptured, 3)),
    rates={k: round(v, 4) for k, v in out["rides_per_bike_day"]["values"].items()},
), ensure_ascii=False, indent=1))
