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

# ------------------------------------------------- Chao1 and effort benchmark
# Two windows give only (n1, n2, m).  A homogeneous model (N, p1, p2) fits
# those three statistics exactly, so the raw Chao1 value is a deterministic
# function of the same three numbers; with p1 != p2 it is NOT a lower bound.
f1 = n1 + n2 - 2 * m          # seen in exactly one window
f2 = m                        # seen in both
f0_raw = f1 ** 2 / (2 * f2)
chao1_raw = union + f0_raw

N_h = chap
p1_h = n1 / N_h
p2_h = n2 / N_h
E_f1 = N_h * (p1_h * (1 - p2_h) + (1 - p1_h) * p2_h)
E_f2 = N_h * p1_h * p2_h
E_union = N_h * (p1_h + p2_h - p1_h * p2_h)
chao1_homog = E_union + E_f1 ** 2 / (2 * E_f2)
effort_factor_obs = chao1_raw / N_h
exact_factor = (p1_h + p2_h - p1_h * p2_h
                + (p1_h + p2_h - 2 * p1_h * p2_h) ** 2
                / (2 * p1_h * p2_h))


def equal_effort_factor(p):
    """Chao1/N under equal effort p in both windows."""
    return 1.0 + (1.0 - p) ** 2


# 200 draws of five first-window days, second window fixed.
w1_days = sorted(set(int(d) for d in day[W1]))
per_day = {d: np.unique(bike[day == d]) for d in w1_days}
rng = np.random.default_rng(20261118)
boot = []
for _ in range(200):
    sub = rng.choice(w1_days, 5, replace=False)
    u = np.unique(np.concatenate([per_day[int(d)] for d in sub]))
    mr = int(np.intersect1d(u, u2).size)
    dr = len(u) + n2 - mr
    f1r = len(u) + n2 - 2 * mr
    boot.append(dr + f1r ** 2 / (2 * mr))
boot = np.array(boot)


def chao_split(days):
    u = np.unique(np.concatenate([per_day[d] for d in days]))
    mr = int(np.intersect1d(u, u2).size)
    dr = len(u) + n2 - mr
    f1r = len(u) + n2 - 2 * mr
    return dict(n1=int(len(u)), m=mr, chao1=float(dr + f1r ** 2 / (2 * mr)))


named = {"10-14": chao_split([0, 1, 2, 3, 4]),
         "15-19": chao_split([5, 6, 8, 9])}

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
                     "down' is wrong for this data definition.  The net sign "
                     "of the openness terms is not identified here."),
        w2_only_usage=dict(
            mean=zero["w2_trips_mean"],
            recaptured_mean=zero["recaptured_w2_trips_mean"],
            share_ge4=zero["w2_trips_share_ge4"],
            recaptured_share_ge4=zero["recaptured_w2_trips_share_ge4"],
            note=("first-seen-in-window-2 bicycles are only 13% less used in "
                  "window 2 than recaptured bicycles; a pure zero-window-1 "
                  "tail would be far less active.  This is suggestive of some "
                  "new deployment, but it is an activity difference, not an "
                  "identification of a deployment flow: treat it as a hint, "
                  "not a measured quantity")),
        w1_only_usage=dict(
            mean=float(c1_only.mean()), recaptured_mean=float(c1_both.mean()),
            note="all possible retirees are inside the window-1-only group")),
    chao1_raw=dict(
        value=float(chao1_raw), f1=int(f1), f2=int(f2),
        status=("withdrawn as a lower bound.  Two windows reduce the data to "
                "(n1,n2,m); the homogeneous model (N,p1,p2) fits all three "
                "exactly, and Chao1 is the same deterministic function of "
                "them.  The excess over the Chapman point measures the "
                "unequal window effort (p1=0.934 vs p2=0.670), not "
                "heterogeneity."),
        note="reported as a number only; not used as a bound anywhere"),
    homogeneity_benchmark=dict(
        N=float(N_h), p1=float(p1_h), p2=float(p2_h),
        E_f1=float(E_f1), E_f2=float(E_f2), E_union=float(E_union),
        observed_f1=int(f1), observed_f2=int(f2), observed_union=int(union),
        chao1_homog=float(chao1_homog), chao1_observed=float(chao1_raw),
        diff_chao1=float(chao1_raw - chao1_homog),
        max_stat_diff=float(max(abs(E_f1 - f1), abs(E_f2 - f2),
                                abs(E_union - union))),
        note=("the homogeneous model reproduces the three observed statistics "
              "and the Chao1 value to within 0.05 bicycles, so the raw Chao1 "
              "excess is an effort artefact")),
    chao_effort_bias=dict(
        equal_effort_formula="Chao1/N = 1 + (1-p)^2",
        at_p=dict(p_0934=equal_effort_factor(0.934),
                  p_0801=equal_effort_factor(0.801),
                  p_0670=equal_effort_factor(0.670)),
        exact_at_observed=float(exact_factor),
        observed_factor=float(effort_factor_obs),
        statement=("Chao1 assumes equal per-bicycle capture propensity in the "
                   "two windows; Chapman does not.  With p1=0.934 and "
                   "p2=0.670 the homogeneous model alone produces the +7.8% "
                   "excess, so Chao1 cannot separate heterogeneity from the "
                   "9-day vs 5-day effort difference")),
    resample_5day=dict(
        draws=200, seed=20261118,
        design="five of the nine recorded first-window days, second window fixed",
        p5=float(np.percentile(boot, 5)),
        p50=float(np.percentile(boot, 50)),
        p95=float(np.percentile(boot, 95)),
        min=float(boot.min()), max=float(boot.max()),
        share_below_chapman=float((boot < chap).mean()),
        distinct_values=int(len(set(np.round(boot, 6)))),
        named_splits=named,
        named_note=("reviewer's 30-draw check quoted 540,086 (10-14) and "
                    "494,845 (15-19); this pipeline gives 539,180 and "
                    "494,284, a 0.1-0.2% day-boundary offset, same sign and "
                    "same crossing below the Chapman point"),
        note=("the spread and the 15-19 block falling below 495,540 show that "
              "Chao1 is not a stable lower bound in this design")),
    interval=dict(
        design_independent_lower=484754.0,
        chao1_scenario=float(chao1_raw),
        old=[484754.0, 506635.0],
        note=("M16 does not revise the paper's model interval.  The only "
              "design-independent statement is N >= the observed 14-day "
              "union 484,754.  534,039 is a scenario reference, not a lower "
              "bound")),
    rides_per_bike_day=dict(
        numerator=RIDES_DAY,
        numerator_source="q1_params.json daily_trips / 0.128",
        values={
            "F_484754_ceiling": float(RIDES_DAY / 484754.0),
            "F_495540": float(old_rate),
            "F_534039_scenario": float(RIDES_DAY / chao1_raw)},
        conclusion=("4.33 is an upper estimate because the Chapman fleet is "
                    "biased low; the design-independent ceiling is 4.42 "
                    "(N >= 484,754).  The 4.01 figure holds only in the "
                    "scenario F = 534,039")),
)

with open(os.path.join(OUT, "fleet_estimates.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)

print(json.dumps(dict(
    counts=out["counts"], chapman=out["chapman_pooled"]["estimate"],
    strata=[(s["label"], s["n1"], s["n2"], round(s["recapture_rate"], 4))
            for s in strata],
    zero=(zero["n2"], zero["w2_trips_mean"], zero["recaptured_w2_trips_mean"]),
    trend=(round(z_trend, 1), f"{p:.3g}"),
    homogeneity=(round(N_h, 1), round(p1_h, 4), round(p2_h, 4),
                 round(chao1_homog, 1), round(chao1_raw - chao1_homog, 2)),
    resample=(round(float(np.percentile(boot, 5)), 0),
              round(float(np.percentile(boot, 50)), 0),
              round(float(np.percentile(boot, 95)), 0),
              int((boot < chap).sum())),
    dN=(round(dN_new_captured, 3), round(dN_retire_recaptured, 3)),
    rates={k: round(v, 4) for k, v in out["rides_per_bike_day"]["values"].items()},
), ensure_ascii=False, indent=1))
