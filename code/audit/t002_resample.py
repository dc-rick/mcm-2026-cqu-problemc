"""T002 step 3: thinning resampling of the fleet estimator.

For retention q, keep each observed order independently with probability q,
recompute T' = orders kept, D' = distinct bikes kept, solve
mu'/(1-exp(-mu')) = T'/D', and F' = T'/mu'.  The Poisson-occupancy algebra says
F' is invariant to q; the run measures the finite-sample deviation.

Window/filter: lib.load_trips (39.4<lat<40.4, 115.8<lon<117.0), reference day
2017-05-16 (paper's reference weekday), plus the seven typical days pooled.
"""
import io
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code"))
import lib  # noqa: E402

RUN = os.path.join(ROOT, "data", "runs", "T002")
os.makedirs(RUN, exist_ok=True)


def fleet(trips, bikes):
    r = trips / bikes
    mu = brentq(lambda m: m / (1 - np.exp(-m)) - r, 1e-6, 50)
    return trips / mu, mu, r


df = lib.load_trips(None)
day = df[df.date == "2017-05-16"]
T0, D0 = len(day), day.bikeid.nunique()
F0, mu0, r0 = fleet(T0, D0)

RATES = (0.05, 0.10, 0.20, 0.33, 0.50)
NSEED = 200
rng = np.random.default_rng(20260517)
rows = []
for q in RATES:
    fs = []
    for s in range(NSEED):
        keep = rng.random(T0) < q
        sub = day[keep]
        T, D = len(sub), sub.bikeid.nunique()
        F, mu, r = fleet(T, D)
        fs.append(F)
    fs = np.array(fs)
    rows.append(dict(
        q=q, n_seed=NSEED, mean=float(fs.mean()), sd=float(fs.std(ddof=1)),
        p05=float(np.percentile(fs, 5)), p50=float(np.percentile(fs, 50)),
        p95=float(np.percentile(fs, 95)),
        min=float(fs.min()), max=float(fs.max()),
        dev_mean=float(fs.mean() / F0 - 1),
        dev_max_abs=float(np.abs(fs / F0 - 1).max()),
        share_gt4pct=float((np.abs(fs / F0 - 1) > 0.04).mean()),
        example_T=int(round(T0 * q)),
    ))

# original paper grid, same day, for comparison with sensitivity.json
orig = []
for q in (0.05, 0.128, 0.25, 0.5):
    keep = rng.random(T0) < q
    sub = day[keep]
    F, mu, r = fleet(len(sub), sub.bikeid.nunique())
    orig.append(dict(q=q, F=float(F), dev=float(F / F0 - 1)))
# exact reproduction of code/60_sensitivity.py seed 0
rng0 = np.random.default_rng(0)
orig_seed0 = []
for q in (0.05, 0.128, 0.25, 0.5):
    keep = rng0.random(T0) < q
    sub = day[keep]
    F, mu, r = fleet(len(sub), sub.bikeid.nunique())
    orig_seed0.append(dict(q=q, F=float(F), dev=float(F / F0 - 1)))

# pooled typical days: thin each day at q, estimate F per day, average; the
# reference is the average full-sample F over the same seven days (not F0,
# which is 5/16 alone).
typ = df[df.date.isin(lib.TYP_DAYS)]
full_by_day = []
for d, g in typ.groupby("date"):
    F, mu, r = fleet(len(g), g.bikeid.nunique())
    full_by_day.append(F)
F_typ = float(np.mean(full_by_day))
pooled = []
for q in RATES:
    fs = []
    for d, g in typ.groupby("date"):
        T = len(g)
        keep = rng.random(T) < q
        sub = g[keep]
        F, mu, r = fleet(len(sub), sub.bikeid.nunique())
        fs.append(F)
    pooled.append(dict(q=q, mean=float(np.mean(fs)), sd=float(np.std(fs)),
                        dev=float(np.mean(fs) / F_typ - 1)))

out = dict(
    reference=dict(date="2017-05-16", T=T0, D=D0, ratio=r0, mu=mu0, F=F0,
                   filter="lib.load_trips: 39.4<start_lat<40.4, "
                          "115.8<start_lon<117.0"),
    rates=list(RATES), n_seed=NSEED, resample=rows,
    paper_grid=orig, paper_grid_seed0=orig_seed0,
    pooled_typical_days=dict(reference_F=F_typ, rows=pooled),
    verdict=dict(
        max_abs_dev_all=float(max(r["dev_max_abs"] for r in rows)),
        any_mean_gt4=bool(any(abs(r["dev_mean"]) > 0.04 for r in rows)),
        any_seed_gt4=bool(any(r["share_gt4pct"] > 0 for r in rows)),
    ),
)
json.dump(out, io.open(os.path.join(RUN, "resample.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
pd.DataFrame(rows).to_csv(os.path.join(RUN, "resample.csv"), index=False,
                          encoding="utf-8-sig")
print(json.dumps(out, ensure_ascii=False, indent=1))
