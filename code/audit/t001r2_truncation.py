"""T001-R2 step 1: is 20-24 May a truncated sample?  Per-day volume, hourly
coverage and distinct bicycles, against 10-19 May.
"""
import io
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
RUN = os.path.join(DATA, "runs", "T001R2")
os.makedirs(RUN, exist_ok=True)

df = pd.read_parquet(os.path.join(DATA, "trips.parquet"))
df = df[(df.start_lat > 39.4) & (df.start_lon > 115.8) &
        (df.start_lat < 40.4) & (df.start_lon < 117.0)].copy()
df["date"] = df.starttime.dt.normalize()
df["hour"] = df.starttime.dt.hour

rows = []
hourly = {}
for d, g in df.groupby("date"):
    hc = g.groupby("hour").size().reindex(range(24), fill_value=0)
    hourly[str(d)[:10]] = hc
    rows.append(dict(
        date=str(d)[:10], trips=int(len(g)), bikes=int(g.bikeid.nunique()),
        hours_with_orders=int((hc > 0).sum()),
        min_hour_count=int(hc.min()), zero_hours=int((hc == 0).sum()),
        zero_hour_list=[int(h) for h in hc.index[hc == 0]],
        night_share=float(hc.loc[[0, 1, 2, 3, 4, 5]].sum() / len(g)),
    ))
daily = pd.DataFrame(rows)
full = daily[daily.date.isin(["2017-05-10", "2017-05-11", "2017-05-12",
                              "2017-05-13", "2017-05-14", "2017-05-15",
                              "2017-05-16", "2017-05-18", "2017-05-19"])]
trunc = daily[daily.date.between("2017-05-20", "2017-05-24")]

# hourly profile comparison: normalise each day to shares, compare with the
# mean profile of the nine full days
prof = {}
for d, g in df.groupby("date"):
    hc = g.groupby("hour").size().reindex(range(24), fill_value=0).to_numpy(float)
    prof[str(d)[:10]] = hc / hc.sum()
base = np.mean([prof[d] for d in full.date], axis=0)
prof_cmp = {}
for d in list(full.date) + list(trunc.date):
    p = prof[d]
    prof_cmp[d] = dict(
        corr=float(np.corrcoef(p, base)[0, 1]),
        max_abs_pp=float(np.abs(p - base).max() * 100),
        l1_pp=float(np.abs(p - base).sum() * 100),
    )

# effort inside the hours a truncated day actually covers: compare its volume
# with the full-day mean volume restricted to the same hours.
full_hour_mean = np.mean([hourly[d].to_numpy(float) for d in full.date], axis=0)
effort = {}
for d in trunc.date:
    hc = hourly[d].to_numpy(float)
    covered = hc > 0
    expected = full_hour_mean[covered].sum()
    effort[d] = dict(
        covered_hours=int(covered.sum()),
        observed=int(hc.sum()),
        expected_at_full_rate=float(expected),
        ratio_within_covered_hours=float(hc.sum() / expected),
        full_day_ratio=float(hc.sum() / full_hour_mean.sum()),
    )

# distinct-bike overlap between the two windows
w1 = set(df[df.date <= "2017-05-19"].bikeid.unique())
w2 = set(df[df.date >= "2017-05-20"].bikeid.unique())
n1, n2, m = len(w1), len(w2), len(w1 & w2)
chapman = (n1 + 1) * (n2 + 1) / (m + 1) - 1

out = dict(
    daily=rows,
    summary=dict(
        full_mean_trips=float(full.trips.mean()),
        trunc_mean_trips=float(trunc.trips.mean()),
        volume_ratio=float(trunc.trips.mean() / full.trips.mean()),
        full_mean_bikes=float(full.bikes.mean()),
        trunc_mean_bikes=float(trunc.bikes.mean()),
        bike_ratio=float(trunc.bikes.mean() / full.bikes.mean()),
        full_min_hours=int(full.hours_with_orders.min()),
        trunc_min_hours=int(trunc.hours_with_orders.min()),
        full_zero_hours=int(full.zero_hours.sum()),
        trunc_zero_hours=int(trunc.zero_hours.sum()),
    ),
    hourly_profile=prof_cmp,
    effort_within_covered_hours=effort,
    union=dict(n1=n1, n2=n2, overlap=m,
               union14=len(w1 | w2), chapman=float(chapman)),
)
json.dump(out, io.open(os.path.join(RUN, "truncation.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
