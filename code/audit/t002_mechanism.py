"""T002 step 2: mechanistic estimate of p from the adjacency test.

Model.  Each true order is retained independently with probability p.
For two consecutive *observed* orders of one bicycle, the number of missed
orders between them is geometric, so P(no missed order) = p.  If no order is
missed the bicycle was not moved, and the second order starts at the same
physical point as the first ended; decoded geohash cells may differ because of
geohash/GPS jitter.  If at least one order is missed the bicycle was moved, and
the endpoint distance follows the displacement distribution of a trip.

Hence, for a distance threshold d (end of trip k to start of trip k+1):
    m_d = P(observed distance <= d) = p*a_d + (1-p)*b_d
where a_d = P(<=d | adjacent) and b_d = P(<=d | moved).

b_d is estimated from placebo pairs (k, k+2) and (k, k+3): an observed trip
between them proves the bicycle moved.  With a_250 ~ 1 (a same-spot return is
within one geohash cell diagonal ~192 m plus jitter), p is identified at
d = 250 m:  p = (m_250 - b_250) / (1 - b_250).  a_0 and a_160 then follow from
the lower bands and serve as a consistency check (they must lie in [0,1]).
"""
import io
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
RUN = os.path.join(DATA, "runs", "T002")
os.makedirs(RUN, exist_ok=True)
R = 6371000.0

df = pd.read_parquet(os.path.join(DATA, "trips.parquet"))
df = df[(df.start_lat > 39.4) & (df.start_lon > 115.8) &
        (df.start_lat < 40.4) & (df.start_lon < 117.0)].copy()
df["date"] = df.starttime.dt.normalize()
core = df[(df.date >= "2017-05-10") & (df.date <= "2017-05-19")].copy()
core = core.sort_values(["bikeid", "starttime"], kind="mergesort").reset_index(drop=True)

g = core.groupby("bikeid", sort=False)


def shiftcol(col, k):
    return g[col].shift(-k)


core["next_loc"] = shiftcol("geohashed_start_loc", 1)
core["next_lat"] = shiftcol("start_lat", 1)
core["next_lon"] = shiftcol("start_lon", 1)
core["next_start"] = shiftcol("starttime", 1)

for K in (2, 3):
    core[f"k{K}_loc"] = shiftcol("geohashed_start_loc", K)
    core[f"k{K}_lat"] = shiftcol("start_lat", K)
    core[f"k{K}_lon"] = shiftcol("start_lon", K)
    core[f"k{K}_start"] = shiftcol("starttime", K)


def dist_m(lat1, lon1, lat2, lon2):
    p1 = np.radians(lat1.astype(float))
    p2 = np.radians(lat2.astype(float))
    dl = np.radians(lon2.astype(float) - lon1.astype(float))
    dp = p2 - p1
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def band_stats(mask, dist, exact):
    m = mask
    n = int(m.sum())
    d = dist[m]
    e = exact[m]
    return dict(
        n=n,
        exact=float(e.mean()) if n else None,
        le160=float((d <= 160).mean()) if n else None,
        le192=float((d <= 192.6).mean()) if n else None,
        le250=float((d <= 250).mean()) if n else None,
        le330=float((d <= 330).mean()) if n else None,
    )


# real consecutive observed pairs (may be adjacent or not)
mask1 = core.next_start.notna()
d1 = dist_m(core.end_lat, core.end_lon, core.next_lat, core.next_lon)
e1 = core.geohashed_end_loc == core.next_loc
m = band_stats(mask1, d1, e1)

placebo = {}
for K in (2, 3):
    mask = core[f"k{K}_start"].notna()
    dk = dist_m(core.end_lat, core.end_lon, core[f"k{K}_lat"], core[f"k{K}_lon"])
    ek = core.geohashed_end_loc == core[f"k{K}_loc"]
    placebo[f"skip{K-1}"] = band_stats(mask, dk, ek)

# identification at 250 m (a_250 = 1)
b250 = placebo["skip1"]["le250"]
p_hat = (m["le250"] - b250) / (1 - b250)
# implied a_d at the lower bands, using b_d from skip-1 placebo
a = {}
for key, band in (("exact", "exact"), ("le160", "le160"),
                  ("le192", "le192"), ("le250", "le250")):
    bd = placebo["skip1"][band]
    md = m[band]
    a[key] = (md - (1 - p_hat) * bd) / p_hat
# same with skip-2 placebo as sensitivity
b250_2 = placebo["skip2"]["le250"]
p_hat2 = (m["le250"] - b250_2) / (1 - b250_2)
a2 = {}
for key, band in (("exact", "exact"), ("le160", "le160"),
                  ("le192", "le192"), ("le250", "le250")):
    bd = placebo["skip2"][band]
    a2[key] = (m[band] - (1 - p_hat2) * bd) / p_hat2

# Bracket from the two limiting assumptions:
# lower: exact match, ignoring false matches -> p >= exact match (if b0 ~ 0)
# upper: 250 m, ignoring false matches -> p <= m_250 (if a_250 = 1)
bracket = dict(lower_exact=m["exact"], upper_le250=m["le250"],
               lower_bracket_after_b0=(m["exact"] - placebo["skip1"]["exact"]) /
               (1 - placebo["skip1"]["exact"]))

out = dict(
    definition=dict(
        window="2017-05-10..2017-05-19 (10 calendar days, 9 with records)",
        bbox="39.4<start_lat<40.4, 115.8<start_lon<117.0",
        pairing="same bikeid, consecutive observed trips inside the window, next start after current start; cross-day pairs kept; userid not required",
        distance="haversine between decoded geohash cell centroids of trip k end and trip k+1 start (m)",
        bands="exact = same geohash string; le160 = <=160 m; le250 = <=250 m; inclusive",
    ),
    consecutive=m,
    placebo=placebo,
    identification=dict(
        p_250_skip1=p_hat, p_250_skip2=p_hat2,
        b250_skip1=b250, b250_skip2=b250_2,
        a_implied_skip1=a, a_implied_skip2=a2,
        bracket=bracket,
        note=("a_250 is set to 1; a_d below 1 measures geohash/GPS jitter. "
              "a values must lie in [0,1].  Placebo pairs (k,k+2) and (k,k+3) "
              "are definitely moved, so b_d from them approximates the "
              "false-match rate; the two legs make them slightly conservative "
              "(too large) as a null for exactly one missed order."),
    ),
)
json.dump(out, io.open(os.path.join(RUN, "p_estimate.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
