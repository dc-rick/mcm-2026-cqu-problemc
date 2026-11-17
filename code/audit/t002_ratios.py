"""T002 step 1: recompute the consecutive-order match ratios under explicit
definitions.  Read-only on data/trips.parquet; outputs to data/runs/T002/.
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

df = pd.read_parquet(os.path.join(DATA, "trips.parquet"))
n_raw = len(df)
df = df[(df.start_lat > 39.4) & (df.start_lon > 115.8) &
        (df.start_lat < 40.4) & (df.start_lon < 117.0)].copy()
n_bbox = len(df)
df["date"] = df.starttime.dt.normalize()
R = 6371000.0


def make_pairs(sub):
    """Sort one window and pair each trip with the next trip of the same bike
    inside that window (matches code/04_behaviour.py)."""
    sub = sub.sort_values(["bikeid", "starttime"], kind="mergesort").reset_index(drop=True)
    g = sub.groupby("bikeid", sort=False)
    sub["next_start_loc"] = g.geohashed_start_loc.shift(-1)
    sub["next_start"] = g.starttime.shift(-1)
    sub["next_user"] = g.userid.shift(-1)
    sub["next_start_lat"] = g.start_lat.shift(-1)
    sub["next_start_lon"] = g.start_lon.shift(-1)
    sub["next_date"] = sub.next_start.dt.normalize()
    pair = sub.dropna(subset=["next_start"]).copy()
    p1 = np.radians(pair.end_lat.astype(float))
    p2 = np.radians(pair.next_start_lat.astype(float))
    dl = np.radians(pair.next_start_lon.astype(float) - pair.end_lon.astype(float))
    dp = p2 - p1
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    pair["gap_m"] = 2 * R * np.arcsin(np.sqrt(a))
    pair["exact"] = (pair.geohashed_end_loc == pair.next_start_loc)
    pair["same_user"] = pair.userid == pair.next_user
    pair["same_day"] = pair.date == pair.next_date
    return pair


core = make_pairs(df[(df.date >= "2017-05-10") & (df.date <= "2017-05-19")])
all14 = make_pairs(df[(df.date >= "2017-05-10") & (df.date <= "2017-05-24")])


def rates(sub):
    n = len(sub)
    return dict(
        n_pairs=int(n),
        exact=float(sub.exact.mean()),
        le160=float((sub.gap_m <= 160).mean()),
        le250=float((sub.gap_m <= 250).mean()),
        same_user_share=float(sub.same_user.mean()),
        same_day_share=float(sub.same_day.mean()),
        gap_p50=float(sub.gap_m.median()),
        gap_p90=float(sub.gap_m.quantile(0.90)),
    )


variants = {
    "A_paper_core_allpairs": rates(core),
    "B_core_sameday_only": rates(core[core.same_day]),
    "C_core_sameuser_only": rates(core[core.same_user]),
    "D_core_sameuser_sameday": rates(core[core.same_user & core.same_day]),
    "E_all14_allpairs": rates(all14),
    "F_all14_sameday": rates(all14[all14.same_day]),
    "G_core_crossday_only": rates(core[~core.same_day]),
}
out = dict(raw_rows=n_raw, bbox_rows=n_bbox,
           core_rows=int(len(df[(df.date >= "2017-05-10") &
                               (df.date <= "2017-05-19")])),
           variants=variants)
json.dump(out, io.open(os.path.join(RUN, "ratios.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

# Distance histogram of the paper-core pairs near the thresholds and the
# distance values themselves (decoded cell centroids -> lattice distances).
core = core.copy()
core["dist_round"] = core.gap_m.round(1)
top = core.dist_round.value_counts().head(20)
io.open(os.path.join(RUN, "distance_modes.txt"), "w", encoding="utf-8").write(
    "paper-core pairs: top-20 rounded endpoint distances (m)\n"
    + top.to_string() + "\n\nthreshold counts\n"
    + core.groupby(core.gap_m <= 160).size().to_string() + "\n"
    + core.groupby(core.gap_m <= 250).size().to_string() + "\n")

print(json.dumps(out, ensure_ascii=False, indent=1))
