"""T005: return_share definitions and recomputation.

Definitions compared:
  same_cell      trip starts and ends in the same 0.005-degree cell (o == d)
  origin_revisit 11_q1_extra.py's feature: a trip starts in an origin cell the
                 same user already departed from earlier that day
  dest_revisit   a trip ends in a cell the same user used as an origin earlier
                 that day (including its own origin, so same_cell implies it)

Sample: all 10 days (2017-05-10..19), both cell ids present, matching
11_q1_extra.py's behaviour sample.  Segments are read from the existing
data/user_seg.parquet (read-only).
"""
import io
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import lib  # noqa: E402
import t005_common as C  # noqa: E402

RUN = os.path.join(ROOT, "data", "runs", "T005")
os.makedirs(RUN, exist_ok=True)

df, gt, cells, D = C.load_model_data()
n = len(cells)
prob = gt[["userid", "date", "hour", "starttime", "o", "d"]].copy()
print("sample trips", len(prob), "users", prob.userid.nunique())

seg = pd.read_parquet(os.path.join(lib.DATA, "user_seg.parquet"))
print("user_seg", seg.shape, seg.columns.tolist(), seg.index.name)
if "userid" not in seg.columns:
    seg = seg.reset_index()
seg = seg[["userid", "seg"]].drop_duplicates("userid")
prob = prob.merge(seg, on="userid", how="left")
print("unmatched segment users trips", int(prob.seg.isna().sum()))


def rate(x):
    return float(np.mean(x))


def wilson(p, n):
    z = 1.959963984540054
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(mid - half), float(mid + half)]


# ---------------------------------------------------------------- same cell
prob["same_cell"] = (prob.o.values == prob.d.values).astype(np.float64)
overall_p = rate(prob.same_cell)
print("same-cell overall", overall_p, "n", len(prob),
      "CI", wilson(overall_p, len(prob)))

by_band = {}
prob["band"] = pd.cut(prob.hour, [-1, 5, 9, 15, 19, 23],
                      labels=False).astype(int)
for b, name in enumerate(C.BAND_NAMES):
    s = prob[prob.band == b]
    by_band[name] = dict(trips=int(len(s)), same_cell=rate(s.same_cell))
by_date = {d: dict(trips=int(len(s)), same_cell=rate(s.same_cell))
           for d, s in prob.groupby("date")}

# ---------------------------------------------------------- origin revisit
u = prob.sort_values(["userid", "starttime"])
key = (u.userid.astype(str) + "|" + u.date + "|" + u.o.astype(str))
u["origin_revisit"] = key.duplicated()
prob["origin_revisit"] = 0.0
prob.loc[u.index, "origin_revisit"] = u.origin_revisit.astype(np.float64)
print("origin_revisit overall", rate(prob.origin_revisit))

# ------------------------------------------------------------ dest revisit
o_ev = prob[["userid", "date", "o", "starttime"]].rename(
    columns={"o": "cell"}).copy()
o_ev["origin_time"] = o_ev.starttime
o_first = o_ev.groupby(["userid", "date", "cell"], as_index=False).origin_time.min()
d_ev = prob[["userid", "date", "d", "starttime"]].rename(
    columns={"d": "cell"}).copy()
m = d_ev.merge(o_first, on=["userid", "date", "cell"], how="inner")
m = m[m.origin_time <= m.starttime]
hit = m.drop_duplicates(["userid", "date", "cell", "starttime"])
hit = hit[["userid", "date", "cell", "starttime"]].copy()
hit["dest_revisit"] = 1.0
prob = prob.reset_index(drop=True)
prob = prob.merge(hit, left_on=["userid", "date", "d", "starttime"],
                  right_on=["userid", "date", "cell", "starttime"],
                  how="left", suffixes=("", "_hit"))
prob["dest_revisit"] = prob.dest_revisit.fillna(0.0)
print("dest_revisit overall", rate(prob.dest_revisit), "merge rows", len(m),
      "trips", len(prob))

# ----------------------------------------------------- per-user aggregation
g = prob.groupby("userid")
per_user = pd.DataFrame({
    "trips": g.size(),
    "n_days": g.date.nunique(),
    "same_cell": g.same_cell.mean(),
    "origin_revisit": g.origin_revisit.mean(),
    "dest_revisit": g.dest_revisit.mean(),
}).reset_index()
per_user["trips_per_active_day"] = per_user.trips / per_user.n_days
per_user = per_user.merge(seg, on="userid", how="left")

seg_rows = []
for sid, s in prob.groupby("seg"):
    pu = per_user[per_user.seg == sid]
    rec = dict(
        seg=int(sid), users=int(len(pu)), trips=int(len(s)),
        trip_weighted_same_cell=rate(s.same_cell),
        user_mean_same_cell=float(pu.same_cell.mean()),
        user_median_same_cell=float(pu.same_cell.median()),
        trip_weighted_origin_revisit=rate(s.origin_revisit),
        user_mean_origin_revisit=float(pu.origin_revisit.mean()),
        trip_weighted_dest_revisit=rate(s.dest_revisit),
        user_mean_dest_revisit=float(pu.dest_revisit.mean()),
        user_mean_n_days=float(pu.n_days.mean()),
        user_mean_trips_total=float(pu.trips.mean()),
        user_mean_trips_per_active_day=float(pu.trips_per_active_day.mean()),
    )
    seg_rows.append(rec)
seg_rows.sort(key=lambda r: -r["user_mean_trips_total"])

# paper segment labels, keyed by the seg id published in data/q1_segments.csv
paper = {0: "Regular multi-purpose", 1: "Low frequency", 2: "Commuter",
         3: "Heavy short-trip", 4: "Weekend / leisure"}
for r in seg_rows:
    r["paper_label"] = paper.get(r["seg"], "unknown")
    r["paper_claim_quarter"] = (r["paper_label"] == "Heavy short-trip")

overall = dict(
    trips=int(len(prob)), users=int(prob.userid.nunique()),
    trip_weighted_same_cell=overall_p,
    same_cell_ci95=wilson(overall_p, len(prob)),
    user_mean_same_cell=float(per_user.same_cell.mean()),
    user_median_same_cell=float(per_user.same_cell.median()),
    trip_weighted_origin_revisit=rate(prob.origin_revisit),
    user_mean_origin_revisit=float(per_user.origin_revisit.mean()),
    trip_weighted_dest_revisit=rate(prob.dest_revisit),
    user_mean_dest_revisit=float(per_user.dest_revisit.mean()),
)

heavy = [r for r in seg_rows if r["paper_label"] == "Heavy short-trip"][0]
out = dict(
    definition=dict(
        same_cell="origin cell == destination cell on the 0.005-degree grid "
                  "(the task's return_share definition)",
        origin_revisit="11_q1_extra.py feature: current origin cell equals an "
                       "origin this user already used earlier the same day",
        dest_revisit="trip destination equals a cell this user used as an "
                     "origin earlier the same day (including its own start)"),
    sample=dict(days=lib.ALL_DAYS, trips=int(len(prob)),
                note="both origin and destination inside the 5,263-cell study "
                     "area; same sample as 11_q1_extra.py"),
    overall=overall,
    by_band=by_band,
    by_date=by_date,
    by_segment=seg_rows,
    heavy_short_trip=dict(
        paper_sentence="The heavy short-trip user rides 14 times a day over "
                       "0.67 km, with a quarter of trips returning to the "
                       "same cell (paper/sec4_q1.tex:190)",
        feature_value_in_q1_segments=0.2571796072029219,
        recomputed_origin_revisit_trip_weighted=heavy[
            "trip_weighted_origin_revisit"],
        recomputed_origin_revisit_user_mean=heavy[
            "user_mean_origin_revisit"],
        same_cell_trip_weighted=heavy["trip_weighted_same_cell"],
        same_cell_user_mean=heavy["user_mean_same_cell"],
        dest_revisit_trip_weighted=heavy["trip_weighted_dest_revisit"],
    ),
    reading=("The quarter in q1_segments.csv is the origin-revisit feature, "
             "not a same-cell round trip.  Same-cell (o==d) shares are "
             "reported separately in overall/by_segment."),
)
json.dump(out, io.open(os.path.join(RUN, "return_share.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
pd.DataFrame(seg_rows).to_csv(os.path.join(RUN, "return_share_segments.csv"),
                              index=False)
print("overall", json.dumps(overall, indent=1))
print(pd.DataFrame(seg_rows)[["seg", "paper_label", "trips",
                              "trip_weighted_same_cell", "user_mean_same_cell",
                              "trip_weighted_origin_revisit",
                              "trip_weighted_dest_revisit"]].to_string(index=False))
print("saved", os.path.join(RUN, "return_share.json"))
