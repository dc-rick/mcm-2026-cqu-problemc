"""Step 4: behavioural structure - bike continuity, gaps, distance-by-hour, user segments."""
import argparse
import os
import numpy as np, pandas as pd
from lib import DATA

parser = argparse.ArgumentParser(description="Behavioural diagnostics and user features.")
parser.add_argument("--force", action="store_true",
                    help="overwrite an existing data/user_features.parquet")
args = parser.parse_args()

pd.set_option("display.width", 220)
df = pd.read_parquet(os.path.join(DATA, "trips.parquet"))
df = df[(df.start_lat > 39.4) & (df.start_lon > 115.8) & (df.start_lat < 40.4) & (df.start_lon < 117.0)].copy()
df["hour"] = df.starttime.dt.hour
df["date"] = df.starttime.dt.normalize()
core = df[(df.date >= "2017-05-10") & (df.date <= "2017-05-19")].copy()

R = 6371.0
p1, p2 = np.radians(core.start_lat), np.radians(core.end_lat)
dl = np.radians(core.end_lon - core.start_lon); dp = p2 - p1
a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
core["dist_km"] = 2*R*np.arcsin(np.sqrt(a))

print("== bike continuity: does trip k end where trip k+1 starts? ==")
d = core.sort_values(["bikeid", "starttime"])[["bikeid", "starttime", "geohashed_start_loc", "geohashed_end_loc"]].copy()
d["next_start_loc"] = d.groupby("bikeid").geohashed_start_loc.shift(-1)
d["next_start"] = d.groupby("bikeid").starttime.shift(-1)
pair = d.dropna(subset=["next_start"])
match = (pair.geohashed_end_loc == pair.next_start_loc)
print("pairs", len(pair), "exact end==next start", match.mean())
print("gap minutes quantiles:", (pair.next_start - pair.starttime).dt.total_seconds().div(60)
      .describe(percentiles=[.01,.05,.1,.25,.5,.75,.9,.99]).round(1).to_string())

print("\n== implication for ride duration ==")
gapmin = (pair.next_start - pair.starttime).dt.total_seconds()/60
for thr in (5, 10, 15, 20, 30):
    print(f"  share of gaps < {thr} min: {(gapmin < thr).mean():.4f}")

print("\n== distance by hour ==")
print(core.groupby("hour").dist_km.agg(["mean","median","count"]).round(3).to_string())

print("\n== trip purpose proxy: is the trip a round trip within the day? ==")
# same user same day, start of trip equals end of an earlier trip
u = core.sort_values(["userid", "starttime"])
u["key_s"] = u.userid.astype(str) + "_" + u.date.astype(str) + "_" + u.geohashed_start_loc
u["key_e"] = u.userid.astype(str) + "_" + u.date.astype(str) + "_" + u.geohashed_end_loc
first_s = u.drop_duplicates("key_s").set_index("key_s").index
pair_hits = u.key_e.isin(set(first_s))
print("trips whose end cell later appears as that user's start cell the same day:",
      pair_hits.mean())

print("\n== repeat OD per user across days ==")
od = core.groupby(["userid", "geohashed_start_loc", "geohashed_end_loc"]).size()
print("OD pairs per user: mean", od.groupby("userid").size().mean(),
      "share of trips in an OD pair reused >=2 times:",
      od[od >= 2].sum()/od.sum())

print("\n== per-user behavioural features (for segmentation) ==")
g = core.groupby("userid")
feat = pd.DataFrame({
    "trips": g.size(),
    "dist_mean": g.dist_km.mean(),
    "dist_std": g.dist_km.std(),
    "n_hours": g.hour.nunique(),
    "peak_share": core.assign(pk=core.hour.isin([7,8,9,17,18,19])).groupby("userid").pk.mean(),
    "weekend_share": core.assign(we=core.starttime.dt.dayofweek >= 5).groupby("userid").we.mean(),
    "n_days": g.date.nunique(),
})
print(feat.describe().round(3).to_string())
out = os.path.join(DATA, "user_features.parquet")
if os.path.exists(out) and not args.force:
    print("exists, not overwriting:", out, "(use --force to overwrite)")
else:
    feat.to_parquet(out)
    print("wrote", out)
print("\ncorrelation matrix:\n", feat.corr().round(2).to_string())
