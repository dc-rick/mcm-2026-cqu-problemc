"""Step 1: exploratory facts about the dataset (all numbers used in the paper come from here)."""
import os
import numpy as np, pandas as pd
from lib import DATA

df = pd.read_parquet(os.path.join(DATA, "trips.parquet"))
pd.set_option("display.width", 200)

df["date"] = df.starttime.dt.date
df["hour"] = df.starttime.dt.hour
df["dow"] = df.starttime.dt.dayofweek

print("== basic ==")
print("rows", len(df), "unique orderid", df.orderid.nunique(),
      "users", df.userid.nunique(), "bikes", df.bikeid.nunique())
print("date span", df.starttime.min(), df.starttime.max(),
      "days", df.date.nunique())
print("biketype values", df.biketype.value_counts().to_dict())
print("is biketype constant per bike?",
      (df.groupby("bikeid").biketype.nunique() > 1).sum(), "bikes with >1 type")
print("is biketype constant per user?",
      (df.groupby("userid").biketype.nunique() > 1).sum(), "users with >1 type")

print("\n== trips per day ==")
g = df.groupby("date").agg(trips=("orderid", "size"),
                           users=("userid", "nunique"),
                           bikes=("bikeid", "nunique"))
print(g)
gd = g[g.index.map(str).isin([f"2017-05-{d:02d}" for d in range(10,25)])]
gd = gd[gd.trips > 200000]
print("median daily trips over well-covered days:", gd.trips.median(), "n=", len(gd))

print("\n== hourly ==")
print(df.groupby("hour").size().reindex(range(24)).to_string())

print("\n== ride distance ==")
R = 6371.0
p1, p2 = np.radians(df.start_lat), np.radians(df.end_lat)
dl = np.radians(df.end_lon - df.start_lon); dp = p2 - p1
a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
df["dist_km"] = 2 * R * np.arcsin(np.sqrt(a))
print(df.dist_km.describe(percentiles=[.05, .25, .5, .75, .9, .95, .99]).to_string())
print("share same start/end geohash:", (df.geohashed_start_loc == df.geohashed_end_loc).mean())
print("share < 100m:", (df.dist_km < 0.1).mean(), " > 5km:", (df.dist_km > 5).mean(),
      " > 20km:", (df.dist_km > 20).mean())

print("\n== spatial coverage ==")
s = df.groupby("geohashed_start_loc").size().sort_values(ascending=False)
print("distinct start cells", len(s), "end cells", df.geohashed_end_loc.nunique())
print("top 10 cells share:", s.head(10).sum() / len(df),
      "| cells covering 80% of trips:", (s.cumsum() / len(df) <= .8).sum() + 1)

print("\n== outliers in coordinates ==")
print("start lat<30 or lon<110:", ((df.start_lat < 30) | (df.start_lon < 110)).sum())
print(df.loc[(df.start_lat < 30) | (df.start_lon < 110), "geohashed_start_loc"].value_counts().head(5))

print("\n== per-user activity ==")
u = df.groupby("userid").size()
print(u.describe(percentiles=[.5, .9, .99]).to_string())
print("share of trips from top 1% users:", u.nlargest(int(len(u) * .01)).sum() / len(df))
print("\n== per-bike activity ==")
b = df.groupby("bikeid").size()
print(b.describe(percentiles=[.5, .9, .99]).to_string())

print("\n== data quality: same bike overlapping trips ==")
d = df.sort_values(["bikeid", "starttime"])[["bikeid", "starttime", "dist_km"]].copy()
d["end_est"] = d.starttime + pd.to_timedelta(np.maximum(d.dist_km, 0.2) / 12.0 * 3600, unit="s")
d["next_start"] = d.groupby("bikeid").starttime.shift(-1)
d["next_bike"] = d.groupby("bikeid").bikeid.shift(-1)
same = d.next_bike == d.bikeid
print("consecutive pairs", same.sum(), "overlap (next starts before est. end):",
      (same & (d.next_start < d.end_est)).sum(),
      "=", (same & (d.next_start < d.end_est)).sum() / same.sum())

print("\n== duplicate orderid ==", df.orderid.duplicated().sum())
