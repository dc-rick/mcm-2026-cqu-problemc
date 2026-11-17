"""Q1 (cont.): user segments, duration model, zone system, typical-day scenario for Q2-Q5."""
import gc
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import lib
from lib import DATA

R = 6371.0
COLS = ["userid", "biketype", "starttime", "hour", "qtr", "date", "o", "d",
        "start_lat", "start_lon", "end_lat", "end_lon"]


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1); dp = p2 - p1
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


df = lib.load_trips(lib.ALL_DAYS)
cells, df = lib.make_cells(df, lib.THR)
df = df[COLS]
n = len(cells)

# ------------------------------------------------- probabilistic 10-day set
prob = df[df.date.isin(lib.ALL_DAYS[:10]) & df.o.notna() & df.d.notna()].copy()
prob["o"] = prob.o.astype(int); prob["d"] = prob.d.astype(int)
prob["dist"] = haversine(prob.start_lat.values, prob.start_lon.values,
                         prob.end_lat.values, prob.end_lon.values)
del df; gc.collect()
print("behaviour sample:", len(prob))

# ------------------------------------------------ user behavioural features
g = prob.groupby("userid")
feat = pd.DataFrame({"trips": g.size(), "dist_mean": g.dist.mean(),
                     "n_days": g.date.nunique(), "n_cells": g.o.nunique()})
feat["peak_share"] = prob.assign(pk=prob.hour.isin([7, 8, 9, 17, 18, 19])).groupby("userid").pk.mean()
feat["weekend_share"] = prob.assign(we=prob.starttime.dt.dayofweek >= 5).groupby("userid").we.mean()

# hourly entropy, vectorised through a (user, hour) count table
ch = prob.groupby(["userid", "hour"]).size().rename("c").reset_index()
ch["p"] = ch.c / ch.groupby("userid").c.transform("sum")
ch["plogp"] = ch.p * np.log(ch.p + 1e-12)
feat["hour_entropy"] = -ch.groupby("userid").plogp.sum()

# return propensity: revisiting an origin cell already used that day
u2 = prob[["userid", "date", "o", "starttime"]].sort_values(["userid", "starttime"])
key = u2.userid.astype(str) + "|" + u2.date + "|" + u2.o.astype(str)
u2["is_return"] = key.duplicated()
feat["return_share"] = u2.groupby("userid").is_return.mean()
del ch, u2, key; gc.collect()

X = feat[["trips", "dist_mean", "hour_entropy", "peak_share", "weekend_share", "return_share"]].copy()
X["trips"] = np.log1p(X.trips)
X = X.fillna(X.median())
X = (X - X.mean()) / X.std().replace(0, 1.0)
X = X.astype(np.float32)
print("features:", X.shape)

sil = {}
for k in (3, 4, 5):
    km = KMeans(k, n_init=10, random_state=0).fit(X)
    idx = np.random.default_rng(0).choice(len(X), min(40000, len(X)), replace=False)
    sil[k] = silhouette_score(X.iloc[idx], km.labels_[idx])
print("silhouette", {k: round(v, 3) for k, v in sil.items()})
K = max(sil, key=sil.get)
km = KMeans(K, n_init=20, random_state=0).fit(X)
feat["seg"] = km.labels_

seg = feat.groupby("seg").agg(users=("trips", "size"), trips=("trips", "mean"),
                              dist=("dist_mean", "mean"), ent=("hour_entropy", "mean"),
                              peak=("peak_share", "mean"), weekend=("weekend_share", "mean"),
                              ret=("return_share", "mean"))
seg["share_users"] = seg.users / seg.users.sum()
seg["share_trips"] = feat.groupby("seg").trips.sum() / feat.trips.sum()
seg = seg.sort_values("trips")
print(seg.round(3).to_string())
seg.to_csv(DATA + r"\q1_segments.csv")
feat[["seg"]].to_parquet(DATA + r"\user_seg.parquet")

# ------------------------------------------------------- duration / speed
sub = prob[prob.date.isin(lib.TYP_DAYS)].copy()
rng = np.random.default_rng(1)
v = np.clip(rng.normal(11.5, 2.5, len(sub)), 6, 20)
dur = (1.35 * sub.dist.values / v) * 60 + 1.0
print("\nduration min: median", round(float(np.median(dur)), 2),
      "p10", round(float(np.percentile(dur, 10)), 2),
      "p90", round(float(np.percentile(dur, 90)), 2))
print("model P(duration>7.1)=", round(float((dur > 7.1).mean()), 3),
      "bound 0.01/0.128 =", round(0.01 / 0.128, 3))

from scipy.stats import lognorm
shape, loc, scale = lognorm.fit(sub.dist.values[sub.dist.values > 0], floc=0)
print(f"lognormal(distance): shape {shape:.3f}, scale {scale:.3f}, median {scale:.3f} km")

# ------------------------------------------------------------- zone system
NZ = 50
w = (np.bincount(sub.o.values, minlength=n) + np.bincount(sub.d.values, minlength=n)).astype(float)
xy = lib.cell_xy(cells)
cent = KMeans(NZ, n_init=20, random_state=0).fit(xy, sample_weight=w).cluster_centers_
zid = np.argmin(((xy[:, None, :] - cent[None, :, :]) ** 2).sum(-1), axis=1)
vc = pd.Series(zid).value_counts()
print("\nzones", vc.size, "cells/zone min", vc.min(), "median", int(vc.median()), "max", vc.max())

cells["zone"] = zid
cells.to_parquet(DATA + r"\cells.parquet")

Nz = NZ
d_hour = np.zeros((Nz, 24)); a_hour = np.zeros((Nz, 24))
np.add.at(d_hour, (zid[sub.o.values], sub.hour.values), 1.0)
np.add.at(a_hour, (zid[sub.d.values], sub.hour.values), 1.0)
d_hour /= len(lib.TYP_DAYS); a_hour /= len(lib.TYP_DAYS)
od_z = np.zeros((24, Nz, Nz), dtype=np.float32)
np.add.at(od_z, (sub.hour.values, zid[sub.o.values], zid[sub.d.values]), 1.0)
od_z /= len(lib.TYP_DAYS)
d_q = np.zeros((Nz, 96))
np.add.at(d_q, (zid[sub.o.values], sub.qtr.values), 1.0)
d_q /= len(lib.TYP_DAYS)

np.savez(DATA + r"\typical_day.npz", d_hour=d_hour, a_hour=a_hour, od_z=od_z,
         d_q=d_q, zid=zid, cent=cent, weight=w, dur_median=np.median(dur))

net = a_hour.sum(1) - d_hour.sum(1)
print("\nsample daily trips:", round(float(d_hour.sum())),
      "| zone demand min", round(float(d_hour.sum(1).min())),
      "max", round(float(d_hour.sum(1).max())))
print("zones with |net|>5% of own demand:", int((np.abs(net) > 0.05 * d_hour.sum(1)).sum()))
print("sum |net| / trips =", round(float(np.abs(net).sum() / d_hour.sum()), 4))
print("max hourly net / hourly demand:",
      round(float((a_hour - d_hour).max() / d_hour.sum()), 4))
