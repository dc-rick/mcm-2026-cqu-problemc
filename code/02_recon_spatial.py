"""Step 2: spatial/temporal structure -> choose zoning, check sampling, fleet estimate."""
import numpy as np, pandas as pd
pd.set_option("display.width", 220)

df = pd.read_parquet(r"D:\code\mcm\2026CQU_C\data\trips.parquet")
df = df[(df.start_lat > 35) & (df.start_lon > 110) &
        (df.end_lat > 35) & (df.end_lon > 110)].copy()
print("after dropping out-of-city records:", len(df))

df["date"] = df.starttime.dt.normalize()
df["hour"] = df.starttime.dt.hour
df["dow"] = df.starttime.dt.dayofweek

print("\n== bbox of the 0.5%-99.5% mass ==")
print("lat", np.percentile(df.start_lat, [0.5, 50, 99.5]))
print("lon", np.percentile(df.start_lon, [0.5, 50, 99.5]))

print("\n== 6-char geohash cells ==")
g6 = df.geohashed_start_loc.str[:6]
vc = g6.value_counts()
print("n6 cells", len(vc), "cells for 90% trips", (vc.cumsum()/len(df) <= .9).sum()+1,
      "for 99%", (vc.cumsum()/len(df) <= .99).sum()+1)

print("\n== grid candidates ==")
for step in (0.01, 0.02, 0.03):
    gx = np.floor(df.start_lon/step).astype(int); gy = np.floor(df.start_lat/step).astype(int)
    c = pd.Series(list(zip(gx, gy))).value_counts()
    print(f"step {step}: cells {len(c)}, >=50 trips {int((c>=50).sum())}, "
          f"cover of top {int((c>=50).sum())} cells {(c[c>=50].sum()/len(df)):.3f}, "
          f"largest cell share {c.iloc[0]/len(df):.3f}")

print("\n== hourly shape by day (is the low-volume week a sampling artifact?) ==")
piv = df.groupby(["date", "hour"]).size().unstack(fill_value=0)
pivn = piv.div(piv.sum(1), axis=0)
for d in piv.index:
    print(d.date(), piv.loc[d].sum(), "peak hr", int(pivn.loc[d].idxmax()),
          "peak share", round(pivn.loc[d].max(), 4),
          "8-10 share", round(pivn.loc[d, [8, 9]].sum(), 4),
          "17-19 share", round(pivn.loc[d, [17, 18]].sum(), 4))

print("\n== weekday/weekend profile ==")
df["is_wd"] = df.dow < 5
full = [d for d in piv.index if piv.loc[d].sum() > 200000]
prof = df[df.date.isin(full)].groupby(["is_wd", "hour"]).size().unstack(0)
prof = prof/prof.sum()
print((prof*100).round(2).to_string())

print("\n== fleet size estimate (Poisson occupancy) ==")
from scipy.optimize import brentq
for d, g in df.groupby("date"):
    T = len(g); D = g.bikeid.nunique()
    if T < 200000: continue
    r = T/D
    f = lambda mu: mu/(1-np.exp(-mu)) - r
    mu = brentq(f, 1e-6, 50)
    F = T/mu
    print(d.date(), f"trips {T} used-bikes {D} T/D {r:.3f} mu* {mu:.3f} fleet {F:,.0f} "
          f"util {D/F:.3f}")

print("\n== OD matrix sparsity at 0.02 deg ==")
step = 0.02
sx = np.floor(df.start_lon/step).astype(int); sy = np.floor(df.start_lat/step).astype(int)
ex = np.floor(df.end_lon/step).astype(int); ey = np.floor(df.end_lat/step).astype(int)
so = sy.astype(str)+"_"+sx.astype(str); de = ey.astype(str)+"_"+ex.astype(str)
od = pd.crosstab(so, de)
print("zones", od.shape, "density", (od.values > 0).mean())
print("diagonal share (same cell)", np.trace(od.values)/od.values.sum())
