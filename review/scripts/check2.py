import sys, numpy as np, pandas as pd
D = sys.argv[1]
df = pd.read_parquet(D + "/trips.parquet", columns=["bikeid", "userid", "starttime",
        "geohashed_start_loc", "geohashed_end_loc", "start_lat", "start_lon", "end_lat", "end_lon"])
df = df[(df.start_lat > 39.4) & (df.start_lon > 115.8) & (df.start_lat < 40.4) & (df.start_lon < 117.0)]
df["date"] = df.starttime.dt.strftime("%Y-%m-%d")
print("== A. fleet estimate vs distinct bicycles ==")
days = sorted(df.date.unique()); print("dates:", days)
seen = set(); rows = []
for d in days:
    g = df[df.date == d]
    seen |= set(g.bikeid.unique())
    rows.append((d, len(g), g.bikeid.nunique(), len(seen)))
for r in rows: print(f"  {r[0]}  trips {r[1]:7d}  bikes that day {r[2]:7d}  cumulative distinct {r[3]:7d}")
core = df[df.date <= "2017-05-19"]
print("  distinct bikes 10-19 May:", core.bikeid.nunique())
F, mu = 343638.0, 0.840
print(f"  homogeneous-Poisson prediction for 9 days of data: {F*(1-np.exp(-mu*9)):.0f}")
per_bike_days = core.groupby("bikeid").date.nunique()
print("  bikes seen on exactly 1 of 9 days:", (per_bike_days == 1).mean().round(3),
      " predicted by the model:", round(9*(1-np.exp(-mu))*np.exp(-mu*8)/(1-np.exp(-9*mu)), 4))

print("\n== B. sampling-rate test: jump between a bike's trip end and its next start ==")
s = core.sort_values(["bikeid", "starttime"])
nb = s.bikeid.shift(-1) == s.bikeid
ns_lat = s.start_lat.shift(-1); ns_lon = s.start_lon.shift(-1); ns_gh = s.geohashed_start_loc.shift(-1)
m = nb.values
dy = (ns_lat.values[m] - s.end_lat.values[m]) * 111.2
dx = (ns_lon.values[m] - s.end_lon.values[m]) * 85.2
jump = np.sqrt(dx**2 + dy**2)
exact = (s.geohashed_end_loc.values[m] == ns_gh.values[m])
print(f"  pairs {m.sum()}; exact geohash match {exact.mean():.4f}")
for thr in (0.16, 0.25, 0.35, 0.5):
    print(f"  share with jump <= {thr*1000:.0f} m: {(jump <= thr).mean():.4f}")
h = np.histogram(jump, bins=[0, 0.001, 0.16, 0.25, 0.35, 0.5, 0.75, 1.0, 2.0, 100])[0] / len(jump)
print("  jump histogram [0,1m,160m,250m,350m,500m,750m,1km,2km,inf):", h.round(4).tolist())

print("\n== C. other cited numbers ==")
typ = ["2017-05-10","2017-05-11","2017-05-12","2017-05-15","2017-05-16","2017-05-18","2017-05-19"]
vol = core.groupby("date").size()
print("  CV of daily volume, all 9 days:", round(vol.std()/vol.mean(), 4), " typical days only:",
      round(vol[typ].std()/vol[typ].mean(), 4))
q4 = np.load(D + "/q4_frontier.npz", allow_pickle=False)
caps = q4["caps"]; c450 = q4["c450"]
print("  fences >150 slots:", round((caps > 150).mean(), 4), " coverage after 200/400/626 fences:",
      [round(float(c450[i]), 4) for i in (199, 399, len(c450)-1)])
q3 = np.load(D + "/q3_cell.npz", allow_pickle=False); Dday = q3["Dday"]
top = np.sort(Dday)[::-1]
print("  share of arrivals in busiest 10% of cells:", round(top[:len(top)//10].sum()/top.sum(), 4))
print("  cell fleet curve (alpha, unmet, sd):", np.load(D + "/cell_fleet_curve.npy").round(4).tolist())
