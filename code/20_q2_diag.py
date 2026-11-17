"""Q2 step 0: how large is the imbalance? fluid diagnostic before building the optimiser.

--od q1 (default) takes the demand surface from data/runs/T006/q1_od.npz;
--od obs takes the observed typical day.  --out defaults to
data/runs/T006/q2_state_q1.npz for q1 and data/q2_state.npz for obs, so the
tracked state file stays read-only under the default new input.
"""
import argparse
import os

import numpy as np, pandas as pd, lib
from lib import DATA

ap = argparse.ArgumentParser(description="Q2 imbalance diagnostic")
ap.add_argument("--od", choices=("q1", "obs"), default="q1")
ap.add_argument("--out", default=None)
args = ap.parse_args()

if args.od == "q1":
    z = np.load(os.path.join(DATA, "runs", "T006", "q1_od.npz"))
    od = z["od_q1"].astype(np.float64)
    d_hr = od.sum(2).T
    a_hr = od.sum(1).T
    zid = np.load(os.path.join(DATA, "typical_day.npz"),
                  allow_pickle=True)["zid"]
    cent = np.load(os.path.join(DATA, "typical_day.npz"),
                   allow_pickle=True)["cent"]
    out = args.out or os.path.join(DATA, "runs", "T006", "q2_state_q1.npz")
else:
    z = np.load(os.path.join(DATA, "typical_day.npz"), allow_pickle=True)
    d_hr, a_hr, zid, cent = z["d_hour"], z["a_hour"], z["zid"], z["cent"]
    out = args.out or os.path.join(DATA, "q2_state.npz")
NZ = d_hr.shape[0]
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
print("demand source:", args.od)

# --- scale of the sampled system -> full system
P_SAMPLE = 0.128            # estimated order sampling rate (bike-continuity test)
FLEET = 343638.0            # Poisson-occupancy fleet estimate for the observed system

# --- initial inventory: distribution of the last recorded trip end of the window
df = lib.load_trips(lib.ALL_DAYS)
cells, df = lib.make_cells(df, lib.THR)
ok = df.o.notna() & df.d.notna()
last = (df[ok].sort_values("starttime").groupby("bikeid").tail(1))
first = (df[ok].sort_values("starttime").groupby("bikeid").head(1))
for nm, sub in (("last-end", last.d.astype(int)), ("first-start", first.o.astype(int))):
    sh = np.bincount(zid[sub.values], minlength=NZ) / len(sub)
    print(nm, "share min/max", sh.min().round(4), sh.max().round(4),
          "corr with daily demand share",
          np.corrcoef(sh, d_hr.sum(1) / d_hr.sum())[0, 1].round(3))
    if nm == "last-end":
        init_share = sh
I0 = init_share * FLEET

net = a_hr - d_hr
cum = np.cumsum(net, axis=1)
IQ = I0[:, None] + cum
print("\nfleet", FLEET, " sum I0", I0.sum().round(0))
print("bikes per zone: min %.0f median %.0f max %.0f" % (I0.min(), np.median(I0), I0.max()))
deficit = np.maximum(0, -IQ)
print("zones ever negative:", int((IQ.min(1) < 0).sum()), "/", NZ)
print("fluid stockout (bike-hours):", deficit.sum().round(0),
      " share of hourly inventory-mass:", (deficit.sum() / np.maximum(IQ, 1e-9).sum()).round(4))
print("worst zone-day deficit:", deficit.max().round(0))

# peak-to-trough swing per zone (how many bikes must move to flatten the day)
req = cum.max(1) - cum.min(1)
print("required buffer per zone: min %.0f median %.0f max %.0f  total %.0f"
      % (req.min(), np.median(req), req.max(), req.sum()))
print("buffer / initial stock:", (req / I0).round(2)[:10])
print("\nhourly total departures:", d_hr.sum().round(0))
print("max hourly net flow (+/-):", net.max().round(0), net.min().round(0))
print("sum |net| over day:", np.abs(net).sum().round(0), "=",
      (np.abs(net).sum() / d_hr.sum()).round(3), "of trips")

# where is the imbalance concentrated?
order = np.argsort(-req)
print("\ntop-8 zones by required buffer (x km, y km, demand/day, buffer, I0):")
for i in order[:8]:
    print(f"  z{i:2d} ({cent[i,0]:5.1f},{cent[i,1]:5.2f})  d={d_hr[i].sum():8.0f}  "
          f"req={req[i]:6.0f}  I0={I0[i]:6.0f}  ratio={req[i]/I0[i]:.2f}")
np.savez(out, I0=I0, net=net, cum=cum, req=req, init_share=init_share)
print("wrote", out)
