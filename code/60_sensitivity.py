"""Sensitivity experiments quoted in Section 9.

1. Fleet estimate: Poisson-occupancy model applied day by day, plus a check that
   the estimate does not depend on the (unknown) order sampling rate.
2. Q3: one-at-a-time sweep of the three logit parameters.
3. Q2: the price of a truck-kilometre, re-solving the LP at three values.
"""
import json
import importlib
import numpy as np
import lib
from lib import DATA
from scipy.optimize import brentq

out = {}

# ---------------------------------------------------------------- fleet size
print("== Poisson-occupancy fleet estimate by day ==")
df = lib.load_trips(None)
rows = []
for d, g in df.groupby("date"):
    T = len(g); D = g.bikeid.nunique()
    if T < 200000:
        continue
    r = T / D
    mu = brentq(lambda m: m / (1 - np.exp(-m)) - r, 1e-6, 50)
    rows.append(dict(date=str(d), trips=int(T), used=int(D), ratio=float(r),
                     mu=float(mu), fleet=float(T / mu)))
    print(f"  {d}  T={T:7d} D={D:6d} T/D={r:.3f} mu={mu:.3f} F={T/mu:9,.0f}")
F = np.array([r["fleet"] for r in rows]); R = np.array([r["ratio"] for r in rows])
print(f"  fleet range {F.min():,.0f} - {F.max():,.0f}, mean {F.mean():,.0f}, "
      f"sd {F.std():,.0f}")
out["fleet_days"] = rows

# sampling-rate invariance: resample orders at rate p, re-estimate
print("\n== invariance of the estimate to the order sampling rate ==")
g0 = df[df.date == "2017-05-16"]
rng = np.random.default_rng(0)
inv = []
for p in (0.05, 0.128, 0.25, 0.5):
    sub = g0[rng.random(len(g0)) < p]
    T = len(sub); D = sub.bikeid.nunique(); r = T / D
    mu = brentq(lambda m: m / (1 - np.exp(-m)) - r, 1e-6, 50)
    inv.append((p, T / mu))
    print(f"  p={p:5.3f}  T={T:7d}  D={D:6d}  F_hat={T/mu:9,.0f}")
out["sampling_invariance"] = inv

# ------------------------------------------------------------ Q3 parameters
print("\n== Q3 logit parameters ==")
cells = lib.get_cells()
need = np.load(DATA + r"\q3_cell.npz")["need"]
Dstar = np.load(DATA + r"\q3_cell.npz")["Dstar"]
xy = cells[["x", "y"]].to_numpy()
dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
EDGES = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45])
MID = 0.5 * (EDGES[:-1] + EDGES[1:])
V_BIKE, WALK, TRUCK = 10.0, 80.0, 3.4778
counts = np.array([((dist >= a) & (dist < b)).astype(np.float32) @ Dstar
                   for a, b in zip(EDGES[:-1], EDGES[1:])])
DEL = [0.5 * (a + b) for a, b in zip(EDGES[:-1], EDGES[1:])]
det = np.array([d / V_BIKE * 60 + d * 1000 / WALK for d in DEL])


def optimum(KAPPA, RHO, C0, grid=np.arange(0, 6.001, 0.05)):
    q = np.empty(len(grid))
    for i, r in enumerate(grid):
        p = 1.0 / (1.0 + np.exp(-(KAPPA * r - RHO * det - C0)))
        q[i] = np.minimum(need, (counts * p[:, None]).sum(0)).sum()
    tot = grid * q + TRUCK * (need.sum() - q)
    i = int(np.argmin(tot))
    return float(grid[i]), float(q[i]), float(tot[i])


base = optimum(2.0, 0.35, 0.8)
print(f"  base: r*={base[0]:.2f} q*={base[1]:.0f} total={base[2]:.0f}")
rows = [("base", 2.0, 0.35, 0.8, *base)]
for name, k, rho, c0 in (("kappa low", 1.0, .35, .8), ("kappa high", 3.0, .35, .8),
                         ("rho low", 2.0, .20, .8), ("rho high", 2.0, .50, .8),
                         ("inertia low", 2.0, .35, .4), ("inertia high", 2.0, .35, 1.2)):
    v = optimum(k, rho, c0)
    rows.append((name, k, rho, c0, *v))
    print(f"  {name:12s} r*={v[0]:.2f} q*={v[1]:7.0f} total={v[2]:9.0f}")
out["q3_sensitivity"] = rows

# ------------------------------------------------------------- Q2 truck cost
print("\n== Q2: price of a truck-kilometre ==")
mod = importlib.import_module("21_q2_opt")
q2 = []
for ck in (3.0, 6.0, 12.0):
    mod.COST_KM = ck
    r = mod.solve(alpha=1.0, K=60, time_limit=300, fixed_K_cost=600.0)
    if r:
        q2.append(dict(cost_km=ck, obj=r["obj"], moved=r["moved"],
                       truck_km=r["truck_km"], unmet=r["unmet"]))
        print(f"  c_km={ck:5.1f}  moved={r['moved']:8.0f}  truck-km={r['truck_km']:8.0f} "
              f" obj={r['obj']:9.0f}")
out["q2_cost_km"] = q2

json.dump(out, open(DATA + r"\sensitivity.json", "w"), indent=1)
print("\nwritten", DATA + r"\sensitivity.json")
