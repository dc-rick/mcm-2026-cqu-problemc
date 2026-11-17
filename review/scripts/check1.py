import sys, numpy as np, pandas as pd
D = sys.argv[1]
SCALE = 1/0.128
z = np.load(D + "/typical_day.npz", allow_pickle=False)
st = np.load(D + "/q2_state.npz", allow_pickle=False)
I0 = st["I0"]; d_hr = z["d_hour"]; a_hr = z["a_hour"]
print("== 1. Q2 diagnostic scaling ==")
print("sample trips/day in typical_day.npz:", round(d_hr.sum()))
print("I0 sum:", round(I0.sum()), " min:", round(I0.min()), " median:", round(np.median(I0)))
for lab, s in (("as coded (unscaled flows)", 1.0), ("scaled by 1/p (as paper states)", SCALE)):
    cum = np.cumsum((a_hr - d_hr) * s, axis=1)
    IQ = I0[:, None] + cum
    print(f"  {lab}: worst cum excursion {cum.min():.0f}; min inventory {IQ.min():.0f}; "
          f"zones ever negative {(IQ.min(1) < 0).sum()}; sum|end drift| {np.abs(cum[:, -1]).sum():.0f}")
cum_s = np.cumsum((a_hr - d_hr) * SCALE, axis=1)
w = np.argmin(cum_s.min(1))
print(f"  worst zone {w}: I0={I0[w]:.0f}, worst scaled excursion {cum_s[w].min():.0f}")
lo = np.argsort(I0)[:5]
print("  5 smallest zones: I0", I0[lo].round(0), " scaled min cum", cum_s[lo].min(1).round(0))
print("  ratio |worst excursion| / I0 (scaled), top 5:",
      np.sort(-cum_s.min(1) / I0)[::-1][:5].round(3))

print("\n== 2. Q3 detour bands on the cell grid ==")
cells = pd.read_parquet(D + "/cells.parquet")
xy = cells[["x", "y"]].to_numpy()
q3 = np.load(D + "/q3_cell.npz", allow_pickle=False)
need, Dstar = q3["need"], q3["Dstar"]
dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
u = np.unique(np.round(dist[dist < 1.0], 3))
print("  distinct centroid distances below 1 km:", u[:12])
EDGES = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45])
MID = 0.5 * (EDGES[:-1] + EDGES[1:])
counts = np.array([((dist >= a) & (dist < b)).astype(np.float32) @ Dstar
                   for a, b in zip(EDGES[:-1], EDGES[1:])])
for k in range(6):
    print(f"  band [{EDGES[k]*1000:.0f},{EDGES[k+1]*1000:.0f}) m: candidate trips {counts[k].sum():10.0f}")
def tau(d): return d / 10 * 60 + d * 1000 / 80
def pi(r, d): return 1 / (1 + np.exp(-(2.0 * r - 0.35 * tau(d) - 0.8)))
def q_of(r, cnt):
    return np.minimum(need, (cnt * np.array([pi(r, m) for m in MID])[:, None]).sum(0)).sum()
cnt_noself = counts.copy(); cnt_noself[0] = 0
for r in (0.2, 0.6, 1.0, 1.2, 2.0, 3.5):
    print(f"  r={r:3.1f}: q as coded {q_of(r, counts):9.0f} | without the trips that already end in the deficit cell {q_of(r, cnt_noself):9.0f}")
rr = np.linspace(0, 6, 241)
need_tot = need.sum(); TRU = 83325 / 23959
for lab, cnt in (("as coded", counts), ("self-cell removed", cnt_noself)):
    qq = np.array([q_of(r, cnt) for r in rr]); tot = rr * qq + TRU * (need_tot - qq)
    i = int(np.argmin(tot))
    print(f"  optimum {lab}: r*={rr[i]:.2f}, q*={qq[i]:.0f} ({100*qq[i]/need_tot:.1f}%), saving vs all-truck {100*(1-tot[i]/(TRU*need_tot)):.1f}%")
near = dist <= 0.45
pot_self = Dstar
cov_self_only = np.minimum(pot_self, need).sum() / need_tot
print(f"  catchment coverage from the deficit cell's own arrivals alone: {100*cov_self_only:.1f}%  (paper's 450 m figure: 71.9%)")

print("\n== 3. Cell simulator walking radius ==")
from scipy.spatial import cKDTree
nb = cKDTree(xy).query_ball_point(xy, 0.30)
print("  cells whose 300 m neighbourhood contains any other cell:", sum(len(v) > 1 for v in nb), "of", len(nb))

print("\n== 4. Q2 plan timing and truck utilisation ==")
p = np.load(D + "/q2_plan.npz", allow_pickle=False)
X, M = p["X"], p["M"]
hb = X.sum(0); ht = M.sum(0)
print("  bikes loaded per hour:", hb.round(0).astype(int).tolist())
night = list(range(21, 24)) + list(range(0, 6))
print(f"  share of bikes moved 21:00-06:00: {hb[night].sum()/hb.sum():.3f}")
print(f"  truck-legs per day {ht.sum():.1f}; per truck {ht.sum()/60:.1f} h of driving")
print("  truck-legs per hour:", ht.round(1).tolist())
