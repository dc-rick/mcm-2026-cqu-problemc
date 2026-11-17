"""Q4: electronic-fence layout and capacity planning.

Demand  : arrivals per cell per 15 min (Q1 model / data)
Siting  : greedy maximal coverage with walking radius R (fence = a cell site)
Sizing  : capacity = 95th percentile of the daily peak occupancy, obtained by
          Monte-Carlo on the (arrival - departure) random walk of the served set
"""
import json
import numpy as np
import lib
from lib import DATA

SCALE = 1.0 / 0.128
cells = lib.get_cells()
C = len(cells)
xy = cells[["x", "y"]].to_numpy()
dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))

df = lib.load_trips(lib.TYP_DAYS)
df = lib.assign_cells(df, cells)
ok = df.o.notna() & df.d.notna()
s = df[ok].copy(); s["o"] = s.o.astype(int); s["d"] = s.d.astype(int)
nd = len(lib.TYP_DAYS)
q = s.starttime.dt.hour.values * 4 + s.starttime.dt.minute.values // 15
arr = np.zeros((96, C)); dep = np.zeros((96, C))
np.add.at(arr, (q, s.d.values), 1.0)
np.add.at(dep, (q, s.o.values), 1.0)
arr = arr / nd * SCALE; dep = dep / nd * SCALE
Ad = arr.sum(0)
print(f"arrivals/day {Ad.sum():.0f}, cells {C}")

# ---------- siting: greedy maximal coverage ---------------------------------
def greedy_cover(R, Nmax, cand_top=2500, frac_stop=0.0005):
    """Greedy maximal coverage: pick the cell whose catchment contains the most
    still-uncovered arrivals.  Returns the full coverage path, not just the end."""
    near = (dist <= R).astype(np.float32)
    cand = np.argsort(-Ad)[:cand_top]
    Nmat = near[cand]                       # (ncand, C)
    covered = np.zeros(C, bool)
    chosen = []
    gain_hist = []
    cov_hist = []
    for _ in range(Nmax):
        g = Nmat @ (Ad * (~covered))
        if gain_hist:
            g[chosen] = -1.0
        k = int(np.argmax(g))
        if g[k] <= frac_stop * Ad.sum():
            break
        chosen.append(k)
        covered |= near[cand[k]] > 0
        gain_hist.append(float(g[k]))
        cov_hist.append(float(Ad[covered].sum() / Ad.sum()))
    return [int(cand[k]) for k in chosen], covered, cov_hist


results = {}
curves = {}
for R, Nmax in ((0.25, 2500), (0.45, 1500), (0.65, 900)):
    ch, cov, cvh = greedy_cover(R, Nmax)
    share = Ad[cov].sum() / Ad.sum()
    served = np.where(cov)[0]
    print(f"R={R*1000:.0f} m: {len(ch)} fences cover {100*share:.1f}% of arrivals")
    results[R] = (ch, cov, share)
    curves[R] = np.array(cvh)
    if R == 0.45:
        np.save(DATA + r"\q4_sites.npy", np.array(ch))

# ---------- capacity sizing ------------------------------------------------
def peak_occupancy(served_cells, reps=200, seed=0):
    """Monte-Carlo peak occupancy of one fence serving `served_cells`."""
    rng = np.random.default_rng(seed)
    a = arr[:, served_cells].sum(1)
    d = dep[:, served_cells].sum(1)
    peaks = np.empty(reps)
    for i in range(reps):
        occ = 0.0; peak = 0.0
        ai = rng.poisson(a); di = rng.poisson(d)
        for t in range(96):
            occ = max(0.0, occ + ai[t] - di[t])
            peak = max(peak, occ)
        peaks[i] = peak
    return float(np.percentile(peaks, 95))


caps_by_R = {}
for R in (0.25, 0.45, 0.65):
    ch, cov, share = results[R]
    print(f"\ncapacity sizing for the R={R*1000:.0f} m plan ({len(ch)} fences)")
    caps = []
    for c in ch:
        members = np.where(dist[c] <= R)[0]
        caps.append(peak_occupancy(members, reps=300))
    caps = np.array(caps)
    caps_by_R[R] = caps
    print(f"capacity: total {caps.sum():.0f} slots, median {np.median(caps):.0f}, "
          f"p90 {np.percentile(caps,90):.0f}, max {caps.max():.0f}")
    print(f"land at 1.4 m2/slot: {caps.sum()*1.4/1e4:.1f} ha")
    print(f"median peak arrivals served per fence: "
          f"{np.median([arr[:, np.where(dist[c]<=R)[0]].sum() for c in ch]):.0f}")

R = 0.45
ch, cov, share = results[R]
caps = caps_by_R[R]

# ---------- coverage / cost frontier ---------------------------------------
frontier = []
for R in (0.25, 0.45, 0.65):
    for N in (200, 400, 800, 1200, 2000, 2500):
        ch2, cov2, gh2 = results[R][0][:N], None, None
        near = dist <= R
        covered = np.zeros(C, bool)
        for c in ch2:
            covered |= near[c]
        frontier.append((R, min(N, len(results[R][0])),
                         float(Ad[covered].sum() / Ad.sum())))
fr = np.array(frontier)
np.savez(DATA + r"\q4_frontier.npz", fr=fr, caps=caps,
         sites=np.array(ch), share=share,
         c250=curves[0.25], c450=curves[0.45], c650=curves[0.65])
json.dump(dict(n_fences=len(ch), coverage=float(share),
               total_capacity=float(caps.sum()),
               median_capacity=float(np.median(caps)),
               land_ha=float(caps.sum() * 1.4 / 1e4),
               by_radius={f"{k:.2f}": dict(
                   fences=len(results[k][0]),
                   coverage=float(results[k][2]),
                   total=float(v.sum()),
                   median=float(np.median(v)),
                   p90=float(np.percentile(v, 90)),
                   max=float(v.max()))
                   for k, v in caps_by_R.items()}),
          open(DATA + r"\q4_summary.json", "w"), indent=1)
print("\nfrontier (R, fences, coverage):")
for r in frontier[::4]:
    print(f"  {r[0]*1000:.0f} m  N={int(r[1]):5d}  cov={100*r[2]:5.1f}%")
