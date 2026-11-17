"""Q5: fragmentation cost of multi-platform competition + external validation with OSM."""
import os
import json
import numpy as np
import pandas as pd
import lib
from lib import DATA
import sim

SCALE = 1.0 / 0.128
z = np.load(os.path.join(DATA, "typical_day.npz"), allow_pickle=True)
od = z["od_z"] * SCALE
dep = od.sum(2)
cent = z["cent"]
NZ = od.shape[1]
st = np.load(os.path.join(DATA, "q2_state.npz"), allow_pickle=True)
I0 = st["I0"]

# ---------------- fragmentation experiment ---------------------------------
print("== two-platform split (users single-home, fleet split) ==")
res = {}
for share in (1.0, 0.7, 0.5, 0.35):
    r = []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        # platform A gets share of the fleet and the same share of the demand
        I0A = I0 * share
        depA = dep * share
        # the OD stays spatially identical (users of A are a random subset)
        S = sim.Sim(depA.T, od, I0A, seed=seed)
        r.append(S.run()["unmet_share"])
    res[share] = (float(np.mean(r)), float(np.std(r)))
    print(f"  fleet & demand share {share:.2f}: unmet {100*np.mean(r):5.2f}%")

# closed-form check: if the probability a user finds a bike on one platform is p,
# with two platforms each holding half the fleet and single-homing users, the
# effective availability is that of a system with half the density.
print("\n  interpretation: single-homing splits the effective density; the loss is")
print("  first order in the density, so service falls roughly linearly in the share.")

# ---------------- external validation with OSM subway entrances ------------
print("\n== OSM cross-check of the identified attractors ==")
osm = json.load(open(os.path.join(DATA, "osm_subway_entrances.json"), encoding="utf-8"))
pts = np.array([[e["lat"], e["lon"]] for e in osm["elements"]])
xy_osm = np.c_[ (pts[:, 1] - 116.0) * 85.2, (pts[:, 0] - 39.0) * 111.2 ]
cells = lib.get_cells()
xy = cells[["x", "y"]].to_numpy()

df = lib.load_trips(lib.TYP_DAYS)
df = lib.assign_cells(df, cells)
ok = df.o.notna() & df.d.notna()
A = np.bincount(df[ok].d.astype(int).values, minlength=len(cells)).astype(float)
Ad = A / A.sum()

from scipy.spatial import cKDTree
tree = cKDTree(xy_osm)
d_ent, _ = tree.query(xy)
print(f"subway entrances fetched: {len(xy_osm)}")
print(f"cells within 300 m of an entrance: {(d_ent<=0.3).sum()}")
print(f"share of all arrivals in those cells: {Ad[d_ent<=0.3].sum():.3f}")
top = np.argsort(-Ad)[:300]
print(f"share of arrivals in the top-300 cells: {Ad[top].sum():.3f}")
print(f"share of the top-300 cells that lie within 300 m of an entrance: "
      f"{(d_ent[top]<=0.3).mean():.3f}")
print(f"baseline share of all cells within 300 m: {(d_ent<=0.3).mean():.3f}")
print(f"median distance of the top-100 cells to the nearest entrance: "
      f"{np.median(d_ent[np.argsort(-Ad)[:100]]):.3f} km")
print(f"median distance of all cells: {np.median(d_ent):.3f} km")

# ---------- capacity duplication: each platform sizes its own fence -------
print("\n== fence capacity under fragmenting competition ==")
q15 = df[ok].starttime.dt.hour.values * 4 + df[ok].starttime.dt.minute.values // 15
arrc = np.zeros((96, len(cells))); depc = np.zeros((96, len(cells)))
np.add.at(arrc, (q15, df[ok].d.values.astype(int)), 1.0)
np.add.at(depc, (q15, df[ok].o.values.astype(int)), 1.0)
arrc = arrc / len(lib.TYP_DAYS) / 0.128
depc = depc / len(lib.TYP_DAYS) / 0.128
distc = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
fr = np.load(os.path.join(DATA, "q4_frontier.npz"))
caps, sites = fr["caps"], fr["sites"]


def peak(a, d, share, reps=300, seed=0):
    rng = np.random.default_rng(seed); out = np.empty(reps)
    for k in range(reps):
        occ = 0.0; pk = 0.0
        ai = rng.poisson(a * share); di = rng.poisson(d * share)
        for t in range(96):
            occ = max(0.0, occ + ai[t] - di[t]); pk = max(pk, occ)
        out[k] = pk
    return float(np.percentile(out, 95))


dup = {}
for name, c in (("median", sites[int(np.argsort(caps)[len(caps) // 2])]),
                ("largest", sites[int(np.argmax(caps))])):
    mem = np.where(distc[c] <= 0.45)[0]
    a = arrc[:, mem].sum(1); d = depc[:, mem].sum(1)
    p1 = peak(a, d, 1.0)
    row = {"one": p1}
    for n in (2, 3):
        pn = peak(a, d, 1 / n)
        row[f"{n}"] = (float(pn), float(n * pn / p1 - 1))
    dup[name] = row
    print(f"  {name}: one platform {p1:.0f} slots; " +
          "; ".join(f"{n} platforms {row[str(n)][0]:.0f} each, total "
                    f"{n*row[str(n)][0]:.0f} ({100*row[str(n)][1]:+.1f}%)" for n in (2, 3)))

json.dump(dict(split={str(k): v for k, v in res.items()},
               osm_entrances=int(len(xy_osm)),
               share_arrivals_near_entrance=float(Ad[d_ent <= 0.3].sum()),
               top300_near_entrance=float((d_ent[top] <= 0.3).mean()),
               baseline_near_entrance=float((d_ent <= 0.3).mean()),
               median_dist_top100=float(np.median(d_ent[np.argsort(-Ad)[:100]])),
               median_dist_all=float(np.median(d_ent)),
               capacity_duplication=dup),
          open(os.path.join(DATA, "q5_summary.json"), "w"), indent=1)
