"""T004 boundary check: fleet conservation and non-negativity.

Runs a copy of the CellSim loop (same update order) with bookkeeping of every
borrowed bicycle and the final in-transit pool.  After the T004 fix the residual
I_start - (I_end + in_transit) must be exactly 0 for every grid, radius and
seed; before the fix the same run reports the bicycles deleted by the walk
branch.
"""
import io
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code"))
import sim_cell  # noqa: E402

RUN = os.path.join(ROOT, "data", "runs", "T004")
SEEDS = list(range(5))


def run_count(npz_path, walk_km, seed, fixed=True):
    z = np.load(npz_path)
    dep5, Pest, I0 = z["dep5"], z["Pest"], z["I0c"]
    xy, zid = z["xy"], z["zid"]
    C, Z, T = dep5.shape[1], Pest.shape[0], dep5.shape[0]
    rng = np.random.default_rng(seed)
    I = I0.astype(float).copy()
    I_start = I.sum()
    nbrs = cKDTree(xy).query_ball_point(xy, walk_km)
    zcells = [np.where(zid == zz)[0] for zz in range(Z)]
    inflight = np.zeros((2, C))
    unmet = served = borrowed = 0.0
    min_I = np.inf
    for k in range(T):
        d = rng.poisson(dep5[k])
        take = np.minimum(d, np.floor(I).astype(int))
        I -= take
        resid = d - take
        served += take.sum()
        borrow = np.zeros(C, dtype=int)
        for c in np.where(resid > 0)[0]:
            need = resid[c]
            while need > 0:
                nb = [j for j in nbrs[c] if I[j] >= 1]
                if not nb:
                    break
                j = max(nb, key=lambda q: I[q])
                I[j] -= 1
                borrow[c] += 1
                need -= 1
                served += 1
                borrowed += 1
            unmet += need
        for zz in range(Z):
            idx = zcells[zz]
            n = int(take[idx].sum() + (borrow[idx].sum() if fixed else 0))
            if n > 0:
                dst = rng.choice(C, size=n, p=Pest[zz])
                np.add.at(inflight[1], dst, 1)
        I += inflight[0]
        inflight[0] = inflight[1]
        inflight[1] = 0.0
        min_I = min(min_I, float(I.min()))
    total_end = float(I.sum() + inflight[0].sum())
    return dict(borrowed=int(borrowed), inventory_residual=float(I_start - total_end),
                min_inventory=min_I,
                unmet_share=float(unmet / max(unmet + served, 1)))


coarse = os.path.join(ROOT, "data", "cell_sim_inputs.npz")
fine = os.path.join(RUN, "fine_sim_inputs.npz")
out = {"coarse": {}, "fine": {}, "before_fix_seed0": {}}
for label, path in (("coarse", coarse), ("fine", fine)):
    for R in (0.30, 0.45, 0.60):
        rows = [run_count(path, R, sd, fixed=True) for sd in SEEDS]
        out[label][f"R{int(R*1000)}"] = dict(
            seeds=SEEDS,
            inventory_residual=[r["inventory_residual"] for r in rows],
            min_inventory=[r["min_inventory"] for r in rows],
            borrowed=[r["borrowed"] for r in rows])
        print(label, f"R={R:.2f}", "max_abs_residual",
              max(abs(r["inventory_residual"]) for r in rows),
              "min_I", min(r["min_inventory"] for r in rows))
        if label == "coarse":
            out["before_fix_seed0"][f"R{int(R*1000)}"] = run_count(
                path, R, 0, fixed=False)
json.dump(out, io.open(os.path.join(RUN, "conservation.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
