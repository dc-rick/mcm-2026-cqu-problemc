"""T004 step 6: reachable-neighbour geometry on the 0.0025-degree grid."""
import io
import json
import os

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = os.path.join(ROOT, "data", "runs", "T004")
z = np.load(os.path.join(RUN, "fine_sim_inputs.npz"))
xy = z["xy"]
C = len(xy)
tree = cKDTree(xy)

out = {"n_cells": C, "radii": {}}
for R in (0.30, 0.45, 0.60):
    nbrs = tree.query_ball_point(xy, R)
    counts = np.array([len(n) - 1 for n in nbrs])
    hist = {int(k): int((counts == k).sum()) for k in range(counts.max() + 1)}
    dists = []
    for i, ns in enumerate(nbrs):
        for j in ns:
            if j != i:
                dists.append(float(np.hypot(*(xy[i] - xy[j]))))
    dists = np.array(dists)
    out["radii"][f"{R:.2f}"] = dict(
        cells_with_neighbour=int((counts >= 1).sum()),
        cells_without_neighbour=int((counts == 0).sum()),
        neighbour_count_hist=hist,
        mean_neighbours=float(counts.mean()),
        median_neighbours=float(np.median(counts)),
        unique_distances_km=sorted({round(float(d), 4) for d in dists}),
        n_directed_edges=int(len(dists)))
    print(f"R={R:.2f}", out["radii"][f"{R:.2f}"])

# explicit lattice distances
nbrs8 = tree.query_ball_point(xy, 0.45)
lattice = {}
for i in range(C):
    for j in nbrs8[i]:
        if j <= i:
            continue
        dx, dy = xy[j] - xy[i]
        key = (round(abs(float(dx)), 3), round(abs(float(dy)), 3))
        lattice[key] = lattice.get(key, 0) + 1
out["lattice"] = [
    dict(dx_km=k[0], dy_km=k[1], distance_km=round(float(np.hypot(*k)), 3),
         distance_m=round(float(np.hypot(*k)) * 1000, 1), pairs=v)
    for k, v in sorted(lattice.items())]

json.dump(out, io.open(os.path.join(RUN, "neighbours_fine.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
