"""T004 step 1: reachable neighbours on the 0.005 deg cell lattice.

The cell centres sit on a regular lattice: 0.005 deg in longitude = 0.426 km,
0.005 deg in latitude = 0.556 km, diagonal = 0.700 km.  A circular walking
radius therefore reaches neighbours anisotropically.
"""
import io
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = os.path.join(ROOT, "data", "runs", "T004")
os.makedirs(RUN, exist_ok=True)

z = np.load(os.path.join(ROOT, "data", "cell_sim_inputs.npz"))
xy = z["xy"]
C = len(xy)
tree = cKDTree(xy)

out = {"n_cells": C, "radii": {}}
for R in (0.30, 0.45, 0.60, 0.70, 0.80):
    nbrs = tree.query_ball_point(xy, R)
    counts = np.array([len(n) - 1 for n in nbrs])  # exclude self
    hist = {int(k): int((counts == k).sum()) for k in range(counts.max() + 1)}
    # distance profile of the non-self neighbours
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
        unique_distances_km=sorted({round(float(d), 4) for d in dists}),
        n_directed_edges=int(len(dists)),
    )

# explicit lattice distances: pick a cell with E, W, N, S neighbours
nbrs = tree.query_ball_point(xy, 0.80)
lattice = {}
for i in range(C):
    for j in nbrs[i]:
        if j <= i:
            continue
        dx, dy = xy[j] - xy[i]
        key = (round(abs(float(dx)), 3), round(abs(float(dy)), 3))
        lattice[key] = lattice.get(key, 0) + 1
lat_rows = [dict(dx_km=k[0], dy_km=k[1], distance_km=round(float(np.hypot(*k)), 3),
                 pairs=v) for k, v in sorted(lattice.items())]
out["lattice"] = lat_rows

json.dump(out, io.open(os.path.join(RUN, "neighbours.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
