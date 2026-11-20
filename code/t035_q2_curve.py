"""T035: joint Q2 cost against fleet size, for the Q2 figure panel (d).

Scales the final executable allocation by a factor, rounds to integers,
and evaluates the joint conserving simulation.  The curve is a global
one-parameter slice; the optimum itself comes from T028's coordinate
descent and is not re-derived here.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
import t024_q2_tier2 as t24
import t028_q2_conserve as t28

RUN = os.path.join(lib.DATA, "runs", "T035")
os.makedirs(RUN, exist_ok=True)
FACTORS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15,
           1.20, 1.25, 1.30]


def main():
    ls = t28.load_json("local_search.json")
    alloc = np.array(ls["exec_allocation"], dtype=int)
    inp, _ = t24.prep()
    plan, _ = t24.load_plan()
    ctx = t28.build(inp, plan)
    rows = []
    for f in FACTORS:
        nv = np.maximum(0, np.round(alloc * f)).astype(int)
        rep = t28.simulate(ctx, nv)
        cost = float(t28.truck_obj + t24.TRI_FIXED * nv.sum()
                     + t24.TRI_KM * rep["tri_vehicle_km"].sum()
                     + t24.PENALTY * rep["unmet"])
        rows.append(dict(factor=float(f), n=int(nv.sum()), cost=cost,
                         unmet=float(rep["unmet"]),
                         tri_vkm=float(rep["tri_vehicle_km"].sum())))
        print(rows[-1], flush=True)
    out = dict(task="T035 Q2 joint fleet curve", date="2026-11-20",
               base_n=int(alloc.sum()), rows=rows,
               note=("uniform scaling of the T028 executable allocation; "
                     "each point is a full joint conserving replay"))
    path = os.path.join(RUN, "joint_curve.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
