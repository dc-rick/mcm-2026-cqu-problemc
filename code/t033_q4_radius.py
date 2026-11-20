"""T033: Q4 radius sweep on the conserved demand (analytic + stored plan).

On this 0.005-degree grid the smallest positive cell-centre distance is
0.426 km, so R = 150 m and R = 300 m admit no neighbour and the capped
plan is the top-n cells by min(a_i, cap).  R = 450 m is the stored T030
knee plan.  Compliance factors are T019's zero-incentive rates.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
import t019_q4_cap as t19

RUN = os.path.join(lib.DATA, "runs", "T033")
os.makedirs(RUN, exist_ok=True)
COMPLY = {"150": 0.1454, "300": 0.0605, "450": 0.0238}
KNEE = {"meas343": 3500, "fleet495": 4000}


def self_only(a, arr_day, n, cap=150.0):
    v = np.minimum(a, cap)
    order = np.argsort(-v)
    sel = np.zeros(len(a), bool)
    sel[order[:n]] = True
    u = np.where(sel, np.maximum(a - cap, 0.0), a)
    cov = float(1.0 - u.sum() / a.sum())
    acov = float(t19.arrival_coverage(u, a, arr_day))
    return dict(coverage=cov, arrival_coverage=acov,
                assigned=float((a - u).sum()),
                unassigned=float(u.sum()), n_sites=int(n),
                slots=float(cap * n))


def main():
    z = np.load(os.path.join(lib.DATA, "runs", "T030",
                             "cell_demand_cons.npz"))
    arr_day = z["arr"].sum(0)
    out = {}
    for level in ("meas343", "fleet495"):
        a = z[f"a_{level}"]
        n = KNEE[level]
        plan = np.load(os.path.join(lib.DATA, "runs", "T030",
                                    f"plan_{level}_main.npz"))
        out[level] = dict(selected_n=n, radii={})
        for lab, R in (("150", 0.150), ("300", 0.300)):
            rec = self_only(a, arr_day, n)
            rec.update(R_m=float(R * 1000),
                       comply_zero_incentive=COMPLY[lab],
                       effective_coverage=rec["coverage"] * COMPLY[lab],
                       effective_arrival_coverage=(
                           rec["arrival_coverage"] * COMPLY[lab]),
                       n_arcs=0)
            out[level]["radii"][lab] = rec
        # 450 m: the stored knee plan
        u = plan["u"]
        cov = float(plan["f"].sum() / a.sum())
        acov = float(t19.arrival_coverage(u, a, arr_day))
        out[level]["radii"]["450"] = dict(
            R_m=450.0, n_sites=n, slots=float(150.0 * n),
            coverage=cov, arrival_coverage=acov,
            assigned=float(plan["f"].sum()),
            unassigned=float(u.sum()),
            comply_zero_incentive=COMPLY["450"],
            effective_coverage=cov * COMPLY["450"],
            effective_arrival_coverage=acov * COMPLY["450"],
            n_arcs=int(len(plan["ii"])))
    path = os.path.join(RUN, "radius_cons.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(task="T033 Q4 radius sweep", date="2026-11-20",
                       grid_min_positive_km=0.426,
                       note=("R=150/300 m admit no neighbour cell centre on "
                             "this grid; their plan is the top-n self-only "
                             "selection. R=450 m is the stored T030 knee "
                             "plan. Compliance is T019's zero-incentive "
                             "rate."), levels=out), fh,
                  ensure_ascii=False, indent=1)
    print("wrote", path)
    for level, v in out.items():
        print(level)
        for lab, r in v["radii"].items():
            print("  R=%s cov=%.4f arr=%.4f eff=%.4f"
                  % (lab, r["coverage"], r["arrival_coverage"],
                     r["effective_coverage"]))
    return out


if __name__ == "__main__":
    main()
