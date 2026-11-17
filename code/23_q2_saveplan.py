"""Save a Q2 LP plan as an NPZ for 22 (and downstream) to read.

Defaults to data/runs/T007/q2_plan_q1.npz so the tracked data/q2_plan.npz is
not overwritten by a T007 rerun.  The NPZ carries XM/X/M/I/U plus MK, the
hourly truck flow per arc used for truck-km shares.
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

import lib
from lib import DATA

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description="Save a Q2 plan")
    ap.add_argument("--od", choices=("q1", "obs"), default="q1")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=60)
    ap.add_argument("--conserve", action="store_true")
    ap.add_argument("--fixed-k-cost", type=float, default=600.0)
    ap.add_argument("--time-limit", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(DATA, "runs", "T007",
                                                  "q2_plan_q1.npz"))
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location(
        "q2", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "21_q2_opt.py"))
    q2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q2)
    q2.DEP, q2.ARR, q2.PZ, q2.CENT = q2.load_demand(args.od)
    q2.DIST, q2.ARCS, q2.OUT_TO, q2.IN_FROM = q2.load_network(q2.CENT)
    q2.NZ = len(q2.CENT)
    r = q2.solve(alpha=args.alpha, K=args.K, time_limit=args.time_limit,
                 fixed_K_cost=args.fixed_k_cost, conserve=args.conserve,
                 od=args.od, verbose=True)
    if not r:
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, XM=r["XM"], X=r["X"], M=r["M"], I=r["I"], U=r["U"],
             MK=r["MK"])
    print("saved", args.out, r["XM"].shape, "bikes moved",
          round(float(r["XM"].sum()), 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
