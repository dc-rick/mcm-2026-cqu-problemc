"""T039: Q3 payment-ledger sensitivity to the acceptance constant c0.

The Q3 answer depends on the assumed inertia constant c0 through the
acceptance probability pi(d, r).  The paper reports the truck-caliber
payment ladder at c0 = 0.8 and re-optimises it at 0.8x and 1.2x its value,
holding kappa and rho fixed (a full kappa/rho sweep was not run).

For each c0 the static pool LP of T020 is rebuilt on the T030 fence
capacities and scanned over the 0.05-CNY reward grid under both source
caps.  Two budget rules are reported for each scan point:

  payment   ledger: feasible iff r * P <= 30,000  (the primary bill)
  increment ledger: feasible iff r * q <= 30,000  (the narrower accounting)

and the chosen row minimises payment + TRUCK_UNIT * (N - q).

Writes data/runs/T039/q3_c0.json.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
import t008_q3_pool as t8
import t020_q3_budget as t20
import t031_q3_payment as t31

RUN = os.path.join(lib.DATA, "runs", "T039")
os.makedirs(RUN, exist_ok=True)

BUDGET = 30000.0
C0S = [0.64, 0.80, 0.96]
GRID = np.round(np.arange(0.0, 1.6 + 1e-9, 0.05), 2)


def scan(lp, N, unit, pool):
    out = {}
    for cap_name in ("arrivals", "surplus"):
        caps = np.empty(lp.ns)
        for k, (h, s) in enumerate(lp.src_key_list):
            if cap_name == "arrivals":
                caps[k] = pool["arr"][h, s]
            else:
                caps[k] = max(0.0, pool["arr"][h, s] - pool["dep"][h, s])
        lp.b_src = caps
        rows = []
        for r in GRID:
            got = lp.solve(float(r), budget=None, cap_on=True)
            q, P = float(got["q"]), float(got["P"])
            rows.append(dict(r=float(r), q=q, P=P, payment=float(r * P),
                             bill_increment=float(r * q),
                             total=float(r * P + unit * (N - q))))
        pay_ok = [x for x in rows if x["payment"] <= BUDGET + 1e-9]
        inc_ok = [x for x in rows if x["bill_increment"] <= BUDGET + 1e-9]
        free = [x for x in rows if x["payment"] > BUDGET + 1e-9]
        out[cap_name] = dict(
            rows=rows,
            best_payment=min(pay_ok, key=lambda x: x["total"]),
            best_increment=min(inc_ok, key=lambda x: x["total"]),
            unconstrained=min(rows, key=lambda x: x["total"]),
            first_infeasible_payment=free[0] if free else None,
        )
    return out


def main():
    t0 = time.time()
    pool = t20.load_pool()
    capinfo = t31.build_capacity_t30(pool)
    N = float(pool["N"])
    unit = float(t31.truck_unit())
    allt = unit * N
    old = t8.PARAMS[0]
    out = dict(task="T039 Q3 payment ledger, c0 sweep", date="2026-11-20",
               budget=BUDGET, N=N, truck_unit=unit, all_truck=allt,
               grid=[float(GRID[0]), float(GRID[-1]),
                     float(GRID[1] - GRID[0])],
               kappa=float(t8.KAPPA), rho=float(t8.RHO),
               levels={})
    try:
        for c0 in C0S:
            t8.PARAMS[0] = (float(t8.KAPPA), float(t8.RHO), float(c0))
            lp = t20.PoolLP(pool, capinfo)
            got = scan(lp, N, unit, pool)
            for cap_name, d in got.items():
                for rule in ("best_payment", "best_increment", "unconstrained"):
                    x = d[rule]
                    x["saving"] = float(allt - x["total"])
                    x["saving_share"] = float(1.0 - x["total"] / allt)
            out["levels"]["%.2f" % c0] = got
            b1 = got["arrivals"]["best_payment"]
            b2 = got["arrivals"]["best_increment"]
            print("c0=%.2f pay-ledger r=%.2f q=%.1f pay=%.2f total=%.2f "
                  "save=%.2f | inc-ledger r=%.2f pay=%.2f total=%.2f "
                  "save=%.2f" % (
                      c0, b1["r"], b1["q"], b1["payment"], b1["total"],
                      b1["saving"], b2["r"], b2["payment"], b2["total"],
                      b2["saving"]), flush=True)
    finally:
        t8.PARAMS[0] = old
    out["runtime_s"] = float(time.time() - t0)
    p = os.path.join(RUN, "q3_c0.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
