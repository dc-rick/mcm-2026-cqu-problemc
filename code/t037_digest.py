"""T037: digest of the accepted conserving numbers, for the paper.

Reads the result files and prints one block per section.  Every number the
paper quotes from Q2, Q3 (truck caliber) and Q4 can be traced here.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
import t024_q2_tier2 as t24
import t028_q2_conserve as t28

D = lib.DATA
RUN28 = os.path.join(D, "runs", "T028")
RUN30 = os.path.join(D, "runs", "T030")
RUN31 = os.path.join(D, "runs", "T031")


def j(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def q2():
    ls = j(os.path.join(RUN28, "local_search.json"))
    zb = j(os.path.join(RUN28, "zero_baselines.json"))
    n = int(ls["exec_n"])
    cost = float(ls["exec_cost"])
    unmet = float(ls["exec_unmet"])
    vkm = float(ls["exec_tri_vkm"])
    fixed = t24.TRI_FIXED * n
    kmc = t24.TRI_KM * vkm
    pen = t24.PENALTY * unmet
    base = float(ls["baseline_total"])
    print("Q2 executable")
    print("  fleet", n, "cost", round(cost, 4))
    print("  unmet", round(unmet, 4), "tri_vkm", round(vkm, 4))
    print("  fixed", round(fixed, 4), "km", round(kmc, 4),
          "penalty", round(pen, 4), "sum", round(fixed + kmc + pen, 4))
    print("  baseline", round(base, 4), "saving", round(base - cost, 4),
          "share", round(100 * (base - cost) / base, 6))
    print("  lp_n", ls["lp_n"], "lp_cost", round(ls["lp_cost"], 4),
          "lp_unmet", round(ls["lp_unmet"], 4))
    print("  no_truck_no_tri", zb["no_truck_no_tri"])
    print("  truck_only", zb["truck_only"])
    alloc = np.array(ls["exec_allocation"], dtype=int)
    print("  zones with 0:", int((alloc == 0).sum()),
          "max zone", int(alloc.max()), "median", float(np.median(alloc)))
    tot = ls["exec_schedule"]["totals"]
    print("  schedule totals", tot)
    print("  tri per bike",
          round((fixed + kmc) / max(1e-9, tot.get("bikes", 1)), 4))


def q3():
    d = j(os.path.join(RUN31, "truck_payment.json"))
    N, u = d["N"], d["truck_unit"]
    print("Q3 truck caliber")
    print("  N", N, "unit", u, "all_truck", round(u * N, 4))
    for lvl in ("arrivals", "surplus"):
        b = d["levels"][lvl]["best"]
        rows = d["levels"][lvl]["rows"]
        unc = min([dict(x, total=x["payment"] + u * (N - x["q"]))
                   for x in rows], key=lambda z: z["total"])
        allt = u * N
        print(" ", lvl, "best r", b["r"], "q", round(b["q"], 4),
              "P", round(b["P"], 4), "pay", round(b["payment"], 4),
              "total", round(b["total_spend"], 4),
              "saving", round(b["total_spend"] and
                              100 * (1 - b["total_spend"] / allt), 6),
              "share", round(100 * (1 - b["total_spend"] / allt), 4))
        print("    unconstrained r", unc["r"], "pay",
              round(unc["payment"], 4), "total", round(unc["total"], 4),
              "saving %", round(100 * (1 - unc["total"] / allt), 4))
        print("    cost per placement", round(b["total_spend"] - unc["total"],
                                              4) if False else "")


def q4():
    fr = j(os.path.join(RUN30, "frontier_cons.json"))
    rp = j(os.path.join(RUN30, "replay_cons.json"))
    print("Q4")
    for lv in ("meas343", "fleet495"):
        v = fr["levels"][lv]
        print(" ", lv, "sum a", round(v["demand_sum"], 4),
              "knee", v["knee"]["n"], round(v["knee"]["coverage"], 6),
              "dev", round(v["knee"]["deviation_normalised"], 6))
        s = v["selected"]
        print("   selected cov", round(s["coverage"], 6), "arr",
              round(s["arrival_coverage"], 6), "assigned",
              round(s["assigned"], 4), "unassigned",
              round(s["unassigned"], 4), "slots", s["slots"],
              "gap", s["gap"])
        a = v["all_open_milp"]
        print("   all-open cov", round(a["coverage"], 6), "arr",
              round(a["arrival_coverage"], 6), "unassigned",
              round(a["unassigned"], 4))
        for k in ("design_15", "design_60"):
            x = rp["levels"][lv][k]
            print("   ", k, "unmet", round(x["unmet"], 4), "ov",
                  round(x["overflow"], 4), "cov",
                  round(x["realized_coverage"], 6), "outside",
                  round(x["outside_end"], 4), "full_ever",
                  x["full_sites_ever"], "rate",
                  round(x["departure_shortfall_rate"], 6))


if __name__ == "__main__":
    q2()
    q3()
    q4()
