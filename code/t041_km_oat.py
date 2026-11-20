"""T041: one-at-a-time truck-kilometre response on the conserving Q2 chain.

Holds the T032 reference context (demand, inventory, all other costs) and
re-solves the 50-zone truck LP at COST_KM = 3.0 and 9.0.  The conserving
replay does not depend on the truck-kilometre price, so the change in the LP
objective is the change in the Q2 two-tier total with the plan held fixed.
Writes data/runs/T041/km_oat.json.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
import t023_uncertainty as t23

RUN = os.path.join(lib.DATA, "runs", "T041")
os.makedirs(RUN, exist_ok=True)


def main():
    prep, _ = t23.run_prep()
    ctx = t23.make_ctx(prep)
    out = dict(task="T041 truck-km one-at-a-time", date="2026-11-20",
               plan="T032 reference context, truck plan fixed at K=60",
               rows=[])
    vals = {}
    for km in (3.0, 6.0, 9.0):
        c = t23.cvec_full({"COST_KM": km})
        t0 = time.time()
        r = t23.solve18(ctx.dep_zone, ctx.PZ, ctx.I0z, 60, c,
                        integer=False, time_limit=180)
        obj = float(r["obj"])
        vals[km] = obj
        out["rows"].append(dict(cost_km=km, truck_obj=obj,
                                runtime_s=round(time.time() - t0, 1)))
        print("km=%.1f obj=%.4f" % (km, obj), flush=True)
    out["delta"] = float(vals[9.0] - vals[3.0])
    out["per_yuan_km"] = float(out["delta"] / 6.0)
    path = os.path.join(RUN, "km_oat.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("wrote", path, "delta=%.4f" % out["delta"], flush=True)


if __name__ == "__main__":
    main()
