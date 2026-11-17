"""T004 step 2: cell-level simulator at R = 300/450/600 m.

Runs code/sim_cell.py.  T004 fixed a fleet-conservation bug in the walk branch:
a bicycle borrowed from a neighbouring cell now enters the in-flight arrival
pool.  At R=300 m no neighbour is reachable, so the run reproduces the
published curve exactly; at R>=426 m the fix is active.

Two fleet calibrations are reported:
  * FLEET = 343,638: the calibration that produced the paper's 13.4%, used to
    isolate the radius effect and to reproduce the published number.
  * FLEET = 495,540: the closed-fleet centre from T001-R2, used as a bridge.
"""
import io
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code"))
import sim_cell  # noqa: E402

RUN = os.path.join(ROOT, "data", "runs", "T004")
os.makedirs(RUN, exist_ok=True)
z = np.load(os.path.join(ROOT, "data", "cell_sim_inputs.npz"))
dep5, Pest, I0c, zid, xy = z["dep5"], z["Pest"], z["I0c"], z["zid"], z["xy"]

FLEET_OLD = 343638.0
FLEET_NEW = 495540.0
SEEDS = list(range(5))


def run_one(walk_km, alpha, seed, fleet_scale=1.0):
    S = sim_cell.CellSim(dep5, Pest, I0c * alpha * fleet_scale, xy, zid,
                         rng=np.random.default_rng(seed), walk_km=walk_km)
    return float(S.run()["unmet_share"])


results = {}
t0 = time.time()
for R in (0.30, 0.45, 0.60):
    for a in (1.0, 0.8):
        vals = [run_one(R, a, s) for s in SEEDS]
        results[f"coarse_343638_R{int(R*1000)}_a{a:.1f}"] = dict(
            R_m=int(R * 1000), alpha=a, fleet=FLEET_OLD * a,
            seeds=SEEDS, values=vals,
            mean=float(np.mean(vals)), std=float(np.std(vals, ddof=1)),
        )
        print(f"R={R:.2f} a={a:.1f} mean={np.mean(vals):.4f} sd={np.std(vals,ddof=1):.4f}")

# bridge to the closed fleet at the main radius
for a in (1.0, 0.8):
    vals = [run_one(0.45, a, s, fleet_scale=FLEET_NEW / FLEET_OLD) for s in SEEDS]
    results[f"coarse_495540_R450_a{a:.1f}"] = dict(
        R_m=450, alpha=a, fleet=FLEET_NEW * a,
        seeds=SEEDS, values=vals,
        mean=float(np.mean(vals)), std=float(np.std(vals, ddof=1)),
    )
    print(f"R=450 a={a:.1f} fleet495540 mean={np.mean(vals):.4f} "
          f"sd={np.std(vals,ddof=1):.4f}")

# exact reproduction check against data/cell_fleet_curve.npy (seeds 0-2)
old_curve = np.load(os.path.join(ROOT, "data", "cell_fleet_curve.npy"))
repro = {}
for i, a in enumerate((1.0, 0.8)):
    key = f"coarse_343638_R300_a{a:.1f}"
    vals012 = results[key]["values"][:3]
    repro[f"a{a:.1f}"] = dict(
        old_curve_mean=float(old_curve[i, 1]),
        old_curve_std=float(old_curve[i, 2]),
        rerun_seeds_0_2_mean=float(np.mean(vals012)),
        rerun_seeds_0_2_std=float(np.std(vals012)),
        delta=float(np.mean(vals012) - old_curve[i, 1]),
    )

out = dict(
    simulator=("code/sim_cell.py with the T004 fleet-conservation fix; "
               "5-min steps, 288 steps/day"),
    inputs="data/cell_sim_inputs.npz (dep5, Pest, I0c, zid, xy)",
    fleet_calibrations=dict(old=FLEET_OLD, new=FLEET_NEW,
                            note=("old is the calibration behind the paper's "
                                  "13.4%; new is T001-R2's closed-fleet centre. "
                                  "The radius comparison uses old; the new-fleet "
                                  "row is a bridge only.")),
    seeds=SEEDS,
    results=results,
    reproduce_old_curve=repro,
    runtime_s=float(time.time() - t0),
)
json.dump(out, io.open(os.path.join(RUN, "sim_results.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
