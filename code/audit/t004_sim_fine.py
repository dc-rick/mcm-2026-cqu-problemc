"""T004 step 4: simulator runs on the coarse (0.005 deg) and fine (0.0025 deg)
grids at the closed-fleet centre 495,540.

Main specification asked by the task sheet: coarse grid, R=450 m, alpha=1 and
0.8, five seeds.  Sensitivity: R=300 and R=600.  The fine grid repeats the same
grid with its own last-trip-end I0 distribution (total fixed at 495,540), plus
two checks: zero-weight allocation for the 25 no-evidence cells, and a
demand-matched fleet (fine grid captures 90.2% of coarse typical departures).
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
SEEDS = list(range(5))
FLEET = 495540.0
ALPHAS = (1.0, 0.8)


def run_grid(npz_path, R, alpha, seed, i0_key="I0c", fleet_scale=1.0):
    z = np.load(npz_path)
    S = sim_cell.CellSim(z["dep5"], z["Pest"], z[i0_key] * alpha * fleet_scale,
                         z["xy"], z["zid"], rng=np.random.default_rng(seed),
                         walk_km=R)
    return float(S.run()["unmet_share"])


def block(npz, label, R, alpha, i0_key="I0c", fleet_scale=1.0,
          base_fleet=FLEET):
    vals = [run_grid(npz, R, alpha, s, i0_key, fleet_scale) for s in SEEDS]
    rec = dict(grid=label, R_m=int(R * 1000), alpha=alpha,
               fleet=base_fleet * alpha * fleet_scale, i0=i0_key,
               seeds=SEEDS, values=vals, mean=float(np.mean(vals)),
               std=float(np.std(vals, ddof=1)))
    print(f"{label:6s} R={R:.2f} a={alpha:.1f} {i0_key:8s} "
          f"mean={rec['mean']:.5f} sd={rec['std']:.5f}")
    return rec


coarse = os.path.join(ROOT, "data", "cell_sim_inputs.npz")
fine = os.path.join(RUN, "fine_sim_inputs.npz")
out = dict(fleet=FLEET, seeds=SEEDS,
           inputs=dict(coarse="data/cell_sim_inputs.npz",
                       fine="data/runs/T004/fine_sim_inputs.npz"),
           runs={}, runtime_s=None)
t0 = time.time()
# data/cell_sim_inputs.npz was built with FLEET=343,638; scale it to the
# closed-fleet centre.  The fine inputs were built at 495,540 directly.
COARSE_SCALE = FLEET / 343638.0
for R in (0.30, 0.45, 0.60):
    for a in ALPHAS:
        key = f"coarse_R{int(R*1000)}_a{a:.1f}"
        out["runs"][key] = block(coarse, "coarse", R, a,
                                 fleet_scale=COARSE_SCALE,
                                 base_fleet=343638.0)
for R in (0.30, 0.45, 0.60):
    for a in ALPHAS:
        key = f"fine_R{int(R*1000)}_a{a:.1f}"
        out["runs"][key] = block(fine, "fine", R, a)

# no-evidence cells start empty instead of receiving the zone-median default
for a in ALPHAS:
    key = f"fine_R450_a{a:.1f}_I0zero"
    out["runs"][key] = block(fine, "fine", 0.45, a, i0_key="I0_zero")

# demand-matched: fine grid has 90.2% of coarse typical departures
scale = 247460.57142857136 / 274357.4285714286
for a in ALPHAS:
    key = f"fine_R450_a{a:.1f}_demandmatched"
    out["runs"][key] = block(fine, "fine", 0.45, a, fleet_scale=scale)

out["runtime_s"] = float(time.time() - t0)
json.dump(out, io.open(os.path.join(RUN, "sim_fine.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved", os.path.join(RUN, "sim_fine.json"),
      f"runtime {out['runtime_s']:.1f}s")
