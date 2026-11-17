"""T007: rerun Q2 under H13 arrival conservation and assemble the evidence.

Runs the Q1-demand LP at alpha=1 K in {20,40,60,80}, alpha=0.8/0.6 at K=80,
and the observed-demand baseline at alpha=1 K=60; every run uses
--conserve, fixed_K_cost=600 and the full 343,638-bike scale.  It then checks
hourly city/zone inventory balances, computes the 21:00-06:00 shares from the
main plan, and calls 22_q2_sim.py for the three-policy stochastic comparison.

Outputs (all under data/runs/T007/):
  q2_results_q1.json, q2_plan_q1.npz, q2_conservation.json,
  q2_night.json, q2_sim_q1.json, q2_greedy_check.json,
  plans/<scenario>.json and plans/<scenario>.npz.
"""
import json
import os
import subprocess
import sys

import numpy as np

import lib
from lib import DATA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, "code")
RUN = os.path.join(DATA, "runs", "T007")
PLANS = os.path.join(RUN, "plans")
SCALE = 1.0 / 0.128
FIXED_K = 600.0
TOL = 1e-6

SCEN = [
    ("q1_a1_K60", "q1", 1.0, 60),
    ("q1_a1_K40", "q1", 1.0, 40),
    ("q1_a1_K20", "q1", 1.0, 20),
    ("q1_a1_K80", "q1", 1.0, 80),
    ("q1_a0.8_K80", "q1", 0.8, 80),
    ("q1_a0.6_K80", "q1", 0.6, 80),
    ("obs_a1_K60", "obs", 1.0, 60),
]


def load_demand(kind):
    if kind == "q1":
        z = np.load(os.path.join(DATA, "runs", "T006", "q1_od.npz"))
        od = z["od_q1"].astype(np.float64) * SCALE
        cent = np.load(os.path.join(DATA, "typical_day.npz"),
                       allow_pickle=True)["cent"]
    else:
        z = np.load(os.path.join(DATA, "typical_day.npz"), allow_pickle=True)
        od = z["od_z"].astype(np.float64) * SCALE
        cent = z["cent"]
    dep = od.sum(2).T
    row = od.sum(2)
    PZ = np.divide(od, row[:, :, None], out=np.zeros_like(od),
                   where=row[:, :, None] > 0)
    return dep, PZ, cent


def run_lp(name, kind, alpha, K):
    out_json = os.path.join(PLANS, name + ".json")
    out_npz = os.path.join(PLANS, name + ".npz")
    cmd = [sys.executable, os.path.join(CODE, "21_q2_opt.py"),
           "--od", kind, "--alpha", str(alpha), "--K", str(K),
           "--conserve", "--fixed-k-cost", str(FIXED_K),
           "--out", out_json, "--plan-out", out_npz]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed\n{proc.stdout}\n{proc.stderr}")
    with open(out_json, encoding="utf-8") as fh:
        res = json.load(fh)
    plan = np.load(out_npz)
    return res, plan


def conservation(plan, dep, PZ, cent, alpha):
    I = plan["I"].astype(np.float64)
    U = plan["U"].astype(np.float64)
    XM = plan["XM"].astype(np.float64)
    served = dep - U
    Z, T = dep.shape
    arr = np.stack([PZ[t].T @ served[:, t] for t in range(T)], axis=1)
    city_res = []
    zone_res = []
    for t in range(T - 1):
        pred = arr[:, t].sum() - served[:, t].sum()
        actual = I[:, t + 1].sum() - I[:, t].sum()
        city_res.append(float(actual - pred))
        zone_res.append(float(np.abs(
            I[:, t + 1] - I[:, t] - arr[:, t] + served[:, t]
            + XM[t].sum(1) - XM[t].sum(0)).max()))
    st = np.load(os.path.join(DATA, "q2_state.npz"), allow_pickle=True)
    I0 = st["I0"] * alpha
    day_end = float(I[:, -1].sum() + arr[:, -1].sum() - served[:, -1].sum()
                    - I0.sum())
    u_over = float(np.max(U - dep))
    return {
        "city_residual_max_abs": max(abs(v) for v in city_res),
        "city_residual_by_hour": city_res,
        "zone_residual_max_abs": max(zone_res),
        "day_end_city_diff": day_end,
        "initial_city_diff": float(I[:, 0].sum() - I0.sum()),
        "u_minus_dep_max": u_over,
        "u_min": float(U.min()),
        "passes": (max(abs(v) for v in city_res) <= TOL
                   and max(zone_res) <= TOL
                   and abs(day_end) <= TOL and u_over <= TOL),
    }


def main():
    os.makedirs(PLANS, exist_ok=True)
    dist_cache = {}
    results = {}
    cons = {}
    plans = {}
    for name, kind, alpha, K in SCEN:
        print(f"\n=== {name}: od={kind} alpha={alpha} K={K} ===")
        res, plan = run_lp(name, kind, alpha, K)
        dep, PZ, cent = load_demand(kind)
        c = conservation(plan, dep, PZ, cent, alpha)
        cons[name] = c
        moved = float(plan["XM"].sum())
        results[name] = {
            "od": kind, "alpha": alpha, "K": K, "conserve": True,
            "fixed_k_cost": FIXED_K, "status": res["status"],
            "obj": res["obj"], "obj_variable": res["obj_variable"],
            "moved": moved, "truck_km": res["truck_km"],
            "unmet": res["unmet"], "dev": res["dev"],
            "trucks_used": res["trucks_used"], "trips": res["trips"],
            "gap": res["gap"],
            "plan_json": f"data/runs/T007/plans/{name}.json",
            "plan_npz": f"data/runs/T007/plans/{name}.npz",
        }
        plans[name] = plan
        # sanity: XM row sums reproduce moved, and no negative flows
        assert abs(moved - res["moved"]) <= 1e-6 * max(1.0, moved)
        assert float(plan["XM"].min()) >= -1e-9
        print(f"  moved={moved:.2f} truck-km={res['truck_km']:.2f} "
              f"unmet={res['unmet']:.2f} dev={res['dev']:.2f} "
              f"obj={res['obj']:.2f} cons={c['passes']}")

    # main plan = alpha=1 K=60 q1, with MK for the hourly truck-km split
    main_plan = plans["q1_a1_K60"]
    np.savez(os.path.join(RUN, "q2_plan_q1.npz"),
             XM=main_plan["XM"], X=main_plan["X"], M=main_plan["M"],
             I=main_plan["I"], U=main_plan["U"], MK=main_plan["MK"])

    dep, PZ, cent = load_demand("q1")
    dist = np.sqrt(((cent[:, None, :] - cent[None, :, :]) ** 2).sum(-1))
    moved_h = main_plan["XM"].sum((1, 2))
    truck_km_h = np.einsum("tij,ij->t", main_plan["MK"], dist)
    night_hours = [0, 1, 2, 3, 4, 5, 21, 22, 23]
    day_hours = [h for h in range(24) if h not in night_hours]
    night = {
        "definition": "21:00-06:00 = hours 21,22,23,0,1,2,3,4,5 (9 h)",
        "night_hours": night_hours,
        "day_hours": day_hours,
        "moved_night": float(moved_h[night_hours].sum()),
        "moved_day": float(moved_h[day_hours].sum()),
        "moved_night_share": float(moved_h[night_hours].sum()
                                   / max(moved_h.sum(), 1e-12)),
        "truck_km_night": float(truck_km_h[night_hours].sum()),
        "truck_km_day": float(truck_km_h[day_hours].sum()),
        "truck_km_night_share": float(truck_km_h[night_hours].sum()
                                      / max(truck_km_h.sum(), 1e-12)),
        "moved_by_hour": [float(v) for v in moved_h],
        "truck_km_by_hour": [float(v) for v in truck_km_h],
        "note": "Shares are descriptive; the LP has no night cost and this "
                "task adds none.",
        "scenario": "q1_a1_K60",
    }
    with open(os.path.join(RUN, "q2_night.json"), "w", encoding="utf-8") as fh:
        json.dump(night, fh, ensure_ascii=False, indent=1)

    cc = {"tolerance": TOL,
          "same_hour_arrival_convention":
              "arr_served[z',t] is credited in the same hour t as the served "
              "departure dep[z,t] - u[z,t]; no cross-hour lag.",
          "scenarios": cons}
    with open(os.path.join(RUN, "q2_conservation.json"), "w",
              encoding="utf-8") as fh:
        json.dump(cc, fh, ensure_ascii=False, indent=1)

    q1 = results["q1_a1_K60"]
    obs = results["obs_a1_K60"]
    keys = ["moved", "truck_km", "unmet", "dev", "obj"]
    comparison = {
        "settings": {"alpha": 1.0, "K": 60, "conserve": True,
                     "fixed_k_cost": FIXED_K},
        "q1": q1, "obs": obs,
        "q1_minus_obs": {k: q1[k] - obs[k] for k in keys},
        "relative_q1_vs_obs": {k: (q1[k] - obs[k]) / obs[k] if obs[k] else None
                               for k in keys},
    }
    out = {
        "settings": {
            "scale": SCALE, "I0_full": 343638.0,
            "fixed_k_cost": FIXED_K, "conserve": True,
            "same_hour_arrival": True,
            "note": "Q1 demand is primary; observed OD is the baseline. All "
                    "runs use --conserve with u<=dep.",
        },
        "scenarios": {k: v for k, v in results.items()
                      if k.startswith("q1_")},
        "obs_baseline": obs,
        "q1_vs_obs": comparison,
        "unrerun": "The full alpha=0.4-1.0 x K grid was not run; only "
                   "alpha=0.8 and 0.6 at K=80 were added to the alpha=1 "
                   "K=20/40/60/80 set.",
    }
    with open(os.path.join(RUN, "q2_results_q1.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)

    sim_cmd = [sys.executable, os.path.join(CODE, "22_q2_sim.py"),
               "--od", "q1", "--plan", os.path.join(RUN, "q2_plan_q1.npz"),
               "--out", os.path.join(RUN, "q2_sim_q1.json"),
               "--greedy-check-out", os.path.join(RUN, "q2_greedy_check.json"),
               "--K", "60", "--reps", "20"]
    print("\n=== 22_q2_sim.py ===")
    proc = subprocess.run(sim_cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError("22_q2_sim.py failed")

    with open(os.path.join(RUN, "q2_greedy_check.json"), encoding="utf-8") as fh:
        g = json.load(fh)
    print(json.dumps({"conservation_all_pass": all(v["passes"] for v in cons.values()),
                      "night_share": night["moved_night_share"],
                      "truck_km_night_share": night["truck_km_night_share"],
                      "greedy_max_hourly": g["max_hourly_moved"],
                      "greedy_max_ratio": g["max_source_bikes_per_truck"]},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
