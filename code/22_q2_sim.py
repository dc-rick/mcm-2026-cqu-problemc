"""Q2 evaluation: stochastic simulation of the LP schedule against baselines.

--od q1 (default) takes the demand surface from data/runs/T006/q1_od.npz;
--od obs takes the observed typical day.  --plan points at the saved LP plan
(default data/q2_plan.npz; T007 passes data/runs/T007/q2_plan_q1.npz).
--out writes the simulation table (default data/q2_sim.json).

H14 fix: the greedy baseline now accounts for trucks, not just bikes.
trucks_at[z] is the number of trucks at z at the start of the hour; the bikes
movable from z are Q*trucks_at[z]; moving `amt` bikes consumes amt/Q trucks,
which then sit at the destination.  All three policies share the same
position, travel-time and inventory rules inside sim.Sim.
"""
import argparse
import json
import os
import numpy as np
import lib
from lib import DATA
import sim

ap = argparse.ArgumentParser(description="Q2 stochastic evaluation")
ap.add_argument("--od", choices=("q1", "obs"), default="q1")
ap.add_argument("--plan", default=os.path.join(DATA, "q2_plan.npz"))
ap.add_argument("--out", default=os.path.join(DATA, "q2_sim.json"))
ap.add_argument("--greedy-check-out", default=None)
ap.add_argument("--K", type=int, default=60)
ap.add_argument("--Q", type=int, default=40)
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--alphas", default="1.0,0.8,0.6")
args = ap.parse_args()

SCALE = 1.0 / 0.128
if args.od == "q1":
    z = np.load(os.path.join(DATA, "runs", "T006", "q1_od.npz"))
    od_hr = z["od_q1"].astype(np.float64) * SCALE   # (24,Z,Z) true-scale hourly OD
    cent = np.load(os.path.join(DATA, "typical_day.npz"),
                   allow_pickle=True)["cent"]
else:
    z = np.load(os.path.join(DATA, "typical_day.npz"), allow_pickle=True)
    od_hr = z["od_z"] * SCALE                 # (24,Z,Z) true-scale hourly OD
    cent = z["cent"]
dep_hr = od_hr.sum(2)
NZ = dep_hr.shape[1]

st = np.load(os.path.join(DATA, "q2_state.npz"), allow_pickle=True)
I0_base = st["I0"]
sol = np.load(args.plan)

dist = np.sqrt(((cent[:, None, :] - cent[None, :, :]) ** 2).sum(-1))


def greedy_plan(K, Q=40, alpha=1.0):
    """Rule: each hour, move bikes from the zones with the largest surplus over
    the initial share to the zones with the largest forecast deficit, nearest
    first.  Capacity is truck-based: Q bikes per truck at the source, and each
    moved batch takes its trucks to the destination."""
    plan = np.zeros((24, NZ, NZ))
    trucks_at = np.full(NZ, K / NZ, dtype=float)
    trucks_start = np.zeros((24, NZ))
    for t in range(24):
        trucks_start[t] = trucks_at
        need = od_hr[t].sum(1) - od_hr[t].sum(0)          # net outflow this hour
        surplus = np.argsort(need)                     # most negative first
        deficit = np.argsort(need)[::-1]
        avail = Q * trucks_at                          # bikes movable at source
        for s in surplus[:15]:
            for d in deficit[:15]:
                if need[s] >= 0 or need[d] <= 0:
                    continue
                amt = min(-need[s], need[d], avail[s], float(Q))
                if amt <= 1:
                    continue
                plan[t, s, d] += amt
                avail[s] -= amt
                trucks_at[s] -= amt / Q
                trucks_at[d] += amt / Q
                need[s] += amt
                need[d] -= amt
    return plan, trucks_start


def greedy_check(plan, trucks_start, K, Q):
    moved_h = plan.sum((1, 2))
    ratio = np.divide(plan.sum(2), trucks_start, out=np.zeros_like(trucks_start),
                      where=trucks_start > 0)
    return {
        "K": K, "Q": Q, "trucks_total": float(K),
        "max_hourly_moved": float(moved_h.max()),
        "hourly_moved": [float(v) for v in moved_h],
        "hourly_limit": float(Q * K),
        "max_source_bikes_per_truck": float(ratio.max()),
        "source_limit": float(Q),
        "hourly_violations": int((moved_h > Q * K + 1e-9).sum()),
        "source_violations": int((ratio > Q + 1e-9).sum()),
        "trucks_start_sum_min": float(trucks_start.sum(1).min()),
        "trucks_start_sum_max": float(trucks_start.sum(1).max()),
    }


def evaluate(plan, alpha=1.0, reps=20, label=""):
    I0 = I0_base * alpha
    res = []
    for r in range(reps):
        S = sim.Sim(dep_hr.T, od_hr, I0, seed=r)
        out = S.run(move_plan=plan if plan is not None else None)
        res.append(out["unmet_share"])
    m = np.mean(res); sd = np.std(res)
    print(f"{label:22s} unmet {100*m:6.2f}% +/- {100*sd:.2f}")
    return m, sd


if __name__ == "__main__":
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    greedy, trucks_start = greedy_plan(args.K, Q=args.Q)
    gcheck = greedy_check(greedy, trucks_start, args.K, args.Q)
    if args.greedy_check_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.greedy_check_out)),
                    exist_ok=True)
        with open(args.greedy_check_out, "w", encoding="utf-8") as fh:
            json.dump(gcheck, fh, ensure_ascii=False, indent=1)

    rows = []
    for a in alphas:
        for tag, plan in (("no trucks", None),
                          ("LP plan", sol["XM"] if a == 1.0 else None),
                          ("greedy", greedy)):
            if tag == "LP plan" and a != 1.0:
                continue
            m, sd = evaluate(plan, alpha=a, reps=args.reps,
                             label=f"a={a} {tag}")
            rows.append(dict(alpha=a, policy=tag, unmet=m, sd=sd))
    out = {"od": args.od, "plan": os.path.relpath(args.plan, DATA).replace(os.sep, "/")
           if os.path.isabs(args.plan) else args.plan,
           "K": args.K, "Q": args.Q, "reps": args.reps,
           "alphas": alphas, "seed_base": 0, "rows": rows,
           "greedy_check": gcheck}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("wrote", args.out)
