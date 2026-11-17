"""Q2 evaluation: stochastic simulation of the LP schedule against baselines.

--od q1 (default) takes the demand surface from data/runs/T006/q1_od.npz;
--od obs takes the observed typical day.  The simulation equations are unchanged.
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

st = np.load(DATA + r"\q2_state.npz", allow_pickle=True)
I0_base = st["I0"]
sol = np.load(DATA + r"\q2_plan.npz")

dist = np.sqrt(((cent[:, None, :] - cent[None, :, :]) ** 2).sum(-1))


def greedy_plan(K, Q=40, alpha=1.0):
    """Rule: each hour, move bikes from the zones with the largest surplus over the
    initial share to the zones with the largest forecast deficit, nearest first."""
    I0 = I0_base * alpha
    plan = np.zeros((24, NZ, NZ))
    trucks_at = np.full(NZ, K / NZ)
    for t in range(24):
        # projected end-of-day imbalance resolved per hour: simple target = I0 share
        need = od_hr[t].sum(1) - od_hr[t].sum(0)          # net outflow this hour
        surplus = np.argsort(need)                     # most negative first
        deficit = np.argsort(need)[::-1]
        cap = Q * trucks_at
        for s in surplus[:15]:
            for d in deficit[:15]:
                if need[s] >= 0 or need[d] <= 0:
                    continue
                amt = min(-need[s], need[d], cap[s], 40.0)
                if amt <= 1:
                    continue
                plan[t, s, d] += amt
                cap[s] -= amt / Q
                need[s] += amt
                need[d] -= amt
        trucks_at = np.minimum(trucks_at + 0.0, trucks_at.max())
    return plan


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
    rows = []
    for a in (1.0, 0.8, 0.6):
        for tag, plan in (("no trucks", None),
                          ("LP plan", sol["XM"] if a == 1.0 else None),
                          ("greedy", greedy_plan(60, alpha=a))):
            if tag == "LP plan" and a != 1.0:
                continue
            m, sd = evaluate(plan, alpha=a, reps=20, label=f"a={a} {tag}")
            rows.append(dict(alpha=a, policy=tag, unmet=m, sd=sd))
    json.dump(rows, open(DATA + r"\q2_sim.json", "w"), indent=1)
