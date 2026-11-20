"""T030: Q4 rebuild with a conserved fleet.

T019 sizes each cell from an independent Poisson random walk
`occ <- max(0, occ + A - D)`.  Arrivals are not tied to departures, so the
system total grows to 630,002.78 even though only 343,638 bicycles exist.
This module replaces that demand stage with a conserving process:

  * a departure takes a bicycle from the cell's stock; if the cell is empty
    the departure is unmet and generates no arrival;
  * the served departures of a 15-minute step are routed over the Q1 arrival
    surface, so the city total is exactly I0 at every step;
  * the per-cell 95th percentile of the peak is the new sizing demand a_i.

The layout MILP, the 150-slot cap, the Kneedle rule and the 450 m radius are
unchanged; only `a` and the replay change.  The replay assigns each cell to
its serving fence and keeps an outside state, so an overflow bicycle is not
deleted: inside + outside = fleet at every step.

Stages: demand | solve | replay | all
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
import t019_q4_cap as t19

RUN = os.path.join(lib.DATA, "runs", "T030")
os.makedirs(RUN, exist_ok=True)
CAP = t19.CAP
R = t19.R_MAIN
REPS = t19.REPS
SEED = t19.SEED


def dump(obj, name):
    path = os.path.join(RUN, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    print("wrote", path, flush=True)


def load_json(name):
    with open(os.path.join(RUN, name), encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
def conserved_demand(arr15, dep15, I0, reps=REPS, seed=SEED):
    """Per-cell P95 peak occupancy of a conserving fleet process."""
    T, C = arr15.shape
    p = np.divide(arr15, arr15.sum(1, keepdims=True),
                  out=np.full_like(arr15, 1.0 / C),
                  where=arr15.sum(1, keepdims=True) > 0)
    peak = np.zeros((reps, C))
    total_resid = 0.0
    for r in range(reps):
        rng = np.random.default_rng(seed + r)
        I = I0.astype(float).copy()
        pk = I.copy()
        for t in range(T):
            d = rng.poisson(dep15[t])
            take = np.minimum(d, np.floor(I))
            I -= take
            served = int(round(float(take.sum())))
            if served > 0:
                I += rng.multinomial(served, p[t])
            pk = np.maximum(pk, I)
        total_resid = max(total_resid, abs(float(I.sum()) - float(I0.sum())))
        peak[r] = pk
    return dict(a=np.percentile(peak, 95, axis=0), peak=peak,
                total_resid=float(total_resid))


def demand(force=False):
    path = os.path.join(RUN, "cell_demand_cons.npz")
    if os.path.exists(path) and not force:
        print("reuse", path)
        return dict(np.load(path, allow_pickle=True))
    d = t19.load_inputs()
    arr15 = t19.to15(d["arr"])
    dep15 = t19.to15(d["dep"])
    out = {}
    meta = {}
    for lvl, I0 in (("meas343", d["I0_meas"]), ("fleet495", d["I0_fleet"])):
        t0 = time.time()
        r = conserved_demand(arr15, dep15, I0)
        out[f"a_{lvl}"] = r["a"]
        meta[lvl] = dict(
            sum=float(r["a"].sum()), max=float(r["a"].max()),
            median=float(np.median(r["a"])),
            over_cap=int((r["a"] > CAP).sum()),
            total_resid=float(r["total_resid"]),
            runtime_s=float(time.time() - t0))
        print(lvl, meta[lvl], flush=True)
    np.savez(path, **out, I0_meas=d["I0_meas"], I0_fleet=d["I0_fleet"],
             arr=d["arr"], dep=d["dep"], xy=d["xy"],
             arr15=arr15, dep15=dep15)
    dump(dict(task="T030 conserved Q4 demand", date="2026-11-19",
              reps=REPS, seed=SEED, levels=meta,
              note=("arrivals are multinomial over the planned Q1 arrival "
                    "surface with size equal to the served departures of "
                    "the same 15-minute step; total = I0 at every step; "
                    "P95 over reps of the peak including t=0")),
         "cell_demand_cons.json")
    return dict(np.load(path, allow_pickle=True))


# --------------------------------------------------------------------------
def solve(force=False):
    path = os.path.join(RUN, "frontier_cons.json")
    if os.path.exists(path) and not force:
        print("reuse", path)
        return load_json("frontier_cons.json")
    z = np.load(os.path.join(RUN, "cell_demand_cons.npz"))
    arr_day = z["arr"].sum(0)
    D = np.load(os.path.join(lib.DATA, "dist_matrix.npy"))
    ii, jj = t19.build_arcs(D, R)
    out = {}
    for level in ("meas343", "fleet495"):
        a = z[f"a_{level}"]
        I0 = z["I0_meas" if level == "meas343" else "I0_fleet"]
        print(f"=== {level}: sum a={a.sum():.1f} ===", flush=True)
        ckpt = os.path.join(RUN, f"frontier_points_{level}.json")
        points = []
        if os.path.exists(ckpt):
            with open(ckpt, encoding="utf-8") as fh:
                points = json.load(fh)
            print(f"resume from {ckpt}: {len(points)} points", flush=True)
        done = {p["n"] for p in points}
        r_all = t19.solve_cap(a, ii, jj, n=None, integer=False,
                              time_limit=300)
        for n in t19.N_GRID:
            if int(n) in done:
                continue
            r = t19.solve_cap(a, ii, jj, n=n, integer=True,
                              time_limit=300, mip_gap=0.005)
            rec = dict(n=int(n), coverage=float(r["coverage"]),
                       arrival_coverage=float(t19.arrival_coverage(
                           r["u"], a, arr_day)),
                       assigned=float(r["assigned"]),
                       unassigned=float(r["unassigned"]),
                       slots=float(CAP * n), obj=float(r["obj"]),
                       bound=float(r["bound"]), gap=float(r["gap"]),
                       runtime=float(r["runtime"]), status=int(r["status"]))
            points.append(rec)
            points.sort(key=lambda p: p["n"])
            with open(ckpt, "w", encoding="utf-8") as fh:
                json.dump(points, fh, ensure_ascii=False, indent=1)
            print(f"  n={n:5d} cov={100*r['coverage']:6.3f}% "
                  f"arr={100*rec['arrival_coverage']:6.3f}% "
                  f"gap={100*r['gap']:.3f}% t={r['runtime']:.1f}s",
                  flush=True)
        pn = np.array([p["n"] for p in points], dtype=float)
        pc = np.array([p["coverage"] for p in points], dtype=float)
        xn = (pn - pn[0]) / (pn[-1] - pn[0])
        yn = (pc - pc[0]) / (pc[-1] - pc[0])
        dev = yn - xn
        kbest = int(np.argmax(dev))
        knee = dict(n=int(pn[kbest]), coverage=float(pc[kbest]),
                    deviation_normalised=float(dev[kbest]),
                    rule="point farthest above the chord on the normalised "
                         "(n, coverage) frontier")
        sel = int(knee["n"])
        rs = t19.solve_cap(a, ii, jj, n=sel, integer=True,
                           time_limit=300, mip_gap=0.005)
        rfull = t19.solve_cap(a, ii, jj, n=int(len(a)), integer=True,
                              time_limit=300, mip_gap=0.005)
        np.savez(os.path.join(RUN, f"plan_{level}_main.npz"),
                 s=rs["s"], f=rs["f"], u=rs["u"], ii=ii, jj=jj,
                 a=a, I0=I0)
        np.savez(os.path.join(RUN, f"plan_{level}_all.npz"),
                 s=rfull["s"], f=rfull["f"], u=rfull["u"], ii=ii, jj=jj,
                 a=a, I0=I0)
        out[level] = dict(
            all_open_lp=dict(
                coverage=float(r_all["coverage"]),
                arrival_coverage=float(t19.arrival_coverage(
                    r_all["u"], a, arr_day)),
                assigned=float(r_all["assigned"]),
                unassigned=float(r_all["unassigned"]),
                bound=float(r_all["bound"])),
            all_open_milp=dict(
                coverage=float(rfull["coverage"]),
                arrival_coverage=float(t19.arrival_coverage(
                    rfull["u"], a, arr_day)),
                assigned=float(rfull["assigned"]),
                unassigned=float(rfull["unassigned"]),
                n_open=int(rfull["n_open"]), gap=float(rfull["gap"])),
            knee=knee, selected_n=sel,
            selected=dict(
                coverage=float(rs["coverage"]),
                arrival_coverage=float(t19.arrival_coverage(
                    rs["u"], a, arr_day)),
                assigned=float(rs["assigned"]),
                unassigned=float(rs["unassigned"]),
                slots=float(CAP * sel), gap=float(rs["gap"]),
                bound=float(rs["bound"])),
            points=points, demand_sum=float(a.sum()),
            arrival_sum=float(arr_day.sum()), n_cells=int(len(a)))
    dump(dict(task="T030 Q4 conserved layout", date="2026-11-19",
              radius_m=float(R * 1000), cap=float(CAP),
              levels=out,
              note=("same MILP, Kneedle rule and grid as T019; only the "
                    "per-cell sizing demand a_i is replaced by the "
                    "conserved-process P95")),
         "frontier_cons.json")
    return out


# --------------------------------------------------------------------------
def replay_plan(plan, arr15, dep15, cap=CAP):
    s, f, a, ii, jj, I0 = (plan["s"], plan["f"], plan["a"],
                           plan["ii"], plan["jj"], plan["I0"])
    C = len(a)
    T = arr15.shape[0]
    open_sites = np.where(s > 0.5)[0]
    site_pos = {int(j): k for k, j in enumerate(open_sites)}
    spare = np.zeros((T, len(open_sites)))
    frac = np.divide(f, a[ii], out=np.zeros_like(f), where=a[ii] > 0)
    assigned = np.zeros(C)
    np.add.at(assigned, ii, frac)
    occ = np.zeros(C)
    np.add.at(occ, jj, I0[ii] * frac)
    outside = np.zeros(C)
    oa = np.maximum(0.0, occ - cap)
    outside += oa
    occ = np.minimum(occ, cap)
    outside_unassigned = float((I0 * (1.0 - assigned)).sum())
    fleet = float(I0.sum())
    full_ever = np.zeros(C, bool)
    unmet = 0.0
    overflow = 0.0
    covered = 0.0
    planned_arr = 0.0
    actual_arr = 0.0
    full_steps = 0
    max_resid = 0.0
    covered_dep = 0.0
    unmet_h = np.zeros(24)
    overflow_h = np.zeros(24)
    full_h = np.zeros(24, int)
    for t in range(T):
        req = np.zeros(C)
        np.add.at(req, jj, dep15[t, ii] * frac)
        covered_dep += float(req.sum())
        take = np.minimum(req, occ)
        resid = req - take
        to = np.minimum(resid, outside)
        occ -= take
        outside -= to
        unmet += float((resid - to).sum())
        unmet_h[t * 24 // T] += float((resid - to).sum())
        served = float(take.sum() + to.sum())
        # planned arrival pattern, rescaled to the served departures
        ain = np.zeros(C)
        np.add.at(ain, jj, arr15[t, ii] * frac)
        planned = float(arr15[t].sum())
        planned_arr += planned
        actual = served
        actual_arr += actual
        if ain.sum() > 1e-12 and actual > 0:
            ain = ain * (actual / ain.sum())
        elif actual > 0:
            ain = np.full(C, actual / C)
        newocc = occ + ain
        over = np.maximum(0.0, newocc - cap)
        occ = np.minimum(newocc, cap)
        outside += over
        overflow += float(over.sum())
        overflow_h[t * 24 // T] += float(over.sum())
        covered += float((ain - over).sum())
        full = occ >= cap - 1e-9
        full_ever |= full
        full_steps += int(full.sum())
        full_h[t * 24 // T] += int(full.sum())
        for j in open_sites:
            spare[t, site_pos[int(j)]] = cap - occ[j]
        max_resid = max(max_resid,
                        abs(float(occ.sum()) + float(outside.sum())
                            + outside_unassigned - fleet))
    return dict(
        unmet=float(unmet), overflow=float(overflow),
        covered_arrivals=float(covered), planned_arrivals=float(planned_arr),
        actual_arrivals=float(actual_arr),
        realized_coverage=float(covered / max(planned_arr, 1e-9)),
        realized_coverage_actual=float(covered / max(actual_arr, 1e-9)),
        outside_end=float(outside.sum() + outside_unassigned),
        outside_assigned_end=float(outside.sum()),
        outside_unassigned=float(outside_unassigned),
        covered_departures=float(covered_dep),
        departure_shortfall_rate=float(unmet / max(covered_dep, 1e-9)),
        fleet=float(fleet),
        max_conservation_resid=float(max_resid),
        full_sites_ever=int(full_ever.sum()),
        full_site_steps=int(full_steps),
        departure_shortfall=float(unmet),
        unmet_h=unmet_h, overflow_h=overflow_h, full_sites_h=full_h,
        spare=spare, open_sites=open_sites)


def replay(force=False):
    path = os.path.join(RUN, "replay_cons.json")
    if os.path.exists(path) and not force:
        print("reuse", path)
        return load_json("replay_cons.json")
    z = np.load(os.path.join(RUN, "cell_demand_cons.npz"))
    out = {}
    for level in ("meas343", "fleet495"):
        plan = dict(np.load(os.path.join(RUN, f"plan_{level}_main.npz")))
        r15 = replay_plan(plan, z["arr15"], z["dep15"])
        r60 = replay_plan(plan, z["arr"], z["dep"])
        for rr in (r15, r60):
            rr["unmet_h"] = [float(x) for x in rr["unmet_h"]]
            rr["overflow_h"] = [float(x) for x in rr["overflow_h"]]
            rr["full_sites_h"] = [int(x) for x in rr["full_sites_h"]]
        np.savez(os.path.join(RUN, f"spare_{level}.npz"),
                 spare=r15.pop("spare"), sites=r15.pop("open_sites"),
                 time_index=np.arange(96))
        r60.pop("spare"), r60.pop("open_sites")
        out[level] = dict(
            design_15=r15, design_60=r60,
            delta_coverage=r60["realized_coverage"]
            - r15["realized_coverage"],
            delta_shortfall=r60["departure_shortfall"]
            - r15["departure_shortfall"])
        print(level, "15m", r15, flush=True)
        print(level, "60m", r60, flush=True)
    dump(dict(task="T030 Q4 conserved replay", date="2026-11-19",
              levels=out,
              note=("site-level replay: departures take inside first, then "
                    "the outside pool; arrivals are rescaled to the served "
                    "departures of the step; inside + outside = fleet")),
         "replay_cons.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=("demand", "solve", "replay", "all"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.stage in ("demand", "all"):
        demand(force=args.force)
    if args.stage in ("solve", "all"):
        solve(force=args.force)
    if args.stage in ("replay", "all"):
        replay(force=args.force)
