"""T019: Q4 capacitated fence siting/allocation/capacity.

Demand unit: each of the 5,263 cells has a parking demand a_i = the 95th
percentile of its peak 15-minute occupancy, from the same Monte Carlo scheme
as T009 (300 Poisson replications, trajectory starts at the cell's standing
inventory, peak includes t=0).  Sites are cell centres; a demand cell may be
split across sites within R; each site holds at most 150 slots; unassigned
demand is explicit.

Stages (each writes only data/runs/T019/):
  demand   cell-level a_i for meas343 and fleet495, comparison to T009
  solve    capacitated MILP frontier over the number of sites n
  radius   R in {150, 300, 450} m sweep at the selected n
  replay   15-minute replay on the design day and on 2017-05-12/19
  walk     walking distance of uncovered demand to the nearest open site
  summary  assemble summary.json, radius_sweep.json, uncovered_walk.json,
           spare_capacity.npz

Run with python -I code/t019_q4_cap.py --stage all
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

import gurobipy as gp
from gurobipy import GRB

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
from lib import DATA

RUN = os.path.join(DATA, "runs", "T019")
os.makedirs(RUN, exist_ok=True)

SCALE = 1.0 / 0.128
FLEET_MEAS = 343638.0
FLEET_CLOSED = 495540.0
CAP = 150.0
REPS = 300
RADII = {"150": 0.150, "300": 0.300, "450": 0.450}
R_MAIN = 0.450
SNAP = pd.Timestamp("2017-05-16 00:00:00")
SEED = 20261118
TARGET = 0.7685847656798482        # T009 705-fence coverage
COMPLY = {"450": 0.0238, "300": 0.0605, "225": 0.0948, "150": 0.1454}
N_GRID = [100, 200, 300, 400, 500, 600, 705, 800, 1000, 1250, 1500,
          1750, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5263]
TE_DAYS = ["2017-05-12", "2017-05-19"]
TR_DAYS = ["2017-05-10", "2017-05-11", "2017-05-15", "2017-05-16",
           "2017-05-18"]
BANDS = [(0, 5), (6, 9), (10, 15), (16, 19), (20, 23)]
BAND_NAMES = ["night 0-5", "morning 6-9", "midday 10-15",
              "evening 16-19", "late 20-23"]
SIGMA_A = 1.2


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(obj, name):
    path = os.path.join(RUN, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    print("wrote", path)
    return path


def load_json(name):
    with open(os.path.join(RUN, name), encoding="utf-8") as fh:
        return json.load(fh)


def load_json_any(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# inputs and demand
# --------------------------------------------------------------------------
def norm_logp(logW):
    logW = logW.astype(np.float32)
    mx = logW.max(1, keepdims=True)
    E = np.exp(logW - mx)
    return logW - mx - np.log(E.sum(1, keepdims=True))


def smooth_attraction(Dj, cells, sigma=SIGMA_A):
    ix = cells.ix.values - cells.ix.values.min()
    iy = cells.iy.values - cells.iy.values.min()
    g = np.zeros((ix.max() + 1, iy.max() + 1))
    g[ix, iy] = Dj
    g = gaussian_filter(g, sigma)
    return g[ix, iy].astype(np.float64)


def build_arr_train(force=False):
    """Cell-level Q1 arrival surface from training-day departures only (M11).

    Same band betas, same training-day attraction A_j, same P(j|i,h) as
    code/12_q1_od.py; only dep_obs uses the five training days instead of
    the seven TYP_DAYS.  Also builds the seven-day version so the T006
    regression and the two-caliber difference can be reported.
    """
    path = os.path.join(RUN, "arr_q1_cell_train.npz")
    diff_path = os.path.join(RUN, "arr_train_diff.json")
    if os.path.exists(path) and not force:
        return dict(np.load(path)), load_json_any(diff_path)
    cells = lib.get_cells()
    n = len(cells)
    D = np.load(os.path.join(DATA, "dist_matrix.npy")).astype(np.float32)
    params = load_json_any(os.path.join(DATA, "q1_params.json"))
    beta = np.array([float(params["bands"][nm]["beta"]) for nm in BAND_NAMES],
                    dtype=np.float32)
    df = lib.load_trips(lib.ALL_DAYS)
    cells2, df = lib.make_cells(df, lib.THR)
    assert len(cells2) == n
    df["band"] = pd.cut(df.hour, [-1, 5, 9, 15, 19, 23],
                        labels=False).astype("Int64")
    ok = df.o.notna() & df.d.notna()
    gt = df[ok].copy()
    gt["o"] = gt.o.astype(int)
    gt["d"] = gt.d.astype(int)

    dep5 = np.zeros((24, n))
    dep7 = np.zeros((24, n))
    s5 = gt[gt.date.isin(TR_DAYS)]
    s7 = gt[gt.date.isin(lib.TYP_DAYS)]
    np.add.at(dep5, (s5.hour.values, s5.o.values), 1.0)
    np.add.at(dep7, (s7.hour.values, s7.o.values), 1.0)
    dep5 /= len(TR_DAYS)
    dep7 /= len(lib.TYP_DAYS)

    hours_of_band = {b: [h for h in range(24) if BANDS[b][0] <= h <= BANDS[b][1]]
                     for b in range(len(BANDS))}
    arr5 = np.zeros((24, n))
    arr7 = np.zeros((24, n))
    for b in range(len(BANDS)):
        tr = gt[(gt.band == b) & gt.date.isin(TR_DAYS)]
        Dj = np.bincount(tr.d.values, minlength=n).astype(np.float64)
        A = smooth_attraction(Dj, cells)
        logA = np.log(np.maximum(A, 1e-12)).astype(np.float32)
        logp = norm_logp(-beta[b] * D + logA[None, :])
        P = np.exp(logp)
        for h in hours_of_band[b]:
            arr5[h] = dep5[h].astype(np.float32) @ P
            arr7[h] = dep7[h].astype(np.float32) @ P
        del P, logp
        print(f"  band {BAND_NAMES[b]}: arrival surface built")
    t6 = np.load(os.path.join(DATA, "runs", "T006", "q1_od.npz"))["arr_q1_cell"]
    t6 = t6.astype(np.float64)
    diff = dict(
        train_days=TR_DAYS, typ_days=lib.TYP_DAYS,
        total_train5=float(arr5.sum()), total_typ7=float(arr7.sum()),
        total_t006=float(t6.sum()),
        ratio_train5_over_typ7=float(arr5.sum() / arr7.sum()),
        l1_train5_vs_typ7=float(np.abs(arr5 - arr7).sum()),
        l1_relative=float(np.abs(arr5 - arr7).sum() / arr7.sum()),
        max_abs_train5_vs_typ7=float(np.abs(arr5 - arr7).max()),
        max_abs_typ7_vs_t006=float(np.abs(arr7 - t6).max()),
        l1_typ7_vs_t006=float(np.abs(arr7 - t6).sum()),
        note="arr7 reproduces T006 arr_q1_cell; arr5 is the M11-compliant "
             "training-day surface used by T019.")
    np.savez(path, arr5=arr5, arr7=arr7, dep5=dep5, dep7=dep7)
    dump(diff, "arr_train_diff.json")
    print("wrote", path)
    return dict(np.load(path)), diff


def load_inputs():
    zt, _ = build_arr_train()
    arr = zt["arr5"].astype(np.float64) * SCALE      # (24, C), training days
    dep = zt["dep5"].astype(np.float64) * SCALE
    cells = lib.get_cells()
    C = len(cells)
    xy = cells[["x", "y"]].to_numpy()
    zid = cells.zone.to_numpy().astype(int)
    D = np.load(os.path.join(DATA, "dist_matrix.npy")).astype(np.float32)

    df = lib.load_trips(lib.ALL_DAYS)
    df = lib.assign_cells(df, cells)
    ok = df.o.notna() & df.d.notna()
    df = df[ok].copy()
    df["o"] = df.o.astype(int)
    df["d"] = df.d.astype(int)

    before = df[df.starttime < SNAP]
    last = before.sort_values("starttime").groupby("bikeid").tail(1)
    cnt = np.bincount(last.d.values, minlength=C).astype(np.float64)
    I0_meas = cnt / cnt.sum() * FLEET_MEAS

    cum = np.cumsum(arr - dep, axis=0)
    raw = np.maximum(0.0, -cum.min(axis=0))
    I0_fleet = raw / raw.sum() * FLEET_CLOSED
    return dict(arr=arr, dep=dep, cells=cells, xy=xy, zid=zid, D=D,
                I0_meas=I0_meas, I0_fleet=I0_fleet, df=df, C=C)


def to15(x):
    return np.repeat(x / 4.0, 4, axis=0)


def cell_demand(arr15, dep15, I0, reps=REPS, seed=SEED):
    """95th percentile of the peak 15-min occupancy, per cell (T009 scheme)."""
    C = arr15.shape[1]
    out = np.empty(C)
    for i in range(C):
        rng = np.random.default_rng(seed + i)
        A = rng.poisson(arr15[:, i], size=(reps, arr15.shape[0]))
        Dp = rng.poisson(dep15[:, i], size=(reps, dep15.shape[0]))
        occ = np.full(reps, float(I0[i]))
        peak = occ.copy()
        for t in range(arr15.shape[0]):
            occ = np.maximum(0.0, occ + A[:, t] - Dp[:, t])
            peak = np.maximum(peak, occ)
        out[i] = float(np.percentile(peak, 95))
    return out


def arrival_weights(a, arr_day):
    """Per-cell weight turning occupancy demand into arrival volume:
    w_i = arr_i / a_i, so unassigned arrivals = u_i * w_i and the plan's
    arrival-share coverage is 1 - sum(u_i w_i) / sum(arr)."""
    return np.divide(arr_day, a, out=np.zeros_like(a), where=a > 1e-12)


def arrival_coverage(u, a, arr_day):
    tot = float(arr_day.sum())
    un = float((u * arrival_weights(a, arr_day)).sum())
    return 1.0 - un / max(tot, 1e-12)


def run_demand(force=False):
    path = os.path.join(RUN, "cell_demand.npz")
    if os.path.exists(path) and not force:
        print("reuse", path)
        return dict(np.load(path, allow_pickle=True))
    d = load_inputs()
    arr15 = to15(d["arr"])
    dep15 = to15(d["dep"])
    out = dict()
    for lvl, I0 in (("meas343", d["I0_meas"]), ("fleet495", d["I0_fleet"])):
        print(f"computing cell demand for {lvl}")
        t0 = time.time()
        a = cell_demand(arr15, dep15, I0)
        out[f"a_{lvl}"] = a
        print(f"  {lvl}: sum={a.sum():.1f} max={a.max():.1f} "
              f"median={np.median(a):.1f} >150={int((a > CAP).sum())} "
              f"({time.time()-t0:.1f}s)")
    np.savez(path, **{k: v.astype(np.float64) for k, v in out.items()},
             I0_meas=d["I0_meas"], I0_fleet=d["I0_fleet"],
             arr=d["arr"], dep=d["dep"], xy=d["xy"])
    print("wrote", path)
    return dict(np.load(path, allow_pickle=True))


def demand_stats(d):
    t9 = load_json_any(os.path.join(DATA, "runs", "T009", "q4_results.json"))
    q1 = t9["surfaces"]["q1"]
    f705 = pd.read_csv(os.path.join(DATA, "runs", "T009", "q4_fences.csv"))
    covered = np.zeros(len(d["a_meas343"]), bool)
    for s in f705.assigned_cells.fillna(""):
        for tok in str(s).split():
            covered[int(tok)] = True
    out = {}
    for lvl in ("meas343", "fleet495"):
        a = d[f"a_{lvl}"]
        out[lvl] = dict(
            sum_all=float(a.sum()),
            max=float(a.max()),
            median=float(np.median(a)),
            cells_over_cap=int((a > CAP).sum()),
            cells_over_own_plus_two=int((a > 3 * CAP).sum()),
            sum_over_705_covered=float(a[covered].sum()),
            n_705_covered_cells=int(covered.sum()),
            t009_fence_capacity_total=float(q1[lvl]["capacity_total"]),
            t009_standing_total=float(q1[lvl]["standing_total"]),
            ratio_sum_over_t009=float(a[covered].sum()
                                      / max(q1[lvl]["capacity_total"], 1e-9)),
            note=("cell-level 95th percentiles summed over the 705 covered "
                  "cells exceed the 705 fence-level 95th percentiles because "
                  "the percentile of a sum is below the sum of percentiles; "
                  "the cell-level demand is therefore conservative."),
        )
    out["cells"] = int(len(d["a_meas343"]))
    out["cap_per_site"] = CAP
    out["all_sites_capacity"] = float(CAP * len(d["a_meas343"]))
    dump(out, "demand_stats.json")
    return out


# --------------------------------------------------------------------------
# capacitated siting / allocation MILP
# --------------------------------------------------------------------------
def build_arcs(D, R):
    # dist_matrix.npy stores 0.35/sqrt(2)=0.2475 km on the diagonal as the
    # mean intra-cell offset for the gravity model.  For siting, the site at
    # a cell centre is at distance 0 from that cell, so the diagonal is
    # reset to 0 before the radius test.
    Ds = D.copy()
    np.fill_diagonal(Ds, 0.0)
    ii, jj = np.where(Ds <= R + 1e-9)
    return ii.astype(np.int64), jj.astype(np.int64)


def solve_cap(a, ii, jj, n=None, integer=True, cap=CAP, time_limit=300,
              mip_gap=0.005, fix_sites=None):
    C = len(a)
    K = len(ii)
    out_arcs = [[] for _ in range(C)]
    in_arcs = [[] for _ in range(C)]
    for k in range(K):
        out_arcs[int(ii[k])].append(k)
        in_arcs[int(jj[k])].append(k)

    m = gp.Model("q4cap")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = time_limit
    if integer:
        m.Params.MIPGap = mip_gap
    vt = GRB.BINARY if integer else GRB.CONTINUOUS
    s = m.addVars(C, lb=0, ub=1, vtype=vt, name="s")
    f = m.addVars(K, lb=0, name="f")
    u = m.addVars(C, lb=0, name="u")
    m.setObjective(gp.quicksum(f[k] for k in range(K)), GRB.MAXIMIZE)
    for i in range(C):
        m.addConstr(gp.quicksum(f[k] for k in out_arcs[i]) + u[i]
                    == float(a[i]), name=f"as{i}")
    for j in range(C):
        if in_arcs[j]:
            m.addConstr(gp.quicksum(f[k] for k in in_arcs[j]) <= cap * s[j],
                        name=f"cap{j}")
        else:
            m.addConstr(s[j] == 0)
    if fix_sites is not None:
        fs = set(int(x) for x in fix_sites)
        for j in range(C):
            m.addConstr(s[j] == (1 if j in fs else 0))
    elif n is not None:
        m.addConstr(gp.quicksum(s[j] for j in range(C)) == n, name="nsites")
    m.optimize()
    try:
        if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            print("  solve failed, status", m.Status)
            return None
        fv = np.array([f[k].X for k in range(K)])
        sv = np.array([s[j].X for j in range(C)])
        uv = np.array([u[i].X for i in range(C)])
        assigned = float(fv.sum())
        return dict(coverage=assigned / float(a.sum()), assigned=assigned,
                    unassigned=float(a.sum()) - assigned,
                    n_open=int(round(float(sv.sum()))),
                    obj=float(m.ObjVal),
                    bound=float(getattr(m, "ObjBound", np.nan)),
                    gap=float(getattr(m, "MIPGap", 0.0)) if integer else 0.0,
                    runtime=float(m.Runtime), status=int(m.Status),
                    s=sv, f=fv, u=uv, ii=ii, jj=jj)
    finally:
        m.dispose()


def load_level(level, R=R_MAIN):
    z = np.load(os.path.join(RUN, "cell_demand.npz"))
    a = z[f"a_{level}"]
    I0 = z["I0_meas" if level == "meas343" else "I0_fleet"]
    D = np.load(os.path.join(DATA, "dist_matrix.npy"))
    ii, jj = build_arcs(D, R)
    return a, I0, ii, jj, D, z["xy"]


C_CACHE = 5263


def save_plan(level, tag, a, I0, res):
    np.savez(os.path.join(RUN, f"plan_{level}_{tag}.npz"),
             s=res["s"], f=res["f"], u=res["u"], ii=res["ii"],
             jj=res["jj"], a=a, I0=I0)
    print("wrote", os.path.join(RUN, f"plan_{level}_{tag}.npz"))


def load_plan(level, tag):
    return dict(np.load(os.path.join(RUN, f"plan_{level}_{tag}.npz")))


def run_solve(force=False):
    path = os.path.join(RUN, "frontier.json")
    if os.path.exists(path) and not force:
        print("reuse", path)
        return load_json("frontier.json")
    out = {}
    z = np.load(os.path.join(RUN, "cell_demand.npz"))
    arr_day = z["arr"].sum(0)
    for level in ("meas343", "fleet495"):
        a, I0, ii, jj, D, xy = load_level(level)
        print(f"\n=== {level}: {len(a)} demand cells, {len(ii)} arcs ===")
        t0 = time.time()
        r_all = solve_cap(a, ii, jj, n=None, integer=False, time_limit=300)
        print(f"  all-open LP: coverage {100*r_all['coverage']:.3f}% "
              f"assigned {r_all['assigned']:.1f} ({r_all['runtime']:.1f}s)")
        points = []
        for n in N_GRID:
            r = solve_cap(a, ii, jj, n=n, integer=True, time_limit=300,
                          mip_gap=0.005)
            rec = dict(n=n, coverage=r["coverage"],
                       arrival_coverage=arrival_coverage(r["u"], a, arr_day),
                       assigned=r["assigned"],
                       unassigned=r["unassigned"], slots=float(CAP * n),
                       obj=r["obj"], bound=r["bound"], gap=r["gap"],
                       runtime=r["runtime"], status=r["status"])
            points.append(rec)
            print(f"  n={n:5d} cov={100*r['coverage']:6.3f}% "
                  f"unassigned={r['unassigned']:9.1f} "
                  f"gap={100*r['gap']:.3f}% t={r['runtime']:.1f}s")
        # bisection to the T009 target if it is bracketed
        covs = np.array([p["coverage"] for p in points])
        ns = np.array([p["n"] for p in points])
        crossing = None
        if r_all["coverage"] >= TARGET:
            lo = None
            hi = None
            for k in range(len(ns)):
                if covs[k] >= TARGET:
                    hi = int(ns[k])
                    lo = int(ns[k - 1]) if k > 0 else 0
                    break
            if hi is not None and lo is not None:
                while hi - lo > 25:
                    mid = (lo + hi) // 2
                    rm = solve_cap(a, ii, jj, n=mid, integer=True,
                                   time_limit=300, mip_gap=0.005)
                    rec = dict(n=mid, coverage=rm["coverage"],
                               arrival_coverage=arrival_coverage(
                                   rm["u"], a, arr_day),
                               assigned=rm["assigned"],
                               unassigned=rm["unassigned"],
                               slots=float(CAP * mid), obj=rm["obj"],
                               bound=rm["bound"], gap=rm["gap"],
                               runtime=rm["runtime"], status=rm["status"])
                    points.append(rec)
                    print(f"  bisect n={mid:5d} cov={100*rm['coverage']:6.3f}%")
                    if rm["coverage"] >= TARGET:
                        hi = mid
                    else:
                        lo = mid
                crossing = dict(n=hi, n_below=lo,
                                coverage_at_hi=next(p["coverage"] for p in points
                                                    if p["n"] == hi))
                print(f"  target {100*TARGET:.2f}% reached at n={hi}")
        points.sort(key=lambda p: p["n"])
        # selection: the same chord rule used for eq. (29): normalise
        # (n, coverage) to the unit square and take the point farthest above
        # the chord joining the first and last frontier points.
        pn = np.array([p["n"] for p in points], dtype=float)
        pc = np.array([p["coverage"] for p in points], dtype=float)
        xn = (pn - pn[0]) / (pn[-1] - pn[0])
        yn = (pc - pc[0]) / (pc[-1] - pc[0])
        dev = yn - xn
        kbest = int(np.argmax(dev))
        sel = int(pn[kbest])
        knee = dict(n=sel, coverage=float(pc[kbest]),
                    deviation_normalised=float(dev[kbest]),
                    rule="point farthest above the chord on the normalised "
                         "(n, coverage) frontier; Kneedle analogue of the "
                         "eq. (29) rule")
        # the target plan, if the target is reachable
        if crossing is not None:
            sel = int(crossing["n"])
        rs = solve_cap(a, ii, jj, n=sel, integer=True, time_limit=300,
                       mip_gap=0.005)
        save_plan(level, "main", a, I0, rs)
        rfull = solve_cap(a, ii, jj, n=int(len(a)), integer=True,
                          time_limit=300, mip_gap=0.005)
        save_plan(level, "all", a, I0, rfull)
        out[level] = dict(
            all_open_lp=dict(coverage=r_all["coverage"],
                             arrival_coverage=arrival_coverage(
                                 r_all["u"], a, arr_day),
                             assigned=r_all["assigned"],
                             unassigned=r_all["unassigned"],
                             bound=r_all["bound"], runtime=r_all["runtime"]),
            all_open_milp=dict(coverage=rfull["coverage"],
                               arrival_coverage=arrival_coverage(
                                   rfull["u"], a, arr_day),
                               assigned=rfull["assigned"],
                               unassigned=rfull["unassigned"],
                               n_open=rfull["n_open"], gap=rfull["gap"]),
            target=TARGET,
            target_reached=bool(r_all["coverage"] >= TARGET),
            crossing=crossing,
            knee=knee,
            selection_rule=("target plan if reachable, else Kneedle knee"),
            selected_n=int(sel),
            selected_is_knee=bool(sel == knee["n"]),
            selected=dict(coverage=rs["coverage"],
                          arrival_coverage=arrival_coverage(
                              rs["u"], a, arr_day),
                          assigned=rs["assigned"],
                          unassigned=rs["unassigned"], slots=float(CAP * sel),
                          gap=rs["gap"], bound=rs["bound"],
                          runtime=rs["runtime"], status=rs["status"]),
            points=points,
            n_cells=int(len(a)),
            demand_sum=float(a.sum()),
            arrival_sum=float(arr_day.sum()),
            coverage_basis=("coverage = assigned peak-occupancy demand / "
                            "total; arrival_coverage reweights each cell's "
                            "unassigned share u_i/a_i by its daily arrivals "
                            "arr_i, 1 - sum(u_i arr_i/a_i)/sum(arr_i)"),
            all_sites_capacity=float(CAP * len(a)),
        )
    dump(out, "frontier.json")
    print(f"solve stage total {time.time()-t0:.1f}s")
    return out


# --------------------------------------------------------------------------
# radius sweep
# --------------------------------------------------------------------------
def run_radius(force=False):
    path = os.path.join(RUN, "radius_sweep.json")
    if os.path.exists(path) and not force:
        print("reuse", path)
        return load_json("radius_sweep.json")
    fr = load_json("frontier.json")
    z = np.load(os.path.join(RUN, "cell_demand.npz"))
    arr_day = z["arr"].sum(0)
    out = {}
    for level in ("meas343", "fleet495"):
        n = int(fr[level]["selected_n"])
        out[level] = dict(selected_n=n, radii={})
        for r_lab, R in RADII.items():
            a, I0, ii, jj, D, xy = load_level(level, R)
            r = solve_cap(a, ii, jj, n=n, integer=True, time_limit=300,
                          mip_gap=0.005)
            save_plan(level, f"R{r_lab}", a, I0, r)
            acov = arrival_coverage(r["u"], a, arr_day)
            eff = r["coverage"] * COMPLY[r_lab]
            out[level]["radii"][r_lab] = dict(
                R_m=float(R * 1000), n_sites=n, slots=float(CAP * n),
                coverage=r["coverage"], assigned=r["assigned"],
                unassigned=r["unassigned"], gap=r["gap"],
                arrival_coverage=acov,
                comply_zero_incentive=COMPLY[r_lab],
                effective_coverage=eff,
                effective_coverage_basis="(a) coverage x eq.(19) compliance",
                effective_arrival_coverage=acov * COMPLY[r_lab],
                mandatory_parking_coverage=r["coverage"],
                mandatory_parking_arrival_coverage=acov,
                mandatory_parking_basis="(b) regulated parking inside fences; "
                                        "coverage not multiplied by compliance",
                n_arcs=int(len(ii)),
            )
            print(f"{level} R={r_lab}: cov={100*r['coverage']:.3f}% "
                  f"eff={100*eff:.3f}%")
    dump(out, "radius_sweep.json")
    return out


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------
def build_test_days(force=False):
    path = os.path.join(RUN, "test_days.npz")
    if os.path.exists(path) and not force:
        return dict(np.load(path, allow_pickle=True))
    d = load_inputs()
    z = np.load(os.path.join(RUN, "cell_demand.npz"))
    C = len(z["a_meas343"])
    out = {}
    for k, day in enumerate(TE_DAYS):
        s = d["df"][d["df"].date == day]
        arr = np.zeros((24, C))
        dep = np.zeros((24, C))
        np.add.at(arr, (s.hour.values, s.d.values), 1.0)
        np.add.at(dep, (s.hour.values, s.o.values), 1.0)
        out[f"arr_{k}"] = arr * SCALE
        out[f"dep_{k}"] = dep * SCALE
    np.savez(path, **out)
    print("wrote", path)
    return dict(np.load(path, allow_pickle=True))


def replay_sites(plan, arr15, dep15, cap=CAP):
    s = plan["s"]
    f = plan["f"]
    a = plan["a"]
    ii = plan["ii"]
    jj = plan["jj"]
    I0 = plan["I0"]
    C = len(a)
    T15 = arr15.shape[0]
    frac = np.divide(f, a[ii], out=np.zeros_like(f), where=a[ii] > 0)
    open_sites = np.where(s > 0.5)[0]
    occ = np.zeros(C)
    np.add.at(occ, jj, I0[ii] * frac)
    over_init = float(np.maximum(0.0, occ - cap).sum())
    occ = np.minimum(occ, cap)
    unassigned_frac = 1.0 - np.bincount(ii, weights=frac, minlength=C)
    full_ever = np.zeros(C, bool)
    full_steps = 0
    overflow_arr = 0.0
    dep_short = 0.0
    arr_total = 0.0
    spare = np.zeros((T15, len(open_sites)))
    site_pos = {int(j): k for k, j in enumerate(open_sites)}
    for t in range(T15):
        req = np.zeros(C)
        np.add.at(req, jj, dep15[t, ii] * frac)
        out = np.minimum(req, occ)
        dep_short += float((req - out).sum())
        occ -= out
        ain = np.zeros(C)
        np.add.at(ain, jj, arr15[t, ii] * frac)
        newocc = occ + ain
        over = np.maximum(0.0, newocc - cap)
        overflow_arr += float(over.sum())
        occ = np.minimum(newocc, cap)
        full = occ >= cap - 1e-9
        full_ever |= full
        full_steps += int(full.sum())
        arr_total += float(arr15[t].sum())
        for j in open_sites:
            spare[t, site_pos[int(j)]] = cap - occ[j]
    unassigned_arr = float((arr15 * unassigned_frac[None, :]).sum())
    covered_arr = arr_total - unassigned_arr - overflow_arr
    return dict(
        total_arrivals=arr_total, unassigned_arrivals=unassigned_arr,
        overflow_arrivals=overflow_arr, covered_arrivals=covered_arr,
        realized_coverage=covered_arr / max(arr_total, 1e-9),
        initial_overflow=over_init,
        full_sites_ever=int(full_ever.sum()),
        full_site_steps=full_steps,
        max_full_sites_per_step=None,
        departure_shortfall=dep_short,
        spare=spare, open_sites=open_sites,
    )


def run_replay(force=False):
    path = os.path.join(RUN, "replay.json")
    if os.path.exists(path) and not force:
        print("reuse", path)
        return load_json("replay.json")
    z = np.load(os.path.join(RUN, "cell_demand.npz"))
    td = build_test_days()
    out = {}
    for level in ("meas343", "fleet495"):
        plan = load_plan(level, "main")
        plan_all = load_plan(level, "all")
        arr15 = to15(z["arr"])
        dep15 = to15(z["dep"])
        des = replay_sites(plan, arr15, dep15)
        des60 = replay_sites(plan, z["arr"], z["dep"])
        rec = dict(design=dict(realized_coverage=des["realized_coverage"],
                               overflow_arrivals=des["overflow_arrivals"],
                               unassigned_arrivals=des["unassigned_arrivals"],
                               initial_overflow=des["initial_overflow"],
                               full_sites_ever=des["full_sites_ever"],
                               full_site_steps=des["full_site_steps"],
                               departure_shortfall=des["departure_shortfall"],
                               total_arrivals=des["total_arrivals"]),
                   stride_minutes=15)
        rec["design_1h"] = dict(
            stride_minutes=60,
            realized_coverage=des60["realized_coverage"],
            overflow_arrivals=des60["overflow_arrivals"],
            unassigned_arrivals=des60["unassigned_arrivals"],
            initial_overflow=des60["initial_overflow"],
            full_sites_ever=des60["full_sites_ever"],
            full_site_steps=des60["full_site_steps"],
            departure_shortfall=des60["departure_shortfall"],
            total_arrivals=des60["total_arrivals"],
            delta_realized_coverage=des60["realized_coverage"]
            - des["realized_coverage"],
            delta_departure_shortfall=des60["departure_shortfall"]
            - des["departure_shortfall"],
            delta_overflow=des60["overflow_arrivals"]
            - des["overflow_arrivals"])
        da = replay_sites(plan_all, arr15, dep15)
        da60 = replay_sites(plan_all, z["arr"], z["dep"])
        rec["all_sites"] = dict(
            design=dict(realized_coverage=da["realized_coverage"],
                        overflow_arrivals=da["overflow_arrivals"],
                        unassigned_arrivals=da["unassigned_arrivals"],
                        initial_overflow=da["initial_overflow"],
                        full_sites_ever=da["full_sites_ever"],
                        full_site_steps=da["full_site_steps"],
                        departure_shortfall=da["departure_shortfall"]),
            design_1h=dict(
                stride_minutes=60,
                realized_coverage=da60["realized_coverage"],
                overflow_arrivals=da60["overflow_arrivals"],
                unassigned_arrivals=da60["unassigned_arrivals"],
                full_sites_ever=da60["full_sites_ever"],
                full_site_steps=da60["full_site_steps"],
                departure_shortfall=da60["departure_shortfall"],
                delta_realized_coverage=da60["realized_coverage"]
                - da["realized_coverage"],
                delta_departure_shortfall=da60["departure_shortfall"]
                - da["departure_shortfall"],
                delta_overflow=da60["overflow_arrivals"]
                - da["overflow_arrivals"]))
        np.savez(os.path.join(RUN, f"spare_{level}.npz"),
                 spare=des["spare"], sites=des["open_sites"],
                 time_index=np.arange(arr15.shape[0]))
        for k, day in enumerate(TE_DAYS):
            arr15d = to15(td[f"arr_{k}"])
            dep15d = to15(td[f"dep_{k}"])
            r = replay_sites(plan, arr15d, dep15d)
            r60 = replay_sites(plan, td[f"arr_{k}"], td[f"dep_{k}"])
            rec[day] = dict(realized_coverage=r["realized_coverage"],
                            overflow_arrivals=r["overflow_arrivals"],
                            unassigned_arrivals=r["unassigned_arrivals"],
                            initial_overflow=r["initial_overflow"],
                            full_sites_ever=r["full_sites_ever"],
                            full_site_steps=r["full_site_steps"],
                            departure_shortfall=r["departure_shortfall"],
                            total_arrivals=r["total_arrivals"],
                            stride_minutes=15,
                            hourly=dict(
                                stride_minutes=60,
                                realized_coverage=r60["realized_coverage"],
                                overflow_arrivals=r60["overflow_arrivals"],
                                departure_shortfall=r60["departure_shortfall"],
                                delta_realized_coverage=r60["realized_coverage"]
                                - r["realized_coverage"],
                                delta_departure_shortfall=r60[
                                    "departure_shortfall"]
                                - r["departure_shortfall"]))
            ra = replay_sites(plan_all, arr15d, dep15d)
            rec["all_sites"][day] = dict(
                realized_coverage=ra["realized_coverage"],
                overflow_arrivals=ra["overflow_arrivals"],
                unassigned_arrivals=ra["unassigned_arrivals"],
                full_sites_ever=ra["full_sites_ever"],
                full_site_steps=ra["full_site_steps"],
                departure_shortfall=ra["departure_shortfall"])
            print(f"{level} {day}: realized cov={100*r['realized_coverage']:.2f}% "
                  f"overflow={r['overflow_arrivals']:.1f} "
                  f"(all sites {100*ra['realized_coverage']:.2f}%) "
                  f"shortfall={r['departure_shortfall']:.1f}")
        out[level] = rec
    dump(out, "replay.json")
    return out


# --------------------------------------------------------------------------
# uncovered walk
# --------------------------------------------------------------------------
def weighted_percentile(x, w, q):
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return float("nan")
    return float(np.interp(q * cw[-1], cw, x))


def run_walk(force=False):
    path = os.path.join(RUN, "uncovered_walk.json")
    if os.path.exists(path) and not force:
        print("reuse", path)
        return load_json("uncovered_walk.json")
    z = np.load(os.path.join(RUN, "cell_demand.npz"))
    arr_day = z["arr"].sum(0)
    out = {}
    for level in ("meas343", "fleet495"):
        plan = load_plan(level, "main")
        s = plan["s"]
        a = plan["a"]
        u = plan["u"]
        f = plan["f"]
        ii = plan["ii"]
        jj = plan["jj"]
        D = np.load(os.path.join(DATA, "dist_matrix.npy"))
        Dw = D.copy()
        np.fill_diagonal(Dw, 0.0)
        open_sites = np.where(s > 0.5)[0]
        nearest = np.empty(len(a))
        for i0 in range(0, len(a), 400):
            i1 = min(i0 + 400, len(a))
            nearest[i0:i1] = Dw[i0:i1, open_sites].min(1)
        unassigned_frac = np.divide(u, a, out=np.zeros_like(u), where=a > 0)
        w_arr = arr_day * unassigned_frac
        mask = w_arr > 1e-9
        x = nearest[mask]
        w = w_arr[mask]
        # cells with no assigned arc at all: uncovered by any fence in the plan
        per_cell = np.bincount(ii, weights=f, minlength=len(a))
        nos = (per_cell <= 1e-9) & (a > 1e-9)
        x0 = nearest[nos]
        beyond = x > R_MAIN + 1e-9
        out[level] = dict(
            n_open=int(len(open_sites)),
            unassigned_cells=int(mask.sum()),
            unassigned_arrivals_day=float(w.sum()),
            nearest_m=dict(
                p50=weighted_percentile(x, w, 0.50),
                p90=weighted_percentile(x, w, 0.90),
                max=float(x.max()) if len(x) else None,
                weighted_by="uncovered arrivals = daily arrivals * u_i/a_i"),
            share_within_450m=float(w[x <= 0.45].sum() / max(w.sum(), 1e-9)),
            share_beyond_450m=float(w[beyond].sum() / max(w.sum(), 1e-9)),
            beyond_450m=dict(
                cells=int(beyond.sum()),
                unassigned_arrivals_day=float(w[beyond].sum()),
                p50=weighted_percentile(x[beyond], w[beyond], 0.50),
                p90=weighted_percentile(x[beyond], w[beyond], 0.90),
                max=float(x[beyond].max()) if beyond.any() else None),
            cells_no_fence=int(nos.sum()),
            arrivals_at_cells_no_fence=float(arr_day[nos].sum()),
            nearest_m_cells_no_fence=dict(
                p50=float(np.percentile(x0, 50)) if len(x0) else None,
                p90=float(np.percentile(x0, 90)) if len(x0) else None,
                max=float(x0.max()) if len(x0) else None,
                weighted_by="unweighted over cells with a_i>0 and no arc"),
            note=("Euclidean straight-line distance from the cell centre to "
                  "the nearest open site; capacity-unassigned demand is "
                  "included even when a fence is within R but full.  With "
                  "sites at cell centres, most uncovered arrivals are at "
                  "their own full site (distance 0); the beyond_450m block "
                  "is the strictly distance-uncovered set."),
        )
    dump(out, "uncovered_walk.json")
    return out


# --------------------------------------------------------------------------
# acceptance checks
# --------------------------------------------------------------------------
def run_check(force=False):
    path = os.path.join(RUN, "check.json")
    if os.path.exists(path) and not force:
        print("reuse", path)
        return load_json("check.json")
    z = np.load(os.path.join(RUN, "cell_demand.npz"))
    cells = lib.get_cells()
    xy = cells[["x", "y"]].to_numpy()
    D = np.load(os.path.join(DATA, "dist_matrix.npy"))
    out = {}
    for level in ("meas343", "fleet495"):
        plan = load_plan(level, "main")
        a = plan["a"]
        f = plan["f"]
        u = plan["u"]
        s = plan["s"]
        ii = plan["ii"]
        jj = plan["jj"]
        C = len(a)
        open_sites = np.where(s > 0.5)[0]
        per_cell = np.bincount(ii, weights=f, minlength=C)
        cons = np.abs(per_cell + u - a)
        site_assigned = np.bincount(jj, weights=f, minlength=C)
        big = f > 1e-9
        self_arc = ii[big] == jj[big]
        d_big = D[ii[big], jj[big]].astype(np.float64)
        dmax_nonself = float(d_big[~self_arc].max()) if (~self_arc).any() else 0.0
        viol = int((d_big[~self_arc] > R_MAIN + 1e-6).sum())
        over = site_assigned[open_sites] - CAP
        csv = pd.read_csv(os.path.join(RUN, f"fences_cap150_{level}.csv"))
        out[level] = dict(
            n_open=int(len(open_sites)),
            csv_rows=int(len(csv)),
            csv_rows_ok=bool(len(csv) == len(open_sites)),
            csv_capacity_max=float(csv.capacity.max()),
            csv_capacity_ok=bool(float(csv.capacity.max()) <= CAP + 1e-9),
            csv_assigned_max=float(csv.assigned_demand.max()),
            csv_assigned_le_capacity=bool(
                (csv.assigned_demand <= csv.capacity + 1e-6).all()),
            csv_assigned_matches_plan=bool(
                float(np.abs(csv.assigned_demand.to_numpy()
                             - site_assigned[open_sites]).max()) <= 1e-6),
            assignment_consistency_max_abs=float(cons.max()),
            assignment_consistency_ok=bool(float(cons.max()) <= 1e-6),
            n_arcs_used=int(big.sum()),
            max_distance_km_nonself=dmax_nonself,
            distance_violations=viol,
            distance_ok=bool(viol == 0),
            site_over_capacity_max=float(max(over.max(), 0.0)),
            site_capacity_ok=bool(float(over.max()) <= 1e-6),
            s_integrality_max_dev=float(np.abs(s - np.round(s)).max()),
            s_integrality_ok=bool(float(np.abs(s - np.round(s)).max()) <= 1e-6),
            covered_demand=float(f.sum()),
            unassigned_demand=float(u.sum()),
        )
    out["all_pass"] = all(
        out[l][k] for l in ("meas343", "fleet495")
        for k in ("csv_rows_ok", "csv_capacity_ok", "csv_assigned_le_capacity",
                  "csv_assigned_matches_plan", "assignment_consistency_ok",
                  "distance_ok", "site_capacity_ok", "s_integrality_ok"))
    dump(out, "check.json")
    return out


# --------------------------------------------------------------------------
# denominator conservativeness
# --------------------------------------------------------------------------
def run_denominator(force=False):
    """How conservative is sum_i a_i as a system-level demand?

    a_i is the 95th percentile (over 300 Poisson replays) of each cell's
    own peak 15-min occupancy.  The sum of those percentiles equals the
    95th percentile of the system total only if every cell peaks in the
    same replay at the same step.  This stage re-runs the identical 300
    replays (same per-cell seeds and draw order as cell_demand), forms the
    system total occupancy at every 15-min step, takes each replay's peak
    of that total, and reports
        ratio = P95_r( max_t sum_i occ_ir(t) ) / sum_i a_i.
    """
    path = os.path.join(RUN, "denominator.json")
    if os.path.exists(path) and not force:
        print("reuse", path)
        return load_json("denominator.json")
    z = np.load(os.path.join(RUN, "cell_demand.npz"))
    arr15 = to15(z["arr"])
    dep15 = to15(z["dep"])
    C = len(z["a_meas343"])
    T = arr15.shape[0]
    sysm = np.zeros((T, REPS))
    sysf = np.zeros((T, REPS))
    for i in range(C):
        rng = np.random.default_rng(SEED + i)
        A = rng.poisson(arr15[:, i], size=(REPS, T))
        Dp = rng.poisson(dep15[:, i], size=(REPS, T))
        for I0, sys in ((z["I0_meas"], sysm), (z["I0_fleet"], sysf)):
            occ = np.full(REPS, float(I0[i]))
            for t in range(T):
                occ = np.maximum(0.0, occ + A[:, t] - Dp[:, t])
                sys[t] += occ
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{C} cells")
    out = dict(reps=REPS, seed=SEED, step_minutes=15, cells=int(C),
               definition=("ratio = P95 over replays of max_t sum_i occ_ir(t) "
                           "(system total occupancy) / sum_i a_i, with a_i = "
                           "P95 over the same replays of max_t occ_ir(t); "
                           "same per-cell seeds and draw order as the demand "
                           "stage"),
               note=("ratio < 1 measures how much sum_i a_i overstates a "
                     "system total: per-cell peaks are not simultaneous."))
    for lvl in ("meas343", "fleet495"):
        sysm_ = sysm if lvl == "meas343" else sysf
        peak_sys = sysm_.max(0)
        p95 = float(np.percentile(peak_sys, 95))
        a = z[f"a_{lvl}"]
        out[lvl] = dict(
            sum_cell_p95=float(a.sum()),
            system_p95_total_occupancy=p95,
            ratio=p95 / float(a.sum()),
            system_max_replay=float(peak_sys.max()),
            system_mean_replay=float(peak_sys.mean()))
        print(f"{lvl}: sum cell P95 {a.sum():.1f}, system P95 {p95:.1f}, "
              f"ratio {out[lvl]['ratio']:.4f}")
    dump(out, "denominator.json")
    return out


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------
def run_summary(force=False):
    path = os.path.join(RUN, "summary.json")
    if os.path.exists(path) and not force:
        print("reuse", path)
        return load_json("summary.json")
    fr = load_json("frontier.json")
    rs = load_json("radius_sweep.json")
    t9 = load_json_any(os.path.join(DATA, "runs", "T009", "q4_results.json"))
    q1 = t9["surfaces"]["q1"]
    f705 = pd.read_csv(os.path.join(DATA, "runs", "T009", "q4_fences.csv"))
    sites705 = []
    for s in f705.assigned_cells.fillna(""):
        toks = str(s).split()
        if toks:
            sites705.append(int(toks[0]))
    # the T009 file has one fence per row; recover the site cell from x,y
    cells = lib.get_cells()
    xy = cells[["x", "y"]].to_numpy()
    site_cells = []
    for _, row in f705.iterrows():
        d2 = ((xy - np.array([row.x, row.y])) ** 2).sum(1)
        site_cells.append(int(np.argmin(d2)))
    site_cells = np.array(site_cells)
    z = np.load(os.path.join(RUN, "cell_demand.npz"))
    arr_day = z["arr"].sum(0)
    D = np.load(os.path.join(DATA, "dist_matrix.npy"))
    diff = load_json_any(os.path.join(RUN, "arr_train_diff.json"))
    den = (load_json("denominator.json")
           if os.path.exists(os.path.join(RUN, "denominator.json")) else None)
    rep = (load_json("replay.json")
           if os.path.exists(os.path.join(RUN, "replay.json")) else None)
    out = {"settings": dict(cap=CAP, radii_m={k: float(v * 1000) for k, v in
                                              RADII.items()},
                            demand="cell-level MC 95th percentile peak "
                                   "occupancy, 300 reps, I0 included at t=0",
                            demand_source="Q1 arrival surface rebuilt on "
                                          "training days only (M11): " +
                                          ", ".join(TR_DAYS),
                            demand_diff_vs_typ7=dict(
                                total_train5=diff["total_train5"],
                                total_typ7=diff["total_typ7"],
                                ratio=diff["ratio_train5_over_typ7"],
                                l1_relative=diff["l1_relative"],
                                note=diff["note"]),
                            target=TARGET, target_source="T009 705-fence "
                                   "arrival coverage",
                            comply=COMPLY,
                            note=("two coverage rulers are reported side by "
                                  "side.  occupancy_coverage = assigned "
                                  "peak-occupancy demand / total cell demand.  "
                                  "arrival_coverage = 1 - sum_i u_i*arr_i/a_i "
                                  "/ sum_i arr_i, i.e. the same assignment "
                                  "reweighted to daily arrivals; this is the "
                                  "ruler on which T009's 76.86% was "
                                  "computed and is the one to quote in "
                                  "comparisons with T009.")),
           "denominator_conservativeness": den,
           "levels": {}}
    for level in ("meas343", "fleet495"):
        a = z[f"a_{level}"]
        plan = load_plan(level, "main")
        assigned_by_site = np.bincount(plan["jj"], weights=plan["f"],
                                       minlength=len(a))
        open_sites = np.where(plan["s"] > 0.5)[0]
        # same 705 T009 sites, cap 150
        r705 = solve_cap(a, plan["ii"], plan["jj"], n=None, integer=False,
                         fix_sites=site_cells, time_limit=120)
        out["levels"][level] = dict(
            selected_n=int(fr[level]["selected_n"]),
            selection_rule=fr[level]["selection_rule"],
            knee=fr[level]["knee"],
            slots=float(CAP * fr[level]["selected_n"]),
            occupancy_coverage=fr[level]["selected"]["coverage"],
            arrival_coverage=fr[level]["selected"]["arrival_coverage"],
            coverage=fr[level]["selected"]["coverage"],
            assigned=fr[level]["selected"]["assigned"],
            unassigned=fr[level]["selected"]["unassigned"],
            target_reached=fr[level]["target_reached"],
            max_occupancy_coverage_all_sites=fr[level]["all_open_milp"]["coverage"],
            max_arrival_coverage_all_sites=fr[level]["all_open_milp"][
                "arrival_coverage"],
            max_coverage_all_sites=fr[level]["all_open_milp"]["coverage"],
            max_coverage_all_sites_n=int(fr[level]["n_cells"]),
            all_open_lp_coverage=fr[level]["all_open_lp"]["coverage"],
            all_open_lp_arrival_coverage=fr[level]["all_open_lp"][
                "arrival_coverage"],
            demand_sum=float(a.sum()),
            arrival_sum=float(arr_day.sum()),
            cells_over_150=int((a > CAP).sum()),
            t009_705_sites_cap150=dict(
                n=705, slots=float(CAP * 705), coverage=r705["coverage"],
                arrival_coverage=arrival_coverage(r705["u"], a, arr_day),
                unassigned=r705["unassigned"]),
            csv=f"fences_cap150_{level}.csv",
        )
        if rep is not None and level in rep:
            rd = rep[level]["design"]
            r1h = rep[level].get("design_1h", {})
            out["levels"][level]["replay_design"] = dict(
                step_minutes=rd.get("stride_minutes", 15),
                realized_coverage=rd["realized_coverage"],
                unassigned_arrivals=rd["unassigned_arrivals"],
                overflow_arrivals=rd["overflow_arrivals"],
                full_sites_ever=rd["full_sites_ever"],
                full_site_steps=rd["full_site_steps"],
                departure_shortfall=rd["departure_shortfall"],
                hour_step=dict(realized_coverage=r1h.get("realized_coverage"),
                               departure_shortfall=r1h.get(
                                   "departure_shortfall"),
                               overflow_arrivals=r1h.get("overflow_arrivals"),
                               delta_realized_coverage=r1h.get(
                                   "delta_realized_coverage")),
                note=("full sites and departure shortfall are two different "
                      "service failures: a full fence turns away arriving "
                      "bikes (overflow), an empty fence cannot lend bikes "
                      "(departure_shortfall).  The design replay has zero "
                      "overflow but a large shortfall."))
        df = pd.DataFrame(dict(
            site_cell=open_sites,
            x=xy[open_sites, 0],
            y=xy[open_sites, 1],
            capacity=float(CAP),
            assigned_demand=assigned_by_site[open_sites]))
        df.to_csv(os.path.join(RUN, f"fences_cap150_{level}.csv"), index=False)
        print("wrote", os.path.join(RUN, f"fences_cap150_{level}.csv"),
              len(df), "sites")
    out["old_705_uncapped_reference"] = dict(
        n=705, coverage_arrivals=float(q1["coverage_selected"]),
        slots_meas343=float(q1["meas343"]["capacity_total"]),
        slots_fleet495=float(q1["fleet495"]["capacity_total"]),
        standing_meas343=float(q1["meas343"]["standing_total"]),
        standing_fleet495=float(q1["fleet495"]["standing_total"]),
        note="T009 Kneedle knee, arrival-share coverage, per-fence 95th "
             "percentile with no hard cap; kept as the reference row only.")
    out["reconciliation_7686"] = dict(
        old_T009_coverage=float(q1["coverage_selected"]),
        old_basis="arrival share, uncapped per-fence 95th percentile",
        capped_all_sites_arrival=dict(
            meas343=out["levels"]["meas343"][
                "max_arrival_coverage_all_sites"],
            fleet495=out["levels"]["fleet495"][
                "max_arrival_coverage_all_sites"]),
        capped_knee_arrival=dict(
            meas343=out["levels"]["meas343"]["arrival_coverage"],
            fleet495=out["levels"]["fleet495"]["arrival_coverage"]),
        capped_all_sites_occupancy=dict(
            meas343=out["levels"]["meas343"]["max_occupancy_coverage_all_sites"],
            fleet495=out["levels"]["fleet495"][
                "max_occupancy_coverage_all_sites"]),
        capped_knee_occupancy=dict(
            meas343=out["levels"]["meas343"]["occupancy_coverage"],
            fleet495=out["levels"]["fleet495"]["occupancy_coverage"]),
        note=("the old and the new numbers are different rulers.  Compared "
              "on T009's arrival-share ruler the capped plans reach "
              "56.75% (all 5,263 sites) and 55.80% (4,000-site knee) under "
              "the 343,638 inventory; occupancy shares are higher "
              "(61.10%/59.53%) because capacity shortfalls sit in cells "
              "whose peaks are large relative to their daily arrivals.  "
              "Quote the arrival-share figure whenever the 76.86% reference "
              "is in the same sentence."))
    dump(out, "summary.json")
    # marginal curve
    curve = {}
    for level in ("meas343", "fleet495"):
        pts = fr[level]["points"]
        nn = np.array([p["n"] for p in pts], dtype=float)
        cc = np.array([p["coverage"] for p in pts], dtype=float)
        eff = np.gradient(cc, nn) * 10000.0 if len(pts) > 2 else None
        curve[level] = dict(
            n=[p["n"] for p in pts], coverage=[p["coverage"] for p in pts],
            arrival_coverage=[p["arrival_coverage"] for p in pts],
            slots=[p["slots"] for p in pts],
            unassigned=[p["unassigned"] for p in pts],
            marginal_pp_per_100_sites=None if eff is None else
            [float(v) for v in eff],
            knee=fr[level]["knee"],
            knee_arrival_coverage=fr[level]["selected"][
                "arrival_coverage"] if fr[level]["selected_is_knee"] else None,
            selected_n=fr[level]["selected_n"],
            selected_arrival_coverage=fr[level]["selected"][
                "arrival_coverage"],
            target=TARGET, target_reached=fr[level]["target_reached"],
            max_coverage_all_sites=fr[level]["all_open_milp"]["coverage"],
            max_arrival_coverage_all_sites=fr[level]["all_open_milp"][
                "arrival_coverage"])
    dump(curve, "marginal_curve.json")
    # spare capacity bundle (both levels)
    sp = {}
    for level in ("meas343", "fleet495"):
        f = os.path.join(RUN, f"spare_{level}.npz")
        if os.path.exists(f):
            q = np.load(f)
            sp[f"spare_{level}"] = q["spare"]
            sp[f"sites_{level}"] = q["sites"]
    if sp:
        np.savez(os.path.join(RUN, "spare_capacity.npz"), **sp)
        print("wrote", os.path.join(RUN, "spare_capacity.npz"))
    return out


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="demand",
                    choices=("demand", "solve", "radius", "replay", "walk",
                             "denominator", "check", "summary", "all"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.stage in ("demand", "all"):
        d = run_demand(force=args.force)
        demand_stats(d)
    if args.stage in ("solve", "all"):
        run_solve(force=args.force)
    if args.stage in ("radius", "all"):
        run_radius(force=args.force)
    if args.stage in ("replay", "all"):
        run_replay(force=args.force)
    if args.stage in ("walk", "all"):
        run_walk(force=args.force)
    if args.stage in ("denominator", "all"):
        run_denominator(force=args.force)
    if args.stage in ("summary", "all"):
        run_summary(force=args.force)
    if args.stage in ("check", "all"):
        run_check(force=args.force)
