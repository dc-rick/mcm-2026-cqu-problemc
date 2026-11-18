"""T018: integer per-truck Q2 schedule with a locked hourly event order.

Model changes relative to 21_q2_opt.py (T007):
  * event-order constraint  sum_j x[z,j,t] + served[z,t] <= I[z,t]
    (hour-start inventory is shared by truck pickup and user pickup; user
    returns and truck drops are credited at the end of the hour),
  * truck counts mt, nn integer (x stays continuous, as the task specifies),
  * design demand rebuilt from the five training days only (M11).

Stages (each writes only data/runs/T018/):
  demand    rebuild the Q1 OD on training days, report 5-day vs 7-day diff
  oldviol   T007 K=60 LP solution measured against the event-order constraint
  lp        continuous lower bound for K in {20, 40, 60, 80}
  int       integer MILP for K in {20, 40, 60, 80}
  schedule  per-truck flow decomposition, schedule CSV, replay checks
  check     constraint_check.json (all violations must be 0)
  baselines no-truck / greedy / recommended table + integer-LP gap
  holdout   05-12 and 05-19 replay with the fixed design plan

Run with python -I code/t018_q2_int.py --stage all
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
from scipy.optimize import linprog

import gurobipy as gp
from gurobipy import GRB

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
from lib import DATA

RUN = os.path.join(DATA, "runs", "T018")
os.makedirs(RUN, exist_ok=True)

P_SAMPLE = 0.128
SCALE = 1.0 / P_SAMPLE
FLEET = 343638.0
Q_TRUCK = 40
COST_KM = 6.0
COST_LOAD = 0.5
PENALTY = 6.0
COST_DEV = 3.0
FIXED_K = 600.0
T = 24
NZ = 50

TR_DAYS = ["2017-05-10", "2017-05-11", "2017-05-15", "2017-05-16",
           "2017-05-18"]
TE_DAYS = ["2017-05-12", "2017-05-19"]
BANDS = [(0, 5), (6, 9), (10, 15), (16, 19), (20, 23)]
BAND_NAMES = ["night 0-5", "morning 6-9", "midday 10-15",
              "evening 16-19", "late 20-23"]
SIGMA_A = 1.2

# executability assumptions (each value documented in the delivery)
SPEED_KMH = 20.0
HANDLE_MIN = 0.25          # per bike per operation, two-person crew
K_LIST = [20, 40, 60, 80]

GUROBI_VERSION = ".".join(str(v) for v in gp.gurobi.version())


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
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


def prep(od):
    """Hourly zone OD (sampled scale) -> full-scale dep/arr and P(j|i,t)."""
    od = np.asarray(od, dtype=np.float64)
    dep = od.sum(2).T
    arr = od.sum(1).T
    row = od.sum(2)
    PZ = np.divide(od, row[:, :, None], out=np.zeros_like(od),
                   where=row[:, :, None] > 0)
    return dep * SCALE, arr * SCALE, PZ


def load_network(cent):
    dist = np.sqrt(((cent[:, None, :] - cent[None, :, :]) ** 2).sum(-1))
    NEIGH = 8
    order = np.argsort(dist, axis=1)[:, 1:NEIGH + 1]
    arcs = [(i, int(j)) for i in range(len(cent)) for j in order[i]
            if dist[i, j] <= 9.0]
    out_to = {i: [] for i in range(len(cent))}
    in_from = {i: [] for i in range(len(cent))}
    for (i, j) in arcs:
        out_to[i].append(j)
        in_from[j].append(i)
    return dist, arcs, out_to, in_from


# --------------------------------------------------------------------------
# stage: demand
# --------------------------------------------------------------------------
def build_demand(force=False):
    path = os.path.join(RUN, "q1_od_train.npz")
    diff_path = os.path.join(RUN, "demand_diff.json")
    if os.path.exists(path) and not force:
        print("reuse", path)
        return dict(np.load(path, allow_pickle=True)), load_json_any(diff_path)

    print("building Q1 demand on training days", TR_DAYS)
    cells = lib.get_cells()
    n = len(cells)
    zid = cells.zone.to_numpy().astype(int)
    Z = np.zeros((n, NZ), dtype=np.float32)
    Z[np.arange(n), zid] = 1.0
    D = np.load(os.path.join(DATA, "dist_matrix.npy")).astype(np.float32)
    cent = np.load(os.path.join(DATA, "typical_day.npz"),
                   allow_pickle=True)["cent"]

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
    sub5 = gt[gt.date.isin(TR_DAYS)]
    sub7 = gt[gt.date.isin(lib.TYP_DAYS)]
    np.add.at(dep5, (sub5.hour.values, sub5.o.values), 1.0)
    np.add.at(dep7, (sub7.hour.values, sub7.o.values), 1.0)
    dep5 /= len(TR_DAYS)
    dep7 /= len(lib.TYP_DAYS)

    dep_te = np.zeros((len(TE_DAYS), 24, n))
    arr_te = np.zeros((len(TE_DAYS), 24, n))
    for k, day in enumerate(TE_DAYS):
        s = gt[gt.date == day]
        np.add.at(dep_te[k], (s.hour.values, s.o.values), 1.0)
        np.add.at(arr_te[k], (s.hour.values, s.d.values), 1.0)

    hours_of_band = {b: [h for h in range(24) if BANDS[b][0] <= h <= BANDS[b][1]]
                     for b in range(len(BANDS))}
    od5 = np.zeros((24, NZ, NZ))
    od7 = np.zeros((24, NZ, NZ))
    for b in range(len(BANDS)):
        tr = gt[(gt.band == b) & gt.date.isin(TR_DAYS)]
        Dj = np.bincount(tr.d.values, minlength=n).astype(np.float64)
        A = smooth_attraction(Dj, cells)
        logA = np.log(np.maximum(A, 1e-12)).astype(np.float32)
        logp = norm_logp(-beta[b] * D + logA[None, :])
        P = np.exp(logp)                       # rows already normalised
        R = P @ Z                              # (n, NZ) destination-zone mass
        for h in hours_of_band[b]:
            for dep_obs, od_out in ((dep5, od5), (dep7, od7)):
                d = dep_obs[h].astype(np.float32)
                od_out[h] = Z.T @ (d[:, None] * R)
        del P, R, logp
        print(f"  band {BAND_NAMES[b]}: P built")

    dep5_z = np.stack([np.bincount(zid, weights=dep5[h], minlength=NZ)
                       for h in range(24)])
    dep7_z = np.stack([np.bincount(zid, weights=dep7[h], minlength=NZ)
                       for h in range(24)])
    dep_te_z = np.stack([np.stack(
        [np.bincount(zid, weights=dep_te[k, h], minlength=NZ)
         for h in range(24)]) for k in range(len(TE_DAYS))])
    arr_te_z = np.stack([np.stack(
        [np.bincount(zid, weights=arr_te[k, h], minlength=NZ)
         for h in range(24)]) for k in range(len(TE_DAYS))])

    dep5_full = dep5_z * SCALE
    dep7_full = dep7_z * SCALE
    l1 = float(np.abs(dep5_full - dep7_full).sum())
    mx = float(np.abs(dep5_full - dep7_full).max())
    rel = l1 / max(float(dep7_full.sum()), 1.0)

    t6 = np.load(os.path.join(DATA, "runs", "T006", "q1_od.npz"))["od_q1"]
    reg_abs = np.abs(od7 - t6.astype(np.float64))
    reg = dict(max_abs=float(reg_abs.max()),
               l1=float(reg_abs.sum()),
               total_7day=float(od7.sum()),
               total_t006=float(t6.sum()),
               max_rel=float((reg_abs / np.maximum(np.abs(t6), 1e-9)).max()))

    diff = dict(
        task="T018 demand rebuild", date="2026-11-18",
        train_days=TR_DAYS, typ_days=lib.TYP_DAYS,
        daily_mean_departures_sampled=dict(
            train5=float(dep5_z.sum()), typ7=float(dep7_z.sum())),
        daily_mean_departures_full=dict(
            train5=float(dep5_full.sum()), typ7=float(dep7_full.sum()),
            ratio_train5_over_typ7=float(dep5_full.sum() / dep7_full.sum())),
        zone_hour_difference_full=dict(l1=l1, max_abs=mx,
                                       l1_relative_to_typ7=rel),
        od_total_sampled=dict(train5=float(od5.sum()),
                              typ7=float(od7.sum())),
        od_total_full=dict(train5=float(od5.sum() * SCALE),
                           typ7=float(od7.sum() * SCALE)),
        t006_regression_on_7day=reg,
        note=("train5 = mean over 2017-05-10/11/15/16/18; typ7 = mean over "
              "lib.TYP_DAYS (adds 05-12 and 05-19).  Only the departure "
              "counts differ: both use the same band betas and the same "
              "training-day attraction A_j(h), so the 7-day rebuild should "
              "reproduce data/runs/T006/q1_od.npz."),
    )
    np.savez(path,
             od_q1=od5.astype(np.float32),
             od_q1_7day=od7.astype(np.float32),
             dep_zone=dep5_z.astype(np.float32),
             dep_zone_7=dep7_z.astype(np.float32),
             dep_test=dep_te_z.astype(np.float32),
             arr_test=arr_te_z.astype(np.float32),
             zid=zid.astype(np.int32),
             cent=cent.astype(np.float64),
             train_days=np.array(TR_DAYS),
             test_days=np.array(TE_DAYS))
    dump(diff, "demand_diff.json")
    print(json.dumps(diff, ensure_ascii=False, indent=1))
    return dict(np.load(path, allow_pickle=True)), diff


# --------------------------------------------------------------------------
# MILP / LP
# --------------------------------------------------------------------------
def solve(dep, arr, PZ, net, I0, K, integer, time_limit=600, mip_gap=0.01,
          event_order=True, fixed_cost=FIXED_K, verbose=False):
    dist, arcs, out_to, in_from = net
    m = gp.Model("t018")
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.TimeLimit = time_limit
    if integer:
        m.Params.MIPGap = mip_gap
    m.Params.Method = 2

    vt = GRB.INTEGER if integer else GRB.CONTINUOUS
    I = m.addVars(NZ, T, lb=0, name="I")
    u = m.addVars(NZ, T, lb=0, name="u")
    x = m.addVars(arcs, T, lb=0, name="x")
    mt = m.addVars(arcs, T, lb=0, vtype=vt, name="m")
    nn = m.addVars(NZ, T, lb=0, vtype=vt, name="n")
    dp = m.addVars(NZ, lb=0, name="dp")
    dm = m.addVars(NZ, lb=0, name="dm")

    m.setObjective(
        COST_KM * gp.quicksum(dist[i, j] * mt[i, j, t]
                              for (i, j) in arcs for t in range(T))
        + COST_LOAD * gp.quicksum(x[i, j, t] for (i, j) in arcs
                                  for t in range(T))
        + PENALTY * gp.quicksum(u[z, t] for z in range(NZ) for t in range(T))
        + COST_DEV * gp.quicksum(dp[z] + dm[z] for z in range(NZ))
        + fixed_cost * K,
        GRB.MINIMIZE)

    served = {(z, t): dep[z, t] - u[z, t]
              for z in range(NZ) for t in range(T)}
    arr_served = {}
    for z2 in range(NZ):
        for t in range(T):
            arr_served[(z2, t)] = gp.quicksum(
                PZ[t, z1, z2] * served[(z1, t)] for z1 in range(NZ))
    for z in range(NZ):
        for t in range(T):
            m.addConstr(u[z, t] <= dep[z, t], name=f"ule{z}_{t}")

    for z in range(NZ):
        m.addConstr(I[z, 0] == I0[z])
        arr_end = arr_served[(z, T - 1)]
        serve_end = served[(z, T - 1)]
        m.addConstr(I[z, T - 1] + arr_end - serve_end
                    - gp.quicksum(x[z, j, T - 1] for j in out_to[z])
                    + gp.quicksum(x[i, z, T - 1] for i in in_from[z])
                    - I0[z] == dp[z] - dm[z], name=f"dev{z}")
        for t in range(T):
            if event_order:
                # hour-start inventory is shared by truck pickup and users
                m.addConstr(
                    gp.quicksum(x[z, j, t] for j in out_to[z])
                    + served[(z, t)] <= I[z, t], name=f"order{z}_{t}")
            else:
                m.addConstr(served[(z, t)] <= I[z, t], name=f"serv{z}_{t}")
            if t < T - 1:
                m.addConstr(I[z, t + 1] == I[z, t] + arr_served[(z, t)]
                            - served[(z, t)]
                            - gp.quicksum(x[z, j, t] for j in out_to[z])
                            + gp.quicksum(x[i, z, t] for i in in_from[z]),
                            name=f"inv{z}_{t}")

    for z in range(NZ):
        for t in range(T):
            if t == 0:
                m.addConstr(nn[z, 0] <= K)
            else:
                m.addConstr(nn[z, t] == nn[z, t - 1]
                            - gp.quicksum(mt[z, j, t - 1] for j in out_to[z])
                            + gp.quicksum(mt[i, z, t - 1] for i in in_from[z]),
                            name=f"n{z}_{t}")
            m.addConstr(gp.quicksum(mt[z, j, t] for j in out_to[z]) <= nn[z, t],
                        name=f"navail{z}_{t}")
        m.addConstr(gp.quicksum(nn[q, 0] for q in range(NZ)) == K, name="fleet0")
    m.addConstr(gp.quicksum(nn[q, T - 1] for q in range(NZ)) == K,
                name="fleetT")
    for (i, j) in arcs:
        for t in range(T):
            m.addConstr(x[i, j, t] <= Q_TRUCK * mt[i, j, t],
                        name=f"cap{i}_{j}_{t}")

    m.optimize()
    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        print("  solve failed, status", m.Status)
        return None

    XM = np.zeros((T, NZ, NZ))
    MK = np.zeros((T, NZ, NZ))
    for (i, j) in arcs:
        for t in range(T):
            XM[t, i, j] = x[i, j, t].X
            MK[t, i, j] = mt[i, j, t].X
    NN = np.array([[nn[z, t].X for t in range(T)] for z in range(NZ)])
    II = np.array([[I[z, t].X for t in range(T)] for z in range(NZ)])
    UU = np.array([[u[z, t].X for t in range(T)] for z in range(NZ)])

    peak_moved = float(XM.sum((1, 2)).max())
    peak_hour = int(XM.sum((1, 2)).argmax())
    peak_trips = float(MK.sum((1, 2)).max())
    res = dict(K=K, integer=bool(integer), event_order=bool(event_order),
               obj=float(m.ObjVal),
               obj_variable=float(m.ObjVal - fixed_cost * K),
               status=int(m.Status),
               bound=float(getattr(m, "ObjBound", np.nan)),
               gap=float(getattr(m, "MIPGap", 0.0)) if integer else 0.0,
               runtime=float(m.Runtime),
               unmet=float(sum(u[z, t].X for z in range(NZ) for t in range(T))),
               truck_km=float(sum(dist[i, j] * mt[i, j, t].X
                                  for (i, j) in arcs for t in range(T))),
               moved=float(sum(x[i, j, t].X for (i, j) in arcs
                               for t in range(T))),
               trucks_used=float(sum(mt[i, j, t].X for (i, j) in arcs
                                     for t in range(T))),
               dev=float(sum(dp[q].X + dm[q].X for q in range(NZ))),
               trips=float(dep.sum()),
               peak_moved=peak_moved, peak_hour=peak_hour,
               peak_truck_trips=peak_trips)
    print(f"  K={K:3d} {'int' if integer else 'LP ':3s} obj={res['obj']:10.1f} "
          f"unmet={res['unmet']:8.1f} dev={res['dev']:7.1f} "
          f"moved={res['moved']:8.1f} km={res['truck_km']:8.1f} "
          f"peak={peak_moved:7.1f}@{peak_hour} gap={100*res['gap']:.2f}% "
          f"t={res['runtime']:.1f}s")
    res.update(XM=XM, MK=MK, NN=NN, I=II, U=UU)
    return res


def save_solution(res, tag):
    arrays = {k: res[k] for k in ("XM", "MK", "NN", "I", "U")}
    np.savez(os.path.join(RUN, f"plan_{tag}.npz"), **arrays)
    scalars = {k: v for k, v in res.items() if k not in arrays}
    dump(scalars, f"result_{tag}.json")
    return scalars


def load_solution(tag):
    z = np.load(os.path.join(RUN, f"plan_{tag}.npz"))
    res = {k: z[k] for k in z.files}
    res.update(load_json(f"result_{tag}.json"))
    return res


def solve_schedule(dep, arr, PZ, cent, I0, K, time_limit=600, mip_gap=0.01):
    """T020 interface: zone-hour demand -> cost decomposition + truck schedule.

    Inputs
      dep, arr : (Z, 24) full-scale departures/arrivals
      PZ       : (24, Z, Z) destination probabilities P(j|i,t)
      cent     : (Z, 2) zone centres, km
      I0       : (Z,) initial inventory
      K        : truck fleet size
    Returns dict(cost=..., schedule=DataFrame, result=scalars).
    """
    net = load_network(cent)
    res = solve(dep, arr, PZ, net, I0, K, integer=True,
                time_limit=time_limit, mip_gap=mip_gap, event_order=True)
    if res is None:
        return None
    df = decompose(res["MK"], res["XM"], res["NN"], net[0])
    cost = dict(truck_km=res["truck_km"],
                moved=res["moved"] * COST_LOAD,
                unmet=res["unmet"] * PENALTY,
                dev=res["dev"] * COST_DEV,
                fixed=FIXED_K * K,
                total=res["obj"])
    scalars = {k: v for k, v in res.items()
               if k not in ("XM", "MK", "NN", "I", "U")}
    return dict(cost=cost, schedule=df, result=scalars)


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------
def replay(XM, dep, arr, I0, PZ=None, tol=1e-9):
    """Hourly replay under the locked event order.

    hour start: I -> truck pickup (out) -> user pickup (served) -> at hour end
    add observed/conserved arrivals and truck drops.  Truck pickup and users
    share the hour-start inventory, which is exactly the T018 constraint.
    """
    Z_, T_ = dep.shape
    I = np.asarray(I0, dtype=np.float64).copy()
    unmet = 0.0
    moved = 0.0
    capped_zh = 0
    capped_bikes = 0.0
    traj = np.zeros((Z_, T_))
    out_h = np.zeros(T_)
    for t in range(T_):
        traj[:, t] = I
        out_plan = XM[t].sum(1)
        out = np.minimum(out_plan, I)
        scale = np.divide(out, out_plan, out=np.zeros_like(out),
                          where=out_plan > tol)
        capped_zh += int((out < out_plan - 1e-7).sum())
        capped_bikes += float((out_plan - out).sum())
        drop = (XM[t] * scale[:, None]).sum(0)
        avail = I - out
        served = np.minimum(dep[:, t], avail)
        avail = avail - served
        arr_t = (PZ[t].T @ served) if PZ is not None else arr[:, t]
        I = avail + drop + arr_t
        unmet += float((dep[:, t] - served).sum())
        moved += float(out.sum())
        out_h[t] = out.sum()
    return dict(unmet=unmet, dev=float(np.abs(I - np.asarray(I0)).sum()),
                moved=moved, capped_zone_hours=capped_zh,
                capped_bikes=capped_bikes, I_end=I.copy(), traj=traj,
                out_h=out_h)


# --------------------------------------------------------------------------
# stage: old LP violations
# --------------------------------------------------------------------------
def old_violations():
    plan = np.load(os.path.join(DATA, "runs", "T007", "plans",
                                "q1_a1_K60.npz"))
    od_old = np.load(os.path.join(DATA, "runs", "T006", "q1_od.npz"))["od_q1"]
    dep, arr, PZ = prep(od_old)
    XM = plan["XM"]
    II = plan["I"]
    UU = plan["U"]
    served = dep - UU
    out = XM.sum(2).T
    gap = out + served - II
    viol = gap > 1e-6
    zh = int(viol.sum())
    mx = float(gap.max())
    idx = np.argwhere(viol)
    top = sorted(
        [dict(zone=int(z), hour=int(t), gap=float(gap[z, t]),
              truck_out=float(out[z, t]), served=float(served[z, t]),
              I_start=float(II[z, t]))
         for z, t in idx], key=lambda d: -d["gap"])[:10]
    out_ = dict(
        task="T018 old-solution event-order violations",
        source=("data/runs/T007/plans/q1_a1_K60.npz (T007 K=60 LP, "
                "fixed 600/truck/day, conserve arrivals)"),
        demand=("data/runs/T006/q1_od.npz (the demand the old solution was "
                "optimised on; 7-day TYP_DAYS departure counts)"),
        rule="sum_j XM[t,z,j] + (DEP[z,t]-U[z,t]) - I[z,t] > 1e-6",
        zone_hours_violated=zh,
        zone_hours_total=NZ * T,
        max_gap_bikes=mx,
        total_gap_bikes=float(gap[viol].sum()) if zh else 0.0,
        truck_bikes_in_violating_hours=float(out[viol].sum()) if zh else 0.0,
        top10=top,
        note=("measured on the old solution's own demand; the new MILP "
              "enforces this constraint exactly."),
    )
    dump(out_, "old_violations.json")
    return out_


# --------------------------------------------------------------------------
# stage: schedule decomposition
# --------------------------------------------------------------------------
def decompose(MK, XM, NN, dist):
    T_, Z_ = MK.shape[0], MK.shape[1]
    present = {z: [] for z in range(Z_)}
    tid = 0
    for z in range(Z_):
        cnt = int(round(NN[z, 0]))
        for _ in range(cnt):
            present[z].append(tid)
            tid += 1
    rows = []
    for t in range(T_):
        nxt = {z: [] for z in range(Z_)}
        for z in range(Z_):
            ids = present[z]
            if len(ids) != int(round(NN[z, t])):
                raise RuntimeError(
                    f"truck count mismatch at t={t} z={z}: "
                    f"{len(ids)} vs {NN[z, t]}")
            k = 0
            order = sorted(range(Z_), key=lambda j: (-MK[t, z, j], j))
            for j in order:
                m_ = int(round(MK[t, z, j]))
                if m_ <= 0:
                    continue
                if k + m_ > len(ids):
                    raise RuntimeError(f"dispatch exceeds trucks at {t},{z}")
                load = XM[t, z, j] / m_
                for tid_ in ids[k:k + m_]:
                    rows.append(dict(truck_id=tid_, hour=t, from_zone=z,
                                     to_zone=j, km=float(dist[z, j]),
                                     load_bikes=float(load),
                                     unload_bikes=float(load),
                                     load_after=0.0))
                    nxt[j].append(tid_)
                k += m_
            for tid_ in ids[k:]:
                rows.append(dict(truck_id=tid_, hour=t, from_zone=z,
                                 to_zone=z, km=0.0, load_bikes=0.0,
                                 unload_bikes=0.0, load_after=0.0))
                nxt[z].append(tid_)
        present = nxt
    df = pd.DataFrame(rows).sort_values(["truck_id", "hour"])
    return df.reset_index(drop=True)


def check_schedule(df, res, K, tag):
    XM = res["XM"]
    MK = res["MK"]
    NN = res["NN"]
    dist = res["dist"]
    moved_csv = float(df.load_bikes.sum())
    moved_milp = float(XM.sum())
    XM_csv = np.zeros_like(XM)
    for row in df.itertuples(index=False):
        XM_csv[row.hour, row.from_zone, row.to_zone] += row.load_bikes
    max_elem = float(np.abs(XM_csv - XM).max())
    per_truck = df.groupby("truck_id").size()
    out = dict(
        tag=tag, K=K, rows=int(len(df)),
        expected_rows=int(K * 24),
        rows_ok=bool(len(df) == K * 24),
        trucks=int(df.truck_id.nunique()),
        every_truck_24=bool((per_truck == 24).all()),
        moved_csv=moved_csv, moved_milp=moved_milp,
        moved_diff=abs(moved_csv - moved_milp),
        max_abs_rebuilt_plan=float(max_elem),
        moved_ok=bool(abs(moved_csv - moved_milp) <= 1e-6),
    )
    return out


# --------------------------------------------------------------------------
# stage: constraint checks
# --------------------------------------------------------------------------
def deadhead(NE, NS, dist):
    Z_ = len(NE)
    c = dist.ravel()
    A_eq = np.zeros((2 * Z_, Z_ * Z_))
    for i in range(Z_):
        A_eq[i, i * Z_:(i + 1) * Z_] = 1.0
    for j in range(Z_):
        A_eq[Z_ + j, j::Z_] = 1.0
    b_eq = np.concatenate([NE.astype(float), NS.astype(float)])
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not res.success:
        return dict(feasible=False, message=res.message)
    flow = res.x.reshape(Z_, Z_)
    km = float((flow * dist).sum())
    return dict(feasible=True, deadhead_km=km,
                deadhead_cost_yuan=km * COST_KM,
                deadhead_truck_trips=float(flow.sum()),
                max_supply_error=float(np.abs(flow.sum(1) - NE).max()),
                max_demand_error=float(np.abs(flow.sum(0) - NS).max()))


def constraint_check(res, K, name="best"):
    dep, arr, PZ = res["dep"], res["arr"], res["PZ"]
    XM, MK, NN, II, UU = res["XM"], res["MK"], res["NN"], res["I"], res["U"]
    dist, arcs, out_to, in_from = res["net"]
    I0 = res["I0"]

    out = XM.sum(2).T
    served = dep - UU
    gap = out + served - II
    order_v = int((gap > 1e-6).sum())
    order_mx = float(gap.max())

    mask = MK > 1e-9
    loads = np.zeros_like(XM)
    loads[mask] = XM[mask] / MK[mask]
    load_v = int((loads > Q_TRUCK + 1e-6).sum())
    load_mx = float(loads.max())

    minutes = np.zeros_like(XM)
    for t in range(T):
        for (i, j) in arcs:
            if MK[t, i, j] > 1e-9:
                minutes[t, i, j] = (dist[i, j] / SPEED_KMH * 60.0
                                    + 2.0 * loads[t, i, j] * HANDLE_MIN)
    dur_v = int((minutes > 60.0 + 1e-7).sum())
    dur_mx = float(minutes.max())

    def sensitivity(factor):
        m = np.zeros_like(XM)
        for t in range(T):
            for (i, j) in arcs:
                if MK[t, i, j] > 1e-9:
                    m[t, i, j] = (dist[i, j] / SPEED_KMH * 60.0
                                  + 2.0 * loads[t, i, j] * HANDLE_MIN * factor)
        return dict(factor=factor,
                    handling_min_per_bike_operation=HANDLE_MIN * factor,
                    violations=int((m > 60.0 + 1e-7).sum()),
                    max_minutes=float(m.max()))

    cons = np.zeros((NZ, T))
    for t in range(1, T):
        cons[:, t] = (NN[:, t] - (NN[:, t - 1]
                                  - MK[t - 1].sum(1)
                                  + MK[t - 1].sum(0)))
    cons_v = int((np.abs(cons) > 1e-6).sum())
    cons_mx = float(np.abs(cons).max())

    NE = (NN[:, T - 1] - MK[T - 1].sum(1) + MK[T - 1].sum(0))
    NS = NN[:, 0]
    dh = deadhead(NE, NS, dist)

    rep = replay(XM, dep, arr, I0, PZ=PZ)
    inv_v = int((rep["traj"] < -1e-6).sum())
    inv_min = float(rep["traj"].min())
    traj_err = float(np.abs(rep["traj"] - II).max())
    unmet_err = abs(rep["unmet"] - res["unmet"])

    out_ = dict(
        task="T018 constraint check", name=name, K=K,
        event_order=dict(violations=order_v, max_gap_bikes=order_mx,
                         rule="sum_j x[z,j,t] + served[z,t] <= I[z,t]"),
        load=dict(violations=load_v, max_load_bikes=load_mx,
                  limit=Q_TRUCK),
        duration=dict(violations=dur_v, max_trip_minutes=dur_mx,
                      speed_kmh=SPEED_KMH,
                      handling_min_per_bike_operation=HANDLE_MIN,
                      hour_limit_min=60.0,
                      sensitivity=dict(minus50=sensitivity(0.5),
                                       plus50=sensitivity(1.5))),
        truck_conservation=dict(violations=cons_v, max_residual=cons_mx,
                                fleet_start=float(NN[:, 0].sum()),
                                fleet_end=float(NE.sum()), K=K),
        inventory_nonneg=dict(violations=inv_v, min_inventory=inv_min),
        end_positions=dict(violations=0 if dh["feasible"] else 1,
                           end_total=float(NE.sum()),
                           start_total=float(NS.sum()),
                           deadhead_km=dh.get("deadhead_km"),
                           deadhead_cost_yuan=dh.get("deadhead_cost_yuan"),
                           deadhead_truck_trips=dh.get("deadhead_truck_trips"),
                           note=("end-of-day trucks matched to the design "
                                 "day's 00:00 start positions; the matching "
                                 "km is the extra cost of making the daily "
                                 "plan repeatable")),
        replay=dict(unmet=rep["unmet"], dev=rep["dev"], moved=rep["moved"],
                    capped_zone_hours=rep["capped_zone_hours"],
                    max_traj_error_vs_milp=traj_err,
                    unmet_error_vs_milp=unmet_err),
        note=("all violation counts must be zero; max_gap_bikes and "
              "max_residual are reported even when zero."),
    )
    return out_


# --------------------------------------------------------------------------
# stage: greedy
# --------------------------------------------------------------------------
def greedy_plan(dep, arr, K=60, Q=40):
    """Exact copy of 22_q2_sim.py greedy_plan on a (Z,T) demand surface."""
    plan = np.zeros((T, NZ, NZ))
    trucks_at = np.full(NZ, K / NZ, dtype=float)
    trucks_start = np.zeros((T, NZ))
    for t in range(T):
        trucks_start[t] = trucks_at
        need = dep[:, t] - arr[:, t]
        surplus = np.argsort(need)
        deficit = np.argsort(need)[::-1]
        avail = Q * trucks_at
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


def greedy_metrics(plan, trucks_start, dep, arr, I0, K=60, Q=40):
    rep = replay(plan, dep, arr, I0, PZ=None)
    dist, arcs, out_to, in_from = NET
    km = 0.0
    for t in range(T):
        for i in range(NZ):
            for j in range(NZ):
                if plan[t, i, j] > 0:
                    km += (plan[t, i, j] / Q) * dist[i, j]
    moved_plan = float(plan.sum())
    cost = (COST_KM * km + COST_LOAD * rep["moved"] + PENALTY * rep["unmet"]
            + COST_DEV * rep["dev"] + FIXED_K * K)
    return dict(
        K=K, Q=Q,
        unmet=rep["unmet"], dev=rep["dev"], truck_km=km,
        moved=rep["moved"], moved_planned=moved_plan,
        peak_moved=float(rep["out_h"].max()),
        peak_moved_planned=float(plan.sum((1, 2)).max()),
        peak_hour=int(plan.sum((1, 2)).argmax()),
        trucks_used=float(moved_plan / Q),
        capped_zone_hours=rep["capped_zone_hours"],
        capped_bikes=rep["capped_bikes"],
        cost=cost,
    )


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
NET = None
DEP = ARR = PZ = I0 = None


def setup_design():
    global NET, DEP, ARR, PZ, I0
    z = np.load(os.path.join(RUN, "q1_od_train.npz"), allow_pickle=True)
    DEP, ARR, PZ = prep(z["od_q1"])
    cent = z["cent"]
    NET = load_network(cent)
    I0 = np.load(os.path.join(DATA, "q2_state.npz"))["I0"].astype(float)
    print(f"design demand: trips={DEP.sum():.1f}  I0={I0.sum():.1f}")


def main():
    ap = argparse.ArgumentParser(description="T018 integer Q2 schedule")
    ap.add_argument("--stage", default="all",
                    choices=("demand", "oldviol", "lp", "int", "schedule",
                             "check", "baselines", "holdout", "all"))
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--time-limit", type=int, default=600)
    ap.add_argument("--mip-gap", type=float, default=0.01)
    ap.add_argument("--force-solve", action="store_true")
    args = ap.parse_args()

    stage = args.stage
    if stage in ("demand", "all"):
        build_demand(force=args.rebuild)
    if stage == "demand":
        return 0

    setup_design()

    if stage in ("oldviol", "all"):
        old_violations()

    if stage in ("lp", "all"):
        for K in K_LIST:
            tag = f"lp_K{K}"
            if os.path.exists(os.path.join(RUN, f"result_{tag}.json")) \
                    and not args.force_solve:
                print("reuse", tag)
                continue
            r = solve(DEP, ARR, PZ, NET, I0, K, integer=False,
                      time_limit=args.time_limit, event_order=True)
            if r is not None:
                save_solution(r, tag)

    if stage in ("int", "all"):
        for K in K_LIST:
            tag = f"int_K{K}"
            if os.path.exists(os.path.join(RUN, f"result_{tag}.json")) \
                    and not args.force_solve:
                print("reuse", tag)
                continue
            r = solve(DEP, ARR, PZ, NET, I0, K, integer=True,
                      time_limit=args.time_limit, mip_gap=args.mip_gap,
                      event_order=True)
            if r is not None:
                save_solution(r, tag)

    if stage in ("schedule", "all"):
        for K in K_LIST:
            tag = f"int_K{K}"
            f = os.path.join(RUN, f"result_{tag}.json")
            if not os.path.exists(f):
                continue
            res = load_solution(tag)
            res.update(dep=DEP, arr=ARR, PZ=PZ, I0=I0, net=NET, dist=NET[0])
            df = decompose(res["MK"], res["XM"], res["NN"], NET[0])
            df.to_csv(os.path.join(RUN, f"schedule_K{K}.csv"), index=False,
                      float_format="%.17g")
            print("wrote", os.path.join(RUN, f"schedule_K{K}.csv"),
                  len(df), "rows")

    if stage in ("check", "all"):
        rec = pick_recommended() if stage == "all" else pick_recommended()
        res = load_solution(f"int_K{rec['K']}")
        res.update(dep=DEP, arr=ARR, PZ=PZ, I0=I0, net=NET, dist=NET[0])
        cc = constraint_check(res, rec["K"], name=f"recommended_K{rec['K']}")
        cc["old_lp_T007_K60"] = load_json("old_violations.json") \
            if os.path.exists(os.path.join(RUN, "old_violations.json")) \
            else {}
        all_zero = all(v["violations"] == 0 for k, v in cc.items()
                       if isinstance(v, dict) and "violations" in v)
        cc["all_zero"] = bool(all_zero)
        cc["status"] = "delivered" if all_zero else "partial"
        dump(cc, "constraint_check.json")

        # schedule-level checks for every K
        sched = {}
        for K in K_LIST:
            f = os.path.join(RUN, f"schedule_K{K}.csv")
            if not os.path.exists(f):
                continue
            r = load_solution(f"int_K{K}")
            r.update(dep=DEP, arr=ARR, PZ=PZ, I0=I0, net=NET, dist=NET[0])
            df = pd.read_csv(f)
            sched[f"K{K}"] = check_schedule(df, r, K, f"int_K{K}")
        dump(sched, "schedule_check.json")

    if stage in ("baselines", "all"):
        run_baselines()

    if stage in ("holdout", "all"):
        run_holdout()

    return 0


def pick_recommended():
    rows = []
    for K in K_LIST:
        f = os.path.join(RUN, f"result_int_K{K}.json")
        if os.path.exists(f):
            rows.append(load_json(f"result_int_K{K}.json"))
    if not rows:
        return None
    return min(rows, key=lambda d: d["obj"])


def run_baselines():
    rows = {}
    # no trucks
    f = os.path.join(RUN, "result_lp_K0.json")
    if not os.path.exists(f):
        r0 = solve(DEP, ARR, PZ, NET, I0, 0, integer=False,
                   time_limit=120, event_order=True)
        if r0 is not None:
            save_solution(r0, "lp_K0")
    r0 = load_solution("lp_K0")
    rows["no_trucks"] = dict(
        K=0, unmet=r0["unmet"], dev=r0["dev"], truck_km=0.0, moved=0.0,
        peak_moved=0.0, peak_hour=None,
        cost=PENALTY * r0["unmet"] + COST_DEV * r0["dev"])

    # greedy on the same design demand
    plan, trucks_start = greedy_plan(DEP, ARR, K=60, Q=Q_TRUCK)
    g = greedy_metrics(plan, trucks_start, DEP, ARR, I0, K=60, Q=Q_TRUCK)
    rows["greedy"] = g

    # regression: greedy on the old 7-day demand should give 623.17
    od_old = np.load(os.path.join(DATA, "runs", "T006", "q1_od.npz"))["od_q1"]
    dep_o, arr_o, _ = prep(od_old)
    plan_o, _ = greedy_plan(dep_o, arr_o, K=60, Q=Q_TRUCK)
    ref = load_json_any(os.path.join(DATA, "runs", "T007",
                                     "q2_greedy_check.json"))
    reg = dict(peak_moved_old_demand=float(plan_o.sum((1, 2)).max()),
               peak_moved_reference=float(ref["max_hourly_moved"]),
               diff=float(plan_o.sum((1, 2)).max() - ref["max_hourly_moved"]))

    lp_rows, int_rows = {}, {}
    for K in K_LIST:
        fp = os.path.join(RUN, f"result_lp_K{K}.json")
        if os.path.exists(fp):
            lp_rows[f"K{K}"] = load_json(f"result_lp_K{K}.json")
        fi = os.path.join(RUN, f"result_int_K{K}.json")
        if os.path.exists(fi):
            int_rows[f"K{K}"] = load_json(f"result_int_K{K}.json")

    best = pick_recommended()
    rows["recommended_integer"] = dict(
        K=best["K"], unmet=best["unmet"], dev=best["dev"],
        truck_km=best["truck_km"], moved=best["moved"],
        peak_moved=best["peak_moved"], peak_hour=best["peak_hour"],
        peak_truck_trips=best["peak_truck_trips"],
        trucks_used=best["trucks_used"], cost=best["obj"])
    gap_rows = {}
    for K in K_LIST:
        l = lp_rows.get(f"K{K}")
        i = int_rows.get(f"K{K}")
        if l and i:
            gap_rows[f"K{K}"] = dict(
                lp_obj=l["obj"], int_obj=i["obj"],
                diff_yuan=i["obj"] - l["obj"],
                diff_pct=100.0 * (i["obj"] - l["obj"]) / max(l["obj"], 1e-9),
                int_gap=i["gap"], runtime=i["runtime"])

    out = dict(
        task="T018 baselines", date="2026-11-18",
        design=dict(train_days=TR_DAYS, trips=float(DEP.sum()),
                    I0=float(I0.sum()), p=P_SAMPLE, scale=SCALE,
                    event_order="truck pickup and users share hour-start I",
                    arrivals="conserve rule: arrivals from served departures "
                             "via the fitted OD matrix",
                    objective=("6*truck_km + 0.5*moved + 6*unmet + 3*dev "
                               "+ 600*K")),
        rows=rows,
        lp_rows={k: {kk: vv for kk, vv in v.items() if kk != "X"}
                 for k, v in lp_rows.items()},
        int_rows={k: {kk: vv for kk, vv in v.items() if kk != "X"}
                  for k, v in int_rows.items()},
        integer_vs_lp=gap_rows,
        recommended=dict(K=best["K"], obj=best["obj"],
                         peak_moved=best["peak_moved"],
                         peak_hour=best["peak_hour"],
                         peak_truck_trips=best["peak_truck_trips"]),
        greedy_regression=reg,
        note=("623.17 bikes/hour is the greedy baseline under the old 7-day "
              "demand, not the recommended plan.  The recommended plan's peak "
              "is reported in recommended.peak_moved."),
    )
    dump(out, "baselines.json")
    return out


def run_holdout():
    best = pick_recommended()
    K = best["K"]
    res = load_solution(f"int_K{K}")
    z = np.load(os.path.join(RUN, "q1_od_train.npz"), allow_pickle=True)
    dep_test = z["dep_test"].astype(np.float64) * SCALE
    arr_test = z["arr_test"].astype(np.float64) * SCALE
    design_trips = float(DEP.sum())
    rows = {}
    for k, day in enumerate(TE_DAYS):
        dep = dep_test[k].T
        arr = arr_test[k].T
        rep = replay(res["XM"], dep, arr, I0, PZ=None)
        rows[day] = dict(
            observed_departures_full=float(dep.sum()),
            observed_arrivals_full=float(arr.sum()),
            design_departures_full=design_trips,
            ratio_to_design=float(dep.sum() / max(design_trips, 1e-9)),
            unmet=rep["unmet"],
            dev=rep["dev"],
            moved=rep["moved"],
            moved_planned=float(res["XM"].sum()),
            capped_zone_hours=rep["capped_zone_hours"],
            capped_bikes=rep["capped_bikes"],
        )
        print(day, rows[day])
    out = dict(
        task="T018 holdout replay", date="2026-11-18",
        plan=f"integer K={K} designed on training days only",
        information_set=("when the plan was made the dispatcher knew only "
                         "the training-day (05-10/11/15/16/18) OD, the "
                         "fitted destination model, I0=343,638 and the cost "
                         "parameters.  The test day's departures and "
                         "arrivals enter only in the replay; the plan "
                         "itself is unchanged."),
        arrival_rule=("holdout replay uses the test day's observed arrivals "
                      "(not the design conserve rule), as the task "
                      "specifies."),
        rows=rows,
    )
    dump(out, "holdout.json")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
