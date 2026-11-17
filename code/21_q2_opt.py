"""Q2: optimal repositioning on a typical day (time-space network, Gurobi).

State I[z,t] = bikes parked in zone z at the start of period t (1 h).
Trucks are an integral flow on the same time-expanded graph: one arc = one hour,
capacity Q bikes, drop-all at the destination.

    min  c_km * sum dist*m + c_ld * sum x + w_u * sum u + c_dev * sum(dev+ + dev-)/2
    s.t. inventory balance, truck conservation, capacity,  end-of-day deviation

Demand input is selectable: --od q1 (T006 Q1 model, default) or --od obs
(observed typical day).  --conserve switches the H13 arrival rule: arrivals are
generated only by served departures, using the same OD matrix.
"""
import argparse
import json
import os

import gurobipy as gp
import numpy as np
from gurobipy import GRB

import lib
from lib import DATA

P_SAMPLE = 0.128
SCALE = 1.0 / P_SAMPLE
FLEET = 343638.0
Q_TRUCK = 40
COST_KM = 6.0
COST_LOAD = 0.5
PENALTY = 6.0
COST_DEV = 3.0
T = 24


def load_demand(kind):
    """Return dep[z,t], arr[z,t] and P[t,z,z'] at the sampled scale.

    q1 reads data/runs/T006/q1_od.npz (T006); obs reads data/typical_day.npz.
    Both are daily averages over lib.TYP_DAYS.  The 1/p expansion stays a
    downstream SCALE multiplication applied by the caller.
    """
    if kind == "q1":
        z = np.load(os.path.join(DATA, "runs", "T006", "q1_od.npz"))
        od = z["od_q1"].astype(np.float64)
        cent = np.load(os.path.join(DATA, "typical_day.npz"),
                       allow_pickle=True)["cent"]
    elif kind == "obs":
        z = np.load(os.path.join(DATA, "typical_day.npz"), allow_pickle=True)
        od = z["od_z"].astype(np.float64)
        cent = z["cent"]
    else:
        raise ValueError(kind)
    dep = od.sum(2).T.copy()          # (Z,T)
    arr = od.sum(1).T.copy()
    row = od.sum(2)
    Pz = np.divide(od, row[:, :, None],
                   out=np.zeros_like(od), where=row[:, :, None] > 0)
    return dep, arr, Pz, cent


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


DEP, ARR, PZ, CENT = load_demand("q1")
DIST, ARCS, OUT_TO, IN_FROM = load_network(CENT)
NZ = len(CENT)
RQ = np.load(os.path.join(DATA, "q2_state.npz"), allow_pickle=True)
I0_base = RQ["I0"] / FLEET


def solve(alpha=1.0, K=60, time_limit=300, integer=False, verbose=False,
          fixed_K_cost=0.0, conserve=False, od="q1"):
    I0 = I0_base * FLEET * alpha
    m = gp.Model("rebal")
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = 0.02
    m.Params.Method = 2

    vt = GRB.INTEGER if integer else GRB.CONTINUOUS
    I = m.addVars(NZ, T, lb=0, name="I")
    u = m.addVars(NZ, T, lb=0, name="u")
    x = m.addVars(ARCS, T, lb=0, name="x")
    mt = m.addVars(ARCS, T, lb=0, vtype=vt, name="m")
    nn = m.addVars(NZ, T, lb=0, vtype=vt, name="n")
    dp = m.addVars(NZ, lb=0, name="dp")
    dm = m.addVars(NZ, lb=0, name="dm")

    m.setObjective(
        COST_KM * gp.quicksum(DIST[i, j] * mt[i, j, t]
                              for (i, j) in ARCS for t in range(T))
        + COST_LOAD * gp.quicksum(x[i, j, t] for (i, j) in ARCS
                                  for t in range(T))
        + PENALTY * gp.quicksum(u[z_, t] for z_ in range(NZ) for t in range(T))
        + COST_DEV * gp.quicksum(dp[z_] + dm[z_] for z_ in range(NZ))
        + fixed_K_cost * K,
        GRB.MINIMIZE)

    served = {(z_, t): DEP[z_, t] - u[z_, t]
              for z_ in range(NZ) for t in range(T)}
    if conserve:
        # H13: arrivals come only from departures that were actually served.
        arr_served = {}
        for z2 in range(NZ):
            for t in range(T):
                arr_served[(z2, t)] = gp.quicksum(
                    PZ[t, z1, z2] * served[(z1, t)] for z1 in range(NZ))
        for z_ in range(NZ):
            for t in range(T):
                m.addConstr(u[z_, t] <= DEP[z_, t], name=f"ule{z_}_{t}")
    else:
        arr_served = None

    for z_ in range(NZ):
        m.addConstr(I[z_, 0] == I0[z_])
        arr_end = (arr_served[(z_, T - 1)] if conserve else ARR[z_, T - 1])
        serve_end = served[(z_, T - 1)]
        m.addConstr(I[z_, T - 1] + arr_end - serve_end
                    - gp.quicksum(x[z_, j, T - 1] for j in OUT_TO[z_])
                    + gp.quicksum(x[i, z_, T - 1] for i in IN_FROM[z_])
                    - I0[z_] == dp[z_] - dm[z_], name=f"dev{z_}")
        for t in range(T):
            m.addConstr(served[(z_, t)] <= I[z_, t], name=f"serv{z_}_{t}")
            if t < T - 1:
                arr_t = (arr_served[(z_, t)] if conserve else ARR[z_, t])
                m.addConstr(I[z_, t + 1] == I[z_, t] + arr_t - served[(z_, t)]
                            - gp.quicksum(x[z_, j, t] for j in OUT_TO[z_])
                            + gp.quicksum(x[i, z_, t] for i in IN_FROM[z_]),
                            name=f"inv{z_}_{t}")

    # truck conservation: n[z,0] free, fleet total K, movement uses one period
    for z_ in range(NZ):
        for t in range(T):
            if t == 0:
                m.addConstr(nn[z_, 0] <= K)
            else:
                m.addConstr(nn[z_, t] == nn[z_, t - 1]
                            - gp.quicksum(mt[z_, j, t - 1] for j in OUT_TO[z_])
                            + gp.quicksum(mt[i, z_, t - 1] for i in IN_FROM[z_]),
                            name=f"n{z_}_{t}")
            m.addConstr(gp.quicksum(mt[z_, j, t] for j in OUT_TO[z_]) <= nn[z_, t],
                        name=f"navail{z_}_{t}")
        m.addConstr(gp.quicksum(nn[q, 0] for q in range(NZ)) == K, name="fleet0")
    m.addConstr(gp.quicksum(nn[q, T - 1] for q in range(NZ)) == K, name="fleetT")
    for (i, j) in ARCS:
        for t in range(T):
            m.addConstr(x[i, j, t] <= Q_TRUCK * mt[i, j, t],
                        name=f"cap{i}_{j}_{t}")

    m.optimize()
    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        print("  status", m.Status)
        return None
    res = dict(alpha=alpha, K=K, od=od, conserve=bool(conserve),
               fixed_K_cost=fixed_K_cost, obj=m.ObjVal,
               obj_variable=m.ObjVal - fixed_K_cost * K,
               status=int(m.Status),
               bound=float(getattr(m, "ObjBound", np.nan)),
               gap=float(getattr(m, "MIPGap", 0.0)),
               unmet=float(sum(u[z_, t].X for z_ in range(NZ) for t in range(T))),
               truck_km=float(sum(DIST[i, j] * mt[i, j, t].X
                                  for (i, j) in ARCS for t in range(T))),
               moved=float(sum(x[i, j, t].X for (i, j) in ARCS for t in range(T))),
               trucks_used=float(sum(mt[i, j, t].X for (i, j) in ARCS
                                     for t in range(T))),
               dev=float(sum(dp[q].X + dm[q].X for q in range(NZ))),
               trips=float(DEP.sum()))
    print(f"  od={od} conserve={conserve} a={alpha:.2f} K={K:3d}  "
          f"moved={res['moved']:8.0f}  truck-km={res['truck_km']:8.0f}  "
          f"trips/truck={res['trucks_used']:7.1f}  unmet={res['unmet']:7.0f} "
          f"({100*res['unmet']/res['trips']:.2f}%)  dev={res['dev']:7.0f}  "
          f"obj={res['obj']:10.0f}")
    res["X"] = np.array([[sum(x[z_, j, t].X for j in OUT_TO[z_])
                          for t in range(T)] for z_ in range(NZ)])
    res["M"] = np.array([[sum(mt[z_, j, t].X for j in OUT_TO[z_])
                          for t in range(T)] for z_ in range(NZ)])
    res["I"] = np.array([[I[z_, t].X for t in range(T)] for z_ in range(NZ)])
    res["U"] = np.array([[u[z_, t].X for t in range(T)] for z_ in range(NZ)])
    XM = np.zeros((T, NZ, NZ))
    for (i, j) in ARCS:
        for t in range(T):
            XM[t, i, j] = x[i, j, t].X
    res["XM"] = XM
    return res


def main():
    ap = argparse.ArgumentParser(description="Q2 repositioning LP")
    ap.add_argument("--od", choices=("q1", "obs"), default="q1")
    ap.add_argument("--conserve", action="store_true",
                    help="H13: arrivals from served departures only")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=60)
    ap.add_argument("--time-limit", type=int, default=300)
    ap.add_argument("--fixed-k-cost", type=float, default=0.0)
    ap.add_argument("--integer", action="store_true")
    ap.add_argument("--out", default=None, help="result JSON path")
    ap.add_argument("--plan-out", default=None, help="plan NPZ path")
    args = ap.parse_args()

    global DEP, ARR, PZ, CENT, DIST, ARCS, OUT_TO, IN_FROM, NZ
    DEP, ARR, PZ, CENT = load_demand(args.od)
    DIST, ARCS, OUT_TO, IN_FROM = load_network(CENT)
    NZ = len(CENT)
    r = solve(alpha=args.alpha, K=args.K, time_limit=args.time_limit,
              integer=args.integer, verbose=True,
              fixed_K_cost=args.fixed_k_cost, conserve=args.conserve,
              od=args.od)
    if not r:
        return 1
    if args.out or args.plan_out:
        arrays = {k: r.pop(k) for k in ("X", "M", "I", "U", "XM") if k in r}
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as handle:
                json.dump(r, handle, ensure_ascii=False, indent=1)
            print("wrote", args.out)
        if args.plan_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.plan_out)),
                        exist_ok=True)
            np.savez(args.plan_out, **arrays)
            print("wrote", args.plan_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
