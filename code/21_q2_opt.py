"""Q2: optimal repositioning on a typical day (time-space network, Gurobi).

State I[z,t] = bikes parked in zone z at the start of period t (1 h).
Trucks are an integral flow on the same time-expanded graph: one arc = one hour,
capacity Q bikes, drop-all at the destination.

    min  c_km * sum dist*m + c_ld * sum x + w_u * sum u + c_dev * sum(dev+ + dev-)/2
    s.t. inventory balance, truck conservation, capacity,  end-of-day deviation
"""
import json
import numpy as np
import gurobipy as gp
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

z = np.load(DATA + r"\typical_day.npz", allow_pickle=True)
cent = z["cent"]
NZ = cent.shape[0]
RQ = np.load(DATA + r"\q2_state.npz", allow_pickle=True)
I0_base = RQ["I0"] / FLEET

cells = lib.get_cells()
df = lib.load_trips(lib.TYP_DAYS)
df = lib.assign_cells(df, cells)
ok = df.o.notna() & df.d.notna()
s = df[ok].copy(); s["o"] = s.o.astype(int); s["d"] = s.d.astype(int)
zid = cells.zone.to_numpy()
nd = len(lib.TYP_DAYS)
T = 24
dep = np.zeros((NZ, T)); arr = np.zeros((NZ, T))
np.add.at(dep, (zid[s.o.values], s.starttime.dt.hour.values), 1.0)
np.add.at(arr, (zid[s.d.values], s.starttime.dt.hour.values), 1.0)
dep = dep / nd * SCALE
arr = arr / nd * SCALE

dist = np.sqrt(((cent[:, None, :] - cent[None, :, :]) ** 2).sum(-1))
NEIGH = 8
order = np.argsort(dist, axis=1)[:, 1:NEIGH + 1]
arcs = [(i, int(j)) for i in range(NZ) for j in order[i] if dist[i, j] <= 9.0]
out_to = {i: [] for i in range(NZ)}; in_from = {i: [] for i in range(NZ)}
for (i, j) in arcs:
    out_to[i].append(j); in_from[j].append(i)
print(f"zones {NZ}, arcs {len(arcs)}, mean arc {np.mean([dist[i,j] for i,j in arcs]):.2f} km")


def solve(alpha=1.0, K=60, time_limit=300, integer=False, verbose=False,
          fixed_K_cost=0.0):
    I0 = I0_base * FLEET * alpha
    m = gp.Model("rebal")
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = 0.02
    m.Params.Method = 2

    vt = GRB.INTEGER if integer else GRB.CONTINUOUS
    I = m.addVars(NZ, T, lb=0, name="I")
    u = m.addVars(NZ, T, lb=0, name="u")
    x = m.addVars(arcs, T, lb=0, name="x")
    mt = m.addVars(arcs, T, lb=0, vtype=vt, name="m")
    nn = m.addVars(NZ, T, lb=0, vtype=vt, name="n")
    dp = m.addVars(NZ, lb=0, name="dp"); dm = m.addVars(NZ, lb=0, name="dm")

    m.setObjective(
        COST_KM * gp.quicksum(dist[i, j] * mt[i, j, t] for (i, j) in arcs for t in range(T))
        + COST_LOAD * gp.quicksum(x[i, j, t] for (i, j) in arcs for t in range(T))
        + PENALTY * gp.quicksum(u[z_, t] for z_ in range(NZ) for t in range(T))
        + COST_DEV * gp.quicksum(dp[z_] + dm[z_] for z_ in range(NZ))
        + fixed_K_cost * K,
        GRB.MINIMIZE)

    for z_ in range(NZ):
        m.addConstr(I[z_, 0] == I0[z_])
        m.addConstr(I[z_, T - 1] + arr[z_, T - 1] - (dep[z_, T - 1] - u[z_, T - 1])
                    - gp.quicksum(x[z_, j, T - 1] for j in out_to[z_])
                    + gp.quicksum(x[i, z_, T - 1] for i in in_from[z_])
                    - I0[z_] == dp[z_] - dm[z_], name=f"dev{z_}")
        for t in range(T):
            m.addConstr(dep[z_, t] - u[z_, t] <= I[z_, t], name=f"serv{z_}_{t}")
            if t < T - 1:
                m.addConstr(I[z_, t + 1] == I[z_, t] + arr[z_, t] - (dep[z_, t] - u[z_, t])
                            - gp.quicksum(x[z_, j, t] for j in out_to[z_])
                            + gp.quicksum(x[i, z_, t] for i in in_from[z_]),
                            name=f"inv{z_}_{t}")

    # truck conservation: n[z,0] free, fleet total K, movement uses one period
    for z_ in range(NZ):
        for t in range(T):
            if t == 0:
                m.addConstr(nn[z_, 0] <= K)
            else:
                m.addConstr(nn[z_, t] == nn[z_, t - 1]
                            - gp.quicksum(mt[z_, j, t - 1] for j in out_to[z_])
                            + gp.quicksum(mt[i, z_, t - 1] for i in in_from[z_]),
                            name=f"n{z_}_{t}")
            m.addConstr(gp.quicksum(mt[z_, j, t] for j in out_to[z_]) <= nn[z_, t],
                        name=f"navail{z_}_{t}")
        m.addConstr(gp.quicksum(nn[q, 0] for q in range(NZ)) == K, name=f"fleet0")
    m.addConstr(gp.quicksum(nn[q, T - 1] for q in range(NZ)) == K, name="fleetT")
    for (i, j) in arcs:
        for t in range(T):
            m.addConstr(x[i, j, t] <= Q_TRUCK * mt[i, j, t], name=f"cap{i}_{j}_{t}")

    m.optimize()
    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        print("  status", m.Status); return None
    res = dict(alpha=alpha, K=K, obj=m.ObjVal, status=int(m.Status),
               bound=float(getattr(m, "ObjBound", np.nan)),
               gap=float(getattr(m, "MIPGap", 0.0)),
               unmet=float(sum(u[z_, t].X for z_ in range(NZ) for t in range(T))),
               truck_km=float(sum(dist[i, j] * mt[i, j, t].X for (i, j) in arcs for t in range(T))),
               moved=float(sum(x[i, j, t].X for (i, j) in arcs for t in range(T))),
               trucks_used=float(sum(mt[i, j, t].X for (i, j) in arcs for t in range(T))),
               dev=float(sum(dp[q].X + dm[q].X for q in range(NZ))),
               trips=float(dep.sum()))
    print(f"  a={alpha:.2f} K={K:3d}  moved={res['moved']:8.0f}  truck-km={res['truck_km']:8.0f}  "
          f"trips/truck={res['trucks_used']:7.1f}  unmet={res['unmet']:7.0f} "
          f"({100*res['unmet']/res['trips']:.2f}%)  dev={res['dev']:7.0f}  obj={res['obj']:10.0f}")
    if verbose:
        res["X"] = np.array([[sum(x[z_, j, t].X for j in out_to[z_]) for t in range(T)]
                             for z_ in range(NZ)])
        res["M"] = np.array([[sum(mt[z_, j, t].X for j in out_to[z_]) for t in range(T)]
                             for z_ in range(NZ)])
        res["I"] = np.array([[I[z_, t].X for t in range(T)] for z_ in range(NZ)])
        res["U"] = np.array([[u[z_, t].X for t in range(T)] for z_ in range(NZ)])
        XM = np.zeros((T, NZ, NZ))
        for (i, j) in arcs:
            for t in range(T):
                XM[t, i, j] = x[i, j, t].X
        res["XM"] = XM
    return res


if __name__ == "__main__":
    out = []
    for a in (1.0, 0.8, 0.6, 0.4):
        r = solve(alpha=a, K=0, time_limit=300)
        if r: out.append({**r, "mode": "no trucks"})
    for a in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4):
        for K in (20, 40, 60, 80, 100, 150):
            r = solve(alpha=a, K=K, time_limit=300, fixed_K_cost=600.0,
                      verbose=(K == 60 and a == 1.0))
            if r:
                out.append({**r, "mode": "trucks"})
                if K == 60 and a == 1.0:
                    np.savez(DATA + r"\q2_solution.npz", X=r.pop("X"), M=r.pop("M"),
                             I=r.pop("I"), U=r.pop("U"))
    json.dump([{k: v for k, v in o.items() if not isinstance(v, np.ndarray)} for o in out],
              open(DATA + r"\q2_results.json", "w"), indent=1)
