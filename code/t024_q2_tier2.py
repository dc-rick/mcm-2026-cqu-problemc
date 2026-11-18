"""T024: two-tier Q2 schedule -- trucks between zones, tricycles inside zones.

Tier 1 is T018's integer zone-level MILP, unchanged: it fixes each zone's
hour-start inventory I[z,h] and its truck flows.  Tier 2 is a cell-level
transport model inside each zone:

  * zone totals are inherited from T018 (truck pickups pro-rata to stock,
    truck drops pro-rata to the hour's arrival surface), so the two tiers
    agree exactly at the zone level;
  * arrivals at a cell are Pcell[h,z1,c] * served_z1(h), where Pcell is the
    count-based destination kernel rescaled so that its zone sums equal
    T018's PZ -- the cell layer reproduces T018's zone arrivals exactly;
  * users first walk to a neighbour within 300 m (CellSim rule), then
    tricycles move bikes from donor cells to the residual deficits, subject
    to a per-tricycle time budget; unmet pays the same 6 CNY as T018.

Stages: prep | cellunmet | dyn | fleet | schedule | check | all
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
import t018_q2_int as t18

RUN = os.path.join(lib.DATA, "runs", "T024")
os.makedirs(RUN, exist_ok=True)

P_SAMPLE = 0.128
SCALE = 1.0 / P_SAMPLE
WALK_KM = 0.30
PENALTY = 6.0

# tricycle parameters: planning assumptions (see vehicle_params.json)
TRI_Q = 20.0            # bikes per trip
TRI_V = 15.0            # km/h loaded
TRI_HANDLE = 0.5        # minutes per bike loaded or unloaded
TRI_FIXED = 300.0       # CNY per tricycle per day
TRI_KM = 0.20           # CNY per vehicle-km
TRI_MAX_TRIP = 5.0      # km
TRI_NEAREST = 12        # donor candidates per deficit cell
TRI_NMAX = 60           # priced per-zone cap (R2: cap 60 is not binding)
TRI_NCURVE = 60         # curves evaluated to 60
TRI_HOUR0 = 7           # one 13-hour shift, 07:00-20:00
TRI_HOUR1 = 20

FLEETS = list(range(0, 2001, 100))

PLAN_K = 60


def dump(obj, name):
    path = os.path.join(RUN, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    print("wrote", path)
    return path


def load_json(name):
    with open(os.path.join(RUN, name), encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
def prep(force=False):
    path = os.path.join(RUN, "cell_inputs.npz")
    if os.path.exists(path) and not force:
        print("reuse", path)
        return dict(np.load(path, allow_pickle=True)), load_json("cell_inputs.json")

    t18.setup_design()
    DEP, PZ, I0 = t18.DEP, t18.PZ, t18.I0       # (Z,24), (24,Z,Z), (Z,)
    cells = lib.get_cells()
    C = len(cells)
    zid = cells.zone.to_numpy().astype(int)
    Z = int(zid.max()) + 1

    df = lib.load_trips(lib.ALL_DAYS)
    cells2, df = lib.make_cells(df, lib.THR)
    assert len(cells2) == C
    df = df[df.date.isin(t18.TR_DAYS)].copy()
    ok = df.o.notna() & df.d.notna()
    s = df[ok].copy()
    s["o"] = s.o.astype(int)
    s["d"] = s.d.astype(int)
    step5 = ((s.starttime.dt.hour * 60 + s.starttime.dt.minute) // 5).values
    dep5 = np.zeros((288, C))
    np.add.at(dep5, (step5, s.o.values), 1.0)
    dep5 = dep5 / len(t18.TR_DAYS) * SCALE
    dep_cell = dep5.reshape(24, 12, C).sum(1)      # (24, C)

    cnt = np.zeros((Z, C))
    np.add.at(cnt, (zid[s.o.values], s.d.values), 1.0)
    # pseudo-counts: a zone pair with no sampled trip must still receive the
    # model's PZ mass; 1e-3 is negligible against the smallest real count
    cnt = cnt + 1e-3
    pest = cnt / np.maximum(cnt.sum(1, keepdims=True), 1e-9)

    Zm = np.zeros((C, Z))
    Zm[np.arange(C), zid] = 1.0
    Rz = pest @ Zm                                  # (Z,Z) zone mass per origin
    Pcell = np.empty((24, Z, C), dtype=np.float32)
    for h in range(24):
        fac = PZ[h][:, zid] / np.maximum(Rz[:, zid], 1e-12)
        Pcell[h] = (pest * fac).astype(np.float32)

    cs = np.load(os.path.join(lib.DATA, "cell_sim_inputs.npz"),
                 allow_pickle=True)
    I0c = cs["I0c"].astype(np.float64)
    for z in range(Z):
        m = zid == z
        I0c[m] *= I0[z] / max(I0c[m].sum(), 1e-12)
    xy = cs["xy"].astype(np.float64)

    np.savez(path, dep_cell=dep_cell, Pcell=Pcell, I0c=I0c,
             zid=zid.astype(np.int32), xy=xy,
             dep_zone=DEP.astype(np.float64), PZ=PZ.astype(np.float64),
             lon=cells.lon.to_numpy(), lat=cells.lat.to_numpy())
    # exactness of the zone aggregation: sum_c in z Pcell[h,z1,c] = PZ[h,z1,z]
    agg = np.zeros((24, Z, Z))
    for z in range(Z):
        m = zid == z
        agg[:, :, z] = Pcell[:, :, m].sum(2)
    meta = dict(
        task="T024 cell inputs", date="2026-11-18",
        train_days=t18.TR_DAYS, n_cells=int(C), n_zones=int(Z),
        dep_cell_total=float(dep_cell.sum()),
        dep_zone_total=float(DEP.sum()),
        i0_total=float(I0c.sum()),
        pcell_row_sum_max_err=float(np.abs(
            Pcell.astype(np.float64).sum(2) - 1.0)[DEP.T > 1e-9].max()),
        pcell_zone_match_max_err=float(np.abs(agg - PZ).max()),
        note=("dep_cell from the five T018 training days, 5-min bins, "
              "scaled by 1/0.128; Pcell = count kernel rescaled per "
              "(hour, origin zone, destination zone) to T018's PZ; I0c = "
              "observed end-of-day cell stock rescaled within each zone to "
              "T018's I0."))
    dump(meta, "cell_inputs.json")
    return dict(dep_cell=dep_cell, Pcell=Pcell, I0c=I0c, zid=zid, xy=xy,
                dep_zone=DEP, PZ=PZ, lon=cells.lon.to_numpy(),
                lat=cells.lat.to_numpy()), meta


# --------------------------------------------------------------------------
def load_plan(k=PLAN_K):
    z = np.load(os.path.join(lib.DATA, "runs", "T018", f"plan_int_K{k}.npz"))
    with open(os.path.join(lib.DATA, "runs", "T018",
                           f"result_int_K{k}.json"), encoding="utf-8") as fh:
        scal = json.load(fh)
    return z, scal


def walk_borrow0(I, resid, nbrs):
    """CellSim greedy: the largest-residual cell takes from its richest
    neighbour within 300 m; stock moved is a real ride."""
    served = np.zeros_like(resid)
    for c in np.argsort(-resid):
        need = resid[c]
        if need <= 1e-9:
            continue
        nb = np.asarray(nbrs[c])
        while need > 1e-9 and len(nb):
            j = nb[np.argmax(I[nb])]
            if I[j] <= 1e-9:
                break
            take = min(need, I[j])
            I[j] -= take
            need -= take
            served[c] += take
            nb = nb[I[nb] > 1e-9]
    return served


def static_cellunmet(inp, plan):
    """Disaggregate T018's zone stock by the hourly arrival surface."""
    dep = inp["dep_cell"]                       # (24,C)
    Pc = inp["Pcell"].astype(np.float64)        # (24,Z,C)
    zid = inp["zid"]
    Z = int(zid.max()) + 1
    I = plan["I"]                               # (Z,24)
    XM = plan["XM"]                             # (24,Z,Z)
    U = plan["U"]                               # (Z,24)
    served_zone = inp["dep_zone"] - U           # (Z,24)
    arr_cell = np.einsum("hzc,zh->hc", Pc, served_zone)   # (24,C)
    out = XM.sum(1)                              # (24,Z) truck pickups
    tot_strict = tot_walk = 0.0
    by_hour = []
    cell_short = np.zeros(len(zid))
    cell_hits = np.zeros(len(zid))
    cell_mean = np.zeros(len(zid))
    nbrs = cKDTree(inp["xy"]).query_ball_point(inp["xy"], WALK_KM)
    for h in range(24):
        avail = I[:, h] - out[h]
        stock = np.zeros(len(zid))
        for z in range(Z):
            m = zid == z
            share = arr_cell[h, m]
            if share.sum() <= 1e-12:
                share = dep[h, m]
            if share.sum() <= 1e-12:
                share = np.ones(m.sum())
            stock[m] = avail[z] * share / share.sum()
        short = np.maximum(dep[h] - stock, 0.0)
        tot_strict += short.sum()
        cell_short += short
        cell_hits += short > 1e-9
        cell_mean += np.where(short > 1e-9, dep[h] - stock, 0.0)
        Iw = stock.copy()
        served_own = np.minimum(dep[h], Iw)
        Iw -= served_own
        resid = dep[h] - served_own
        served = walk_borrow0(Iw, resid, nbrs)
        un = np.maximum(resid - served, 0.0)
        tot_walk += un.sum()
        by_hour.append(dict(hour=h, strict=float(short.sum()),
                            walk300=float(un.sum()),
                            zone_unmet=float(U[:, h].sum())))
    order = np.argsort(-cell_short)[:10]
    return dict(
        total_strict=float(tot_strict), total_walk300=float(tot_walk),
        zone_unmet_total=float(U.sum()),
        strict_share_of_trips=float(tot_strict / max(dep.sum(), 1e-9)),
        walk_share_of_trips=float(tot_walk / max(dep.sum(), 1e-9)),
        by_hour=by_hour,
        worst_cells=[dict(cell=int(c), shortfall=float(cell_short[c]),
                          hours=int(cell_hits[c]),
                          lon=float(inp["lon"][c]),
                          lat=float(inp["lat"][c])) for c in order],
        method=("Zone stock = I[z,h] - truck pickups at hour start, split "
                "across cells proportional to the hour's expected arrivals "
                "arr_c(h); fallback to departures when the zone-hour has no "
                "arrivals. strict unmet = max(0, dep_c - stock_c); walk300 "
                "adds the CellSim greedy borrow within 300 m."))


# --------------------------------------------------------------------------
def _tri_lp(sink, donor, resid, I, D, n):
    """Transportation LP: donors -> sinks in one zone-hour, time budget n."""
    dd = D[np.ix_(sink, donor)]                       # (S,Dn)
    arcs = []
    for a in range(len(sink)):
        js = np.where(dd[a] <= TRI_MAX_TRIP)[0]
        if len(js) == 0:
            continue
        js = js[np.argsort(dd[a, js])[:TRI_NEAREST]]
        for b in js:
            arcs.append((a, b))
    if not arcs or n <= 1e-9:
        return None, 0.0
    S, nd = len(sink), len(donor)
    na = len(arcs)
    nv = na + S
    c = np.zeros(nv)
    for k, (a, b) in enumerate(arcs):
        c[k] = TRI_KM * dd[a, b] / TRI_Q
    c[na:] = PENALTY
    rows, cols, vals = [], [], []
    for k, (a, b) in enumerate(arcs):
        rows.append(a); cols.append(k); vals.append(1.0)
    for a in range(S):
        rows.append(a); cols.append(na + a); vals.append(1.0)
    A_eq = coo_matrix((vals, (rows, cols)), shape=(S, nv)).tocsr()
    b_eq = np.array([resid[sink[a]] for a in range(S)])
    rows2 = [b for (a, b) in arcs]
    cols2 = list(range(na))
    vals2 = [1.0] * na
    tcoef = np.array([2 * dd[a, b] / (TRI_V * TRI_Q)
                      + 2 * TRI_HANDLE / 60.0 for (a, b) in arcs])
    rows3 = [nd] * na
    cols3 = list(range(na))
    A_ub = coo_matrix(
        (np.concatenate([vals2, tcoef]),
         (np.array(rows2 + rows3), np.array(cols2 + cols3))),
        shape=(nd + 1, nv)).tocsr()
    b_ub = np.concatenate([np.array([I[donor[b]] for b in range(nd)]), [n]])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=(0, None), method="highs")
    if not res.success:
        return None, 0.0, 0.0
    x = res.x
    flows = [(int(sink[a]), int(donor[b]), float(x[k]))
             for k, (a, b) in enumerate(arcs) if x[k] > 1e-9]
    km = float(sum(dd[a, b] * x[k] for k, (a, b) in enumerate(arcs)))
    return flows, km, km / TRI_Q


def dyn_sim(inp, plan, nvec, collect_trips=False, record=False):
    """Cell-level day with walking, optional tricycles, T018 zone totals."""
    dep = inp["dep_cell"]
    Pc = inp["Pcell"].astype(np.float64)
    zid = inp["zid"]
    xy = inp["xy"]
    Z = int(zid.max()) + 1
    C = len(zid)
    I = inp["I0c"].copy()
    Ipl = plan["I"]
    XM = plan["XM"]
    U = plan["U"]
    served_zone = inp["dep_zone"] - U
    arr_cell = np.einsum("hzc,zh->hc", Pc, served_zone)   # (24,C)

    nbrs = cKDTree(xy).query_ball_point(xy, WALK_KM)
    zcells = [np.where(zid == z)[0] for z in range(Z)]
    D = np.load(os.path.join(lib.DATA, "dist_matrix.npy"))

    unmet_h = np.zeros(24)
    strict_h = np.zeros(24)
    tri_bikes = np.zeros(Z)
    tri_bike_km = np.zeros(Z)
    tri_vehicle_km = np.zeros(Z)
    trips = []
    traj = np.zeros((24, C)) if record else None

    for h in range(24):
        for z in range(Z):
            idx = zcells[z]
            p = XM[h, z].sum()
            tot = I[idx].sum()
            if p > 1e-12 and tot > 1e-12:
                I[idx] *= max(0.0, 1.0 - p / tot)
        strict_h[h] = np.maximum(dep[h] - I, 0.0).sum()
        take = np.minimum(dep[h], I)
        I -= take
        resid = dep[h] - take
        walk = walk_borrow0(I, resid, nbrs)
        resid = np.maximum(resid - walk, 0.0)
        if nvec is not None and resid.sum() > 1e-9:
            for z in range(Z):
                n = nvec[z]
                if n <= 1e-9:
                    continue
                idx = zcells[z]
                sink = idx[resid[idx] > 1e-9]
                donor = idx[I[idx] > 1e-9]
                if len(sink) == 0 or len(donor) == 0:
                    continue
                flows, bkm, vkm = _tri_lp(sink, donor, resid, I, D, n)
                if not flows:
                    continue
                for (d, s, f) in flows:
                    I[s] -= f
                    I[d] += f
                    resid[d] -= f
                tri_bikes[z] += sum(f for (_, _, f) in flows)
                tri_bike_km[z] += bkm
                tri_vehicle_km[z] += vkm
                if collect_trips:
                    for (d, s, f) in flows:
                        trips.append(dict(hour=h, zone=int(z), donor=s,
                                          sink=d, bikes=f,
                                          km=float(D[s, d])))
        for z in range(Z):
            idx = zcells[z]
            d = XM[h, :, z].sum()
            if d > 1e-12:
                m = arr_cell[h, idx]
                if m.sum() <= 1e-12:
                    m = np.ones(len(idx))
                I[idx] += d * m / m.sum()
        I += arr_cell[h]
        unmet_h[h] = np.maximum(resid, 0.0).sum()
        if record:
            traj[h] = I
    return dict(unmet=float(unmet_h.sum()), strict=float(strict_h.sum()),
                unmet_by_hour=unmet_h, strict_by_hour=strict_h,
                tri_bikes=tri_bikes, tri_bike_km=tri_bike_km,
                tri_vehicle_km=tri_vehicle_km, trips=trips,
                traj=traj)


# --------------------------------------------------------------------------
def zone_arrays(inp, plan):
    """Per-zone cell arrays, with walking restricted to the same zone."""
    dep = inp["dep_cell"]
    Pc = inp["Pcell"].astype(np.float64)
    zid = inp["zid"]
    xy = inp["xy"]
    Z = int(zid.max()) + 1
    served_zone = inp["dep_zone"] - plan["U"]
    arr_cell = np.einsum("hzc,zh->hc", Pc, served_zone)   # (24,C)
    XM = plan["XM"]
    tree = cKDTree(xy)
    out = []
    for z in range(Z):
        idx = np.where(zid == z)[0]
        loc = {int(c): k for k, c in enumerate(idx)}
        nb = []
        for c in idx:
            nb.append(np.array([loc[int(j)] for j in tree.query_ball_point(xy[c], WALK_KM)
                                if int(j) in loc], dtype=int))
        out.append(dict(z=z, idx=idx, dep=dep[:, idx], arr=arr_cell[:, idx],
                        i0=inp["I0c"][idx], nbrs=nb,
                        pickup=np.array([XM[h, z].sum() for h in range(24)]),
                        drop=np.array([XM[h, :, z].sum() for h in range(24)])))
    return out


def dyn_zone(za, D, n, collect_trips=False):
    """One zone's day at a fixed tricycle count n."""
    dep, arr, I = za["dep"], za["arr"], za["i0"].copy()
    Cz = len(I)
    unmet = 0.0
    strict = 0.0
    tri_bikes = 0.0
    tri_bike_km = 0.0
    tri_vehicle_km = 0.0
    trips = []
    sub = D[np.ix_(za["idx"], za["idx"])]
    for h in range(24):
        p = za["pickup"][h]
        tot = I.sum()
        if p > 1e-12 and tot > 1e-12:
            I *= max(0.0, 1.0 - p / tot)
        strict += np.maximum(dep[h] - I, 0.0).sum()
        take = np.minimum(dep[h], I)
        I -= take
        resid = dep[h] - take
        walk = walk_borrow0(I, resid, za["nbrs"])
        resid = np.maximum(resid - walk, 0.0)
        if n > 1e-9 and TRI_HOUR0 <= h < TRI_HOUR1:
            sink = np.where(resid > 1e-9)[0]
            donor = np.where(I > 1e-9)[0]
            if len(sink) and len(donor):
                I_before = I.copy() if collect_trips else None
                resid_before = resid.copy() if collect_trips else None
                flows, bkm, vkm = _tri_lp_local(sink, donor, resid, I, sub, n)
                if flows:
                    for (d, s, f) in flows:
                        I[s] -= f
                        I[d] += f
                        resid[d] -= f
                        tri_bikes += f
                        if collect_trips:
                            trips.append(dict(hour=h, zone=int(za["z"]),
                                              donor=int(za["idx"][s]),
                                              sink=int(za["idx"][d]),
                                              bikes=float(f),
                                              avail=int(np.floor(
                                                  I_before[s])),
                                              room=int(np.floor(
                                                  resid_before[d])),
                                              km=float(sub[s, d])))
                    tri_bike_km += bkm
                    tri_vehicle_km += vkm
        d = za["drop"][h]
        if d > 1e-12:
            m = arr[h]
            I += d * (m / m.sum() if m.sum() > 1e-12 else 1.0 / Cz)
        I += arr[h]
        unmet += np.maximum(resid, 0.0).sum()
    return dict(unmet=float(unmet), strict=float(strict),
                tri_bikes=float(tri_bikes),
                tri_bike_km=float(tri_bike_km),
                tri_vehicle_km=float(tri_vehicle_km),
                trips=trips)


def _tri_lp_local(sink, donor, resid, I, sub, n):
    """Same LP as _tri_lp but on the zone-local distance submatrix."""
    dd = sub[np.ix_(sink, donor)]
    arcs = []
    for a in range(len(sink)):
        js = np.where(dd[a] <= TRI_MAX_TRIP)[0]
        if not len(js):
            continue
        js = js[np.argsort(dd[a, js])[:TRI_NEAREST]]
        arcs += [(a, int(b)) for b in js]
    if not arcs or n <= 1e-9:
        return None, 0.0
    S, nd, na = len(sink), len(donor), len(arcs)
    nv = na + S
    c = np.zeros(nv)
    for k, (a, b) in enumerate(arcs):
        c[k] = TRI_KM * dd[a, b] / TRI_Q
    c[na:] = PENALTY
    rows = [a for (a, b) in arcs] + list(range(S))
    cols = list(range(na)) + [na + a for a in range(S)]
    A_eq = coo_matrix((np.ones(na + S), (rows, cols)),
                      shape=(S, nv)).tocsr()
    b_eq = np.array([resid[sink[a]] for a in range(S)])
    rows2 = [b for (a, b) in arcs] + [nd] * na
    cols2 = list(range(na)) * 2
    tcoef = np.array([2 * dd[a, b] / (TRI_V * TRI_Q)
                      + 2 * TRI_HANDLE / 60.0 for (a, b) in arcs])
    A_ub = coo_matrix((np.concatenate([np.ones(na), tcoef]),
                       (rows2, cols2)), shape=(nd + 1, nv)).tocsr()
    b_ub = np.concatenate([[I[donor[b]] for b in range(nd)], [n]])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=(0, None), method="highs")
    if not res.success:
        return None, 0.0, 0.0
    x = res.x
    flows = [(int(sink[a]), int(donor[b]), float(x[k]))
             for k, (a, b) in enumerate(arcs) if x[k] > 1e-9]
    km = float(sum(dd[a, b] * x[k] for k, (a, b) in enumerate(arcs)))
    return flows, km, km / TRI_Q


def _dp_curves(curves, Z, nmax, ncap):
    """Min-cost allocation DP: dp[N] = min cost with exactly N tricycles.
    Vectorised over the fleet axis; choice[zi, N] stores the tricycles given
    to zone zi in the reconstruction of N."""
    dp = np.full(ncap + 1, np.inf)
    dp[0] = 0.0
    choice = np.zeros((Z, ncap + 1), dtype=np.int16)
    for zi in range(Z):
        c = curves[zi]
        ndp = np.full(ncap + 1, np.inf)
        for n in range(min(nmax, ncap) + 1):
            if n == 0:
                cand = dp + c[0]
                better = cand < ndp
                ndp[better] = cand[better]
                continue
            cand = dp[:ncap + 1 - n] + c[n]
            upto = ndp[n:]
            better = cand < upto
            upto[better] = cand[better]
            ch = choice[zi, n:]
            ch[better] = n
        dp = ndp
    return dp, choice


def _dp_recon(choice, N, Z):
    nz = np.zeros(Z, dtype=int)
    left = int(N)
    for zi in range(Z - 1, -1, -1):
        n = int(choice[zi, left])
        nz[zi] = n
        left -= n
    return nz


def run_fleet(force=False):
    path = os.path.join(RUN, "compare.json")
    if os.path.exists(path) and not force:
        print("reuse", path)
        return load_json("compare.json")
    inp, _ = prep()
    plan, scal = load_plan()
    t0 = time.time()
    D = np.load(os.path.join(lib.DATA, "dist_matrix.npy"))
    zas = zone_arrays(inp, plan)
    Z = len(zas)
    # per-zone cost curve C_z(n), n = 0..TRI_NCURVE.  The priced allocation
    # caps at TRI_NMAX=40; the extra points let the cap-60 check reuse them.
    curves = np.zeros((Z, TRI_NCURVE + 1))
    unmet_curve = np.zeros((Z, TRI_NCURVE + 1))
    km_curve = np.zeros((Z, TRI_NCURVE + 1))
    bike_km_curve = np.zeros((Z, TRI_NCURVE + 1))
    strict_curve = np.zeros((Z, TRI_NCURVE + 1))
    bikes_curve = np.zeros((Z, TRI_NCURVE + 1))
    base_by_zone = np.zeros(Z)
    for zi, za in enumerate(zas):
        for n in range(TRI_NCURVE + 1):
            r = dyn_zone(za, D, n)
            curves[zi, n] = (TRI_FIXED * n + TRI_KM * r["tri_vehicle_km"]
                             + PENALTY * r["unmet"])
            unmet_curve[zi, n] = r["unmet"]
            km_curve[zi, n] = r["tri_vehicle_km"]
            bike_km_curve[zi, n] = r["tri_bike_km"]
            strict_curve[zi, n] = r["strict"]
            bikes_curve[zi, n] = r["tri_bikes"]
        base_by_zone[zi] = unmet_curve[zi, 0]
        print(f"zone {zi}: unmet n=0 {unmet_curve[zi,0]:.1f} -> "
              f"n={TRI_NMAX} {unmet_curve[zi,TRI_NMAX]:.1f}  "
              f"n={TRI_NCURVE} {unmet_curve[zi,TRI_NCURVE]:.1f}  "
              f"t={time.time()-t0:.0f}s")
    # integer allocation by DP: minimise sum C_z(n_z), sum n_z <= N
    NCAP = max(FLEETS)
    dp, choice = _dp_curves(curves, Z, TRI_NMAX, NCAP)
    alloc = {}
    for N in FLEETS:
        u = int(np.argmin(dp[:N + 1]))
        alloc[N] = _dp_recon(choice, u, Z)
    best_used = int(np.argmin(dp))
    best_cost = float(dp[best_used])
    rows = []
    base_total = float(scal["obj"]) + PENALTY * float(base_by_zone.sum())
    for N in FLEETS:
        nz = alloc[N]
        unmet = float(sum(unmet_curve[zi, nz[zi]] for zi in range(Z)))
        km = float(sum(km_curve[zi, nz[zi]] for zi in range(Z)))
        bike_km = float(sum(bike_km_curve[zi, nz[zi]] for zi in range(Z)))
        tri_cost = float(TRI_FIXED * nz.sum() + TRI_KM * km)
        total = float(scal["obj"]) + tri_cost + PENALTY * unmet
        rows.append(dict(
            fleet_N=int(N), tricycles=int(nz.sum()),
            zones_with_tricycles=int((nz > 0).sum()),
            max_zone_tricycles=int(nz.max()) if Z else 0,
            cell_unmet=unmet, pre_tricycle_shortfall=float(
                sum(strict_curve[zi, nz[zi]] for zi in range(Z))),
            tricycle_vehicle_km=km, tricycle_bike_km=bike_km,
            tricycle_bikes=float(
                sum(bikes_curve[zi, nz[zi]] for zi in range(Z))),
            tricycle_cost=tri_cost, truck_obj=float(scal["obj"]),
            total=total, saving_vs_no_tricycle=float(base_total - total),
            marginal_saving=None))
    for k in range(1, len(rows)):
        rows[k]["marginal_saving"] = float(
            rows[k - 1]["total"] - rows[k]["total"])
    base_unmet = float(base_by_zone.sum())
    # penalty sensitivity: re-allocate at each penalty p -----------------
    pen_rows = []
    for p in (1.0, 2.0, 3.0, 4.0, 6.0, 10.0):
        cp = (TRI_FIXED * np.arange(TRI_NCURVE + 1)[None, :]
              + TRI_KM * km_curve + p * unmet_curve)
        dpp, chp = _dp_curves(cp, Z, TRI_NMAX, NCAP)
        N = int(np.argmin(dpp))
        nz = _dp_recon(chp, N, Z)
        unmet = float(sum(unmet_curve[zi, nz[zi]] for zi in range(Z)))
        km = float(sum(km_curve[zi, nz[zi]] for zi in range(Z)))
        tri = float(TRI_FIXED * N + TRI_KM * km)
        removed = base_unmet - unmet
        pen_rows.append(dict(
            penalty=p, best_used=N, tricycles=int(nz.sum()),
            nmax_used=int(TRI_NMAX),
            zones_at_nmax=int((nz >= TRI_NMAX).sum()),
            cell_unmet=unmet, removed_unmet=removed,
            tricycle_cost=tri, vehicle_km=km,
            penalty_cost=float(p * unmet), total=float(tri + p * unmet),
            threshold=float(tri / removed) if removed > 1e-9 else None))
    thr = []
    for N in FLEETS:
        nz = alloc[N]
        unmet = float(sum(unmet_curve[zi, nz[zi]] for zi in range(Z)))
        km = float(sum(km_curve[zi, nz[zi]] for zi in range(Z)))
        tri = float(TRI_FIXED * N + TRI_KM * km)
        removed = base_unmet - unmet
        if removed > 1e-9:
            thr.append(dict(fleet_N=int(N), tricycle_cost=tri,
                            removed_unmet=removed,
                            threshold=float(tri / removed)))
    gmin = min(thr, key=lambda r: r["threshold"]) if thr else None
    nmax_sweep = []
    for nmx in (20, 25, 30, 40, 60):
        dpn, chn = _dp_curves(curves[:, :nmx + 1], Z, nmx, NCAP)
        Nn = int(np.argmin(dpn))
        nzn = _dp_recon(chn, Nn, Z)
        un = float(sum(unmet_curve[zi, nzn[zi]] for zi in range(Z)))
        kn = float(sum(km_curve[zi, nzn[zi]] for zi in range(Z)))
        trn = float(TRI_FIXED * Nn + TRI_KM * kn)
        nmax_sweep.append(dict(
            nmax=nmx, best_used=Nn, cell_unmet=un, vehicle_km=kn,
            tricycle_cost=trn,
            total=float(scal["obj"]) + trn + PENALTY * un,
            zones_at_cap=int((nzn >= nmx).sum())))
    best_row = max(rows, key=lambda r: r["saving_vs_no_tricycle"])
    out = dict(
        K=int(scal["K"]), zone_unmet=float(scal["unmet"]),
        truck_obj=float(scal["obj"]), truck_km=float(scal["truck_km"]),
        truck_moved=float(scal["moved"]), trucks=int(scal["K"]),
        no_tricycle=dict(cell_unmet=float(base_by_zone.sum()),
                         total=base_total),
        best_used=int(best_used), best_layer_cost=float(best_cost),
        best_allocation=[int(v) for v in _dp_recon(choice, best_used, Z)],
        rows=rows,
        penalty_scan=dict(
            rows=pen_rows,
            global_threshold=gmin,
            nmax_sweep=nmax_sweep,
            note=("N*(p) re-optimises the per-zone curves at each penalty p "
                  "with the same integer DP; global_threshold is the "
                  "smallest tricycle cost per removed unmet trip among the "
                  "fleet-scan rows, i.e. the lowest penalty at which any "
                  "scale breaks even; nmax_sweep re-solves the allocation "
                  "with the per-zone upper bound capped at 20/25/30/40/60. "
                  "The priced cap is 60; rows report zones_at_nmax so any "
                  "row still truncated by the cap is visible (p=10 keeps "
                  "zones at the cap and is therefore a lower bound).")),
        cell_unmet_interval=dict(
            static_lower=float(load_json("cell_unmet.json")["static"]["total_strict"])
            if os.path.exists(os.path.join(RUN, "cell_unmet.json")) else None,
            static_share=float(load_json("cell_unmet.json")["static"]
                               ["strict_share_of_trips"])
            if os.path.exists(os.path.join(RUN, "cell_unmet.json")) else None,
            dynamic_upper=float(base_unmet),
            dynamic_share=float(base_unmet / 2132542.1875),
            note=("the static hourly arrival-surface split is a lower bound "
                  "(bikes are re-split for free every hour); the dynamic "
                  "no-tricycle replay is the operational upper bound. "
                  "Walking across cell boundaries is not modelled at 300 m "
                  "because no neighbour centre is within reach on this grid")),
        curves=dict(zones=[int(z) for z in range(Z)],
                    cost=curves.tolist(),
                    unmet=unmet_curve.tolist(),
                    km=km_curve.tolist(),
                    bike_km=bike_km_curve.tolist()),
        allocation={str(N): [int(v) for v in alloc[N]] for N in FLEETS},
        params=dict(Q=TRI_Q, V=TRI_V, handle_min=TRI_HANDLE,
                    fixed=TRI_FIXED, km=TRI_KM, max_trip=TRI_MAX_TRIP,
                    nearest=TRI_NEAREST, nmax=TRI_NMAX, ncurve=TRI_NCURVE),
        runtime_s=float(time.time() - t0),
        note=("Tier 1 (T018) is identical in every row; the comparison is "
              "the tricycle layer plus the cell-level unmet penalty. Walking "
              "is restricted to same-zone neighbours; the share of 300 m "
              "pairs that cross a zone boundary is reported in check.json."))
    dump(out, "compare.json")
    return out


# --------------------------------------------------------------------------
def run_cellunmet(force=False):
    path = os.path.join(RUN, "cell_unmet.json")
    if os.path.exists(path) and not force:
        print("reuse", path)
        return load_json("cell_unmet.json")
    inp, _ = prep()
    plan, scal = load_plan()
    st = static_cellunmet(inp, plan)
    out = dict(
        task="T024 cell-level unmet under the T018 recommended plan",
        K=int(scal["K"]), zone_unmet=float(scal["unmet"]),
        zone_unmet_saved=float(plan["U"].sum()),
        static=st,
        note=("static disaggregation of the T018 plan: each zone's hour-start "
              "stock is split across its cells by the hour's expected arrival "
              "surface. It is an upper bound on cell-level mismatch because "
              "bikes are re-split every hour for free; the dynamic model in "
              "dyn_baseline.json is the operational number."))
    dump(out, "cell_unmet.json")
    return out


def run_dyn(force=False):
    path = os.path.join(RUN, "dyn_baseline.json")
    if os.path.exists(path) and not force:
        print("reuse", path)
        return load_json("dyn_baseline.json")
    inp, _ = prep()
    plan, scal = load_plan()
    t0 = time.time()
    D = np.load(os.path.join(lib.DATA, "dist_matrix.npy"))
    zas = zone_arrays(inp, plan)
    unmet = strict = 0.0
    for za in zas:
        r = dyn_zone(za, D, 0)
        unmet += r["unmet"]
        strict += r["strict"]
    out = dict(K=int(scal["K"]), zone_unmet=float(scal["unmet"]),
               cell_unmet=float(unmet), strict_unmet=float(strict),
               trips=float(inp["dep_cell"].sum()),
               runtime_s=float(time.time() - t0),
               note=("dynamic cell model with no tricycles: own service, "
                     "CellSim 300 m walking (same zone), T018 truck flows at "
                     "zone level; zone totals follow T018 exactly."))
    dump(out, "dyn_baseline.json")
    return out


# --------------------------------------------------------------------------
def _int_trips(za, D, n_lp):
    """Integer trip list from the LP solved at n_lp tricycles.

    Each LP arc flow is rounded to whole bicycles under the donor's integer
    stock, split into loads of at most Q, and zero-load flows are merged away.
    The trip list is fixed from here on: the fleet-size scenarios below only
    differ in how many tricycles execute the same trips.
    """
    r = dyn_zone(za, D, n_lp, collect_trips=True)
    budget = {}
    for t in r["trips"]:
        key = (t["hour"], t["donor"])
        if key not in budget:
            budget[key] = t["avail"]
    room = {}
    for t in r["trips"]:
        key = (t["hour"], t["sink"])
        if key not in room:
            room[key] = t["room"]
    flows_bikes = float(sum(t["bikes"] for t in r["trips"]))
    dropped_flows = 0
    dropped_bikes = 0.0
    trips = []
    q = int(round(TRI_Q))
    for t in r["trips"]:
        key = (t["hour"], t["donor"])
        n_int = int(round(t["bikes"]))
        n_int = min(n_int, max(0, budget[key]))
        budget[key] -= n_int
        if n_int <= 0:
            dropped_flows += 1
            dropped_bikes += t["bikes"]
            continue
        n_full, rem = divmod(n_int, q)
        for _ in range(n_full):
            trips.append(dict(hour=t["hour"], donor=t["donor"],
                              sink=t["sink"], bikes=q, km=t["km"]))
        if rem:
            trips.append(dict(hour=t["hour"], donor=t["donor"],
                              sink=t["sink"], bikes=rem, km=t["km"]))
    int_bikes = float(sum(t["bikes"] for t in trips))
    deliv = {}
    for t in trips:
        key = (t["hour"], t["sink"])
        deliv[key] = deliv.get(key, 0) + t["bikes"]
    sink_excess = max((deliv[k] - room.get(k, 0) for k in deliv), default=0)
    return dict(zone=int(za["z"]), n_lp=int(n_lp), r=r, trips=trips,
                flows_bikes=flows_bikes, dropped_flows=int(dropped_flows),
                dropped_bikes=float(dropped_bikes), int_bikes=int_bikes,
                sink_excess_bikes=float(sink_excess),
                lp_vehicle_km=float(r["tri_vehicle_km"]),
                lp_bike_km=float(r["tri_bike_km"]))


def _assign(trips, n, D):
    """Greedy earliest-finish assignment of a fixed trip list to n tricycles."""
    order = sorted(trips, key=lambda t: (t["hour"], t["donor"]))
    free = [0.0] * n
    pos = [None] * n
    sched = []
    unassigned = 0
    for t in order:
        best_k, best_finish, best_start, best_dh = -1, None, 0.0, 0.0
        for k in range(n):
            dh = 0.0 if pos[k] is None else D[pos[k], t["donor"]] / TRI_V
            start = max(free[k] + dh, t["hour"], TRI_HOUR0)
            finish = (start + 2 * t["km"] / TRI_V
                      + 2 * t["bikes"] * TRI_HANDLE / 60.0)
            if finish <= TRI_HOUR1 and (best_finish is None
                                        or finish < best_finish):
                best_k, best_finish, best_start, best_dh = (
                    k, finish, start, dh)
        if best_k < 0:
            unassigned += 1
            continue
        sched.append(dict(tricycle=best_k + 1, hour=int(t["hour"]),
                          donor=int(t["donor"]), sink=int(t["sink"]),
                          bikes=float(t["bikes"]), km=float(t["km"]),
                          deadhead_km=float(best_dh),
                          start_h=float(best_start),
                          end_h=float(best_finish)))
        free[best_k] = best_finish
        pos[best_k] = t["sink"]
    by_tri = {}
    for s in sched:
        by_tri.setdefault(s["tricycle"], []).append(s)
    return dict(n=int(n), sched=sched, unassigned=int(unassigned),
                scheduled=int(len(sched)),
                max_load=max((s["bikes"] for s in sched), default=0.0),
                max_end_h=max((s["end_h"] for s in sched), default=0.0),
                max_deadhead_km=max((s["deadhead_km"] for s in sched),
                                    default=0.0),
                vehicle_km=float(sum(s["km"] for s in sched)),
                bike_km=float(sum(s["km"] * s["bikes"] for s in sched)),
                per_tricycle={str(k): dict(
                    trips=len(v),
                    bikes=int(sum(s["bikes"] for s in v)),
                    km=float(sum(s["km"] for s in v)),
                    end_h=float(v[-1]["end_h"])) for k, v in by_tri.items()})


def _case(built, n, D, name):
    a = _assign(built["trips"], n, D)
    return dict(
        name=name, tricycles_in_zone=int(n),
        dispatch_buffer=int(n - built["n_lp"]),
        trips=int(len(built["trips"])),
        scheduled=int(a["scheduled"]), unassigned=int(a["unassigned"]),
        max_load=float(a["max_load"]),
        max_load_ok=bool(a["max_load"] <= TRI_Q + 1e-9),
        loads_integer=bool(all(float(s["bikes"]).is_integer()
                               for s in a["sched"])),
        max_end_h=float(a["max_end_h"]),
        shift_ok=bool(a["max_end_h"] <= TRI_HOUR1 + 1e-9),
        max_deadhead_km=float(a["max_deadhead_km"]),
        vehicle_km=float(a["vehicle_km"]), bike_km=float(a["bike_km"]),
        per_tricycle=a["per_tricycle"], sched=a["sched"])


def _min_fleet(built, D, lo, hi):
    """Smallest fleet that schedules a fixed trip list; binary search.

    Monotonicity in the fleet size is assumed for the greedy rule and then
    verified (hi-1 must fail, hi must pass).  Returns None if hi fails.
    """
    def ok(k):
        return _assign(built["trips"], k, D)["unassigned"] == 0
    if not ok(hi):
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid + 1
    return int(lo)


def _exec_n(za, n0, D, cap=400):
    """Fixed point: LP at n, then raise n to the fleet its own trip list needs."""
    n = int(n0)
    b = _int_trips(za, D, n)
    for it in range(10):
        hi = min(cap, max(n + 10, 2 * n))
        k = _min_fleet(b, D, max(1, n - 5), hi)
        while k is None and hi < cap:
            hi = min(cap, 2 * hi)
            k = _min_fleet(b, D, max(1, n - 5), hi)
        if k is None:
            return dict(n_priced=int(n0), n_exec=None, feasible=False,
                        iterations=it, built=b)
        if k <= n:
            return dict(n_priced=int(n0), n_exec=int(n), feasible=True,
                        iterations=it, built=b, min_fleet_for_last=k)
        n = int(k)
        b = _int_trips(za, D, n)
    return dict(n_priced=int(n0), n_exec=None, feasible=False,
                iterations=10, built=b)


def run_schedule(force=False):
    path = os.path.join(RUN, "schedule_check.json")
    if os.path.exists(path) and not force:
        print("reuse", path)
        return load_json("schedule_check.json")
    cmp = load_json("compare.json")
    best = max(cmp["rows"], key=lambda r: r["saving_vs_no_tricycle"])
    N = int(best["fleet_N"])
    nz = np.array(cmp["allocation"][str(N)])
    zmax = int(np.argmax(nz))
    n_alloc = int(nz[zmax])
    inp, _ = prep()
    plan, _ = load_plan()
    D = np.load(os.path.join(lib.DATA, "dist_matrix.npy"))
    zas_all = zone_arrays(inp, plan)
    za = zas_all[zmax]
    built = _int_trips(za, D, n_alloc)

    n_buffer = int(np.ceil(1.1 * n_alloc))
    case_plan = _case(built, n_alloc, D, "alloc_no_buffer")
    case_buf = _case(built, n_buffer, D, "alloc_buffer10")
    # minimum fleet that executes the same integer trip list within the shift
    min_feasible = None
    scan = []
    for k in range(1, min(80, n_alloc + 25) + 1):
        a = _assign(built["trips"], k, D)
        scan.append(dict(tricycles=k, unassigned=int(a["unassigned"])))
        if a["unassigned"] == 0:
            min_feasible = k
            break

    # city-wide executability at every zone's priced allocation: the sample
    # zone is the largest, but the claim "no buffer is priced" needs all 50
    built_zones = []
    for zi, za_i in enumerate(zas_all):
        n_z = int(nz[zi])
        built_zones.append(
            (za_i, n_z, _int_trips(za_i, D, n_z) if n_z > 0 else None))
    city = []
    for za_i, n_z, b in built_zones:
        if n_z <= 0:
            city.append(dict(zone=int(za_i["z"]), tricycles=0, trips=0,
                             unassigned=0, vehicle_km=0.0, deadhead_km=0.0,
                             bike_km=0.0, lp_vehicle_km=0.0))
            continue
        a = _assign(b["trips"], n_z, D)
        city.append(dict(
            zone=int(za_i["z"]), tricycles=int(n_z),
            trips=int(len(b["trips"])), unassigned=int(a["unassigned"]),
            vehicle_km=float(a["vehicle_km"]),
            deadhead_km=float(sum(s["deadhead_km"] for s in a["sched"])),
            bike_km=float(a["bike_km"]),
            lp_vehicle_km=float(b["lp_vehicle_km"])))
    city_tot = dict(
        tricycles=int(sum(c["tricycles"] for c in city)),
        trips=int(sum(c["trips"] for c in city)),
        unassigned=int(sum(c["unassigned"] for c in city)),
        zones_with_unassigned=int(sum(1 for c in city
                                      if c["unassigned"] > 0)),
        vehicle_km=float(sum(c["vehicle_km"] for c in city)),
        deadhead_km=float(sum(c["deadhead_km"] for c in city)),
        bike_km=float(sum(c["bike_km"] for c in city)),
        lp_vehicle_km=float(sum(c["lp_vehicle_km"] for c in city)))
    city_tot["actual_mileage_cost"] = float(
        TRI_KM * (city_tot["vehicle_km"] + city_tot["deadhead_km"]))
    city_tot["fullload_mileage_cost"] = float(
        TRI_KM * city_tot["lp_vehicle_km"])
    # how much dispatch buffer does the same trip list need city-wide?
    buffer_sweep = []
    for fac in (1.0, 1.05, 1.10, 1.15, 1.20):
        un = 0
        zones_un = 0
        n_tot = 0
        for za_i, n_z, b in built_zones:
            if n_z <= 0:
                continue
            k_z = int(np.ceil(fac * n_z))
            a = _assign(b["trips"], k_z, D)
            un += int(a["unassigned"])
            zones_un += int(a["unassigned"] > 0)
            n_tot += k_z
        buffer_sweep.append(dict(factor=float(fac), tricycles=int(n_tot),
                                 unassigned=int(un),
                                 zones_with_unassigned=int(zones_un)))
    # fixed-point executable variant: for each zone raise the fleet until its
    # own LP trip list is schedulable, then price that fleet.  This uses the
    # same greedy rule as the schedule check and the actual achieved loads.
    exec_rows = []
    for za_i, n_z, b in built_zones:
        if n_z <= 0:
            continue
        e = _exec_n(za_i, n_z, D)
        bf = e["built"]
        a = _assign(bf["trips"], e["n_exec"], D) if e["feasible"] else None
        exec_rows.append(dict(
            zone=int(za_i["z"]), n_priced=int(n_z),
            n_exec=e["n_exec"], feasible=bool(e["feasible"]),
            iterations=int(e["iterations"]),
            trips=int(len(bf["trips"])),
            unassigned=int(a["unassigned"]) if a else None,
            unmet_lp=float(bf["r"]["unmet"]),
            int_short=float(bf["flows_bikes"] - bf["int_bikes"]),
            unmet=float(bf["r"]["unmet"] + bf["flows_bikes"]
                        - bf["int_bikes"]),
            lp_vehicle_km=float(bf["lp_vehicle_km"]),
            vehicle_km=float(a["vehicle_km"]) if a else None,
            deadhead_km=float(sum(s["deadhead_km"] for s in a["sched"]))
            if a else None,
            bike_km=float(a["bike_km"]) if a else None))
    exec_tot = dict(
        n_exec=int(sum(r["n_exec"] or 0 for r in exec_rows)),
        feasible=bool(all(r["feasible"] for r in exec_rows)),
        zones_raised=int(sum(1 for r in exec_rows
                             if r["n_exec"] is not None
                             and r["n_exec"] > r["n_priced"])),
        unmet_lp=float(sum(r["unmet_lp"] for r in exec_rows)),
        int_short=float(sum(r["int_short"] for r in exec_rows)),
        unmet=float(sum(r["unmet"] for r in exec_rows)),
        lp_vehicle_km=float(sum(r["lp_vehicle_km"] for r in exec_rows)),
        vehicle_km=float(sum(r["vehicle_km"] or 0 for r in exec_rows)),
        deadhead_km=float(sum(r["deadhead_km"] or 0 for r in exec_rows)),
        bike_km=float(sum(r["bike_km"] or 0 for r in exec_rows)))
    exec_tot["layer_cost_fullload"] = float(
        TRI_FIXED * exec_tot["n_exec"]
        + TRI_KM * exec_tot["lp_vehicle_km"]
        + PENALTY * exec_tot["unmet"])
    exec_tot["layer_cost_actual"] = float(
        TRI_FIXED * exec_tot["n_exec"]
        + TRI_KM * (exec_tot["vehicle_km"] + exec_tot["deadhead_km"])
        + PENALTY * exec_tot["unmet"])
    exec_tot["total_fullload"] = float(float(cmp["truck_obj"])
                                       + exec_tot["layer_cost_fullload"])
    exec_tot["total_actual"] = float(float(cmp["truck_obj"])
                                     + exec_tot["layer_cost_actual"])
    exec_tot["no_tricycle_total"] = float(
        float(cmp["truck_obj"])
        + PENALTY * float(cmp["no_tricycle"]["cell_unmet"]))
    exec_tot["saving_fullload"] = float(
        exec_tot["no_tricycle_total"] - exec_tot["total_fullload"])
    exec_tot["saving_actual"] = float(
        exec_tot["no_tricycle_total"] - exec_tot["total_actual"])
    exec_tot["saving_actual_share"] = float(
        exec_tot["saving_actual"] / exec_tot["no_tricycle_total"])
    exec_tot["saving_fullload_share"] = float(
        exec_tot["saving_fullload"] / exec_tot["no_tricycle_total"])
    exec_tot["removed_unmet"] = float(
        float(cmp["no_tricycle"]["cell_unmet"]) - exec_tot["unmet"])
    exec_tot["penalty_ceiling_exec"] = float(
        (TRI_FIXED * exec_tot["n_exec"]
         + TRI_KM * exec_tot["lp_vehicle_km"])
        / max(exec_tot["removed_unmet"], 1e-9))
    exec_tot["fixed_ceiling_exec"] = float(
        (PENALTY * exec_tot["removed_unmet"]
         - TRI_KM * exec_tot["lp_vehicle_km"])
        / max(exec_tot["n_exec"], 1))

    import csv
    for cname, c in (("schedule_sample.csv", case_plan),
                     ("schedule_sample_buffer.csv", case_buf)):
        with open(os.path.join(RUN, cname), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(c["sched"][0].keys())
                               if c["sched"] else ["tricycle"])
            w.writeheader()
            w.writerows(c["sched"])

    def public(c):
        out = {k: v for k, v in c.items() if k != "sched"}
        return out

    checks = dict(
        zone=int(za["z"]), fleet_N=N,
        tricycles_allocated=n_alloc, tricycles_in_zone=n_alloc,
        dispatch_buffer=0,
        lp_solved_at_n=int(built["n_lp"]),
        lp_vehicle_km=float(built["lp_vehicle_km"]),
        lp_bike_km=float(built["lp_bike_km"]),
        bike_km_matches_curve=bool(abs(
            built["lp_bike_km"]
            - float(cmp["curves"]["bike_km"][za["z"]][n_alloc])) < 1e-6),
        trips=int(len(built["trips"])), scheduled=int(case_plan["scheduled"]),
        unassigned=int(case_plan["unassigned"]),
        max_load=float(case_plan["max_load"]),
        max_load_ok=bool(case_plan["max_load_ok"]),
        loads_integer=bool(case_plan["loads_integer"]),
        max_end_h=float(case_plan["max_end_h"]),
        shift_ok=bool(case_plan["shift_ok"]),
        max_deadhead_km=float(case_plan["max_deadhead_km"]),
        vehicle_km=float(case_plan["vehicle_km"]),
        bike_km=float(case_plan["bike_km"]),
        lp_bikes=float(built["flows_bikes"]),
        int_bikes=float(built["int_bikes"]),
        dropped_flows=int(built["dropped_flows"]),
        dropped_bikes=float(built["dropped_bikes"]),
        sink_excess_bikes=float(built["sink_excess_bikes"]),
        rounding_bound=0.5 * len(built["r"]["trips"]),
        rounding_loss=float(built["flows_bikes"] - built["int_bikes"]),
        rounding_share=float((built["flows_bikes"] - built["int_bikes"])
                             / max(built["flows_bikes"], 1e-9)),
        lp_unmet_zone=float(built["r"]["unmet"]),
        int_unmet_zone=float(built["r"]["unmet"] + built["flows_bikes"]
                             - built["int_bikes"]),
        min_feasible_tricycles=min_feasible,
        min_feasible_scan=scan,
        city=city,
        city_totals=city_tot,
        buffer_sweep=buffer_sweep,
        exec_rows=exec_rows,
        exec_totals=exec_tot,
        plan=public(case_plan), buffer=public(case_buf),
        per_tricycle={str(k): dict(trips=len(v),
                                   bikes=int(sum(s["bikes"] for s in v)),
                                   km=float(sum(s["km"] for s in v)),
                                   end_h=float(v[-1]["end_h"]))
                      for k, v in
                      ((k, [s for s in case_plan["sched"]
                            if s["tricycle"] == k])
                       for k in sorted({s["tricycle"]
                                        for s in case_plan["sched"]}))},
        note=("one zone's complete day at the fleet that maximises the "
              "two-tier saving. The LP is solved once at the allocated "
              "n (the quantity the cost curve prices); its arc flows are "
              "rounded to whole bicycles under the donor's integer stock and "
              "split into loads of at most Q. plan assigns exactly n_alloc "
              "tricycles (no buffer); buffer adds 10%; min_feasible_tricycles "
              "is the smallest fleet, under the same greedy earliest-finish "
              "rule, that schedules every trip of that same list. city "
              "repeats the integer schedule for all 50 zones at their priced "
              "allocations, so city_totals.unassigned is the city-wide "
              "executability evidence and city_totals.actual_mileage_cost "
              "prices loaded travel plus deadhead at the achieved loads, "
              "against fullload_mileage_cost which is the 0.2*vehicle_km "
              "figure used in compare.json. vehicle_km is loaded tricycle "
              "travel; bike_km is the bicycle-kilometre work done."))
    dump(checks, "schedule_check.json")
    return checks


# --------------------------------------------------------------------------
def run_params(force=False):
    path = os.path.join(RUN, "vehicle_params.json")
    cmp = load_json("compare.json") if os.path.exists(
        os.path.join(RUN, "compare.json")) else None
    base = dict(
        task="T024 tricycle parameters", date="2026-11-18",
        status="planning assumptions (web lookup unavailable at run time)",
        params=[
            dict(name="capacity_bikes", value=TRI_Q, unit="bikes/trip",
                 range=[12, 30], source="planning assumption",
                 rationale=("a cargo tricycle carries 200-500 kg; a dockless "
                            "bike is ~16 kg, so 12-30 bikes; 20 is the "
                            "midpoint of the plausible range")),
            dict(name="speed", value=TRI_V, unit="km/h",
                 range=[10, 20], source="planning assumption",
                 rationale="loaded electric cargo tricycle in city traffic"),
            dict(name="handling", value=TRI_HANDLE, unit="min/bike",
                 range=[0.3, 1.0], source="planning assumption",
                 rationale="one driver loads/unloads each bike"),
            dict(name="fixed_cost", value=TRI_FIXED, unit="CNY/tricycle-day",
                 range=[200, 500], source="planning assumption",
                 rationale="vehicle rental + driver wage + battery for one day"),
            dict(name="distance_cost", value=TRI_KM, unit="CNY/vehicle-km",
                 range=[0.10, 0.40], source="planning assumption",
                 rationale="electricity and maintenance"),
            dict(name="max_trip_km", value=TRI_MAX_TRIP, unit="km",
                 range=[3, 8], source="planning assumption",
                 rationale="intra-zone redistribution trips are 1-4 km")],
        note=("No public source was reachable in this run; every parameter is "
              "labelled a planning assumption with a range, as the task "
              "allows. The break-even threshold below shows how far the "
              "conclusion moves inside those ranges."))
    if cmp:
        rows = cmp["rows"]
        best = max(rows, key=lambda r: r["saving_vs_no_tricycle"])
        removed = cmp["no_tricycle"]["cell_unmet"] - best["cell_unmet"]
        base["break_even"] = dict(
            best_fleet=int(best["fleet_N"]),
            fixed_cost_ceiling=float(
                (PENALTY * removed - TRI_KM * best["tricycle_vehicle_km"])
                / max(best["tricycles"], 1)),
            penalty_ceiling=float(
                (TRI_FIXED * best["tricycles"]
                 + TRI_KM * best["tricycle_vehicle_km"])
                / max(removed, 1e-12)),
            note=("the layer pays for itself iff the fixed cost per "
                  "tricycle-day is below fixed_cost_ceiling or the value of "
                  "one unserved trip is above penalty_ceiling: "
                  "cost = fixed*N + km_cost*km < p*(unmet0-unmet)"))
    dump(base, "vehicle_params.json")
    return base


def run_check(force=False):
    path = os.path.join(RUN, "check.json")
    if os.path.exists(path) and not force:
        print("reuse", path)
        return load_json("check.json")
    inp, meta = prep()
    cmp = load_json("compare.json")
    sch = load_json("schedule_check.json")
    dep_c, dep_z = float(inp["dep_cell"].sum()), float(inp["dep_zone"].sum())
    rows = cmp["rows"]
    cost_ok = all(abs(r["total"] - (r["truck_obj"] + r["tricycle_cost"]
                                    + PENALTY * r["cell_unmet"])) < 1e-6
                  for r in rows)
    zid = inp["zid"]
    tree = cKDTree(inp["xy"])
    nb = tree.query_ball_point(inp["xy"], WALK_KM)
    tot = sum(len(v) for v in nb) - len(zid)
    cross = sum(1 for c, v in enumerate(nb) for j in v
                if j != c and zid[j] != zid[c])
    nmax = int(cmp["params"]["nmax"])
    best_alloc = np.array(cmp["best_allocation"])
    sw = {r["nmax"]: r for r in cmp["penalty_scan"]["nmax_sweep"]}
    cap60 = sw.get(int(cmp["params"].get("ncurve", 60)))
    city = sch.get("city_totals", {})
    ex = sch.get("exec_totals", {})
    out = dict(
        pcell_zone_match_max_err=meta["pcell_zone_match_max_err"],
        pcell_row_sum_max_err=meta["pcell_row_sum_max_err"],
        dep_cell_over_dep_zone=float(dep_c / dep_z),
        cost_identity_ok=bool(cost_ok),
        schedule_unassigned=int(sch["unassigned"]),
        schedule_plan_unassigned=int(sch["plan"]["unassigned"]),
        schedule_buffer_unassigned=int(sch["buffer"]["unassigned"]),
        schedule_min_feasible=sch["min_feasible_tricycles"],
        schedule_bike_km_matches_curve=bool(
            sch.get("bike_km_matches_curve", False)),
        schedule_city_unassigned=int(city.get("unassigned", -1)),
        schedule_city_zones_with_unassigned=int(
            city.get("zones_with_unassigned", -1)),
        schedule_exec_feasible=bool(ex.get("feasible", False)),
        schedule_exec_n=int(ex.get("n_exec", -1)),
        schedule_exec_unmet_lp=float(ex.get("unmet_lp", 0.0)),
        schedule_exec_int_short=float(ex.get("int_short", 0.0)),
        schedule_exec_unmet_int=float(ex.get("unmet", 0.0)),
        schedule_exec_total_fullload=float(ex.get("total_fullload", 0.0)),
        schedule_exec_saving_fullload=float(ex.get("saving_fullload", 0.0)),
        schedule_exec_saving_share=float(
            ex.get("saving_fullload_share", 0.0)),
        schedule_exec_penalty_ceiling=float(
            ex.get("penalty_ceiling_exec", 0.0)),
        schedule_exec_fixed_ceiling=float(ex.get("fixed_ceiling_exec", 0.0)),
        schedule_lp_unmet_zone=float(sch.get("lp_unmet_zone", 0.0)),
        schedule_int_unmet_zone=float(sch.get("int_unmet_zone", 0.0)),
        schedule_city_priced_actual_mileage_cost=float(
            city.get("actual_mileage_cost", 0.0)),
        schedule_city_priced_fullload_mileage_cost=float(
            city.get("fullload_mileage_cost", 0.0)),
        schedule_exec_actual_mileage_cost=float(
            TRI_KM * (ex.get("vehicle_km", 0.0)
                      + ex.get("deadhead_km", 0.0))),
        schedule_exec_fullload_mileage_cost=float(
            TRI_KM * ex.get("lp_vehicle_km", 0.0)),
        schedule_load_ok=bool(sch["max_load_ok"]),
        schedule_loads_integer=bool(sch.get("loads_integer", False)),
        schedule_shift_ok=bool(sch["shift_ok"]),
        walk_cross_zone_share=float(cross / max(tot, 1)),
        nmax=nmax,
        best_used=int(cmp["best_used"]),
        zones_at_nmax=int((best_alloc >= nmax).sum()),
        cap60_best_used=int(cap60["best_used"]) if cap60 else None,
        cap60_not_binding=bool(cap60 is not None
                               and cap60["best_used"] == cmp["best_used"]),
        all_pass=bool(meta["pcell_zone_match_max_err"] < 1e-6
                      and meta["pcell_row_sum_max_err"] < 1e-6
                      and cost_ok and sch["unassigned"] == 0
                      and sch["max_load_ok"] and sch["shift_ok"]
                      and sch.get("loads_integer", False)
                      and sch.get("bike_km_matches_curve", False)
                      and ex.get("feasible", False)))
    dump(out, "check.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=("prep", "cellunmet", "dyn", "fleet", "schedule",
                             "params", "check", "all"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.stage in ("prep", "all"):
        prep(force=args.force)
    if args.stage in ("cellunmet", "all"):
        run_cellunmet(force=args.force)
    if args.stage in ("dyn", "all"):
        run_dyn(force=args.force)
    if args.stage in ("fleet", "all"):
        run_fleet(force=args.force)
    if args.stage in ("schedule", "all"):
        run_schedule(force=args.force)
    if args.stage in ("params", "all"):
        run_params(force=args.force)
    if args.stage in ("check", "all"):
        run_check(force=args.force)
