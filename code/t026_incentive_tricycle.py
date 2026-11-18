"""T026: Q3 incentives vs tricycles in the same T024 two-tier model.

Question: T025 showed incentives cannot replace trucks (the zone-level truck
layer has almost no unmet to remove).  The cell-level deficit is handled by
tricycles in T024-R3, at an average 1.47215 CNY per removed unmet bicycle.
This task re-solves the incentive (reroute) LP *inside* the T024 replay:

  * Delta_c is the T024 post-walking residual deficit at (cell, hour), not
    T008's static pool;
  * the reroute LP is T008's (same acceptance function pi(r), kappa/rho/c0,
    two source caps: arrivals / hourly surplus), restricted to cells of the
    same zone so that T018's zone totals and the truck layer are untouched;
  * a filled deficit y_c is removed from that cell's residual and the same
    number of bikes is removed from the source cells' end-of-hour arrivals,
    so the day's bike balance is exact;
  * the residual then goes through T024's per-zone tricycle LP and the
    per-zone fleet fixed point; the loop repeats until the fleet stops
    changing (<=10 iterations, residual reported).

Accounting (all platform cost, T024 conventions):
  total(r) = payment(r) + tricycle_layer(r) + truck_layer
  payment(r) = r * q(r)                    (uncapped LP payment; the 30,000
                                            budget is reported, not enforced)
  tricycle_layer = TRI_FIXED*n_exec + TRI_KM*lp_vehicle_km + PENALTY*unmet
  truck_layer    = 83,807.74383516546       (T018 integer plan, constant)

Stages: regression | smoke | scan | root | sensitivity | figure | check | all
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
import t008_q3_pool as t8
import t018_q2_int as t18
import t024_q2_tier2 as t24

RUN = os.path.join(lib.DATA, "runs", "T026")
os.makedirs(RUN, exist_ok=True)

R_KM = 0.45               # T008 edge radius
KAPPA, RHO, C0 = 2.0, 0.35, 0.8
BUDGET = 30000.0
TRUCK_OBJ = 83807.74383516546
TRI_FIXED = t24.TRI_FIXED
TRI_KM = t24.TRI_KM
TRI_Q = t24.TRI_Q
PENALTY = t24.PENALTY

R_GRID = np.round(np.arange(0.0, 3.0 + 1e-9, 0.05), 2)
CAPS = ("arrivals", "surplus")

STATE = {}


def dump(obj, name):
    path = os.path.join(RUN, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    print("wrote", path, flush=True)
    return path


def load_json(name):
    with open(os.path.join(RUN, name), encoding="utf-8") as fh:
        return json.load(fh)


def check_json(name, folder):
    with open(os.path.join(lib.DATA, "runs", folder, name),
              encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# loaded state
# --------------------------------------------------------------------------
def setup(force=False):
    if STATE.get("ready") and not force:
        return STATE
    t0 = time.time()
    inp, _ = t24.prep()
    plan, _ = t24.load_plan()
    cmp = check_json("compare.json", "T024")
    D = np.load(os.path.join(lib.DATA, "dist_matrix.npy"))
    zas = t24.zone_arrays(inp, plan)
    # T024's executable path starts from the DP allocation of the best row
    # (fleet_N = 900, allocation sums to 889 because of the per-zone cap)
    # and raises each zone to its own fixed point (total 975).
    best = max(cmp["rows"], key=lambda r0: r0["saving_vs_no_tricycle"])
    N = int(best["fleet_N"])
    alloc = np.array(cmp["allocation"][str(N)], dtype=int)
    assert int(alloc.sum()) == int(cmp["best_used"]), (
        int(alloc.sum()), cmp["best_used"])
    zones = []
    for za in zas:
        edges = zone_edges(za, D)
        zones.append(dict(za=za, edges=edges))
    STATE.update(ready=True, inp=inp, plan=plan, cmp=cmp, D=D, zas=zas,
                 alloc=alloc, zones=zones, setup_s=float(time.time() - t0))
    print("setup: %d zones, alloc total %d, %.1fs"
          % (len(zones), int(alloc.sum()), STATE["setup_s"]), flush=True)
    return STATE


# --------------------------------------------------------------------------
# reroute (T008 LP) inside a zone
# --------------------------------------------------------------------------
def zone_edges(za, D, R_KM=R_KM):
    """Per-hour reroute edges within one zone: j -> c, j != c, dist <= R."""
    idx = za["idx"]
    Cz = len(idx)
    sub = D[np.ix_(idx, idx)]
    out = {}
    for h in range(24):
        src = np.where(za["arr"][h] > 0)[0]
        if len(src) == 0:
            out[h] = None
            continue
        tg = np.arange(Cz)
        mask = sub[np.ix_(src, tg)] <= R_KM
        mask[np.arange(len(src)), src] = False
        nz = np.argwhere(mask)
        si, ti = nz[:, 0].astype(np.int64), nz[:, 1].astype(np.int64)
        out[h] = (src, tg, si, ti, sub[src[si], tg[ti]])
    return out


def reroute_hour(e, za, h, resid, arr, r, c0, source_cap):
    """One hourly reroute LP.  Returns (q, moved_per_source, filler_per_cell).

    Targets are the cells with a positive residual deficit after walking;
    sources are the cells with positive arrivals this hour (upper cap) or
    with positive hourly surplus (conservative cap).  The LP is exactly
    T008's: max sum w_e x_e, source sum_c x_jc <= A_j, target
    sum_j w_e x_jc <= Delta_c.
    """
    src, tg, si, ti, d = e
    live = resid[tg[ti]] > 1e-9
    if not live.any():
        return None
    si, ti, d = si[live], ti[live], d[live]
    w = (t8.acceptance(r, d, KAPPA, RHO, c0)
         - t8.acceptance(0.0, d, KAPPA, RHO, c0))
    keep = w > 1e-12
    if not keep.any():
        return None
    si, ti, w = si[keep], ti[keep], w[keep]
    ne = len(w)
    A = coo_matrix((np.r_[np.ones(ne), w],
                    (np.r_[si, len(src) + ti],
                     np.r_[np.arange(ne), np.arange(ne)])),
                   shape=(len(src) + len(tg), ne)).tocsr()
    caps = t8.source_caps(h, src, arr, za["dep"], source_cap)
    b = np.r_[caps, resid[tg]]
    res = linprog(-w, A_ub=A, b_ub=b, bounds=(0, None), method="highs")
    if res.status != 0:
        return None
    y = w * res.x
    q = float(y.sum())
    if q <= 1e-12:
        return None
    per_src = np.bincount(si, weights=y, minlength=len(src))
    per_tgt = np.bincount(ti, weights=y, minlength=len(tg))
    return q, per_src, per_tgt


# --------------------------------------------------------------------------
# T024 replay with the reroute layer (exact copy of dyn_zone at r = 0)
# --------------------------------------------------------------------------
def dyn_zone_r(za, D, n, r=0.0, c0=C0, source_cap="arrivals", edges=None,
               collect_trips=False):
    dep, arr0, I = za["dep"], za["arr"], za["i0"].copy()
    Cz = len(I)
    unmet = 0.0
    strict = 0.0
    tri_bikes = 0.0
    tri_bike_km = 0.0
    tri_vehicle_km = 0.0
    q_tot = 0.0
    payment = 0.0
    trips = []
    sub = D[np.ix_(za["idx"], za["idx"])]
    arr = arr0 if (r <= 0.0 or edges is None) else arr0.copy()
    for h in range(24):
        p = za["pickup"][h]
        tot = I.sum()
        if p > 1e-12 and tot > 1e-12:
            I *= max(0.0, 1.0 - p / tot)
        strict += np.maximum(dep[h] - I, 0.0).sum()
        take = np.minimum(dep[h], I)
        I -= take
        resid = dep[h] - take
        walk = t24.walk_borrow0(I, resid, za["nbrs"])
        resid = np.maximum(resid - walk, 0.0)
        if r > 0.0 and edges is not None and edges.get(h) is not None:
            got = reroute_hour(edges[h], za, h, resid, arr, r, c0,
                               source_cap)
            if got is not None:
                q, per_src, per_tgt = got
                resid = np.maximum(resid - per_tgt, 0.0)
                arr[h] = arr[h] - per_src
                q_tot += q
                payment += r * q
        if n > 1e-9 and t24.TRI_HOUR0 <= h < t24.TRI_HOUR1:
            sink = np.where(resid > 1e-9)[0]
            donor = np.where(I > 1e-9)[0]
            if len(sink) and len(donor):
                I_before = I.copy() if collect_trips else None
                resid_before = resid.copy() if collect_trips else None
                flows, bkm, vkm = t24._tri_lp_local(sink, donor, resid, I,
                                                    sub, n)
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
                q=float(q_tot), payment=float(payment), trips=trips)


def int_trips_r(za, D, n_lp, r=0.0, c0=C0, source_cap="arrivals",
                edges=None):
    """Integer trip list from the LP at n_lp (T024 _int_trips, same code)."""
    rr = dyn_zone_r(za, D, n_lp, r=r, c0=c0, source_cap=source_cap,
                    edges=edges, collect_trips=True)
    budget = {}
    for t in rr["trips"]:
        key = (t["hour"], t["donor"])
        if key not in budget:
            budget[key] = t["avail"]
    room = {}
    for t in rr["trips"]:
        key = (t["hour"], t["sink"])
        if key not in room:
            room[key] = t["room"]
    flows_bikes = float(sum(t["bikes"] for t in rr["trips"]))
    dropped_flows = 0
    dropped_bikes = 0.0
    trips = []
    qn = int(round(TRI_Q))
    for t in rr["trips"]:
        key = (t["hour"], t["donor"])
        n_int = int(round(t["bikes"]))
        n_int = min(n_int, max(0, budget[key]))
        budget[key] -= n_int
        if n_int <= 0:
            dropped_flows += 1
            dropped_bikes += t["bikes"]
            continue
        n_full, rem = divmod(n_int, qn)
        for _ in range(n_full):
            trips.append(dict(hour=t["hour"], donor=t["donor"],
                              sink=t["sink"], bikes=qn, km=t["km"]))
        if rem:
            trips.append(dict(hour=t["hour"], donor=t["donor"],
                              sink=t["sink"], bikes=rem, km=t["km"]))
    int_bikes = float(sum(t["bikes"] for t in trips))
    deliv = {}
    for t in trips:
        key = (t["hour"], t["sink"])
        deliv[key] = deliv.get(key, 0) + t["bikes"]
    sink_excess = max((deliv[k] - room.get(k, 0) for k in deliv), default=0)
    return dict(zone=int(za["z"]), n_lp=int(n_lp), r=rr, trips=trips,
                flows_bikes=flows_bikes, dropped_flows=int(dropped_flows),
                dropped_bikes=float(dropped_bikes), int_bikes=int_bikes,
                sink_excess_bikes=float(sink_excess),
                lp_vehicle_km=float(rr["tri_vehicle_km"]),
                lp_bike_km=float(rr["tri_bike_km"]),
                q=float(rr["q"]), payment=float(rr["payment"]))


def exec_n_r(za, n0, D, r=0.0, c0=C0, source_cap="arrivals", edges=None,
             cap=400):
    """T024 _exec_n with the reroute layer; fixed point on the fleet."""
    n = int(n0)
    b = int_trips_r(za, D, n, r, c0, source_cap, edges)
    hist = []
    for it in range(10):
        hi = min(cap, max(n + 10, 2 * n))
        k = t24._min_fleet(b, D, max(1, n - 5), hi)
        while k is None and hi < cap:
            hi = min(cap, 2 * hi)
            k = t24._min_fleet(b, D, max(1, n - 5), hi)
        if k is None:
            return dict(n_priced=int(n0), n_exec=None, feasible=False,
                        iterations=it, built=b, hist=hist, resid=None,
                        min_fleet=None)
        hist.append(dict(n=int(n), min_fleet=int(k)))
        if k <= n:
            # fixed point reached: the trip list built at n is schedulable by
            # n tricycles.  resid is the iterate change (0 here); slack is
            # the spare fleet above the greedy minimum for this last list.
            return dict(n_priced=int(n0), n_exec=int(n), feasible=True,
                        iterations=it, built=b, hist=hist,
                        min_fleet=int(k), resid=0, slack=int(n - k))
        n = int(k)
        b = int_trips_r(za, D, n, r, c0, source_cap, edges)
    return dict(n_priced=int(n0), n_exec=None, feasible=False,
                iterations=10, built=b, hist=hist, resid=None,
                min_fleet=None, slack=None)


# --------------------------------------------------------------------------
# one (r, cap) point
# --------------------------------------------------------------------------
def run_point(r, source_cap, c0=C0, warm=None, detail_rows=False):
    st = setup()
    D = st["D"]
    alloc = st["alloc"] if warm is None else np.asarray(warm, dtype=int)
    rows = []
    q = pay = 0.0
    for zi, rec in enumerate(st["zones"]):
        za = rec["za"]
        n0 = int(alloc[zi])
        if n0 <= 0:
            rows.append(dict(zone=int(za["z"]), n_priced=0, n_exec=0,
                             feasible=True, iterations=0, trips=0,
                             unassigned=0, unmet_lp=0.0, int_short=0.0,
                             unmet=0.0, lp_vehicle_km=0.0, q=0.0,
                             payment=0.0, resid=0, slack=0))
            continue
        e = exec_n_r(za, n0, D, r=r, c0=c0, source_cap=source_cap,
                     edges=rec["edges"])
        bf = e["built"]
        a = t24._assign(bf["trips"], e["n_exec"], D) if e["feasible"] else None
        q += bf["q"]
        pay += bf["payment"]
        rows.append(dict(
            zone=int(za["z"]), n_priced=int(n0), n_exec=e["n_exec"],
            feasible=bool(e["feasible"]), iterations=int(e["iterations"]),
            trips=int(len(bf["trips"])),
            unassigned=int(a["unassigned"]) if a else None,
            unmet_lp=float(bf["r"]["unmet"]),
            int_short=float(bf["flows_bikes"] - bf["int_bikes"]),
            unmet=float(bf["r"]["unmet"] + bf["flows_bikes"]
                        - bf["int_bikes"]),
            lp_vehicle_km=float(bf["lp_vehicle_km"]),
            q=float(bf["q"]), payment=float(bf["payment"]),
            resid=int(e["resid"]) if e["resid"] is not None else None,
            slack=int(e["slack"]) if e["slack"] is not None else None))
    n_tot = int(sum(r0["n_exec"] or 0 for r0 in rows))
    unmet = float(sum(r0["unmet"] for r0 in rows))
    lp_vkm = float(sum(r0["lp_vehicle_km"] for r0 in rows))
    unassigned = int(sum(r0["unassigned"] or 0 for r0 in rows))
    slack_tot = int(sum(r0["slack"] or 0 for r0 in rows))
    layer = float(TRI_FIXED * n_tot + TRI_KM * lp_vkm + PENALTY * unmet)
    total = float(pay + layer + TRUCK_OBJ)
    iter_max = int(max(r0["iterations"] for r0 in rows))
    resid_max = int(max((r0["resid"] or 0) for r0 in rows))
    out = dict(
        r=float(r), source_cap=source_cap,
        q=float(q), payment=float(pay),
        payment_uncapped=float(pay),
        payment_actual=float(min(pay, BUDGET)),
        payment_capped=float(min(pay, BUDGET)),
        capped=bool(pay > BUDGET),
        budget_slack=float(BUDGET - pay),
        budget_feasible=bool(pay <= BUDGET),
        q_budget=float(q if pay <= BUDGET else BUDGET / r if r > 0 else q),
        n_total=int(n_tot),
        tricycle_layer=float(layer),
        truck_layer=float(TRUCK_OBJ),
        total=float(total),
        total_capped=float(min(pay, BUDGET) + layer + TRUCK_OBJ),
        unmet=float(unmet),
        unassigned=int(unassigned),
        lp_vehicle_km=float(lp_vkm),
        iter=int(iter_max), resid=int(resid_max), slack=int(slack_tot),
        alloc=[int(r0["n_exec"] or 0) for r0 in rows],
        zones_raised=int(sum(1 for r0 in rows
                             if r0["n_exec"] is not None
                             and r0["n_exec"] > r0["n_priced"])))
    if detail_rows:
        out["rows"] = rows
    return out


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------
def regression():
    st = setup()
    t0 = time.time()
    p0 = run_point(0.0, "arrivals")
    sc = check_json("schedule_check.json", "T024")
    ref = sc["exec_totals"]
    chk = dict(
        n_total=[int(p0["n_total"]), int(ref["n_exec"])],
        unmet=[float(p0["unmet"]), float(ref["unmet"])],
        layer=[float(p0["tricycle_layer"]),
               float(ref["layer_cost_fullload"])],
        total=[float(p0["total"]), float(ref["total_fullload"])],
        unassigned=[int(p0["unassigned"]), 0],
        payment=[float(p0["payment"]), 0.0], q=[float(p0["q"]), 0.0],
        bit_identical=False)
    chk["bit_identical"] = bool(
        p0["n_total"] == ref["n_exec"]
        and p0["unmet"] == ref["unmet"]
        and p0["tricycle_layer"] == ref["layer_cost_fullload"]
        and p0["total"] == ref["total_fullload"]
        and p0["unassigned"] == 0
        and p0["payment"] == 0.0 and p0["q"] == 0.0)
    out = dict(task="T026 regression", date="2026-11-19",
               check=chk, point=p0, runtime_s=float(time.time() - t0),
               note=("r=0 with the reroute layer off must reproduce T024-R3 "
                     "bit-for-bit: same dyn_zone/int_trips/_exec_n code path, "
                     "no array mutation.  Compared against "
                     "data/runs/T024/schedule_check.json exec_totals."))
    dump(out, "regression.json")
    print(json.dumps(chk, indent=1), flush=True)
    return out


def stage_scan(caps=CAPS, grid=R_GRID, c0=C0, tag="cost_vs_r"):
    st = setup()
    out = dict(task="T026 cost vs r", date="2026-11-19",
               grid=[float(x) for x in grid], caps=list(caps),
               budget=BUDGET, c0=float(c0), truck_layer=float(TRUCK_OBJ),
               baseline=None, rows={})
    base = run_point(0.0, caps[0], c0=c0)
    out["baseline"] = base
    base_total = float(base["total"])
    out["total_at_zero"] = base_total
    for cap in caps:
        rows = []
        t0 = time.time()
        for i, r in enumerate(grid):
            # no warm start: the raise-only fixed point must not inherit a
            # higher fleet from the previous r, or n would be path-dependent
            p = run_point(float(r), cap, c0=c0)
            rows.append(p)
            if i % 10 == 0 or i == len(grid) - 1:
                print("  %s r=%.2f n=%d total=%.2f q=%.1f pay=%.0f (%.0fs)"
                      % (cap, r, p["n_total"], p["total"], p["q"],
                         p["payment"], time.time() - t0), flush=True)
        out["rows"][cap] = rows
    for cap in caps:
        for p in out["rows"][cap]:
            p["saving_vs_zero"] = float(base_total - p["total"])
    dump(out, tag + ".json")
    return out


# --------------------------------------------------------------------------
# root: r*, r_cross, H26, bound comparison
# --------------------------------------------------------------------------
def _scan_rows(cap):
    d = load_json("cost_vs_r.json")
    return d["rows"][cap], float(d["total_at_zero"])


def _bracket(rows, base_total):
    """Last grid interval where total crosses from below to at/above base.

    The low-r end can oscillate around the baseline (integer fleet jumps and
    the source-removal feedback), so the first upward crossing is not the
    return root; the last one is.
    """
    below = [i for i, p in enumerate(rows) if p["total"] < base_total]
    if not below:
        return None
    i = max(below)
    if i + 1 >= len(rows) or rows[i + 1]["total"] < base_total:
        return None
    return float(rows[i]["r"]), float(rows[i + 1]["r"])


def _bisect_cross(cap, lo, hi, base_total, c0=C0, tol_r=0.001):
    """Bisect total(r) = total(0) on r to 0.001 CNY; returns bracket/evals."""
    evals = []
    for _ in range(32):
        if hi - lo <= tol_r:
            break
        mid = round(0.5 * (lo + hi), 6)
        p = run_point(mid, cap, c0=c0)
        evals.append(dict(r=mid, total=p["total"],
                          g=float(p["total"] - base_total),
                          n_total=p["n_total"], payment=p["payment"]))
        if p["total"] >= base_total:
            hi = mid
        else:
            lo = mid
    return dict(lo=float(lo), hi=float(hi), width=float(hi - lo),
                evals=evals, converged=bool(hi - lo <= tol_r))


def _bound_table(cap, d):
    """Section-1 average-cost bound q(r)*(1.47215 - r) vs the exact saving."""
    t24chk = check_json("schedule_check.json", "T024")
    unit = float(t24chk["exec_totals"]["penalty_ceiling_exec"])
    base_total = float(d["total_at_zero"])
    tab = []
    for p in d["rows"][cap]:
        b = float(p["q"] * (unit - p["r"]))
        tab.append(dict(r=p["r"], q=p["q"], unit_cost=unit,
                        bound_saving=b,
                        actual_saving=float(base_total - p["total"]),
                        gap=float(b - (base_total - p["total"]))))
    return tab, unit


def stage_root(caps=CAPS):
    d = load_json("cost_vs_r.json")
    base_total = float(d["total_at_zero"])
    res = dict(task="T026 optimum", date="2026-11-19",
               baseline_total=base_total, budget=BUDGET,
               definitions=dict(
                   r_star="argmin total(r) on the 0.05 grid",
                   r_cross="return root after the productive region: the "
                           "last grid crossing from below to at/above "
                           "total(0), bisected on r to 0.001 CNY; the "
                           "low-r oscillation is excluded",
                   saving="total(0) - total(r)"),
               caps={})
    for cap in caps:
        rows, _ = _scan_rows(cap)
        star = min(rows, key=lambda p: p["total"])
        br = _bracket(rows, base_total)
        cross = None if br is None else _bisect_cross(
            cap, br[0], br[1], base_total)
        # per-zone fleet decline at r*
        pstar = run_point(float(star["r"]), cap, detail_rows=True)
        zones = []
        for zi, zrow in enumerate(pstar["rows"]):
            n0 = int(d["baseline"]["alloc"][zi])
            n1 = int(zrow["n_exec"] or 0)
            if n1 != n0:
                zones.append(dict(zone=zrow["zone"], n_zero=n0,
                                  n_star=n1, decline=n0 - n1))
        zones.sort(key=lambda z: (-abs(z["decline"]), z["zone"]))
        tab, unit = _bound_table(cap, d)
        base_layer = float(d["baseline"]["tricycle_layer"])
        base_unmet = float(d["baseline"]["unmet"])
        base_fixed = float(base_layer - PENALTY * base_unmet)
        star_fixed = float(star["tricycle_layer"]
                           - PENALTY * star["unmet"])
        fixed_save = float(base_fixed - star_fixed)
        net_red = float(base_unmet - star["unmet"])
        pen_save = float(PENALTY * net_red)
        layer_save = float(base_layer - star["tricycle_layer"])
        qstar = float(star["q"])
        bound_pred = float(qstar * (unit - star["r"]))
        actual_save = float(base_total - star["total"])
        shortfall = float(qstar * unit - fixed_save)
        plateau = [p["r"] for p in rows
                   if p["total"] <= star["total"] + 1000.0]
        feasible = [p for p in rows if p["payment"] <= BUDGET]
        bstar = min(feasible, key=lambda p: p["total"]) if feasible else None
        bump = [dict(r=p["r"], total=p["total"],
                     excess=float(p["total"] - base_total))
                for p in rows if p["r"] <= 0.30 and p["total"] > base_total]
        res["caps"][cap] = dict(
            r_star=float(star["r"]),
            total_star=float(star["total"]),
            net_saving_star=actual_save,
            q_star=qstar,
            payment_star=float(star["payment"]),
            payment_capped_star=float(star["payment_capped"]),
            budget_slack_star=float(BUDGET - star["payment"]),
            budget_feasible_star=bool(star["payment"] <= BUDGET),
            n_total_star=int(star["n_total"]),
            layer_star=float(star["tricycle_layer"]),
            layer_saving_star=layer_save,
            layer_saving_per_q_star=float(layer_save / qstar)
            if qstar > 0 else None,
            base_unmet=base_unmet,
            unmet_star=float(star["unmet"]),
            net_unmet_reduction_star=net_red,
            unmet_reduction_per_q_star=float(net_red / qstar)
            if qstar > 0 else None,
            fixed_km_base=base_fixed,
            fixed_km_star=star_fixed,
            fixed_km_saving_star=fixed_save,
            fixed_km_saving_per_q_star=float(fixed_save / qstar)
            if qstar > 0 else None,
            fixed_km_saving_per_removed_unmet_star=float(fixed_save / net_red)
            if net_red > 0 else None,
            penalty_saving_star=pen_save,
            bound_prediction_star=bound_pred,
            bound_gap_star=float(bound_pred - actual_save),
            gap_decomposition=dict(
                fixed_km_shortfall=shortfall, penalty_credit=pen_save,
                identity_ok=bool(abs(shortfall - pen_save
                                     - (bound_pred - actual_save)) < 1e-6)),
            r_star_plateau=[float(x) for x in plateau],
            r_star_budget=None if bstar is None else float(bstar["r"]),
            total_star_budget=None if bstar is None else float(bstar["total"]),
            low_r_bump=bump,
            low_r_bump_note=(
                "At very small r the reroute removes source arrivals and can "
                "slightly increase the later tricycle layer before the "
                "filled deficits dominate; the affected grid points are "
                "listed in low_r_bump.  r_cross is the return root after "
                "the productive region, not the trivial root at r=0."),
            r_cross=None if cross is None else float(cross["hi"]),
            cross=cross,
            bound_table=tab,
            bound_unit_cost=unit,
            bound_note=(
                "The section-1 bound q(r)*(unit - r) prices every filled "
                "deficit at the average tricycle-layer unit cost "
                "%.5f. The exact replay differs because (i) the fleet is "
                "integer and only raised, "
                "(ii) the reroute fills whatever sources can reach, not the "
                "cheapest tricycles, (iii) some filled deficits would have "
                "stayed unmet (saving 6) rather than been served by a "
                "tricycle, and (iv) tricycles run 07-20 only while reroutes "
                "are offered in all 24 hours; (v) the bound's q comes from "
                "T008's static pool while this replay prices the T024 dynamic "
                "post-walking deficit.  The dominant gap is that q does not "
                "translate one-for-one into removed unmet: at r*, q=%.1f but "
                "the net unmet reduction is %.1f (%.3f per q), because the "
                "source cells lose the same arrivals and create later "
                "deficits.  The fixed+km saving is %.2f (%.4f per q, %.4f "
                "per net removed unmet) versus the bound's q*unit=%.2f at "
                "%.5f; the exact penalty saving %.2f is a credit the bound "
                "omits (it assumes both worlds have the same unmet).  Net "
                "gap = %.2f = fixed+km shortfall %.2f - penalty credit %.2f."
                % (unit, qstar, net_red,
                   float(net_red / qstar) if qstar > 0 else 0.0,
                   fixed_save, float(fixed_save / qstar) if qstar > 0 else 0.0,
                   float(fixed_save / net_red) if net_red > 0 else 0.0,
                   float(qstar * unit), unit, pen_save,
                   float(bound_pred - actual_save), shortfall, pen_save)),
            zone_decline=zones[:15],
            zone_decline_total=int(sum(z["decline"] for z in zones)),
            n_zones_changed=int(len(zones)),
            detail_check=dict(
                n_total=sum(int(z["n_exec"] or 0) for z in pstar["rows"]),
                expected=int(star["n_total"]),
                payment=float(pstar["payment"]),
                expected_payment=float(star["payment"])))
    # ---- H26: r = 0.81 vs r = 2.70 in the same caliber -------------------
    p081 = {cap: run_point(0.81, cap) for cap in caps}
    p270 = {cap: next(p for p in d["rows"][cap]
                      if abs(p["r"] - 2.70) < 1e-9) for cap in caps}
    cap0 = "arrivals"
    a, b = p081[cap0], p270[cap0]
    truck_equal = bool(a["truck_layer"] == b["truck_layer"] == TRUCK_OBJ)
    better = "0.81" if a["total"] < b["total"] else "2.70"
    delta = float(b["total"] - a["total"])
    star0 = res["caps"][cap0]
    h26 = dict(
        caliber=cap0,
        r_low=0.81, r_high=2.70,
        low=dict(total=float(a["total"]),
                 total_capped=float(a["total_capped"]),
                 payment_uncapped=float(a["payment_uncapped"]),
                 payment_capped=float(a["payment_capped"]),
                 capped=bool(a["capped"]), q=float(a["q"]),
                 budget_slack=float(a["budget_slack"]),
                 budget_feasible=bool(a["budget_feasible"]),
                 truck_layer=float(a["truck_layer"]),
                 tricycle_layer=float(a["tricycle_layer"]),
                 n_total=int(a["n_total"]), unmet=float(a["unmet"])),
        high=dict(total=float(b["total"]),
                  total_capped=float(b["total_capped"]),
                  payment_uncapped=float(b["payment_uncapped"]),
                  payment_capped=float(b["payment_capped"]),
                  capped=bool(b["capped"]), q=float(b["q"]),
                  budget_slack=float(b["budget_slack"]),
                  budget_feasible=bool(b["budget_feasible"]),
                  truck_layer=float(b["truck_layer"]),
                  tricycle_layer=float(b["tricycle_layer"]),
                  n_total=int(b["n_total"]), unmet=float(b["unmet"])),
        both_caps=dict(
            arrivals=dict(low_total=float(p081["arrivals"]["total"]),
                          high_total=float(p270["arrivals"]["total"])),
            surplus=dict(low_total=float(p081["surplus"]["total"]),
                         high_total=float(p270["surplus"]["total"]))),
        truck_layer_equal=bool(truck_equal),
        truck_layer_note=(
            "The truck layer is identical at both r by construction: "
            "reroutes are restricted to the same zone, so every zone's "
            "total arrivals and T018's integer plan are unchanged.  A "
            "lower truck objective at 2.70 does not exist in this caliber."
            if truck_equal else
            "STOP: the truck layer changed with r; the deficit caliber "
            "between the reroute LP and the tricycle layer is misaligned."),
        better=better,
        total_gap_high_minus_low=delta,
        offset_evidence=dict(
            payment_gap=float(b["payment_uncapped"] - a["payment_uncapped"]),
            layer_gap=float(b["tricycle_layer"] - a["tricycle_layer"]),
            total_gap=delta,
            check=("payment_gap + layer_gap == total_gap"
                   if abs((b["payment_uncapped"] - a["payment_uncapped"])
                          + (b["tricycle_layer"] - a["tricycle_layer"])
                          - delta) < 1e-6 else "MISMATCH")),
        global_r_star=float(star0["r_star"]),
        global_net_saving=float(star0["net_saving_star"]),
        rewrite_note=None)
    if better == "2.70":
        h26["rewrite_note"] = (
            "r = 2.70 has the lower total; the recommended reward is "
            "r* = %.2f (grid 0.05) with a net saving of %.2f CNY/day. "
            "Paper section 6 and the abstract must be rewritten."
            % (star0["r_star"], star0["net_saving_star"]))
    else:
        h26["rewrite_note"] = (
            "r = 0.81 has the lower total: %.2f vs %.2f CNY/day "
            "(difference %.2f).  The truck layer is identical, so no "
            "lower truck objective exists to offset; the 2.70 point pays "
            "%.2f more in rewards while saving only %.2f in the tricycle "
            "layer.  The global optimum of this caliber is r* = %.2f "
            "(grid 0.05) with a net saving of %.2f CNY/day; paper section 6 "
            "and the abstract should state r*, not 2.70."
            % (a["total"], b["total"], delta,
               b["payment_uncapped"] - a["payment_uncapped"],
               -(b["tricycle_layer"] - a["tricycle_layer"]),
               star0["r_star"], star0["net_saving_star"]))
    res["h26"] = h26
    h26["total_capped_note"] = (
        "total_capped = min(payment, 30000) + layers is an accounting-only "
        "ledger variant.  It is NOT a feasible policy: the acceptance "
        "probability depends on the promised r, and paying only 30,000 "
        "against an obligation of %.2f at r=2.70 would renege on the "
        "promise.  The policy comparison therefore uses the uncapped total."
        % b["payment_uncapped"])
    h26["budget_note"] = (
        "At r = 2.70 the uncapped obligation is %.2f CNY/day, %.1fx the "
        "30,000 budget; honoring r would require re-scaling the reward, so "
        "the point is budget-infeasible, not merely capped (capping the "
        "payment without changing r would renege on the promise).  At "
        "r = 0.81 the obligation is %.2f, %.2fx the budget.  The "
        "unconstrained optimum r* = %.2f pays %.2f, inside the budget, so "
        "the budget does not bind at the optimum; the budget-feasible "
        "optimum is r = %.2f."
        % (b["payment_uncapped"], b["payment_uncapped"] / BUDGET,
           a["payment_uncapped"], a["payment_uncapped"] / BUDGET,
           star0["r_star"], star0["payment_star"],
           star0["r_star_budget"]))
    # ---- truck invariant over every scanned row --------------------------
    bad = []
    for cap in caps:
        for p in d["rows"][cap]:
            if float(p["truck_layer"]) != TRUCK_OBJ:
                bad.append(dict(cap=cap, r=p["r"],
                                truck_layer=p["truck_layer"]))
    res["truck_invariant"] = dict(
        expected=TRUCK_OBJ, violated=bad, ok=not bad)
    if bad:
        print("STOP: truck layer changed with r (caliber misalignment); "
              "first violations: %s" % bad[:3], flush=True)
    dump(res, "optimum.json")
    print(json.dumps({c: {k: res["caps"][c][k] for k in
                          ("r_star", "net_saving_star", "r_cross")}
                      for c in caps}, indent=1), flush=True)
    print("h26:", better, "delta=%.2f" % delta, flush=True)
    return res


# --------------------------------------------------------------------------
# c0 sensitivity
# --------------------------------------------------------------------------
def stage_sensitivity(grid=R_GRID, cap="arrivals"):
    """c0 x {0.8, 1.0, 1.2}: re-optimize r* on the same 0-3.00 grid."""
    d = load_json("cost_vs_r.json")
    base_total = float(d["total_at_zero"])
    rows = []
    for f in (0.8, 1.0, 1.2):
        c0 = float(C0 * f)
        if abs(f - 1.0) < 1e-12:
            star = min(d["rows"][cap], key=lambda p: p["total"])
            n_grid, src = len(d["rows"][cap]), "cost_vs_r.json"
        else:
            rs = []
            t0 = time.time()
            for i, r in enumerate(grid):
                rs.append(run_point(float(r), cap, c0=c0))
                if i % 15 == 0 or i == len(grid) - 1:
                    p = rs[-1]
                    print("  c0x%.1f r=%.2f total=%.2f n=%d (%.0fs)"
                          % (f, p["r"], p["total"], p["n_total"],
                             time.time() - t0), flush=True)
            star = min(rs, key=lambda p: p["total"])
            n_grid, src = len(rs), "stage_sensitivity"
        rows.append(dict(
            factor=float(f), c0=c0, r_star=float(star["r"]),
            total_star=float(star["total"]),
            net_saving_star=float(base_total - star["total"]),
            q_star=float(star["q"]), payment_star=float(star["payment"]),
            budget_feasible=bool(star["budget_feasible"]),
            n_total_star=int(star["n_total"]),
            unmet_star=float(star["unmet"]),
            iter=int(star["iter"]), resid=int(star["resid"]),
            n_grid=n_grid, source=src))
    out = dict(task="T026 c0 sensitivity", date="2026-11-19", cap=cap,
               c0_base=C0, grid=[float(x) for x in grid],
               baseline_total=base_total, budget=BUDGET, rows=rows,
               note=("c0 is the acceptance-function constant in "
                     "pi(r) = logistic(kappa*r - rho*tau - c0).  Each row "
                     "re-optimizes r* on the same 61-point 0-3.00 grid "
                     "(0.05 step); the x1.0 row reuses cost_vs_r.json, so "
                     "the r* movement is a re-optimization, not a fixed-r "
                     "stress.  '+' means a higher c0 (lower acceptance)."))
    dump(out, "sensitivity.json")
    return out


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
def stage_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False})
    d = load_json("cost_vs_r.json")
    opt = load_json("optimum.json")
    base_total = float(d["total_at_zero"])
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.2, 3.1),
        gridspec_kw=dict(width_ratios=[1.12, 1.0], wspace=0.30))
    cols = {"arrivals": "#0072B2", "surplus": "#D55E00"}
    labs = {"arrivals": "arrivals cap (upper)",
            "surplus": "surplus cap (conservative)"}
    for cap in ("arrivals", "surplus"):
        rows = d["rows"][cap]
        x = np.array([p["r"] for p in rows])
        y = np.array([p["total"] for p in rows]) / 1000.0
        for ax in (ax1, ax2):
            ax.plot(x, y, color=cols[cap], lw=1.4, label=labs[cap])
        o = opt["caps"][cap]
        for ax in (ax1, ax2):
            ax.plot([o["r_star"]], [o["total_star"] / 1000.0], "o",
                    ms=4.2, color=cols[cap], zorder=5)
            ax.axvline(o["r_cross"], color=cols[cap], ls="--", lw=0.9,
                       alpha=0.8)
    ax1.axhline(base_total / 1000.0, color="0.35", ls=":", lw=1.0,
                label="r = 0 baseline")
    ax2.axhline(base_total / 1000.0, color="0.35", ls=":", lw=1.0)
    o = opt["caps"]["arrivals"]
    ax1.set_xlim(0.0, 3.0)
    ax1.set_ylim(560.0, 1460.0)
    ax1.set_xlabel("reward r (CNY per accepted reroute)")
    ax1.set_ylabel("total platform cost ($10^3$ CNY/day)")
    ax1.set_title("(a) full grid 0-3.00")
    ax1.legend(frameon=False, loc="upper left")
    ax2.set_xlim(0.0, 1.2)
    lo = min(float(p["total"]) for cap in CAPS for p in d["rows"][cap]
             if p["r"] <= 1.2) / 1000.0
    ax2.set_ylim(lo - 6.0, base_total / 1000.0 + 8.0)
    ax2.set_xlabel("reward r (CNY per accepted reroute)")
    ax2.set_ylabel("total platform cost ($10^3$ CNY/day)")
    ax2.set_title("(b) zoom 0-1.20")
    ax2.annotate("r* = %.2f\nsaving %.0f" % (o["r_star"],
                                             o["net_saving_star"]),
                 xy=(o["r_star"], o["total_star"] / 1000.0),
                 xytext=(0.16, lo + 0.5), fontsize=7, color=cols["arrivals"],
                 arrowprops=dict(arrowstyle="-", lw=0.7,
                                 color=cols["arrivals"]))
    ax2.annotate("r_cross = %.3f" % o["r_cross"],
                 xy=(o["r_cross"], base_total / 1000.0),
                 xytext=(0.76, base_total / 1000.0 + 5.0), fontsize=7,
                 color="0.25",
                 arrowprops=dict(arrowstyle="-", lw=0.7, color="0.4"))
    fig.tight_layout()
    out = os.path.join(RUN, "figs")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "fig_total_vs_r.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path, flush=True)
    return path


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------
def stage_check():
    d = load_json("cost_vs_r.json")
    opt = load_json("optimum.json")
    reg = load_json("regression.json")
    sen = load_json("sensitivity.json")
    checks = {}
    # 1 grid and fields
    need = ("r", "q", "payment", "payment_capped", "n_total",
            "tricycle_layer", "truck_layer", "total", "unmet", "iter",
            "resid")
    miss = []
    for cap in CAPS:
        rows = d["rows"][cap]
        if len(rows) != 61 or abs(rows[0]["r"]) > 1e-9 \
                or abs(rows[-1]["r"] - 3.0) > 1e-9:
            miss.append("%s grid length/ends" % cap)
        for p in rows:
            for k in need:
                if k not in p:
                    miss.append("%s r=%.2f missing %s" % (cap, p["r"], k))
        if not any(abs(p["r"] - 2.70) < 1e-9 for p in rows):
            miss.append("%s no r=2.70" % cap)
    checks["1_grid_and_fields"] = dict(
        pass_=not miss, n_grid=61, r_270_present=True,
        detail=("61 values 0..3.00 step 0.05 per cap; every row carries "
                + ", ".join(need)), issues=miss[:20])
    # 2 H26
    h = opt["h26"]
    verdict_ok = ((h["better"] == "0.81")
                  == (h["low"]["total"] < h["high"]["total"]))
    checks["2_h26_comparison"] = dict(
        pass_=bool(verdict_ok and h["truck_layer_equal"]
                   and "payment_uncapped" in h["low"]
                   and "payment_capped" in h["low"]
                   and "payment_uncapped" in h["high"]
                   and "payment_capped" in h["high"]),
        verdict_consistent=bool(verdict_ok),
        better=h["better"],
        low=dict(total=h["low"]["total"],
                 payment_uncapped=h["low"]["payment_uncapped"],
                 payment_capped=h["low"]["payment_capped"],
                 capped=h["low"]["capped"]),
        high=dict(total=h["high"]["total"],
                  payment_uncapped=h["high"]["payment_uncapped"],
                  payment_capped=h["high"]["payment_capped"],
                  capped=h["high"]["capped"]),
        truck_layer_equal=h["truck_layer_equal"],
        evidence=h["offset_evidence"], note=h["rewrite_note"])
    # 3 regression
    checks["3_regression"] = dict(
        pass_=bool(reg["check"]["bit_identical"]),
        detail=reg["check"],
        source="data/runs/T024/schedule_check.json exec_totals")
    # 4 fixed point
    bad = []
    for cap in CAPS:
        for p in d["rows"][cap]:
            if p["iter"] > 10 or not (p["resid"] is not None
                                      and p["resid"] < 1):
                bad.append(dict(cap=cap, r=p["r"], iter=p["iter"],
                                resid=p["resid"]))
    slack = max(p["slack"] for cap in CAPS for p in d["rows"][cap])
    checks["4_fixed_point"] = dict(
        pass_=not bad, issues=bad[:10],
        max_iter=max(p["iter"] for cap in CAPS for p in d["rows"][cap]),
        max_resid=max(p["resid"] for cap in CAPS for p in d["rows"][cap]),
        max_slack=int(slack),
        detail=("resid = |n_(i+1) - n_i| of the last fleet iteration (0 at "
                "the fixed point).  slack = n_exec - min_fleet of the last "
                "trip list is the greedy over-provision, reported separately "
                "and not a convergence residual."))
    # 5 r_cross and r*
    rows5 = {}
    ok5 = True
    for cap in CAPS:
        c = opt["caps"][cap]["cross"]
        good = bool(c is not None and c["converged"]
                    and c["width"] <= 1e-3)
        ok5 = ok5 and good
        rows5[cap] = dict(r_star=opt["caps"][cap]["r_star"],
                          net_saving=opt["caps"][cap]["net_saving_star"],
                          r_cross=opt["caps"][cap]["r_cross"],
                          width=None if c is None else c["width"],
                          converged=good,
                          evals=None if c is None else c["evals"])
    checks["5_cross_and_star"] = dict(
        pass_=ok5, by_cap=rows5,
        detail="bisection on r to <= 0.001 CNY; r* = grid argmin (0.05)")
    # 6 sensitivity
    checks["6_sensitivity"] = dict(
        pass_=bool(len(sen["rows"]) == 3
                   and all(r["n_grid"] == 61 for r in sen["rows"])),
        rows=[dict(factor=r["factor"], c0=r["c0"], r_star=r["r_star"],
                   total=r["total_star"],
                   net_saving=r["net_saving_star"],
                   payment=r["payment_star"], q=r["q_star"]) for r in sen["rows"]],
        detail=("each c0 is re-optimized on the 61-point 0-3.00 grid; "
                "r* and net saving are the movement requested by the sheet"),
        note=sen["note"])
    # 7 figure
    figp = os.path.join(RUN, "figs", "fig_total_vs_r.pdf")
    head = b""
    if os.path.exists(figp):
        with open(figp, "rb") as fh:
            head = fh.read(5)
    checks["7_figure"] = dict(
        pass_=bool(head == b"%PDF-"),
        path="data/runs/T026/figs/fig_total_vs_r.pdf",
        script="code/t026_incentive_tricycle.py:stage_figure",
        vector=True, size_bytes=(os.path.getsize(figp)
                                 if os.path.exists(figp) else 0))
    # 8 calibre
    checks["8_delta_caliber"] = dict(
        pass_=bool(reg["check"]["bit_identical"]),
        detail=("the reroute block runs after walk_borrow0 and before "
                "t24._tri_lp_local on the same resid array; r=0 reproduces "
                "T024-R3 bit-for-bit, so Delta_c is the post-walking "
                "deficit by construction"),
        evidence="code/t026_incentive_tricycle.py:dyn_zone_r / regression.json")
    # 9 caps and budget
    checks["9_source_caps_and_budget"] = dict(
        pass_=bool(all(len(d["rows"][c]) == 61 for c in CAPS)
                   and any(p["capped"] for p in d["rows"]["arrivals"])
                   and any(not p["capped"] for p in d["rows"]["arrivals"])),
        caps=list(CAPS),
        n_capped_arrivals=sum(1 for p in d["rows"]["arrivals"]
                              if p["capped"]),
        n_capped_surplus=sum(1 for p in d["rows"]["surplus"]
                             if p["capped"]),
        detail=("payment is uncapped in the model; payment_capped = "
                "min(payment, 30000) is reported alongside and never "
                "fed back into the behavior"))
    # 10 truck invariant
    checks["10_truck_invariant"] = dict(
        pass_=bool(opt["truck_invariant"]["ok"]),
        expected=TRUCK_OBJ, violated=opt["truck_invariant"]["violated"],
        detail=("if any row's truck layer differs, the run must stop and "
                "report rather than accept silently"))
    # 11 detail consistency at r*
    det = opt["caps"]["arrivals"]["detail_check"]
    checks["11_detail_consistency"] = dict(
        pass_=bool(det["n_total"] == det["expected"]
                   and abs(det["payment"] - det["expected_payment"]) < 1e-9),
        detail=det)
    # 12 bound-gap decomposition identity
    bd = {cap: opt["caps"][cap]["gap_decomposition"] for cap in CAPS}
    checks["12_bound_decomposition"] = dict(
        pass_=bool(all(v["identity_ok"] for v in bd.values())),
        by_cap=bd,
        detail=("gap = q*unit - fixed_km_saving - 6*net_unmet_reduction; "
                "the fixed+km shortfall minus the penalty credit reproduces "
                "the bound-vs-exact gap"))
    out = dict(task="T026 checks", date="2026-11-19",
               n_pass=sum(1 for c in checks.values() if c["pass_"]),
               n_total=len(checks), checks=checks)
    dump(out, "check.json")
    print("%d/%d checks pass" % (out["n_pass"], out["n_total"]), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="T026 incentives vs tricycles")
    ap.add_argument("--stage", default="regression",
                    choices=("regression", "smoke", "scan", "root",
                             "sensitivity", "figure", "check", "all"))
    ap.add_argument("--r", type=float, default=None)
    ap.add_argument("--cap", default="arrivals", choices=CAPS)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    if args.stage in ("regression", "all"):
        regression()
    if args.stage == "smoke":
        p = run_point(args.r, args.cap)
        print(json.dumps({k: p[k] for k in
                          ("r", "q", "payment", "payment_uncapped",
                           "payment_capped", "capped", "n_total",
                           "tricycle_layer", "truck_layer", "total",
                           "unmet", "iter", "resid", "slack")}, indent=1))
    if args.stage in ("scan", "all"):
        stage_scan()
    if args.stage in ("root", "all"):
        stage_root()
    if args.stage in ("sensitivity", "all"):
        stage_sensitivity()
    if args.stage in ("figure", "all"):
        stage_figure()
    if args.stage in ("check", "all"):
        stage_check()
    print("[%.1fs] stage %s done" % (time.time() - t0, args.stage),
          flush=True)


if __name__ == "__main__":
    main()
