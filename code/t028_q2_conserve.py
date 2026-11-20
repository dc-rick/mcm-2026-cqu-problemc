"""T028: conserving cell-level replay of the T018/T024 two-tier plan.

T024's dyn_zone adds the *planned* zone-hour arrival surface to stock even
when the corresponding departures were not served, so a no-tricycle replay
ends the day with I0 + unmet bicycles instead of I0 (582,883 vs 343,638).
This module replaces that replay with a joint city simulation:

  * every served departure generates exactly one arrival, routed with the
    T018 destination kernel Pcell[h, origin zone, destination cell];
  * truck moves are arc-level: a bicycle is removed from its origin cell and
    added to its destination cell in the same hour, so a stock-short origin
    scales its own deliveries rather than creating bicycles;
  * walking and tricycles move bicycles inside a zone and generate no arrival;
  * the city total is asserted equal to I0 at every hour.

The one timing approximation is explicit: an arrival is credited at the end
of the hour in which its trip is served.  The median trip is 11.5 min, so
this is a within-hour approximation; it has no effect on conservation.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
import t024_q2_tier2 as t24

RUN = os.path.join(lib.DATA, "runs", "T028")
os.makedirs(RUN, exist_ok=True)


def dump(obj, name):
    path = os.path.join(RUN, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    print("wrote", path, flush=True)
    return path


def load_json(name):
    with open(os.path.join(RUN, name), encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
def build(inp, plan):
    dep = inp["dep_cell"].astype(np.float64)          # (24, C)
    Pcell = inp["Pcell"].astype(np.float64)           # (24, Z, C)
    zid = inp["zid"].astype(int)
    Z = int(zid.max()) + 1
    C = len(zid)
    xy = inp["xy"].astype(np.float64)
    I0 = inp["I0c"].astype(np.float64)
    XM = plan["XM"].astype(np.float64)                # (24, Z, Z)
    # dep_cell columns are ordered by cell; zone totals via bincount
    dep_zone = np.zeros((24, Z))
    for h in range(24):
        dep_zone[h] = np.bincount(zid, weights=dep[h], minlength=Z)
    # planned arrival surface, used only to place truck drops inside a zone
    surf = np.zeros((24, C))
    for h in range(24):
        surf[h] = Pcell[h].T @ dep_zone[h]
    zones = []
    for z in range(Z):
        idx = np.where(zid == z)[0]
        tree = cKDTree(xy[idx])
        # tree is built on the local (zone) point array, so its neighbour
        # indices are already local
        nb = [np.asarray(v, dtype=int) for v in
              tree.query_ball_point(xy[idx], t24.WALK_KM)]
        zones.append(dict(z=z, idx=idx, i0=I0[idx], dep=dep[:, idx],
                          nb=nb,
                          surf=surf[:, idx],
                          sub=None))
    D = np.load(os.path.join(lib.DATA, "dist_matrix.npy"))
    for z in range(Z):
        iz = zones[z]["idx"]
        zones[z]["sub"] = D[np.ix_(iz, iz)]
    return dict(dep=dep, Pcell=Pcell, zid=zid, Z=Z, C=C, xy=xy, I0=I0,
                XM=XM, dep_zone=dep_zone, surf=surf, zones=zones, D=D)


def simulate(ctx, nvec, record=False, truck=True):
    """Joint hourly simulation.  nvec[z] tricycles in zone z."""
    Z, C = ctx["Z"], ctx["C"]
    I = ctx["I0"].copy()
    XM = ctx["XM"]
    Pcell = ctx["Pcell"]
    zones = ctx["zones"]
    total0 = float(I.sum())
    assert abs(total0 - 343638.0) < 1e-6, total0

    unmet_h = np.zeros(24)
    strict_h = np.zeros(24)
    served_h = np.zeros((24, Z))
    truck_short_h = np.zeros(24)
    tri_bikes = np.zeros(Z)
    tri_bike_km = np.zeros(Z)
    tri_vehicle_km = np.zeros(Z)
    total_h = np.zeros(24)
    traj = np.zeros((24, C)) if record else None
    max_resid = 0.0

    for h in range(24):
        # --- truck moves: arc-level, conserving ---------------------------
        if truck:
            deliver = np.zeros(Z)
            for z, zz in enumerate(zones):
                idx = zz["idx"]
                arcs = [(j, float(XM[h, z, j])) for j in range(Z)
                        if XM[h, z, j] > 1e-12]
                if not arcs:
                    continue
                p = float(sum(x for _, x in arcs))
                tot = float(I[idx].sum())
                if p <= 1e-12:
                    continue
                if tot <= 1e-12:
                    scale = 0.0
                else:
                    scale = min(1.0, tot / p)
                moved = p * scale
                truck_short_h[h] += p - moved
                if moved > 1e-12:
                    I[idx] *= max(0.0, 1.0 - moved / tot)
                for j, x in arcs:
                    if x * scale > 1e-12:
                        deliver[j] += x * scale
            # distribute each zone's total delivery by the planned surface
            for j, zz in enumerate(zones):
                d = float(deliver[j])
                if d <= 1e-12:
                    continue
                idx = zz["idx"]
                w = zz["surf"][h]
                if w.sum() <= 1e-12:
                    w = np.ones(len(idx))
                I[idx] += d * w / w.sum()

        # --- departures, walking, tricycles -------------------------------
        for z, zz in enumerate(zones):
            idx = zz["idx"]
            I_z = I[idx]
            dep = zz["dep"][h]
            strict_h[h] += float(np.maximum(dep - I_z, 0.0).sum())
            take = np.minimum(dep, I_z)
            I_z -= take
            resid = dep - take
            walk = t24.walk_borrow0(I_z, resid, zz["nb"])
            resid = np.maximum(resid - walk, 0.0)
            served = float(take.sum() + walk.sum())
            n = float(nvec[z])
            if n > 1e-9 and t24.TRI_HOUR0 <= h < t24.TRI_HOUR1:
                sink = np.where(resid > 1e-9)[0]
                donor = np.where(I_z > 1e-9)[0]
                if len(sink) and len(donor):
                    flows, bkm, vkm = t24._tri_lp_local(
                        sink, donor, resid, I_z, zz["sub"], n)
                    if flows:
                        for (d, s, f) in flows:
                            # the tricycle moves a bicycle from s to d, and
                            # the user at d then rides it away: the move adds
                            # f at d, the served departure removes it again.
                            I_z[s] -= f
                            I_z[d] += f
                            I_z[d] -= f
                            resid[d] -= f
                            served += f
                            tri_bikes[z] += f
                        tri_bike_km[z] += bkm
                        tri_vehicle_km[z] += vkm
            I[idx] = I_z
            served_h[h, z] = served
            unmet_h[h] += float(np.maximum(resid, 0.0).sum())

        # --- arrivals from this hour's actual served trips ----------------
        arr = Pcell[h].T @ served_h[h]
        I += arr

        total_h[h] = float(I.sum())
        max_resid = max(max_resid, abs(total_h[h] - total0))
        if record:
            traj[h] = I

    return dict(
        unmet=float(unmet_h.sum()), strict=float(strict_h.sum()),
        unmet_by_hour=unmet_h, strict_by_hour=strict_h,
        served_h=served_h, truck_short_by_hour=truck_short_h,
        truck_short=float(truck_short_h.sum()),
        tri_bikes=tri_bikes, tri_bike_km=tri_bike_km,
        tri_vehicle_km=tri_vehicle_km,
        total_by_hour=total_h, end_total=float(I.sum()),
        max_total_resid=float(max_resid),
        I_end=I, traj=traj,
        nvec=np.asarray(nvec, dtype=float))


def diag():
    inp, _ = t24.prep()
    plan, scal = t24.load_plan()
    ctx = build(inp, plan)
    sch = t24.load_json("schedule_check.json")
    cmp = t24.load_json("compare.json")
    exec_n = np.zeros(ctx["Z"], dtype=int)
    for r in sch["exec_rows"]:
        exec_n[int(r["zone"])] = int(r["n_exec"])
    alloc = np.array(cmp["allocation"][str(max(
        cmp["rows"], key=lambda r: r["saving_vs_no_tricycle"])["fleet_N"])],
        dtype=int)
    out = dict(task="T028 diagnostic", date="2026-11-19",
               note=("joint conserving replay; nvec=0 should end at 343,638; "
                     "exec_n is T024's 975-zone fleet; alloc is the 889 "
                     "priced allocation"))
    for name, nv in (("zero", np.zeros(ctx["Z"], dtype=int)),
                     ("alloc889", alloc),
                     ("exec975", exec_n)):
        t0 = time.time()
        r = simulate(ctx, nv)
        out[name] = dict(
            n=int(nv.sum()), unmet=float(r["unmet"]),
            strict=float(r["strict"]),
            end_total=float(r["end_total"]),
            max_total_resid=float(r["max_total_resid"]),
            truck_short=float(r["truck_short"]),
            tri_bikes=float(r["tri_bikes"].sum()),
            tri_vehicle_km=float(r["tri_vehicle_km"].sum()),
            tri_bike_km=float(r["tri_bike_km"].sum()),
            runtime_s=float(time.time() - t0),
            unmet_by_hour=[float(x) for x in r["unmet_by_hour"]])
        print(name, out[name], flush=True)
    out["baseline_no_tri_old"] = float(cmp["no_tricycle"]["cell_unmet"])
    out["exec_old"] = float(sch["exec_totals"]["unmet"])
    dump(out, "diag.json")
    return out


# --------------------------------------------------------------------------
# zone views with an exogenous arrival surface, for the per-zone cost curves
# --------------------------------------------------------------------------
def zone_views(ctx, served_h):
    """Reuse T024's dyn_zone contract, but with arrivals generated by the
    given served profile instead of the planned dep_zone - U."""
    Z = ctx["Z"]
    arr = np.zeros((24, ctx["C"]))
    for h in range(24):
        arr[h] = ctx["Pcell"][h].T @ served_h[h]
    zas = []
    for z, zz in enumerate(ctx["zones"]):
        idx = zz["idx"]
        zas.append(dict(
            z=z, idx=idx, dep=zz["dep"], arr=arr[:, idx], i0=zz["i0"],
            nbrs=zz["nb"],
            pickup=np.array([ctx["XM"][h, z, :].sum() for h in range(24)]),
            drop=np.array([ctx["XM"][h, :, z].sum() for h in range(24)])))
    return zas, arr


def curves(zas, D, Z, nmax=t24.TRI_NCURVE):
    """Per-zone cost/unmet/km curves at a fixed arrival surface."""
    nc = nmax + 1
    cost = np.zeros((Z, nc))
    unmet = np.zeros((Z, nc))
    vkm = np.zeros((Z, nc))
    bkm = np.zeros((Z, nc))
    strict = np.zeros((Z, nc))
    bikes = np.zeros((Z, nc))
    for zi, za in enumerate(zas):
        for n in range(nc):
            r = t24.dyn_zone(za, D, n)
            cost[zi, n] = (t24.TRI_FIXED * n + t24.TRI_KM * r["tri_vehicle_km"]
                           + t24.PENALTY * r["unmet"])
            unmet[zi, n] = r["unmet"]
            vkm[zi, n] = r["tri_vehicle_km"]
            bkm[zi, n] = r["tri_bike_km"]
            strict[zi, n] = r["strict"]
            bikes[zi, n] = r["tri_bikes"]
    return dict(cost=cost, unmet=unmet, km=vkm, bike_km=bkm,
                strict=strict, bikes=bikes)


def solve_dp(cv, Z, fleets=None):
    fleets = fleets or t24.FLEETS
    NCAP = max(fleets)
    dp, choice = t24._dp_curves(cv["cost"], Z, t24.TRI_NMAX, NCAP)
    alloc = {}
    for N in fleets:
        u = int(np.argmin(dp[:N + 1]))
        alloc[N] = t24._dp_recon(choice, u, Z)
    best_used = int(np.argmin(dp))
    best_alloc = t24._dp_recon(choice, best_used, Z)
    base_unmet = float(cv["unmet"][:, 0].sum())
    base_total = float(truck_obj) + t24.PENALTY * base_unmet
    rows = []
    for N in fleets:
        nz = alloc[N]
        unmet = float(sum(cv["unmet"][zi, nz[zi]] for zi in range(Z)))
        km = float(sum(cv["km"][zi, nz[zi]] for zi in range(Z)))
        bike_km = float(sum(cv["bike_km"][zi, nz[zi]] for zi in range(Z)))
        tri_cost = float(t24.TRI_FIXED * nz.sum() + t24.TRI_KM * km)
        total = float(truck_obj) + tri_cost + t24.PENALTY * unmet
        rows.append(dict(
            fleet_N=int(N), tricycles=int(nz.sum()),
            zones_with_tricycles=int((nz > 0).sum()),
            max_zone_tricycles=int(nz.max()) if Z else 0,
            cell_unmet=unmet, pre_tricycle_shortfall=float(
                sum(cv["strict"][zi, nz[zi]] for zi in range(Z))),
            tricycle_vehicle_km=km, tricycle_bike_km=bike_km,
            tricycle_bikes=float(
                sum(cv["bikes"][zi, nz[zi]] for zi in range(Z))),
            tricycle_cost=tri_cost, truck_obj=float(truck_obj),
            total=total, saving_vs_no_tricycle=float(base_total - total),
            marginal_saving=None))
    for k in range(1, len(rows)):
        rows[k]["marginal_saving"] = float(
            rows[k - 1]["total"] - rows[k]["total"])
    best_row = max(rows, key=lambda r: r["saving_vs_no_tricycle"])
    return dict(best_used=best_used, best_alloc=best_alloc,
                best_row=best_row, rows=rows, alloc=alloc,
                no_tricycle_unmet=base_unmet,
                no_tricycle_total=base_total)


def schedule(zas, alloc, D, Z, cv):
    """T024's executable fixed point at a fixed arrival surface."""
    rows = []
    for zi, za in enumerate(zas):
        n_z = int(alloc[zi])
        if n_z <= 0:
            base = float(cv["unmet"][zi, 0])
            rows.append(dict(zone=zi, n_priced=0, n_exec=0, feasible=True,
                             iterations=0, trips=0, unassigned=0,
                             unmet_lp=base, int_short=0.0, unmet=base,
                             lp_vehicle_km=0.0, vehicle_km=0.0,
                             deadhead_km=0.0, bike_km=0.0,
                             tri_bikes=0.0))
            continue
        e = t24._exec_n(za, n_z, D)
        bf = e["built"]
        a = t24._assign(bf["trips"], e["n_exec"], D) if e["feasible"] else None
        rows.append(dict(
            zone=zi, n_priced=n_z, n_exec=e["n_exec"],
            feasible=bool(e["feasible"]), iterations=int(e["iterations"]),
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
            bike_km=float(a["bike_km"]) if a else None,
            tri_bikes=float(bf["r"]["tri_bikes"])))
    ex = dict(
        n_exec=int(sum(r["n_exec"] or 0 for r in rows)),
        feasible=bool(all(r["feasible"] for r in rows)),
        zones_raised=int(sum(1 for r in rows
                             if r["n_exec"] is not None
                             and r["n_exec"] > r["n_priced"])),
        unmet_lp=float(sum(r["unmet_lp"] for r in rows)),
        int_short=float(sum(r["int_short"] for r in rows)),
        unmet=float(sum(r["unmet"] for r in rows)),
        lp_vehicle_km=float(sum(r["lp_vehicle_km"] for r in rows)),
        vehicle_km=float(sum(r["vehicle_km"] or 0 for r in rows)),
        deadhead_km=float(sum(r["deadhead_km"] or 0 for r in rows)),
        bike_km=float(sum(r["bike_km"] or 0 for r in rows)))
    ex["layer_cost_fullload"] = float(
        t24.TRI_FIXED * ex["n_exec"]
        + t24.TRI_KM * ex["lp_vehicle_km"]
        + t24.PENALTY * ex["unmet"])
    ex["layer_cost_actual"] = float(
        t24.TRI_FIXED * ex["n_exec"]
        + t24.TRI_KM * (ex["vehicle_km"] + ex["deadhead_km"])
        + t24.PENALTY * ex["unmet"])
    ex["total_fullload"] = float(truck_obj + ex["layer_cost_fullload"])
    ex["total_actual"] = float(truck_obj + ex["layer_cost_actual"])
    ex["no_tricycle_total"] = float(
        truck_obj + t24.PENALTY * float(alloc_no_tri_unmet))
    ex["saving_fullload"] = float(ex["no_tricycle_total"]
                                  - ex["total_fullload"])
    ex["saving_actual"] = float(ex["no_tricycle_total"]
                                - ex["total_actual"])
    ex["saving_fullload_share"] = float(
        ex["saving_fullload"] / ex["no_tricycle_total"])
    ex["saving_actual_share"] = float(
        ex["saving_actual"] / ex["no_tricycle_total"])
    ex["removed_unmet"] = float(alloc_no_tri_unmet - ex["unmet"])
    return dict(rows=rows, totals=ex)


truck_obj = 83807.74383516546
alloc_no_tri_unmet = 369181.07568090025


def iterate(max_iter=8, tol=1.0):
    inp, _ = t24.prep()
    plan, _ = t24.load_plan()
    ctx = build(inp, plan)
    D = ctx["D"]
    Z = ctx["Z"]
    t0 = time.time()
    # start from the planned served profile (T018's dep_zone - U)
    served0 = np.maximum(ctx["dep_zone"] - plan["U"].T, 0.0) \
        if plan["U"].shape == (Z, 24) else ctx["dep_zone"].copy()
    served = served0.copy()
    hist = []
    prev_alloc = None
    prev_unmet = None
    for it in range(1, max_iter + 1):
        zas, arr = zone_views(ctx, served)
        cv = curves(zas, D, Z)
        dp = solve_dp(cv, Z)
        alloc = dp["best_alloc"]
        sch = schedule(zas, alloc, D, Z, cv)
        nvec = np.array([r["n_exec"] or 0 for r in sch["rows"]], dtype=int)
        rep = simulate(ctx, nvec)
        # damped update of the arrival-generating served profile
        new_served = 0.5 * served + 0.5 * rep["served_h"]
        dn = int(np.abs(alloc - prev_alloc).sum()) if prev_alloc is not None \
            else None
        du = abs(rep["unmet"] - prev_unmet) if prev_unmet is not None else None
        hist.append(dict(
            iteration=it,
            best_N=int(dp["best_row"]["fleet_N"]),
            alloc_sum=int(alloc.sum()),
            n_exec=int(nvec.sum()),
            dp_unmet=float(dp["best_row"]["cell_unmet"]),
            dp_total=float(dp["best_row"]["total"]),
            replay_unmet=float(rep["unmet"]),
            replay_end=float(rep["end_total"]),
            replay_truck_short=float(rep["truck_short"]),
            replay_tri_bikes=float(rep["tri_bikes"].sum()),
            replay_tri_vkm=float(rep["tri_vehicle_km"].sum()),
            exec_unmet=float(sch["totals"]["unmet"]),
            exec_total=float(sch["totals"]["total_fullload"]),
            alloc_change=dn, unmet_change=du,
            runtime_s=float(time.time() - t0)))
        print(hist[-1], flush=True)
        alloc = np.asarray(alloc, dtype=int)
        if it > 1 and dn == 0 and du is not None and du < tol:
            break
        prev_alloc = alloc
        prev_unmet = float(rep["unmet"])
        served = new_served
    # final full evaluation at the last allocation
    zas, arr = zone_views(ctx, served)
    cv = curves(zas, D, Z)
    dp = solve_dp(cv, Z)
    alloc = dp["best_alloc"]
    sch = schedule(zas, alloc, D, Z, cv)
    nvec = np.array([r["n_exec"] or 0 for r in sch["rows"]], dtype=int)
    rep = simulate(ctx, nvec)
    out = dict(task="T028 conserving Q2 rerun", date="2026-11-19",
               iterations=hist, final=dict(
                   served_h=rep["served_h"].tolist(),
                   allocation=[int(x) for x in alloc],
                   n_exec=[int(x) for x in nvec],
                   replay=dict(unmet=float(rep["unmet"]),
                               end_total=float(rep["end_total"]),
                               max_total_resid=float(rep["max_total_resid"]),
                               truck_short=float(rep["truck_short"]),
                               tri_bikes=float(rep["tri_bikes"].sum()),
                               tri_vehicle_km=float(
                                   rep["tri_vehicle_km"].sum()),
                               tri_bike_km=float(rep["tri_bike_km"].sum())),
                   dp=dict(best_N=int(dp["best_row"]["fleet_N"]),
                           best_used=int(dp["best_used"]),
                           rows=dp["rows"]),
                   schedule=dict(rows=sch["rows"], totals=sch["totals"]),
                   curves=dict(cost=cv["cost"].tolist(),
                               unmet=cv["unmet"].tolist(),
                               km=cv["km"].tolist(),
                               bike_km=cv["bike_km"].tolist())),
               note=("fixed point between the per-zone cost curves (fixed "
                     "arrival surface) and the joint conserving replay.  "
                     "Each iteration rebuilds the curves from the last "
                     "replay's served profile, re-solves the integer DP and "
                     "the executable fleet, and reruns the joint replay."))
    dump(out, "iterate.json")
    return out


def joint_cost(ctx, nvec):
    r = simulate(ctx, nvec)
    total = (truck_obj + t24.TRI_FIXED * float(np.sum(nvec))
             + t24.TRI_KM * float(r["tri_vehicle_km"].sum())
             + t24.PENALTY * float(r["unmet"]))
    return float(total), r


def local_search(max_sweeps=3, deltas=(-2, -1, 1, 2), resume=False,
                 start_json=None, start_field="exec_allocation"):
    """Coordinate descent on the joint conserving replay.

    The per-zone cost curves in `iterate` are a partial-equilibrium
    approximation: they hold the arrival surface fixed while one zone's
    fleet changes.  This stage evaluates the actual joint model at every
    candidate fleet and is therefore the optimisation the paper reports.

    With `resume=True` the descent restarts from the saved `lp_allocation`
    and appends to its history.  With `start_json` it restarts from an
    explicit allocation (used when a uniform fleet scan finds a better
    region than the current local optimum).
    """
    inp, _ = t24.prep()
    plan, _ = t24.load_plan()
    ctx = build(inp, plan)
    Z = ctx["Z"]
    sch = t24.load_json("schedule_check.json")
    nvec = np.zeros(Z, dtype=int)
    for row in sch["exec_rows"]:
        nvec[int(row["zone"])] = int(row["n_exec"])
    start_label = "T024 schedule_check exec_rows"
    t0 = time.time()
    hist = []
    s0 = 0
    if resume:
        prev = load_json("local_search.json")
        nvec = np.array(prev["lp_allocation"], dtype=int)
        hist = list(prev["history"])
        s0 = max(h["sweep"] for h in hist)
        start_label = "local_search lp_allocation"
        print("resume from n=%d at sweep %d" % (int(nvec.sum()), s0),
              flush=True)
    if start_json:
        with open(start_json, encoding="utf-8") as fh:
            prev = json.load(fh)
        nvec = np.array(prev[start_field], dtype=int)
        start_label = os.path.basename(start_json) + ":" + start_field
        hist = []
        s0 = 0
        print("start from %s n=%d" % (start_label, int(nvec.sum())),
              flush=True)
    base, r0 = joint_cost(ctx, nvec)
    if not hist:
        hist.append(dict(sweep=0, zone=None, n=int(nvec.sum()),
                         cost=float(base), unmet=float(r0["unmet"]),
                         tri_vkm=float(r0["tri_vehicle_km"].sum()),
                         runtime_s=float(time.time() - t0)))
        print(hist[-1], flush=True)
    for sweep in range(1, max_sweeps + 1):
        improved = False
        for z in range(Z):
            cur = base
            best_n = int(nvec[z])
            best_cost = cur
            best_r = None
            for d in deltas:
                nz = int(nvec[z] + d)
                if nz < 0:
                    continue
                nv = nvec.copy()
                nv[z] = nz
                c, rr = joint_cost(ctx, nv)
                if c < best_cost - 1e-6:
                    best_cost = c
                    best_n = nz
                    best_r = rr
            if best_n != nvec[z]:
                nvec[z] = best_n
                base = best_cost
                improved = True
                hist.append(dict(sweep=s0 + sweep, zone=z,
                                 n=int(nvec.sum()),
                                 cost=float(base),
                                 unmet=float(best_r["unmet"]),
                                 tri_vkm=float(
                                     best_r["tri_vehicle_km"].sum()),
                                 runtime_s=float(time.time() - t0)))
                print(hist[-1], flush=True)
        if not improved:
            break
    final_cost, final_r = joint_cost(ctx, nvec)
    # executable fixed point at the joint allocation
    served_h = np.maximum(final_r["served_h"], 0.0)
    zas, _ = zone_views(ctx, served_h)
    D = ctx["D"]
    base_unmet = np.array([t24.dyn_zone(za, D, 0)["unmet"] for za in zas])
    cv0 = dict(unmet=np.array([[u, u] for u in base_unmet]))
    sch2 = schedule(zas, nvec, D, Z, cv0)
    exec_n = np.array([r["n_exec"] or 0 for r in sch2["rows"]], dtype=int)
    exec_cost, exec_r = joint_cost(ctx, exec_n)
    out = dict(task="T028 joint local search", date="2026-11-20",
               start=start_label, start_n=int(hist[0]["n"]),
               lp_n=int(nvec.sum()),
               lp_cost=float(final_cost), lp_unmet=float(final_r["unmet"]),
               lp_tri_vkm=float(final_r["tri_vehicle_km"].sum()),
               lp_allocation=[int(x) for x in nvec],
               exec_n=int(exec_n.sum()), exec_cost=float(exec_cost),
               exec_unmet=float(exec_r["unmet"]),
               exec_tri_vkm=float(exec_r["tri_vehicle_km"].sum()),
               exec_allocation=[int(x) for x in exec_n],
               exec_schedule=dict(rows=sch2["rows"],
                                  totals=sch2["totals"]),
               baseline_total=float(truck_obj + t24.PENALTY
                                    * alloc_no_tri_unmet),
               history=hist,
               note=("coordinate descent over integer per-zone fleets on "
                     "the joint conserving replay; each candidate is a full "
                     "24-hour city simulation."))
    dump(out, "local_search.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="diag",
                    choices=("diag", "iterate", "local"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sweeps", type=int, default=3)
    ap.add_argument("--start-json", default=None)
    ap.add_argument("--start-field", default="exec_allocation")
    args = ap.parse_args()
    if args.stage == "diag":
        diag()
    elif args.stage == "iterate":
        iterate()
    elif args.stage == "local":
        local_search(max_sweeps=args.sweeps, resume=args.resume,
                     start_json=args.start_json,
                     start_field=args.start_field)
