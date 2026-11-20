"""T031: Q3 rebased on the conserving Q2/Q4 models, payment ledger primary.

Two calibers are rebuilt:

1. Truck-caliber ladder (T020's static pool LP).  The fence capacities and
   the demand they serve changed when Q4 was rebuilt on the conserved fleet
   (T030), so the payment ladder (r, q, P, rP) is re-solved against the T030
   plan and spare capacities.  The bill is the payment ledger ``r P`` with
   ``P = sum_e pi(r) x_e``; the increment ledger ``r q`` is reported as a
   secondary column.

2. Tricycle-caliber replay (T026's experiment) on the T028 joint conserving
   simulation.  A rerouted trip moves one arrival from its source cell to
   the target cell and serves one waiting departure there; the served
   departures of the hour generate the hour's arrivals, and the source
   cells lose exactly the rerouted arrivals.  The city total stays I0 at
   every hour.  The bill is again ``r P``; the deficit-filling count ``q``
   and the increment bill ``r q`` are reported alongside.

The executable tricycle fleet is re-solved at the rerouted served profile.

Stages: truck | tricycle | all
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
import t008_q3_pool as t8
import t019_q4_cap as t19
import t020_q3_budget as t20
import t024_q2_tier2 as t24
import t026_incentive_tricycle as t26
import t028_q2_conserve as t28

RUN = os.path.join(lib.DATA, "runs", "T031")
os.makedirs(RUN, exist_ok=True)

BUDGET = 30000.0
LEVEL = "meas343"
R_GRID = np.round(np.arange(0.0, 3.0 + 1e-9, 0.05), 2)


def dump(obj, name):
    path = os.path.join(RUN, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    print("wrote", path, flush=True)
    return path


def load_json(name):
    with open(os.path.join(RUN, name), encoding="utf-8") as fh:
        return json.load(fh)


def truck_unit():
    """Cost of one truck-placed bicycle, from the stored K=60 plan.

    The static pool LP prices every unplaced bicycle at the average cost the
    truck layer pays per bicycle it moves.  That ratio was a typed constant
    in T020 (82,194.80/24,679.28 = 3.3305 from an earlier plan); it is
    re-derived here from the T018 result that the paper reports, so the
    unit and the plan come from the same file.
    """
    with open(os.path.join(lib.DATA, "runs", "T018",
                           "result_int_K60.json"), encoding="utf-8") as fh:
        d = json.load(fh)
    return float(d["obj"]) / float(d["moved"])


# --------------------------------------------------------------------------
# truck caliber: T020's pool LP against the T030 plan and spare capacity
# --------------------------------------------------------------------------
def build_capacity_t30(pool, level=LEVEL):
    plan = np.load(os.path.join(lib.DATA, "runs", "T030",
                                f"plan_{level}_main.npz"))
    s, f, ii, jj, a = (plan[k] for k in ("s", "f", "ii", "jj", "a"))
    sites = np.where(s > 0.5)[0]
    spos = {int(j): k for k, j in enumerate(sites)}
    sp = np.load(os.path.join(lib.DATA, "runs", "T030",
                              f"spare_{level}.npz"))
    sites_arr = sp["sites"]
    if not np.array_equal(sites, sites_arr):
        raise RuntimeError("T030 site order mismatch")
    spare = sp["spare"]
    cap = spare.reshape(24, 4, len(sites)).min(1)
    serv = {}
    for k in range(len(f)):
        if f[k] <= 0 or a[ii[k]] <= 0:
            continue
        j = int(jj[k])
        if s[j] <= 0.5:
            continue
        serv.setdefault(int(ii[k]), []).append(
            (spos[j], float(f[k] / a[ii[k]])))
    z = np.load(os.path.join(lib.DATA, "runs", "T030",
                             "cell_demand_cons.npz"))
    outside = np.ones(len(z["a_meas343"]), bool)
    for c in serv:
        outside[c] = False
    return dict(cap=cap, serv=serv, outside=outside, sites=sites,
                n_sites=len(sites))


def truck_ledger(force=False):
    path = os.path.join(RUN, "truck_payment.json")
    if os.path.exists(path) and not force:
        print("reuse", path)
        return load_json("truck_payment.json")
    t0 = time.time()
    pool = t20.load_pool()
    capinfo = build_capacity_t30(pool)
    lp = t20.PoolLP(pool, capinfo)
    N = pool["N"]
    unit = truck_unit()
    out = dict(task="T031 truck-caliber payment ledger", date="2026-11-20",
               budget=BUDGET, level=LEVEL, N=float(N),
               truck_unit=float(unit),
               cap_source="data/runs/T030/plan_meas343_main.npz + "
                          "spare_meas343.npz",
               note=("static pool LP of T020 rebuilt on the conserved Q4 "
                     "plan; payment ledger r*P is the primary bill; the "
                     "increment ledger r*q is secondary"),
               levels={})
    for cap_name in ("arrivals", "surplus"):
        caps = np.empty(lp.ns)
        for k, (h, s) in enumerate(lp.src_key_list):
            if cap_name == "arrivals":
                caps[k] = pool["arr"][h, s]
            else:
                caps[k] = max(0.0, pool["arr"][h, s] - pool["dep"][h, s])
        lp.b_src = caps
        rows = []
        for r in R_GRID:
            got = lp.solve(r, budget=None, cap_on=True)
            rows.append(dict(r=float(r), q=float(got["q"]),
                             P=float(got["P"]),
                             invited=float(got["invited"]),
                             payment=float(r * got["P"]),
                             bill_old=float(r * got["q"]),
                             budget_feasible=bool(r * got["P"]
                                                  <= BUDGET + 1e-9)))
        feasible = [x for x in rows if x["budget_feasible"]]
        # cheapest total = payment + truck_unit * (N - q); q enters through
        # the truck residual, so the optimum is the feasible point with the
        # lowest total spend, not the lowest bill.
        for x in feasible:
            x["total_spend"] = float(
                x["payment"] + unit * (N - x["q"]))
        best = min(feasible, key=lambda x: x["total_spend"])
        out["levels"][cap_name] = dict(
            rows=rows, best=best,
            all_truck_cost=float(unit * N))
        print(f"{cap_name}: best r={best['r']:.2f} q={best['q']:.2f} "
              f"P={best['P']:.2f} pay={best['payment']:.2f} "
              f"total={best['total_spend']:.2f}", flush=True)
    out["runtime_s"] = float(time.time() - t0)
    dump(out, "truck_payment.json")
    return out


# --------------------------------------------------------------------------
# tricycle caliber: reroute inside the joint conserving replay
# --------------------------------------------------------------------------
def reroute_hour_p(e, dep, arr, h, resid, r, c0, source_cap):
    """T026's hourly reroute LP, also returning the payment head count.

    q = sum_e (pi(r)-pi(0)) x_e   (deficit-filling placements)
    P = sum_e pi(r) x_e           (accepted reroutes, payment ledger)
    """
    src, tg, si, ti, d = e
    live = resid[tg[ti]] > 1e-9
    if not live.any():
        return None
    si, ti, d = si[live], ti[live], d[live]
    w = (t8.acceptance(r, d, t26.KAPPA, t26.RHO, c0)
         - t8.acceptance(0.0, d, t26.KAPPA, t26.RHO, c0))
    keep = w > 1e-12
    if not keep.any():
        return None
    si, ti, w = si[keep], ti[keep], w[keep]
    pi = w + t8.acceptance(0.0, d[keep], t26.KAPPA, t26.RHO, c0)
    ne = len(w)
    A = sparse.coo_matrix(
        (np.r_[np.ones(ne), w],
         (np.r_[si, len(src) + ti], np.r_[np.arange(ne), np.arange(ne)])),
        shape=(len(src) + len(tg), ne)).tocsr()
    caps = t8.source_caps(h, src, arr, dep, source_cap)
    b = np.r_[caps, resid[tg]]
    res = linprog(-w, A_ub=A, b_ub=b, bounds=(0, None), method="highs")
    if res.status != 0:
        return None
    q = float(w @ res.x)
    if q <= 1e-12:
        return None
    P = float(pi @ res.x)
    per_src = np.bincount(si, weights=w * res.x, minlength=len(src))
    per_tgt = np.bincount(ti, weights=w * res.x, minlength=len(tg))
    return q, P, per_src, per_tgt


def simulate_r(ctx, nvec, r=0.0, c0=t26.C0, source_cap="arrivals",
               record=False):
    """T028's conserving joint replay plus an intra-zone reroute layer.

    The reroute serves a waiting departure at the target cell from the
    rerouted arrival, and removes the same number of arrivals from the
    source cells.  Arrivals of the hour are

        arr = Pcell[h] @ served_new - q_src,

    where served_new includes the rerouted waiting departures and q_src is
    the per-cell count whose trip no longer ends at the source.  Every
    served departure generates one arrival, so the city total is I0.
    """
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
    q_h = np.zeros(24)
    P_h = np.zeros(24)
    q_zone = np.zeros((24, Z))
    total_h = np.zeros(24)
    traj = np.zeros((24, C)) if record else None
    max_resid = 0.0
    max_arr_neg = 0.0

    for h in range(24):
        if True:                                    # truck moves
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
                scale = 0.0 if tot <= 1e-12 else min(1.0, tot / p)
                moved = p * scale
                truck_short_h[h] += p - moved
                if moved > 1e-12:
                    I[idx] *= max(0.0, 1.0 - moved / tot)
                for j, x in arcs:
                    if x * scale > 1e-12:
                        deliver[j] += x * scale
            for j, zz in enumerate(zones):
                d = float(deliver[j])
                if d <= 1e-12:
                    continue
                idx = zz["idx"]
                w = zz["surf"][h]
                if w.sum() <= 1e-12:
                    w = np.ones(len(idx))
                I[idx] += d * w / w.sum()

        q_src_city = np.zeros(C)
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
            if r > 0.0:
                e = ctx["edges"][z][h]
                if e is not None:
                    arr_plan = ctx["surf"][:, idx]
                    got = reroute_hour_p(e, zz["dep"], arr_plan, h, resid,
                                         r, c0, source_cap)
                    if got is not None:
                        q, P, per_src, per_tgt = got
                        resid = np.maximum(resid - per_tgt, 0.0)
                        served += q
                        q_h[h] += q
                        q_zone[h, z] += q
                        P_h[h] += P
                        q_src_city[idx] += per_src
            n = float(nvec[z])
            if n > 1e-9 and t24.TRI_HOUR0 <= h < t24.TRI_HOUR1:
                sink = np.where(resid > 1e-9)[0]
                donor = np.where(I_z > 1e-9)[0]
                if len(sink) and len(donor):
                    flows, bkm, vkm = t24._tri_lp_local(
                        sink, donor, resid, I_z, zz["sub"], n)
                    if flows:
                        for (d, s, f) in flows:
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

        arr = Pcell[h].T @ served_h[h]
        # reroutes remove q_src arrivals; if a source cell has fewer
        # modelled arrivals than rerouted trips, scale the whole hour's
        # reroute vector (including the served departures it created)
        qs = q_src_city
        if qs.sum() > 1e-12:
            ratio = np.divide(arr, qs, out=np.full_like(arr, np.inf),
                              where=qs > 1e-9)
            fscale = float(min(1.0, ratio.min()))
            if fscale < 1.0:
                served_h[h] -= q_zone[h] * (1.0 - fscale)
                arr = Pcell[h].T @ served_h[h]
                q_h[h] *= fscale
                P_h[h] *= fscale
                qs = qs * fscale
            arr = arr - qs
            max_arr_neg = max(max_arr_neg, float(-arr.min()))
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
        q=float(q_h.sum()), P=float(P_h.sum()), q_by_hour=q_h,
        P_by_hour=P_h, total_by_hour=total_h, end_total=float(I.sum()),
        max_total_resid=float(max_resid),
        max_arr_neg=float(max_arr_neg),
        I_end=I, traj=traj, nvec=np.asarray(nvec, dtype=float))


def setup_ctx():
    inp, _ = t24.prep()
    plan, _ = t24.load_plan()
    ctx = t28.build(inp, plan)
    D = ctx["D"]
    edges = []
    for zz in ctx["zones"]:
        idx = zz["idx"]
        za = dict(idx=idx, arr=ctx["surf"][:, idx],
                  dep=zz["dep"])
        edges.append(t26.zone_edges(za, D))
    ctx["edges"] = edges
    return ctx


def tricycle_ledger(force=False, grid=None):
    path = os.path.join(RUN, "tricycle_payment.json")
    if os.path.exists(path) and not force:
        print("reuse", path)
        return load_json("tricycle_payment.json")
    t0 = time.time()
    grid = R_GRID if grid is None else np.asarray(grid, dtype=float)
    ctx = setup_ctx()
    Z = ctx["Z"]
    D = ctx["D"]
    ls = t28.load_json("local_search.json")
    start_n = np.array(ls["exec_allocation"], dtype=int)
    base = simulate_r(ctx, start_n, r=0.0)
    base_cost = float(t28.truck_obj + t24.TRI_FIXED * start_n.sum()
                      + t24.TRI_KM * base["tri_vehicle_km"].sum()
                      + t24.PENALTY * base["unmet"])
    print(f"baseline n={start_n.sum()} unmet={base['unmet']:.2f} "
          f"vkm={base['tri_vehicle_km'].sum():.2f} cost={base_cost:.2f}",
          flush=True)
    rows = {}
    for cap_name in ("arrivals", "surplus"):
        scan = []
        ck = os.path.join(RUN, f"scan_{cap_name}.json")
        if os.path.exists(ck) and not force:
            with open(ck, encoding="utf-8") as fh:
                scan = json.load(fh)
            print(f"resume {cap_name}: {len(scan)} r values", flush=True)
        done_r = {round(x["r"], 4) for x in scan}
        for r in grid:
            if round(float(r), 4) in done_r:
                continue
            rep = simulate_r(ctx, start_n, r=float(r), source_cap=cap_name)
            layer = float(t24.TRI_FIXED * start_n.sum()
                          + t24.TRI_KM * rep["tri_vehicle_km"].sum()
                          + t24.PENALTY * rep["unmet"])
            pay = float(r * rep["P"])
            rec = dict(
                r=float(r), q=float(rep["q"]), P=float(rep["P"]),
                payment=pay, bill_old=float(r * rep["q"]),
                unmet=float(rep["unmet"]),
                tri_vehicle_km=float(rep["tri_vehicle_km"].sum()),
                tricycle_layer=layer,
                total=float(layer + pay + t28.truck_obj),
                end_total=float(rep["end_total"]),
                max_total_resid=float(rep["max_total_resid"]),
                max_arr_neg=float(rep["max_arr_neg"]),
                budget_feasible=bool(pay <= BUDGET + 1e-9))
            scan.append(rec)
            scan.sort(key=lambda x: x["r"])
            with open(ck, "w", encoding="utf-8") as fh:
                json.dump(scan, fh, ensure_ascii=False, indent=1)
            print(f"  {cap_name} r={rec['r']:.2f} q={rec['q']:.1f} "
                  f"P={rec['P']:.1f} pay={rec['payment']:.1f} "
                  f"unmet={rec['unmet']:.1f} "
                  f"total={rec['total']:.1f}", flush=True)
        feas = [x for x in scan if x["budget_feasible"]]
        best = min(feas, key=lambda x: x["total"])
        best_fixed = dict(best)
        print(f"{cap_name}: best r={best['r']:.2f} q={best['q']:.1f} "
              f"P={best['P']:.1f} pay={best['payment']:.1f} "
              f"unmet={best['unmet']:.1f} total={best['total']:.1f}",
              flush=True)
        # executable fleet at the rerouted served profile of the best r
        rep = simulate_r(ctx, start_n, r=best["r"], source_cap=cap_name)
        zas, _ = t28.zone_views(ctx, rep["served_h"])
        base_unmet = np.array([t24.dyn_zone(za, D, 0)["unmet"] for za in zas])
        cv0 = dict(unmet=np.array([[u, u] for u in base_unmet]))
        sch = t28.schedule(zas, start_n, D, Z, cv0)
        exec_n = np.array([x["n_exec"] or 0 for x in sch["rows"]], dtype=int)
        rep2 = simulate_r(ctx, exec_n, r=best["r"], source_cap=cap_name)
        # second fixed-point pass: rebuild the views from the new served
        # profile and re-solve the executable fleet
        zas2, _ = t28.zone_views(ctx, rep2["served_h"])
        base_unmet2 = np.array(
            [t24.dyn_zone(za, D, 0)["unmet"] for za in zas2])
        cv2 = dict(unmet=np.array([[u, u] for u in base_unmet2]))
        sch2 = t28.schedule(zas2, exec_n, D, Z, cv2)
        exec_n2 = np.array([x["n_exec"] or 0 for x in sch2["rows"]],
                           dtype=int)
        rep3 = simulate_r(ctx, exec_n2, r=best["r"], source_cap=cap_name)
        layer2 = float(t24.TRI_FIXED * exec_n2.sum()
                       + t24.TRI_KM * rep3["tri_vehicle_km"].sum()
                       + t24.PENALTY * rep3["unmet"])
        pay2 = float(best["r"] * rep3["P"])
        best_exec = dict(
            r=float(best["r"]), n_exec=int(exec_n2.sum()),
            alloc=[int(x) for x in exec_n2], q=float(rep3["q"]),
            P=float(rep3["P"]), payment=pay2,
            unmet=float(rep3["unmet"]),
            tri_vehicle_km=float(rep3["tri_vehicle_km"].sum()),
            tricycle_layer=layer2,
            total=float(layer2 + pay2 + t28.truck_obj),
            end_total=float(rep3["end_total"]),
            max_total_resid=float(rep3["max_total_resid"]),
            budget_feasible=bool(pay2 <= BUDGET + 1e-9),
            schedule_totals=sch2["totals"])
        rows[cap_name] = dict(scan=scan, best_fixed_fleet=best_fixed,
                              best_exec=best_exec)
        print(f"{cap_name}: exec n={exec_n2.sum()} unmet={rep3['unmet']:.2f} "
              f"pay={pay2:.2f} total={best_exec['total']:.2f}", flush=True)
    out = dict(task="T031 tricycle-caliber payment ledger", date="2026-11-20",
               budget=BUDGET, grid=[float(grid[0]), float(grid[-1]),
                                    float(grid[1] - grid[0])],
               baseline=dict(n=int(start_n.sum()),
                             unmet=float(base["unmet"]),
                             tri_vehicle_km=float(
                                 base["tri_vehicle_km"].sum()),
                             total=base_cost),
               levels=rows,
               note=("joint conserving replay of T028 with an intra-zone "
                     "reroute layer; arr = Pcell @ served_new - q_src; "
                     "payment ledger r*P primary, increment r*q secondary; "
                     "executable fleet re-solved at the rerouted served "
                     "profile (two fixed-point passes)"))
    out["runtime_s"] = float(time.time() - t0)
    dump(out, "tricycle_payment.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=("truck", "tricycle", "all"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rmax", type=float, default=3.0)
    args = ap.parse_args()
    if args.stage in ("truck", "all"):
        truck_ledger(force=args.force)
    if args.stage in ("tricycle", "all"):
        g = np.round(np.arange(0.0, args.rmax + 1e-9, 0.05), 2)
        tricycle_ledger(force=args.force, grid=g)
