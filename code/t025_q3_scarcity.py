"""T025: does the Q3 incentive pay off when the truck fleet is scarce?

T020 showed that at K = 60 trucks (zone-level unmet 0) the incentive is a
net cost.  T025 asks the complementary question: shrink the fleet until
zone-level scarcity is real, and see whether the incentive substitutes for
trucks at some K.  Same caliber as T020 throughout: T018 integer MILP,
event order, bike integer loads, training-day demand, I0 = 343,638, the
same cost constants; incentive LP identical to T020 and PZ re-fitted by the
same IPF.

Stages: fleet | hybrid | breakeven | check | all
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
import t018_q2_int as t18
import t020_q3_budget as t20

RUN = os.path.join(lib.DATA, "runs", "T025")
os.makedirs(RUN, exist_ok=True)

K_SCAN = [0, 2, 5, 10, 20]
K_REG = [20, 40, 60]
# R1: the pure-truck unmet at K=10/20 (64.4 / 95.0) sat inside the 1% MIP gap,
# so K=5/10/20 are re-solved at 0.1% for both the pure and the hybrid models.
K_TIGHT = [5, 10, 20]
SOURCES = ["surplus", "arrivals"]
BUDGETS = [30000.0, 60000.0, 100000.0, None]
TIME_LIMIT = 300
MIP_GAP = 0.01
MIP_GAP_TIGHT = 0.001


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
def run_fleet(force=False):
    path = os.path.join(RUN, "fleet_scan.json")
    if os.path.exists(path) and not force:
        print("reuse", path)
        return load_json("fleet_scan.json")
    t18.setup_design()
    DEP, PZ, NET, I0 = t18.DEP, t18.PZ, t18.NET, t18.I0
    rows = []
    for K in sorted(set(K_SCAN) | set(K_REG)):
        t0 = time.time()
        res = t18.solve(DEP, t18.ARR, PZ, NET, I0, K=K, integer=True,
                        time_limit=TIME_LIMIT, mip_gap=MIP_GAP,
                        event_order=True, bike_integer=True)
        if res is None:
            rows.append(dict(K=K, status="solve_failed"))
            continue
        rows.append(dict(
            K=K, obj=float(res["obj"]), variable=float(res["obj_variable"]),
            unmet=float(res["unmet"]), dev=float(res["dev"]),
            truck_km=float(res["truck_km"]), moved=float(res["moved"]),
            trucks_used=float(res["trucks_used"]),
            peak_moved=float(res["peak_moved"]), peak_hour=int(res["peak_hour"]),
            runtime=float(time.time() - t0), gap=float(res["gap"]),
            status=int(res["status"])))

    # regression against T018's saved integer solutions
    reg = {}
    for K in K_REG:
        p = os.path.join(lib.DATA, "runs", "T018", f"result_int_K{K}.json")
        with open(p, encoding="utf-8") as fh:
            old = json.load(fh)
        new = next(r for r in rows if r["K"] == K)
        reg[K] = {k: dict(t018=float(old[k]), t025=float(new[k]),
                          abs_diff=float(abs(old[k] - new[k])))
                  for k in ("obj", "unmet", "dev", "truck_km", "moved")}

    # R1: tight-gap re-solve of the non-monotone pure-truck points.  K=0 is an
    # LP and K=2 stays at the coarse gap per the review, so only K=5/10/20 are
    # re-solved here.  Both sets are kept: the 1% unmet is the upper bound the
    # solver returned, the 0.1% value is the one the paper should quote.
    tight = {}
    keys = ("obj", "unmet", "dev", "truck_km", "moved",
            "trucks_used", "peak_moved", "gap", "status")
    for K in K_TIGHT:
        t0 = time.time()
        res = t18.solve(DEP, t18.ARR, PZ, NET, I0, K=K, integer=True,
                        time_limit=TIME_LIMIT, mip_gap=MIP_GAP_TIGHT,
                        event_order=True, bike_integer=True)
        coarse = next(r for r in rows if r["K"] == K)
        coarse_sub = {k: coarse[k] for k in
                      keys + ("variable", "peak_hour")}
        if res is None:
            tight[str(K)] = dict(coarse=coarse_sub, tight=None,
                                 status="solve_failed")
            continue
        tr = {k: float(res[k]) for k in keys}
        tr["variable"] = float(res["obj_variable"])
        tr["peak_hour"] = int(res["peak_hour"])
        tr["status"] = int(res["status"])
        tr["runtime"] = float(time.time() - t0)
        tight[str(K)] = dict(
            coarse=coarse_sub, tight=tr,
            unmet_upper_bound_1pct=float(coarse["unmet"]),
            unmet_tight=float(res["unmet"]),
            obj_delta=float(res["obj"] - coarse["obj"]),
            gap_achieved=float(res["gap"]),
            gap_was_inside_coarse_gap=bool(
                abs(res["obj"] - coarse["obj"])
                <= coarse["gap"] * max(abs(coarse["obj"]), 1.0) + 1e-6))
        print(f"tight K={K}: obj {tr['obj']:.4f} (coarse {coarse['obj']:.4f}) "
              f"unmet {tr['unmet']:.4f} (coarse {coarse['unmet']:.4f}) "
              f"gap {100 * tr['gap']:.3f}% in {tr['runtime']:.1f}s")

    scan = [r for r in rows if r["K"] in K_SCAN]
    unmet_by_K = {r["K"]: r["unmet"] for r in scan}
    for K in K_TIGHT:
        t = tight.get(str(K))
        if t and t.get("tight") is not None:
            unmet_by_K[K] = t["tight"]["unmet"]
    positive = [K for K in K_SCAN if unmet_by_K[K] > 1e-6]
    out = dict(
        ks=K_SCAN, rows=rows, regression_t018=reg,
        first_positive_unmet_K=int(min(positive)) if positive else None,
        first_positive_unmet_K_tight=(
            int(min(positive)) if positive else None),
        dev_share_at_first_positive=(
            float(next(r for r in scan
                       if r["K"] == min(positive))["dev"]) if positive else None),
        gap_tight=dict(
            gap=MIP_GAP_TIGHT, K=tight,
            note=("K in K_TIGHT re-solved at the tight gap; coarse is the 1% "
                  "gap row in rows. unmet_upper_bound_1pct is what the 1% "
                  "solve returned; unmet_tight is the 0.1% value. K=0 is an "
                  "LP (gap 0) and K=2 stays at 1%.")),
        note=("T018 integer MILP, K in {0,2,5,10,20}; K=40/60 re-run only for "
              "the regression check against data/runs/T018. Same demand, I0, "
              "event order, bike_integer loads and costs as T018."),
        runtime_s=float(sum(r.get("runtime", 0.0) for r in rows)
                        + sum(t.get("tight", {}).get("runtime", 0.0)
                              for t in tight.values()
                              if t.get("tight") is not None)))
    dump(out, "fleet_scan.json")
    return out


# --------------------------------------------------------------------------
def _policy_frames():
    """Solve the T020 LP once per (source, budget) and return the refitted PZ."""
    pol = t20.load_json("policy_table.json")
    pool = t20.load_pool()
    capinfo = t20.build_capacity(pool)
    lp = t20.PoolLP(pool, capinfo)
    frames = []
    for cap_name in SOURCES:
        caps = np.empty(lp.ns)
        for k, (h, s) in enumerate(lp.src_key_list):
            if cap_name == "arrivals":
                caps[k] = pool["arr"][h, s]
            else:
                caps[k] = max(0.0, pool["arr"][h, s] - pool["dep"][h, s])
        lp.b_src = caps
        for B in BUDGETS:
            row = next(z for z in pol["levels"][cap_name]["rows"]
                       if z["budget"] == B)
            got = lp.solve(row["r"], budget=B, cap_on=True, want_x=True)
            dA = t20.zone_delta(got["x"], lp.w, pool)
            frames.append(dict(source_cap=cap_name, budget=B, r=row["r"],
                               q=float(got["q"]), P=float(got["P"]),
                               invited=float(got["invited"]),
                               payment=float(got["payment"]),
                               dA=dA))
            print(f"  LP {cap_name} B={B}: r={row['r']} q={got['q']:.1f} "
                  f"pay={got['payment']:.1f}")
    return frames


def run_hybrid(force=False):
    path = os.path.join(RUN, "hybrid_by_K.json")
    if os.path.exists(path) and not force:
        print("reuse", path)
        return load_json("hybrid_by_K.json")
    fleet = load_json("fleet_scan.json")
    pure = {r["K"]: r for r in fleet["rows"]}
    k0 = fleet["first_positive_unmet_K"]
    k_above = min(k for k in sorted(set(K_SCAN) | set(K_REG)) if k > k0)
    # all five scan points: the unmet curve is non-monotone (K=5 serves all
    # zone-level departures while K=10/20 leave 64/95), so the break-even
    # question is asked at every K rather than only at the first scarce one.
    kset = sorted(set(K_SCAN) | {0, k0, k_above})
    print("hybrid K set", kset)

    t18.setup_design()
    DEP, ARR, NET, I0 = t18.DEP, t18.ARR, t18.NET, t18.I0
    frames = _policy_frames()
    # refit PZ once per policy (K-independent)
    for f in frames:
        PZ2 = np.empty_like(t18.PZ)
        errs = []
        for t in range(24):
            PZ2[t], re_, ce_ = t20.ipf_adjust(t18.PZ[t], DEP[:, t], f["dA"][t])
            errs.append((re_, ce_))
        f["PZ2"] = PZ2
        f["ipf_row_err_max"] = float(max(e[0] for e in errs))
        f["ipf_col_err_max"] = float(max(e[1] for e in errs))

    rows = []
    for K in kset:
        base = pure[K]
        for f in frames:
            res = t18.solve(DEP, ARR, f["PZ2"], NET, I0, K=K, integer=True,
                            time_limit=TIME_LIMIT, mip_gap=MIP_GAP,
                            event_order=True, bike_integer=True)
            if res is None:
                rows.append(dict(K=K, source_cap=f["source_cap"],
                                 budget=f["budget"], status="solve_failed"))
                continue
            total = float(res["obj"]) + f["payment"]
            removed = float(base["unmet"] - res["unmet"])
            rows.append(dict(
                K=K, source_cap=f["source_cap"], budget=f["budget"],
                r=f["r"], q=f["q"], P=f["P"], invited=f["invited"],
                payment=f["payment"], truck_obj=float(res["obj"]),
                truck_unmet=float(res["unmet"]), truck_dev=float(res["dev"]),
                truck_km=float(res["truck_km"]), truck_moved=float(res["moved"]),
                total=total, pure_obj=float(base["obj"]),
                pure_unmet=float(base["unmet"]), pure_dev=float(base["dev"]),
                diff=float(base["obj"] - total),
                diff_pct=float(100.0 * (base["obj"] - total) / base["obj"]),
                unmet_removed=removed,
                payment_ceiling=float(base["obj"] - res["obj"]),
                payment_per_unmet_removed=(
                    float(f["payment"] / removed) if removed > 1e-9 else None),
                ipf_row_err_max=f["ipf_row_err_max"],
                ipf_col_err_max=f["ipf_col_err_max"],
                gap_mode="coarse", gap=float(res["gap"])))
            print(f"hybrid K={K} {f['source_cap']} B={f['budget']}: "
                  f"truck={res['obj']:.1f} pay={f['payment']:.1f} "
                  f"total={total:.1f} pure={base['obj']:.1f} "
                  f"diff={base['obj'] - total:.1f}")

    # R1: same eight policies at the tight gap for K=5/10/20, compared against
    # the pure-truck tight-gap objective.  Coarse rows stay for the record.
    for K in K_TIGHT:
        t = fleet["gap_tight"]["K"].get(str(K))
        if not t or t.get("tight") is None:
            print(f"tight hybrid K={K}: no tight pure row, skipped")
            continue
        base_t = dict(obj=float(t["tight"]["obj"]),
                      unmet=float(t["tight"]["unmet"]),
                      dev=float(t["tight"]["dev"]))
        for f in frames:
            res = t18.solve(DEP, ARR, f["PZ2"], NET, I0, K=K, integer=True,
                            time_limit=TIME_LIMIT, mip_gap=MIP_GAP_TIGHT,
                            event_order=True, bike_integer=True)
            if res is None:
                rows.append(dict(K=K, source_cap=f["source_cap"],
                                 budget=f["budget"], gap_mode="tight",
                                 status="solve_failed"))
                continue
            total = float(res["obj"]) + f["payment"]
            removed = float(base_t["unmet"] - res["unmet"])
            rows.append(dict(
                K=K, source_cap=f["source_cap"], budget=f["budget"],
                r=f["r"], q=f["q"], P=f["P"], invited=f["invited"],
                payment=f["payment"], truck_obj=float(res["obj"]),
                truck_unmet=float(res["unmet"]), truck_dev=float(res["dev"]),
                truck_km=float(res["truck_km"]), truck_moved=float(res["moved"]),
                total=total, pure_obj=base_t["obj"],
                pure_unmet=base_t["unmet"], pure_dev=base_t["dev"],
                diff=float(base_t["obj"] - total),
                diff_pct=float(100.0 * (base_t["obj"] - total)
                               / base_t["obj"]),
                unmet_removed=removed,
                payment_ceiling=float(base_t["obj"] - res["obj"]),
                payment_per_unmet_removed=(
                    float(f["payment"] / removed) if removed > 1e-9 else None),
                ipf_row_err_max=f["ipf_row_err_max"],
                ipf_col_err_max=f["ipf_col_err_max"],
                gap_mode="tight", gap=float(res["gap"])))
            print(f"tight hybrid K={K} {f['source_cap']} B={f['budget']}: "
                  f"truck={res['obj']:.1f} pay={f['payment']:.1f} "
                  f"total={total:.1f} pure={base_t['obj']:.1f} "
                  f"diff={base_t['obj'] - total:.1f} gap={100*res['gap']:.3f}%")

    out = dict(
        ks=kset, k_first_positive_unmet=k0, rows=rows,
        gap_coarse=MIP_GAP, gap_tight=MIP_GAP_TIGHT, K_TIGHT=K_TIGHT,
        n_rows=len(rows),
        caliber=dict(
            truck_model="T018 integer MILP, event_order, bike_integer, K as "
                        "in fleet_scan",
            incentive_lp="T020 PoolLP, cap_on=True, r from T020 policy_table",
            pz_refit="T020 ipf_adjust, rows=departures, columns=new "
                     "zone-hour arrival totals",
            costs=dict(COST_KM=t18.COST_KM, COST_LOAD=t18.COST_LOAD,
                       PENALTY=t18.PENALTY, COST_DEV=t18.COST_DEV,
                       FIXED_K=t18.FIXED_K, Q_TRUCK=t18.Q_TRUCK),
            note=("total = T018 objective (including 600*K fixed) + payment; "
                  "payment_ceiling is the largest payment at which the mixed "
                  "policy would still beat pure truck at this K. Rows with "
                  "gap_mode=tight re-solve the same policy at 0.1% and use "
                  "the pure-truck 0.1% objective as the baseline; coarse rows "
                  "use the 1% gap on both sides.")),
        runtime_s=None)
    dump(out, "hybrid_by_K.json")
    return out


# --------------------------------------------------------------------------
def run_plans(force=False):
    """Re-solve the pure-truck scan and keep the plan arrays in this run dir."""
    tag_ok = all(os.path.exists(os.path.join(RUN, f"plan_int_K{K}.npz"))
                 for K in K_SCAN)
    if tag_ok and not force:
        print("reuse plan_int_K*.npz")
        return
    t18.setup_design()
    DEP, PZ, NET, I0 = t18.DEP, t18.PZ, t18.NET, t18.I0
    fleet = {r["K"]: r for r in load_json("fleet_scan.json")["rows"]}
    for K in K_SCAN:
        res = t18.solve(DEP, t18.ARR, PZ, NET, I0, K=K, integer=True,
                        time_limit=TIME_LIMIT, mip_gap=MIP_GAP,
                        event_order=True, bike_integer=True)
        if res is None:
            raise RuntimeError(f"K={K} failed")
        assert abs(res["obj"] - fleet[K]["obj"]) < 1e-6, (K, res["obj"])
        arrays = {k: res[k] for k in ("XM", "MK", "NN", "I", "U")}
        np.savez(os.path.join(RUN, f"plan_int_K{K}.npz"), **arrays)
        print("saved", os.path.join(RUN, f"plan_int_K{K}.npz"))


def _model_arrivals(PZ, DEP, U):
    served = DEP - U                       # (Z, T)
    arr = np.empty((24, PZ.shape[1]))
    for t in range(24):
        arr[t] = PZ[t].T @ served[:, t]
    return arr


def run_alignment(force=False):
    path = os.path.join(RUN, "mechanism.json")
    if os.path.exists(path) and not force:
        print("reuse", path)
        return load_json("mechanism.json")
    t18.setup_design()
    DEP, PZ = t18.DEP, t18.PZ
    frames = _policy_frames()
    out = dict(
        metric=("net[t,z] = model arrivals - departures in the pure-truck plan; "
                "aligned_share = (sum|net| - sum|net+dA|) / (2*sum positive dA); "
                "+1 = every cross-zone reroute reduces a zone-hour imbalance, "
                "0 = orthogonal, -1 = every reroute worsens one"),
        per_K=[])
    for K in K_SCAN:
        z = np.load(os.path.join(RUN, f"plan_int_K{K}.npz"))
        arr = _model_arrivals(PZ, DEP, z["U"])
        net = arr - DEP.T                      # (24, Z)
        base = float(np.abs(net).sum())
        rows = []
        for f in frames:
            dA = f["dA"]
            pos = float(np.maximum(dA, 0.0).sum())
            after = float(np.abs(net + dA).sum())
            rows.append(dict(
                source_cap=f["source_cap"], budget=f["budget"],
                q=float(f["q"]), cross_zone_bikes=pos,
                imbalance_before=base, imbalance_after=after,
                imbalance_change=after - base,
                aligned_share=float((base - after) / max(2 * pos, 1e-12))))
        out["per_K"].append(dict(K=K, imbalance_abs_sum=base, policies=rows))
        best = max(rows, key=lambda r: r["aligned_share"])
        print(f"K={K}: best aligned_share={best['aligned_share']:+.3f} "
              f"({best['source_cap']} B={best['budget']}), "
              f"imbalance {base:.1f} -> {best['imbalance_after']:.1f}")
    dump(out, "mechanism.json")
    return out


# --------------------------------------------------------------------------
def run_breakeven(force=False):
    path = os.path.join(RUN, "breakeven.json")
    if os.path.exists(path) and not force:
        print("reuse", path)
        return load_json("breakeven.json")
    fleet = load_json("fleet_scan.json")
    hyb = load_json("hybrid_by_K.json")
    pure = {r["K"]: r for r in fleet["rows"]}
    per_k = []
    for K in hyb["ks"]:
        rows_all = [r for r in hyb["rows"]
                    if r["K"] == K and r.get("status") != "solve_failed"]
        tight_rows = [r for r in rows_all if r.get("gap_mode") == "tight"]
        rows = tight_rows if tight_rows else rows_all
        gap_mode = "tight" if tight_rows else "coarse"
        if gap_mode == "tight":
            t = fleet["gap_tight"]["K"][str(K)]["tight"]
            pure_obj = float(t["obj"])
            pure_unmet = float(t["unmet"])
            pure_dev = float(t["dev"])
            gap_used = MIP_GAP_TIGHT
        else:
            pure_obj = float(pure[K]["obj"])
            pure_unmet = float(pure[K]["unmet"])
            pure_dev = float(pure[K]["dev"])
            gap_used = MIP_GAP
        best = max(rows, key=lambda r: r["diff"]) if rows else None
        ceilings = [r["payment_ceiling"] for r in rows]
        max_ceiling = float(max(ceilings)) if ceilings else None
        tol = float(gap_used * pure_obj)
        per_k.append(dict(
            K=K, gap_mode=gap_mode, gap_used=gap_used,
            pure_obj=pure_obj, pure_unmet=pure_unmet, pure_dev=pure_dev,
            best_policy=dict(source_cap=best["source_cap"],
                             budget=best["budget"], payment=best["payment"],
                             truck_obj=best["truck_obj"],
                             total=best["total"], diff=best["diff"],
                             unmet_removed=best["unmet_removed"],
                             payment_ceiling=best["payment_ceiling"],
                             gap=best.get("gap")) if best else None,
            max_payment_ceiling=max_ceiling,
            ceiling_tolerance=tol,
            ceiling_zero_within_tolerance=(
                bool(max_ceiling is not None and max_ceiling <= tol)),
            n_policies=len(rows),
            min_payment=float(min(r["payment"] for r in rows)) if rows else None,
            all_negative=bool(all(r["diff"] < 0 for r in rows)) if rows else None,
            coverage_ratio=(float(best["unmet_removed"] /
                                  max(pure_unmet, 1e-9))
                            if best else None),
            q_over_unmet=(float(best["q"] / max(pure_unmet, 1e-9))
                          if best else None)))
    any_positive = any(p["max_payment_ceiling"] is not None
                       and p["max_payment_ceiling"] > 0 for p in per_k)
    best_ceiling = max((p["max_payment_ceiling"] for p in per_k
                        if p["max_payment_ceiling"] is not None), default=None)
    mech = None
    mpath = os.path.join(RUN, "mechanism.json")
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as fh:
            mech = json.load(fh)
        for p in per_k:
            m = next(x for x in mech["per_K"] if x["K"] == p["K"])
            if p["best_policy"] is None:
                continue
            a = next(r for r in m["policies"]
                     if r["source_cap"] == p["best_policy"]["source_cap"]
                     and r["budget"] == p["best_policy"]["budget"])
            p["alignment_of_best"] = dict(
                aligned_share=a["aligned_share"],
                cross_zone_bikes=a["cross_zone_bikes"],
                imbalance_before=a["imbalance_before"],
                imbalance_after=a["imbalance_after"])
    out = dict(
        per_K=per_k,
        any_K_worthwhile=bool(any_positive),
        gap_modes={str(p["K"]): p["gap_mode"] for p in per_k},
        ceiling_statement=(
            "the payment ceiling pure_obj(K) - hybrid_truck_obj(K) is 0 or "
            "negative within solver tolerance at every K in the scan"
            if all(p["ceiling_zero_within_tolerance"] for p in per_k)
            else "at least one K has a positive payment ceiling outside the "
                 "solver tolerance"),
        conclusion=("incentive is worthwhile at some K in the scan"
                    if any_positive else
                    "within the fleet sizes compared (K = "
                    f"{hyb['ks']}), the incentive is never worthwhile: even a "
                    "zero payment leaves the mixed policy at or above the pure "
                    "truck cost at every K"),
        max_payment_ceiling=best_ceiling,
        ceiling_formula=("payment < pure_obj(K) - truck_obj_hybrid(K); "
                         "payment_ceiling in hybrid_by_K.json is the "
                         "right-hand side. A negative ceiling means no "
                         "non-negative payment can break even at that K. "
                         "K=5/10/20 use the 0.1% re-solve on both sides; "
                         "K=0/2 stay at 1%."),
        mechanism=None)
    # mechanism numbers at the first scarce K
    k0 = hyb["k_first_positive_unmet"]
    rows0 = [r for r in hyb["rows"] if r["K"] == k0]
    if rows0:
        best0 = max(rows0, key=lambda r: r["diff"])
        out["mechanism"] = dict(
            K=k0, pure_unmet=float(best0["pure_unmet"]),
            hybrid_unmet=float(best0["truck_unmet"]),
            unmet_removed=float(best0["unmet_removed"]),
            incentive_q=float(best0["q"]),
            q_over_unmet=float(best0["q"] / max(best0["pure_unmet"], 1e-9)),
            payment_per_unmet_removed=best0["payment_per_unmet_removed"],
            penalty_per_unmet=t18.PENALTY,
            note=("the incentive moves q cell-level reroutes; only the part "
                  "that changes a zone-hour arrival total can remove a "
                  "zone-level unmet trip, and zone-level unmet is what the "
                  "truck objective prices at 6 CNY."))
    dump(out, "breakeven.json")
    return out


# --------------------------------------------------------------------------
def run_check(force=False):
    path = os.path.join(RUN, "check.json")
    if os.path.exists(path) and not force:
        print("reuse", path)
        return load_json("check.json")
    fleet = load_json("fleet_scan.json")
    hyb = load_json("hybrid_by_K.json")
    be = load_json("breakeven.json")
    reg = fleet["regression_t018"]
    reg_ok = all(v["abs_diff"] < 1e-6 for K in reg for v in reg[K].values())
    budgets_ok = all(r["payment"] <= r["budget"] + 1e-6
                     for r in hyb["rows"]
                     if r.get("budget") is not None
                     and r.get("status") != "solve_failed")
    totals_ok = all(abs(r["total"] - (r["truck_obj"] + r["payment"])) < 1e-6
                    for r in hyb["rows"] if r.get("status") != "solve_failed")
    diff_ok = all(abs(r["diff"] - (r["pure_obj"] - r["total"])) < 1e-6
                  for r in hyb["rows"] if r.get("status") != "solve_failed")
    # R1 tight-gap checks
    tight_pure = fleet.get("gap_tight", {}).get("K", {})
    tight_pure_ok = all(
        tight_pure.get(str(K), {}).get("tight") is not None
        and tight_pure[str(K)]["tight"]["gap"] <= MIP_GAP_TIGHT + 1e-9
        for K in K_TIGHT)
    tight_unmet = {K: tight_pure[str(K)]["tight"]["unmet"]
                   if tight_pure.get(str(K), {}).get("tight") else None
                   for K in K_TIGHT}
    tight_hyb = [r for r in hyb["rows"]
                 if r.get("gap_mode") == "tight"
                 and r.get("status") != "solve_failed"]
    tight_hyb_ok = all(
        sum(1 for r in tight_hyb if r["K"] == K) == len(SOURCES) * len(BUDGETS)
        and all(r["gap"] <= MIP_GAP_TIGHT + 1e-9
                for r in tight_hyb if r["K"] == K)
        for K in K_TIGHT)
    ceiling_ok = all(p["ceiling_zero_within_tolerance"]
                     for p in be["per_K"])
    out = dict(regression_t018_ok=bool(reg_ok),
               regression_max_abs_diff=float(max(
                   v["abs_diff"] for K in reg for v in reg[K].values())),
               budgets_respected=bool(budgets_ok),
               total_identity_ok=bool(totals_ok),
               diff_identity_ok=bool(diff_ok),
               conclusion_any_worthwhile=bool(be["any_K_worthwhile"]),
               gap_tight_pure_ok=bool(tight_pure_ok),
               gap_tight_hybrid_ok=bool(tight_hyb_ok),
               gap_tight_hybrid_rows=int(len(tight_hyb)),
               tight_unmet_by_K=tight_unmet,
               ceiling_zero_within_tolerance=bool(ceiling_ok),
               ceiling_statement=be["ceiling_statement"],
               all_pass=bool(reg_ok and budgets_ok and totals_ok and diff_ok
                             and tight_pure_ok and tight_hyb_ok and ceiling_ok))
    dump(out, "check.json")
    return out


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=("fleet", "hybrid", "plans", "mechanism",
                             "breakeven", "check", "all"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    if args.stage in ("fleet", "all"):
        run_fleet(force=args.force)
    if args.stage in ("hybrid", "all"):
        run_hybrid(force=args.force)
    if args.stage in ("plans", "all"):
        run_plans(force=args.force)
    if args.stage in ("mechanism", "all"):
        run_alignment(force=args.force)
    if args.stage in ("breakeven", "all"):
        run_breakeven(force=True)
    if args.stage in ("check", "all"):
        run_check(force=True)
    print(f"done in {time.time() - t0:.1f}s")
