"""T008: rebuild the Q3 executable relocation pool.

Pool definition (task sheet):
  target index  : (c, h_c), where h_c is cell c's own deficit hour
                  argmax_t (departures - arrivals)
  candidate     : a trip ending at j != c in hour h_c, with dist(j,c) <= R
  source cap    : A[j,h_c], the full-scale arrivals at j in that hour
  one reroute   : the assignment LP uses each source hour's trips at most once

For a reward r the incremental acceptance probability is
  w(d,r) = pi(d,r) - pi(d,0),  pi = logistic(kappa*r - rho*tau(d) - c0),
and the per-hour assignment solves
  max sum_e w_e x_e
  s.t. sum_{c} x_{jc} <= A[j,h]          (source, one reroute per trip)
       sum_{j} w_e x_e <= Delta_c        (target expected placements)
       x >= 0
The LP is solved with HiGHS on a sparse matrix.  q(r) is the sum of the
objective values over the 24 hourly subproblems.

Outputs: data/runs/T008/q3_pool.npz, q3_qr.json, q3_results.json,
q3_pool_audit.json.
"""
import json
import os

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

import lib
from lib import DATA

RUN = os.path.join(DATA, "runs", "T008")
SCALE = 1.0 / 0.128
R = 0.45
R_KM = R
KAPPA, RHO, C0 = 2.0, 0.35, 0.8
V_BIKE, WALK_M_MIN = 10.0, 80.0
TRUCK_MAIN = 82194.7968 / 24679.2791
TRUCK_LOW = 46194.7968 / 24679.2791
PARAMS = [
    (2.0, 0.35, 0.8),
    (1.0, 0.35, 0.8),
    (3.0, 0.35, 0.8),
    (2.0, 0.20, 0.8),
    (2.0, 0.50, 0.8),
    (2.0, 0.35, 0.4),
    (2.0, 0.35, 1.2),
]


def load_data():
    cells = lib.get_cells()
    C = len(cells)
    xy = cells[["x", "y"]].to_numpy()
    D = np.load(os.path.join(DATA, "dist_matrix.npy")).astype(np.float64)
    df = lib.load_trips(lib.TYP_DAYS)
    df = lib.assign_cells(df, cells)
    df = df[df.o.notna() & df.d.notna()].copy()
    df["o"] = df.o.astype(int)
    df["d"] = df.d.astype(int)
    hour = df.starttime.dt.hour.values
    nd = len(lib.TYP_DAYS)
    arr = np.zeros((24, C))
    dep = np.zeros((24, C))
    arx = np.zeros((24, C))
    np.add.at(arr, (hour, df.d.values), 1.0)
    np.add.at(dep, (hour, df.o.values), 1.0)
    diff = df.o.values != df.d.values
    np.add.at(arx, (hour[diff], df.d.values[diff]), 1.0)
    arr = arr / nd * SCALE
    dep = dep / nd * SCALE
    arx = arx / nd * SCALE
    return cells, xy, D, arr, dep, arx


def build_edges(arr, D, need, tstar, pos):
    near = D <= R_KM
    cache = {}
    for h in range(24):
        tg = np.where(pos & (tstar == h))[0]
        src = np.where(arr[h] > 0)[0]
        if len(tg) == 0 or len(src) == 0:
            cache[h] = None
            continue
        sub = near[np.ix_(src, tg)].copy()
        for k, c in enumerate(tg):
            j = np.where(src == c)[0]
            if len(j):
                sub[j[0], k] = False
        nz = np.argwhere(sub)
        si, ti = nz[:, 0], nz[:, 1]
        cache[h] = (src, tg, si, ti, D[src[si], tg[ti]])
    return cache


def tau_min(d):
    return d / V_BIKE * 60.0 + d * 1000.0 / WALK_M_MIN


def acceptance(r, d, kappa=KAPPA, rho=RHO, c0=C0):
    return 1.0 / (1.0 + np.exp(-(kappa * r - rho * tau_min(d) - c0)))


def q_curve(rs, cache, arr, need, params):
    kappa, rho, c0 = params
    out = np.zeros(len(rs))
    statuses = {"optimal": 0, "other": 0}
    for idx, r in enumerate(rs):
        total = 0.0
        for h, e in cache.items():
            if e is None:
                continue
            src, tg, si, ti, d = e
            w = acceptance(r, d, kappa, rho, c0) \
                - acceptance(0.0, d, kappa, rho, c0)
            keep = w > 1e-12
            if not keep.any():
                continue
            si, ti, w = si[keep], ti[keep], w[keep]
            ne = len(w)
            rows = np.r_[si, len(src) + ti]
            cols = np.r_[np.arange(ne), np.arange(ne)]
            vals = np.r_[np.ones(ne), w]
            A = coo_matrix((vals, (rows, cols)),
                           shape=(len(src) + len(tg), ne)).tocsr()
            b = np.r_[arr[h, src], need[tg]]
            res = linprog(-w, A_ub=A, b_ub=b, bounds=(0, None),
                          method="highs")
            if res.status == 0:
                statuses["optimal"] += 1
                total += -float(res.fun)
            else:
                statuses["other"] += 1
        out[idx] = total
    return out, statuses


def solve_audit(r, cache, arr, need, params):
    kappa, rho, c0 = params
    max_src = 0.0
    max_tgt = 0.0
    max_offer = 0.0
    q = 0.0
    for h, e in cache.items():
        if e is None:
            continue
        src, tg, si, ti, d = e
        w = acceptance(r, d, kappa, rho, c0) \
            - acceptance(0.0, d, kappa, rho, c0)
        keep = w > 1e-12
        if not keep.any():
            continue
        si, ti, w = si[keep], ti[keep], w[keep]
        ne = len(w)
        A = coo_matrix((np.r_[np.ones(ne), w],
                        (np.r_[si, len(src) + ti],
                         np.r_[np.arange(ne), np.arange(ne)])),
                       shape=(len(src) + len(tg), ne)).tocsr()
        b = np.r_[arr[h, src], need[tg]]
        res = linprog(-w, A_ub=A, b_ub=b, bounds=(0, None), method="highs")
        if res.status != 0:
            continue
        x = res.x
        q += float(w @ x)
        src_use = np.bincount(si, weights=x, minlength=len(src))
        max_src = max(max_src, float(np.max(src_use / np.maximum(
            arr[h, src], 1e-12))))
        tgt_use = np.bincount(ti, weights=w * x, minlength=len(tg))
        max_tgt = max(max_tgt, float(np.max(tgt_use / np.maximum(
            need[tg], 1e-12))))
        max_offer = max(max_offer, float(x.sum()))
    return {"q": q, "max_source_usage_ratio": max_src,
            "max_target_usage_ratio": max_tgt,
            "max_hourly_offer_sum": max_offer}


def main():
    os.makedirs(RUN, exist_ok=True)
    cells, xy, D, arr, dep, arx = load_data()
    C = len(cells)
    net = arr - dep
    need = np.maximum(0.0, -net).max(0)
    tstar = np.argmax(-net, axis=0)
    pos = need > 0
    N = float(need.sum())
    cache = build_edges(arr, D, need, tstar, pos)

    targets = np.where(pos)[0]
    edges = []
    for h, e in cache.items():
        if e is None:
            continue
        src, tg, si, ti, d = e
        for a, b, dd in zip(si, ti, d):
            edges.append((int(src[a]), int(tg[b]), int(h), float(dd)))
    edge_src = np.array([e[0] for e in edges], dtype=np.int32)
    edge_dst = np.array([e[1] for e in edges], dtype=np.int32)
    edge_hour = np.array([e[2] for e in edges], dtype=np.int8)
    edge_dist = np.array([e[3] for e in edges], dtype=np.float64)
    edge_src_arrivals = arr[edge_hour, edge_src]
    edge_dst_need = need[edge_dst]

    naive_units = float(edge_src_arrivals.sum())
    group = {(int(h), int(s)) for s, h in zip(edge_src, edge_hour)}
    distinct_units = float(sum(arr[h, s] for h, s in group))
    same_units = float(arr[tstar[targets], targets].sum())

    np.savez(os.path.join(RUN, "q3_pool.npz"),
             edge_src=edge_src, edge_dst=edge_dst, edge_hour=edge_hour,
             edge_dist_km=edge_dist, edge_src_arrivals=edge_src_arrivals,
             edge_dst_need=edge_dst_need,
             target_cell=targets.astype(np.int32),
             target_hour=tstar[targets].astype(np.int8),
             target_need=need[targets],
             target_arrivals=arr[tstar[targets], targets],
             target_free_riders=arx[tstar[targets], targets],
             need=need, tstar=tstar,
             source_hour_arrivals=arr)

    rs_dense = np.round(np.arange(0.0, 6.0001, 0.01), 4)
    q_dense, status = q_curve(rs_dense, cache, arr, need, PARAMS[0])

    def pack_opt(rs, qs, unit, free=0.0):
        total = rs * (qs + free) + unit * (N - qs)
        i = int(np.argmin(total))
        j = min(i + 1, len(rs) - 1)
        if qs[i] <= 1e-12:
            mc = None
        elif j > i and qs[j] > qs[i]:
            mc = ((rs[j] * qs[j] - rs[i] * qs[i]) / (qs[j] - qs[i]))
        else:
            mc = None
        return {
            "r_star": float(rs[i]), "q_star": float(qs[i]),
            "share_of_need": float(qs[i] / N),
            "bill": float(rs[i] * qs[i]),
            "free_rider_bill": float(rs[i] * free),
            "truck_residual": float(unit * (N - qs[i])),
            "total": float(total[i]),
            "all_truck": float(unit * N),
            "saving": float(1.0 - total[i] / (unit * N)),
            "marginal_cost_at_star": None if mc is None else float(mc),
            "unit": float(unit),
        }

    targeted = pack_opt(rs_dense, q_dense, TRUCK_MAIN)
    free_units = float(arx[tstar[targets], targets].sum())
    public = pack_opt(rs_dense, q_dense, TRUCK_MAIN, free=free_units)
    TRUCK_OLD = 3.478
    public_old = pack_opt(rs_dense, q_dense, TRUCK_OLD, free=free_units)
    pos_mask = rs_dense > 0
    total_pos = (rs_dense[pos_mask] * (q_dense[pos_mask] + free_units)
                 + TRUCK_MAIN * (N - q_dense[pos_mask]))
    ipos = int(np.argmax(-total_pos))
    idx_pos = np.where(pos_mask)[0][ipos]
    public_positive = {
        "r_star": float(rs_dense[idx_pos]),
        "q_star": float(q_dense[idx_pos]),
        "share_of_need": float(q_dense[idx_pos] / N),
        "bill": float(rs_dense[idx_pos] * q_dense[idx_pos]),
        "free_rider_bill": float(rs_dense[idx_pos] * free_units),
        "truck_residual": float(TRUCK_MAIN * (N - q_dense[idx_pos])),
        "total": float(total_pos[ipos]),
        "all_truck": float(TRUCK_MAIN * N),
        "saving": float(1.0 - total_pos[ipos] / (TRUCK_MAIN * N)),
        "unit": float(TRUCK_MAIN),
    }
    i12 = int(np.argmin(np.abs(rs_dense - 1.2)))
    total12 = (rs_dense[i12] * (q_dense[i12] + free_units)
               + TRUCK_MAIN * (N - q_dense[i12]))
    public_at_12 = {
        "r": float(rs_dense[i12]), "q": float(q_dense[i12]),
        "bill": float(rs_dense[i12] * q_dense[i12]),
        "free_rider_bill": float(rs_dense[i12] * free_units),
        "truck_residual": float(TRUCK_MAIN * (N - q_dense[i12])),
        "total": float(total12),
        "saving": float(1.0 - total12 / (TRUCK_MAIN * N)),
        "unit": float(TRUCK_MAIN),
    }

    # parameter / truck-unit sensitivity, coarser reward grid
    rs_coarse = np.round(np.arange(0.0, 6.0001, 0.05), 4)
    rows = []
    for params in PARAMS:
        qs, st = q_curve(rs_coarse, cache, arr, need, params)
        for unit_name, unit in (("main", TRUCK_MAIN), ("lower", TRUCK_LOW)):
            t = pack_opt(rs_coarse, qs, unit)
            p = pack_opt(rs_coarse, qs, unit, free=free_units)
            rows.append({"kappa": params[0], "rho": params[1],
                         "c0": params[2], "unit": unit_name,
                         "unit_value": float(unit),
                         "targeted": t, "public": p,
                         "lp_status": st})

    sav_t = [row["targeted"]["saving"] for row in rows]
    sav_p = [row["public"]["saving"] for row in rows]
    interval = {
        "targeted_saving_min": float(min(sav_t)),
        "targeted_saving_max": float(max(sav_t)),
        "public_saving_min": float(min(sav_p)),
        "public_saving_max": float(max(sav_p)),
        "r_star_min": float(min(row["targeted"]["r_star"] for row in rows)),
        "r_star_max": float(max(row["targeted"]["r_star"] for row in rows)),
        "q_star_min": float(min(row["targeted"]["q_star"] for row in rows)),
        "q_star_max": float(max(row["targeted"]["q_star"] for row in rows)),
        "sources": "7 acceptance parameter sets x {main, lower} truck unit",
    }

    audit = solve_audit(targeted["r_star"], cache, arr, need, PARAMS[0])
    q3_audit = {
        "R_m": R,
        "n_cells": int(C),
        "n_targets": int(pos.sum()),
        "need_total": N,
        "n_edges": int(len(edge_src)),
        "same_cell_excluded_units": same_units,
        "same_cell_share_of_naive_pair": float(
            same_units / max(naive_units, 1e-12)),
        "same_cell_share_of_naive_plus_same": float(
            same_units / max(naive_units + same_units, 1e-12)),
        "old_34_105_reproduced": bool(
            abs(same_units / max(naive_units, 1e-12) - 0.34105) < 0.002),
        "naive_pair_units": naive_units,
        "distinct_units": distinct_units,
        "duplicate_ratio": float(naive_units / distinct_units),
        "old_1_09_reproduced": bool(abs(naive_units / distinct_units
                                        - 1.09) < 0.02),
        "old_2_93_reproduced": bool(abs(naive_units / 113936.0 - 2.93) < 0.05),
        "naive_over_old_113936": float(naive_units / 113936.0),
        "duplicate_source": "a trip group (j,h) can be within R of several "
                            "target cells; the naive pair pool counts it once "
                            "per target.",
        "duplicate_resolution": "the LP constrains sum_c x[j,c,h] <= "
                                "A[j,h], so each source-hour trip group is "
                                "used at most once; q(r) is therefore <= "
                                "distinct_units.",
        "old_2_93_note": "the old 2.93 ratio compared the pair pool with "
                         "113,936 trips counted under each source's own "
                         "deficit hour; the corrected target-hour pool has "
                         "307,187.5 distinct units and a 1.0878 pair ratio.",
        "tstar_counts": [int(v) for v in np.bincount(tstar[pos], minlength=24)],
        "audit_at_r_star": {
            "r": targeted["r_star"],
            "q": float(audit["q"]),
            "q_over_distinct": float(audit["q"] / distinct_units),
            "max_source_usage_ratio": float(audit["max_source_usage_ratio"]),
            "max_target_usage_ratio": float(audit["max_target_usage_ratio"]),
            "max_hourly_offer_sum": float(audit["max_hourly_offer_sum"]),
            "one_reroute_per_trip": True,
        },
        "feasibility": {
            "q_le_distinct": bool(audit["q"] <= distinct_units * (1 + 1e-9)),
            "source_usage_le_arrivals": bool(
                audit["max_source_usage_ratio"] <= 1 + 1e-6),
            "target_usage_le_need": bool(
                audit["max_target_usage_ratio"] <= 1 + 1e-6),
            "same_cell_edges": 0,
        },
        "lp_status_full_curve": status,
    }

    qr = {
        "R_m": R,
        "r_grid": [float(v) for v in rs_dense],
        "q": [float(v) for v in q_dense],
        "bill": [float(r * q) for r, q in zip(rs_dense, q_dense)],
        "marginal_cost": [
            None if i == 0 or q_dense[i] <= q_dense[i - 1] else
            float((rs_dense[i] * q_dense[i] - rs_dense[i - 1] * q_dense[i - 1])
                  / (q_dense[i] - q_dense[i - 1]))
            for i in range(len(rs_dense))],
        "r_star": targeted["r_star"],
        "q_star": targeted["q_star"],
        "marginal_cost_at_star": targeted["marginal_cost_at_star"],
        "need_total": N,
        "distinct_candidate_trips": distinct_units,
        "acceptance": {"kappa": KAPPA, "rho": RHO, "c0": C0,
                       "form": "incremental pi(r)-pi(0), logit"},
        "solver": "scipy.optimize.linprog(method='highs'), one LP per hour",
        "status_counts": status,
    }

    results = {
        "settings": {
            "demand": "observed typical days (7 weekdays), full scale 1/0.128",
            "R_m": R,
            "need_definition": "Delta_c = max_t max(0, dep[c,t]-arr[c,t]); "
                               "h_c = argmax_t (dep-arr); N=sum Delta_c",
            "pool_index": "(target cell c, target deficit hour h_c)",
            "source_cap": "A[j,h_c] = full-scale arrivals at j in h_c",
            "acceptance": {"kappa": KAPPA, "rho": RHO, "c0": C0,
                           "incremental": True},
            "truck_unit_main": float(TRUCK_MAIN),
            "truck_unit_lower": float(TRUCK_LOW),
            "truck_unit_source": "T007 q1 alpha=1 K=60: main 82,194.7968 / "
                                 "24,679.2791; lower 46,194.7968 / "
                                 "24,679.2791 (fixed 600*60 removed). Old "
                                 "3.478 is obsolete.",
            "provisional": [],
        },
        "targeted": targeted,
        "public_push": public,
        "public_push_positive_reward": public_positive,
        "public_push_at_r_1_2": public_at_12,
        "public_push_old_unit_3_478": {
            "note": "B3's historical ~2.1% used c_truck=3.478 and a "
                    "non-flow pool; with the corrected pool but the old unit "
                    "the public-push saving is shown here.",
            **public_old,
        },
        "free_rider": {
            "units": free_units,
            "definition": "existing non-same-cell arrivals into target cells "
                          "in their deficit hours (arx[h_c,c]); these riders "
                          "would be paid under a public push without changing "
                          "their destination.",
            "bill_at_public_optimum": public["free_rider_bill"],
            "bill_at_positive_reward": public_positive["free_rider_bill"],
            "bill_at_old_unit_optimum": public_old["free_rider_bill"],
            "targeted_free_rider_bill": 0.0,
        },
        "interval": interval,
        "parameter_sensitivity": rows,
        "pool": {
            "naive_pair_units": naive_units,
            "distinct_units": distinct_units,
            "duplicate_ratio": float(naive_units / distinct_units),
            "same_cell_excluded_units": same_units,
            "same_cell_share": float(same_units / max(naive_units + same_units,
                                                      1e-12)),
            "n_edges": int(len(edge_src)),
        },
        "text_code_consistency": {
            "status": "code changed to match the section text",
            "old": "each source cell's own deficit hour supplied the candidate "
                   "count (H16 wrong-hour path)",
            "new": "candidate arrivals are counted in the target cell's "
                   "deficit hour h_c for both the target and its sources",
        },
        "notes": [
            "The pool excludes j=c: no same-cell trip can be counted as a "
            "redirection. The same-cell units removed are reported in "
            "q3_pool_audit.json.",
            "One reroute per trip is enforced by the source constraint "
            "sum_c x[j,c,h] <= A[j,h]; the LP never counts a source-hour "
            "group more than once.",
            "The 1.09 duplicate ratio is the naive pair count over distinct "
            "source-hour trips; the LP resolves it exactly.",
            "Public push pays the free riders in addition to the induced "
            "placements; its bill and saving are reported separately and are "
            "the lower-bound scenario required by B3. With the corrected "
            "TRUCK_UNIT=3.3305 the public push is not profitable (best saving "
            "0, at r=0); the best positive-reward point loses money. The old "
            "3.478 unit gives about 1% with this flow pool, not the historical "
            "2.1% (that figure used a non-flow pool); both are reported.",
            "Acceptance parameters are not identified from May 2017 (no "
            "reward was offered); the interval row sweeps kappa/rho/c0 and "
            "the two truck units.",
            "No paper/review/data file outside data/runs/T008 is written.",
        ],
        "outputs": {
            "pool": "data/runs/T008/q3_pool.npz",
            "qr": "data/runs/T008/q3_qr.json",
            "results": "data/runs/T008/q3_results.json",
            "audit": "data/runs/T008/q3_pool_audit.json",
        },
    }

    with open(os.path.join(RUN, "q3_qr.json"), "w", encoding="utf-8") as fh:
        json.dump(qr, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(RUN, "q3_results.json"), "w",
              encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(RUN, "q3_pool_audit.json"), "w",
              encoding="utf-8") as fh:
        json.dump(q3_audit, fh, ensure_ascii=False, indent=1)

    print(json.dumps({
        "N": N, "edges": len(edge_src), "naive": naive_units,
        "distinct": distinct_units, "dup": naive_units / distinct_units,
        "same": same_units,
        "targeted": targeted, "public": public,
        "interval": interval,
        "audit": q3_audit["audit_at_r_star"],
    }, ensure_ascii=False, indent=1))
    print("wrote", RUN)


if __name__ == "__main__":
    main()
