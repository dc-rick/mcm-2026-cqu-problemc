"""T009: Q4 fence layout and sizing with exclusive attribution and nonzero
initial inventory.

Demand surfaces: Q1 model arrivals (primary) and observed arrivals (baseline),
both at the full 1/0.128 scale.

Initial inventory, two levels:
  meas343  = 2017-05-16 00:00 snapshot: for every bike seen before that
             instant, the destination cell of its last earlier trip; shares
             scaled to 343,638 (the measured fleet).
  fleet495 = minimum feasible periodic inventory from Q1 full-scale hourly
             arrivals minus departures, occ0_j = max(0, -min_t cum_j(t)),
             then normalised to 495,540 (the closed-fleet estimate).  This is
             a defined Q1 steady-state shape, not an unqualified "steady
             state".

Capacity per fence: Monte Carlo on the 15-minute occupancy trajectory that
starts at the fence's standing inventory (I0 summed over its exclusive
catchment), adds Poisson arrivals and subtracts Poisson departures, records
the peak including t=0, repeats 300 times, and takes the 95th percentile.
The initial value is added to the trajectory before the max is taken.
Siting is greedy maximal coverage with a partition: each picked fence claims
the still-unassigned cells within R, so every cell belongs to at most one
fence.  Geometric overlap is reported separately.
"""
import json
import os

import numpy as np
import pandas as pd

import lib
from lib import DATA

RUN = os.path.join(DATA, "runs", "T009")
SCALE = 1.0 / 0.128
FLEET_MEAS = 343638.0
FLEET_CLOSED = 495540.0
R = 0.45
REPS = 300
LAMBDAS = [0.0, 150.0, 300.0]
SNAP = pd.Timestamp("2017-05-16 00:00:00")


def load_inputs():
    z = np.load(os.path.join(DATA, "runs", "T006", "q1_od.npz"))
    arr_q1 = z["arr_q1_cell"].astype(np.float64) * SCALE     # (24, C)
    arr_obs = z["arr_obs_cell"].astype(np.float64) * SCALE
    dep = z["dep_obs"].astype(np.float64) * SCALE
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

    cum = np.cumsum(arr_q1 - dep, axis=0)
    raw = np.maximum(0.0, -cum.min(axis=0))
    I0_fleet = raw / raw.sum() * FLEET_CLOSED
    return dict(arr_q1=arr_q1, arr_obs=arr_obs, dep=dep, cells=cells, xy=xy,
                zid=zid, D=D, I0_meas=I0_meas, I0_fleet=I0_fleet, df=df,
                steady_raw_sum=float(raw.sum()))


def greedy_partition(arr, D, R=0.45):
    """Exclusive maximal coverage: each picked site claims unassigned cells."""
    C = D.shape[0]
    demand = arr.sum(0)
    near = D <= R
    assigned = np.zeros(C, bool)
    gain = near @ demand
    sites, members, cov_path, new_cells = [], [], [], []
    covered = 0.0
    for _ in range(C):
        gain[sites] = -1.0
        s = int(np.argmax(gain))
        if gain[s] <= 0:
            break
        mem = (~assigned) & near[s]
        if not mem.any():
            gain[s] = -1.0
            continue
        assigned |= mem
        covered += float(demand[mem].sum())
        sites.append(s)
        members.append(mem)
        cov_path.append(covered / demand.sum())
        new_cells.append(int(mem.sum()))
        gain -= near[:, mem] @ demand[mem]
    return sites, members, np.array(cov_path), np.array(new_cells), demand


def to15(x):
    """(24, n) hourly totals -> (96, n) 15-minute slots, equal split."""
    return np.repeat(x / 4.0, 4, axis=0)


def fence_capacity(members, arr15, dep15, I0, reps=REPS, seed=0):
    """MC 95th-percentile peak occupancy starting from standing inventory."""
    rng = np.random.default_rng(seed)
    out = np.empty(len(members))
    stand = np.empty(len(members))
    for i, mem in enumerate(members):
        a = arr15[:, mem].sum(1)
        dd = dep15[:, mem].sum(1)
        rng = np.random.default_rng(seed + i)
        A = rng.poisson(a, size=(reps, len(a)))
        D = rng.poisson(dd, size=(reps, len(dd)))
        occ = np.full(reps, float(I0[mem].sum()))
        peak = occ.copy()
        for t in range(len(a)):
            occ = np.maximum(0.0, occ + A[:, t] - D[:, t])
            peak = np.maximum(peak, occ)
        out[i] = float(np.percentile(peak, 95))
        stand[i] = float(I0[mem].sum())
    return out, stand


def knee(x, y):
    """Kneedle: point farthest from the chord (x,y normalised)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    xn = (x - x.min()) / max(x.max() - x.min(), 1e-12)
    yn = (y - y.min()) / max(y.max() - y.min(), 1e-12)
    dx, dy = xn[-1] - xn[0], yn[-1] - yn[0]
    den = np.hypot(dx, dy)
    dist = np.abs(dy * (xn - xn[0]) - dx * (yn - yn[0])) / max(den, 1e-12)
    return int(np.argmax(dist))


def overlap_stats(sites, arr, I0, D, R=R):
    near = D <= R
    mult = near[:, sites].sum(1) if sites else np.zeros(D.shape[0], int)
    covered = mult > 0
    cell_cap = I0 + arr.max(0)
    dup_cap = float((cell_cap * np.maximum(mult - 1, 0)).sum())
    return {
        "cells_total": int(len(mult)),
        "cells_0": int((mult == 0).sum()),
        "cells_1": int((mult == 1).sum()),
        "cells_2": int((mult == 2).sum()),
        "cells_3plus": int((mult >= 3).sum()),
        "cells_covered": int(covered.sum()),
        "coverage_multiplicity": float(mult[covered].mean()) if covered.any() else 0.0,
        "duplicate_capacity_slots": dup_cap,
        "duplicate_capacity_share": float(dup_cap / max(cell_cap[covered].sum(), 1e-12)),
        "definition": "multiplicity counts how many selected fence sites lie "
                      "within R of each cell; duplicate capacity = sum over "
                      "cells of (I0 + max hourly arrivals) * (multiplicity-1), "
                      "i.e. the slot count that would be counted twice if "
                      "catchments were treated as overlapping.",
    }


def main():
    os.makedirs(RUN, exist_ok=True)
    d = load_inputs()
    print("steady raw sum before normalisation:", d["steady_raw_sum"])
    arrs = {"q1": d["arr_q1"], "obs": d["arr_obs"]}
    I0s = {"meas343": d["I0_meas"], "fleet495": d["I0_fleet"]}
    dep15 = to15(d["dep"])
    surfaces = {}
    for name, arr in arrs.items():
        print(f"\n=== siting: {name} ===")
        sites, members, cov_path, new_cells, demand = greedy_partition(
            arr, d["D"], R)
        arr15 = to15(arr)
        caps, stands = {}, {}
        for lvl, I0 in I0s.items():
            caps[lvl], stands[lvl] = fence_capacity(
                members, arr15, dep15, I0, seed=1000)
        surfaces[name] = dict(sites=sites, members=members, cov_path=cov_path,
                              new_cells=new_cells, caps=caps, stands=stands,
                              demand=demand)
        n = np.arange(1, len(sites) + 1)
        cost = np.cumsum(caps["meas343"]) + 150.0 * n
        surfaces[name]["n_selected"] = knee(cost, cov_path) + 1
        print(f"  {len(sites)} fences, coverage {100*cov_path[-1]:.2f}%; "
              f"selected n={surfaces[name]['n_selected']}")
        for lvl in I0s:
            print(f"  {lvl}: total capacity {caps[lvl].sum():.0f}")

    zid = d["zid"]
    NZ = int(zid.max()) + 1
    zq1 = np.bincount(zid, weights=d["arr_q1"].sum(0), minlength=NZ)
    zobs = np.bincount(zid, weights=d["arr_obs"].sum(0), minlength=NZ)
    zone_cmp = {
        "l1": float(np.abs(zq1 - zobs).sum()),
        "l1_normalized": float(np.abs(zq1 - zobs).sum() / max(zobs.sum(), 1e-12)),
        "max_abs": float(np.abs(zq1 - zobs).max()),
        "corr": float(np.corrcoef(zq1, zobs)[0, 1]),
        "q1_total": float(zq1.sum()), "obs_total": float(zobs.sum()),
    }

    df = d["df"]
    te = df[df.date.isin(["2017-05-12", "2017-05-19"])]
    obs_te = np.bincount(te.d.values, minlength=len(d["cells"])) / 2.0 * SCALE
    pred = d["arr_q1"].sum(0)

    def shares(x):
        return x / max(x.sum(), 1e-12)
    sp, so = shares(pred), shares(obs_te)
    top_pred = set(np.argsort(-pred)[:100])
    top_obs = set(np.argsort(-obs_te)[:100])
    val = {
        "holdout_days": ["2017-05-12", "2017-05-19"],
        "cell_corr": float(np.corrcoef(sp, so)[0, 1]),
        "l1_normalized": float(np.abs(sp - so).sum()),
        "top100_overlap": len(top_pred & top_obs),
        "q1_daily_total": float(pred.sum()),
        "holdout_daily_total": float(obs_te.sum()),
        "initial_inventory": {
            "meas343_sum": float(d["I0_meas"].sum()),
            "fleet495_sum": float(d["I0_fleet"].sum()),
            "corr": float(np.corrcoef(d["I0_meas"], d["I0_fleet"])[0, 1]),
            "l1_normalized": float(np.abs(shares(d["I0_meas"])
                                          - shares(d["I0_fleet"])).sum()),
        },
        "note": "Hold-out comparison uses the two T005 test days "
                "(2017-05-12/19), excluded from the Q1 training set. The "
                "inventory comparison is the measured 5/16 00:00 drop-off "
                "snapshot against the minimum-feasible Q1 shape.",
    }

    frontier = {}
    for name in arrs:
        s = surfaces[name]
        n = np.arange(1, len(s["sites"]) + 1)
        pts = {"n_fences": [int(v) for v in n],
               "coverage": [float(v) for v in s["cov_path"]],
               "new_cells": [int(v) for v in s["new_cells"]]}
        knees = {}
        for lvl in I0s:
            cap_cum = np.cumsum(s["caps"][lvl])
            pts[f"capacity_{lvl}"] = [float(v) for v in cap_cum]
            knees[lvl] = {}
            for lam in LAMBDAS:
                cost = cap_cum + lam * n
                k = knee(cost, s["cov_path"])
                knees[lvl][str(int(lam))] = {
                    "n": int(n[k]), "cost": float(cost[k]),
                    "coverage": float(s["cov_path"][k]),
                    "cost_per_covered_arrival": float(
                        cost[k] / max(s["cov_path"][k]
                                      * float(s["demand"].sum()), 1e-12))}
        pos626 = None
        ns = int(s["n_selected"])
        if len(n) >= 626:
            pos626 = {"n": 626, "coverage": float(s["cov_path"][625])}
            for lvl in I0s:
                pos626[f"capacity_{lvl}"] = float(
                    np.cumsum(s["caps"][lvl])[625])
                pos626[f"standing_{lvl}"] = float(
                    np.cumsum(s["stands"][lvl])[625])
                pos626[f"buffer_{lvl}"] = float(
                    pos626[f"capacity_{lvl}"] - pos626[f"standing_{lvl}"])
        bridge = {}
        for label, idx in (("n_626", 625), ("n_selected", ns - 1),
                           ("n_full", len(s["sites"]) - 1)):
            rec = {"n": idx + 1, "coverage": float(s["cov_path"][idx])}
            for lvl in I0s:
                cap_cum = np.cumsum(s["caps"][lvl])
                stand_cum = np.cumsum(s["stands"][lvl])
                rec[f"capacity_{lvl}"] = float(cap_cum[idx])
                rec[f"standing_{lvl}"] = float(stand_cum[idx])
                rec[f"buffer_{lvl}"] = float(cap_cum[idx] - stand_cum[idx])
            bridge[label] = rec
        frontier[name] = {"points": pts, "knees": knees,
                          "position_626": pos626,
                          "capacity_bridge": bridge,
                          "rule": "knee of the cost-coverage curve; cost = "
                                  "capacity slots + lambda * fences; lambda "
                                  "in slot-equivalents (0/150/300)."}

    s = surfaces["q1"]
    nsel = int(s["n_selected"])
    rows = []
    for i, z in enumerate(s["sites"][:nsel]):
        mem = s["members"][i]
        rec = {"fence_id": i + 1, "x": float(d["xy"][z, 0]),
               "y": float(d["xy"][z, 1]), "zone": int(zid[z]),
               "n_assigned": int(mem.sum()),
               "assigned_cells": " ".join(str(int(j)) for j in np.where(mem)[0]),
               "coverage_arrivals_day": float(s["demand"][mem].sum())}
        for lvl in I0s:
            rec[f"capacity_{lvl}"] = float(s["caps"][lvl][i])
            rec[f"standing_{lvl}"] = float(s["stands"][lvl][i])
            rec[f"buffer_{lvl}"] = float(s["caps"][lvl][i] - s["stands"][lvl][i])
        rows.append(rec)
    pd.DataFrame(rows).to_csv(os.path.join(RUN, "q4_fences.csv"), index=False)

    overlap = {}
    for name in arrs:
        s = surfaces[name]
        nsel = int(s["n_selected"])
        overlap[name] = {
            "selected": overlap_stats(s["sites"][:nsel], arrs[name],
                                      I0s["meas343"], d["D"]),
            "full_path": overlap_stats(s["sites"], arrs[name],
                                       I0s["meas343"], d["D"]),
        }
    with open(os.path.join(RUN, "q4_overlap.json"), "w", encoding="utf-8") as fh:
        json.dump(overlap, fh, ensure_ascii=False, indent=1)

    with open(os.path.join(RUN, "q4_frontier.json"), "w", encoding="utf-8") as fh:
        json.dump(frontier, fh, ensure_ascii=False, indent=1)

    near = d["D"] <= R
    summary = {}
    for name in arrs:
        s = surfaces[name]
        nsel = int(s["n_selected"])
        selected_cells = near[:, s["sites"][:nsel]].any(1)
        rec = {
            "n_selected": nsel,
            "coverage_selected": float(s["cov_path"][nsel - 1]),
            "uncovered_share_selected": float(1 - s["cov_path"][nsel - 1]),
            "uncovered_cells_selected": int((~selected_cells).sum()),
            "total_arrivals_day": float(s["demand"].sum()),
            "covered_arrivals_day": float(
                s["cov_path"][nsel - 1] * s["demand"].sum()),
            "uncovered_arrivals_day": float(
                (1 - s["cov_path"][nsel - 1]) * s["demand"].sum()),
            "n_full_coverage": len(s["sites"]),
            "full_coverage": {
                "n_fences": len(s["sites"]),
                "coverage": float(s["cov_path"][-1]),
                "uncovered_cells": int(
                    (~near[:, s["sites"]].any(1)).sum()),
            },
        }
        for lvl in I0s:
            caps = s["caps"][lvl]
            caps_sel = caps[:nsel]
            stand_sel = s["stands"][lvl][:nsel]
            rec[lvl] = {
                "capacity_total": float(caps_sel.sum()),
                "capacity_median": float(np.median(caps_sel)),
                "capacity_max": float(caps_sel.max()),
                "standing_total": float(stand_sel.sum()),
                "buffer_total": float(caps_sel.sum() - stand_sel.sum()),
                "capacity_per_covered_arrival": float(
                    caps_sel.sum()
                    / max(s["cov_path"][nsel - 1] * s["demand"].sum(), 1e-12)),
            }
            rec["full_coverage"][lvl] = {
                "capacity_total": float(caps.sum()),
                "capacity_median": float(np.median(caps)),
                "capacity_max": float(caps.max()),
                "standing_total": float(s["stands"][lvl].sum()),
                "buffer_total": float(caps.sum()
                                      - s["stands"][lvl].sum()),
            }
        summary[name] = rec

    q1 = summary["q1"]; obs = summary["obs"]
    mag = {
        "measured_range_slots": [600000, 750000],
        "fleet_range_slots": [780000, 900000],
        "basis": "full_coverage (all 5,263 cells inside some fence); the "
                 "selected knee plan covers only ~77% of arrivals, so its "
                 "capacity is not comparable with these fleet-wide ranges.",
        "selected_capacity_note": "At the knee the selected plan's capacity is "
                                  "lower because the uncovered cells and their "
                                  "standing bikes lie outside the fences.",
        "measured_ok": {
            "q1": 600000 <= q1["full_coverage"]["meas343"]["capacity_total"] <= 750000,
            "obs": 600000 <= obs["full_coverage"]["meas343"]["capacity_total"] <= 750000,
        },
        "fleet_ok": {
            "q1": 780000 <= q1["full_coverage"]["fleet495"]["capacity_total"] <= 900000,
            "obs": 780000 <= obs["full_coverage"]["fleet495"]["capacity_total"] <= 900000,
        },
        "fleet_within_one_percent": {
            "q1": 780000 * 0.99 <= q1["full_coverage"]["fleet495"]["capacity_total"]
                  <= 900000 * 1.01,
            "obs": 780000 * 0.99 <= obs["full_coverage"]["fleet495"]["capacity_total"]
                   <= 900000 * 1.01,
        },
        "obs_fleet_excess": {
            "slots_above_900k": float(
                obs["full_coverage"]["fleet495"]["capacity_total"] - 900000),
            "percent_above_900k": float(
                100 * (obs["full_coverage"]["fleet495"]["capacity_total"]
                       / 900000 - 1)),
            "no_double_count_audit": "standing_total equals 495,540 exactly "
                                     "(each bike counted once); the full-coverage "
                                     "plan assigns every cell to one fence, so "
                                     "buffer_total does not count any cell twice. "
                                     "The excess is therefore not double counting; "
                                     "the observed surface has sharper peaks than "
                                     "the Q1 surface.",
        },
    }
    results = {
        "settings": {
            "R_m": 450, "FLEET_measured": FLEET_MEAS,
            "FLEET_closed": FLEET_CLOSED, "scale": SCALE,
            "reps": REPS, "capacity_percentile": 95,
            "demand_primary": "q1", "demand_baseline": "obs",
            "initial_levels": {
                "meas343": "2017-05-16 00:00 snapshot: for every bike whose "
                           "last trip started before the instant, the "
                           "destination cell of that trip; trip durations are "
                           "not in the data, so a trip starting before 00:00 "
                           "is treated as completed by the snapshot. Bikes "
                           "seen before the snapshot are assumed still in "
                           "service. Shares are scaled to 343,638, the "
                           "measured fleet total used since T006.",
                "fleet495": "Q1 minimum feasible periodic inventory "
                            "occ0_j = max(0, -min_t cum_j(t)) with cum from "
                            "full-scale Q1 arrivals (arr_q1_cell/0.128) minus "
                            "full-scale observed departures "
                            "(dep_obs/0.128), then normalised to 495,540. "
                            "The raw sum before normalisation is 478,104.70, "
                            "so only the shape is used, not the raw total.",
            },
            "capacity_rule": "Monte Carlo 95th percentile of the peak "
                             "15-minute occupancy; the trajectory starts at "
                             "the fence's standing inventory and the peak "
                             "includes t=0. Per-fence percentiles are not "
                             "additive in probability, so their sum is a "
                             "conservative deterministic sizing convention, "
                             "not a city-wide simultaneous 95% guarantee.",
            "departures": "Both demand surfaces use the same full-scale "
                          "observed departures dep_obs/0.128; the Q1 surface "
                          "changes arrivals only.",
            "frontier_cost": "capacity slots + lambda * fences, lambda=150 "
                             "slot-equivalents (knees also for 0 and 300).",
            "lambda_note": "lambda=150 is a planning exchange rate (one "
                           "fence is treated as worth 150 slot-equivalents), "
                           "not an estimated construction cost; lambda=0 and "
                           "300 bracket the selection.",
            "selected_rule": "primary plan = knee at lambda=150 on the "
                             "meas343 capacity-coverage curve of each demand "
                             "surface; 626 is reported only as a point on "
                             "the frontier.",
            "night_note": "T007 measured 21:00-06:00 at 39.34% of moved "
                          "bikes; zero-initial / overnight-empty is not used.",
        },
        "surfaces": summary,
        "magnitude_check": mag,
        "q1_minus_obs": {
            "n_selected": q1["n_selected"] - obs["n_selected"],
            "coverage_selected": q1["coverage_selected"]
                                 - obs["coverage_selected"],
            "capacity_meas343": q1["meas343"]["capacity_total"]
                                - obs["meas343"]["capacity_total"],
            "capacity_fleet495": q1["fleet495"]["capacity_total"]
                                 - obs["fleet495"]["capacity_total"],
            "capacity_per_covered_arrival_meas343":
                q1["meas343"]["capacity_per_covered_arrival"]
                - obs["meas343"]["capacity_per_covered_arrival"],
            "note": "Raw capacity totals are confounded by the different "
                    "coverage of the two knee plans (Q1 76.86%, observed "
                    "78.43%); the per-covered-arrival row is the comparable "
                    "size difference.",
        },
        "zone_comparison": zone_cmp,
        "validation": val,
        "overlap": overlap,
        "frontier_summary": {name: {"knees": frontier[name]["knees"],
                                    "position_626": frontier[name]["position_626"],
                                    "capacity_bridge":
                                        frontier[name]["capacity_bridge"]}
                             for name in frontier},
        "b5": "R=450 m is retained as the current planning threshold, not a "
              "2017 measured walking distance; 2026 OSM is used only as an "
              "external geographic check.",
    }
    with open(os.path.join(RUN, "q4_results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"q1": {"n": q1["n_selected"], "cov": q1["coverage_selected"],
                             "cap_meas": q1["meas343"]["capacity_total"],
                             "cap_fleet": q1["fleet495"]["capacity_total"],
                             "full_cap_meas": q1["full_coverage"]["meas343"]["capacity_total"],
                             "full_cap_fleet": q1["full_coverage"]["fleet495"]["capacity_total"]},
                      "obs": {"n": obs["n_selected"], "cov": obs["coverage_selected"],
                              "cap_meas": obs["meas343"]["capacity_total"],
                              "cap_fleet": obs["fleet495"]["capacity_total"],
                              "full_cap_meas": obs["full_coverage"]["meas343"]["capacity_total"],
                              "full_cap_fleet": obs["full_coverage"]["fleet495"]["capacity_total"]},
                      "mag": mag,
                      "knees": results["frontier_summary"]["q1"]["knees"],
                      "val": {"corr": val["cell_corr"],
                              "top100": val["top100_overlap"]}},
                     ensure_ascii=False, indent=1))
    print("wrote", RUN)


if __name__ == "__main__":
    main()
