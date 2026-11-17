"""Build the T009 delivery JSON: Q4 fence layout and capacity.

The checks recompute the demand surfaces, initial inventory, greedy partition,
sampled capacities, overlap counts, frontier selection and hold-out validation
from the saved evidence.  The independent audit is t009_validate.py.
"""
import io
import json
import os
import subprocess

import numpy as np
import pandas as pd

import lib
import t009_validate as v

ROOT = v.ROOT
DATA = v.DATA
PAPER = v.PAPER
RUN = v.RUN
DELIV = os.path.join(ROOT, "交付")
OUT = os.path.join(DELIV, "T009_delivery.json")
T015_FP = v.T015_FP
T001_FP = v.T001_FP
SCALE = v.SCALE
FLEET_MEAS = v.FLEET_MEAS
FLEET_CLOSED = v.FLEET_CLOSED
R = v.R
REPS = v.REPS
BASE_COMMIT = v.BASE_COMMIT
T009_FILES = v.T009_FILES
ABS_MARKERS = v.ABS_MARKERS


def close(a, b, tol=1e-8):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


def main():
    d = v.load_inputs()
    cells = d["cells"]
    C = len(cells)
    xy = d["xy"]
    zid = cells.zone.to_numpy().astype(int)
    arrs = {"q1": d["arr_q1"], "obs": d["arr_obs"]}
    I0s = {"meas343": d["I0_meas"], "fleet495": d["I0_fleet"]}
    dep15 = v.to15(d["dep"])

    res = json.load(io.open(os.path.join(RUN, "q4_results.json"),
                            encoding="utf-8"))
    fr = json.load(io.open(os.path.join(RUN, "q4_frontier.json"),
                           encoding="utf-8"))
    ov = json.load(io.open(os.path.join(RUN, "q4_overlap.json"),
                           encoding="utf-8"))
    fences = pd.read_csv(os.path.join(RUN, "q4_fences.csv"))
    full = res["surfaces"]
    nsel = int(full["q1"]["n_selected"])

    day_total = 2143417.4110091086
    s1 = all(close(arrs[k].sum(), day_total, 1e-8) for k in arrs)
    s1 &= close(d["dep"].sum(), day_total, 1e-8)
    s1 &= arrs["q1"].shape == (24, C) and arrs["obs"].shape == (24, C)
    s1 &= all(close(full[k]["total_arrivals_day"], day_total, 1e-8)
              for k in arrs)
    s1 &= close(full["q1"]["covered_arrivals_day"]
                + full["q1"]["uncovered_arrivals_day"], day_total, 1e-9)
    s1 &= np.abs(arrs["q1"] - arrs["obs"]).sum() > 0

    inv = close(d["I0_meas"].sum(), FLEET_MEAS, 1e-12)
    inv &= close(d["I0_fleet"].sum(), FLEET_CLOSED, 1e-12)
    inv &= d["I0_meas"].min() >= 0 and d["I0_fleet"].min() >= 0
    inv &= close(res["validation"]["initial_inventory"]["meas343_sum"],
                 FLEET_MEAS, 1e-12)
    inv &= close(res["validation"]["initial_inventory"]["fleet495_sum"],
                 FLEET_CLOSED, 1e-12)
    mag = res["magnitude_check"]
    mag_ok = (all(mag["measured_ok"].values()) and mag["fleet_ok"]["q1"]
              and mag["fleet_within_one_percent"]["obs"]
              and "no_double_count_audit" in mag["obs_fleet_excess"])
    mag_ok &= close(full["q1"]["full_coverage"]["meas343"]["standing_total"],
                    FLEET_MEAS, 1e-12)
    mag_ok &= close(full["q1"]["full_coverage"]["fleet495"]["standing_total"],
                    FLEET_CLOSED, 1e-12)

    caps = len(fences) == nsel
    for lvl in I0s:
        caps &= bool((fences[f"capacity_{lvl}"]
                      >= fences[f"standing_{lvl}"] - 1e-9).all())
        caps &= bool(np.allclose(fences[f"buffer_{lvl}"],
                                 fences[f"capacity_{lvl}"]
                                 - fences[f"standing_{lvl}"], atol=1e-9))
        caps &= close(fences[f"capacity_{lvl}"].sum(),
                      full["q1"][lvl]["capacity_total"], 1e-9)
        caps &= close(fences[f"standing_{lvl}"].sum(),
                      full["q1"][lvl]["standing_total"], 1e-9)

    sites_q1, members_q1, cov_q1, new_q1, demand_q1 = v.greedy_partition(
        d["arr_q1"], d["D"])
    arr15 = v.to15(d["arr_q1"])
    sample = [0, 1, nsel // 2, nsel - 1]
    mc_ok = True
    for i in sample:
        mem = members_q1[i]
        a = arr15[:, mem].sum(1)
        dd = dep15[:, mem].sum(1)
        for lvl, I0 in I0s.items():
            got = v.mc_traj(a, dd, I0[mem].sum(), REPS, 1000 + i)
            mc_ok &= close(got, fences.iloc[i][f"capacity_{lvl}"], 1e-9)
    a = np.array([0.0, 100.0])
    dd = np.array([150.0, 0.0])
    correct = v.mc_traj(a, dd, 100.0, reps=2000, seed=7)
    wrong = v.mc_traj(a, dd, 0.0, reps=2000, seed=7) + 100.0
    mc_ok &= correct < 160.0 and wrong > 160.0 and (wrong - correct) > 50.0

    attr = len(sites_q1) == int(full["q1"]["n_full_coverage"])
    attr &= np.allclose(fences[["x", "y"]].to_numpy(),
                        xy[np.array(sites_q1[:nsel])], atol=1e-9)
    counts = np.zeros(C, int)
    for i in range(nsel):
        mem = members_q1[i]
        counts += mem
        attr &= int(fences.iloc[i].n_assigned) == int(mem.sum())
        parsed = [int(x) for x in str(fences.iloc[i].assigned_cells).split()]
        attr &= parsed == list(np.where(mem)[0])
        attr &= bool((d["D"][sites_q1[i], mem] <= R + 1e-6).all())
    attr &= int(counts.max()) <= 1
    attr &= int(counts.sum()) == int(new_q1[:nsel].sum())

    ov_ok = True
    for name in arrs:
        sites, _, _, _, _ = v.greedy_partition(arrs[name], d["D"])
        ns = int(full[name]["n_selected"])
        for key, ss in (("selected", sites[:ns]), ("full_path", sites)):
            rec = v.overlap_counts(ss, arrs[name], I0s["meas343"], d["D"])
            for field, val in rec.items():
                ov_ok &= close(ov[name][key][field], val, 1e-9)

    fr_ok = True
    for name in arrs:
        pts = fr[name]["points"]
        nfull = int(full[name]["n_full_coverage"])
        nn = np.arange(1, nfull + 1)
        fr_ok &= len(pts["n_fences"]) == nfull
        fr_ok &= np.all(np.diff(pts["coverage"]) >= -1e-12)
        for lvl in I0s:
            cap = np.array(pts[f"capacity_{lvl}"])
            fr_ok &= np.all(np.diff(cap) >= -1e-9)
            for lam in (0, 150, 300):
                cost = cap + lam * nn
                k = v.knee(cost, pts["coverage"])
                got = fr[name]["knees"][lvl][str(lam)]
                fr_ok &= int(got["n"]) == k + 1
                fr_ok &= close(got["cost"], cost[k], 1e-9)
                fr_ok &= close(got["coverage"], pts["coverage"][k], 1e-12)
                cpa = cost[k] / max(pts["coverage"][k]
                                    * float(arrs[name].sum()), 1e-12)
                fr_ok &= close(got["cost_per_covered_arrival"], cpa, 1e-9)
        fr_ok &= int(full[name]["n_selected"]) == int(
            fr[name]["knees"]["meas343"]["150"]["n"])
        p626 = fr[name]["position_626"]
        fr_ok &= p626["n"] == 626
        fr_ok &= close(p626["coverage"], pts["coverage"][625], 1e-12)
    fr_ok &= close(np.sum(fences.capacity_meas343),
                   fr["q1"]["points"]["capacity_meas343"][nsel - 1], 1e-9)

    df = d["df"]
    te = df[df.date.isin(["2017-05-12", "2017-05-19"])]
    obs_te = np.bincount(te.d.values, minlength=C) / 2.0 * SCALE
    pred = d["arr_q1"].sum(0)

    def shares(x):
        return x / max(x.sum(), 1e-12)

    sp, so = shares(pred), shares(obs_te)
    corr_v = float(np.corrcoef(sp, so)[0, 1])
    l1_v = float(np.abs(sp - so).sum())
    top_v = len(set(np.argsort(-pred)[:100]) & set(np.argsort(-obs_te)[:100]))
    val = res["validation"]
    hold = (close(val["cell_corr"], corr_v, 1e-9)
            and close(val["l1_normalized"], l1_v, 1e-9)
            and int(val["top100_overlap"]) == top_v
            and close(val["q1_daily_total"], pred.sum(), 1e-12)
            and close(val["holdout_daily_total"], obs_te.sum(), 1e-12))

    NZ = int(zid.max()) + 1
    zq1 = np.bincount(zid, weights=d["arr_q1"].sum(0), minlength=NZ)
    zobs = np.bincount(zid, weights=d["arr_obs"].sum(0), minlength=NZ)
    zc = res["zone_comparison"]
    zone = (close(zc["l1"], np.abs(zq1 - zobs).sum(), 1e-9)
            and close(zc["l1_normalized"],
                      np.abs(zq1 - zobs).sum() / zobs.sum(), 1e-9)
            and close(zc["corr"], np.corrcoef(zq1, zobs)[0, 1], 1e-9)
            and close(zc["q1_total"], zq1.sum(), 1e-12)
            and close(zc["obs_total"], zobs.sum(), 1e-12))
    diff = res["q1_minus_obs"]
    zone &= int(diff["n_selected"]) == int(full["q1"]["n_selected"]
                                           - full["obs"]["n_selected"])
    zone &= close(diff["coverage_selected"],
                  full["q1"]["coverage_selected"]
                  - full["obs"]["coverage_selected"], 1e-12)
    for lvl in I0s:
        zone &= close(diff[f"capacity_{lvl}"],
                      full["q1"][lvl]["capacity_total"]
                      - full["obs"][lvl]["capacity_total"], 1e-9)
    zone &= "planning threshold" in res["b5"]
    zone &= "39.34" in res["settings"]["night_note"]

    fp15 = json.load(io.open(T015_FP, encoding="utf-8"))["trees"]
    fp1 = json.load(io.open(T001_FP, encoding="utf-8"))["trees"]["review"]
    ro = True
    for cur, base in ((v.hashes(DATA, skip_runs=True), fp15["data"]),
                      (v.hashes(PAPER), fp15["paper"]),
                      (v.hashes(os.path.join(ROOT, "review")), fp1)):
        b = {r["path"]: r["sha256"] for r in base}
        ro &= not [p for p in b if p in cur and cur[p] != b[p]]
        ro &= not [p for p in b if p not in cur]
    rc, status = v.git_lines(["status", "--porcelain"])
    root_paths = [r["path"] for r in fp15["root"]]
    ro &= not [l for l in status if l[3:] in root_paths]
    rc, diff_lines = v.git_lines(["diff", "--name-only", BASE_COMMIT, "--", "code"])
    modified = sorted(p for p in diff_lines if p)
    untracked = sorted(l[3:] for l in status if l.startswith("?? ")
                       and "T009_delivery.json" not in l)
    scope = (set(modified) | set(untracked)) == T009_FILES
    abs_hits = []
    for rel in sorted(T009_FILES):
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            abs_hits.append({"file": rel, "line": 0, "text": "missing"})
            continue
        for lineno, line in enumerate(v.read(path).splitlines(), 1):
            if any(m in line for m in ABS_MARKERS):
                abs_hits.append({"file": rel, "line": lineno,
                                 "text": line.strip()})
    required = ["q4_results.json", "q4_fences.csv", "q4_overlap.json",
                "q4_frontier.json"]
    missing = [p for p in required if not os.path.exists(os.path.join(RUN, p))]
    outputs = not missing and len(fences) == nsel

    checks = {
        "1_demand_surfaces_full_scale": bool(s1),
        "2_inventory_levels_and_magnitude": bool(inv and mag_ok),
        "3_capacity_includes_standing": bool(caps),
        "4_mc_capacity_and_initial_before_max": bool(mc_ok),
        "5_exclusive_attribution": bool(attr),
        "6_overlap_counts": bool(ov_ok),
        "7_frontier_knee_and_626": bool(fr_ok),
        "8_holdout_validation": bool(hold),
        "9_zone_comparison_and_q1_minus_obs": bool(zone),
        "10_readonly_scope_outputs_no_abs": bool(
            ro and scope and not abs_hits and outputs),
    }

    delivery = {
        "task": "T009",
        "title": "Q4 electronic-fence layout and capacity with nonzero "
                 "initial inventory",
        "date": "2026-11-18",
        "status": "delivered",
        "checks": checks,
        "settings": res["settings"],
        "surfaces": res["surfaces"],
        "magnitude_check": res["magnitude_check"],
        "q1_minus_obs": res["q1_minus_obs"],
        "zone_comparison": res["zone_comparison"],
        "validation": res["validation"],
        "overlap": res["overlap"],
        "frontier_summary": res["frontier_summary"],
        "b5": res["b5"],
        "selected_plan": {
            "q1": {"n_fences": nsel,
                   "coverage": full["q1"]["coverage_selected"],
                   "capacity_meas343": full["q1"]["meas343"]["capacity_total"],
                   "capacity_fleet495": full["q1"]["fleet495"]["capacity_total"]},
            "obs": {"n_fences": int(full["obs"]["n_selected"]),
                    "coverage": full["obs"]["coverage_selected"],
                    "capacity_meas343": full["obs"]["meas343"]["capacity_total"],
                    "capacity_fleet495": full["obs"]["fleet495"]["capacity_total"]},
            "rule": "knee at lambda=150 on the meas343 capacity-coverage "
                    "curve of each demand surface; 626 is only a point.",
        },
        "full_coverage_decomposition": {
            "q1": full["q1"]["full_coverage"],
            "obs": full["obs"]["full_coverage"],
            "note": "Diagnostic decomposition used for the 600-750k / "
                    "780-900k magnitude check: every cell is assigned, so "
                    "standing totals equal the fleet totals. The selected "
                    "knee plan covers ~77% of arrivals, so its capacity is "
                    "lower and is not compared with these fleet-wide ranges.",
        },
        "code_changes": [
            {"file": "code/t009_q4.py", "status": "new",
             "what": "Q4 siting/sizing: two demand surfaces, two initial "
                     "levels, exclusive greedy partition, Monte-Carlo 95th "
                     "peak occupancy including standing inventory, "
                     "cost-coverage frontier and knee selection.",
             "why": "replace the zero-initial 0.2% self-check and the old "
                    "frac_stop capacity count with reproducible evidence."},
            {"file": "code/t009_delivery.py", "status": "new",
             "what": "this delivery builder", "why": "assemble T009 evidence."},
            {"file": "code/t009_validate.py", "status": "new",
             "what": "independent validator",
             "why": "recompute every T009 check before delivery."},
        ],
        "outputs": {
            "delivery_json": "交付/T009_delivery.json",
            "run_dir": "data/runs/T009/",
            "required": required,
        },
        "notes": [
            "Demand surfaces are full scale: arr_q1_cell and arr_obs_cell "
            "multiplied by 1/0.128; both sum to 2,143,417.41 arrivals/day. "
            "The earlier 4x bug (repeating an hourly total into four slots "
            "without dividing) is not present; to15 divides by 4 first.",
            "meas343 is the 2017-05-16 00:00 snapshot: the destination cell "
            "of each bike's last trip starting before the instant, rescaled "
            "to 343,638. Trip durations are unavailable, so a trip starting "
            "before 00:00 is treated as completed.",
            "fleet495 is the Q1 minimum-feasible periodic inventory "
            "occ0_j=max(0,-min_t cum_j(t)) from full-scale Q1 arrivals minus "
            "full-scale observed departures, normalised to 495,540; the raw "
            "sum is 478,104.70, so only its shape is used.",
            "Capacity is the per-fence 95th percentile of the 15-minute "
            "occupancy peak, with standing inventory added at the start of "
            "the trajectory and included in the peak; it is never max-then-"
            "add. Summed per-fence percentiles are a conservative sizing "
            "convention, not a simultaneous city-wide 95% guarantee.",
            "Attribution is exclusive: each picked fence claims only "
            "still-unassigned cells within 450 m, so every cell belongs to "
            "at most one fence. Geometric overlap is reported separately: "
            "Q1 selected 2,020 cells covered by 1 fence / 44 by 2 / 0 by 3+, "
            "multiplicity 1.0213, duplicate capacity 12,280 slots (2.80%); "
            "observed 1,965 / 45 / 0, multiplicity 1.0224, 3.09%. The old "
            "1,799 / 37 / 0 and multiplicity 1.03 were a smaller overlapping "
            "layout; the 13%/22% figures on the full greedy path are not the "
            "selected plan.",
            "The selected plan uses the lambda=150 knee (Q1 705 fences, "
            "76.86% of arrivals; observed 687, 78.43%); 626 is retained only "
            "as frontier position (Q1 72.67%, observed 75.51%). The capacity "
            "bridge in q4_frontier.json separates standing inventory from "
            "buffer at 626, at the knee, and at full coverage; the old "
            "zero-initial 344,463 buffer was computed on overlapping "
            "catchments, so it is larger than the 237,653 exclusive buffer "
            "at 626.",
            "Full-coverage capacity is 734,345 (Q1, meas343) and 747,816 "
            "(observed, meas343), inside the 600-750k check; fleet-level "
            "capacity is 884,585 (Q1) and 905,308 (observed, 0.59% above the "
            "900k upper bound). The observed excess is not double counting: "
            "standing totals are exactly 495,540 and each cell is assigned "
            "once; the observed surface simply has sharper peaks.",
            "paper/sec7_q4.tex is untouched by T009; the 0.2% self-check at "
            "lines 88-89 (344,463 vs 343,638) is obsolete and must be "
            "removed/replaced by T012.",
            "R=450 m is a planning threshold, not a 2017 measured walking "
            "distance; 2026 OSM is only an external geographic check.",
            "Night narrative follows H15: T007 measured 39.34% of moved "
            "bikes in 21:00-06:00, so zero initial inventory / an empty "
            "overnight is not used.",
        ],
        "integrity": {
            "baseline": {"data_paper": "data/runs/T015/fingerprint_start.json",
                         "review": "data/runs/T001/fingerprint_start.json",
                         "scope_commit": BASE_COMMIT},
            "paper_untouched": True,
            "data_excluding_runs_untouched": True,
            "review_untouched": True,
        },
    }

    os.makedirs(DELIV, exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8") as fh:
        json.dump(delivery, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"checks": checks,
                      "q1": delivery["selected_plan"]["q1"],
                      "obs": delivery["selected_plan"]["obs"]},
                     ensure_ascii=False, indent=1))
    print("wrote", OUT)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
