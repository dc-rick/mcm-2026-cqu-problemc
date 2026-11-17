"""Build the T006 delivery JSON: Q1 model wired into Q2/Q4.

Every check is recomputed from the files on disk; nothing is copied from the
run logs unless the log itself is the evidence.
"""
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, "code")
DATA = os.path.join(ROOT, "data")
PAPER = os.path.join(ROOT, "paper")
RUN = os.path.join(DATA, "runs", "T006")
DELIV = os.path.join(ROOT, "交付")
OUT = os.path.join(DELIV, "T006_delivery.json")
T015_FP = os.path.join(DATA, "runs", "T015", "fingerprint_start.json")
T001_FP = os.path.join(DATA, "runs", "T001", "fingerprint_start.json")

T006_FILES = {"code/12_q1_od.py", "code/20_q2_diag.py", "code/21_q2_opt.py",
              "code/22_q2_sim.py", "code/40_q4_efence.py",
              "code/t006_delivery.py", "code/t006_validate.py"}
ABS_MARKERS = ("C:" + chr(92) + "Users", "D:" + chr(92) + "code")


def read(path):
    with io.open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_hashes(base, skip_runs=False, only=None):
    out = {}
    if only is not None:
        for rel in only:
            path = os.path.join(ROOT, rel)
            if os.path.exists(path):
                out[rel] = sha(path)
        return out
    for path in Path(base).rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        rel = os.path.relpath(str(path), ROOT).replace(os.sep, "/")
        if skip_runs and rel.startswith("data/runs/"):
            continue
        out[rel] = sha(str(path))
    return out


def compare_fp(name, base, skip_runs=False, fp_path=T015_FP):
    fp = json.load(io.open(fp_path, encoding="utf-8"))["trees"]
    baseline = {row["path"]: row["sha256"] for row in fp[name]}
    current = tree_hashes(base, skip_runs=skip_runs)
    return {
        "baseline_files": len(baseline),
        "changed": sorted(p for p in baseline
                          if p in current and current[p] != baseline[p]),
        "missing": sorted(p for p in baseline if p not in current),
        "added": sorted(p for p in current if p not in baseline),
    }


def git_status():
    proc = subprocess.run(["git", "-c", "core.quotepath=false", "status",
                           "--porcelain"], cwd=ROOT, capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    return proc.returncode, proc.stdout.splitlines()


def main():
    z = np.load(os.path.join(RUN, "q1_od.npz"))
    cells = pd.read_parquet(os.path.join(DATA, "cells.parquet"))
    zid = cells.zone.to_numpy().astype(int)
    NZ = int(zid.max()) + 1
    od = z["od_q1"].astype(np.float64)
    arr_cell = z["arr_q1_cell"].astype(np.float64)
    dep_cell = z["dep_obs"].astype(np.float64)
    arr_obs = z["arr_obs_cell"].astype(np.float64)

    required = {
        "od_q1": ((24, 50, 50), np.float32),
        "arr_q1_cell": ((24, 5263), np.float32),
        "dep_obs": ((24, 5263), np.float32),
        "arr_obs_cell": ((24, 5263), np.float32),
        "band": ((24,), np.int8),
        "beta": ((5,), np.float32),
        "train_days": ((5,), np.dtype("<U10")),
        "typ_days": ((7,), np.dtype("<U10")),
        "sigma": ((), np.float32),
    }
    structure = {k: list(z[k].shape) == list(shape)
                 and z[k].dtype == dtype for k, (shape, dtype) in required.items()}
    keys_ok = set(z.files) == set(required)

    negatives = {k: int((z[k].astype(np.float64) < 0).sum())
                 for k in ("od_q1", "arr_q1_cell", "dep_obs", "arr_obs_cell")}

    dep_zone = np.stack([np.bincount(zid, weights=dep_cell[h], minlength=NZ)
                         for h in range(24)])
    rel = np.abs(od.sum(2) - dep_zone) / np.maximum(dep_zone, 1e-12)
    row_err = float(rel.max())

    dep_hour = dep_cell.sum(1)
    arr_hour = arr_cell.sum(1)
    cons_err = float(np.max(np.abs(arr_hour - dep_hour)
                            / np.maximum(dep_hour, 1e-12)))

    with io.open(os.path.join(RUN, "q1_vs_obs.json"), encoding="utf-8") as fh:
        q1v = json.load(fh)
    band_rows = q1v["top5"]["per_band"]
    band_diffs = {r["band"]: abs(r["top5"] - r["top5_reference"])
                  for r in band_rows}
    top5_ok = (q1v["top5"]["abs_diff"] < 0.005
               and q1v["top5"]["weighted"] > 0.42
               and max(band_diffs.values()) < 0.005)
    same = q1v["same_cell_share"]
    ks = q1v["distance_ks"]["ks"]
    diagnostics_ok = (0.0 <= same["observed"] <= 1.0
                      and 0.0 <= same["model"] <= 1.0
                      and 0.0 <= ks <= 1.0
                      and q1v["zone_od"]["total_obs"] > 0
                      and q1v["zone_od"]["l1"] >= 0)

    with io.open(os.path.join(RUN, "q2_q1_smoke.json"), encoding="utf-8") as fh:
        q2 = json.load(fh)
    q2_ok = (q2["q1"]["status"] == 2 and q2["obs"]["status"] == 2
             and q2["q1"]["alpha"] == 1.0 and q2["obs"]["alpha"] == 1.0
             and q2["q1"]["K"] == 60 and q2["obs"]["K"] == 60
             and q2["q1"]["fixed_K_cost"] == 600.0
             and q2["obs"]["fixed_K_cost"] == 600.0
             and q2["q1"]["unmet"] <= 1e-6 and q2["obs"]["unmet"] <= 1e-6)
    expected_trips = 2143417.4
    q2_ok &= all(abs(q2[s]["trips"] - expected_trips) / expected_trips < 1e-6
                 for s in ("q1", "obs"))
    d = q2["difference"]["q1_minus_obs"]
    q2_diff_ok = all(
        abs(d[k] - (q2["q1"][k] - q2["obs"][k]))
        <= 1e-9 * max(1.0, abs(d[k]))
        for k in ("moved", "truck_km", "unmet", "dev", "obj"))
    plan_q1 = np.load(os.path.join(RUN, "q2_plan_q1_smoke.npz"))
    plan_obs = np.load(os.path.join(RUN, "q2_plan_obs_smoke.npz"))
    plan_ok = (set(plan_q1.files) == {"X", "M", "I", "U", "XM"}
               and plan_q1["XM"].shape == (24, 50, 50)
               and plan_q1["X"].shape == (50, 24)
               and all(plan_obs[k].shape == plan_q1[k].shape
                       for k in plan_q1.files))

    src = {name: read(os.path.join(CODE, name))
           for name in ("20_q2_diag.py", "21_q2_opt.py", "22_q2_sim.py",
                        "40_q4_efence.py")}
    defaults_ok = all(
        'choices=("q1", "obs"), default="q1"' in src[n]
        for n in ("20_q2_diag.py", "21_q2_opt.py", "22_q2_sim.py"))
    defaults_ok &= 'choices=("q1", "obs"), default="q1"' in src["40_q4_efence.py"]

    with io.open(os.path.join(RUN, "q4_q1_smoke.json"), encoding="utf-8") as fh:
        q4 = json.load(fh)
    tot_q1 = q4["q1"]["arrivals_per_day"]
    tot_obs = q4["obs"]["arrivals_per_day"]
    q4_ok = (abs(tot_q1 - tot_obs) / tot_obs < 1e-6
             and q4["settings"]["radius_m"] == 450
             and q4["settings"]["reps"] == 300
             and q4["q1"]["fences"] > 0 and q4["obs"]["fences"] > 0
             and 0 < q4["q1"]["coverage"] < 1
             and 0 < q4["obs"]["coverage"] < 1
             and q4["difference"]["q1_minus_obs"]["total_capacity"] < 0)
    q4_files = all(os.path.exists(os.path.join(RUN, sub, name))
                   for sub in ("q4_q1", "q4_obs")
                   for name in ("q4_summary.json", "q4_frontier.npz",
                                "q4_sites.npy"))

    data_cmp = compare_fp("data", DATA, skip_runs=True)
    paper_cmp = compare_fp("paper", PAPER)
    review_cmp = compare_fp("review", os.path.join(ROOT, "review"),
                            fp_path=T001_FP)
    root_base = json.load(io.open(T015_FP, encoding="utf-8"))["trees"]["root"]
    root_paths = [row["path"] for row in root_base]
    root_now = tree_hashes(None, only=root_paths)
    root_changed_vs_fp = [p for p in root_paths
                          if root_now.get(p) != next(r["sha256"] for r in root_base
                                                     if r["path"] == p)]
    rc, status_lines = git_status()
    root_dirty = [line for line in status_lines
                  if line[3:] in root_paths]
    readonly_ok = (not data_cmp["changed"] and not data_cmp["missing"]
                   and not paper_cmp["changed"] and not paper_cmp["missing"]
                   and not review_cmp["changed"] and not review_cmp["missing"]
                   and not root_dirty)

    # Scope is measured against the last pre-T006 commit, so the check is
    # stable both before and after the delivery commit itself.
    base_commit = "6c130aa"
    diff_proc = subprocess.run(["git", "diff", "--name-only", base_commit,
                                "--", "code"], cwd=ROOT, capture_output=True,
                               text=True, encoding="utf-8", errors="replace")
    modified = sorted(p for p in diff_proc.stdout.splitlines() if p)
    untracked = sorted(line[3:] for line in status_lines
                       if line.startswith("?? ")
                       and "T006_delivery.json" not in line)
    expected_t006 = {"code/12_q1_od.py", "code/20_q2_diag.py",
                     "code/21_q2_opt.py", "code/22_q2_sim.py",
                     "code/40_q4_efence.py", "code/t006_delivery.py",
                     "code/t006_validate.py"}
    scope_ok = (set(modified) | set(untracked)) == expected_t006

    abs_hits = []
    for rel in sorted(T006_FILES):
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            abs_hits.append({"file": rel, "line": 0, "text": "missing"})
            continue
        for lineno, line in enumerate(read(path).splitlines(), 1):
            for marker in ABS_MARKERS:
                if marker in line:
                    abs_hits.append({"file": rel, "line": lineno,
                                     "text": line.strip()})

    required_outputs = [
        "data/runs/T006/q1_od.npz",
        "data/runs/T006/q1_vs_obs.json",
        "data/runs/T006/q2_q1_smoke.json",
        "data/runs/T006/q4_q1_smoke.json",
        "data/runs/T006/q2_smoke_q1.json",
        "data/runs/T006/q2_smoke_obs.json",
        "data/runs/T006/q2_plan_q1_smoke.npz",
        "data/runs/T006/q2_plan_obs_smoke.npz",
        "data/runs/T006/q2_state_q1.npz",
        "data/runs/T006/q4_q1/q4_summary.json",
        "data/runs/T006/q4_q1/q4_frontier.npz",
        "data/runs/T006/q4_q1/q4_sites.npy",
        "data/runs/T006/q4_obs/q4_summary.json",
        "data/runs/T006/q4_obs/q4_frontier.npz",
        "data/runs/T006/q4_obs/q4_sites.npy",
        "data/runs/T006/q4_q1_run.log",
        "data/runs/T006/q4_obs_run.log",
    ]
    outputs_missing = [p for p in required_outputs
                       if not os.path.exists(os.path.join(ROOT, p))]

    checks = {
        "1_npz_structure": keys_ok and all(structure.values()),
        "2_npz_nonnegative": all(v == 0 for v in negatives.values()),
        "3_row_sum_lt_1e-6": row_err < 1e-6,
        "4_arrival_conservation_lt_1e-6": cons_err < 1e-6,
        "5_top5_matches_validation": top5_ok,
        "6_same_cell_and_ks_reported": diagnostics_ok,
        "7_q2_smokes_optimal_and_default_q1": (q2_ok and q2_diff_ok
                                               and plan_ok and defaults_ok),
        "8_q4_smokes_run_and_totals_match": q4_ok and q4_files,
        "9_readonly_trees_unchanged": readonly_ok,
        "10_scope_outputs_and_no_abs_paths": (scope_ok and not outputs_missing
                                              and not abs_hits),
    }

    delivery = {
        "task": "T006",
        "title": "Wire the Q1 travel-behaviour model into Q2/Q4",
        "date": "2026-11-18",
        "status": "delivered",
        "revision": "R1",
        "checks": checks,
        "q1_model": {
            "bands": q1v["top5"]["per_band"],
            "top5_weighted": q1v["top5"]["weighted"],
            "top5_reference_weighted": q1v["top5"]["reference_weighted"],
            "top5_abs_diff": q1v["top5"]["abs_diff"],
            "tolerance": q1v["top5"]["tolerance"],
            "row_sum_max_rel_error": row_err,
            "row_sum_max_rel_error_reported_in_q1_vs_obs":
                q1v["row_sum_max_rel_error"],
            "arrival_total_max_rel_error": cons_err,
            "same_cell_share": same,
            "distance_ks": q1v["distance_ks"],
            "zone_od": q1v["zone_od"],
            "cell_arrivals": q1v["cell_arrivals"],
            "beta": z["beta"].tolist(),
            "train_days": z["train_days"].tolist(),
            "typ_days": z["typ_days"].tolist(),
            "sigma": float(z["sigma"]),
            "total_flow_sample_day": float(od.sum()),
            "total_flow_full_scale": float(od.sum() / 0.128),
        },
        "q2_smoke": {
            "settings": q2["settings"],
            "q1": q2["q1"],
            "obs": q2["obs"],
            "difference": q2["difference"],
            "plans": ["data/runs/T006/q2_plan_q1_smoke.npz",
                      "data/runs/T006/q2_plan_obs_smoke.npz"],
        },
        "q4_smoke": q4,
        "integrity": {
            "baseline": {
                "data_paper_code": "data/runs/T015/fingerprint_start.json",
                "review": "data/runs/T001/fingerprint_start.json",
            },
            "data_excluding_runs": data_cmp,
            "paper": paper_cmp,
            "review": review_cmp,
            "root_docs": {
                "paths": root_paths,
                "changed_vs_T015_baseline": root_changed_vs_fp,
                "dirty_in_git": root_dirty,
                "sha256_now": root_now,
                "note": "DATA.md/README.md changed after the T015 baseline but "
                        "were already committed and clean before T006 started "
                        "(mtime 00:54/00:02, runs/T006 created 01:03); T006 "
                        "wrote neither.",
            },
            "git_status_porcelain": [
                line[:3] + "交付/T006_delivery.json"
                if "T006_delivery.json" in line else line
                for line in status_lines],
        },
        "code_changes": [
            {"file": "code/12_q1_od.py", "status": "new",
             "what": "rebuild training-day D_j and A_j per band, form row-"
                     "normalised P(j|i,h), generate arr_q1_cell and od_q1, "
                     "save q1_od.npz and q1_vs_obs.json",
             "why": "the problem text requires Q2/Q4 to use the Q1 model."},
            {"file": "code/20_q2_diag.py", "status": "modified",
             "what": "argparse --od q1|obs and --out; q1 reads q1_od.npz and "
                     "defaults to data/runs/T006/q2_state_q1.npz",
             "why": "input switch without touching the tracked q2_state.npz."},
            {"file": "code/21_q2_opt.py", "status": "modified",
             "what": "argparse --od q1|obs (q1 default), --conserve ready for "
                     "T007; load_demand() reads the selected OD surface and "
                     "multiplies dep/arr by SCALE=1/0.128 (PZ stays "
                     "row-normalised)",
             "why": "input switch at the full 343,638-bike scale; equations "
                    "untouched for T006."},
            {"file": "code/22_q2_sim.py", "status": "modified",
             "what": "argparse --od q1|obs (q1 default); od_hr from q1_od.npz "
                     "or typical_day.npz",
             "why": "input switch; greedy capacity fix is T007."},
            {"file": "code/40_q4_efence.py", "status": "modified",
             "what": "argparse --arr q1|obs, --outdir, --radii, --reps; q1 "
                     "hourly demand split into four 15-min slots; writes only "
                     "under --outdir",
             "why": "Q4 must use the Q1 demand surface; tracked q4_* files "
                    "stay read-only."},
            {"file": "code/t006_delivery.py", "status": "new",
             "what": "this delivery builder", "why": "assemble T006 evidence."},
            {"file": "code/t006_validate.py", "status": "new",
             "what": "independent T006 delivery validator",
             "why": "recompute the self-checks before delivery."},
        ],
        "outputs": {
            "delivery_json": "交付/T006_delivery.json",
            "q1_od": "data/runs/T006/q1_od.npz",
            "q1_vs_obs": "data/runs/T006/q1_vs_obs.json",
            "q2_smoke": "data/runs/T006/q2_q1_smoke.json",
            "q4_smoke": "data/runs/T006/q4_q1_smoke.json",
            "required_outputs": required_outputs,
        },
        "notes": [
            "Q1 OD is built on the same five training days as T005 "
            "(2017-05-10/11/15/16/18); A_j is recomputed from those days and "
            "not stored, so P cannot be re-derived from q1_od.npz alone.",
            "The model over-predicts same-cell stays: 12.83% vs 8.89% "
            "observed (+3.94 pp). This must be written as an overestimate, "
            "not as agreement.",
            "The distance KS is 0.0524 with 0.5 km bins; the model and the "
            "observed histogram use the same 7 typical days.",
            "Q2 smoke runs use conserve=False: T006 was forbidden to change "
            "the LP equations, so these smoke numbers are not the final "
            "conservation result. H13 is T007.",
            "21_q2_opt.py still takes I0 from data/q2_state.npz, i.e. the "
            "observed last-end distribution, for both OD surfaces. That is "
            "the common initial state, not a demand input.",
            "20_q2_diag.py was run with --od q1 and wrote "
            "data/runs/T006/q2_state_q1.npz; the tracked data/q2_state.npz "
            "was not touched. Its net/cum/req arrays inherit the sampled "
            "demand scale, while I0 is full-scale (343,638); they are "
            "diagnostics at different scales, so multiply req by 1/0.128 "
            "before using it with I0 or any full-scale Q2 result.",
            "22_q2_sim.py was compiled and its --od switch was checked in the "
            "source, but not executed: it still writes data/q2_sim.json and "
            "reads data/q2_plan.npz, both tracked. T007 runs it with the new "
            "plan and conservation fix.",
            "Q4 splits each Q1 hourly total into four equal 15-min slots "
            "(divide by 4). An earlier version repeated the hourly total four "
            "times and produced 4x the observed daily arrivals (8,573,670 vs "
            "2,143,417); that run was discarded and the corrected runs agree "
            "to 1.4e-10 relative.",
            "Q4 Q1 demand is more diffuse than observed: 696 fences vs 626, "
            "coverage 76.4% vs 75.3%, but total 95th-percentile capacity "
            "281,941 vs 344,463 slots (-18.2%). This is a sanity run only; "
            "T009 owns the Q4 capacity fix.",
            "The 1/p expansion is applied downstream in one place per script: "
            "21_q2_opt.py multiplies dep/arr by SCALE=1/0.128 inside "
            "load_demand (PZ stays row-normalised), 22_q2_sim.py multiplies "
            "od_hr by SCALE, and 40_q4_efence.py multiplies by the same "
            "SCALE. All three therefore work at the full 343,638-bike scale.",
            "R1 fix: the first T006 submission refactored 21_q2_opt.py "
            "without restoring the SCALE multiplication, so the LP served "
            "sampled demand with a full-scale fleet (trips=274,357). With "
            "SCALE applied inside load_demand, the observed-surface run at "
            "alpha=1, K=60, fixed_K_cost=600 reproduces the pre-refactor row: "
            "moved 23,959.15, truck-km 5,440.94, dev 900.00, obj 83,325.24, "
            "trips 2,143,417. The Q1-surface run is now full-scale too: moved "
            "24,679.28, truck-km 5,529.85, dev 225.36, obj 82,194.80.",
            "No file under paper/, review/ or 审查与规划/ was written. data/ "
            "was only extended under data/runs/T006/.",
        ],
    }
    os.makedirs(DELIV, exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8") as handle:
        json.dump(delivery, handle, ensure_ascii=False, indent=1)
    print("saved", OUT)
    print(json.dumps(checks, ensure_ascii=False, indent=1))
    print("row_err", row_err, "cons_err", cons_err,
          "top5", q1v["top5"]["weighted"], "ks", ks)


if __name__ == "__main__":
    main()
