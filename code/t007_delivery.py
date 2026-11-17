"""Build the T007 delivery JSON: Q2 rerun under arrival conservation.

All checks are recomputed from the saved plans and JSON files.
"""
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, "code")
DATA = os.path.join(ROOT, "data")
PAPER = os.path.join(ROOT, "paper")
RUN = os.path.join(DATA, "runs", "T007")
DELIV = os.path.join(ROOT, "交付")
OUT = os.path.join(DELIV, "T007_delivery.json")
T015_FP = os.path.join(DATA, "runs", "T015", "fingerprint_start.json")
T001_FP = os.path.join(DATA, "runs", "T001", "fingerprint_start.json")
T006_SMOKE = os.path.join(DATA, "runs", "T006", "q2_q1_smoke.json")

SCALE = 1.0 / 0.128
TOL = 1e-6
BASE_COMMIT = "1b563da"
EXPECTED_SCEN = {"q1_a1_K20", "q1_a1_K40", "q1_a1_K60", "q1_a1_K80",
                 "q1_a0.8_K80", "q1_a0.6_K80"}
T007_FILES = {"code/21_q2_opt.py", "code/22_q2_sim.py",
              "code/23_q2_saveplan.py", "code/t007_q2_run.py",
              "code/t007_delivery.py", "code/t007_validate.py"}
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


def hashes(base, skip_runs=False, only=None):
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
    current = hashes(base, skip_runs=skip_runs)
    return {
        "baseline_files": len(baseline),
        "changed": sorted(p for p in baseline
                          if p in current and current[p] != baseline[p]),
        "missing": sorted(p for p in baseline if p not in current),
        "added": sorted(p for p in current if p not in baseline),
    }


def git_lines(args):
    proc = subprocess.run(["git", "-c", "core.quotepath=false"] + args,
                          cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    return proc.returncode, proc.stdout.splitlines()


def main():
    results = json.load(io.open(os.path.join(RUN, "q2_results_q1.json"),
                                encoding="utf-8"))
    cons = json.load(io.open(os.path.join(RUN, "q2_conservation.json"),
                             encoding="utf-8"))
    night = json.load(io.open(os.path.join(RUN, "q2_night.json"),
                              encoding="utf-8"))
    sim = json.load(io.open(os.path.join(RUN, "q2_sim_q1.json"),
                            encoding="utf-8"))
    greedy = json.load(io.open(os.path.join(RUN, "q2_greedy_check.json"),
                               encoding="utf-8"))

    scenario_keys = set(results["scenarios"].keys())
    plan_keys = {"XM", "X", "M", "I", "U", "MK"}
    structure_ok = scenario_keys == EXPECTED_SCEN
    structure_ok &= "obs_baseline" in results and "q1_vs_obs" in results
    structure_ok &= set(cons["scenarios"].keys()) == scenario_keys | {"obs_a1_K60"}
    for name in scenario_keys | {"obs_a1_K60"}:
        p = np.load(os.path.join(RUN, "plans", name + ".npz"))
        structure_ok &= set(p.files) == plan_keys
        structure_ok &= p["XM"].shape == (24, 50, 50)
        structure_ok &= p["MK"].shape == (24, 50, 50)
        structure_ok &= p["I"].shape == (50, 24)
        structure_ok &= os.path.exists(os.path.join(RUN, "plans", name + ".json"))
    main_plan = np.load(os.path.join(RUN, "q2_plan_q1.npz"))
    structure_ok &= plan_keys <= set(main_plan.files)

    full_scale = True
    for name, rec in results["scenarios"].items():
        p = np.load(os.path.join(RUN, "plans", name + ".npz"))
        full_scale &= abs(float(p["XM"].sum()) - rec["moved"]) <= 1e-6 * max(1.0, rec["moved"])
        full_scale &= abs(rec["trips"] - 2143417.4) / 2143417.4 < 1e-6
        full_scale &= abs(float(p["I"][:, 0].sum()) - 343638.0 * rec["alpha"]) \
            <= 1e-6 * 343638.0 * rec["alpha"]
    full_scale &= abs(float(main_plan["XM"].sum())
                      - results["scenarios"]["q1_a1_K60"]["moved"]) <= 1e-6

    cons_ok = all(v["passes"] for v in cons["scenarios"].values())
    cons_ok &= max(v["city_residual_max_abs"] for v in cons["scenarios"].values()) <= TOL
    cons_ok &= max(v["zone_residual_max_abs"] for v in cons["scenarios"].values()) <= TOL
    cons_ok &= max(abs(v["day_end_city_diff"]) for v in cons["scenarios"].values()) <= TOL
    cons_ok &= max(v["u_minus_dep_max"] for v in cons["scenarios"].values()) <= TOL
    cons_ok &= min(v["u_min"] for v in cons["scenarios"].values()) >= -1e-9

    t6 = json.load(io.open(T006_SMOKE, encoding="utf-8"))
    repro = {}
    for kind, t6key in (("q1", "q1"), ("obs", "obs")):
        name = "q1_a1_K60" if kind == "q1" else "obs_a1_K60"
        rec = results["scenarios"][name] if kind == "q1" else results["obs_baseline"]
        repro[kind] = all(
            abs(rec[k] - t6[t6key][k]) <= 1e-6 * max(1.0, abs(t6[t6key][k]))
            for k in ("moved", "truck_km", "unmet", "dev", "obj"))
    repro_ok = all(repro.values())

    fingerprint_ok = True
    for rec in list(results["scenarios"].values()) + [results["obs_baseline"]]:
        if rec["unmet"] > 1e-6:
            fingerprint_ok &= abs(rec["dev"] - rec["unmet"]) > 1e-6 * max(1.0, rec["unmet"])

    coverage_ok = (EXPECTED_SCEN == scenario_keys
                   and {results["scenarios"][n]["K"] for n in
                        ("q1_a1_K20", "q1_a1_K40", "q1_a1_K60", "q1_a1_K80")}
                   == {20, 40, 60, 80}
                   and results["scenarios"]["q1_a0.8_K80"]["alpha"] == 0.8
                   and results["scenarios"]["q1_a0.6_K80"]["alpha"] == 0.6
                   and "not run" in results["unrerun"])

    cmp_ok = set(results["q1_vs_obs"]["q1_minus_obs"]) == {
        "moved", "truck_km", "unmet", "dev", "obj"}
    cmp_ok &= all(
        abs(results["q1_vs_obs"]["q1_minus_obs"][k]
            - (results["scenarios"]["q1_a1_K60"][k]
               - results["obs_baseline"][k])) <= 1e-9 * max(1.0, abs(
                   results["q1_vs_obs"]["q1_minus_obs"][k]))
        for k in ("moved", "truck_km", "unmet", "dev", "obj"))

    greedy_ok = (greedy["max_hourly_moved"] <= greedy["hourly_limit"] + 1e-9
                 and greedy["max_source_bikes_per_truck"] <= greedy["source_limit"] + 1e-9
                 and greedy["hourly_violations"] == 0
                 and greedy["source_violations"] == 0
                 and abs(greedy["trucks_start_sum_min"] - greedy["K"]) <= 1e-9
                 and abs(greedy["trucks_start_sum_max"] - greedy["K"]) <= 1e-9)
    policies = {(row["alpha"], row["policy"]) for row in sim["rows"]}
    greedy_ok &= {1.0} == {a for a, p in policies if p == "LP plan"}
    greedy_ok &= {(1.0, "no trucks"), (1.0, "LP plan"), (1.0, "greedy")} <= policies
    greedy_ok &= len(sim["rows"]) == 7

    z = np.load(os.path.join(DATA, "runs", "T006", "q1_od.npz"))
    cent = np.load(os.path.join(DATA, "typical_day.npz"),
                   allow_pickle=True)["cent"]
    dist = np.sqrt(((cent[:, None, :] - cent[None, :, :]) ** 2).sum(-1))
    moved_h = main_plan["XM"].sum((1, 2))
    km_h = np.einsum("tij,ij->t", main_plan["MK"], dist)
    night_hours = [0, 1, 2, 3, 4, 5, 21, 22, 23]
    night_share = float(moved_h[night_hours].sum() / moved_h.sum())
    km_share = float(km_h[night_hours].sum() / km_h.sum())
    night_ok = (night_hours == night["night_hours"]
                and abs(night_share - night["moved_night_share"]) <= 1e-9
                and abs(km_share - night["truck_km_night_share"]) <= 1e-9
                and abs(moved_h.sum() - main_plan["XM"].sum()) <= 1e-9
                and night["note"].find("no night cost") >= 0)

    data_cmp = compare_fp("data", DATA, skip_runs=True)
    paper_cmp = compare_fp("paper", PAPER)
    review_cmp = compare_fp("review", os.path.join(ROOT, "review"),
                            fp_path=T001_FP)
    rc, status_lines = git_lines(["status", "--porcelain"])
    root_paths = [r["path"] for r in
                  json.load(io.open(T015_FP, encoding="utf-8"))["trees"]["root"]]
    root_dirty = [line for line in status_lines if line[3:] in root_paths]
    readonly_ok = (not data_cmp["changed"] and not data_cmp["missing"]
                   and not paper_cmp["changed"] and not paper_cmp["missing"]
                   and not review_cmp["changed"] and not review_cmp["missing"]
                   and not root_dirty)

    rc, diff_lines = git_lines(["diff", "--name-only", BASE_COMMIT, "--", "code"])
    modified = sorted(p for p in diff_lines if p)
    untracked = sorted(line[3:] for line in status_lines
                       if line.startswith("?? ")
                       and "T007_delivery.json" not in line)
    scope_ok = (set(modified) | set(untracked)) == T007_FILES

    abs_hits = []
    for rel in sorted(T007_FILES):
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            abs_hits.append({"file": rel, "line": 0, "text": "missing"})
            continue
        for lineno, line in enumerate(read(path).splitlines(), 1):
            for marker in ABS_MARKERS:
                if marker in line:
                    abs_hits.append({"file": rel, "line": lineno,
                                     "text": line.strip()})

    outputs = ["q2_results_q1.json", "q2_plan_q1.npz", "q2_sim_q1.json",
               "q2_conservation.json", "q2_night.json", "q2_greedy_check.json"]
    outputs += [f"plans/{n}.json" for n in scenario_keys | {"obs_a1_K60"}]
    outputs += [f"plans/{n}.npz" for n in scenario_keys | {"obs_a1_K60"}]
    missing = [p for p in outputs if not os.path.exists(os.path.join(RUN, p))]

    checks = {
        "1_outputs_and_plan_structure": structure_ok,
        "2_full_scale_plans": full_scale,
        "3_conservation_residuals": cons_ok,
        "4_u0_reproduces_T006_smoke": repro_ok,
        "5_u_le_dep_and_no_dev_fingerprint": cons_ok and fingerprint_ok,
        "6_scenario_coverage_and_unrerun_note": coverage_ok,
        "7_q1_vs_obs_comparison": cmp_ok,
        "8_greedy_capacity_and_three_policies": greedy_ok,
        "9_night_share_recomputed": night_ok,
        "10_readonly_scope_outputs_no_abs": (readonly_ok and scope_ok
                                             and not missing and not abs_hits),
    }

    delivery = {
        "task": "T007",
        "title": "Q2 rerun: arrival conservation, greedy units, night share",
        "date": "2026-11-18",
        "status": "delivered",
        "checks": checks,
        "settings": results["settings"],
        "scenarios": results["scenarios"],
        "obs_baseline": results["obs_baseline"],
        "q1_vs_obs": results["q1_vs_obs"],
        "conservation": {
            "tolerance": TOL,
            "convention": cons["same_hour_arrival_convention"],
            "scenarios": cons["scenarios"],
        },
        "night": night,
        "greedy": greedy,
        "simulation": {"meta": {k: sim[k] for k in
                                ("od", "plan", "K", "Q", "reps", "alphas",
                                 "seed_base")},
                       "rows": sim["rows"]},
        "integrity": {
            "baseline": {"data_paper_code":
                         "data/runs/T015/fingerprint_start.json",
                         "review": "data/runs/T001/fingerprint_start.json",
                         "scope_commit": BASE_COMMIT},
            "data_excluding_runs": data_cmp,
            "paper": paper_cmp,
            "review": review_cmp,
            "root_docs": {"dirty_in_git": root_dirty,
                          "note": "root docs were already committed before "
                                  "T007 and are clean."},
            "git_status_porcelain": [
                line[:3] + "交付/T007_delivery.json"
                if "T007_delivery.json" in line else line
                for line in status_lines],
        },
        "code_changes": [
            {"file": "code/21_q2_opt.py", "status": "modified",
             "what": "expose MK (hourly truck flow per arc) in the result and "
                     "the plan NPZ, for the night truck-km split",
             "why": "T007 needs truck-km by hour, which M (zone totals) "
                    "cannot provide."},
            {"file": "code/22_q2_sim.py", "status": "modified",
             "what": "H14: greedy capacity in bikes with Q*trucks_at, amt/Q "
                     "trucks moved from source to destination; add --plan, "
                     "--out, --greedy-check-out, --K/--Q/--reps/--alphas",
             "why": "the old cap[s]-=amt/Q mixed bike and truck units and "
                    "never moved trucks."},
            {"file": "code/23_q2_saveplan.py", "status": "modified",
             "what": "argparse --od/--alpha/--K/--conserve/--out (default "
                     "data/runs/T007/q2_plan_q1.npz); saves MK too",
             "why": "avoid overwriting tracked data/q2_plan.npz and keep the "
                    "plan in one directory."},
            {"file": "code/t007_q2_run.py", "status": "new",
             "what": "run the six Q1 scenarios plus the observed baseline, "
                     "compute conservation/night evidence, call 22",
             "why": "assemble T007 reproducibly."},
            {"file": "code/t007_delivery.py", "status": "new",
             "what": "this delivery builder", "why": "assemble T007 evidence."},
            {"file": "code/t007_validate.py", "status": "new",
             "what": "independent T007 delivery validator",
             "why": "recompute the checks before delivery."},
        ],
        "outputs": {
            "delivery_json": "交付/T007_delivery.json",
            "run_dir": "data/runs/T007/",
            "required": outputs,
        },
        "notes": [
            "All runs use --conserve with u<=dep and the same-hour arrival "
            "convention: arrivals at t come only from served departures at t, "
            "weighted by the row-normalised Q1 (or observed) P.",
            "u=0 scenarios reproduce the T006 R1 smoke rows bit-for-bit: "
            "q1 alpha=1 K=60 moved 24,679.2791, truck-km 5,529.8473, dev "
            "225.3579, obj 82,194.7968; obs 23,959.1518 / 5,440.9441 / "
            "900.0000 / 83,325.2404. With u=0, conserve changes no arrival.",
            "Conservation is exact to solver tolerance: the largest city "
            "residual is 3.23e-9 (alpha=0.6), the largest zone residual "
            "3.97e-10, and the day-end city difference 7.39e-9; all below "
            "1e-6.",
            "Unmet appears only when the initial fleet is scaled: alpha=0.8 "
            "K=80 has 447.52 unmet, alpha=0.6 K=80 has 59,517.66. The LP "
            "trades unmet against end-of-day deviation; dev is not equal to "
            "unmet in either row.",
            "Scenario coverage is alpha=1 K=20/40/60/80 plus alpha=0.8/0.6 at "
            "K=80. The full alpha=0.4-1.0 x K grid was NOT run and is marked "
            "in q2_results_q1.json; T012 must not quote unrun grid values.",
            "K=20 and K=40 are capacity-binding: moved is exactly 9,600 and "
            "19,200 bikes; K=60 moves 24,679 and K=80 only 24,753, so the "
            "returns flatten above K=60.",
            "H14: greedy max hourly moved is 623.17 (limit 2,400) and max "
            "source ratio exactly 40 bikes/truck (limit 40), with zero "
            "violations. Trucks start each hour summing to K and move with "
            "their bikes.",
            "Night (21:00-06:00, 9 h) carries 39.34% of moved bikes and "
            "37.12% of truck-km, close to its 37.5% share of clock time; the "
            "paper's 'daytime barely moves' wording is not supported. No "
            "night cost was added.",
            "The stochastic simulation reports 0.00% unmet for every policy "
            "and alpha, including no trucks: with I0 ~343,638 the zone "
            "inventories never hit zero at 5-min resolution, so the "
            "simulation is uninformative about the LP's alpha<1 unmet. This "
            "is a known limitation, not evidence that unmet is zero.",
            "22 was run at reps=20 with the T007 plan; 23 was compiled but "
            "not executed (it duplicates the runner's plan save).",
            "data/ (except runs/T007), paper/ and review/ hashes are "
            "unchanged; paper/, review/ and 审查与规划/ were not written.",
        ],
    }
    os.makedirs(DELIV, exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8") as handle:
        json.dump(delivery, handle, ensure_ascii=False, indent=1)
    print("saved", OUT)
    print(json.dumps(checks, ensure_ascii=False, indent=1))
    print("max city resid",
          max(v["city_residual_max_abs"] for v in cons["scenarios"].values()),
          "night share", night_share, "greedy max", greedy["max_hourly_moved"])


if __name__ == "__main__":
    main()
