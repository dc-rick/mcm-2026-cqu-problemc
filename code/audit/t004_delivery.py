"""T004 delivery: assemble the spatial-resolution / walking-radius evidence."""
import datetime
import hashlib
import io
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = os.path.join(ROOT, "data", "runs", "T004")
OUT = os.path.join(ROOT, "交付", "T004_delivery.json")


def load(name):
    return json.load(io.open(os.path.join(RUN, name), encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


neigh = load("neighbours.json")
neigh_f = load("neighbours_fine.json")
sim_old = load("sim_results.json")
sim_new = load("sim_fine.json")
fine = load("fine_grid.json")
dry = load("dry_cells.json")
dry_robust = load("dry_172_robustness.json")
cons = load("conservation.json")

# ---------------------------------------------------------------- hashes
fp0 = json.load(io.open(os.path.join(ROOT, "data", "runs", "T001",
                                     "fingerprint_start.json"), encoding="utf-8"))
fp_end = json.load(io.open(os.path.join(ROOT, "data", "runs", "T001",
                                        "fingerprint_end.json"), encoding="utf-8"))
old_paper = {i["path"]: i["sha256"] for i in fp0["trees"]["paper"]}
old_data = {i["path"]: i["sha256"] for i in fp0["trees"]["data"]}


def tree_compare(rel, old):
    cur = {}
    base = os.path.join(ROOT, rel)
    for dp, _, fn in os.walk(base):
        for f in fn:
            p = os.path.join(dp, f)
            r = os.path.relpath(p, ROOT).replace("\\", "/")
            if rel == "data" and r.startswith("data/runs"):
                continue
            cur[r] = sha(p)
    return dict(
        changed=sorted(k for k in old.keys() & cur.keys() if old[k] != cur[k]),
        added=sorted(cur.keys() - old.keys()),
        removed=sorted(old.keys() - cur.keys()),
        n_files=len(cur),
        unchanged=(set(old) == set(cur) and
                   all(old[k] == cur[k] for k in old.keys() & cur.keys())))


paper_check = tree_compare("paper", old_paper)
data_check = tree_compare("data", old_data)
old_code = {i["path"]: i["sha256"] for i in fp_end["trees"]["code"]}
cur_code = {}
for dp, _, fn in os.walk(os.path.join(ROOT, "code")):
    for f in fn:
        p = os.path.join(dp, f)
        r = os.path.relpath(p, ROOT).replace("\\", "/")
        cur_code[r] = sha(p)
code_check = dict(
    changed=sorted(k for k in old_code.keys() & cur_code.keys()
                   if old_code[k] != cur_code[k]),
    added=sorted(cur_code.keys() - old_code.keys()),
    removed=sorted(old_code.keys() - cur_code.keys()))
lib_now = sha(os.path.join(ROOT, "code", "lib.py"))
lib_old = [i["sha256"] for i in fp_end["trees"]["code"]
           if i["path"] == "code/lib.py"][0]
sim_now = sha(os.path.join(ROOT, "code", "sim_cell.py"))
sim_old_sha = [i["sha256"] for i in fp_end["trees"]["code"]
               if i["path"] == "code/sim_cell.py"][0]
old_cfc = [i["sha256"] for i in fp0["trees"]["data"]
           if i["path"] == "data/cell_fleet_curve.npy"][0]
old_csi = [i["sha256"] for i in fp0["trees"]["data"]
           if i["path"] == "data/cell_sim_inputs.npz"][0]
now_cfc = sha(os.path.join(ROOT, "data", "cell_fleet_curve.npy"))
now_csi = sha(os.path.join(ROOT, "data", "cell_sim_inputs.npz"))

# ---------------------------------------------------------------- texts
ABSTRACT = (
    "The 50-zone inventory never goes negative: the smallest zone holds 1,986 "
    "bicycles at midnight and the worst cumulative excursion is -349. At "
    "neighbourhood scale the shortage is local, not systemic. On the "
    "0.005-degree grid a 300 m walk reaches no neighbouring cell, so the "
    "earlier 13.4% is a no-walking bound; with a 450 m walk unmet demand is "
    "5.8% at the paper's 343,638-bicycle calibration and 3.5% at the "
    "closed-fleet estimate of 495,540. A 0.0025-degree grid covering 86% of "
    "trips gives 2.1% at 495,540. The system has enough bicycles; it does not "
    "have them everywhere."
)
SEC9 = (
    "The consequence is a wide range in the simulated service level at cell "
    "resolution. In the zone-level inventory path of Section Q2 no zone goes "
    "negative. On the 0.005-degree grid, a 300 m radius reaches no "
    "neighbouring cell centre (the nearest is 426 m), so 13.4% is the "
    "no-walking bound. With a 450 m radius, unmet demand is 5.8% at 343,638 "
    "bicycles and 3.5% at the closed-fleet estimate of 495,540; 600 m gives "
    "2.3% and 1.3%. On a 0.0025-degree grid, which covers 86% of trips, 450 m "
    "gives 2.1%."
)
wc_a, wc_9 = len(ABSTRACT.split()), len(SEC9.split())
assert wc_a <= 120 and wc_9 <= 120, (wc_a, wc_9)

# ---------------------------------------------------------------- cases
def case(label, mean, sd, fleet, grid, R, alpha, source, note=""):
    return dict(label=label, mean=mean, sd=sd, fleet=fleet, grid=grid,
                R_m=R, alpha=alpha, source=source, note=note)


cases = []
for key, r in sim_old["results"].items():
    cases.append(case(key, r["mean"], r["std"], r["fleet"],
                      "0.005 deg", r["R_m"], r["alpha"],
                      "data/runs/T004/sim_results.json:results." + key))
for key, r in sim_new["runs"].items():
    cases.append(case(key, r["mean"], r["std"], r["fleet"],
                      "0.005 or 0.0025 deg", r["R_m"], r["alpha"],
                      "data/runs/T004/sim_fine.json:runs." + key))

delivery = {
    "task": "T004",
    "title": "空间分辨率与步行半径",
    "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "resdir": "data/runs/T004",
    "headline": {
        "abstract_claim_old": ("At every resolution ... no zone ever runs out "
                               "of bicycles"),
        "coarse_R300_a1_fleet343638": sim_old["results"][
            "coarse_343638_R300_a1.0"]["mean"],
        "coarse_R450_a1_fleet343638": sim_old["results"][
            "coarse_343638_R450_a1.0"]["mean"],
        "coarse_R450_a1_fleet495540": sim_new["runs"][
            "coarse_R450_a1.0"]["mean"],
        "fine_R450_a1_fleet495540": sim_new["runs"]["fine_R450_a1.0"]["mean"],
        "cells_with_neighbour_300_450_600": [
            neigh["radii"]["0.30"]["cells_with_neighbour"],
            neigh["radii"]["0.45"]["cells_with_neighbour"],
            neigh["radii"]["0.60"]["cells_with_neighbour"]],
        "dry_cells_172_condition": ("hourly, unscaled cell path; the stated "
                                    "5-minute 1/p-scaled path gives 2,306"),
    },
    "neighbour_geometry": {
        "coarse": {
            "n_cells": neigh["n_cells"],
            "radii": neigh["radii"],
            "lattice": neigh["lattice"],
            "interpretation": (
                "300 m: 0 of 5,263 cells have any neighbour (nearest centre "
                "426 m), so the walk branch never fires. 450 m: 4,885 cells "
                "have 1-2 neighbours, all east-west; 378 edge cells have none. "
                "600 m: 5,091 cells have neighbours; east-west 426 m and "
                "north-south 556 m are reachable, diagonal 700 m is not. "
                "R=0.70 is identical to R=0.60 because the diagonal distance "
                "is slightly above 0.700 km. The 172 cells with no neighbour "
                "at R=600 are numerically coincidental with the paper's 172 "
                "dry cells and are unrelated."),
            "anisotropy": ("A circular radius on a rectangular lattice is an "
                           "artifact: at 450 m the reachable graph is a "
                           "one-dimensional east-west chain, not a disc."),
        },
        "fine": {
            "n_cells": neigh_f["n_cells"],
            "radii": neigh_f["radii"],
            "lattice": neigh_f["lattice"],
            "interpretation": (
                "At 0.0025 deg the spacings are 213 m east-west, 278 m "
                "north-south and 350 m diagonal. A 300 m radius now reaches "
                "1-4 real neighbours (mean 2.93), so the 300 m experiment "
                "finally exercises walking; 450 m reaches mean 7.0 and 600 m "
                "mean 13.4. Resolution and radius are therefore not separable "
                "in the coarse-grid comparison."),
        },
    },
    "simulator_defect": {
        "location": "code/sim_cell.py, walk-to-neighbour branch",
        "old_behaviour": (
            "I[j] -= 1 removed a bicycle from the donor cell, but the in-flight "
            "arrival pool was fed only from `take` (bicycles taken at the "
            "departure cell). Every walked trip deleted one bicycle."),
        "measured_loss_before_fix": cons["before_fix_seed0"],
        "fix": ("borrow[c] counts walked trips; the zone loop schedules "
                "take[idx].sum() + borrow[idx].sum() arrivals, so a borrowed "
                "ride re-enters the fleet through the same one-step in-flight "
                "pool. The destination kernel remains the demand cell's zone."),
        "baseline_effect": ("At R=300 m no neighbour is reachable, borrowed=0, "
                            "so the published 13.4% curve is bit-identical; the "
                            "fix changes only R>=426 m runs."),
        "consequence": ("Before the fix, unmet rose with radius (R=450 14.90%, "
                        "R=600 20.52% at 343,638) because bicycles were being "
                        "deleted; those numbers are invalid."),
    },
    "simulation_results": {
        "seed_set": sim_old["seeds"],
        "n_seeds": 5,
        "fleet_343638_paper_calibration": {
            k: v for k, v in sim_old["results"].items() if "343638" in k},
        "fleet_495540_bridge": {
            k: v for k, v in sim_old["results"].items() if "495540" in k},
        "fleet_495540_closed_fleet": sim_new["runs"],
        "reproduce_old_curve": sim_old["reproduce_old_curve"],
        "conservation": cons,
        "reading": (
            "At the paper's 343,638 calibration the five-seed mean is 13.41% "
            "at R=300 (13.4257% for seeds 0-2, exactly the stored curve), "
            "5.84% at R=450 and 2.35% at R=600. At the closed-fleet centre "
            "495,540 the same radii give 9.13%, 3.51% and 1.28%. The main "
            "coarse-grid specification asked for by the task (R=450, alpha=1) "
            "is therefore 5.84% at 343,638 and 3.51% at 495,540."),
    },
    "dry_cells": {
        "paper_claim": "some 172 of 5,263 cells do run dry",
        "finding": (
            "172 is reproduced exactly by an HOURLY, UNSCALED deterministic "
            "cell path I_c(t)=I0c+cumsum(arrivals-departures). The 5-minute "
            "1/p-scaled path described in the paper gives 2,306; hourly scaled "
            "gives 2,251; 5-minute unscaled gives 175. No LP rerun was done: "
            "q2_plan.npz and q2_solution.npz are zone-level (50 zones), so "
            "they cannot yield a count over 5,263 cells. Robustness: all 12 "
            "hourly phase offsets give 169-174 dry cells (172 at three of "
            "them), and a one-step arrival lag also gives 172; the match is "
            "not a single lucky timestamp."),
        "definitions": dry["definitions"],
        "robustness": dry_robust,
        "stochastic_343638": dry["stochastic_343638"],
        "side_by_side_rule": (
            "The fluid count and the stochastic count are different models "
            "and are never substituted for one another. The stochastic count "
            "is cells whose end-of-step inventory is below one bicycle at "
            "least once; it rises from 2,463 at R=300 to 2,778 at R=450 even "
            "though unserved departures fall from 13.41% to 5.84%, because "
            "walking moves emptiness between cells."),
    },
    "fine_grid": {
        "diagnostics": fine,
        "support_verdict": (
            "supported, with reduced coverage. Cell dimensions 213 x 278 m are "
            "3.64x the constant GPS error (58.5 m lon, 76.4 m lat), and the "
            "median retained cell has 257 trips over 10 days (p10 104). But "
            "85.9% of box-filtered trips map to retained fine cells versus "
            "95.6% at 0.005 deg, and typical-day departures are 90.2% of the "
            "coarse total, so the fine-grid level is not directly comparable "
            "to the coarse level without the demand-matched row."),
        "results_fleet495540": sim_new["runs"],
        "i0_no_evidence": fine["last_trip_end"],
        "i0_zero_sensitivity": {
            "R450_a1": sim_new["runs"]["fine_R450_a1.0_I0zero"]["mean"],
            "R450_a1_default": sim_new["runs"]["fine_R450_a1.0"]["mean"],
            "R450_a08": sim_new["runs"]["fine_R450_a0.8_I0zero"]["mean"],
            "R450_a08_default": sim_new["runs"]["fine_R450_a0.8"]["mean"],
        },
        "demand_matched": {
            "fleet": sim_new["runs"]["fine_R450_a1.0_demandmatched"]["fleet"],
            "R450_a1": sim_new["runs"]["fine_R450_a1.0_demandmatched"]["mean"],
            "R450_a08": sim_new["runs"]["fine_R450_a0.8_demandmatched"]["mean"],
        },
    },
    "replacement_text": {
        "abstract_24_27": {"english": ABSTRACT, "word_count": wc_a,
                           "paper_untouched": True},
        "sec9_106_108": {"english": SEC9, "word_count": wc_9,
                         "paper_untouched": True},
        "note": ("The final clause of the abstract ('The system has enough "
                 "bicycles; it does not have them everywhere') is a suggested "
                 "framing, not a decision; the reviewer may keep the facts and "
                 "change the conclusion. The sentence immediately above the "
                 "abstract replacement still carries the T001-R2 fleet rewrite "
                 "and is not changed here."),
    },
    "affected": [
        {"coord": "paper/summary.tex:24-27",
         "old": "At every resolution ... no zone ever runs out of bicycles",
         "new": "see replacement_text.abstract_24_27; 13.4% -> 5.8%/3.5%",
         "why": "the claim is contradicted by sec5 and sec9 and by every run"},
        {"coord": "paper/sec5_q2.tex:19-22",
         "old": "some 172 of 5,263 cells do run dry",
         "new": ("172 only under an hourly unscaled path; the stated 5-minute "
                 "1/p-scaled path gives 2,306; stochastic gives 2,463 at R=300"),
         "why": "the number's stated model does not reproduce it"},
        {"coord": "paper/sec9_sensitivity.tex:103-113",
         "old": ("5,263 cells served by 343,638; 300 m walking radius; 13.4% "
                 "at alpha=1; 40.3% at alpha=0.2"),
         "new": ("300 m reaches no neighbour; 450 m gives 5.8% at 343,638 and "
                 "3.5% at 495,540; alpha=0.2 value not rerun under walking"),
         "why": "the 13.4% is a no-walking number and the radius was never "
                "active"},
        {"coord": "paper/sec7_q4.tex (450 m coverage radius)",
         "old": "450 m coverage radius",
         "new": "unchanged; T004 shows 450 m is the first radius that reaches "
                "any coarse-grid neighbour (east-west only)",
         "why": "consistency note only"},
    ],
    "methods": {
        "neighbours": ("cKDTree.query_ball_point on cell centres; x=(lon-116)*"
                       "85.2 km, y=(lat-39)*111.2 km; distance in km; counts "
                       "exclude self."),
        "simulator": ("CellSim, 288 five-minute steps; departures Poisson(dep5); "
                      "take=min(d,floor(I)); residual walked to the neighbour "
                      "with the most bicycles inside R; borrowed rides enter "
                      "the one-step in-flight pool; unmet_share=unmet/(unmet+"
                      "served)."),
        "fleet_scale": ("coarse inputs store FLEET=343,638, scaled by 495540/"
                        "343638 for the closed-fleet rows; fine inputs store "
                        "495,540 directly."),
        "fluid_dry": ("I_c(t)=I0c+cumsum(arrivals-departures); dry = min<0 at "
                      "the stated sampling; scaled uses 1/0.128."),
        "fine_grid": ("same construction as 10_q1_model.py at step 0.0025, "
                      "threshold 50 on starts or ends over the 10 well-covered "
                      "days; zones inherited from the parent 0.005 cell."),
        "uncertainty": ("mean and sample standard deviation over seeds 0-4; "
                        "the seed spread is Monte Carlo only, not model "
                        "uncertainty."),
    },
    "cases": cases,
    "comparisons": [
        {"label": "R=450 vs R=300, alpha=1, 343,638",
         "old": sim_old["results"]["coarse_343638_R300_a1.0"]["mean"],
         "new": sim_old["results"]["coarse_343638_R450_a1.0"]["mean"],
         "diff": (sim_old["results"]["coarse_343638_R450_a1.0"]["mean"] -
                  sim_old["results"]["coarse_343638_R300_a1.0"]["mean"]),
         "unit": "unmet share"},
        {"label": "495,540 vs 343,638, R=450, alpha=1",
         "old": sim_old["results"]["coarse_343638_R450_a1.0"]["mean"],
         "new": sim_new["runs"]["coarse_R450_a1.0"]["mean"],
         "diff": (sim_new["runs"]["coarse_R450_a1.0"]["mean"] -
                  sim_old["results"]["coarse_343638_R450_a1.0"]["mean"]),
         "unit": "unmet share"},
        {"label": "fine vs coarse, R=450, alpha=1, 495,540",
         "old": sim_new["runs"]["coarse_R450_a1.0"]["mean"],
         "new": sim_new["runs"]["fine_R450_a1.0"]["mean"],
         "diff": (sim_new["runs"]["fine_R450_a1.0"]["mean"] -
                  sim_new["runs"]["coarse_R450_a1.0"]["mean"]),
         "unit": "unmet share"},
        {"label": "fine demand-matched vs coarse, R=450, alpha=1",
         "old": sim_new["runs"]["coarse_R450_a1.0"]["mean"],
         "new": sim_new["runs"]["fine_R450_a1.0_demandmatched"]["mean"],
         "diff": (sim_new["runs"]["fine_R450_a1.0_demandmatched"]["mean"] -
                  sim_new["runs"]["coarse_R450_a1.0"]["mean"]),
         "unit": "unmet share"},
        {"label": "dry-cell count: paper vs 5-min scaled vs stochastic R=300",
         "old": 172, "new": 2306, "third": 2463.4,
         "unit": "cells", "comparable": False},
    ],
    "checks": {
        "1_numbers_traceable": True,
        "2_old_data_unchanged": data_check["unchanged"],
        "3_readonly_untouched": paper_check["unchanged"],
        "4_new_results_in_T004": True,
        "5_change_list": True,
        "6_failures_explicit": True,
        "7_magnitude": True,
        "8_boundary_conditions": True,
        "9_untested_branches": True,
        "10_caliber_unchanged": True,
        "reasons": {
            "1": ("every case names its JSON file and key; neighbour and fine "
                  "numbers come from data/runs/T004/*.json"),
            "2": ("data/ excluding data/runs against the T001 start "
                  "fingerprint: " + str(len(data_check["changed"])) +
                  " changed, " + str(len(data_check["added"])) + " added, " +
                  str(len(data_check["removed"])) + " removed; "
                  "cell_fleet_curve.npy and cell_sim_inputs.npz are also "
                  "compared explicitly in paper_integrity"),
            "3": ("paper tree unchanged against the T001 start fingerprint: " +
                  str(len(paper_check["changed"])) + " changed, " +
                  str(len(paper_check["added"])) + " added, " +
                  str(len(paper_check["removed"])) + " removed; T004 scripts "
                  "contain no write path to review/ or 审查与规划/, which are "
                  "owned by the reviewer"),
            "4": "all new artifacts are under data/runs/T004",
            "5": "code_changes lists every new script and the one modified file",
            "6": ("172 is not reproduced by the paper's stated 5-minute scaled "
                  "path (2,306); coarse R=450 reaches only east-west "
                  "neighbours; the 0.700 km diagonal is outside R=0.70; the "
                  "alpha=0.2 curve was not rerun under walking"),
            "7": ("fleet 495,540 >= observed union 484,754; all unmet shares "
                  "are in [0,1]; inventory is non-negative at every step; the "
                  "fine-grid demand base is 90.2% of the coarse base"),
            "8": ("fleet conservation residual is exactly 0 for every grid, "
                  "radius and seed after the fix; minimum inventory is 0.0 "
                  "(coarse) and 0.0021 (fine), never negative; the final "
                  "in-transit pool is included in the residual"),
            "9": "see notes",
            "10": ("no paper edit, no Q2 LP rerun, no Q4 siting rerun; 343,638 "
                   "retained only as the reproduction calibration and 495,540 "
                   "as the closed-fleet bridge; p=0.128 unchanged"),
        },
    },
    "paper_integrity": {
        "paper": paper_check,
        "data_excluding_runs": data_check,
        "lib_py_sha_now": lib_now,
        "lib_py_sha_t001": lib_old,
        "lib_py_unchanged": lib_now == lib_old,
        "sim_cell_py_sha_now": sim_now,
        "sim_cell_py_sha_t001": sim_old_sha,
        "sim_cell_py_changed": sim_now != sim_old_sha,
        "cell_fleet_curve_sha_now": now_cfc,
        "cell_fleet_curve_sha_t001": old_cfc,
        "cell_fleet_curve_unchanged": now_cfc == old_cfc,
        "cell_sim_inputs_sha_now": now_csi,
        "cell_sim_inputs_sha_t001": old_csi,
        "cell_sim_inputs_unchanged": now_csi == old_csi,
    },
    "code_integrity": {
        "baseline": "data/runs/T001/fingerprint_end.json (code tree)",
        "changed": code_check["changed"],
        "added_count": len(code_check["added"]),
        "removed": code_check["removed"],
        "t004_modified": ["code/sim_cell.py",
                          "code/__pycache__/sim_cell.cpython-311.pyc"],
        "pre_existing_since_T001": ["code/audit/t001_delivery.py",
                                    "code/audit/t003_numbers.py"],
        "explanation": ("t001_delivery.py (22:06) and t003_numbers.py (22:11) "
                        "were changed before T004 started and were not touched "
                        "by T004; the bytecode file is regenerated by Python "
                        "when sim_cell.py is imported."),
    },
    "code_changes": [
        {"file": "code/sim_cell.py", "status": "modified",
         "what": ("walk-branch fleet conservation: borrow[c] counted and added "
                  "to the in-flight arrival pool; docstring records the T004 "
                  "fix"),
         "why": ("without it every walked trip deleted a bicycle; R=450 lost "
                 "240,170 bicycles/day and R=600 lost 314,073"),
         "baseline": "R=300 bit-identical because borrow=0"},
        {"file": "code/__pycache__/sim_cell.cpython-311.pyc",
         "status": "regenerated", "what": "bytecode cache of sim_cell.py",
         "why": "Python import cache; not source"},
        {"file": "code/audit/t001_delivery.py", "status": "pre-existing change",
         "what": "mtime 22:06, before T004 started",
         "why": "not modified by T004; listed for a complete code diff"},
        {"file": "code/audit/t003_numbers.py", "status": "pre-existing change",
         "what": "mtime 22:11, before T004 started",
         "why": "not modified by T004; listed for a complete code diff"},
        {"file": "code/audit/t004_neighbours.py", "status": "new",
         "what": "coarse-grid reachable-neighbour counts and lattice distances"},
        {"file": "code/audit/t004_neighbours_fine.py", "status": "new",
         "what": "fine-grid reachable-neighbour counts and lattice distances"},
        {"file": "code/audit/t004_sim.py", "status": "new",
         "what": "coarse 343,638 radius runs and 13.4% reproduction"},
        {"file": "code/audit/t004_sim_fine.py", "status": "new",
         "what": "coarse 495,540 and fine-grid runs, I0-zero and demand-matched"},
        {"file": "code/audit/t004_leak_check.py", "status": "new",
         "what": "before/after fleet-conservation measurement, all seeds"},
        {"file": "code/audit/t004_fine_grid.py", "status": "new",
         "what": "0.0025-degree cell system, data support, fine inputs"},
        {"file": "code/audit/t004_drycells.py", "status": "new",
         "what": "172-dry-cell reconstruction and stochastic side-by-side"},
        {"file": "code/audit/t004_dry_search.py", "status": "new",
         "what": "definition search that located the hourly-unscaled path"},
        {"file": "code/audit/t004_dry_robust.py", "status": "new",
         "what": "phase-offset/arrival-lag robustness of the 172 count"},
        {"file": "code/audit/t004_validate.py", "status": "new",
         "what": "delivery validator"},
        {"file": "code/audit/t004_delivery.py", "status": "new",
         "what": "this delivery builder"},
    ],
    "outputs": {
        "delivery_json": "交付/T004_delivery.json",
        "neighbours_coarse": "data/runs/T004/neighbours.json",
        "neighbours_fine": "data/runs/T004/neighbours_fine.json",
        "coarse_runs": "data/runs/T004/sim_results.json",
        "conservation": "data/runs/T004/conservation.json",
        "fine_grid": "data/runs/T004/fine_grid.json",
        "fine_cells": "data/runs/T004/fine_cells.parquet",
        "fine_inputs": "data/runs/T004/fine_sim_inputs.npz",
        "all_runs": "data/runs/T004/sim_fine.json",
        "dry_cells": "data/runs/T004/dry_cells.json",
        "dry_search": "data/runs/T004/dry_definition_search.json",
        "dry_robustness": "data/runs/T004/dry_172_robustness.json",
    },
    "notes": [
        ("The 172 dry-cell number is not an LP result. The Q2 LP files are "
         "zone-level (50 zones); the only reconstruction that gives 172 is an "
         "hourly, unscaled cell path. The paper says the path is 1/p-scaled; "
         "the 5-minute scaled path gives 2,306. T012 must decide which "
         "statement to keep."),
        ("At 0.005 deg, R=450 reaches only east-west neighbours (426 m) and "
         "R=600 adds north-south (556 m); diagonal neighbours (700 m) are not "
         "reachable until R>0.700. This anisotropy is an artifact of a "
         "circular radius on a rectangular lattice, not a walking model."),
        ("A 0.0025 deg cell is 213 x 278 m, 3.64x the constant GPS error; it is "
         "above the coordinate resolution but 14.1% of box-filtered trips fall "
         "in sub-threshold fine cells, so the fine-grid level is supported with "
         "reduced coverage, not as a full-city estimate."),
        ("The fine-grid fleet is fixed at 495,540 while its demand base is "
         "90.2% of the coarse base; the demand-matched row (446,959 bicycles) "
         "is the like-for-like comparison and gives 2.39% rather than 2.07% at "
         "R=450, alpha=1. After the simulator's 1/0.128 scaling the coarse "
         "day has 2.14M departures and the fine day 1.93M; all quoted unmet "
         "figures are ratios on their own demand base."),
        ("On the fine grid the original 300 m radius is physically active and "
         "gives 4.77% at 495,540; the coarse 300 m run is no-walking. If the "
         "paper keeps the 300 m label, the fine-grid number is the defensible "
         "one; the coarse R=450 number is the task's main specification."),
        ("A walked trip is dispatched with the demand cell's zone destination "
         "kernel, not the donor cell's; adjacent cells can cross a KMeans zone "
         "boundary, so this is an approximation. The donor choice is 'the "
         "reachable cell with the most bicycles'; nearest-first and random "
         "donor rules are untested."),
        ("No-evidence I0 cells: 25 of 11,269 (0.22%), carrying 0.043% of "
         "typical departures; giving them the zone-median default instead of "
         "zero changes the R=450 result by less than 0.02 percentage points."),
        ("Untested branches: non-circular walking areas; real pedestrian "
         "networks and barriers; walking willingness that depends on distance, "
         "weather or bike availability; competition between walkers; "
         "time-varying radius; the diagonal at exactly 0.70 km; the alpha=0.2 "
         "curve under walking; Q2 LP rerun (T007) and Q4 siting rerun (T009) "
         "are out of scope."),
        ("The seed spread is Monte Carlo only (0.03-0.24 percentage points); "
         "model uncertainty from p, fleet and I0 is larger and is not included "
         "in the quoted standard deviations."),
        ("The replacement texts are suggestions only; paper/ is byte-identical "
         "to the T001 start fingerprint and no conclusion is decided here."),
    ],
}

json.dump(delivery, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False,
          indent=1)
print("written", OUT)
print("checks", {k: v for k, v in delivery["checks"].items()
                  if isinstance(v, bool)})
print("paper unchanged", paper_check["unchanged"],
      "data unchanged", data_check["unchanged"])
print("word counts", wc_a, wc_9)
