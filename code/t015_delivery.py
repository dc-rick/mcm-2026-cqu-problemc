"""Build the T015 delivery JSON: portability cleanup evidence."""
import hashlib
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, "code")
DATA = os.path.join(ROOT, "data")
PAPER = os.path.join(ROOT, "paper")
DELIV = os.path.join(ROOT, "\u4ea4\u4ed8")
RUNS = os.path.join(DATA, "runs", "T015")
FP_PATH = os.path.join(RUNS, "fingerprint_start.json")
T001_FP_PATH = os.path.join(DATA, "runs", "T001", "fingerprint_start.json")
OUT = os.path.join(DELIV, "T015_delivery.json")

sys.path.insert(0, CODE)
import run_all  # noqa: E402

ABS_MARKERS = ("C:" + chr(92) + "Users", "D:" + chr(92) + "code")
EXPECTED_MODIFIED = {"00_prepare.py", "01_recon.py", "02_recon_spatial.py",
                     "03_diag.py", "04_behaviour.py"}
EXPECTED_ADDED = {"build_dist_matrix.py", "run_all.py",
                  "t015_delivery.py", "t015_validate.py"}


def read(path):
    with io.open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def current_tree(base, skip_runs=False):
    out = {}
    for path in Path(base).rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(ROOT).as_posix()
        if skip_runs and rel.startswith("data/runs/"):
            continue
        out[rel] = sha(str(path))
    return out


def compare_tree(name, skip_runs=False):
    fp = json.load(io.open(FP_PATH, encoding="utf-8"))["trees"]
    baseline = {row["path"]: row["sha256"] for row in fp[name]}
    if name == "root":
        current = {p: sha(os.path.join(ROOT, p)) for p in baseline
                   if os.path.exists(os.path.join(ROOT, p))}
        added = []
    else:
        base = DATA if name == "data" else PAPER if name == "paper" else CODE
        current = current_tree(base, skip_runs=skip_runs)
        added = sorted(p for p in current if p not in baseline)
    changed = sorted(p for p in baseline if p in current and current[p] != baseline[p])
    missing = sorted(p for p in baseline if p not in current)
    return {"baseline_files": len(baseline), "changed": changed,
            "missing": missing, "added": added}


def path_audit():
    hits = []
    for path in sorted(Path(CODE).rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(ROOT).as_posix()
        for lineno, line in enumerate(read(str(path)).splitlines(), 1):
            for marker in ABS_MARKERS:
                if marker in line:
                    hits.append({"file": rel, "line": lineno,
                                 "text": line.strip()})
    return hits


def run(cmd):
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    return proc


def main():
    fp = json.load(io.open(FP_PATH, encoding="utf-8"))
    abs_hits = path_audit()
    data_cmp = compare_tree("data", skip_runs=True)
    paper_cmp = compare_tree("paper")
    root_cmp = compare_tree("root")
    code_cmp = compare_tree("code")
    t001 = json.load(io.open(T001_FP_PATH, encoding="utf-8"))["trees"]
    review_baseline = {row["path"]: row["sha256"] for row in t001["review"]}
    review_now = current_tree(os.path.join(ROOT, "review"))
    review_cmp = {
        "baseline_files": len(review_baseline),
        "changed": sorted(p for p in review_baseline
                          if p in review_now and review_now[p] != review_baseline[p]),
        "missing": sorted(p for p in review_baseline if p not in review_now),
        "added": sorted(p for p in review_now if p not in review_baseline),
    }
    code_modified = sorted(os.path.basename(p) for p in code_cmp["changed"])
    code_added = sorted(os.path.basename(p) for p in code_cmp["added"])

    help_run = run([sys.executable, "-I", os.path.join(CODE, "00_prepare.py"),
                    "--help"])
    guard_run = run([sys.executable, "-I", os.path.join(CODE, "00_prepare.py"),
                     "--out", os.path.join(DATA, "trips.parquet")])
    help_log = read(os.path.join(RUNS, "00_prepare_help.log"))
    guard_log = read(os.path.join(RUNS, "00_prepare_overwrite_guard.log"))
    prepare_log = read(os.path.join(RUNS, "00_prepare_run.log"))
    behaviour_log = read(os.path.join(RUNS, "04_behaviour_noforce.log"))

    ref_trips = pd.read_parquet(os.path.join(DATA, "trips.parquet"))
    check_trips = pd.read_parquet(os.path.join(RUNS, "trips_check.parquet"))
    trips_sha = sha(os.path.join(DATA, "trips.parquet"))
    trips_sha_start = {r["path"]: r["sha256"] for r in fp["trees"]["data"]}[
        "data/trips.parquet"]
    rows_written = None
    match = re.search(r"wrote .*?\((\d+), (\d+)\)", prepare_log)
    if match:
        rows_written, cols_written = int(match.group(1)), int(match.group(2))
    else:
        cols_written = None

    dm = json.load(io.open(os.path.join(RUNS, "dist_matrix_check.json"),
                           encoding="utf-8"))
    dm["output"] = os.path.relpath(dm["output"], ROOT).replace(os.sep, "/")
    dm["reference"] = os.path.relpath(dm["reference"], ROOT).replace(os.sep, "/")
    dist_sha = sha(os.path.join(DATA, "dist_matrix.npy"))
    dist_sha_start = {r["path"]: r["sha256"] for r in fp["trees"]["data"]}[
        "data/dist_matrix.npy"]

    dry_run = run([sys.executable, os.path.join(CODE, "run_all.py"), "--dry-run"])
    dry_log = read(os.path.join(RUNS, "run_all_dry_run.log"))
    probe = os.path.join(RUNS, "run_all_fail_probe.py")
    fail_run = run([sys.executable, os.path.join(CODE, "run_all.py"),
                    "--scripts", probe + ",10_q1_model.py"])
    fail_log = read(os.path.join(RUNS, "run_all_fail_test.log"))
    order_ok = all(name in dry_run.stdout.splitlines()[i + 1]
                   for i, name in enumerate(run_all.STEPS))

    user_features_sha = sha(os.path.join(DATA, "user_features.parquet"))
    uf_start = {r["path"]: r["sha256"] for r in fp["trees"]["data"]}[
        "data/user_features.parquet"]

    checks = {
        "1_absolute_paths_zero": len(abs_hits) == 0,
        "2_prepare_help_ok": help_run.returncode == 0
        and all(flag in help_run.stdout for flag in ("--raw", "--out", "--force")),
        "3_prepare_refuses_overwrite": guard_run.returncode == 1
        and "exists; use --out" in guard_run.stderr
        and trips_sha == trips_sha_start,
        "4_prepare_run_rows_and_schema": rows_written == 3214096
        and cols_written == 15
        and list(check_trips.columns) == list(ref_trips.columns)
        and len(check_trips) == 3214096,
        "5_dist_matrix_rebuilt": dm["shape"] == [5263, 5263]
        and dm["dtype"] == "float32"
        and dm["reference_shape"] == [5263, 5263]
        and dm["reference_dtype"] == "float32"
        and dm["max_abs_diff"] == 0.0
        and dm["bitwise_equal"] is True
        and dm["sha256_output_array"] == dm["sha256_reference_array"],
        "6_dist_matrix_reference_untouched": dist_sha == dist_sha_start,
        "7_run_all_dry_run": dry_run.returncode == 0 and order_ok,
        "8_run_all_stops_on_failure": fail_run.returncode == 3
        and "FAILED" in fail_run.stdout
        and "[2/2]" not in fail_run.stdout,
        "9_readonly_trees_unchanged": not data_cmp["changed"]
        and not data_cmp["missing"] and not paper_cmp["changed"]
        and not paper_cmp["missing"] and not root_cmp["changed"]
        and not root_cmp["missing"] and not review_cmp["changed"]
        and not review_cmp["missing"],
        "10_scope_limited": set(code_modified) == EXPECTED_MODIFIED
        and set(code_added) == EXPECTED_ADDED
        and "exists, not overwriting" in behaviour_log
        and user_features_sha == uf_start,
    }

    delivery = {
        "task": "T015",
        "title": "Repository cleanup and portability",
        "date": "2026-11-18",
        "status": "delivered",
        "checks": checks,
        "absolute_path_audit": {
            "markers": ["C:\\\\Users", "D:\\\\code"],
            "files_scanned": len([p for p in Path(CODE).rglob("*")
                                  if p.is_file()
                                  and "__pycache__" not in p.parts]),
            "hits": abs_hits,
            "note": "grep-equivalent scan of every file under code/, "
                    "__pycache__ excluded.",
        },
        "portability_changes": [
            {"file": "code/00_prepare.py",
             "old": "hard-coded raw CSV path and output path",
             "new": "--raw (default data/raw/*.csv) and --out "
                    "(default data/trips.parquet), --force to overwrite",
             "evidence": "code/00_prepare.py:20-21,51-81"},
            {"file": "code/01_recon.py", "old": "absolute trips path",
             "new": "os.path.join(DATA, 'trips.parquet')",
             "evidence": "code/01_recon.py:4-6"},
            {"file": "code/02_recon_spatial.py", "old": "absolute trips path",
             "new": "os.path.join(DATA, 'trips.parquet')",
             "evidence": "code/02_recon_spatial.py:4-8"},
            {"file": "code/03_diag.py", "old": "absolute trips and figure paths",
             "new": "os.path.join(DATA, 'trips.parquet') and "
                    "os.path.join(FIG, 'diag_maps.png')",
             "evidence": "code/03_diag.py:6-8,42-43"},
            {"file": "code/04_behaviour.py", "old": "absolute trips and "
                    "user-features paths; unconditional write",
             "new": "os.path.join(DATA, ...); write only with --force",
             "evidence": "code/04_behaviour.py:5-13,71-75"},
        ],
        "prepare": {
            "help_exit": help_run.returncode,
            "help_stdout_head": help_run.stdout.splitlines()[:3],
            "overwrite_guard_exit": guard_run.returncode,
            "overwrite_guard_stderr":
                guard_run.stderr.strip().replace(ROOT, "<repo>"),
            "run_command": "python -I code/00_prepare.py --raw <raw csv> "
                           "--out data/runs/T015/trips_check.parquet",
            "rows_written": rows_written,
            "columns_written": cols_written,
            "columns_match_reference": list(check_trips.columns)
            == list(ref_trips.columns),
            "reference_rows": len(ref_trips),
            "reference_sha256": trips_sha,
            "reference_unchanged": trips_sha == trips_sha_start,
            "log": "data/runs/T015/00_prepare_run.log",
        },
        "dist_matrix": dict(dm, reference_sha256_unchanged=dist_sha == dist_sha_start),
        "run_all": {
            "steps": run_all.STEPS,
            "dry_run_exit": dry_run.returncode,
            "dry_run_log": "data/runs/T015/run_all_dry_run.log",
            "failure_test_exit": fail_run.returncode,
            "failure_test_output": [line.replace(ROOT, "<repo>")
                                    for line in fail_run.stdout.strip().splitlines()],
            "failure_test_log": "data/runs/T015/run_all_fail_test.log",
            "failure_probe": "data/runs/T015/run_all_fail_probe.py",
            "full_run_executed": False,
        },
        "integrity": {
            "baseline": "data/runs/T015/fingerprint_start.json",
            "data_excluding_runs": data_cmp,
            "paper": paper_cmp,
            "review": review_cmp,
            "root_files": root_cmp,
            "code": {"baseline_files": code_cmp["baseline_files"],
                     "modified": code_cmp["changed"],
                     "added": code_cmp["added"],
                     "missing": code_cmp["missing"]},
        },
        "code_changes": [
            {"file": "code/00_prepare.py", "status": "modified",
             "what": "argparse --raw/--out/--force; raw glob resolution; "
                     "refuse to overwrite without --force; DATA from lib",
             "why": "remove the hard-coded desktop input path and the "
                    "hard-coded output path without changing decoding."},
            {"file": "code/01_recon.py", "status": "modified",
             "what": "read trips via DATA", "why": "absolute path removal."},
            {"file": "code/02_recon_spatial.py", "status": "modified",
             "what": "read trips via DATA", "why": "absolute path removal."},
            {"file": "code/03_diag.py", "status": "modified",
             "what": "read trips via DATA; write figure via FIG",
             "why": "absolute path removal."},
            {"file": "code/04_behaviour.py", "status": "modified",
             "what": "read trips via DATA; write user features via DATA; "
                     "skip an existing output unless --force",
             "why": "fix the stale output directory and protect existing "
                    "data."},
            {"file": "code/build_dist_matrix.py", "status": "new",
             "what": "rebuild the 5263x5263 matrix from cells.parquet, "
                     "write data/runs/T015/dist_matrix.npy, compare shape, "
                     "dtype, max abs diff, bitwise equality and SHA-256",
             "why": "the generator was missing from code/."},
            {"file": "code/run_all.py", "status": "new",
             "what": "ordered 10->95 runner with --dry-run, stop on first "
                     "non-zero exit, optional --scripts override",
             "why": "document and enforce the stage order."},
            {"file": "code/t015_delivery.py", "status": "new",
             "what": "this delivery builder", "why": "assemble T015 evidence."},
            {"file": "code/t015_validate.py", "status": "new",
             "what": "independent T015 delivery validator",
             "why": "recompute the self-checks before delivery."},
        ],
        "outputs": {
            "delivery_json": "交付/T015_delivery.json",
            "start_fingerprint": "data/runs/T015/fingerprint_start.json",
            "rebuilt_trips": "data/runs/T015/trips_check.parquet",
            "rebuilt_dist_matrix": "data/runs/T015/dist_matrix.npy",
            "dist_matrix_check": "data/runs/T015/dist_matrix_check.json",
            "logs": ["data/runs/T015/00_prepare_help.log",
                     "data/runs/T015/00_prepare_overwrite_guard.log",
                     "data/runs/T015/00_prepare_run.log",
                     "data/runs/T015/01_recon_run.log",
                     "data/runs/T015/02_recon_spatial_run.log",
                     "data/runs/T015/04_behaviour_noforce.log",
                     "data/runs/T015/run_all_dry_run.log",
                     "data/runs/T015/run_all_fail_test.log"],
        },
        "notes": [
            "data/dist_matrix.npy is reproduced by Euclidean distances in the "
            "projected km frame with the diagonal set to 0.35/sqrt(2), exactly "
            "as 10_q1_model.py:34-35 generated it. It is not a haversine "
            "matrix: haversine on the cell centres differs on the diagonal by "
            "0.2475 km. The task sheet's word 'spherical' therefore describes "
            "a different construction; build_dist_matrix.py follows the "
            "actual generator so the rebuild is bit-identical.",
            "data/raw/ does not exist in this workspace, so the default --raw "
            "glob was exercised only through --help. The verification run "
            "passed the existing raw CSV with an explicit --raw value; that "
            "path is not stored in code or in this delivery.",
            "The --force overwrite paths of 00_prepare.py and 04_behaviour.py "
            "were deliberately not executed, to protect the existing data. "
            "Only the refusal path was tested.",
            "03_diag.py's FIG-joined save path was changed but not executed, "
            "to avoid overwriting an existing figure; its syntax and the DATA "
            "read path were checked separately.",
            "run_all.py was not run end-to-end, because that would rerun "
            "Q1-Q5. Only the dry-run order and the stop-on-failure probe were "
            "executed.",
            "Portability is still Windows-first. lib.py and several later "
            "scripts concatenate paths with backslashes (for example "
            "DATA + r'\\trips.parquet' or FIG + backslash), which works on "
            "Windows but not on POSIX. T015's scope was the absolute-path "
            "removal in 00-04; a full os.path.join sweep is a separate "
            "change.",
            "trips_check.parquet matches data/trips.parquet in row count and "
            "column names, not byte-for-byte; parquet metadata and "
            "compression can differ between pyarrow versions.",
            "README.md and DATA.md still mention the old script names; the "
            "reviewer owns their wording, and they were not edited.",
            "No git command was run; .git/, .gitignore, .gitattributes, "
            "README.md and DATA.md were not modified.",
            "The code/ audit files were included in the absolute-path scan; "
            "they contain no matches.",
        ],
    }
    os.makedirs(DELIV, exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8") as handle:
        json.dump(delivery, handle, ensure_ascii=False, indent=1)
    print("saved", OUT)
    print(json.dumps(checks, ensure_ascii=False, indent=1))
    print("rows", rows_written, "cols", cols_written,
          "dist maxdiff", dm["max_abs_diff"],
          "bitwise", dm["bitwise_equal"])


if __name__ == "__main__":
    main()
