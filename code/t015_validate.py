"""Independent validator for the T015 delivery."""
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
DELIV = os.path.join(ROOT, "交付")
RUNS = os.path.join(DATA, "runs", "T015")
FP_PATH = os.path.join(RUNS, "fingerprint_start.json")
T001_FP_PATH = os.path.join(DATA, "runs", "T001", "fingerprint_start.json")
DELIVERY = os.path.join(DELIV, "T015_delivery.json")
ABS_MARKERS = ("C:" + chr(92) + "Users", "D:" + chr(92) + "code")
EXPECTED_MODIFIED = {"00_prepare.py", "01_recon.py", "02_recon_spatial.py",
                     "03_diag.py", "04_behaviour.py"}
EXPECTED_ADDED = {"build_dist_matrix.py", "run_all.py",
                  "t015_delivery.py", "t015_validate.py"}

ok = []


def check(name, cond, detail=""):
    ok.append((name, bool(cond), detail))
    print(("PASS " if cond else "FAIL ") + name
          + (" | " + detail if detail else ""))


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


def run(cmd):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def main():
    d = json.load(io.open(DELIVERY, encoding="utf-8"))
    fp = json.load(io.open(FP_PATH, encoding="utf-8"))["trees"]
    t001 = json.load(io.open(T001_FP_PATH, encoding="utf-8"))["trees"]

    check("delivery checks all true", all(d["checks"].values()),
          json.dumps(d["checks"], ensure_ascii=False))

    hits = []
    for path in sorted(Path(CODE).rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        for marker in ABS_MARKERS:
            if marker in read(str(path)):
                hits.append(str(path))
    check("absolute paths zero in code", not hits, str(hits))

    help_run = run([sys.executable, "-I", os.path.join(CODE, "00_prepare.py"),
                    "--help"])
    check("prepare --help exits 0", help_run.returncode == 0)
    check("prepare help lists --raw/--out/--force",
          all(flag in help_run.stdout for flag in ("--raw", "--out", "--force")))

    guard_run = run([sys.executable, "-I", os.path.join(CODE, "00_prepare.py"),
                     "--out", os.path.join(DATA, "trips.parquet")])
    check("prepare refuses overwrite without --force",
          guard_run.returncode == 1 and "exists; use --out" in guard_run.stderr,
          guard_run.stderr.strip())
    trips_sha = sha(os.path.join(DATA, "trips.parquet"))
    baseline = {row["path"]: row["sha256"] for tree in fp.values() for row in tree}
    check("data/trips.parquet unchanged",
          trips_sha == baseline["data/trips.parquet"])

    ref = pd.read_parquet(os.path.join(DATA, "trips.parquet"))
    chk = pd.read_parquet(os.path.join(RUNS, "trips_check.parquet"))
    check("trips_check 3,214,096 rows", len(chk) == 3214096, str(len(chk)))
    check("trips_check columns match reference",
          list(chk.columns) == list(ref.columns))
    check("trips_check shape matches reference", chk.shape == ref.shape,
          str(chk.shape))

    dm = json.load(io.open(os.path.join(RUNS, "dist_matrix_check.json"),
                           encoding="utf-8"))
    check("dist matrix shape/dtype",
          dm["shape"] == [5263, 5263] and dm["dtype"] == "float32")
    check("dist matrix reference shape/dtype",
          dm["reference_shape"] == [5263, 5263]
          and dm["reference_dtype"] == "float32")
    check("dist matrix max abs diff 0", dm["max_abs_diff"] == 0.0)
    check("dist matrix bitwise equal", dm["bitwise_equal"] is True)
    check("dist matrix SHA equal",
          dm["sha256_output_array"] == dm["sha256_reference_array"])
    check("dist matrix output file hash recorded",
          sha(os.path.join(RUNS, "dist_matrix.npy")) == dm["sha256_output_file"])
    dist_sha = sha(os.path.join(DATA, "dist_matrix.npy"))
    check("tracked dist matrix unchanged",
          dist_sha == baseline["data/dist_matrix.npy"])
    build_src = read(os.path.join(CODE, "build_dist_matrix.py"))
    check("build script mirrors 10_q1_model construction",
          "lib.dist_matrix(xy)" in build_src
          and "np.fill_diagonal(D, 0.35 / np.sqrt(2))" in build_src)

    dry = run([sys.executable, os.path.join(CODE, "run_all.py"), "--dry-run"])
    lines = dry.stdout.splitlines()
    steps = [ln.split(". ", 1)[1] for ln in lines[1:] if ". " in ln]
    check("run_all dry-run exits 0", dry.returncode == 0)
    check("run_all dry-run has 13 stages", len(steps) == 13, str(len(steps)))
    check("run_all dry-run order 10 -> 95",
          all(f"{n:02d}_" in steps[i] for i, n in enumerate(
              [10, 11, 20, 21, 22, 23, 24, 30, 40, 50, 60, 90, 95])))

    probe = os.path.join(RUNS, "run_all_fail_probe.py")
    fail = run([sys.executable, os.path.join(CODE, "run_all.py"),
                "--scripts", probe + ",10_q1_model.py"])
    check("run_all stops on failure",
          fail.returncode == 3 and "FAILED" in fail.stdout
          and "[2/2]" not in fail.stdout,
          f"rc={fail.returncode}")

    behaviour_log = read(os.path.join(RUNS, "04_behaviour_noforce.log"))
    check("04 skips existing output without --force",
          "exists, not overwriting" in behaviour_log)
    check("data/user_features.parquet unchanged",
          sha(os.path.join(DATA, "user_features.parquet"))
          == baseline["data/user_features.parquet"])

    data_now = current_tree(DATA, skip_runs=True)
    data_base = {row["path"]: row["sha256"] for row in fp["data"]}
    check("data excluding runs unchanged",
          data_now == data_base,
          str(sorted(set(data_now) ^ set(data_base))))

    paper_now = current_tree(PAPER)
    paper_base = {row["path"]: row["sha256"] for row in fp["paper"]}
    check("paper unchanged", paper_now == paper_base,
          str(sorted(set(paper_now) ^ set(paper_base))))

    root_paths = [row["path"] for row in fp["root"]]
    check("README/DATA/.git* unchanged",
          all(os.path.exists(os.path.join(ROOT, p))
              and sha(os.path.join(ROOT, p)) == baseline[p]
              for p in root_paths))

    review_now = current_tree(os.path.join(ROOT, "review"))
    review_base = {row["path"]: row["sha256"] for row in t001["review"]}
    check("review unchanged", review_now == review_base,
          str(sorted(set(review_now) ^ set(review_base))))

    code_now = current_tree(CODE)
    code_base = {row["path"]: row["sha256"] for row in fp["code"]}
    modified = sorted(os.path.basename(p) for p in code_now
                      if p in code_base and code_now[p] != code_base[p])
    added = sorted(os.path.basename(p) for p in code_now if p not in code_base)
    missing = sorted(p for p in code_base if p not in code_now)
    check("code modified exactly 00-04", set(modified) == EXPECTED_MODIFIED,
          str(modified))
    check("code added exactly four T015 scripts", set(added) == EXPECTED_ADDED,
          str(added))
    check("no code files missing", not missing, str(missing))

    documented = {item["file"] for item in d["code_changes"]}
    expected_docs = {"code/" + name for name in EXPECTED_MODIFIED | EXPECTED_ADDED}
    check("every code change documented", documented == expected_docs,
          str(sorted(documented ^ expected_docs)))

    src = read(os.path.join(CODE, "00_prepare.py"))
    check("00_prepare argparse flags present",
          all(flag in src for flag in ("--raw", "--out", "--force")))
    for name in ("01_recon.py", "02_recon_spatial.py", "03_diag.py",
                 "04_behaviour.py"):
        text = read(os.path.join(CODE, name))
        check(name + " uses lib DATA", "from lib import DATA" in text)

    check("delivery date present", bool(d.get("date")))
    check("notes record untested branches",
          len(d["notes"]) >= 9
          and any("Windows-first" in n for n in d["notes"])
          and any("not run end-to-end" in n for n in d["notes"]))
    check("start fingerprint has four trees",
          set(fp) == {"data", "paper", "code", "root"})
    check("rebuilt outputs exist",
          all(os.path.exists(os.path.join(RUNS, name)) for name in
              ("trips_check.parquet", "dist_matrix.npy",
               "dist_matrix_check.json", "run_all_dry_run.log",
               "run_all_fail_test.log")))

    failed = [name for name, value, _ in ok if not value]
    print(f"\n{len(ok) - len(failed)}/{len(ok)} checks passed")
    if failed:
        raise SystemExit("FAILED: " + "; ".join(failed))
    print("T015 delivery consistent")


if __name__ == "__main__":
    main()
