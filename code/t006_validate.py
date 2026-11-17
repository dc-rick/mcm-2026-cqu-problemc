"""Independent validator for the T006 delivery.

Recomputes every self-check from the arrays and JSON files; it does not import
t006_delivery.py.  Prints one line per check and the total.
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
T015_FP = os.path.join(DATA, "runs", "T015", "fingerprint_start.json")
T001_FP = os.path.join(DATA, "runs", "T001", "fingerprint_start.json")

sys.path.insert(0, CODE)
import lib  # noqa: E402

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


def hashes(base, skip_runs=False):
    out = {}
    for path in Path(base).rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        rel = os.path.relpath(str(path), ROOT).replace(os.sep, "/")
        if skip_runs and rel.startswith("data/runs/"):
            continue
        out[rel] = sha(str(path))
    return out


def main():
    results = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))
        print(("PASS " if ok else "FAIL ") + name + (("  " + detail) if detail else ""))

    z = np.load(os.path.join(RUN, "q1_od.npz"))
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
    struct_ok = set(z.files) == set(required)
    for key, (shape, dtype) in required.items():
        if key not in z.files:
            struct_ok = False
            continue
        struct_ok &= list(z[key].shape) == list(shape) and z[key].dtype == dtype
    check("1 structure: keys/shapes/dtypes", struct_ok, f"{len(z.files)} keys")

    neg = {k: int((z[k].astype(np.float64) < 0).sum())
           for k in ("od_q1", "arr_q1_cell", "dep_obs", "arr_obs_cell")}
    check("2 no negative demand", all(v == 0 for v in neg.values()), str(neg))

    cells = pd.read_parquet(os.path.join(DATA, "cells.parquet"))
    zid = cells.zone.to_numpy().astype(int)
    NZ = int(zid.max()) + 1
    od = z["od_q1"].astype(np.float64)
    dep = z["dep_obs"].astype(np.float64)
    dep_zone = np.stack([np.bincount(zid, weights=dep[h], minlength=NZ)
                         for h in range(24)])
    row_err = float((np.abs(od.sum(2) - dep_zone)
                     / np.maximum(dep_zone, 1e-12)).max())
    check("3 od_q1 row sums == dep zone totals", row_err < 1e-6,
          f"max rel {row_err:.3e}")

    dep_h = dep.sum(1)
    arr_h = z["arr_q1_cell"].astype(np.float64).sum(1)
    cons = float(np.max(np.abs(arr_h - dep_h) / np.maximum(dep_h, 1e-12)))
    check("4 hourly arrivals == hourly departures", cons < 1e-6,
          f"max rel {cons:.3e}")

    q1v = json.load(io.open(os.path.join(RUN, "q1_vs_obs.json"),
                            encoding="utf-8"))
    ref = pd.read_csv(os.path.join(DATA, "q1_validation.csv")).set_index("band")
    rows = q1v["top5"]["per_band"]
    band_ok = all(abs(r["top5"] - float(ref.loc[r["band"], "top5_model"])) < 0.005
                  for r in rows)
    w = np.array([r["trips"] for r in rows], dtype=float)
    w_top = float(sum(r["top5"] * r["trips"] for r in rows) / w.sum())
    w_ref = float(sum(float(ref.loc[r["band"], "top5_model"]) * r["trips"]
                      for r in rows) / w.sum())
    check("5 top-5 matches q1_validation",
          band_ok and abs(w_top - q1v["top5"]["weighted"]) < 1e-9
          and abs(w_ref - q1v["top5"]["reference_weighted"]) < 1e-9
          and abs(w_top - w_ref) < 0.005,
          f"weighted {w_top:.7f} vs {w_ref:.7f}")

    # independent observed same-cell share on the seven typical days
    df = lib.load_trips(lib.TYP_DAYS)
    df = lib.assign_cells(df, cells)
    okd = df.o.notna() & df.d.notna()
    sub = df[okd]
    obs_same = float((sub.o.astype(int).values == sub.d.astype(int).values).mean())
    same = q1v["same_cell_share"]
    ks = q1v["distance_ks"]["ks"]
    check("6 same-cell and KS reported",
          0 <= same["model"] <= 1 and 0 <= same["observed"] <= 1
          and 0 <= ks <= 1
          and abs(obs_same - same["observed"]) < 1e-9,
          f"model {100*same['model']:.2f}% obs {100*obs_same:.2f}% KS {ks:.4f}")

    q2 = json.load(io.open(os.path.join(RUN, "q2_q1_smoke.json"),
                           encoding="utf-8"))
    q2_ok = (q2["q1"]["status"] == 2 and q2["obs"]["status"] == 2
             and q2["q1"]["alpha"] == q2["obs"]["alpha"] == 1.0
             and q2["q1"]["K"] == q2["obs"]["K"] == 60
             and q2["q1"]["unmet"] <= 1e-6 and q2["obs"]["unmet"] <= 1e-6)
    d = q2["difference"]["q1_minus_obs"]
    q2_ok &= all(abs(d[k] - (q2["q1"][k] - q2["obs"][k]))
                 <= 1e-9 * max(1.0, abs(d[k]))
                 for k in ("moved", "truck_km", "unmet", "dev", "obj"))
    plans = [np.load(os.path.join(RUN, f"q2_plan_{s}_smoke.npz"))
             for s in ("q1", "obs")]
    q2_ok &= all(set(p.files) == {"X", "M", "I", "U", "XM"}
                 and p["XM"].shape == (24, 50, 50)
                 and p["I"].shape == (50, 24) for p in plans)
    defaults_ok = all(
        'choices=("q1", "obs"), default="q1"'
        in read(os.path.join(CODE, n))
        for n in ("20_q2_diag.py", "21_q2_opt.py", "22_q2_sim.py",
                  "40_q4_efence.py"))
    check("7 Q2 smokes optimal, default q1", q2_ok and defaults_ok,
          f"q1 obj {q2['q1']['obj']:.0f}, obs obj {q2['obs']['obj']:.0f}")

    q4 = json.load(io.open(os.path.join(RUN, "q4_q1_smoke.json"),
                           encoding="utf-8"))
    t1, t2 = q4["q1"]["arrivals_per_day"], q4["obs"]["arrivals_per_day"]
    q4_ok = abs(t1 - t2) / t2 < 1e-6
    q4_ok &= (q4["settings"]["radius_m"] == 450 and q4["settings"]["reps"] == 300
              and q4["q1"]["fences"] > 0 and q4["obs"]["fences"] > 0
              and 0 < q4["q1"]["coverage"] < 1 and 0 < q4["obs"]["coverage"] < 1)
    q4_ok &= all(os.path.exists(os.path.join(RUN, s, n))
                 for s in ("q4_q1", "q4_obs")
                 for n in ("q4_summary.json", "q4_frontier.npz", "q4_sites.npy"))
    check("8 Q4 R=450 both surfaces, equal totals", q4_ok,
          f"{t1:.1f} vs {t2:.1f}")

    fp15 = json.load(io.open(T015_FP, encoding="utf-8"))["trees"]
    data_cmp = hashes(DATA, skip_runs=True)
    paper_cmp = hashes(PAPER)
    fp1 = json.load(io.open(T001_FP, encoding="utf-8"))["trees"]["review"]
    review_cmp = hashes(os.path.join(ROOT, "review"))
    ro_ok = True
    for name, cur, base in (("data", data_cmp, fp15["data"]),
                            ("paper", paper_cmp, fp15["paper"]),
                            ("review", review_cmp, fp1)):
        b = {r["path"]: r["sha256"] for r in base}
        ro_ok &= not [p for p in b if p in cur and cur[p] != b[p]]
        ro_ok &= not [p for p in b if p not in cur]
    status_lines = subprocess.run(["git", "-c", "core.quotepath=false",
                                   "status", "--porcelain"], cwd=ROOT,
                                  capture_output=True, text=True,
                                  encoding="utf-8", errors="replace").stdout.splitlines()
    root_paths = [r["path"] for r in fp15["root"]]
    ro_ok &= not [line for line in status_lines if line[3:] in root_paths]
    check("9 data/paper/review and root docs unchanged", ro_ok)

    modified = sorted(line[3:] for line in status_lines if line.startswith(" M "))
    untracked = sorted(line[3:] for line in status_lines
                       if line.startswith("?? ")
                       and line[3:] != "交付/T006_delivery.json")
    scope_ok = (set(modified) == {"code/20_q2_diag.py", "code/21_q2_opt.py",
                                  "code/22_q2_sim.py", "code/40_q4_efence.py"}
                and set(untracked) == {"code/12_q1_od.py",
                                       "code/t006_delivery.py",
                                       "code/t006_validate.py"})
    abs_hits = []
    for rel in sorted(set(modified) | set(untracked)):
        for lineno, line in enumerate(read(os.path.join(ROOT, rel)).splitlines(), 1):
            if any(m in line for m in ABS_MARKERS):
                abs_hits.append((rel, lineno))
    required_outputs = [
        "q1_od.npz", "q1_vs_obs.json", "q2_q1_smoke.json", "q4_q1_smoke.json",
        "q2_smoke_q1.json", "q2_smoke_obs.json", "q2_plan_q1_smoke.npz",
        "q2_plan_obs_smoke.npz", "q2_state_q1.npz", "q4_q1_run.log",
        "q4_obs_run.log",
    ]
    missing = [p for p in required_outputs
               if not os.path.exists(os.path.join(RUN, p))]
    check("10 scope, outputs, no absolute paths",
          scope_ok and not abs_hits and not missing,
          f"modified {len(modified)}, untracked {len(untracked)}")

    with io.open(os.path.join(ROOT, "交付", "T006_delivery.json"),
                 encoding="utf-8") as handle:
        delivery = json.load(handle)
    all_true = all(delivery["checks"].values())
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nvalidator: {passed}/{len(results)} passed; "
          f"delivery checks all true: {all_true}")
    return 0 if passed == len(results) and all_true else 1


if __name__ == "__main__":
    raise SystemExit(main())
