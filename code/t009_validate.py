"""Independent validator for the T009 Q4 delivery.

Recomputes the demand surfaces, both initial-inventory vectors, the greedy
partition, sampled Monte-Carlo capacities, overlap counts, frontier knees,
hold-out validation and the read-only/scope checks.  It does not import
t009_delivery.py or t009_q4.py.
"""
import hashlib
import io
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import lib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, "code")
DATA = os.path.join(ROOT, "data")
PAPER = os.path.join(ROOT, "paper")
RUN = os.path.join(DATA, "runs", "T009")
DELIV = os.path.join(ROOT, "交付")
T015_FP = os.path.join(DATA, "runs", "T015", "fingerprint_start.json")
T001_FP = os.path.join(DATA, "runs", "T001", "fingerprint_start.json")
Q1_OD = os.path.join(DATA, "runs", "T006", "q1_od.npz")

SCALE = 1.0 / 0.128
FLEET_MEAS = 343638.0
FLEET_CLOSED = 495540.0
R = 0.45
REPS = 300
SNAP = pd.Timestamp("2017-05-16 00:00:00")
TOL = 1e-6
BASE_COMMIT = "d731585"
T009_FILES = {"code/t009_q4.py", "code/t009_delivery.py",
              "code/t009_validate.py"}
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


def git_lines(args):
    proc = subprocess.run(["git", "-c", "core.quotepath=false"] + args,
                          cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    return proc.returncode, proc.stdout.splitlines()


def load_inputs():
    z = np.load(Q1_OD)
    arr_q1 = z["arr_q1_cell"].astype(np.float64) * SCALE
    arr_obs = z["arr_obs_cell"].astype(np.float64) * SCALE
    dep = z["dep_obs"].astype(np.float64) * SCALE
    cells = lib.get_cells()
    xy = cells[["x", "y"]].to_numpy()
    D = np.load(os.path.join(DATA, "dist_matrix.npy")).astype(np.float32)
    df = lib.load_trips(lib.ALL_DAYS)
    df = lib.assign_cells(df, cells)
    df = df[df.o.notna() & df.d.notna()].copy()
    df["o"] = df.o.astype(int)
    df["d"] = df.d.astype(int)
    before = df[df.starttime < SNAP]
    last = before.sort_values("starttime").groupby("bikeid").tail(1)
    cnt = np.bincount(last.d.values, minlength=len(cells)).astype(np.float64)
    I0_meas = cnt / cnt.sum() * FLEET_MEAS
    cum = np.cumsum(arr_q1 - dep, axis=0)
    raw = np.maximum(0.0, -cum.min(axis=0))
    I0_fleet = raw / raw.sum() * FLEET_CLOSED
    return dict(arr_q1=arr_q1, arr_obs=arr_obs, dep=dep, cells=cells,
                xy=xy, D=D, df=df, I0_meas=I0_meas, I0_fleet=I0_fleet,
                raw_sum=float(raw.sum()))


def greedy_partition(arr, D, R=R):
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
    return np.repeat(x / 4.0, 4, axis=0)


def mc_traj(a, dd, i0, reps=REPS, seed=0):
    """Local copy of the capacity rule: initial on the trajectory at t=0."""
    rng = np.random.default_rng(seed)
    A = rng.poisson(a, size=(reps, len(a)))
    D = rng.poisson(dd, size=(reps, len(dd)))
    occ = np.full(reps, float(i0))
    peak = occ.copy()
    for t in range(len(a)):
        occ = np.maximum(0.0, occ + A[:, t] - D[:, t])
        peak = np.maximum(peak, occ)
    return float(np.percentile(peak, 95))


def knee(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xn = (x - x.min()) / max(x.max() - x.min(), 1e-12)
    yn = (y - y.min()) / max(y.max() - y.min(), 1e-12)
    dx, dy = xn[-1] - xn[0], yn[-1] - yn[0]
    den = np.hypot(dx, dy)
    dist = np.abs(dy * (xn - xn[0]) - dx * (yn - yn[0])) / max(den, 1e-12)
    return int(np.argmax(dist))


def overlap_counts(sites, arr, I0, D):
    near = D <= R
    mult = near[:, sites].sum(1) if sites else np.zeros(D.shape[0], int)
    covered = mult > 0
    cell_cap = I0 + arr.max(0)
    dup = float((cell_cap * np.maximum(mult - 1, 0)).sum())
    return {
        "cells_total": int(len(mult)),
        "cells_0": int((mult == 0).sum()),
        "cells_1": int((mult == 1).sum()),
        "cells_2": int((mult == 2).sum()),
        "cells_3plus": int((mult >= 3).sum()),
        "cells_covered": int(covered.sum()),
        "coverage_multiplicity": float(mult[covered].mean()) if covered.any() else 0.0,
        "duplicate_capacity_slots": dup,
        "duplicate_capacity_share": float(
            dup / max(cell_cap[covered].sum(), 1e-12)),
    }


def close(a, b, tol=1e-8):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


def main():
    d = load_inputs()
    cells = d["cells"]
    C = len(cells)
    xy = d["xy"]
    zid = cells.zone.to_numpy().astype(int)
    arrs = {"q1": d["arr_q1"], "obs": d["arr_obs"]}
    I0s = {"meas343": d["I0_meas"], "fleet495": d["I0_fleet"]}
    dep15 = to15(d["dep"])

    res = json.load(io.open(os.path.join(RUN, "q4_results.json"),
                            encoding="utf-8"))
    fr = json.load(io.open(os.path.join(RUN, "q4_frontier.json"),
                           encoding="utf-8"))
    ov = json.load(io.open(os.path.join(RUN, "q4_overlap.json"),
                           encoding="utf-8"))
    fences = pd.read_csv(os.path.join(RUN, "q4_fences.csv"))

    results = []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(("PASS " if ok else "FAIL ") + name + (("  " + detail) if detail else ""))

    # 1 --- demand surfaces -------------------------------------------------
    day_total = 2143417.4110091086
    surf_ok = all(close(arrs[k].sum(), day_total, 1e-8) for k in arrs)
    surf_ok &= close(d["dep"].sum(), day_total, 1e-8)
    surf_ok &= arrs["q1"].shape == arrs["obs"].shape == (24, C)
    surf_ok &= np.abs(arrs["q1"] - arrs["obs"]).sum() > 0
    surf_ok &= all(close(res["surfaces"][k]["total_arrivals_day"],
                         day_total, 1e-8) for k in arrs)
    surf_ok &= close(res["surfaces"]["q1"]["covered_arrivals_day"]
                     + res["surfaces"]["q1"]["uncovered_arrivals_day"],
                     day_total, 1e-9)
    check("1 demand surfaces full scale", surf_ok,
          f"q1/obs daily total {arrs['q1'].sum():,.2f}")

    # 2 --- inventory levels and magnitude ---------------------------------
    inv_ok = close(d["I0_meas"].sum(), FLEET_MEAS, 1e-12)
    inv_ok &= close(d["I0_fleet"].sum(), FLEET_CLOSED, 1e-12)
    inv_ok &= d["I0_meas"].min() >= 0 and d["I0_fleet"].min() >= 0
    inv_ok &= close(res["validation"]["initial_inventory"]["meas343_sum"],
                    FLEET_MEAS, 1e-12)
    inv_ok &= close(res["validation"]["initial_inventory"]["fleet495_sum"],
                    FLEET_CLOSED, 1e-12)
    corr_i = float(np.corrcoef(d["I0_meas"], d["I0_fleet"])[0, 1])
    inv_ok &= close(res["validation"]["initial_inventory"]["corr"], corr_i, 1e-9)
    full = res["surfaces"]
    mag = res["magnitude_check"]
    mag_ok = all(mag["measured_ok"].values())
    mag_ok &= mag["fleet_ok"]["q1"] and mag["fleet_within_one_percent"]["obs"]
    mag_ok &= mag["obs_fleet_excess"]["slots_above_900k"] > 0
    mag_ok &= "no_double_count_audit" in mag["obs_fleet_excess"]
    mag_ok &= close(full["q1"]["full_coverage"]["meas343"]["standing_total"],
                    FLEET_MEAS, 1e-12)
    mag_ok &= close(full["q1"]["full_coverage"]["fleet495"]["standing_total"],
                    FLEET_CLOSED, 1e-12)
    check("2 inventory levels and magnitude", inv_ok and mag_ok,
          f"I0 {d['I0_meas'].sum():,.0f}/{d['I0_fleet'].sum():,.0f}; "
          f"full capacity {full['q1']['full_coverage']['meas343']['capacity_total']:,.0f}")

    # 3 --- capacity includes standing inventory ---------------------------
    cap_ok = len(fences) == int(full["q1"]["n_selected"])
    for lvl in I0s:
        cap_ok &= bool((fences[f"capacity_{lvl}"]
                        >= fences[f"standing_{lvl}"] - 1e-9).all())
        cap_ok &= bool(np.allclose(fences[f"buffer_{lvl}"],
                                   fences[f"capacity_{lvl}"]
                                   - fences[f"standing_{lvl}"], atol=1e-9))
        cap_ok &= close(fences[f"capacity_{lvl}"].sum(),
                        full["q1"][lvl]["capacity_total"], 1e-9)
        cap_ok &= close(fences[f"standing_{lvl}"].sum(),
                        full["q1"][lvl]["standing_total"], 1e-9)
    check("3 capacity includes standing inventory", cap_ok,
          f"{len(fences)} fences, min capacity "
          f"{fences.capacity_meas343.min():,.1f}")

    # 4 --- Monte-Carlo capacity and initial-before-max --------------------
    sites_q1, members_q1, cov_q1, new_q1, demand_q1 = greedy_partition(
        d["arr_q1"], d["D"])
    arr15 = to15(d["arr_q1"])
    nsel = int(full["q1"]["n_selected"])
    sample = [0, 1, nsel // 2, nsel - 1]
    mc_ok = True
    worst = 0.0
    for i in sample:
        mem = members_q1[i]
        a = arr15[:, mem].sum(1)
        dd = dep15[:, mem].sum(1)
        for lvl, I0 in I0s.items():
            got = mc_traj(a, dd, I0[mem].sum(), REPS, 1000 + i)
            want = float(fences.iloc[i][f"capacity_{lvl}"])
            worst = max(worst, abs(got - want))
            mc_ok &= close(got, want, 1e-9)
    # semantics: starting at I0 and clipping departures must NOT give
    # (zero-initial peak) + I0
    a = np.array([0.0, 100.0])
    dd = np.array([150.0, 0.0])
    correct = mc_traj(a, dd, 100.0, reps=2000, seed=7)
    zero = mc_traj(a, dd, 0.0, reps=2000, seed=7)
    wrong = zero + 100.0
    mc_ok &= correct < 160.0 and wrong > 160.0 and (wrong - correct) > 50.0
    check("4 MC capacity and initial-before-max", mc_ok,
          f"sample max diff {worst:.2e}; synthetic {correct:.1f} vs {wrong:.1f}")

    # 5 --- exclusive attribution ------------------------------------------
    nsel = int(full["q1"]["n_selected"])
    attr_ok = len(sites_q1) == int(full["q1"]["n_full_coverage"])
    attr_ok &= len(fences) == nsel
    attr_ok &= np.allclose(fences[["x", "y"]].to_numpy(),
                           xy[np.array(sites_q1[:nsel])], atol=1e-9)
    counts = np.zeros(C, int)
    for i in range(nsel):
        mem = members_q1[i]
        counts += mem
        attr_ok &= int(fences.iloc[i].n_assigned) == int(mem.sum())
        parsed = [int(v) for v in str(fences.iloc[i].assigned_cells).split()]
        attr_ok &= parsed == list(np.where(mem)[0])
        attr_ok &= bool((d["D"][sites_q1[i], mem] <= R + 1e-6).all())
    attr_ok &= int(counts.max()) <= 1
    attr_ok &= int(counts.sum()) == int(new_q1[:nsel].sum())
    attr_ok &= int(new_q1[:nsel].sum()) == int(
        (d["D"][:, np.array(sites_q1[:nsel])] <= R).any(1).sum())
    check("5 exclusive attribution", attr_ok,
          f"max multiplicity {counts.max()}, assigned cells {counts.sum()}")

    # 6 --- overlap counts --------------------------------------------------
    ov_ok = True
    for name in arrs:
        sites, members, _, _, _ = greedy_partition(arrs[name], d["D"])
        ns = int(full[name]["n_selected"])
        for key, ss in (("selected", sites[:ns]), ("full_path", sites)):
            rec = overlap_counts(ss, arrs[name], I0s["meas343"], d["D"])
            for field, val in rec.items():
                ov_ok &= close(ov[name][key][field], val, 1e-9)
        ov_ok &= (ov[name]["selected"]["cells_0"]
                  + ov[name]["selected"]["cells_1"]
                  + ov[name]["selected"]["cells_2"]
                  + ov[name]["selected"]["cells_3plus"]) == C
    check("6 overlap counts recomputed", ov_ok,
          f"q1 selected multiplicity {ov['q1']['selected']['coverage_multiplicity']:.6f}")

    # 7 --- frontier, knee, 626 --------------------------------------------
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
                k = knee(cost, pts["coverage"])
                got = fr[name]["knees"][lvl][str(lam)]
                fr_ok &= int(got["n"]) == k + 1
                fr_ok &= close(got["cost"], cost[k], 1e-9)
                fr_ok &= close(got["coverage"], pts["coverage"][k], 1e-12)
                cpa = cost[k] / max(pts["coverage"][k]
                                    * float(arrs[name].sum()), 1e-12)
                fr_ok &= close(got["cost_per_covered_arrival"], cpa, 1e-9)
        fr_ok &= int(full[name]["n_selected"]) == int(
            fr[name]["knees"]["meas343"]["150"]["n"])
        fr_ok &= close(full[name]["coverage_selected"],
                       pts["coverage"][int(full[name]["n_selected"]) - 1],
                       1e-12)
        p626 = fr[name]["position_626"]
        fr_ok &= p626["n"] == 626
        fr_ok &= close(p626["coverage"], pts["coverage"][625], 1e-12)
        for lvl in I0s:
            fr_ok &= close(p626[f"capacity_{lvl}"],
                           pts[f"capacity_{lvl}"][625], 1e-9)
    for lvl in I0s:
        fr_ok &= close(np.sum(fences[f"capacity_{lvl}"]),
                       fr["q1"]["points"][f"capacity_{lvl}"][nsel - 1], 1e-9)
    check("7 frontier monotone, knee and 626", fr_ok,
          f"q1 selected n={nsel}, 626 coverage "
          f"{fr['q1']['position_626']['coverage']:.4f}")

    # 8 --- hold-out validation --------------------------------------------
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
    hold_ok = close(val["cell_corr"], corr_v, 1e-9)
    hold_ok &= close(val["l1_normalized"], l1_v, 1e-9)
    hold_ok &= int(val["top100_overlap"]) == top_v
    hold_ok &= close(val["q1_daily_total"], pred.sum(), 1e-12)
    hold_ok &= close(val["holdout_daily_total"], obs_te.sum(), 1e-12)
    hold_ok &= abs(corr_v - 0.8041443151446922) <= 1e-9
    check("8 hold-out validation", hold_ok,
          f"corr {corr_v:.5f}, L1 {l1_v:.5f}, top100 {top_v}")

    # 9 --- zone comparison and Q1-minus-obs -------------------------------
    NZ = int(zid.max()) + 1
    zq1 = np.bincount(zid, weights=d["arr_q1"].sum(0), minlength=NZ)
    zobs = np.bincount(zid, weights=d["arr_obs"].sum(0), minlength=NZ)
    zc = res["zone_comparison"]
    z_ok = close(zc["l1"], np.abs(zq1 - zobs).sum(), 1e-9)
    z_ok &= close(zc["l1_normalized"],
                  np.abs(zq1 - zobs).sum() / zobs.sum(), 1e-9)
    z_ok &= close(zc["corr"], np.corrcoef(zq1, zobs)[0, 1], 1e-9)
    z_ok &= close(zc["q1_total"], zq1.sum(), 1e-12)
    z_ok &= close(zc["obs_total"], zobs.sum(), 1e-12)
    diff = res["q1_minus_obs"]
    z_ok &= int(diff["n_selected"]) == int(full["q1"]["n_selected"]
                                           - full["obs"]["n_selected"])
    z_ok &= close(diff["coverage_selected"],
                  full["q1"]["coverage_selected"]
                  - full["obs"]["coverage_selected"], 1e-12)
    for lvl in I0s:
        z_ok &= close(diff[f"capacity_{lvl}"],
                      full["q1"][lvl]["capacity_total"]
                      - full["obs"][lvl]["capacity_total"], 1e-9)
    z_ok &= "planning threshold" in res["b5"]
    z_ok &= "39.34" in res["settings"]["night_note"]
    check("9 zone comparison and Q1-minus-obs", z_ok)

    # 10 --- read-only, scope, outputs, no absolute paths ------------------
    fp15 = json.load(io.open(T015_FP, encoding="utf-8"))["trees"]
    fp1 = json.load(io.open(T001_FP, encoding="utf-8"))["trees"]["review"]
    ro = True
    for cur, base in ((hashes(DATA, skip_runs=True), fp15["data"]),
                      (hashes(PAPER), fp15["paper"]),
                      (hashes(os.path.join(ROOT, "review")), fp1)):
        b = {r["path"]: r["sha256"] for r in base}
        ro &= not [p for p in b if p in cur and cur[p] != b[p]]
        ro &= not [p for p in b if p not in cur]
    rc, status = git_lines(["status", "--porcelain"])
    root_paths = [r["path"] for r in fp15["root"]]
    ro &= not [l for l in status if l[3:] in root_paths]
    rc, diff = git_lines(["diff", "--name-only", BASE_COMMIT, "--", "code"])
    modified = sorted(p for p in diff if p)
    untracked = sorted(l[3:] for l in status if l.startswith("?? ")
                       and "T009_delivery.json" not in l)
    scope = (set(modified) | set(untracked)) == T009_FILES
    abs_hits = []
    for rel in sorted(T009_FILES):
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            abs_hits.append({"file": rel, "line": 0, "text": "missing"})
            continue
        for lineno, line in enumerate(read(path).splitlines(), 1):
            if any(m in line for m in ABS_MARKERS):
                abs_hits.append({"file": rel, "line": lineno,
                                 "text": line.strip()})
    required = ["q4_results.json", "q4_fences.csv", "q4_overlap.json",
                "q4_frontier.json"]
    missing = [p for p in required if not os.path.exists(os.path.join(RUN, p))]
    csv_cols = {"fence_id", "x", "y", "zone", "n_assigned", "assigned_cells",
                "coverage_arrivals_day", "capacity_meas343",
                "standing_meas343", "buffer_meas343", "capacity_fleet495",
                "standing_fleet495", "buffer_fleet495"}
    outputs_ok = not missing and set(fences.columns) == csv_cols
    outputs_ok &= len(fences) == nsel
    dpath = os.path.join(DELIV, "T009_delivery.json")
    delivery_ok = os.path.exists(dpath)
    if delivery_ok:
        delivery = json.load(io.open(dpath, encoding="utf-8"))
        delivery_ok = (len(delivery["checks"]) == 10
                       and all(delivery["checks"].values()))
    check("10 readonly, scope, outputs, delivery, no abs paths",
          ro and scope and not abs_hits and outputs_ok and delivery_ok,
          f"modified {len(modified)}, untracked {len(untracked)}")

    passed = sum(results)
    print(f"\nvalidator: {passed}/{len(results)} passed")
    if delivery_ok:
        print("delivery checks all true")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
