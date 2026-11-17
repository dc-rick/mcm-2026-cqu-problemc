"""Independent validator for the T007 delivery.

Recomputes conservation, night shares and the greedy baseline from the saved
plans; it does not import t007_delivery.py or t007_q2_run.py.
"""
import hashlib
import io
import json
import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, "code")
DATA = os.path.join(ROOT, "data")
PAPER = os.path.join(ROOT, "paper")
RUN = os.path.join(DATA, "runs", "T007")
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


def load_demand(kind):
    if kind == "q1":
        z = np.load(os.path.join(DATA, "runs", "T006", "q1_od.npz"))
        od = z["od_q1"].astype(np.float64) * SCALE
    else:
        z = np.load(os.path.join(DATA, "typical_day.npz"), allow_pickle=True)
        od = z["od_z"].astype(np.float64) * SCALE
    dep = od.sum(2).T
    row = od.sum(2)
    PZ = np.divide(od, row[:, :, None], out=np.zeros_like(od),
                   where=row[:, :, None] > 0)
    return dep, PZ


def cons_recompute(kind, plan, alpha):
    dep, PZ = load_demand(kind)
    I = plan["I"].astype(np.float64)
    U = plan["U"].astype(np.float64)
    XM = plan["XM"].astype(np.float64)
    served = dep - U
    Z, T = dep.shape
    arr = np.stack([PZ[t].T @ served[:, t] for t in range(T)], axis=1)
    city, zone = [], []
    for t in range(T - 1):
        city.append(abs((I[:, t + 1].sum() - I[:, t].sum())
                        - (arr[:, t].sum() - served[:, t].sum())))
        zone.append(np.abs(I[:, t + 1] - I[:, t] - arr[:, t] + served[:, t]
                          + XM[t].sum(1) - XM[t].sum(0)).max())
    st = np.load(os.path.join(DATA, "q2_state.npz"), allow_pickle=True)
    I0 = st["I0"] * alpha
    day_end = abs(I[:, -1].sum() + arr[:, -1].sum() - served[:, -1].sum()
                  - I0.sum())
    return max(city), max(zone), day_end, float(np.max(U - dep))


def greedy_recompute(K=60, Q=40):
    dep, _ = load_demand("q1")
    od = np.load(os.path.join(DATA, "runs", "T006", "q1_od.npz"))["od_q1"].astype(float) * SCALE
    Z = dep.shape[0]
    plan = np.zeros((24, Z, Z))
    trucks = np.full(Z, K / Z, dtype=float)
    start = np.zeros((24, Z))
    for t in range(24):
        start[t] = trucks
        need = od[t].sum(1) - od[t].sum(0)
        surplus = np.argsort(need)
        deficit = np.argsort(need)[::-1]
        avail = Q * trucks
        for s in surplus[:15]:
            for d in deficit[:15]:
                if need[s] >= 0 or need[d] <= 0:
                    continue
                amt = min(-need[s], need[d], avail[s], float(Q))
                if amt <= 1:
                    continue
                plan[t, s, d] += amt
                avail[s] -= amt
                trucks[s] -= amt / Q
                trucks[d] += amt / Q
                need[s] += amt
                need[d] -= amt
    ratio = np.divide(plan.sum(2), start, out=np.zeros_like(start),
                      where=start > 0)
    return plan.sum((1, 2)).max(), ratio.max(), start.sum(1)


def main():
    results = []
    rj = json.load(io.open(os.path.join(RUN, "q2_results_q1.json"), encoding="utf-8"))
    cj = json.load(io.open(os.path.join(RUN, "q2_conservation.json"), encoding="utf-8"))
    nj = json.load(io.open(os.path.join(RUN, "q2_night.json"), encoding="utf-8"))
    sj = json.load(io.open(os.path.join(RUN, "q2_sim_q1.json"), encoding="utf-8"))
    gj = json.load(io.open(os.path.join(RUN, "q2_greedy_check.json"), encoding="utf-8"))

    def check(name, ok, detail=""):
        results.append(ok)
        print(("PASS " if ok else "FAIL ") + name + (("  " + detail) if detail else ""))

    scen = set(rj["scenarios"])
    struct = scen == EXPECTED_SCEN
    for n in scen | {"obs_a1_K60"}:
        p = np.load(os.path.join(RUN, "plans", n + ".npz"))
        struct &= set(p.files) == {"XM", "X", "M", "I", "U", "MK"}
        struct &= p["XM"].shape == (24, 50, 50) and p["MK"].shape == (24, 50, 50)
        struct &= p["I"].shape == (50, 24) and p["U"].shape == (50, 24)
    mp = np.load(os.path.join(RUN, "q2_plan_q1.npz"))
    struct &= {"XM", "X", "M", "I", "U", "MK"} <= set(mp.files)
    check("1 scenario set and plan structure", struct, f"{len(scen)} q1 scenarios")

    full = True
    for n, rec in rj["scenarios"].items():
        p = np.load(os.path.join(RUN, "plans", n + ".npz"))
        full &= abs(float(p["XM"].sum()) - rec["moved"]) <= 1e-6 * rec["moved"]
        full &= abs(rec["trips"] - 2143417.4) / 2143417.4 < 1e-6
        full &= abs(float(p["I"][:, 0].sum()) - 343638.0 * rec["alpha"]) \
            <= 1e-6 * 343638.0 * rec["alpha"]
    full &= abs(float(mp["XM"].sum()) - rj["scenarios"]["q1_a1_K60"]["moved"]) <= 1e-6
    check("2 plans are full scale", full, "trips 2,143,417")

    worst = 0.0
    all_ok = True
    for n in scen | {"obs_a1_K60"}:
        rec = rj["scenarios"].get(n, rj["obs_baseline"])
        p = np.load(os.path.join(RUN, "plans", n + ".npz"))
        city, zone, de, uover = cons_recompute(rec["od"], p, rec["alpha"])
        worst = max(worst, city, zone, de, uover)
        all_ok &= max(city, zone, de, uover) <= TOL
    check("3 conservation residuals <= 1e-6", all_ok, f"worst {worst:.2e}")

    t6 = json.load(io.open(T006_SMOKE, encoding="utf-8"))
    repro = True
    for kind, key in (("q1", "q1"), ("obs", "obs")):
        rec = rj["scenarios"]["q1_a1_K60"] if kind == "q1" else rj["obs_baseline"]
        repro &= all(abs(rec[k] - t6[key][k]) <= 1e-6 * max(1.0, abs(t6[key][k]))
                     for k in ("moved", "truck_km", "unmet", "dev", "obj"))
    check("4 u=0 rows reproduce T006 smoke", repro,
          "q1 24,679.28 / obs 23,959.15")

    fb = True
    for n in scen | {"obs_a1_K60"}:
        rec = rj["scenarios"].get(n, rj["obs_baseline"])
        if rec["unmet"] > 1e-6:
            fb &= abs(rec["dev"] - rec["unmet"]) > 1e-6 * max(1.0, rec["unmet"])
    check("5 u<=dep and dev != unmet", all_ok and fb)

    cover = (scen == EXPECTED_SCEN
             and {rj["scenarios"][n]["K"] for n in
                  ("q1_a1_K20", "q1_a1_K40", "q1_a1_K60", "q1_a1_K80")}
             == {20, 40, 60, 80}
             and rj["scenarios"]["q1_a0.8_K80"]["alpha"] == 0.8
             and rj["scenarios"]["q1_a0.6_K80"]["alpha"] == 0.6
             and "not run" in rj["unrerun"])
    check("6 coverage and unrerun note", cover)

    d = rj["q1_vs_obs"]["q1_minus_obs"]
    cmp_ok = all(abs(d[k] - (rj["scenarios"]["q1_a1_K60"][k]
                             - rj["obs_baseline"][k]))
                 <= 1e-9 * max(1.0, abs(d[k]))
                 for k in ("moved", "truck_km", "unmet", "dev", "obj"))
    check("7 Q1-minus-obs comparison", cmp_ok)

    gmax, gratio, tsum = greedy_recompute()
    greedy_ok = (abs(gmax - gj["max_hourly_moved"]) <= 1e-9
                 and abs(gratio - gj["max_source_bikes_per_truck"]) <= 1e-9
                 and gmax <= 2400 + 1e-9 and gratio <= 40 + 1e-9
                 and gj["hourly_violations"] == 0 and gj["source_violations"] == 0
                 and abs(tsum.min() - 60) <= 1e-9 and abs(tsum.max() - 60) <= 1e-9)
    pol = {(r["alpha"], r["policy"]) for r in sj["rows"]}
    greedy_ok &= {(1.0, "no trucks"), (1.0, "LP plan"), (1.0, "greedy")} <= pol
    greedy_ok &= len(sj["rows"]) == 7
    check("8 greedy <=2,400/h and <=40x trucks, 3 policies", greedy_ok,
          f"max hour {gmax:.2f}, ratio {gratio:.2f}")

    cent = np.load(os.path.join(DATA, "typical_day.npz"), allow_pickle=True)["cent"]
    dist = np.sqrt(((cent[:, None, :] - cent[None, :, :]) ** 2).sum(-1))
    mh = mp["XM"].sum((1, 2))
    kh = np.einsum("tij,ij->t", mp["MK"], dist)
    nh = [0, 1, 2, 3, 4, 5, 21, 22, 23]
    night_ok = (nh == nj["night_hours"]
                and abs(mh[nh].sum() / mh.sum() - nj["moved_night_share"]) <= 1e-9
                and abs(kh[nh].sum() / kh.sum() - nj["truck_km_night_share"]) <= 1e-9)
    check("9 night shares recomputed", night_ok,
          f"{100*mh[nh].sum()/mh.sum():.2f}% moved, "
          f"{100*kh[nh].sum()/kh.sum():.2f}% truck-km")

    fp15 = json.load(io.open(T015_FP, encoding="utf-8"))["trees"]
    fp1 = json.load(io.open(T001_FP, encoding="utf-8"))["trees"]["review"]
    ro = True
    for cur, base in ((hashes(DATA, skip_runs=True), fp15["data"]),
                      (hashes(PAPER), fp15["paper"]),
                      (hashes(os.path.join(ROOT, "review")), fp1)):
        b = {r["path"]: r["sha256"] for r in base}
        ro &= not [p for p in b if p in cur and cur[p] != b[p]]
        ro &= not [p for p in b if p not in cur]
    status = subprocess.run(["git", "-c", "core.quotepath=false", "status",
                             "--porcelain"], cwd=ROOT, capture_output=True,
                            text=True, encoding="utf-8", errors="replace").stdout.splitlines()
    root_paths = [r["path"] for r in fp15["root"]]
    ro &= not [l for l in status if l[3:] in root_paths]
    diff = subprocess.run(["git", "diff", "--name-only", BASE_COMMIT, "--", "code"],
                          cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout.splitlines()
    modified = sorted(p for p in diff if p)
    untracked = sorted(l[3:] for l in status if l.startswith("?? ")
                       and "T007_delivery.json" not in l)
    scope = (set(modified) | set(untracked)) == T007_FILES
    abs_hits = []
    for rel in sorted(T007_FILES):
        if not os.path.exists(os.path.join(ROOT, rel)):
            abs_hits.append(rel)
            continue
        for i, line in enumerate(read(os.path.join(ROOT, rel)).splitlines(), 1):
            if any(m in line for m in ABS_MARKERS):
                abs_hits.append((rel, i))
    outputs = ["q2_results_q1.json", "q2_plan_q1.npz", "q2_sim_q1.json",
               "q2_conservation.json", "q2_night.json", "q2_greedy_check.json"]
    missing = [p for p in outputs if not os.path.exists(os.path.join(RUN, p))]
    check("10 readonly, scope, outputs, no abs paths",
          ro and scope and not abs_hits and not missing,
          f"modified {len(modified)}, untracked {len(untracked)}")

    with io.open(os.path.join(ROOT, "交付", "T007_delivery.json"),
                 encoding="utf-8") as fh:
        delivery = json.load(fh)
    passed = sum(results)
    print(f"\nvalidator: {passed}/{len(results)} passed; "
          f"delivery checks all true: {all(delivery['checks'].values())}")
    return 0 if passed == len(results) and all(delivery["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
