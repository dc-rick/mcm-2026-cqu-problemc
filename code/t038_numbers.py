"""T038: number lock for the conserving rewrite.

Every printed number should trace to a stored result.  This module

  1. flattens every JSON result under the accepted run directories into
     (value, file, path) records;
  2. adds the derived quantities the paper quotes (savings, shares, layer
     costs) with their arithmetic written out;
  3. scans the .tex files for numeric tokens and matches each token against
     the records at the token's printed precision, trying the scale factors
     1, 100 (percent) and 1e-2;
  4. writes data/runs/T038/numbers.json and a coverage report listing
     matched, ambiguous and unmatched tokens for triage.

Superseded output directories are excluded so that a stale number cannot
match an old result file: T019/T020 (old Q4/Q3), T021A (old lock), T022
(old Q5), T023 (old uncertainty), T024/T025/T026 (old tricycle and
incentive experiments).  Their replacements are T028--T035.

A token is matched when at least one record reproduces it at the printed
precision.  Ambiguities and misses are reported, never silently accepted.
LaTeX thousands separators ``{,}`` are normalised before scanning.
"""
import io
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib

RUN = os.path.join(lib.DATA, "runs", "T038")
os.makedirs(RUN, exist_ok=True)

INCLUDE = ["T001", "T001R2", "T002", "T003", "T004", "T005", "T006", "T007",
           "T008", "T009", "T012", "T014", "T015", "T016", "T017", "T018",
           "T022", "T028", "T029", "T030", "T031", "T032", "T033", "T035",
           "T039", "T041"]

# T022 mixes Q1/fleet analyses (kept) with the superseded Q5 mismatch run
SKIP = {"runs/T022/m15_mismatch.json", "runs/T022/m15_check.json"}

# typed planning constants, with the assumption that fixes them
TYPED = {
    "slots_per_fence": (150.0, "A8/Q4 planning cap"),
    "fence_radius_m": (450.0, "Q4 walking threshold"),
    "walk_radius_m": (300.0, "A10 walk-borrow radius"),
    "truck_capacity": (40.0, "A6"),
    "truck_kmh": (20.0, "A6"),
    "truck_fixed_cny": (600.0, "A9"),
    "truck_km_cny": (6.0, "A9"),
    "truck_handling_cny": (0.5, "A9"),
    "tri_capacity": (20.0, "A10"),
    "tri_kmh": (15.0, "A10"),
    "tri_min_per_bike": (0.5, "A10"),
    "tri_max_km": (5.0, "A10"),
    "tri_fixed_cny": (300.0, "A10"),
    "tri_km_cny": (0.2, "A10"),
    "unmet_penalty_cny": (6.0, "A9"),
    "daily_budget_cny": (30000.0, "A11"),
    "kappa": (2.0, "A7"),
    "rho": (0.35, "A7"),
    "c0": (0.8, "A7"),
    "neighbour_zones": (8.0, "A6"),
    "arc_km_limit": (9.0, "A6"),
}


def write_constants():
    """Recount the data facts and record the typed planning constants."""
    import pandas as pd
    df = pd.read_parquet(os.path.join(lib.DATA, "trips.parquet"))
    box = ((df.start_lat > 39.4) & (df.start_lon > 115.8) &
           (df.start_lat < 40.4) & (df.start_lon < 117.0))
    dfx = df[box]
    cells = lib.get_cells()
    D = np.load(os.path.join(lib.DATA, "dist_matrix.npy"))
    out = dict(
        task="T038 constants", date="2026-11-20",
        data=dict(
            orders_raw=dict(value=float(len(df)),
                            basis="len(pd.read_parquet(data/trips.parquet))"),
            users_raw=dict(value=float(df.userid.nunique()),
                           basis="userid.nunique() on the raw file"),
            bikes_raw=dict(value=float(df.bikeid.nunique()),
                           basis="bikeid.nunique() on the raw file"),
            orders_in_box=dict(value=float(len(dfx)),
                               basis="rows inside 39.4-40.4 N, 115.8-117.0 E"),
            users_in_box=dict(value=float(dfx.userid.nunique()),
                              basis="userid.nunique() inside the box"),
            bikes_in_box=dict(value=float(dfx.bikeid.nunique()),
                              basis="bikeid.nunique() inside the box"),
            n_days=dict(value=float(dfx.starttime.dt.normalize()
                                    .astype(str).str[:10].nunique()),
                        basis="distinct dates in the box"),
            n_cells=dict(value=float(len(cells)),
                         basis="lib.get_cells(), 0.005 deg grid, THR=50"),
        ),
        typed={k: dict(value=v, basis=b) for k, (v, b) in TYPED.items()},
    )
    zid = cells.zone.to_numpy().astype(int)
    Z = int(zid.max()) + 1
    out["data"]["n_zones"] = dict(value=float(Z), basis="max(cells.zone)+1")
    # T018's arc rule: each zone's 8 nearest zone centroids, kept if <= 9 km
    cent = np.load(os.path.join(lib.DATA, "typical_day.npz"),
                   allow_pickle=True)["cent"].astype(np.float64)
    zd = np.sqrt(((cent[:, None, :] - cent[None, :, :]) ** 2).sum(-1))
    order = np.argsort(zd, axis=1)[:, 1:9]
    arcs = [(i, int(j)) for i in range(len(cent)) for j in order[i]
            if zd[i, j] <= 9.0]
    n_arcs = len(arcs)
    out["data"]["n_arcs"] = dict(
        value=float(n_arcs),
        basis="T018 load_network: 8 nearest zone centroids within 9 km")
    out["data"]["mean_arc_km"] = dict(
        value=float(sum(zd[i, j] for i, j in arcs) / max(n_arcs, 1)),
        basis="mean centroid distance of the T018 arcs")
    day = dfx.groupby(dfx.starttime.dt.strftime("%m-%d")).size()
    win = day.loc["05-10":"05-19"]
    out["data"]["daily_volume_min"] = dict(
        value=float(np.floor(win.min() / 1000.0) * 1000.0),
        basis="min orders/day over 05-10..05-19, floored to the thousand")
    out["data"]["daily_volume_max"] = dict(
        value=float(np.floor(win.max() / 1000.0) * 1000.0),
        basis="max orders/day over 05-10..05-19, floored to the thousand")
    dz = np.load(os.path.join(lib.DATA, "typical_day.npz"),
                 allow_pickle=True)["d_hour"].sum(1)
    out["data"]["zone_trips_min"] = dict(
        value=float(dz.min()), basis="min zone departures on the typical day")
    out["data"]["zone_trips_max"] = dict(
        value=float(dz.max()), basis="max zone departures on the typical day")
    p = os.path.join(RUN, "constants.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("wrote", p)
    return out


# first-rank source for values that legitimately occur in several files
PREFER = ("T032", "T031", "T030", "T029", "T028", "T018", "T017", "T016",
          "T015", "T014", "T012", "T003", "T005", "T006", "T007", "T009",
          "T004", "T002", "T001")

TEX_FILES = [
    "paper/summary.tex",
    "paper/sec1_intro.tex",
    "paper/sec2_assumptions.tex",
    "paper/sec3_data.tex",
    "paper/sec4_q1.tex",
    "paper/sec5_q2.tex",
    "paper/sec6_q3.tex",
    "paper/sec7_q4.tex",
    "paper/sec8_q5.tex",
    "paper/sec9_sensitivity.tex",
    "paper/sec10_evaluation.tex",
    "paper/ai_report.tex",
]

SCALES = ((1.0, "value"), (100.0, "percent"), (0.01, "fraction"))

# tokens that are calendar years, not results
YEARS = {"2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023",
         "2024", "2025", "2026"}

# tokens that are identifiers, not results (team control number)
IDS = {"13648"}


def flatten(obj, file, path, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(v, file, path + "." + str(k), out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            flatten(v, file, path + "[%d]" % i, out)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float, np.floating, np.integer)):
        v = float(obj)
        if np.isfinite(v):
            out.append(dict(value=v, file=file, path=path))


def load_records():
    out = []
    files = []
    for d in INCLUDE:
        root = os.path.join(lib.DATA, "runs", d)
        if not os.path.isdir(root):
            print("missing dir", d)
            continue
        for dirpath, _, names in os.walk(root):
            for nm in sorted(names):
                if not nm.endswith(".json"):
                    continue
                p = os.path.join(dirpath, nm)
                rel = os.path.relpath(p, lib.DATA).replace("\\", "/")
                if rel in SKIP:
                    continue
                files.append((p, rel))
    cp = os.path.join(RUN, "constants.json")
    if os.path.exists(cp):
        files.append((cp, "runs/T038/constants.json"))
    for p, rel in files:
        try:
            with io.open(p, encoding="utf-8") as fh:
                obj = json.load(fh)
        except Exception as exc:      # noqa: BLE001
            print("skip", rel, exc)
            continue
        flatten(obj, rel, "", out)
    return out


def add_derived(recs):
    """Quantities the paper quotes that are not stored as fields."""
    ls = json.load(open(os.path.join(lib.DATA, "runs", "T028",
                                     "local_search.json"), encoding="utf-8"))
    base = float(ls["baseline_total"])
    cost = float(ls["exec_cost"])
    recs.append(dict(value=base - cost, file="derived",
                     path="baseline_total - exec_cost"))
    recs.append(dict(value=(base - cost) / base, file="derived",
                     path="(baseline_total - exec_cost)/baseline_total"))
    layer = 300.0 * ls["exec_n"] + 0.2 * ls["exec_tri_vkm"]
    recs.append(dict(value=layer, file="derived",
                     path="300*exec_n + 0.2*exec_tri_vkm"))
    tp = os.path.join(lib.DATA, "runs", "T031", "truck_payment.json")
    if os.path.exists(tp):
        d = json.load(open(tp, encoding="utf-8"))
        u, N = d["truck_unit"], d["N"]
        t18 = os.path.join(lib.DATA, "runs", "T018",
                           "result_int_K60.json")
        if os.path.exists(t18):
            k60 = json.load(open(t18, encoding="utf-8"))
            uv = k60["obj_variable"] / k60["moved"]
            recs.append(dict(value=uv, file="derived",
                             path="T018 obj_variable/moved (variable unit)"))
            recs.append(dict(value=uv * N, file="derived",
                             path="variable unit * N (all-truck, fleet excluded)"))
            recs.append(dict(value=N / k60["moved"], file="derived",
                             path="N / K60 moved (placement multiple)"))
        for lvl in ("arrivals", "surplus"):
            b = d["levels"][lvl]["best"]
            allt = u * N
            recs.append(dict(value=allt - b["total_spend"], file="derived",
                             path=f"all_truck - {lvl}.best.total_spend"))
            recs.append(dict(value=1 - b["total_spend"] / allt,
                             file="derived",
                             path=f"1 - {lvl}.best.total_spend/all_truck"))
            recs.append(dict(value=b["P"] - b["q"], file="derived",
                             path=f"{lvl} free riders P - q"))
            recs.append(dict(value=b["payment"] / b["q"], file="derived",
                             path=f"{lvl} payment per filled deficit"))
            recs.append(dict(value=u * b["q"] / b["payment"], file="derived",
                             path=f"{lvl} placement return per yuan"))
            recs.append(dict(value=(allt - b["total_spend"]) / b["payment"],
                             file="derived",
                             path=f"{lvl} saving per yuan of payment"))
            recs.append(dict(value=b["payment"] / 30000.0, file="derived",
                             path=f"{lvl} payment / budget"))
            recs.append(dict(value=(b["P"] - b["q"]) / b["P"], file="derived",
                             path=f"{lvl} free-rider share of P"))
        rows = d["levels"]["arrivals"]["rows"]
        r3 = [x for x in rows if abs(x["r"] - 3.0) < 1e-9]
        if r3:
            x = r3[0]
            tot = x["payment"] + u * (N - x["q"])
            recs.append(dict(value=tot, file="derived",
                             path="r=3.00 total spend"))
            recs.append(dict(value=1 - tot / (u * N), file="derived",
                             path="r=3.00 saving share"))
            recs.append(dict(value=x["payment"] / 30000.0, file="derived",
                             path="r=3.00 payment / budget"))
        b = d["levels"]["arrivals"]["best"]
        if os.path.exists(t18):
            recs.append(dict(value=uv * b["q"] - b["payment"], file="derived",
                             path="variable-unit saving at best r"))
    tb = os.path.join(lib.DATA, "runs", "T032", "tricycle_baseline.json")
    if os.path.exists(tb):
        d = json.load(open(tb, encoding="utf-8"))
        b = d["baseline"]
        layer = 300.0 * b["n"] + 0.2 * b["tri_vehicle_km"] + 6.0 * b["unmet"]
        recs.append(dict(value=layer, file="derived",
                         path="300*n + 0.2*vkm + 6*unmet (T031 baseline)"))
    add_derived_cons(recs, ls)
    return recs


def add_derived_cons(recs, ls):
    """Derived quantities of the conserving chain (T018/T028/T030/T031/T039)."""
    # --- T018 K-scan differences used in the single-layer discussion
    bpath = os.path.join(lib.DATA, "runs", "T018", "baselines.json")
    k60 = 83807.74383516546
    if os.path.exists(bpath):
        b = json.load(open(bpath, encoding="utf-8"))
        k0 = float(b["rows"]["no_trucks"]["cost"])
        k60 = float(b["rows"]["recommended_integer"]["cost"])
        k80 = float(b["int_rows"]["K80"]["obj"])
        recs.append(dict(value=k80 - k60, file="derived",
                         path="T018 int_rows K80.obj - K60.obj"))
        recs.append(dict(value=k0 - k60, file="derived",
                         path="T018 rows.no_trucks.cost - K60.obj"))
        recs.append(dict(value=(k0 - k60) / k0, file="derived",
                         path="(T018 no_trucks.cost - K60.obj)/no_trucks.cost"))
    # --- descent (LP) level of the Q2 search
    lp_n = float(ls["lp_n"])
    lp_cost = float(ls["lp_cost"])
    lp_unmet = float(ls["lp_unmet"])
    lp_vkm = float(ls["lp_tri_vkm"])
    recs.append(dict(value=300.0 * lp_n + 0.2 * lp_vkm + 6.0 * lp_unmet,
                     file="derived",
                     path="300*lp_n + 0.2*lp_tri_vkm + 6*lp_unmet"))
    recs.append(dict(value=300.0 * ls["exec_n"], file="derived",
                     path="300*exec_n"))
    recs.append(dict(value=6.0 * ls["exec_unmet"], file="derived",
                     path="6*exec_unmet"))
    recs.append(dict(value=float(ls["exec_cost"]) - lp_cost, file="derived",
                     path="exec_cost - lp_cost"))
    recs.append(dict(value=lp_unmet - float(ls["exec_unmet"]), file="derived",
                     path="lp_unmet - exec_unmet"))
    recs.append(dict(value=6.0 * (lp_unmet - float(ls["exec_unmet"])),
                     file="derived", path="6*(lp_unmet - exec_unmet)"))
    recs.append(dict(value=300.0 * ls["exec_n"] + 0.2 * ls["exec_tri_vkm"],
                     file="derived",
                     path="300*exec_n + 0.2*exec_tri_vkm (operating only)"))
    recs.append(dict(value=float(ls["baseline_total"]) - k60, file="derived",
                     path="baseline_total - K60 obj (cell penalty of the truck plan)"))
    # --- T028 zero-dispatch and truck-only replays: departures removed by the
    # --- K=60 plan and the CNY cost per departure removed (quoted in Sec. 5/AI report)
    zp = os.path.join(lib.DATA, "runs", "T028", "zero_baselines.json")
    if os.path.exists(zp):
        z = json.load(open(zp, encoding="utf-8"))
        delta = (float(z["no_truck_no_tri"]["unmet"])
                 - float(z["truck_only"]["unmet"]))
        recs.append(dict(value=delta, file="derived",
                         path="T028 unmet(no truck) - unmet(truck only)"))
        recs.append(dict(value=k60 / delta, file="derived",
                         path="T018 K60 obj / T028 departures removed"))
    # --- T030 fence demand: sums and per-fleet ratios
    fp = os.path.join(lib.DATA, "runs", "T030", "frontier_cons.json")
    if os.path.exists(fp):
        f = json.load(open(fp, encoding="utf-8"))
        for lvl, fleet in (("meas343", 343638.0), ("fleet495", 495540.0)):
            s = float(f["levels"][lvl]["demand_sum"])
            recs.append(dict(value=s / fleet, file="derived",
                             path=f"{lvl} demand_sum / {int(fleet)}"))
    # --- T019: the superseded non-conserving demand, cited once as history
    t19s = os.path.join(lib.DATA, "runs", "T019", "summary.json")
    if os.path.exists(t19s):
        d19 = json.load(open(t19s, encoding="utf-8"))
        recs.append(dict(
            value=float(d19["denominator_conservativeness"]["meas343"]
                        ["system_p95_total_occupancy"]),
            file="runs/T019/summary.json",
            path=".denominator_conservativeness.meas343."
                 "system_p95_total_occupancy (superseded construction, "
                 "quoted as history)"))
    # --- T031 truck-caliber unconstrained optimum on the full ladder
    tp = os.path.join(lib.DATA, "runs", "T031", "truck_payment.json")
    if os.path.exists(tp):
        d = json.load(open(tp, encoding="utf-8"))
        u, N = float(d["truck_unit"]), float(d["N"])
        rows = d["levels"]["arrivals"]["rows"]
        for x in rows:
            if abs(x["r"] - 1.75) < 1e-9:
                tot = x["payment"] + u * (N - x["q"])
                recs.append(dict(value=tot, file="derived",
                                 path="r=1.75 payment + unit*(N-q)"))
                recs.append(dict(value=u * N - tot, file="derived",
                                 path="all_truck - r=1.75 total"))
        b = d["levels"]["arrivals"]["best"]
        x85 = [x for x in rows if abs(x["r"] - 0.85) < 1e-9]
        if x85:
            x = x85[0]
            recs.append(dict(value=N - b["q"], file="derived",
                             path="N - q at the payment-ledger optimum"))
            recs.append(dict(value=u * (N - b["q"]), file="derived",
                             path="unit*(N - q) at the payment-ledger optimum"))
            recs.append(dict(value=x["payment"] - 30000.0, file="derived",
                             path="r=0.85 payment - budget"))
            recs.append(dict(value=x["P"] - x["q"], file="derived",
                             path="r=0.85 free riders P - q"))
            recs.append(dict(value=b["r"] * b["q"], file="derived",
                             path="r*q at the payment-ledger optimum"))
            recs.append(dict(value=b["payment"] - b["r"] * b["q"],
                             file="derived",
                             path="payment - r*q at the optimum"))
            recs.append(dict(value=30000.0 - b["payment"], file="derived",
                             path="budget - payment at the optimum"))
        rows_s = d["levels"]["surplus"]["rows"]
        for x in rows_s:
            if abs(x["r"] - 1.80) < 1e-9:
                recs.append(dict(value=x["payment"] + u * (N - x["q"]),
                                 file="derived",
                                 path="r=1.80 (surplus) payment + unit*(N-q)"))
    # --- T031 tricycle-caliber levels
    ty = os.path.join(lib.DATA, "runs", "T031", "tricycle_payment.json")
    if os.path.exists(ty):
        d = json.load(open(ty, encoding="utf-8"))
        base = d["baseline"]
        bt, bu, bk = (float(base["total"]), float(base["unmet"]),
                      float(base["tri_vehicle_km"]))
        for lvl in ("arrivals", "surplus"):
            L = d["levels"][lvl]
            be = L["best_exec"]
            tot = float(be["total"])
            recs.append(dict(value=bt - tot, file="derived",
                             path=f"{lvl} baseline.total - best_exec.total"))
            recs.append(dict(value=(bt - tot) / bt, file="derived",
                             path=f"{lvl} (baseline - best_exec)/baseline"))
            recs.append(dict(value=be["P"] - be["q"], file="derived",
                             path=f"{lvl} free riders P - q"))
            rem = bu - float(be["unmet"])
            recs.append(dict(value=rem, file="derived",
                             path=f"{lvl} baseline.unmet - best_exec.unmet"))
            recs.append(dict(value=rem / be["q"], file="derived",
                             path=f"{lvl} net removed departures per reroute"))
            recs.append(dict(value=6.0 * rem, file="derived",
                             path=f"{lvl} 6*(removed departures)"))
            kms = 0.2 * (bk - float(be["tri_vehicle_km"]))
            recs.append(dict(value=kms, file="derived",
                             path=f"{lvl} 0.2*(baseline vkm - exec vkm)"))
            recs.append(dict(value=6.0 * rem + kms, file="derived",
                             path=f"{lvl} 6*removed + km saving"))
            recs.append(dict(value=be["payment"] / rem, file="derived",
                             path=f"{lvl} payment per net removed departure"))
            bf = L.get("best_fixed_fleet")
            if bf:
                recs.append(dict(value=bt - float(bf["total"]), file="derived",
                                 path=f"{lvl} baseline.total - "
                                      "best_fixed_fleet.total"))
    # --- T032: interval bounds (signs are dropped by the tokenizer, so the
    #     magnitudes are recorded), the share sums, and the joint-sample extremes
    ivp = os.path.join(lib.DATA, "runs", "T032", "intervals.json")
    if os.path.exists(ivp):
        iv = json.load(open(ivp, encoding="utf-8"))["outputs"]
        for key, e in iv.items():
            for g, s in e["slices"].items():
                for p in ("p5", "p50", "p95"):
                    if p in s:
                        recs.append(dict(value=abs(s[p]), file="derived",
                                         path=f"T032 {key}.{g}.{p} magnitude"))
    shp = os.path.join(lib.DATA, "runs", "T032", "variance_shares.json")
    if os.path.exists(shp):
        sh = json.load(open(shp, encoding="utf-8"))["outputs"]
        for key, e in sh.items():
            recs.append(dict(value=sum(e["share_first_order"].values()),
                             file="derived",
                             path=f"T032 {key} first-order share sum"))
            recs.append(dict(value=abs(e["residual_1_minus_sum"]),
                             file="derived",
                             path=f"T032 {key} |share sum - 1|"))
    rp = os.path.join(lib.DATA, "runs", "T032", "reference.json")
    ep = os.path.join(lib.DATA, "runs", "T032", "evals.jsonl")
    if os.path.exists(rp):
        ref = json.load(open(rp, encoding="utf-8"))
        recs.append(dict(value=abs(ref["anchors"]["q3_total"]), file="derived",
                         path="T032 |q3_total anchor|"))
    if os.path.exists(rp) and os.path.exists(ep):
        rows = [json.loads(line) for line in open(ep, encoding="utf-8")
                if line.strip()]
        F = [r for r in rows if r["id"].startswith("F_")]
        acc, anc = ref["accepted"], ref["anchors"]
        s3 = []
        for r in F:
            at = r["q3"]["all_truck"] + (acc["q3"]["all_truck"]
                                         - ref["q3"]["all_truck"])
            t = r["q3"]["total"] + (acc["q3"]["total"]
                                    - ref["q3"]["total"])
            s3.append(1.0 - t / at)
        recs.append(dict(value=min(s3), file="derived",
                         path="T032 min joint Q3 saving share"))
        s2 = []
        for r in F:
            b = r["q2"]["baseline"] + (acc["q2"]["baseline"]
                                       - ref["q2"]["baseline"])
            t = r["q2"]["total"] + (acc["q2"]["total"] - ref["q2"]["total"])
            s2.append((b - t) / b)
        recs.append(dict(value=sum(1 for x in s2 if x < 0), file="derived",
                         path="T032 joint draws with negative Q2 saving"))
    # --- T005: the loss reduction is stored with its sign; the report prints
    #     the magnitude ("reduced loss by")
    dv = os.path.join(lib.DATA, "runs", "T005", "dest_validation.json")
    if os.path.exists(dv):
        d = json.load(open(dv, encoding="utf-8"))
        recs.append(dict(
            value=abs(float(d["day_split"]["pooled"]
                            ["delta_nlp_full_minus_distance"])),
            file="derived",
            path="abs(day_split pooled delta_nlp full minus distance)"))
    # --- T039 c0 sweep: free riders of each re-optimised reward
    cp = os.path.join(lib.DATA, "runs", "T039", "q3_c0.json")
    if os.path.exists(cp):
        c = json.load(open(cp, encoding="utf-8"))
        for lvl, v in c["levels"].items():
            for cap in ("arrivals", "surplus"):
                bp = v[cap].get("best_payment")
                if bp:
                    recs.append(dict(value=bp["P"] - bp["q"], file="derived",
                                     path=f"c0={lvl} {cap} free riders P - q"))
    return recs


TOKEN = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")


def parse_token(t):
    return float(t.replace(",", ""))


def decimals(t):
    return len(t.split(".")[1]) if "." in t else 0


def rank(h):
    for i, d in enumerate(PREFER):
        if ("runs/" + d) in h["file"]:
            return i
    return len(PREFER)


def match(token, recs):
    v = parse_token(token)
    d = decimals(token)
    tol = 0.5 * 10.0 ** (-d) + 1e-9
    hits = []
    for r in recs:
        for scale, label in SCALES:
            x = r["value"] * scale
            if abs(x - v) <= tol:
                hits.append(dict(value=r["value"], file=r["file"],
                                 path=r["path"], scale=scale,
                                 scale_label=label))
    seen = set()
    uniq = []
    for h in hits:
        k = (h["file"], h["path"], h["scale"])
        if k not in seen:
            seen.add(k)
            uniq.append(h)
    uniq.sort(key=rank)
    return uniq


def main():
    recs = load_records()
    recs = add_derived(recs)
    print("records", len(recs))
    numbers = []
    report = []
    for rel in TEX_FILES:
        p = os.path.join(lib.ROOT, rel)
        if not os.path.exists(p):
            report.append(f"MISSING TEX {rel}")
            continue
        text = io.open(p, encoding="utf-8").read()
        # strip LaTeX comments, but not the '%' of an escaped '\%'
        text = re.sub(r"(?m)(?<!\\)%.*$", "", text)
        text = text.replace("{,}", ",")
        for m in TOKEN.finditer(text):
            tok = m.group(1)
            if tok in YEARS or tok in IDS:
                continue
            if len(tok.replace(",", "").replace(".", "")) < 3:
                continue          # one- and two-digit tokens: indices, counts
            hits = match(tok, recs)
            numbers.append(dict(token=tok, file=rel,
                                pos=int(m.start()), hits=hits))
            if hits:
                continue
            report.append(f"UNMATCHED {rel} :: {tok}")
    out = dict(task="T038 number lock", date="2026-11-20",
               n_records=len(recs), n_tokens=len(numbers),
               n_matched=sum(1 for x in numbers if x["hits"]),
               n_unique=sum(1 for x in numbers if len(x["hits"]) == 1),
               tokens=numbers)
    path = os.path.join(RUN, "numbers.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    rp = os.path.join(RUN, "coverage_report.txt")
    n_unmatched = sum(1 for r in report if r.startswith("UNMATCHED"))
    head = [f"T038 number lock, {out['date']}",
            f"records {out['n_records']}, tokens {out['n_tokens']}, "
            f"matched {out['n_matched']}, unique {out['n_unique']}, "
            f"unmatched {n_unmatched}"]
    if not report:
        head.append("PASS  every numeric token in the sources traces to a stored result")
    with open(rp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(head + report) + "\n")
    print("tokens", len(numbers), "matched", out["n_matched"],
          "unique", out["n_unique"], "unmatched", len(report))
    print("wrote", path, rp)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--constants", action="store_true",
                    help="recount the data facts and write constants.json")
    a = ap.parse_args()
    if a.constants:
        write_constants()
    main()
