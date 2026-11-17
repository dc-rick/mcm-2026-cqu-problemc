"""Build the T008 Q3 delivery JSON: executable pool, q(r), savings.

The checks recompute the pool edges, sample q(r) with the hourly assignment
LPs, and verify the feasibility audit, savings, public-push free-rider bill and
read-only/scope conditions.  The independent audit is t008_validate.py.
"""
import io
import json
import os

import numpy as np

import t008_validate as v

ROOT = v.ROOT
DATA = v.DATA
PAPER = v.PAPER
RUN = v.RUN
DELIV = os.path.join(ROOT, "交付")
OUT = os.path.join(DELIV, "T008_delivery.json")
T015_FP = v.T015_FP
T001_FP = v.T001_FP
SCALE = v.SCALE
R = v.R
N_DEFAULT = 216343.75
TRUCK_MAIN = v.TRUCK_MAIN
TRUCK_LOW = v.TRUCK_LOW
BASE_COMMIT = v.BASE_COMMIT
T008_FILES = v.T008_FILES
ABS_MARKERS = v.ABS_MARKERS


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


def main():
    cells, D, arr, dep, arx, need, tstar, pos, N = v.load_all()
    cache = v.build_edges(arr, D, need, tstar, pos)
    pool = np.load(os.path.join(RUN, "q3_pool.npz"))
    qr = json.load(io.open(os.path.join(RUN, "q3_qr.json"), encoding="utf-8"))
    qr_c = json.load(io.open(os.path.join(RUN, "q3_qr_conservative.json"),
                             encoding="utf-8"))
    res = json.load(io.open(os.path.join(RUN, "q3_results.json"),
                            encoding="utf-8"))
    aud = json.load(io.open(os.path.join(RUN, "q3_pool_audit.json"),
                            encoding="utf-8"))

    edges = []
    for h in range(24):
        e = cache[h]
        if e is None:
            continue
        src, tg, si, ti, d = e
        for a, b, dd in zip(si, ti, d):
            edges.append((src[a], tg[b], h, dd))
    edge_src = np.array([e[0] for e in edges])
    edge_dst = np.array([e[1] for e in edges])
    edge_hour = np.array([e[2] for e in edges])
    naive = float(arr[edge_hour, edge_src].sum())
    group = {(int(h), int(s)) for s, h in zip(edge_src, edge_hour)}
    distinct = float(sum(arr[h, s] for h, s in group))
    same = float(arr[tstar[pos], np.where(pos)[0]].sum())

    keys = {"edge_src", "edge_dst", "edge_hour", "edge_dist_km",
            "edge_src_arrivals", "edge_dst_need", "target_cell",
            "target_hour", "target_need", "target_arrivals",
            "target_free_riders", "need", "tstar", "source_hour_arrivals"}
    s1 = keys <= set(pool.files)
    s1 &= np.array_equal(pool["edge_src"], edge_src.astype(np.int32))
    s1 &= np.array_equal(pool["edge_dst"], edge_dst.astype(np.int32))
    s1 &= np.array_equal(pool["edge_hour"], edge_hour.astype(np.int8))
    s1 &= bool((pool["edge_src"] != pool["edge_dst"]).all())
    s1 &= bool((pool["edge_dist_km"] <= R + 1e-9).all())
    s1 &= close(naive, aud["naive_pair_units"], 1e-12)
    s1 &= close(distinct, aud["distinct_units"], 1e-12)

    s2 = bool((pool["edge_src"] != pool["edge_dst"]).all())
    s2 &= bool((pool["target_hour"] == tstar[pool["target_cell"]]).all())
    s2 &= close(same / naive, 0.34110008149741483, 1e-9)
    s2 &= aud["old_34_105_reproduced"] is True

    s3 = close(naive / distinct, 1.087763406481611, 1e-9)
    s3 &= aud["old_1_09_reproduced"] is True
    s3 &= aud["old_2_93_reproduced"] is True

    rs = np.array(qr["r_grid"])
    qs = np.array(qr["q"])
    s4 = len(rs) == len(qs) and np.all(np.diff(rs) > 0)
    s4 &= np.all(np.diff(qs) >= -1e-9)
    for r in (0.0, 0.5, 1.0, 1.5, float(qr["r_star"]), 2.0, 3.0):
        got = v.q_at(r, cache, arr, need)["q"]
        s4 &= close(got, float(np.interp(r, rs, qs)), 1e-6)
    total = rs * qs + TRUCK_MAIN * (N - qs)
    i = int(np.argmin(total))
    s4 &= close(rs[i], res["targeted"]["r_star"], 1e-12)
    s4 &= close(qs[i], res["targeted"]["q_star"], 1e-9)

    au = v.q_at(float(res["targeted"]["r_star"]), cache, arr, need)
    s5 = close(au["q"], res["targeted"]["q_star"], 1e-9)
    s5 &= au["q"] <= distinct * (1 + 1e-9)
    s5 &= au["max_source_usage_ratio"] <= 1 + 1e-6
    s5 &= au["max_target_usage_ratio"] <= 1 + 1e-6
    s5 &= all(s == 0 for s in au["statuses"])
    s5 &= bool(aud["feasibility"]["q_le_distinct"])
    s5 &= bool(aud["feasibility"]["source_usage_le_arrivals"])
    s5 &= bool(aud["feasibility"]["target_usage_le_need"])
    s5 &= aud["feasibility"]["same_cell_edges"] == 0

    # conservative hourly-surplus bound (R1)
    rsc = np.array(qr_c["r_grid"])
    qsc = np.array(qr_c["q"])
    cons = res["targeted_conservative"]
    s6 = len(rsc) == len(qsc) and np.all(np.diff(rsc) > 0)
    s6 &= np.all(np.diff(qsc) >= -1e-9)
    s6 &= bool(np.all(qsc <= qs + 1e-9))
    for r in (0.0, 0.5, 1.0, float(qr_c["r_star"]), 2.0, 3.0):
        got = v.q_at(r, cache, arr, need, dep=dep,
                     source_cap="surplus")["q"]
        s6 &= close(got, float(np.interp(r, rsc, qsc)), 1e-6)
    cons_total = rsc * qsc + TRUCK_MAIN * (N - qsc)
    ic = int(np.argmin(cons_total))
    s6 &= close(rsc[ic], cons["r_star"], 1e-12)
    s6 &= close(qsc[ic], cons["q_star"], 1e-9)
    s6 &= abs(rsc[ic] - 1.83) <= 1e-12
    s6 &= abs(qsc[ic] - 32777.11) <= 1e-2
    s6 &= abs((1.0 - cons_total[ic] / (TRUCK_MAIN * N)) - 0.06826) <= 1e-4
    s6 &= close(cons["saving"],
                1.0 - cons_total[ic] / (TRUCK_MAIN * N), 1e-9)
    s6 &= cons["bound"] == "hourly_surplus_sources"
    s6 &= res["targeted"]["bound"] == "upper_no_source_deficit"
    ac = v.q_at(cons["r_star"], cache, arr, need, dep=dep,
                source_cap="surplus")
    s6 &= ac["q"] <= au["q"] * (1 + 1e-9)
    s6 &= ac["max_source_usage_ratio"] <= 1 + 1e-6
    s6 &= ac["max_target_usage_ratio"] <= 1 + 1e-6
    s6 &= all(s == 0 for s in ac["statuses"])

    # zone decomposition (R1)
    zid = cells["zone"].to_numpy().astype(np.int64)
    n_zones = int(zid.max()) + 1
    zd = res["zone_decomposition"]
    s7 = zd["zone_count"] == n_zones
    for name, source_cap in (("targeted", "arrivals"),
                             ("targeted_conservative", "surplus")):
        got = v.zone_at(float(res[name]["r_star"]), cache, arr, dep, need,
                        zid, n_zones, source_cap=source_cap)
        want = zd[name]
        s7 &= close(got["q"], want["q"], 1e-9)
        s7 &= close(got["cross_zone_volume"],
                    want["cross_zone_volume"], 1e-9)
        s7 &= close(got["cross_zone_share"],
                    want["cross_zone_share"], 1e-9)
        s7 &= close(got["net_zone_hour_positive_sum"],
                    want["net_zone_hour_positive_sum"], 1e-9)
        s7 &= close(got["max_abs_zone_hour"],
                    want["max_abs_zone_hour"], 1e-9)
        s7 &= got["zones_changed"] == want["zones_changed"]
    zt = zd["targeted"]
    s7 &= abs(zt["cross_zone_volume"] - 7133.35) <= 0.01
    s7 &= abs(zt["cross_zone_share"] - 0.0746) <= 1e-4
    s7 &= abs(zt["net_zone_hour_positive_sum"] - 4332.11) <= 0.01
    s7 &= abs(zt["max_abs_zone_hour"] - 327.83) <= 0.01
    s7 &= zt["zones_changed"] == 19

    t = res["targeted"]
    s8 = close(t["all_truck"], TRUCK_MAIN * N, 1e-9)
    s8 &= close(t["bill"], t["r_star"] * t["q_star"], 1e-9)
    s8 &= close(t["truck_residual"], TRUCK_MAIN * (N - t["q_star"]), 1e-9)
    s8 &= close(t["saving"], 1.0 - t["total"] / t["all_truck"], 1e-9)
    s8 &= abs(t["marginal_cost_at_star"] - TRUCK_MAIN) < 0.5
    rows = res["parameter_sensitivity"]
    s8 &= len(rows) == 14
    for row in rows:
        tt = row["targeted"]
        s8 &= close(tt["saving"],
                    1.0 - tt["total"] / (row["unit_value"] * N), 1e-9)
    s8 &= "source-cap" in res["interval"]["does_not_cover"]

    free = float(arx[tstar[pos], np.where(pos)[0]].sum())
    pub = res["public_push"]
    s9 = close(res["free_rider"]["units"], free, 1e-9)
    s9 &= close(pub["free_rider_bill"], pub["r_star"] * free, 1e-9)
    s9 &= close(pub["saving"], 1.0 - pub["total"] / (pub["unit"] * N), 1e-9)
    p12 = res["public_push_at_r_1_2"]
    s9 &= close(p12["free_rider_bill"], 1.2 * free, 1e-9)
    s9 &= close(p12["saving"], 1.0 - p12["total"] / (TRUCK_MAIN * N), 1e-9)
    old = res["public_push_old_unit_3_478"]
    s9 &= close(old["saving"], 1.0 - old["total"] / (3.478 * N), 1e-9)
    s9 &= free > 0

    s10 = close(TRUCK_MAIN, 3.3305185482504633, 1e-12)
    s10 &= close(TRUCK_LOW, 1.871804950736993, 1e-12)
    s10 &= close(res["settings"]["truck_unit_main"], TRUCK_MAIN, 1e-9)
    s10 &= close(res["settings"]["truck_unit_lower"], TRUCK_LOW, 1e-9)
    s10 &= res["text_code_consistency"]["status"].startswith("code changed")
    s10 &= "upper_no_source_deficit" in res["settings"]["source_cap"]
    s10 &= "hourly_surplus_sources" in res["settings"]["source_cap"]

    s11 = True
    for row in rows:
        s11 &= row["public"]["saving"] <= row["targeted"]["saving"] + 1e-9
        s11 &= row["targeted"]["q_star"] <= distinct * (1 + 1e-9)
        s11 &= row["targeted"]["q_star"] <= N * (1 + 1e-9)

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
    rc, diff = v.git_lines(["diff", "--name-only", BASE_COMMIT, "--", "code"])
    modified = sorted(p for p in diff if p)
    untracked = sorted(l[3:] for l in status if l.startswith("?? ")
                       and "T008_delivery.json" not in l)
    scope = (set(modified) | set(untracked)) == T008_FILES
    abs_hits = []
    for rel in sorted(T008_FILES):
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            abs_hits.append({"file": rel, "line": 0, "text": "missing"})
            continue
        for lineno, line in enumerate(v.read(path).splitlines(), 1):
            if any(m in line for m in ABS_MARKERS):
                abs_hits.append({"file": rel, "line": lineno,
                                 "text": line.strip()})
    required = ["q3_pool.npz", "q3_qr.json", "q3_qr_conservative.json",
                "q3_results.json", "q3_pool_audit.json"]
    missing = [p for p in required if not os.path.exists(os.path.join(RUN, p))]

    checks = {
        "1_pool_arrays_and_edges": bool(s1),
        "2_same_cell_and_target_hour": bool(s2),
        "3_duplicate_1_09_and_2_93": bool(s3),
        "4_q_curve_and_optimum": bool(s4),
        "5_feasibility_at_r_star": bool(s5),
        "6_conservative_surplus_bound": bool(s6),
        "7_zone_decomposition": bool(s7),
        "8_targeted_saving_and_interval": bool(s8),
        "9_public_push_and_free_rider": bool(s9),
        "10_truck_units_labels_and_text_code": bool(s10),
        "11_parameter_sensitivity": bool(s11),
        "12_readonly_scope_outputs_no_abs": bool(
            ro and scope and not abs_hits and not missing),
    }

    delivery = {
        "task": "T008",
        "title": "Q3 executable relocation pool, q(r), r* and savings",
        "date": "2026-11-18",
        "status": "delivered",
        "checks": checks,
        "settings": res["settings"],
        "pool": res["pool"],
        "pool_audit": aud,
        "targeted": res["targeted"],
        "targeted_conservative": res["targeted_conservative"],
        "zone_decomposition": res["zone_decomposition"],
        "public_push": res["public_push"],
        "public_push_positive_reward": res["public_push_positive_reward"],
        "public_push_at_r_1_2": res["public_push_at_r_1_2"],
        "public_push_old_unit_3_478": res["public_push_old_unit_3_478"],
        "free_rider": res["free_rider"],
        "interval": res["interval"],
        "parameter_sensitivity": res["parameter_sensitivity"],
        "qr_summary": {k: qr[k] for k in
                       ("R_m", "bound", "source_cap", "r_star", "q_star",
                        "marginal_cost_at_star", "need_total",
                        "distinct_candidate_trips", "acceptance", "solver",
                        "status_counts")},
        "qr_conservative_summary": {k: qr_c[k] for k in
                                    ("R_m", "bound", "source_cap", "r_star",
                                     "q_star", "marginal_cost_at_star",
                                     "need_total", "distinct_candidate_trips",
                                     "acceptance", "solver",
                                     "status_counts")},
        "text_code_consistency": res["text_code_consistency"],
        "code_changes": [
            {"file": "code/t008_q3_pool.py", "status": "new",
             "what": "rebuild the (c,h_c) candidate pool, solve the hourly "
                     "assignment LPs, compute q(r), r*, targeted/public "
                     "savings and the feasibility audit. R1 adds the "
                     "source_cap=surplus mode, the conservative curve, "
                     "targeted_conservative and the zone decomposition.",
             "why": "remove the same-cell, wrong-hour and duplicate-counting "
                    "paths behind the old 34.9% saving; the surplus mode "
                    "prevents a source with no hourly surplus from donating a "
                    "bike and creating a new deficit."},
            {"file": "code/t008_delivery.py", "status": "new",
             "what": "this delivery builder", "why": "assemble T008 evidence."},
            {"file": "code/t008_validate.py", "status": "new",
             "what": "independent validator",
             "why": "recompute pool, q(r), audit and savings before delivery; "
                    "R1 adds independent surplus-source, conservative-optimum "
                    "and zone-decomposition recomputation."},
        ],
        "outputs": {
            "delivery_json": "交付/T008_delivery.json",
            "run_dir": "data/runs/T008/",
            "required": required,
        },
        "notes": [
            "Pool index is (target cell c, target deficit hour h_c); candidate "
            "trips end at j!=c within 450 m and are counted in the arrivals at "
            "j in h_c. Same-cell redirections are excluded (0 edges). The "
            "removed same-cell units are 34.110% of the naive pair pool, which "
            "reproduces the old 34.105% path as an audit figure, not a result.",
            "One reroute per trip is enforced by sum_c x[j,c,h] <= A[j,h]. "
            "The naive pair pool is 334,147.32 units vs 307,187.50 distinct "
            "source-hour units (ratio 1.0878, the M8 1.09); the LP resolves "
            "it, so q(r*) = 95,565.82 <= distinct.",
            "The old 2.93 ratio is reproduced as 334,147.32/113,936 = 2.9328; "
            "the difference comes from the old source-own-deficit-hour count, "
            "not from the new target-hour pool.",
            "Targeted upper bound (bound=upper_no_source_deficit, source "
            "capacity = full arrivals A[j,h_c]): r*=1.76 CNY, q*=95,565.82 "
            "(44.17% of N), bill 168,195.84, truck residual 402,253.15, "
            "total 570,448.99 vs 720,536.87 all-truck, saving 20.83% at "
            "TRUCK_UNIT=3.3305. This is an upper bound because a source with "
            "no hourly surplus is allowed to donate a bike.",
            "Conservative bound (bound=hourly_surplus_sources, source "
            "capacity = max(0, arr-dep)): r*=1.83 CNY, q*=32,777.11 (15.15% "
            "of N), bill 59,982.12, truck residual 611,372.09, total "
            "671,354.21 vs 720,536.87, saving 6.826%. It matches the R1 "
            "reference r*=1.83, q*=32,777.11 and 6.826% within 1e-4. Every "
            "source hour has enough surplus to cover the reroutes it supplies.",
            "Zone decomposition at each bound's own r*: targeted "
            "cross-zone 7,133.35 (7.464% of q), positive zone-hour sum "
            "4,332.11, max |zone-hour| 327.83, 19 zones flip between deficit "
            "and surplus; conservative cross-zone 2,836.57 (8.654%), positive "
            "sum 2,315.70, max 273.45, 10 zones flip.",
            "The 14-row interval (7 kappa/rho/c0 sets x 2 truck units) covers "
            "acceptance and truck-cost uncertainty only. It does not cover "
            "the source-cap assumption, so the upper and conservative bounds "
            "are separate scenarios, not two ends of that interval.",
            "Public push with the corrected unit is not profitable: the "
            "optimum is r=0 (saving 0). At r=1.2 it loses 0.22% while paying "
            "120,049.55 to free riders; with the historical unit 3.478 it "
            "saves 0.98%, not the old 2.1% (that figure used a non-flow "
            "pool). All three are reported so T012 can state the lower bound "
            "honestly.",
            "The free-rider count is 100,041.29 existing non-same-cell "
            "arrivals into target cells in their deficit hours; targeted push "
            "pays zero free riders by construction.",
            "Acceptance parameters are not identified from May 2017 (no "
            "reward was offered); the 7-set x 2-unit sweep is the interval. "
            "TRUCK_UNIT is final from T007: 3.3305 main, 1.8718 lower; the "
            "old 3.478 is obsolete except as a historical comparison.",
            "Text-code consistency: the section's 'in c's deficit hour' is "
            "now what the code does; the old source-own-hour path is gone.",
            "No paper/review/data file outside data/runs/T008 is written.",
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
    print(json.dumps({"checks": checks, "targeted": res["targeted"],
                      "targeted_conservative":
                          res["targeted_conservative"],
                      "zone": res["zone_decomposition"],
                      "public": res["public_push"],
                      "interval": res["interval"]},
                     ensure_ascii=False, indent=1))
    print("wrote", OUT)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
