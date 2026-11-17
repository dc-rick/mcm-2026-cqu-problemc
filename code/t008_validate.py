"""Independent validator for the T008 Q3 delivery.

Rebuilds the (c,h_c) candidate pool, re-solves the hourly assignment LPs at
sample rewards, and recomputes the feasibility audit, savings, public-push
free-rider bill and the read-only/scope checks.  It does not import
t008_delivery.py or t008_q3_pool.py.
"""
import hashlib
import io
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

import lib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
PAPER = os.path.join(ROOT, "paper")
RUN = os.path.join(DATA, "runs", "T008")
DELIV = os.path.join(ROOT, "交付")
T015_FP = os.path.join(DATA, "runs", "T015", "fingerprint_start.json")
T001_FP = os.path.join(DATA, "runs", "T001", "fingerprint_start.json")

SCALE = 1.0 / 0.128
R = 0.45
KAPPA, RHO, C0 = 2.0, 0.35, 0.8
V_BIKE, WALK_M_MIN = 10.0, 80.0
TRUCK_MAIN = 82194.7968 / 24679.2791
TRUCK_LOW = 46194.7968 / 24679.2791
TOL = 1e-9
BASE_COMMIT = "7a3c71b"
T008_FILES = {"code/t008_q3_pool.py", "code/t008_delivery.py",
              "code/t008_validate.py"}
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


def load_all():
    cells = lib.get_cells()
    C = len(cells)
    D = np.load(os.path.join(DATA, "dist_matrix.npy")).astype(np.float64)
    df = lib.load_trips(lib.TYP_DAYS)
    df = lib.assign_cells(df, cells)
    df = df[df.o.notna() & df.d.notna()].copy()
    df["o"] = df.o.astype(int)
    df["d"] = df.d.astype(int)
    hour = df.starttime.dt.hour.values
    nd = len(lib.TYP_DAYS)
    arr = np.zeros((24, C))
    dep = np.zeros((24, C))
    arx = np.zeros((24, C))
    np.add.at(arr, (hour, df.d.values), 1.0)
    np.add.at(dep, (hour, df.o.values), 1.0)
    diff = df.o.values != df.d.values
    np.add.at(arx, (hour[diff], df.d.values[diff]), 1.0)
    arr = arr / nd * SCALE
    dep = dep / nd * SCALE
    arx = arx / nd * SCALE
    net = arr - dep
    need = np.maximum(0.0, -net).max(0)
    tstar = np.argmax(-net, axis=0)
    pos = need > 0
    return cells, D, arr, dep, arx, need, tstar, pos, float(need.sum())


def build_edges(arr, D, need, tstar, pos):
    near = D <= R
    cache = {}
    for h in range(24):
        tg = np.where(pos & (tstar == h))[0]
        src = np.where(arr[h] > 0)[0]
        if len(tg) == 0 or len(src) == 0:
            cache[h] = None
            continue
        sub = near[np.ix_(src, tg)].copy()
        for k, c in enumerate(tg):
            j = np.where(src == c)[0]
            if len(j):
                sub[j[0], k] = False
        nz = np.argwhere(sub)
        si, ti = nz[:, 0], nz[:, 1]
        cache[h] = (src, tg, si, ti, D[src[si], tg[ti]])
    return cache


def tau(d):
    return d / V_BIKE * 60.0 + d * 1000.0 / WALK_M_MIN


def pi(r, d, kappa=KAPPA, rho=RHO, c0=C0):
    return 1.0 / (1.0 + np.exp(-(kappa * r - rho * tau(d) - c0)))


def source_caps(h, src, arr, dep, source_cap):
    caps = arr[h, src].astype(np.float64)
    if source_cap == "surplus":
        caps = np.maximum(0.0, caps - dep[h, src])
    elif source_cap != "arrivals":
        raise ValueError("unknown source_cap %r" % (source_cap,))
    return caps


def q_at(r, cache, arr, need, params=(KAPPA, RHO, C0), dep=None,
         source_cap="arrivals", want_x=False):
    kappa, rho, c0 = params
    total = 0.0
    max_src = 0.0
    max_tgt = 0.0
    max_offer = 0.0
    statuses = []
    for h, e in cache.items():
        if e is None:
            continue
        src, tg, si, ti, d = e
        w = pi(r, d, kappa, rho, c0) - pi(0.0, d, kappa, rho, c0)
        keep = w > 1e-12
        if not keep.any():
            continue
        si, ti, w = si[keep], ti[keep], w[keep]
        ne = len(w)
        A = coo_matrix((np.r_[np.ones(ne), w],
                        (np.r_[si, len(src) + ti],
                         np.r_[np.arange(ne), np.arange(ne)])),
                       shape=(len(src) + len(tg), ne)).tocsr()
        caps = source_caps(h, src, arr, dep, source_cap)
        b = np.r_[caps, need[tg]]
        res = linprog(-w, A_ub=A, b_ub=b, bounds=(0, None), method="highs")
        statuses.append(int(res.status))
        if res.status != 0:
            continue
        x = res.x
        total += float(w @ x)
        use = np.bincount(si, weights=x, minlength=len(src))
        max_src = max(max_src, float(np.max(use / np.maximum(
            caps, 1e-12))))
        tu = np.bincount(ti, weights=w * x, minlength=len(tg))
        max_tgt = max(max_tgt, float(np.max(tu / np.maximum(
            need[tg], 1e-12))))
        max_offer = max(max_offer, float(x.sum()))
    return {"q": total, "max_source_usage_ratio": max_src,
            "max_target_usage_ratio": max_tgt,
            "max_hourly_offer_sum": max_offer,
            "statuses": statuses}


def zone_at(r, cache, arr, dep, need, zid, n_zones,
            params=(KAPPA, RHO, C0), source_cap="arrivals"):
    """Independent recomputation of the zone decomposition."""
    kappa, rho, c0 = params
    net = np.zeros((24, n_zones))
    base = np.zeros((24, n_zones))
    for h in range(24):
        base[h] = np.bincount(zid, weights=arr[h] - dep[h],
                              minlength=n_zones)
    cross = 0.0
    total_q = 0.0
    for h, e in cache.items():
        if e is None:
            continue
        src, tg, si, ti, d = e
        w = pi(r, d, kappa, rho, c0) - pi(0.0, d, kappa, rho, c0)
        keep = w > 1e-12
        if not keep.any():
            continue
        si, ti, w = si[keep], ti[keep], w[keep]
        ne = len(w)
        A = coo_matrix((np.r_[np.ones(ne), w],
                        (np.r_[si, len(src) + ti],
                         np.r_[np.arange(ne), np.arange(ne)])),
                       shape=(len(src) + len(tg), ne)).tocsr()
        caps = source_caps(h, src, arr, dep, source_cap)
        res = linprog(-w, A_ub=A, b_ub=np.r_[caps, need[tg]],
                      bounds=(0, None), method="highs")
        if res.status != 0:
            continue
        y = w * res.x
        total_q += float(y.sum())
        sc, dc = src[si], tg[ti]
        cross += float(y[zid[sc] != zid[dc]].sum())
        cn = (np.bincount(dc, weights=y, minlength=len(zid))
              - np.bincount(sc, weights=y, minlength=len(zid)))
        net[h] = np.bincount(zid, weights=cn, minlength=n_zones)
    tot = base + net
    sb = np.sign(np.where(np.abs(base) < 1e-9, 0.0, base))
    st = np.sign(np.where(np.abs(tot) < 1e-9, 0.0, tot))
    changed = int(np.count_nonzero((sb != st).any(0)))
    return {
        "q": total_q,
        "cross_zone_volume": cross,
        "cross_zone_share": cross / total_q if total_q > 0 else None,
        "net_zone_hour_positive_sum": float(np.maximum(net, 0).sum()),
        "max_abs_zone_hour": float(np.abs(net).max()),
        "zones_changed": changed,
    }


def pack(r, q, unit, free=0.0, N=216343.75):
    total = r * (q + free) + unit * (N - q)
    return {"total": total, "saving": 1.0 - total / (unit * N)}


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


def main():
    cells, D, arr, dep, arx, need, tstar, pos, N = load_all()
    C = len(cells)
    cache = build_edges(arr, D, need, tstar, pos)
    pool = np.load(os.path.join(RUN, "q3_pool.npz"))
    qr = json.load(io.open(os.path.join(RUN, "q3_qr.json"), encoding="utf-8"))
    res = json.load(io.open(os.path.join(RUN, "q3_results.json"),
                            encoding="utf-8"))
    aud = json.load(io.open(os.path.join(RUN, "q3_pool_audit.json"),
                            encoding="utf-8"))
    results = []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(("PASS " if ok else "FAIL ") + name + (("  " + detail) if detail else ""))

    # rebuild edge arrays in a stable order
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
    edge_dist = np.array([e[3] for e in edges])
    naive = float(arr[edge_hour, edge_src].sum())
    group = {(int(h), int(s)) for s, h in zip(edge_src, edge_hour)}
    distinct = float(sum(arr[h, s] for h, s in group))
    same = float(arr[tstar[pos], np.where(pos)[0]].sum())

    # 1 --- pool arrays and feasibility ------------------------------------
    keys = {"edge_src", "edge_dst", "edge_hour", "edge_dist_km",
            "edge_src_arrivals", "edge_dst_need", "target_cell",
            "target_hour", "target_need", "target_arrivals",
            "target_free_riders", "need", "tstar", "source_hour_arrivals"}
    pool_ok = keys <= set(pool.files)
    pool_ok &= np.array_equal(pool["edge_src"], edge_src.astype(np.int32))
    pool_ok &= np.array_equal(pool["edge_dst"], edge_dst.astype(np.int32))
    pool_ok &= np.array_equal(pool["edge_hour"], edge_hour.astype(np.int8))
    pool_ok &= np.allclose(pool["edge_dist_km"], edge_dist, atol=1e-12)
    pool_ok &= np.allclose(pool["edge_src_arrivals"],
                           arr[edge_hour, edge_src], atol=1e-12)
    pool_ok &= np.allclose(pool["edge_dst_need"], need[edge_dst], atol=1e-12)
    pool_ok &= np.allclose(pool["need"], need, atol=1e-12)
    pool_ok &= np.array_equal(pool["tstar"], tstar.astype(pool["tstar"].dtype))
    pool_ok &= bool((edge_src != edge_dst).all())
    pool_ok &= bool((edge_dist <= R + 1e-9).all())
    pool_ok &= bool((edge_hour == tstar[edge_dst]).all())
    pool_ok &= close(naive, aud["naive_pair_units"], 1e-12)
    pool_ok &= close(distinct, aud["distinct_units"], 1e-12)
    pool_ok &= close(same, aud["same_cell_excluded_units"], 1e-12)
    check("1 pool arrays and edge validity", pool_ok,
          f"{len(edges)} edges, N={N:,.2f}")

    # 2 --- same-cell and hour indexing ------------------------------------
    same_ok = (int(pool["edge_src"].size) > 0
               and bool((pool["edge_src"] != pool["edge_dst"]).all()))
    same_ok &= close(same / naive, 0.34110008149741483, 1e-9)
    same_ok &= abs(same / naive - 0.34105) < 0.002
    same_ok &= bool((pool["target_hour"] == tstar[pool["target_cell"]]).all())
    same_ok &= aud["same_cell_share_of_naive_pair"] > 0.34
    same_ok &= aud["old_34_105_reproduced"] is True
    check("2 same-cell removed and target-hour index", same_ok,
          f"same-cell share of naive pair pool {100*same/naive:.3f}%")

    # 3 --- 1.09 duplicate resolution --------------------------------------
    dup_ok = close(naive / distinct, 1.087763406481611, 1e-9)
    dup_ok &= aud["old_1_09_reproduced"] is True
    dup_ok &= aud["old_2_93_reproduced"] is True
    dup_ok &= close(naive / 113936.0, 2.9327633182538566, 1e-9)
    dup_ok &= distinct > naive / 1.5
    check("3 duplicate 1.09 and 2.93 reproduced", dup_ok,
          f"naive {naive:,.2f} / distinct {distinct:,.2f} = {naive/distinct:.6f}")

    # 4 --- q(r) curve and optimum -----------------------------------------
    rs = np.array(qr["r_grid"])
    qs = np.array(qr["q"])
    curve_ok = len(rs) == len(qs) and np.all(np.diff(rs) > 0)
    curve_ok &= np.all(np.diff(qs) >= -1e-9)
    for r in (0.0, 0.5, 1.0, 1.5, float(qr["r_star"]), 2.0, 3.0):
        got = q_at(r, cache, arr, need)["q"]
        want = float(np.interp(r, rs, qs))
        curve_ok &= close(got, want, 1e-6)
    total = rs * qs + TRUCK_MAIN * (N - qs)
    i = int(np.argmin(total))
    curve_ok &= close(rs[i], res["targeted"]["r_star"], 1e-12)
    curve_ok &= close(qs[i], res["targeted"]["q_star"], 1e-9)
    curve_ok &= close(res["targeted"]["saving"], 1.0 - total[i] / (TRUCK_MAIN * N), 1e-9)
    check("4 q(r) curve, optimum and saving", curve_ok,
          f"r*={rs[i]:.2f}, q*={qs[i]:,.1f}")

    # 5 --- feasibility audit at r* ----------------------------------------
    au = q_at(float(res["targeted"]["r_star"]), cache, arr, need)
    f_ok = close(au["q"], res["targeted"]["q_star"], 1e-9)
    f_ok &= au["q"] <= distinct * (1 + 1e-9)
    f_ok &= au["max_source_usage_ratio"] <= 1 + 1e-6
    f_ok &= au["max_target_usage_ratio"] <= 1 + 1e-6
    f_ok &= all(s == 0 for s in au["statuses"])
    f_ok &= bool(aud["feasibility"]["q_le_distinct"])
    f_ok &= bool(aud["feasibility"]["source_usage_le_arrivals"])
    f_ok &= bool(aud["feasibility"]["target_usage_le_need"])
    f_ok &= aud["feasibility"]["same_cell_edges"] == 0
    f_ok &= close(aud["audit_at_r_star"]["q"], au["q"], 1e-9)
    check("5 feasibility at r*", f_ok,
          f"q={au['q']:,.1f} <= distinct {distinct:,.1f}; "
          f"source max {au['max_source_usage_ratio']:.12f}")

    # 6 --- conservative hourly-surplus bound ------------------------------
    qr_c = json.load(io.open(os.path.join(RUN, "q3_qr_conservative.json"),
                             encoding="utf-8"))
    rsc = np.array(qr_c["r_grid"])
    qsc = np.array(qr_c["q"])
    cons_ok = len(rsc) == len(qsc) and np.all(np.diff(rsc) > 0)
    cons_ok &= np.all(np.diff(qsc) >= -1e-9)
    cons_ok &= np.all(qsc <= qs + 1e-9)
    for r in (0.0, 0.5, 1.0, float(qr_c["r_star"]), 2.0, 3.0):
        got = q_at(r, cache, arr, need, dep=dep,
                   source_cap="surplus")["q"]
        cons_ok &= close(got, float(np.interp(r, rsc, qsc)), 1e-6)
    cons_total = rsc * qsc + TRUCK_MAIN * (N - qsc)
    ic = int(np.argmin(cons_total))
    cons_ok &= close(rsc[ic], res["targeted_conservative"]["r_star"], 1e-12)
    cons_ok &= close(qsc[ic], res["targeted_conservative"]["q_star"], 1e-9)
    cons_ok &= close(rsc[ic], 1.83, 1e-12)
    cons_ok &= abs(qsc[ic] - 32777.11) <= 1e-2
    cons_save = 1.0 - cons_total[ic] / (TRUCK_MAIN * N)
    cons_ok &= abs(cons_save - 0.06826) <= 1e-4
    cons_ok &= close(cons_save,
                     res["targeted_conservative"]["saving"], 1e-9)
    cons_ok &= (res["targeted_conservative"]["bound"]
                == "hourly_surplus_sources")
    cons_ok &= (res["targeted"]["bound"] == "upper_no_source_deficit")
    check("6 conservative surplus-source bound", cons_ok,
          f"r*={rsc[ic]:.2f}, q*={qsc[ic]:,.2f}, "
          f"saving {100*cons_save:.4f}%")

    # 7 --- zone decomposition --------------------------------------------
    zid = cells["zone"].to_numpy().astype(np.int64)
    n_zones = int(zid.max()) + 1
    zd = res["zone_decomposition"]
    z_ok = True
    for name, source_cap in (("targeted", "arrivals"),
                             ("targeted_conservative", "surplus")):
        got = zone_at(float(res[name]["r_star"]), cache, arr, dep, need,
                      zid, n_zones, source_cap=source_cap)
        want = zd[name]
        z_ok &= close(got["q"], want["q"], 1e-9)
        z_ok &= close(got["cross_zone_volume"],
                      want["cross_zone_volume"], 1e-9)
        z_ok &= close(got["cross_zone_share"],
                      want["cross_zone_share"], 1e-9)
        z_ok &= close(got["net_zone_hour_positive_sum"],
                      want["net_zone_hour_positive_sum"], 1e-9)
        z_ok &= close(got["max_abs_zone_hour"],
                      want["max_abs_zone_hour"], 1e-9)
        z_ok &= got["zones_changed"] == want["zones_changed"]
    zt = zd["targeted"]
    z_ok &= abs(zt["cross_zone_volume"] - 7133.35) <= 0.01
    z_ok &= abs(zt["cross_zone_share"] - 0.0746) <= 1e-4
    z_ok &= abs(zt["net_zone_hour_positive_sum"] - 4332.11) <= 0.01
    z_ok &= abs(zt["max_abs_zone_hour"] - 327.83) <= 0.01
    z_ok &= zt["zones_changed"] == 19
    check("7 zone decomposition targeted and conservative", z_ok,
          f"cross {zt['cross_zone_volume']:,.2f}, share "
          f"{zt['cross_zone_share']:.4f}, zones {zt['zones_changed']}")

    # 8 --- targeted savings and interval ----------------------------------
    t = res["targeted"]
    sav_ok = close(t["all_truck"], TRUCK_MAIN * N, 1e-9)
    sav_ok &= close(t["bill"], t["r_star"] * t["q_star"], 1e-9)
    sav_ok &= close(t["truck_residual"],
                    TRUCK_MAIN * (N - t["q_star"]), 1e-9)
    sav_ok &= close(t["total"], t["bill"] + t["truck_residual"], 1e-9)
    sav_ok &= close(t["saving"], 1.0 - t["total"] / t["all_truck"], 1e-9)
    sav_ok &= abs(t["marginal_cost_at_star"] - TRUCK_MAIN) < 0.5
    rows = res["parameter_sensitivity"]
    sav_ok &= len(rows) == 14
    for row in rows:
        tt = row["targeted"]
        sav_ok &= close(tt["saving"],
                        1.0 - tt["total"] / (row["unit_value"] * N), 1e-9)
    sav_ok &= close(res["interval"]["targeted_saving_min"],
                    min(row["targeted"]["saving"] for row in rows), 1e-9)
    sav_ok &= close(res["interval"]["targeted_saving_max"],
                    max(row["targeted"]["saving"] for row in rows), 1e-9)
    check("8 targeted saving and interval", sav_ok,
          f"saving {100*t['saving']:.2f}%, interval "
          f"[{100*res['interval']['targeted_saving_min']:.2f}, "
          f"{100*res['interval']['targeted_saving_max']:.2f}]%")

    # 9 --- public push and free-rider cost --------------------------------
    free = float(arx[tstar[pos], np.where(pos)[0]].sum())
    pub = res["public_push"]
    pub_ok = close(res["free_rider"]["units"], free, 1e-9)
    pub_ok &= close(pub["free_rider_bill"], pub["r_star"] * free, 1e-9)
    pub_ok &= close(pub["saving"],
                    1.0 - pub["total"] / (pub["unit"] * N), 1e-9)
    p12 = res["public_push_at_r_1_2"]
    pub_ok &= close(p12["free_rider_bill"], 1.2 * free, 1e-9)
    pub_ok &= close(p12["saving"],
                    1.0 - p12["total"] / (TRUCK_MAIN * N), 1e-9)
    old = res["public_push_old_unit_3_478"]
    pub_ok &= close(old["bill"], old["r_star"] * old["q_star"], 1e-9)
    pub_ok &= close(old["free_rider_bill"], old["r_star"] * free, 1e-9)
    pub_ok &= close(old["saving"],
                    1.0 - old["total"] / (3.478 * N), 1e-9)
    pub_ok &= free > 0 and res["free_rider"]["targeted_free_rider_bill"] == 0
    check("9 public push and free-rider bill", pub_ok,
          f"free riders {free:,.2f}; public new-unit saving "
          f"{100*pub['saving']:.2f}%, old-unit {100*old['saving']:.2f}%")

    # 10 --- truck units, labels and text-code consistency ----------------
    unit_ok = close(TRUCK_MAIN, 3.3305185482504633, 1e-12)
    unit_ok &= close(TRUCK_LOW, 1.871804950736993, 1e-12)
    unit_ok &= close(res["settings"]["truck_unit_main"], TRUCK_MAIN, 1e-9)
    unit_ok &= close(res["settings"]["truck_unit_lower"], TRUCK_LOW, 1e-9)
    unit_ok &= "3.478" in res["settings"]["truck_unit_source"]
    unit_ok &= res["text_code_consistency"]["status"].startswith("code changed")
    unit_ok &= "target cell" in res["text_code_consistency"]["new"]
    unit_ok &= bool(res["settings"]["source_cap"]["upper_no_source_deficit"])
    unit_ok &= bool(res["settings"]["source_cap"]["hourly_surplus_sources"])
    unit_ok &= "does_not_cover" in res["interval"]
    unit_ok &= "source-cap" in res["interval"]["does_not_cover"]
    check("10 truck units, labels and interval caveat", unit_ok,
          f"main {TRUCK_MAIN:.4f}, lower {TRUCK_LOW:.4f}")

    # 11 --- parameter sensitivity and monotonicity -----------------------
    ps_ok = True
    for row in rows:
        ps_ok &= row["public"]["saving"] <= row["targeted"]["saving"] + 1e-9
        ps_ok &= row["targeted"]["q_star"] <= distinct * (1 + 1e-9)
        ps_ok &= row["targeted"]["q_star"] <= N * (1 + 1e-9)
    ps_ok &= res["interval"]["r_star_min"] <= res["interval"]["r_star_max"]
    ps_ok &= res["interval"]["q_star_min"] <= res["interval"]["q_star_max"]
    ps_ok &= res["interval"]["public_saving_min"] <= res["interval"]["public_saving_max"]
    check("11 parameter sensitivity consistency", ps_ok,
          f"{len(rows)} rows")

    # 12 --- read-only, scope, outputs, delivery, no abs paths ------------
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
                       and "T008_delivery.json" not in l)
    scope = (set(modified) | set(untracked)) == T008_FILES
    abs_hits = []
    for rel in sorted(T008_FILES):
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            abs_hits.append({"file": rel, "line": 0, "text": "missing"})
            continue
        for lineno, line in enumerate(read(path).splitlines(), 1):
            if any(m in line for m in ABS_MARKERS):
                abs_hits.append({"file": rel, "line": lineno,
                                 "text": line.strip()})
    required = ["q3_pool.npz", "q3_qr.json", "q3_qr_conservative.json",
                "q3_results.json", "q3_pool_audit.json"]
    missing = [p for p in required if not os.path.exists(os.path.join(RUN, p))]
    dpath = os.path.join(DELIV, "T008_delivery.json")
    delivery_ok = os.path.exists(dpath)
    if delivery_ok:
        delivery = json.load(io.open(dpath, encoding="utf-8"))
        delivery_ok = (len(delivery["checks"]) == 12
                       and all(delivery["checks"].values()))
    check("12 readonly, scope, outputs, delivery, no abs paths",
          ro and scope and not abs_hits and not missing and delivery_ok,
          f"modified {len(modified)}, untracked {len(untracked)}")

    passed = sum(results)
    print(f"\nvalidator: {passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
