"""T032: sensitivity rerun on the conserving Q2/Q3/Q4 models.

T023's 128-draw pick-freeze design is reused unchanged (same prep, same
A/B/C draws, same job ids).  Only the evaluations change:

  Q2  joint conserving replay (T028) at the draw's demand, inventory and
      tricycle constants, on the reference truck plan and the new
      executable tricycle allocation;
  Q3  truck-caliber payment ceiling at the new payment-ladder reward, with
      the T030 fence capacities and the V_BIKE/WALK_M_MIN draws actually
      applied to tau (T023 drew them but never patched t8);
  Q4  conserving fleet demand (T030) at the draw's arrival/departure ratio
      and inventory, then the 3,500-site MILP.

Outputs: evals.jsonl, reference.json, intervals.json, variance_shares.json
in data/runs/T032/.  The design files are read from T023, never modified.
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
import t008_q3_pool as t8
import t019_q4_cap as t19
import t020_q3_budget as t20
import t023_uncertainty as t23
import t024_q2_tier2 as t24
import t028_q2_conserve as t28
import t030_q4_fleet as t30
import t031_q3_payment as t31

RUN = os.path.join(lib.DATA, "runs", "T032")
os.makedirs(RUN, exist_ok=True)

Q4_SITES = 3500
Q3_R = 0.80
WORKERS = 4

_CAP_T30 = None


def dump(obj, name):
    path = os.path.join(RUN, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    print("wrote", path, flush=True)
    return path


def load_json(name):
    with open(os.path.join(RUN, name), encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
def ref_numbers():
    """Accepted conserving-model reference values the intervals anchor to."""
    ls = t28.load_json("local_search.json")
    fr = json.load(open(os.path.join(lib.DATA, "runs", "T030",
                                     "frontier_cons.json"),
                        encoding="utf-8"))
    q4 = fr["levels"]["meas343"]["knee"]
    q4sel = fr["levels"]["meas343"]["selected"]
    q3 = json.load(open(os.path.join(lib.DATA, "runs", "T031",
                                     "truck_payment.json"),
                        encoding="utf-8"))["levels"]["arrivals"]
    q3b = q3["best"]
    baseline = float(t28.truck_obj + t24.PENALTY * t28.alloc_no_tri_unmet)
    return dict(
        q2=dict(total=float(ls["exec_cost"]), unmet=float(ls["exec_unmet"]),
                fleet=int(ls["exec_n"]),
                baseline=float(baseline),
                saving=float(baseline - ls["exec_cost"]),
                saving_share=float((baseline - ls["exec_cost"]) / baseline),
                truck_obj=float(t28.truck_obj)),
        q3=dict(all_truck=float(q3["all_truck_cost"]),
                total=float(q3b["total_spend"]),
                payment=float(q3b["payment"]), r=float(q3b["r"]),
                q=float(q3b["q"]), P=float(q3b["P"]),
                saving=float(1.0 - q3b["total_spend"]
                             / q3["all_truck_cost"])),
        q4=dict(n_sites=Q4_SITES, coverage=float(q4["coverage"]),
                arrival_coverage=float(q4sel["arrival_coverage"]),
                unassigned=float(q4sel["unassigned"])))


# --------------------------------------------------------------------------
# Q2: conserving draw simulation
# --------------------------------------------------------------------------
def simulate_draw(ctx, nvec, expect_total):
    Z, C = ctx["Z"], ctx["C"]
    I = ctx["I0"].copy()
    XM = ctx["XM"]
    Pcell = ctx["Pcell"]
    zones = ctx["zones"]
    total0 = float(I.sum())
    assert abs(total0 - expect_total) < 1e-6, (total0, expect_total)

    unmet_h = np.zeros(24)
    strict_h = np.zeros(24)
    served_h = np.zeros((24, Z))
    tri_bikes = np.zeros(Z)
    tri_bike_km = np.zeros(Z)
    tri_vehicle_km = np.zeros(Z)
    max_resid = 0.0
    for h in range(24):
        deliver = np.zeros(Z)
        for z, zz in enumerate(zones):
            idx = zz["idx"]
            arcs = [(j, float(XM[h, z, j])) for j in range(Z)
                    if XM[h, z, j] > 1e-12]
            if not arcs:
                continue
            p = float(sum(x for _, x in arcs))
            tot = float(I[idx].sum())
            if p <= 1e-12:
                continue
            scale = 0.0 if tot <= 1e-12 else min(1.0, tot / p)
            moved = p * scale
            if moved > 1e-12:
                I[idx] *= max(0.0, 1.0 - moved / tot)
            for j, x in arcs:
                if x * scale > 1e-12:
                    deliver[j] += x * scale
        for j, zz in enumerate(zones):
            d = float(deliver[j])
            if d <= 1e-12:
                continue
            idx = zz["idx"]
            w = zz["surf"][h]
            if w.sum() <= 1e-12:
                w = np.ones(len(idx))
            I[idx] += d * w / w.sum()
        for z, zz in enumerate(zones):
            idx = zz["idx"]
            I_z = I[idx]
            dep = zz["dep"][h]
            strict_h[h] += float(np.maximum(dep - I_z, 0.0).sum())
            take = np.minimum(dep, I_z)
            I_z -= take
            resid = dep - take
            walk = t24.walk_borrow0(I_z, resid, zz["nb"])
            resid = np.maximum(resid - walk, 0.0)
            served = float(take.sum() + walk.sum())
            n = float(nvec[z])
            if n > 1e-9 and t24.TRI_HOUR0 <= h < t24.TRI_HOUR1:
                sink = np.where(resid > 1e-9)[0]
                donor = np.where(I_z > 1e-9)[0]
                if len(sink) and len(donor):
                    flows, bkm, vkm = t24._tri_lp_local(
                        sink, donor, resid, I_z, zz["sub"], n)
                    if flows:
                        for (d, s, f) in flows:
                            I_z[s] -= f
                            I_z[d] += f
                            I_z[d] -= f
                            resid[d] -= f
                            served += f
                            tri_bikes[z] += f
                        tri_bike_km[z] += bkm
                        tri_vehicle_km[z] += vkm
            I[idx] = I_z
            served_h[h, z] = served
            unmet_h[h] += float(np.maximum(resid, 0.0).sum())
        arr = Pcell[h].T @ served_h[h]
        I += arr
        max_resid = max(max_resid, abs(float(I.sum()) - total0))
    return dict(unmet=float(unmet_h.sum()), strict=float(strict_h.sum()),
                tri_bikes=float(tri_bikes.sum()),
                tri_bike_km=float(tri_bike_km.sum()),
                tri_vehicle_km=float(tri_vehicle_km.sum()),
                end_total=float(I.sum()), max_total_resid=float(max_resid))


def q2_ctx(prep, ctx, nvec):
    inp = dict(dep_cell=ctx.depc, Pcell=t23.pcell(prep, ctx.PZ),
               I0c=ctx.I0c, zid=prep["zid"], xy=prep["xy"])
    return t28.build(inp, dict(XM=prep["XM_plan"]))


def eval_q2(prep, ctx, nvec):
    lp = t23.solve18(ctx.dep_zone, ctx.PZ, ctx.I0z, 60, ctx.c,
                     integer=False, time_limit=180)
    truck_obj = float(lp["obj"])
    c2 = q2_ctx(prep, ctx, nvec)
    with t23.patch24(ctx.c):
        rep = simulate_draw(c2, nvec, float(ctx.I0c.sum()))
        base = simulate_draw(c2, np.zeros_like(nvec),
                             float(ctx.I0c.sum()))
    tri_cost = float(ctx.c["TRI_FIXED"] * float(nvec.sum())
                     + ctx.c["TRI_KM"] * rep["tri_vehicle_km"])
    total = float(truck_obj + tri_cost + ctx.c["PENALTY"] * rep["unmet"])
    baseline = float(truck_obj + ctx.c["PENALTY"] * base["unmet"])
    return dict(total=total, unmet=float(rep["unmet"]),
                truck_obj=truck_obj, tri_cost=tri_cost,
                tri_vkm=float(rep["tri_vehicle_km"]),
                tri_bike_km=float(rep["tri_bike_km"]),
                fleet=float(nvec.sum()), baseline=baseline,
                unmet_n0=float(base["unmet"]),
                saving=float(baseline - total),
                end_total=float(rep["end_total"]),
                max_total_resid=float(rep["max_total_resid"]))


def t18_ready():
    return True


# --------------------------------------------------------------------------
# Q3: truck-caliber ceiling with the T030 capacities
# --------------------------------------------------------------------------
def cap_t30(cap_draw=150.0):
    """T030's per-site spare room, shifted to the draw's slot cap.

    The stored spare is `150 - occupancy`; a draw that changes the cap keeps
    the occupancy path and changes the headroom, so `cap = max(0, cap_draw -
    (150 - spare_150))`.  This is an approximation: re-solving the Q4 MILP at
    every draw's cap would change the layout as well as the headroom.
    """
    global _CAP_T30
    if _CAP_T30 is None:
        _CAP_T30 = t31.build_capacity_t30(None)
    base = float(t19.CAP)
    if abs(float(cap_draw) - base) < 1e-12:
        return _CAP_T30
    out = dict(_CAP_T30)
    out["cap"] = np.maximum(0.0, float(cap_draw) - (base - _CAP_T30["cap"]))
    return out


@contextlib.contextmanager
def patch8(c):
    keys = ("KAPPA", "RHO", "C0", "V_BIKE", "WALK_M_MIN")
    old = {k: getattr(t8, k) for k in keys}
    old_params = t8.PARAMS[0]
    try:
        t8.KAPPA, t8.RHO, t8.C0 = c["KAPPA"], c["RHO"], c["C0"]
        t8.V_BIKE, t8.WALK_M_MIN = c["V_BIKE"], c["WALK_M_MIN"]
        t8.PARAMS[0] = (c["KAPPA"], c["RHO"], c["C0"])
        yield
    finally:
        for k, v in old.items():
            setattr(t8, k, v)
        t8.PARAMS[0] = old_params


def q3_pool(ctx):
    arr, dep = ctx.arrc, ctx.depc
    net = arr - dep
    need = np.maximum(0.0, -net).max(0)
    tstar = np.argmax(-net, axis=0)
    pos = need > 0
    cache = t23.build_edges_cached(arr, t23.dist_matrix(), need, tstar, pos)
    rows = []
    for h, e in cache.items():
        if e is None:
            continue
        src, tg, si, ti, dd = e
        for k in range(len(si)):
            rows.append((h, int(src[si[k]]), int(tg[ti[k]]), float(dd[k])))
    E = np.array(rows)
    cells = t23.cells_table()
    pool = dict(arr=arr, dep=dep, need=need, tstar=tstar, cells=cells,
                e_hour=E[:, 0].astype(int), e_src=E[:, 1].astype(int),
                e_tgt=E[:, 2].astype(int), e_dist=E[:, 3], n_zones=50,
                zid=cells.zone.to_numpy().astype(int),
                N=float(need.sum()))
    return pool, t20.PoolLP(pool, cap_t30(ctx.c["CAP"]))


def eval_q3(prep, ctx):
    """Payment-ledger optimum on the static pool at the draw's parameters.

    The reward grid is 0.00--1.60 in steps of 0.10; the bill is r*P and the
    budget is 30,000 CNY/day.  The reported total is the payment plus the
    truck residual priced at T008's per-bicycle unit cost.
    """
    pool, lp = q3_pool(ctx)
    N = float(pool["N"])
    best = {}
    for cap_name in ("arrivals", "surplus"):
        caps = np.empty(lp.ns)
        for k, (h, s) in enumerate(lp.src_key_list):
            if cap_name == "arrivals":
                caps[k] = pool["arr"][h, s]
            else:
                caps[k] = max(0.0, pool["arr"][h, s] - pool["dep"][h, s])
        lp.b_src = caps
        rows = []
        for r in np.round(np.arange(0.0, 1.6 + 1e-9, 0.10), 2):
            got = lp.solve(float(r), budget=None, cap_on=True)
            pay = float(r * got["P"])
            if pay <= 30000.0 + 1e-9:
                rows.append(dict(
                    r=float(r), q=float(got["q"]), P=float(got["P"]),
                    payment=pay,
                    total=float(pay + t31.truck_unit() * (N - got["q"]))))
        best[cap_name] = min(rows, key=lambda x: x["total"])
    b = best["arrivals"]
    return dict(best_r=b["r"], best_q=b["q"], best_P=b["P"],
                payment=b["payment"], total=b["total"],
                all_truck=float(t31.truck_unit() * N), N=N,
                saving=float(1.0 - b["total"] / (t31.truck_unit() * N)),
                best_r_surplus=best["surplus"]["r"],
                best_total_surplus=best["surplus"]["total"],
                saving_surplus=float(1.0 - best["surplus"]["total"]
                                     / (t31.truck_unit() * N)))


# --------------------------------------------------------------------------
# Q4: conserving demand at the draw's surface and inventory
# --------------------------------------------------------------------------
_T19_IN = None


def t19_inputs():
    global _T19_IN
    if _T19_IN is None:
        _T19_IN = t19.load_inputs()
    return _T19_IN


def q4_ratio(draw, ref):
    pos = ref[ref > 0]
    reg = 1e-2 * float(np.median(pos)) if len(pos) else 1.0
    return np.clip((draw + reg) / (ref + reg), 0.0, 5.0)


def eval_q4(prep, ctx):
    d = t19_inputs()
    ra = q4_ratio(ctx.arrc.sum(0), prep["arrc_ref"].sum(0))
    rd = q4_ratio(ctx.depc.sum(0), prep["depc_ref"].sum(0))
    arr_d = d["arr"] * ra[None, :]
    dep_d = d["dep"] * rd[None, :]
    dem = t30.conserved_demand(t19.to15(arr_d), t19.to15(dep_d), ctx.I0c)
    a = dem["a"]
    ii, jj = t23.q4_arcs(ctx.c["R_WALK"])
    r = t19.solve_cap(a, ii, jj, n=Q4_SITES, integer=True, time_limit=60,
                      cap=ctx.c["CAP"], mip_gap=0.005)
    return dict(coverage=float(r["coverage"]),
                arrival_coverage=float(
                    t19.arrival_coverage(r["u"], a, arr_d.sum(0))),
                unassigned=float(r["unassigned"]),
                assigned=float(r["assigned"]), a_total=float(a.sum()),
                gap=float(r.get("gap", 0.0)),
                demand_resid=float(dem["total_resid"]))


# --------------------------------------------------------------------------
def nvec_ref():
    return np.array(t28.load_json("local_search.json")["exec_allocation"],
                    dtype=int)


def eval_point(prep, ctx, nvec):
    return dict(q2=eval_q2(prep, ctx, nvec), q3=eval_q3(prep, ctx),
                q4=eval_q4(prep, ctx))


# --------------------------------------------------------------------------
# runners
# --------------------------------------------------------------------------
_W = {}


def _worker_init():
    t23._worker_init()
    _W["prep"], _ = t23.run_prep()
    _W["nvec"] = nvec_ref()
    cap_t30()


def _job(job):
    if "prep" not in _W:
        _worker_init()
    prep = _W["prep"]
    ctx = t23.make_ctx(prep, job["a"], job["b"], job["c"])
    ev = eval_point(prep, ctx, _W["nvec"])
    return dict(id=job["id"], tag=job["tag"], a=job["a"], b=job["b"],
                c=job["c"], **ev)


def run_reference(force=False):
    path = os.path.join(RUN, "reference.json")
    if os.path.exists(path) and not force:
        print("reuse", path, flush=True)
        return load_json("reference.json")
    prep, _ = t23.run_prep()
    nvec = nvec_ref()
    ctx = t23.make_ctx(prep)
    t0 = time.time()
    ev = eval_point(prep, ctx, nvec)
    ref = ref_numbers()
    ev["anchors"] = dict(
        q2_total=ref["q2"]["total"] - ev["q2"]["total"],
        q2_unmet=ref["q2"]["unmet"] - ev["q2"]["unmet"],
        q2_baseline=ref["q2"]["baseline"] - ev["q2"]["baseline"],
        q3_all_truck=ref["q3"]["all_truck"] - ev["q3"]["all_truck"],
        q3_total=ref["q3"]["total"] - ev["q3"]["total"],
        q3_payment=ref["q3"]["payment"] - ev["q3"]["payment"],
        q3_saving=ref["q3"]["saving"] - ev["q3"]["saving"],
        q4_coverage=ref["q4"]["coverage"] - ev["q4"]["coverage"],
        q4_arrival=(ref["q4"]["arrival_coverage"]
                    - ev["q4"]["arrival_coverage"]),
        q4_unassigned=ref["q4"]["unassigned"] - ev["q4"]["unassigned"])
    ev["accepted"] = ref
    ev["runtime_s"] = float(time.time() - t0)
    ev["note"] = ("reference evaluation of the conserving chain; anchors "
                  "shift the raw values to the accepted T028/T031/T030 "
                  "numbers.  Q4 at the reference draw reuses the T030 "
                  "arrival/departure surfaces (ratio 1) and should anchor "
                  "at zero.  Q3 capacity uses the stored T030 spare scaled "
                  "to the draw's slot cap.")
    dump(ev, "reference.json")
    return ev


def run_design(force=False, limit=None):
    path = os.path.join(RUN, "evals.jsonl")
    prep, _ = t23.run_prep()
    inputs = t23.run_inputs()
    jobs = t23.build_jobs(inputs)
    done = t23._done_ids(path) if not force else set()
    todo = [j for j in jobs if j["id"] not in done]
    if limit:
        todo = todo[:limit]
    print("jobs %d done %d todo %d workers %d"
          % (len(jobs), len(done), len(todo), WORKERS), flush=True)
    t0 = time.time()
    n = 0
    fh = open(path, "a", encoding="utf-8")
    try:
        for row in t23.pmap_iter(_job, todo, WORKERS,
                                 initializer=_worker_init):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            n += 1
            if n % 8 == 0:
                t23.banner("design %d/%d %s"
                           % (n, len(todo), row["id"]), t0)
    finally:
        fh.close()
    t23.banner("design done", t0)


# --------------------------------------------------------------------------
def raw_value(row, key):
    q2, q3, q4 = row["q2"], row["q3"], row["q4"]
    return {
        "q2_total": q2["total"], "q2_unmet": q2["unmet"],
        "q2_saving_share": q2["saving"] / max(q2["baseline"], 1e-12),
        "q3_all_truck": q3["all_truck"], "q3_total": q3["total"],
        "q3_payment": q3["payment"], "q3_saving": q3["saving"],
        "q4_coverage": q4["coverage"], "q4_arrival": q4["arrival_coverage"],
        "q4_unassigned": q4["unassigned"]}[key]


def anchored(ref, row, key):
    anc = ref["anchors"]
    acc = ref["accepted"]
    if key == "q2_saving_share":
        b = row["q2"]["baseline"] + (acc["q2"]["baseline"]
                                     - ref["q2"]["baseline"])
        t = row["q2"]["total"] + (acc["q2"]["total"]
                                  - ref["q2"]["total"])
        return (b - t) / b
    if key == "q3_saving":
        at = row["q3"]["all_truck"] + (acc["q3"]["all_truck"]
                                       - ref["q3"]["all_truck"])
        t = row["q3"]["total"] + (acc["q3"]["total"]
                                  - ref["q3"]["total"])
        return 1.0 - t / max(at, 1e-12)
    return raw_value(row, key) + anc[key]


def load_evals():
    rows = {}
    with open(os.path.join(RUN, "evals.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
    return rows


OUT_KEYS = [
    ("q2_total", "Q2 two-tier total cost", "CNY/day"),
    ("q2_unmet", "Q2 executable unmet", "bikes/day"),
    ("q2_saving_share", "Q2 saving share vs no-tricycle", "share"),
    ("q3_all_truck", "Q3 all-truck pool cost", "CNY/day"),
    ("q3_total", "Q3 payment-ledger optimum total", "CNY/day"),
    ("q3_payment", "Q3 payment-ledger bill", "CNY/day"),
    ("q3_saving", "Q3 saving share vs all-truck", "share"),
    ("q4_coverage", "Q4 occupancy coverage at 3,500 sites", "share"),
    ("q4_arrival", "Q4 arrival-volume coverage", "share"),
    ("q4_unassigned", "Q4 unassigned occupancy demand", "bike-slots"),
]


def ref_value(ref, key):
    acc = ref["accepted"]
    if key.startswith("q2_"):
        if key == "q2_total":
            return acc["q2"]["total"]
        if key == "q2_unmet":
            return acc["q2"]["unmet"]
        return acc["q2"]["saving_share"]
    if key == "q3_all_truck":
        return acc["q3"]["all_truck"]
    if key == "q3_total":
        return acc["q3"]["total"]
    if key == "q3_payment":
        return acc["q3"]["payment"]
    if key == "q3_saving":
        return acc["q3"]["saving"]
    if key == "q4_coverage":
        return acc["q4"]["coverage"]
    if key == "q4_arrival":
        return acc["q4"]["arrival_coverage"]
    return acc["q4"]["unassigned"]


def run_intervals():
    ref = load_json("reference.json")
    rows = load_evals()
    slices = dict(
        F=[rows["F_%03d" % i] for i in range(t23.N_HALF)],
        A=[rows["MA_%03d" % i] for i in range(t23.N_HALF)],
        B=[rows["BE_%02d" % k] for k in range(len(t23.B_COMBOS))],
        C=[rows["MC_%03d" % i] for i in range(t23.N_HALF)])
    out = dict(task="T032 intervals", date="2026-11-19",
               method=("P5/P50/P95 of the anchored output over each group's "
                       "marginal slice; F is the joint sample"),
               outputs={})
    for key, label, unit in OUT_KEYS:
        entry = dict(label=label, unit=unit, reference=ref_value(ref, key),
                     slices={})
        for sname, srows in slices.items():
            v = np.array([anchored(ref, r, key) for r in srows])
            entry["slices"][sname] = dict(
                n=int(len(v)),
                p5=float(np.percentile(v, 5)),
                p50=float(np.percentile(v, 50)),
                p95=float(np.percentile(v, 95)),
                mean=float(v.mean()), sd=float(v.std(ddof=1)))
        out["outputs"][key] = entry
    # K* is the truck LP scan, unchanged by the replay; copy T023's
    kp = os.path.join(lib.DATA, "runs", "T023", "kstar.json")
    if os.path.exists(kp):
        ks = json.load(open(kp, encoding="utf-8"))["rows"]
        v = np.array([r["kstar"] for r in ks], dtype=float)
        out["outputs"]["q2_kstar"] = dict(
            label="Q2 optimal truck K*", unit="trucks", reference=60,
            slices=dict(F=dict(
                n=int(len(v)), p5=float(np.percentile(v, 5)),
                p50=float(np.percentile(v, 50)),
                p95=float(np.percentile(v, 95)),
                distribution={str(int(k)): int((v == k).sum())
                              for k in sorted(set(v))})))
    dump(out, "intervals.json")
    return out


def run_shares():
    ref = load_json("reference.json")
    rows = load_evals()

    def series(key, ids):
        return np.array([anchored(ref, rows[i], key) for i in ids])

    out = dict(task="T032 variance shares", date="2026-11-19",
               groups=["A", "B", "C"], outputs={})
    for key, label, unit in OUT_KEYS:
        f = series(key, ["F_%03d" % i for i in range(t23.N_HALF)])
        var = float(f.var(ddof=1))
        res = dict(label=label, unit=unit, var_F=var, mean_F=float(f.mean()),
                   n_joint=t23.N_HALF, n_marginal=t23.N_HALF,
                   n_B=len(t23.B_COMBOS))
        sj = {}
        for g in "ABC":
            s = series(key, ["S%s_%03d" % (g, i) for i in range(t23.N_HALF)])
            m = series(key, ["M%s_%03d" % (g, i) for i in range(t23.N_HALF)])
            sj[g] = float((np.mean(f * s) - f.mean() * s.mean()) / var)
            res["share_marginal_" + g] = float(m.var(ddof=1) / var)
        res["share_first_order"] = sj
        res["residual_1_minus_sum"] = float(1.0 - sum(sj.values()))
        out["outputs"][key] = res
    dump(out, "variance_shares.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=("ref", "design", "intervals", "shares", "all"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.stage in ("ref", "all"):
        run_reference(force=args.force)
    if args.stage in ("design", "all"):
        run_design(force=args.force, limit=args.limit)
    if args.stage in ("intervals", "all"):
        run_intervals()
    if args.stage in ("shares", "all"):
        run_shares()
