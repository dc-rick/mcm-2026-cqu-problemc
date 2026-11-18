"""T020: Q3 pay-on-completion incentives under a hard budget.

Two ledgers, one LP
-------------------
T008 billed ``r * q`` where ``q = sum_e (pi(r)-pi(0)) x_e`` is the
*incremental* reroute volume.  Under pay-on-completion every invited rider
who completes is paid, including the share ``pi(0)`` who would have chosen
the target anyway, so the payable head count is ``P = sum_e pi(r) x_e`` and
the bill is ``r P``.  This module keeps both ledgers and reports ``q``,
``P``, invitations and payment for every row.

Budget in the optimisation
--------------------------
The bill constraint ``r sum_e pi(r) x_e <= B`` couples the 24 hourly LPs,
so the pool is solved as one stacked LP per reward.  Rows: source-hour
caps, target-cell need, the T019 fence-capacity rows (optional), and the
budget row (optional).  Scanning r on a 0.01 grid and, for each B, taking
the r that minimises ``payment + unit * (N - q)`` is the joint (r, x)
optimisation the task asks for.

Fence capacity
--------------
T019's ``spare_capacity.npz`` gives each selected site's 15-min headroom on
the design day.  The replay attribution rule is used: an arrival at cell c
adds ``f[c,j]/a[c]`` bikes to site j, so the hourly constraint is

    sum_{e: target c} (f[c,j]/a[c]) * w_e * x_e <= min_{4 steps of h} spare[j,h]

for every open site j.  Reroutes whose target cell is served by no site
are counted separately (they are outside the fence system).

Stages: repro | policy | hardbudget | hybrid | check | all.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib                                            # noqa: E402
from lib import DATA                                  # noqa: E402
import t008_q3_pool as t8                             # noqa: E402

RUN = os.path.join(DATA, "runs", "T020")
os.makedirs(RUN, exist_ok=True)

BUDGETS = [30000.0, 60000.0, 100000.0, 170000.0, None]
R_GRID = np.round(np.arange(0.01, 3.0001, 0.01), 4)
TRUCK_UNIT = t8.TRUCK_MAIN                 # 82,194.7968 / 24,679.2791
SIM_DAYS = 1000
SIM_SEED = 20261118
LEVEL = "meas343"


def dump(obj, name):
    path = os.path.join(RUN, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    print("wrote", path)
    return path


def load_json(name):
    with open(os.path.join(RUN, name), encoding="utf-8") as fh:
        return json.load(fh)


def load_json_any(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# pool and fence capacity
# --------------------------------------------------------------------------
def load_pool():
    cells, xy, D, arr, dep, arx = t8.load_data()
    net = arr - dep
    need = np.maximum(0.0, -net).max(0)
    tstar = np.argmax(-net, axis=0)
    pos = need > 0
    cache = t8.build_edges(arr, D, need, tstar, pos)
    rows = []
    for h, e in cache.items():
        if e is None:
            continue
        src, tg, si, ti, d = e
        for k in range(len(si)):
            rows.append((h, int(src[si[k]]), int(tg[ti[k]]), float(d[k])))
    E = np.array(rows)
    out = dict(arr=arr, dep=dep, need=need, tstar=tstar, cells=cells,
               e_hour=E[:, 0].astype(int), e_src=E[:, 1].astype(int),
               e_tgt=E[:, 2].astype(int), e_dist=E[:, 3])
    out["zid"] = cells["zone"].to_numpy().astype(int)
    out["n_zones"] = int(out["zid"].max()) + 1
    out["N"] = float(need.sum())
    return out


def build_capacity(pool, level=LEVEL):
    """Site-level hourly headroom and per-edge attribution coefficients.

    Returns cap (nsites, 24) and attrib: for each edge the list of
    (site_pos, f[c,j]/a[c]) pairs of its target cell.
    """
    plan = np.load(os.path.join(DATA, "runs", "T019",
                                f"plan_{level}_main.npz"))
    s, f, ii, jj, a = (plan[k] for k in ("s", "f", "ii", "jj", "a"))
    sites = np.where(s > 0.5)[0]
    spos = {int(j): k for k, j in enumerate(sites)}
    sp = np.load(os.path.join(DATA, "runs", "T019", "spare_capacity.npz"))
    sites_arr = sp[f"sites_{level}"]
    if not np.array_equal(sites, sites_arr):
        raise RuntimeError("T019 site order mismatch")
    spare = sp[f"spare_{level}"]
    cap = spare.reshape(24, 4, len(sites)).min(1)      # (24, nsites)
    serv = {}
    for k in range(len(f)):
        if f[k] <= 0 or a[ii[k]] <= 0:
            continue
        j = int(jj[k])
        if s[j] <= 0.5:
            continue
        serv.setdefault(int(ii[k]), []).append(
            (spos[j], float(f[k] / a[ii[k]])))
    z = np.load(os.path.join(DATA, "runs", "T019", "cell_demand.npz"))
    outside = np.ones(len(z["a_meas343"]), bool)
    for c in serv:
        outside[c] = False
    return dict(cap=cap, serv=serv, outside=outside, sites=sites,
                n_sites=len(sites))


class PoolLP:
    """One stacked LP over all 24 hours; rebuilt per reward value."""

    def __init__(self, pool, capinfo):
        self.p = pool
        self.capinfo = capinfo
        self.E = len(pool["e_src"])
        self.w = None
        self.pi = None
        src_key, tgt_key = {}, {}
        srow = np.empty(self.E, int)
        trow = np.empty(self.E, int)
        for k in range(self.E):
            ks = (int(pool["e_hour"][k]), int(pool["e_src"][k]))
            kt = (int(pool["e_hour"][k]), int(pool["e_tgt"][k]))
            if ks not in src_key:
                src_key[ks] = len(src_key)
            if kt not in tgt_key:
                tgt_key[kt] = len(tgt_key)
            srow[k] = src_key[ks]
            trow[k] = tgt_key[kt]
        self.src_key, self.tgt_key = src_key, tgt_key
        self.ns, self.nt = len(src_key), len(tgt_key)
        self.srow, self.trow = srow, trow
        # site rows (edge -> (h, site) with coefficient frac * w)
        site_key = {}
        sk, sc, sfr = [], [], []
        for k in range(self.E):
            c = int(pool["e_tgt"][k])
            h = int(pool["e_hour"][k])
            for (pj, fr) in capinfo["serv"].get(c, ()):
                kk = (h, pj)
                if kk not in site_key:
                    site_key[kk] = len(site_key)
                sk.append(site_key[kk])
                sc.append(k)
                sfr.append(fr)
        self.site_key = site_key
        self.sk = np.array(sk, dtype=int)
        self.sc = np.array(sc, dtype=int)
        self.sfr = np.array(sfr, dtype=float)
        self.nsite = len(site_key)
        # constant bounds
        self.b_src = np.array([pool["arr"][h, x]
                               for (h, x) in src_key], dtype=float)
        self.b_tgt = np.array([pool["need"][c]
                               for (h, c) in tgt_key], dtype=float)
        self.b_site = (np.array([capinfo["cap"][h, pj]
                                 for (h, pj) in site_key], dtype=float)
                       if site_key else np.zeros(0))
        self.site_key_list = list(site_key.keys())
        self.src_key_list = list(src_key.keys())
        self.tgt_key_list = list(tgt_key.keys())
        self.edge_outside = np.array(
            [capinfo["outside"][int(c)] for c in pool["e_tgt"]], bool)
        self.idx = np.arange(self.E)

    def set_r(self, r):
        d = self.p["e_dist"]
        self.pi = t8.acceptance(r, d, *t8.PARAMS[0])
        self.w = self.pi - t8.acceptance(0.0, d, *t8.PARAMS[0])

    def solve(self, r, budget=None, cap_on=True, want_x=False):
        self.set_r(r)
        w = self.w
        nrow = self.ns + self.nt + (self.nsite if cap_on else 0)
        rows = [self.srow, self.trow + self.ns]
        cols = [self.idx, self.idx]
        data = [np.ones(self.E), w]
        b = [self.b_src, self.b_tgt]
        if cap_on:
            rows.append(self.sk + self.ns + self.nt)
            cols.append(self.sc)
            data.append(w[self.sc] * self.sfr)
            b.append(self.b_site)
        if budget is not None:
            rows.append(np.full(self.E, nrow))
            cols.append(self.idx)
            data.append(self.pi)
            b.append(np.array([float(budget) / float(r)]))
            nrow += 1
        A = sparse.csr_matrix(
            (np.concatenate(data),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(nrow, self.E))
        res = linprog(-w, A_ub=A, b_ub=np.concatenate(b),
                      bounds=(0, None), method="highs")
        if res.status != 0:
            raise RuntimeError(f"LP failed r={r} B={budget} "
                               f"status={res.status}")
        x = res.x
        q = float(w @ x)
        P = float(self.pi @ x)
        out = dict(r=float(r), q=q, P=P, invited=float(x.sum()),
                   payment=float(r * P), bill_old=float(r * q),
                   total=float(r * P + TRUCK_UNIT * (self.p["N"] - q)),
                   budget=budget)
        if want_x:
            out["x"] = x
        return out

    def outside_volume(self, x):
        m = self.edge_outside
        return float(self.w[m] @ x[m])


# --------------------------------------------------------------------------
# stage: reproduce the old ledger
# --------------------------------------------------------------------------
def run_repro():
    pool = load_pool()
    capinfo = build_capacity(pool)
    lp = PoolLP(pool, capinfo)
    rows = {}
    for cap_name, r in (("arrivals", 1.76), ("surplus", 1.83)):
        # no fence rows, no budget: exactly T008's hourly LP
        self_w = None
        lp.set_r(r)
        w = lp.w
        d = pool["e_dist"]
        # rebuild the source caps with the T008 rule
        caps = np.empty(lp.ns)
        for k, (h, s) in enumerate(lp.src_key_list):
            if cap_name == "arrivals":
                caps[k] = pool["arr"][h, s]
            else:
                caps[k] = max(0.0, pool["arr"][h, s] - pool["dep"][h, s])
        A = sparse.csr_matrix(
            (np.r_[np.ones(lp.E), w],
             (np.r_[lp.srow, lp.trow + lp.ns],
              np.r_[lp.idx, lp.idx])),
            shape=(lp.ns + lp.nt, lp.E))
        b = np.r_[caps, lp.b_tgt]
        res = linprog(-w, A_ub=A, b_ub=b, bounds=(0, None), method="highs")
        x = res.x
        q = float(w @ x)
        P = float(lp.pi @ x)
        rows[cap_name] = dict(
            r=r, q=q, P=P, invited=float(x.sum()), bill_new=float(r * P),
            bill_old=float(r * q),
            t008_q={"arrivals": 95565.815471, "surplus": 32777.112129}[
                cap_name],
            reviewer_P={"arrivals": 101286.09, "surplus": 34601.17}[
                cap_name],
            q_abs_diff=abs(q - {"arrivals": 95565.815471,
                                "surplus": 32777.112129}[cap_name]),
            P_abs_diff=abs(P - {"arrivals": 101286.09,
                                "surplus": 34601.17}[cap_name]))
        print(f"repro {cap_name}: q={q:.6f} P={P:.4f} invited={x.sum():.2f} "
              f"bill_new={r*P:.4f}")
    out = dict(rows=rows, N=pool["N"], truck_unit=TRUCK_UNIT,
               note=("no budget, no fence-capacity rows; identical LP to "
                     "T008's hourly formulation, stacked over the 24 hours"))
    dump(out, "repro.json")
    return out


# --------------------------------------------------------------------------
# stage: policy table
# --------------------------------------------------------------------------
def run_policy(force=False):
    path = os.path.join(RUN, "policy_table.json")
    if os.path.exists(path) and not force:
        print("reuse", path)
        return load_json("policy_table.json")
    pool = load_pool()
    capinfo = build_capacity(pool)
    lp = PoolLP(pool, capinfo)
    N = pool["N"]
    t0 = time.time()

    def scan(cap_on):
        unc = []
        for r in R_GRID:
            unc.append(lp.solve(r, budget=None, cap_on=cap_on))
        return unc

    out = dict(settings=dict(
        r_grid=[float(R_GRID[0]), float(R_GRID[-1]),
                float(R_GRID[1] - R_GRID[0])],
        budgets=BUDGETS, truck_unit=TRUCK_UNIT, N=N,
        level=LEVEL,
        payment_ledger="r * P, P = sum pi(r) x (pay on completion)",
        old_ledger="r * q (increment only), kept as bill_old",
        total_spend="payment + truck_unit * (N - q)",
        target_capacity=("sum_c (f[c,j]/a[c]) w x <= min_{15-min steps of h} "
                         "spare[j,h] for every open site j (T019)"),
        cap_source="data/runs/T019/spare_capacity.npz, "
                   "plan_meas343_main.npz"),
        levels={})

    for cap_name in ("arrivals", "surplus"):
        # set the caliber's source caps before scanning
        caps = np.empty(lp.ns)
        for k, (h, s) in enumerate(lp.src_key_list):
            if cap_name == "arrivals":
                caps[k] = pool["arr"][h, s]
            else:
                caps[k] = max(0.0, pool["arr"][h, s] - pool["dep"][h, s])
        lp.b_src = caps
        base = scan(True)
        unc_no = scan(False)       # reference without fence rows
        rows = []
        for B in BUDGETS:
            best = None
            for row_u in base:
                r = row_u["r"]
                if B is None or row_u["payment"] <= B + 1e-9:
                    cand = row_u
                else:
                    cand = lp.solve(r, budget=B, cap_on=True)
                if best is None or cand["total"] < best[1]["total"]:
                    best = (r, cand)
            r, cand = best
            rows.append(dict(
                budget=B, source_cap=cap_name, target_cap=True,
                r=r, q=cand["q"], P=cand["P"], invited=cand["invited"],
                payment=cand["payment"], bill_old=cand["bill_old"],
                truck_residual=float(TRUCK_UNIT * (N - cand["q"])),
                truck_cost_change=float(-TRUCK_UNIT * cand["q"]),
                total_spend=cand["total"],
                saving_vs_all_truck=float(
                    1.0 - cand["total"] / (TRUCK_UNIT * N)),
                q_outside_sites=None,   # filled in run_check
            ))
            print(f"{cap_name} B={B}: r={r:.2f} q={cand['q']:.1f} "
                  f"P={cand['P']:.1f} pay={cand['payment']:.1f} "
                  f"total={cand['total']:.1f}")
        # fill the outside-fence split for every row (target cell served by
        # no T019 site)
        for row in rows:
            got = lp.solve(row["r"], budget=row["budget"], cap_on=True,
                           want_x=True)
            row["q_outside_sites"] = lp.outside_volume(got["x"])
            row["q_inside_sites"] = float(got["q"]
                                          - row["q_outside_sites"])
        # no-fence reference at its own optimum, no budget
        best_no = min(unc_no, key=lambda z: z["total"])
        out["levels"][cap_name] = dict(
            rows=rows,
            no_target_cap_reference=dict(
                r=best_no["r"], q=best_no["q"], P=best_no["P"],
                payment=best_no["payment"],
                total_spend=best_no["total"],
                saving_vs_all_truck=float(
                    1.0 - best_no["total"] / (TRUCK_UNIT * N)),
                capacity_deduction_q=float(best_no["q"]
                                           - next(x["q"] for x in base
                                                  if x["r"] == best_no["r"])),
            ),
            all_truck_cost=float(TRUCK_UNIT * N))
    out["runtime_s"] = float(time.time() - t0)
    dump(out, "policy_table.json")
    return out


# --------------------------------------------------------------------------
# stage: hard-budget execution rule
# --------------------------------------------------------------------------
def run_hardbudget(force=False):
    path = os.path.join(RUN, "hard_budget.json")
    if os.path.exists(path) and not force:
        print("reuse", path)
        return load_json("hard_budget.json")
    pool = load_pool()
    capinfo = build_capacity(pool)
    lp = PoolLP(pool, capinfo)
    pol = load_json("policy_table.json")
    rng = np.random.default_rng(SIM_SEED)
    out = dict(seed=SIM_SEED, sim_days=SIM_DAYS,
               rule=("reserve r per invitation (worst case: every invitee "
                     "completes); issue hours in order until the cumulative "
                     "reserve reaches B, then stop"),
               rows=[])
    for cap_name in ("arrivals", "surplus"):
        caps = np.empty(lp.ns)
        for k, (h, s) in enumerate(lp.src_key_list):
            if cap_name == "arrivals":
                caps[k] = pool["arr"][h, s]
            else:
                caps[k] = max(0.0, pool["arr"][h, s] - pool["dep"][h, s])
        lp.b_src = caps
        for row in pol["levels"][cap_name]["rows"]:
            B = row["budget"]
            r = row["r"]
            got = lp.solve(r, budget=B, cap_on=True, want_x=True)
            x = got["x"]
            Ih = np.bincount(pool["e_hour"], weights=x, minlength=24)
            Ph = np.bincount(pool["e_hour"], weights=lp.pi * x,
                             minlength=24)
            reserve_h = r * Ih
            cum = np.cumsum(reserve_h)
            info = dict(budget=B, source_cap=cap_name, r=r,
                        reserve_total=float(reserve_h.sum()),
                        expected_payment=float(got["payment"]))
            if B is None or reserve_h.sum() <= B + 1e-9:
                info.update(stop_hour=None, issued_fraction=1.0,
                            truncated_q=float(got["q"]),
                            truncated_payment=float(got["payment"]))
                issued = Ih.copy()
            else:
                hstar = int(np.argmax(cum > B))
                prev = float(cum[hstar - 1]) if hstar > 0 else 0.0
                frac = (B - prev) / max(reserve_h[hstar], 1e-12)
                issued = Ih.copy()
                issued[hstar] *= frac
                issued[hstar + 1:] = 0.0
                q_tr = float((lp.w * np.where(
                    pool["e_hour"] < hstar, 1.0,
                    np.where(pool["e_hour"] == hstar, frac, 0.0))) @ x)
                p_tr = float((lp.pi * np.where(
                    pool["e_hour"] < hstar, 1.0,
                    np.where(pool["e_hour"] == hstar, frac, 0.0))) @ x)
                info.update(stop_hour=hstar, issued_fraction=float(frac),
                            truncated_q=q_tr,
                            truncated_payment=float(r * p_tr))
            # simulate realized acceptances at the hour level
            n_int = np.round(issued).astype(int)
            pbar = np.divide(Ph, Ih, out=np.zeros(24), where=Ih > 0)
            pay = np.zeros(SIM_DAYS)
            for h in range(24):
                if n_int[h] <= 0:
                    continue
                K = rng.binomial(n_int[h], np.clip(pbar[h], 0, 1),
                                 size=SIM_DAYS)
                pay += r * K
            info["sim_max_payment"] = float(pay.max())
            info["sim_mean_payment"] = float(pay.mean())
            info["overrun_days"] = int((pay > (B if B is not None
                                               else np.inf) + 1e-6).sum())
            info["overrun_probability"] = float(info["overrun_days"]
                                                / SIM_DAYS)
            out["rows"].append(info)
            print(f"hard {cap_name} B={B}: stop={info['stop_hour']} "
                  f"sim_max={pay.max():.1f} overrun={info['overrun_days']}")
    out["all_zero_overrun"] = all(x["overrun_days"] == 0
                                  for x in out["rows"])
    dump(out, "hard_budget.json")
    return out


# --------------------------------------------------------------------------
# stage: feed the OD change into the T018 integer scheduler
# --------------------------------------------------------------------------
def ipf_adjust(PZ_t, dep_t, dA, iters=3000, tol=1e-9):
    od = PZ_t * dep_t[:, None]
    A = od.sum(0)
    target = np.maximum(A + dA, 0.0)
    if target.sum() > 0 and dep_t.sum() > 0:
        target *= dep_t.sum() / target.sum()
    for it in range(iters):
        cs = od.sum(0)
        od *= np.divide(target, np.maximum(cs, 1e-12))[None, :]
        rs = od.sum(1)
        od *= np.divide(dep_t, np.maximum(rs, 1e-12))[:, None]
        if (np.abs(od.sum(0) - target).max() < tol
                and np.abs(od.sum(1) - dep_t).max() < tol):
            break
    row_err = float(np.abs(od.sum(1) - dep_t).max())
    col_err = float(np.abs(od.sum(0) - target).max())
    return od / np.maximum(dep_t, 1e-12)[:, None], row_err, col_err


def zone_delta(x, w, pool):
    dA = np.zeros((24, pool["n_zones"]))
    zw = w * x
    np.add.at(dA, (pool["e_hour"], pool["zid"][pool["e_tgt"]]), zw)
    np.add.at(dA, (pool["e_hour"], pool["zid"][pool["e_src"]]), -zw)
    return dA


def run_hybrid(force=False):
    path = os.path.join(RUN, "hybrid.json")
    if os.path.exists(path) and not force:
        print("reuse", path)
        return load_json("hybrid.json")
    import t018_q2_int as t18
    pool = load_pool()
    capinfo = build_capacity(pool)
    lp = PoolLP(pool, capinfo)
    pol = load_json("policy_table.json")
    t18.setup_design()
    DEP, PZ0, NET, I0 = t18.DEP, t18.PZ, t18.NET, t18.I0
    pure = t18.load_solution("int_K60")
    pure_obj = float(pure["obj"])
    out = dict(pure_truck=dict(
                   obj=pure_obj, unmet=float(pure["unmet"]),
                   dev=float(pure["dev"]), truck_km=float(pure["truck_km"]),
                   moved=float(pure["moved"]), K=int(pure["K"])),
               note=("hybrid = T018 integer MILP on the same DEP/I0/K/costs "
                     "with PZ re-fitted by IPF to the incentive's zone-hour "
                     "arrival change (rows=departures fixed, columns=new "
                     "arrival totals); payment added to the truck objective"),
               policies=[])
    for cap_name, B in (("surplus", None), ("arrivals", None),
                        ("surplus", 60000.0)):
        caps = np.empty(lp.ns)
        for k, (h, s) in enumerate(lp.src_key_list):
            if cap_name == "arrivals":
                caps[k] = pool["arr"][h, s]
            else:
                caps[k] = max(0.0, pool["arr"][h, s] - pool["dep"][h, s])
        lp.b_src = caps
        row = next(z for z in pol["levels"][cap_name]["rows"]
                   if z["budget"] == B)
        got = lp.solve(row["r"], budget=B, cap_on=True, want_x=True)
        dA = zone_delta(got["x"], lp.w, pool)
        PZ2 = np.empty_like(PZ0)
        errs = []
        for t in range(24):
            PZ2[t], re_, ce_ = ipf_adjust(PZ0[t], DEP[:, t], dA[t])
            errs.append((re_, ce_))
        res = t18.solve(DEP, t18.ARR, PZ2, NET, I0, K=60, integer=True,
                        time_limit=600, mip_gap=0.01, event_order=True,
                        bike_integer=True)
        if res is None:
            out["policies"].append(dict(source_cap=cap_name, budget=B,
                                        status="solve_failed"))
            continue
        payment = float(got["payment"])
        total = float(res["obj"]) + payment
        out["policies"].append(dict(
            source_cap=cap_name, budget=B, r=row["r"],
            q=float(got["q"]), P=float(got["P"]), payment=payment,
            delta_zone_hour_abs_sum=float(np.abs(dA).sum()),
            delta_positive=float(np.maximum(dA, 0).sum()),
            ipf_row_err_max=float(max(e[0] for e in errs)),
            ipf_col_err_max=float(max(e[1] for e in errs)),
            truck_obj=float(res["obj"]), truck_unmet=float(res["unmet"]),
            truck_dev=float(res["dev"]), truck_km=float(res["truck_km"]),
            truck_moved=float(res["moved"]),
            total=total, saving_vs_pure=float(pure_obj - total),
            saving_pct=float(100.0 * (pure_obj - total) / pure_obj)))
        print(f"hybrid {cap_name} B={B}: obj={res['obj']:.2f} "
              f"pay={payment:.2f} total={total:.2f} "
              f"(pure {pure_obj:.2f})")
    out["old_cell_level"] = dict(
        targeted_upper_pct=20.83, conservative_pct=6.83,
        note=("T008 q-based, cell-level equivalent-cost model; not a system "
              "saving"))
    dump(out, "hybrid.json")
    return out


# --------------------------------------------------------------------------
# stage: checks
# --------------------------------------------------------------------------
def run_check(force=False):
    path = os.path.join(RUN, "check.json")
    if os.path.exists(path) and not force:
        print("reuse", path)
        return load_json("check.json")
    pool = load_pool()
    capinfo = build_capacity(pool)
    lp = PoolLP(pool, capinfo)
    repro = load_json("repro.json")
    pol = load_json("policy_table.json")
    hard = load_json("hard_budget.json")
    out = dict(
        repro_q_ok=bool(all(v["q_abs_diff"] <= 1e-6
                            for v in repro["rows"].values())),
        repro_P_ok=bool(all(v["P_abs_diff"] <= 0.01
                            for v in repro["rows"].values())),
        budgets_respected={}, capacity={}, outside={})

    for cap_name in ("arrivals", "surplus"):
        caps = np.empty(lp.ns)
        for k, (h, s) in enumerate(lp.src_key_list):
            if cap_name == "arrivals":
                caps[k] = pool["arr"][h, s]
            else:
                caps[k] = max(0.0, pool["arr"][h, s] - pool["dep"][h, s])
        lp.b_src = caps
        for row in pol["levels"][cap_name]["rows"]:
            key = f"{cap_name}_B{row['budget']}"
            out["budgets_respected"][key] = dict(
                B=row["budget"], payment=row["payment"],
                ok=bool(row["budget"] is None
                        or row["payment"] <= row["budget"] + 1.0))
            got = lp.solve(row["r"], budget=row["budget"], cap_on=True,
                           want_x=True)
            x = got["x"]
            # site headroom: recompute the loading on every binding row
            load = np.zeros((24, len(capinfo["sites"])))
            pos = {int(j): k for k, j in enumerate(capinfo["sites"])}
            for k in range(lp.E):
                if x[k] <= 0:
                    continue
                c = int(pool["e_tgt"][k])
                h = int(pool["e_hour"][k])
                for (pj, fr) in capinfo["serv"].get(c, ()):
                    load[h, pj] += fr * lp.w[k] * x[k]
            over = load - capinfo["cap"]
            worst = float(over.max())
            out["capacity"][key] = dict(
                max_site_over_capacity=worst,
                ok=bool(worst <= 1e-6),
                n_binding_sites=int((np.abs(over) <= 1e-6).sum()))
            out["outside"][key] = dict(
                q_outside_sites=lp.outside_volume(x),
                q_total=float(got["q"]),
                share=float(lp.outside_volume(x) / max(got["q"], 1e-12)),
                n_outside_cells=int(capinfo["outside"].sum()))
    out["hard_budget_zero_overrun"] = bool(hard["all_zero_overrun"])
    out["all_pass"] = bool(
        out["repro_q_ok"] and out["repro_P_ok"]
        and all(v["ok"] for v in out["budgets_respected"].values())
        and all(v["ok"] for v in out["capacity"].values())
        and out["hard_budget_zero_overrun"])
    dump(out, "check.json")
    return out


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=("repro", "policy", "hardbudget", "hybrid",
                             "check", "all"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.stage in ("repro", "all"):
        run_repro()
    if args.stage in ("policy", "all"):
        run_policy(force=args.force)
    if args.stage in ("hardbudget", "all"):
        run_hardbudget(force=args.force)
    if args.stage in ("hybrid", "all"):
        run_hybrid(force=args.force)
    if args.stage in ("check", "all"):
        run_check(force=args.force)
