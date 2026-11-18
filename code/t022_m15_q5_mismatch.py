"""T022-M15: Q5 two-platform spatial mismatch, independent vs shared fences.

The existing Q5 experiment splits fleet and demand proportionally between
platforms and finds zero unserved demand, so it cannot discriminate between
competition regimes.  This module adds the missing margin: each platform's
riders are displaced east or west of its own fleet (platform A's users are
eastern, platform B's western), while total fleet and total demand are held
fixed.  Each platform keeps half of the observed initial inventory at every
cell, so the mismatch is between where a platform's riders are and where its
bicycles are deployed.  Under that mismatch we compare

  * independent operation, each platform with its own 150-slot fence per site;
  * independent operation at equal total capacity (75 slots per platform);
  * shared fences, one 150-slot fence per site that both platforms park in.

Riders are single-homing throughout: they take only their own platform's
bicycles, even from a shared fence.  Sharing is therefore parking capacity,
not bicycles, exactly the policy in the paper's Q5 section.

The fence layout, 150-slot capacity and 450 m assignment are T019's capacitated
Q4 plan for the measured 343,638 inventory (training-day demand, M11 caliber).
The replay is 15-minute, T019's replay clock.

Outputs (data/runs/T022/):
  m15_mismatch.json   metrics table, controls, required capacity, mechanism
  m15_capacity.npz    per-site required capacity arrays
  m15_check.json      consistency checks

Run:  python -I code/t022_m15_q5_mismatch.py --stage all --force
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
from scipy.sparse import csr_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RUN = os.path.join(DATA, "runs", "T022")
T019 = os.path.join(DATA, "runs", "T019")
CAP = 150.0                 # slots per fence (T019)
STEP_MIN = 15               # replay clock (T019)
SEEDS = list(range(20))     # task asks >= 6
MISMATCH = [0.0, 0.5, 1.0, 1.5]   # exp(+/- m * x_tilde) fleet skew
SHARE = 0.5                 # each platform's share of fleet and demand
# (name, cap_A, cap_B, pooled)
REGIMES = [
    ("independent", 150.0, 150.0, False),
    ("independent_equal", 75.0, 75.0, False),
    ("shared", 150.0, 150.0, True),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(obj, name):
    path = os.path.join(RUN, name)
    os.makedirs(RUN, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    print("wrote", path)


def load_json(name):
    with open(os.path.join(RUN, name), encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def demand_shares(dep_cell, xt, m):
    """Platform A's share of each origin cell's trips.

    share_A(i) = sigmoid(a + 2*m*x_tilde(i)) with a bisected so that A still
    carries half of all trips; B carries the complement.  At m=0 the share is
    0.5 everywhere.
    """
    w = dep_cell / dep_cell.sum()
    target = 0.5

    def f(a):
        return float((w * _sigmoid(a + 2.0 * m * xt)).sum())

    lo, hi = -30.0, 30.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    a = 0.5 * (lo + hi)
    sA = _sigmoid(a + 2.0 * m * xt)
    return sA, a


def prep():
    """Site-level flows, initial occupancy and the mismatch weights."""
    plan = np.load(os.path.join(T019, "plan_meas343_main.npz"))
    z = np.load(os.path.join(T019, "cell_demand.npz"))
    xy = z["xy"]
    dep = z["dep"]                      # (24, C) training-day departures
    arr = z["arr"]                      # (24, C) training-day arrivals
    C = dep.shape[1]
    s, f, a, ii, jj, I0 = (plan["s"], plan["f"], plan["a"],
                           plan["ii"], plan["jj"], plan["I0"])
    frac = np.divide(f, a[ii], out=np.zeros_like(f), where=a[ii] > 0)
    W = csr_matrix((frac, (ii, jj)), shape=(C, C))
    dep15 = np.repeat(dep / 4.0, 4, axis=0)
    arr15 = np.repeat(arr / 4.0, 4, axis=0)
    dep_site = np.ascontiguousarray(W.T.dot(dep15.T).T)
    arr_site = np.ascontiguousarray(W.T.dot(arr15.T).T)
    I0_site = np.asarray(W.T.dot(I0)).ravel()
    x = xy[:, 0]
    xt = 2.0 * (x - x.min()) / max(x.max() - x.min(), 1e-12) - 1.0
    dep_tot = dep.sum(0)
    fleets = {m: (SHARE * I0, SHARE * I0) for m in MISMATCH}
    shares = {m: demand_shares(dep_tot, xt, m) for m in MISMATCH}
    return dict(plan=plan, z=z, C=C, W=W, s=s, dep_site=dep_site,
                arr_site=arr_site, dep15=dep15, arr15=arr15,
                I0_site=I0_site, I0=I0, xt=xt,
                fleets=fleets, shares=shares, xy=xy,
                open_sites=int((s > 0.5).sum()),
                plan_path=os.path.join(T019, "plan_meas343_main.npz"),
                demand_path=os.path.join(T019, "cell_demand.npz"))


def occ_from_cells(I0_cell, W):
    """Initial site occupancy: sum_i I0[i] * frac(i, j)."""
    return np.asarray(W.T.dot(I0_cell)).ravel()


def draw_flows(dep_rate, arr_rate, seed):
    """Poisson draws at the given site-level rates (already platform-scaled)."""
    rng = np.random.default_rng(seed)
    d = rng.poisson(dep_rate)
    a = rng.poisson(arr_rate)
    return d, a


def simulate(depA, arrA, occA0, depB, arrB, occB0, capA, capB, pooled,
             uncapped=False, record_peak=False):
    """One day at 15 minutes.  Flows are pre-drawn so every regime shares the
    same randomness."""
    occA = occA0.astype(float).copy()
    occB = occB0.astype(float).copy()
    depA = depA.astype(float)
    depB = depB.astype(float)
    arrA = arrA.astype(float)
    arrB = arrB.astype(float)
    unmetA = unmetB = 0.0
    overA = overB = 0.0
    dep_drawnA = dep_drawnB = 0.0
    arr_drawnA = arr_drawnB = 0.0
    unmetA_site = np.zeros_like(occA)
    unmetB_site = np.zeros_like(occB)
    overA_site = np.zeros_like(occA)
    overB_site = np.zeros_like(occB)
    if record_peak:
        peakA = occA.copy()
        peakB = occB.copy()
        peakS = occA + occB
    # initial capacity handling
    if not uncapped:
        if not pooled:
            oa = np.maximum(0.0, occA - capA)
            ob = np.maximum(0.0, occB - capB)
            overA += float(oa.sum())
            overB += float(ob.sum())
            overA_site += oa
            overB_site += ob
            occA = np.minimum(occA, capA)
            occB = np.minimum(occB, capB)
        else:
            tot = occA + occB
            scale = np.divide(capA, tot, out=np.ones_like(tot),
                              where=tot > 0)
            scale = np.minimum(scale, 1.0)
            oa = occA * (1.0 - scale)
            ob = occB * (1.0 - scale)
            overA += float(oa.sum())
            overB += float(ob.sum())
            overA_site += oa
            overB_site += ob
            occA = occA * scale
            occB = occB * scale
    T = depA.shape[0]
    for t in range(T):
        dA, dB = depA[t], depB[t]
        aA, aB = arrA[t], arrB[t]
        takeA = np.minimum(dA, occA)
        takeB = np.minimum(dB, occB)
        ua = dA - takeA
        ub = dB - takeB
        unmetA += float(ua.sum())
        unmetB += float(ub.sum())
        unmetA_site += ua
        unmetB_site += ub
        dep_drawnA += float(dA.sum())
        dep_drawnB += float(dB.sum())
        occA -= takeA
        occB -= takeB
        arr_drawnA += float(aA.sum())
        arr_drawnB += float(aB.sum())
        if uncapped:
            occA += aA
            occB += aB
        elif not pooled:
            nA = occA + aA
            nB = occB + aB
            oa = np.maximum(0.0, nA - capA)
            ob = np.maximum(0.0, nB - capB)
            overA += float(oa.sum())
            overB += float(ob.sum())
            overA_site += oa
            overB_site += ob
            occA = np.minimum(nA, capA)
            occB = np.minimum(nB, capB)
        else:
            tot = occA + occB
            inc = aA + aB
            room = np.maximum(0.0, capA - tot)
            scale = np.divide(room, inc, out=np.ones_like(inc), where=inc > 0)
            scale = np.minimum(scale, 1.0)
            oa = aA * (1.0 - scale)
            ob = aB * (1.0 - scale)
            overA += float(oa.sum())
            overB += float(ob.sum())
            overA_site += oa
            overB_site += ob
            occA = occA + aA * scale
            occB = occB + aB * scale
        if record_peak:
            peakA = np.maximum(peakA, occA)
            peakB = np.maximum(peakB, occB)
            peakS = np.maximum(peakS, occA + occB)
    out = dict(unmet=unmetA + unmetB, unmetA=unmetA, unmetB=unmetB,
               overflow=overA + overB, overflowA=overA, overflowB=overB,
               dep=dep_drawnA + dep_drawnB, depA=dep_drawnA, depB=dep_drawnB,
               arr=arr_drawnA + arr_drawnB, arrA=arr_drawnA, arrB=arr_drawnB,
               unmetA_site=unmetA_site, unmetB_site=unmetB_site,
               overA_site=overA_site, overB_site=overB_site)
    if record_peak:
        out.update(peakA=peakA, peakB=peakB, peakS=peakS)
    return out


def run_sim(force=False):
    path = os.path.join(RUN, "m15_mismatch.json")
    cap_path = os.path.join(RUN, "m15_capacity.npz")
    if os.path.exists(path) and os.path.exists(cap_path) and not force:
        print("reuse", path)
        return load_json("m15_mismatch.json")
    t0 = time.time()
    P = prep()
    dep_site, arr_site = P["dep_site"], P["arr_site"]
    W = P["W"]
    east = P["xy"][:, 0] > np.median(P["xy"][:, 0])
    rows = []
    per_platform = {}
    regional = {}
    cap_arrays = {}

    # ---- single-platform reference (the same fences, one operator) ------
    ref = {k: [] for k in ("unmet", "overflow", "dep", "arr")}
    ref_peaks = []
    for seed in SEEDS:
        d, a = draw_flows(dep_site, arr_site, seed + 1000)
        occ0 = P["I0_site"].copy()
        r = simulate(d, a, occ0, np.zeros_like(d), np.zeros_like(a),
                     np.zeros_like(occ0), CAP, CAP, False,
                     record_peak=True)
        for k in ref:
            ref[k].append(r[k])
        ref_peaks.append(r["peakA"])
    ref_peaks = np.array(ref_peaks)
    ur = np.array([u / max(d, 1e-9) for u, d in zip(ref["unmet"], ref["dep"])])
    fr = np.array([o / max(a, 1e-9) for o, a in zip(ref["overflow"], ref["arr"])])
    ref_rec = dict(
        unmet_rate=float(ur.mean()), unmet_rate_sd=float(ur.std()),
        full_rate=float(fr.mean()), full_rate_sd=float(fr.std()),
        departures=float(np.mean(ref["dep"])),
        arrivals=float(np.mean(ref["arr"])),
        unmet_departures=float(np.mean(ref["unmet"])),
        overflow_arrivals=float(np.mean(ref["overflow"])),
        required_capacity_slots=float(
            np.percentile(ref_peaks, 95, axis=0).sum()))
    print(f"single platform: unmet {100*ref_rec['unmet_rate']:.2f}% "
          f"full {100*ref_rec['full_rate']:.2f}% "
          f"cap {ref_rec['required_capacity_slots']:.0f} "
          f"({time.time()-t0:.0f}s)")

    # ---- mismatch x regime --------------------------------------------
    for m in MISMATCH:
        IA, IB = P["fleets"][m]
        sA, shift_a = P["shares"][m]
        depA_site = np.ascontiguousarray(
            W.T.dot((P["dep15"] * sA[None, :]).T).T)
        depB_site = np.ascontiguousarray(
            W.T.dot((P["dep15"] * (1.0 - sA)[None, :]).T).T)
        arrA_site = arr_site * SHARE
        arrB_site = arr_site * SHARE
        occA0 = occ_from_cells(IA, W)
        occB0 = occ_from_cells(IB, W)
        # uncapped runs: required capacity, shared across regimes
        peaksA, peaksB, peaksS = [], [], []
        for seed in SEEDS:
            dA, aA = draw_flows(depA_site, arrA_site, seed + 1)
            dB, aB = draw_flows(depB_site, arrB_site, seed + 5001)
            r = simulate(dA, aA, occA0, dB, aB, occB0, CAP, CAP, False,
                         uncapped=True, record_peak=True)
            peaksA.append(r["peakA"])
            peaksB.append(r["peakB"])
            peaksS.append(r["peakS"])
        peaksA = np.array(peaksA)
        peaksB = np.array(peaksB)
        peaksS = np.array(peaksS)
        qA = np.percentile(peaksA, 95, axis=0)
        qB = np.percentile(peaksB, 95, axis=0)
        qS = np.percentile(peaksS, 95, axis=0)
        req_ind = qA + qB
        req_sh = qS
        cap_arrays[f"m{m}_q95peakA"] = qA.astype(np.float32)
        cap_arrays[f"m{m}_q95peakB"] = qB.astype(np.float32)
        cap_arrays[f"m{m}_q95peakS"] = qS.astype(np.float32)
        cap_arrays[f"m{m}_req_ind"] = req_ind.astype(np.float32)
        cap_arrays[f"m{m}_req_shared"] = req_sh.astype(np.float32)
        cap_arrays[f"m{m}_I0A_site"] = occA0.astype(np.float32)
        cap_arrays[f"m{m}_I0B_site"] = occB0.astype(np.float32)
        nz_ind = req_ind[req_ind > 0]
        nz_sh = req_sh[req_sh > 0]
        print(f"m={m:.1f} capacity: ind {req_ind.sum():.0f} "
              f"shared {req_sh.sum():.0f} "
              f"ratio {req_ind.sum()/max(req_sh.sum(),1e-9):.3f} "
              f"({time.time()-t0:.0f}s)")

        for name, capA, capB, pooled in REGIMES:
            rec = {k: [] for k in ("unmet", "overflow", "dep", "arr",
                                   "unmetA", "unmetB", "overflowA",
                                   "overflowB", "depA", "depB",
                                   "arrA", "arrB")}
            ua_site = np.zeros(P["C"])
            ub_site = np.zeros(P["C"])
            da_site = np.zeros(P["C"])
            db_site = np.zeros(P["C"])
            for seed in SEEDS:
                dA, aA = draw_flows(depA_site, arrA_site, seed + 1)
                dB, aB = draw_flows(depB_site, arrB_site, seed + 5001)
                r = simulate(dA, aA, occA0, dB, aB, occB0, capA, capB,
                             pooled)
                for k in rec:
                    rec[k].append(r[k])
                ua_site += r["unmetA_site"]
                ub_site += r["unmetB_site"]
                da_site += dA.sum(0)
                db_site += dB.sum(0)
            ur = np.array([u / max(d, 1e-9) for u, d in
                           zip(rec["unmet"], rec["dep"])])
            fr = np.array([o / max(a, 1e-9) for o, a in
                           zip(rec["overflow"], rec["arr"])])
            uA = np.array([u / max(d, 1e-9) for u, d in
                           zip(rec["unmetA"], rec["depA"])])
            uB = np.array([u / max(d, 1e-9) for u, d in
                           zip(rec["unmetB"], rec["depB"])])
            fA = np.array([o / max(a, 1e-9) for o, a in
                           zip(rec["overflowA"], rec["arrA"])])
            fB = np.array([o / max(a, 1e-9) for o, a in
                           zip(rec["overflowB"], rec["arrB"])])
            rows.append(dict(
                mismatch=m, regime=name,
                unmet_rate=float(ur.mean()), unmet_rate_sd=float(ur.std()),
                full_rate=float(fr.mean()), full_rate_sd=float(fr.std()),
                unmet_A=float(uA.mean()), unmet_B=float(uB.mean()),
                full_A=float(fA.mean()), full_B=float(fB.mean()),
                departures=float(np.mean(rec["dep"])),
                arrivals=float(np.mean(rec["arr"])),
                unmet_departures=float(np.mean(rec["unmet"])),
                overflow_arrivals=float(np.mean(rec["overflow"])),
                required_capacity_slots=float((req_ind if not pooled
                                               else req_sh).sum()),
                req_median_site=float(np.median(nz_ind if not pooled
                                                else nz_sh))
                if (nz_ind.size if not pooled else nz_sh.size) else 0.0,
                req_p95_site=float(np.percentile(nz_ind if not pooled
                                                 else nz_sh, 95))
                if (nz_ind.size if not pooled else nz_sh.size) else 0.0,
                req_max_site=float((req_ind if not pooled
                                    else req_sh).max()),
                sites_with_required_capacity=int(
                    ((req_ind if not pooled else req_sh) > 0).sum())))
            print(f"m={m:.1f} {name:18s}: unmet {100*ur.mean():6.2f}% "
                  f"full {100*fr.mean():6.2f}% "
                  f"({time.time()-t0:.0f}s)")
            per_platform[f"m{m}_{name}"] = dict(
                unmet_A=float(uA.mean()), unmet_A_sd=float(uA.std()),
                unmet_B=float(uB.mean()), unmet_B_sd=float(uB.std()),
                full_A=float(fA.mean()), full_B=float(fB.mean()))
            regional[f"m{m}_{name}"] = {}
            for reg in ("east", "west"):
                sel = east if reg == "east" else ~east
                da = float(da_site[sel].sum())
                db = float(db_site[sel].sum())
                regional[f"m{m}_{name}"][reg] = dict(
                    depA=da / len(SEEDS), depB=db / len(SEEDS),
                    unmetA=float(ua_site[sel].sum()) / len(SEEDS),
                    unmetB=float(ub_site[sel].sum()) / len(SEEDS),
                    rateA=float(ua_site[sel].sum() / max(da, 1e-9)),
                    rateB=float(ub_site[sel].sum() / max(db, 1e-9)))

    ctrl = None
    q5 = os.path.join(DATA, "q5_summary.json")
    if os.path.exists(q5):
        with open(q5, encoding="utf-8") as f:
            ctrl = json.load(f)
    out = dict(
        task="T022-M15 Q5 spatial mismatch",
        date="2026-11-18",
        revision="R0",
        settings=dict(
            plan="T019 plan_meas343_main (4,000 sites, 150 slots, 450 m)",
            level="meas343", cap=CAP, step_minutes=STEP_MIN,
            seeds=SEEDS, n_seeds=len(SEEDS),
            mismatch_levels=MISMATCH, share=SHARE,
            regimes=[r[0] for r in REGIMES],
            skew_rule=("platform A's share of each origin cell's trips is "
                       "sigmoid(a + 2*m*x_tilde(i)), platform B takes the "
                       "complement; a is bisected so each platform still "
                       "carries half of all trips; x_tilde is the east-west "
                       "cell coordinate scaled to [-1,1].  Both platforms "
                       "keep half of the observed initial inventory at "
                       "every cell, so the mismatch is between where a "
                       "platform's riders are and where its fleet sits.  A's "
                       "destinations follow the citywide arrival surface at "
                       "half rate (origins-only skew)"),
            platform_rule=("single-homing: each platform's flows are "
                           "independent Poisson draws at half the site "
                           "arrivals/departures"),
            independent_rule=("each platform owns a 150-slot fence at each "
                              "open site; only own bikes park there"),
            independent_equal_rule=("same, but 75 slots per platform, so the "
                                    "concrete per site equals the shared "
                                    "regime's 150"),
            shared_rule=("one 150-slot fence per site; both platforms park "
                         "in it, riders still take only their own bikes; "
                         "overflow is rationed in proportion to that "
                         "step's arrivals"),
            demand_source="T019 training-day Q1 surface (M11)"),
        single_platform=ref_rec,
        control_proportional_zone=ctrl,
        rows=rows,
        per_platform=per_platform,
        regional=regional,
        note=("The control is the old zone-level proportional split (zero "
              "unmet in 24 runs).  The m=0 row is the same proportional split "
              "at cell/fence resolution and is the model-consistent control. "
              "Required capacity is the sum over sites of the 95th percentile "
              "over seeds of peak occupancy; independent = q95(A)+q95(B), "
              "shared = q95(A+B), so the ratio is the capacity duplication "
              "from separate fences.  Shared fences pool parking slots only, "
              "not bicycles."),
        runtime_s=float(time.time() - t0),
        inputs=dict(plan_sha256=sha256(P["plan_path"]),
                    demand_sha256=sha256(P["demand_path"])))
    dump(out, "m15_mismatch.json")
    np.savez(os.path.join(RUN, "m15_capacity.npz"), **cap_arrays)
    print("wrote", os.path.join(RUN, "m15_capacity.npz"))
    return out


def run_check(force=False):
    path = os.path.join(RUN, "m15_check.json")
    if os.path.exists(path) and not force:
        print("reuse", path)
        return load_json("m15_check.json")
    P = prep()
    out = run_sim()
    checks = {}
    d = P["dep_site"].sum()
    a = P["arr_site"].sum()
    m = 1.0
    sA, shift_a = P["shares"][m]
    depA = np.ascontiguousarray(P["W"].T.dot(
        (P["dep15"] * sA[None, :]).T).T)
    depB = np.ascontiguousarray(P["W"].T.dot(
        (P["dep15"] * (1.0 - sA)[None, :]).T).T)
    dA, aA = draw_flows(depA, P["arr_site"] * SHARE, 1)
    dB, aB = draw_flows(depB, P["arr_site"] * SHARE, 5001)
    checks["flow_conservation"] = dict(
        total_dep_site=float(d), total_arr_site=float(a),
        seed1_dep_A=float(dA.sum()), seed1_dep_B=float(dB.sum()),
        dep_ratio=float((dA.sum() + dB.sum()) / max(d, 1e-9)),
        arr_ratio=float((aA.sum() + aB.sum()) / max(a, 1e-9)),
        share_sum_max_err=float(np.abs(sA + (1.0 - sA) - 1.0).max()),
        shift_a=float(shift_a))
    IA, IB = P["fleets"][0.0]
    sA0, _ = P["shares"][0.0]
    checks["m0_fleet_split"] = dict(
        sum_A=float(IA.sum()), sum_B=float(IB.sum()),
        max_abs_A_vs_half=float(np.abs(IA - 0.5 * P["I0"]).max()),
        max_abs_B_vs_half=float(np.abs(IB - 0.5 * P["I0"]).max()),
        max_abs_share_vs_half=float(np.abs(sA0 - 0.5).max()),
        target=0.5 * float(P["I0"].sum()))
    checks["rows"] = dict(
        n=len(out["rows"]),
        expected=len(MISMATCH) * len(REGIMES),
        mismatches=sorted({r["mismatch"] for r in out["rows"]}),
        regimes=sorted({r["regime"] for r in out["rows"]}))
    checks["rates_in_range"] = bool(all(
        0.0 <= r["unmet_rate"] <= 1.0 and 0.0 <= r["full_rate"] <= 1.0
        for r in out["rows"]))
    by = {(r["mismatch"], r["regime"]): r for r in out["rows"]}
    ratios = {}
    for m in MISMATCH:
        ri = by[(m, "independent")]["required_capacity_slots"]
        rs = by[(m, "shared")]["required_capacity_slots"]
        ratios[str(m)] = float(ri / max(rs, 1e-9))
    checks["capacity_ratio_ind_over_shared"] = ratios
    checks["capacity_pooling_ok"] = bool(all(v >= 1.0 - 1e-9
                                             for v in ratios.values()))
    sm = by[(0.0, "shared")]["required_capacity_slots"]
    checks["m0_shared_vs_single"] = dict(
        shared=sm, single=out["single_platform"]["required_capacity_slots"],
        ratio=float(sm / max(out["single_platform"]["required_capacity_slots"],
                             1e-9)))
    checks["all_pass"] = bool(
        abs(checks["flow_conservation"]["dep_ratio"] - 1.0) < 5e-3
        and abs(checks["flow_conservation"]["arr_ratio"] - 1.0) < 5e-3
        and checks["flow_conservation"]["share_sum_max_err"] < 1e-12
        and checks["m0_fleet_split"]["max_abs_A_vs_half"] < 1e-9
        and checks["m0_fleet_split"]["max_abs_B_vs_half"] < 1e-9
        and checks["m0_fleet_split"]["max_abs_share_vs_half"] < 1e-9
        and checks["rows"]["n"] == checks["rows"]["expected"]
        and checks["rates_in_range"]
        and checks["capacity_pooling_ok"])
    dump(checks, "m15_check.json")
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["sim", "check", "all"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.stage in ("sim", "all"):
        run_sim(force=args.force)
    if args.stage in ("check", "all"):
        run_check(force=args.force)


if __name__ == "__main__":
    main()
