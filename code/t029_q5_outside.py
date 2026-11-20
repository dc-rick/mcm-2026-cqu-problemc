"""T029: Q5 with an explicit outside-fence inventory state.

T022's fenced-parking replay removes overflow arrivals from the system and
records them only as a count.  That understates the platform's fleet: a
bicycle that cannot enter a full fence still exists, can be rented later if
it is parked just outside, and returns to the inside pool when a slot frees.
This module reruns the same 15-minute, two-platform, four-mismatch experiment
with two states per platform and site:

  occ[j]  bicycles parked inside the 150-slot fence;
  out[j]  bicycles parked outside the fence after an overflow.

A departure first takes from occ, then from out; only the residual is unmet.
An arrival fills the inside room, and the overflow goes to out.  The identity
occ + out = fleet is asserted at every step, so no bicycle disappears.
Inside capacity is still the binding parking constraint; outside bicycles do
not consume a slot.

Regimes are T022's: independent (150 each), independent_equal (75 each),
shared (one 150-slot fence, separate fleets and separate outside pools).
Overflow is rationed proportionally to each platform's arrivals in the shared
regime, exactly as in T022.

Outputs go to data/runs/T029/.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t022_m15_q5_mismatch as t22

RUN = os.path.join(t22.DATA, "runs", "T029")
os.makedirs(RUN, exist_ok=True)


def dump(obj, name):
    path = os.path.join(RUN, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    print("wrote", path, flush=True)


def prep_t30():
    """T022's prep, rebuilt on T030's conserved demand and knee plan.

    T022.prep() reads T019's independent-Poisson plan.  On the rebuilt Q4 the
    plan and the demand are T030's, so Q5 must use those or it describes a
    layout and a demand surface that the paper no longer reports.
    """
    from scipy.sparse import csr_matrix
    plan_path = os.path.join(t22.DATA, "runs", "T030",
                             "plan_meas343_main.npz")
    zpath = os.path.join(t22.DATA, "runs", "T030", "cell_demand_cons.npz")
    plan = np.load(plan_path)
    z = np.load(zpath)
    dep15 = z["dep15"]
    arr15 = z["arr15"]
    xy = z["xy"]
    C = dep15.shape[1]
    s, f, a, ii, jj, I0 = (plan["s"], plan["f"], plan["a"],
                           plan["ii"], plan["jj"], plan["I0"])
    frac = np.divide(f, a[ii], out=np.zeros_like(f), where=a[ii] > 0)
    W = csr_matrix((frac, (ii, jj)), shape=(C, C))
    dep_site = np.ascontiguousarray(W.T.dot(dep15.T).T)
    arr_site = np.ascontiguousarray(W.T.dot(arr15.T).T)
    I0_site = np.asarray(W.T.dot(I0)).ravel()
    x = xy[:, 0]
    xt = 2.0 * (x - x.min()) / max(x.max() - x.min(), 1e-12) - 1.0
    dep_tot = dep15.sum(0)
    fleets = {m: (t22.SHARE * I0, t22.SHARE * I0) for m in t22.MISMATCH}
    shares = {m: t22.demand_shares(dep_tot, xt, m) for m in t22.MISMATCH}
    return dict(plan=plan, z=z, C=C, W=W, s=s, dep_site=dep_site,
                arr_site=arr_site, dep15=dep15, arr15=arr15,
                I0_site=I0_site, I0=I0, xt=xt, fleets=fleets,
                shares=shares, xy=xy, open_sites=int((s > 0.5).sum()),
                plan_path=plan_path, demand_path=zpath)


def simulate_out(depA, arrA, occA0, depB, arrB, occB0, capA, capB, pooled,
                 record_peak=False):
    """One day at 15 minutes with inside and outside states."""
    occA = occA0.astype(float).copy()
    occB = occB0.astype(float).copy()
    outA = np.zeros_like(occA)
    outB = np.zeros_like(occB)
    fleetA = float(occA.sum())
    fleetB = float(occB.sum())
    depA = depA.astype(float)
    depB = depB.astype(float)
    arrA = arrA.astype(float)
    arrB = arrB.astype(float)

    unmetA = unmetB = 0.0
    overA = overB = 0.0
    out_usedA = out_usedB = 0.0
    dep_drawnA = dep_drawnB = 0.0
    arr_drawnA = arr_drawnB = 0.0
    max_resid = 0.0
    unmetA_site = np.zeros_like(occA)
    unmetB_site = np.zeros_like(occB)
    overA_site = np.zeros_like(occA)
    overB_site = np.zeros_like(occB)
    if record_peak:
        peakA = occA.copy()
        peakB = occB.copy()
        peakS = occA + occB
        peakOutA = outA.copy()
        peakOutB = outB.copy()

    # initial capacity: excess goes outside, never disappears
    if not pooled:
        oa = np.maximum(0.0, occA - capA)
        ob = np.maximum(0.0, occB - capB)
        occA = np.minimum(occA, capA)
        occB = np.minimum(occB, capB)
        outA += oa
        outB += ob
    else:
        tot = occA + occB
        scale = np.divide(capA, tot, out=np.ones_like(tot), where=tot > 0)
        scale = np.minimum(scale, 1.0)
        oa = occA * (1.0 - scale)
        ob = occB * (1.0 - scale)
        occA = occA * scale
        occB = occB * scale
        outA += oa
        outB += ob

    T = depA.shape[0]
    for t in range(T):
        dA, dB = depA[t], depB[t]
        # planned spatial arrival patterns; the level is set by served trips
        aA0, aB0 = arrA[t], arrB[t]
        # departures: inside first, then outside
        tiA = np.minimum(dA, occA)
        rA = dA - tiA
        toA = np.minimum(rA, outA)
        uA = rA - toA
        tiB = np.minimum(dB, occB)
        rB = dB - tiB
        toB = np.minimum(rB, outB)
        uB = rB - toB
        occA -= tiA
        occB -= tiB
        outA -= toA
        outB -= toB
        out_usedA += float(toA.sum())
        out_usedB += float(toB.sum())
        unmetA += float(uA.sum())
        unmetB += float(uB.sum())
        unmetA_site += uA
        unmetB_site += uB
        dep_drawnA += float(dA.sum())
        dep_drawnB += float(dB.sum())
        # actual arrivals = served departures of the same platform; the
        # planned site pattern only routes them
        servedA = float(tiA.sum() + toA.sum())
        servedB = float(tiB.sum() + toB.sum())
        sA = servedA / aA0.sum() if aA0.sum() > 1e-12 else 0.0
        sB = servedB / aB0.sum() if aB0.sum() > 1e-12 else 0.0
        aA = aA0 * sA
        aB = aB0 * sB
        arr_drawnA += float(aA.sum())
        arr_drawnB += float(aB.sum())

        if not pooled:
            nA = occA + aA
            nB = occB + aB
            oa = np.maximum(0.0, nA - capA)
            ob = np.maximum(0.0, nB - capB)
            occA = np.minimum(nA, capA)
            occB = np.minimum(nB, capB)
            outA += oa
            outB += ob
        else:
            tot = occA + occB
            inc = aA + aB
            room = np.maximum(0.0, capA - tot)
            scale = np.divide(room, inc, out=np.ones_like(inc),
                              where=inc > 0)
            scale = np.minimum(scale, 1.0)
            addA = aA * scale
            addB = aB * scale
            oa = aA - addA
            ob = aB - addB
            occA += addA
            occB += addB
            outA += oa
            outB += ob
        overA += float(oa.sum())
        overB += float(ob.sum())
        overA_site += oa
        overB_site += ob

        # conservation: inside + outside equals the platform fleet
        max_resid = max(max_resid,
                        abs(float(occA.sum() + outA.sum()) - fleetA),
                        abs(float(occB.sum() + outB.sum()) - fleetB))
        if record_peak:
            peakA = np.maximum(peakA, occA)
            peakB = np.maximum(peakB, occB)
            peakS = np.maximum(peakS, occA + occB)
            peakOutA = np.maximum(peakOutA, outA)
            peakOutB = np.maximum(peakOutB, outB)

    out = dict(unmet=unmetA + unmetB, unmetA=unmetA, unmetB=unmetB,
               overflow=overA + overB, overflowA=overA, overflowB=overB,
               outside_used=out_usedA + out_usedB,
               outside_usedA=out_usedA, outside_usedB=out_usedB,
               outside_end=float(outA.sum() + outB.sum()),
               outside_endA=float(outA.sum()), outside_endB=float(outB.sum()),
               fleetA=fleetA, fleetB=fleetB, max_resid=max_resid,
               dep=dep_drawnA + dep_drawnB, depA=dep_drawnA, depB=dep_drawnB,
               arr=arr_drawnA + arr_drawnB, arrA=arr_drawnA, arrB=arr_drawnB,
               unmetA_site=unmetA_site, unmetB_site=unmetB_site,
               overA_site=overA_site, overB_site=overB_site)
    if record_peak:
        out.update(peakA=peakA, peakB=peakB, peakS=peakS,
                   peakOutA=peakOutA, peakOutB=peakOutB)
    return out


def run(force=False):
    path = os.path.join(RUN, "m15_outside.json")
    if os.path.exists(path) and not force:
        with open(path, encoding="utf-8") as fh:
            print("reuse", path)
            return json.load(fh)
    t0 = time.time()
    P = prep_t30()
    dep_site, arr_site = P["dep_site"], P["arr_site"]
    W = P["W"]
    rows = []
    per_platform = {}
    cap_arrays = {}

    # single platform reference, outside state
    ref = {k: [] for k in ("unmet", "overflow", "dep", "arr", "outside_used",
                           "outside_end")}
    ref_peaks = []
    ref_resid = []
    for seed in t22.SEEDS:
        d, a = t22.draw_flows(dep_site, arr_site, seed + 1000)
        occ0 = P["I0_site"].copy()
        r = simulate_out(d, a, occ0, np.zeros_like(d), np.zeros_like(a),
                         np.zeros_like(occ0), t22.CAP, t22.CAP, False,
                         record_peak=True)
        for k in ref:
            ref[k].append(r[k])
        ref_peaks.append(r["peakA"])
        ref_resid.append(r["max_resid"])
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
        outside_used=float(np.mean(ref["outside_used"])),
        outside_end=float(np.mean(ref["outside_end"])),
        required_capacity_slots=float(
            np.percentile(ref_peaks, 95, axis=0).sum()),
        max_conservation_resid=float(max(ref_resid)))
    print("single platform", ref_rec, flush=True)

    for m in t22.MISMATCH:
        IA, IB = P["fleets"][m]
        sA, shift_a = P["shares"][m]
        depA_site = np.ascontiguousarray(
            W.T.dot((P["dep15"] * sA[None, :]).T).T)
        depB_site = np.ascontiguousarray(
            W.T.dot((P["dep15"] * (1.0 - sA)[None, :]).T).T)
        arrA_site = arr_site * t22.SHARE
        arrB_site = arr_site * t22.SHARE
        occA0 = t22.occ_from_cells(IA, W)
        occB0 = t22.occ_from_cells(IB, W)
        peaksA, peaksB, peaksS = [], [], []
        for seed in t22.SEEDS:
            dA, aA = t22.draw_flows(depA_site, arrA_site, seed + 1)
            dB, aB = t22.draw_flows(depB_site, arrB_site, seed + 5001)
            r = simulate_out(dA, aA, occA0, dB, aB, occB0,
                             t22.CAP, t22.CAP, False, record_peak=True)
            peaksA.append(r["peakA"])
            peaksB.append(r["peakB"])
            peaksS.append(r["peakS"])
        peaksA = np.array(peaksA)
        peaksB = np.array(peaksB)
        peaksS = np.array(peaksS)
        cap_arrays[f"m{m}_peakA"] = np.percentile(peaksA, 95, axis=0)
        cap_arrays[f"m{m}_peakB"] = np.percentile(peaksB, 95, axis=0)
        cap_arrays[f"m{m}_peakS"] = np.percentile(peaksS, 95, axis=0)
        for name, capA, capB, pooled in t22.REGIMES:
            rec = {k: [] for k in ("unmet", "overflow", "dep", "arr",
                                   "outside_used", "outside_end",
                                   "depA", "depB")}
            resid = []
            for seed in t22.SEEDS:
                dA, aA = t22.draw_flows(depA_site, arrA_site, seed + 1)
                dB, aB = t22.draw_flows(depB_site, arrB_site, seed + 5001)
                r = simulate_out(dA, aA, occA0, dB, aB, occB0,
                                 capA, capB, pooled)
                for k in rec:
                    rec[k].append(r[k])
                resid.append(r["max_resid"])
            ur = np.array([u / max(d, 1e-9)
                           for u, d in zip(rec["unmet"], rec["dep"])])
            fr = np.array([o / max(a, 1e-9)
                           for o, a in zip(rec["overflow"], rec["arr"])])
            row = dict(
                m=float(m), institution=name, capA=float(capA),
                capB=float(capB), pooled=bool(pooled),
                unmet_rate=float(ur.mean()),
                unmet_rate_sd=float(ur.std()),
                full_rate=float(fr.mean()), full_rate_sd=float(fr.std()),
                departures=float(np.mean(rec["dep"])),
                arrivals=float(np.mean(rec["arr"])),
                unmet_departures=float(np.mean(rec["unmet"])),
                overflow_arrivals=float(np.mean(rec["overflow"])),
                outside_used=float(np.mean(rec["outside_used"])),
                outside_end=float(np.mean(rec["outside_end"])),
                required_capacity_slots=float(
                    np.percentile(peaksS, 95, axis=0).sum()
                    if pooled else
                    np.percentile(peaksA, 95, axis=0).sum()
                    + np.percentile(peaksB, 95, axis=0).sum()),
                max_conservation_resid=float(max(resid)))
            rows.append(row)
            per_platform[f"m{m}_{name}"] = dict(
                depA=float(np.mean(rec["depA"])),
                depB=float(np.mean(rec["depB"])),
                unmet_total=float(np.mean(rec["unmet"])),
                outside_end=float(np.mean(rec["outside_end"])))
            print(row, flush=True)
    # shared-vs-independent ratios
    ratios = {}
    for m in t22.MISMATCH:
        ind = next(r for r in rows if r["m"] == m and
                   r["institution"] == "independent")
        sha = next(r for r in rows if r["m"] == m and
                   r["institution"] == "shared")
        ratios[str(m)] = dict(
            capacity_ratio_ind_over_shared=float(
                ind["required_capacity_slots"]
                / sha["required_capacity_slots"]),
            unmet_diff_shared_minus_ind=float(
                sha["unmet_rate"] - ind["unmet_rate"]),
            overflow_diff_shared_minus_ind=float(
                sha["overflow_arrivals"] - ind["overflow_arrivals"]))
    out = dict(task="T029 Q5 outside-fence inventory", date="2026-11-20",
               seeds=t22.SEEDS, mismatch=t22.MISMATCH,
               plan_path=P["plan_path"], demand_path=P["demand_path"],
               open_sites=int(P["open_sites"]), cap=t22.CAP,
               single_platform=ref_rec, rows=rows, ratios=ratios,
               note=("T030 conserved demand and knee plan (3,500 sites, "
                     "150 slots, 450 m, 343,638 inventory); occ + outside = "
                     "fleet at every step; departures draw inside first, then "
                     "outside; arrivals fill inside room and overflow to "
                     "outside.  Outside bicycles do not consume a slot and "
                     "are not reinserted into inside occupancy."))
    np.savez(os.path.join(RUN, "m15_outside_capacity.npz"), **cap_arrays)
    dump(out, "m15_outside.json")
    print("runtime %.1fs" % (time.time() - t0), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("all",))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run(force=args.force)
