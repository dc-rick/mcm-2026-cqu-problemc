"""T022-M12: value of the five time-band betas against one shared beta.

Same training days (10/11/15/16/18), same test days (12/19), same candidate
destination set (all 5,263 cells).  The only difference between the two models
is the distance-decay parameter: five band-specific betas versus one pooled
beta fitted on the summed band likelihoods.

Reported per test day:
  - destination log-loss (negative log probability per trip),
  - top-5 hit rate,
  - zone-hour net-flow error (L1 and max absolute),
with five-band minus pooled differences and their signs.

Reads data/ only; writes data/runs/T022/.  Python must be run with -I.
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
import t005_common as C

RUN = os.path.join(lib.DATA, "runs", "T022")
os.makedirs(RUN, exist_ok=True)
BANDS = C.BANDS
BAND_NAMES = C.BAND_NAMES
TR_DAYS = C.TR_DAYS
TE_DAYS = C.TE_DAYS
K = 5


def band_terms(sub, cells, n, D):
    """Sufficient statistics for the band likelihood, full candidate set."""
    Dj = np.bincount(sub.d.values, minlength=n).astype(np.float64)
    A = C.smooth_attraction(Dj, cells)
    logA = np.log(np.maximum(A, 1e-12)).astype(np.float32)
    key = sub.o.values.astype(np.int64) * n + sub.d.values
    uk, cnt = np.unique(key, return_counts=True)
    ii = (uk // n).astype(np.int64)
    jj = (uk % n).astype(np.int64)
    T1 = np.bincount(ii, weights=cnt * logA[jj], minlength=n)
    T2 = np.bincount(ii, weights=cnt * D[ii, jj], minlength=n)
    N = np.bincount(ii, weights=cnt.astype(float), minlength=n)
    origins = np.where(N > 0)[0]
    return dict(A=A, logA=logA, T1=T1[origins], T2=T2[origins],
                N=N[origins], origins=origins, trips=int(cnt.sum()))


def log_s(terms, beta, D, origins=None):
    o = terms["origins"] if origins is None else origins
    W = terms["logA"][None, :] - np.float32(beta) * D[o, :]
    m = W.max(1)
    S = np.exp(W - m[:, None]).sum(1)
    return (m + np.log(S)).astype(np.float64)


def nll_one(beta, terms, D):
    ls = log_s(terms, beta, D)
    return -(float(terms["T1"].sum()) - beta * float(terms["T2"].sum())
             - float((terms["N"] * ls).sum()))


def fit_one(terms, D):
    res = minimize_scalar(nll_one, bounds=(0.05, 8.0), method="bounded",
                          args=(terms, D), options={"xatol": 1e-4})
    return float(res.x), float(-res.fun)


def nll_pooled(beta, terms_list, D):
    return sum(nll_one(beta, t, D) for t in terms_list)


def fit_pooled(terms_list, D):
    res = minimize_scalar(nll_pooled, bounds=(0.05, 8.0), method="bounded",
                          args=(terms_list, D), options={"xatol": 1e-4})
    return float(res.x), float(-res.fun)


def top5_membership(logA, beta, D, origins, jj, inv, K=5):
    """Deterministic top-5 set per origin: weight desc, cell index asc on ties."""
    hit = np.zeros(len(jj))
    for a in range(0, len(origins), 600):
        o = origins[a:a + 600]
        W = logA[None, :] - np.float32(beta) * D[o, :]
        top = np.argsort(-W, axis=1, kind="stable")[:, :K]
        sel = (inv >= a) & (inv < a + len(o))
        row = inv[sel] - a
        hit[sel] = (top[row] == jj[sel][:, None]).any(1).astype(float)
    return hit


def paired_band(terms, beta_five, beta_shared, D, sub, n):
    """Per-trip paired differences: log-loss (five - shared) and top-5 hit."""
    key = sub.o.values.astype(np.int64) * n + sub.d.values
    uk, cnt = np.unique(key, return_counts=True)
    ii = (uk // n).astype(np.int64)
    jj = (uk % n).astype(np.int64)
    origins, inv = np.unique(ii, return_inverse=True)
    ls_f = log_s(terms, beta_five, D, origins=origins)
    ls_s = log_s(terms, beta_shared, D, origins=origins)
    logp_f = (terms["logA"][jj] - ls_f[inv].astype(np.float32)
              - np.float32(beta_five) * D[ii, jj]).astype(np.float64)
    logp_s = (terms["logA"][jj] - ls_s[inv].astype(np.float32)
              - np.float32(beta_shared) * D[ii, jj]).astype(np.float64)
    # nll_five - nll_shared: negative favours the five-band model
    d_nll = logp_s - logp_f
    hit_f = top5_membership(terms["logA"], beta_five, D, origins, jj, inv)
    hit_s = top5_membership(terms["logA"], beta_shared, D, origins, jj, inv)
    return cnt.astype(np.float64), d_nll, hit_f - hit_s


def paired_stats(weights, d):
    """Weighted mean of per-trip paired differences with a paired 95% CI."""
    n = float(weights.sum())
    mean = float((weights * d).sum() / n)
    var = float((weights * (d - mean) ** 2).sum() / (n - 1.0))
    se = float(np.sqrt(var / n))
    z = float(mean / se) if se > 0 else float("nan")
    return dict(n=int(round(n)), diff=mean, se=se, z=z,
                ci95=[mean - 1.96 * se, mean + 1.96 * se],
                sd=float(np.sqrt(var)))


def softmax_rows(logA, beta, D):
    W = logA[None, :] - np.float32(beta) * D
    m = W.max(1)
    E = np.exp(W - m[:, None])
    return (E / E.sum(1)[:, None]).astype(np.float32)


def score_band(terms, beta, D, sub, n, zid, NZ, Z, dep_hours, arr_hours):
    """Log-loss, top-5, and zone-hour net-flow error for one band one day."""
    key = sub.o.values.astype(np.int64) * n + sub.d.values
    uk, cnt = np.unique(key, return_counts=True)
    ii = (uk // n).astype(np.int64)
    jj = (uk % n).astype(np.int64)
    origins, inv = np.unique(ii, return_inverse=True)
    ls = log_s(terms, beta, D, origins=origins)
    logp = (terms["logA"][jj] - np.float32(beta) * D[ii, jj]
            - ls[inv].astype(np.float32)).astype(np.float64)
    nlp = float(-(cnt * logp).sum() / cnt.sum())

    # expected top-5 hit with uniform tie-breaking, chunked over origins
    hit = 0.0
    for a in range(0, len(origins), 600):
        o = origins[a:a + 600]
        W = terms["logA"][None, :] - np.float32(beta) * D[o, :]
        kk = min(K, W.shape[1])
        thr = np.partition(W, W.shape[1] - kk, axis=1)[:, W.shape[1] - kk]
        above = W > thr[:, None]
        tie = W == thr[:, None]
        n_above = above.sum(1).astype(np.float64)
        n_tie = tie.sum(1).astype(np.float64)
        sel = (inv >= a) & (inv < a + len(o))
        row = inv[sel] - a
        j = jj[sel]
        w = W[row, j]
        p = np.where(w > thr[row], 1.0,
                     np.where(w == thr[row],
                              (kk - n_above[row]) / np.maximum(n_tie[row], 1),
                              0.0))
        hit += float((cnt[sel] * p).sum())
    hit /= cnt.sum()

    P = softmax_rows(terms["logA"], beta, D)
    l1 = 0.0
    mx = 0.0
    for h in dep_hours:
        arr_pred = dep_hours[h] @ P
        pred_z = Z.T @ arr_pred
        obs_z = Z.T @ arr_hours[h]
        dep_z = Z.T @ dep_hours[h]
        net_pred = pred_z - dep_z
        net_obs = obs_z - dep_z
        err = net_pred - net_obs
        l1 += float(np.abs(err).sum())
        mx = max(mx, float(np.abs(err).max()))
    del P
    return dict(nlp=nlp, top5=float(hit), l1=l1, max_abs=mx,
                trips=int(cnt.sum()), pairs=int(len(cnt)))


def main():
    df, gt, cells, D = C.load_model_data()
    n = len(cells)
    zcells = lib.get_cells()
    assert len(zcells) == n
    zid = zcells.zone.to_numpy().astype(int)
    NZ = int(zid.max()) + 1
    Z = np.zeros((n, NZ), dtype=np.float32)
    Z[np.arange(n), zid] = 1.0
    hours_of_band = {b: [h for h in range(24) if BANDS[b][0] <= h <= BANDS[b][1]]
                     for b in range(len(BANDS))}

    print("building band statistics on training days")
    terms = []
    for b in range(len(BANDS)):
        tr = gt[(gt.band == b) & gt.date.isin(TR_DAYS)]
        terms.append(band_terms(tr, cells, n, D))

    beta_bands = []
    ll_bands = []
    for b, name in enumerate(BAND_NAMES):
        beta, ll = fit_one(terms[b], D)
        beta_bands.append(beta)
        ll_bands.append(ll)
        print(f"band {name}: beta={beta:.6f} loglik={ll:.1f}")

    beta_pool, ll_pool = fit_pooled(terms, D)
    print(f"pooled beta={beta_pool:.6f} (sum loglik={ll_pool:.1f}; "
          f"five-band sum={sum(ll_bands):.1f})")

    per_day = {}
    for day in TE_DAYS:
        day_res = {}
        dep_hours = {}
        arr_hours = {}
        sub_day = gt[gt.date == day]
        for h in range(24):
            dep_hours[h] = np.bincount(
                sub_day.o.values[sub_day.hour.values == h], minlength=n
            ).astype(np.float64)
            arr_hours[h] = np.bincount(
                sub_day.d.values[sub_day.hour.values == h], minlength=n
            ).astype(np.float64)
        for model, betas in (("five_band", beta_bands), ("shared_beta", [beta_pool])):
            rec = {"per_band": []}
            num_nlp = num_hit = 0.0
            trips_total = 0
            l1 = 0.0
            mx = 0.0
            for b, name in enumerate(BAND_NAMES):
                sub = gt[(gt.date == day) & (gt.band == b)]
                beta = betas[b] if model == "five_band" else betas[0]
                r = score_band(terms[b], beta, D, sub, n, zid, NZ, Z,
                               {h: dep_hours[h] for h in hours_of_band[b]},
                               {h: arr_hours[h] for h in hours_of_band[b]})
                rec["per_band"].append(dict(band=name, beta=beta, **r))
                num_nlp += r["nlp"] * r["trips"]
                num_hit += r["top5"] * r["trips"]
                trips_total += r["trips"]
                l1 += r["l1"]
                mx = max(mx, r["max_abs"])
            rec.update(nlp=num_nlp / trips_total,
                       top5=num_hit / trips_total,
                       l1=l1, max_abs=mx, trips=trips_total)
            day_res[model] = rec
            print(f"{day} {model}: nlp={rec['nlp']:.6f} top5={rec['top5']:.6f} "
                  f"l1={rec['l1']:.1f} max={rec['max_abs']:.1f}")
        a, p = day_res["five_band"], day_res["shared_beta"]
        weights = []
        d_nll = []
        d_hit = []
        for b in range(len(BANDS)):
            sub = gt[(gt.date == day) & (gt.band == b)]
            w_b, nll_b, hit_b = paired_band(terms[b], beta_bands[b],
                                            beta_pool, D, sub, n)
            weights.append(w_b)
            d_nll.append(nll_b)
            d_hit.append(hit_b)
        w_all = np.concatenate(weights)
        nll_all = np.concatenate(d_nll)
        hit_all = np.concatenate(d_hit)
        paired = dict(
            nll=paired_stats(w_all, nll_all),
            top5=paired_stats(w_all, hit_all),
            note=("per-trip paired difference (five-band minus shared); "
                  "nll < 0 and top5 > 0 favour the five-band model.  "
                  "SE is the paired standard error over all test trips."))
        per_day[day] = dict(five_band=a, shared_beta=p,
                            paired=paired,
                            diff=dict(nlp=a["nlp"] - p["nlp"],
                                      top5=a["top5"] - p["top5"],
                                      l1=a["l1"] - p["l1"],
                                      max_abs=a["max_abs"] - p["max_abs"],
                                      per_band_nlp=[
                                          x["nlp"] - y["nlp"]
                                          for x, y in zip(a["per_band"],
                                                          p["per_band"])],
                                      per_band_top5=[
                                          x["top5"] - y["top5"]
                                          for x, y in zip(a["per_band"],
                                                          p["per_band"])]))
        print(f"{day} delta: nlp={per_day[day]['diff']['nlp']:+.6f} "
              f"top5={per_day[day]['diff']['top5']:+.6f} "
              f"l1={per_day[day]['diff']['l1']:+.1f} "
              f"max={per_day[day]['diff']['max_abs']:+.1f}")
        pn = paired["nll"]
        pt = paired["top5"]
        print(f"{day} paired nll: {pn['diff']:+.6f} "
              f"[{pn['ci95'][0]:+.6f}, {pn['ci95'][1]:+.6f}] z={pn['z']:+.2f}; "
              f"top5: {pt['diff']:+.6f} "
              f"[{pt['ci95'][0]:+.6f}, {pt['ci95'][1]:+.6f}] z={pt['z']:+.2f} "
              f"(n={pn['n']})")

    out = dict(
        task="T022-M12", date="2026-11-18",
        setup=dict(train_days=TR_DAYS, test_days=TE_DAYS,
                   candidate_set="all 5,263 cells",
                   attraction="per-band Gaussian-smoothed training-day arrivals "
                              "(sigma=1.2 cells), identical in both models",
                   difference="distance-decay beta only: five band betas vs one "
                              "pooled beta fitted on the summed band likelihood",
                   predictions="predicted arrivals = the test day's observed "
                               "departures times P(j|i,band); both models use "
                               "the same departure input, so the net-flow error "
                               "isolates the destination model",
                   top5_rule="expected hit with uniform tie-breaking"),
        beta=dict(five_band=dict(zip(BAND_NAMES, beta_bands)),
                  shared=beta_pool,
                  loglik=dict(five_band_sum=float(sum(ll_bands)),
                              shared_sum=float(ll_pool),
                              five_band_each=[float(v) for v in ll_bands])),
        per_day=per_day,
        summary=dict(
            nlp_mean=float(np.mean([per_day[d]["five_band"]["nlp"]
                                    for d in TE_DAYS])),
            nlp_shared_mean=float(np.mean([per_day[d]["shared_beta"]["nlp"]
                                           for d in TE_DAYS])),
            note="negative nlp difference means the five-band model is "
                 "better; positive top5 difference means the five-band "
                 "model is better; inferential claims use the per-day paired "
                 "95% CI in per_day.*.paired, not the sign of a point "
                 "estimate"),
    )
    with open(os.path.join(RUN, "shared_beta.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("wrote", os.path.join(RUN, "shared_beta.json"))


if __name__ == "__main__":
    main()
