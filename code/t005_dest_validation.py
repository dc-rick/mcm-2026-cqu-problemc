"""T005: four-model out-of-sample comparison for Q1 destination choice.

Models per band:
  full        A_j(h) exp(-beta d_ij), beta fitted by MLE
  distance    exp(-beta d_ij), A=1, beta fitted by its own MLE
  attraction  A_j(h), beta=0
  uniform     1/n

Main split: fit on 5 typical days, test on 2 held-out typical days.
Control:   random split of trips from the same 7 days, with the same per-band
           training count as the day split (so training size is matched).
No AIC is used.  The old q1_validation.csv distance column used the full-model
beta and is not a distance-model optimum.

Top-5 is the expected hit rate under uniform random tie-breaking (exact for
the tied lattice distances and tied zero attraction); the paper's own
argpartition convention is reported alongside for reproduction.
"""
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import t005_common as C  # noqa: E402

RUN = os.path.join(ROOT, "data", "runs", "T005")
os.makedirs(RUN, exist_ok=True)
NB = 500
SEVEN = ["2017-05-10", "2017-05-11", "2017-05-12", "2017-05-15",
         "2017-05-16", "2017-05-18", "2017-05-19"]


def boot_ci(diff, nboot=NB, seed=0):
    rng = np.random.default_rng(seed)
    m = np.array([diff[rng.integers(0, len(diff), len(diff))].mean()
                  for _ in range(nboot)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def eval_band(tr, te, cells, D, n, choice_sens=False):
    O, Dj, OD = C.band_stats(tr, n)
    A = C.smooth_attraction(Dj, cells)
    bf, llf = C.fit_beta(OD, A, D)
    bd, lld = C.fit_beta(OD, np.ones(n), D)
    Ote, Dte, ODte = C.band_stats(te, n)
    models = {
        "full": C.logp_full(bf, A, D),
        "distance": C.logp_dist(bd, D),
        "attraction": C.logp_att(A, D),
        "uniform": C.logp_uniform(n),
    }
    res, per = {}, {}
    for name, logp in models.items():
        nlp, hit_ap, nlp_pair, hit_pair_ap = C.score(
            logp, ODte, uniform=(name == "uniform"))
        if name == "uniform":
            hit_te = 5.0 / n
            hit_pair = np.full(len(nlp_pair), hit_te)
        else:
            hit_te, hit_pair = C.score_expected_top5(logp, ODte)
        res[name] = dict(nlp=nlp, top5=hit_te, top5_argpartition=hit_ap)
        per[name] = dict(nlp_pair=nlp_pair, hit_pair=hit_pair,
                         hit_pair_ap=hit_pair_ap)
    m = ODte > 0
    cnt = ODte[m].astype(int)
    d_nlp = np.repeat(per["full"]["nlp_pair"] - per["distance"]["nlp_pair"], cnt)
    d_hit = np.repeat(per["full"]["hit_pair"] - per["distance"]["hit_pair"],
                      cnt)
    rec = dict(
        n_train=int(len(tr)), n_test=int(len(te)),
        beta_full=bf, beta_distance=bd, beta_diff=bd - bf,
        train_ll_full=llf, train_ll_distance=lld,
        nlp={k: res[k]["nlp"] for k in res},
        top5={k: res[k]["top5"] for k in res},
        top5_argpartition={k: res[k]["top5_argpartition"] for k in res},
        delta_nlp_full_minus_distance=float(d_nlp.mean()),
        delta_nlp_ci95=boot_ci(d_nlp),
        delta_top5_full_minus_distance=float(d_hit.mean()),
        delta_top5_ci95=boot_ci(d_hit),
        _d_nlp=d_nlp, _d_hit=d_hit,
    )
    if choice_sens:
        bfa, _ = C.fit_beta(OD, A, D, choice="all")
        bda, _ = C.fit_beta(OD, np.ones(n), D, choice="all")
        nfa, _, _, _ = C.score(C.logp_full(bfa, A, D), ODte)
        nda, _, _, _ = C.score(C.logp_dist(bda, D), ODte)
        rec["full_choice_set"] = dict(
            beta_full=bfa, beta_distance=bda, beta_diff=bda - bfa,
            nlp_full=nfa, nlp_distance=nda,
            beta_full_shift=bfa - bf, beta_distance_shift=bda - bd,
            nlp_full_shift=nfa - res["full"]["nlp"],
            nlp_distance_shift=nda - res["distance"]["nlp"])
    return rec


def pool(rows):
    d_nlp = np.concatenate([r["_d_nlp"] for r in rows])
    d_hit = np.concatenate([r["_d_hit"] for r in rows])
    nt = np.array([r["n_test"] for r in rows], dtype=float)
    return dict(
        n_test=int(nt.sum()),
        nlp={k: float(sum(r["nlp"][k] * r["n_test"] for r in rows) / nt.sum())
             for k in ("full", "distance", "attraction", "uniform")},
        top5={k: float(sum(r["top5"][k] * r["n_test"] for r in rows) / nt.sum())
              for k in ("full", "distance", "attraction", "uniform")},
        top5_argpartition={
            k: float(sum(r["top5_argpartition"][k] * r["n_test"] for r in rows)
                     / nt.sum())
            for k in ("full", "distance", "attraction", "uniform")},
        delta_nlp_full_minus_distance=float(d_nlp.mean()),
        delta_nlp_ci95=boot_ci(d_nlp, seed=1),
        delta_top5_full_minus_distance=float(d_hit.mean()),
        delta_top5_ci95=boot_ci(d_hit, seed=2),
    )


def strip(rows):
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]


t0 = time.time()
df, gt, cells, D = C.load_model_data()
n = len(cells)
print("data", len(gt), "cells", n)

# ---------------------------------------------------------------- day split
day_rows = []
for b, name in enumerate(C.BAND_NAMES):
    tr = gt[(gt.band == b) & gt.date.isin(C.TR_DAYS)]
    te = gt[(gt.band == b) & gt.date.isin(C.TE_DAYS)]
    r = eval_band(tr, te, cells, D, n, choice_sens=True)
    r["band"] = name
    day_rows.append(r)
    print(f"{name}: bf={r['beta_full']:.4f} bd={r['beta_distance']:.4f} "
          f"nlp full={r['nlp']['full']:.4f} dist={r['nlp']['distance']:.4f} "
          f"att={r['nlp']['attraction']:.4f} unif={r['nlp']['uniform']:.3f} "
          f"top5 {r['top5']['full']:.4f}/{r['top5']['distance']:.4f}",
          flush=True)
day_out = dict(train_days=C.TR_DAYS, test_days=C.TE_DAYS,
               train_trips=int(sum(r["n_train"] for r in day_rows)),
               test_trips=int(sum(r["n_test"] for r in day_rows)),
               bands=strip(day_rows), pooled=pool(day_rows))
print("day split done", f"{time.time()-t0:.0f}s", flush=True)

# ------------------------------------------------------------- random split
rand_seeds = [0, 1, 2]
rand_rows = {name: [] for name in C.BAND_NAMES}
rand_pool = []
seven = gt[gt.date.isin(SEVEN)]
for seed in rand_seeds:
    rng = np.random.default_rng(seed)
    rows = []
    for b, name in enumerate(C.BAND_NAMES):
        sub = seven[seven.band == b]
        n_train = int(((gt.band == b) & gt.date.isin(C.TR_DAYS)).sum())
        idx = rng.permutation(len(sub))
        tr = sub.iloc[idx[:n_train]]
        te = sub.iloc[idx[n_train:]]
        r = eval_band(tr, te, cells, D, n)
        r["band"] = name
        rand_rows[name].append(r)
        rows.append(r)
    rand_pool.append(pool(rows))
    print("random seed", seed, "done", f"{time.time()-t0:.0f}s", flush=True)

rand_out = {"seeds": rand_seeds, "source_days": SEVEN,
            "train_size": "same per-band count as the day split", "bands": {},
            "pooled": {}}
for name in C.BAND_NAMES:
    rows = rand_rows[name]
    rec = {"beta_full_mean": float(np.mean([r["beta_full"] for r in rows])),
           "beta_full_sd": float(np.std([r["beta_full"] for r in rows], ddof=1)),
           "beta_distance_mean": float(np.mean([r["beta_distance"] for r in rows])),
           "beta_distance_sd": float(np.std([r["beta_distance"] for r in rows], ddof=1))}
    for k in ("full", "distance", "attraction", "uniform"):
        rec[f"nlp_{k}_mean"] = float(np.mean([r["nlp"][k] for r in rows]))
        rec[f"nlp_{k}_sd"] = float(np.std([r["nlp"][k] for r in rows], ddof=1))
        rec[f"top5_{k}_mean"] = float(np.mean([r["top5"][k] for r in rows]))
        rec[f"top5_{k}_sd"] = float(np.std([r["top5"][k] for r in rows], ddof=1))
    rec["delta_nlp_mean"] = float(np.mean(
        [r["delta_nlp_full_minus_distance"] for r in rows]))
    rec["delta_nlp_sd"] = float(np.std(
        [r["delta_nlp_full_minus_distance"] for r in rows], ddof=1))
    rec["delta_top5_mean"] = float(np.mean(
        [r["delta_top5_full_minus_distance"] for r in rows]))
    rec["delta_top5_sd"] = float(np.std(
        [r["delta_top5_full_minus_distance"] for r in rows], ddof=1))
    rand_out["bands"][name] = rec
rand_out["pooled"] = {
    "nlp": {k: float(np.mean([p["nlp"][k] for p in rand_pool]))
            for k in ("full", "distance", "attraction", "uniform")},
    "top5": {k: float(np.mean([p["top5"][k] for p in rand_pool]))
             for k in ("full", "distance", "attraction", "uniform")},
    "delta_nlp_mean": float(np.mean([p["delta_nlp_full_minus_distance"]
                                     for p in rand_pool])),
    "delta_nlp_sd": float(np.std([p["delta_nlp_full_minus_distance"]
                                  for p in rand_pool], ddof=1)),
    "delta_top5_mean": float(np.mean([p["delta_top5_full_minus_distance"]
                                      for p in rand_pool])),
    "delta_top5_sd": float(np.std([p["delta_top5_full_minus_distance"]
                                   for p in rand_pool], ddof=1)),
}

out = dict(
    day_split=day_out,
    random_split=rand_out,
    old_q1_validation_status=(
        "data/q1_validation.csv's ll_distance and top5_dist used the full "
        "model's beta (10_q1_model.py:136), not the distance model's MLE; the "
        "old distance column is invalid as a distance-model optimum and is "
        "superseded by the distance values here."),
    methods=dict(
        distance_matrix="data/dist_matrix.npy; diagonal 0.35/sqrt(2)=0.2475 km",
        A="Gaussian-smoothed training arrivals, sigma=1.2 cells",
        beta="bounded scalar MLE, 0.05..8.0 1/km, xatol=1e-4; paper's "
             "choice-set convention (non-zero training destinations), with a "
             "full-choice-set sensitivity",
        nlp="negative log-likelihood per test trip (nats)",
        top5="expected share of test trips whose realised destination is in "
             "the origin's top-5, random tie-breaking; uniform = 5/n = "
             f"{5/n:.6f}; argpartition convention also saved",
        ci="500-draw trip bootstrap, 95th percentile interval"),
    runtime_s=float(time.time() - t0),
)
json.dump(out, io.open(os.path.join(RUN, "dest_validation.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
pd.DataFrame(strip(day_rows)).to_csv(
    os.path.join(RUN, "dest_validation.csv"), index=False)
print("saved", os.path.join(RUN, "dest_validation.json"),
      f"{time.time()-t0:.0f}s")
