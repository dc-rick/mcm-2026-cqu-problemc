"""T005: stability of the user segmentation for k = 3..6.

Mirrors 11_q1_extra.py's feature matrix exactly (same 10-day sample, same six
features, log1p trips, median fill, z-score, float32).  For each k, 20 separate
KMeans runs with different random states (k-means++, n_init=1); within-k
stability is the pairwise adjusted Rand index.  Silhouette is evaluated on the
same fixed 40,000-user sample used by the paper.  Cross-k ARI compares the
best-inertia labelling of each k.  Also reports ARI against the published
data/user_seg.parquet labels (k=5).
"""
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import lib  # noqa: E402
import t005_common as C  # noqa: E402

RUN = os.path.join(ROOT, "data", "runs", "T005")
os.makedirs(RUN, exist_ok=True)
KS = [3, 4, 5, 6]
NSEED = 20
SIL_N = 10000          # all-run silhouette: O(n^2), so keep this small
SIL_PAPER_N = 40000    # paper's sample, representative run only


def build_features(gt):
    prob = gt[["userid", "biketype", "starttime", "hour", "qtr", "date",
               "o", "d", "start_lat", "start_lon", "end_lat",
               "end_lon"]].copy()
    prob["dist"] = C.haversine(prob.start_lat.values, prob.start_lon.values,
                               prob.end_lat.values, prob.end_lon.values)
    g = prob.groupby("userid")
    feat = pd.DataFrame({"trips": g.size(), "dist_mean": g.dist.mean(),
                         "n_days": g.date.nunique(), "n_cells": g.o.nunique()})
    feat["peak_share"] = prob.assign(
        pk=prob.hour.isin([7, 8, 9, 17, 18, 19])).groupby("userid").pk.mean()
    feat["weekend_share"] = prob.assign(
        we=prob.starttime.dt.dayofweek >= 5).groupby("userid").we.mean()
    ch = prob.groupby(["userid", "hour"]).size().rename("c").reset_index()
    ch["p"] = ch.c / ch.groupby("userid").c.transform("sum")
    ch["plogp"] = ch.p * np.log(ch.p + 1e-12)
    feat["hour_entropy"] = -ch.groupby("userid").plogp.sum()
    u2 = prob[["userid", "date", "o", "starttime"]].sort_values(
        ["userid", "starttime"])
    key = u2.userid.astype(str) + "|" + u2.date + "|" + u2.o.astype(str)
    u2["is_return"] = key.duplicated()
    feat["return_share"] = u2.groupby("userid").is_return.mean()
    X = feat[["trips", "dist_mean", "hour_entropy", "peak_share",
              "weekend_share", "return_share"]].copy()
    X["trips"] = np.log1p(X.trips)
    X = X.fillna(X.median())
    X = (X - X.mean()) / X.std().replace(0, 1.0)
    return X.astype(np.float32)


t0 = time.time()
df, gt, cells, D = C.load_model_data()
X = build_features(gt)
Xa = np.ascontiguousarray(X.to_numpy())
print("features", Xa.shape, "users", X.index.nunique(), f"{time.time()-t0:.0f}s")

seg = pd.read_parquet(os.path.join(lib.DATA, "user_seg.parquet"))
if "userid" not in seg.columns:
    seg = seg.reset_index()
seg = seg[["userid", "seg"]].drop_duplicates("userid")
published = seg.set_index("userid").seg.reindex(X.index)
print("published labels matched", int(published.notna().sum()), "of", len(X))
pub = published.to_numpy()

idx = np.random.default_rng(0).choice(len(Xa), min(SIL_N, len(Xa)),
                                      replace=False)
Xs = Xa[idx]
idx_p = np.random.default_rng(0).choice(len(Xa), min(SIL_PAPER_N, len(Xa)),
                                        replace=False)
Xp = Xa[idx_p]

out = {"k": {}, "n_users": int(len(Xa)), "n_seeds": NSEED,
       "silhouette_sample": int(len(idx)),
       "silhouette_paper_sample": int(len(idx_p)), "ks": KS,
       "features": ["log1p(trips)", "dist_mean", "hour_entropy",
                    "peak_share", "weekend_share", "return_share"],
       "published_seg_ari": {}}
labels_by_k = {}
for k in KS:
    tk = time.time()
    labs, inertias, sils = [], [], []
    niters = []
    for s in range(NSEED):
        km = KMeans(k, n_init=1, init="k-means++", random_state=s,
                    max_iter=300)
        km.fit(Xa)
        labs.append(km.labels_.astype(np.int32))
        inertias.append(float(km.inertia_))
        sils.append(float(silhouette_score(Xs, km.labels_[idx])))
        niters.append(int(km.n_iter_))
    L = np.vstack(labs)
    n = len(L)
    aris = []
    for a in range(n):
        for b in range(a + 1, n):
            aris.append(float(adjusted_rand_score(L[a], L[b])))
    aris = np.array(aris)
    ref_ari = np.array([adjusted_rand_score(L[0], L[s]) for s in range(n)])
    rep = int(np.argmin(inertias))
    labels_by_k[k] = L[rep]
    sil_rep_40k = float(silhouette_score(Xp, L[rep][idx_p]))
    sizes = np.bincount(L[rep]).tolist()
    rec = dict(
        inertia_mean=float(np.mean(inertias)),
        inertia_min=float(np.min(inertias)),
        inertia_max=float(np.max(inertias)),
        inertia_best_run=rep,
        silhouette_mean=float(np.mean(sils)),
        silhouette_min=float(np.min(sils)),
        silhouette_max=float(np.max(sils)),
        silhouette_per_seed=sils,
        silhouette_40k_representative=sil_rep_40k,
        ari_pairwise_mean=float(aris.mean()),
        ari_pairwise_min=float(aris.min()),
        ari_pairwise_p05=float(np.percentile(aris, 5)),
        ari_vs_run0_mean=float(ref_ari.mean()),
        ari_vs_run0_min=float(ref_ari.min()),
        n_aris=int(len(aris)),
        representative_sizes=sorted(sizes, reverse=True),
        representative_cluster_sizes_unsorted=sizes,
        n_iter_mean=float(np.mean(niters)),
        n_iter_max=int(np.max(niters)),
    )
    if pub is not None and not np.isnan(pub).any():
        rec_ari_pub = float(adjusted_rand_score(pub, L[rep]))
        rec["ari_vs_published_k5"] = rec_ari_pub
        out["published_seg_ari"][f"k{k}"] = rec_ari_pub
    out["k"][str(k)] = rec
    print(f"k={k}: sil10k {np.mean(sils):.4f} [{np.min(sils):.4f},"
          f"{np.max(sils):.4f}] sil40k(rep) {sil_rep_40k:.4f} "
          f"ARI pair {aris.mean():.4f} min {aris.min():.4f} "
          f"| vs published {rec.get('ari_vs_published_k5', float('nan')):.4f} "
          f"| {time.time()-tk:.0f}s", flush=True)

cross = {}
for a in KS:
    for b in KS:
        cross[f"{a}-{b}"] = float(adjusted_rand_score(labels_by_k[a],
                                                      labels_by_k[b]))
out["cross_k_ari"] = cross
out["cross_k_matrix"] = {str(a): [cross[f"{a}-{b}"] for b in KS] for a in KS}
out["best_silhouette_k"] = max(
    KS, key=lambda k: out["k"][str(k)]["silhouette_40k_representative"])
out["best_silhouette_k_10k"] = max(
    KS, key=lambda k: out["k"][str(k)]["silhouette_mean"])
out["runtime_s"] = float(time.time() - t0)

json.dump(out, io.open(os.path.join(RUN, "k_stability.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
np.savez_compressed(os.path.join(RUN, "k_stability_labels.npz"),
                    **{f"k{k}": labels_by_k[k] for k in KS},
                    userid=X.index.to_numpy())
print("cross-k ARI")
print(pd.DataFrame(out["cross_k_matrix"], index=[str(k) for k in KS]).round(4))
print("best silhouette k", out["best_silhouette_k"],
      "saved", os.path.join(RUN, "k_stability.json"),
      f"{time.time()-t0:.0f}s")
