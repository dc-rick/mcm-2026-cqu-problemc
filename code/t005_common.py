"""Shared Q1 model pieces for T005: bands, data, fit, scoring.

Mirrors code/10_q1_model.py exactly (same cells, days, smoothing, distance
matrix) so the T005 comparison is against the paper's model, not a variant.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize_scalar

import lib

BANDS = [(0, 5), (6, 9), (10, 15), (16, 19), (20, 23)]
BAND_NAMES = ["night 0-5", "morning 6-9", "midday 10-15",
              "evening 16-19", "late 20-23"]
TR_DAYS = ["2017-05-10", "2017-05-11", "2017-05-15", "2017-05-16",
           "2017-05-18"]
TE_DAYS = ["2017-05-12", "2017-05-19"]
SIGMA_A = 1.2
SIGMA_RHO = 1.0
R_KM = 6371.0


def load_model_data():
    df = lib.load_trips(lib.ALL_DAYS)
    cells, df = lib.make_cells(df, lib.THR)
    df["band"] = pd.cut(df.hour, [-1, 5, 9, 15, 19, 23],
                        labels=False).astype("Int64")
    ok = df.o.notna() & df.d.notna()
    gt = df[ok].copy()
    gt["o"] = gt.o.astype(int)
    gt["d"] = gt.d.astype(int)
    D = np.load(lib.DATA + r"\dist_matrix.npy").astype(np.float32)
    return df, gt, cells, D


def band_stats(sub, n):
    O = np.bincount(sub.o.values, minlength=n).astype(np.float64)
    Dj = np.bincount(sub.d.values, minlength=n).astype(np.float64)
    OD = np.zeros((n, n), dtype=np.float32)
    np.add.at(OD, (sub.o.values, sub.d.values), 1.0)
    return O, Dj, OD


def smooth_attraction(Dj, cells, sigma=SIGMA_A):
    ix = cells.ix.values - cells.ix.values.min()
    iy = cells.iy.values - cells.iy.values.min()
    g = np.zeros((ix.max() + 1, iy.max() + 1))
    g[ix, iy] = Dj
    g = gaussian_filter(g, sigma)
    return g[ix, iy].astype(np.float64)


def fit_beta(OD, A, D, lo=0.05, hi=8.0, xatol=1e-4, choice="paper"):
    """MLE of beta for P(j|i) ∝ A_j exp(-beta d_ij).

    choice="paper" reproduces 10_q1_model.py exactly: the normaliser only runs
    over destinations with a non-zero training count (nz_j).  choice="all"
    uses the full destination choice set, as the scored model does.
    """
    logA = np.log(np.maximum(A, 1e-12))
    rows = OD.sum(1)
    nz = np.where(rows > 0)[0]
    if choice == "paper":
        nz_j = np.where(OD.sum(0) > 0)[0]
    elif choice == "all":
        nz_j = np.arange(OD.shape[1])
    else:
        raise ValueError(choice)
    Osub = OD[np.ix_(nz, nz_j)]
    Asub = logA[nz_j]
    Dsub = D[np.ix_(nz, nz_j)]

    def nll(beta):
        W = (-beta) * Dsub + Asub[None, :]
        m = W.max(1, keepdims=True)
        E = np.exp(W - m)
        S = E.sum(1, keepdims=True)
        logp = W - m - np.log(S)
        return float(-(Osub * logp).sum())

    res = minimize_scalar(nll, bounds=(lo, hi), method="bounded",
                          options={"xatol": xatol})
    return float(res.x), float(-res.fun)


def norm_logp(logW):
    logW = logW.astype(np.float32)
    mx = logW.max(1, keepdims=True)
    E = np.exp(logW - mx)
    return logW - mx - np.log(E.sum(1, keepdims=True))


def logp_full(beta, A, D):
    logA = np.log(np.maximum(A, 1e-12)).astype(np.float32)
    return norm_logp((-beta) * D + logA[None, :])


def logp_dist(beta, D):
    return norm_logp((-beta) * D)


def logp_att(A, D):
    logA = np.log(np.maximum(A, 1e-12)).astype(np.float32)
    return norm_logp(np.tile(logA, (D.shape[0], 1)))


def logp_uniform(n):
    return np.full((n, n), -np.log(n), dtype=np.float32)


def score(logp, ODte, k=5, uniform=False):
    """Return (nlp, top5, per-pair nlp, per-pair hit) on observed test cells."""
    m = ODte > 0
    ii, jj = np.where(m)
    cnt = ODte[m]
    w = logp[ii, jj]
    nlp = float(-(cnt * w).sum() / cnt.sum())
    if uniform:
        hit = float(k / logp.shape[1])
        hit_pair = np.full(len(ii), hit / k)  # placeholder, unused
    else:
        hit = 0.0
        hit_pair = np.zeros(len(ii), dtype=np.float64)
        for i0 in np.unique(ii):
            sel = ii == i0
            kk = min(k, logp.shape[1])
            top = np.argpartition(-logp[i0], kk - 1)[:kk]
            in_top = np.isin(jj[sel], top)
            hit += cnt[sel][in_top].sum()
            hit_pair[sel] = in_top.astype(np.float64)
        hit = float(hit / cnt.sum())
    return nlp, hit, -w, hit_pair


def score_expected_top5(logp, ODte, k=5):
    """Expected top-k hit rate with uniform random tie-breaking.

    Destinations strictly above the k-th largest score count with probability
    1; destinations tied at the threshold share the remaining places.  This is
    exact for the grid's tied lattice distances and for tied zero attraction.
    Uniform 1/n scores give exactly k/n.
    """
    m = ODte > 0
    ii, jj = np.where(m)
    cnt = ODte[m]
    hit_pair = np.zeros(len(ii), dtype=np.float64)
    for i0 in np.unique(ii):
        row = logp[i0]
        kk = min(k, row.size)
        thr = np.partition(row, row.size - kk)[row.size - kk]
        above = row > thr
        n_above = int(above.sum())
        tie = row == thr
        n_tie = int(tie.sum())
        sel = ii == i0
        p = np.where(above[jj[sel]], 1.0,
                     np.where(tie[jj[sel]], (kk - n_above) / n_tie, 0.0))
        hit_pair[sel] = p
    hit = float((cnt * hit_pair).sum() / cnt.sum())
    return hit, hit_pair


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    dp = p2 - p1
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
