"""Q1: trip-generation + destination-choice model, estimation, and validation.

Model
-----
Trip generation:   N_i(t) ~ Poisson( Lambda * pi(t) * rho_i(t) )
Destination:       P(j | i, h) = A_j(h) exp(-beta_h d_ij) / sum_k A_k(h) exp(-beta_h d_ik)
with A_j(h) = spatially smoothed arrivals in band h (attractiveness field).

Outputs: params.json, od_model.npz, validation table, figures.
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.ndimage import gaussian_filter
import lib
from lib import DATA, COL, save, style
import matplotlib.pyplot as plt

style()
BANDS = [(0, 5), (6, 9), (10, 15), (16, 19), (20, 23)]
BAND_NAMES = ["night 0-5", "morning 6-9", "midday 10-15", "evening 16-19", "late 20-23"]

df = lib.load_trips(lib.ALL_DAYS)
cells, df = lib.make_cells(df, lib.THR)
n = len(cells)
df["band"] = pd.cut(df.hour, [-1, 5, 9, 15, 19, 23], labels=False).astype("Int64")
print(f"cells {n}, trips {len(df)}, both-in {(df.o.notna() & df.d.notna()).mean():.4f}")

ok = df.o.notna() & df.d.notna()
gt = df[ok].copy()
gt["o"] = gt.o.astype(int); gt["d"] = gt.d.astype(int)
xy = lib.cell_xy(cells)
D = lib.dist_matrix(xy)
np.fill_diagonal(D, 0.35 / np.sqrt(2))     # intra-cell distance: mean offset (km)

tr_days = ["2017-05-10", "2017-05-11", "2017-05-15", "2017-05-16", "2017-05-18"]
te_days = ["2017-05-12", "2017-05-19"]


def band_stats(sub):
    """Return O_i, D_j, OD counts for one subset."""
    O = np.bincount(sub.o.values, minlength=n).astype(np.float64)
    Dj = np.bincount(sub.d.values, minlength=n).astype(np.float64)
    OD = np.zeros((n, n), dtype=np.float32)
    np.add.at(OD, (sub.o.values, sub.d.values), 1.0)
    return O, Dj, OD


def smooth_attraction(Dj, sigma=1.2):
    """Gaussian smoothing of the arrival field on the cell lattice (regularisation)."""
    ix = cells.ix.values - cells.ix.values.min()
    iy = cells.iy.values - cells.iy.values.min()
    g = np.zeros((ix.max() + 1, iy.max() + 1))
    g[ix, iy] = Dj
    g = gaussian_filter(g, sigma)
    return g[ix, iy]


def fit_beta(OD, A, mask=None, lo=0.05, hi=8.0):
    """MLE of beta for P(j|i) ∝ A_j exp(-beta d_ij). Returns beta, loglik."""
    logA = np.log(np.maximum(A, 1e-12))
    rows = OD.sum(1)
    nz = np.where(rows > 0)[0]
    Osub = OD[nz]
    nz_j = np.where(OD.sum(0) > 0)[0]
    Osub = Osub[:, nz_j]
    Asub = logA[nz_j]
    Dsub = D[np.ix_(nz, nz_j)]

    def nll(beta):
        W = -beta * Dsub + Asub[None, :]
        m = W.max(1, keepdims=True)
        E = np.exp(W - m)
        S = E.sum(1, keepdims=True)
        logp = W - m - np.log(S)
        return -(Osub * logp).sum()

    res = minimize_scalar(nll, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-4})
    return res.x, -res.fun


# ---------------------------------------------------------------- estimate
params = {}
res_rows = []

for b, name in enumerate(BAND_NAMES):
    tr = gt[gt.band == b]
    tr = tr[tr.date.isin(tr_days)]
    O, Dj, OD = band_stats(tr)
    A = smooth_attraction(Dj)
    beta, ll = fit_beta(OD, A)
    # baselines on the same training data
    _, ll_att = fit_beta(OD, A, lo=1e-6, hi=1e-6) if False else (None, None)
    print(f"band {name}: beta={beta:.3f} /km, loglik={ll:.1f}, trips={len(tr)}")

    # validation on held-out days
    te = gt[(gt.band == b) & gt.date.isin(te_days)]
    Ote, Dte, ODte = band_stats(te)
    m = ODte > 0
    ii, jj = np.where(m)
    cnt = ODte[m]

    def score(pred_logp_fn, k=5):
        """mean log-lik per trip and top-k hit rate (memory-safe)."""
        logp = pred_logp_fn()
        w = logp[ii, jj]
        ll_tot = float((cnt * w).sum())
        tot = cnt.sum()
        hit = 0.0
        for i0 in np.unique(ii):
            sel = ii == i0
            kk = min(k, logp.shape[1])
            top = np.argpartition(-logp[i0], kk - 1)[:kk]
            hit += cnt[sel][np.isin(jj[sel], top)].sum()
        return ll_tot / tot, hit / tot

    def norm_logp(logW):
        logW = logW.astype(np.float32)
        mx = logW.max(1, keepdims=True)
        E = np.exp(logW - mx)
        return logW - mx - np.log(E.sum(1, keepdims=True))

    def model_w():
        logW = -beta * D + np.log(np.maximum(A, 1e-12)).astype(np.float32)[None, :]
        return norm_logp(logW)

    def uniform_w():
        return np.full((n, n), -np.log(n), dtype=np.float32)

    def att_w():
        return norm_logp(np.tile(np.log(np.maximum(A, 1e-12)).astype(np.float32), (n, 1)))

    def dist_w():
        return norm_logp(-beta * D)

    ll_m, hit_m = score(model_w)
    ll_u, hit_u = score(uniform_w, k=5)
    ll_a, hit_a = score(att_w)
    ll_d, hit_d = score(dist_w)
    res_rows.append(dict(band=name, trips=len(te), beta=beta,
                         ll_model=ll_m, ll_attraction=ll_a, ll_distance=ll_d,
                         ll_uniform=ll_u, top5_model=hit_m, top5_att=hit_a,
                         top5_dist=hit_d))
    params[name] = dict(beta=float(beta), loglik=float(ll), n_train=int(len(tr)))
    print(f"   held-out: model {ll_m:.3f} | attraction-only {ll_a:.3f} | "
          f"distance-only {ll_d:.3f} | uniform {ll_u:.3f}")
    print(f"   top-5: model {hit_m:.3f} | att {hit_a:.3f} | dist {hit_d:.3f}")

# ------------------------------------------------------- generation model
sub = gt[(gt.date.isin(lib.TYP_DAYS))]
prof96 = np.zeros(96)
np.add.at(prof96, sub.qtr.values.astype(int), 1.0)
prof96 /= len(lib.TYP_DAYS)
daily = len(sub) / len(lib.TYP_DAYS)
# spatial intensity by hour band (smoothed)
rho = {}
for b, name in enumerate(BAND_NAMES):
    s = sub[sub.band == b]
    cnt = np.bincount(s.o.values, minlength=n).astype(float)
    rho[name] = (smooth_attraction(cnt, 1.0) / cnt.sum()).tolist()

out = dict(cells=int(n), daily_trips=float(daily), bands=params,
           band_names=BAND_NAMES, profile96=prof96.tolist(),
           rho=rho, step=lib.STEP, thr=lib.THR)
json.dump(out, open(DATA + r"\q1_params.json", "w"), indent=1)

pd.DataFrame(res_rows).to_csv(DATA + r"\q1_validation.csv", index=False)
np.save(DATA + r"\dist_matrix.npy", D.astype(np.float32))
cells.to_parquet(DATA + r"\cells.parquet")
print("\nsaved q1_params.json, q1_validation.csv")
print(pd.DataFrame(res_rows).round(3).to_string(index=False))
