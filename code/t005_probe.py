"""Timing/numeral probe for T005 (one band)."""
import time

import numpy as np

import t005_common as C

t0 = time.time()
df, gt, cells, D = C.load_model_data()
n = len(cells)
print("loaded", len(gt), "n", n, "D", D.shape, f"{time.time()-t0:.1f}s")

tr = gt[(gt.band == 1) & gt.date.isin(C.TR_DAYS)]
te = gt[(gt.band == 1) & gt.date.isin(C.TE_DAYS)]
O, Dj, OD = C.band_stats(tr, n)
A = C.smooth_attraction(Dj, cells)
print("train", len(tr), "test", len(te), "A", A.min(), A.max())

t = time.time()
bf, llf = C.fit_beta(OD, A, D)
print("full beta", bf, "ll", llf, f"{time.time()-t:.2f}s")
t = time.time()
bd, lld = C.fit_beta(OD, np.ones(n), D)
print("dist beta", bd, "ll", lld, f"{time.time()-t:.2f}s")

Ote, Dte, ODte = C.band_stats(te, n)
for name, logp in (("full", C.logp_full(bf, A, D)),
                   ("dist", C.logp_dist(bd, D)),
                   ("att", C.logp_att(A, D)),
                   ("unif", C.logp_uniform(n))):
    t = time.time()
    nlp, hit, _, _ = C.score(logp, ODte, uniform=(name == "unif"))
    print(name, "nlp", round(nlp, 4), "top5", round(hit, 4),
          f"{time.time()-t:.2f}s")
