"""Probe fit-choice-set and top-5 tie conventions (morning band)."""
import numpy as np

import t005_common as C

df, gt, cells, D = C.load_model_data()
n = len(cells)
tr = gt[(gt.band == 1) & gt.date.isin(C.TR_DAYS)]
te = gt[(gt.band == 1) & gt.date.isin(C.TE_DAYS)]
O, Dj, OD = C.band_stats(tr, n)
A = C.smooth_attraction(Dj, cells)
Ote, Dte, ODte = C.band_stats(te, n)
print("train", len(tr), "test", len(te),
      "cols with train dest", int((OD.sum(0) > 0).sum()), "of", n,
      "A>0", int((A > 0).sum()))

for choice in ("paper", "all"):
    bf, lf = C.fit_beta(OD, A, D, choice=choice)
    bd, ld = C.fit_beta(OD, np.ones(n), D, choice=choice)
    print(choice, "bf", round(bf, 6), "bd", round(bd, 6),
          "diff", round(bd - bf, 6), "llf", round(lf, 1), "lld", round(ld, 1))
    for nm, logp in (("full", C.logp_full(bf, A, D)),
                     ("dist", C.logp_dist(bd, D)),
                     ("att", C.logp_att(A, D)),
                     ("unif", C.logp_uniform(n))):
        nlp, hit_ap, _, _ = C.score(logp, ODte, uniform=(nm == "unif"))
        hit_te, _ = C.score_expected_top5(logp, ODte)
        print(" ", nm, "nlp", round(nlp, 4), "top5 argpart", round(hit_ap, 4),
              "tie", round(hit_te, 4))
