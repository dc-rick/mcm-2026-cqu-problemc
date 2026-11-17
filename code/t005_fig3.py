"""T005: redraw Q1 validation figure 3 with the full model.

Panel (a): a synthetic day drawn from the full model (A included, all five
bands, beta from q1_params.json) against held-out observed trips.  Cross-cell
trips only; same-cell trips are counted separately and never assigned a
fabricated 0.30 km.  Distances are cell-centre distances on both sides for an
apples-to-apples comparison; the observed GPS curve is shown as a reference.
Panel (b): full-model beta vs the distance-only MLE beta.
Panel (c): difference plot, nlp(model) - nlp(distance-only), linear axis
(replaces the old log-scale panel that flattened the differences).
Panel (d): top-5 hit rate, full vs distance-only.

Also reproduces the old fig3a (A=1, morning only, 0.30 km same-cell, GPS
observed) to quantify how far the published KS moved.
"""
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import lib  # noqa: E402
from lib import COL, style  # noqa: E402
import t005_common as C  # noqa: E402

RUN = os.path.join(ROOT, "data", "runs", "T005")
os.makedirs(RUN, exist_ok=True)
style()
rng = np.random.default_rng(20260517)

t0 = time.time()
df, gt, cells, D = C.load_model_data()
n = len(cells)
q1 = json.load(io.open(os.path.join(lib.DATA, "q1_params.json"),
                       encoding="utf-8"))
dv = json.load(io.open(os.path.join(RUN, "dest_validation.json"),
                       encoding="utf-8"))
band_val = {b["band"]: b for b in dv["day_split"]["bands"]}


def sample_dest(logp, counts):
    ii = np.repeat(np.arange(n), counts)
    jj = np.empty(len(ii), dtype=np.int32)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    for i in np.where(counts > 0)[0]:
        c = counts[i]
        p = np.exp(logp[i].astype(np.float64))
        cs = np.cumsum(p)
        u = rng.random(c)
        j = np.searchsorted(cs, u * cs[-1], side="right")
        jj[starts[i]:starts[i] + c] = np.minimum(j, n - 1)
    return ii, jj


bands = {}
syn_dist, syn_same = [], 0
obs_dist, obs_gps, obs_same = [], [], 0
for b, name in enumerate(C.BAND_NAMES):
    tr = gt[(gt.band == b) & gt.date.isin(C.TR_DAYS)]
    te = gt[(gt.band == b) & gt.date.isin(C.TE_DAYS)]
    Otr, Djtr, _ = C.band_stats(tr, n)
    A = C.smooth_attraction(Djtr, cells, sigma=C.SIGMA_A)
    rho = C.smooth_attraction(Otr, cells, sigma=C.SIGMA_RHO)
    rho = rho / rho.sum()
    beta = float(q1["bands"][name]["beta"])
    logp = C.logp_full(beta, A, D)
    n_syn = int(round(len(te) / len(C.TE_DAYS)))
    counts = rng.multinomial(n_syn, rho)
    ii, jj = sample_dest(logp, counts)
    same = ii == jj
    syn_same += int(same.sum())
    syn_dist.append(D[ii[~same], jj[~same]].astype(np.float64))
    model_same = float(np.dot(rho, np.exp(
        logp[np.arange(n), np.arange(n)].astype(np.float64))))

    o = te.o.values
    d = te.d.values
    same_o = o == d
    obs_same += int(same_o.sum())
    obs_dist.append(D[o[~same_o], d[~same_o]].astype(np.float64))
    gps = C.haversine(te.start_lat.values[~same_o], te.start_lon.values[~same_o],
                      te.end_lat.values[~same_o], te.end_lon.values[~same_o])
    obs_gps.append(gps)
    ks_b = ks_2samp(obs_dist[-1], syn_dist[-1]).statistic
    bands[name] = dict(
        beta_full=beta, beta_distance=band_val[name]["beta_distance"],
        n_test_cross=int((~same_o).sum()), n_syn_cross=int((~same).sum()),
        n_syn_total=n_syn,
        observed_same_cell_share=float(same_o.mean()),
        model_implied_same_cell_share=model_same,
        synthetic_realised_same_cell_share=float(same.mean()),
        ks_cross_cell=float(ks_b),
        ks_cross_cell_vs_gps=float(ks_2samp(gps, syn_dist[-1]).statistic),
    )
    print(name, "KS", round(ks_b, 4), "same obs/model",
          round(float(same_o.mean()), 4), round(model_same, 4),
          f"{time.time()-t0:.0f}s", flush=True)

syn_all = np.concatenate(syn_dist)
obs_all = np.concatenate(obs_dist)
gps_all = np.concatenate(obs_gps)
ks_all = float(ks_2samp(obs_all, syn_all).statistic)
ks_all_gps = float(ks_2samp(gps_all, syn_all).statistic)
n_syn_total = int(sum(bands[b]["n_syn_total"] for b in C.BAND_NAMES))
print("pooled KS cell-centre", ks_all, "vs GPS", ks_all_gps)

# --------------------------------------------- old fig3a reproduction
rho_old = np.array(q1["rho"]["morning 6-9"], dtype=float)
rho_old /= rho_old.sum()
beta_old = float(q1["bands"]["morning 6-9"]["beta"])
rng_old = np.random.default_rng(0)
o_old = rng_old.choice(n, 12000, p=rho_old)
sims_old = np.empty(12000)
for t, i in enumerate(o_old):
    w = np.exp(-beta_old * D[i])
    w /= w.sum()
    j = rng_old.choice(n, p=w)
    sims_old[t] = D[i, j] if i != j else 0.30
df_old = lib.load_trips(lib.TYP_DAYS)
p1 = np.radians(df_old.start_lat.values)
p2 = np.radians(df_old.end_lat.values)
dl = np.radians((df_old.end_lon - df_old.start_lon).values)
dp = p2 - p1
aa = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
dob_old = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(aa, 0, 1)))
ks_old = float(ks_2samp(dob_old, sims_old).statistic)
print("old fig3a KS reproduced", ks_old, f"{time.time()-t0:.0f}s")

obs_share = float(obs_same / (obs_same + len(obs_all)))
syn_share = float(syn_same / n_syn_total)
model_share = float(np.average(
    [bands[b]["model_implied_same_cell_share"] for b in C.BAND_NAMES],
    weights=[bands[b]["n_syn_total"] for b in C.BAND_NAMES]))

out = dict(
    synthetic_day=dict(seed=20260517, bands=C.BAND_NAMES,
                       trips=n_syn_total,
                       band_trips={k: bands[k]["n_syn_total"] for k in bands},
                       parameters="beta from data/q1_params.json (fitted on "
                                  "5 training days); A from training arrival "
                                  "field, sigma=1.2; origins from training "
                                  "origin field, sigma=1.0",
                       observed="held-out test days " + ",".join(C.TE_DAYS)),
    distances="cell-centre Euclidean distance for both observed and synthetic "
              "(data/dist_matrix.npy); observed GPS haversine shown as "
              "reference only",
    same_cell=dict(
        definition="origin cell == destination cell; excluded from all "
                   "distance panels and never assigned a fabricated distance",
        observed_share=obs_share,
        synthetic_share=syn_share,
        model_implied_share=model_share,
        by_band={b: dict(
            observed=bands[b]["observed_same_cell_share"],
            model_implied=bands[b]["model_implied_same_cell_share"],
            synthetic=bands[b]["synthetic_realised_same_cell_share"])
            for b in C.BAND_NAMES}),
    ks=dict(pooled_cross_cell=ks_all,
            pooled_cross_cell_vs_gps=ks_all_gps,
            by_band={b: dict(cell_centre=bands[b]["ks_cross_cell"],
                             vs_gps=bands[b]["ks_cross_cell_vs_gps"])
                     for b in C.BAND_NAMES},
            n_observed=len(obs_all), n_synthetic=len(syn_all)),
    old_fig3a=dict(procedure="90_figs.py:131-143 reproduction: morning band "
                             "only, A=1, 12,000 draws, same-cell forced to "
                             "0.30 km, observed GPS distances over the 7 "
                             "typical days",
                   ks=ks_old, synthetic_same_cell_share=float(
                       np.mean(sims_old == 0.30))),
    bands=bands,
    caption_suggestion=(
        "Panel (a) compares cross-cell trips only: same-cell trips are "
        f"{100*obs_share:.1f}% of held-out observed trips and are excluded "
        "because the 0.005-degree cell model carries no intra-cell geometry. "
        f"The model implies {100*model_share:.1f}% same-cell trips; the "
        "synthetic day realises "
        f"{100*syn_share:.1f}%. Distances are cell-centre distances on both "
        f"sides; KS = {ks_all:.3f}. Panel (c) plots differences, not a log "
        "axis, so the distance-only baseline is not visually flattened."),
    runtime_s=float(time.time() - t0),
)
json.dump(out, io.open(os.path.join(RUN, "fig3_redraw.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(2, 2, figsize=(7.2, 4.8), constrained_layout=True)
a = ax[0, 0]
lo = max(0.32, float(np.percentile(obs_all, 0.1)))
hi = float(np.percentile(obs_all, 99.8))
xs = np.logspace(np.log10(lo), np.log10(hi), 300)
a.plot(xs, [(obs_all <= v).mean() for v in xs], color=COL["blue"], lw=1.5,
       label="observed, cell-centre")
a.plot(xs, [(syn_all <= v).mean() for v in xs], color=COL["orange"], lw=1.3,
       ls="--", label="full model, synthetic day")
a.plot(xs, [(gps_all <= v).mean() for v in xs], color=COL["grey"], lw=0.9,
       ls=":", label="observed, GPS")
a.set_xscale("log")
a.set_xlim(lo, hi)
a.set_ylim(0, 1)
a.set_xticks([0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0])
a.set_xticklabels(["0.5", "0.7", "1", "1.5", "2", "3", "5"])
a.minorticks_off()
a.set_xlabel("cross-cell trip distance (km)")
a.set_ylabel("cumulative share")
a.text(0.04, 0.80, f"KS = {ks_all:.3f}", transform=a.transAxes, fontsize=8)
a.text(0.04, 0.70,
       f"same-cell excluded: obs {100*obs_share:.1f}%"
       f" / model {100*model_share:.1f}%",
       transform=a.transAxes, fontsize=6.8, color=COL["dark"],
       bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                 boxstyle="round,pad=0.2"))
a.legend(loc="lower right", fontsize=6.8, frameon=False)
a.set_title("(a) full-model synthetic day vs held-out", loc="left")

b = ax[0, 1]
x = np.arange(len(C.BAND_NAMES))
b.bar(x - 0.18, [bands[k]["beta_full"] for k in C.BAND_NAMES], 0.34,
      color=COL["blue"], label="full (A + distance)")
b.bar(x + 0.18, [bands[k]["beta_distance"] for k in C.BAND_NAMES], 0.34,
      color=COL["grey"], label="distance only (own MLE)")
for i, k in enumerate(C.BAND_NAMES):
    b.text(i, max(bands[k]["beta_full"], bands[k]["beta_distance"]) + 0.05,
           f"{bands[k]['beta_distance']-bands[k]['beta_full']:+.3f}",
           ha="center", fontsize=6.5)
b.set_xticks(x)
b.set_xticklabels(["0-5", "6-9", "10-15", "16-19", "20-23"], fontsize=7.5)
b.set_xlabel("time band")
b.set_ylabel("distance decay (1/km)")
b.set_ylim(0, 3.0)
b.legend(fontsize=6.6, frameon=False, loc="upper left")
b.set_title("(b) beta: full vs distance-only MLE", loc="left")

c = ax[1, 0]
d_full = np.array([band_val[k]["delta_nlp_full_minus_distance"]
                   for k in C.BAND_NAMES])
ci = np.array([band_val[k]["delta_nlp_ci95"] for k in C.BAND_NAMES])
d_att = np.array([band_val[k]["nlp"]["attraction"]
                  - band_val[k]["nlp"]["distance"] for k in C.BAND_NAMES])
c.axhline(0, color=COL["dark"], lw=0.7)
c.bar(x, d_full, 0.5, color=COL["blue"], label="full - distance")
c.errorbar(x, d_full, yerr=[d_full - ci[:, 0], ci[:, 1] - d_full],
           fmt="none", ecolor=COL["dark"], elinewidth=0.8, capsize=2)
c.set_xticks(x)
c.set_xticklabels(["0-5", "6-9", "10-15", "16-19", "20-23"], fontsize=7.5)
c.set_xlabel("time band")
c.set_ylabel("nlp difference (nats/trip)")
c.set_ylim(min(ci[:, 0]) * 1.12, 0.015)
c.legend(fontsize=6.6, frameon=False, loc="lower right")
c.text(0.38, 0.62,
       "negative = better than distance-only\n"
       f"attraction - distance: +{d_att.min():.1f} to +{d_att.max():.1f} "
       "nats (off scale)\n"
       "uniform - distance: "
       f"{np.mean([band_val[k]['nlp']['uniform']-band_val[k]['nlp']['distance'] for k in C.BAND_NAMES]):.2f} nats",
       transform=c.transAxes, fontsize=6.3, color=COL["dark"],
       bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                 boxstyle="round,pad=0.2"))
c.set_title("(c) out-of-sample difference plot (linear)", loc="left")

d = ax[1, 1]
w = 0.34
d.bar(x - w / 2, [band_val[k]["top5"]["full"] for k in C.BAND_NAMES], w,
      color=COL["blue"], label="full")
d.bar(x + w / 2, [band_val[k]["top5"]["distance"] for k in C.BAND_NAMES], w,
      color=COL["grey"], label="distance only")
for i, k in enumerate(C.BAND_NAMES):
    d.text(i, band_val[k]["top5"]["full"] + 0.012,
           f"{band_val[k]['top5']['full']:.2f}", ha="center", fontsize=6.2)
d.set_xticks(x)
d.set_xticklabels(["0-5", "6-9", "10-15", "16-19", "20-23"], fontsize=7.5)
d.set_xlabel("time band")
d.set_ylabel("top-5 hit rate")
d.set_ylim(0, 0.58)
d.legend(fontsize=6.6, frameon=False, loc="upper right")
d.set_title("(d) top-5 hit rate (linear)", loc="left")

fig.savefig(os.path.join(RUN, "t005_fig3_redraw.pdf"))
fig.savefig(os.path.join(RUN, "t005_fig3_redraw.png"), dpi=300)
plt.close(fig)
print("saved figure and", os.path.join(RUN, "fig3_redraw.json"),
      f"{time.time()-t0:.0f}s")
