"""T012: regenerate figures 3-6 with the locked numbers.

All quantities are read from frozen result files; the only recomputation is the
Q1 synthetic day needed for figure 3(a), which mirrors code/t005_fig3.py.
"""
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LogNorm
from scipy.stats import ks_2samp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import lib  # noqa: E402
import t005_common as C  # noqa: E402

RUN = os.path.join(ROOT, "data", "runs", "T012")
os.makedirs(RUN, exist_ok=True)
lib.style()
rng = np.random.default_rng(20260517)

out = {}

# ======================================================================
# Figure 3 - Q1 validation
# ======================================================================
df, gt, cells, D = C.load_model_data()
n = len(cells)
q1 = json.load(io.open(os.path.join(lib.DATA, "q1_params.json"),
                       encoding="utf-8"))
dv = json.load(io.open(os.path.join(ROOT, "data", "runs", "T005",
                                     "dest_validation.json"), encoding="utf-8"))
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


syn_dist, obs_dist = [], []
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
    syn_dist.append(D[ii[~same], jj[~same]].astype(np.float64))
    o = te.o.values
    d = te.d.values
    same_o = o == d
    obs_dist.append(D[o[~same_o], d[~same_o]].astype(np.float64))
syn_all = np.concatenate(syn_dist)
obs_all = np.concatenate(obs_dist)
ks_all = float(ks_2samp(obs_all, syn_all).statistic)

fig, ax = plt.subplots(2, 2, figsize=(7.2, 4.9), constrained_layout=True)
a = ax[0, 0]
lo = max(0.32, float(np.percentile(obs_all, 0.1)))
hi = float(np.percentile(obs_all, 99.8))
xs = np.logspace(np.log10(lo), np.log10(hi), 300)
a.plot(xs, np.searchsorted(np.sort(obs_all), xs, side="right") / len(obs_all),
       color=lib.COL["blue"], lw=1.4, label="observed, held-out days")
a.plot(xs, np.searchsorted(np.sort(syn_all), xs, side="right") / len(syn_all),
       color=lib.COL["orange"], lw=1.4, ls="--", label="full model, synthetic day")
a.set_xscale("log")
a.set_xticks([0.5, 0.7, 1, 1.5, 2, 3, 5])
a.set_xticklabels(["0.5", "0.7", "1", "1.5", "2", "3", "5"])
a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a.set_xlim(lo, hi)
a.set_ylim(0, 1.02)
a.set_xlabel("cross-cell trip distance (km)")
a.set_ylabel("cumulative share")
a.set_title("(a) full-model synthetic day vs held-out", loc="left")
a.text(0.04, 0.95, f"KS = {ks_all:.3f}", transform=a.transAxes)
a.text(0.40, 0.32, "same-cell (full sample):\nmodel 12.83% / observed 8.89%",
       transform=a.transAxes, fontsize=7.5)
a.legend(loc="lower right")

b = ax[0, 1]
names = C.BAND_NAMES
bf = [band_val[k]["beta_full"] for k in names]
bd = [band_val[k]["beta_distance"] for k in names]
xx = np.arange(len(names))
b.bar(xx - 0.18, bf, 0.34, color=lib.COL["blue"], label="full")
b.bar(xx + 0.18, bd, 0.34, color=lib.COL["grey"], label="distance only")
for i, (f, d0) in enumerate(zip(bf, bd)):
    b.text(i, max(f, d0) + 0.05, f"+{f-d0:.3f}", ha="center", fontsize=7)
b.set_xticks(xx, ["0-5", "6-9", "10-15", "16-19", "20-23"])
b.set_ylabel("distance decay (1/km)")
b.set_title("(b) full vs distance-only MLE", loc="left")
b.set_ylim(0, max(bf + bd) * 1.28)
b.legend(loc="upper right")

c = ax[1, 0]
dnlp = [band_val[k]["nlp"]["full"] - band_val[k]["nlp"]["distance"]
        for k in names]
c.bar(xx, dnlp, 0.55, color=lib.COL["blue"])
c.axhline(0, color=lib.COL["dark"], lw=0.7)
c.set_xticks(xx, ["0-5", "6-9", "10-15", "16-19", "20-23"])
c.set_ylabel("log-loss difference (nats/trip)")
c.set_title("(c) out-of-sample difference, linear axis", loc="left")
c.text(0.30, 0.12, "negative = better than distance-only",
       transform=c.transAxes, fontsize=7.5)

d = ax[1, 1]
tf = [band_val[k]["top5"]["full"] for k in names]
td = [band_val[k]["top5"]["distance"] for k in names]
d.bar(xx - 0.18, tf, 0.34, color=lib.COL["blue"], label="full")
d.bar(xx + 0.18, td, 0.34, color=lib.COL["grey"], label="distance only")
for i, (f, d0) in enumerate(zip(tf, td)):
    d.text(i - 0.03, f + 0.01, f"{f:.2f}", ha="center", fontsize=7)
d.set_xticks(xx, ["0-5", "6-9", "10-15", "16-19", "20-23"])
d.set_ylabel("top-5 hit rate")
d.set_title("(d) top-5 hit rate, linear axis", loc="left")
d.set_ylim(0, 0.64)
d.legend(loc="upper right")

lib.save(fig, "fig3_q1_validation")
out["fig3"] = {"ks": ks_all, "same_cell_full": [0.12833723401989766,
                                                0.08886322430281249]}

# ======================================================================
# Figure 4 - Q2
# ======================================================================
od = np.load(os.path.join(ROOT, "data", "runs", "T006", "q1_od.npz"))["od_q1"]
plan = np.load(os.path.join(ROOT, "data", "runs", "T007", "q2_plan_q1.npz"))
q2 = json.load(io.open(os.path.join(ROOT, "data", "runs", "T007",
                                     "q2_results_q1.json"), encoding="utf-8"))
cz = pd.read_parquet(os.path.join(lib.DATA, "cells.parquet"))[["key", "zone"]]
cells_z = cells.merge(cz, on="key", how="left")
zones = cells_z.groupby("zone")[["x", "y"]].mean().sort_index()
zid = zones.index.values

morn = od[6:10].sum(axis=0)          # (i, j) over 06-09
arr_z = morn.sum(axis=0)
dep_z = morn.sum(axis=1)
net = arr_z - dep_z

fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
a = ax[0, 0]
v = np.abs(net).max()
norm = TwoSlopeNorm(vcenter=0, vmin=-v, vmax=v)
sc = a.scatter(zones.x, zones.y, s=8 + 55 * np.abs(net) / v,
               c=net, cmap="RdBu_r", norm=norm, edgecolor="none")
a.set_aspect("equal")
a.set_xlabel("east (km)")
a.set_ylabel("north (km)")
a.set_title("(a) morning net flow by zone, 06-09", loc="left")
cb = fig.colorbar(sc, ax=a, shrink=0.8, pad=0.02)
cb.set_label("arrivals - departures")

b = ax[0, 1]
X = plan["X"].T                      # (hour, zone)
im = b.imshow(X, aspect="auto", origin="lower", cmap="magma",
              extent=[0.5, 50.5, -0.5, 23.5])
b.set_xlabel("zone")
b.set_ylabel("hour")
b.set_title("(b) bicycles moved, $\\alpha=1$, $K=60$", loc="left")
cb = fig.colorbar(im, ax=b, shrink=0.9, pad=0.02)
cb.set_label("bicycles moved")

c = ax[1, 0]
ks = ["q1_a1_K20", "q1_a1_K40", "q1_a1_K60", "q1_a1_K80"]
kk = [20, 40, 60, 80]
cost = [q2["scenarios"][s]["obj"] for s in ks]
dev = [q2["scenarios"][s]["dev"] for s in ks]
c.plot(kk, cost, "o-", color=lib.COL["blue"], label="objective")
c.set_xlabel("trucks $K$")
c.set_ylabel("objective (CNY/day)")
c.set_title("(c) objective and end-of-day deviation", loc="left")
c2 = c.twinx()
c2.plot(kk, dev, "s--", color=lib.COL["orange"], label="deviation")
c2.set_ylabel("end-of-day deviation")
c2.set_yscale("log")
c2.set_ylim(60, 40000)
c.axvline(60, color=lib.COL["grey"], lw=0.6, ls=":")
c.text(0.03, 0.06, "$K=60$", transform=c.transAxes, fontsize=7.5)

d = ax[1, 1]
red = [("full", q2["scenarios"]["q1_a1_K80"]),
       ("$\\alpha=0.8$", q2["scenarios"]["q1_a0.8_K80"]),
       ("$\\alpha=0.6$", q2["scenarios"]["q1_a0.6_K80"])]
labels = [r[0] for r in red]
costs = [r[1]["obj"] for r in red]
unmet = [r[1]["unmet"] for r in red]
bars = d.bar(labels, costs, color=[lib.COL["blue"], lib.COL["orange"],
                                   lib.COL["red"]])
for bar, u in zip(bars, unmet):
    d.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
           f"unmet {u:,.0f}", ha="center", fontsize=7)
d.set_ylabel("objective (CNY/day)")
d.set_title("(d) three reduced-fleet points at $K=80$", loc="left")
d.set_ylim(0, max(costs) * 1.12)
d.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

lib.save(fig, "fig4_q2")
out["fig4"] = {"K": kk, "obj": cost, "dev": dev}

# ======================================================================
# Figure 5 - Q3
# ======================================================================
q3 = json.load(io.open(os.path.join(ROOT, "data", "runs", "T008",
                                     "q3_results.json"), encoding="utf-8"))
qr = json.load(io.open(os.path.join(ROOT, "data", "runs", "T008",
                                     "q3_qr.json"), encoding="utf-8"))
qrc = json.load(io.open(os.path.join(ROOT, "data", "runs", "T008",
                                      "q3_qr_conservative.json"), encoding="utf-8"))
acc = q3["settings"]["acceptance"]


def tau(d):
    return d / 10.0 * 60.0 + d * 1000.0 / 80.0


def pi(d, r):
    z = acc["kappa"] * r - acc["rho"] * tau(d) - acc["c0"]
    return 1.0 / (1.0 + np.exp(-z))


fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
a = ax[0, 0]
rr = np.linspace(0, 4, 400)
for d0, col in ((0.05, lib.COL["blue"]), (0.225, lib.COL["teal"]),
                (0.45, lib.COL["orange"])):
    a.plot(rr, pi(d0, rr), color=col, lw=1.4,
           label=f"{int(d0*1000)} m detour")
a.axvline(q3["targeted"]["r_star"], color=lib.COL["grey"], lw=0.6, ls=":")
a.set_xlabel("reward $r$ (CNY)")
a.set_ylabel("acceptance probability")
a.set_title("(a) acceptance vs reward", loc="left")
a.legend(loc="lower right")

b = ax[0, 1]
rg = np.array(qr["r_grid"])
b.plot(rg, np.array(qr["q"]), color=lib.COL["blue"], lw=1.4,
       label="upper: full arrivals")
b.plot(rg, np.array(qrc["q"]), color=lib.COL["orange"], lw=1.4,
       label="conservative: hourly surplus")
b.plot([qr["r_star"]], [qr["q_star"]], "o", color=lib.COL["blue"], ms=5)
b.plot([qrc["r_star"]], [qrc["q_star"]], "o", color=lib.COL["orange"], ms=5)
b.set_xlabel("reward $r$ (CNY)")
b.set_ylabel("placements $q(r)$")
b.set_title("(b) the two supply curves at 450 m", loc="left")
b.set_xlim(0, 4)
b.legend(loc="upper left")

c = ax[1, 0]
names = ["all truck", "upper\n$r^*=1.76$", "conservative\n$r^*=1.83$"]
vals = [q3["targeted"]["all_truck"], q3["targeted"]["total"],
        q3["targeted_conservative"]["total"]]
bars = c.bar(names, vals, color=[lib.COL["grey"], lib.COL["blue"],
                                 lib.COL["orange"]])
for bar, v0 in zip(bars, vals):
    c.text(bar.get_x() + bar.get_width() / 2, v0 + 12000, f"{v0:,.0f}",
           ha="center", fontsize=7)
c.set_ylabel("daily cost (CNY)")
c.set_title("(c) cost of the two optimal mixes", loc="left")
c.set_ylim(0, max(vals) * 1.12)

d = ax[1, 1]
zt = q3["zone_decomposition"]["targeted"]
zc = q3["zone_decomposition"]["targeted_conservative"]
labels = ["upper", "conservative"]
cross = [zt["cross_zone_volume"], zc["cross_zone_volume"]]
same = [zt["q"] - zt["cross_zone_volume"], zc["q"] - zc["cross_zone_volume"]]
xx = np.arange(2)
d.bar(xx, same, 0.5, color=lib.COL["blue"], label="same zone")
d.bar(xx, cross, 0.5, bottom=same, color=lib.COL["orange"],
      label="cross zone")
for i in range(2):
    d.text(i, same[i] + cross[i] + 1800,
           f"{100*cross[i]/(same[i]+cross[i]):.2f}% cross", ha="center",
           fontsize=7.5)
d.set_xticks(xx, labels)
d.set_ylabel("rerouted bicycles")
d.set_title("(d) zone decomposition of the optimum", loc="left")
d.set_ylim(0, (max(np.array(same) + np.array(cross))) * 1.15)
d.legend(loc="upper right")

lib.save(fig, "fig5_q3")
out["fig5"] = {"cross_share": [zt["cross_zone_share"],
                               zc["cross_zone_share"]]}

# ======================================================================
# Figure 6 - Q4
# ======================================================================
fen = pd.read_csv(os.path.join(ROOT, "data", "runs", "T009",
                               "q4_fences.csv"))
front = json.load(io.open(os.path.join(ROOT, "data", "runs", "T009",
                                       "q4_frontier.json"), encoding="utf-8"))
q5 = json.load(io.open(os.path.join(lib.DATA, "q5_summary.json"),
                       encoding="utf-8"))

fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
a = ax[0, 0]
sc = a.scatter(fen.x, fen.y, s=5 + 26 * fen.capacity_meas343 /
               fen.capacity_meas343.max(), c=fen.capacity_meas343,
               cmap="viridis", norm=LogNorm(), alpha=0.85,
               edgecolor="none")
a.set_aspect("equal")
a.set_xlabel("east (km)")
a.set_ylabel("north (km)")
a.set_title("(a) 705-fence knee layout, $R=450$ m", loc="left")
cb = fig.colorbar(sc, ax=a, shrink=0.85, pad=0.02)
cb.set_label("slots at 343,638")

b = ax[0, 1]
for key, col, lab in (("q1", lib.COL["blue"], "Q1 demand"),
                      ("obs", lib.COL["orange"], "observed demand")):
    pts = front[key]["points"]
    nf = np.array(pts["n_fences"])
    cov = np.array(pts["coverage"])
    b.plot(nf, 100 * cov, color=col, lw=1.3, label=lab)
    knee = front[key]["knees"]["meas343"]["150"]
    b.plot([knee["n"]], [100 * knee["coverage"]], "o", color=col, ms=5)
b.set_xlabel("number of fences")
b.set_ylabel("coverage of arrivals (%)")
b.set_title("(b) coverage frontier", loc="left")
b.legend(loc="lower right")

c = ax[1, 0]
bins = np.linspace(0, 1600, 60)
c.hist(fen.capacity_meas343, bins=bins, color=lib.COL["blue"], alpha=0.65,
       label="343,638 inventory")
c.hist(fen.capacity_fleet495, bins=bins, color=lib.COL["orange"], alpha=0.45,
       label="495,540 inventory")
c.axvline(150, color=lib.COL["red"], lw=1.0, ls="--", label="150-slot limit")
c.set_xlabel("fence capacity (slots, clipped at 1,600)")
c.set_ylabel("number of fences")
c.set_title("(c) capacity distribution of the knee", loc="left")
c.legend(loc="upper right")

d = ax[1, 1]
vals = [100 * q5["baseline_near_entrance"],
        100 * q5["top300_near_entrance"],
        100 * q5["share_arrivals_near_entrance"]]
names = ["all cells", "top-300\nattractors", "arrivals"]
bars = d.bar(names, vals, color=[lib.COL["grey"], lib.COL["blue"],
                                 lib.COL["teal"]])
for bar, v0 in zip(bars, vals):
    d.text(bar.get_x() + bar.get_width() / 2, v0 + 0.3, f"{v0:.1f}%",
           ha="center", fontsize=7.5)
d.set_ylabel("share within 300 m of a subway entrance (%)")
d.set_title("(d) OSM cross-check", loc="left")
d.set_ylim(0, max(vals) * 1.2)

lib.save(fig, "fig6_q4")
out["fig6"] = {"n": 705, "coverage": 0.7685847656798482,
               "osm": vals}

json.dump(out, io.open(os.path.join(RUN, "figures.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", os.path.join(RUN, "figures.json"))
