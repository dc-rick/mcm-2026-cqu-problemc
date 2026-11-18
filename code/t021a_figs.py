"""T021A figures: regenerated Q2 four-panel figure, Q3 cost-vs-reward, tornado copy.

Reads verified result files only:
  data/typical_day.npz                     (Q1 demand surface)
  data/runs/T018/schedule_K60.csv          (integer truck schedule)
  data/runs/T018/baselines.json            (integer K scan)
  data/runs/T024/compare.json              (tricycle layer vs fleet)
  data/runs/T026/cost_vs_r.json            (Q3 tricycle-caliber cost curve)
  data/runs/T026/optimum.json              (Q3 optimum and crossings)
  data/runs/T023/figs/fig_tornado.pdf      (already 8 pt; copied)

Writes figs/fig4_q2.pdf, figs/fig_q3_cost_vs_r.pdf, figs/fig9_tornado.pdf.
All text at 8 pt or larger (no mathtext superscripts).
"""
import csv
import io
import json
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib

ROOT = lib.ROOT
FIG = lib.FIG
RUN18 = os.path.join(lib.DATA, "runs", "T018")
RUN24 = os.path.join(lib.DATA, "runs", "T024")
RUN26 = os.path.join(lib.DATA, "runs", "T026")
RUN23 = os.path.join(lib.DATA, "runs", "T023")

C_A = "#0072B2"
C_B = "#D55E00"
C_C = "#009E73"
C_X = "#7F7F7F"

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def load_json(path):
    with io.open(path, encoding="utf-8") as fh:
        return json.load(fh)


def fig_q2():
    z = np.load(os.path.join(lib.DATA, "typical_day.npz"), allow_pickle=True)
    od = z["od_z"].astype(float)          # (24, 50, 50)
    cent = z["cent"]                       # (50, 2)
    n = cent.shape[0]

    # (a) net inflow 07:00-09:00
    net = od[7:9].sum(axis=0).sum(axis=1) - od[7:9].sum(axis=0).sum(axis=0)
    # sanity: total net across all zones is zero
    assert abs(net.sum()) < 1e-6

    # (b) delivered bicycles by hour and receiving zone
    sched = list(csv.DictReader(open(os.path.join(RUN18, "schedule_K60.csv"))))
    mat = np.zeros((24, n))
    for r in sched:
        mat[int(r["hour"]), int(r["to_zone"])] += float(r["unload_bikes"])
    order = np.argsort(mat.sum(axis=0))[::-1]
    mat = mat[:, order]

    # (c) integer K scan
    b = load_json(os.path.join(RUN18, "baselines.json"))
    ks = [0]
    objs = [b["rows"]["no_trucks"]["cost"]]
    unmet = [b["rows"]["no_trucks"]["unmet"]]
    for key in ("K20", "K40", "K60", "K80"):
        row = b["int_rows"][key]
        ks.append(row["K"])
        objs.append(row["obj"])
        unmet.append(row["unmet"])
    ks = np.asarray(ks)
    objs = np.asarray(objs) / 1e3
    unmet = np.asarray(unmet)

    # (d) tricycle layer: LP total against fleet, plus LP optimum and executable point
    cmp_ = load_json(os.path.join(RUN24, "compare.json"))
    rows = [r for r in cmp_["rows"] if r["fleet_N"] <= 800]
    fx = np.array([r["fleet_N"] for r in rows])
    fy = np.array([r["total"] for r in rows]) / 1e3
    lp_total = cmp_["truck_obj"] + cmp_["best_layer_cost"]
    lp_n = cmp_["best_used"]
    exec_total = 617055.7515153277
    exec_n = 975

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    a, bb, c, d = axes.ravel()

    # (a)
    v = np.percentile(np.abs(net), 95)
    sc = a.scatter(cent[:, 0], cent[:, 1], c=net, cmap="RdBu_r", vmin=-v, vmax=v,
                   s=34, edgecolor="white", linewidth=0.4, zorder=2)
    a.set_xlim(*np.percentile(cent[:, 0], [0.5, 99.5]))
    a.set_ylim(*np.percentile(cent[:, 1], [0.5, 99.5]))
    a.set_xlabel("x (km)")
    a.set_ylabel("y (km)")
    a.set_title("(a) net inflow, 07:00--09:00", loc="left")
    cb = fig.colorbar(sc, ax=a, fraction=0.045, pad=0.02)
    cb.set_label("bicycles (red = gain)")
    cb.ax.tick_params(labelsize=8)

    # (b)
    from matplotlib.colors import LogNorm
    mm = np.ma.masked_where(mat <= 0, mat)
    im = bb.imshow(mm.T, aspect="auto", origin="lower", interpolation="nearest",
                   extent=[-0.5, 23.5, -0.5, n - 0.5],
                   norm=LogNorm(vmin=1, vmax=mat.max()), cmap="viridis")
    bb.set_xlabel("hour")
    bb.set_ylabel("receiving zone (ranked)")
    bb.set_title("(b) bicycles delivered, $K=60$", loc="left")
    cb = fig.colorbar(im, ax=bb, fraction=0.045, pad=0.02)
    cb.set_label("bicycles (log scale)")
    cb.ax.tick_params(labelsize=8)

    # (c)
    c.plot(ks, objs, "-o", color=C_A, ms=4, lw=1.2, label="objective")
    c.set_xlabel("trucks $K$")
    c.set_ylabel("objective (thousand CNY/day)", color=C_A)
    c.tick_params(axis="y", labelcolor=C_A)
    c.set_title("(c) integer truck scan", loc="left")
    ax2 = c.twinx()
    ax2.plot(ks, unmet, "--s", color=C_B, ms=4, lw=1.2, label="zone unmet")
    ax2.set_ylabel("zone unmet (bicycles/day)", color=C_B)
    ax2.tick_params(axis="y", labelcolor=C_B)
    ax2.spines["right"].set_visible(True)
    c.axvline(60, color=C_X, lw=0.8, ls=":")
    c.annotate("K = 60", xy=(60, objs.min()), xytext=(64, objs.max() * 0.72),
               fontsize=8, color=C_X,
               arrowprops=dict(arrowstyle="-", lw=0.7, color=C_X))

    # (d)
    d.plot(fx, fy, "-o", color=C_A, ms=3.2, lw=1.2, label="LP relaxation")
    d.plot([lp_n], [lp_total / 1e3], "o", ms=7, mfc="none", mec=C_C, mew=1.3,
           label="LP optimum, 889")
    d.plot([exec_n], [exec_total / 1e3], "s", ms=6, mfc="none", mec=C_B, mew=1.3,
           label="executable, 975")
    d.axhline(exec_total / 1e3, color=C_X, lw=0.8, ls=":")
    d.set_xlabel("tricycles")
    d.set_ylabel("total cost (thousand CNY/day)")
    d.set_title("(d) tricycle layer vs fleet", loc="left")
    d.legend(loc="upper right", frameon=False, fontsize=8)

    fig.tight_layout(h_pad=1.0, w_pad=0.8)
    out = os.path.join(FIG, "fig4_q2.pdf")
    # R1/D6: panel (b) is a resampled imshow; save at 300 dpi so the embedded
    # raster is >=300 dpi at placement size (vector panels are unaffected).
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", out)


def fig_q3_cost():
    d = load_json(os.path.join(RUN26, "cost_vs_r.json"))
    opt = load_json(os.path.join(RUN26, "optimum.json"))
    base = d["baseline"]["total"] / 1e3
    r_star = opt["caps"]["arrivals"]["r_star"]
    crossings = {cap: opt["caps"][cap]["r_cross"] for cap in ("arrivals", "surplus")}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, xmax in zip(axes, (3.05, 1.85)):
        for cap, col, lab in (("arrivals", C_A, "arrivals cap"),
                              ("surplus", C_B, "surplus cap")):
            rows = [r for r in d["rows"][cap] if r["r"] <= xmax + 1e-9]
            x = np.array([r["r"] for r in rows])
            y = np.array([r["total"] for r in rows]) / 1e3
            ax.plot(x, y, "-", color=col, lw=1.2, label=lab)
        ax.axhline(base, color=C_X, lw=0.8, ls=":")
        ax.set_xlim(-0.02, xmax)
        ax.set_xlabel("reward $r$ (CNY per reroute)")
        ax.set_title("(a) full grid" if xmax > 2 else "(b) productive region",
                     loc="left")
        ax.legend(loc="upper left", frameon=False, fontsize=8)
    axes[0].set_ylabel("total cost (thousand CNY/day)")

    ax = axes[1]
    ax.set_ylim(578, 872)
    r_star_y = opt["caps"]["arrivals"]["total_star"] / 1e3
    ax.plot([r_star], [r_star_y], "o", ms=6, mfc="none",
            mec=C_A, mew=1.3, zorder=5)
    ax.annotate("r* = 0.60", xy=(r_star, r_star_y), xytext=(0.68, 588),
                fontsize=8, color=C_A,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=C_A))
    for cap, col in (("arrivals", C_A), ("surplus", C_B)):
        rc = crossings[cap]
        ax.axvline(rc, color=col, lw=0.8, ls="--")
        ax.text(rc - 0.03, 840, "%.2f" % rc, ha="right", va="top",
                fontsize=8, color=col)

    fig.tight_layout(w_pad=1.0)
    out = os.path.join(FIG, "fig_q3_cost_vs_r.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    fig_q2()
    fig_q3_cost()
    src = os.path.join(RUN23, "figs", "fig_tornado.pdf")
    dst = os.path.join(FIG, "fig9_tornado.pdf")
    shutil.copyfile(src, dst)
    print("copied", dst)


if __name__ == "__main__":
    main()
