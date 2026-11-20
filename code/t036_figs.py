"""T036: figures rebuilt on the conserving model.

  fig4_q2.pdf    panels (a)-(c) unchanged; (d) the joint fleet curve of T035
  fig5_q3.pdf    Q3 payment-ledger ladders (T031) and total against reward
  fig6_q4.pdf    Q4 conserved frontier and the knee plan's replay profile

Reads verified result files only.  All text at 8 pt or larger.
"""
import argparse
import csv
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib

FIG = lib.FIG
RUN18 = os.path.join(lib.DATA, "runs", "T018")
RUN28 = os.path.join(lib.DATA, "runs", "T028")
RUN30 = os.path.join(lib.DATA, "runs", "T030")
RUN31 = os.path.join(lib.DATA, "runs", "T031")
RUN35 = os.path.join(lib.DATA, "runs", "T035")

PNG = {"dir": None}


def save(fig, name):
    out = os.path.join(FIG, name)
    fig.savefig(out, bbox_inches="tight", **({"dpi": 300} if name ==
                                             "fig4_q2.pdf" else {}))
    if PNG["dir"]:
        fig.savefig(os.path.join(PNG["dir"], name.replace(".pdf", ".png")),
                    bbox_inches="tight", dpi=150)
    print("wrote", out)

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


# --------------------------------------------------------------------------
def fig_q2():
    z = np.load(os.path.join(lib.DATA, "typical_day.npz"), allow_pickle=True)
    od = z["od_z"].astype(float)
    cent = z["cent"]
    n = cent.shape[0]

    net = od[7:9].sum(axis=0).sum(axis=1) - od[7:9].sum(axis=0).sum(axis=0)
    assert abs(net.sum()) < 1e-6

    sched = list(csv.DictReader(open(os.path.join(RUN18,
                                                  "schedule_K60.csv"))))
    mat = np.zeros((24, n))
    for r in sched:
        mat[int(r["hour"]), int(r["to_zone"])] += float(r["unload_bikes"])
    order = np.argsort(mat.sum(axis=0))[::-1]
    mat = mat[:, order]

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

    cur = load_json(os.path.join(RUN35, "joint_curve.json"))
    ls = load_json(os.path.join(RUN28, "local_search.json"))
    fx = np.array([r["n"] for r in cur["rows"]])
    fy = np.array([r["cost"] for r in cur["rows"]]) / 1e3

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    a, bb, c, d = axes.ravel()

    v = np.percentile(np.abs(net), 95)
    sc = a.scatter(cent[:, 0], cent[:, 1], c=net, cmap="RdBu_r", vmin=-v,
                   vmax=v, s=34, edgecolor="white", linewidth=0.4, zorder=2)
    a.set_xlim(*np.percentile(cent[:, 0], [0.5, 99.5]))
    a.set_ylim(*np.percentile(cent[:, 1], [0.5, 99.5]))
    a.set_xlabel("x (km)")
    a.set_ylabel("y (km)")
    a.set_title("(a) net inflow, 07:00--09:00", loc="left")
    cb = fig.colorbar(sc, ax=a, fraction=0.045, pad=0.02)
    cb.set_label("bicycles (red = gain)")
    cb.ax.tick_params(labelsize=8)

    from matplotlib.colors import LogNorm
    mm = np.ma.masked_where(mat <= 0, mat)
    im = bb.imshow(mm.T, aspect="auto", origin="lower",
                   interpolation="nearest",
                   extent=[-0.5, 23.5, -0.5, n - 0.5],
                   norm=LogNorm(vmin=1, vmax=mat.max()), cmap="viridis")
    bb.set_xlabel("hour")
    bb.set_ylabel("receiving zone (ranked)")
    bb.set_title("(b) bicycles delivered, $K=60$", loc="left")
    cb = fig.colorbar(im, ax=bb, fraction=0.045, pad=0.02)
    cb.set_label("bicycles (log scale)")
    cb.ax.tick_params(labelsize=8)

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

    d.plot(fx, fy, "-o", color=C_A, ms=3.2, lw=1.2, label="joint conserving replay")
    d.plot([ls["exec_n"]], [ls["exec_cost"] / 1e3], "s", ms=6.5, mfc="none",
           mec=C_B, mew=1.3, label="executable, %d" % ls["exec_n"])
    d.axhline(ls["exec_cost"] / 1e3, color=C_X, lw=0.8, ls=":")
    d.set_xlabel("tricycles")
    d.set_ylabel("total cost (thousand CNY/day)")
    d.set_title("(d) tricycle layer vs fleet", loc="left")
    d.legend(loc="upper right", frameon=False, fontsize=8)

    fig.tight_layout(h_pad=1.0, w_pad=0.8)
    save(fig, "fig4_q2.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_q3():
    d = load_json(os.path.join(RUN31, "truck_payment.json"))
    lv = d["levels"]
    unit, N = d["truck_unit"], d["N"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    a, b = axes

    for key, lab, col, ls_ in (("arrivals", "arrivals cap", C_A, "-"),
                               ("surplus", "surplus cap", C_B, "--")):
        rows = lv[key]["rows"]
        rr = np.array([x["r"] for x in rows])
        tot = np.array([x["payment"] + unit * (N - x["q"]) for x in rows])
        pay = np.array([x["payment"] for x in rows])
        feas = pay <= d["budget"] + 1e-9
        a.plot(rr[feas], tot[feas] / 1e3, ls_, color=col, lw=1.3, label=lab)
        a.plot(rr[~feas], tot[~feas] / 1e3, ":", color=col, lw=1.0)
        b.plot(rr, pay / 1e3, ls_, color=col, lw=1.2, label=lab)
        best = lv[key]["best"]
        a.plot([best["r"]], [best["total_spend"] / 1e3], "o", ms=5.5,
               mfc="none", mec=col, mew=1.3)
        b.plot([best["r"]], [best["payment"] / 1e3], "o", ms=5.5,
               mfc="none", mec=col, mew=1.3)

    a.axhline(lv["arrivals"]["all_truck_cost"] / 1e3, color=C_X, lw=0.8, ls=":")
    a.annotate("all-truck", xy=(0.05, lv["arrivals"]["all_truck_cost"] / 1e3),
               fontsize=8, color=C_X, va="bottom")
    a.set_xlabel("reward $r$ (CNY)")
    a.set_ylabel("total spend (thousand CNY/day)")
    a.set_title("(a) total: payment + truck residual", loc="left")
    a.set_xlim(0, 1.6)
    a.legend(loc="lower left", frameon=False, fontsize=8)
    a.text(1.05, lv["arrivals"]["all_truck_cost"] / 1e3 - 8,
           "dotted: budget-infeasible", fontsize=7, color=C_X)

    b.axhline(d["budget"] / 1e3, color=C_X, lw=0.8, ls=":")
    b.annotate("budget 30,000", xy=(1.15, d["budget"] / 1e3), fontsize=8,
               color=C_X, va="bottom")
    b.set_xlabel("reward $r$ (CNY)")
    b.set_ylabel("payment ledger $rP$ (thousand CNY/day)")
    b.set_title("(b) payment against the budget", loc="left")
    b.set_xlim(0, 1.6)
    b.legend(loc="upper left", frameon=False, fontsize=8)

    fig.tight_layout(h_pad=1.0, w_pad=1.2)
    save(fig, "fig_q3_cost_vs_r.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_q4():
    fr = load_json(os.path.join(RUN30, "frontier_cons.json"))
    rp = load_json(os.path.join(RUN30, "replay_cons.json"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    a, b = axes

    for lv, lab, col, mk in (("meas343", "343,638 inventory", C_A, "o"),
                             ("fleet495", "495,540 inventory", C_B, "s")):
        pts = fr["levels"][lv]["points"]
        n = np.array([p["n"] for p in pts])
        cov = np.array([p["coverage"] for p in pts]) * 100
        a.plot(n, cov, mk + "-", color=col, ms=3.2, lw=1.2, label=lab)
        knee = fr["levels"][lv]["knee"]
        a.plot([knee["n"]], [knee["coverage"] * 100], mk, ms=8, mfc="none",
               mec=col, mew=1.4)
    a.set_xlabel("fences $n$")
    a.set_ylabel("occupancy coverage (\\%)")
    a.set_title("(a) capped frontier, 150 slots", loc="left")
    a.legend(loc="lower right", frameon=False, fontsize=8)

    u = np.array(rp["levels"]["meas343"]["design_15"]["unmet_h"])
    o = np.array(rp["levels"]["meas343"]["design_15"]["overflow_h"])
    hours = np.arange(24)
    b.bar(hours - 0.2, u / 1e3, width=0.4, color=C_A, label="unmet")
    b.bar(hours + 0.2, o / 1e3, width=0.4, color=C_B, label="overflow")
    b.set_xlabel("hour")
    b.set_ylabel("thousand bicycles")
    b.set_title("(b) replay at 343,638", loc="left")
    b.set_xlim(-0.6, 23.6)
    b.legend(loc="upper left", frameon=False, fontsize=8)

    fig.tight_layout(h_pad=1.0, w_pad=1.2)
    save(fig, "fig6_q4.pdf")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all",
                    choices=("q2", "q3", "q4", "all"))
    ap.add_argument("--pngdir", default=None)
    x = ap.parse_args()
    PNG["dir"] = x.pngdir
    if x.which in ("q2", "all"):
        fig_q2()
    if x.which in ("q3", "all"):
        fig_q3()
    if x.which in ("q4", "all"):
        fig_q4()
