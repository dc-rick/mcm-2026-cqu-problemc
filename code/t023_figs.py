"""T023 figures: tornado, variance shares, intervals.

Reads data/runs/T023/{tornado,variance_shares,intervals}.json and writes
vector PDFs to data/runs/T023/figs/.  All text at 8 pt or larger.
"""
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

RUN = os.path.join(lib.DATA, "runs", "T023")
FIGS = os.path.join(RUN, "figs")
os.makedirs(FIGS, exist_ok=True)

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
C_A = "#0072B2"
C_B = "#D55E00"
C_C = "#009E73"
C_X = "#7F7F7F"


def load(name):
    with io.open(os.path.join(RUN, name), encoding="utf-8") as fh:
        return json.load(fh)


def fig_tornado():
    d = load("tornado.json")
    panels = [("q2_total", "Q2 total cost (CNY/day)", 1.0),
              ("q3_ceiling", "Q3 payment ceiling (CNY/day)", 1.0),
              ("q4_coverage", "Q4 occupancy coverage (pp)", 100.0)]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.5),
                             gridspec_kw=dict(width_ratios=[1, 1, 1],
                                              wspace=0.62))
    for ax, (key, title, scale) in zip(axes, panels):
        rank = [r for r in d["rank"][key] if r["range"] > 0][:10]
        names = [r["param"] for r in rank][::-1]
        lo = np.array([d["bars"][n]["dev_low"][key] for n in names]) * scale
        hi = np.array([d["bars"][n]["dev_high"][key] for n in names]) * scale
        grp = [d["bars"][n]["group"] for n in names]
        cols = {"A": C_A, "B": C_B, "C": C_C}
        y = np.arange(len(names))
        for i in range(len(names)):
            ax.barh(y[i], lo[i], height=0.68, color=cols[grp[i]], alpha=0.55,
                    edgecolor="none")
            ax.barh(y[i], hi[i], height=0.68, color=cols[grp[i]],
                    edgecolor="none")
        ax.axvline(0.0, color="black", lw=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_title(title)
        if key != "q4_coverage":
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(
                    lambda v, _p: ("%.0fk" % (v / 1000.0))
                    if abs(v) >= 1000 else ("%.0f" % v)))
        ax.set_xlabel("anchored change vs reference")
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_A, label="A data"),
               plt.Rectangle((0, 0), 1, 1, color=C_B, label="B loads"),
               plt.Rectangle((0, 0), 1, 1, color=C_C, label="C constants")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(FIGS, "fig_tornado.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_shares():
    d = load("variance_shares.json")
    keys = [k for k in ("q2_total", "q2_unmet", "q3_pure_obj", "q3_ceiling",
                        "q4_coverage", "q4_arrival")
            if not d["outputs"][k].get("degenerate")]
    labels = {"q2_total": "Q2 total", "q2_unmet": "Q2 unmet",
              "q3_pure_obj": "Q3 pure obj", "q3_ceiling": "Q3 ceiling",
              "q4_coverage": "Q4 coverage", "q4_arrival": "Q4 arrival"}
    sa = np.array([d["outputs"][k]["share_first_order"]["A"] for k in keys])
    sb = np.array([d["outputs"][k]["share_first_order"]["B"] for k in keys])
    sc = np.array([d["outputs"][k]["share_first_order"]["C"] for k in keys])
    ea = np.array([d["outputs"][k]["share_se_boot"]["A"] for k in keys]) * 1.96
    eb = np.array([d["outputs"][k]["share_se_boot"]["B"] for k in keys]) * 1.96
    ec = np.array([d["outputs"][k]["share_se_boot"]["C"] for k in keys]) * 1.96
    ssum = np.array([d["outputs"][k]["sum_shares"] for k in keys])
    esum = np.array([d["outputs"][k]["sum_shares_se"] for k in keys]) * 1.96
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.bar(x, sa, color=C_A, label="A data-identified", width=0.62,
           yerr=ea, capsize=1.5, error_kw=dict(lw=0.7, ecolor="0.25"))
    ax.bar(x, sb, bottom=sa, color=C_B, label="B assumed loads", width=0.62,
           yerr=eb, capsize=1.5, error_kw=dict(lw=0.7, ecolor="0.25"))
    ax.bar(x, sc, bottom=sa + sb, color=C_C, width=0.62,
           label="C planning constants",
           yerr=ec, capsize=1.5, error_kw=dict(lw=0.7, ecolor="0.25"))
    ax.errorbar(x, ssum, yerr=esum, fmt="D", ms=3.2, color="black",
                lw=0.8, capsize=2,
                label=r"$\Sigma_g S_g$ (residual $= 1-\Sigma_g S_g$)")
    ax.axhline(1.0, color="0.35", lw=0.8, ls="--",
               label=r"additive reference ($\Sigma_g S_g = 1$)")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[k] for k in keys], rotation=15, ha="right")
    ax.set_ylabel("share of output variance")
    ax.set_ylim(0, max(1.45, float((ssum + esum).max()) * 1.25))
    ax.legend(frameon=False, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), fontsize=7)
    for i, k in enumerate(keys):
        ax.text(i, float(ssum[i] + esum[i]) + 0.03,
                "sd=%.3g" % np.sqrt(d["outputs"][k]["var_F"]),
                ha="center", va="bottom", fontsize=7, color="0.25")
    fig.tight_layout()
    out = os.path.join(FIGS, "fig_shares.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_intervals():
    d = load("intervals.json")
    keys = list(d["outputs"].keys())
    labels = [d["outputs"][k]["label"] for k in keys]
    slices = [("A", C_A, -0.18), ("B", C_B, -0.06), ("C", C_C, 0.06),
              ("F", "black", 0.18)]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y = np.arange(len(keys))[::-1]
    for sname, col, dx in slices:
        lo, med, hi = [], [], []
        for k in keys:
            ref = d["outputs"][k]["reference"]
            e = d["outputs"][k]["slices"][sname]
            scale = max(abs(ref), 1e-12)
            lo.append((e["p5"] - ref) / scale)
            med.append((e["p50"] - ref) / scale)
            hi.append((e["p95"] - ref) / scale)
        ypos = y + dx
        ax.hlines(ypos, np.array(lo) * 100, np.array(hi) * 100, color=col,
                  lw=1.6, label={"A": "A slice", "B": "B slice",
                                 "C": "C slice", "F": "all groups"}[sname])
        ax.plot(np.array(med) * 100, ypos, "o", ms=3.0, color=col,
                markerfacecolor="white", markeredgewidth=1.0)
    ax.axvline(0.0, color="black", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("P5 - P50 - P95 relative to the locked reference (%)")
    ax.legend(frameon=False, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    out = os.path.join(FIGS, "fig_intervals.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_tornado()
    fig_shares()
    fig_intervals()
