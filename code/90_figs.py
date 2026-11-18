"""All paper figures."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch
import lib
from lib import DATA, FIG, COL, save, style

style()
C = COL
SCALE = 1.0 / 0.128
cells = lib.get_cells()
xy = cells[["x", "y"]].to_numpy()
zid = cells.zone.to_numpy()
Z = int(zid.max()) + 1

# ===================================================== FIG 1 data overview
def fig1():
    df = lib.load_trips(lib.ALL_DAYS)
    daily = lib.load_trips(None).groupby("date").size()
    dates = sorted(daily.index)
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)

    a = ax[0, 0]
    full = pd.date_range("2017-05-10", "2017-05-24").strftime("%Y-%m-%d")
    x = np.arange(len(full))
    vals = [daily.get(d, 0) / 1000 for d in full]
    cols = [C["blue"] if d < "2017-05-20" else "#B9C4D0" for d in full]
    cols[full.get_loc("2017-05-17")] = "none"
    a.bar(x, vals, color=cols, width=0.72, edgecolor="none")
    a.axvline(9.5, color=C["red"], ls=":", lw=1)
    a.set_xticks(x[::2]); a.set_xticklabels([d[5:] for d in full][::2])
    a.set_ylabel("orders (thousand / day)"); a.set_xlabel("date, 2017")
    a.annotate("no records", xy=(7, 6), xytext=(4.6, 330), fontsize=8.6,
               color=C["dark"], ha="center",
               arrowprops=dict(arrowstyle="->", lw=.7, color=C["dark"]))
    a.annotate("sampling truncated", xy=(10.4, 90), xytext=(11.6, 210), fontsize=8.6,
               color=C["grey"], ha="center",
               arrowprops=dict(arrowstyle="->", lw=.7, color=C["grey"]))
    a.set_ylim(0, 360)
    a.set_title("(a) daily volume", loc="left")

    b = ax[0, 1]
    sub = df[df.date.isin(lib.ALL_DAYS[:10])]
    for k, (lab, c) in enumerate([("weekday", C["blue"]), ("weekend", C["orange"])]):
        m = (sub.starttime.dt.dayofweek >= 5) == (k == 1)
        h = sub[m].hour.value_counts().sort_index() / m.sum()
        b.plot(h.index, h.values * 100, marker="o", ms=2.6, lw=1.2, color=c, label=lab)
    b.set_xlabel("hour of day"); b.set_ylabel("share of daily trips (%)")
    b.legend(loc="upper left"); b.set_xlim(0, 23)
    b.set_title("(b) temporal profile", loc="left")

    c = ax[1, 0]
    R = 6371.0
    p1 = np.radians(sub.start_lat.values); p2 = np.radians(sub.end_lat.values)
    dl = np.radians((sub.end_lon - sub.start_lon).values); dp = p2 - p1
    aa = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    d = 2 * R * np.arcsin(np.sqrt(np.clip(aa, 0, 1)))
    xs = np.logspace(np.log10(0.12), np.log10(5), 300)
    c.plot(xs, [(d <= v).mean() for v in xs], color=C["blue"], lw=1.5, label="observed")
    from scipy.stats import lognorm
    sh, loc, sc = lognorm.fit(d, floc=0)
    c.plot(xs, lognorm.cdf(xs, sh, 0, sc), color=C["red"], lw=1.2, ls="--",
           label=f"lognormal ($\sigma$={sh:.2f}, median {sc:.2f} km)")
    c.set_xscale("log"); c.set_xlabel("straight-line trip distance (km)")
    c.set_ylabel("cumulative share"); c.legend(loc="lower right", fontsize=8.6)
    c.set_title("(c) trip length", loc="left")

    e = ax[1, 1]
    h = e.hist2d(sub.start_lon, sub.start_lat, bins=[240, 180], cmap="magma",
                 norm=LogNorm(), cmin=12)
    e.set_xlabel("longitude"); e.set_ylabel("latitude")
    e.set_title("(d) spatial density of trip starts", loc="left")
    cb = plt.colorbar(h[3], ax=e, fraction=0.046, pad=0.02)
    cb.set_label("trips per cell", fontsize=8.6)
    cb.ax.tick_params(labelsize=8.6)
    save(fig, "fig1_data")


# ===================================================== FIG 2 segments
def fig2():
    seg = pd.read_csv(os.path.join(DATA, "q1_segments.csv"))
    names = {4: "weekend / leisure", 1: "low frequency", 2: "commuter",
             0: "regular multi-purpose", 3: "heavy short-trip"}
    short = {4: "weekend", 1: "low-freq.", 2: "commuter",
             0: "multi-purpose", 3: "heavy short"}
    seg["name"] = seg.seg.map(names)
    seg = seg.sort_values("share_trips", ascending=False).reset_index(drop=True)
    fig = plt.figure(figsize=(7.2, 4.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)

    a = fig.add_subplot(gs[:, :2])
    y = np.arange(len(seg))
    a.barh(y - .18, seg.share_users * 100, height=.34, color=C["teal"],
           label="share of users")
    a.barh(y + .18, seg.share_trips * 100, height=.34, color=C["blue"],
           label="share of trips")
    a.set_yticks(y); a.set_yticklabels(seg.name, fontsize=8.6)
    a.set_xlabel("%"); a.legend(loc="lower right"); a.invert_yaxis()
    a.set_title("(a) segment size", loc="left")

    panels = [("dist", "mean trip distance (km)", "(b) distance"),
              ("peak", "peak-hour share", "(c) temporal concentration"),
              ("weekend", "weekend share", "(d) weekend share"),
              ("ret", "return-trip share", "(e) round trips")]
    cols = [C["blue"], C["orange"], C["green"], C["purple"], C["red"]]
    for k, (key, lab, title) in enumerate(panels):
        ax = fig.add_subplot(gs[k // 2, 2 + k % 2])
        ax.bar(np.arange(len(seg)), seg[key], color=cols, width=.62)
        ax.set_xticks(np.arange(len(seg)))
        ax.set_xticklabels([short[s] for s in seg.seg], rotation=32, ha="right",
                           fontsize=8.6)
        ax.set_ylabel(lab, fontsize=8.6)
        ax.set_title(title, loc="left", fontsize=8.6)
    save(fig, "fig2_segments")


# ===================================================== FIG 3 validation
def fig3():
    q1 = json.load(open(os.path.join(DATA, "q1_params.json"), encoding="utf-8"))
    val = pd.read_csv(os.path.join(DATA, "q1_validation.csv"))
    Dm = np.load(os.path.join(DATA, "dist_matrix.npy"))
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 4.6), constrained_layout=True)

    # (a) end-to-end: draw a synthetic day from the model, compare trip lengths
    a = ax[0, 0]
    rho = np.array(q1["rho"]["morning 6-9"], dtype=float); rho /= rho.sum()
    rng = np.random.default_rng(0)
    o = rng.choice(len(rho), 12000, p=rho)
    beta = q1["bands"]["morning 6-9"]["beta"]
    A = np.maximum(np.array(q1["bands"]["morning 6-9"]["beta"]) * 0 +
                   np.array([q1["bands"]["morning 6-9"]["beta"]]) * 0 +
                   np.asarray(sorted(q1["bounds"]) if False else [1.0]), 1e-9)
    sims = []
    for i in o:
        w = np.exp(-beta * Dm[i])
        w /= w.sum()
        j = rng.choice(len(w), p=w)
        sims.append(Dm[i, j] if i != j else 0.30)
    sims = np.array(sims)

    df = lib.load_trips(lib.TYP_DAYS)
    R = 6371.0
    p1 = np.radians(df.start_lat.values); p2 = np.radians(df.end_lat.values)
    dl = np.radians((df.end_lon - df.start_lon).values); dp = p2 - p1
    aa = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    dob = 2 * R * np.arcsin(np.sqrt(np.clip(aa, 0, 1)))
    xs = np.logspace(np.log10(0.12), np.log10(5), 250)
    a.plot(xs, [(dob <= v).mean() for v in xs], color=C["blue"], lw=1.5, label="observed")
    a.plot(xs, [(sims <= v).mean() for v in xs], color=C["orange"], lw=1.3, ls="--",
           label="model")
    from scipy.stats import ks_2samp
    ks = ks_2samp(dob, sims).statistic
    a.set_xscale("log"); a.set_xlabel("trip distance (km)")
    a.set_ylabel("cumulative share")
    a.text(0.35, 0.72, f"KS = {ks:.3f}", fontsize=7.5, transform=a.transAxes)
    a.legend(loc="lower right", fontsize=7)
    a.set_title("(a) end-to-end check", loc="left")

    # (b) beta by band
    b = ax[0, 1]
    names = list(q1["bands"])
    beta_v = [q1["bands"][k]["beta"] for k in names]
    b.bar(np.arange(len(names)), beta_v, color=C["teal"], width=.6)
    b.set_xticks(np.arange(len(names)))
    b.set_xticklabels(["0-5", "6-9", "10-15", "16-19", "20-23"], fontsize=7.5)
    b.set_xlabel("time band"); b.set_ylabel("distance decay (1/km)")
    for i, v in enumerate(beta_v):
        b.text(i, v + .03, f"{v:.2f}", ha="center", fontsize=7)
    b.set_title("(b) estimated deterrence", loc="left")

    # (c) out-of-sample log loss
    c = ax[1, 0]
    x = np.arange(len(val)); w = .26
    c.bar(x - w, -val.ll_distance, w, color=C["grey"], label="distance only")
    c.bar(x, -val.ll_model, w, color=C["blue"], label="gravity (ours)")
    c.bar(x + w, -val.ll_attraction, w, color=C["orange"], label="attraction only")
    c.set_xticks(x); c.set_xticklabels(["0-5", "6-9", "10-15", "16-19", "20-23"], fontsize=7.5)
    c.set_yscale("log"); c.set_ylabel("held-out log-loss / trip")
    c.set_ylim(2.6, 30)
    c.legend(fontsize=6.5, ncol=1, loc="upper right", frameon=False)
    c.set_title("(c) out-of-sample fit", loc="left")

    # (d) top-5
    d = ax[1, 1]
    d.bar(x - w / 2, val.top5_dist, w, color=C["grey"], label="distance only")
    d.bar(x + w / 2, val.top5_model, w, color=C["blue"], label="gravity (ours)")
    d.set_xticks(x); d.set_xticklabels(["0-5", "6-9", "10-15", "16-19", "20-23"], fontsize=7.5)
    d.set_ylabel("top-5 destination hit rate")
    d.set_ylim(0, 0.62)
    d.legend(fontsize=7, loc="upper center", ncol=2, frameon=False)
    d.set_title("(d) destination accuracy", loc="left")
    save(fig, "fig3_q1_validation")


# ===================================================== FIG 4 Q2
def fig4():
    z = np.load(os.path.join(DATA, "typical_day.npz"))
    cent = z["cent"]; od = z["od_z"] * SCALE
    plan = np.load(os.path.join(DATA, "q2_plan.npz")); XM = plan["XM"]
    res = pd.DataFrame(json.load(open(os.path.join(DATA, "q2_results.json"))))
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)

    a = ax[0, 0]
    arr = od[7:9].sum((0, 1)); dep = od[7:9].sum((0, 2))
    net = arr - dep
    a.scatter(xy[:, 0], xy[:, 1], s=0.7, c="#E4E8ED", edgecolor="none", zorder=0)
    v = np.percentile(np.abs(net), 95)
    sc = a.scatter(cent[:, 0], cent[:, 1], c=net, cmap="RdBu_r", vmin=-v, vmax=v,
                   s=6 + 190 * (arr + dep) / (arr + dep).max(),
                   edgecolor="#333333", linewidth=0.25, zorder=2, alpha=.9)
    a.set_xlim(*np.percentile(xy[:, 0], [0.5, 99.5]))
    a.set_ylim(*np.percentile(xy[:, 1], [0.5, 99.5]))
    a.set_xlabel("x (km, east of 116.0$^\\circ$E)")
    a.set_ylabel("y (km, north of 39.0$^\\circ$N)")
    a.set_title("(a) net flow 07-09", loc="left")
    cb = plt.colorbar(sc, ax=a, fraction=.046, pad=.02)
    cb.set_label("arrivals $-$ departures", fontsize=7); cb.ax.tick_params(labelsize=6)
    a.set_aspect("equal")

    b = ax[0, 1]
    deliv = XM.sum(1)                              # (T, zone) bikes delivered
    tot = deliv.sum(0)
    ordz = np.argsort(-tot)[:30].astype(int)       # 30 busiest receiving zones
    ordz = ordz[np.argsort(-tot[ordz])]
    im = b.imshow(deliv[:, ordz].T, aspect="auto", cmap="magma_r",
                  norm=LogNorm(vmin=max(1.0, deliv.max() * 1e-3), vmax=deliv.max()),
                  extent=[0, 24, 0, len(ordz)], origin="lower", interpolation="nearest")
    b.set_xlabel("hour"); b.set_ylabel("busiest 30 zones (ranked)")
    b.set_title("(b) relocation schedule", loc="left")
    cb = plt.colorbar(im, ax=b, fraction=.046, pad=.02)
    cb.set_label("bikes delivered / h", fontsize=7); cb.ax.tick_params(labelsize=6)

    c = ax[1, 0]
    tr = res[res["mode"] == "trucks"]
    for al, mk, col in ((1.0, "o", C["blue"]), (0.9, "s", C["orange"]),
                        (0.8, "^", C["green"]), (0.6, "d", C["red"])):
        s = tr[tr.alpha == al].sort_values("K")
        c.plot(s.K, s.unmet / s.trips * 100, marker=mk, ms=3, lw=1.1, color=col,
               label=f"fleet $\\times${al:.1f}")
    c.set_xlabel("trucks in service"); c.set_ylabel("unserved demand (%)")
    c.legend(fontsize=7); c.set_title("(c) service vs truck fleet", loc="left")
    c.set_ylim(-0.06, 5.2)
    c.text(0.36, 0.13, "$\\times$1.0 and $\\times$0.9: zero unmet at every $K$",
           transform=c.transAxes, fontsize=7, color=C["grey"])

    d = ax[1, 1]
    c_bike = 1.0
    best = []
    for al in sorted(tr.alpha.unique()):
        s = tr[tr.alpha == al]
        tot = c_bike * 343638 * al + s.obj
        k = int(s.K.iloc[np.argmin(tot.values)])
        best.append((al, c_bike * 343638 * al + tr[(tr.alpha == al) & (tr.K == k)].obj.iloc[0]))
    best = np.array(best)
    d.plot(best[:, 0], best[:, 1] / 1e3, "o-", color=C["blue"], ms=3.5, lw=1.3)
    k = int(np.argmin(best[:, 1]))
    d.scatter([best[k, 0]], [best[k, 1] / 1e3], s=60, facecolor="none",
              edgecolor=C["red"], lw=1.2, zorder=5)
    d.annotate(f"optimum $\\alpha$={best[k,0]:.1f}\n{best[k,1]/1e3:.0f}k CNY/day",
               xy=(best[k, 0], best[k, 1] / 1e3), xytext=(0.5, 0.45),
               textcoords="axes fraction", fontsize=7, color=C["red"],
               arrowprops=dict(arrowstyle="->", lw=.7, color=C["red"]))
    d.set_xlabel("fleet size relative to observed ($\\alpha$)")
    d.set_ylabel("daily cost incl. fleet capital ($10^3$ CNY)")
    d.set_ylim(300, 1160)
    d.set_title("(d) right-sizing the fleet", loc="left")
    save(fig, "fig4_q2")


# ===================================================== FIG 5 Q3
def fig5():
    q3 = json.load(open(os.path.join(DATA, "q3_summary.json")))
    cell = np.load(os.path.join(DATA, "q3_cell.npz"))
    need = q3["need_total"]
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 4.8), constrained_layout=True)
    V, W, K, RHO, C0 = 10.0, 80.0, 2.0, 0.35, 0.8

    def dt(r): return r / V * 60 + r * 1000 / W

    def pa(r, R): return 1 / (1 + np.exp(-(K * r - RHO * dt(R) - C0)))

    rs = np.linspace(0, 5, 200)
    a = ax[0, 0]
    for R, col, lab in ((0.45, C["blue"], "450 m"), (0.60, C["orange"], "600 m"),
                        (0.85, C["green"], "850 m")):
        a.plot(rs, pa(rs, R), color=col, lw=1.3, label=rf"detour {lab}")
    a.axvline(q3["opt"]["reward"], color=C["grey"], ls=":", lw=1)
    a.set_xlabel("reward (CNY)"); a.set_ylabel("acceptance probability")
    a.legend(fontsize=7); a.set_title("(a) logit response", loc="left")
    a.set_ylim(0, 1)

    b = ax[0, 1]
    sup = np.array(q3["supply"])
    b.plot(sup[:, 0], sup[:, 1] / 1e3, "o-", color=C["blue"], ms=3.5, lw=1.2)
    b.axhline(need / 1e3, color=C["grey"], ls="--", lw=1)
    b.text(1.9, need / 1e3 - 16, "peak-hour deficit", fontsize=7, color=C["grey"])
    b.annotate("pool exhausted", xy=(3.2, sup[-1, 1] / 1e3), xytext=(2.0, 175),
               fontsize=7, color=C["blue"],
               arrowprops=dict(arrowstyle="->", lw=.7, color=C["blue"]))
    b.scatter([q3["opt"]["reward"]], [q3["opt"]["q"] / 1e3], s=55,
              facecolor="none", edgecolor=C["red"], lw=1.2, zorder=5)
    b.annotate(f"$r^*$={q3['opt']['reward']:.2f} CNY\n{q3['opt']['q']/1e3:.0f}k placements",
               xy=(q3["opt"]["reward"], q3["opt"]["q"] / 1e3), xytext=(0.42, 0.12),
               textcoords="axes fraction", fontsize=7, color=C["red"],
               arrowprops=dict(arrowstyle="->", lw=.7, color=C["red"]))
    b.set_xlabel("uniform reward (CNY)"); b.set_ylabel("placements (10$^3$)")
    b.set_title("(b) supply of voluntary relocations", loc="left")

    c = ax[1, 0]
    base = q3["truck_unit"] * need / 1e3
    bill = q3["opt"]["bill"] / 1e3
    mix = [bill, q3["truck_unit"] * (need - q3["opt"]["q"]) / 1e3]
    c.bar([0, 1], [base, sum(mix)], color=[C["grey"], C["blue"]], width=.5)
    c.bar([1], [mix[0]], color=C["green"], width=.5)
    c.text(0, base * 1.02, f"{base:.0f}k", ha="center", fontsize=7.5)
    c.text(1, sum(mix) * 1.02, f"{sum(mix):.0f}k", ha="center", fontsize=7.5)
    c.text(1, mix[0] / 2, "incentive", ha="center", va="center", fontsize=7, color="white")
    c.text(1, mix[0] + mix[1] / 2, "trucks", ha="center", va="center", fontsize=7, color="white")
    c.set_xticks([0, 1]); c.set_xticklabels(["all trucks", "optimal mix"])
    c.set_ylabel("daily cost ($10^3$ CNY)")
    c.set_ylim(0, base * 1.14)
    c.text(0.55, 0.92, f"saving {100*(1-sum(mix)/base):.0f}%", transform=c.transAxes,
           ha="center", fontsize=7.5, color=C["red"])
    c.set_title("(c) cost of the two instruments", loc="left")

    d = ax[1, 1]
    d.plot(cell["radii"] * 1000, cell["curve"] * 100, color=C["blue"], lw=1.4)
    d.axhline(95, color=C["grey"], ls=":", lw=1)
    d.set_xlabel("catchment radius around a deficit cell (m)")
    d.set_ylabel("deficit coverable by incentives (%)")
    d.set_title("(d) what a walkable detour can absorb", loc="left")
    d.set_ylim(0, 102)
    save(fig, "fig5_q3")


# ===================================================== FIG 6 Q4
def fig6():
    q4 = json.load(open(os.path.join(DATA, "q4_summary.json")))
    q5 = json.load(open(os.path.join(DATA, "q5_summary.json")))
    fr = np.load(os.path.join(DATA, "q4_frontier.npz"))
    sites = fr["sites"]; caps = fr["caps"]
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    d3 = np.load(os.path.join(DATA, "q3_cell.npz")); Dday = d3["Dday"]

    a = ax[0, 0]
    order = np.argsort(-Dday)
    a.scatter(xy[order[800:], 0], xy[order[800:], 1], s=0.7, c="#E4E8ED", edgecolor="none")
    s = a.scatter(xy[sites, 0], xy[sites, 1], s=2.5 + 22 * np.log10(caps / caps.min()),
                  c=caps, cmap="plasma", norm=LogNorm(), edgecolor="none", alpha=.95)
    a.set_xlim(*np.percentile(xy[:, 0], [0.5, 99.5]))
    a.set_ylim(*np.percentile(xy[:, 1], [0.5, 99.5]))
    a.set_aspect("equal")
    a.set_xlabel("x (km, east of 116.0$^\\circ$E)")
    a.set_ylabel("y (km, north of 39.0$^\\circ$N)")
    a.set_title("(a) fence layout, sized by capacity", loc="left")
    cb = plt.colorbar(s, ax=a, fraction=.046, pad=.02)
    cb.set_label("slots", fontsize=7); cb.ax.tick_params(labelsize=6)

    b = ax[0, 1]
    for R, col, key in ((0.25, C["blue"], "c250"), (0.45, C["orange"], "c450"),
                        (0.65, C["green"], "c650")):
        cv = fr[key]
        b.plot(np.arange(1, len(cv) + 1), cv * 100, lw=1.3, color=col,
               label=rf"$R$ = {R*1000:.0f} m")
    b.set_xlabel("number of fences"); b.set_ylabel("arrivals covered (%)")
    b.legend(fontsize=7, loc="lower right"); b.set_title("(b) coverage", loc="left")
    b.set_ylim(0, 92)

    c = ax[1, 0]
    c.hist(caps, bins=np.logspace(0, np.log10(caps.max()), 40), color=C["blue"],
           alpha=.7, edgecolor="none")
    c.set_xscale("log"); c.set_xlabel("fence capacity (slots)"); c.set_ylabel("fences")
    c.axvline(150, color=C["red"], ls="--", lw=1)
    c.text(165, c.get_ylim()[1] * .78, "practical\nlimit", fontsize=7, color=C["red"])
    c.text(0.03, 0.90, f"median {np.median(caps):.0f}, max {caps.max():.0f}",
           transform=c.transAxes, fontsize=7)
    c.set_title("(c) capacity distribution", loc="left")

    e = ax[1, 1]
    vals = [q5["baseline_near_entrance"], q5["top300_near_entrance"],
            q5["share_arrivals_near_entrance"]]
    e.bar(["all cells", "top-300\nattractors", "trip\narrivals"],
          [100 * v for v in vals], color=[C["grey"], C["blue"], C["teal"]], width=.55)
    for i, v in enumerate(vals):
        e.text(i, 100 * v + 0.35, f"{100*v:.1f}%", ha="center", fontsize=7.5)
    e.set_ylabel("% within 300 m of a subway entrance")
    e.set_ylim(0, 20)
    e.set_title("(d) OSM cross-check", loc="left")
    save(fig, "fig6_q4")


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ["1", "2", "3", "4", "5", "6"]
    for t in todo:
        globals()[f"fig{t}"]()
