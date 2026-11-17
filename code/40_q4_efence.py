"""Q4: electronic-fence layout and capacity planning.

Demand  : arrivals per cell per 15 min (Q1 model / data)
Siting  : greedy maximal coverage with walking radius R (fence = a cell site)
Sizing  : capacity = 95th percentile of the daily peak occupancy, obtained by
          Monte-Carlo on the (arrival - departure) random walk of the served set

--arr q1 (default) reads data/runs/T006/q1_od.npz (hourly, replicated to the
four 15-min slots of each hour); --arr obs uses the observed typical day.
Only the demand surface changes -- siting and sizing are untouched.  Results
go to --outdir (default data/runs/T006/q4_<arr>), so data/q4_* stays read-only.
"""
import argparse
import json
import os

import numpy as np

import lib
from lib import DATA

SCALE = 1.0 / 0.128
RUN = os.path.join(DATA, "runs", "T006")


def load_demand(kind):
    """Return arr/dep as (96, C) at the full-scale daily-average level."""
    if kind == "q1":
        z = np.load(os.path.join(RUN, "q1_od.npz"))
        # hourly totals -> four equal 15-min slots (divide, do not repeat the
        # hourly total four times: that would quadruple the daily demand)
        arr = np.repeat(z["arr_q1_cell"].astype(np.float64) / 4.0, 4, axis=0)
        dep = np.repeat(z["dep_obs"].astype(np.float64) / 4.0, 4, axis=0)
        arr = arr * SCALE
        dep = dep * SCALE
        return arr, dep

    cells = lib.get_cells()
    C = len(cells)
    df = lib.load_trips(lib.TYP_DAYS)
    df = lib.assign_cells(df, cells)
    ok = df.o.notna() & df.d.notna()
    s = df[ok].copy()
    s["o"] = s.o.astype(int)
    s["d"] = s.d.astype(int)
    nd = len(lib.TYP_DAYS)
    q = s.starttime.dt.hour.values * 4 + s.starttime.dt.minute.values // 15
    arr = np.zeros((96, C))
    dep = np.zeros((96, C))
    np.add.at(arr, (q, s.d.values), 1.0)
    np.add.at(dep, (q, s.o.values), 1.0)
    arr = arr / nd * SCALE
    dep = dep / nd * SCALE
    return arr, dep


def main():
    ap = argparse.ArgumentParser(description="Q4 fence siting and capacity")
    ap.add_argument("--arr", choices=("q1", "obs"), default="q1")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--radii", default="0.25,0.45,0.65")
    ap.add_argument("--reps", type=int, default=300)
    args = ap.parse_args()

    radii = [float(x) for x in args.radii.split(",") if x.strip()]
    outdir = args.outdir or os.path.join(RUN, "q4_" + args.arr)
    os.makedirs(outdir, exist_ok=True)

    cells = lib.get_cells()
    C = len(cells)
    xy = cells[["x", "y"]].to_numpy()
    dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    arr, dep = load_demand(args.arr)
    Ad = arr.sum(0)
    print(f"arrivals/day {Ad.sum():.0f}, cells {C}, source {args.arr}")

    # ---------- siting: greedy maximal coverage --------------------------
    def greedy_cover(R, Nmax, cand_top=2500, frac_stop=0.0005):
        """Greedy maximal coverage: pick the cell whose catchment contains the
        most still-uncovered arrivals.  Returns the full coverage path."""
        near = (dist <= R).astype(np.float32)
        cand = np.argsort(-Ad)[:cand_top]
        Nmat = near[cand]                       # (ncand, C)
        covered = np.zeros(C, bool)
        chosen = []
        gain_hist = []
        cov_hist = []
        for _ in range(Nmax):
            g = Nmat @ (Ad * (~covered))
            if gain_hist:
                g[chosen] = -1.0
            k = int(np.argmax(g))
            if g[k] <= frac_stop * Ad.sum():
                break
            chosen.append(k)
            covered |= near[cand[k]] > 0
            gain_hist.append(float(g[k]))
            cov_hist.append(float(Ad[covered].sum() / Ad.sum()))
        return [int(cand[k]) for k in chosen], covered, cov_hist

    n_max = {0.25: 2500, 0.45: 1500, 0.65: 900}
    results = {}
    curves = {}
    for R in radii:
        ch, cov, cvh = greedy_cover(R, n_max.get(R, 1500))
        share = Ad[cov].sum() / Ad.sum()
        print(f"R={R*1000:.0f} m: {len(ch)} fences cover {100*share:.1f}% of arrivals")
        results[R] = (ch, cov, share)
        curves[R] = np.array(cvh)
        if abs(R - 0.45) < 1e-9:
            np.save(os.path.join(outdir, "q4_sites.npy"), np.array(ch))

    # ---------- capacity sizing ------------------------------------------
    def peak_occupancy(served_cells, reps=300, seed=0):
        """Monte-Carlo peak occupancy of one fence serving `served_cells`."""
        rng = np.random.default_rng(seed)
        a = arr[:, served_cells].sum(1)
        d = dep[:, served_cells].sum(1)
        peaks = np.empty(reps)
        for _ in range(reps):
            occ = 0.0
            peak = 0.0
            ai = rng.poisson(a)
            di = rng.poisson(d)
            for t in range(96):
                occ = max(0.0, occ + ai[t] - di[t])
                peak = max(peak, occ)
            peaks[_] = peak
        return float(np.percentile(peaks, 95))

    caps_by_R = {}
    for R in radii:
        ch, cov, share = results[R]
        print(f"\ncapacity sizing for the R={R*1000:.0f} m plan ({len(ch)} fences)")
        caps = []
        for c in ch:
            members = np.where(dist[c] <= R)[0]
            caps.append(peak_occupancy(members, reps=args.reps))
        caps = np.array(caps)
        caps_by_R[R] = caps
        print(f"capacity: total {caps.sum():.0f} slots, median {np.median(caps):.0f}, "
              f"p90 {np.percentile(caps,90):.0f}, max {caps.max():.0f}")
        print(f"land at 1.4 m2/slot: {caps.sum()*1.4/1e4:.1f} ha")

    R_main = 0.45 if 0.45 in results else radii[-1]
    ch, cov, share = results[R_main]
    caps = caps_by_R[R_main]

    # ---------- coverage / cost frontier ---------------------------------
    frontier = []
    for R in radii:
        for N in (200, 400, 800, 1200, 2000, 2500):
            ch2 = results[R][0][:N]
            near = dist <= R
            covered = np.zeros(C, bool)
            for c in ch2:
                covered |= near[c]
            frontier.append((R, min(N, len(results[R][0])),
                             float(Ad[covered].sum() / Ad.sum())))
    fr = np.array(frontier)
    arrays = dict(fr=fr, caps=caps, sites=np.array(ch), share=share)
    for R in radii:
        arrays[f"c{int(round(R * 1000))}"] = curves[R]
    np.savez(os.path.join(outdir, "q4_frontier.npz"), **arrays)
    with open(os.path.join(outdir, "q4_summary.json"), "w",
              encoding="utf-8") as handle:
        json.dump(dict(arr=args.arr,
                       radii=radii,
                       reps=args.reps,
                       arrivals_per_day=float(Ad.sum()),
                       n_fences=len(ch),
                       coverage=float(share),
                       total_capacity=float(caps.sum()),
                       median_capacity=float(np.median(caps)),
                       land_ha=float(caps.sum() * 1.4 / 1e4),
                       by_radius={f"{k:.2f}": dict(
                           fences=len(results[k][0]),
                           coverage=float(results[k][2]),
                           total=float(v.sum()),
                           median=float(np.median(v)),
                           p90=float(np.percentile(v, 90)),
                           max=float(v.max()))
                           for k, v in caps_by_R.items()}),
                  handle, ensure_ascii=False, indent=1)
    print("\nfrontier (R, fences, coverage):")
    for r in frontier[::4]:
        print(f"  {r[0]*1000:.0f} m  N={int(r[1]):5d}  cov={100*r[2]:5.1f}%")
    print("wrote", outdir)


if __name__ == "__main__":
    main()
