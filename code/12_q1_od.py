"""T006: turn the fitted Q1 destination model into a typical-day OD input for Q2/Q4.

Reads data/q1_params.json (five beta values), data/cells.parquet (cells and the
50-zone assignment), data/dist_matrix.npy and the TYP_DAYS trips.  Rebuilds the
training-day attraction field A_j(h) exactly as 10_q1_model.py did, forms
P(j|i,h) proportional to A_j(h) exp(-beta_h d_ij), and saves the cell and zone
arrays Q2/Q4 need.  Observations stay at the sampled scale: no 1/p expansion.

Output: data/runs/T006/q1_od.npz and q1_vs_obs.json.
"""
import io
import json
import os

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

import lib
from lib import DATA

RUN = os.path.join(DATA, "runs", "T006")
BANDS = [(0, 5), (6, 9), (10, 15), (16, 19), (20, 23)]
BAND_NAMES = ["night 0-5", "morning 6-9", "midday 10-15",
              "evening 16-19", "late 20-23"]
TR_DAYS = ["2017-05-10", "2017-05-11", "2017-05-15", "2017-05-16",
           "2017-05-18"]
TE_DAYS = ["2017-05-12", "2017-05-19"]
SIGMA_A = 1.2


def band_stats(sub, n):
    O = np.bincount(sub.o.values, minlength=n).astype(np.float64)
    Dj = np.bincount(sub.d.values, minlength=n).astype(np.float64)
    OD = np.zeros((n, n), dtype=np.float32)
    np.add.at(OD, (sub.o.values, sub.d.values), 1.0)
    return O, Dj, OD


def smooth_attraction(Dj, cells, sigma=SIGMA_A):
    ix = cells.ix.values - cells.ix.values.min()
    iy = cells.iy.values - cells.iy.values.min()
    g = np.zeros((ix.max() + 1, iy.max() + 1))
    g[ix, iy] = Dj
    g = gaussian_filter(g, sigma)
    return g[ix, iy].astype(np.float64)


def norm_logp(logW):
    logW = logW.astype(np.float32)
    mx = logW.max(1, keepdims=True)
    E = np.exp(logW - mx)
    return logW - mx - np.log(E.sum(1, keepdims=True))


def top5(logp, ODte, k=5):
    m = ODte > 0
    ii, jj = np.where(m)
    cnt = ODte[m]
    hit = 0.0
    for i0 in np.unique(ii):
        sel = ii == i0
        top = np.argpartition(-logp[i0], k - 1)[:k]
        hit += cnt[sel][np.isin(jj[sel], top)].sum()
    return float(hit / cnt.sum())


def weighted_ks(hist_a, hist_b, edges):
    ca = np.cumsum(hist_a) / max(hist_a.sum(), 1.0)
    cb = np.cumsum(hist_b) / max(hist_b.sum(), 1.0)
    return float(np.abs(ca - cb).max())


def main():
    os.makedirs(RUN, exist_ok=True)
    params = json.load(io.open(os.path.join(DATA, "q1_params.json"),
                               encoding="utf-8"))
    beta_by_name = {name: float(params["bands"][name]["beta"])
                    for name in BAND_NAMES}
    beta = np.array([beta_by_name[name] for name in BAND_NAMES],
                    dtype=np.float32)

    cells = lib.get_cells()
    n = len(cells)
    zid = cells.zone.to_numpy().astype(int)
    NZ = int(zid.max()) + 1
    Z = np.zeros((n, NZ), dtype=np.float64)
    Z[np.arange(n), zid] = 1.0
    D = np.load(os.path.join(DATA, "dist_matrix.npy")).astype(np.float32)

    df = lib.load_trips(lib.ALL_DAYS)
    cells2, df = lib.make_cells(df, lib.THR)
    assert len(cells2) == n
    df["band"] = pd.cut(df.hour, [-1, 5, 9, 15, 19, 23],
                        labels=False).astype("Int64")
    ok = df.o.notna() & df.d.notna()
    gt = df[ok].copy()
    gt["o"] = gt.o.astype(int)
    gt["d"] = gt.d.astype(int)

    # --- observed typical-day demand, sampled scale, no 1/p expansion
    sub = gt[gt.date.isin(lib.TYP_DAYS)]
    nd = len(lib.TYP_DAYS)
    dep_obs = np.zeros((24, n), dtype=np.float64)
    arr_obs_cell = np.zeros((24, n), dtype=np.float64)
    np.add.at(dep_obs, (sub.hour.values, sub.o.values), 1.0)
    np.add.at(arr_obs_cell, (sub.hour.values, sub.d.values), 1.0)
    dep_obs /= nd
    arr_obs_cell /= nd

    od_obs = np.zeros((24, NZ, NZ), dtype=np.float64)
    np.add.at(od_obs, (sub.hour.values, zid[sub.o.values], zid[sub.d.values]),
              1.0)
    od_obs /= nd
    obs_hist = np.bincount(
        np.clip((D[sub.o.values, sub.d.values] / 0.5).astype(int), 0, 179),
        minlength=180).astype(np.float64)

    hours_of_band = {b: [h for h in range(24) if BANDS[b][0] <= h <= BANDS[b][1]]
                     for b in range(5)}
    band = np.array([next(b for b, (lo, hi) in enumerate(BANDS)
                          if lo <= h <= hi) for h in range(24)],
                    dtype=np.int8)

    arr_q1_cell = np.zeros((24, n), dtype=np.float64)
    od_q1 = np.zeros((24, NZ, NZ), dtype=np.float64)
    model_hist = np.zeros(180, dtype=np.float64)
    model_same = 0.0
    per_band = []
    ref_table = pd.read_csv(os.path.join(DATA, "q1_validation.csv")).set_index("band")

    for b, name in enumerate(BAND_NAMES):
        tr = gt[(gt.band == b) & gt.date.isin(TR_DAYS)]
        _, Dj, _ = band_stats(tr, n)
        A = smooth_attraction(Dj, cells)
        logA = np.log(np.maximum(A, 1e-12))
        logp = norm_logp((-beta[b] * D + logA.astype(np.float32)[None, :]))
        te = gt[(gt.band == b) & gt.date.isin(TE_DAYS)]
        _, _, ODte = band_stats(te, n)
        hit = top5(logp, ODte)

        P = np.exp(logp.astype(np.float64))
        P /= P.sum(1, keepdims=True)
        diagP = np.diagonal(P).copy()
        R = P @ Z                       # (n, NZ): destination-zone mass per origin

        for h in hours_of_band[b]:
            d = dep_obs[h]
            arr_q1_cell[h] = d @ P
            od_q1[h] = Z.T @ (d[:, None] * R)
            model_same += float(d @ diagP)
            for j0 in range(0, n, 800):
                j1 = min(j0 + 800, n)
                w = P[:, j0:j1] * d[:, None]   # d_i * P(j|i), every origin
                bins = np.clip((D[:, j0:j1] / 0.5).astype(int), 0, 179)
                model_hist += np.bincount(
                    bins.ravel(), weights=w.ravel(), minlength=180)
        per_band.append({"band": name, "trips": int(len(te)),
                         "top5": hit,
                         "top5_reference": float(ref_table.loc[name, "top5_model"])})
        print(f"{name}: top5 {hit:.6f} "
              f"(reference {per_band[-1]['top5_reference']:.6f})")

    # --- row-sum and comparison diagnostics
    dep_zone = np.stack([np.bincount(zid, weights=dep_obs[h], minlength=NZ)
                         for h in range(24)])
    od32 = od_q1.astype(np.float32)
    row = od32.sum(2, dtype=np.float64)
    rel = np.abs(row - dep_zone) / np.maximum(dep_zone, 1e-12)
    row_err = float(rel.max())

    trips = per_band
    w = np.array([r["trips"] for r in trips], dtype=float)
    top5_weighted = float(sum(r["top5"] * r["trips"] for r in trips) / w.sum())
    top5_ref = float(sum(r["top5_reference"] * r["trips"] for r in trips)
                     / w.sum())

    # observed same-cell share and distance histogram on the same seven days
    obs_same = float((sub.o.values == sub.d.values).mean())
    dep_total = float(dep_obs.sum())
    model_same_share = model_same / max(dep_total, 1.0)
    hist_edges = np.arange(0, 90.5, 0.5)

    zone_diff = od_q1 - od_obs
    cell_arr_diff = arr_q1_cell - arr_obs_cell

    out = {
        "top5": {
            "per_band": trips,
            "weighted": top5_weighted,
            "reference_weighted": top5_ref,
            "abs_diff": abs(top5_weighted - top5_ref),
            "tolerance": 0.005,
        },
        "row_sum_max_rel_error": row_err,
        "same_cell_share": {
            "observed": obs_same,
            "model": float(model_same_share),
            "diff": float(model_same_share - obs_same),
        },
        "distance_ks": {
            "ks": weighted_ks(model_hist, obs_hist, hist_edges),
            "bin_km": 0.5,
            "range_km": [0.0, 90.0],
        },
        "zone_od": {
            "l1": float(np.abs(zone_diff).sum()),
            "l1_normalized": float(np.abs(zone_diff).sum()
                                   / max(od_obs.sum(), 1.0)),
            "max_abs": float(np.abs(zone_diff).max()),
            "total_q1": float(od_q1.sum()),
            "total_obs": float(od_obs.sum()),
        },
        "cell_arrivals": {
            "l1": float(np.abs(cell_arr_diff).sum()),
            "max_abs": float(np.abs(cell_arr_diff).max()),
            "total_q1": float(arr_q1_cell.sum()),
            "total_obs": float(arr_obs_cell.sum()),
        },
    }

    np.savez(
        os.path.join(RUN, "q1_od.npz"),
        od_q1=od_q1.astype(np.float32),
        arr_q1_cell=arr_q1_cell.astype(np.float32),
        dep_obs=dep_obs.astype(np.float32),
        arr_obs_cell=arr_obs_cell.astype(np.float32),
        band=band,
        beta=beta,
        train_days=np.array(TR_DAYS),
        typ_days=np.array(lib.TYP_DAYS),
        sigma=np.float32(SIGMA_A),
    )
    with io.open(os.path.join(RUN, "q1_vs_obs.json"), "w",
                 encoding="utf-8") as handle:
        json.dump(out, handle, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print("saved", os.path.join(RUN, "q1_od.npz"))


if __name__ == "__main__":
    main()
