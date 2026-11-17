"""T004 step 3: build the 0.0025-degree (quarter-size) cell system.

Same construction as the 0.005-degree grid in code/10_q1_model.py:
  * all 10 well-covered days (lib.ALL_DAYS), threshold 50 trips on starts or ends
  * cell key = floor(lon/step) + 100000 * floor(lat/step)
  * zones inherited from the parent 0.005-degree cell, so the 50-zone system is
    unchanged and the two resolutions are nested.

Fleet for the fine-grid runs is fixed at 495,540 (T001-R2 centre).  I0 is
redistributed by the fine-grid last-trip-end distribution.  Cells with no
last-trip-end evidence receive a default weight (zone median of positive
counts); the zero-weight variant is saved for a sensitivity check.
"""
import io
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code"))
import lib  # noqa: E402

RUN = os.path.join(ROOT, "data", "runs", "T004")
os.makedirs(RUN, exist_ok=True)
FLEET = 495540.0
SCALE = 1.0 / 0.128
STEP_FINE = 0.0025

# ---- fine cells ---------------------------------------------------------
lib.STEP = STEP_FINE
df_all = lib.load_trips(lib.ALL_DAYS)
n_all = len(df_all)
cells_f, df_f = lib.make_cells(df_all, 50)
C = len(cells_f)
ok = df_f.o.notna() & df_f.d.notna()
n_mapped = int(ok.sum())

# parent zone from the 0.005-degree table
coarse = pd.read_parquet(os.path.join(ROOT, "data", "cells.parquet"))
zone_of_key = dict(zip(coarse.key.astype(int), coarse.zone.astype(int)))
parent_key = (cells_f.iy // 2) * 100000 + (cells_f.ix // 2)
zid = np.array([zone_of_key[int(k)] for k in parent_key], dtype=int)
assert np.isfinite(zid).all()
Z = int(zid.max()) + 1

# ---- typical-day departures and destination kernel ----------------------
df_typ = lib.load_trips(lib.TYP_DAYS)
# Re-use the fine cell map built on ALL_DAYS: map by key, not by a second threshold
key_map = {int(k): i for i, k in enumerate(cells_f.key)}
df_typ = lib.add_cells(df_typ)
df_typ["o"] = (df_typ.iy_s * 100000 + df_typ.ix_s).map(key_map)
df_typ["d"] = (df_typ.iy_e * 100000 + df_typ.ix_e).map(key_map)
okt = df_typ.o.notna() & df_typ.d.notna()
st = df_typ[okt].copy()
st["o"] = st.o.astype(int); st["d"] = st.d.astype(int)
nd = len(lib.TYP_DAYS)
step5 = (st.starttime.dt.hour.values * 60 + st.starttime.dt.minute.values) // 5
dep5 = np.zeros((288, C))
np.add.at(dep5, (step5, st.o.values), 1.0)
dep5 = dep5 / nd * SCALE

cnt = np.zeros((Z, C))
np.add.at(cnt, (zid[st.o.values], st.d.values), 1.0)
Pest = cnt / np.maximum(cnt.sum(1, keepdims=True), 1e-9)

# ---- initial inventory: last trip end per bicycle, all 10 days ----------
dfl = df_all.copy()
dfl = lib.add_cells(dfl)
dfl["o"] = (dfl.iy_s * 100000 + dfl.ix_s).map(key_map)
dfl["d"] = (dfl.iy_e * 100000 + dfl.ix_e).map(key_map)
okl = dfl.o.notna() & dfl.d.notna()
lastd = (dfl[okl].sort_values("starttime").groupby("bikeid").tail(1)
         .d.dropna().astype(int).values)
n_last = len(lastd)
last_cnt = np.bincount(lastd, minlength=C).astype(float)
zero_ev = last_cnt == 0
# default weight for cells without last-trip-end evidence: zone median of
# positive counts; if a zone has none, the global median
pos = last_cnt[last_cnt > 0]
glob_med = float(np.median(pos))
w = last_cnt.copy()
for z in range(Z):
    zz = np.where((zid == z) & zero_ev)[0]
    if len(zz) == 0:
        continue
    zp = last_cnt[(zid == z) & (last_cnt > 0)]
    w[zz] = float(np.median(zp)) if len(zp) else glob_med
I0_default = w / w.sum() * FLEET
I0_zero = last_cnt / last_cnt.sum() * FLEET

# ---- diagnostics --------------------------------------------------------
trips_per_cell = (np.bincount(df_f.o.dropna().astype(int), minlength=C) +
                  np.bincount(df_f.d.dropna().astype(int), minlength=C))
typ_dep = dep5.sum(0) / SCALE  # observed typical-day departures, unscaled
gps = float(pd.read_parquet(os.path.join(ROOT, "data", "trips.parquet"),
                            columns=["start_lat_err"]).start_lat_err.iloc[0])
# coarse-grid support (0.005 deg) on the same trip set, for comparison
coarse_keys = coarse.key.astype(int).to_numpy()
ks_c = (np.floor(df_all.start_lat.values / 0.005).astype(int) * 100000
        + np.floor(df_all.start_lon.values / 0.005).astype(int))
ke_c = (np.floor(df_all.end_lat.values / 0.005).astype(int) * 100000
        + np.floor(df_all.end_lon.values / 0.005).astype(int))
coarse_both = np.isin(ks_c, coarse_keys) & np.isin(ke_c, coarse_keys)
typ_mask = df_all.date.isin(lib.TYP_DAYS).to_numpy()
diag = dict(
    step_deg=STEP_FINE,
    n_cells=C,
    n_coarse_cells=len(coarse),
    n_zones=Z,
    all_days=lib.ALL_DAYS,
    trips_all_days=n_all,
    trips_with_both_cells_mapped=n_mapped,
    share_trips_mapped=float(n_mapped / n_all),
    coarse_support=dict(
        n_cells=len(coarse),
        trips_mapped=int(coarse_both.sum()),
        share_trips_mapped=float(coarse_both.mean()),
        typical_day_departures=float((coarse_both & typ_mask).sum() / nd),
        share_of_fine_typical_departures=float(
            ((coarse_both & typ_mask).sum()) / len(st))),
    per_cell_trips_10days=dict(
        min=float(trips_per_cell.min()), p10=float(np.percentile(trips_per_cell, 10)),
        median=float(np.median(trips_per_cell)), mean=float(trips_per_cell.mean()),
        p90=float(np.percentile(trips_per_cell, 90)), max=float(trips_per_cell.max())),
    typical_day_departures=dict(
        zero_cells=int((typ_dep == 0).sum()),
        min=float(typ_dep.min()), median=float(np.median(typ_dep)),
        mean=float(typ_dep.mean()), max=float(typ_dep.max()),
        total=float(typ_dep.sum())),
    gps_error_deg=gps,
    gps_error_m_lat=gps * lib.LAT_KM * 1000,
    gps_error_m_lon=gps * lib.LON_KM * 1000,
    cell_m_lat=STEP_FINE * lib.LAT_KM * 1000,
    cell_m_lon=STEP_FINE * lib.LON_KM * 1000,
    cell_over_gps_lat=float(STEP_FINE / gps),
    cell_over_gps_lon=float(STEP_FINE / gps),
    last_trip_end=dict(
        bikes=int(n_last),
        cells_with_evidence=int((~zero_ev).sum()),
        cells_without_evidence=int(zero_ev.sum()),
        departures_in_no_evidence_cells=float(typ_dep[zero_ev].sum()),
        share_departures_no_evidence=float(typ_dep[zero_ev].sum() / typ_dep.sum()),
        default_weight=float(glob_med),
        note=("main I0 gives each no-evidence cell its zone's median positive "
              "last-trip-end count; the zero-evidence variant leaves them at 0")),
    geohash_cell_m=[153.0, 117.0],
)
assert len(parent_key) == C and np.all(np.isin(parent_key, coarse.key.astype(int)))

cells_f["zone"] = zid
cells_f.to_parquet(os.path.join(RUN, "fine_cells.parquet"))
np.savez(os.path.join(RUN, "fine_sim_inputs.npz"),
         dep5=dep5, Pest=Pest, I0c=I0_default, I0_zero=I0_zero,
         zid=zid.astype(np.int32), xy=cells_f[["x", "y"]].to_numpy())
json.dump(diag, io.open(os.path.join(RUN, "fine_grid.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(diag, ensure_ascii=False, indent=1))
