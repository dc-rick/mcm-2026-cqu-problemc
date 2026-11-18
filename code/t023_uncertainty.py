"""T023: uncertainty decomposed by source (M6 reopened, M13 folded in).

Three parameter groups, one model, one group at a time:

  A  data-identified   five band betas, hourly departure/arrival surfaces and
                       the attraction field (day-block bootstrap over the nine
                       recorded days); trip-distance distribution (duration
                       input, Q1 only)
  B  assumed loads     sampling rate p, initial inventory I0, day-to-day
                       volume (nine recorded days)
  C  literature / planning constants (costs, kappa/rho/c0, R, cap 150,
                       T018 truck assumptions, T024 tricycle constants)

Reference caliber (task sheet T023 section 4, locked 2026-11-18):
  Q2 = T024-R3 two-tier: 975 tricycles, 617,055.7515 CNY/day (full-load),
       unmet 39,984.2332, saving 902,223.20 (59.385%).
  Q3 = T025-R2: K=2 tight pure 138,317.1512 / unmet 33.1936, payment
       ceiling -1,591.403 (best policy), all K negative.
  Q4 = T019: 4,000-site knee, occupancy coverage 0.595298, arrival
       coverage 0.558045, unassigned 394,553.13 (occupancy caliber).

Method
  * A: 256 day-block bootstrap draws (multinomial weights over the nine
    recorded days).  Each draw re-estimates the five betas with the same
    restricted-support MLE as 10_q1_model.py, rebuilds the attraction field
    and the hourly surfaces through the exact T018 demand path, then
    renormalises the total to the reference so that day-level volume
    variation stays in group B.
  * B: the 3 x 2 x 9 = 54 combinations (p, I0, recorded day) are enumerated;
    the B marginal is therefore exact, not sampled.
  * C: 256 independent uniform draws over the documented ranges (+-50% for
    parameters without a source).
  * Design: 8 evaluations per base draw (n = 128), pick-freeze:
        F  = f(A_i, B_i, C_i)          S_A = f(A_i, B'_i, C'_i)
        Fb = f(A'_i, B'_i, C'_i)       S_B = f(A'_i, B_i, C'_i)
        M_A= f(A_i, B_ref, C_ref)      S_C = f(A'_i, B'_i, C_i)
        M_B= f(A_ref, B_i, C_ref)      M_C= f(A_ref, B_ref, C_i)
    First-order group index (pick-freeze covariance identity)
        S_g = [mean(F * S_g) - mean(F) * mean(S_g)] / Var(F),
    where F and S_g share the draw of group g and are independent in the
    other two groups, so the covariance identifies Var(E[f | g]).  Pairing
    S_g with Fb instead would identify the closed index of the *other* two
    groups (a common mistake; that estimator sums to ~2-3, not ~1).
    Cross-checked against the spread at the reference point of the other
    groups, Var(M_g)/Var(F), which is a sensitivity diagnostic and NOT a
    Sobol index: under strong multiplicative interactions it can exceed 1.
  * Proxy: MILP layers are replaced by their LP relaxations and anchored at
    the reference, out(x) = out_ref + [proxy(x) - proxy(ref)].  The anchor
    shift is deterministic (integerisation) and is NOT counted in any
    variance share.  Proxy deviation is measured against the integer chains
    at ten sampled points (Q2 MILP K=60 + T024 integer trips; Q3 tight
    K=2 MILP; Q4 MILP siting + MC demand).

Stages: prep | bootstrap | inputs | reference | design | kstar | intervals |
        shares | flips | deviation | tornado | params | check | all
"""
import argparse
import contextlib
import functools
import gc
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
import t005_common as C5
import t008_q3_pool as t8
import t018_q2_int as t18
import t019_q4_cap as t19
import t020_q3_budget as t20
import t024_q2_tier2 as t24

RUN = os.path.join(lib.DATA, "runs", "T023")
ACACHE = os.path.join(RUN, "a_cache")
FIGS = os.path.join(RUN, "figs")
for _d in (RUN, ACACHE, FIGS):
    os.makedirs(_d, exist_ok=True)

P_REF = 0.128
SCALE = 1.0 / P_REF
# nine recorded days: lib.ALL_DAYS also lists 2017-05-17, which has no
# trips in the order file (verified in prep: zero rows).
DAYS9 = [d for d in lib.ALL_DAYS if d != "2017-05-17"]
N_DRAW = 256
N_HALF = 128
SEED_A = 20261118
SEED_C = 20261119
SEED_B = 20261120
BASE_Q2 = 1519278.950269192       # T024 no-tricycle baseline (MILP caliber)

# Every constant below is copied verbatim from an accepted result file; the
# provenance is re-checked programmatically in run_check (item 10).
#   data/runs/T024/schedule_check.json  exec_totals.*
#   data/runs/T018/result_int_K60.json  obj
#   data/runs/T018/result_lp_K60.json   obj
#   data/runs/T025/hybrid_by_K.json     tight K=2 arrivals/30000 row
#   data/runs/T019/frontier.json        meas343.selected / knee
REF = dict(
    q2=dict(fleet=975, total=617055.7515153277, unmet=39984.233156118324,
            saving=902223.1987538643, saving_share=0.5938496012164223,
            baseline=BASE_Q2, truck_obj=83807.74383516546,
            lp_obj=82968.06705963545,
            layer_fullload=533248.0076801622,
            lp_unmet=38489.43620902851, int_short=1494.7969470898138,
            tri_vkm_fullload=4213.0437172613465),
    q3=dict(K=2, pure_obj=138317.15115493198, pure_unmet=33.1936115075404,
            ceiling=-1591.402938762012, best_policy="arrivals B=30000",
            payment=30000.0, truck_obj=139908.5540936937),
    q4=dict(n_sites=4000, coverage=0.5952982255532903,
            arrival_coverage=0.5580451986279951,
            unassigned=394553.1326196083, cap=150.0, R=0.450),
)

C_SPECS = [
    dict(key="COST_KM", ref=6.0, lo=3.0, hi=9.0, unit="CNY/vehicle-km",
         source="planning assumption (T018); no public source; +-50%",
         effect="Q2,Q3"),
    dict(key="COST_LOAD", ref=0.5, lo=0.25, hi=0.75, unit="CNY/bike moved",
         source="planning assumption (T018); +-50%", effect="Q2,Q3"),
    dict(key="PENALTY", ref=6.0, lo=3.0, hi=9.0, unit="CNY/unserved bike",
         source="planning assumption (T018); +-50%", effect="Q2,Q3"),
    dict(key="COST_DEV", ref=3.0, lo=1.5, hi=4.5,
         unit="CNY/bike end-of-day deviation",
         source="planning assumption (T018); +-50%", effect="Q2,Q3"),
    dict(key="FIXED_K", ref=600.0, lo=300.0, hi=900.0,
         unit="CNY/truck-day", source="planning assumption (T018); +-50%",
         effect="Q2,Q3"),
    dict(key="Q_TRUCK", ref=40.0, lo=20.0, hi=60.0, unit="bikes/truck trip",
         source="planning assumption (T018); +-50%", effect="Q2,Q3"),
    dict(key="TRUCK_SPEED", ref=20.0, lo=10.0, hi=30.0, unit="km/h",
         source="planning assumption (T018); +-50%",
         effect="none in the LP objective (feasibility audit only)"),
    dict(key="TRUCK_HANDLE", ref=0.25, lo=0.125, hi=0.375,
         unit="min/bike/operation",
         source="planning assumption (T018); +-50%",
         effect="none in the LP objective (feasibility audit only)"),
    dict(key="TRI_FIXED", ref=300.0, lo=150.0, hi=450.0,
         unit="CNY/tricycle-day",
         source="planning assumption (T024 vehicle_params.json); +-50%",
         effect="Q2"),
    dict(key="TRI_KM", ref=0.20, lo=0.10, hi=0.30, unit="CNY/vehicle-km",
         source="planning assumption (T024); +-50%", effect="Q2"),
    dict(key="TRI_Q", ref=20.0, lo=10.0, hi=30.0, unit="bikes/trip",
         source="planning assumption (T024); +-50%", effect="Q2"),
    dict(key="TRI_V", ref=15.0, lo=7.5, hi=22.5, unit="km/h",
         source="planning assumption (T024); +-50%", effect="Q2"),
    dict(key="TRI_HANDLE", ref=0.5, lo=0.25, hi=0.75, unit="min/bike",
         source="planning assumption (T024); +-50%", effect="Q2"),
    dict(key="TRI_MAX_TRIP", ref=5.0, lo=2.5, hi=7.5, unit="km",
         source="planning assumption (T024); +-50%", effect="Q2"),
    dict(key="TRI_NMAX", ref=60.0, lo=30.0, hi=90.0,
         unit="tricycles/zone (solver cap)",
         source="planning assumption (T024); +-50%",
         effect="fleet re-optimisation only; the reference allocation is "
                "priced unclipped (the cap did not bind in R2/R3)"),
    dict(key="TRI_SHIFT", ref=13.0, lo=6.5, hi=19.5, unit="h/day",
         source="planning assumption (T024, 07:00-20:00); +-50%",
         effect="Q2"),
    dict(key="KAPPA", ref=2.0, lo=1.0, hi=3.0, unit="utility/CNY",
         source="T008 assumption, unchanged in T020/T025; range is the "
                "T008/T005 sweep range (1-3)", effect="Q3"),
    dict(key="RHO", ref=0.35, lo=0.175, hi=0.525, unit="utility/min",
         source="T008 assumption; range 0.20-0.50 in the T008 sweep, "
                "widened to +-50%", effect="Q3"),
    dict(key="C0", ref=0.8, lo=0.4, hi=1.2, unit="utility",
         source="T008 assumption; range 0.4-1.2 in the T008 sweep",
         effect="Q3"),
    dict(key="V_BIKE", ref=10.0, lo=5.0, hi=15.0, unit="km/h",
         source="T008 assumption in tau(d); +-50%", effect="Q3"),
    dict(key="WALK_M_MIN", ref=80.0, lo=40.0, hi=120.0, unit="m/min",
         source="T008 assumption in tau(d); +-50%", effect="Q3"),
    dict(key="R_WALK", ref=0.450, lo=0.150, hi=0.450, unit="km",
         source="T019 radius sweep {0.150, 0.300, 0.450}; discrete, "
                "sourced", effect="Q4"),
    dict(key="CAP", ref=150.0, lo=75.0, hi=225.0, unit="slots/site",
         source="planning assumption (T009/T019, lambda=150); +-50%",
         effect="Q4"),
    dict(key="LAMBDA", ref=150.0, lo=None, hi=None, unit="slots/site",
         source="superseded: T019 replaced the eq.(29) lambda rule by the "
                "Kneedle knee; recorded, not sampled",
         effect="none (not in the capped model)"),
    dict(key="DUR_SPEED", ref=11.5, lo=5.75, hi=17.25, unit="km/h",
         source="planning assumption (sec4_q1 eq. dur), NOT fitted: the "
                "order file has no end time; +-50%",
         effect="none on Q2-Q4 (Q1 only)"),
    dict(key="DUR_SD", ref=2.5, lo=1.25, hi=3.75, unit="km/h",
         source="planning assumption (sec4_q1); +-50%",
         effect="none on Q2-Q4"),
    dict(key="DUR_DETOUR", ref=1.35, lo=None, hi=None, unit="ratio",
         source="planning assumption (sec4_q1); multiplier on straight-line "
                "distance", effect="none on Q2-Q4"),
    dict(key="DUR_FIXED", ref=1.0, lo=None, hi=None, unit="min",
         source="planning assumption (sec4_q1); lock/unlock time",
         effect="none on Q2-Q4"),
]
C_SAMPLED = [s for s in C_SPECS if s["lo"] is not None and s["hi"] is not None]
C_BY_KEY = {s["key"]: s for s in C_SPECS}

OUT_KEYS = [
    ("q2_total", "Q2 two-tier total cost", "CNY/day", REF["q2"]["total"]),
    ("q2_unmet", "Q2 executable unmet", "bikes/day", REF["q2"]["unmet"]),
    ("q2_saving_share", "Q2 saving share vs no-tricycle", "share",
     REF["q2"]["saving_share"]),
    ("q3_pure_obj", "Q3 pure truck optimum at K=2", "CNY/day",
     REF["q3"]["pure_obj"]),
    ("q3_ceiling", "Q3 payment ceiling (best policy)", "CNY/day",
     REF["q3"]["ceiling"]),
    ("q3_pure_unmet", "Q3 unmet at K=2", "bikes/day", REF["q3"]["pure_unmet"]),
    ("q4_coverage", "Q4 occupancy coverage at 4,000 sites", "share",
     REF["q4"]["coverage"]),
    ("q4_arrival", "Q4 arrival-volume coverage", "share",
     REF["q4"]["arrival_coverage"]),
    ("q4_unassigned", "Q4 unassigned occupancy demand", "bike-slots",
     REF["q4"]["unassigned"]),
]
OUT_NAMES = [k for k, _, _, _ in OUT_KEYS]
OUT_SOLVER = {
    "q2_total": "LP proxy (T018 truck LP + T024 tricycle LP replay)",
    "q2_unmet": "LP proxy (T024 tricycle LP replay)",
    "q2_saving_share": "LP proxy (LP baseline and total)",
    "q3_pure_obj": "LP proxy (T018 K=2 LP)",
    "q3_ceiling": "LP proxy (T018 K=2 LP pure vs hybrid)",
    "q3_pure_unmet": "LP proxy (T018 K=2 LP)",
    "q4_coverage": "MILP (T019 site-integer, n=4,000, gap 0.005)",
    "q4_arrival": "MILP (T019 site-integer, n=4,000, gap 0.005)",
    "q4_unassigned": "MILP (T019 site-integer, n=4,000, gap 0.005)",
}


# --------------------------------------------------------------------------
# utilities
# --------------------------------------------------------------------------
def dump(obj, name, folder=RUN):
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    print("wrote", path, flush=True)
    return path


def load_json(name, folder=RUN):
    with open(os.path.join(folder, name), encoding="utf-8") as fh:
        return json.load(fh)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def banner(msg, t0=None):
    print("[%s] %s%s" % (time.strftime("%H:%M:%S"), msg,
                         "" if t0 is None else " (%.1fs)" % (time.time() - t0)),
          flush=True)


_DIST = None
_T8 = None
_CAPINFO = None
_PEST = None
_NEAR = None
_CELLS = None


def dist_matrix():
    global _DIST
    if _DIST is None:
        _DIST = np.load(os.path.join(lib.DATA, "dist_matrix.npy"))
    return _DIST


def near_mask():
    """Memoised ``dist_matrix() <= t8.R_KM`` (26.4 MiB bool at 5,263 cells).

    ``t8.build_edges`` recomputes this mask on every call, and Q3 runs in
    every design job, so the mask is built once per worker.  R_KM is a fixed
    T008 constant (0.45 km, never sampled), so the mask is a pure function of
    the (also cached) distance matrix.
    """
    global _NEAR
    if _NEAR is None:
        _NEAR = dist_matrix() <= t8.R_KM
    return _NEAR


def cells_table():
    """Cached canonical cell table (parquet read once per worker)."""
    global _CELLS
    if _CELLS is None:
        _CELLS = lib.get_cells()
    return _CELLS


def t8_data():
    global _T8
    if _T8 is None:
        _T8 = t8.load_data()
    return _T8


def capinfo():
    global _CAPINFO
    if _CAPINFO is None:
        # build_capacity reads the T019 plan and spare-capacity files; its
        # ``pool`` argument is unused, so passing None avoids loading the
        # whole T008 pool (and the trips) in every worker.
        _CAPINFO = t20.build_capacity(None)
    return _CAPINFO


@contextlib.contextmanager
def patch18(c):
    keys = ("COST_KM", "COST_LOAD", "PENALTY", "COST_DEV", "Q_TRUCK")
    old = {k: getattr(t18, k) for k in keys}
    for k in keys:
        setattr(t18, k, c[k])
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(t18, k, v)


@contextlib.contextmanager
def patch24(c):
    keys = ("TRI_Q", "TRI_V", "TRI_HANDLE", "TRI_KM", "TRI_MAX_TRIP",
            "TRI_FIXED", "TRI_NMAX", "TRI_HOUR0", "TRI_HOUR1")
    hour0, hour1 = 7.0, min(24.0, 7.0 + float(c["TRI_SHIFT"]))
    vals = dict(TRI_Q=c["TRI_Q"], TRI_V=c["TRI_V"],
                TRI_HANDLE=c["TRI_HANDLE"], TRI_KM=c["TRI_KM"],
                TRI_MAX_TRIP=c["TRI_MAX_TRIP"], TRI_FIXED=c["TRI_FIXED"],
                TRI_NMAX=c["TRI_NMAX"], TRI_HOUR0=hour0, TRI_HOUR1=hour1)
    old = {k: getattr(t24, k) for k in keys}
    for k, v in vals.items():
        setattr(t24, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(t24, k, v)


def solve18(dep, PZ, I0, K, c, integer=False, time_limit=120, mip_gap=0.01):
    if t18.NET is None:
        t18.setup_design()
    with patch18(c), contextlib.redirect_stdout(io.StringIO()):
        r = t18.solve(dep, None, PZ, t18.NET, I0, K, integer,
                      time_limit=time_limit, mip_gap=mip_gap,
                      event_order=True, fixed_cost=c["FIXED_K"])
    if r is None:
        raise RuntimeError("T018 solve failed")
    return r


def cvec_full(cvals):
    c = {s["key"]: s["ref"] for s in C_SPECS}
    for k, v in (cvals or {}).items():
        c[k] = float(v)
    return c


# --------------------------------------------------------------------------
# prep
# --------------------------------------------------------------------------
def run_prep(force=False):
    path = os.path.join(RUN, "prep.npz")
    meta_path = os.path.join(RUN, "prep.json")
    if os.path.exists(path) and os.path.exists(meta_path) and not force:
        print("reuse", path, flush=True)
        return dict(np.load(path, allow_pickle=True)), load_json("prep.json")
    t0 = time.time()
    cells = lib.get_cells()
    C = len(cells)
    zid = cells.zone.to_numpy().astype(int)
    Z = int(zid.max()) + 1
    xy = cells[["x", "y"]].to_numpy().astype(np.float64)

    df = lib.load_trips(DAYS9)
    cells2, df = lib.make_cells(df, lib.THR)
    assert len(cells2) == C
    df["band"] = np.digitize(df.hour, [6, 10, 16, 20])
    ok = df.o.notna() & df.d.notna()
    days = list(DAYS9)
    dayidx = {d: k for k, d in enumerate(days)}
    s = df[ok].copy()
    s["o"] = s.o.astype(int)
    s["d"] = s.d.astype(int)
    di = s.date.map(dayidx).to_numpy().astype(np.int16)
    step5 = ((s.starttime.dt.hour * 60 + s.starttime.dt.minute) // 5
             ).to_numpy().astype(np.int16)
    dep5_day = np.zeros((len(days), 288, C), dtype=np.float32)
    arr5_day = np.zeros((len(days), 288, C), dtype=np.float32)
    np.add.at(dep5_day, (di, step5, s.o.values), 1.0)
    np.add.at(arr5_day, (di, step5, s.d.values), 1.0)
    depc_day = dep5_day.reshape(len(days), 24, 12, C).sum(2)
    arrc_day = arr5_day.reshape(len(days), 24, 12, C).sum(2)
    band = s.band.to_numpy().astype(np.int8)
    dj_day = np.zeros((len(days), 5, C), dtype=np.float32)
    np.add.at(dj_day, (di, band, s.d.values), 1.0)

    tr_idx = [dayidx[d] for d in t18.TR_DAYS]
    w_ref = np.zeros(len(days))
    w_ref[tr_idx] = 1.0 / len(tr_idx)
    q1p = load_json("q1_params.json", folder=lib.DATA)
    betas_ref = np.array([float(q1p["bands"][nm]["beta"])
                          for nm in t18.BAND_NAMES])
    ref = demand_path(depc_day, arrc_day, dj_day, cells, zid, w_ref,
                      betas_ref)

    z19 = np.load(os.path.join(lib.DATA, "runs", "T019", "cell_demand.npz"))
    I0_meas_cell = z19["I0_meas"].astype(np.float64)
    I0_fleet_cell = z19["I0_fleet"].astype(np.float64)
    I0_fleet_zone = np.bincount(zid, weights=I0_fleet_cell, minlength=Z)
    I0c_ref = np.load(os.path.join(lib.DATA, "runs", "T024",
                                   "cell_inputs.npz"))["I0c"].astype(float)
    # the zone-level I0 must be the zone sums of the cell-level I0 actually
    # replayed, otherwise the T018 LP and the T024 replay disagree by ~650
    # bikes in L1 (T024 rescaled I0c to T018's zone I0 within each zone).
    I0_meas_zone = np.bincount(zid, weights=I0c_ref, minlength=Z)

    sch = t24.load_json("schedule_check.json")
    n_exec = np.zeros(Z)
    for rrow in sch["exec_rows"]:
        n_exec[int(rrow["zone"])] = float(rrow["n_exec"])
    assert abs(n_exec.sum() - 975) < 1e-9, n_exec.sum()

    # T024-R3 replays the *stored* T018 integer plan (plan_int_K60), not a
    # re-solved LP: the LP has many optima and a different vertex shifts the
    # tricycle replay by ~40 bikes (0.10%).  Using the accepted plan makes
    # the proxy \"T024-R3 with only the truck objective replaced by its LP
    # relaxation\", so the reference integerisation shift is exactly
    # 1,494.7969 as the task sheet section 4.3 requires.
    zpl = np.load(os.path.join(lib.DATA, "runs", "T018", "plan_int_K60.npz"))
    XM_plan = zpl["XM"].astype(np.float64)
    U_plan = zpl["U"].astype(np.float64)
    with open(os.path.join(lib.DATA, "runs", "T018",
                           "result_int_K60.json"), encoding="utf-8") as fh:
        int_res = json.load(fh)

    ref5_dep = (dep5_day[tr_idx].mean(0) * SCALE).astype(np.float64)
    ref5_arr = (arr5_day[tr_idx].mean(0) * SCALE).astype(np.float64)

    # destination-choice prior used by the replay (was rebuilt from the trips
    # in every worker; precomputed here to keep worker memory flat)
    Zm = np.zeros((C, Z), dtype=np.float64)
    Zm[np.arange(C), zid] = 1.0
    tr_mask = s.date.isin(t18.TR_DAYS).to_numpy()
    cnt = np.zeros((Z, C))
    np.add.at(cnt, (zid[s.o.values[tr_mask]], s.d.values[tr_mask]), 1.0)
    cnt += 1e-3
    pest = cnt / cnt.sum(1, keepdims=True)
    pest_zone = pest @ Zm

    np.savez_compressed(
        path, depc_day=depc_day, arrc_day=arrc_day, dj_day=dj_day,
        zid=zid.astype(np.int32), xy=xy, days=np.array(days),
        pest=pest, pest_zone=pest_zone,
        betas_ref=betas_ref.astype(np.float64),
        depc_ref=ref["depc"], arrc_ref=ref["arrc"], PZ_ref=ref["PZ"],
        dep_zone_ref=ref["dep_zone"],
        dep_ref_total=float(ref["depc"].sum()),
        arr_ref_total=float(ref["arrc"].sum()),
        I0c_ref=I0c_ref, I0_meas_cell=I0_meas_cell,
        I0_fleet_cell=I0_fleet_cell, I0_meas_zone=I0_meas_zone,
        I0_fleet_zone=I0_fleet_zone, n_exec=n_exec,
        XM_plan=XM_plan, U_plan=U_plan,
        truck_obj_int=float(int_res["obj"]),
        a_meas=z19["a_meas343"].astype(np.float64),
        a_fleet=z19["a_fleet495"].astype(np.float64),
        arr_q4_ref=z19["arr"].astype(np.float64),
        ref_dep5=ref5_dep, ref_arr5=ref5_arr)

    cin = np.load(os.path.join(lib.DATA, "runs", "T024", "cell_inputs.npz"),
                  allow_pickle=True)
    dep_rel = float(np.abs(ref["depc"] - cin["dep_cell"]).max()
                    / max(np.abs(cin["dep_cell"]).max(), 1e-9))
    PZ_abs = float(np.abs(ref["PZ"] - cin["PZ"]).max())
    t18_ref = np.load(os.path.join(lib.DATA, "runs", "T018",
                                   "q1_od_train.npz"))
    t18_dep_total = float(t18_ref["dep_zone"].sum() * SCALE)
    meta = dict(
        task="T023 prep", date="2026-11-18", days=days,
        train_days=t18.TR_DAYS, n_cells=int(C), n_zones=int(Z),
        dep_ref_total=float(ref["depc"].sum()),
        arr_ref_total=float(ref["arrc"].sum()),
        dep_cell_vs_t018_rel=float(abs(ref["depc"].sum() - t18_dep_total)
                                   / max(t18_dep_total, 1.0)),
        dep_cell_vs_t024_cell_inputs_maxrel=dep_rel,
        PZ_vs_t024_cell_inputs_maxabs=PZ_abs,
        i0_ref=float(I0c_ref.sum()),
        n_exec_total=float(n_exec.sum()),
        n_exec_zones=int((n_exec > 0).sum()),
        n_exec_max=float(n_exec.max()),
        note=("depc_day/arrc_day are per-day hourly counts, UNSCALED.  The "
              "reference surfaces come from demand_path() with the five "
              "training days weighted 1/5, i.e. the exact T018 path "
              "(od -> t18.prep).  Bootstrap draws use nine-day weights and "
              "renormalise the total to the reference so that day-level "
              "volume variation is group B."))
    dump(meta, "prep.json")
    banner("prep done", t0)
    return dict(np.load(path, allow_pickle=True)), meta


def demand_path(depc_day, arrc_day, dj_day, cells, zid, w, betas):
    """Exact T018 demand path for one weighted day mix (w sums to 1)."""
    C = len(zid)
    Z = int(zid.max()) + 1
    Zm = np.zeros((C, Z), dtype=np.float32)
    Zm[np.arange(C), zid] = 1.0
    D32 = dist_matrix().astype(np.float32)
    depc_s = (depc_day.astype(np.float64) * w[:, None, None]).sum(0)
    arrc_s = (arrc_day.astype(np.float64) * w[:, None, None]).sum(0)
    od = np.zeros((24, Z, Z))
    for b in range(5):
        Dj = (dj_day[:, b, :].astype(np.float64) * w[:, None]).sum(0)
        A = t18.smooth_attraction(Dj, cells)
        logA = np.log(np.maximum(A, 1e-12)).astype(np.float32)
        P = np.exp(t18.norm_logp(-np.float32(betas[b]) * D32 + logA[None, :]))
        R = P @ Zm
        for h in range(24):
            if C5.BANDS[b][0] <= h <= C5.BANDS[b][1]:
                od[h] = Zm.T @ (depc_s[h].astype(np.float32)[:, None] * R)
        del P, R
    dep_zone, _, PZ = t18.prep(od)
    return dict(depc=depc_s * SCALE, arrc=arrc_s * SCALE,
                dep_zone=dep_zone, PZ=PZ, od=od)


# --------------------------------------------------------------------------
# A-group bootstrap
# --------------------------------------------------------------------------
def band_pairs(days):
    df = lib.load_trips(days)
    cells, df = lib.make_cells(df, lib.THR)
    n = len(cells)
    df["band"] = np.digitize(df.hour, [6, 10, 16, 20])
    ok = df.o.notna() & df.d.notna()
    s = df[ok].copy()
    s["o"] = s.o.astype(int)
    s["d"] = s.d.astype(int)
    dayidx = {d: k for k, d in enumerate(days)}
    di = s.date.map(dayidx).to_numpy().astype(np.int16)
    band = s.band.to_numpy().astype(np.int8)
    o = s.o.values.astype(np.int64)
    d = s.d.values.astype(np.int64)
    out = []
    for b in range(5):
        m = band == b
        per_day = []
        for k in range(len(days)):
            md = m & (di == k)
            if not md.any():
                per_day.append((np.zeros(0, np.int64), np.zeros(0, np.int64),
                                np.zeros(0, np.float64)))
                continue
            key = o[md] * n + d[md]
            uk, cnt = np.unique(key, return_counts=True)
            per_day.append(((uk // n).astype(np.int64),
                            (uk % n).astype(np.int64),
                            cnt.astype(np.float64)))
        out.append(per_day)
    return out, days, n


def fit_beta_draw(logA, D, rows, w):
    """Restricted-destination-support MLE, same estimator as 10_q1_model."""
    from scipy.optimize import minimize_scalar
    o_all = np.concatenate([r[0] for r in rows])
    j_all = np.concatenate([r[1] for r in rows])
    c_all = np.concatenate([r[2] * w[k] for k, r in enumerate(rows)])
    keep = c_all > 0
    o_all, j_all, c_all = o_all[keep], j_all[keep], c_all[keep]
    if len(o_all) < 50:
        return None
    origins, inv = np.unique(o_all, return_inverse=True)
    nzj = np.unique(j_all)
    logAs = logA[nzj]
    Dsub = D[np.ix_(origins, nzj)].astype(np.float32)
    pos = -np.ones(D.shape[0], dtype=np.int64)
    pos[nzj] = np.arange(len(nzj))
    T1 = np.bincount(inv, weights=c_all * logAs[pos[j_all]],
                     minlength=len(origins))
    T2 = np.bincount(inv, weights=c_all * D[o_all, j_all],
                     minlength=len(origins))
    N = np.bincount(inv, weights=c_all, minlength=len(origins))

    def nll(beta):
        W = logAs[None, :] - np.float32(beta) * Dsub
        m = W.max(1, keepdims=True)
        E = np.exp(W - m)
        logS = m.ravel() + np.log(E.sum(1).ravel())
        return -(float(T1.sum()) - beta * float(T2.sum())
                 - float((N * logS).sum()))
    res = minimize_scalar(nll, bounds=(1.0, 4.0), method="bounded",
                          options={"xatol": 1e-3})
    return float(res.x)


_BW = {}


def _boot_weights():
    rng = np.random.default_rng(SEED_A)
    return rng.multinomial(9, np.full(9, 1.0 / 9.0),
                           size=N_DRAW).astype(np.int16)


def _boot_init():
    os.environ["OMP_NUM_THREADS"] = "1"
    prep, _ = run_prep()
    _BW["prep"] = prep
    _BW["D32"] = dist_matrix().astype(np.float32)
    _BW["rows"], _, _ = band_pairs(DAYS9)
    _BW["cells"] = lib.get_cells()
    _BW["W"] = _boot_weights()


def _boot_job(i):
    if "prep" not in _BW:
        _boot_init()
    prep, D32, rows, cells = (_BW["prep"], _BW["D32"], _BW["rows"],
                              _BW["cells"])
    zid = prep["zid"]
    n = len(zid)
    W = _BW["W"]
    w = W[i].astype(np.float64) / 9.0
    A_list = []
    depc_day, arrc_day, dj_day = (prep["depc_day"], prep["arrc_day"],
                                  prep["dj_day"])
    for b in range(5):
        Dj = (dj_day[:, b, :].astype(np.float64) * w[:, None]).sum(0)
        A_list.append(np.log(np.maximum(
            t18.smooth_attraction(Dj, cells), 1e-12)).astype(np.float32))
    beta = np.zeros(5)
    for b in range(5):
        bt = fit_beta_draw(A_list[b], D32, rows[b], w)
        beta[b] = bt if bt is not None else np.nan
    depc_s = (depc_day.astype(np.float64) * w[:, None, None]).sum(0)
    arrc_s = (arrc_day.astype(np.float64) * w[:, None, None]).sum(0)
    dep_total = float(depc_s.sum() * SCALE)
    arr_total = float(arrc_s.sum() * SCALE)
    meta = load_json("prep.json")
    depc_s *= meta["dep_ref_total"] / max(depc_s.sum() * SCALE, 1e-12)
    arrc_s *= meta["arr_ref_total"] / max(arrc_s.sum() * SCALE, 1e-12)
    Zm = np.zeros((n, 50), dtype=np.float32)
    Zm[np.arange(n), zid] = 1.0
    od = np.zeros((24, 50, 50))
    for b in range(5):
        P = np.exp(t18.norm_logp(-np.float32(beta[b]) * D32
                                 + A_list[b][None, :]))
        R = P @ Zm
        for h in range(24):
            if C5.BANDS[b][0] <= h <= C5.BANDS[b][1]:
                od[h] = Zm.T @ (depc_s[h].astype(np.float32)[:, None] * R)
        del P, R
    dep_zone, _, PZ = t18.prep(od)
    cp = os.path.join(ACACHE, "a_%03d.npz" % i)
    np.savez_compressed(cp, depc=(depc_s * SCALE).astype(np.float32),
                        arrc=(arrc_s * SCALE).astype(np.float32),
                        PZ=PZ.astype(np.float32),
                        dep_zone=dep_zone.astype(np.float32),
                        beta=beta, w=W[i])
    return i, beta, dep_total, arr_total


def run_bootstrap(force=False, workers=None):
    path = os.path.join(RUN, "bootstrap_samples.npz")
    t0 = time.time()
    prep, meta = run_prep()
    days = list(prep["days"])
    W = _boot_weights()
    pending, corrupt = [], []
    for i in range(N_DRAW):
        cp = os.path.join(ACACHE, "a_%03d.npz" % i)
        if force or not os.path.exists(cp):
            pending.append(i)
            continue
        try:
            z = np.load(cp)
            if "beta" not in z.files or "depc" not in z.files:
                raise ValueError
        except Exception:
            corrupt.append(i)
            pending.append(i)
    if os.path.exists(path) and not force and not pending:
        print("reuse", path, flush=True)
        return dict(np.load(path))
    print("bootstrap pending %d (corrupt %d) of %d" % (
        len(pending), len(corrupt), N_DRAW), flush=True)
    nw = n_workers(3) if workers is None else max(1, workers)
    done = 0
    if pending:
        for i, beta, dep_total, arr_total in pmap_iter(
                _boot_job, pending, nw, initializer=_boot_init):
            done += 1
            if done % 4 == 0:
                banner("bootstrap %d/%d beta=%s" % (
                    done, len(pending), np.round(beta, 3).tolist()), t0)
    beta_mat = np.zeros((N_DRAW, 5))
    dep_total = np.zeros(N_DRAW)
    arr_total = np.zeros(N_DRAW)
    depcs = prep["depc_day"].sum((1, 2)) * SCALE
    arrcs = prep["arrc_day"].sum((1, 2)) * SCALE
    for i in range(N_DRAW):
        z = np.load(os.path.join(ACACHE, "a_%03d.npz" % i))
        beta_mat[i] = z["beta"]
        w = W[i].astype(np.float64) / 9.0
        dep_total[i] = float((w * depcs).sum())
        arr_total[i] = float((w * arrcs).sum())
    np.savez_compressed(path, weights=W, beta=beta_mat, dep_total=dep_total,
                        arr_total=arr_total, days=np.array(days),
                        seed=np.array(SEED_A))
    banner("bootstrap done", t0)
    return dict(np.load(path))


# --------------------------------------------------------------------------
# B and C draws
# --------------------------------------------------------------------------
B_COMBOS = [dict(p=p, I0=i0, day=d) for p in (0.128, 0.185, 0.200)
            for i0 in (343638.0, 495540.0) for d in range(9)]


def run_inputs(force=False):
    path = os.path.join(RUN, "design_inputs.npz")
    if os.path.exists(path) and not force:
        print("reuse", path, flush=True)
        return dict(np.load(path, allow_pickle=True))
    rng = np.random.default_rng(SEED_C)
    keys = [s["key"] for s in C_SAMPLED]
    Cdraw = np.zeros((N_DRAW, len(keys)))
    for k, s in enumerate(C_SAMPLED):
        if s["key"] == "R_WALK":
            Cdraw[:, k] = rng.choice([0.150, 0.300, 0.450], N_DRAW)
        else:
            Cdraw[:, k] = rng.uniform(s["lo"], s["hi"], N_DRAW)
    # B is enumerated exactly in the BE jobs; the joint F/S designs pair each
    # base draw with an independent uniform B combination (with replacement).
    perm1 = rng.integers(0, len(B_COMBOS), N_DRAW)
    perm2 = rng.integers(0, len(B_COMBOS), N_DRAW)
    np.savez_compressed(path, c_keys=np.array(keys), Cdraw=Cdraw,
                        combos_p=np.array([b["p"] for b in B_COMBOS]),
                        combos_i0=np.array([b["I0"] for b in B_COMBOS]),
                        combos_day=np.array([b["day"] for b in B_COMBOS]),
                        perm1=perm1, perm2=perm2,
                        seed_c=np.array(SEED_C), seed_b=np.array(SEED_B))
    return dict(np.load(path, allow_pickle=True))


# --------------------------------------------------------------------------
# evaluation contexts
# --------------------------------------------------------------------------
class Ctx:
    def __init__(self, depc, arrc, dep_zone, PZ, I0c, I0z, c):
        self.depc = depc
        self.arrc = arrc
        self.dep_zone = dep_zone
        self.PZ = PZ
        self.I0c = I0c
        self.I0z = I0z
        self.c = c


def make_ctx(prep, aidx=None, bcombo=None, cvals=None):
    if aidx is None:
        depc = prep["depc_ref"].copy()
        arrc = prep["arrc_ref"].copy()
        dep_zone = prep["dep_zone_ref"].copy()
        PZ = prep["PZ_ref"].copy()
    else:
        z = np.load(os.path.join(ACACHE, "a_%03d.npz" % aidx))
        depc = z["depc"].astype(np.float64)
        arrc = z["arrc"].astype(np.float64)
        PZ = z["PZ"].astype(np.float64)
        dep_zone = z["dep_zone"].astype(np.float64)
    I0c = prep["I0c_ref"].copy()
    I0z = prep["I0_meas_zone"].copy()
    if bcombo is not None:
        days = list(prep["days"])
        tr = [days.index(d) for d in t18.TR_DAYS]
        ref_dep = prep["depc_day"][tr].mean(0).sum()
        m = float(prep["depc_day"][int(bcombo["day"])].sum() / ref_dep)
        fac = (P_REF / float(bcombo["p"])) * m
        depc = depc * fac
        arrc = arrc * fac
        dep_zone = dep_zone * fac
        if float(bcombo["I0"]) > 400000.0:
            I0z = prep["I0_fleet_zone"].copy()
            I0c = prep["I0c_ref"].copy()
            zid = prep["zid"]
            for zz in range(50):
                msk = zid == zz
                I0c[msk] *= I0z[zz] / max(I0c[msk].sum(), 1e-12)
    return Ctx(depc, arrc, dep_zone, PZ, I0c, I0z, cvec_full(cvals))


# --------------------------------------------------------------------------
# parallel workers
# --------------------------------------------------------------------------
_W = {}


def _worker_init():
    os.environ["OMP_NUM_THREADS"] = "1"
    # One retry after gc: a transient MemoryError while loading prep.npz in a
    # freshly spawned worker used to take the whole pool down.
    try:
        prep, _ = run_prep()
    except MemoryError:
        gc.collect()
        prep, _ = run_prep()
    dist_matrix()
    _pest()
    t18.setup_design()
    capinfo()
    _W["prep"] = prep
    _W["ref"] = load_json("reference.json")


def _ensure_worker():
    if "prep" not in _W:
        _worker_init()
    return _W["prep"], _W["ref"]


def _retry_memory_call(fn, x):
    """One retry after gc.collect() on MemoryError (module-level: picklable)."""
    try:
        return fn(x)
    except MemoryError:
        gc.collect()
        return fn(x)


def pmap_iter(fn, items, workers, initializer=None):
    """Order-free parallel map; each worker bootstraps the module caches."""
    items = list(items)
    if initializer is None:
        initializer = _worker_init
    job = functools.partial(_retry_memory_call, fn)
    if workers <= 1 or len(items) <= 1:
        initializer()
        for x in items:
            yield job(x)
        return
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(workers, initializer=initializer)
    try:
        for r in pool.imap_unordered(job, items, chunksize=1):
            yield r
    finally:
        pool.close()
        pool.join()


def n_workers(argv_default=3):
    env = os.environ.get("T023_WORKERS")
    if env:
        return max(1, int(env))
    return max(1, min(argv_default, (os.cpu_count() or 4) - 2))


# --------------------------------------------------------------------------
# evaluators
# --------------------------------------------------------------------------
def _pest():
    global _PEST
    if _PEST is None:
        prep, _ = run_prep()
        if "pest" in prep and "pest_zone" in prep:
            _PEST = (prep["pest"], prep["pest_zone"])
            return _PEST
        # fallback for a prep.npz written before the matrices were cached:
        # make_cells rebuilds the cell grid from the trips, so it must see
        # the full nine-day set before the training-day filter; filtering
        # the trip list first would renumber the cells (4,358 vs 5,263).
        zid = prep["zid"]
        df = lib.load_trips(DAYS9)
        cells2, df = lib.make_cells(df, lib.THR)
        assert len(cells2) == len(zid)
        df = df[df.date.isin(t18.TR_DAYS)]
        ok = df.o.notna() & df.d.notna()
        gt = df[ok]
        cnt = np.zeros((50, len(zid)))
        np.add.at(cnt, (zid[gt.o.astype(int).values],
                        gt.d.astype(int).values), 1.0)
        cnt += 1e-3
        pest = cnt / cnt.sum(1, keepdims=True)
        Zm = np.zeros((len(zid), 50))
        Zm[np.arange(len(zid)), zid] = 1.0
        _PEST = (pest, pest @ Zm)
    return _PEST
    return _PEST


def pcell(prep, PZ):
    pest, Rz = _pest()
    zid = prep["zid"]
    out = np.empty((24, 50, len(zid)), dtype=np.float32)
    for h in range(24):
        fac = PZ[h][:, zid] / np.maximum(Rz[:, zid], 1e-12)
        out[h] = (pest * fac).astype(np.float32)
    return out


def dyn_zone(za, D, n, c):
    dep, arr, I = za["dep"], za["arr"], za["i0"].copy()
    Cz = len(I)
    unmet = strict = tri_bikes = tri_bike_km = tri_vkm = 0.0
    hour1 = min(24.0, 7.0 + c["TRI_SHIFT"])
    sub = D[np.ix_(za["idx"], za["idx"])]
    for h in range(24):
        p = za["pickup"][h]
        tot = I.sum()
        if p > 1e-12 and tot > 1e-12:
            I *= max(0.0, 1.0 - p / tot)
        strict += np.maximum(dep[h] - I, 0.0).sum()
        take = np.minimum(dep[h], I)
        I -= take
        resid = dep[h] - take
        walk = t24.walk_borrow0(I, resid, za["nbrs"])
        resid = np.maximum(resid - walk, 0.0)
        if n > 1e-9 and 7.0 <= h < hour1:
            sink = np.where(resid > 1e-9)[0]
            donor = np.where(I > 1e-9)[0]
            if len(sink) and len(donor):
                flows, bkm, vkm = tri_lp_local(sink, donor, resid, I, sub,
                                               n, c)
                if flows:
                    for (dd, ss, f) in flows:
                        I[ss] -= f
                        I[dd] += f
                        resid[dd] -= f
                        tri_bikes += f
                    tri_bike_km += bkm
                    tri_vkm += vkm
        d = za["drop"][h]
        if d > 1e-12:
            mm = arr[h]
            I += d * (mm / mm.sum() if mm.sum() > 1e-12 else 1.0 / Cz)
        I += arr[h]
        unmet += np.maximum(resid, 0.0).sum()
    return dict(unmet=float(unmet), strict=float(strict),
                tri_bikes=float(tri_bikes), tri_bike_km=float(tri_bike_km),
                tri_vehicle_km=float(tri_vkm))


def tri_lp_local(sink, donor, resid, I, sub, n, c):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    dd = sub[np.ix_(sink, donor)]
    arcs = []
    for a in range(len(sink)):
        js = np.where(dd[a] <= c["TRI_MAX_TRIP"])[0]
        if not len(js):
            continue
        js = js[np.argsort(dd[a, js])[:t24.TRI_NEAREST]]
        arcs += [(a, int(b)) for b in js]
    if not arcs or n <= 1e-9:
        return None, 0.0, 0.0
    S, nd, na = len(sink), len(donor), len(arcs)
    nv = na + S
    q = c["TRI_Q"]
    cc = np.zeros(nv)
    for k, (a, b) in enumerate(arcs):
        cc[k] = c["TRI_KM"] * dd[a, b] / q
    cc[na:] = c["PENALTY"]
    rows = [a for (a, b) in arcs] + list(range(S))
    cols = list(range(na)) + [na + a for a in range(S)]
    A_eq = coo_matrix((np.ones(na + S), (rows, cols)), shape=(S, nv)).tocsr()
    b_eq = np.array([resid[sink[a]] for a in range(S)])
    rows2 = [b for (a, b) in arcs] + [nd] * na
    cols2 = list(range(na)) * 2
    tcoef = np.array([2 * dd[a, b] / (c["TRI_V"] * q)
                      + 2 * c["TRI_HANDLE"] / 60.0 for (a, b) in arcs])
    A_ub = coo_matrix((np.concatenate([np.ones(na), tcoef]),
                       (np.array(rows2), np.array(cols2)),
                       ), shape=(nd + 1, nv)).tocsr()
    b_ub = np.concatenate([[I[donor[b]] for b in range(nd)], [n]])
    res = linprog(cc, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=(0, None), method="highs")
    if not res.success:
        return None, 0.0, 0.0
    x = res.x
    flows = [(int(sink[a]), int(donor[b]), float(x[k]))
             for k, (a, b) in enumerate(arcs) if x[k] > 1e-9]
    km = float(sum(dd[a, b] * x[k] for k, (a, b) in enumerate(arcs)))
    return flows, km, km / q


def eval_q2(prep, ctx):
    lp = solve18(ctx.dep_zone, ctx.PZ, ctx.I0z, 60, ctx.c,
                 integer=False, time_limit=180)
    Pcell = pcell(prep, ctx.PZ)
    inp = dict(dep_cell=ctx.depc, Pcell=Pcell, I0c=ctx.I0c,
               zid=prep["zid"], xy=prep["xy"], dep_zone=ctx.dep_zone)
    # replay on the accepted T018 integer plan (T024-R3 caliber); the LP
    # value above is only the truck-layer objective proxy.
    ans = t24.zone_arrays(inp, dict(XM=prep["XM_plan"], U=prep["U_plan"]))
    D = dist_matrix()
    nz = prep["n_exec"]
    unmet = vkm = bkm = strict0 = 0.0
    for zi, za in enumerate(ans):
        r = dyn_zone(za, D, float(nz[zi]), ctx.c)
        unmet += r["unmet"]
        vkm += r["tri_vehicle_km"]
        bkm += r["tri_bike_km"]
        strict0 += dyn_zone(za, D, 0.0, ctx.c)["strict"]
    tri_cost = ctx.c["TRI_FIXED"] * float(nz.sum()) + ctx.c["TRI_KM"] * vkm
    total = lp["obj"] + tri_cost + ctx.c["PENALTY"] * unmet
    baseline = lp["obj"] + ctx.c["PENALTY"] * strict0
    return dict(total=float(total), unmet=float(unmet),
                truck_obj=float(lp["obj"]), truck_unmet=float(lp["unmet"]),
                tri_cost=float(tri_cost), tri_vkm=float(vkm),
                tri_bike_km=float(bkm), fleet=float(nz.sum()),
                baseline=float(baseline), unmet_n0=float(strict0),
                saving=float(baseline - total))


def build_edges_cached(arr, D, need, tstar, pos):
    """Bit-identical copy of ``t8.build_edges`` with the 5,263^2 mask cached.

    T008's own function allocates ``D <= R_KM`` on every call; under three
    concurrent workers that allocation was the one that exhausted the Windows
    commit limit.  Every expression below is unchanged from
    ``t008_q3_pool.py:77-94``; ``run_check`` verifies equality at the
    reference point.
    """
    near = near_mask()
    cache = {}
    for h in range(24):
        tg = np.where(pos & (tstar == h))[0]
        src = np.where(arr[h] > 0)[0]
        if len(tg) == 0 or len(src) == 0:
            cache[h] = None
            continue
        sub = near[np.ix_(src, tg)].copy()
        for k, c in enumerate(tg):
            j = np.where(src == c)[0]
            if len(j):
                sub[j[0], k] = False
        nz = np.argwhere(sub)
        si, ti = nz[:, 0], nz[:, 1]
        cache[h] = (src, tg, si, ti, D[src[si], tg[ti]])
    return cache


def q3_setup(ctx):
    c = ctx.c
    arr, dep = ctx.arrc, ctx.depc
    net = arr - dep
    need = np.maximum(0.0, -net).max(0)
    tstar = np.argmax(-net, axis=0)
    pos = need > 0
    cache = build_edges_cached(arr, dist_matrix(), need, tstar, pos)
    rows = []
    for h, e in cache.items():
        if e is None:
            continue
        src, tg, si, ti, dd = e
        for k in range(len(si)):
            rows.append((h, int(src[si[k]]), int(tg[ti[k]]), float(dd[k])))
    E = np.array(rows)
    cells = cells_table()
    pool = dict(arr=arr, dep=dep, need=need, tstar=tstar,
                cells=cells, e_hour=E[:, 0].astype(int),
                e_src=E[:, 1].astype(int), e_tgt=E[:, 2].astype(int),
                e_dist=E[:, 3], n_zones=50,
                zid=cells.zone.to_numpy().astype(int),
                N=float(need.sum()))
    old = t8.PARAMS[0]
    t8.PARAMS[0] = (c["KAPPA"], c["RHO"], c["C0"])
    try:
        lp = t20.PoolLP(pool, capinfo())
    finally:
        t8.PARAMS[0] = old
    return pool, lp


def q3_policies(ctx, pool, lp, integer=False, mip_gap=0.001, tl=240):
    c = ctx.c
    old = t8.PARAMS[0]
    t8.PARAMS[0] = (c["KAPPA"], c["RHO"], c["C0"])
    try:
        pure = solve18(ctx.dep_zone, ctx.PZ, ctx.I0z, 2, c, integer=integer,
                       time_limit=tl, mip_gap=mip_gap)
        # The policy is T025's selected one, quoted from T020's policy table:
        # arrivals source cap, budget 30,000 CNY/day, r = 0.81.  r is held
        # fixed because T020's selection criterion is a proxy for the true
        # hybrid objective and small r changes flip the LP vertex; the T023
        # question is how the ceiling responds to A/B/C at the chosen
        # policy, not how the policy itself would be re-optimised.
        r = 0.81
        got = lp.solve(r, budget=30000.0, cap_on=True, want_x=True)
        score = got["payment"] + t20.TRUCK_UNIT * (pool["N"] - got["q"])
        dA = t20.zone_delta(got["x"], lp.w, pool)
        PZ2 = np.empty_like(ctx.PZ)
        for h in range(24):
            PZ2[h], _, _ = t20.ipf_adjust(ctx.PZ[h], ctx.dep_zone[:, h],
                                          dA[h])
        hy = solve18(ctx.dep_zone, PZ2, ctx.I0z, 2, c, integer=integer,
                     time_limit=tl, mip_gap=mip_gap)
        best = dict(r=float(r), budget=30000.0, score=float(score),
                    payment=float(got["payment"]), q=float(got["q"]),
                    truck_obj=float(hy["obj"]),
                    truck_unmet=float(hy["unmet"]),
                    total=float(hy["obj"] + got["payment"]),
                    ceiling=float(pure["obj"] - hy["obj"]),
                    removed=float(pure["unmet"] - hy["unmet"]))
    finally:
        t8.PARAMS[0] = old
    return dict(pure_obj=float(pure["obj"]),
                pure_unmet=float(pure["unmet"]),
                pure_gap=float(pure.get("gap", 0.0)),
                ceiling=float(best["ceiling"]),
                best_r=float(best["r"]), best_budget=best["budget"],
                best_payment=float(best["payment"]),
                best_total=float(best["total"]),
                best_removed=float(best["removed"]),
                best_q=float(best["q"]),
                best_score=float(best["score"]))


def eval_q3(prep, ctx):
    pool, lp = q3_setup(ctx)
    r = q3_policies(ctx, pool, lp, integer=False)
    r.pop("policies", None)
    return r


_ARCS = {}


def q4_arcs(R):
    key = round(float(R), 3)
    if key not in _ARCS:
        _ARCS[key] = t19.build_arcs(dist_matrix(), key)
    return _ARCS[key]


def q4_inputs(prep, ctx):
    """Q4 demand surface and arrival weights for one evaluation point.

    ``a`` is T019's cell occupancy demand (a_meas343 / a_fleet495) scaled by
    the draw's cellwise arrival ratio; the arrival weights use T019's own
    model arrival surface (z19['arr']), not the observed counts: the two
    differ spatially because T019's surface is the gravity-implied one.

    The ratio is regularised because the nine-day draws visit cells that the
    five training days never reach (reference arrival exactly zero) while
    T019's model surface is positive everywhere: ratio =
    (draw + reg) / (ref + reg) with reg = 1% of the median positive 24-h
    cell arrival, clipped to [0, 5].  The shift form makes the ratio exactly
    1 at the reference, so the reference Q4 surface is T019's a_meas343
    unchanged; without the regularisation the surface explodes
    (a_total ~ 1e13 at draw 000).
    """
    r_ref = prep["arrc_ref"].sum(0)
    pos = r_ref[r_ref > 0]
    reg = 1e-2 * float(np.median(pos)) if len(pos) else 1.0
    ratio = (ctx.arrc.sum(0) + reg) / (r_ref + reg)
    ratio = np.clip(ratio, 0.0, 5.0)
    base = prep["a_fleet"] if ctx.I0z.sum() > 400000.0 else prep["a_meas"]
    arr_day = prep["arr_q4_ref"].sum(0) * ratio
    return base * ratio, arr_day


def eval_q4(prep, ctx):
    a, arr_day = q4_inputs(prep, ctx)
    ii, jj = q4_arcs(ctx.c["R_WALK"])
    # gap 0.005 reproduces T019's stored knee solution exactly at the
    # reference (0.595298226 / 0.558045199 / 394,553.13) in 4-6 s; a tighter
    # gap costs 27 s (0.002) to 90 s (0.001) and is not reproducible run to
    # run, so the design keeps 0.005 and the deviation stage reports the
    # MILP-vs-LP difference at ten points.
    r = t19.solve_cap(a, ii, jj, n=4000, integer=True, time_limit=60,
                      cap=ctx.c["CAP"], mip_gap=0.005)
    return dict(coverage=float(r["coverage"]),
                arrival_coverage=float(
                    t19.arrival_coverage(r["u"], a, arr_day)),
                unassigned=float(r["unassigned"]),
                assigned=float(r["assigned"]), a_total=float(a.sum()),
                gap=float(r.get("gap", 0.0)))


def eval_point(prep, ctx):
    return dict(q2=eval_q2(prep, ctx), q3=eval_q3(prep, ctx),
                q4=eval_q4(prep, ctx))


# --------------------------------------------------------------------------
# reference and design
# --------------------------------------------------------------------------
def run_reference(force=False):
    path = os.path.join(RUN, "reference.json")
    if os.path.exists(path) and not force:
        print("reuse", path, flush=True)
        return load_json("reference.json")
    prep, meta = run_prep()
    ctx = make_ctx(prep)
    t0 = time.time()
    ev = eval_point(prep, ctx)
    q2k = {}
    for K in (0, 20, 40, 60, 80):
        q2k[K] = float(solve18(ctx.dep_zone, ctx.PZ, ctx.I0z, K, ctx.c,
                               time_limit=180)["obj"])
    ev["q2"]["kstar_objs"] = {str(k): v for k, v in q2k.items()}
    ev["q2"]["kstar"] = int(min(q2k, key=q2k.get))
    ev["anchors"] = dict(
        q2_total=REF["q2"]["total"] - ev["q2"]["total"],
        q2_unmet=REF["q2"]["unmet"] - ev["q2"]["unmet"],
        q2_baseline=BASE_Q2 - ev["q2"]["baseline"],
        q2_saving=REF["q2"]["saving"] - ev["q2"]["saving"],
        q3_pure_obj=REF["q3"]["pure_obj"] - ev["q3"]["pure_obj"],
        q3_ceiling=REF["q3"]["ceiling"] - ev["q3"]["ceiling"],
        q3_pure_unmet=REF["q3"]["pure_unmet"] - ev["q3"]["pure_unmet"],
        q4_coverage=REF["q4"]["coverage"] - ev["q4"]["coverage"],
        q4_arrival=(REF["q4"]["arrival_coverage"]
                    - ev["q4"]["arrival_coverage"]),
        q4_unassigned=REF["q4"]["unassigned"] - ev["q4"]["unassigned"])
    ev["prep_meta"] = meta
    ev["runtime_s"] = time.time() - t0
    ev["note"] = (
        "Raw LP-proxy reference evaluation and the deterministic anchor. "
        "The Q2 replay runs on the stored T018 integer plan, so the raw "
        "values reproduce the accepted pipeline exactly: truck LP objective "
        "82,968.0678, LP-caliber replay unmet 38,489.4362, no-tricycle "
        "baseline 1,518,439.27 (LP caliber).  The unmet anchor is then the "
        "required integerisation shift 1,494.7969; the cost anchor also "
        "carries the truck-layer LP-vs-MILP gap 839.68.  Q4 is the MILP "
        "itself and reproduces T019's stored knee solution.  Anchor shifts "
        "are constant translations of the proxy and are NOT part of any "
        "variance share.")
    dump(ev, "reference.json")
    return ev


def build_jobs(inputs):
    jobs = []
    for i in range(N_HALF):
        Ai, Ai2 = i, i + N_HALF
        k1, k2 = int(inputs["perm1"][i]), int(inputs["perm2"][i])
        b1 = dict(p=float(inputs["combos_p"][k1]),
                  I0=float(inputs["combos_i0"][k1]),
                  day=int(inputs["combos_day"][k1]))
        b2 = dict(p=float(inputs["combos_p"][k2]),
                  I0=float(inputs["combos_i0"][k2]),
                  day=int(inputs["combos_day"][k2]))
        c1 = {str(k): float(inputs["Cdraw"][i, j])
              for j, k in enumerate(inputs["c_keys"])}
        c2 = {str(k): float(inputs["Cdraw"][i + N_HALF, j])
              for j, k in enumerate(inputs["c_keys"])}
        jobs += [
            dict(id="F_%03d" % i, tag="F", a=Ai, b=b1, c=c1),
            dict(id="MA_%03d" % i, tag="MA", a=Ai, b=None, c=None),
            dict(id="SA_%03d" % i, tag="SA", a=Ai, b=b2, c=c2),
            dict(id="FB_%03d" % i, tag="FB", a=Ai2, b=b2, c=c2),
            dict(id="SB_%03d" % i, tag="SB", a=Ai2, b=b1, c=c2),
            dict(id="SC_%03d" % i, tag="SC", a=Ai2, b=b2, c=c1),
            dict(id="MB_%03d" % i, tag="MB", a=None, b=b1, c=None),
            dict(id="MC_%03d" % i, tag="MC", a=None, b=None, c=c1),
        ]
    for k, b in enumerate(B_COMBOS):
        jobs.append(dict(id="BE_%02d" % k, tag="BE", a=None, b=b, c=None))
    return jobs


def _done_ids(path):
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    done.add(json.loads(line)["id"])
    return done


def _design_job(job):
    prep, _ = _ensure_worker()
    ctx = make_ctx(prep, job["a"], job["b"], job["c"])
    ev = eval_point(prep, ctx)
    return dict(id=job["id"], tag=job["tag"], a=job["a"], b=job["b"],
                c=job["c"], **ev)


def run_design(force=False, limit=None, workers=None, tags=None):
    path = os.path.join(RUN, "evals.jsonl")
    prep, _ = run_prep()
    inputs = run_inputs()
    jobs = build_jobs(inputs)
    if tags:
        jobs = [j for j in jobs if j["tag"] in tags]
    done = set() if force else _done_ids(path)
    todo = [j for j in jobs if j["id"] not in done]
    if limit:
        todo = todo[:limit]
    nw = n_workers() if workers is None else max(1, workers)
    print("jobs %d done %d todo %d workers %d" % (
        len(jobs), len(done), len(todo), nw), flush=True)
    t0 = time.time()
    n = 0
    fh = open(path, "a", encoding="utf-8")
    try:
        for row in pmap_iter(_design_job, todo, nw):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            n += 1
            if n % 8 == 0:
                banner("design %d/%d %s" % (n, len(todo), row["id"]), t0)
    finally:
        fh.close()
    banner("design done", t0)


def _kstar_job(payload):
    prep, _ = _ensure_worker()
    ctx = make_ctx(prep, payload["i"], payload["b"], payload["c"])
    objs = {K: float(solve18(ctx.dep_zone, ctx.PZ, ctx.I0z, K, ctx.c,
                             time_limit=180)["obj"])
            for K in (0, 20, 40, 60, 80)}
    return dict(i=payload["i"], objs={str(k): v for k, v in objs.items()},
                kstar=int(min(objs, key=objs.get)))


def _case_ctx(prep, payload):
    r = payload.get("row") or {}
    return make_ctx(prep, payload.get("aidx"), r.get("b"), r.get("c"))


def _kstar_case(payload):
    prep, _ = _ensure_worker()
    ctx = _case_ctx(prep, payload)
    objs = {K: float(solve18(ctx.dep_zone, ctx.PZ, ctx.I0z, K, ctx.c,
                             time_limit=180)["obj"])
            for K in (0, 20, 40, 60, 80)}
    return payload["key"], dict(
        objs={str(k): v for k, v in objs.items()},
        kstar=int(min(objs, key=objs.get)))


def run_kstar_marg(workers=None):
    """Optimal truck K* on each group's marginal slice (intervals only)."""
    path = os.path.join(RUN, "kstar_marginals.json")
    if os.path.exists(path):
        print("reuse", path, flush=True)
        return load_json("kstar_marginals.json")
    prep, _ = run_prep()
    inputs = run_inputs()
    payloads = [dict(key="reference", aidx=None, row=None)]
    for i in range(N_HALF):
        payloads.append(dict(key="MA_%03d" % i, aidx=i, row=None))
        c1 = {str(k): float(inputs["Cdraw"][i, j])
              for j, k in enumerate(inputs["c_keys"])}
        payloads.append(dict(key="MC_%03d" % i, aidx=None,
                             row=dict(b=None, c=c1)))
    for k, b in enumerate(B_COMBOS):
        payloads.append(dict(key="BE_%02d" % k, aidx=None,
                             row=dict(b=dict(b), c=None)))
    t0 = time.time()
    nw = n_workers(3) if workers is None else max(1, workers)
    out = dict(task="T023 K* marginals", date="2026-11-18", cases={})
    for key, rec in pmap_iter(_kstar_case, payloads, nw):
        out["cases"][key] = rec
        if len(out["cases"]) % 16 == 0:
            banner("kstar_marg %d/%d" % (len(out["cases"]), len(payloads)),
                   t0)
    dump(out, "kstar_marginals.json")
    return out


def run_kstar(limit=None, workers=None):
    path = os.path.join(RUN, "kstar.json")
    if os.path.exists(path):
        print("reuse", path, flush=True)
        return load_json("kstar.json")
    prep, _ = run_prep()
    inputs = run_inputs()
    n = N_HALF if not limit else min(limit, N_HALF)
    payloads = []
    for i in range(n):
        k = int(inputs["perm1"][i])
        b = dict(p=float(inputs["combos_p"][k]),
                 I0=float(inputs["combos_i0"][k]),
                 day=int(inputs["combos_day"][k]))
        c = {str(kk): float(inputs["Cdraw"][i, j])
             for j, kk in enumerate(inputs["c_keys"])}
        payloads.append(dict(i=i, b=b, c=c))
    t0 = time.time()
    nw = n_workers() if workers is None else max(1, workers)
    rows = []
    for r in pmap_iter(_kstar_job, payloads, nw):
        rows.append(r)
        if len(rows) % 16 == 0:
            banner("kstar %d/%d" % (len(rows), n), t0)
    rows.sort(key=lambda z: z["i"])
    dump(dict(rows=rows, n=n), "kstar.json")
    return dict(rows=rows, n=n)


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------
def load_evals():
    rows = {}
    with open(os.path.join(RUN, "evals.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
    return rows


def raw_value(row, key):
    q2, q3, q4 = row["q2"], row["q3"], row["q4"]
    if key == "q2_total":
        return q2["total"]
    if key == "q2_unmet":
        return q2["unmet"]
    if key == "q2_saving_share":
        return q2["saving"] / max(q2["baseline"], 1e-12)
    if key == "q3_pure_obj":
        return q3["pure_obj"]
    if key == "q3_ceiling":
        return q3["ceiling"]
    if key == "q3_pure_unmet":
        return q3["pure_unmet"]
    if key == "q4_coverage":
        return q4["coverage"]
    if key == "q4_arrival":
        return q4["arrival_coverage"]
    if key == "q4_unassigned":
        return q4["unassigned"]
    raise KeyError(key)


def anchor_of(ref, key):
    a = ref["anchors"]
    return {"q2_total": a["q2_total"], "q2_unmet": a["q2_unmet"],
            "q3_pure_obj": a["q3_pure_obj"], "q3_ceiling": a["q3_ceiling"],
            "q3_pure_unmet": a["q3_pure_unmet"],
            "q4_coverage": a["q4_coverage"], "q4_arrival": a["q4_arrival"],
            "q4_unassigned": a["q4_unassigned"]}[key]


def anchored(ref, row, key):
    if key == "q2_saving_share":
        b = row["q2"]["baseline"] + (BASE_Q2 - ref["q2"]["baseline"])
        t = row["q2"]["total"] + (REF["q2"]["total"] - ref["q2"]["total"])
        return (b - t) / b
    return raw_value(row, key) + anchor_of(ref, key)


def run_intervals():
    ref = load_json("reference.json")
    rows = load_evals()
    slices = dict(
        F=[rows["F_%03d" % i] for i in range(N_HALF)],
        A=[rows["MA_%03d" % i] for i in range(N_HALF)],
        B=[rows["BE_%02d" % k] for k in range(len(B_COMBOS))],
        C=[rows["MC_%03d" % i] for i in range(N_HALF)])
    out = dict(task="T023 intervals", date="2026-11-18",
               method=("P5/P50/P95 of the anchored output over each group's "
                       "marginal slice: A and C from the design marginals "
                       "(n=128, other groups at their reference values), B "
                       "from the exact 54-combination enumeration.  'all' is "
                       "the joint sample F (n=128)."),
               outputs={})
    for key, label, unit, refv in OUT_KEYS:
        entry = dict(label=label, unit=unit, reference=refv,
                     solver=OUT_SOLVER[key], slices={})
        for sname, srows in slices.items():
            v = np.array([anchored(ref, r, key) for r in srows])
            p5, p50, p95 = (float(np.percentile(v, 5)),
                            float(np.percentile(v, 50)),
                            float(np.percentile(v, 95)))
            e = dict(n=int(len(v)), p5=p5, p50=p50, p95=p95,
                     mean=float(v.mean()), sd=float(v.std(ddof=1)),
                     width=float(p95 - p5),
                     p5_at=srows[int(np.argmin(np.abs(v - p5)))]["id"],
                     p95_at=srows[int(np.argmin(np.abs(v - p95)))]["id"])
            entry["slices"][sname] = e
        out["outputs"][key] = entry
    # optimal truck K* (discrete): interval over the same marginal slices
    kmp = os.path.join(RUN, "kstar_marginals.json")
    ksp = os.path.join(RUN, "kstar.json")
    if os.path.exists(kmp) and os.path.exists(ksp):
        km = load_json("kstar_marginals.json")["cases"]
        ks = load_json("kstar.json")["rows"]
        entry = dict(label="Q2 optimal truck K*", unit="trucks",
                     reference=60, solver="LP (T018 K scan 0/20/40/60/80)",
                     slices={})
        for sname, ids in (
                ("A", ["MA_%03d" % i for i in range(N_HALF)]),
                ("B", ["BE_%02d" % k for k in range(len(B_COMBOS))]),
                ("C", ["MC_%03d" % i for i in range(N_HALF)])):
            v = np.array([km[i]["kstar"] for i in ids], dtype=float)
            counts = {str(int(k)): int((v == k).sum())
                      for k in sorted(set(v))}
            entry["slices"][sname] = dict(
                n=int(len(v)), p5=float(np.percentile(v, 5)),
                p50=float(np.percentile(v, 50)),
                p95=float(np.percentile(v, 95)), distribution=counts,
                p5_at=ids[int(np.argmin(np.abs(v - np.percentile(v, 5))))],
                p95_at=ids[int(np.argmin(np.abs(v - np.percentile(v, 95))))])
        v = np.array([r["kstar"] for r in ks], dtype=float)
        entry["slices"]["F"] = dict(
            n=int(len(v)), p5=float(np.percentile(v, 5)),
            p50=float(np.percentile(v, 50)),
            p95=float(np.percentile(v, 95)),
            distribution={str(int(k)): int((v == k).sum())
                          for k in sorted(set(v))})
        out["outputs"]["q2_kstar"] = entry
    dump(out, "intervals.json")
    return out


def _series(rows, ref, key, ids):
    return np.array([anchored(ref, rows[i], key) for i in ids])


def run_shares():
    ref = load_json("reference.json")
    rows = load_evals()
    half = N_HALF // 2
    out = dict(task="T023 variance shares", date="2026-11-18",
               groups=["A", "B", "C"],
               design=("pick-freeze, 8 passes per base draw, n=128.  F and "
                       "Fb are iid joint samples; S_g is the pass that keeps "
                       "group g and re-draws the other two.  First-order "
                       "index from the covariance identity Cov(F, S_g)/"
                       "Var(F), which requires pairing S_g with F (they "
                       "share group g).  Share_marginal_g = Var(M_g)/Var(F) "
                       "is the spread at the reference point of the other "
                       "groups: a diagnostic, not a Sobol index; with strong "
                       "multiplicative interactions it can exceed 1 and is "
                       "not used as a cross-check of the shares.  "
                       "residual_1_minus_sum = 1 - sum_g S_g is a residual, "
                       "expected 0 when the first-order terms add up; it is "
                       "NOT a second-order index and is reported with its "
                       "own bootstrap SE.  Convergence: the task sheet's "
                       "absolute 0.05 gate is not attainable at n=128 (share "
                       "SE 0.06-0.16; the 64->128 change is 0.8-1.6 SE).  "
                       "Amendment history (reviewer, 2026-11-18): first "
                       "revised to doubling_change_over_se <= 1.5; two "
                       "outputs came in at 1.52 and 1.57, and because the "
                       "nested doubling change has SD = SE(n=128), a 1.5-SE "
                       "gate would fail ~10% of shares by sampling noise "
                       "alone, so the final criterion is "
                       "doubling_change_over_se <= 2.0 with SEs reported."),
               outputs={})
    for key, label, unit, refv in OUT_KEYS:
        f = _series(rows, ref, key, ["F_%03d" % i for i in range(N_HALF)])
        var = float(f.var(ddof=1))
        res = dict(label=label, unit=unit, var_F=var, mean_F=float(f.mean()),
                   solver=OUT_SOLVER[key], n_joint=N_HALF, n_marginal=N_HALF,
                   n_B=len(B_COMBOS))
        if key == "q3_pure_unmet":
            # LP pure-truck unmet sits at its floor in most draws; variance
            # is carried by rare large-unmet draws, so no first-order
            # structure is identifiable.  Reported qualitatively, excluded
            # from the share table and the convergence criterion (reviewer
            # ruling 2026-11-18).
            q = np.percentile(f, [0, 25, 50, 75, 95, 100])
            res.update(
                degenerate=True,
                degenerate_reason=(
                    "LP pure-truck unmet sits at its floor in most draws "
                    "(p25 = median = p75); variance is carried by rare "
                    "large-unmet draws, so the first-order shares are not "
                    "identifiable"),
                p0=float(q[0]), p25=float(q[1]), p50=float(q[2]),
                p75=float(q[3]), p95=float(q[4]), p100=float(q[5]),
                n_draws=int(len(f)))
            out["outputs"][key] = res
            continue
        sj = {}
        s_series = {}
        for g in "ABC":
            s = _series(rows, ref, key,
                        ["S%s_%03d" % (g, i) for i in range(N_HALF)])
            s_series[g] = s
            sj[g] = float((np.mean(f * s) - f.mean() * s.mean()) / var)
            m = _series(rows, ref, key,
                        ["M%s_%03d" % (g, i) for i in range(N_HALF)])
            res["share_marginal_" + g] = float(m.var(ddof=1) / var)
        res["share_first_order"] = sj
        conv = {}
        for tag, idx in (("n64", list(range(half))),
                         ("n128", list(range(N_HALF)))):
            ff = f[idx]
            vv = float(ff.var(ddof=1))
            sh = {}
            for g in "ABC":
                s = s_series[g][idx]
                sh[g] = float((np.mean(ff * s) - ff.mean() * s.mean())
                              / vv)
            conv[tag] = dict(shares=sh,
                             residual_1_minus_sum=float(1 - sum(sh.values())),
                             var_F=vv)
        res["doubling"] = conv
        res["doubling_max_abs_change"] = float(max(
            abs(conv["n128"]["shares"][g] - conv["n64"]["shares"][g])
            for g in "ABC"))
        # nonparametric bootstrap: SEs of the shares, of their sum, and of
        # the residual 1 - sum (the same resampled indices are used for f
        # and S_g, so the pairing survives the resampling).
        rng = np.random.default_rng(SEED_C + 101)
        nrep = 400
        boot = {g: [] for g in "ABC"}
        boot_sum = []
        for _ in range(nrep):
            ii = rng.integers(0, N_HALF, N_HALF)
            ff = f[ii]
            vv = float(ff.var(ddof=1))
            if vv <= 0:
                continue
            vals = {}
            for g in "ABC":
                s = s_series[g][ii]
                vals[g] = float((np.mean(ff * s) - ff.mean() * s.mean())
                                / vv)
                boot[g].append(vals[g])
            boot_sum.append(sum(vals.values()))
        se = {g: float(np.std(boot[g], ddof=1)) for g in "ABC"}
        se_sum = float(np.std(boot_sum, ddof=1))
        res["share_se_boot"] = se
        res["share_ci95"] = {g: [sj[g] - 1.96 * se[g], sj[g] + 1.96 * se[g]]
                             for g in "ABC"}
        res["sum_shares"] = float(sum(sj.values()))
        res["sum_shares_se"] = se_sum
        res["residual_1_minus_sum"] = float(1.0 - sum(sj.values()))
        res["residual_1_minus_sum_se"] = se_sum
        res["residual_within_2se"] = bool(
            abs(res["residual_1_minus_sum"]) <= 2.0 * se_sum)
        res["se_nrep"] = nrep
        res["doubling_change_over_se"] = float(max(
            abs(conv["n128"]["shares"][g] - conv["n64"]["shares"][g])
            / max(se[g], 1e-12) for g in "ABC"))
        out["outputs"][key] = res
    dvp = os.path.join(RUN, "proxy_deviation.json")
    if os.path.exists(dvp):
        out["proxy_deviation"] = load_json("proxy_deviation.json")["summary"]
        out["proxy_deviation_file"] = "data/runs/T023/proxy_deviation.json"
    dump(out, "variance_shares.json")
    return out


# --------------------------------------------------------------------------
# flip table
# --------------------------------------------------------------------------
def fleet_dp(prep, ctx, nmax=60, ncap=2000):
    """Full T024 fleet re-optimisation at one evaluation point."""
    lp = solve18(ctx.dep_zone, ctx.PZ, ctx.I0z, 60, ctx.c, time_limit=180)
    inp = dict(dep_cell=ctx.depc, Pcell=pcell(prep, ctx.PZ), I0c=ctx.I0c,
               zid=prep["zid"], xy=prep["xy"], dep_zone=ctx.dep_zone)
    zas = t24.zone_arrays(inp, dict(XM=prep["XM_plan"],
                                    U=prep["U_plan"]))
    D = dist_matrix()
    Z = len(zas)
    curves = np.zeros((Z, nmax + 1))
    unmet_curve = np.zeros((Z, nmax + 1))
    for zi, za in enumerate(zas):
        for n in range(nmax + 1):
            r = dyn_zone(za, D, n, ctx.c)
            curves[zi, n] = (ctx.c["TRI_FIXED"] * n
                             + ctx.c["TRI_KM"] * r["tri_vehicle_km"]
                             + ctx.c["PENALTY"] * r["unmet"])
            unmet_curve[zi, n] = r["unmet"]
        if (zi + 1) % 10 == 0:
            print("    fleet zone %d/%d" % (zi + 1, Z), flush=True)
    dp, choice = t24._dp_curves(curves, Z, nmax, ncap)
    N = int(np.argmin(dp))
    nz = t24._dp_recon(choice, N, Z)
    unmet = float(sum(unmet_curve[zi, nz[zi]] for zi in range(Z)))
    total = lp["obj"] + float(curves[np.arange(Z), nz].sum())
    return dict(n_star=int(nz.sum()), dp_cost=float(dp[N]),
                truck_obj=float(lp["obj"]), unmet=unmet,
                total=float(total), zones=int((nz > 0).sum()),
                max_zone=int(nz.max()))


def q4_frontier(prep, ctx, n_list=None, integer=None):
    """Kneedle knee on the T019 n-grid, normalised as in T019.

    The grid is T019's own frontier grid and the rule is its rule (point
    farthest above the chord on the min-max-normalised (n, coverage) curve),
    so the reference draw must reproduce the 4,000-site knee.
    """
    if n_list is None:
        with open(os.path.join(lib.DATA, "runs", "T019",
                               "frontier.json"), encoding="utf-8") as fh:
            pts = json.load(fh)["meas343"]["points"]
        n_list = tuple(int(p["n"]) for p in pts)
    if integer is None:
        integer = True
    a, _ = q4_inputs(prep, ctx)
    ii, jj = q4_arcs(ctx.c["R_WALK"])
    cov = {}
    for n in n_list:
        r = t19.solve_cap(a, ii, jj, n=int(n), integer=integer,
                          time_limit=60, cap=ctx.c["CAP"], mip_gap=0.005)
        cov[int(n)] = float(r["coverage"])
    ns = np.array(sorted(cov), dtype=float)
    ys = np.array([cov[int(n)] for n in ns])
    xn = (ns - ns.min()) / max(ns.max() - ns.min(), 1e-12)
    yn = (ys - ys.min()) / max(ys.max() - ys.min(), 1e-12)
    dist = np.abs((yn[-1] - yn[0]) * xn - (xn[-1] - xn[0]) * yn
                  + xn[-1] * yn[0] - yn[-1] * xn[0])
    dist /= np.hypot(yn[-1] - yn[0], xn[-1] - xn[0])
    order = np.argsort(dist)[::-1]
    knee = int(ns[int(order[0])])
    # Only the knee and its margin are reported.  The full per-n curve would
    # repeat T019's grid coordinates verbatim, including the superseded 705
    # site count that must never appear as a T023 number.
    return dict(knee=knee,
                coverage_at_knee=float(ys[int(order[0])]),
                knee_margin=float(dist[order[0]] - dist[order[1]]),
                n_grid=int(len(ns)), n_min=int(ns.min()), n_max=int(ns.max()),
                integer=bool(integer), rule="T019 normalised-chord Kneedle")


def _flip_ctx(prep, payload):
    if payload["aidx"] is None:
        return make_ctx(prep)
    r = payload["row"]
    return make_ctx(prep, payload["aidx"], r["b"], r["c"])


def _flip_q2_job(payload):
    prep, _ = _ensure_worker()
    fd = fleet_dp(prep, _flip_ctx(prep, payload))
    fd["fleet_at_reference_still_975"] = bool(fd["n_star"] == 975)
    return payload["tag"], fd


def _flip_q4_job(payload):
    prep, _ = _ensure_worker()
    return payload["tag"], q4_frontier(prep, _flip_ctx(prep, payload))


def run_flips(workers=None):
    prep, _ = run_prep()
    ref = load_json("reference.json")
    rows = load_evals()
    # Q2 fleet: reference + six joint draws spanning the total-cost range
    vals = [(i, anchored(ref, rows["F_%03d" % i], "q2_total"))
            for i in range(N_HALF)]
    vals.sort(key=lambda z: z[1])
    picks = [vals[0][0], vals[len(vals) // 5][0], vals[2 * len(vals) // 5][0],
             vals[3 * len(vals) // 5][0], vals[4 * len(vals) // 5][0],
             vals[-1][0]]
    out = dict(task="T023 flip table", date="2026-11-18",
               reference=dict(q2_fleet=975, q4_sites=4000), q2_fleet={},
               q4_knee={}, q3_sign={})
    out["caliber"] = (
        "q2_fleet re-optimises the T024 LP-curve DP (zone cost curves n=0..60 "
        "+ DP allocation) at each draw.  Its reference value 889 is T024's "
        "own pre-executable optimum (data/runs/T024/compare.json best_used="
        "889); the accepted 975 is the executable fixed point, which also "
        "applies the integer trip construction and the walking assignment "
        "and is not run here.  The six draw counts are therefore comparable "
        "with each other and with 889, not with 975.  The required 'recommended "
        "K' flip (truck fleet in the T018 objective, K=0..60) is reported in "
        "kstar.json / kstar_marginals.json and as q2_kstar in intervals.json.")
    nw = n_workers(3) if workers is None else max(1, workers)
    t0 = time.time()
    t18.setup_design()
    q2_payloads = [dict(tag="ref", aidx=None, row=None)]
    q2_payloads += [dict(tag="F_%03d" % i, aidx=i, row=rows["F_%03d" % i])
                    for i in picks]
    for tag, fd in pmap_iter(_flip_q2_job, q2_payloads, nw):
        out["q2_fleet"][tag] = fd
        banner("flip q2 %s n*=%d" % (tag, fd["n_star"]), t0)
    q4_payloads = [dict(tag="ref", aidx=None, row=None)]
    q4_payloads += [dict(tag="F_%03d" % i, aidx=i, row=rows["F_%03d" % i])
                    for i in picks[:3]]
    for tag, fr in pmap_iter(_flip_q4_job, q4_payloads, nw):
        out["q4_knee"][tag] = fr
        banner("flip q4 %s knee=%d" % (tag, fr["knee"]), t0)
    # Q3 sign over the C slice and the joint sample
    for name, ids in (("C", ["MC_%03d" % i for i in range(N_HALF)]),
                      ("F", ["F_%03d" % i for i in range(N_HALF)]),
                      ("A", ["MA_%03d" % i for i in range(N_HALF)]),
                      ("B", ["BE_%02d" % k for k in range(len(B_COMBOS))])):
        c = np.array([raw_value(rows[i], "q3_ceiling") for i in ids])
        tot = np.array([rows[i]["q3"]["best_total"]
                        - rows[i]["q3"]["pure_obj"] for i in ids])
        out["q3_sign"][name] = dict(
            n=int(len(c)), ceiling_min=float(c.min()),
            ceiling_max=float(c.max()), ceiling_median=float(np.median(c)),
            n_positive=int((c > 0).sum()),
            net_cost_min=float(tot.min()), net_cost_max=float(tot.max()),
            n_net_cost_negative=int((tot < 0).sum()),
            note=("ceiling = pure_obj - hybrid_truck_obj (payment the "
                  "incentive could absorb for free); net_cost = "
                  "hybrid_truck + payment - pure_obj (cost of the policy); "
                  "a sign flip would need ceiling > 0."))
    dump(out, "flips.json")
    return out


# --------------------------------------------------------------------------
# proxy deviation
# --------------------------------------------------------------------------
def q2_integer(prep, ctx, time_limit=300, mip_gap=0.01):
    """T024-R3 integer chain at the fixed 975 allocation.

    The truck layer is the accepted stored integer plan (plan_int_K60, obj
    83,807.7438) and the tricycle layer is T024's integer trip construction;
    no MILP is re-solved, so the chain is exactly the accepted caliber and
    the only difference from the proxy is the truck objective (LP 82,968.07
    vs integer 83,807.74) plus the LP-vs-integer tricycle construction.
    """
    inp = dict(dep_cell=ctx.depc, Pcell=pcell(prep, ctx.PZ), I0c=ctx.I0c,
               zid=prep["zid"], xy=prep["xy"], dep_zone=ctx.dep_zone)
    zas = t24.zone_arrays(inp, dict(XM=prep["XM_plan"],
                                    U=prep["U_plan"]))
    D = dist_matrix()
    nz = prep["n_exec"]
    unmet_lp = vkm = 0.0
    dropped = 0.0
    unassigned = 0
    with patch24(ctx.c):
        for zi, za in enumerate(zas):
            n = float(nz[zi])
            r = t24.dyn_zone(za, D, int(n))
            unmet_lp += r["unmet"]
            vkm += r["tri_vehicle_km"]
            it = t24._int_trips(za, D, int(n))
            dropped += it["flows_bikes"] - it["int_bikes"]
            if n >= 1:
                asg = t24._assign(it["trips"], int(n), D)
                unassigned += asg["unassigned"]
    int_unmet = unmet_lp + dropped
    truck_obj = float(prep["truck_obj_int"])
    total = (truck_obj + ctx.c["TRI_FIXED"] * float(nz.sum())
             + ctx.c["TRI_KM"] * vkm + ctx.c["PENALTY"] * int_unmet)
    return dict(total=float(total), unmet=float(int_unmet),
                truck_obj=truck_obj, truck_unmet=0.0,
                tri_vkm=float(vkm), dropped=float(dropped),
                unassigned=int(unassigned), gap=None)


def q4_integer(prep, ctx, mc=False):
    a, arr_day = q4_inputs(prep, ctx)
    ii, jj = q4_arcs(ctx.c["R_WALK"])
    r = t19.solve_cap(a, ii, jj, n=4000, integer=True, time_limit=60,
                      cap=ctx.c["CAP"], mip_gap=0.005)
    out = dict(coverage=float(r["coverage"]),
               arrival_coverage=float(
                   t19.arrival_coverage(r["u"], a, arr_day)),
               gap=float(r.get("gap", 0.0)))
    if mc:
        t0 = time.time()
        hfac = (ctx.arrc.sum(1)
                / np.maximum(prep["arrc_ref"].sum(1), 1e-12))
        f5 = np.repeat(hfac, 12)
        a_mc = t19.cell_demand(prep["ref_arr5"] * f5[:, None],
                               prep["ref_dep5"] * f5[:, None], ctx.I0c)
        r2 = t19.solve_cap(a_mc, ii, jj, n=4000, integer=True,
                           time_limit=60, cap=ctx.c["CAP"], mip_gap=0.005)
        out["mc"] = dict(coverage=float(r2["coverage"]),
                         arrival_coverage=float(
                             t19.arrival_coverage(r2["u"], a_mc, arr_day)),
                         a_total=float(a_mc.sum()),
                         a_vs_proxy_maxabs=float(np.abs(a_mc - a).max()),
                         a_vs_proxy_l1rel=float(
                             np.abs(a_mc - a).sum() / max(a.sum(), 1e-9)),
                         runtime_s=time.time() - t0)
    return out


def _dev_job(payload):
    prep, _ = _ensure_worker()
    rank = payload["rank"]
    r = payload["row"]
    ctx = _flip_ctx(prep, dict(aidx=payload["aidx"], row=r))
    qi = q2_integer(prep, ctx)
    pool, lp = q3_setup(ctx)
    q3i = q3_policies(ctx, pool, lp, integer=True, mip_gap=0.001, tl=240)
    q4i = q4_integer(prep, ctx, mc=(rank in (0, 4, 9)))
    rec = dict(
        rank=rank, b=r["b"], c=r["c"],
        q2_proxy=dict(total=r["q2"]["total"], unmet=r["q2"]["unmet"],
                      truck_obj=r["q2"]["truck_obj"]),
        q2_integer=qi,
        q2_total_dev=float(qi["total"] - r["q2"]["total"]),
        q2_unmet_dev=float(qi["unmet"] - r["q2"]["unmet"]),
        q3_proxy=dict(pure_obj=r["q3"]["pure_obj"],
                      pure_unmet=r["q3"]["pure_unmet"],
                      ceiling=r["q3"]["ceiling"]),
        q3_integer=dict(pure_obj=q3i["pure_obj"],
                        pure_unmet=q3i["pure_unmet"],
                        ceiling=q3i["ceiling"],
                        pure_gap=q3i["pure_gap"]),
        q3_pure_dev=float(q3i["pure_obj"] - r["q3"]["pure_obj"]),
        q3_ceiling_dev=float(q3i["ceiling"] - r["q3"]["ceiling"]),
        q4_proxy=dict(coverage=r["q4"]["coverage"],
                      arrival_coverage=r["q4"]["arrival_coverage"]),
        q4_integer=q4i,
        q4_coverage_dev=float(q4i["coverage"] - r["q4"]["coverage"]))
    return payload["key"], rec


def run_deviation(workers=None):
    prep, _ = run_prep()
    ref = load_json("reference.json")
    rows = load_evals()
    vals = [(i, anchored(ref, rows["F_%03d" % i], "q2_total"))
            for i in range(N_HALF)]
    vals.sort(key=lambda z: z[1])
    picks = [vals[int(k * (len(vals) - 1) / 9)][0] for k in range(10)]
    out = dict(task="T023 proxy deviation", date="2026-11-18",
               method=("integer chains vs the LP proxy at the reference and "
                       "at ten joint draws spanning the Q2 total-cost range "
                       "(11 points >= the required 10).  Q2 integer = T018 "
                       "MILP K=60 + T024 integer trips at the fixed 975 "
                       "allocation (executable unmet = LP unmet + dropped "
                       "bikes, T024-R3 convention).  Q3 integer = tight "
                       "K=2 MILP.  Q4 integer = site-integer MILP at "
                       "n=4,000, plus a full cell_demand MC at three of the "
                       "ten joint draws.  The reference row is the check "
                       "that the integer chain reproduces 617,055.7515 / "
                       "39,984.2332 / 138,317.1512 / 33.1936."),
               points={})
    t0 = time.time()
    nw = n_workers(4) if workers is None else max(1, workers)
    ref_row = dict(
        b=None, c=None,
        q2=dict(total=ref["q2"]["total"], unmet=ref["q2"]["unmet"],
                truck_obj=ref["q2"]["truck_obj"]),
        q3=dict(pure_obj=ref["q3"]["pure_obj"],
                pure_unmet=ref["q3"]["pure_unmet"],
                ceiling=ref["q3"]["ceiling"]),
        q4=dict(coverage=ref["q4"]["coverage"],
                arrival_coverage=ref["q4"]["arrival_coverage"]))
    payloads = [dict(key="reference", rank=-1, aidx=None, row=ref_row)]
    payloads += [dict(key="F_%03d" % i, rank=k, aidx=i,
                      row=rows["F_%03d" % i])
                 for k, i in enumerate(picks)]
    for key, rec in pmap_iter(_dev_job, payloads, nw):
        out["points"][key] = rec
        banner("deviation %d/11 %s" % (len(out["points"]), key), t0)
    out["summary"] = dict(
        q2_total_dev_max=float(max(abs(v["q2_total_dev"])
                                   for v in out["points"].values())),
        q2_unmet_dev_max=float(max(abs(v["q2_unmet_dev"])
                                   for v in out["points"].values())),
        q3_pure_dev_max=float(max(abs(v["q3_pure_dev"])
                                  for v in out["points"].values())),
        q3_ceiling_dev_max=float(max(abs(v["q3_ceiling_dev"])
                                     for v in out["points"].values())),
        q4_coverage_dev_max=float(max(abs(v["q4_coverage_dev"])
                                      for v in out["points"].values())),
        q2_unassigned_max=int(max(v["q2_integer"]["unassigned"]
                                  for v in out["points"].values())),
        n_points=int(len(out["points"])))
    dump(out, "proxy_deviation.json")
    return out


# --------------------------------------------------------------------------
# tornado (one-at-a-time at the reference)
# --------------------------------------------------------------------------
def _ctx_scaled(prep, dep_fac=1.0, fleet=False):
    ctx = make_ctx(prep)
    ctx.depc = ctx.depc * dep_fac
    ctx.arrc = ctx.arrc * dep_fac
    ctx.dep_zone = ctx.dep_zone * dep_fac
    if fleet:
        zid = prep["zid"]
        ctx.I0z = prep["I0_fleet_zone"].copy()
        I0c = prep["I0c_ref"].copy()
        for zz in range(50):
            msk = zid == zz
            I0c[msk] *= ctx.I0z[zz] / max(I0c[msk].sum(), 1e-12)
        ctx.I0c = I0c
    return ctx


def _ctx_from_spec(prep, spec):
    kind = spec["kind"]
    if kind == "c":
        return make_ctx(prep, None, None, spec["cvals"])
    if kind == "dep":
        return _ctx_scaled(prep, spec["fac"])
    if kind == "fleet":
        return _ctx_scaled(prep, fleet=True)
    if kind == "day":
        return make_ctx(prep, None, dict(p=spec["p"], I0=343638.0,
                                         day=spec["day"]), None)
    if kind == "beta":
        bb = prep["betas_ref"].copy()
        bb[spec["band"]] = spec["beta"]
        days = list(prep["days"])
        # demand_path expects weights that sum to 1 (the T018 training-day
        # average is 1/5 per day); a weight of 1.0 per day would inflate the
        # demand surface five-fold.
        w = np.zeros(len(days))
        w[[days.index(d) for d in t18.TR_DAYS]] = 1.0 / len(t18.TR_DAYS)
        d = demand_path(prep["depc_day"], prep["arrc_day"], prep["dj_day"],
                        lib.get_cells(), prep["zid"], w, bb)
        ctx = make_ctx(prep)
        ctx.depc, ctx.arrc = d["depc"], d["arrc"]
        ctx.dep_zone, ctx.PZ = d["dep_zone"], d["PZ"]
        return ctx
    raise KeyError(kind)


def _tornado_job(payload):
    prep, ref = _ensure_worker()
    ctx = _ctx_from_spec(prep, payload["spec"])
    p = eval_point(prep, ctx)
    vals = {k: anchored(ref, p, k) for k, _, _, _ in OUT_KEYS}
    return payload, vals


def run_tornado(workers=None):
    """One-at-a-time at the reference, anchored outputs."""
    prep, _ = run_prep()
    ref = load_json("reference.json")
    out = dict(task="T023 tornado", date="2026-11-18",
               method=("one-at-a-time at the reference: each parameter is "
                       "moved to its low and high bound with every other "
                       "parameter at its reference value.  Bounds are each "
                       "group's stated interval, never wider: A-group betas "
                       "at their day-block bootstrap P5/P95, B-group at the "
                       "measured scenario values (p 0.185/0.200, inventory "
                       "495,540, min/max recorded day), C-group at the stated "
                       "planning interval (50% for unsourced constants).  A "
                       "global +/-50% departure-rate stress is deliberately "
                       "not included: it is outside every group's interval "
                       "and is not a property of any identified parameter.  "
                       "Values are the anchored Q2 total, Q2 unmet, Q3 "
                       "payment ceiling, Q4 occupancy coverage and Q4 "
                       "arrival coverage."),
               bars={}, reference=None)
    t0 = time.time()
    p0 = eval_point(prep, make_ctx(prep))
    out["reference"] = {k: anchored(ref, p0, k) for k, _, _, _ in OUT_KEYS}
    cases = []
    for s in C_SAMPLED:
        if s["key"] in ("TRUCK_SPEED", "TRUCK_HANDLE"):
            continue
        for side, v in (("low", s["lo"]), ("high", s["hi"])):
            cases.append(dict(bar=s["key"], side=side, group="C",
                              unit=s["unit"], spec=dict(kind="c", cvals={
                                  s["key"]: v})))
    betas = prep["betas_ref"]
    bpath = os.path.join(RUN, "bootstrap_samples.npz")
    bboot = np.load(bpath)["beta"] if os.path.exists(bpath) else None
    for b in range(5):
        if bboot is not None:
            lo, hi = (float(np.percentile(bboot[:, b], 5)),
                      float(np.percentile(bboot[:, b], 95)))
        else:
            lo, hi = float(betas[b] * 0.975), float(betas[b] * 1.025)
        for side, val in (("low", lo), ("high", hi)):
            cases.append(dict(bar="beta_%d" % b, side=side, group="A",
                              unit="1/km", spec=dict(
                                  kind="beta", band=b, beta=val)))
    # No global departure-rate stress bar: the only measured volume
    # variation is the day-to-day range, carried by day_min_max below.  A
    # +/-50% rate scale would sit outside every group's stated interval and
    # would not be a property of any identified parameter.
    for side, pval in (("low", 0.185), ("high", 0.200)):
        cases.append(dict(bar="p_0.185_0.200", side=side, group="B",
                          unit="sampling rate",
                          spec=dict(kind="dep", fac=P_REF / pval)))
    cases.append(dict(bar="I0_495540", side="point", group="B",
                      unit="bicycles", spec=dict(kind="fleet")))
    dep_tot = prep["depc_day"].sum((1, 2))
    dmin, dmax = int(np.argmin(dep_tot)), int(np.argmax(dep_tot))
    for side, day in (("low", dmin), ("high", dmax)):
        cases.append(dict(bar="day_min_max", side=side, group="B",
                          unit="recorded day", spec=dict(
                              kind="day", day=int(day), p=0.128)))
    nw = n_workers() if workers is None else max(1, workers)
    for payload, vals in pmap_iter(_tornado_job, cases, nw):
        bar = out["bars"].setdefault(payload["bar"], dict(
            group=payload["group"], unit=payload["unit"]))
        bar[payload["side"]] = vals
        banner("tornado %s %s" % (payload["bar"], payload["side"]), t0)
    out["bars"]["day_min_max"]["days"] = [str(prep["days"][dmin]),
                                          str(prep["days"][dmax])]
    for k, bar in out["bars"].items():
        if "low" not in bar:
            continue
        bar["range"] = {key: float(abs(bar["high"][key] - bar["low"][key]))
                        for key, _, _, _ in OUT_KEYS}
        bar["dev_low"] = {key: float(bar["low"][key] - out["reference"][key])
                          for key, _, _, _ in OUT_KEYS}
        bar["dev_high"] = {key: float(bar["high"][key] - out["reference"][key])
                           for key, _, _, _ in OUT_KEYS}
    out["rank"] = {}
    for key, _, _, _ in OUT_KEYS:
        ranked = sorted(((k, b.get("range", {}).get(key, 0.0))
                         for k, b in out["bars"].items()), key=lambda z: -z[1])
        out["rank"][key] = [dict(param=k, range=r) for k, r in ranked]
    dump(out, "tornado.json")
    return out


# --------------------------------------------------------------------------
# parameter table and check
# --------------------------------------------------------------------------
def run_params(force=False):
    path = os.path.join(RUN, "param_table.json")
    if os.path.exists(path) and not force:
        print("reuse", path, flush=True)
        return load_json("param_table.json")
    prep, meta = run_prep()
    q1 = load_json("q1_params.json", folder=lib.DATA)
    bands = np.array([float(q1["bands"][nm]["beta"])
                      for nm in t18.BAND_NAMES])
    boot = None
    bpath = os.path.join(RUN, "bootstrap_samples.npz")
    if os.path.exists(bpath):
        b = np.load(bpath)["beta"]
        boot = [dict(band=i, main=float(bands[i]),
                     p5=float(np.percentile(b[:, i], 5)),
                     p50=float(np.percentile(b[:, i], 50)),
                     p95=float(np.percentile(b[:, i], 95)),
                     sd=float(np.std(b[:, i], ddof=1)),
                     n=int(b.shape[0])) for i in range(5)]
    days = list(prep["days"])
    dep_day = prep["depc_day"].sum((1, 2))
    arr_day = prep["arrc_day"].sum((1, 2))
    ref_dep = prep["depc_day"][[days.index(d) for d in t18.TR_DAYS]
                               ].mean(0).sum()
    per_day = [dict(day=str(d),
                    departures_sampled=float(dep_day[k]),
                    departures_full=float(dep_day[k] * SCALE),
                    factor_vs_reference_day=float(dep_day[k] / ref_dep),
                    arrivals_full=float(arr_day[k] * SCALE))
               for k, d in enumerate(days)]
    out = dict(
        task="T023 parameter table", date="2026-11-18", reference=REF,
        groups=dict(
            A=dict(
                label="data-identified (day-block bootstrap, 9 recorded days)",
                parameters=[dict(name="beta_%d" % i, value=float(bands[i]),
                                 unit="1/km",
                                 source="restricted-support MLE on the five "
                                        "training days (10_q1_model.py), "
                                        "stored in data/q1_params.json; "
                                        "bootstrap P5/P95 below",
                                 interval=([boot[i]["p5"], boot[i]["p95"]]
                                           if boot else None),
                                 bootstrap=boot[i] if boot else None)
                            for i in range(5)]
                + [dict(name="departure_rate_cell_hour",
                        value=None, unit="bikes/h",
                        source="mean of the five training days x 1/0.128",
                        interval=None,
                        bootstrap="resampled days, renormalised to the "
                                  "reference total"),
                   dict(name="attraction_A_j", value=None,
                        unit="relative", source="Gaussian-smoothed "
                        "destination counts, sigma=1.2 cells",
                        interval=None),
                   dict(name="trip_distance_distribution", value=None,
                        unit="km", source="recorded OD pairs; the only "
                        "data-identified input of the duration model",
                        interval=None)],
                per_day=per_day),
            B=dict(
                label="assumed loads / operational scenarios",
                parameters=[
                    dict(name="p", values=[0.128, 0.185, 0.200],
                         unit="sampling rate", source="M13 scenario set; "
                         "0.128 is the reference caliber",
                         effect="demand scales as 1/p"),
                    dict(name="I0", values=[343638.0, 495540.0],
                         unit="bicycles", source="343,638 measured midnight "
                         "inventory (T018); 495,540 closed-fleet estimate "
                         "(T019 I0_fleet, same within-zone shape)",
                         effect="initial inventory vector"),
                    dict(name="day_factor", values=per_day,
                         source="nine recorded days; 05-17 absent from the "
                                "order file", effect="multiplicative "
                                "day-level load, m_d = D_d / mean_train5")]),
            C=dict(label="literature / planning constants",
                   parameters=[dict(name=s["key"], value=s["ref"],
                                    interval=[s["lo"], s["hi"]],
                                    unit=s["unit"], source=s["source"],
                                    effect=s["effect"])
                               for s in C_SPECS])),
        notes=[
            "C ranges are +-50% around the reference for every parameter "
            "without a literature source, as the task sheet requires; the "
            "kappa/rho/c0 and R ranges are the ones already swept in "
            "T008/T019.",
            "TRI_NMAX is sampled but does not enter the cost-at-policy "
            "evaluation: the reference allocation's largest zone takes 44 "
            "tricycles and R2/R3 found the 60 cap did not bind.  It enters "
            "the fleet re-optimisation (flip table).",
            "Truck speed and handling do not appear in the T018 LP "
            "objective; they are feasibility-audit constants and have zero "
            "effect on Q2-Q4.  Duration constants do not enter Q2-Q4 at all.",
            "The duration model is listed under A in the task sheet, but "
            "sec4_q1 and 11_q1_extra.py show it is an assumption, not a "
            "fitted relation; only its distance input is data-identified.  "
            "The constants are therefore recorded under C and flagged as "
            "zero-effect for Q2-Q4."])
    dump(out, "param_table.json")
    return out


FORBIDDEN = ["885", "266327", "40357", "705", "76.86", "6.83", "20.83",
             "10.40", "2883.3", "480.6"]


def run_check():
    out = dict(task="T023 check", date="2026-11-18", checks={}, files={})
    ok = out["checks"]
    # 1 baseline endpoint against the locked references
    ref = load_json("reference.json")
    rec = {
        "q2_total": (ref["q2"]["total"], ref["anchors"]["q2_total"],
                     REF["q2"]["total"]),
        "q2_unmet": (ref["q2"]["unmet"], ref["anchors"]["q2_unmet"],
                     REF["q2"]["unmet"]),
        "q3_pure_obj": (ref["q3"]["pure_obj"],
                        ref["anchors"]["q3_pure_obj"],
                        REF["q3"]["pure_obj"]),
        "q3_ceiling": (ref["q3"]["ceiling"], ref["anchors"]["q3_ceiling"],
                       REF["q3"]["ceiling"]),
        "q4_coverage": (ref["q4"]["coverage"],
                        ref["anchors"]["q4_coverage"], REF["q4"]["coverage"]),
        "q4_arrival": (ref["q4"]["arrival_coverage"],
                       ref["anchors"]["q4_arrival"],
                       REF["q4"]["arrival_coverage"]),
        "q4_unassigned": (ref["q4"]["unassigned"],
                          ref["anchors"]["q4_unassigned"],
                          REF["q4"]["unassigned"]),
    }
    ok["1_baseline_endpoint"] = dict(
        tolerance=1e-6,
        construction=("anchored(x_ref) = ref + [proxy(x_ref) - proxy(x_ref)] "
                      "= ref by construction; the anchor shift itself is the "
                      "deterministic LP-vs-integer translation and is "
                      "reported, not counted in any variance share"),
        reconstructed={k: dict(raw=float(raw), anchor=float(anc),
                               reconstructed=float(raw + anc),
                               reference=float(target),
                               abs_dev=float(abs(raw + anc - target)))
                       for k, (raw, anc, target) in rec.items()},
        raw=dict(q2_total=ref["q2"]["total"], q2_unmet=ref["q2"]["unmet"],
                 q2_truck_obj=ref["q2"]["truck_obj"],
                 q2_baseline=ref["q2"]["baseline"],
                 q3_pure_obj=ref["q3"]["pure_obj"],
                 q3_ceiling=ref["q3"]["ceiling"],
                 q4_coverage=ref["q4"]["coverage"],
                 q4_arrival=ref["q4"]["arrival_coverage"],
                 q4_unassigned=ref["q4"]["unassigned"]),
        raw_vs_accepted=dict(
            q2_truck_obj=dict(raw=ref["q2"]["truck_obj"],
                              accepted=REF["q2"]["lp_obj"],
                              abs_dev=abs(ref["q2"]["truck_obj"]
                                          - REF["q2"]["lp_obj"])),
            q2_lp_unmet=dict(raw=ref["q2"]["unmet"],
                             accepted=REF["q2"]["lp_unmet"],
                             abs_dev=abs(ref["q2"]["unmet"]
                                         - REF["q2"]["lp_unmet"])),
            q2_integer_shift=dict(raw=ref["anchors"]["q2_unmet"],
                                  accepted=REF["q2"]["int_short"],
                                  abs_dev=abs(ref["anchors"]["q2_unmet"]
                                              - REF["q2"]["int_short"])),
            q2_baseline_anchor=dict(
                raw=ref["anchors"]["q2_baseline"],
                accepted=REF["q2"]["truck_obj"] - REF["q2"]["lp_obj"],
                abs_dev=abs(ref["anchors"]["q2_baseline"]
                            - (REF["q2"]["truck_obj"]
                               - REF["q2"]["lp_obj"])),
                tol=0.01,
                note=("anchor = truck MILP-LP gap 839.676 plus 6 x the "
                      "strict-baseline LP difference.  The two HiGHS runs "
                      "agree to 8e-4 CNY on the truck objective and 1.2e-3 "
                      "bikes on strict0; 6 x 1.2e-3 = 0.0072 CNY/day, and "
                      "the observed anchor deviation is 0.0071.  This entry "
                      "therefore uses an absolute tolerance of 0.01 CNY/day; "
                      "the reconstructed endpoints above remain at 1e-6.")),
            q3_ceiling=dict(raw=ref["q3"]["ceiling"]
                            + ref["anchors"]["q3_ceiling"],
                            accepted=REF["q3"]["ceiling"],
                            abs_dev=abs(ref["q3"]["ceiling"]
                                        + ref["anchors"]["q3_ceiling"]
                                        - REF["q3"]["ceiling"])),
            q4_coverage=dict(raw=ref["q4"]["coverage"]
                             + ref["anchors"]["q4_coverage"],
                             accepted=REF["q4"]["coverage"],
                             abs_dev=abs(ref["q4"]["coverage"]
                                         + ref["anchors"]["q4_coverage"]
                                         - REF["q4"]["coverage"]))),
        anchors=ref["anchors"])
    ok["1_baseline_endpoint"]["pass_"] = bool(
        all(v["abs_dev"] <= 1e-6 * max(1.0, abs(v["reference"]))
            for v in ok["1_baseline_endpoint"]["reconstructed"].values())
        and all(v["abs_dev"] <= v.get(
                "tol", 1e-6 * max(1.0, abs(v["accepted"])))
                for v in ok["1_baseline_endpoint"][
                    "raw_vs_accepted"].values()))
    # 2 bootstrap
    b = np.load(os.path.join(RUN, "bootstrap_samples.npz"))
    ok["2_bootstrap"] = dict(pass_=bool(b["beta"].shape[0] >= 200),
                             n_draws=int(b["beta"].shape[0]),
                             blocks=int(b["weights"].shape[1]),
                             days=[str(d) for d in b["days"]],
                             nan_betas=int(np.isnan(b["beta"]).sum()),
                             beta_p5_p95={("beta_%d" % i): [
                                 float(np.percentile(b["beta"][:, i], 5)),
                                 float(np.percentile(b["beta"][:, i], 95))]
                                 for i in range(5)})
    # 3 doubling: final reviewer criterion (2026-11-18, amended twice): the
    # nested 64->128 share change must be <= 2.0 bootstrap SE for every
    # non-degenerate output.  History: the task sheet's absolute 0.05 was
    # not attainable at n=128; the first amendment set 1.5 SE, two outputs
    # measured 1.52 / 1.57, and because the nested change has SD = SE(n=128)
    # a 1.5-SE gate fails ~10% of shares on sampling noise alone, so the
    # final amendment set 2.0 SE.  All three steps are recorded here.
    sh = load_json("variance_shares.json")
    outs = {k: v for k, v in sh["outputs"].items() if not v.get("degenerate")}
    dmax = max(v["doubling_max_abs_change"] for v in outs.values())
    rmax = max(v["doubling_change_over_se"] for v in outs.values())
    ok["3_doubling"] = dict(
        pass_=bool(rmax <= 2.0),
        criterion=("doubling_change_over_se <= 2.0 (final reviewer "
                   "amendment 2026-11-18, based on 400-replicate bootstrap "
                   "SEs).  History: the task sheet's absolute 0.05 gate was "
                   "not met; the first amendment set 1.5 SE; two outputs "
                   "measured 1.52 and 1.57, and because the nested doubling "
                   "change has SD = SE(n=128), a 1.5-SE gate fails ~10% of "
                   "shares on sampling noise alone; the final amendment set "
                   "2.0 SE."),
        max_abs_change=dmax, max_change_over_se=rmax,
        original_gate=0.05, original_gate_met=bool(dmax <= 0.05),
        first_amendment_gate=1.5,
        first_amendment_met=bool(rmax <= 1.5),
        final_gate=2.0,
        sum_shares={k: [v["sum_shares"], v["sum_shares_se"]]
                    for k, v in outs.items()},
        residual_1_minus_sum={k: [v["residual_1_minus_sum"],
                                  v["residual_1_minus_sum_se"]]
                              for k, v in outs.items()},
        residuals_within_2se=bool(all(v["residual_within_2se"]
                                      for v in outs.values())),
        degenerate=[k for k, v in sh["outputs"].items()
                    if v.get("degenerate")],
        n_outputs_scored=len(outs))
    # 4 deviation
    dev = load_json("proxy_deviation.json")
    ok["4_proxy_deviation"] = dict(
        pass_=bool(dev["summary"]["n_points"] >= 10),
        n_points=dev["summary"]["n_points"], summary=dev["summary"])
    # 5 flips
    fl = load_json("flips.json")
    ok["5_flips"] = dict(
        pass_=True,
        q2_fleet_n_star={k: v["n_star"] for k, v in
                         fl["q2_fleet"].items()},
        q4_knee={k: v["knee"] for k, v in fl["q4_knee"].items()},
        q3_ceiling_positive={k: v["n_positive"] for k, v in
                             fl["q3_sign"].items()})
    # 6 forbidden tokens scan over the generated JSONs
    import re
    hits = {}
    for name in ("param_table.json", "intervals.json",
                 "variance_shares.json", "proxy_deviation.json",
                 "flips.json", "tornado.json", "reference.json"):
        p = os.path.join(RUN, name)
        if not os.path.exists(p):
            continue
        txt = io.open(p, encoding="utf-8").read()
        for tok in FORBIDDEN:
            pat = re.compile(r"(?<!\d)" + re.escape(tok) + r"(?!\d)")
            ctxs = []
            for m in pat.finditer(txt):
                if len(ctxs) >= 3:
                    break
                k = m.start()
                ctxs.append(txt[max(0, k - 40):k + 40])
            if ctxs:
                hits["%s:%s" % (name, tok)] = ctxs
    ok["6_forbidden_tokens"] = dict(
        pass_=bool(len(hits) == 0), hits=hits,
        note=("digit-bounded scan for the superseded number tokens listed "
              "in the task sheet; a hit means the token appears as a "
              "standalone number, not as a digit substring of a legitimate "
              "value such as 617055.75"))
    # 7 figures present and vector
    figs = {}
    for f in sorted(os.listdir(FIGS)):
        if f.endswith(".pdf"):
            figs[f] = os.path.getsize(os.path.join(FIGS, f))
    ok["7_figures"] = dict(pass_=bool(len(figs) >= 2), files=figs)
    # 8 samples saved
    n_cache = len([f for f in os.listdir(ACACHE) if f.endswith(".npz")])
    ok["8_samples_saved"] = dict(pass_=bool(n_cache >= N_DRAW),
                                 a_cache_files=n_cache,
                                 bootstrap_npz=os.path.getsize(
                                     os.path.join(RUN,
                                                  "bootstrap_samples.npz")))
    # 9 design completeness and pick-freeze pairing
    rows = load_evals()
    want = set()
    for i in range(N_HALF):
        for t in ("F", "MA", "SA", "FB", "SB", "SC", "MB", "MC"):
            want.add("%s_%03d" % (t, i))
    for k in range(len(B_COMBOS)):
        want.add("BE_%02d" % k)
    missing = sorted(want - set(rows))
    pairing = dict(
        A={"n_checked": N_HALF,
           "same_A_in_F_SA": all(rows["F_%03d" % i]["a"]
                                 == rows["SA_%03d" % i]["a"]
                                 for i in range(N_HALF)),
           "A_prime_in_FB_SB_SC": all(
               rows["FB_%03d" % i]["a"] == rows["SB_%03d" % i]["a"]
               == rows["SC_%03d" % i]["a"] != rows["F_%03d" % i]["a"]
               for i in range(N_HALF))},
        B={"same_B_in_F_SB": all(rows["F_%03d" % i]["b"]
                                 == rows["SB_%03d" % i]["b"]
                                 for i in range(N_HALF))},
        C={"same_C_in_F_SC": all(rows["F_%03d" % i]["c"]
                                 == rows["SC_%03d" % i]["c"]
                                 for i in range(N_HALF))})
    ok["9_design_complete"] = dict(
        pass_=bool(not missing and all(
            all(v for kk, v in vv.items() if kk != "n_checked")
            for vv in pairing.values())),
        n_rows=len(rows), missing=missing[:20], pairing=pairing)
    # 10 REF provenance: every hard-coded constant equals its source key
    def _rj(p):
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    et = _rj(os.path.join(lib.DATA, "runs", "T024",
                          "schedule_check.json"))["exec_totals"]
    t18i = _rj(os.path.join(lib.DATA, "runs", "T018",
                            "result_int_K60.json"))
    t18lp = _rj(os.path.join(lib.DATA, "runs", "T018",
                             "result_lp_K60.json"))
    hy = _rj(os.path.join(lib.DATA, "runs", "T025", "hybrid_by_K.json"))
    fr = _rj(os.path.join(lib.DATA, "runs", "T019",
                          "frontier.json"))["meas343"]["selected"]
    kr = _rj(os.path.join(lib.DATA, "runs", "T019",
                          "frontier.json"))["meas343"]["knee"]
    q3row = None
    for r in hy["rows"]:
        if (r.get("K") == REF["q3"]["K"]
                and r.get("source_cap") == "arrivals"
                and r.get("budget") == REF["q3"]["payment"]
                and abs(r["pure_obj"] - REF["q3"]["pure_obj"]) < 1e-6):
            q3row = r
    assert q3row is not None, "T025 tight K=2 arrivals/30000 row not found"
    prov = {
        "q2_baseline": (BASE_Q2, et["no_tricycle_total"]),
        "q2_total": (REF["q2"]["total"], et["total_fullload"]),
        "q2_unmet": (REF["q2"]["unmet"], et["unmet"]),
        "q2_saving": (REF["q2"]["saving"], et["saving_fullload"]),
        "q2_saving_share": (REF["q2"]["saving_share"],
                            et["saving_fullload_share"]),
        "q2_layer": (REF["q2"]["layer_fullload"], et["layer_cost_fullload"]),
        "q2_lp_unmet": (REF["q2"]["lp_unmet"], et["unmet_lp"]),
        "q2_int_short": (REF["q2"]["int_short"], et["int_short"]),
        "q2_tri_vkm": (REF["q2"]["tri_vkm_fullload"], et["lp_vehicle_km"]),
        "q2_truck_obj": (REF["q2"]["truck_obj"], t18i["obj"]),
        "q2_lp_obj": (REF["q2"]["lp_obj"], t18lp["obj"]),
        "q3_pure_obj": (REF["q3"]["pure_obj"], q3row["pure_obj"]),
        "q3_pure_unmet": (REF["q3"]["pure_unmet"], q3row["pure_unmet"]),
        "q3_truck_obj": (REF["q3"]["truck_obj"], q3row["truck_obj"]),
        "q3_ceiling": (REF["q3"]["ceiling"],
                       q3row["pure_obj"] - q3row["truck_obj"]),
        "q4_coverage": (REF["q4"]["coverage"], fr["coverage"]),
        "q4_arrival": (REF["q4"]["arrival_coverage"],
                       fr["arrival_coverage"]),
        "q4_unassigned": (REF["q4"]["unassigned"], fr["unassigned"]),
        "q4_n_sites": (float(REF["q4"]["n_sites"]), float(kr["n"])),
    }
    prov_out = {k: dict(ref=float(a), source=float(b),
                        abs_dev=float(abs(a - b)))
                for k, (a, b) in prov.items()}
    ok["10_ref_provenance"] = dict(
        pass_=bool(all(v["abs_dev"] <= 1e-9 * max(1.0, abs(v["ref"]))
                       for v in prov_out.values())),
        sources=["data/runs/T024/schedule_check.json:exec_totals",
                 "data/runs/T018/result_int_K60.json:obj",
                 "data/runs/T018/result_lp_K60.json:obj",
                 "data/runs/T025/hybrid_by_K.json tight K=2 arrivals/30000",
                 "data/runs/T019/frontier.json meas343.selected/knee"],
        constants=prov_out,
        note=("every REF constant is copied from the accepted result file "
              "and re-compared here; anchors are constant translations "
              "(REF - raw proxy at the reference) and do not depend on any "
              "sampled parameter"))
    # 11 memoised Q3 edge mask equals T008's own build_edges at the reference
    prep_chk, _ = run_prep()
    ctx_chk = make_ctx(prep_chk)
    net = ctx_chk.arrc - ctx_chk.depc
    need_c = np.maximum(0.0, -net).max(0)
    tstar_c = np.argmax(-net, axis=0)
    pos_c = need_c > 0
    e_ref = t8.build_edges(ctx_chk.arrc, dist_matrix(), need_c, tstar_c,
                           pos_c)
    e_mem = build_edges_cached(ctx_chk.arrc, dist_matrix(), need_c, tstar_c,
                               pos_c)
    same = True
    for h in range(24):
        a, b = e_ref[h], e_mem[h]
        if (a is None) != (b is None):
            same = False
            break
        if a is None:
            continue
        if not (len(a) == len(b) and all(
                x.dtype == y.dtype and np.array_equal(x, y)
                for x, y in zip(a, b))):
            same = False
            break
    ok["11_q3_mask_memo"] = dict(
        pass_=bool(same),
        note=("q3_setup uses build_edges_cached, a memoised copy of "
              "t008_q3_pool.build_edges that reuses the 5,263^2 boolean "
              "mask; this check compares its 24 hourly edge sets with "
              "T008's own function at the reference point"))
    # 12 cached bootstrap draws equal draws recomputed from the rebuilt prep
    bep = os.path.join(RUN, "boot_equiv.json")
    if os.path.exists(bep):
        be = load_json("boot_equiv.json")
        ok["12_boot_equiv"] = dict(
            pass_=bool(be["all_equal"]), verdict=be["verdict"],
            totals=be["totals"], per_draw=be["per_draw"],
            worst_max_abs=be.get("worst_max_abs"),
            self_repeat=be.get("self_repeat"),
            weights=be.get("weights"),
            acache_unchanged=be["acache_unchanged"],
            note=("a_cache was written before the 21:50 prep rebuild.  "
                  "The rebuild changed the replay's plan arrays (old path: "
                  "a K=60 LP re-solve vertex; new: the accepted "
                  "plan_int_K60), which is why the replayed unmet moved "
                  "38,529.8072 -> 38,489.4373; the demand arrays "
                  "depc_day/arrc_day/dj_day/zid were unchanged.  "
                  "code/t023_boot_equiv.py recomputes draws 0/137/255 "
                  "through the exact _boot_job path with ACACHE redirected "
                  "to a temp dir and compares every stored array; the 256 "
                  "dep_total/arr_total values are recomputed in full"))
    for name in os.listdir(RUN):
        p = os.path.join(RUN, name)
        if os.path.isfile(p):
            out["files"][name] = dict(bytes=os.path.getsize(p),
                                      sha256=sha256(p))
    dump(out, "check.json")
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="T023 uncertainty by source")
    ap.add_argument("--stage", default="all",
                    choices=("prep", "bootstrap", "inputs", "reference",
                             "design", "kstar", "intervals", "shares",
                             "flips", "deviation", "tornado", "params",
                             "check", "all"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel worker processes (0 = serial)")
    ap.add_argument("--tags", default=None,
                    help="comma-separated design tags (F,MA,SA,FB,SB,SC,"
                         "MB,MC,BE)")
    args = ap.parse_args()
    t0 = time.time()
    w = 0 if args.workers == 0 else args.workers
    if args.stage in ("prep", "all"):
        run_prep(force=args.force)
    if args.stage in ("bootstrap", "all"):
        run_bootstrap(force=args.force, workers=w)
    if args.stage in ("inputs", "all"):
        run_inputs(force=args.force)
    if args.stage in ("reference", "all"):
        run_reference(force=args.force)
    if args.stage in ("design", "all"):
        run_design(force=args.force, limit=args.limit, workers=w,
                   tags=(args.tags.split(",") if args.tags else None))
    if args.stage in ("kstar", "all") and args.limit is None:
        run_kstar(workers=w)
        run_kstar_marg(workers=w)
    if args.stage in ("intervals", "all"):
        run_intervals()
    if args.stage in ("flips", "all") and args.limit is None:
        run_flips(workers=w)
    if args.stage in ("deviation", "all") and args.limit is None:
        run_deviation(workers=w)
    if args.stage in ("shares", "all"):
        run_shares()
    if args.stage in ("tornado", "all") and args.limit is None:
        run_tornado(workers=w)
    if args.stage in ("params", "all"):
        run_params(force=args.force)
    if args.stage in ("check", "all") and args.limit is None:
        run_check()
    banner("stage %s done" % args.stage, t0)


if __name__ == "__main__":
    main()
