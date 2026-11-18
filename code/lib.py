"""Shared library: data loading, cell/zone system, distances, typical day, plotting style."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project root = parent of this file's directory, so the scripts run from any cwd.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "figs")

# --- study area / zoning -------------------------------------------------
STEP = 0.005              # degrees (~555 m x 426 m)
THR = 50                  # min trips for a cell to enter the study area
LAT_KM = 111.2            # km per degree latitude
LON_KM = 85.2             # km per degree longitude at 39.9 N
TYP_DAYS = ["2017-05-10", "2017-05-11", "2017-05-12", "2017-05-15",
            "2017-05-16", "2017-05-18", "2017-05-19"]   # well-covered weekdays
ALL_DAYS = ["2017-05-%02d" % d for d in range(10, 20)]


def load_trips(days=None):
    df = pd.read_parquet(os.path.join(DATA, "trips.parquet"))
    df = df[(df.start_lat > 39.4) & (df.start_lon > 115.8) &
            (df.start_lat < 40.4) & (df.start_lon < 117.0)].copy()
    df["date"] = df.starttime.dt.normalize().astype(str).str[:10]
    if days is not None:
        df = df[df.date.isin(days)].copy()
    df["hour"] = df.starttime.dt.hour
    df["qtr"] = df.starttime.dt.hour * 4 + df.starttime.dt.minute // 15
    return df


def add_cells(df):
    df["ix_s"] = np.floor(df.start_lon / STEP).astype(int)
    df["iy_s"] = np.floor(df.start_lat / STEP).astype(int)
    df["ix_e"] = np.floor(df.end_lon / STEP).astype(int)
    df["iy_e"] = np.floor(df.end_lat / STEP).astype(int)
    return df


def make_cells(df, thr=THR):
    """Cell table for the study area; returns (cells, trips with cell ids)."""
    df = add_cells(df)
    key_s = df.iy_s * 100000 + df.ix_s
    key_e = df.iy_e * 100000 + df.ix_e
    ks = key_s.value_counts()
    ke = key_e.value_counts()
    keep = sorted(set(ks[ks >= thr].index) | set(ke[ke >= thr].index))
    cmap = {k: i for i, k in enumerate(keep)}
    cells = pd.DataFrame({"key": keep})
    cells["ix"] = cells.key % 100000
    cells["iy"] = cells.key // 100000
    cells["lon"] = (cells.ix + 0.5) * STEP
    cells["lat"] = (cells.iy + 0.5) * STEP
    cells["x"] = (cells.lon - 116.0) * LON_KM
    cells["y"] = (cells.lat - 39.0) * LAT_KM
    df["o"] = key_s.map(cmap)
    df["d"] = key_e.map(cmap)
    return cells, df


def cell_xy(cells):
    return cells[["x", "y"]].to_numpy()


def dist_matrix(xy, chunk=800):
    """Euclidean km distance matrix (N x N)."""
    n = len(xy)
    D = np.empty((n, n), dtype=np.float32)
    for i in range(0, n, chunk):
        d = xy[i:i + chunk, None, :] - xy[None, :, :]
        D[i:i + chunk] = np.sqrt((d ** 2).sum(-1))
    return D


# --- typical day ---------------------------------------------------------
def typical_day_od(df, cells, days=TYP_DAYS):
    """Mean weekday OD counts and 15-min temporal profile."""
    sub = df[df.date.isin(days) & df.o.notna() & df.d.notna()]
    n = len(cells)
    nday = sub.date.nunique()
    od = np.zeros((n, n), dtype=np.float32)
    np.add.at(od, (sub.o.values.astype(int), sub.d.values.astype(int)), 1.0)
    od /= (nday * 1.0)
    prof = np.zeros(96)
    np.add.at(prof, sub.qtr.values.astype(int), 1.0)
    prof /= nday
    return od, prof, len(sub) / nday


# --- style ---------------------------------------------------------------
def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.6,
        "ytick.labelsize": 8.6,
        "legend.fontsize": 8.6,
        "axes.linewidth": 0.7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "figure.dpi": 150,
        "legend.frameon": False,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
    })


COL = {"blue": "#2C6FBB", "orange": "#E07B39", "green": "#3C8D5A",
       "red": "#C0392B", "purple": "#7B5EA7", "grey": "#6B7280",
       "teal": "#2A9D8F", "dark": "#1F2937"}


def save(fig, name):
    fig.savefig(os.path.join(FIG, f"{name}.pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("fig ->", name)


def get_cells():
    """The canonical cell table (built once from all 14 days)."""
    return pd.read_parquet(os.path.join(DATA, "cells.parquet"))


def assign_cells(df, cells):
    """Map trips onto an existing cell table (NaN outside the study area)."""
    df = add_cells(df)
    key = cells.key.to_numpy()
    idx = {int(k): i for i, k in enumerate(key)}
    ks = (df.iy_s * 100000 + df.ix_s).map(idx)
    ke = (df.iy_e * 100000 + df.ix_e).map(idx)
    df["o"] = ks.astype("Float64")
    df["d"] = ke.astype("Float64")
    return df
