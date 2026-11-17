"""Step 0: load the raw Mobike CSV, decode geohash -> lat/lon, cache as parquet.

Raw file: 2026_CQU_MCM_Problem_C_DATA.csv (3.21 M trips, 2017-05-10..2017-05-24, Beijing)

Usage:
    python code/00_prepare.py --raw data/raw/*.csv --out data/trips.parquet
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

# Make ``python -I code/00_prepare.py`` work: -I drops the script directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import DATA

DEFAULT_RAW = os.path.join(DATA, "raw", "*.csv")
DEFAULT_OUT = os.path.join(DATA, "trips.parquet")

_B32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_MAP = {c: i for i, c in enumerate(_B32)}


def decode_geohash_np(gh_in: np.ndarray):
    """Vectorised geohash decode. gh_in: array of equal-length strings.
    Returns (lat_center, lon_center, lat_err, lon_err)."""
    gh = np.asarray(gh_in).astype("U")
    gh = gh.view("U1").reshape(len(gh), -1)   # (n, L) array of single chars
    n, L = gh.shape
    bits = np.empty((n, L * 5), dtype=np.int8)
    for j in range(L):
        vals = np.array([_MAP[c] for c in gh[:, j]], dtype=np.int8)
        for b in range(5):
            bits[:, j * 5 + b] = (vals >> (4 - b)) & 1
    lat = np.zeros(n); lat_err = np.full(n, 90.0)
    lon = np.zeros(n); lon_err = np.full(n, 180.0)
    for b in range(L * 5):
        even = (b % 2 == 0)
        if even:
            lon_err /= 2.0
            lon += np.where(bits[:, b] == 1, lon_err, -lon_err)
        else:
            lat_err /= 2.0
            lat += np.where(bits[:, b] == 1, lat_err, -lat_err)
    return lat, lon, lat_err, lon_err


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decode the raw Mobike order CSV into a parquet table.")
    parser.add_argument(
        "--raw", default=DEFAULT_RAW,
        help="raw CSV file or glob (default: %(default)s)")
    parser.add_argument(
        "--out", default=DEFAULT_OUT,
        help="output parquet (default: %(default)s)")
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite an existing output file")
    return parser.parse_args()


def resolve_raw(pattern):
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise SystemExit("no CSV matched --raw %r" % pattern)
    if len(matches) > 1:
        raise SystemExit("--raw matched several files; pass one explicitly:\n  "
                         + "\n  ".join(matches))
    return matches[0]


def main():
    args = parse_args()
    out = os.path.abspath(args.out)
    if os.path.exists(out) and not args.force:
        raise SystemExit("%s exists; use --out for a new file or --force to "
                         "overwrite" % out)
    raw = resolve_raw(args.raw)
    print("raw:", raw)
    print("out:", out)
    print("reading csv ...")
    df = pd.read_csv(
        raw,
        dtype={"orderid": "int64", "userid": "int64", "bikeid": "int64",
               "biketype": "int8", "geohashed_start_loc": "str",
               "geohashed_end_loc": "str"},
        parse_dates=["starttime"],
    )
    print(df.shape)
    print(df.head())
    print("time range:", df.starttime.min(), df.starttime.max())
    print("geohash length:", df.geohashed_start_loc.str.len().value_counts())
    print("biketype:", df.biketype.value_counts().to_dict())

    for tag in ("start", "end"):
        la, lo, lae, loe = decode_geohash_np(df[f"geohashed_{tag}_loc"].values)
        df[f"{tag}_lat"] = la.astype("float32")
        df[f"{tag}_lon"] = lo.astype("float32")
        df[f"{tag}_lat_err"] = lae.astype("float32")
        df[f"{tag}_lon_err"] = loe.astype("float32")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_parquet(out, index=False)
    print("wrote", out, df.shape)
    print(df[["start_lat", "start_lon", "end_lat", "end_lon"]].describe())


if __name__ == "__main__":
    main()
