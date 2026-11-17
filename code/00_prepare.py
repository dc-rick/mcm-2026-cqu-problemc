"""Step 0: load the raw Mobike CSV, decode geohash -> lat/lon, cache as parquet.

Raw file: 2026_CQU_MCM_Problem_C_DATA.csv (3.21 M trips, 2017-05-10..2017-05-24, Beijing)
"""
import numpy as np
import pandas as pd
import os

RAW = r"C:\Users\Administrator\Desktop\2026年重庆大学数学建模秋季赛题目\2026_CQU_MCM_Problem_C_DATA.csv"
OUT = r"D:\code\mcm\2026CQU_C\data\trips.parquet"

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


def main():
    print("reading csv ...")
    df = pd.read_csv(
        RAW,
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

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_parquet(OUT, index=False)
    print("wrote", OUT, df.shape)
    print(df[["start_lat", "start_lon", "end_lat", "end_lon"]].describe())


if __name__ == "__main__":
    main()
