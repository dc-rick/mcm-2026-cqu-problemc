"""Rebuild the 5263 x 5263 cell distance matrix from data/cells.parquet.

The tracked data/dist_matrix.npy is reproduced exactly as it was generated in
10_q1_model.py: Euclidean distances between cell centres in the projected km
frame from lib.cell_xy, with the diagonal set to the mean intra-cell offset
0.35 / sqrt(2) km.  The file is written to data/runs/T015/ and the tracked
matrix is only read, never overwritten.

Usage:
    python code/build_dist_matrix.py
"""
import argparse
import hashlib
import io
import json
import os

import numpy as np

import lib
from lib import DATA

OUT_DIR = os.path.join(DATA, "runs", "T015")
DEFAULT_OUT = os.path.join(OUT_DIR, "dist_matrix.npy")
DEFAULT_REF = os.path.join(DATA, "dist_matrix.npy")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build():
    cells = lib.get_cells()
    xy = lib.cell_xy(cells)
    D = lib.dist_matrix(xy)
    np.fill_diagonal(D, 0.35 / np.sqrt(2))   # same intra-cell offset as 10_q1_model.py
    return D.astype(np.float32), len(cells)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="output .npy (default: %(default)s)")
    parser.add_argument("--reference", default=DEFAULT_REF,
                        help="tracked matrix to compare against (default: %(default)s)")
    args = parser.parse_args()

    D, n = build()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, D)
    print("cells", n, "matrix", D.shape, D.dtype)
    print("wrote", args.out)

    check = {
        "cells": n,
        "output": args.out,
        "reference": args.reference,
        "shape": list(D.shape),
        "dtype": str(D.dtype),
        "sha256_output_file": sha256_file(args.out),
        "sha256_output_array": hashlib.sha256(D.tobytes()).hexdigest(),
        "reference_exists": os.path.exists(args.reference),
    }
    if check["reference_exists"]:
        ref = np.load(args.reference)
        check["reference_shape"] = list(ref.shape)
        check["reference_dtype"] = str(ref.dtype)
        check["same_shape"] = D.shape == ref.shape
        check["same_dtype"] = D.dtype == ref.dtype
        if D.shape == ref.shape:
            check["max_abs_diff"] = float(np.abs(D - ref).max())
            check["bitwise_equal"] = bool(np.array_equal(D, ref))
        else:
            check["max_abs_diff"] = None
            check["bitwise_equal"] = False
        check["sha256_reference_file"] = sha256_file(args.reference)
        check["sha256_reference_array"] = hashlib.sha256(ref.tobytes()).hexdigest()
        print("reference", ref.shape, ref.dtype)
        print("max abs diff", check["max_abs_diff"],
              "bitwise equal", check["bitwise_equal"])
        print("sha256 output file", check["sha256_output_file"])
        print("sha256 reference", check["sha256_reference_file"])

    check_path = os.path.join(OUT_DIR, "dist_matrix_check.json")
    with io.open(check_path, "w", encoding="utf-8") as handle:
        json.dump(check, handle, ensure_ascii=False, indent=1)
    print("check ->", check_path)


if __name__ == "__main__":
    main()
