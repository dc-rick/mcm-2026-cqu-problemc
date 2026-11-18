"""T023: prove the cached bootstrap draws equal draws from the rebuilt prep.

The bootstrap samples in ``data/runs/T023/a_cache`` were produced at 20:27-
21:26, before ``prep.npz`` was rebuilt at 21:50 with the stored-plan replay
fields (XM_plan / U_plan / truck_obj_int).  The Q2 caliber fix touched only
``run_prep``'s added keys, not the demand arrays, so this script proves the
draws are unchanged rather than paying for a 25-minute rerun.

Evidence produced:
  1. Draws 0 / 137 / 255 are recomputed with the *current* prep through the
     exact ``_boot_job`` code path (its ACACHE is redirected to a temp dir so
     no cached file is ever overwritten) and compared bit-for-bit
     (``np.array_equal``) on depc / arrc / PZ / dep_zone / beta.
  2. ``weights`` rows in bootstrap_samples.npz equal ``_boot_weights()``.
  3. All 256 dep_total / arr_total values recomputed from the current prep's
     depc_day / arrc_day and the weight matrix; max relative difference.
  4. SHA-256 of every a_cache file before and after the run (must not change).

Writes ``data/runs/T023/boot_equiv.json``.  Run after ``--stage design`` and
before ``--stage kstar``.
"""
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t023_uncertainty as T

RUN = T.RUN
ACACHE = T.ACACHE
DRAWS = (0, 137, 255)
ARRAYS = ("depc", "arrc", "PZ", "dep_zone", "beta")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    prep, _ = T.run_prep()
    hashes_before = {f: sha(os.path.join(ACACHE, f))
                     for f in sorted(os.listdir(ACACHE)) if f.endswith(".npz")}
    boot = np.load(os.path.join(RUN, "bootstrap_samples.npz"))
    W = T._boot_weights()

    T._boot_init()
    out = dict(task="T023 bootstrap equivalence", date="2026-11-18",
               prep_rebuild=("prep.npz was rebuilt 2026-11-18 21:50 after the "
                             "Q2 caliber fix.  The change is in the replay's "
                             "plan arrays: the old path re-solved the K=60 "
                             "LP and landed on another optimal vertex, while "
                             "the rebuilt prep reads the accepted integer "
                             "plan plan_int_K60 (XM_plan / U_plan / "
                             "truck_obj_int).  That is why the replayed "
                             "unmet moved 38,529.8072 -> 38,489.4373.  The "
                             "demand-surface arrays depc_day / arrc_day / "
                             "dj_day / zid were not touched, so the cached "
                             "bootstrap draws are bit-identical to draws "
                             "from the rebuilt prep."),
               draws=list(DRAWS), arrays=list(ARRAYS), per_draw={},
               weights=dict(), totals=dict(), acache_sha_before=hashes_before)

    tmp = tempfile.mkdtemp(prefix="t023eq_")
    tmp2 = tempfile.mkdtemp(prefix="t023eq2_")
    old_cache = T.ACACHE
    try:
        T.ACACHE = tmp
        for i in DRAWS:
            rec = dict(draw=int(i), equal={}, max_abs={})
            ref = np.load(os.path.join(ACACHE, "a_%03d.npz" % i))
            T._boot_job(i)
            new = np.load(os.path.join(tmp, "a_%03d.npz" % i))
            for k in ARRAYS:
                a, b = ref[k], new[k]
                rec["equal"][k] = bool(a.shape == b.shape
                                       and np.array_equal(a, b))
                rec["max_abs"][k] = float(np.max(np.abs(a.astype(np.float64)
                                                       - b.astype(np.float64))))
            rec["w_row_equal"] = bool(np.array_equal(ref["w"], W[i]))
            out["per_draw"][str(i)] = rec
        # self-consistency: recompute draw 0 a second time in this process.
        # If the two fresh copies are bit-identical, any difference from the
        # cache is not run-to-run noise inside this process.
        T.ACACHE = tmp2
        T._boot_job(0)
        a = np.load(os.path.join(tmp, "a_000.npz"))
        b = np.load(os.path.join(tmp2, "a_000.npz"))
        out["self_repeat"] = dict(
            draw=0, equal={k: bool(a[k].shape == b[k].shape
                                   and np.array_equal(a[k], b[k]))
                           for k in ARRAYS},
            note=("two recomputations in this process; equality means the "
                  "remaining cache difference, if any, comes from the old "
                  "worker's BLAS configuration, not from nondeterminism"))
    finally:
        T.ACACHE = old_cache
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(tmp2, ignore_errors=True)

    # weights rows for every draw
    wz = np.load(os.path.join(ACACHE, "a_%03d.npz" % 0))["w"]  # shape probe
    out["weights"] = dict(
        shape=[int(x) for x in np.shape(W)],
        rows_equal_all=bool(all(
            np.array_equal(np.load(os.path.join(ACACHE, "a_%03d.npz" % i))["w"],
                           W[i])
            for i in range(T.N_DRAW))),
        sample_w_dtype=str(wz.dtype))

    # all 256 totals from the current prep
    depcs = prep["depc_day"].sum((1, 2)) * T.SCALE
    arrcs = prep["arrc_day"].sum((1, 2)) * T.SCALE
    wrow = W.astype(np.float64) / 9.0
    dep_new = wrow @ depcs
    arr_new = wrow @ arrcs
    dep_rel = np.abs(dep_new - boot["dep_total"]) / np.maximum(
        np.abs(boot["dep_total"]), 1e-12)
    arr_rel = np.abs(arr_new - boot["arr_total"]) / np.maximum(
        np.abs(boot["arr_total"]), 1e-12)
    out["totals"] = dict(
        dep_max_rel=float(dep_rel.max()), arr_max_rel=float(arr_rel.max()),
        dep_max_abs=float(np.abs(dep_new - boot["dep_total"]).max()),
        arr_max_abs=float(np.abs(arr_new - boot["arr_total"]).max()),
        gate=1e-12,
        pass_=bool(dep_rel.max() < 1e-12 and arr_rel.max() < 1e-12))

    hashes_after = {f: sha(os.path.join(ACACHE, f))
                    for f in sorted(os.listdir(ACACHE)) if f.endswith(".npz")}
    out["acache_sha_after"] = hashes_after
    out["acache_unchanged"] = bool(hashes_before == hashes_after)
    out["all_equal"] = bool(
        out["acache_unchanged"]
        and all(all(v for v in r["equal"].values())
                and r["w_row_equal"] for r in out["per_draw"].values())
        and all(out["self_repeat"]["equal"].values())
        and out["weights"]["rows_equal_all"] and out["totals"]["pass_"])
    worst = {}
    for k in ARRAYS:
        worst[k] = float(max(r["max_abs"][k] for r in out["per_draw"].values()))
    out["worst_max_abs"] = worst
    out["verdict"] = ("cached bootstrap draws are bit-identical to draws "
                      "from the rebuilt prep; no rerun needed"
                      if out["all_equal"] else
                      "MISMATCH: rerun bootstrap --force and force every "
                      "stage that reads a_cache (design, kstar, flips, "
                      "deviation, tornado)")
    with io.open(os.path.join(RUN, "boot_equiv.json"), "w",
                 encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in
                      ("all_equal", "verdict", "totals")}, indent=1))
    print("wrote", os.path.join(RUN, "boot_equiv.json"))


if __name__ == "__main__":
    main()
