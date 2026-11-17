"""T004 step 5: the paper's "172 of 5,263 cells run dry".

Finding: 172 is reproduced only by an HOURLY, UNSCALED cell inventory path
    I_c(t) = I0c + cumsum_hourly(arrivals - departures)
with observed counts (no 1/p factor).  The 5-minute scaled path stated in the
paper gives 2,306 dry cells.  We report both, plus the stochastic simulator's
dry-cell count as a third, separate model.  No LP is re-solved and no existing
file is overwritten.
"""
import io
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code"))
import lib  # noqa: E402
import sim_cell  # noqa: E402

RUN = os.path.join(ROOT, "data", "runs", "T004")
SCALE = 1.0 / 0.128

z = np.load(os.path.join(ROOT, "data", "cell_sim_inputs.npz"))
dep5, Pest, I0c, xy, zid = z["dep5"], z["Pest"], z["I0c"], z["xy"], z["zid"]
C = len(I0c)

df = lib.load_trips(lib.TYP_DAYS)
cells = lib.get_cells()
df = lib.assign_cells(df, cells)
ok = df.o.notna() & df.d.notna()
s = df[ok].copy()
s["o"] = s.o.astype(int); s["d"] = s.d.astype(int)
step5 = (s.starttime.dt.hour.values * 60 + s.starttime.dt.minute.values) // 5
nd = len(lib.TYP_DAYS)
arr5 = np.zeros((288, C))
np.add.at(arr5, (step5, s.d.values), 1.0)
arr5 = arr5 / nd                    # observed arrivals per typical day
dep_obs = dep5 / SCALE              # observed departures per typical day


def fluid(kind, scale, samp):
    net = (arr5 - dep_obs) * scale
    I = I0c[:, None] + np.cumsum(net, axis=0).T
    mn = I[:, samp].min(1)
    return I, mn


out = {"n_cells": C, "I0_sum": float(I0c.sum()),
       "definitions": {},
       "finding": ("172 is the hourly unscaled path; the paper's stated "
                   "5-minute 1/p-scaled path gives 2,306")}
hour_end = np.arange(11, 288, 12)
five = np.arange(288)
for tag, scale, samp in (("hourly_unscaled", 1.0, hour_end),
                         ("hourly_scaled", SCALE, hour_end),
                         ("5min_unscaled", 1.0, five),
                         ("5min_scaled", SCALE, five)):
    I, mn = fluid(tag, scale, samp)
    out["definitions"][tag] = dict(
        cells_negative=int((mn < 0).sum()),
        cells_below_one=int((mn < 1).sum()),
        min_inventory=float(mn.min()))
    if tag == "hourly_unscaled":
        dry = mn < 0
        out["definitions"][tag].update(
            I0_of_dry=dict(
                zero=int((I0c[dry] == 0).sum()),
                below_one=int((I0c[dry] < 1).sum()),
                median=float(np.median(I0c[dry])),
                max=float(I0c[dry].max())),
            dry_cells_typical_departures=float(dep_obs.sum(0)[dry].sum()),
            dry_share_of_typical_departures=float(
                dep_obs.sum(0)[dry].sum() / dep_obs.sum()),
            note=("all dry cells are low-stock cells; the path is hourly and "
                  "unscaled, so only small I0 cells cross zero"))
print(json.dumps(out["definitions"], ensure_ascii=False, indent=1))

# stochastic simulator dry counts: same model as the 13.4%, different radius
stoch = {}
for R in (0.30, 0.45, 0.60):
    vals = []
    for sd in range(5):
        S = sim_cell.CellSim(dep5, Pest, I0c, xy, zid,
                             rng=np.random.default_rng(sd), walk_km=R)
        r = S.run(record=True)
        vals.append(dict(cells_below_one=int((r["traj"] < 1).any(0).sum()),
                         unmet_share=float(r["unmet_share"])))
    stoch[f"R{int(R*1000)}"] = dict(
        seeds=list(range(5)),
        cells_below_one=[v["cells_below_one"] for v in vals],
        unmet_share=[v["unmet_share"] for v in vals],
        mean_cells_below_one=float(np.mean([v["cells_below_one"] for v in vals])),
        mean_unmet=float(np.mean([v["unmet_share"] for v in vals])))
    print(f"stochastic R={R:.2f}", stoch[f"R{int(R*1000)}"])
out["stochastic_343638"] = dict(
    model=("CellSim at FLEET=343,638, 5-min steps; 'dry' = end-of-step "
           "inventory < 1 at least once"),
    note="separate model; never substituted for the fluid count",
    by_radius=stoch)

json.dump(out, io.open(os.path.join(RUN, "dry_cells.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
