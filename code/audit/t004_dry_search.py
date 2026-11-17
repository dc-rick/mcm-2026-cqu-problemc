"""T004 dry-cell definition search: which 5-minute cell-level path yields 172?

Candidate definitions, all rebuilt from existing typical-day data and I0c:
  * scaled (1/0.128) vs unscaled departures/arrivals
  * 5-minute vs hourly vs 15-minute sampling
  * strictly negative vs empty (<1)
No existing file is modified; this is a diagnostic to locate the paper number.
"""
import io
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code"))
import lib  # noqa: E402

RUN = os.path.join(ROOT, "data", "runs", "T004")
z = np.load(os.path.join(ROOT, "data", "cell_sim_inputs.npz"))
dep5, I0c = z["dep5"], z["I0c"]
C = len(I0c)
SCALE = 1.0 / 0.128

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
arr5 = arr5 / nd

out = {}
for tag, scale in (("scaled", SCALE), ("unscaled", 1.0)):
    net = (arr5 - dep5 / SCALE) * scale
    I = I0c[:, None] + np.cumsum(net, axis=0).T
    for samp, idx in (("5min", np.arange(288)),
                      ("15min", np.arange(2, 288, 3)),
                      ("hourly", np.arange(11, 288, 12))):
        mn = I[:, idx].min(1)
        out[f"{tag}_{samp}"] = dict(
            cells_negative=int((mn < 0).sum()),
            cells_below_one=int((mn < 1).sum()),
            min_inventory=float(mn.min()))
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, io.open(os.path.join(RUN, "dry_definition_search.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
