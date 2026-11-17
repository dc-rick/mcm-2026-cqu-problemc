"""T004: robustness of the 172 dry-cell reconstruction."""
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
dep_obs = dep5 / SCALE

out = {"phase_offsets_hourly_unscaled": {}, "arrival_lag": {},
       "first_start_I0": {}}
net = arr5 - dep_obs
cum = np.cumsum(net, axis=0)
I = I0c[:, None] + cum.T
for off in range(12):
    idx = np.arange(off, 288, 12)
    out["phase_offsets_hourly_unscaled"][str(off)] = int(
        (I[:, idx].min(1) < 0).sum())
print("phase offsets", out["phase_offsets_hourly_unscaled"])

# one-step arrival lag
cum_lag = np.cumsum(np.vstack([np.zeros((1, C)), arr5[:-1]]) - dep_obs, axis=0)
I_lag = I0c[:, None] + cum_lag.T
for off in (0, 11):
    idx = np.arange(off, 288, 12)
    out["arrival_lag"][str(off)] = int((I_lag[:, idx].min(1) < 0).sum())

# first-start initial inventory
I0f = np.load(os.path.join(ROOT, "data", "I0_cell_first.npy"))
I_f = I0f[:, None] + cum.T
for off in (0, 11):
    idx = np.arange(off, 288, 12)
    out["first_start_I0"][str(off)] = int((I_f[:, idx].min(1) < 0).sum())
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, io.open(os.path.join(RUN, "dry_172_robustness.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
