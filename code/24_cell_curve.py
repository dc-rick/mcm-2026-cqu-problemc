"""Build cell-level inputs (5-min departures, destination kernels) and run the
fleet-sizing curve at neighbourhood resolution."""
import os
import numpy as np
import lib
from lib import DATA
import sim_cell

SCALE = 1.0 / 0.128
FLEET = 343638.0
cells = lib.get_cells()
C = len(cells)
zid = cells.zone.to_numpy()
Z = zid.max() + 1

df = lib.load_trips(lib.TYP_DAYS)
df = lib.assign_cells(df, cells)
ok = df.o.notna() & df.d.notna()
s = df[ok].copy(); s["o"] = s.o.astype(int); s["d"] = s.d.astype(int)
nd = len(lib.TYP_DAYS)
step5 = (s.starttime.dt.hour.values * 60 + s.starttime.dt.minute.values) // 5

dep5 = np.zeros((288, C))
np.add.at(dep5, (step5, s.o.values), 1.0)
dep5 = dep5 / nd * SCALE

# destination kernel: P(cell | origin zone) using the calibrated counts
cnt = np.zeros((Z, C))
np.add.at(cnt, (zid[s.o.values], s.d.values), 1.0)
Pest = cnt / np.maximum(cnt.sum(1, keepdims=True), 1e-9)

st = np.load(os.path.join(DATA, "q2_state.npz"), allow_pickle=True)
I0c = np.bincount(s.d.values, minlength=C)  # placeholder, replaced below
dfl = lib.load_trips(lib.ALL_DAYS); dfl = lib.assign_cells(dfl, cells)
ok2 = dfl.o.notna() & dfl.d.notna()
lastd = dfl[ok2].sort_values("starttime").groupby("bikeid").tail(1).d.astype(int).values
I0c = np.bincount(lastd, minlength=C) / len(lastd) * FLEET

xy = cells[["x", "y"]].to_numpy()
np.savez(os.path.join(DATA, "cell_sim_inputs.npz"), dep5=dep5, Pest=Pest, I0c=I0c, zid=zid,
         xy=xy, zone_cent=cells.groupby("zone")[["x", "y"]].mean().values)

if __name__ == "__main__":
    rows = []
    for a in (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2):
        r = []
        for sd in range(3):
            S = sim_cell.CellSim(dep5, Pest, I0c * a, xy, zid,
                                 rng=np.random.default_rng(sd))
            r.append(S.run()["unmet_share"])
        rows.append((a, float(np.mean(r)), float(np.std(r))))
        print(f"alpha {a:4.2f} fleet {FLEET*a:8.0f}  unmet {100*np.mean(r):6.2f}%"
              f" +/- {100*np.std(r):.2f}")
    np.save(os.path.join(DATA, "cell_fleet_curve.npy"), np.array(rows))
