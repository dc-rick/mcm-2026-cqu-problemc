"""T001 impact analysis: how every FLEET-dependent quantity moves.

Replicates the diagnostic of 20_q2_diag.py in memory (no file is written), and
re-reads q2_results.json to recompute the right-sizing panel of fig 4(d) for
candidate fleet values.  Writes only data/runs/T001/.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import lib  # noqa: E402

OUT = os.path.join(lib.ROOT, "data", "runs", "T001")
os.makedirs(OUT, exist_ok=True)
F_OLD = 343638.0
CAND = {"lower_bound_9d": 462893.0, "chapman_first5_last4": 477565.0,
        "beta_binomial": 483787.0, "chao1_first5_last4": 492306.0}

# ---------------------------------------------------------------- diagnostics
z = np.load(os.path.join(lib.DATA, "typical_day.npz"), allow_pickle=True)
d_hr, a_hr, zid, cent = z["d_hour"], z["a_hour"], z["zid"], z["cent"]
NZ = d_hr.shape[0]

df = lib.load_trips(lib.ALL_DAYS)
cells, df = lib.make_cells(df, lib.THR)
ok = df.o.notna() & df.d.notna()
last = df[ok].sort_values("starttime").groupby("bikeid").tail(1)
init_share = np.bincount(zid[last.d.astype(int)], minlength=NZ) / len(last)
assert abs(init_share.sum() - 1) < 1e-9, init_share.sum()

net = a_hr - d_hr
cum = np.cumsum(net, axis=1)
req = cum.max(1) - cum.min(1)


def diag(F):
    I0 = init_share * F
    IQ = I0[:, None] + cum
    deficit = np.maximum(0, -IQ)
    return dict(
        F=float(F), sum_I0=float(I0.sum()),
        zones_ever_negative=int((IQ.min(1) < 0).sum()),
        stockout_bike_hours=float(deficit.sum()),
        stockout_share_of_mass=float(deficit.sum() / np.maximum(IQ, 1e-9).sum()),
        worst_zone_deficit=float(deficit.max()),
        buffer_over_I0_median=float(np.median(req / I0)),
        buffer_over_I0_min=float((req / I0).min()),
        min_inventory_ratio=float((IQ.min(1) / I0).min()),
    )


diags = [diag(F) for F in [F_OLD] + list(CAND.values())]

# ---------------------------------------------------------------- fig 4(d)
# The LP's initial inventory is share * (alpha*FLEET), i.e. it depends on the
# PHYSICAL fleet P = alpha*FLEET, not on alpha and FLEET separately.  Every
# solved point therefore belongs to a curve total(P) that does not move when
# FLEET is re-estimated; only the alpha label of each point changes.
q2 = pd.DataFrame(json.load(open(os.path.join(lib.DATA, "q2_results.json"))))
tr = q2[q2["mode"] == "trucks"].copy()
curve = []
for al in sorted(tr.alpha.unique()):
    s = tr[tr.alpha == al]
    tot = 1.0 * F_OLD * al + s.obj
    k = int(np.argmin(tot.values))
    curve.append(dict(alpha_old=float(al), P=float(al * F_OLD), K=int(s.K.iloc[k]),
                      obj=float(s.obj.iloc[k]), capital=float(F_OLD * al),
                      total=float(tot.iloc[k])))
best = min(curve, key=lambda r: r["total"])
panel = dict(curve=curve, best=best)
alpha_star = {name: best["P"] / F for name, F in [("old", F_OLD)] + list(CAND.items())}
for name, F in [("old", F_OLD)] + list(CAND.items()):
    panel.setdefault("relabelled", {})[name] = dict(
        F=F, alpha_star=best["P"] / F, P_star=best["P"], total_at_star=best["total"])

# ---------------------------------------------------------------- text numbers
TRIPS_SCALED = 274357 / 0.128          # 2.1434e6, same as 95_numbers.py
rides = {name: TRIPS_SCALED / F for name, F in [("old", F_OLD)] + list(CAND.items())}
slots = 344463.0                        # q4_summary total capacity, paper Sec 7.3

# ---------------------------------------------------------------- report
scale = {name: dict(F=F, ratio_vs_old=F / F_OLD,
                    I0_multiplier=F / F_OLD,
                    rides_per_bike=TRIPS_SCALED / F,
                    slots_over_fleet=slots / F)
         for name, F in [("old", F_OLD)] + list(CAND.items())}

out = dict(old=F_OLD, candidates=CAND, diagnostics=diags, panel_d=panel,
           scaled_trips_per_day=float(TRIPS_SCALED), rides_per_bike=rides,
           scale=scale)
with open(os.path.join(OUT, "impact_analysis.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=float)

pd.DataFrame(diags).to_csv(os.path.join(OUT, "diagnostics_by_fleet.csv"),
                           index=False, encoding="utf-8-sig")
pd.DataFrame([dict(candidate=k, **{kk: vv for kk, vv in v.items()})
              for k, v in scale.items()]).to_csv(
    os.path.join(OUT, "scaling_table.csv"), index=False, encoding="utf-8-sig")

print("diagnostics")
for dgp in diags:
    print(f"  F={dgp['F']:9,.0f} zones_neg={dgp['zones_ever_negative']:3d} "
          f"stockout={dgp['stockout_bike_hours']:12,.0f} "
          f"worst={dgp['worst_zone_deficit']:9,.0f} "
          f"min_inv/I0={dgp['min_inventory_ratio']:.3f}")
print("\nfig4(d) optimum in physical-fleet terms")
b = panel["best"]
print(f"  P*={b['P']:,.0f} bikes (old alpha {b['alpha_old']:.1f}), K={b['K']}, "
      f"total={b['total']/1e3:.0f}k CNY")
for k, v in panel["relabelled"].items():
    print(f"  {k:22s} F={v['F']:9,.0f} alpha*={v['alpha_star']:.3f} "
          f"total at P*={v['total_at_star']/1e3:.0f}k")
print("\nrides/bike/day")
for k, v in rides.items():
    print(f"  {k:22s} {v:.2f}")
print("\nslots/fleet")
for k, v in scale.items():
    print(f"  {k:22s} {v['slots_over_fleet']:.4f}")
print("saved", OUT)
