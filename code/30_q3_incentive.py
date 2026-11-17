"""Q3: user incentive design, computed at neighbourhood (cell) resolution.

The truck plan of Q2 moves bikes BETWEEN zones.  An incentive can only shift a
trip's END by a short distance, so its natural scale is the cell: it recruits
users who are already heading within a few hundred metres of a cell that will
run short, and pays them to finish there.

Outputs: absorption potential, logit acceptance curve, cost curve, optimal mix.
"""
import os
import json
import numpy as np
import lib
from lib import DATA

SCALE = 1.0 / 0.128
FLEET = 343638.0
cells = lib.get_cells()
C = len(cells)
zid = cells.zone.to_numpy()
xy = cells[["x", "y"]].to_numpy()
dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))

df = lib.load_trips(lib.TYP_DAYS)
df = lib.assign_cells(df, cells)
ok = df.o.notna() & df.d.notna()
s = df[ok].copy(); s["o"] = s.o.astype(int); s["d"] = s.d.astype(int)
nd = len(lib.TYP_DAYS)
q = s.starttime.dt.hour.values * 4 + s.starttime.dt.minute.values // 15
dep = np.zeros((96, C)); arr = np.zeros((96, C))
np.add.at(dep, (q, s.o.values), 1.0); np.add.at(arr, (q, s.d.values), 1.0)
dep = dep / nd * SCALE; arr = arr / nd * SCALE

# resupply requirement = peak hourly net outflow (flow-only, no inventory needed)
# need_c is measured in the cell's own worst hour, so the pool of users who can
# be recruited to fill it must be counted in that same hour, not over the day.
net15 = arr - dep
net_hr = net15.reshape(24, 4, C).sum(1)          # (24, C) hourly net flow
arr_hr = arr.reshape(24, 4, C).sum(1)            # (24, C) hourly arrivals
need = np.maximum(0, -net_hr).max(0)             # bikes a cell loses at its worst hour
tstar = np.argmax(-net_hr, axis=0)               # hour in which that happens
Dstar = arr_hr[tstar, np.arange(C)]              # arrivals during that hour
print(f"cells with positive peak net outflow: {(need>0).sum()}, "
      f"total resupply requirement {need.sum():.0f} bikes per peak hour")
print(f"need quantiles {np.percentile(need[need>0],[50,90,99]).round(1)}")

# daily trips ending in each cell
Dday = arr.sum(0)
print(f"total arrivals/day {Dday.sum():.0f}")

# ---- absorption potential: trips ending within R of a short cell ----------
Rs = (0.45, 0.60, 0.85)
near_pools = []
for R in Rs:
    near = dist <= R
    pot = near.astype(float) @ Dstar         # trips that could be shifted in
    pot = np.maximum(pot, 0)
    near_pools.append(pot)
    cover = np.minimum(pot, need).sum()
    print(f"R={R*1000:.0f} m: cover {cover:.0f} of {need.sum():.0f} "
          f"= {100*cover/need.sum():.1f}% (candidate trips counted per cell)")

# ---- logit acceptance -----------------------------------------------------
V_BIKE, WALK_M_MIN = 10.0, 80.0
KAPPA, RHO, C0 = 2.0, 0.35, 0.8          # utilities per CNY / per minute, inertia


def detour_min(r_km):
    return r_km / V_BIKE * 60 + r_km * 1000 / WALK_M_MIN


def p_acc(r, r_km):
    return 1.0 / (1.0 + np.exp(-(KAPPA * r - RHO * detour_min(r_km) - C0)))


print("\nacceptance probability")
print("reward   R=450m   R=600m   R=850m")
for r in (0.5, 1.0, 1.5, 2.0, 3.0):
    print(f"{r:4.1f}   " + "   ".join(f"{p_acc(r,R):.3f}" for R in (0.45, 0.60, 0.85)))

# ---- the supply curve of voluntary relocations ---------------------------
# A user whose destination lies within R of a deficit cell is offered r CNY to
# finish at the deficit cell instead.  The extra distance is not a single
# number: trips end at every distance in [0, R], and those that end closest
# accept most readily.  We therefore integrate the logit over the observed
# distribution of detours rather than evaluating it at the mean.
# For a reward r the expected number of placements in cell c is
#     q_c(r) = min( need_c , sum_b n_{c,b} * p(d_b, r) )
# where n_{c,b} counts the trips ending in band b around cell c.  Multiplying
# r by q gives the bill; q(r) is the incentive's supply curve.
R0 = 0.45
EDGES = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30, R0])       # km
MID = 0.5 * (EDGES[:-1] + EDGES[1:])
counts = np.array([((dist >= a) & (dist < b)).astype(np.float32) @ Dstar
                   for a, b in zip(EDGES[:-1], EDGES[1:])])      # (band, cell)
print(f"\n450 m band: {counts.shape[0]} detour bands, "
      f"candidate trips per deficit cell = "
      f"{np.median(counts.sum(0)[need > 0]):.0f} vs median need "
      f"{np.median(need[need > 0]):.1f}")

band_p = lambda r: 1.0 / (1.0 + np.exp(-(KAPPA * r
                                         - RHO * np.array([detour_min(d) for d in MID])
                                         - C0)))


def placements(r):
    return float(np.minimum(need, (counts * band_p(r)[:, None]).sum(0)).sum())


TRUCK_UNIT = 83325.0 / 23959.0
print(f"truck unit cost {TRUCK_UNIT:.2f} CNY per bike moved\n")
print("reward   placements   %need    bill (CNY)   cost/bike")
rs = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
supply = []
for r in rs:
    q = placements(r)
    supply.append((float(r), q, float(r * q)))
    print(f"{r:5.2f}   {q:10.0f}   {100*q/need.sum():5.1f}%   {r*q:10.0f}   {r:5.2f}")

q_cross = placements(TRUCK_UNIT)
print(f"\nat the truck cost ({TRUCK_UNIT:.2f} CNY) the incentive buys "
      f"{q_cross:.0f} placements = {100*q_cross/need.sum():.1f}% of the daily need")
print(f"deficit with no qualifying trip within {R0*1000:.0f} m: "
      f"{need.sum() - np.minimum(near_pools[0], need).sum():.0f} bikes/day")

# ---- optimal uniform reward ----------------------------------------------
# The platform pays the same reward r to every adopter, so the bill is r*q(r)
# and what it leaves to the trucks costs TRUCK_UNIT each.  Minimise the two
# together; the first-order condition is that the marginal cost of an extra
# incentive placement equals the truck cost.
rr = np.linspace(0.0, 6.0, 241)
qq = np.array([placements(r) for r in rr])
tot = rr * qq + TRUCK_UNIT * (need.sum() - qq)
i = int(np.argmin(tot))
mc = np.gradient(rr * qq, qq)
print(f"\noptimal uniform reward {rr[i]:.2f} CNY -> {qq[i]:.0f} placements "
      f"({100*qq[i]/need.sum():.1f}% of need), incentive bill {rr[i]*qq[i]:.0f} CNY")
print(f"  trucks serve the remaining {need.sum()-qq[i]:.0f} bikes "
      f"({TRUCK_UNIT*(need.sum()-qq[i]):.0f} CNY), total {tot[i]:.0f} CNY")
print(f"  all-truck baseline {TRUCK_UNIT*need.sum():.0f} CNY, "
      f"saving {100*(1-tot[i]/(TRUCK_UNIT*need.sum())):.1f}%")
print(f"  marginal cost at the optimum {mc[i]:.2f} CNY/bike (truck {TRUCK_UNIT:.2f})")

# ---- how far the catchment must reach ------------------------------------
radii = np.linspace(0.05, 1.60, 63)
curve = []
for R in radii:
    pot = np.maximum((dist <= R).astype(np.float32) @ Dstar, 0)
    curve.append(float(np.minimum(pot, need).sum() / need.sum()))
curve = np.array(curve)
j = int(np.argmax(curve >= 0.95))
print(f"\ncatchment needed for 95% coverage: {radii[j]*1000:.0f} m "
      f"(at 450 m: {100*curve[np.argmin(abs(radii-0.45))]:.1f}%)")

json.dump(dict(need_total=float(need.sum()), truck_unit=float(TRUCK_UNIT),
               pool450=float(counts.sum()), detour_mean=float(detour_min(R0 / 2)),
               cover450=float(np.minimum(near_pools[0], need).sum() / need.sum()),
               q_at_truck=float(q_cross),
               opt=dict(reward=float(rr[i]), q=float(qq[i]),
                        bill=float(rr[i] * qq[i]), total=float(tot[i]),
                        mc=float(mc[i])),
               pot={f"{int(R*1000)}": float(np.minimum(near_pools[k], need).sum())
                    for k, R in enumerate(Rs)},
               supply=[[float(a), float(b), float(c)] for a, b, c in supply]),
          open(os.path.join(DATA, "q3_summary.json"), "w"), indent=1)
np.savez(os.path.join(DATA, "q3_cell.npz"), need=need, Dday=Dday, Dstar=Dstar,
         pot450=near_pools[0], pot600=near_pools[1], pot850=near_pools[2],
         radii=radii, curve=curve)
