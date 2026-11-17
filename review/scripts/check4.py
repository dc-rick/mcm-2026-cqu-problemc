import sys, numpy as np, pandas as pd
D = sys.argv[1]; P = 0.128; SC = 1 / P
TYP = ["2017-05-10","2017-05-11","2017-05-12","2017-05-15","2017-05-16","2017-05-18","2017-05-19"]
cells = pd.read_parquet(D + "/cells.parquet"); C = len(cells)
xy = cells[["x", "y"]].to_numpy()
dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
df = pd.read_parquet(D + "/trips.parquet", columns=["starttime","start_lat","start_lon","end_lat","end_lon"])
df = df[(df.start_lat > 39.4) & (df.start_lon > 115.8) & (df.start_lat < 40.4) & (df.start_lon < 117.0)]
df = df[df.starttime.dt.strftime("%Y-%m-%d").isin(TYP)]
idx = {int(k): i for i, k in enumerate(cells.key.to_numpy())}
ks = (np.floor(df.start_lat / 0.005).astype(int) * 100000 + np.floor(df.start_lon / 0.005).astype(int)).map(idx)
ke = (np.floor(df.end_lat / 0.005).astype(int) * 100000 + np.floor(df.end_lon / 0.005).astype(int)).map(idx)
ok = ks.notna() & ke.notna()
o = ks[ok].astype(int).values; d = ke[ok].astype(int).values; h = df.starttime[ok].dt.hour.values
nd = len(TYP)
arr = np.zeros((24, C)); dep = np.zeros((24, C)); arx = np.zeros((24, C))
np.add.at(arr, (h, d), 1); np.add.at(dep, (h, o), 1); np.add.at(arx, (h[o != d], d[o != d]), 1)
arr_s, dep_s, arx_s = arr / nd * SC, dep / nd * SC, arx / nd * SC
net = arr_s - dep_s
need = np.maximum(0, -net).max(0); ts = np.argmax(-net, axis=0); pos = need > 0
N = need.sum(); ar = np.arange(C)
ring = (dist > 0.3) & (dist < 0.45)
print("ring distances (km):", np.unique(np.round(dist[ring], 3)))
Dst = arr_s[ts, ar]
ring_code = ring.astype(float) @ Dst
ring_right = np.array([arr_s[ts[c], ring[c]].sum() for c in range(C)])
IM = arx_s[ts, ar][pos].sum()
TRU = 83325.24050274817 / 23959.151785714257
def tau(x): return x / 10 * 60 + x * 1000 / 80
def pi(r, x, k, rho, c0): return 1 / (1 + np.exp(-(k * r - rho * tau(x) - c0)))
def solve(terms, k=2.0, rho=0.35, c0=0.8, free=0.0):
    best = None
    for r in np.round(np.arange(0, 6.0001, 0.05), 4):
        q = np.minimum(need, sum(cnt * pi(r, x, k, rho, c0) for cnt, x in terms)).sum()
        tot = r * (q + free) + TRU * (N - q)
        if best is None or tot < best[2]: best = (r, q, tot)
    return best
paper = [(Dst, 0.025), (ring_code, 0.375)]
fixA = [(ring_code, 0.375)]                      # drop own cell, keep code timing
fixB = [(ring_right, 0.375)]                     # drop own cell, deficit-hour timing (paper text)
fixC = [(ring_right, 0.426)]                     # + true neighbour distance
base = TRU * N
print(f"\nN={N:.0f}  all-truck={base:.0f}  free riders IM={IM:.0f}")
print("coverage potential: own cell only %.3f | code pool %.3f | neighbours, deficit hour %.3f" % (
    np.minimum(need, Dst).sum() / N, np.minimum(need, Dst + ring_code).sum() / N, np.minimum(need, ring_right).sum() / N))
r0 = np.minimum(need, sum(c * pi(0, x, 2, .35, .8) for c, x in paper)).sum()
print(f"paper model at r=0: q={r0:.0f} ({100*r0/N:.1f}% of N) placements for no reward")
print("\nvariant                    kappa rho  c0  | paper r* q* save | A save | B save | C save | C+free r* save")
for k, rho, c0 in [(2, .35, .8), (1, .35, .8), (3, .35, .8), (2, .2, .8), (2, .5, .8), (2, .35, .4), (2, .35, 1.2)]:
    p = solve(paper, k, rho, c0); a = solve(fixA, k, rho, c0); b = solve(fixB, k, rho, c0)
    c = solve(fixC, k, rho, c0); cf = solve(fixC, k, rho, c0, free=IM)
    s = lambda t: 100 * (1 - t[2] / base)
    print(f"{'':26s} {k:4.1f} {rho:4.2f} {c0:4.1f} | {p[0]:4.2f} {p[1]:7.0f} {s(p):5.1f} | {s(a):5.1f} | {s(b):5.1f} | {s(c):5.1f} | {cf[0]:4.2f} {s(cf):5.1f}")
use = np.zeros((24, C))
for c in np.where(pos)[0]: use[ts[c], ring[c]] += 1
print(f"\nneighbour pool multiplicity (same rider counted for several deficit cells): {(use*arr_s).sum()/(arr_s*(use>0)).sum():.2f}")
rng = np.random.default_rng(0); Ns = []
for b in range(40):
    nb = (rng.poisson(arr) - rng.poisson(dep)) / nd * SC
    Ns.append(np.maximum(0, -nb).max(0).sum())
Ns = np.array(Ns)
print(f"\nbootstrap of N: observed {N:.0f}, resampled mean {Ns.mean():.0f} (sd {Ns.std():.0f}); bias-corrected ~{2*N-Ns.mean():.0f}")
se = np.sqrt(arr[ts, ar] + dep[ts, ar]) / (nd * P)
print(f"share of N in cells whose deficit exceeds 2 SE: {need[need > 2*se].sum()/N:.3f}; cells: {(need > 2*se).sum()} of {pos.sum()}")
z = np.load(D + "/typical_day.npz", allow_pickle=False); st = np.load(D + "/q2_state.npz", allow_pickle=False)
I0 = st["I0"]; cum = np.concatenate([np.zeros((len(I0), 1)), np.cumsum((z["a_hour"] - z["d_hour"]) * SC, 1)], 1)
I = I0[:, None] + cum
print(f"\nQ2: min zone stock {I0.min():.0f}, median {np.median(I0):.0f}; worst excursion {cum.min():.0f} (zone {np.unravel_index(cum.argmin(), cum.shape)[0]}, I0={I0[np.unravel_index(cum.argmin(), cum.shape)[0]]:.0f})")
rel = I.min(1) / I0; zr = rel.argmin()
print(f"    deepest relative drop: zone {zr}, min/I0={rel[zr]:.3f}, I0={I0[zr]:.0f}; drift sum|I24-I0|={np.abs(cum[:, -1]).sum():.0f}")
