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
def solve2(terms, k=2.0, rho=0.35, c0=0.8, free=0.0, incr=False):
    best = None
    for r in np.round(np.arange(0, 6.0001, 0.01), 4):
        f = (lambda x: pi(r, x, k, rho, c0) - pi(0, x, k, rho, c0)) if incr else (lambda x: pi(r, x, k, rho, c0))
        q = np.minimum(need, sum(cnt * f(x) for cnt, x in terms)).sum()
        tot = r * (q + free) + TRU * (N - q)
        if best is None or tot < best[2]: best = (r, q, tot)
    return best
paper = [(Dst, 0.025), (ring_code, 0.375)]
V = [("paper (reproduced)", paper, 0, False), ("A drop own cell, code timing", [(ring_code, .375)], 0, False),
     ("B drop own cell, deficit-hour timing", [(ring_right, .375)], 0, False), ("C = B + true distance 426 m", [(ring_right, .426)], 0, False),
     ("paper + zero-reward baseline removed", paper, 0, True), ("C + zero-reward baseline removed", [(ring_right, .426)], 0, True),
     ("C + pay outside riders (paper 6.7 rule)", [(ring_right, .426)], IM, False), ("C + pay outside + baseline removed", [(ring_right, .426)], IM, True)]
base = TRU * N
print(f"N={N:.0f} base={base:.0f} IM={IM:.0f}; outside riders' bill at r=1.2: {1.2*IM:.0f} CNY")
for name, t, fr, inc in V:
    r, q, tot = solve2(t, free=fr, incr=inc)
    print(f"  {name:42s} r*={r:4.2f} q*={q:7.0f} ({100*q/N:4.1f}%) total={tot:7.0f} saving={100*(1-tot/base):5.1f}%")
print("Table-9 parameter sets, variant 'C + pay outside + baseline removed':")
for k, rho, c0 in [(2, .35, .8), (1, .35, .8), (3, .35, .8), (2, .2, .8), (2, .5, .8), (2, .35, .4), (2, .35, 1.2)]:
    r, q, tot = solve2([(ring_right, .426)], k, rho, c0, free=IM, incr=True)
    r2, q2, tot2 = solve2([(ring_right, .426)], k, rho, c0, incr=True)
    print(f"  k={k} rho={rho} c0={c0}: with outside riders paid r*={r:4.2f} saving={100*(1-tot/base):5.1f}% | targeted r*={r2:4.2f} saving={100*(1-tot2/base):5.1f}%")
