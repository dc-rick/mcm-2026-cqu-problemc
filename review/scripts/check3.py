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
arr = np.zeros((24, C)); dep = np.zeros((24, C)); arr_ext = np.zeros((24, C))
np.add.at(arr, (h, d), 1); np.add.at(dep, (h, o), 1); np.add.at(arr_ext, (h[o != d], d[o != d]), 1)
arr_s, dep_s, arrx_s = arr / nd * SC, dep / nd * SC, arr_ext / nd * SC
net = arr_s - dep_s
need = np.maximum(0, -net).max(0); tstar = np.argmax(-net, axis=0)
q3 = np.load(D + "/q3_cell.npz", allow_pickle=False)
print("== Q3 ==")
print(f"  need recomputed {need.sum():.0f} vs file {q3['need'].sum():.0f}")
pos = need > 0
IM = arrx_s[tstar, np.arange(C)][pos].sum()
print(f"  trips from other cells already ending in a deficit cell during its worst hour (free riders): {IM:.0f}")
nb = (dist > 0.3) & (dist < 0.45)
cand_right = np.array([arr_s[tstar[c], nb[c]].sum() for c in range(C)])
Dstar = arr_s[tstar, np.arange(C)]
cand_code = nb.astype(float) @ Dstar
print(f"  neighbour pool: code (neighbour's own worst hour) {cand_code[pos].sum():.0f} vs paper's definition (deficit cell's hour) {cand_right[pos].sum():.0f}")
nbr_def = np.array([(arr_s[tstar[c], nb[c]] * (net[tstar[c], nb[c]] < 0)).sum() for c in range(C)])
print(f"  share of that pool coming from neighbours that are themselves losing bikes that hour: {nbr_def[pos].sum()/cand_right[pos].sum():.3f}")
def tau(x): return x / 10 * 60 + x * 1000 / 80
def pi(r, x): return 1 / (1 + np.exp(-(2.0 * r - 0.35 * tau(x) - 0.8)))
TRU = 83325 / 23959; N = need.sum()
rr = np.linspace(0, 6, 241)
rows = []
for r in rr:
    qn = np.minimum(need, cand_right * pi(r, 0.375)).sum()
    rows.append((r, qn, r * qn + TRU * (N - qn), r * (qn + IM) + TRU * (N - qn)))
rows = np.array(rows)
i1 = int(np.argmin(rows[:, 2])); i2 = int(np.argmin(rows[:, 3]))
print(f"  all-truck baseline {TRU*N:.0f} CNY")
print(f"  targeted offers, neighbours only, correct hour: r*={rows[i1,0]:.2f} q*={rows[i1,1]:.0f} ({100*rows[i1,1]/N:.1f}%) saving {100*(1-rows[i1,2]/(TRU*N)):.1f}%")
print(f"  public offer incl. free riders:              r*={rows[i2,0]:.2f} q*={rows[i2,1]:.0f} saving {100*(1-rows[i2,3]/(TRU*N)):.1f}%")
for r in (0.5, 1.0, 1.2, 2.0):
    j = int(np.argmin(abs(rows[:, 0] - r)))
    print(f"    r={r}: public-offer total {rows[j,3]:.0f}")

print("\n== Q4 catchment overlap (626 fences, 450 m) ==")
sites = np.load(D + "/q4_sites.npy", allow_pickle=False)
Ad = arr_s.sum(0)
mem = dist[sites] <= 0.45
tot_member = (mem * Ad[None, :]).sum()
union = Ad[mem.any(0)].sum()
mult = mem.sum(0)[mem.any(0)]
print(f"  sum over fences of catchment arrivals / arrivals in the union = {tot_member/union:.2f}")
print(f"  covered cells counted by 1/2/3+ fences: {(mult==1).sum()}/{(mult==2).sum()}/{(mult>=3).sum()}")

print("\n== Q5 standard error of the median cell deficit ==")
med = np.median(need[pos]); c = int(np.argmin(abs(need - med) + (~pos) * 1e9))
a_c, d_c = arr_s[tstar[c], c], dep_s[tstar[c], c]
na, ndp = arr[tstar[c], c], dep[tstar[c], c]
print(f"  cell {c}: deficit {need[c]:.1f}, scaled arrivals {a_c:.1f}, departures {d_c:.1f}; 7-day sample counts {na:.0f}/{ndp:.0f}")
print(f"  Poisson SE of the scaled mean net flow = sqrt(na+nd)/(7p) = {np.sqrt(na+ndp)/(nd*P):.1f}  (paper: 4.2)")

print("\n== Q2 stock depth (scaled flows, no trucks) ==")
z = np.load(D + "/typical_day.npz", allow_pickle=False); st = np.load(D + "/q2_state.npz", allow_pickle=False)
I0 = st["I0"]; dz = z["d_hour"] * SC; az = z["a_hour"] * SC
I = I0[:, None] + np.concatenate([np.zeros((len(I0), 1)), np.cumsum(az - dz, 1)[:, :-1]], 1)
ratio = I / np.maximum(dz, 1e-9)
print(f"  min over zone-hours of (stock at start of hour / departures in that hour) = {ratio.min():.2f}")
print(f"  zone-hours where one hour of departures exceeds 50% of the stock: {(ratio < 2).sum()}")
print(f"  min of stock / sqrt(departures) = {(I/np.sqrt(np.maximum(dz,1))).min():.0f}")
