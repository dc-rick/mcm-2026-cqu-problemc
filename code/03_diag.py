"""Step 3: quick diagnostic maps (density, morning/evening net flow) to guide zoning."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_parquet(r"D:\code\mcm\2026CQU_C\data\trips.parquet")
df = df[(df.start_lat > 39.4) & (df.start_lon > 115.8) & (df.start_lat < 40.4) & (df.start_lon < 117.0)]
df["hour"] = df.starttime.dt.hour
df["date"] = df.starttime.dt.normalize()
core = df[df.date < "2017-05-20"]

LON0, LON1, LAT0, LAT1, STEP = 116.1, 116.75, 39.65, 40.15, 0.01
NX = int(round((LON1 - LON0) / STEP)); NY = int(round((LAT1 - LAT0) / STEP))


def gridcount(x, y):
    ix = np.clip(((x - LON0) / STEP).astype(int), 0, NX - 1)
    iy = np.clip(((y - LAT0) / STEP).astype(int), 0, NY - 1)
    out = np.zeros((NX, NY))
    np.add.at(out, (ix, iy), 1.0)
    return out


fig, ax = plt.subplots(1, 3, figsize=(19, 6))
h = ax[0].hist2d(core.start_lon, core.start_lat, bins=[280, 200], cmap="inferno",
                 norm=matplotlib.colors.LogNorm(), range=[[LON0, LON1], [LAT0, LAT1]])
plt.colorbar(h[3], ax=ax[0]); ax[0].set_title("trip starts (log density)")

for a, hrs, t in [(ax[1], (7, 8, 9), "morning 7-9"), (ax[2], (17, 18, 19), "evening 17-19")]:
    s = core[core.hour.isin(hrs)]
    outs = gridcount(s.start_lon.values, s.start_lat.values)
    ins = gridcount(s.end_lon.values, s.end_lat.values)
    net = ins - outs
    v = np.nanpercentile(np.abs(net), 99)
    im = a.imshow(net.T, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto",
                  extent=[LON0, LON1, LAT0, LAT1])
    plt.colorbar(im, ax=a); a.set_title(f"net arrivals - starts, {t}")
    a.set_xlabel("lon"); a.set_ylabel("lat")
plt.tight_layout(); plt.savefig(r"D:\code\mcm\2026CQU_C\figs\diag_maps.png", dpi=110)
print("saved fig")

s = core[core.hour.isin((7, 8, 9))]
for hrs, t in [((7, 8, 9), "am"), ((17, 18, 19), "pm")]:
    s = core[core.hour.isin(hrs)]
    outs = gridcount(s.start_lon.values, s.start_lat.values)
    ins = gridcount(s.end_lon.values, s.end_lat.values)
    net = ins - outs
    idx = np.dstack(np.unravel_index(np.argsort(net.ravel()), net.shape))[0]
    print(f"\n{t}: biggest exporters")
    for i, j in idx[:5]:
        print(f"  lon {LON0+i*STEP:.2f} lat {LAT0+j*STEP:.2f}  net {net[i,j]:+.0f}")
    print(f"{t}: biggest importers")
    for i, j in idx[-5:]:
        print(f"  lon {LON0+i*STEP:.2f} lat {LAT0+j*STEP:.2f}  net {net[i,j]:+.0f}")

print("\ncell trip counts (0.01 deg over 9 days):")
c = gridcount(core.start_lon.values, core.start_lat.values)
print("nonzero", (c > 0).sum(), ">=500", (c >= 500).sum(), ">=2000", (c >= 2000).sum(),
      "share of trips in top 200 cells", np.sort(c.ravel())[-200:].sum() / c.sum())
