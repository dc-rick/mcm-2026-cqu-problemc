"""Print every number the paper cites, read from the result files actually written
by the modelling scripts.  Nothing here is typed by hand."""
import json
import numpy as np
import pandas as pd
import lib
from lib import DATA

pd.set_option("display.width", 200)
print("=" * 70, "\nDATA")
df = lib.load_trips(None)
print("trips total:", len(df))
print("users:", df.userid.nunique(), " bikes:", df.bikeid.nunique())
print("dates:", sorted(df.date.unique()))
print("date range:", df.date.min(), df.date.max())
cells = lib.get_cells()
print("study cells:", len(cells), " zones:", int(cells.zone.max()) + 1)
d = lib.load_trips(lib.TYP_DAYS)
d = lib.assign_cells(d, cells)
ok = d.o.notna() & d.d.notna()
print("typical-day sample trips:", int(ok.sum()), " share of TYP_DAYS trips:",
      round(100 * ok.sum() / len(d), 2))
print("cells in typical day:", len(np.unique(np.r_[d.o[ok], d.d[ok]])))

z = np.load(DATA + r"\typical_day.npz", allow_pickle=True)
od = z["od_z"]
print("od_z sum (sample):", od.sum(), " scaled:",
      od.sum() / 0.128)
print("zone demand min/max:", od.sum((0, 2)).min(), od.sum((0, 2)).max())

print("=" * 70, "\nQ1")
q1 = json.load(open(DATA + r"\q1_params.json"))
print(json.dumps(q1, indent=1)[:1200])
val = pd.read_csv(DATA + r"\q1_validation.csv")
print(val.round(3).to_string())
seg = pd.read_csv(DATA + r"\q1_segments.csv")
print(seg.round(3).to_string())

print("=" * 70, "\nQ2")
r = pd.DataFrame(json.load(open(DATA + r"\q2_results.json")))
print(r[r["mode"] == "no trucks"][["alpha", "K", "obj", "unmet", "moved"]].round(1).to_string())
tr = r[r["mode"] == "trucks"]
print(tr[tr.K == 60][["alpha", "K", "obj", "unmet", "truck_km", "moved",
                      "dev"]].round(1).to_string())
print("unit cost at alpha=1,K=60:",
      round(tr[(tr.alpha == 1) & (tr.K == 60)].obj.iloc[0] /
            tr[(tr.alpha == 1) & (tr.K == 60)].moved.iloc[0], 3))
print("moved share of trips:", round(100 * tr[(tr.alpha == 1) & (tr.K == 60)].moved.iloc[0]
                                     / tr[(tr.alpha == 1) & (tr.K == 60)].trips.iloc[0], 3))
st = np.load(DATA + r"\q2_state.npz", allow_pickle=True)
print("I0 quantiles:", np.percentile(st["I0"], [5, 50, 95]).round(1),
      " sum:", st["I0"].sum().round(0))

print("=" * 70, "\nQ3")
q3 = json.load(open(DATA + r"\q3_summary.json"))
for k, v in q3.items():
    if k != "supply":
        print(" ", k, v)
print("  supply:", [[round(a, 2), round(b, 0), round(c, 0)] for a, b, c in q3["supply"]])

print("=" * 70, "\nQ4")
q4 = json.load(open(DATA + r"\q4_summary.json"))
print(q4)
fr = np.load(DATA + r"\q4_frontier.npz")
for k in ("c250", "c450", "c650"):
    c = fr[k]
    print(k, "n=", len(c), "cov at 100/200/400/end:",
          [round(100 * c[min(i, len(c) - 1)], 1) for i in (99, 199, 399, len(c) - 1)])

print("=" * 70, "\nQ5")
q5 = json.load(open(DATA + r"\q5_summary.json"))
print(json.dumps(q5, indent=1))
