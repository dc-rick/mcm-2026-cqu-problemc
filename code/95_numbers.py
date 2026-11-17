"""Print every number the paper cites, read from the result files actually
written by the modelling scripts.  Nothing here is typed by hand.

Run from code/:  python 95_numbers.py
Every value printed below is reproduced from a file under data/runs/T0xx/ or
data/*.json.  The paper is allowed to cite only numbers that appear here.
"""
import io
import json
import os

import numpy as np
import pandas as pd

import lib
from lib import DATA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(name, file):
    return os.path.join(DATA, "runs", name, file)


def js(name, file):
    return json.load(io.open(run(name, file), encoding="utf-8"))


def head(title):
    print("=" * 70, "\n" + title)


pd.set_option("display.width", 220)

# ------------------------------------------------------------------ DATA
head("DATA")
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

z = np.load(os.path.join(DATA, "typical_day.npz"), allow_pickle=True)
od = z["od_z"]
print("od_z sum (sample):", od.sum(), " scaled:", od.sum() / 0.128)
print("zone demand min/max:", od.sum((0, 2)).min(), od.sum((0, 2)).max())

daily = df.groupby("date").size()
typ = daily.loc[[str(x) for x in lib.TYP_DAYS]]
print("typical-day volume CV %:", round(100 * typ.std() / typ.mean(), 3))

fleet = js("T001R2", "fleet_closed.json")
trunc = js("T001R2", "truncation.json")
print("fleet central:", fleet["central"]["F"])
print("fleet interval:", fleet["central"]["model_interval"])
print("fleet method:", fleet["central"]["method"])
print("fleet sampling se:", fleet["methods"]["B_chapman_9day_vs_5day"]["se"])
print("fleet sampling ci95:",
      fleet["methods"]["B_chapman_9day_vs_5day"]["ci95_sampling"])
print("rides/bike/day:", fleet["conversions"]["rides_per_bike_per_day"])
print("alpha*:", fleet["conversions"]["alpha_star"])
print("truncation summary:", trunc["summary"])

p = js("T002", "p_estimate.json")
print("p exact:", p["consecutive"]["exact"])
print("p le160:", p["consecutive"]["le160"])
print("p le250:", p["consecutive"]["le250"])
print("p bracket:", p["identification"]["bracket"])
rez = js("T002", "resample.json")
print("resample reference F:", rez["reference"]["F"])
for row in rez["resample"]:
    print("resample q=%.2f mean %.1f dev_mean %.4f dev_max_abs %.4f "
          "share_gt4pct %.3f" % (row["q"], row["mean"], row["dev_mean"],
                                 row["dev_max_abs"], row["share_gt4pct"]))

# ------------------------------------------------------------------ Q1
head("Q1")
q1 = json.load(io.open(os.path.join(DATA, "q1_params.json"), encoding="utf-8"))
print("q1 beta:", {k: round(v["beta"], 6) for k, v in q1["bands"].items()})
dv = js("T005", "dest_validation.json")
print("day split:", dv["day_split"]["train_days"], dv["day_split"]["test_days"],
      dv["day_split"]["train_trips"], dv["day_split"]["test_trips"])
for b in dv["day_split"]["bands"]:
    print(b["band"], "n", b["n_train"], "beta", round(b["beta_full"], 6),
          "full", round(b["nlp"]["full"], 6),
          "dist", round(b["nlp"]["distance"], 6),
          "attr", round(b["nlp"]["attraction"], 6),
          "unif", round(b["nlp"]["uniform"], 6),
          "top5", round(b["top5"]["full"], 6),
          "top5d", round(b["top5"]["distance"], 6),
          "dnlp", round(b["delta_nlp_full_minus_distance"], 6),
          "dtop5", round(b["delta_top5_full_minus_distance"], 6))
fig3 = js("T005", "fig3_redraw.json")
print("KS pooled cross-cell:", fig3["ks"]["pooled_cross_cell"])
print("KS old simplified:", fig3["old_fig3a"]["ks"])
print("ki stability:")
ks = js("T005", "k_stability.json")["k"]
for kk in ("3", "4", "5", "6"):
    x = ks[kk]
    print(" k", kk, "sil", round(x["silhouette_40k_representative"], 6),
          "ari", round(x["ari_pairwise_mean"], 6),
          "ari_min", round(x["ari_pairwise_min"], 6))
rs = js("T005", "return_share.json")
print("same-cell overall:", rs["overall"]["trip_weighted_same_cell"])
for s in rs["by_segment"]:
    print("seg", s["paper_label"], "users", s["users"], "same",
          round(s["trip_weighted_same_cell"], 6), "origin_revisit",
          round(s["trip_weighted_origin_revisit"], 6), "days",
          round(s["user_mean_n_days"], 3), "trips_total",
          round(s["user_mean_trips_total"], 3), "per_active_day",
          round(s["user_mean_trips_per_active_day"], 3))
q1o = js("T006", "q1_vs_obs.json")
print("Q1 same-cell model/obs/diff:", q1o["same_cell_share"])
print("Q1 distance KS:", q1o["distance_ks"]["ks"])
print("Q1 top5 weighted:", q1o["top5"]["weighted"])
print("Q1 zone L1:", q1o["zone_od"])
seg = pd.read_csv(os.path.join(DATA, "q1_segments.csv"))
print(seg.round(3).to_string())

# ------------------------------------------------------------------ Q2
head("Q2")
q2 = js("T007", "q2_results_q1.json")
for name, row in q2["scenarios"].items():
    print(name, "obj", round(row["obj"], 2), "moved", round(row["moved"], 2),
          "truck_km", round(row["truck_km"], 2), "unmet", round(row["unmet"], 2),
          "dev", round(row["dev"], 2), "trips", round(row["trips"], 2))
o = q2["obs_baseline"]
print("obs K60 obj", round(o["obj"], 2), "moved", round(o["moved"], 2),
      "truck_km", round(o["truck_km"], 2), "unmet", round(o["unmet"], 2),
      "dev", round(o["dev"], 2))
print("q1 minus obs:", q2["q1_vs_obs"]["q1_minus_obs"])
cons = js("T007", "q2_conservation.json")
for name, row in cons["scenarios"].items():
    print("conservation", name, "city", row["city_residual_max_abs"],
          "zone", row["zone_residual_max_abs"],
          "u-dep", row["u_minus_dep_max"], "pass", row["passes"])
gr = js("T007", "q2_greedy_check.json")
print("greedy peak", gr["max_hourly_moved"], "limit", gr["hourly_limit"],
      "per_truck", gr["max_source_bikes_per_truck"], "viol",
      gr["hourly_violations"], gr["source_violations"])
ng = js("T007", "q2_night.json")
print("night moved share", ng["moved_night_share"], "truck-km share",
      ng["truck_km_night_share"])

# ------------------------------------------------------------------ Q3
head("Q3")
q3 = js("T008", "q3_results.json")
pool = np.load(run("T008", "q3_pool.npz"))
need = pool["need"]
print("need total:", q3["settings"]["need_definition"] and need.sum())
print("need cells >0:", int((need > 0).sum()))
print("need median/p90/p99:", np.percentile(need[need > 0], [50, 90, 99]))
print("pool:", q3["pool"])
for key in ("targeted", "targeted_conservative"):
    t = q3[key]
    print(key, "bound", t["bound"], "r*", t["r_star"], "q*", t["q_star"],
          "share", t["share_of_need"], "bill", t["bill"],
          "truck", t["truck_residual"], "total", t["total"],
          "all_truck", t["all_truck"], "saving", t["saving"],
          "marginal", t["marginal_cost_at_star"], "unit", t["unit"])
print("free rider:", q3["free_rider"])
print("public push:", {k: q3["public_push"][k] for k in
                       ("r_star", "q_star", "bill", "free_rider_bill",
                        "truck_residual", "total", "saving")})
print("public r=1.2:", {k: q3["public_push_at_r_1_2"][k] for k in
                        ("r", "q", "bill", "free_rider_bill", "total",
                         "saving")})
print("public old 3.478:", {k: q3["public_push_old_unit_3_478"][k] for k in
                            ("r_star", "q_star", "bill", "free_rider_bill",
                             "total", "all_truck", "saving")})
print("interval:", q3["interval"])
print("zone decomposition:")
for key in ("targeted", "targeted_conservative"):
    print(" ", key, q3["zone_decomposition"][key])
qr = js("T008", "q3_qr.json")
qrc = js("T008", "q3_qr_conservative.json")
acc = q3["settings"]["acceptance"]
for d in (0.05, 0.225, 0.45):
    tau = d / 10.0 * 60.0 + d * 1000.0 / 80.0
    z = acc["kappa"] * q3["targeted"]["r_star"] - acc["rho"] * tau - acc["c0"]
    print("acceptance r=%.2f d=%.3f tau=%.4f p=%.7f" %
          (q3["targeted"]["r_star"], d, tau, 1.0 / (1.0 + np.exp(-z))))
print("q(6) upper:", qr["q"][-1], "share", qr["q"][-1] / need.sum())
print("q(6) conservative:", qrc["q"][-1], "share",
      qrc["q"][-1] / need.sum())
for r in (1.0, 1.5, 1.76, 1.83, 2.0, 3.0):
    qu = np.interp(r, qr["r_grid"], qr["q"])
    qc = np.interp(r, qrc["r_grid"], qrc["q"])
    print(" r", r, "upper", qu, "cons", qc)

# ------------------------------------------------------------------ Q4
head("Q4")
q4 = js("T009", "q4_results.json")
print("settings:", q4["settings"])
print("surfaces:", q4["surfaces"])
print("validation:", q4["validation"])
print("overlap:", q4["overlap"])
print("frontier summary:", q4["frontier_summary"])
fr = js("T009", "q4_frontier.json")
for surf in ("q1", "obs"):
    pts = fr[surf]["points"]
    print(surf, "points end:", pts["n_fences"][-1], pts["coverage"][-1],
          pts["capacity_meas343"][-1], pts["capacity_fleet495"][-1])
fences = pd.read_csv(run("T009", "q4_fences.csv"))
for col in ("capacity_meas343", "capacity_fleet495"):
    print("selected", col, "n", len(fences), "median",
          fences[col].median(), "p90", fences[col].quantile(0.9),
          "max", fences[col].max(), "sum", fences[col].sum(),
          "share>150", (fences[col] > 150).mean())
print("selected covered arrivals:",
      fences["coverage_arrivals_day"].sum())

# ------------------------------------------------------------------ Q5
head("Q5")
q5 = json.load(open(DATA + r"\q5_summary.json"))
print(json.dumps(q5, indent=1))
