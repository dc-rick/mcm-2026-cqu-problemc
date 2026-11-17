"""T005 delivery validator: recompute the self-checks and inspect the table."""
import io
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(ROOT, "data", "runs", "T005")
OUT = os.path.join(ROOT, "交付", "T005_delivery.json")

d = json.load(io.open(OUT, encoding="utf-8"))
dv = json.load(io.open(os.path.join(RUN, "dest_validation.json"),
                       encoding="utf-8"))
fg = json.load(io.open(os.path.join(RUN, "fig3_redraw.json"),
                       encoding="utf-8"))
kk = json.load(io.open(os.path.join(RUN, "k_stability.json"),
                       encoding="utf-8"))
rs = json.load(io.open(os.path.join(RUN, "return_share.json"),
                       encoding="utf-8"))

ok = []


def check(name, cond, detail=""):
    ok.append((name, bool(cond), detail))
    print(("PASS " if cond else "FAIL ") + name + (" | " + detail if detail else ""))


# 1-10 declared checks
for k, v in d["checks"].items():
    check("check " + k, v)

# four models x five bands
bands = dv["day_split"]["bands"]
check("five bands", len(bands) == 5)
check("four models per band", all(
    set(b["nlp"]) == {"full", "distance", "attraction", "uniform"} and
    set(b["top5"]) == {"full", "distance", "attraction", "uniform"}
    for b in bands))
check("day split declared", dv["day_split"]["train_days"] == [
    "2017-05-10", "2017-05-11", "2017-05-15", "2017-05-16", "2017-05-18"]
    and dv["day_split"]["test_days"] == ["2017-05-12", "2017-05-19"])
check("test size", dv["day_split"]["test_trips"] ==
      sum(b["n_test"] for b in bands) == 555675)
check("train size", dv["day_split"]["train_trips"] ==
      sum(b["n_train"] for b in bands))

# pooled consistency
pool = dv["day_split"]["pooled"]
nt = sum(b["n_test"] for b in bands)
for m in ("full", "distance", "attraction", "uniform"):
    w = sum(b["nlp"][m] * b["n_test"] for b in bands) / nt
    check("pooled nlp " + m, abs(w - pool["nlp"][m]) < 1e-9,
          f"{w:.6f} vs {pool['nlp'][m]:.6f}")
    w = sum(b["top5"][m] * b["n_test"] for b in bands) / nt
    check("pooled top5 " + m, abs(w - pool["top5"][m]) < 1e-9)
check("uniform top5 = 5/n", abs(pool["top5"]["uniform"] - 5 / 5263) < 1e-12)
check("uniform nlp = ln(5263)",
      abs(pool["nlp"]["uniform"] - 8.568456) < 1e-5)
check("pooled delta nlp", abs(
    pool["delta_nlp_full_minus_distance"] -
    (pool["nlp"]["full"] - pool["nlp"]["distance"])) < 1e-7)

# distance model own MLE
for b in bands:
    check("distance beta is own MLE " + b["band"],
          abs(b["beta_distance"] - b["beta_full"]) > 0.001,
          f"diff {b['beta_distance']-b['beta_full']:+.4f}")
check("old distance column invalid",
      "invalid" in d["dest_validation"]["old_q1_validation_status"] and
      "full model" in d["dest_validation"]["old_q1_validation_status"])
check("old top5 reproduced by argpartition",
      all(abs(r["old_top5_dist"] - r["new_top5_argpartition"]) < 1e-6
          for r in d["dest_validation"]["distance_only_refit"]))
check("refit improves old nlp",
      all(r["ll_improvement"] > 0 for r in
          d["dest_validation"]["distance_only_refit"]))

# figure
check("fig3 script exists", os.path.exists(os.path.join(
    ROOT, "code", "t005_fig3.py")))
check("fig3 files exist", all(os.path.getsize(os.path.join(RUN, f)) > 1000
                              for f in ("t005_fig3_redraw.pdf",
                                        "t005_fig3_redraw.png")))
check("new KS below old", fg["ks"]["pooled_cross_cell"] <
      fg["old_fig3a"]["ks"])
check("GPS comparison reproduces old geometry",
      abs(fg["ks"]["pooled_cross_cell_vs_gps"] -
          fg["old_fig3a"]["ks"]) < 0.02)
check("same-cell excluded from KS",
      "excluded" in fg["same_cell"]["definition"].lower())
check("model same-cell share in [0,1]",
      0 <= fg["same_cell"]["model_implied_share"] <= 1)
check("observed same-cell share in [0,1]",
      0 <= fg["same_cell"]["observed_share"] <= 1)

# k stability
check("k list 3-6", kk["ks"] == [3, 4, 5, 6])
check("20 seeds per k", kk["n_seeds"] >= 20)
check("four k rows", len(d["k_stability"]["table"]) == 4)
check("cross-k 4x4", len(kk["cross_k_matrix"]) == 4 and
      all(len(v) == 4 for v in kk["cross_k_matrix"].values()))
for a in ("3", "4", "5", "6"):
    for b in ("3", "4", "5", "6"):
        v = kk["cross_k_matrix"][a][int(b) - 3]
        if a == b:
            check(f"cross-k diag {a}", abs(v - 1) < 1e-12)
        else:
            check(f"cross-k symmetric {a}-{b}",
                  abs(v - kk["cross_k_matrix"][b][int(a) - 3]) < 1e-12)
for k in ("3", "4", "5", "6"):
    r = kk["k"][k]
    check("ARI range k" + k, -1 <= r["ari_pairwise_mean"] <= 1 and
          -1 <= r["ari_pairwise_min"] <= 1)
    check("silhouette range k" + k, 0 <= r["silhouette_40k_representative"]
          <= 1)

# return share
h = rs["heavy_short_trip"]
check("same-cell overall in [0,1]",
      0 <= rs["overall"]["trip_weighted_same_cell"] <= 1)
check("heavy same-cell < origin revisit",
      h["same_cell_trip_weighted"] < h["recomputed_origin_revisit_trip_weighted"])
check("heavy origin revisit near quarter",
      0.24 < h["recomputed_origin_revisit_trip_weighted"] < 0.28)
check("heavy same-cell below quarter",
      h["same_cell_trip_weighted"] < 0.25)
check("return_share definition present",
      "origin cell == destination cell" in rs["definition"]["same_cell"])

# integrity
check("paper unchanged", d["paper_integrity"]["paper"]["unchanged"])
check("data excluding runs unchanged",
      d["paper_integrity"]["data_excluding_runs"]["unchanged"])
check("q1_params unchanged", d["paper_integrity"]["q1_params_unchanged"])
check("q1_validation unchanged",
      d["paper_integrity"]["q1_validation_unchanged"])
check("lib unchanged", d["paper_integrity"]["lib_py_unchanged"])
check("no unexpected code changes",
      not d["code_integrity"]["unexpected_changes"])
check("nine t005 scripts", len(d["code_integrity"]["t005_new"]) == 9)

failed = [n for n, v, _ in ok if not v]
print(f"\n{len(ok)-len(failed)}/{len(ok)} checks passed")
if failed:
    raise SystemExit("FAILED: " + "; ".join(failed))
print("T005 delivery consistent")
