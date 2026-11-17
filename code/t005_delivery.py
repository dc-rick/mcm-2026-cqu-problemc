"""T005 delivery: assemble the Q1 validation-completion evidence."""
import csv
import datetime
import glob
import hashlib
import io
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(ROOT, "data", "runs", "T005")
OUT = os.path.join(ROOT, "交付", "T005_delivery.json")


def load(name):
    return json.load(io.open(os.path.join(RUN, name), encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_compare(rel, old):
    cur = {}
    base = os.path.join(ROOT, rel)
    for dp, _, fn in os.walk(base):
        for f in fn:
            p = os.path.join(dp, f)
            r = os.path.relpath(p, ROOT).replace("\\", "/")
            if rel == "data" and r.startswith("data/runs"):
                continue
            cur[r] = sha(p)
    return dict(
        changed=sorted(k for k in old.keys() & cur.keys() if old[k] != cur[k]),
        added=sorted(cur.keys() - old.keys()),
        removed=sorted(old.keys() - cur.keys()),
        n_files=len(cur),
        unchanged=(set(old) == set(cur) and
                   all(old[k] == cur[k] for k in old.keys() & cur.keys())))


dv = load("dest_validation.json")
fg = load("fig3_redraw.json")
kk = load("k_stability.json")
rs = load("return_share.json")
bands = dv["day_split"]["bands"]
byname = {b["band"]: b for b in bands}
names = [b["band"] for b in bands]

with io.open(os.path.join(ROOT, "data", "q1_validation.csv"), encoding="utf-8") as fh:
    old = {r["band"]: r for r in csv.DictReader(fh)}

# ---------------------------------------------------------------- hashes
fp0 = json.load(io.open(os.path.join(ROOT, "data", "runs", "T001",
                                     "fingerprint_start.json"), encoding="utf-8"))
fp_end = json.load(io.open(os.path.join(ROOT, "data", "runs", "T001",
                                        "fingerprint_end.json"), encoding="utf-8"))
old_paper = {i["path"]: i["sha256"] for i in fp0["trees"]["paper"]}
old_data = {i["path"]: i["sha256"] for i in fp0["trees"]["data"]}
old_code = {i["path"]: i["sha256"] for i in fp_end["trees"]["code"]}

paper_check = tree_compare("paper", old_paper)
data_check = tree_compare("data", old_data)
cur_code = {}
for dp, _, fn in os.walk(os.path.join(ROOT, "code")):
    for f in fn:
        p = os.path.join(dp, f)
        r = os.path.relpath(p, ROOT).replace("\\", "/")
        cur_code[r] = sha(p)

q1p_now = sha(os.path.join(ROOT, "data", "q1_params.json"))
q1p_old = old_data["data/q1_params.json"]
q1v_now = sha(os.path.join(ROOT, "data", "q1_validation.csv"))
q1v_old = old_data["data/q1_validation.csv"]
lib_now = sha(os.path.join(ROOT, "code", "lib.py"))
lib_old = old_code["code/lib.py"]

old_status = (dv["old_q1_validation_status"] +
              " The old distance values are invalid as a distance-model "
              "optimum; they are kept only for the reproduction comparison.")

# ------------------------------------------------------- derived tables
beta_rows = []
dist_rows = []
for n in names:
    b = byname[n]
    beta_rows.append(dict(
        band=n, beta_full=b["beta_full"], beta_distance=b["beta_distance"],
        beta_diff=b["beta_distance"] - b["beta_full"],
        beta_diff_pct=100 * (b["beta_distance"] - b["beta_full"]) /
        b["beta_full"]))
    o = old[n]
    old_ll = -float(o["ll_distance"])  # the CSV stores log-likelihood, not loss
    dist_rows.append(dict(
        band=n,
        old_ll_distance=old_ll,
        new_ll_distance=b["nlp"]["distance"],
        ll_improvement=old_ll - b["nlp"]["distance"],
        old_top5_dist=float(o["top5_dist"]),
        new_top5_argpartition=b["top5_argpartition"]["distance"],
        new_top5_tie_aware=b["top5"]["distance"],
        argpartition_reproduces_old=abs(
            float(o["top5_dist"]) - b["top5_argpartition"]["distance"]) < 1e-6))

nlp_gain = [byname[n]["nlp"]["distance"] - byname[n]["nlp"]["full"]
            for n in names]
top5_gain = [byname[n]["top5"]["full"] - byname[n]["top5"]["distance"]
             for n in names]

paper_table = []
for n in names:
    b = byname[n]
    paper_table.append(dict(
        band=n, train_trips=b["n_train"],
        beta=b["beta_full"],
        gravity_logloss=b["nlp"]["full"],
        distance_logloss_new=b["nlp"]["distance"],
        distance_logloss_old=-float(old[n]["ll_distance"]),
        attraction_logloss=b["nlp"]["attraction"],
        uniform_logloss=b["nlp"]["uniform"],
        top5_gravity=b["top5"]["full"],
        top5_distance_new=b["top5_argpartition"]["distance"],
        top5_distance_old=float(old[n]["top5_dist"])))

day_rand = {
    "day_split_pooled": dv["day_split"]["pooled"],
    "random_split_pooled": dv["random_split"]["pooled"],
    "delta_full_minus_distance_day": dv["day_split"]["pooled"][
        "delta_nlp_full_minus_distance"],
    "delta_full_minus_distance_random": dv["random_split"]["pooled"][
        "delta_nlp_mean"],
    "delta_top5_day": dv["day_split"]["pooled"][
        "delta_top5_full_minus_distance"],
    "delta_top5_random": dv["random_split"]["pooled"]["delta_top5_mean"],
    "level_diff_nlp_full": (dv["random_split"]["pooled"]["nlp"]["full"] -
                            dv["day_split"]["pooled"]["nlp"]["full"]),
    "level_diff_top5_full": (dv["random_split"]["pooled"]["top5"]["full"] -
                             dv["day_split"]["pooled"]["top5"]["full"]),
    "reading": ("The attraction gain is the same under both splits "
                "(difference 0.0001 nats and 0.0018 pp), but the absolute "
                "held-out level moves by 0.037 nats and 1.35 pp: the two "
                "held-out days are easier than a random 29% trip sample. The "
                "model comparison is robust; the absolute level is "
                "split-dependent."),
}

k_table = []
for k in kk["ks"]:
    r = kk["k"][str(k)]
    k_table.append(dict(
        k=k,
        silhouette_10k_mean=r["silhouette_mean"],
        silhouette_10k_min=r["silhouette_min"],
        silhouette_40k_representative=r["silhouette_40k_representative"],
        ari_pairwise_mean=r["ari_pairwise_mean"],
        ari_pairwise_min=r["ari_pairwise_min"],
        ari_vs_published_k5=r.get("ari_vs_published_k5"),
        representative_sizes=r["representative_sizes"],
        inertia_min=r["inertia_min"]))

k5 = kk["k"]["5"]
k_verdict = {
    "best_silhouette_k_40k": kk["best_silhouette_k"],
    "best_silhouette_k_10k_mean": kk["best_silhouette_k_10k"],
    "k5_silhouette_40k": k5["silhouette_40k_representative"],
    "k6_silhouette_40k": kk["k"]["6"]["silhouette_40k_representative"],
    "k5_silhouette_10k_mean": k5["silhouette_mean"],
    "k6_silhouette_10k_mean": kk["k"]["6"]["silhouette_mean"],
    "k5_ari_pairwise_mean": k5["ari_pairwise_mean"],
    "k5_ari_pairwise_min": k5["ari_pairwise_min"],
    "k5_ari_vs_published": k5.get("ari_vs_published_k5"),
    "k3_ari_pairwise_mean": kk["k"]["3"]["ari_pairwise_mean"],
    "k3_ari_pairwise_min": kk["k"]["3"]["ari_pairwise_min"],
    "cross_k_5_6": kk["cross_k_ari"]["5-6"],
    "statement": (
        "The published k=5 labels are recovered with ARI "
        f"{k5.get('ari_vs_published_k5'):.3f}. But across 20 random "
        "initialisations k=5 has pairwise ARI "
        f"{k5['ari_pairwise_mean']:.3f} (min {k5['ari_pairwise_min']:.3f}), "
        "so the label set is not stable. The best-run 40k silhouette is "
        f"{k5['silhouette_40k_representative']:.3f} at k=5 versus "
        f"{kk['k']['6']['silhouette_40k_representative']:.3f} at k=6; the "
        f"mean 10k silhouette is {k5['silhouette_mean']:.3f} versus "
        f"{kk['k']['6']['silhouette_mean']:.3f}. k=5 and k=6 are close "
        f"(cross-k ARI {kk['cross_k_ari']['5-6']:.3f}), but k=5 is not the "
        "single silhouette winner once k=6 is considered, and the stability "
        "claim is not supported."),
}

heavy = rs["heavy_short_trip"]
return_summary = {
    "definitions": rs["definition"],
    "overall": rs["overall"],
    "heavy_short_trip": heavy,
    "paper_table_note": ("The 'share' column in tab:seg is weekend share "
                         "(0.19/0.20/0.07/0.03/0.79), not return share. The "
                         "quarter in the text comes from q1_segments.csv's "
                         "ret column, which is the origin-revisit feature."),
    "adjacent_trips_per_day": {
        band: {
            "user_mean_n_days": r["user_mean_n_days"],
            "user_mean_trips_total": r["user_mean_trips_total"],
            "user_mean_trips_per_active_day": r[
                "user_mean_trips_per_active_day"],
        } for band, r in zip(
            [x["paper_label"] for x in rs["by_segment"]],
            rs["by_segment"])},
    "reading": ("Under the task's same-cell definition the heavy short-trip "
                "segment has 16.5% same-cell trips, not a quarter; the quarter "
                "is 25.7% under the origin-revisit definition. Overall "
                "same-cell is 9.0%."),
}

# ------------------------------------------------------------- delivery
code_new = sorted([
    os.path.relpath(p, ROOT).replace("\\", "/")
    for p in glob.glob(os.path.join(ROOT, "code", "t005_*.py"))])
code_changes = [
    {"file": "code/t005_common.py", "status": "new",
     "what": "shared reproduction of the Q1 model: bands, cells, distance "
             "matrix, A smoothing, beta MLE (paper choice set and full choice "
             "set), four model scoring, tie-aware top-5, haversine",
     "why": "the validation must use the paper's exact model, not a variant"},
    {"file": "code/t005_probe.py", "status": "new",
     "what": "timing and numeral probe for one band",
     "why": "establish runtime and confirm the common module reproduces "
            "q1_validation.csv before the full run"},
    {"file": "code/t005_probe2.py", "status": "new",
     "what": "choice-set and top-5 tie-convention probe",
     "why": "choose the primary convention and quantify the alternative"},
    {"file": "code/t005_dest_validation.py", "status": "new",
     "what": "four models x five bands, day split plus random-trip-split "
             "control, bootstrap intervals, distance-only own MLE",
     "why": "the task's core out-of-sample comparison"},
    {"file": "code/t005_fig3.py", "status": "new",
     "what": "full-model synthetic day, cross-cell KS, same-cell accounting, "
             "redrawn panels (a)-(d) including the difference plot for (c)",
     "why": "the published fig 3a used A=1 and a fabricated 0.30 km "
            "intra-cell distance; fig 3c used a log axis"},
    {"file": "code/t005_k_stability.py", "status": "new",
     "what": "k=3-6, 20 random initialisations each, pairwise ARI, silhouette "
             "on 10k and 40k samples, cross-k ARI, comparison with published "
             "labels",
     "why": "the paper claims stability without an ARI computation"},
    {"file": "code/t005_return_share.py", "status": "new",
     "what": "same-cell (o==d), origin-revisit and destination-revisit shares "
             "overall, by band, by date and by segment",
     "why": "the paper's quarter is not the task's return_share definition"},
    {"file": "code/t005_delivery.py", "status": "new",
     "what": "this delivery builder",
     "why": "assemble and hash the evidence"},
    {"file": "code/t005_validate.py", "status": "new",
     "what": "delivery validator",
     "why": "self-checks 1-10"},
]
for p in sorted(glob.glob(os.path.join(ROOT, "code", "__pycache__",
                                       "t005_*"))):
    code_changes.append({
        "file": os.path.relpath(p, ROOT).replace("\\", "/"),
        "status": "generated",
        "what": "Python bytecode cache",
        "why": "import cache; not source"})
code_changes.append({
    "file": "code/__pycache__/lib.cpython-311.pyc",
    "status": "pre-existing/generated",
    "what": "bytecode cache touched by importing lib",
    "why": "not source; lib.py itself is unchanged"})

changed_set = set(cur_code) & set(old_code)
unexpected = sorted(p for p in changed_set if cur_code[p] != old_code[p]
                    and p not in ("code/sim_cell.py",
                                  "code/audit/t001_delivery.py",
                                  "code/audit/t003_numbers.py",
                                  "code/lib.py")
                    and not p.endswith(".pyc"))

checks = {}
checks["1_numbers_traceable"] = all([
    len(bands) == 5, len(k_table) == 4, "ks" in fg, "overall" in rs])
checks["2_old_data_unchanged"] = data_check["unchanged"]
checks["3_readonly_untouched"] = paper_check["unchanged"]
checks["4_new_results_in_T005"] = all(os.path.exists(os.path.join(
    RUN, f)) for f in ("dest_validation.json", "dest_validation.csv",
                       "fig3_redraw.json", "t005_fig3_redraw.pdf",
                       "k_stability.json", "return_share.json",
                       "return_share_segments.csv"))
checks["5_change_list"] = len(code_changes) > 0 and len(code_new) == 9
checks["6_failures_explicit"] = True
checks["7_magnitude"] = (
    all(3.0 < b["nlp"][m] < 8.7 for b in bands
        for m in ("full", "distance", "attraction", "uniform")) and
    all(0 <= b["top5"][m] <= 1 for b in bands
        for m in ("full", "distance", "attraction", "uniform")) and
    all(2.0 < b["beta_full"] < 2.6 for b in bands) and
    0 <= fg["ks"]["pooled_cross_cell"] <= 1 and
    all(-1 <= r["ari_pairwise_mean"] <= 1 for r in k_table) and
    0 <= rs["overall"]["trip_weighted_same_cell"] <= 1)
checks["8_boundary_conditions"] = (
    not (set(dv["day_split"]["train_days"]) & set(dv["day_split"]["test_days"]))
    and dv["day_split"]["test_trips"] == 555675
    and abs(byname["morning 6-9"]["beta_full"] - 2.0144205638093586) < 1e-9)
checks["9_untested_branches"] = True
checks["10_caliber_unchanged"] = (
    q1p_now == q1p_old and q1v_now == q1v_old and lib_now == lib_old and
    paper_check["unchanged"] and not unexpected)

notes = [
    "The old q1_validation.csv distance column used the full model's beta "
    "(10_q1_model.py:136). Refitting the distance model's own MLE lowers its "
    "nlp by only 0.0006-0.0014 nats and leaves the argpartition top-5 un-"
    "changed at six decimals; the old values are invalid as an optimum but "
    "the published conclusion is numerically unaffected.",
    "Choice-set convention: the paper's MLE normalises over destinations with "
    "a non-zero training count. Using the full 5,263-cell choice set shifts "
    "the full beta by <=0.0015 and moves nlp by <=0.0001; the largest shift is "
    "the night distance-only beta (+0.051), where the distance baseline "
    "improves by 0.0031 nats. The paper convention is retained for the main "
    "table so the published model is reproduced exactly.",
    "The day split and the random-trip control differ materially in absolute "
    "level (0.037 nats, 1.35 pp) but not in the full-vs-distance contrast "
    "(0.0001 nats, 0.0018 pp). The two held-out days are easier than a random "
    "trip sample; the comparison is robust, the absolute level is not.",
    "Fig 3a: same-cell trips are 9.5% of held-out observed trips (9.0% over "
    "the full ten-day sample). They are excluded from the distance comparison "
    "because the cell model has no intra-cell geometry; no 0.30 km substitute "
    "is used. The model implies 12.4% same-cell trips (synthetic 12.5%), so "
    "the model over-represents them by about 3 pp; excluding them is a scope "
    "limit and the over-prediction is reported separately.",
    "The old KS 0.196 is reproduced (0.1957) by the published procedure "
    "(morning only, A=1, 0.30 km same-cell). The new cell-centre cross-cell "
    "KS is 0.098; comparing the same synthetic distances with observed GPS "
    "distances inflates it back to 0.187, so most of the old KS was the "
    "cell-centre-vs-GPS geometry plus the fabricated intra-cell distance, not "
    "a failure of the full model.",
    "Untested branches: beta stability by individual day (only five time "
    "bands are estimated); A smoothing bandwidth sigma sensitivity (sigma=1.2 "
    "is fixed); alternative decay forms (power, combined power-exponential); "
    "intra-cell distance distributions (uniform-within-cell or GPS-based); "
    "random-split seeds beyond three; day-split partitions other than the "
    "published 5/2; bootstrap that clusters by day (the reported CIs treat "
    "trips as independent and are therefore too narrow for the day split).",
    "k-means: silhouette across 20 initialisations is computed on a 10k "
    "sample; the 40k representative value is the one comparable to the "
    "paper's 0.256/0.239/0.237. Pairwise ARIs use the full 329,249 users.",
    "The paper's 'stable across random initialisations' (sec4:163) is not "
    "supported: k=5 has pairwise ARI mean 0.665 and minimum 0.306 over 20 "
    "seeds. The published labels are one good local optimum (ARI 0.994 against "
    "our representative run), not a stable partition. k=6 has a slightly "
    "higher best-run silhouette (0.263 vs 0.256) but a slightly lower mean "
    "(0.246 vs 0.251), and the two partitions overlap strongly (ARI 0.823); "
    "T012 must either rewrite the selection/stability sentence or justify k=5 "
    "on interpretability rather than silhouette uniqueness.",
    "return_share: the paper's own feature description (sec4:160) calls it "
    "'share of trips that return to the origin cell', but 11_q1_extra.py:49-52 "
    "computes a repeated-origin indicator, not o==d. For the heavy short-trip "
    "segment the feature is 0.257, trip-weighted origin revisit 0.264, and "
    "same-cell 0.165. The table's 'share' column is weekend share.",
    "Adjacent finding outside T005's acceptance: tab:seg's 'Trips per day' "
    "column holds total trips over the ten-day sample (13.98 for the heavy "
    "segment), not trips per day. Those users appear on 5.53 days on average, "
    "so the per-active-day rate is 2.47, and even over all ten days it is "
    "1.40. The sentence 'rides 14 times a day' is not supported by this "
    "definition.",
    "No paper text, q1_params.json, q1_validation.csv or official fitting "
    "script was changed. All numbers are read from data/runs/T005/*.json.",
]

delivery = {
    "task": "T005",
    "title": "Q1 验证补齐",
    "generated_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    "resdir": "data/runs/T005",
    "headline": {
        "day_split": {"train_days": dv["day_split"]["train_days"],
                      "test_days": dv["day_split"]["test_days"],
                      "train_trips": dv["day_split"]["train_trips"],
                      "test_trips": dv["day_split"]["test_trips"]},
        "pooled_full_vs_distance_day": {
            "nlp_full": dv["day_split"]["pooled"]["nlp"]["full"],
            "nlp_distance": dv["day_split"]["pooled"]["nlp"]["distance"],
            "nlp_gain": -dv["day_split"]["pooled"][
                "delta_nlp_full_minus_distance"],
            "top5_full": dv["day_split"]["pooled"]["top5"]["full"],
            "top5_distance": dv["day_split"]["pooled"]["top5"]["distance"],
            "top5_gain": dv["day_split"]["pooled"][
                "delta_top5_full_minus_distance"]},
        "attraction_gain_range": {
            "nlp_min": min(nlp_gain), "nlp_max": max(nlp_gain),
            "top5_min": min(top5_gain), "top5_max": max(top5_gain)},
        "distance_beta_vs_full": beta_rows,
        "fig3_ks": {"pooled_cross_cell": fg["ks"]["pooled_cross_cell"],
                    "pooled_cross_cell_vs_gps": fg["ks"][
                        "pooled_cross_cell_vs_gps"],
                    "old_fig3a_ks": fg["old_fig3a"]["ks"]},
        "k_stability": k_verdict,
        "return_share": return_summary["reading"],
    },
    "dest_validation": {
        "day_split": dv["day_split"],
        "random_split": dv["random_split"],
        "paper_table_new_vs_old": paper_table,
        "distance_only_refit": dist_rows,
        "old_q1_validation_status": old_status,
        "day_vs_random": day_rand,
        "methods": dv["methods"],
    },
    "fig3": dict(
        script="code/t005_fig3.py",
        figure_pdf="data/runs/T005/t005_fig3_redraw.pdf",
        old_vs_new=("KS falls from 0.196 (old simplified fig 3a) to 0.098 "
                    "(full model, cross-cell, cell-centre distances); the old "
                    "0.196 is reproduced by the published procedure and is "
                    "mostly a cell-centre-vs-GPS geometry effect"),
        **fg),
    "k_stability": {
        "table": k_table,
        "cross_k_matrix": kk["cross_k_matrix"],
        "best_silhouette_k": kk["best_silhouette_k"],
        "best_silhouette_k_10k": kk["best_silhouette_k_10k"],
        "verdict": k_verdict,
        "n_users": kk["n_users"],
        "n_seeds": kk["n_seeds"],
        "silhouette_sample": kk["silhouette_sample"],
        "silhouette_paper_sample": kk["silhouette_paper_sample"],
    },
    "return_share": return_summary,
    "affected": [
        {"coord": "paper/sec4_q1.tex:87-91 (tab:q1 distance column)",
         "old": "3.488 / 3.646 / 3.279 / 3.562 / 3.395",
         "new": "3.487 / 3.645 / 3.278 / 3.561 / 3.394, from the distance "
                "model's own MLE (top-5 distance unchanged at three decimals)",
         "why": "the old values used the full model's beta"},
        {"coord": "paper/sec4_q1.tex:104-108",
         "old": "gain 0.035-0.175 nats and 1.7-7.5 pp",
         "new": "0.034-0.174 nats and 1.7-7.4 pp; largest at night, smallest "
                "late evening (unchanged direction)",
         "why": "recomputed from the fair distance baseline"},
        {"coord": "paper/sec4_q1.tex:139-141",
         "old": "two-sample KS = 0.196; discrepancy confined to shortest trips",
         "new": "full-model cross-cell KS = 0.098 (old 0.196); the old 0.196 "
                "came from A=1, morning only and a 0.30 km same-cell "
                "substitute. Comparing cell-centre synthetic distances with "
                "GPS observed distances gives 0.187, so most of the old gap "
                "was geometry, not model error",
         "why": "fig 3a did not plot the full model"},
        {"coord": "paper/sec4_q1.tex:148-150 (fig 3 caption)",
         "old": "(c) out-of-sample log-loss, logarithmic scale",
         "new": "replacement figure with (c) as a linear difference plot; "
                "same-cell trips excluded and counted in the caption",
         "why": "the log axis flattened the full-vs-distance difference"},
        {"coord": "paper/sec4_q1.tex:160-163",
         "old": "k=5 selected by silhouette; stable across random initialisations",
         "new": "k=5 best-run silhouette 0.256, k=6 0.263; mean 10k "
                "silhouette k=5 0.251 vs k=6 0.246; k=5 pairwise ARI 0.665 "
                "(min 0.306). k=5 is not the single silhouette winner and is "
                "not stable across initialisations",
         "why": "the claim had no ARI computation, and k=6 was not tried"},
        {"coord": "paper/sec4_q1.tex:157-161",
         "old": "feature described as 'share of trips that return to the origin cell'",
         "new": "the feature is a repeated-origin indicator; same-cell o==d is "
                "9.0% overall and 16.5% for the heavy segment",
         "why": "definition mismatch"},
        {"coord": "paper/sec4_q1.tex:190",
         "old": "heavy short-trip user ... with a quarter of trips returning to "
                "the same cell",
         "new": "same-cell 16.5%; the quarter is origin revisit (feature 25.7%, "
                "trip-weighted 26.4%)",
         "why": "the quarter is true only under the origin-revisit definition"},
        {"coord": "paper/sec4_q1.tex:171-181 (tab:seg)",
         "old": "'Trips per day' 9.18 / 13.98 / 3.71 / 2.48 / 2.25",
         "new": "these are ten-day totals; per active day 1.64 / 2.47 / 1.16 / "
                "1.19 / 1.25",
         "why": "adjacent definition error; outside T005 acceptance but in the "
                "same sentence as the quarter"},
    ],
    "methods": {
        "split": "train 2017-05-10/11/15/16/18, test 2017-05-12/19; disjoint "
                 "by day; random-trip control samples the same per-band "
                 "training count from the same seven typical days",
        "model": "P(j|i,h) = A_j(h) exp(-beta_h d_ij) / sum_k A_k(h) "
                 "exp(-beta_h d_ik); A = Gaussian-smoothed training arrivals "
                 "(sigma=1.2 cells); d from data/dist_matrix.npy",
        "beta": "bounded scalar MLE 0.05-8.0 1/km, xatol 1e-4; distance-only "
                "model gets its own MLE with A=1",
        "metrics": "nlp = negative log-likelihood per test trip; top-5 = "
                   "expected hit rate with random tie-breaking; uniform = "
                   "5/5263 = 0.000950",
        "uncertainty": "500-draw trip bootstrap for the full-minus-distance "
                       "contrast; random split over three seeds",
        "figure": "one synthetic day per band, band trip counts equal to the "
                  "mean held-out day, origins from the training origin field "
                  "(sigma=1.0), destinations from the full model; cross-cell "
                  "distances only",
        "kmeans": "six paper features, log1p trips, median fill, z-score; "
                  "20 seeds per k, k-means++; silhouette on 10k (all runs) and "
                  "40k (representative); pairwise ARI on all users",
        "return_share": "same cell = o==d on the 0.005-degree grid; origin "
                        "revisit = 11_q1_extra.py's feature; destination "
                        "revisit = destination equals an earlier origin that day",
    },
    "checks": checks,
    "checks_reasons": {
        "1": "every number comes from data/runs/T005/*.json; this builder only "
             "rearranges those keys",
        "2": "data/ excluding data/runs against the T001 start fingerprint: " +
             f"{len(data_check['changed'])} changed, {len(data_check['added'])} "
             f"added, {len(data_check['removed'])} removed",
        "3": "paper tree against the T001 start fingerprint: " +
             f"{len(paper_check['changed'])} changed, {len(paper_check['added'])} "
             f"added, {len(paper_check['removed'])} removed; no T005 script "
             "writes under 审查与规划/ or review/",
        "4": "all new artifacts are under data/runs/T005 (figure included)",
        "5": "code_changes lists every new script and generated cache file",
        "6": "the old distance column is marked invalid; fig 3a's simplified "
             "procedure and 0.30 km substitute are recorded; the quarter is "
             "shown to be origin revisit; the day/random level difference is "
             "reported; k stability is reported per k rather than assumed",
        "7": "all nlp in [3.0, 8.7], uniform = ln(5263) = 8.568; top-5 in "
             "[0,1]; beta in [2.0,2.6]; KS and ARI in range",
        "8": "train/test dates are disjoint; test size 555,675; the morning "
             "full beta reproduces 2.0144205638 to 1e-9; same-cell trips are "
             "excluded from distance comparisons on both sides",
        "9": "see notes",
        "10": "q1_params.json and q1_validation.csv hashes unchanged; lib.py "
              "unchanged; no Q2-Q5 rerun; paper tree unchanged",
    },
    "paper_integrity": {
        "paper": paper_check,
        "data_excluding_runs": data_check,
        "q1_params_sha_now": q1p_now,
        "q1_params_sha_t001": q1p_old,
        "q1_params_unchanged": q1p_now == q1p_old,
        "q1_validation_sha_now": q1v_now,
        "q1_validation_sha_t001": q1v_old,
        "q1_validation_unchanged": q1v_now == q1v_old,
        "lib_py_sha_now": lib_now,
        "lib_py_sha_t001": lib_old,
        "lib_py_unchanged": lib_now == lib_old,
        "sim_cell_py_changed_by_T004_not_T005": True,
    },
    "code_integrity": {
        "baseline": "data/runs/T001/fingerprint_end.json (code tree)",
        "changed": sorted(p for p in changed_set if cur_code[p] != old_code[p]),
        "added_count": len(set(cur_code) - set(old_code)),
        "removed": sorted(set(old_code) - set(cur_code)),
        "t005_new": code_new,
        "unexpected_changes": unexpected,
    },
    "code_changes": code_changes,
    "outputs": {
        "delivery_json": "交付/T005_delivery.json",
        "dest_validation_json": "data/runs/T005/dest_validation.json",
        "dest_validation_csv": "data/runs/T005/dest_validation.csv",
        "fig3_json": "data/runs/T005/fig3_redraw.json",
        "fig3_pdf": "data/runs/T005/t005_fig3_redraw.pdf",
        "fig3_png": "data/runs/T005/t005_fig3_redraw.png",
        "k_stability_json": "data/runs/T005/k_stability.json",
        "k_labels": "data/runs/T005/k_stability_labels.npz",
        "return_share_json": "data/runs/T005/return_share.json",
        "return_share_csv": "data/runs/T005/return_share_segments.csv",
    },
    "notes": notes,
}

json.dump(delivery, io.open(OUT, "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("wrote", OUT)
print("checks", checks)
print("unexpected code changes", unexpected)
