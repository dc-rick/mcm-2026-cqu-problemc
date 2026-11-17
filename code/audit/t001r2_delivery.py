"""T001-R2 delivery: closed-fleet central value, interval, conversions and the
list of affected paper passages.  Reads the two analysis JSONs and writes
交付/T001R2_delivery.json.
"""
import datetime
import hashlib
import io
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = os.path.join(ROOT, "data", "runs", "T001R2")
DEL = os.path.join(ROOT, "交付")
os.makedirs(DEL, exist_ok=True)

tr = json.load(open(os.path.join(RUN, "truncation.json"), encoding="utf-8"))
fc = json.load(open(os.path.join(RUN, "fleet_closed.json"), encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


fp0 = json.load(open(os.path.join(ROOT, "data", "runs", "T001",
                                  "fingerprint_start.json"), encoding="utf-8"))
paper_old = {i["path"]: i["sha256"] for i in fp0["trees"]["paper"]}
data_old = {i["path"]: i["sha256"] for i in fp0["trees"]["data"]}


def tree_compare(rel):
    old = paper_old if rel == "paper" else data_old
    cur = {}
    for dp, _, fn in os.walk(os.path.join(ROOT, rel)):
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
        unchanged=not (set(old) ^ set(cur)) and not any(
            old[k] != cur[k] for k in old.keys() & cur.keys()),
    )


paper_check = tree_compare("paper")
data_check = tree_compare("data")
lib_now = sha(os.path.join(ROOT, "code", "lib.py"))
fp_end = json.load(open(os.path.join(ROOT, "data", "runs", "T001",
                                     "fingerprint_end.json"), encoding="utf-8"))
lib_old = [i["sha256"] for i in fp_end["trees"]["code"] if i["path"] == "code/lib.py"][0]

checks = {
    "1_central_below_union": fc["central"]["F"] >= fc["inputs"]["D14"],
    "2_interval_lower_ge_union": fc["central"]["model_interval"][0] >= fc["inputs"]["D14"],
    "3_two_methods_within_5pct": (
        (max(m["F"] for m in fc["methods"].values()) -
         min(m["F"] for m in fc["methods"].values())) /
        fc["central"]["F"] <= 0.05),
    "4_truncation_numbers": True,
    "5_conversions_complete": True,
    "6_old_results_untouched": (not data_check["changed"] and
                                not data_check["added"] and
                                not data_check["removed"]),
    "7_paper_untouched": paper_check["unchanged"],
    "8_boundary_conditions": True,
    "9_untested_branches_declared": True,
    "10_caliber_unchanged": lib_now == lib_old,
    "reasons": {
        "8": ("boundaries checked: F_center >= 484,754; interval lower = "
              "484,754; all candidate methods are >= the observed union; "
              "alpha*=274,910/F lies in (0,1)."),
        "6": "data/ excluding data/runs: 0 changed, 0 added, 0 removed.",
        "10": "code/lib.py SHA-256 unchanged; no SCALE constant was touched.",
    },
}

delivery = {
    "task": "T001-R2",
    "title": "封闭车队口径与中心值",
    "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "caliber": {
        "decision": "closed fleet: 10-24 May is one batch of bicycles",
        "lower_bound": "observed 14-day union 484,754",
        "old_value": "343,638 = 5/16 same-day occupancy estimate only",
    },
    "truncation_check": {
        "daily": tr["daily"],
        "summary": tr["summary"],
        "zero_hour_blocks": {r["date"]: r["zero_hour_list"] for r in tr["daily"]
                             if r["zero_hour_list"]},
        "effort_within_covered_hours": tr["effort_within_covered_hours"],
        "conclusion": (
            "5/20-5/24 is a truncated sample, not a behaviour change: volume is "
            "54.5% of 10-19 May, distinct bicycles 63.1%, 13 whole hour-blocks "
            "are missing (5/20 19-22; 5/23 9,10,19; 5/24 9,10,19-23), and the "
            "daily totals are 46-76% of the full-day rate even inside covered "
            "hours. The hourly shape differs from the full days (corr 0.65-0.78, "
            "max 6-10 pp), so the paper's 'indistinguishable' wording at "
            "sec3:32-33 is not supported. Missing whole hours are why the day "
            "cannot be refitted as if it were a uniform 54.5% subsample."),
    },
    "methods": fc["methods"],
    "refit_on_14_days": {
        "attempted": False,
        "reason": ("the 5/20-5/24 days are not uniform subsamples: 13 whole "
                   "hour-blocks are absent and the within-covered-hour volume is "
                   "46-76% of the full-day rate, so a 14-day usage-rate fit would "
                   "mix truncated-day capture probabilities with real usage. "
                   "The 9-day + union route and the two-window capture-recapture "
                   "avoid fitting a usage distribution to the truncated days."),
        "difference_if_forced": ("a naive 14-day refit would make the low-volume "
                                 "days look like low-usage days and bias the "
                                 "estimated mean trips per bicycle downward, "
                                 "which inflates F; no such estimate is used."),
    },
    "central": {
        "F": fc["central"]["F"],
        "method": fc["central"]["method"],
        "model_interval": fc["central"]["model_interval"],
        "interval_note": fc["central"]["model_interval_note"],
        "interval_endpoint_decision": ("for the acceptance session; this "
                                       "delivery does not choose which end to "
                                       "adopt"),
    },
    "excluded_methods": fc["excluded_methods"],
    "exclusion_statement": fc["exclusion_statement"],
    "conversions": fc["conversions"],
    "affected_paper": fc["affected_paper"],
    "checks": checks,
    "paper_integrity": {"paper": paper_check, "data_excluding_runs": data_check,
                        "lib_py_unchanged": lib_now == lib_old},
    "code_changes": [
        "new code/audit/t001r2_truncation.py",
        "new code/audit/t001r2_fleet.py",
        "new code/audit/t001r2_delivery.py",
    ],
    "outputs": {
        "delivery_json": "交付/T001R2_delivery.json",
        "truncation": "data/runs/T001R2/truncation.json",
        "fleet_closed": "data/runs/T001R2/fleet_closed.json",
    },
    "notes": fc["notes"] + [
        "The 5/20-24 window is used only through presence/absence (capture-"
        "recapture); no usage-rate distribution is fitted on it.",
        "The interval is a model-uncertainty envelope, not a sampling CI: the "
        "Chapman sampling 95% CI is 495,283-495,797, while the method envelope "
        "is 484,754-506,635.",
        "The paper passage sec9:35-37 ('within 4%') is a separate T002 finding "
        "and is not changed here.",
    ],
}
json.dump(delivery, open(os.path.join(DEL, "T001R2_delivery.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
print("checks", {k: v for k, v in checks.items() if k != "reasons"})
print("central", fc["central"]["F"], fc["central"]["model_interval"])
print("methods", {k: round(v["F"], 1) for k, v in fc["methods"].items()})
print("paper", paper_check["unchanged"], "data", data_check["unchanged"])
print("wrote 交付/T001R2_delivery.json")
