"""Assemble 交付/T001_delivery.json from the frozen result files."""
import json
import os
import sys
import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data", "runs", "T001")
DEL = os.path.join(ROOT, "交付")
os.makedirs(DEL, exist_ok=True)

fe = json.load(open(os.path.join(OUT, "fleet_estimate.json"), encoding="utf-8"))
imp = json.load(open(os.path.join(OUT, "impact_analysis.json"), encoding="utf-8"))
fp0 = json.load(open(os.path.join(OUT, "fingerprint_start.json"), encoding="utf-8"))
fp1 = json.load(open(os.path.join(OUT, "fingerprint_end.json"), encoding="utf-8"))
fd = json.load(open(os.path.join(OUT, "fingerprint_diff.json"), encoding="utf-8"))

FOLD = fe["old_constant"]
F_LB = float(fe["observed_lower_bound_9d"])
BB = fe["beta_binomial"]["N"]
D = imp["diagnostics"]
scale = imp["scale"]

lib_before = [i for i in fp0["trees"]["code"] if i["path"] == "code/lib.py"][0]
lib_after = [i for i in fp1["trees"]["code"] if i["path"] == "code/lib.py"][0]


def delta(x):
    return x - FOLD


cases = []
for m in fe["methods"]:
    cases.append(dict(
        method=m["method"], point=float(m["point"]),
        lo=None if m["lo"] in (None, "") else float(m["lo"]),
        hi=None if m["hi"] in (None, "") else float(m["hi"]),
        delta_vs_old=float(m["point"]) - FOLD,
        ratio_vs_old=float(m["point"]) / FOLD,
        note=m["note"],
    ))
cases.append(dict(method="0: exact reproduction of the paper constant",
                  point=fe["reproduction"]["fleet"],
                  lo=fe["reproduction"]["fleet"], hi=fe["reproduction"]["fleet"],
                  delta_vs_old=fe["reproduction"]["delta_vs_constant"],
                  ratio_vs_old=fe["reproduction"]["fleet"] / FOLD,
                  note=("F=T/mu on 2017-05-16 under 02_recon_spatial.py's crude "
                        "filter; reproduces 343,637.986 -> 343,638 (|delta|=0.014)")))

methods = {
    "reproduction_343638": dict(
        formula=("single day: T=trips, D=distinct bikes; solve mu/(1-exp(-mu))=T/D; "
                 "F=T/mu. Filter: start/end lat>35, lon>110 (02_recon_spatial.py). "
                 "Day 2017-05-16: T=288,691, D=195,301, T/D=1.478185, mu=0.840102, "
                 "F=343,637.986 -> 343,638"),
        units="bicycles"),
    "lib_filter_difference": dict(
        formula=("lib.load_trips keeps only starts in 39.4<lat<40.4, "
                 "115.8<lon<117.0; the crude filter keeps 12 fringe records "
                 "(lat 38.03-40.17, lon 114.5-117.4) on 5/16: T 288,679->288,691, "
                 "D 195,289->195,301, F 343,604.83->343,637.99 (gap 33.16)"),
        units="bicycles"),
    "observed_lower_bound": dict(
        formula="|{bikeid}| over the window under lib.load_trips filter",
        units="bicycles"),
    "chapman": dict(
        formula=("N=(n1+1)(n2+1)/(m+1)-1; SE^2=(n1+1)(n2+1)(n1-m)(n2-m)/"
                 "((m+1)^2(m+2)); 95% CI = N +/- 1.96 SE"),
        units="bicycles"),
    "chao1": dict(
        formula=("Nhat = S + f1^2/(2 f2), S=distinct observed, f1=seen in exactly one "
                 "period, f2=seen in both; log-normal 95% CI on the undetected part"),
        units="bicycles"),
    "beta_binomial": dict(
        formula=("each bike's daily appearance probability q ~ Beta(a,b); over 9 days "
                 "k~BetaBin(9,a,b); fit zero-truncated MLE (Nelder-Mead in log a,b); "
                 "P0=B(a,9+b)/B(a,b); N=D/(1-P0); 95% CI = percentile bootstrap "
                 "(600 resamples of the day-count histogram)"),
        units="bicycles"),
    "homogeneous_9day": dict(
        formula=("paper's estimator on the 9-day window: solve m/(1-exp(-m))=T9/D9 "
                 "= 5.3278 -> m=5.301 (expected sampled trips per bike over 9 days); "
                 "F=T9/m"),
        units="bicycles"),
    "zt_nb": dict(
        formula=("trips per bike ~ zero-truncated negative binomial "
                 "P(n)=Gamma(n+r)/(Gamma(r)n!) q^r (1-q)^n; MLE in (r,q); "
                 "P0=q^r; N=D/(1-P0); 95% CI = percentile bootstrap"),
        units="bicycles"),
    "rides_per_bike": dict(
        formula="scaled trips per day / fleet = (274,357/0.128)/F",
        units="rides per bicycle per day"),
    "right_sizing": dict(
        formula=("LP initial inventory is share*(alpha*FLEET), i.e. it depends on "
                 "physical P=alpha*FLEET; total(P)=1.0*P+obj(P); P*=274,910 at old "
                 "alpha 0.8, total 390k CNY; alpha*=P*/F under a new constant"),
        units="bicycles, CNY/day"),
}

affected = [
    dict(location="code/20_q2_diag.py:11 and :26",
         quantity="I0 = init_share * FLEET (initial inventory of every zone)",
         relation="linear in FLEET",
         multiplier_at_lower_bound=float(F_LB / FOLD),
         multiplier_at_beta_binomial=float(BB / FOLD),
         detail=("sum I0 goes 343,638 -> 462,893 (LB) / 483,787 (BB). Diagnostics in "
                 "memory: zones ever negative stays 0; min inventory/initial goes "
                 "0.928 -> 0.946 (LB) / 0.949 (BB); stockout bike-hours stays 0.")),
    dict(location="code/21_q2_opt.py:19, :30, :58",
         quantity="I0_base = RQ['I0']/FLEET, then I0 = I0_base*FLEET*alpha",
         relation="cancels inside the file; depends on q2_state.npz, not on the literal",
         multiplier_at_lower_bound="invariant as written",
         multiplier_at_beta_binomial="invariant as written",
         detail=("If q2_state.npz is regenerated with a new constant, I0 scales by "
                 "Fnew/Fold and the LP must be re-solved. At alpha=1 unmet=0 in every "
                 "solved row, so a larger I0 cannot change the alpha=1 solution; at "
                 "alpha<1 unmet>0, so those rows do move.")),
    dict(location="code/22_q2_sim.py:15-16",
         quantity="I0_base = st['I0'] (absolute bicycles) driving the Q2 simulation",
         relation="linear in FLEET through q2_state.npz",
         multiplier_at_lower_bound=float(F_LB / FOLD),
         multiplier_at_beta_binomial=float(BB / FOLD),
         detail=("the simulated service level shifts with I0; direction: a larger "
                 "fleet gives fewer unmet departures, so the 'no shortage' conclusion "
                 "is strengthened, not weakened")),
    dict(location="code/50_q5_platform.py:14-15",
         quantity="I0 = st['I0'] in the Q5 split-fleet experiment",
         relation="linear in FLEET through q2_state.npz",
         multiplier_at_lower_bound=float(F_LB / FOLD),
         multiplier_at_beta_binomial=float(BB / FOLD),
         detail=("the same inventory level is split across platforms; its scale moves "
                 "with the constant, so the 'zero unmet in 24 runs' result is a function "
                 "of the chosen fleet")),
    dict(location="code/95_numbers.py:49-50",
         quantity="I0 quantiles and sum printed for the paper",
         relation="linear in FLEET through q2_state.npz",
         multiplier_at_lower_bound=float(F_LB / FOLD),
         multiplier_at_beta_binomial=float(BB / FOLD),
         detail=("these are the paper's inventory levels in Sec 5.1; with a new "
                 "constant the quantiles scale linearly (1,986 -> 2,675 at the LB, "
                 "2,796 at the beta-binomial; 6,912 -> 9,311 / 9,731)")),
    dict(location="code/24_cell_curve.py:9 and :36",
         quantity="I0c = (last-end share) * FLEET (cell initial inventory)",
         relation="linear in FLEET; the unmet curve is a function of P=alpha*FLEET",
         multiplier_at_lower_bound=float(F_LB / FOLD),
         multiplier_at_beta_binomial=float(BB / FOLD),
         detail=("the curve does not move once plotted against physical inventory P; "
                 "only the alpha labels change: alpha_new = alpha_old*343,638/Fnew")),
    dict(location="code/30_q3_incentive.py:16",
         quantity="FLEET defined but never used (dead constant)",
         relation="no dependency",
         multiplier_at_lower_bound=1.0,
         multiplier_at_beta_binomial=1.0,
         detail="grep shows the name appears only on line 16; Q3 results do not depend on it"),
    dict(location="code/90_figs.py:256 and :258 (two literal 343638, not a reference)",
         quantity="panel (d) total cost = 1.0*343638*alpha + LP objective",
         relation="capital term linear in the literal; optimum in physical terms invariant",
         multiplier_at_lower_bound=float(F_LB / FOLD),
         multiplier_at_beta_binomial=float(BB / FOLD),
         detail=("P*=274,910 and 390k CNY/day do not change; the alpha label of the "
                 "optimum moves 0.800 -> 0.594 (LB) / 0.568 (BB). Editing the literal "
                 "without SAVE-ING the axis label gives a wrong figure.")),
    dict(location="code/40_q4_efence.py (no FLEET at all)",
         quantity="fence capacity is built from scaled arrivals only",
         relation="no code dependency",
         multiplier_at_lower_bound=1.0,
         multiplier_at_beta_binomial=1.0,
         detail=("paper only: sec7_q4.tex:88 compares 344,463 slots with the fleet; "
                 "slots/fleet goes 1.0024 -> 0.7442 (LB) / 0.7120 (BB), so the "
                 "'0.2% margin / not a coincidence' claim fails even harder")),
]

paper_hits = [
    dict(file="paper/sec3_data.tex", lines=[85, 96, 97, 139, 140],
         numbers=["343,638", "6.2", "3.4e5"], effect="fleet estimate and rides/bike"),
    dict(file="paper/summary.tex", lines=[24, 49],
         numbers=["3.4e5", "6.2", "427k", "390k"], effect="abstract fleet + right-sizing"),
    dict(file="paper/sec5_q2.tex", lines=[17, 18, 19],
         numbers=["1,986", "6,912", "-349"],
         effect=("1,986 and 6,912 are I0 min/median and scale linearly with the constant "
                 "(verified: 1,986 = min init_share*343,638); -349 is the flow-only "
                 "excursion min(cum) and does NOT move with the fleet")),
    dict(file="paper/sec7_q4.tex", line=88,
         numbers=["344,463", "343,638"], effect="slots-vs-fleet comparison"),
    dict(file="paper/sec9_sensitivity.tex", lines=[103, 123],
         numbers=["343,638", "6.2"], effect="sensitivity narrative"),
    dict(file="figs/fig4_q2.pdf (panel d)",
         numbers=["343638 literal", "optimum label 390k"],
         effect="axis is relative to the constant; relabel needed if constant changes"),
]

checks = {
    "1_number_traceable": True,
    "2_old_results_untouched": True,
    "3_readonly_trees_untouched": True,
    "4_new_results_in_runs_dir": True,
    "5_change_list_complete": True,
    "6_failures_explicit": True,
    "7_magnitude_sanity": True,
    "8_boundary_conditions": "n/a",
    "9_untested_branches_declared": True,
    "10_caliber_unchanged": True,
    "reasons": {
        "8": ("no optimisation or ODE is solved in T001; residuals are not defined. "
              "The only numerical invariants relevant here (F >= observed union, I0 >= 0) "
              "are checked in check 7."),
        "2": "fingerprint_diff.json: data/ excluding runs/ has 0 added, 0 removed, 0 modified",
        "3": ("paper/, review/ and data/ (excluding runs/) have 0 added, 0 removed, "
              "0 modified. 审查与规划/ changed during the run, but only in files the "
              "peer session writes (its acceptance record and two ledgers); see "
              "fingerprints.peer_workspace_delta. The executor wrote nothing there."),
    },
}

out = {
    "task": "T001",
    "title": "车队规模审计与重估（R1 已补）",
    "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "resdir": "data/runs/T001",
    "checks": checks,
    "fingerprints": {
        "start": "data/runs/T001/fingerprint_start.json",
        "end": "data/runs/T001/fingerprint_end.json",
        "diff": "data/runs/T001/fingerprint_diff.json",
        "lib_py_sha_before": lib_before["sha256"],
        "lib_py_sha_after": lib_after["sha256"],
        "code_added": fd["trees"]["code"]["added"],
        "code_modified": fd["trees"]["code"]["modified"],
        "readonly_unchanged": all(
            not fd["trees"][t]["added"] and not fd["trees"][t]["removed"]
            and not fd["trees"][t]["modified"]
            for t in ("data", "paper", "review")),
        "peer_workspace_delta": (
            fd["trees"]["审查与规划"]["added"]
            + fd["trees"]["审查与规划"]["modified"]),
        "peer_workspace_note": (
            "审查与规划/ changed while T001 ran, but every changed path is a file the "
            "peer session maintains (its own T001 acceptance record and two ledgers). "
            "The executor wrote nothing there; all executor outputs live in "
            "data/runs/T001/ and 交付/."),
    },
    "methods": methods,
    "cases": cases,
    "affected": affected,
    "affected_paper": paper_hits,
    "evidence": dict(
        reproduction=fe["reproduction"],
        per_day=fe["per_day"],
        gof_days_seen=fe["gof_days_seen"],
        capture_recapture=fe["capture_recapture"],
        beta_binomial=fe["beta_binomial"],
        homogeneous_9day=fe["homogeneous_9day"],
        nb=fe["nb"],
        diagnostics=D,
        scale=scale,
        figure="data/runs/T001/fig_fleet_estimates.png",
        tables=["data/runs/T001/methods.csv",
                "data/runs/T001/diagnostics_by_fleet.csv",
                "data/runs/T001/scaling_table.csv",
                "data/runs/T001/gof_days_seen.csv",
                "data/runs/T001/capture_recapture.csv",
                "data/runs/T001/per_day_estimates.csv"],
    ),
    "notes": [
        ("343,638 IS reproducible to 0.014 bicycles, but only for the single day "
         "2017-05-16 under 02_recon_spatial.py's crude filter. The paper calls it "
         "'the fleet' and quotes 6.2 rides per bike per day from it. The comment "
         "20_q2_diag.py:11 is technically true (it is a Poisson-occupancy estimate) "
         "but does not say 'single day'; the constant and the comment are jointly "
         "misleading about the caliber."),
        ("The 33-bicycle gap to the lib-filter value is fully accounted for: 12 fringe "
         "records (lat 38.03-40.17, lon 114.5-117.4, all outside the Beijing bbox) on "
         "5/16, 12 distinct bikes, T +12 and D +12 -> F +33.16."),
        ("All closed-fleet methods give >= 462,893. Point estimates: lower bound 462,893 "
         "(9 d) / 484,754 (14 d); Chapman 470,478-477,565; Chao1 479,222-506,693; "
         "zero-truncated beta-binomial 483,787 [483,506, 484,053]; zero-truncated NB "
         "478,337 [478,152, 478,543]; the paper's own homogeneous model corrected to "
         "9 days gives 465,212."),
        ("The bootstrap/parametric intervals are sampling-only and are tiny because "
         "n is 4.6e5. The real uncertainty is model specification: the methods span "
         "465k-507k. Do not quote 483,787 [483.5k, 484.1k] as if the interval covered "
         "model uncertainty."),
        ("GOF rejects homogeneity: observed share seen exactly 1 day = 13.29%; the "
         "paper's one-day homogeneous model extended to 9 days predicts 0.62%; the "
         "homogeneous model corrected to the 9-day window predicts 3.62%; the "
         "beta-binomial predicts 12.06%."),
        ("Assumptions behind the estimates: closed population over 5/10-5/19; daily "
         "usage iid within a bike given its rate; the order sampling rate p is "
         "irrelevant to all estimators (it only rescales the per-bike rate axis). "
         "Capture-recapture also assumes equal catchability across periods; "
         "heterogeneity biases Lincoln-Petersen downward and Chao1 is a lower-bound "
         "correction."),
        ("Open-fleet alternative (decision item B6, NOT decided here): if Mobike was "
         "still deploying during the window, the 9/14-day union is not a lower bound "
         "for any single-day fleet, and 343,638 could be a legitimate 5/16 snapshot. "
         "Then the errors are the wording ('the fleet'), the 6.2 rides/bike/day "
         "inference, and the direct comparison with 484,754 distinct bikes. This "
         "needs the user: choose closed-fleet interval vs single-day snapshot + "
         "growth narrative."),
        ("Impact: the state variable is the PHYSICAL fleet P=alpha*FLEET, not the "
         "constant by itself. q2_state.npz (written by 20_q2_diag.py:55, where "
         "I0=init_share*FLEET at :26) carries I0 in absolute bicycles and is read by "
         "22_q2_sim.py:15-16 (Q2 simulation), 50_q5_platform.py:14-15 (Q5 split-fleet "
         "experiment) and 95_numbers.py:49-50 (I0 quantiles quoted in the paper). So "
         "the constant DOES reach Q2, Q5 and the paper numbers through that file. What "
         "is invariant is the right-sizing curve as a function of P=alpha*FLEET: "
         "P*=274,910 and 390k CNY/day stay, while the I0 levels scale by Fnew/Fold, the "
         "alpha labels rescale (0.800 -> 0.568 at the beta-binomial), and 6.2 -> 4.43 "
         "rides/bike/day. At alpha=1 unmet=0 in every solved row, so a larger I0 does "
         "not change the alpha=1 LP solution; at alpha<1 unmet>0 the rows move."),
        ("90_figs.py:256/258 hold the literal 343638, not the constant - a second "
         "failure mode: a future edit of FLEET alone leaves the figure stale. It is "
         "listed separately in affected[] for that reason."),
        ("Only existing file changed: code/lib.py lines 8-9 (DATA/FIG) plus 'import os'; "
         "absolute D:\\code\\mcm paths replaced by paths derived from __file__. "
         "code/__pycache__/lib.cpython-311.pyc changed as a side effect of importing "
         "lib; no source semantics changed."),
        ("Not tested: time-varying fleet models (growth curve), day-level capture "
         "occasions with Mh/jackknife estimators, beta-binomial with temporal "
         "correlation, and any re-solve of Q2-Q5. The Q2 invariance statement at "
         "alpha=1 is read off q2_results.json (unmet=0), not re-solved."),
    ],
    "open_decisions": [dict(
        id="B6",
        raised="2026-11-17",
        issue=("Is the fleet a closed population over 5/10-5/19 (then F >= 462,893, "
               "central ~484k, and 343,638 is invalid), or an open/growing population "
               "(then 343,638 may be a 5/16 snapshot and the paper's error is wording "
               "plus the 6.2 rides/bike/day claim)? T001 does not decide this."),
        options=["closed-fleet: adopt ~465k-507k with explicit interval",
                 "open-fleet: keep a single-day estimate but relabel and bound it",
                 "report both in the paper, one as main and one as scenario"],
    )],
}

path = os.path.join(DEL, "T001_delivery.json")
with open(path, "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
print("wrote", os.path.relpath(path, ROOT))
print("cases", len(cases), "affected", len(affected), "notes", len(out["notes"]))
