"""T012 self-checks 1-10: paper text against frozen result files."""
import io
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
LOGN = os.path.join(ROOT, "data", "runs", "T012_numbers.log")
XLOG = os.path.join(PAPER, "main.log")
FIG = os.path.join(ROOT, "figs")


def read(path):
    return io.open(path, encoding="utf-8").read()


def all_text(names):
    return "\n".join(read(os.path.join(PAPER, n)) for n in names)


core = all_text(["summary.tex", "sec1_intro.tex", "sec2_assumptions.tex",
                 "sec3_data.tex", "sec4_q1.tex", "sec5_q2.tex", "sec6_q3.tex",
                 "sec7_q4.tex", "sec8_q5.tex", "sec9_sensitivity.tex",
                 "sec10_evaluation.tex"])
log = read(LOGN)
checks = {}
detail = {}


def has_all(tokens):
    return [t for t in tokens if t not in core]


def has_none(tokens):
    return [t for t in tokens if t in core]


# 1. numbers log carries the locked Q3/Q2/Q4 anchors
anchors = ["r* 1.76", "q* 95565.81547068334", "saving 0.20830007675074336",
           "r* 1.83", "q* 32777.11212887443", "saving 0.06825835929691104",
           "cross_zone_share': 0.07464335953503903",
           "net_zone_hour_positive_sum': 4332.1059422301505",
           "need total: 216343.75", "need cells >0: 5188"]
miss = [a for a in anchors if a not in log]
checks["1_numbers_log_matches_results"] = not miss
detail["1_missing"] = miss

# 2. Q3 locked values, no obsolete Q3 string
need = ["20.83", "6.83", "1.76", "1.83", "95,566", "32,777", "7.46", "4,332",
        "0.22"]
miss = has_all(need)
old = has_none(["34.9", "115,385", "752,404", "71.9", "1,025", "about 2\\%",
                "约 2\\%"])
checks["2_q3_locked_and_no_obsolete"] = not miss and not old
detail["2_missing"] = miss
detail["2_obsolete_present"] = old

# 3. Q2 locked values and no stale K=80 increment
need = ["24,679", "5,530", "82,195", "82,933", "8,269", "225", "39.34",
        "37.12"]
miss = has_all(need)
old = has_none(["41,233", "23,959 bikes", "3.48 CNY"])
checks["3_q2_locked"] = not miss and not old
detail["3_missing"] = miss
detail["3_obsolete_present"] = old

# 4. Q4 locked values, no zero-initial or old 626 numbers
need = ["705", "76.86", "491,692", "621,004", "458,516", "582,977",
        "2,020", "1.0213", "0.804", "0.423", "150"]
miss = has_all(need)
old = has_none(["344,463", "4,827", "89.8", "48.2", "0.2\\%", "344,463"])
checks["4_q4_locked"] = not miss and not old
detail["4_missing"] = miss
detail["4_obsolete_present"] = old

# 5. Q1 locked values
need = ["3.487", "3.645", "3.278", "3.561", "3.394", "0.098", "12.83",
        "8.89", "0.256", "0.263", "0.665", "0.306"]
miss = has_all(need)
checks["5_q1_locked"] = not miss
detail["5_missing"] = miss

# 6. data / fleet locked values
need = ["495,540", "4.33", "343,638", "54.53", "63.11", "12.83", "17.72",
        "19.97"]
miss = has_all(need)
checks["6_data_locked"] = not miss
detail["6_missing"] = miss

# 7. compile: two passes, zero undefined/overfull, body <= 25 pages
xlog = read(XLOG)
undefined = len(re.findall(r"undefined", xlog))
overfull = len(re.findall(r"Overfull", xlog))
m = re.search(r"sec1_intro\.tex\s*\[(\d+)\]", xlog)
ref = re.search(r"references\.tex\s*\[(\d+)\]", xlog)
body = int(ref.group(1)) - int(m.group(1)) if m and ref else -1
checks["7_compile_clean_and_page_limit"] = (undefined == 0 and
                                            overfull == 0 and 0 < body <= 25)
detail["7_undefined"] = undefined
detail["7_overfull"] = overfull
detail["7_body_pages"] = body

# 8. figures regenerated and fig3 annotation matches locked same-cell
figs = ["fig3_q1_validation.pdf", "fig4_q2.pdf", "fig5_q3.pdf",
        "fig6_q4.pdf"]
fig_ok = all(os.path.exists(os.path.join(FIG, f)) for f in figs)
try:
    txt = subprocess.check_output(["pdftotext",
                                   os.path.join(FIG, "fig3_q1_validation.pdf"),
                                   "-"], text=True)
except Exception:
    txt = ""
fig_ok = fig_ok and "12.83%" in txt and "8.89%" in txt and "9.5%" not in txt
checks["8_figures_regenerated"] = fig_ok
detail["8_fig3_text_ok"] = ("12.83%" in txt and "8.89%" in txt)

# 9. references and AI-report wording
refs = read(os.path.join(PAPER, "references.tex"))
air = read(os.path.join(PAPER, "ai_report.tex"))
checks["9_references_and_ai_report"] = ("\\small" in refs and
                                        "next revision" not in air)
detail["9_refs_small"] = "\\small" in refs

# 10. read-only scope: no changes outside paper/, figs/, code/, delivery
try:
    st = subprocess.check_output(["git", "status", "--short"],
                                 cwd=ROOT, text=True)
except Exception:
    st = ""
bad = [ln for ln in st.splitlines()
       if ln.strip() and not re.search(
           r"(paper/|figs/|code/|交付/)", ln)
       and "T012_delivery.json" not in ln]
checks["10_readonly_scope"] = not bad
detail["10_out_of_scope"] = bad

out = {"checks": checks, "detail": detail,
       "source_log": "data/runs/T012_numbers.log",
       "paper_files": sorted(os.listdir(PAPER))}
os.makedirs(os.path.join(ROOT, "data", "runs", "T012"), exist_ok=True)
json.dump(out, io.open(os.path.join(ROOT, "data", "runs", "T012",
                                    "validate.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(json.dumps(checks, indent=1))
print(json.dumps(detail, indent=1))
sys.exit(0 if all(checks.values()) else 1)
