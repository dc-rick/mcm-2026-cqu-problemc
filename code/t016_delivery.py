"""Build 交付/T016_delivery.json: recorded-day wording fix.

Five text replacements only; every check is computed from the current tree.
"""
import json
import os
import re
import subprocess
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
OUTDIR = os.path.join(ROOT, "data", "runs", "T016")
os.makedirs(OUTDIR, exist_ok=True)
BASE_COMMIT = "7ce233c"


def read(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def pdf_text(pdf_path, txt_path):
    subprocess.run(["pdftotext", "-layout", "-enc", "UTF-8", pdf_path, txt_path],
                   check=True)
    return read(txt_path)


pdf = pdf_text(os.path.join(PAPER, "main.pdf"), os.path.join(OUTDIR, "main_txt.txt"))
pdf_norm = re.sub(r"\s+", " ", pdf.replace("–", "-").replace("‒", "-"))

base_pdf = os.path.join(OUTDIR, "base_%s.pdf" % BASE_COMMIT)
with open(base_pdf, "wb") as fh:
    fh.write(subprocess.run(["git", "show", "%s:paper/main.pdf" % BASE_COMMIT],
                            cwd=ROOT, capture_output=True, check=True).stdout)
base_norm = pdf_text(base_pdf, os.path.join(OUTDIR, "base_txt.txt"))
base_norm = re.sub(r"\s+", " ", base_norm.replace("–", "-").replace("‒", "-"))

sec3 = read(os.path.join(PAPER, "sec3_data.tex"))
sec4 = read(os.path.join(PAPER, "sec4_q1.tex"))

# ---------- the five replacements ----------
items = [
    ("1_sec3_estimation_window",
     "All estimation uses the nine recorded days of 10--19 May,",
     "All estimation uses the nine recorded days of 10-19 May,"),
    ("2_sec4_caption",
     "Latent user segments (the nine recorded days of 10--19 May).",
     "Latent user segments (the nine recorded days of 10-19 May)."),
    ("3_sec4_table_head",
     "\\textbf{user (10--19 May)}",
     "user (10-19 May)"),
    ("4_sec4_regular_profile",
     "the nine recorded days of the 10--19 May window",
     "the nine recorded days of the 10-19 May window"),
    ("5_sec4_heavy_profile",
     "13.98 trips over the nine recorded days",
     "13.98 trips over the nine recorded days"),
]
checks = {}
for key, tex_snippet, pdf_snippet in items:
    src = sec3 if key.startswith("1_") else sec4
    checks[key] = bool(re.sub(r"\s+", " ", src).find(tex_snippet) >= 0
                       and pdf_snippet in pdf_norm)

# ---------- ten-days gate ----------
hits = []
for name in sorted(os.listdir(PAPER)):
    if not name.endswith(".tex"):
        continue
    for i, line in enumerate(read(os.path.join(PAPER, name)).splitlines(), 1):
        if re.search(r"ten days|ten-day|10 days", line, re.I):
            hits.append("%s:%d: %s" % (name, i, line.strip()))
checks["6_no_ten_days_wording"] = len(hits) == 0

# ---------- numeric tokens ----------
def tokens(text):
    return Counter(re.findall(r"\d[\d,\.]*", text))


base_tok, new_tok = tokens(base_norm), tokens(pdf_norm)
only_base, only_new = dict(base_tok - new_tok), dict(new_tok - base_tok)
checks["7_tokens_only_date_labels"] = (
    only_base == {} and set(only_new) <= {"10", "19"})

# ---------- compile facts ----------
log = read(os.path.join(PAPER, "main.log"))
total_pages = int(re.search(r"Output written on main\.pdf \((\d+) pages\)", log).group(1))
undefined = len(re.findall(r"Undefined control sequence|Reference .* undefined", log))
overfull = len(re.findall(r"Overfull \\[hv]box", log))
toc = read(os.path.join(PAPER, "main.toc"))
ref_page = int(re.search(r"\\contentsline \{section\}\{References\}\{(\d+)\}",
                         toc).group(1))
body_pages = ref_page - 3
u037e = pdf.count(";")
semicolons = pdf.count(";")
qq = pdf_norm.count("??")
checks["8_compile_unchanged"] = bool(
    total_pages == 30 and body_pages == 24 and undefined == 0 and overfull == 0
    and u037e == 0 and semicolons == 131 and qq == 0 and ref_page == 27)

# ---------- data evidence ----------
import pandas as pd

trips = pd.read_parquet(os.path.join(ROOT, "data", "trips.parquet"),
                        columns=["starttime"])
per_day = pd.to_datetime(trips["starttime"]).dt.date.value_counts().sort_index()
window = {str(k): int(v) for k, v in per_day.items() if 10 <= k.day <= 19}
has_17 = any(k.endswith("-17") for k in window)
checks["9_data_17_may_absent_nine_days"] = (not has_17 and len(window) == 9)

# ---------- scope ----------
status = subprocess.run(["git", "-c", "core.quotepath=false", "status", "--porcelain"],
                        cwd=ROOT, capture_output=True, text=True).stdout.splitlines()


def allowed(line):
    return (line.startswith((" M paper/", " M code/", "?? code/"))
            or (line.startswith("?? ") and "T016_delivery.json" in line))


out_of_scope = [ln for ln in status if not allowed(ln)]
checks["10_scope_clean"] = len(out_of_scope) == 0

delivery = {
    "task": "T016",
    "title": "日数口径修正",
    "date": "2026-11-18",
    "status": "T016 已交付",
    "checks": checks,
    "replacements": [
        {"file": "paper/sec3_data.tex", "line": 37,
         "old": "All estimation uses the ten days 10--19 May,",
         "new": "All estimation uses the nine recorded days of 10--19 May,"},
        {"file": "paper/sec4_q1.tex", "line": 178,
         "old": "Latent user segments (ten days).",
         "new": "Latent user segments (the nine recorded days of 10--19 May)."},
        {"file": "paper/sec4_q1.tex", "line": 186,
         "old": "\\textbf{user (10 days)}",
         "new": "\\textbf{user (10--19 May)}"},
        {"file": "paper/sec4_q1.tex", "line": 199,
         "old": "over the ten-day window",
         "new": "over the nine recorded days of the 10--19 May window"},
        {"file": "paper/sec4_q1.tex", "line": 201,
         "old": "over ten days",
         "new": "over the nine recorded days"},
    ],
    "compile": {"engine": "xelatex", "passes": 2,
                "total_pages": total_pages, "body_pages": body_pages,
                "body_range": "sections 1-10, pages 3-%d" % (ref_page - 1),
                "undefined": undefined, "overfull": overfull, "u037e": u037e,
                "semicolons_in_pdf": semicolons, "double_question_marks": qq,
                "references_page": ref_page},
    "token_diff": {"base": BASE_COMMIT, "only_in_base": only_base,
                   "only_in_new": only_new},
    "per_day_counts_10_19": window,
    "notes": [
        "数据依据：data/trips.parquet 按 starttime 计数，2017-05-17 无任何记录；10--19 May 只有九天有数据（10/11/12/13/14/15/16/18/19，22.5--32.2 万单/日），20--24 May 为截断窗口。code/04_behaviour.py:17 的用户表过滤即 05-10..05-19，故 trips per user 的分母是九个记录日。",
        "只改 5 处文字，未动任何数字、表值、图注数值或公式；未重跑模型。",
        "‘10--19 May’ 保留为窗口标签（其内 17 May 缺记录），文中用 ‘nine recorded days’ 限定，二者不再矛盾。",
        "数字 token 多重集与 7ce233c 的差异只有 10 与 19（日期标签），无删除、无数据值变化。",
        "grep 'ten days|ten-day|10 days' paper/*.tex = 0。",
    ],
    "outputs": ["paper/main.pdf", "paper/sec3_data.tex", "paper/sec4_q1.tex",
                "code/t016_delivery.py", "交付/T016_delivery.json"],
}

for path in (os.path.join(OUTDIR, "delivery.json"),
             os.path.join(ROOT, "交付", "T016_delivery.json")):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(delivery, fh, ensure_ascii=False, indent=1)

print(json.dumps(checks, indent=1))
print("pages", total_pages, "body", body_pages, "refs", ref_page,
      "u037e", u037e, "semicolons", semicolons)
print("only base", only_base, "only new", only_new)
print("days", len(window), "17 present:", has_17)
print("gate hits", len(hits), "out_of_scope", out_of_scope)
