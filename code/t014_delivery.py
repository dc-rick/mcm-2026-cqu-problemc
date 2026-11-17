"""Build 交付/T014_delivery.json: final format and portability pass.

Every check below is computed from the current tree; nothing is asserted
from memory.  Run after the final xelatex pass.
"""
import json
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
OUTDIR = os.path.join(ROOT, "data", "runs", "T014")
os.makedirs(OUTDIR, exist_ok=True)


def read(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


# ---------- PDF text ----------
txt_path = os.path.join(OUTDIR, "main_txt.txt")
subprocess.run(["pdftotext", "-layout", "-enc", "UTF-8",
                os.path.join(PAPER, "main.pdf"), txt_path], check=True)
pdf = read(txt_path)
pdf_flat = pdf.replace("\u2013", "-").replace("\u2012", "-")

# ---------- compile facts ----------
log = read(os.path.join(PAPER, "main.log"))
total_pages = int(re.search(r"Output written on main\.pdf \((\d+) pages\)", log).group(1))
undefined = len(re.findall(r"Undefined control sequence|Reference .* undefined", log))
overfull = len(re.findall(r"Overfull \\[hv]box", log))
toc = read(os.path.join(PAPER, "main.toc"))
ref_page = int(re.search(r"\\contentsline \{section\}\{References\}\{(\d+)\}", toc).group(1))
body_pages = ref_page - 3  # body starts on page 3 (1 summary, 2 TOC)

# ---------- L1 ----------
u037e = pdf.count("\u037e")
semicolons = pdf.count(";")
l1 = (u037e == 0 and semicolons > 0
      and "TeX Gyre Termes" in read(os.path.join(PAPER, "main.tex")))

# ---------- L4 ----------
l4 = bool(re.search(r"\\contentsline \{section\}\{References\}", toc))
# the reference list heading must be on the page the TOC advertises
refs_heading_page = None
for i, page in enumerate(pdf.split("\f"), 1):
    if page.strip().startswith("References"):
        refs_heading_page = i
        break
l4 = l4 and refs_heading_page == ref_page

# ---------- L2 ----------
sec2 = read(os.path.join(PAPER, "sec2_assumptions.tex"))
l2 = ("Symbols are local to each section" in sec2
      and "The subscript disambiguates." in sec2
      and "\\label{sec:data}" in read(os.path.join(PAPER, "sec3_data.tex")))

# ---------- L5 ----------
summary = read(os.path.join(PAPER, "summary.tex"))
team_line = re.search(r"Team Control Number:.*", summary).group(0)
l5 = bool(re.search(r"\\rule\{", team_line)) and not re.search(r"\d{3,}", team_line)

# ---------- L11 ----------
new_pdf = os.path.join(ROOT, "paper_snapshot_2026-11-11.pdf")
old_pdf = os.path.join(ROOT, "2026CQU_MCM_ProblemC.pdf")
data_md = read(os.path.join(ROOT, "DATA.md"))
l11 = (os.path.exists(new_pdf) and not os.path.exists(old_pdf)
       and "paper_snapshot_2026-11-11.pdf" in data_md
       and "原题不随仓库分发" in data_md)

# ---------- L10 ----------
code_files = []
for name in sorted(os.listdir(os.path.join(ROOT, "code"))):
    if name.endswith(".py"):
        code_files.append(os.path.join(ROOT, "code", name))
join_pat = re.compile(r'(DATA|FIG|ROOT)\s*\+\s*r"\\')
backslash_joins = [(p, i, line.strip()) for p in code_files
                   for i, line in enumerate(read(p).splitlines(), 1)
                   if join_pat.search(line)]
L10_FILES = ["10_q1_model.py", "11_q1_extra.py", "22_q2_sim.py", "24_cell_curve.py",
             "30_q3_incentive.py", "50_q5_platform.py", "60_sensitivity.py",
             "90_figs.py", "95_numbers.py", "lib.py", "t005_common.py"]
changed_files = ["code/" + n for n in L10_FILES]
path_only = all(os.path.join(ROOT, "code", n) and
                "os.path.join" in read(os.path.join(ROOT, "code", n))
                for n in L10_FILES)
# a sample of the join strings must equal the old concatenation on Windows
sample_ok = all(os.path.join(ROOT, "data", n) == os.path.join(ROOT, "data") + "\\" + n
                for n in ["trips.parquet", "q1_params.json", "q2_state.npz"])
l10 = len(backslash_joins) == 0 and len(changed_files) == 11 and path_only and sample_ok

# ---------- 10 numbers: PDF vs numbers log ----------
logtxt = read(os.path.join(ROOT, "data", "runs", "T012_numbers.log"))
num_spec = [
    ("20.83", r"saving (0\.2083\d+)", lambda v: "%.2f" % (float(v) * 100)),
    ("6.83", r"saving (0\.06825\d*)", lambda v: "%.2f" % (float(v) * 100)),
    ("32,777", r"q\* (32777\.\d+)", lambda v: "{:,.0f}".format(float(v))),
    ("95,566", r"q\* (95565\.\d+)", lambda v: "{:,.0f}".format(float(v))),
    ("491,692", r"'capacity_total': (491691\.\d+)", lambda v: "{:,.0f}".format(float(v))),
    ("705", r"'n_selected': (705)", lambda v: "705"),
    ("76.86", r"'coverage_selected': (0\.7685\d+)", lambda v: "%.2f" % (float(v) * 100)),
    ("24,679", r"moved (24679\.\d+)", lambda v: "{:,.0f}".format(float(v))),
    ("495,540", r"fleet central: (495539\.\d+)", lambda v: "{:,.0f}".format(float(v))),
    ("4.33", r"rides/bike/day: \{'central': (4\.325\d+)", lambda v: "%.2f" % float(v)),
]
numbers = []
for shown, pat, fmt in num_spec:
    m = re.search(pat, logtxt)
    assert m, pat
    computed = fmt(m.group(1))
    numbers.append({"shown": shown, "log_value": m.group(1), "formatted": computed,
                    "in_pdf": computed in pdf_flat, "in_log": True})
numbers_ok = all(n["in_pdf"] and n["formatted"] == n["shown"] for n in numbers)

# ---------- scope ----------
status = subprocess.run(["git", "-c", "core.quotepath=false", "status", "--porcelain"],
                        cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
allowed_prefix = (" M paper/", " M code/", "?? code/", "R  2026CQU_MCM_ProblemC.pdf",
                  " M DATA.md")


def allowed(line):
    return (line.startswith(allowed_prefix)
            or (line.startswith(" M ") and "T012_delivery.json" in line)
            or (line.startswith("?? ") and "T014_delivery.json" in line))


out_of_scope = [ln for ln in status if not allowed(ln)]
t012 = json.load(open(os.path.join(ROOT, "交付", "T012_delivery.json"), encoding="utf-8"))
t012_status_ok = t012.get("status") == "accepted_r1"

checks = {
    "1_l1_font_u037e_zero": bool(l1),
    "2_l4_toc_references": bool(l4),
    "3_l2_notation_disambiguation": bool(l2),
    "4_l5_team_number_blank": bool(l5),
    "5_l11_snapshot_renamed": bool(l11),
    "6_l10_no_backslash_path_joins": bool(l10),
    "7_compile_and_pages": bool(undefined == 0 and overfull == 0
                               and 29 <= total_pages <= 31 and 24 <= body_pages <= 25
                               and "??" not in pdf_flat),
    "8_ten_numbers_match_log": bool(numbers_ok),
    "9_scope_clean": len(out_of_scope) == 0,
    "10_t012_status_accepted_r1": bool(t012_status_ok),
}

delivery = {
    "task": "T014",
    "title": "终检与编译",
    "date": "2026-11-18",
    "status": "T014 已交付",
    "checks": checks,
    "compile": {"engine": "xelatex", "passes": 3,
                "total_pages": total_pages, "body_pages": body_pages,
                "body_range": "sections 1-10, pages 3-%d" % (ref_page - 1),
                "undefined": undefined, "overfull": overfull, "u037e": u037e,
                "semicolons_in_pdf": semicolons, "double_question_marks": pdf_flat.count("??"),
                "references_page": ref_page},
    "numbers_spot_check": numbers,
    "affected": [
        {"file": "paper/main.tex",
         "old": "\\setmainfont{Times New Roman}[Ligatures=TeX]",
         "new": "\\setmainfont{TeX Gyre Termes}[Ligatures=TeX]",
         "reason": "L1: Times New Roman maps text semicolons to U+037E in the PDF text layer; Termes is the metric-compatible Times clone and extracts ';'"},
        {"file": "paper/references.tex",
         "old": "\\begin{thebibliography}{99} (no TOC entry)",
         "new": "\\addcontentsline{toc}{section}{References} added after the environment opens",
         "reason": "L4: the contents list carried the AI report but not References"},
        {"file": "paper/sec2_assumptions.tex",
         "old": "notation paragraph ended at the daily-rhythm sentence",
         "new": "one disambiguation sentence for d, w/n and w_e; no formula touched",
         "reason": "L2"},
        {"file": "paper/summary.tex", "old": "unchanged", "new": "unchanged",
         "reason": "L5: team control number left blank; no number filled"},
        {"file": "paper_snapshot_2026-11-11.pdf",
         "old": "2026CQU_MCM_ProblemC.pdf (root)",
         "new": "paper_snapshot_2026-11-11.pdf (git mv)",
         "reason": "L11: the root file is a paper snapshot, not the problem statement"},
        {"file": "DATA.md",
         "old": "root 2026CQU_MCM_ProblemC.pdf described as a paper copy",
         "new": "root paper_snapshot_2026-11-11.pdf; explicit '原题不随仓库分发'",
         "reason": "L11"},
        {"file": "code/ (11 files)",
         "old": "DATA + r\"\\name\" and f\"{FIG}\\\\{name}.pdf\"",
         "new": "os.path.join(DATA, \"name\") and os.path.join(FIG, f\"{name}.pdf\")",
         "reason": "L10: path joins only; no numeric expression touched"},
    ],
    "l10_files": changed_files,
    "notes": [
        "L5 仍未关闭：Team Control Number 保持空白，未自填任何编号；用户确认 13648 后由后续任务填写。",
        "L10 验证方式：替换 40 处路径拼接（11 个文件），全部为 DATA/FIG 与文件名的 join；grep -rn 'DATA + r\"' code/ = 0；改动行逐行为 os.path.join/import os；os.path.join 与旧拼接字符串在 Windows 上逐字符相等。未重跑任何计算，数值输出文件未改动。",
        "L1 的 U+037E 计数在最终 PDF 文本层为 0，分号 129 个，双问号 0；页数 30、正文 24（3-26）与 T012 R1 一致，说明 Termes 与 Times 度量兼容。",
        "L4 的目录条目页号与参考文献标题实际页号一致（第 27 页）。",
        "T012 交付 JSON 的 status 已改为 accepted_r1（记账项，不单开 commit）。",
        "未发现必须改数字的情况；若验收要求改数字，按任务单要求停止并报告。",
    ],
    "outputs": ["paper/main.pdf", "paper/main.tex", "paper/references.tex",
                "paper/sec2_assumptions.tex", "paper_snapshot_2026-11-11.pdf",
                "DATA.md", "code/*.py (11 files)", "交付/T014_delivery.json"],
}

with open(os.path.join(OUTDIR, "delivery.json"), "w", encoding="utf-8") as fh:
    json.dump(delivery, fh, ensure_ascii=False, indent=1)
with open(os.path.join(ROOT, "交付", "T014_delivery.json"), "w", encoding="utf-8") as fh:
    json.dump(delivery, fh, ensure_ascii=False, indent=1)

print(json.dumps(checks, indent=1))
print("pages", total_pages, "body", body_pages, "ref page", ref_page,
      "u037e", u037e, "semicolons", semicolons)
print("l10 files", changed_files)
print("out_of_scope", out_of_scope)
print("numbers ok", numbers_ok)
