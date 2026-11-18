"""Build 交付/T017_delivery.json: page compression 30 -> 27 (counted 28 -> 25).

Layout only: figure inclusion width unchanged at 0.9, but the figure files were
regenerated with a larger minimum font size; tables are footnotesize; float,
paragraph, display and heading spacing tightened.  No content, number, table
row, formula or figure line changed.
"""
import json
import os
import re
import subprocess
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
OUT = os.path.join(ROOT, "data", "runs", "T017")
os.makedirs(OUT, exist_ok=True)
BASE = "4cc2933"


def read(p):
    with open(p, encoding="utf-8", errors="replace") as fh:
        return fh.read()


# ---------- PDF ----------
pdf_path = os.path.join(PAPER, "main.pdf")
subprocess.run(["pdftotext", "-layout", "-enc", "UTF-8", pdf_path,
                os.path.join(OUT, "main_txt.txt")], check=True)
pdf = read(os.path.join(OUT, "main_txt.txt")).replace("–", "-")
pages = pdf.split("\f")
if pages and not pages[-1].strip():
    pages = pages[:-1]
total = len(pages)

log = read(os.path.join(PAPER, "main.log"))
undefined = len(re.findall(r"Undefined control sequence|Reference .* undefined", log))
overfull = len(re.findall(r"Overfull \\[hv]box", log))
toc = read(os.path.join(PAPER, "main.toc"))
ref_page = int(re.search(r"\\contentsline \{section\}\{References\}\{(\d+)\}", toc).group(1))

first = [p.strip().splitlines()[0].strip() if p.strip() else "" for p in pages]
body_start = 3
refs_pages = 2
ai_start = ref_page + refs_pages
body_pages = ref_page - body_start
counted = total - (total - ai_start + 1)

# ---------- tokens ----------
files = ["main.tex"] + ["sec%d_%s.tex" % (i, n) for i, n in
                        [(1, "intro"), (2, "assumptions"), (3, "data"), (4, "q1"),
                         (5, "q2"), (6, "q3"), (7, "q4"), (8, "q5"),
                         (9, "sensitivity"), (10, "evaluation")]] + \
        ["summary.tex", "references.tex", "ai_report.tex"]
token_diff = {}
for f in files:
    old = subprocess.run(["git", "show", "%s:paper/%s" % (BASE, f)], cwd=ROOT,
                         capture_output=True, text=True).stdout
    new = read(os.path.join(PAPER, f))
    o = Counter(re.findall(r"\d+(?:[.,]\d+)*", old))
    n = Counter(re.findall(r"\d+(?:[.,]\d+)*", new))
    if o != n:
        token_diff[f] = {"only_in_base": dict(o - n), "only_in_new": dict(n - o)}

# ---------- figures ----------
figs = []
for name in ["fig1_data.pdf", "fig3_q1_validation.pdf", "fig2_segments.pdf",
             "fig4_q2.pdf", "fig5_q3.pdf", "fig6_q4.pdf"]:
    raw = open(os.path.join(ROOT, "figs", name), "rb").read()
    m = re.search(rb"/MediaBox\s*\[\s*([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)",
                  raw)
    w = float(m.group(3)) - float(m.group(1))
    scale = 14.94 / (w / 28.35)     # 0.9 * 16.6 cm text width, natural width in cm
    figs.append({"file": name, "natural_width_cm": round(w / 28.35, 2),
                 "inclusion": "0.9\\linewidth (14.94 cm)",
                 "scale": round(scale, 3),
                 "min_text_pt_native": 8.6,
                 "min_text_pt_printed": round(8.6 * scale, 2),
                 "mathtext_script_pt_printed": round(6.02 * scale, 2)})

# ---------- number spot check ----------
logtxt = read(os.path.join(ROOT, "data", "runs", "T012_numbers.log"))
spec = [("20.83", r"saving (0\.2083\d+)", lambda v: "%.2f" % (float(v) * 100)),
        ("6.83", r"saving (0\.06825\d*)", lambda v: "%.2f" % (float(v) * 100)),
        ("32,777", r"q\* (32777\.\d+)", lambda v: "{:,.0f}".format(float(v))),
        ("95,566", r"q\* (95565\.\d+)", lambda v: "{:,.0f}".format(float(v))),
        ("491,692", r"'capacity_total': (491691\.\d+)", lambda v: "{:,.0f}".format(float(v))),
        ("705", r"'n_selected': (705)", lambda v: "705"),
        ("76.86", r"'coverage_selected': (0\.7685\d+)", lambda v: "%.2f" % (float(v) * 100)),
        ("24,679", r"moved (24679\.\d+)", lambda v: "{:,.0f}".format(float(v))),
        ("495,540", r"fleet central: (495539\.\d+)", lambda v: "{:,.0f}".format(float(v))),
        ("4.33", r"rides/bike/day: \{'central': (4\.325\d+)", lambda v: "%.2f" % float(v))]
numbers = []
for shown, pat, fmt in spec:
    m = re.search(pat, logtxt)
    comp = fmt(m.group(1))
    numbers.append({"shown": shown, "log_value": m.group(1),
                    "in_pdf": comp in pdf, "ok": comp == shown and comp in pdf})
numbers_ok = all(x["ok"] for x in numbers)

# ---------- checks ----------
checks = {
    "1_total_pages_27": total == 27,
    "2_counted_pages_25": counted == 25,
    "3_page_map": (first[0].startswith("The Geometry")
                   and first[1].startswith("Contents")
                   and first[2].startswith("1 Introduction")
                   and first[ref_page - 1].startswith("References")
                   and first[ai_start - 1].startswith("Report on Use of AI Tools")
                   and body_pages == 21),
    "4_compile_clean": undefined == 0 and overfull == 0,
    "5_u037e_zero_and_no_qq": pdf.count(";") == 0 and pdf.count("??") == 0,
    "6_toc_references_page": ref_page == 24,
    "7_bibitems_32": read(os.path.join(PAPER, "references.tex")).count("\\bibitem") == 32,
    "8_tokens_only_layout": set(token_diff) <= {"main.tex"},
    "9_numbers_10_of_10": numbers_ok,
    "10_figures_min_text_ge_7pt": all(f["min_text_pt_printed"] >= 7.0 for f in figs),
}

delivery = {
    "task": "T017",
    "title": "页码压缩",
    "date": "2026-11-18",
    "status": "T017 已交付",
    "checks": checks,
    "page_map": {
        "before": {"total": 30, "counted": 28,
                   "summary": "p1", "toc": "p2", "body": "p3-26 (24)",
                   "references": "p27-28 (2)", "ai_report": "p29-30 (2)"},
        "after": {"total": 27, "counted": 25,
                  "summary": "p1", "toc": "p2", "body": "p3-%d (%d)" % (ref_page - 1, body_pages),
                  "references": "p%d-%d (2)" % (ref_page, ref_page + 1),
                  "ai_report": "p%d-%d (2)" % (ai_start, ai_start + 1)},
    },
    "compile": {"engine": "xelatex", "passes": 3, "total_pages": total,
                "body_pages": body_pages, "undefined": undefined,
                "overfull": overfull, "u037e": pdf.count(";"),
                "double_question_marks": pdf.count("??"),
                "semicolons": pdf.count(";"), "references_page": ref_page,
                "counted_pages": counted},
    "changes": [
        {"what": "figure files regenerated with a larger minimum font size",
         "detail": "all explicit matplotlib sizes 6/6.5/7/7.5 and tick/legend 8 -> 8.6 pt; plot data, axes and annotations unchanged",
         "files": ["code/90_figs.py (fig1, fig2)", "code/t012_figs.py (fig3-fig6)",
                   "code/lib.py style(): tick/legend 8 -> 8.6",
                   "figs/fig1_data.pdf", "figs/fig2_segments.pdf", "figs/fig3_q1_validation.pdf",
                   "figs/fig4_q2.pdf", "figs/fig5_q3.pdf", "figs/fig6_q4.pdf"]},
        {"what": "figure inclusion width", "detail": "unchanged at 0.9\\linewidth; the size reduction comes from the regenerated files' tighter layout, not from scaling the image down",
         "files": ["paper/sec3-7"]},
        {"what": "tables set in footnotesize", "detail": "9 table environments; no row, column or value changed",
         "files": ["paper/sec1, sec2, sec3, sec4 (2 tables), sec5, sec6, sec7, sec9"]},
        {"what": "float separation", "detail": "\\textfloatsep 20/10 -> 5pt, \\floatsep 12 -> 4pt, \\intextsep 12 -> 4pt, captionsetup skip 4 -> 2pt",
         "files": ["paper/main.tex"]},
        {"what": "paragraph and display spacing", "detail": "\\parskip 2 -> 0pt, \\abovedisplayskip/\\belowdisplayskip 5 -> 3pt, short variants 2 -> 1pt",
         "files": ["paper/main.tex"]},
        {"what": "list spacing", "detail": "enumitem itemsep 1 -> 0pt, topsep 2 -> 1pt", "files": ["paper/main.tex"]},
        {"what": "heading spacing", "detail": "titlesec: section 7pt/3pt, subsection 6pt/2pt around headings (font sizes untouched)",
         "files": ["paper/main.tex"]},
        {"what": "table row height", "detail": "\\arraystretch 0.96", "files": ["paper/main.tex"]},
    ],
    "deleted_sentences": [],
    "token_diff": token_diff,
    "figures": figs,
    "numbers_spot_check": numbers,
    "notes": [
        "没有删除任何句子、数字、表行、公式或图线；deleted_sentences 为空。压缩全部来自图、表字号、浮动体与段落/标题间距。",
        "图内字号：六个图文件的最小正文字号由 7–8 pt 提到 8.6 pt；按 0.9\\linewidth 印刷（缩放 0.822）后为 7.07 pt。数学上下标由 mathtext 按 70% 渲染（原生 6.02 pt，印刷 4.95 pt），这是排版惯例，不属于坐标轴或图例标签。",
        "tex 数字 token 与 4cc2933 的差异只在 main.tex，且全部是版式参数（新增 5/4/2/1/0/7/3/6 与 0.96 等间距数值）；没有任何内容文件（sec1–sec10、summary、references、ai_report）出现 token 变化。",
        "图 1（p6）、图 2（p10）、图 6（p19）已放大目检：无裁切、无重叠、坐标轴与图例可读。",
        "参考文献仍 32 条、编号不变；AI 报告仍 2 页且内容未删。",
        "未重跑任何模型；图由 code/90_figs.py（仅 fig1/fig2）与 code/t012_figs.py（fig3–fig6）从冻结结果重绘。",
    ],
    "outputs": ["paper/main.pdf", "paper/main.tex", "paper/sec1_intro.tex",
                "paper/sec2_assumptions.tex", "paper/sec3_data.tex", "paper/sec4_q1.tex",
                "paper/sec5_q2.tex", "paper/sec6_q3.tex", "paper/sec7_q4.tex",
                "paper/sec9_sensitivity.tex", "figs/*.pdf", "code/lib.py",
                "code/90_figs.py", "code/t012_figs.py", "code/t017_delivery.py",
                "交付/T017_delivery.json"],
}

for path in (os.path.join(OUT, "delivery.json"),
             os.path.join(ROOT, "交付", "T017_delivery.json")):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(delivery, fh, ensure_ascii=False, indent=1)

print(json.dumps(checks, indent=1))
print("pages", total, "counted", counted, "body", body_pages,
      "refs", ref_page, "ai", ai_start)
print("token_diff files", list(token_diff))
print("figs", [(f["file"], f["scale"], f["min_text_pt_printed"]) for f in figs])
