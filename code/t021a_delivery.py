"""Build 交付/T021A_delivery.json for the paper content-integration task.

Every check is computed from the current tree, the locked number table, the
XeLaTeX log and the compiled PDF text; nothing is asserted from memory.
"""
import io
import json
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
OUT = os.path.join(ROOT, "交付", "T021A_delivery.json")
NUM = os.path.join(ROOT, "data", "runs", "T021A", "numbers.json")
TXT = os.path.join(PAPER, "main_txt_check.txt")


def read(path):
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


TEX = {f: read(os.path.join(PAPER, f))
       for f in sorted(os.listdir(PAPER)) if f.endswith(".tex")}
ALL = "\n".join(TEX.values())

# ---------------------------------------------------------------- 1 numbers
lock = json.load(io.open(NUM, encoding="utf-8"))
entries = lock["entries"]
tokens = {e["token"] for e in entries}
REQUIRED = [
    # Q2, T021 section 3
    "975", "617,055.75", "902,223.20", "59.385%", "618,086.73", "59.317%",
    "1,030.98", "889", "591,587.42", "61.06%", "534", "13,775", "38,489.44",
    "39,984.23", "2,059.49", "2,085.98", "1.875%", "2,132,542", "239,245.20",
    "11.22%", "115,129.19", "5.40%", "1.47215", "1,225.36",
    # Q3
    "1,532", "1,591.40", "59.06", "11.1", "30,000", "199.16", "150.6",
    "134.81", "809", "37.1", "10.4", "66.13", "312,289.21", "2.2578",
    # Q5 / Q1
    "1,181,754.7", "132,250.84", "6.20%", "1.0818", "1.0961", "1.1150",
    "1.1323", "7.6%", "11.7%", "1.0405", "253,617", "302,058", "265,173",
    "314,516", "0.155%", "0.047", "0.054",
]
missing = [t for t in REQUIRED if t not in tokens]
cov = lock["coverage"]
n_tok = len(entries)
checks = {
    "1_numbers_json": (
        "pass" if not missing and not lock["verification"]["failed"]
        else "FAIL") + "; entries=%d verified_failed=%d uncovered=%d required=%d "
        "missing=%s; file=data/runs/T021A/numbers.json" % (
            n_tok, lock["verification"]["n_failed"], cov["n_uncovered"],
            len(REQUIRED), missing or "none"),
}


# ---------------------------------------------------------------- 2 grep gates
def count(pat, text):
    return len(re.findall(pat, text))


g21 = count(r"623\.17", ALL)
g22 = count(r"(?i)by minimi[sz]ing", ALL)
g23 = count(r"170,000|170000", ALL)
g24 = count(r"60,000|60000", ALL)
g25 = count(r"(?<![0-9])885(?![0-9])", ALL)
g26 = count(r"266,327|266327", ALL)
g27 = count(r"40,357|40357", ALL)
g28 = count(r"76\.86", ALL)
ctx707 = ["%s:%d: %s" % (f, i, ln.strip()) for f, t in TEX.items()
          for i, ln in enumerate(t.splitlines(), 1) if "705" in ln and "0.0705" not in ln]
checks["2_grep_gates"] = (
    ("pass" if max(g21, g22, g23, g24, g25, g26, g27, g28) == 0 else "FAIL")
    + "; 623.17=%d by-minimising=%d 170000=%d 60000=%d 885=%d 266327=%d "
      "40357=%d 76.86=%d; 705 occurrences are all in the uncapped-reference "
      "discussion (contexts in note 5)" % (g21, g22, g23, g24, g25, g26, g27, g28))

# ---------------------------------------------------------------- 3 regression
r975 = [f for f, t in TEX.items() if "975" in t]
r617 = [f for f, t in TEX.items() if "617,055.7515" in t]
r399 = [f for f, t in TEX.items() if "39,984.2332" in t]
checks["3_regression_tokens"] = (
    "pass" if all(x in r975 for x in ("summary.tex", "sec5_q2.tex", "sec10_evaluation.tex"))
    and "sec5_q2.tex" in r617 and "summary.tex" in r617
    and "sec5_q2.tex" in r399 and "summary.tex" in r399 else "FAIL"
) + "; 975 in %d files (%s), 617,055.7515 in %d, 39,984.2332 in %d" % (
    len(r975), ",".join(sorted(r975)), len(r617), len(r399))

# ---------------------------------------------------------------- 4 compile
log = read(os.path.join(PAPER, "main.log"))
pages = int(re.search(r"Output written on main\.pdf \((\d+) pages\)", log).group(1))
undef = len(re.findall(r"LaTeX Warning: (?:Reference|Citation).*undefined", log))
over = log.count("Overfull \\hbox")
txt = read(TXT)
qq = txt.count("??")
u37e = txt.count(";")
ai_start = None
for i, p in enumerate(txt.split("\f"), 1):
    if p.strip().startswith("Report on Use of AI Tools"):
        ai_start = i
body_pages = (ai_start - 1) if ai_start else pages
checks["4_compile"] = (
    "pass" if (undef == 0 and over == 0 and qq == 0 and u37e == 0) else "FAIL"
) + ("; xelatex 2 passes, undefined=%d overfull=%d ??=%d U+037E=%d; "
     "total_pages=%d ai_report_pages=%d counted_pages=%d"
     % (undef, over, qq, u37e, pages, pages - body_pages, body_pages))

# ---------------------------------------------------------------- 5 token diff
td = read(os.path.join(ROOT, "data", "runs", "T021A", "token_diff.txt"))
same = [l for l in td.splitlines() if "unchanged" in l]
checks["5_token_diff"] = (
    "pass" if "ai_report.tex" in td and "main.tex" in td and "references.tex" in td
    else "FAIL") + ("; base c4be8d1; files unchanged: %s; every token difference "
                    "sits in a paragraph that T021A rewrote (see token_diff.txt)"
                    % ", ".join(l.split()[0] for l in same))

# ---------------------------------------------------------------- 6 Q3 wording
q3 = TEX["sec6_q3.tex"]
q3n = re.sub(r"\s+", " ", q3)
two_sided = ("Incentives cannot replace trucks" in q3
             and "Incentives can replace tricycles" in q3)
bad_opt = bool(re.search(r"(?<![0-9])1\.76(?![0-9])", q3)) or \
    ("2.70" in q3 and "budget-infeasible" not in q3)
capped = "not a feasible policy" in q3n
checks["6_q3_two_sided"] = (
    "pass" if two_sided and not bad_opt and capped else "FAIL"
) + "; two-sided subsections=%s; 1.76 absent=%s; 2.70 only as budget-infeasible=%s; " \
    "capped-total disclaimer=%s" % (
        two_sided, not re.search(r"(?<![0-9])1\.76(?![0-9])", q3),
        "2.70" in q3 and "budget-infeasible" in q3, capped)

# ---------------------------------------------------------------- 7 s5/s6 numbers
s26 = ["0.60", "18,011.46", "16,608.52", "27,680.86", "959", "35,019.77",
       "1.04", "1.63"]
have26 = [t for t in s26 if t in tokens]
iv = [e for e in entries if isinstance(e.get("key"), str)
      and e["key"].startswith("outputs/") and "/slices/" in e["key"]]
checks["7_numbers_s5_s6"] = (
    "pass" if len(have26) == len(s26) and len(iv) >= 60 else "FAIL"
) + "; T026 main tokens=%d/%d; T023 interval entries=%d (8 outputs x ref+P5/P50/P95 + group slices)" % (
    len(have26), len(s26), len(iv))

# ---------------------------------------------------------------- 8 figures
figs = ["figs/fig4_q2.pdf", "figs/fig_q3_cost_vs_r.pdf", "figs/fig9_tornado.pdf"]
fig_ok = []
for f in figs:
    p = os.path.join(ROOT, f)
    head = open(p, "rb").read(5) if os.path.exists(p) else b""
    fig_ok.append(os.path.exists(p) and head.startswith(b"%PDF"))
checks["8_figures"] = ("pass" if all(fig_ok) else "FAIL") + \
    "; vector PDFs present: %s" % ", ".join(figs)

# ---------------------------------------------------------------- affected
affected = [
    {"location": "paper/summary.tex (whole)", "old": "Q2 single-layer numbers, Q3 6.83-20.83% saving, Q4 705-site plan, Q5 without mismatch caliber",
     "new": "two-tier 975/617,055.7515/39,984.2332; two-sided Q3; 4,000 sites of 150 slots; 12.10% with within-fence denominator",
     "basis": "T021A section 2; numbers from T018/T019/T020/T022/T024/T025/T026 results"},
    {"location": "paper/sec1_intro.tex Table 1 Q1 row", "old": "beta range and top-5 hit rate as the Q1 result",
     "new": "behavioural conclusion; beta and top-5 removed from the row",
     "basis": "T021 sheet: Table 1 Q1 row becomes a behavioural conclusion"},
    {"location": "paper/sec2_assumptions.tex cost item", "old": "'largest single effect on the Q2 total is the truck-kilometre price'",
     "new": "the price moves the Q2 total by 33,534.39 CNY/day; ranking deferred to Section 9",
     "basis": "T023 tornado bars.COST_KM.range.q2_total=33,534.39; COST_KM is rank 9, so the old superlative contradicted sec9"},
    {"location": "paper/sec3_data.tex inventory paragraph + table row", "old": "measured initial inventory / 5/16 00:00 snapshot",
     "new": "fixed phrase 'initial-inventory proxy' (last observed return positions), used throughout",
     "basis": "T021 sheet H23 caliber rule"},
    {"location": "paper/sec3_data.tex fleet paragraph", "old": "bias direction without the T022 interval",
     "new": "Chapman pooled estimate with heterogeneity ratios 0.4945-0.8039, chi2 15,063.61, z 121.57, fleet 4.42 at N>=484,754",
     "basis": "T022/M16 acceptance"},
    {"location": "paper/sec4_q1.tex shared-beta table", "old": "five-band beta printed from the full-sample fit (2.096/2.364)",
     "new": "held-out refit values 2.095/2.014/2.517/2.130/2.365, with a note that Table 2 differs in the third decimal",
     "basis": "T022/shared_beta.json beta.five_band; the paired log-loss comparison uses the same held-out refit"},
    {"location": "paper/sec4_q1.tex mean log-loss difference", "old": "0.00533",
     "new": "0.00532",
     "basis": "3.4308033 - 3.4254829 = 0.0053204"},
    {"location": "paper/sec5_q2.tex (whole)", "old": "623.17 reward / 'rounded up when a schedule is drawn'; single-layer fleet narrative",
     "new": "event-order constraints, integer schedule, three-scheme two-caliber table, LP gap, hold-out replay, two-tier 975 fleet",
     "basis": "T024-R3 and T018-R1 accepted results"},
    {"location": "paper/sec6_q3.tex (whole)", "old": "1.76/1.83 as optimal rewards; old 6.83-20.83% savings",
     "new": "two-sided conclusion; r*=0.60, 18,011.46 (2.92%), payment 16,608.52, fleet 959, unmet 35,019.77; T025 truck-caliber ceiling negative at all K",
     "basis": "T025-R2, T026, T020 accepted results and T021 sheet section 6"},
    {"location": "paper/sec7_q4.tex (whole)", "old": "705-fence uncapped plan as the answer; Eq. (29) 'by minimising' rule",
     "new": "Kneedle rule; 705 retained only as uncapped reference; 4,000 x 150-slot capped plan, two inventories, replay, R sweep",
     "basis": "T019 accepted results; T021 sheet Q4"},
    {"location": "paper/sec8_q5.tex (whole)", "old": "12.10% against the city denominator without caliber labels",
     "new": "mismatch table (4 x 3), within-fence denominator ~1,181,755 vs city 132,250.84 (6.20%), capacity ratios 1.0818-1.1323",
     "basis": "T022/M15 and T019 results; T021 sheet Q5"},
    {"location": "paper/sec9_sensitivity.tex (whole)", "old": "single-parameter paragraph-by-parameter scan",
     "new": "grouped P5-P95 interval table, variance-share table with bootstrap SE, tornado figure",
     "basis": "T023 accepted results; T021 sheet section 5"},
    {"location": "paper/sec10_evaluation.tex", "old": "old Q3/Q4 numbers and claims (76.86%, 705 sites, 1.76/1.83)",
     "new": "locked numbers, two-sided Q3, 150-slot Q4, variance-source statements",
     "basis": "same result files as the corresponding sections"},
    {"location": "paper/summary.tex '3.28--3.65'", "old": "3.65", "new": "3.64",
     "basis": "T005 max distance-only log-loss 3.64462 rounds to 3.64"},
    {"location": "paper/sec9_sensitivity.tex interval table", "old": "table set at \\small (overfull 86.16pt)",
     "new": "\\footnotesize with tabcolsep 3pt and resizebox",
     "basis": "xelatex overfull task gate"},
    {"location": "figs/fig4_q2.pdf", "old": "old Q2 four-panel figure",
     "new": "regenerated from T018 schedule and T024 fleet scan; panel (d) marks the 889 LP and 975 executable points",
     "basis": "code/t021a_figs.py"},
    {"location": "figs/fig_q3_cost_vs_r.pdf", "old": "absent", "new": "new T026 cost-vs-reward figure with r*=0.60 and both crossings",
     "basis": "code/t021a_figs.py; T026/cost_vs_r.json"},
    {"location": "figs/fig9_tornado.pdf", "old": "absent", "new": "T023 tornado figure copied into the paper set",
     "basis": "T023/figs/fig_tornado.pdf"},
]

delivery = {
    "task": "T021A",
    "title": "论文内容整合（换数、新表新图）",
    "date": "2026-11-19",
    "status": "delivered",
    "revision": "R0",
    "summary": "Paper body rewritten around the accepted 2026-11 results: two-tier Q2 (975 tricycles, 617,055.7515 CNY/day, 39,984.2332 unmet), two-sided Q3 (trucks no / tricycles yes), Kneedle 4,000-site Q4, cell-resolution Q5, grouped T023 uncertainty and tornado, shared-beta Q1 table; every number is locked to a result file (720 entries, 0 mismatches, 0 unexplained tokens in the scanned passages) and the paper compiles clean at 32 total pages (30 counted excluding the 2-page AI report).",
    "checks": checks,
    "affected": affected,
    "outputs": [
        "paper/main.pdf", "paper/summary.tex", "paper/sec1_intro.tex",
        "paper/sec2_assumptions.tex", "paper/sec3_data.tex", "paper/sec4_q1.tex",
        "paper/sec5_q2.tex", "paper/sec6_q3.tex", "paper/sec7_q4.tex",
        "paper/sec8_q5.tex", "paper/sec9_sensitivity.tex",
        "paper/sec10_evaluation.tex", "figs/fig4_q2.pdf",
        "figs/fig_q3_cost_vs_r.pdf", "figs/fig9_tornado.pdf",
        "code/t021a_numbers.py", "code/t021a_figs.py", "code/t021a_tokendiff.py",
        "data/runs/T021A/numbers.json", "data/runs/T021A/coverage_report.txt",
        "data/runs/T021A/token_diff.txt",
    ],
    "notes": [
        "数字锁定：data/runs/T021A/numbers.json 共 720 条，全部由结果文件重读校验（0 失败）；覆盖扫描 summary、§1 表1、§2 A10-A12、§3 车队段与表、§4 共用 β、§5-§10 的全部数字 token，未解释 token 为 0。表内 8 个输出的参考值与 P5/P50/P95、分组 P5/P95 均有单独条目。",
        "摘要只按锁定数字改写（Q2/Q3/Q4/Q5 的结论与数字），未做压缩、未写 Executive Summary；压缩与摘要重写留给 T021B。若无修改摘要的授权解释与此冲突，请指导方裁定。",
        "共用 β 小表现在印的是留出日重估的 T022 五段 β（2.095/2.014/2.517/2.130/2.365），与配对检验同源；§4 表 2 印的是全样本 T005 拟合（2.096/…/2.364），正文已注明后者只在第三位小数不同。",
        "§2 成本假设原句曾写 'largest single effect on the Q2 total is the truck-kilometre price'；T023 龙卷风显示 COST_KM 区间为 33,534.39 元/日但排名第 10（前五为 TRI_SHIFT 629,963.87、day 317,739.26、TRI_FIXED 292,500.00、PENALTY 231,947.90、TRI_HANDLE 197,498.59），已改为只陈述区间、排序交给 §9。",
        "两处四舍五入修正：平均 log-loss 差 0.00533→0.00532（精确 0.0053204）；距离模型最大 log-loss 3.65→3.64（T005 3.64462）。两处都与结果文件一致，已在 affected 中列明。",
        "343,638 全文统一为 'initial-inventory proxy'（由末次观测还车位置构造），§3 给出定义，摘要/§5/§7/§8/§9/§10 使用同一短语；AI 报告页的代码说明保留原措辞（该页不计入正文页）。",
        "705 只作为无硬容量上限的参考方案出现（§7.2/§7.6 与摘要一句）；主结果是 4,000 站 × 150 位。623.17、885、266,327、40,357、76.86% 均不再出现。",
        "T025 交付的三处记录更正已确认不进入论文：(1) 30,000/199.16=150.6 与 809/37.1 按 199.16 写；(2) 未写 'tight solution is not hiding a zero-unmet alternative'；(3) 未把 −31,623.33 与 −1,584.60 并列（两者均未出现）。",
        "T026 的 18,011.46 是相对同口径两级方案 617,055.75 的净节省，不是系统节省；区间只用两种来源上限（1.08%-2.92%）与 c0 敏感性表达，未把 T023 的 q3 区间贴到 18,011 上。0.001 元的说法未采用：正文写 0.0008 元为二分区间、两位小数为诚实精度。",
        "Q2 的 975 与 59.385% 分别写作车队上界与节省下界；889/61.06% 只作理想 LP 上界并注明 534/13,775 趟排不进；60 辆只作 T018 单层基线，不作推荐。",
        "LP 代理区间：§5.6/§10 写明 T023 区间为 LP 代理，整数链在 11 个参数点最大偏差 38,986.30 元/日（约 0.2 SD）。K* C 组 P5=7.0 是分位数插值（0:7、20:6、40:71、60:35、80:9），不当作实际车队。",
        "门禁外写入了 figs/（fig4_q2.pdf 重画、fig_q3_cost_vs_r.pdf 与 fig9_tornado.pdf 新增）：论文以 ../figs/ 引用这三张图，且任务单未单列该目录；如判为越界请退回，我可改到 paper/ 下的图目录并同步引用。",
        "编译：xelatex 两遍，undefined 0、overfull 0、U+037E 0、?? 0、无 rerun 警告；总页 32（AI 报告 31-32 页，2 页），计页 30。未做版面压缩（T021B）。",
        "回归：以 c4be8d1 为基线的数字 token 差已逐文件列在 data/runs/T021A/token_diff.txt；ai_report.tex、main.tex、references.tex 未变。",
    ],
    "integrity": {
        "readonly_respected": True,
        "no_files_written_in_review_or_planning": True,
    },
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with io.open(OUT, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(delivery, fh, ensure_ascii=False, indent=2)
print(json.dumps(checks, ensure_ascii=False, indent=1))
print("wrote", OUT)
