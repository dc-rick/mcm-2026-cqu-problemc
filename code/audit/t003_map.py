"""T003 step 2: attach provenance to every scanned numeric token.

Consumes data/runs/T003/tokens_raw.csv and the hand-audited line rules in
t003_rules.py, aligns one rule entry per token on the same paper line (a
length mismatch is fatal), groups table tokens into cells, and writes

    交付/T003_numbers.csv        one row per in-scope token (UTF-8 BOM)
    交付/T003_delivery.json      summary, conflicts, no-code list, clue verdicts
    data/runs/T003/paper_hashes.json   paper/ integrity vs the T001 snapshot

No paper file is opened for writing and no model is run.
"""
import csv
import datetime
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code", "audit"))
import t003_rules  # noqa: E402

RUN = os.path.join(ROOT, "data", "runs", "T003")
DEL = os.path.join(ROOT, "交付")
os.makedirs(DEL, exist_ok=True)

BODY = t003_rules.BODY
FILES = (["summary.tex"]
         + [f"sec{i}_{n}.tex" for i, n in [
             (1, "intro"), (2, "assumptions"), (3, "data"), (4, "q1"),
             (5, "q2"), (6, "q3"), (7, "q4"), (8, "q5"),
             (9, "sensitivity"), (10, "evaluation")]]
         + ["ai_report.tex"])

TEX = {}
for fn in FILES:
    with open(os.path.join(ROOT, "paper", fn), encoding="utf-8") as fh:
        TEX[fn] = fh.read().splitlines()

rows = list(csv.DictReader(open(os.path.join(RUN, "tokens_raw.csv"),
                                encoding="utf-8-sig")))
scan = json.load(open(os.path.join(RUN, "scan_summary.json"),
                      encoding="utf-8"))
exc = list(csv.DictReader(open(os.path.join(RUN, "exclusions.csv"),
                               encoding="utf-8-sig")))
tcell = list(csv.DictReader(open(os.path.join(RUN, "table_cells.csv"),
                                encoding="utf-8-sig")))
LABEL = {}
for c in tcell:
    LABEL[c["table"]] = c["table_label"] or ""

# ------------------------------------------------------------------ alignment
counter = {}
out = []
errors = []
for r in rows:
    fname = os.path.basename(r["file"])
    ln = int(r["line"])
    key = (fname, ln)
    if key not in BODY:
        errors.append(f"{r['file']}:{ln} has token {r['token']} but no rule")
        continue
    i = counter.get(key, 0)
    if i >= len(BODY[key]):
        errors.append(f"{r['file']}:{ln} has more tokens than rule entries")
        continue
    e = BODY[key][i]
    counter[key] = i + 1
    raw = TEX[fname][ln - 1]
    col = int(r["col"])
    amp = raw[:col].count("&")
    cell = amp + 1 if "&" in raw else None
    out.append(dict(
        id=r["id"], token=r["token"], value=float(r["value"]),
        kind=r["kind"], file=r["file"], fname=fname, line=ln, col=col,
        table=r["table"], cell=cell, table_label=LABEL.get(r["table"], ""),
        meaning=e[0], source=e[1], result=e[2], status=e[3], note=e[4],
        conflict_id="", clue="",
    ))
for key, entries in BODY.items():
    if counter.get(key, 0) != len(entries):
        errors.append(f"{key[0]}:{key[1]} rule has {len(entries)} entries, "
                      f"tokens consumed {counter.get(key, 0)}")
if errors:
    print("ALIGNMENT ERRORS")
    for x in errors[:60]:
        print("  ", x)
    raise SystemExit(f"{len(errors)} alignment error(s)")

# ------------------------------------------------------------------ conflicts
CONFLICTS = [
    dict(
        id="C1", quantity="2017-05-16 车队估计 F=T/μ（同一日同一量）",
        a_coord="paper/sec3_data.tex:85", a_value="343,638",
        b_coord="paper/sec9_sensitivity.tex:18", b_value="343,605",
        note=("两处相差 33 辆。来源分别是 code/02_recon_spatial.py:56 的粗过滤"
              "（start/end lat>35, lon>110；F=343,637.986）与 "
              "code/60_sensitivity.py:29 的 lib 过滤（39.4<lat<40.4, "
              "115.8<lon<117.0；343,604.83）。相差的 12 条边缘记录是过滤口径差异，"
              "不是计算错误；论文两值并存的写法需要统一。"),
        occurrences=[("sec3_data.tex", 85, 343638.0),
                     ("sec3_data.tex", 96, 343638.0),
                     ("sec3_data.tex", 139, 343638.0),
                     ("sec7_q4.tex", 88, 343638.0),
                     ("sec9_sensitivity.tex", 18, 343605.0),
                     ("sec9_sensitivity.tex", 31, 343638.0),
                     ("sec9_sensitivity.tex", 103, 343638.0)],
    ),
]

_SEG_CONF = [
    ("C2a", 177, "Regular multi-purpose", "9.18", "0.92", 9.179683),
    ("C2b", 178, "Heavy short-trip", "13.98", "1.40", 13.979151),
    ("C2c", 179, "Commuter", "3.71", "0.37", 3.705500),
    ("C2d", 180, "Low frequency", "2.48", "0.25", 2.481254),
    ("C2e", 181, "Weekend / leisure", "2.25", "0.23", 2.254097),
]
_C2_R3 = (" 总量反证（R3）：seg0 有 132,984 名用户，若真按 9.18 单/天计，"
          "10 天为 132,984×9.18×10=1,220.8 万单，是 5/10–05/19 窗口"
          "全样本 246.6 万单的 4.95 倍；user_features 的每用户 10 天 9.36 单"
          "（日均 0.94）与“10 天累计”口径一致。")
for cid, ln, name, shown, perday, raw in _SEG_CONF:
    CONFLICTS.append(dict(
        id=cid, quantity=f"{name} 每用户每日出行次数（表 5 Trips per day）",
        a_coord=f"paper/sec4_q1.tex:{ln}", a_value=shown,
        b_coord=f"data/q1_segments.csv:seg={dict(zip([177, 178, 179, 180, 181],[0,3,2,1,4]))[ln]}.trips",
        b_value=f"{perday}（= {raw:.4f} / 10 天）",
        note=("code/11_q1_extra.py:72-78 的 feat.trips = 用户在该 10 天窗口内的"
              "总出行次数（g.size()），没有除以天数；表头却写 Trips per day。"
              f"窗口为 2017-05-10 至 05-19 共 10 个日历日（9 天有数据），"
              f"故每日应为 {perday}，论文值高 10 倍。"
              + ("正文 sec4_q1.tex:187/190 的“9.2 times a day”/"
                 "“14 times a day”是同一错误的复述。"
                 if cid in ("C2a", "C2b") else "")
              + _C2_R3),
        occurrences=[("sec4_q1.tex", ln, float(shown))]
                    + ([("sec4_q1.tex", 187, 9.2)] if cid == "C2a" else [])
                    + ([("sec4_q1.tex", 190, 14.0)] if cid == "C2b" else []),
    ))

_C3_NOTE = ("R1 验收：C3 降级为标签歧义，不是数值冲突。q5_summary 的 "
            "230.05/172.0/2,451/1,656.05 是单平台容量，表内 460/516/"
            "4,902/4,968 是同一地点跨平台总槽位，增量 +10.3%/+23.7% 与 "
            "+1.6%/+2.9% 正是用总量算的；正文“three platforms need 24% more "
            "parking space”也是总空间口径。建议行标签改 "
            "“Total slots per fence location”或在表注说明。")
LABEL_AMBIGUITIES = [
    dict(id="L1", coord="paper/sec8_q5.tex:45", value="460",
         field="data/q5_summary.json:capacity_duplication.median.2[0]=230.05",
         product="2×230.05=460.1", row="表 8 Median fence · 2 平台",
         note=_C3_NOTE),
    dict(id="L2", coord="paper/sec8_q5.tex:45", value="516",
         field="data/q5_summary.json:capacity_duplication.median.3[0]=172.0",
         product="3×172.0=516.0", row="表 8 Median fence · 3 平台",
         note=_C3_NOTE),
    dict(id="L3", coord="paper/sec8_q5.tex:46", value="4,902",
         field="data/q5_summary.json:capacity_duplication.largest.2[0]=2,451",
         product="2×2,451=4,902", row="表 8 Largest fence · 2 平台",
         note=_C3_NOTE),
    dict(id="L4", coord="paper/sec8_q5.tex:46", value="4,968",
         field="data/q5_summary.json:capacity_duplication.largest.3[0]=1,656.05",
         product="3×1,656.05=4,968.15", row="表 8 Largest fence · 3 平台",
         note=_C3_NOTE),
]

by_coord = {}
for r in out:
    by_coord.setdefault((r["fname"], r["line"], round(r["value"], 6)),
                        []).append(r)
for c in CONFLICTS:
    ids = []
    for fn, ln, val in c["occurrences"]:
        hit = by_coord.get((fn, ln, round(val, 6)), [])
        if not hit:
            print(f"WARNING: conflict {c['id']} occurrence not found: "
                  f"{fn}:{ln}={val}")
            continue
        for r in hit:
            r["conflict_id"] = c["id"]
            ids.append(r["id"])
    c["occurrence_ids"] = ids
    c["n_occurrence_tokens"] = len(ids)

unassigned = [r for r in out if r["status"] == "conflict" and not r["conflict_id"]]
if unassigned:
    print("WARNING: conflict rows without a conflict id:",
          [(r["file"], r["line"], r["token"]) for r in unassigned])

# ------------------------------------------------------------------ clues
CLUES = [
    dict(id="K1", title="§10.3 的节省/覆盖结论与表 9 不一致",
         verdict="部分成立",
         evidence=("表 9 的 saving 是 1−total/all-truck（口径 A），§10.3 的"
                   "“cover less than a third of the deficit”是覆盖份额（口径 B），"
                   "两者不可直接相减比较；可比处（κ=1.0 时 saving=23.8%）与"
                   "表 9 一致。表 9 的 κ=1.0（恰为基线的一半）行最优奖励仍为 "
                   "1.20 CNY，说明 §10.3 的“低于 0.6 CNY”只可能发生在 κ<1.0；"
                   "该区间没有已扫描结果（表 9 只扫 κ=1.0/3.0，基线 2.0），"
                   "0.6 CNY 无法核对。"),
         coords=["paper/sec10_evaluation.tex:61",
                 "paper/sec9_sensitivity.tex:55-61"]),
    dict(id="K2", title="表 5 的 Trips per day 单位错误",
         verdict="确认",
         evidence=("q1_segments.csv 的 trips 是 10 天窗口内每用户总出行次数，"
                   "未除以 10；表头写 per day，正文“9.2 times a day”/“14 times a day”"
                   "沿用同一错误。逐行见 C2a–C2e。"),
         coords=["paper/sec4_q1.tex:177", "paper/sec4_q1.tex:178",
                 "paper/sec4_q1.tex:179", "paper/sec4_q1.tex:180",
                 "paper/sec4_q1.tex:181", "paper/sec4_q1.tex:187",
                 "paper/sec4_q1.tex:190"]),
    dict(id="K3", title="K=60 与 K=40 是否矛盾",
         verdict="不成立",
         evidence=("Q=40 是单车容量（code/21_q2_opt.py:20 Q_TRUCK=40，"
                   "sec2_assumptions.tex:41 Q=40），K=60 是卡车数量"
                   "（sec5_q2.tex:104 表 6 行，code/21_q2_opt.py:145-148）。"
                   "两者量纲不同，不构成冲突。"),
         coords=["paper/sec2_assumptions.tex:41", "paper/sec5_q2.tex:104",
                 "paper/sec5_q2.tex:76"]),
    dict(id="K4", title="图 4(c) 成功率与正文 99.96%/98.4% 是否矛盾",
         verdict="不成立",
         evidence=("1−816.42/2,143,414=99.962%；1−34,492.53/2,143,414=98.391%。"
                   "正文两个服务率与 q2_results.json 的 unmet 完全一致"
                   "（sec5_q2.tex:136）。"),
         coords=["paper/sec5_q2.tex:136", "paper/sec5_q2.tex:108",
                 "paper/sec5_q2.tex:110"]),
    dict(id="K5", title="摘要 426k 与 427k 是否两值并存",
         verdict="不成立",
         evidence=("摘要只有 427k（summary.tex:49）；未取整值为 "
                   "343,638+83,325.24=426,963.24，四舍五入即 427k。论文没有 426k。"),
         coords=["paper/summary.tex:49"]),
    dict(id="K6", title="“ten usable days / 十个可用日”是否成立",
         verdict="部分成立",
         evidence=("抽样窗口确为 10 个日历日（ALL_DAYS=2017-05-10 至 05-19，"
                   "code/lib.py:21），但 5 月 17 日无任何订单，实际有数据的只有 "
                   "9 天（trips.parquet 逐日计数；sec3_data.tex:131 的 14 是"
                   "10–24 日整窗的有数据日数）。写成 ten usable days 不准确，"
                   "应写 ten-day window, nine days with data。"),
         coords=["paper/sec4_q1.tex:158", "paper/sec9_sensitivity.tex:131",
                 "paper/sec3_data.tex:131"]),
    dict(id="K7", title="通勤高峰份额、骑行时长、聚类稳定性",
         verdict="部分成立",
         evidence=("高峰份额 91%（sec4_q1.tex:192）有 q1_segments.csv seg2.peak="
                   "0.9104 支撑，成立；骑行时长 5.9/3.0/11.9 min（sec4_q1.tex:130）"
                   "由 code/11_q1_extra.py:88-90 打印，但未落盘到结果文件；"
                   "“stable across random initialisations”（sec4_q1.tex:158）"
                   "没有任何已保存的数值证据。"),
         coords=["paper/sec4_q1.tex:192", "paper/sec4_q1.tex:130",
                 "paper/sec4_q1.tex:158"]),
    dict(id="K8", title="已知 no-code 集合是否确实无代码/无结果字段",
         verdict="确认",
         evidence=("0.83 与 2.04（sec2:19/21，15 分钟计数方差均值比）、"
                   "172 个干涸格点（sec5:20）、Q5 标准误 4.2/7.3（sec8:66/67）、"
                   "1,200 km²（sec9:122）、约 4 km 小区尺度（sec3:113）"
                   "在现有 code/ 与 data/ 中均无产出脚本或结果字段。"),
         coords=["paper/sec2_assumptions.tex:19", "paper/sec2_assumptions.tex:21",
                 "paper/sec5_q2.tex:20", "paper/sec8_q5.tex:66",
                 "paper/sec8_q5.tex:67", "paper/sec9_sensitivity.tex:122",
                 "paper/sec3_data.tex:113"]),
]

# attach clue ids to rows by coordinate prefix
for c in CLUES:
    ids = []
    for coord in c["coords"]:
        path, rng = coord.split(":")
        fn = os.path.basename(path)
        if "-" in rng:
            a, b = [int(x) for x in rng.split("-")]
        else:
            a = b = int(rng)
        for r in out:
            if r["fname"] == fn and a <= r["line"] <= b:
                if not r["clue"]:
                    r["clue"] = c["id"]
                ids.append(r["id"])
    c["row_count"] = len(ids)
    c["row_ids"] = ids[:40]
    c["no_code_ids"] = [r["id"] for r in out
                        if r["status"] == "no-code" and r["clue"] == c["id"]]
    c["no_code_count"] = len(c["no_code_ids"])

# ------------------------------------------------------------------ summary
def status_counts(items, key="status"):
    d = {"ok": 0, "no-code": 0, "conflict": 0}
    for x in items:
        d[x[key]] = d.get(x[key], 0) + 1
    return d

totals = status_counts(out)
CONST_MARKERS = ["formula constant", "model variant definition", "表头",
                 "定义值", "编码常数", "理论常数", "geometry definition"]
const = [r for r in out if r["status"] == "ok" and
         any(m in r["result"] for m in CONST_MARKERS)]
should_cover = len(out) - len(const)
ok_data = totals["ok"] - len(const)
coverage = ok_data / should_cover if should_cover else 0.0

per_file = {}
for r in out:
    per_file.setdefault(r["file"], []).append(r)
per_file_summary = {k: dict(total=len(v), **status_counts(v))
                    for k, v in per_file.items()}

# table cells
tables = {}
for t, ts in sorted({r["table"]: [x for x in out if x["table"] == r["table"]]
                     for r in out if r["table"]}.items()):
    cells = {}
    for r in ts:
        if r["cell"] is None:
            continue
        cells.setdefault((r["line"], r["cell"]), []).append(r)
    cell_status = {"ok": 0, "conflict": 0, "no-code": 0}
    for k, cell_rows in cells.items():
        sts = {x["status"] for x in cell_rows}
        if "no-code" in sts:
            cell_status["no-code"] += 1
        elif "conflict" in sts:
            cell_status["conflict"] += 1
        else:
            cell_status["ok"] += 1
    n_cells = len(cells)
    covered_cells = n_cells - cell_status["no-code"]
    tables[t] = dict(
        label=LABEL.get(t, ""),
        tokens=len(ts), **status_counts(ts),
        caption_tokens=sum(1 for r in ts if r["cell"] is None),
        total_cells=n_cells, covered_cells=covered_cells,
        cell_ok=cell_status["ok"], cell_conflict=cell_status["conflict"],
        cell_no_code=cell_status["no-code"],
        cell_coverage=(covered_cells / n_cells if n_cells else None),
    )

no_code = [dict(id=r["id"], coord=f"{r['file']}:{r['line']}",
                token=r["token"], meaning=r["meaning"],
                source=r["source"], result=r["result"], note=r["note"],
                clue=r["clue"] or None) for r in out if r["status"] == "no-code"]

# ------------------------------------------------------------------ exclusions
EXCL_CATEGORY = [
    ("带单位长度", ("length_with_unit", "pt|cm|mm")),
    ("下标/上标索引", ("subscript_group",)),
    ("交叉引用/标签", ("label", "ref", "cite", "\\input", "includegraphics",
                    "section", "paragraph", "Section", "Table", "Fig", "§")),
    ("排版参数", ("vspace", "hspace", "setlength", "addtolength", "\\rule",
               "multicolumn", "cmidrule", "cline", "addlinespace", "captionsetup",
               "setcounter", "begin{tabular", "fontsize", "renewcommand",
               "tag{", "hfill", "quad")),
    ("时间/日期标签", ("--", "a\\.m\\.", "p\\.m\\.", "\\d{1,2}:", "January",
                   "February", "March", "April", "May", "June", "July", "August",
                   "September", "October", "November", "December", "between")),
    ("日历年", ("(?:19|20)",)),
    ("索引/百分位记号", ("dots", "percentile", "\\[\\s*\\d")),
    ("代码文件名", ("texttt",)),
]


def category(rule):
    if rule.startswith("\\\\\\\\"):
        return "排版参数"          # \\[1ex] row spacing
    for name, needles in EXCL_CATEGORY:
        if any(n in rule for n in needles):
            return name
    return "其他"


rule_counts = {}
for e in exc:
    rule_counts[e["rule"]] = rule_counts.get(e["rule"], 0) + 1
exclusion_rules = [dict(rule=k, count=v, category=category(k))
                   for k, v in sorted(rule_counts.items(),
                                      key=lambda kv: -kv[1])]

# ------------------------------------------------------------------ paper hashes
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


fp0_path = os.path.join(ROOT, "data", "runs", "T001", "fingerprint_start.json")
paper_check = {"baseline": os.path.relpath(fp0_path, ROOT), "changed": [],
               "added": [], "removed": [], "unchanged": None}
if os.path.exists(fp0_path):
    fp0 = json.load(open(fp0_path, encoding="utf-8"))
    old = {i["path"]: i["sha256"] for i in fp0["trees"]["paper"]}
    cur = {}
    paper_dir = os.path.join(ROOT, "paper")
    for dp, _, fn in os.walk(paper_dir):
        for f in fn:
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, ROOT).replace("\\", "/")
            cur[rel] = sha(p)
    paper_check["changed"] = sorted(k for k in old.keys() & cur.keys()
                                    if old[k] != cur[k])
    paper_check["added"] = sorted(cur.keys() - old.keys())
    paper_check["removed"] = sorted(old.keys() - cur.keys())
    paper_check["n_files"] = len(cur)
    paper_check["unchanged"] = not (paper_check["changed"] or
                                    paper_check["added"] or
                                    paper_check["removed"])
    json.dump({"baseline": paper_check["baseline"], "current": cur},
              open(os.path.join(RUN, "paper_hashes.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)

# ------------------------------------------------------------------ write CSV
CSV_COLS = ["id", "数值", "位置(tex:行)", "含义", "来源脚本:行",
            "结果文件:字段", "状态", "备注", "表", "单元格", "冲突编号", "线索"]
with open(os.path.join(DEL, "T003_numbers.csv"), "w", newline="",
          encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(CSV_COLS)
    for r in out:
        w.writerow([r["id"], r["token"].replace("$", ""),
                    f"{r['file']}:{r['line']}",
                    r["meaning"], r["source"], r["result"], r["status"],
                    r["note"], r["table_label"], r["cell"], r["conflict_id"],
                    r["clue"]])

# ------------------------------------------------------------------ write JSON
delivery = {
    "task": "T003",
    "title": "论文数字溯源台账",
    "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "scope": {
        "files": [f"paper/{f}" for f in FILES],
        "n_files": len(FILES),
        "in_scope_tokens": len(out),
        "excluded_tokens": len(exc),
        "exclusion_rules": exclusion_rules,
        "exclusion_rule_count": len(exclusion_rules),
    },
    "totals": {
        "in_scope_tokens": len(out),
        "ok": totals["ok"],
        "no_code": totals["no-code"],
        "conflict": totals["conflict"],
        "covered_ok": ok_data,
        "should_cover": should_cover,
        "coverage": coverage,
        "coverage_all_in_scope": totals["ok"] / len(out),
        "constants_excluded": len(const),
        "constant_filter": ("status=ok and 结果文件:字段 contains one of "
                            + ", ".join(CONST_MARKERS)),
        "denominator_reproduce": ("len(data/runs/T003/tokens_raw.csv) - "
                                  "count(marker-matched ok rows); run "
                                  "python code/audit/t003_map.py"),
        "coverage_numerator_note": ("coverage = covered_ok / should_cover = "
                                    f"{ok_data}/{should_cover}；covered_ok = "
                                    "ok − constants_excluded，"
                                    "不是 ok 本身。"),
    },
    "resubmission": {
        "round": 2,
        "changes": [
            "R1：C3a–C3d 从 conflict 降级为“标签歧义”（数值正确，行标签单数 fence 有歧义），conflict token 18→14、冲突记录 10→6。",
            "R2：totals 增加 covered_ok；覆盖率分子明确为 ok−常数，不是 ok。",
            "R3：C2 备注补总量反证（seg0 用户数×9.18×10 天 = 1,220.8 万单，为窗口全样本 246.6 万单的 4.95 倍）。",
        ],
    },
    "per_file": per_file_summary,
    "tables": tables,
    "conflicts": [{k: v for k, v in c.items() if k != "occurrences"}
                  for c in CONFLICTS],
    "label_ambiguities": LABEL_AMBIGUITIES,
    "no_code": no_code,
    "no_code_summary": {
        "total": len(no_code),
        "mapped_to_clue8": sum(1 for r in no_code if r["clue"] == "K8"),
        "outside_known_clues": sum(1 for r in no_code if not r["clue"]),
        "note": ("线索 1–7 是口径/冲突/无数值文本断言；线索 8 才是 no-code "
                 "集合。其余 no-code 行为本次新增发现：sec2:13、sec3:15、"
                 "sec3:107（95.8）、sec4:57（0.3/30.6）、"
                 "sec7:58/83/84/94/96/97/102、sec8:116（10/40）。"),
    },
    "clues": CLUES,
    "scope_notes": [
        "references.tex 不纳入扫描：其年份/卷号按任务约定整体排除。",
        "表格单元格总数只统计含至少一个数值 token 的单元格；纯标签单元格不计。",
        "同一单元格内的多个 token（如 Top-5 的 0.476 / 0.401）计为一个 cell、两个 token。",
    ],
    "paper_integrity": paper_check,
    "reproduce": {
        "scan": "python code/audit/t003_numbers.py",
        "map": "python code/audit/t003_map.py",
        "scan_outputs": ["data/runs/T003/tokens_raw.csv",
                         "data/runs/T003/exclusions.csv",
                         "data/runs/T003/table_cells.csv",
                         "data/runs/T003/scan_summary.json"],
    },
    "outputs": {
        "numbers_csv": "交付/T003_numbers.csv",
        "delivery_json": "交付/T003_delivery.json",
        "paper_hashes": "data/runs/T003/paper_hashes.json",
    },
    "method_notes": [
        "扫描对象：paper/ 下 12 个文件（summary、sec1–sec10、ai_report）；数字 token 由 code/audit/t003_numbers.py 的正则提取，覆盖科学计数法、1{,}234 千分位、维度对（如 153×153）与普通小数；逐条排除规则及计数见 scope.exclusion_rules。",
        "一个数值 token 一行；表格单元格按 (行号, 第几个 &) 归组，同一单元格内的多个 token（如 Top-5 的 0.476 / 0.401）计为一个 cell、两个 token。",
        "ok = 脚本显式计算/落盘/打印，或为有明确出处的定义参数/公式常数；no-code = 现有 code/ 与 data/ 无产出脚本或结果字段；conflict = 同一量在论文/结果字段中有两个可指认且不兼容的值（数值或口径）；标签歧义单列 label_ambiguities，不计入 conflict。",
        "constants_excluded 仅用于覆盖率分母：把结果字段为 formula constant / model variant definition / 表头 / 定义值 / 编码常数 / 理论常数 / geometry definition 的 ok 行视为无需溯源的常数；coverage_all_in_scope 为不过滤常数的覆盖率。",
        "冲突不改论文、不判定正确值；只登记两坐标与两值，交由验收方决定。",
    ],
}
with open(os.path.join(DEL, "T003_delivery.json"), "w", encoding="utf-8") as fh:
    json.dump(delivery, fh, ensure_ascii=False, indent=1)

print(f"tokens={len(out)} ok={totals['ok']} no-code={totals['no-code']} "
      f"conflict={totals['conflict']}")
print(f"should_cover={should_cover} coverage={coverage:.4f} "
      f"all_in_scope={totals['ok'] / len(out):.4f} constants={len(const)}")
print("tables:")
for t, s in tables.items():
    print(f"  {t} {s['label']}: cells {s['covered_cells']}/{s['total_cells']} "
          f"(ok {s['cell_ok']}, conflict {s['cell_conflict']}, "
          f"no-code {s['cell_no_code']}); tokens {s['tokens']}")
print(f"conflicts={len(CONFLICTS)} (tokens assigned="
      f"{sum(1 for r in out if r['conflict_id'])})")
print(f"paper unchanged={paper_check['unchanged']} "
      f"changed={paper_check['changed']} added={paper_check['added']} "
      f"removed={paper_check['removed']}")
print("wrote 交付/T003_numbers.csv and 交付/T003_delivery.json")
