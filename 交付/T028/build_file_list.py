"""Build 交付/T028/file_list.json: the submission package inventory.

Every listed file gets path (relative to the project root), SHA-256 and size.
Sections follow the task sheet: paper/*.tex, figs/*.pdf, paper/main.pdf,
code/ (all .py), data/runs/T0xx results, DATA.md, README.md.
Scratch/page-measurement files are kept out of the formal list and appear under
non_reproduction.  The reproduction block documents the base pipeline
(run_all.py plus the preparation scripts) and the final-round conserving chain
that produces every number the paper quotes.

Run:  python -I 交付/T028/build_file_list.py
"""
import hashlib
import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "交付", "T028", "file_list.json")

FIG_DIR = os.path.join(ROOT, "figs")
FIG_USED = [
    "fig1_data.pdf", "fig3_q1_validation.pdf", "fig4_q2.pdf",
    "fig_q3_cost_vs_r.pdf",
]
FIG_SUPERSEDED = ["fig2_segments.pdf", "fig5_q3.pdf", "fig6_q4.pdf",
                  "fig9_tornado.pdf"]

EXCLUDE_DIRS = {"__pycache__", ".claude"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entry(rel):
    p = os.path.join(ROOT, rel.replace("/", os.sep))
    return {"path": rel, "sha256": sha256(p), "bytes": os.path.getsize(p)}


def walk(rel_dir, suffix=None, skip_scratch=False, skip_polish=False):
    out = []
    base = os.path.join(ROOT, rel_dir)
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS)
        if skip_scratch and "scratch" in dirnames:
            dirnames.remove("scratch")
        rel = os.path.relpath(dirpath, ROOT).replace(os.sep, "/")
        if skip_polish and rel.startswith("data/runs/polish_2026-11-18"):
            continue
        for name in sorted(filenames):
            if suffix and not name.endswith(suffix):
                continue
            out.append(entry(rel + "/" + name))
    return sorted(out, key=lambda e: e["path"])


def main():
    sections = {}

    tex = sorted(os.listdir(os.path.join(ROOT, "paper")))
    sections["paper_tex"] = [entry("paper/" + n) for n in tex if n.endswith(".tex")]
    sections["paper_pdf"] = [entry("paper/main.pdf")]

    figs = []
    for name in sorted(os.listdir(FIG_DIR)):
        if name.endswith(".pdf"):
            e = entry("figs/" + name)
            e["used_in_paper"] = name in FIG_USED
            figs.append(e)
    sections["figures_pdf"] = figs

    sections["code"] = walk("code", suffix=".py")
    sections["results"] = walk("data/runs", skip_scratch=True, skip_polish=True)
    sections["documents"] = [entry("DATA.md"), entry("README.md")]

    # per-run counts for navigation
    runs = {}
    for e in sections["results"]:
        parts = e["path"].split("/")
        run = parts[2] if len(parts) > 3 else parts[-1]
        runs[run] = runs.get(run, 0) + 1

    non_repro = {
        "note": ("非复现路径：工作记录、页面测量转储与预览图，不属于复现所需，也不进正式清单。"
                 "交付前已从工作区删除页面测量文件；figs/ 中被取代的旧图保留，仅在此单列。"),
        "figures_png_previews": [],
        "runs_scratch": [],
        "runs_polish_preview": [],
    }
    for name in sorted(os.listdir(FIG_DIR)):
        if name.endswith(".png"):
            non_repro["figures_png_previews"].append(entry("figs/" + name))
    non_repro["runs_scratch"] = walk("data/runs", skip_polish=True)
    non_repro["runs_scratch"] = [e for e in non_repro["runs_scratch"] if "/scratch/" in e["path"]]
    non_repro["runs_polish_preview"] = walk("data/runs", skip_scratch=True)
    non_repro["runs_polish_preview"] = [e for e in non_repro["runs_polish_preview"]
                                        if "/polish_2026-11-18/" in e["path"]]

    excluded = [
        {"path": "paper/pgall*.txt, paper/pg_*.txt, paper/blocks.txt, paper/bnd*.txt, paper/fill*.txt, paper/p*.txt, paper/pages_dump.txt, paper/main_txt.txt",
         "reason": "页面测量与 PDF 文本转储（工作文件），交付前删除。"},
        {"path": "paper/page*.png",
         "reason": "页面预览渲染图（工作文件），交付前删除。"},
        {"path": "code/__pycache__/*.pyc",
         "reason": "Python 字节码缓存，生成物。"},
        {"path": "paper/main.{aux,log,out,toc,lof,lot,synctex.gz}",
         "reason": "LaTeX 编译中间文件，.gitignore 排除。"},
        {"path": "data/*(顶层)",
         "reason": "数据输入与派生大文件（trips.parquet、dist_matrix.npy、cells.parquet、typical_day.npz、q1_params.json、q5_summary.json 等）。.gitignore 排除，不随仓库分发；来源与重建方式见 DATA.md。"},
    ]

    reproduction = {
        "entry_point": "code/run_all.py",
        "statement": (
            "run_all.py 依 STEPS 顺序执行 13 个编号阶段，遇第一个失败即停止；"
            "00--04 的数据准备不在其中。所有脚本以 python -I 运行，"
            "lib.py 以自身位置解析 data/ 与 figs/，可从任意当前目录启动。"
            "data/ 不随仓库分发（见 DATA.md），故本清单为 data/runs/ 下每个文件附 SHA-256。"),
        "preparation_not_in_run_all": [
            {"stage": "00_prepare.py", "role": "清洗原始订单文件 -> data/trips.parquet",
             "timing": "未计时（估计 5--15 分钟，191 MB CSV）"},
            {"stage": "01_recon.py", "role": "数据概况诊断（只打印，不写数据文件）",
             "timing": "未计时（估计 <1 分钟）"},
            {"stage": "02_recon_spatial.py", "role": "空间/时间结构诊断（只打印）",
             "timing": "未计时（估计 1--5 分钟）"},
            {"stage": "03_diag.py", "role": "诊断地图（只输出图）",
             "timing": "未计时（估计 1--5 分钟）"},
            {"stage": "04_behaviour.py", "role": "行为诊断；写 data/user_features.parquet",
             "timing": "未计时（估计 1--5 分钟）"},
            {"stage": "build_dist_matrix.py", "role": "从 data/cells.parquet 重建 5,263x5,263 距离矩阵；T015 核对与仓库外原文件逐位相同",
             "timing": "未计时（估计 1--3 分钟）"},
        ],
        "stage_12_gap": (
            "run_all.py 的 STEPS 不含 12_q1_od.py，但 20_q2_diag.py、21_q2_opt.py、40_q4_efence.py "
            "都读 data/runs/T006/q1_od.npz，t018_q2_int.py 也读它。因此从头复现必须在 11 与 20 之间"
            "手动运行 12_q1_od.py；否则 20 阶段即失败。"),
        "final_round_chain": {
            "statement": (
                "基准管线（run_all.py 的 13 个阶段与 12_q1_od.py）只用来产出 data/ 下的输入与 "
                "T018/T024 的两层计划。论文 Q2--Q5 与敏感性一节引用的全部数字来自下面的守恒链，"
                "按此顺序运行；链上脚本均以 python -I 运行，可在任意当前目录启动。"
                "链的输出取代基准管线中 30_q3_incentive.py、40_q4_efence.py、50_q5_platform.py、"
                "60_sensitivity.py、90_figs.py 的对应结果；这些阶段仍须先跑，因为链读 data/ 的派生文件。"),
            "steps": [
                {"step": "t028_q2_conserve.py",
                 "role": "守恒联合重放（到达由已服务出发生成，全市总量固定 343,638）+ 整数三轮车逐区坐标下降至可执行不动点",
                 "reads": "data/（typical_day.npz、q1_od.npz 等）、data/runs/T018（K=60 计划）、data/runs/T024（二层计划）",
                 "writes": "data/runs/T028/（local_search.json、zero_baselines.json、iterate.json、diag.json、start_scale110.json、日志）",
                 "timing_recorded": "坐标下降 90 个候选车队逐次 runtime_s 合计约 40,490 s（单次 1.7--1,352 s）"},
                {"step": "t035_q2_curve.py",
                 "role": "Q2 联合成本—车队规模曲线（图 3 面板 d）",
                 "reads": "data/runs/T028/local_search.json",
                 "writes": "data/runs/T035/joint_curve.json",
                 "timing_recorded": "未计时（日志 curve.log）"},
                {"step": "t030_q4_fleet.py",
                 "role": "Q4 守恒需求重建（15 分钟步进、到达由已服务出发生成）、Kneedle 拐点、上限整数选址与重放",
                 "reads": "data/（q1_od.npz、typical_day.npz、dist_matrix.npy）、import t019_q4_cap（仅加 m.dispose()）",
                 "writes": "data/runs/T030/（cell_demand_cons.npz/json、frontier_cons.json、plan_meas343_main.npz、plan_fleet495_all.npz、replay_cons.json 等）",
                 "timing_recorded": "需求阶段 meas343 11.3 s、fleet495 11.9 s（demand.log）；frontier/replay 未计时"},
                {"step": "t029_q5_outside.py",
                 "role": "Q5 两平台失配实验，显式围栏外库存状态（occ/out 双状态）",
                 "reads": "data/、data/runs/T030/plan_meas343_main.npz 与 cell_demand_cons.npz",
                 "writes": "data/runs/T029/（m15_outside.json、m15_outside_capacity.npz、run_t30.log）",
                 "timing_recorded": "18.1 s（run_t30.log）"},
                {"step": "t031_q3_payment.py",
                 "role": "Q3 支付台账刻度：卡车刻度静态池阶梯（基于 T030 围栏容量）+ 三轮车刻度守恒重放（1,233 固定，重解可执行车队）",
                 "reads": "data/runs/T030/（plan_meas343_main.npz、spare_capacity）、data/runs/T018（K=60 单位成本）、T028 重放模块",
                 "writes": "data/runs/T031/（truck_payment.json、tricycle_payment.json、scan_arrivals.json、scan_surplus.json、tricycle.log）",
                 "timing_recorded": "truck 6.9 s、tricycle 297.8 s（结果文件 runtime_s）"},
                {"step": "t033_q4_radius.py",
                 "role": "Q4 步行半径扫描（守恒需求，解析 + 已存计划）",
                 "reads": "data/runs/T030（cell_demand_cons.npz、plan_meas343_main.npz）",
                 "writes": "data/runs/T033/radius_cons.json",
                 "timing_recorded": "未计时"},
                {"step": "t039_q3_c0.py",
                 "role": "Q3 接受常数 c0 扫描（0.8x/1x/1.2x，支付台账与增量台账并列）",
                 "reads": "data/runs/T030/（计划与容量）",
                 "writes": "data/runs/T039/q3_c0.json",
                 "timing_recorded": "9.6 s（结果文件 runtime_s）"},
                {"step": "t032_sensitivity_conserve.py",
                 "role": "敏感性重跑：T023 的 128 组联合抽样 + 128 A + 128 C + 54 B，逐抽样在守恒链上重解 Q2/Q3/Q4",
                 "reads": "data/runs/T023（设计，只读）、data/runs/T030/frontier_cons.json、data/runs/T031/truck_payment.json",
                 "writes": "data/runs/T032/（evals.jsonl、intervals.json、variance_shares.json、reference.json、run.log）",
                 "timing_recorded": "全量 10,253.5 s（run.log）；reference 上下文 55.8 s"},
                {"step": "t041_km_oat.py",
                 "role": "Q2 卡车公里单价单因素检验（在 T032 参考上下文重解 50 区 LP，COST_KM=3/6/9）",
                 "reads": "data/runs/T032/reference.json 的上下文、import t023_uncertainty",
                 "writes": "data/runs/T041/km_oat.json",
                 "timing_recorded": "3 次 LP 合计 2.9 s（结果文件 runtime_s）"},
                {"step": "t036_figs.py",
                 "role": "重绘 fig4_q2.pdf、fig_q3_cost_vs_r.pdf、fig6_q4.pdf（图 1、2 未随守恒链变化，由 90_figs.py 产出）",
                 "reads": "data/runs/T028、T030、T031、T035（及 Q1 结果）",
                 "writes": "figs/fig4_q2.pdf、figs/fig_q3_cost_vs_r.pdf、figs/fig6_q4.pdf（fig6 未入论文）",
                 "timing_recorded": "未计时（估计 <1 分钟）"},
                {"step": "t038_numbers.py",
                 "role": "数字锁定：展平全部结果，加入带算式的派生记录，逐 token 匹配论文；写 numbers.json 与 coverage_report.txt",
                 "reads": "data/runs/T001R2--T041 的结果文件与 paper/*.tex",
                 "writes": "data/runs/T038/（numbers.json、coverage_report.txt）",
                 "timing_recorded": "未计时（估计 <1 分钟）"},
                {"step": "t037_digest.py / t042_sec9_tables.py / t040_citecheck.py",
                 "role": "只读核对：分区数字表、§9 两表按印刷精度重排、\\cite/\\bibitem 一致性",
                 "reads": "data/runs/ 结果文件与 paper/*.tex", "writes": "无（只打印）",
                 "timing_recorded": "未计时"},
                {"step": "xelatex main.tex（三遍）",
                 "role": "编译论文", "reads": "paper/*.tex、figs/*.pdf", "writes": "paper/main.pdf",
                 "timing_recorded": "未计时（三遍约 1--2 分钟）"},
            ],
        },
        "dependency_answers": {
            "which_stages_read_the_final_chain": "无。run_all.py 的 13 个阶段与 12_q1_od.py 均不读 T028--T041 的结果（方向相反：链读它们的输出）。因此基准管线可独立重跑，链需在此基础上按序运行。",
            "skippable": (
                "22_q2_sim.py 与 60_sensitivity.py 可跳：它们分别写 data/q2_sim.json 与 data/sensitivity.json，"
                "论文引用的数字不来自它们（Q2 来自 T028，敏感性来自 T032）。t036_figs.py 的 fig6_q4.pdf "
                "未入论文，可跳；图 1、2 必须由 90_figs.py 生成。"),
            "requires_stated_order": "是。基准管线靠 data/ 下的中间文件串联；守恒链读 T028/T030/T031/T032 的结果，顺序不可换。",
        },
    }

    doc = {
        "task": "T028",
        "date": "2026-11-20",
        "base_commit": "508a66f",
        "hash": "sha256 (over the bytes on disk as delivered; for tracked text files Git stores LF-normalised blobs with core.autocrlf=true, so a fresh checkout may hash differently)",
        "counts": {k: len(v) for k, v in sections.items()},
        "counts_note": "figures_pdf 含 8 个 PDF：4 个论文实际引用（used_in_paper=true），4 个为被取代的旧图（false），保留但不引用。",
        "runs_breakdown": runs,
        "sections": sections,
        "non_reproduction": non_repro,
        "excluded": excluded,
        "reproduction": reproduction,
    }
    doc["total_files"] = sum(doc["counts"].values())

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print(json.dumps({"counts": doc["counts"], "total_files": doc["total_files"],
                      "non_reproduction": {k: len(v) for k, v in non_repro.items() if isinstance(v, list)}},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
