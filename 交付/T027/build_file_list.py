"""Build 交付/T027/file_list.json: the submission package inventory.

Every listed file gets path (relative to the project root), SHA-256 and size.
Sections follow the task sheet: paper/*.tex, figs/*.pdf, paper/main.pdf,
code/ (all .py), data/runs/T0xx results, DATA.md, README.md.
Scratch/bench files are kept out of the formal list and appear under
non_reproduction, marked as not part of the reproduction path.
The reproduction block documents run_all.py's stage list, the dependencies
of every stage, and which stages can be skipped.

Run:  python -I 交付/T027/build_file_list.py
"""
import hashlib
import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "交付", "T027", "file_list.json")

FIG_DIR = os.path.join(ROOT, "figs")
FIG_USED = [
    "fig1_data.pdf", "fig3_q1_validation.pdf", "fig4_q2.pdf",
    "fig_q3_cost_vs_r.pdf", "fig9_tornado.pdf",
]
FIG_SUPERSEDED = ["fig2_segments.pdf", "fig5_q3.pdf", "fig6_q4.pdf"]

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
        "note": ("非复现路径：工作记录、预览文件与基准脚本，不属于复现所需，也不进正式清单。"
                 "figs/ 与 data/ 不在 T027 的可写范围（任务单只授权 paper/ 与 交付/T027/），"
                 "因此未删除，仅在此单列。"),
        "figures_png_previews": [],
        "runs_scratch": [],
        "runs_polish_preview": [],
    }
    for name in sorted(os.listdir(FIG_DIR)):
        if name.endswith(".png"):
            non_repro["figures_png_previews"].append(entry("figs/" + name))
    non_repro["runs_scratch"] = walk("data/runs", skip_polish=True)
    # keep only scratch entries there
    non_repro["runs_scratch"] = [e for e in non_repro["runs_scratch"] if "/scratch/" in e["path"]]
    non_repro["runs_polish_preview"] = walk("data/runs", skip_scratch=True)
    non_repro["runs_polish_preview"] = [e for e in non_repro["runs_polish_preview"]
                                        if "/polish_2026-11-18/" in e["path"]]

    excluded = [
        {"path": "paper/main_txt_check.txt",
         "reason": "T021A-R1 的 PDF 文本转储（1,638 行），L14 要求提交前删除；已于 T027 用 git rm 删除，内容仍可在 8708d37 取得。"},
        {"path": "code/__pycache__/*.pyc",
         "reason": "Python 字节码缓存，生成物。"},
        {"path": "paper/main.{aux,log,out,toc,lof,lot,synctex.gz}",
         "reason": "LaTeX 编译中间文件，.gitignore 排除。"},
        {"path": "paper/{b1.log,b2.log,build.log,build2.log,pg1.png,pg2.png,pg3.png,pg4.png,preview/}",
         "reason": "旧编译日志与页面预览图（工作文件，非提交物）；保留在仓库中但不入清单。"},
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
        "run_all_steps": [
            {"step": "10_q1_model.py", "writes": "data/q1_params.json, data/q1_validation.csv, data/dist_matrix.npy, data/cells.parquet",
             "depends_on": "data/trips.parquet", "timing": "未计时（估计 5--15 分钟）"},
            {"step": "11_q1_extra.py", "writes": "data/user_seg.parquet, data/q1_segments.csv, data/typical_day.npz",
             "depends_on": "data/trips.parquet, data/q1_params.json", "timing": "未计时（估计 10--30 分钟；k-means 多初值）"},
            {"step": "12_q1_od.py（不在 STEPS 中，见下）", "writes": "data/runs/T006/q1_od.npz, q1_vs_obs.json",
             "depends_on": "data/q1_params.json, data/cells.parquet, data/dist_matrix.npy",
             "timing": "未计时（估计 1--3 分钟）"},
            {"step": "20_q2_diag.py", "writes": "data/typical_day.npz 更新, data/q2_state.npz",
             "depends_on": "data/runs/T006/q1_od.npz", "timing": "未计时（估计 2--10 分钟）"},
            {"step": "21_q2_opt.py", "writes": "data/q2_solution.npz",
             "depends_on": "data/runs/T006/q1_od.npz, data/q2_state.npz", "timing": "未计时（估计 2--10 分钟）"},
            {"step": "22_q2_sim.py", "writes": "data/q2_sim.json",
             "depends_on": "data/q2_state.npz", "timing": "未计时（估计 2--10 分钟）"},
            {"step": "23_q2_saveplan.py", "writes": "计划文件（Q2 资产）",
             "depends_on": "data/q2_solution.npz", "timing": "未计时（估计 <1 分钟）"},
            {"step": "24_cell_curve.py", "writes": "细胞级曲线（Q2 资产）",
             "depends_on": "data/trips.parquet, data/q2_state.npz", "timing": "未计时（估计 <1 分钟）"},
            {"step": "30_q3_incentive.py", "writes": "data/q3_summary.json, data/q3_cell.npz, data/q3_budget.json, data/q3_potential.npz",
             "depends_on": "data/trips.parquet, data/dist_matrix.npy", "timing": "未计时（估计 5--20 分钟）"},
            {"step": "40_q4_efence.py", "writes": "data/q4_summary.json, data/q4_frontier.npz",
             "depends_on": "data/runs/T006/q1_od.npz, data/typical_day.npz", "timing": "未计时（估计 5--20 分钟）"},
            {"step": "50_q5_platform.py", "writes": "data/q5_summary.json",
             "depends_on": "data/typical_day.npz", "timing": "未计时（估计 <1 分钟）"},
            {"step": "60_sensitivity.py", "writes": "data/sensitivity.json",
             "depends_on": "data 各输入", "timing": "未计时（估计 10--40 分钟）"},
            {"step": "90_figs.py", "writes": "figs/fig1_data.pdf, figs/fig3_q1_validation.pdf（并重写已被取代的 fig5_q3.pdf、fig6_q4.pdf）",
             "depends_on": "data/q3_summary.json, data/q3_cell.npz, data/q4_summary.json, data/q4_frontier.npz（30/40 阶段输出）及 Q1 文件",
             "timing": "未计时（估计 <1 分钟）"},
            {"step": "95_numbers.py", "writes": "控制台数字表",
             "depends_on": "data/runs/T001R2、T002、T005、T006、T007、T008、T009，data/typical_day.npz、q1_params.json、q1_segments.csv、q5_summary.json",
             "timing": "未计时（估计 <1 分钟）"},
        ],
        "stage_12_gap": (
            "run_all.py 的 STEPS 不含 12_q1_od.py，但 20_q2_diag.py、21_q2_opt.py、40_q4_efence.py "
            "都读 data/runs/T006/q1_od.npz，t018_q2_int.py 也读它。因此从头复现必须在 11 与 20 之间"
            "手动运行 12_q1_od.py；否则 20 阶段即失败。"),
        "task_scripts_before_95": (
            "95_numbers.py 还读 data/runs/T001R2、T002、T005、T007、T008、T009 的结果，这些目录"
            "由 run_all.py 之外的脚本产出：audit/t001r2_*.py、audit/t002_*.py、t005_*.py、"
            "t007_q2_run.py、t008_q3_pool.py、t009_q4.py（T006 由 12_q1_od.py 产出）。从头复现时"
            "须先运行它们（其顺序为：12 -> t007 -> t008 -> t009；T001R2/T002 读 data/ 即可），"
            "否则 95 阶段失败。"),
        "final_round_order": [
            {"step": "t018_q2_int.py", "reads": "data/runs/T007 的 K=60 LP 计划、T006 q1_od、data/",
             "writes": "data/runs/T018/", "timing_recorded": "整数 MILP：K=20 8.68 s、K=40 8.70 s、K=60 83.97 s、K=80 68.57 s；LP 0.3--0.5 s/K；日循环变体 379.85 s"},
            {"step": "t019_q4_cap.py", "reads": "data/，以及 data/runs/T009 的 q4_results.json/q4_fences.csv（比较与目标覆盖）",
             "writes": "data/runs/T019/", "timing_recorded": "分阶段约：demand 60 s、solve 7 s、radius 8 s、replay 40 s、walk 5 s、denominator 150 s、summary 8 s、check 2 s（合计约 280 s）"},
            {"step": "t020_q3_budget.py", "reads": "data/runs/T019/spare_capacity.npz 等", "writes": "data/runs/T020/",
             "timing_recorded": "单策略 99.87 s"},
            {"step": "t022_fleet.py / t022_shared_beta.py / t022_m15_q5_mismatch.py", "reads": "data/ only",
             "writes": "data/runs/T022/", "timing_recorded": "m16 48 s、m12 54.7 s、m15 23.0 s"},
            {"step": "t024_q2_tier2.py", "reads": "data/runs/T018/（import t018_q2_int）", "writes": "data/runs/T024/",
             "timing_recorded": "fleet 3,050 个 LP 约 140 s；schedule 约 60 s；其余阶段未计"},
            {"step": "t025_q3_scarcity.py", "reads": "data/runs/T018、T020（import t018_q2_int/t020_q3_budget）",
             "writes": "data/runs/T025/", "timing_recorded": "fleet 218.9 s、hybrid 665.1 s、plans 81.1 s、mechanism 3.1 s（合计约 968 s）"},
            {"step": "t023_uncertainty.py", "reads": "data/runs/T018、T019、T024、T025 的结果键（见脚本头注释）",
             "writes": "data/runs/T023/", "timing_recorded": "1,078 个 job 全量 4,845.5 s（约 81 分钟）"},
            {"step": "t023_figs.py", "reads": "data/runs/T023/{tornado,variance_shares,intervals}.json",
             "writes": "data/runs/T023/figs/", "timing_recorded": "未计时（估计 <1 分钟）"},
            {"step": "t026_incentive_tricycle.py", "reads": "data/runs/T024/schedule_check.json；import t008_q3_pool、t018_q2_int、t024_q2_tier2",
             "writes": "data/runs/T026/", "timing_recorded": "未计时"},
            {"step": "t021a_figs.py", "reads": "data/runs/T018/{schedule_K60.csv,baselines.json}、T024/compare.json、T026/{cost_vs_r,optimum}.json、T023/figs/fig_tornado.pdf",
             "writes": "figs/fig4_q2.pdf、figs/fig_q3_cost_vs_r.pdf、figs/fig9_tornado.pdf", "timing_recorded": "未计时（估计 <1 分钟）"},
            {"step": "t021a_numbers.py", "reads": "data/runs/T018、T019、T020、T023、T024、T025、T026，以及 T004/sim_results.json",
             "writes": "data/runs/T021A/numbers.json（724 条）", "timing_recorded": "未计时（估计 <1 分钟）"},
            {"step": "xelatex main.tex（三遍）", "reads": "paper/*.tex、figs/*.pdf", "writes": "paper/main.pdf",
             "timing_recorded": "未计时（三遍约 1--2 分钟）"},
        ],
        "dependency_answers": {
            "which_stages_read_T018_T026": "无。对 run_all.py 的全部 STEPS 脚本与 12_q1_od.py 逐一 grep T018--T026：0 命中；95_numbers.py 也不读 T018--T026（它读 T001R2--T009 与 data/ 下四个文件）。final-round 脚本（t021a_numbers.py 等）读 T018--T026 的结果，方向相反。",
            "skippable": (
                "22_q2_sim.py 与 60_sensitivity.py 可跳：它们分别写 data/q2_sim.json 与 data/sensitivity.json，"
                "而 95_numbers.py、t021a_numbers.py 与 90_figs.py 均不读这两个文件（grep 0），论文引用的数字"
                "不来自它们。其余阶段不建议跳：90_figs.py 读 30/40 阶段写出的 q3/q4 数据文件（缺了会失败），"
                "50 阶段写 95_numbers.py 要读的 q5_summary.json，11 阶段写 95/90 要读的 q1_segments.csv。"),
            "requires_stated_order": "是。脚本靠 data/ 下的中间文件串联（q1_params.json、typical_day.npz、q2_state.npz、q1_od.npz 等），不是单命令可复现。",
        },
    }

    doc = {
        "task": "T027",
        "date": "2026-11-19",
        "baseline": "8708d37",
        "hash": "sha256 (over the bytes on disk as delivered; for tracked text files Git stores LF-normalised blobs with core.autocrlf=true, so a fresh checkout may hash differently)",
        "counts": {k: len(v) for k, v in sections.items()},
        "counts_note": "figures_pdf 含 8 个 PDF：5 个论文实际引用（used_in_paper=true），3 个为被取代的旧图（false），保留但不引用。",
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
