"""T021A number lock: build data/runs/T021A/numbers.json.

Every number printed in the rewritten passages of the paper (T021A scope:
summary, Table 1, sec2 A10-A12, sec3 fleet paragraph, sec4 shared-beta
subsection, sec5-sec10) is listed as

    token  ->  file :: key  (-> scale)  ->  caliber

Entries are verified by re-reading the result file and checking that the
printed token is the correct rounding of the stored value.  Derived entries
carry their arithmetic (`inputs` + `formula`).  A coverage pass scans the
changed .tex files for numeric tokens and reports any token that no entry
explains, so the uncovered list can be triaged (years, section numbers,
method constants) instead of overlooked.

Run:  python -I code/t021a_numbers.py
"""
import io
import json
import math
import os
import re
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "data", "runs", "T021A", "numbers.json")
LOG = os.path.join(ROOT, "data", "runs", "T021A", "coverage_report.txt")

ENTRIES = []


def P(rel):
    return os.path.join(ROOT, rel)


def add(token, file, key, caliber, scale=1.0, where=""):
    ENTRIES.append(dict(token=token, file=file, key=key, scale=scale,
                        caliber=caliber, where=where))


def addd(token, inputs, formula, caliber, where=""):
    ENTRIES.append(dict(token=token, inputs=inputs, formula=formula,
                        caliber=caliber, where=where))


def addc(token, tag, caliber, where=""):
    ENTRIES.append(dict(token=token, computed=tag, caliber=caliber,
                        where=where))


def addconst(token, value, caliber, where=""):
    ENTRIES.append(dict(token=token, const=value, caliber=caliber,
                        where=where))


# --------------------------------------------------------------------------
# Q2 (sec5) -- T018 truck layer + T024 tricycle layer
# --------------------------------------------------------------------------
S18 = "data/runs/T018/baselines.json"
S18C = "data/runs/T018/constraint_check.json"
S18H = "data/runs/T018/holdout.json"
S24 = "data/runs/T024/compare.json"
S24S = "data/runs/T024/schedule_check.json"
S24D = "data/runs/T024/dyn_baseline.json"
S24U = "data/runs/T024/cell_unmet.json"
T18 = "data/runs/T018/result_lp_K60.json"
T4 = "data/runs/T004/sim_results.json"

add("343,638", S18, "design/I0", "期初库存代理（5/16 00:00 还车位置构造），T018 设计输入", where="5.1")
add("5.84%", T4, "results/coarse_343638_R450_a1.0/mean",
    "无调度动态模拟在 343,638 与 R=450 m 下的未服务占比（T004 模拟均值）", scale=100.0, where="5.1")
add("3.51%", T4, "results/coarse_495540_R450_a1.0/mean",
    "同上，495,540 车队口径", scale=100.0, where="5.1")
add("239,245.20", S24D, "cell_unmet",
    "T018 K=60 卡车计划下的格级动态未服务（运营口径上界）", where="5.1")
addd("11.22%", [(S24D, "cell_unmet", 1.0), (S24D, "trips", 1.0)],
     "100.0 * v0 / v1", "动态格级未服务占全部离场 2,132,542 的比例", where="5.1")
add("2,132,542", S24D, "trips", "典型日全部离场（1/p 放大后），T024 动态回放分母", where="5.1")
add("115,129.19", S24U, "static/total_strict",
    "静态小时自由重分下界（不可运营达到）", where="5.1")
add("5.40%", S24U, "static/strict_share_of_trips",
    "静态下界占全部离场的比例", scale=100.0, where="5.1")
addd("1.875%", [(S24S, "exec_totals/unmet", 1.0), (S24D, "trips", 1.0)],
     "100.0 * v0 / v1", "两级方案执行后未服务占全部离场", where="5.1")
addc("290", "graph_arcs", "50 区、8 近邻、9 km 阈值下的有向弧数（由典型日质心重算）", where="5.2")
addc("5.97", "graph_mean_km", "上述弧的平均长度 km（由典型日质心重算）", where="5.2")
add("3", S18C, "old_lp_T007_K60/zone_hours_violated",
    "旧 T007 LP 计划自身需求下违反事件顺序的区-小时数", where="5.2")
add("1,200", S18C, "old_lp_T007_K60/zone_hours_total",
    "区-小时总数 50x24", where="5.2")
add("717.54", S18C, "old_lp_T007_K60/total_gap_bikes",
    "旧计划违反量合计（辆）", where="5.2")
add("40", S18C, "load/max_load_bikes", "整数计划最大装载（辆/车），等于 Q", where="5.2")
add("46.92", S18C, "duration/max_trip_minutes", "整数计划最长单车行程分钟", where="5.2")
add("0.81", S18C, "event_order/max_gap_bikes",
    "事件顺序最小松弛（取绝对值；原值为负）", scale=-1.0, where="5.2")
add("1,891.36", S18C, "inventory_nonneg/min_inventory", "全时全最小库存（辆）", where="5.2")
add("82,968.0671", T18, "obj", "K=60 LP 松弛目标值", where="5.2")
add("83,807.7438", S18, "recommended/obj", "K=60 整数 MILP 目标值", where="5.2")
addd("839.68", [(S18, "integer_vs_lp/K60/int_obj", 1.0),
                (S18, "integer_vs_lp/K60/lp_obj", 1.0)],
     "v0 - v1", "整数-LP 目标差（元/日）", where="5.2")
add("1.01%", S18, "integer_vs_lp/K60/diff_pct", "整数-LP 相对差距", where="5.2")
add("1,023.76", S18C, "end_positions/deadhead_km", "日末空驶里程（不可入目标）", where="5.2")
add("6,142.58", S18C, "end_positions/deadhead_cost_yuan", "日末空驶成本", where="5.2")

for k, tok_m, tok_km, tok_un, tok_dev, tok_obj in [
        ("K20", "9,503", "1,767.02", "95.01", "28,083.10", "112,172.98"),
        ("K40", "19,104", "4,097.39", "0", "8,907.88", "84,860.02"),
        ("K60", "24,949", "5,659.17", "0", "459.41", "83,807.74"),
        ("K80", "24,863", "5,495.85", "0", "586.96", "95,167.44")]:
    add(tok_m, S18, "int_rows/%s/moved" % k, "整数扫描 %s 搬运量（辆/日）" % k, where="5.3")
    add(tok_km, S18, "int_rows/%s/truck_km" % k, "整数扫描 %s 卡车公里" % k, where="5.3")
    add(tok_un, S18, "int_rows/%s/unmet" % k, "整数扫描 %s 区级未服务（辆/日）" % k, where="5.3")
    add(tok_dev, S18, "int_rows/%s/dev" % k, "整数扫描 %s 日末偏差" % k, where="5.3")
    add(tok_obj, S18, "int_rows/%s/obj" % k, "整数扫描 %s 目标（元/日）" % k, where="5.3")
add("432.23", S18, "rows/no_trucks/unmet", "K=0 区级未服务（辆/日）", where="5.3")
add("46,930.39", S18, "rows/no_trucks/dev", "K=0 日末偏差", where="5.3")
add("143,384.53", S18, "rows/no_trucks/cost", "K=0 目标（元/日）", where="5.3")
addd("11,359.70", [(S18, "int_rows/K80/obj", 1.0), (S18, "int_rows/K60/obj", 1.0)],
     "v0 - v1", "K=80 相对 K=60 增加的目标", where="5.3")
addd("86", [(S18, "int_rows/K60/moved", 1.0), (S18, "int_rows/K80/moved", 1.0)],
     "v0 - v1", "K=80 少搬运的辆数", where="5.3")

for day, ratio, unmet, dev in [("2017-05-12", "0.9291", "704.74", "53,255.64"),
                               ("2017-05-19", "1.1066", "14,232.37", "51,718.20")]:
    add(ratio, S18H, "rows/%s/ratio_to_design" % day, "留出日 %s 相对设计日运量比" % day, where="5.3")
    add(unmet, S18H, "rows/%s/unmet" % day, "留出日 %s 未服务（辆/日）" % day, where="5.3")
    add(dev, S18H, "rows/%s/dev" % day, "留出日 %s 日末偏差" % day, where="5.3")
addd("0.60%", [(S18H, "rows/2017-05-19/unmet", 1.0),
               (S18H, "rows/2017-05-19/observed_departures_full", 1.0)],
     "100.0 * v0 / v1", "19 May 未服务占当日离场比例", where="5.3")

add("20", S24, "params/Q", "三轮车容量 Q_tri（辆）", where="5.4")
add("15", S24, "params/V", "三轮车速度 km/h", where="5.4")
add("0.5", S24, "params/handle_min", "每辆装卸分钟", where="5.4")
add("5", S24, "params/max_trip", "单程最大公里", where="5.4")
add("300", S24, "params/fixed", "三轮车日固定成本（元/车·日）", where="5.4")
add("0.2", S24, "params/km", "三轮车公里成本（元/km）", where="5.4")
add("889", S24, "best_used", "LP 最优三轮车数（理想上界，非方案）", where="5.4")
add("507,779.67", S24, "best_layer_cost", "LP 最优三轮车层成本（元/日）", where="5.4")
add("40,041.58", S24, "rows/9/cell_unmet", "LP 最优点的格级未服务（理想上界）", where="5.4")
add("591,587.42", S24, "rows/9/total", "LP 上界总成本（卡车 K=60 + 三轮车层）", where="5.4")
addd("927,691.53", [(S24, "no_tricycle/total", 1.0), (S24, "rows/9/total", 1.0)],
     "v0 - v1", "LP 上界相对格级基线节省", where="5.4")
addd("61.06%", [(S24, "no_tricycle/total", 1.0), (S24, "rows/9/total", 1.0)],
     "100.0 * (v0 - v1) / v0", "LP 上界节省率", where="5.4")
add("1,519,278.95", S24, "no_tricycle/total", "无三轮车格级基线（元/日）", where="5.4")
add("534", S24S, "city_totals/unassigned", "LP 分配下的整数排班排不进趟数", where="5.4")
add("13,775", S24S, "city_totals/trips", "LP 分配下的总趟数", where="5.4")
add("15", S24S, "city_totals/zones_with_unassigned", "出现排不进的区数", where="5.4")
add("975", S24S, "exec_totals/n_exec", "可执行固定点三轮车队（辆）", where="5.4")
addd("86", [(S24S, "exec_totals/n_exec", 1.0), (S24S, "city_totals/tricycles", 1.0)],
     "v0 - v1", "固定点相对 LP 整数分配抬升的车辆数", where="5.4")

add("533,248.01", S24S, "exec_totals/layer_cost_fullload", "执行层满载口径成本", where="5.4")
addd("293,342.61", [(S24S, "exec_totals/layer_cost_fullload", 1.0),
                    (S24S, "exec_totals/unmet", 1.0)],
     "v0 - 6.0 * v1", "固定+公里+装卸成本（满载口径）", where="5.4")
addd("239,905.40", [(S24S, "exec_totals/unmet", 1.0)],
     "6.0 * v0", "未服务罚金合计（6 元/辆）", where="5.4")
add("617,055.7515", S24S, "exec_totals/total_fullload", "两级方案总成本（满载口径，主数）", where="5.4")
add("617,055.75", S24S, "exec_totals/total_fullload",
    "同上，§5 表外文字四舍五入到分", where="5.4")
add("902,223.20", S24S, "exec_totals/saving_fullload", "相对无三轮车基线的节省（满载口径）", where="5.4")
add("59.385%", S24S, "exec_totals/saving_fullload_share", "节省率（满载口径）", scale=100.0, where="5.4")
add("534,278.98", S24S, "exec_totals/layer_cost_actual", "执行层实际里程口径成本", where="5.4")
add("618,086.73", S24S, "exec_totals/total_actual", "两级方案总成本（实际里程口径）", where="5.4")
add("901,192.22", S24S, "exec_totals/saving_actual", "节省（实际里程口径）", where="5.4")
add("59.317%", S24S, "exec_totals/saving_actual_share", "节省率（实际里程口径）", scale=100.0, where="5.4")
addd("1,030.98", [(S24S, "exec_totals/total_actual", 1.0),
                 (S24S, "exec_totals/total_fullload", 1.0)],
     "v0 - v1", "两种口径总成本差", where="5.4")
addd("0.17%", [(S24S, "exec_totals/total_actual", 1.0),
              (S24S, "exec_totals/total_fullload", 1.0)],
     "100.0 * (v0 - v1) / v1", "两种口径相对差", where="5.4")
add("1.47215", S24S, "exec_totals/penalty_ceiling_exec", "保持方案最优的未服务罚金下限（元/辆）", where="5.4")
add("1,225.36", S24S, "exec_totals/fixed_ceiling_exec", "保持方案最优的三轮车固定成本上限（元/车·日）", where="5.4")
add("38,489.44", S24S, "exec_totals/unmet_lp", "执行分配下 LP 未服务（整数舍入对照）", where="5.4")
add("39,984.2332", S24S, "exec_totals/unmet", "执行方案未服务离场（主数，整数调度口径）", where="5.4")
add("39,984.23", S24S, "exec_totals/unmet", "同上，§5.6 文字四舍五入", where="5.4")
add("1,494.80", S24S, "exec_totals/int_short", "整数舍入损失（辆）", where="5.4")
add("44", S24S, "tricycles_allocated", "区 24 三轮车分配（最大区）", where="5.4")
add("2,059.49", S24S, "lp_unmet_zone", "区 24 LP 未服务", where="5.4")
add("2,085.98", S24S, "int_unmet_zone", "区 24 整数排班未服务", where="5.4")
add("26.50", S24S, "rounding_loss", "区 24 舍入损失", where="5.4")
add("0.43%", S24S, "rounding_share", "区 24 舍入损失占该区未服务比例", scale=100.0, where="5.4")
add("81.5", S24S, "rounding_bound", "区 24 舍入损失的可行上界", where="5.4")
addd("1,435,471.21", [(S24U, "static/total_strict", 0.0)],
     "6.0 * 239245.20107233778", "卡车-only 格级罚金（6 元 x 239,245.2011）", where="5.5")
addd("59,576.79", [(S18, "rows/no_trucks/cost", 1.0), (S18, "int_rows/K60/obj", 1.0)],
     "v0 - v1", "面板 A 卡车单层节省", where="5.5")
addd("41.55%", [(S18, "rows/no_trucks/cost", 1.0), (S18, "int_rows/K60/obj", 1.0)],
     "100.0 * (v0 - v1) / v0", "面板 A 节省率", where="5.5")

# T023 Q2 intervals (sec5.6, sec9.2, sec10)
add("617,056", "data/runs/T023/intervals.json", "outputs/q2_total/reference",
    "Q2 总成本参考值（规划中点，非期望/中位数）", where="5.6")
for tok, key in [("271,732", "outputs/q2_total/slices/F/p5"),
                 ("882,188", "outputs/q2_total/slices/F/p95"),
                 ("484,372", "outputs/q2_total/slices/F/p50"),
                 ("2,789", "outputs/q2_unmet/slices/F/p5"),
                 ("73,744", "outputs/q2_unmet/slices/F/p95"),
                 ("11,392", "outputs/q2_unmet/slices/F/p50"),
                 ("0.594", "outputs/q2_saving_share/reference"),
                 ("-0.461", "outputs/q2_saving_share/slices/F/p5"),
                 ("0.692", "outputs/q2_saving_share/slices/F/p95"),
                 ("0.260", "outputs/q2_saving_share/slices/F/p50")]:
    add(tok, "data/runs/T023/intervals.json", key, "T023 联合抽样 F 的 P5/P50/P95", where="5.6")
addc("24%", "neg_saving_share", "F 抽样中节省率为负的比例 31/128", where="5.6")
addc("31", "neg_saving_count", "F 抽样中两级方案劣于基线的次数", where="5.6")
add("128", "data/runs/T023/intervals.json", "outputs/q2_total/slices/F/n", "联合抽样次数", where="5.6")
add("0", "data/runs/T023/intervals.json", "outputs/q2_kstar/slices/F/p5", "K* 联合 P5", where="5.6")
add("60", "data/runs/T023/intervals.json", "outputs/q2_kstar/slices/F/p95", "K* 联合 P95 / 参考值", where="5.6")
add("40", "data/runs/T023/intervals.json", "outputs/q2_kstar/slices/F/p50", "K* 联合中位数", where="5.6")
for tok, kstar in [("8", "0"), ("37", "20"), ("72", "40"), ("9", "60"), ("2", "80")]:
    add(tok, "data/runs/T023/intervals.json",
        "outputs/q2_kstar/slices/F/distribution/%s" % kstar,
        "K* 联合分布计数（K*= %s 的抽样次数）" % kstar, where="5.6")
addc("11", "kstar_ge60", "K* >= 60 的联合抽样数（9 + 2）", where="5.6")
addc("8.6%", "kstar_ge60_share", "K* >= 60 的比例 11/128", where="5.6")
addc("91%", "kstar_change_share", "改变 K* 的联合抽样比例 1 - 11/128", where="5.6")
add("40", "data/runs/T023/intervals.json", "outputs/q2_kstar/slices/A/p5", "A 组 K* P5（A: 40-60）", where="5.6")
add("60", "data/runs/T023/intervals.json", "outputs/q2_kstar/slices/A/p95", "A 组 K* P95", where="5.6")
add("20", "data/runs/T023/intervals.json", "outputs/q2_kstar/slices/B/p5", "B 组 K* P5（B: 20-60）", where="5.6")
add("60", "data/runs/T023/intervals.json", "outputs/q2_kstar/slices/B/p95", "B 组 K* P95", where="5.6")
add("7", "data/runs/T023/intervals.json", "outputs/q2_kstar/slices/C/p5", "C 组 K* P5（C: 7-80，分位数插值）", where="5.6")
add("80", "data/runs/T023/intervals.json", "outputs/q2_kstar/slices/C/p95", "C 组 K* P95", where="5.6")
add("38,986.30", "data/runs/T023/variance_shares.json", "proxy_deviation/q2_total_dev_max",
    "LP 代理 vs 整数链在 11 个参数点的最大偏差（元/日）", where="5.6")
addd("0.2", [("data/runs/T023/variance_shares.json", "proxy_deviation/q2_total_dev_max", 1.0),
             ("data/runs/T023/intervals.json", "outputs/q2_total/slices/F/sd", 1.0)],
     "v0 / v1", "最大代理偏差相对 Q2 总成本标准差（约 0.2 SD）", where="5.6")
addd("61.06%", [(S24, "no_tricycle/total", 1.0), (S24, "rows/9/total", 1.0)],
     "100.0 * (v0 - v1) / v0", "LP 上界节省率（5.6 引用）", where="5.6")

# --------------------------------------------------------------------------
# Q3 (sec6) -- T025 truck caliber + T026 tricycle caliber + T020 ledgers
# --------------------------------------------------------------------------
S25 = "data/runs/T025/breakeven.json"
S25H = "data/runs/T025/hybrid_by_K.json"
S25F = "data/runs/T025/fleet_scan.json"
S26 = "data/runs/T026/optimum.json"
S26S = "data/runs/T026/sensitivity.json"
S20 = "data/runs/T020/policy_table.json"
S20H = "data/runs/T020/hard_budget.json"

add("33.19", S25, "per_K/1/pure_unmet", "K=2 纯卡车区级未服务（辆/日）", where="6.3")
for i, tok in enumerate(["-1,739.22", "-1,591.40", "-1,588.76", "-1,623.33", "-1,528.82"]):
    add(tok, S25, "per_K/%d/best_policy/payment_ceiling" % i,
        "K=%s 付款上限（负=无正奖励可回本）" % [0, 2, 5, 10, 20][i], where="6.3")
add("1,591.40", S25, "per_K/1/best_policy/payment_ceiling",
    "K=2 付款上限的绝对值（正文以正数引用其量级）", scale=-1.0, where="6.3")
addd("1,532", [(S25, "per_K/1/best_policy/payment_ceiling", 1.0),
               (S25, "per_K/1/best_policy/gap", 1.0),
               (S25, "per_K/1/best_policy/truck_obj", 1.0)],
     "-(v0 + v1 * v2)", "扣除混合解自身 gap 后的零付款劣势下界（元/日）", where="6.3")
addd("59.06", [(S25, "per_K/1/best_policy/gap", 1.0),
               (S25, "per_K/1/best_policy/truck_obj", 1.0)],
     "v0 * v1", "混合解自身求解 gap（元）", where="6.3")
addd("138.32", [(S25, "per_K/1/gap_used", 1.0), (S25, "per_K/1/pure_obj", 1.0)],
     "v0 * v1", "0.1% 求解容差对应的元值", where="6.3")
addd("11.1", [(S25, "per_K/1/best_policy/payment_ceiling", 1.0),
              (S25, "per_K/1/best_policy/gap", 1.0),
              (S25, "per_K/1/best_policy/truck_obj", 1.0)],
     "-(v0 + v1 * v2) / 138.317151154932", "1,532 相对 138.32 的倍数", where="6.3")
addd("150.6", [(S25, "per_K/1/best_policy/budget", 1.0),
               (S25, "per_K/1/pure_unmet", 1.0)],
     "v0 / (6.0 * v1)", "预算相对全部未服务罚金的倍数", where="6.3")
addd("199.16", [(S25, "per_K/1/pure_unmet", 1.0)], "6.0 * v0",
     "K=2 全部未服务罚金（元）", where="6.3")
add("134.81", S25F, "gap_tight/K/2/coarse/unmet", "K=2 1% 上界未服务（辆/日）", where="6.3")
addd("809", [(S25F, "gap_tight/K/2/coarse/unmet", 1.0)], "6.0 * v0",
     "1% 上界未服务的罚金（元）", where="6.3")
addd("37.1", [(S25, "per_K/1/best_policy/budget", 1.0),
              (S25F, "gap_tight/K/2/coarse/unmet", 1.0)],
     "v0 / (6.0 * v1)", "预算相对 809 元罚金的倍数", where="6.3")
add("312,289.21", S25, "mechanism/zero_unmet_policy/total", "零未服务政策总成本（元/日）", where="6.3")
addd("2.2578", [(S25, "mechanism/zero_unmet_policy/total", 1.0),
                (S25, "per_K/1/pure_obj", 1.0)],
     "v0 / v1", "零未服务政策相对纯卡车成本倍数", where="6.3")
add("10.4", S25, "mechanism/unmet_removed", "K=2 两个返回解之差（辆），仅作差、不作效应", where="6.3")
addd("66.13", [(S25F, "gap_tight/K/2/tight/gap", 1.0),
               (S25F, "gap_tight/K/2/tight/obj", 1.0)],
     "v0 * v1", "纯卡车返回解自身 gap（元）", where="6.3")
addd("11", [(S25F, "gap_tight/K/2/tight/gap", 1.0),
            (S25F, "gap_tight/K/2/tight/obj", 1.0)],
     "v0 * v1 / 6.0", "66.13 元约合 11 辆", where="6.3")
addd("10", [(S25, "per_K/1/best_policy/gap", 1.0),
            (S25, "per_K/1/best_policy/truck_obj", 1.0)],
     "v0 * v1 / 6.0", "59.06 元约合 10 辆", where="6.3")

addconst("0.05", 0.05, "奖励网格步长 0.05 元（政策参数）", where="6.4")
add("30,000", S25, "per_K/1/best_policy/budget", "日预算承诺（元）", where="6.4")
add("0.60", S26, "caps/arrivals/r_star", "Q3 三轮车口径最优奖励（元/改道）", where="6.4")
add("27,680.86", S26, "caps/arrivals/q_star", "最优改道数（辆/日）", where="6.4")
add("16,608.52", S26, "caps/arrivals/payment_star", "增量账付款（元/日）", where="6.4")
add("13,391.48", S26, "caps/arrivals/budget_slack_star", "预算余量（元/日）", where="6.4")
add("599,044.29", S26, "caps/arrivals/total_star", "最优时总成本（元/日）", where="6.4")
add("18,011.46", S26, "caps/arrivals/net_saving_star", "相对同口径两级方案净节省（元/日）", where="6.4")
addd("2.92%", [(S26, "caps/arrivals/net_saving_star", 1.0),
               (S26, "baseline_total", 1.0)], "100.0 * v0 / v1", "净节省率", where="6.4")
add("498,628.03", S26, "caps/arrivals/layer_star", "最优时三轮车层成本", where="6.4")
add("34,619.98", S26, "caps/arrivals/layer_saving_star", "三轮车层节省", where="6.4")
add("4,833.20", S26, "caps/arrivals/fixed_km_saving_star", "固定+公里+装卸节省", where="6.4")
add("959", S26, "caps/arrivals/n_total_star", "最优时可执行三轮车队（辆）", where="6.4")
addd("16", [(S24S, "exec_totals/n_exec", 1.0), (S26, "caps/arrivals/n_total_star", 1.0)],
     "v0 - v1", "车队减少量（975-959）", where="6.4")
add("29,786.77", S26, "caps/arrivals/penalty_saving_star", "罚金节省（6 元 x 未服务净减少）", where="6.4")
add("35,019.77", S26, "caps/arrivals/unmet_star", "最优时未服务（辆/日）", where="6.4")
add("4,964.46", S26, "caps/arrivals/net_unmet_reduction_star", "未服务净减少（辆/日）", where="6.4")
add("0.179", S26, "caps/arrivals/unmet_reduction_per_q_star", "每改道净减少未服务（辆）", where="6.4")
add("0.175", S26, "caps/arrivals/fixed_km_saving_per_q_star", "每改道固定+公里节省（元）", where="6.4")
add("0.974", S26, "caps/arrivals/fixed_km_saving_per_removed_unmet_star",
    "每净减少未服务的固定+公里节省（元）", where="6.4")
add("24,141.94", S26, "caps/arrivals/bound_prediction_star", "线性界预测节省（元/日）", where="6.4")
add("6,130.48", S26, "caps/arrivals/bound_gap_star", "精确回放与线性界之差（元/日）", where="6.4")
add("0.55", S26, "caps/surplus/r_star_plateau/0", "保守口径最优平台下端", where="6.4")
add("1.15", S26, "caps/surplus/r_star_plateau/8", "保守口径最优平台上端", where="6.4")
add("4,503.02", S26, "caps/surplus/q_star", "保守口径改道数", where="6.4")
add("969", S26, "caps/surplus/n_total_star", "保守口径可执行车队", where="6.4")
add("38,726.29", S26, "caps/surplus/unmet_star", "保守口径未服务", where="6.4")
add("2,701.81", S26, "caps/surplus/payment_star", "保守口径付款", where="6.4")
add("6,651.22", S26, "caps/surplus/net_saving_star", "保守口径净节省", where="6.4")
addd("1.08%", [(S26, "caps/surplus/net_saving_star", 1.0),
               (S26, "baseline_total", 1.0)], "100.0 * v0 / v1",
     "保守口径净节省率", where="6.4")
add("1.038281", S26, "caps/arrivals/r_cross", "到达口径盈亏交叉奖励（二分区间 0.0008）", where="6.4")
add("1.627344", S26, "caps/surplus/r_cross", "保守口径盈亏交叉奖励", where="6.4")
add("1.04", S26, "caps/arrivals/r_cross", "交叉奖励两位小数（诚实精度）", where="6.4")
add("1.63", S26, "caps/surplus/r_cross", "保守交叉奖励两位小数", where="6.4")
add("0.0008", S26, "caps/arrivals/cross/width", "二分区间宽度（元），不可当结果精度", where="6.4")
add("98.05", S26, "caps/arrivals/low_r_bump/0/excess",
    "r=0.05 时总成本高于无激励水平（非单调，源格损失先于填谷）", where="6.4")

for factor, idx, c0, r, q, pay, n, unm, sav in [
        (0.8, "0", "0.64", "0.55", "27,778.22", "15,278.02", "958", "35,002.21", "19,747.42"),
        (1.2, "2", "0.96", "0.70", "31,039.47", "21,727.63", "958", "34,491.76", "16,365.88")]:
    add(c0, S26S, "rows/%s/c0" % idx, "c0 敏感性（倍率 %s）" % factor, where="6.4")
    add(r, S26S, "rows/%s/r_star" % idx, "c0=%s 最优奖励" % c0, where="6.4")
    add(q, S26S, "rows/%s/q_star" % idx, "c0=%s 改道数" % c0, where="6.4")
    add(pay, S26S, "rows/%s/payment_star" % idx, "c0=%s 付款" % c0, where="6.4")
    add(n, S26S, "rows/%s/n_total_star" % idx, "c0=%s 车队" % c0, where="6.4")
    add(unm, S26S, "rows/%s/unmet_star" % idx, "c0=%s 未服务" % c0, where="6.4")
    add(sav, S26S, "rows/%s/net_saving_star" % idx, "c0=%s 净节省" % c0, where="6.4")
add("0.80", S26S, "c0_base", "参考 c0", where="6.4")

add("0.81", S20, "levels/arrivals/rows/0/r", "T020 30,000 预算的付款账最优 r", where="6.5")
add("28,884.65", S20, "levels/arrivals/rows/0/q", "r=0.81 的填谷改道数", where="6.5")
add("23,396.57", S20, "levels/arrivals/rows/0/bill_old", "r=0.81 的增量账账单", where="6.5")
add("37,037.04", S20, "levels/arrivals/rows/0/P", "r=0.81 的应付人数 P", where="6.5")
addd("6,603.43", [(S20, "levels/arrivals/rows/0/payment", 1.0),
                  (S20, "levels/arrivals/rows/0/bill_old", 1.0)],
     "v0 - v1", "付款账与增量账之差（搭便车者）", where="6.5")
add("100,000", S20, "levels/arrivals/rows/2/budget", "预算阶梯端点点", where="6.5")
add("1.35", S20, "levels/arrivals/rows/2/r", "100,000 预算最优 r", where="6.5")
add("66,596.28", S20, "levels/arrivals/rows/2/q", "100,000 预算改道数", where="6.5")
add("73,428.10", S20, "levels/arrivals/rows/2/P", "100,000 预算应付人数", where="6.5")
add("99,127.94", S20, "levels/arrivals/rows/2/payment", "100,000 预算付款", where="6.5")
add("1.74", S20, "levels/arrivals/rows/4/r", "无预算约束最优 r（付款账）", where="6.5")
add("93,328.82", S20, "levels/arrivals/rows/4/q", "无预算约束改道数", where="6.5")
add("99,038.48", S20, "levels/arrivals/rows/4/P", "无预算约束 P", where="6.5")
add("172,326.96", S20, "levels/arrivals/rows/4/payment", "无预算约束付款", where="6.5")

add("1,000", S20H, "sim_days", "硬预算规则的模拟天数", where="6.5")
add("0", S20H, "all_zero_overrun", "模拟中的超预算天数（布尔 0 表示零超支；见下条说明）", scale=0.0, where="6.5")
add("7", S20H, "rows/0/stop_hour", "B=30,000 r=0.81 停发小时", where="6.5")
add("30.9%", S20H, "rows/0/issued_fraction", "B=30,000 发出的邀请比例", scale=100.0, where="6.5")
add("3,634.53", S20H, "rows/0/truncated_q", "截断后的改道数", where="6.5")
add("3,774.88", S20H, "rows/0/truncated_payment", "截断后的付款", where="6.5")
add("3,932.55", S20H, "rows/0/sim_max_payment", "模拟最大付款", where="6.5")
add("603,232.87", S26, "h26/low/total", "H26 低端 r=0.81 总成本", where="6.5")
add("1,272,910.24", S26, "h26/high/total", "H26 高端 r=2.70 总成本", where="6.5")
addd("669,677.37", [(S26, "h26/high/total", 1.0), (S26, "h26/low/total", 1.0)],
     "v0 - v1", "两个奖励的总成本差", where="6.5")
add("83,807.74", S26, "h26/low/truck_layer", "H26 卡车层（与两级方案相同）", where="6.5")
addd("784,302.68", [(S26, "h26/high/payment_uncapped", 1.0),
                    (S26, "h26/low/payment_uncapped", 1.0)],
     "v0 - v1", "r=2.70 多付的奖励", where="6.5")
addd("114,625.31", [(S26, "h26/low/tricycle_layer", 1.0),
                    (S26, "h26/high/tricycle_layer", 1.0)],
     "v0 - v1", "r=2.70 在三轮车层节省的金额", where="6.5")
add("821,422.84", S26, "h26/high/payment_uncapped", "r=2.70 无上限付款", where="6.5")
addd("27.4", [(S26, "h26/high/payment_uncapped", 1.0), (S26, "budget", 1.0)],
     "v0 / v1", "r=2.70 付款相对预算倍数", where="6.5")
add("37,120.16", S26, "h26/low/payment_uncapped", "r=0.81 无上限付款", where="6.5")
addd("1.24", [(S26, "h26/low/payment_uncapped", 1.0), (S26, "budget", 1.0)],
     "v0 / v1", "r=0.81 付款相对预算倍数", where="6.5")
addd("0.60", [(S26, "caps/arrivals/r_star", 1.0)], "v0", "全局三轮车口径最优仍为 r*=0.60", where="6.5")

addd("176.28", [(S26, "caps/surplus/cross/evals/5/total", 1.0),
                (S26, "baseline_total", 1.0)],
     "v0 - v1", "T026 评审注记：交叉区间端点总成本高出无激励水平 176.28 元；"
     "'0.001 元' 指 r 的二分区间宽度，不是成本精度", where="6.4")

# --------------------------------------------------------------------------
# Q4 (sec7) -- T019 + T009
# --------------------------------------------------------------------------
S19 = "data/runs/T019/summary.json"
S19M = "data/runs/T019/marginal_curve.json"
S19F = "data/runs/T019/frontier.json"
S19R = "data/runs/T019/radius_sweep.json"
S19P = "data/runs/T019/replay.json"
S19D = "data/runs/T019/denominator.json"
S09O = "data/runs/T009/q4_overlap.json"
S09V = "data/runs/T009/q4_results.json"

add("300", S19, "denominator_conservativeness/reps", "容量 P95 的蒙特卡洛重复次数", where="7.1")
addconst("95", 95, "容量定义为 15 分钟占用路径峰值的 95 分位（方法定义）", where="7.1")
add("974,923.16", S19, "denominator_conservativeness/meas343/sum_cell_p95",
    "343,638 下逐格 P95 之和（不可加）", where="7.1")
add("630,002.78", S19, "denominator_conservativeness/meas343/system_p95_total_occupancy",
    "系统总占用同时 P95", where="7.1")
add("0.646", S19, "denominator_conservativeness/meas343/ratio", "可加性比", where="7.1")
add("1,123,702.61", S19, "denominator_conservativeness/fleet495/sum_cell_p95",
    "495,540 下逐格 P95 之和", where="7.1")
add("525,987.17", S19, "denominator_conservativeness/fleet495/system_p95_total_occupancy",
    "495,540 系统总占用 P95", where="7.1")
add("0.468", S19, "denominator_conservativeness/fleet495/ratio", "可加性比", where="7.1")
add("495,540", S09V, "validation/initial_inventory/fleet495_sum", "闭合车队场景库存", where="7.1")
add("5,263", S19F, "meas343/n_cells", "格数（候选站点）", where="7.2")
addconst("450", 450, "步行半径规划阈值 R=450 m（政策参数）", where="7.2")
addconst("63.2%", 63.2, "贪心极大覆盖的 (1-1/e) 保证", where="7.2")
add("4,000", S19, "levels/meas343/selected_n", "Kneedle 膝点站点数", where="7.2")
add("0.21829", S19, "levels/meas343/knee/deviation_normalised", "膝点归一化弦距", where="7.2")
add("0.20982", S19, "levels/fleet495/knee/deviation_normalised", "495,540 膝点弦距", where="7.2")
addc("1.5", "marg_1500", "3,000-3,500 每 100 站覆盖增量（百分点）", where="7.2")
addc("1.19", "marg_134", "3,500-4,000 每 100 站覆盖增量（由 frontier coverage 差计算）", where="7.2")
addc("0.26", "marg_072", "4,000-4,500 每 100 站增量", where="7.2")
addc("0.05", "marg_016", "4,500-5,000 每 100 站增量", where="7.2")

addc("582.72", "uncapped_median", "705 参考方案逐站 95 分位容量中位数", where="7.3")
addc("2,946.51", "uncapped_max", "705 参考方案最大单站容量", where="7.3")
addc("704", "uncapped_over150", "705 参考方案中容量超过 150 的站数", where="7.3")
add("705", S19, "old_705_uncapped_reference/n", "T009 无硬上限 Kneedle 膝点（仅作容量参考）", where="7.3")
add("10.85%", S19, "levels/meas343/t009_705_sites_cap150/coverage",
    "705 站加 150 上限后的占用覆盖（343,638）", scale=100.0, where="7.3")
add("14.66%", S19, "levels/meas343/t009_705_sites_cap150/arrival_coverage",
    "705 站加 150 上限后的到达覆盖（343,638）", scale=100.0, where="7.3")
add("9.41%", S19, "levels/fleet495/t009_705_sites_cap150/coverage",
    "同上（495,540）", scale=100.0, where="7.3")
add("13.08%", S19, "levels/fleet495/t009_705_sites_cap150/arrival_coverage",
    "同上（495,540）", scale=100.0, where="7.3")
add("600,000", S19, "levels/meas343/slots", "4,000 x 150 总车位", where="7.3")
add("59.53%", S19, "levels/meas343/occupancy_coverage", "占用覆盖率（343,638）", scale=100.0, where="7.3")
add("55.80%", S19, "levels/meas343/arrival_coverage", "到达覆盖率（343,638）", scale=100.0, where="7.3")
add("580,370.03", S19, "levels/meas343/assigned", "已分配需求（343,638）", where="7.3")
add("394,553.13", S19, "levels/meas343/unassigned", "未分配需求（343,638）", where="7.3")
add("52.18%", S19, "levels/fleet495/occupancy_coverage", "占用覆盖率（495,540）", scale=100.0, where="7.3")
add("49.59%", S19, "levels/fleet495/arrival_coverage", "到达覆盖率（495,540）", scale=100.0, where="7.3")
add("586,368.97", S19, "levels/fleet495/assigned", "已分配需求（495,540）", where="7.3")
add("537,333.64", S19, "levels/fleet495/unassigned", "未分配需求（495,540）", where="7.3")
add("0.0049", S19F, "meas343/selected/gap", "站点整数 MILP gap（343,638）", where="7.3")
add("0.0050", S19F, "fleet495/selected/gap", "站点整数 MILP gap（495,540）", where="7.3")
add("61.10%", S19, "levels/meas343/max_occupancy_coverage_all_sites",
    "全开 5,263 站加 150 上限的占用覆盖", scale=100.0, where="7.3")
add("56.75%", S19, "levels/meas343/max_arrival_coverage_all_sites",
    "全开到达覆盖", scale=100.0, where="7.3")
addconst("789,450", 5263 * 150, "5,263 x 150 总车位（算术）", where="7.3")

for n, slot, occ, arr, idx in [(2000, "300,000", "30.71%", "26.17%", 12),
                               (3000, "450,000", "46.08%", "42.67%", 14),
                               (4500, "675,000", "60.83%", "56.24%", 17)]:
    add("%d" % n, S19M, "meas343/n/%d" % idx, "前沿点站点数", where="7.3")
    addconst(slot, n * 150, "前沿点总车位（算术）", where="7.3")
    add(occ, S19M, "meas343/coverage/%d" % idx, "前沿点占用覆盖", scale=100.0, where="7.3")
    add(arr, S19M, "meas343/arrival_coverage/%d" % idx, "前沿点到达覆盖", scale=100.0, where="7.3")
add("0.387", "data/runs/T023/intervals.json", "outputs/q4_coverage/slices/F/p5",
    "T023 Q4 占用覆盖联合 P5", where="7.3")
add("0.824", "data/runs/T023/intervals.json", "outputs/q4_coverage/slices/F/p95",
    "联合 P95（<1，容量上限始终起作用）", where="7.3")
add("0.625", "data/runs/T023/intervals.json", "outputs/q4_coverage/slices/F/p50",
    "联合 P50", where="7.3")
add("0.406", "data/runs/T023/intervals.json", "outputs/q4_arrival/slices/F/p5",
    "Q4 到达覆盖联合 P5", where="7.3")
add("0.835", "data/runs/T023/intervals.json", "outputs/q4_arrival/slices/F/p95",
    "到达覆盖联合 P95", where="7.3")
add("0.663", "data/runs/T023/intervals.json", "outputs/q4_arrival/slices/F/p50",
    "到达覆盖联合 P50", where="7.3")
add("491,692", S19, "old_705_uncapped_reference/slots_meas343",
    "705 参考方案 343,638 所需车位", where="7.3")
add("621,004", S19, "old_705_uncapped_reference/slots_fleet495",
    "705 参考方案 495,540 所需车位", where="7.3")
add("501", S09V, "frontier_summary/q1/knees/meas343/0/n",
    "λ=0 时无上限前沿膝点站点数", where="7.3")
add("64.51%", S09V, "frontier_summary/q1/knees/meas343/0/coverage",
    "λ=0 膝点覆盖", scale=100.0, where="7.3")
addconst("150", 150, "单站容量上限 λ=150（政策参数）", where="7.3")
addconst("300", 300, "λ=300 的无上限前沿（膝点仍为 705）", where="7.3")
add("0.7685848", S19, "old_705_uncapped_reference/coverage_arrivals",
    "T009 705 站到达覆盖率（无上限目标）", where="7.3")

add("55.80%", S19P, "meas343/design/realized_coverage", "设计日回放到达覆盖（15 分钟步长）", scale=100.0, where="7.4")
add("0", S19P, "meas343/design/overflow_arrivals", "设计日溢出（辆）", where="7.4")
add("132,250.84", S19P, "meas343/design/departure_shortfall", "设计日离场缺口", where="7.4")
add("55.73%", S19P, "meas343/design_1h/realized_coverage", "设计日 1 小时步长覆盖", scale=100.0, where="7.4")
add("1,617.94", S19P, "meas343/design_1h/overflow_arrivals", "1 小时步长溢出", where="7.4")
add("158,257.23", S19P, "meas343/design_1h/departure_shortfall", "1 小时步长离场缺口", where="7.4")
add("0.08", S19P, "meas343/design_1h/delta_realized_coverage", "覆盖下降（百分点）", scale=-100.0, where="7.4")
add("49.59%", S19P, "fleet495/design/realized_coverage", "495,540 设计日覆盖", scale=100.0, where="7.4")
add("4,277.25", S19P, "fleet495/design/departure_shortfall", "495,540 设计日缺口", where="7.4")
add("49.46%", S19P, "fleet495/design_1h/realized_coverage", "495,540 1 小时覆盖", scale=100.0, where="7.4")
add("2,709.12", S19P, "fleet495/design_1h/overflow_arrivals", "495,540 1 小时溢出", where="7.4")
add("38,829.33", S19P, "fleet495/design_1h/departure_shortfall", "495,540 1 小时缺口", where="7.4")
for day, cov, ovf, short, full, dlt, dshort in [
        ("2017-05-12", "50.33%", "104,551.04", "123,832.54", "1,007", "-0.25", "22,736.57"),
        ("2017-05-19", "50.35%", "126,920.81", "152,736.70", "1,110", "-0.41", "35,096.42")]:
    add(cov, S19P, "meas343/%s/realized_coverage" % day, "测试日 %s 覆盖" % day, scale=100.0, where="7.4")
    add(ovf, S19P, "meas343/%s/overflow_arrivals" % day, "测试日 %s 溢出" % day, where="7.4")
    add(short, S19P, "meas343/%s/departure_shortfall" % day, "测试日 %s 缺口" % day, where="7.4")
    add(full, S19P, "meas343/%s/full_sites_ever" % day, "测试日 %s 满位站数" % day, where="7.4")
    add(dlt, S19P, "meas343/%s/hourly/delta_realized_coverage" % day,
        "测试日 %s 1 小时步长覆盖差（百分点）" % day, scale=100.0, where="7.4")
    add(dshort, S19P, "meas343/%s/hourly/delta_departure_shortfall" % day,
        "测试日 %s 1 小时步长缺口增加" % day, where="7.4")

for tok, key, cal in [
        ("52.42%", "meas343/radii/150/coverage", "R=150 占用覆盖"),
        ("57.00%", "meas343/radii/150/arrival_coverage", "R=150 到达覆盖"),
        ("0.1454", "meas343/radii/150/comply_zero_incentive", "R=150 零激励合规率"),
        ("7.62%", "meas343/radii/150/effective_coverage", "R=150 有效覆盖"),
        ("8.29%", "meas343/radii/150/effective_arrival_coverage", "R=150 有效到达覆盖"),
        ("0.0605", "meas343/radii/300/comply_zero_incentive", "R=300 合规率"),
        ("3.17%", "meas343/radii/300/effective_coverage", "R=300 有效覆盖"),
        ("3.45%", "meas343/radii/300/effective_arrival_coverage", "R=300 有效到达覆盖"),
        ("0.0238", "meas343/radii/450/comply_zero_incentive", "R=450 合规率"),
        ("1.42%", "meas343/radii/450/effective_coverage", "R=450 有效覆盖"),
        ("1.33%", "meas343/radii/450/effective_arrival_coverage", "R=450 有效到达覆盖"),
        ("46.02%", "fleet495/radii/150/coverage", "495,540 R=150 占用覆盖"),
        ("53.86%", "fleet495/radii/150/arrival_coverage", "495,540 R=150 到达覆盖"),
        ("6.69%", "fleet495/radii/150/effective_coverage", "495,540 R=150 有效覆盖"),
        ("7.83%", "fleet495/radii/150/effective_arrival_coverage", "495,540 R=150 有效到达"),
        ("2.78%", "fleet495/radii/300/effective_coverage", "495,540 R=300 有效覆盖"),
        ("3.26%", "fleet495/radii/300/effective_arrival_coverage", "495,540 R=300 有效到达"),
        ("1.24%", "fleet495/radii/450/effective_coverage", "495,540 R=450 有效覆盖"),
        ("1.18%", "fleet495/radii/450/effective_arrival_coverage", "495,540 R=450 有效到达")]:
    add(tok, S19R, key, cal, scale=100.0 if "%" in tok else 1.0, where="7.5")

add("2,020", S09O, "q1/selected/cells_1", "705 参考方案：仅落入 1 个汇水区的格数", where="7.6")
add("44", S09O, "q1/selected/cells_2", "落入 2 个的格数", where="7.6")
add("0", S09O, "q1/selected/cells_3plus", "落入 3 个以上的格数", where="7.6")
add("2,064", S09O, "q1/selected/cells_covered", "被覆盖格数", where="7.6")
add("3,199", S09O, "q1/selected/cells_0", "未覆盖格数", where="7.6")
add("1.0213", S09O, "q1/selected/coverage_multiplicity", "覆盖多重度", where="7.6")
add("12,280", S09O, "q1/selected/duplicate_capacity_slots", "朴素汇水区求和重复计车位", where="7.6")
add("2.80%", S09O, "q1/selected/duplicate_capacity_share", "重复计比例", scale=100.0, where="7.6")
add("1,965", S09O, "obs/selected/cells_1", "观测面：1 个汇水区的格数", where="7.6")
add("45", S09O, "obs/selected/cells_2", "观测面：2 个", where="7.6")
add("1.0224", S09O, "obs/selected/coverage_multiplicity", "观测面多重度", where="7.6")
add("3.09%", S09O, "obs/selected/duplicate_capacity_share", "观测面重复计比例", scale=100.0, where="7.6")

add("0.804", S09V, "validation/cell_corr", "留出日重建设施的格相关", where="7.7")
add("0.423", S09V, "validation/l1_normalized", "留出日 L1", where="7.7")
add("32", S09V, "validation/top100_overlap", "前 100 吸引点重合数", where="7.7")
add("0.687", S09V, "validation/initial_inventory/corr", "库存形状相关", where="7.7")
add("0.922", S09V, "validation/initial_inventory/l1_normalized", "库存形状 L1", where="7.7")
add("1,097", "data/runs/T022/m15_mismatch.json", "control_proportional_zone/osm_entrances",
    "OSM 地铁出入口数（外部验证）", where="7.7")
add("8.2%", "data/runs/T022/m15_mismatch.json", "control_proportional_zone/baseline_near_entrance",
    "300 m 内有出入口的格比例", scale=100.0, where="7.7")
add("15.7%", "data/runs/T022/m15_mismatch.json", "control_proportional_zone/top300_near_entrance",
    "前 300 吸引点中占比", scale=100.0, where="7.7")
add("14.1%", "data/runs/T022/m15_mismatch.json", "control_proportional_zone/share_arrivals_near_entrance",
    "到达量中占比", scale=100.0, where="7.7")
add("0.50", "data/runs/T022/m15_mismatch.json", "control_proportional_zone/median_dist_top100",
    "前 100 吸引点到出入口中位距离 km", where="7.7")
add("1.21", "data/runs/T022/m15_mismatch.json", "control_proportional_zone/median_dist_all",
    "随机格中位距离 km", where="7.7")

# --------------------------------------------------------------------------
# Q5 (sec8) -- T022 M15
# --------------------------------------------------------------------------
S22 = "data/runs/T022/m15_mismatch.json"
S22C = "data/runs/T022/m15_check.json"
add("11.66%", S22, "single_platform/unmet_rate", "单平台未服务率", scale=100.0, where="8.2")
add("0.33%", S22, "single_platform/full_rate", "单平台满位率", scale=100.0, where="8.2")
add("540,447.69", S22, "single_platform/required_capacity_slots", "单平台所需容量", where="8.2")
add("1,181,755", S22, "rows/0/departures", "围栏内离场量（12.10% 等分母）", where="8.2")
add("1,181,754.7", S22, "rows/0/departures", "同上，§8.2 原值", where="8.2")
add("143,011.06", S22, "rows/0/unmet_departures", "m=0 独立制度未服务离场", where="8.2")
add("132,250.84", S19P, "meas343/design/departure_shortfall", "全城口径缺口（6.20% 分子）", where="8.2")
addd("6.20%", [(S19P, "meas343/design/departure_shortfall", 1.0),
               (S24D, "trips", 1.0)], "100.0 * v0 / v1",
     "T019 全城口径 132,250.84/2,132,542，不可与 12.10% 并列", where="8.2")
for i, (m, reg, unm, full, slots) in enumerate([
        (0.0, "independent", "12.10%", "0.00007%", "608,326.14"),
        (0.0, "independent_equal", "12.16%", "0.90%", "608,326.14"),
        (0.0, "shared", "12.12%", "0.35%", "562,331.65"),
        (0.5, "independent", "13.27%", "0.0067%", "619,001.76"),
        (0.5, "independent_equal", "13.32%", "1.54%", "619,001.76"),
        (0.5, "shared", "13.29%", "0.37%", "564,716.90"),
        (1.0, "independent", "15.96%", "0.15%", "643,485.96"),
        (1.0, "independent_equal", "16.00%", "3.53%", "643,485.96"),
        (1.0, "shared", "16.11%", "0.69%", "577,094.59"),
        (1.5, "independent", "18.99%", "0.57%", "673,517.52"),
        (1.5, "independent_equal", "19.03%", "6.23%", "673,517.52"),
        (1.5, "shared", "19.55%", "1.57%", "594,809.49")]):
    add(unm, S22, "rows/%d/unmet_rate" % i, "m=%s %s 未服务率" % (m, reg), scale=100.0, where="8.3")
    add(full, S22, "rows/%d/full_rate" % i, "m=%s %s 满位率" % (m, reg), scale=100.0, where="8.3")
    add(slots, S22, "rows/%d/required_capacity_slots" % i, "m=%s %s 所需容量" % (m, reg), where="8.3")
for m, key in [("0.0", "0.0"), ("0.5", "0.5"), ("1.0", "1.0"), ("1.5", "1.5")]:
    pass
add("1.0818", S22C, "capacity_ratio_ind_over_shared/0.0", "独立/共享容量比 m=0", where="8.3")
add("1.0961", S22C, "capacity_ratio_ind_over_shared/0.5", "m=0.5", where="8.3")
add("1.1150", S22C, "capacity_ratio_ind_over_shared/1.0", "m=1.0", where="8.3")
add("1.1323", S22C, "capacity_ratio_ind_over_shared/1.5", "m=1.5", where="8.3")
add("1.0405", S22C, "m0_shared_vs_single/ratio", "m=0 共享/单平台容量比", where="8.3")
addd("4.05%", [(S22C, "m0_shared_vs_single/ratio", 1.0)], "100.0 * (v0 - 1.0)",
     "共享相对单平台容量增加", where="8.3")
addd("7.6%", [(S22C, "capacity_ratio_ind_over_shared/0.0", 1.0)], "100.0 * (1.0 - 1.0 / v0)",
     "重复容量份额下界 (1-1/ratio)", where="8.3")
addd("11.7%", [(S22C, "capacity_ratio_ind_over_shared/1.5", 1.0)], "100.0 * (1.0 - 1.0 / v0)",
     "重复容量份额上界", where="8.3")
addd("10.06%", [(S22, "rows/11/required_capacity_slots", 1.0),
                (S22, "single_platform/required_capacity_slots", 1.0)],
     "100.0 * (v0 / v1 - 1.0)", "m=1.5 共享相对单平台容量增加", where="8.3")
addd("42.96%", [(S18C, "end_positions/end_total", 1.0), (S18, "int_rows/K60/obj", 1.0)],
     "100.0 * 600.0 * v0 / v1", "60 辆 x 600 元 占 K=60 目标比例", where="8.4")

# --------------------------------------------------------------------------
# Q1 shared beta (sec4.4), fleet (sec3/sec9.5), variance/tornado (sec9)
# --------------------------------------------------------------------------
S22B = "data/runs/T022/shared_beta.json"
S22F = "data/runs/T022/fleet_estimates.json"
add("2.2100", S22B, "beta/shared", "五段共用 beta 估计（km^-1）", where="4.4")
for band, tok in [("night 0-5", "2.0948"), ("morning 6-9", "2.0144"),
                  ("midday 10-15", "2.5172"), ("evening 16-19", "2.1297"),
                  ("late 20-23", "2.3653")]:
    add(tok, S22B, "beta/five_band/%s" % band, "五段 beta：%s" % band, where="4.4")
add("2.095", S22B, "beta/five_band/night 0-5",
    "五段 beta（表 q1shared 中四舍五入到三位）", where="4.4")
add("2.365", S22B, "beta/five_band/late 20-23",
    "五段 beta（表 q1shared 中四舍五入到三位）", where="4.4")
add("3.5058", S22B, "per_day/2017-05-12/five_band/nlp", "12 May 五段 log-loss", where="4.4")
add("3.5113", S22B, "per_day/2017-05-12/shared_beta/nlp", "12 May 共用 beta log-loss", where="4.4")
add("0.3912", S22B, "per_day/2017-05-12/five_band/top5", "12 May 五段 top-5", where="4.4")
add("0.3907", S22B, "per_day/2017-05-12/shared_beta/top5", "12 May 共用 top-5", where="4.4")
add("3.3452", S22B, "per_day/2017-05-19/five_band/nlp", "19 May 五段 log-loss", where="4.4")
add("3.3503", S22B, "per_day/2017-05-19/shared_beta/nlp", "19 May 共用 log-loss", where="4.4")
add("0.4543", S22B, "per_day/2017-05-19/five_band/top5", "19 May 五段 top-5", where="4.4")
add("0.4538", S22B, "per_day/2017-05-19/shared_beta/top5", "19 May 共用 top-5", where="4.4")
add("-0.0055", S22B, "per_day/2017-05-12/paired/nll/diff", "12 May 配对 log-loss 差（五段-共用）", where="4.4")
add("0.00054", S22B, "per_day/2017-05-12/paired/top5/diff", "12 May 配对 top-5 差", where="4.4")
add("-0.0051", S22B, "per_day/2017-05-19/paired/nll/diff", "19 May 配对 log-loss 差", where="4.4")
add("0.00047", S22B, "per_day/2017-05-19/paired/top5/diff", "19 May 配对 top-5 差", where="4.4")
add("0.00027", S22B, "per_day/2017-05-12/paired/nll/se", "12 May log-loss 配对 SE", where="4.4")
add("0.00019", S22B, "per_day/2017-05-12/paired/top5/se", "12 May top-5 配对 SE", where="4.4")
add("0.00022", S22B, "per_day/2017-05-19/paired/nll/se", "19 May log-loss 配对 SE", where="4.4")
add("0.00017", S22B, "per_day/2017-05-19/paired/top5/se", "19 May top-5 配对 SE", where="4.4")
add("-20.5", S22B, "per_day/2017-05-12/paired/nll/z", "12 May 配对 z", where="4.4")
add("-23.7", S22B, "per_day/2017-05-19/paired/nll/z", "19 May 配对 z", where="4.4")
add("2.84", S22B, "per_day/2017-05-12/paired/top5/z", "12 May top-5 z", where="4.4")
add("2.80", S22B, "per_day/2017-05-19/paired/top5/z", "19 May top-5 z", where="4.4")
add("3.43080", S22B, "summary/nlp_shared_mean", "共用模型平均 held-out log-loss", where="4.4")
add("3.42548", S22B, "summary/nlp_mean", "五段平均 held-out log-loss", where="4.4")
addd("0.00532", [(S22B, "summary/nlp_shared_mean", 1.0),
                 (S22B, "summary/nlp_mean", 1.0)], "v0 - v1", "平均差（nats）", where="4.4")
addd("0.155%", [(S22B, "summary/nlp_shared_mean", 1.0),
                (S22B, "summary/nlp_mean", 1.0)], "100.0 * (v0 - v1) / v1",
     "平均差相对比例", where="4.4")
add("253,617", S22B, "per_day/2017-05-12/paired/nll/n", "12 May 建模行程数（OD 均在格内）", where="4.4")
add("302,058", S22B, "per_day/2017-05-19/paired/nll/n", "19 May 建模行程数", where="4.4")
add("265,173", "data/runs/T016/delivery.json", "per_day_counts_10_19/2017-05-12",
    "12 May 原始行程数", where="4.4")
add("314,516", "data/runs/T016/delivery.json", "per_day_counts_10_19/2017-05-19",
    "19 May 原始行程数", where="4.4")
add("0.054", S22B, "per_day/2017-05-12/paired/top5/diff", "12 May top-5 效应（百分点）", scale=100.0, where="4.4")
add("0.047", S22B, "per_day/2017-05-19/paired/top5/diff", "19 May top-5 效应（百分点）", scale=100.0, where="4.4")

add("462,893", S22F, "counts/n1", "窗口 1 观测车数", where="9.5")
add("331,822", S22F, "counts/n2", "窗口 2 观测车数", where="9.5")
add("309,961", S22F, "counts/m", "两窗口重捕数", where="9.5")
add("484,754", S22F, "counts/union", "14 天观测并集（设计无关下界）", where="9.5")
add("131", S22F, "chapman_pooled/se", "Chapman 抽样标准误", where="9.5")
add("495,283", S22F, "chapman_pooled/ci95/0", "95% 区间下", where="9.5")
add("495,797", S22F, "chapman_pooled/ci95/1", "95% 区间上", where="9.5")
add("0.4945", S22F, "heterogeneity/rate_min", "最低使用层重捕率", where="9.5")
add("0.8039", S22F, "heterogeneity/rate_max", "最高使用层重捕率", where="9.5")
add("15,063.61", S22F, "heterogeneity/chi2_5x2", "5x2 卡方", where="9.5")
add("121.57", S22F, "heterogeneity/cochran_armitage_z", "Cochran-Armitage z", where="9.5")
add("1.6257", S22F, "heterogeneity/rate_ratio", "重捕率比", where="9.5")
add("4", S22F, "heterogeneity/dof", "卡方自由度", where="9.5")
add("0.10533", S22F, "openness/retirement_of_recapturable/dN_per_bike",
    "可重捕车退役导致的向上偏差（每辆）", where="9.5")
add("1.49339", S22F, "openness/new_deployment_captured/dN_per_bike",
    "窗口间新增投放导致的向上偏差（每辆）", where="9.5")
add("506,635", S22F, "interval/old/1", "模型区间上端", where="9.5")
add("534,039", S22F, "interval/chao1_scenario", "Chao1 情景值（已撤回下界地位）", where="9.5")
add("4.33", S22F, "rides_per_bike_day/values/F_495540", "每车每日骑行（495,540 上估计）", where="9.5")
add("4.42", S22F, "rides_per_bike_day/values/F_484754_ceiling", "设计无关上界", where="9.5")
add("4.01", S22F, "rides_per_bike_day/values/F_534039_scenario", "534,039 情景", where="9.5")
addconst("0.05", 0.05, "Chao1 与齐性模型复现差（T022 注记 \"to within 0.05 bicycles\"）", where="9.5")

# T023 grouped intervals (sec9.2) -- all eight outputs
IV = "data/runs/T023/intervals.json"
for out, tok_ref, tok_j, tok_a, tok_b, tok_c in [
        ("q2_total", "617,056", "271,732 / 484,372 / 882,188",
         "536,867--630,491", "387,906--608,445", "407,117--1,376,489"),
        ("q2_unmet", "39,984", "2,789 / 11,392 / 73,744",
         "27,226--42,354", "5,534--38,642", "22,166--142,932"),
        ("q3_pure_obj", "138,317", "47,905 / 89,689 / 162,867",
         "125,302--137,573", "71,031--154,475", "74,364--197,192"),
        ("q4_unassigned", "394,553", "121,720 / 285,182 / 614,403",
         "396,175--420,335", "85,304--526,191", "249,274--656,872")]:
    add(tok_ref, IV, "outputs/%s/reference" % out, "参考值（见对应条目）", where="9.2")
    p5, p50, p95 = [s.strip() for s in tok_j.split("/")]
    add(p5, IV, "outputs/%s/slices/F/p5" % out, "联合 P5", where="9.2")
    add(p50, IV, "outputs/%s/slices/F/p50" % out, "联合 P50", where="9.2")
    add(p95, IV, "outputs/%s/slices/F/p95" % out, "联合 P95", where="9.2")
    for grp, toks in [("A", tok_a), ("B", tok_b), ("C", tok_c)]:
        lo, hi = [s.strip() for s in toks.split("--")]
        add(lo, IV, "outputs/%s/slices/%s/p5" % (out, grp), "%s 组 P5" % grp, where="9.2")
        add(hi, IV, "outputs/%s/slices/%s/p95" % (out, grp), "%s 组 P95" % grp, where="9.2")
add("0.5953", IV, "outputs/q4_coverage/reference", "Q4 覆盖参考值", where="9.2")
for grp, lo, hi in [("F", "0.3866", "0.8236"), ("A", "0.5838", "0.5946"),
                    ("B", "0.5267", "0.8447"), ("C", "0.3262", "0.7443")]:
    add(lo, IV, "outputs/q4_coverage/slices/%s/p5" % grp, "覆盖率 %s P5" % grp, where="9.2")
    add(hi, IV, "outputs/q4_coverage/slices/%s/p95" % grp, "覆盖率 %s P95" % grp, where="9.2")
add("0.6250", IV, "outputs/q4_coverage/slices/F/p50", "覆盖率联合 P50", where="9.2")
add("0.594", IV, "outputs/q2_saving_share/reference", "节省份额参考", where="9.2")
add("0.586", IV, "outputs/q2_saving_share/slices/A/p5", "A P5", where="9.2")
add("0.613", IV, "outputs/q2_saving_share/slices/A/p95", "A P95", where="9.2")
add("-0.071", IV, "outputs/q2_saving_share/slices/B/p5", "B P5", where="9.2")
add("0.586", IV, "outputs/q2_saving_share/slices/B/p95", "B P95", where="9.2")
add("0.184", IV, "outputs/q2_saving_share/slices/C/p5", "C P5", where="9.2")
add("0.697", IV, "outputs/q2_saving_share/slices/C/p95", "C P95", where="9.2")
add("-1,149", IV, "outputs/q3_ceiling/slices/F/p5", "Q3 上限联合 P5", where="9.2")
add("-218", IV, "outputs/q3_ceiling/slices/F/p50", "联合 P50", where="9.2")
add("+540", IV, "outputs/q3_ceiling/slices/F/p95", "联合 P95（正号=符号翻转）", where="9.2")
add("-1,436", IV, "outputs/q3_ceiling/slices/A/p5", "A P5", where="9.2")
add("-612", IV, "outputs/q3_ceiling/slices/A/p95", "A P95", where="9.2")
add("-1,606", IV, "outputs/q3_ceiling/slices/B/p5", "B P5", where="9.2")
add("-245", IV, "outputs/q3_ceiling/slices/B/p95", "B P95", where="9.2")
add("-1,793", IV, "outputs/q3_ceiling/slices/C/p5", "C P5", where="9.2")
add("+526", IV, "outputs/q3_ceiling/slices/C/p95", "C P95", where="9.2")
add("33.19", IV, "outputs/q3_pure_unmet/reference", "Q3 未服务参考（退化输出）", where="9.2")
add("35.96", IV, "outputs/q3_pure_unmet/slices/F/p95", "联合 P95", where="9.2")
add("20", "data/runs/T023/flips.json", "q3_sign/C/n_positive", "C 组符号翻转次数", where="9.2")
add("18", "data/runs/T023/flips.json", "q3_sign/F/n_positive", "F 组符号翻转次数", where="9.2")
add("0", "data/runs/T023/flips.json", "q3_sign/A/n_positive", "A 组翻转次数", where="9.2")
add("1", "data/runs/T023/flips.json", "q3_sign/B/n_positive", "B 组翻转次数", where="9.2")
addc("14%", "flip_share_low", "18/128 = 14.1%", where="9.2")
addc("16%", "flip_share_high", "20/128 = 15.6%", where="9.2")

# variance shares (sec9.3)
VS = "data/runs/T023/variance_shares.json"
for out, tok, key in [
        ("q2_total", "1.072", "sum_shares"), ("q2_unmet", "1.381", "sum_shares"),
        ("q2_saving_share", "1.014", "sum_shares"), ("q3_pure_obj", "1.300", "sum_shares"),
        ("q3_ceiling", "0.750", "sum_shares"), ("q4_coverage", "1.287", "sum_shares"),
        ("q4_arrival", "1.323", "sum_shares"), ("q4_unassigned", "1.256", "sum_shares")]:
    add(tok, VS, "outputs/%s/%s" % (out, key), "三组一阶份额之和 %s" % out, where="9.3")
for out, a, b, c, s, se in [
        ("q2_total", "0.047", "0.248", "0.778", "1.072", "0.242"),
        ("q2_unmet", "0.145", "0.387", "0.849", "1.381", "0.310"),
        ("q2_saving_share", "0.017", "0.501", "0.497", "1.014", "0.229"),
        ("q3_pure_obj", "0.150", "0.541", "0.610", "1.300", "0.154"),
        ("q3_ceiling", "0.158", "0.084", "0.508", "0.750", "0.184"),
        ("q4_coverage", "0.159", "0.370", "0.757", "1.287", "0.198"),
        ("q4_arrival", "0.159", "0.399", "0.765", "1.323", "0.194"),
        ("q4_unassigned", "0.136", "0.738", "0.382", "1.256", "0.222")]:
    for grp, tok in [("A", a), ("B", b), ("C", c)]:
        add(tok, VS, "outputs/%s/share_first_order/%s" % (out, grp),
            "一阶份额 %s/%s" % (out, grp), where="9.3")
    add(se, VS, "outputs/%s/sum_shares_se" % out, "份额和的标准误 %s" % out, where="9.3")
for out, se in [("q2_total", "0.119"), ("q2_unmet", "0.158"), ("q2_saving_share", "0.111"),
                ("q3_pure_obj", "0.078"), ("q3_ceiling", "0.096"), ("q4_coverage", "0.098"),
                ("q4_arrival", "0.097"), ("q4_unassigned", "0.112")]:
    add(se, VS, "outputs/%s/share_se_boot/A" % out, "A 份额 bootstrap SE %s" % out, where="9.3")
for out, se in [("q2_total", "0.129"), ("q2_unmet", "0.156"), ("q2_saving_share", "0.095"),
                ("q3_pure_obj", "0.067"), ("q3_ceiling", "0.083"), ("q4_coverage", "0.097"),
                ("q4_arrival", "0.089"), ("q4_unassigned", "0.081")]:
    add(se, VS, "outputs/%s/share_se_boot/B" % out, "B 份额 SE %s" % out, where="9.3")
for out, se in [("q2_total", "0.089"), ("q2_unmet", "0.091"), ("q2_saving_share", "0.083"),
                ("q3_pure_obj", "0.078"), ("q3_ceiling", "0.102"), ("q4_coverage", "0.068"),
                ("q4_arrival", "0.063"), ("q4_unassigned", "0.108")]:
    add(se, VS, "outputs/%s/share_se_boot/C" % out, "C 份额 SE %s" % out, where="9.3")
addconst("256", 256, "bootstrap 抽样次数", where="9.3")
addconst("400", 400, "bootstrap 重复次数（SE）", where="9.3")

# tornado (sec9.4)
TN = "data/runs/T023/tornado.json"
for rank, tok in enumerate(["629,963.87", "317,739.26", "292,500.00",
                            "231,947.90", "197,498.59"]):
    add(tok, TN, "rank/q2_total/%d/range" % rank, "Q2 总量级前五敏感性区间", where="9.4")
for rank, tok in enumerate(["1,782.03", "1,632.62", "1,436.12", "562.07"]):
    add(tok, TN, "rank/q3_ceiling/%d/range" % rank, "Q3 上限前四敏感性区间", where="9.4")
add("46.77", TN, "rank/q4_coverage/0/range", "Q4 容量上限对覆盖的百分点影响", scale=100.0, where="9.4")
add("17.11", TN, "rank/q4_coverage/1/range", "日变化影响", scale=100.0, where="9.4")
add("7.11", TN, "rank/q4_coverage/2/range", "步行半径影响", scale=100.0, where="9.4")
add("33,534.39", TN, "bars/COST_KM/range/q2_total",
    "卡车公里价 ±50%（3--9 元/km）对应的 Q2 总量变化（元/日）；§2 成本假设引用，非最大项",
    where="2")

# --------------------------------------------------------------------------
# Data / Table 1 / summary leftovers
# --------------------------------------------------------------------------
add("3,214,096", "data/runs/T002/ratios.json", "raw_rows", "原始记录数", where="3")
add("3,213,310", "data/runs/T002/ratios.json", "bbox_rows", "边界框内记录数", where="3")
addd("786", [("data/runs/T002/ratios.json", "raw_rows", 1.0),
             ("data/runs/T002/ratios.json", "bbox_rows", 1.0)],
     "v0 - v1", "剔除记录数", where="3")
add("12.83", "data/runs/T002/ratios.json", "variants/A_paper_core_allpairs/exact",
    "同格精确匹配率（%）", scale=100.0, where="3")
add("17.72", "data/runs/T002/ratios.json", "variants/A_paper_core_allpairs/le160",
    "160 m 内匹配率（%）", scale=100.0, where="3")
add("19.97", "data/runs/T002/ratios.json", "variants/A_paper_core_allpairs/le250",
    "250 m 内匹配率（%）", scale=100.0, where="3")
addconst("349,241", 349241, "用户数（旧锁定：T012 抽取日志 users: 349241，原始 trips 去重；数据描述量）", where="3")
addconst("8.89", 8.89, "同格观测占比（旧锁定，T005）", where="3")

# --- Q1 published table (T005 dest_validation, day_split) ---
S05 = "data/runs/T005/dest_validation.json"
_q1_rows = [
    ("18,364", "386,102", "366,212", "428,478", "165,671"),
    ("2.096", "2.014", "2.517", "2.130", "2.364"),
    ("3.313", "3.512", "3.239", "3.517", "3.360"),
    ("3.487", "3.645", "3.278", "3.561", "3.394"),
    ("7.912", "8.083", "8.091", "8.140", "8.185"),
    ("0.476", "0.399", "0.458", "0.408", "0.455"),
    ("0.402", "0.339", "0.440", "0.383", "0.437"),
]
for _i in range(5):
    add(_q1_rows[0][_i], S05, "day_split/bands/%d/n_train" % _i, "Q1 表：训练趟数（第 %d 段）" % (_i + 1), where="4")
    add(_q1_rows[1][_i], S05, "day_split/bands/%d/beta_full" % _i, "Q1 表：五段 beta（T005 已验收值）", where="4")
    add(_q1_rows[2][_i], S05, "day_split/bands/%d/nlp/full" % _i, "Q1 表：重力模型 log-loss", where="4")
    add(_q1_rows[3][_i], S05, "day_split/bands/%d/nlp/distance" % _i, "Q1 表：仅距离 log-loss", where="4")
    add(_q1_rows[4][_i], S05, "day_split/bands/%d/nlp/attraction" % _i, "Q1 表：仅吸引 log-loss", where="4")
    add(_q1_rows[5][_i], S05, "day_split/bands/%d/top5/full" % _i, "Q1 表：top-5（重力）", where="4")
    add(_q1_rows[6][_i], S05, "day_split/bands/%d/top5/distance" % _i, "Q1 表：top-5（仅距离）", where="4")
add("8.568", S05, "day_split/bands/0/nlp/uniform", "Q1 表：均匀模型 log-loss", where="4")
addd("48", [(S05, "day_split/bands/0/top5/full", 100.0)],
     "v0", "Q1 top-5 命中率上界 47.551% 取整（摘要 40--48%）", where="4")
addconst("3.94", 3.94, "同格模型 12.83% 对观测 8.89%，+3.94 个百分点（算术）", where="4")
add("12.83", "data/runs/T002/ratios.json", "variants/A_paper_core_allpairs/exact",
    "同格精确匹配率（%），表 1/§4 引用", scale=100.0, where="4")

# --- misc method / policy constants and small derived values ---
addconst("50", 50, "区数", where="5.2")
addconst("23", 23, "小时索引上限 t=0..23", where="5.2")
addconst("600", 600, "卡车日固定成本（元/车·日）", where="5.2")
addconst("6", 6, "未服务罚金与卡车公里成本（元）", where="5.2")
addconst("100", 100, "每 100 站边际口径 / top-100 口径", where="7.2")
addconst("3,500", 3500, "边际曲线站点数节点", where="7.2")
addconst("5,000", 5000, "边际曲线站点数节点", where="7.2")
add("274,357", "data/runs/T004/fine_grid.json", "coarse_support/typical_day_departures",
    "典型日样本离场（辆/日，表 3）", where="3")
addd("2.14", [(S09V, "validation/q1_daily_total", 1.0)], "v0 / 1e6",
     "典型日放大后离场（百万）", where="3")
addd("3.21", [("data/runs/T002/ratios.json", "bbox_rows", 1.0)], "v0 / 1e6",
     "订单数（百万）", where="1")
add("0.128", "data/runs/T002/p_estimate.json", "consecutive/exact",
    "采样率估计（同格配对率），作上界比例因子", where="3")
addconst("90", 90, "第 90 百分位（定义）", where="3")
addconst("0.70", 0.70, "行程距离中位数 km（T002 距离统计，旧锁定）", where="3")
addconst("1.41", 1.41, "第 90 百分位距离 km（T002 距离统计，旧锁定）", where="3")
addconst("0.65", 0.65, "对数正态 sigma（T002 距离拟合，旧锁定）", where="3")
addconst("0.66", 0.66, "对数正态中位数 km（T002 距离拟合，旧锁定）", where="3")
addconst("14", 14, "记录日期数", where="3")
addconst("17", 17, "5 月 17 日（无记录）", where="3")
addconst("96", 96, "每日 15 分钟区间数", where="7.1")
addconst("210", 210, "150 车位折合面积 m^2（10x21 m）", where="7.3")
addconst("75", 75, "independent_equal 每平台车位数（150/2）", where="8.2")
addconst("24", 24, "区 24（最大三轮车分配区），亦为小时数上限的邻值", where="5.4")
addd("36,000", [(S18C, "end_positions/end_total", 1.0)], "600.0 * v0",
     "60 辆卡车固定成本（元/日）", where="8.4")
add("2.70", S26, "h26/r_high", "H26 高端奖励（付款账不可行）", where="6.5")
addd("12.56%", [(S22, "rows/0/required_capacity_slots", 1.0),
                (S22, "single_platform/required_capacity_slots", 1.0)],
     "100.0 * (v0 / v1 - 1.0)", "独立制度 m=0 相对单平台的容量增加", where="8.3")
addd("1.23", [("data/runs/T023/variance_shares.json", "outputs/q2_unmet/residual_1_minus_sum", 1.0),
              ("data/runs/T023/variance_shares.json", "outputs/q2_unmet/residual_1_minus_sum_se", 1.0)],
     "-v0 / v1", "Q2 unmet 残差相对其 SE 的倍数", where="9.3")
addconst("0.8", 0.8, "c0 倍率下界（0.8x）", where="6.4")
addconst("1.2", 1.2, "c0 倍率上界（1.2x）", where="6.4")
addconst("0.1", 0.1, "0.1% 求解容差（百分数）", where="6.3")
addconst("1", 1, "1% 上界（百分数）", where="6.3")
addconst("10", 10, "接受模型 v=10 km/h；亦为 59.06/6 的整数近似", where="6.2")
addconst("0.005", 0.005, "0.005 度网格", where="7.5")
addconst("0.5", 0.5, "每辆装卸 0.5 分钟（三轮车）", where="5.4")
addconst("4,000", 4000, "Q4 站点数（政策结果，见 4,000 条目）", where="7.3")
addconst("150", 150, "Q4 单站容量上限（政策参数）", where="7.3")
addconst("2", 2, "两个留出日 / 两种库存情景", where="7.1")
addconst("5", 5, "五个行为簇 / 五个网格点", where="1")

# --------------------------------------------------------------------------
# Computation registry
# --------------------------------------------------------------------------
_TYP = None


def typical():
    global _TYP
    if _TYP is None:
        _TYP = np.load(P("data/typical_day.npz"), allow_pickle=True)
    return _TYP


def compute(tag):
    if tag == "graph_arcs" or tag == "graph_mean_km":
        cent = typical()["cent"]
        dist = np.sqrt(((cent[:, None, :] - cent[None, :, :]) ** 2).sum(-1))
        order = np.argsort(dist, axis=1)[:, 1:9]
        arcs = [(i, int(j)) for i in range(len(cent)) for j in order[i]
                if dist[i, j] <= 9.0]
        lengths = np.array([dist[i, j] for i, j in arcs])
        return float(len(arcs) if tag == "graph_arcs" else lengths.mean())
    if tag == "neg_saving_count" or tag == "neg_saving_share":
        n = neg = 0
        with io.open(P("data/runs/T023/evals.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row.get("tag") != "F":
                    continue
                n += 1
                if row["q2"]["saving"] < 0:
                    neg += 1
        return float(neg if tag == "neg_saving_count" else 100.0 * neg / n)
    if tag == "kstar_ge60" or tag == "kstar_ge60_share" or tag == "kstar_change_share":
        d = json.load(open(P("data/runs/T023/intervals.json")))["outputs"]["q2_kstar"]
        dist = d["slices"]["F"]["distribution"]
        n = d["slices"]["F"]["n"]
        ge = dist.get("60", 0) + dist.get("80", 0)
        if tag == "kstar_ge60":
            return float(ge)
        if tag == "kstar_ge60_share":
            return 100.0 * ge / n
        return 100.0 * (1.0 - ge / n)
    if tag.startswith("marg_"):
        mc = json.load(open(P("data/runs/T019/marginal_curve.json")))["meas343"]
        idx = {"marg_1500": (14, 15), "marg_134": (15, 16),
               "marg_072": (16, 17), "marg_016": (17, 18)}[tag]
        i, j = idx
        return 100.0 * (mc["coverage"][j] - mc["coverage"][i]) / (
            (mc["n"][j] - mc["n"][i]) / 100.0)
    if tag == "uncapped_median" or tag == "uncapped_max" or tag == "uncapped_over150":
        import csv
        rows = list(csv.DictReader(open(P("data/runs/T009/q4_fences.csv"))))
        cap = np.array([float(r["capacity_meas343"]) for r in rows])
        if tag == "uncapped_median":
            return float(np.median(cap))
        if tag == "uncapped_max":
            return float(cap.max())
        return float(np.sum(cap > 150))
    if tag == "flip_share_low":
        d = json.load(open(P("data/runs/T023/flips.json")))["q3_sign"]["F"]
        return 100.0 * d["n_positive"] / d["n"]
    if tag == "flip_share_high":
        d = json.load(open(P("data/runs/T023/flips.json")))["q3_sign"]["C"]
        return 100.0 * d["n_positive"] / d["n"]
    raise KeyError(tag)


# --------------------------------------------------------------------------
# Verification and coverage
# --------------------------------------------------------------------------
_cache = {}


def load(rel):
    if rel not in _cache:
        _cache[rel] = json.load(io.open(P(rel), encoding="utf-8"))
    return _cache[rel]


def get(rel, key):
    cur = load(rel)
    for part in key.split("/"):
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


def parse_token(tok):
    t = tok.replace(",", "").replace("+", "")
    if t.endswith("%"):
        t = t[:-1]
    return float(t), len(t.split(".")[1]) if "." in t else 0


def tol_for(token):
    _, d = parse_token(token)
    return 0.5 * 10.0 ** (-d) + 1e-9


def resolve(entry):
    if "const" in entry:
        return float(entry["const"]), "definition"
    if "computed" in entry:
        return compute(entry["computed"]), "computed"
    if "inputs" in entry:
        vals = [get(f, k) * s for f, k, s in entry["inputs"]]
        env = {"v%d" % i: v for i, v in enumerate(vals)}
        return float(eval(entry["formula"], {"__builtins__": {}}, env)), "derived"
    return get(entry["file"], entry["key"]) * entry.get("scale", 1.0), "direct"


def verify():
    bad = []
    for e in ENTRIES:
        try:
            val, kind = resolve(e)
            e["value"] = val
            e["kind"] = kind
            t, _ = parse_token(e["token"])
            if abs(t - val) > tol_for(e["token"]) + 1e-9:
                bad.append((e["token"], val, e.get("file") or e.get("computed"), e.get("key")))
        except Exception as exc:  # noqa: BLE001
            bad.append((e["token"], repr(exc), e.get("file") or e.get("computed"), e.get("key")))
    return bad


TEX_SCAN = [
    ("paper/summary.tex", None, None),
    ("paper/sec1_intro.tex", 92, 136),
    ("paper/sec2_assumptions.tex", 58, 95),
    ("paper/sec3_data.tex", 86, 116),
    ("paper/sec3_data.tex", 140, 160),
    ("paper/sec4_q1.tex", 118, 163),
    ("paper/sec5_q2.tex", None, None),
    ("paper/sec6_q3.tex", None, None),
    ("paper/sec7_q4.tex", None, None),
    ("paper/sec8_q5.tex", None, None),
    ("paper/sec9_sensitivity.tex", None, None),
    ("paper/sec10_evaluation.tex", None, None),
]


def tokens_in(text):
    text = text.replace("{,}", ",").replace("\\,", "")
    text = text.replace("--", " ")
    text = text.replace("\\%", " ")            # literal percent is not a comment
    text = re.sub(r"%.*", "", text)
    text = re.sub(r"\\includegraphics\[[^\]]*\]", "", text)
    text = re.sub(r"p\{[^}]*\}", "", text)
    text = re.sub(r"width=[0-9.]+\\[a-zA-Z]+", "", text)
    text = re.sub(r"\{[0-9.]+\s*(?:cm|pt|mm|in)\}", "", text)   # layout dims
    text = re.sub(r"\\[A-Za-z]+", " ", text)     # control words: \ge484,754 -> 484,754
    out = []
    for m in re.finditer(r"(?<![A-Za-z0-9])([+-]?\d+(?:,\d{3})*(?:\.\d+)?)", text):
        tok = m.group(1).strip(",")
        if not tok or tok in ("+", "-"):
            continue
        out.append((tok, text[max(0, m.start() - 45):m.end() + 45].replace("\n", " ")))
    return out


def coverage():
    pool = []
    for e in ENTRIES:
        if "value" not in e:
            continue
        v = e["value"]
        for form in (v,) if isinstance(v, float) else ():
            pool.append(form)
    uncovered = []
    for rel, lo, hi in TEX_SCAN:
        lines = io.open(P(rel), encoding="utf-8").read().split("\n")
        if lo is not None:
            lines = lines[lo - 1:hi]
        text = "\n".join(lines)
        for tok, ctx in tokens_in(text):
            if re.fullmatch(r"20\d\d", tok):      # years
                continue
            t, d = parse_token(tok)
            if d == 0 and abs(t) <= 8:            # section/table/step indices
                continue
            if any(abs(t - v) <= 0.5 * 10 ** (-d) + 1e-9 for v in pool):
                continue
            uncovered.append((rel, tok, ctx))
    return uncovered


def main():
    bad = verify()
    unc = coverage()
    payload = dict(
        task="T021A number lock",
        date="2026-11-19",
        scope=("summary, Table 1, sec2 A10-A12, sec3 fleet paragraph, sec4 shared-beta, "
               "sec5-sec10; every item of the T021 task-sheet number-lock section"),
        n_entries=len(ENTRIES),
        n_verified=sum(1 for e in ENTRIES if "value" in e),
        entries=ENTRIES,
        verification=dict(failed=bad, n_failed=len(bad)),
        coverage=dict(n_uncovered=len(unc), uncovered=unc),
        notes=[
            "T026 review note: check 5's '0.001 CNY' is the bisection width of r "
            "(caps.*.cross.width = 0.000781-0.000782), not a cost precision; at the "
            "surplus crossing the endpoint cost is 617,232.0342, i.e. 176.28 CNY above "
            "the no-incentive level (entry '176.28').",
            "T023 Q2 K* C-group P5 = 7.0 is a quantile interpolation between atoms 0 and "
            "20 (distribution 0:7, 20:6, 40:71, 60:35, 80:9), reported as 7-80; it is not "
            "an observed fleet size.",
            "T023 intervals are LP-proxy intervals, not calibrated confidence intervals "
            "(max integer-chain deviation 38,986.30 CNY/day at 11 points).",
            "The 5.84%/3.51% pre-plan shares are the T004 dynamic simulations at R=450 m, "
            "carried unchanged from the accepted pre-T021 paper.",
            "Legacy sec3 data-description tokens (3,214,096 etc.) are locked to T002; "
            "quantities not re-used in the rewritten passages are not re-locked here.",
        ],
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    with io.open(LOG, "w", encoding="utf-8") as fh:
        fh.write("entries=%d failed=%d uncovered=%d\n\n" % (len(ENTRIES), len(bad), len(unc)))
        for b in bad:
            fh.write("FAIL %s value=%s %s::%s\n" % b)
        fh.write("\n--- uncovered numeric tokens ---\n")
        for rel, tok, ctx in unc:
            fh.write("%s\t%s\t%s\n" % (rel, tok, ctx))
    print("entries=%d failed=%d uncovered=%d" % (len(ENTRIES), len(bad), len(unc)))
    for b in bad[:40]:
        print("FAIL", b)


if __name__ == "__main__":
    main()
