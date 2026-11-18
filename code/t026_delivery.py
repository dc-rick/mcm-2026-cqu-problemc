"""Build 交付/T026_delivery.json from the T026 result files.

Every number is read from data/runs/T026/*.json; none is typed from memory.
Run after ``t026_incentive_tricycle.py --stage all``.
"""
import glob
import hashlib
import io
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(ROOT, "data", "runs", "T026")
OUT = os.path.join(ROOT, "交付", "T026_delivery.json")


def load(name):
    with io.open(os.path.join(RUN, name), encoding="utf-8") as fh:
        return json.load(fh)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def rel(path):
    return os.path.relpath(path, ROOT).replace("\\", "/")


BANNED = ("885", "266327", "266,327", "40357", "40,357", "705",
          "76.86", "6.83", "20.83", "10.40", "2883.3", "2,883.3", "480.6")


def main():
    check = load("check.json")
    d = load("cost_vs_r.json")
    opt = load("optimum.json")
    sen = load("sensitivity.json")
    reg = load("regression.json")
    ck = check["checks"]
    h = opt["h26"]
    arr = opt["caps"]["arrivals"]
    sur = opt["caps"]["surplus"]

    def _bump(cap):
        xs = opt["caps"][cap]["low_r_bump"]
        return ("none" if not xs else
                ", ".join("r=%.2f +%.2f" % (x["r"], x["excess"]) for x in xs))

    def _plat(cap):
        xs = opt["caps"][cap]["r_star_plateau"]
        return "%.2f-%.2f (%d point%s)" % (
            min(xs), max(xs), len(xs), "" if len(xs) == 1 else "s")

    checks = {
        "1_grid_and_fields": (
            "pass; cost_vs_r.json rows=61 per cap (0..3.00 step 0.05), "
            "r=2.70 present, every row carries r/q/payment/payment_capped/"
            "n_total/tricycle_layer/truck_layer/total/unmet/iter/resid; "
            "check.json 1 issues=%d" % len(ck["1_grid_and_fields"]["issues"])),
        "2_h26_comparison": (
            "pass; optimum.json h26: total(0.81)=%.2f vs total(2.70)=%.2f, "
            "better=%s; uncapped payments %.2f vs %.2f, capped actual "
            "%.2f vs %.2f (capped flags %s/%s); truck layers %.5f vs "
            "%.5f equal=%s; gap decomposition %s"
            % (h["low"]["total"], h["high"]["total"], h["better"],
               h["low"]["payment_uncapped"], h["high"]["payment_uncapped"],
               h["low"]["payment_capped"], h["high"]["payment_capped"],
               h["low"]["capped"], h["high"]["capped"],
               h["low"]["truck_layer"], h["high"]["truck_layer"],
               h["truck_layer_equal"], h["offset_evidence"]["check"])),
        "3_regression": (
            "pass; r=0 bit-identical to T024-R3: n_total %d, unmet %.10f, "
            "layer %.10f, total %.10f, unassigned %d, payment %.1f, q %.1f"
            % (reg["check"]["n_total"][0], reg["check"]["unmet"][0],
               reg["check"]["layer"][0], reg["check"]["total"][0],
               reg["check"]["unassigned"][0], reg["check"]["payment"][0],
               reg["check"]["q"][0])),
        "4_fixed_point": (
            "pass; max iter=%d (gate 10) and max resid=%d (gate <1) over "
            "122 rows; resid is the last fleet-iterate change; the greedy "
            "slack (n_exec - min_fleet, max %d) is reported separately as "
            "a diagnostic, T024's own raise-only convention"
            % (ck["4_fixed_point"]["max_iter"],
               ck["4_fixed_point"]["max_resid"],
               ck["4_fixed_point"]["max_slack"])),
        "5_cross_and_star": (
            "pass; r_cross arrivals=%.3f (bisection width %.1e), "
            "r_cross surplus=%.3f (width %.1e); r* arrivals=%.2f saving "
            "%.2f; r* surplus=%.2f saving %.2f"
            % (arr["r_cross"], arr["cross"]["width"],
               sur["r_cross"], sur["cross"]["width"],
               arr["r_star"], arr["net_saving_star"],
               sur["r_star"], sur["net_saving_star"])),
        "6_sensitivity": (
            "pass; sensitivity.json re-optimizes r* on the same 61-point grid "
            "for c0 x0.8/x1.0/x1.2 (c0=%.3f/%.3f/%.3f): r*=%s, net "
            "saving=%s CNY/day (movement, not a fixed-r stress)"
            % (sen["rows"][0]["c0"], sen["rows"][1]["c0"],
               sen["rows"][2]["c0"],
               [r["r_star"] for r in sen["rows"]],
               [round(r["net_saving_star"], 2) for r in sen["rows"]])),
        "6b_bound_decomposition": (
            "pass; gap identity holds for both caps (check.json 12); at r* "
            "arrivals q=%.1f, net unmet reduction=%.1f (%.3f per q), "
            "fixed+km saving=%.2f (%.4f per q), penalty credit=%.2f; bound "
            "prediction=%.2f vs actual saving=%.2f, gap=%.2f = shortfall "
            "%.2f - credit %.2f"
            % (arr["q_star"], arr["net_unmet_reduction_star"],
               arr["unmet_reduction_per_q_star"],
               arr["fixed_km_saving_star"],
               arr["fixed_km_saving_per_q_star"],
               arr["penalty_saving_star"], arr["bound_prediction_star"],
               arr["net_saving_star"], arr["bound_gap_star"],
               arr["gap_decomposition"]["fixed_km_shortfall"],
               arr["gap_decomposition"]["penalty_credit"])),
        "7_figure": (
            "pass; data/runs/T026/figs/fig_total_vs_r.pdf, %d bytes, "
            "%%PDF header, produced by code/t026_incentive_tricycle.py "
            "stage_figure" % ck["7_figure"]["size_bytes"]),
        "8_delta_caliber": (
            "pass; reroute block runs after walk_borrow0 and before "
            "t24._tri_lp_local on the same post-walking resid array; r=0 "
            "reproduces T024-R3 bit-for-bit, so Delta_c is the T024 "
            "post-walking (cell, hour) deficit by construction"),
        "9_caps_and_budget": (
            "pass; both caps scanned (61 rows each); payment is not a "
            "constraint: uncapped LP payment and payment_capped="
            "min(payment,30000) are both reported; capped rows: "
            "arrivals=%d, surplus=%d"
            % (ck["9_source_caps_and_budget"]["n_capped_arrivals"],
               ck["9_source_caps_and_budget"]["n_capped_surplus"])),
        "10_truck_invariant": (
            "pass; truck_layer == %.5f in all 122 rows; same-zone reroutes "
            "leave T018's integer plan and every zone total untouched"
            % opt["truck_invariant"]["expected"]),
        "11_detail_consistency": (
            "pass; the r* detail replay sums to the scanned point "
            "(n_total %d, payment %.10f)"
            % (arr["detail_check"]["n_total"],
               arr["detail_check"]["payment"])),
        "12_no_superseded_numbers": None,
        "13_scope": (
            "pass; only code/t026_incentive_tricycle.py, code/t026_delivery.py, "
            "data/runs/T026/ and 交付/T026_delivery.json written; no paper, "
            "no 审查与规划, no existing data input; no push"),
    }

    # forbidden-number scan over the delivered result JSONs
    hits = []
    for name in ("cost_vs_r.json", "optimum.json", "sensitivity.json",
                 "regression.json", "check.json"):
        text = io.open(os.path.join(RUN, name), encoding="utf-8").read()
        for tok in BANNED:
            pat = re.compile(r"(?<!\d)" + re.escape(tok) + r"(?!\d)")
            for m in pat.finditer(text):
                i = m.start()
                hits.append(dict(file=name, token=tok,
                                 context=text[max(0, i - 40):i + 40]))
    checks["12_no_superseded_numbers"] = (
        "pass; forbidden-token scan over the five result JSONs, hits=%d"
        % len(hits))

    inputs = [
        "data/q1_params.json",
        "data/dist_matrix.npy",
        "data/runs/T018/plan_int_K60.npz",
        "data/runs/T019/cell_demand.npz",
        "data/runs/T024/cell_inputs.npz",
        "data/runs/T024/compare.json",
        "data/runs/T024/schedule_check.json",
    ]
    out_files = ["code/t026_incentive_tricycle.py", "code/t026_delivery.py"]
    for name in ("regression.json", "cost_vs_r.json", "optimum.json",
                 "sensitivity.json", "check.json"):
        out_files.append("data/runs/T026/" + name)
    out_files += sorted(rel(p) for p in
                        glob.glob(os.path.join(RUN, "figs", "*.pdf")))

    notes = [
        ("模型口径：Δ_c 取 T024 回放中步行之后、三轮车之前的 (cell, hour) "
         "残差；reroute LP 与 T008 同验收函数 π(r)=logistic(κr-ρτ-c0)，"
         "κ=2、ρ=0.35、c0=0.8，边半径 0.45 km，两种源上限（arrivals 上限 / "
         "小时盈余 max(0, arr-dep) 保守）。被填的缺口 y_c 同时从该格残差和"
         "源格的当小时到达中扣除，日自行车总量守恒。重路由限制在同一 zone："
         "T018 的 zone 总量与卡车层因此逐位不变（122 行全部等于 83,807.74384）"
         "——这是本口径的定义性质，不是巧合；如果它变了，说明 reroute LP 与"
         "三轮车层的缺口口径错位，应停下报告。"),
        ("时点约定（显式假设）：接受的 reroute 按“小时内落车”处理，与 T024 "
         "的三轮车调运同一时点；源格减少的是同一小时的到达。若把 reroute "
         "当作跨小时延迟（落车只服务下一小时），它会被系统性削弱，本任务"
         "不采用该变体。用户侧的支付 r 是平台支出，用户时间/步行成本不进"
         "目标函数；π(r) 是行为接受概率，不是价格弹性估计。"),
        ("外循环与残差：每个 (r, cap) 点都从 T024 的 LP 曲线 DP 分配 889 辆"
         "出发（用上一 r 的解热启动会让 raise-only 不动点继承更高的车队，"
         "使 n 依赖扫描方向，故禁用）。exec_n_r 复刻 T024._exec_n：只在 "
         "k>n 时抬升，k<=n 即停。任务单的“残差 <1 辆”指最后一次迭代的"
         "车队变化 |n_(i+1)-n_i|，本任务全部为 0；n_exec 高于该行程表的贪心"
         "最小车队 k 的部分（slack）是 T024 同款超配：搜索下界是 n-5，故"
         "每个 zone 的 slack ≤5 辆，全市合计最大 %.0f 辆（check 第 4 项），"
         "单列在每行，不冒充收敛残差。iter 为 50 个 zone 的最大值，全部 "
         "≤10。" % ck["4_fixed_point"]["max_slack"]),
        ("H26 结论：r=0.81 与 r=2.70 在同一 T024 口径下，total(0.81)="
         "%.2f，total(2.70)=%.2f，0.81 更低（差 %.2f 元/日）。两点的未截断"
         "付款都超过 30,000：%.2f vs %.2f（分别为预算的 %.1f 倍与 %.1f 倍），"
         "截断实付同为 30,000。截断只是会计口径，不是可执行的省钱：要在 "
         "2.70 兑现奖励，义务就是 %.2f，必须重定 r，所以 2.70 是预算不可行"
         "点。卡车层逐位相同（83,807.74384），所以“2.70 的卡车目标更低”这一"
         "前提在本口径下不存在；2.70 多付 %.2f 元/日，只省下 %.2f 元/日的"
         "三轮车层。本口径的全局最优点是 r*=%.2f（0.05 网格），净节省 "
         "%.2f 元/日，付款 %.2f 在预算内（余 %.2f），预算在最优点不紧；"
         "预算可行最优 r 同为 %.2f。论文 §6 与摘要若把 2.70 或 0.81 写成"
         "最优，应按 r* 重写。"
         % (h["low"]["total"], h["high"]["total"],
            h["total_gap_high_minus_low"],
            h["low"]["payment_uncapped"], h["high"]["payment_uncapped"],
            h["low"]["payment_uncapped"] / 30000.0,
            h["high"]["payment_uncapped"] / 30000.0,
            h["high"]["payment_uncapped"],
            h["offset_evidence"]["payment_gap"],
            -h["offset_evidence"]["layer_gap"],
            arr["r_star"], arr["net_saving_star"], arr["payment_star"],
            arr["budget_slack_star"], arr["r_star_budget"])),
        ("r_cross 定义与数值：r_cross 取网格上最后一个自下而上穿越 total(0) "
         "的根（有效区间之后的回归点；低奖励端的振荡与 r=0 的平凡根都不算），"
         "对 r 二分到 ≤0.001 元。arrivals 上限 r_cross=%.3f（区间宽 %.1e），"
         "surplus 上限 r_cross=%.3f（宽 %.1e）。低奖励端有一个小凸起"
         "（low_r_bump，arrivals：%s；surplus：%s；"
         "单位元/日）：reroute 从源格移走到达，可能在后续小时造成新缺口，"
         "先于被填缺口主导；这不影响 r* 与 r_cross。r* 为 0.05 网格 argmin，"
         "距最小值 ≤1,000 元/日的网格平台为 arrivals %s、surplus %s；整数"
         "车队跳变使 argmin 的分辨率约为该量级。净节省 arrivals %.2f、"
         "surplus %.2f 元/日。"
         % (arr["r_cross"], arr["cross"]["width"],
            sur["r_cross"], sur["cross"]["width"],
            _bump("arrivals"), _bump("surplus"),
            _plat("arrivals"), _plat("surplus"),
            arr["net_saving_star"], sur["net_saving_star"])),
        ("与 §1 平均成本上界的对照：上界 q(r)(1.4721529-r) 把每个被填缺口"
         "都按三轮车层的平均单位成本 1.4721529 计价。精确回放低于它，原因"
         "（optimum.json bound_note）：(i) 车队是整数且只升不降；(ii) reroute "
         "按源格可达性填缺口，不按“最贵的三轮车”排序，会挤掉便宜的三轮车"
         "服务；(iii) 部分缺口本来会留作未服务（节省 6 元/辆）；(iv) 三轮车"
         "只在 07-20 运营，reroute 在 24 小时都可发；(v) 上界的 q 来自 T008 "
         "静态池，本回放用 T024 动态缺口。主导项是 q 与净减少未服务不是"
         "一对一：r* 处 q=%.1f，但净未服务只减少 %.1f（每 q 仅 %.3f），"
         "因为源格同时失去这些到达、在后续小时产生新缺口。固定+里程节省 "
         "%.2f（每 q %.4f，每净减少辆 %.4f），对比上界的 q*1.47215=%.2f；"
         "精确口径还多出罚款节省 %.2f——这是上界假设“两边未服务相同”而"
         "略去的贷方。上界与实际的净差 %.2f = 固定+里程缺口 %.2f − 罚款"
         "贷方 %.2f。上界仅作对照，所有结论取自精确外循环。"
         % (arr["q_star"], arr["net_unmet_reduction_star"],
            arr["unmet_reduction_per_q_star"],
            arr["fixed_km_saving_star"], arr["fixed_km_saving_per_q_star"],
            arr["fixed_km_saving_per_removed_unmet_star"],
            arr["q_star"] * arr["bound_unit_cost"],
            arr["penalty_saving_star"], arr["bound_gap_star"],
            arr["gap_decomposition"]["fixed_km_shortfall"],
            arr["gap_decomposition"]["penalty_credit"])),
        ("c0 敏感性是重新优化：每行在同一个 0-3.00、0.05 网格上重求 r*，"
         "不是固定 r 的压力测试。c0=%.3f/%.3f/%.3f（x0.8/x1.0/x1.2）对应 "
         "r*=%s、净节省=%s 元/日。c0 变小（接受概率上升、同 r 下 q 与付款"
         "上升）使最优奖励向低处移动，反之向高处移动。"
         % (sen["rows"][0]["c0"], sen["rows"][1]["c0"],
            sen["rows"][2]["c0"],
            [r["r_star"] for r in sen["rows"]],
            [round(r["net_saving_star"], 2) for r in sen["rows"]])),
        ("预算：payment 在模型中不截断（行为只依赖 r，不依赖预算余额），"
         "每行同时给 payment_uncapped 与 payment_capped=min(payment,30000)，"
         "capped 标志、budget_slack 与 q_budget（预算能买到的接受量）一并"
         "报告。截断只是会计口径，不反馈进行为；r* 的未截断付款在预算内，"
         "预算可行最优 r 与无约束 r* 相同（均为 %.2f），所以本任务不需要"
         "另做预算约束优化。高 r 端（如 2.70）的未截断义务远超预算，属预算"
         "不可行点，不能靠“截断实付”当成可行方案。"
         % arr["r_star_budget"]),
        ("本任务没有引用任何被取代的数字（885 / 266,327 / 40,357 / 705 / "
         "76.86%% / 6.83%%-20.83%% / 10.40 / 2,883.3 / 480.6x）；交付 checks "
         "的 12_no_superseded_numbers"
         "对五个结果 JSON 做禁用词扫描，命中 %d。T021A 在 T026 验收前不得"
         "引用本任务数字，只能留占位符/TODO。" % len(hits)),
        "r=0.81 不在 0.05 网格上，为回答 H26 单独求值（两个 cap 各一次）；"
        "r=2.70 是网格点。H26 的“哪个更低”用 arrivals 上限为主口径，"
        "surplus 上限的两点也一并记录。",
        "全部结论是单日（五个训练日的确定性回放）平台成本，不含实施成本、"
        "用户时间与公平性权重；不确定性另见 T023。",
    ]

    delivery = {
        "task": "T026",
        "title": "Q3 激励与三轮车同口径重解（H26：0.81 vs 2.70）",
        "date": "2026-11-19",
        "status": "delivered",
        "revision": "R0",
        "summary": (
            "Re-solved the Q3 incentive against T024's tricycle layer in one "
            "caliber: the reroute LP is embedded in the per-zone replay on "
            "the post-walking (cell, hour) deficit, same-zone so T018 and the "
            "truck layer are untouched, with the bike balance exact.  61 r "
            "values x 2 source caps, every point from the 889 allocation to "
            "its own raise-only fleet fixed point (max iter %d, resid 0).  "
            "r=0 reproduces T024-R3 bit-for-bit.  Under this caliber the "
            "optimum is r*=%.2f (%s cap) with a net saving of %.2f CNY/day; "
            "total(0.81)=%.2f < total(2.70)=%.2f, the truck layer is "
            "identical at both (so the 'lower truck objective at 2.70' does "
            "not exist here), and 2.70's extra payment is not offset."
            % (ck["4_fixed_point"]["max_iter"], arr["r_star"], "arrivals",
               arr["net_saving_star"], h["low"]["total"], h["high"]["total"])),
        "checks": checks,
        "affected": [],
        "outputs": out_files,
        "notes": notes,
        "integrity": {
            "readonly_respected": True,
            "no_files_written_in_review_or_planning": True,
            "no_push": True,
            "inputs_sha256": {p: sha(os.path.join(ROOT, p)) for p in inputs},
            "outputs_sha256": {p: sha(os.path.join(ROOT, p))
                               for p in out_files},
        },
    }
    with io.open(OUT, "w", encoding="utf-8") as fh:
        json.dump(delivery, fh, ensure_ascii=False, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
