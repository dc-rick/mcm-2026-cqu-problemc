"""T003 line rules: one entry per numeric token, in column order.

Entry = (meaning, source_script:line, result_file:field, status, note)
status: ok / no-code / conflict.  The mapper aligns the list with the tokens
found on that paper line; a length mismatch is an error so nothing shifts.
"""

BODY = {}


def R(fname, line, *entries):
    BODY[(fname, line)] = [tuple(e) + ("",) * (5 - len(e)) for e in entries]


OK = "ok"
NC = "no-code"
CF = "conflict"

# ------------------------------------------------------------------ summary
R("summary.tex", 19,
  ("订单记录数（3.21 百万）", "code/00_prepare.py:42-49",
   "data/trips.parquet:n_rows=3,214,096", OK, "3.21e6 为 3,214,096 的四舍五入"),
  ("不同单车数", "code/95_numbers.py:13", "data/trips.parquet:bikeid.nunique=484,754", OK, ""),
  ("不同用户数", "code/95_numbers.py:13", "data/trips.parquet:userid.nunique=349,241", OK, ""))
R("summary.tex", 24,
  ("车队规模（约 3.4×10^5）", "code/02_recon_spatial.py:56",
   "data/runs/T001/fleet_estimate.json:reproduction.value=343,638", OK, "343,638 取整为 3.4e5"),
  ("每车每日骑行次数", "code/95_numbers.py:27-28",
   "q1_params.json:daily_trips ÷ 0.128 ÷ 343,638 = 6.24", OK, "论文四舍五入 6.2"))
R("summary.tex", 26,
  ("交通小区数 Z", "code/11_q1_extra.py:99", "data/cells.parquet:zone.max+1=50", OK, ""),
  ("研究格点数", "code/10_q1_model.py:164", "data/q1_params.json:cells=5,263", OK, ""))
R("summary.tex", 34,
  ("早高峰距离衰减 β", "code/10_q1_model.py:146", "data/q1_params.json:bands.morning 6-9.beta=2.014", OK, ""),
  ("午间距离衰减 β", "code/10_q1_model.py:146", "data/q1_params.json:bands.midday 10-15.beta=2.517", OK, ""))
R("summary.tex", 36,
  ("引力模型 log-loss 下界", "code/10_q1_model.py:142-145", "data/q1_validation.csv:ll_model.min=3.239", OK, ""),
  ("引力模型 log-loss 上界", "code/10_q1_model.py:142-145", "data/q1_validation.csv:ll_model.max=3.517", OK, ""),
  ("仅距离模型 log-loss 下界", "code/10_q1_model.py:142-145", "data/q1_validation.csv:ll_distance.min=3.279", OK, ""),
  ("仅距离模型 log-loss 上界", "code/10_q1_model.py:142-145", "data/q1_validation.csv:ll_distance.max=3.646", OK, ""),
  ("仅吸引模型 log-loss 下界", "code/10_q1_model.py:142-145", "data/q1_validation.csv:ll_attraction.min=7.912", OK, ""),
  ("仅吸引模型 log-loss 上界", "code/10_q1_model.py:142-145", "data/q1_validation.csv:ll_attraction.max=8.185", OK, ""))
R("summary.tex", 37,
  ("Top-5 命中率下界（%）", "code/10_q1_model.py:145", "data/q1_validation.csv:top5_model.min=0.399", OK, "0.399→40%"),
  ("Top-5 命中率上界（%）", "code/10_q1_model.py:145", "data/q1_validation.csv:top5_model.max=0.476", OK, "0.476→48%"))
R("summary.tex", 39,
  ("多用途类出行份额（%）", "code/11_q1_extra.py:77", "data/q1_segments.csv:seg0.share_trips=0.5177", OK, ""),
  ("重短途类出行份额（%）", "code/11_q1_extra.py:77", "data/q1_segments.csv:seg3.share_trips=0.2960", OK, ""))
R("summary.tex", 43,
  ("交通小区数 Z", "code/11_q1_extra.py:99", "data/cells.parquet:zone.max+1=50", OK, ""))
R("summary.tex", 45,
  ("每日搬移自行车数", "code/21_q2_opt.py:121", "data/q2_results.json[alpha=1,K=60].moved=23,959.15", OK, ""),
  ("搬移量占全部出行（%）", "code/21_q2_opt.py:124",
   "q2_results.json[alpha=1,K=60].moved ÷ .trips = 1.118%", OK, "1.118→1.12"),
  ("卡车公里数", "code/21_q2_opt.py:120", "data/q2_results.json[alpha=1,K=60].truck_km=5,440.94", OK, ""))
R("summary.tex", 46,
  ("每车搬移成本（CNY）", "code/95_numbers.py:45-47",
   "q2_results.json[alpha=1,K=60].obj ÷ .moved = 3.478", OK, ""))
R("summary.tex", 48,
  ("最优车队乘数 α", "code/90_figs.py:254-258", "data/q2_results.json:alpha grid 0.4–1.0", OK, ""))
R("summary.tex", 49,
  ("观测车队总成本（千 CNY/日）", "code/90_figs.py:256-258",
   "343,638 + q2_results.json[alpha=1,K=60].obj = 426,963", OK, "→427k"),
  ("最小总成本（千 CNY/日）", "code/90_figs.py:256-258",
   "274,910 + q2_results.json[alpha=0.8,K=80].obj = 390,055", OK, "→390k"))
R("summary.tex", 53,
  ("最优奖励（CNY）", "code/30_q3_incentive.py:158", "data/q3_summary.json:opt.reward=1.200", OK, ""),
  ("最优奖励下投放量", "code/30_q3_incentive.py:158", "data/q3_summary.json:opt.q=115,384.96", OK, ""))
R("summary.tex", 54,
  ("投放量占缺口的份额（%）", "code/30_q3_incentive.py:158",
   "q3_summary.json:opt.q ÷ need_total = 53.33%", OK, ""),
  ("激励账单（千 CNY/日）", "code/30_q3_incentive.py:159", "data/q3_summary.json:opt.bill=138,461.95", OK, "→138k"))
R("summary.tex", 55,
  ("相对全卡车方案节省（%）", "code/30_q3_incentive.py:160",
   "1 − q3_summary.json:opt.total ÷ (truck_unit×need_total) = 34.93%", OK, ""))
R("summary.tex", 56,
  ("激励可达半径（m）", "code/30_q3_incentive.py:51", "q3_summary.json:pot key 450; R0=0.45", OK, ""),
  ("450 m 覆盖缺口份额（%）", "code/30_q3_incentive.py:156", "data/q3_summary.json:cover450=0.71905", OK, ""))
R("summary.tex", 57,
  ("目标覆盖率（%）", "code/30_q3_incentive.py:150", "curve>=0.95", OK, ""),
  ("达到 95% 所需半径（m）", "code/30_q3_incentive.py:150-152",
   "打印 radii[argmax(curve>=0.95)]=1,025 m", OK, ""))
R("summary.tex", 60,
  ("栏杆/电子围栏数", "code/40_q4_efence.py:121", "data/q4_summary.json:n_fences=626", OK, ""))
R("summary.tex", 61,
  ("推荐步行半径（m）", "code/40_q4_efence.py:59", "45_q4 loop R=0.45", OK, ""),
  ("450 m 覆盖率（%）", "code/40_q4_efence.py:121", "data/q4_summary.json:coverage=0.75296", OK, ""))
R("summary.tex", 62,
  ("围栏容量中位数（槽）", "code/40_q4_efence.py:123", "data/q4_summary.json:median_capacity=416.625", OK, ""),
  ("围栏容量最大值（槽）", "code/40_q4_efence.py:125-131", "data/q4_summary.json:by_radius.0.45.max=4,827", OK, ""))
R("summary.tex", 66,
  ("单平台需求份额下限", "code/50_q5_platform.py:21", "data/q5_summary.json:split key 0.35", OK, "s∈{1.0,0.7,0.5,0.35}"))

# ------------------------------------------------------------------ sec1
R("sec1_intro.tex", 19,
  ("过滤后订单数", "code/95_numbers.py:12", "data/trips.parquet:bbox filter rows=3,213,310", OK, ""))
R("sec1_intro.tex", 20,
  ("不同单车数", "code/95_numbers.py:13", "data/trips.parquet:bikeid.nunique=484,754", OK, ""),
  ("不同用户数", "code/95_numbers.py:13", "data/trips.parquet:userid.nunique=349,241", OK, ""))
R("sec1_intro.tex", 56,
  ("1/p 采样放大因子", "code/lib.py:1", "formula constant", OK, "定义式，非数据"))
R("sec1_intro.tex", 123,
  ("午间 β", "code/10_q1_model.py:146", "q1_params.json:bands.midday 10-15.beta=2.517", OK, ""),
  ("早间 β", "code/10_q1_model.py:146", "q1_params.json:bands.morning 6-9.beta=2.014", OK, ""),
  ("Top-5 的 k", "code/10_q1_model.py:105", "k=5 (score 函数默认)", OK, ""),
  ("Top-5 命中率下界", "code/10_q1_model.py:145", "q1_validation.csv:top5_model.min=0.399", OK, ""),
  ("Top-5 命中率上界", "code/10_q1_model.py:145", "q1_validation.csv:top5_model.max=0.476", OK, ""))
R("sec1_intro.tex", 124,
  ("每日搬移量", "code/21_q2_opt.py:121", "q2_results.json[alpha=1,K=60].moved=23,959", OK, ""),
  ("卡车公里数", "code/21_q2_opt.py:120", "q2_results.json[alpha=1,K=60].truck_km=5,441", OK, ""),
  ("单位搬移成本", "code/95_numbers.py:45-47", "q2_results.json:obj/moved=3.478", OK, ""),
  ("最优 α", "code/90_figs.py:254-258", "q2_results.json:alpha grid", OK, ""))
R("sec1_intro.tex", 125,
  ("最优奖励", "code/30_q3_incentive.py:158", "q3_summary.json:opt.reward=1.20", OK, ""),
  ("投放量", "code/30_q3_incentive.py:158", "q3_summary.json:opt.q=115,385", OK, ""),
  ("节省率（%）", "code/30_q3_incentive.py:160", "q3_summary: 1−total/all-truck=34.9%", OK, ""))
R("sec1_intro.tex", 126,
  ("围栏数", "code/40_q4_efence.py:121", "q4_summary.json:n_fences=626", OK, ""),
  ("步行半径（m）", "code/40_q4_efence.py:59", "R=0.45", OK, ""),
  ("覆盖率（%）", "code/40_q4_efence.py:121", "q4_summary.json:coverage=75.30%", OK, ""),
  ("容量中位数", "code/40_q4_efence.py:123", "q4_summary.json:median_capacity=416.6", OK, ""))

# ------------------------------------------------------------------ sec2
R("sec2_assumptions.tex", 13,
  ("日订单量变异系数（%）", "—", "无脚本产出；01_recon.py:23-26 只打印逐日量", NC,
   "可由 trips.parquet 逐日量复算，但无代码行/结果字段"))
R("sec2_assumptions.tex", 17,
  ("15 分钟时间分辨率", "code/lib.py:32", "qtr = hour*4+minute//15", OK, "模型参数"))
R("sec2_assumptions.tex", 19,
  ("15 分钟时间分辨率", "code/lib.py:32", "qtr 定义", OK, ""),
  ("15 分钟计数方差均值比中位数", "—", "无脚本产出", NC, "疑似 no-code（任务单线索）"))
R("sec2_assumptions.tex", 20,
  ("Poisson 参考比值", "—", "定义值 1.0", OK, "理论基准，非数据"))
R("sec2_assumptions.tex", 21,
  ("合并后方差均值比", "—", "无脚本产出", NC, "疑似 no-code（任务单线索）"))
R("sec2_assumptions.tex", 41,
  ("卡车容量 Q（辆/车）", "code/21_q2_opt.py:20", "Q_TRUCK=40", OK, "假设 A6"))
R("sec2_assumptions.tex", 44,
  ("邻域半径上限（km）", "code/21_q2_opt.py:49", "dist<=9.0", OK, ""),
  ("弧数", "code/21_q2_opt.py:53", "len(arcs)=290", OK, ""))
R("sec2_assumptions.tex", 59,
  ("单车日成本（CNY）", "code/90_figs.py:252", "c_bike=1.0", OK, "假设 A9"))
R("sec2_assumptions.tex", 60,
  ("单车三年总成本（CNY）", "—", "假设 A9（1.0×365×3≈1,095）", OK, "文字假设"),
  ("卡车日固定成本（CNY）", "code/21_q2_opt.py:150", "fixed_K_cost=600.0", OK, "假设 A9"))
R("sec2_assumptions.tex", 61,
  ("卡车公里成本（CNY/km）", "code/21_q2_opt.py:21", "COST_KM=6.0", OK, ""))
R("sec2_assumptions.tex", 62,
  ("敏感性扫描下界（CNY/km）", "code/60_sensitivity.py:89", "for ck in (3.0,6.0,12.0)", OK, ""))
R("sec2_assumptions.tex", 63,
  ("敏感性扫描上界（CNY/km）", "code/60_sensitivity.py:89", "12.0", OK, ""))
R("sec2_assumptions.tex", 73,
  ("15 分钟分辨率", "code/lib.py:32", "qtr 定义", OK, ""))
R("sec2_assumptions.tex", 86,
  ("交通小区数 Z", "code/11_q1_extra.py:99", "NZ=50", OK, "notation"))
R("sec2_assumptions.tex", 87,
  ("小时数 T", "code/21_q2_opt.py:39", "T=24", OK, "notation"))

# ------------------------------------------------------------------ sec3
R("sec3_data.tex", 9,
  ("原始订单记录数", "code/00_prepare.py:42-49", "data/trips.parquet:n_rows=3,214,096", OK, ""))
R("sec3_data.tex", 12,
  ("base-32 进制", "—", "geohash 编码定义（literature）", OK, "编码常数"))
R("sec3_data.tex", 14,
  ("geohash 单元纬度边长（m）", "—", "7 字符=35 bit，纬 17 bit → 180/2^17×111.2≈153 m", OK,
   "经向为 117 m；论文写 153×153 为近似"),
  ("geohash 单元纬度边长（m）", "—", "同上", OK, "第二个 153"))
R("sec3_data.tex", 15,
  ("wx4 前缀记录占比（%）", "—", "无脚本产出", NC, "可由 trips.parquet geohash 前缀复算"))
R("sec3_data.tex", 18,
  ("剔除记录数", "—", "3,214,096 − 3,213,310 = 786", OK, "由两个计数相减"))
R("sec3_data.tex", 19,
  ("研究区纬度下界", "code/lib.py:26-27", "load_trips 过滤 39.4–40.4", OK, ""),
  ("研究区纬度上界", "code/lib.py:26-27", "同左", OK, ""),
  ("研究区经度下界", "code/lib.py:26-27", "115.8–117.0", OK, ""),
  ("研究区经度上界", "code/lib.py:26-27", "同左", OK, ""))
R("sec3_data.tex", 20,
  ("离群点纬度（°N）", "code/01_recon.py:52-53", "lat<30 的记录 geohash 解码", OK,
   "01_recon 打印离群 geohash；20°N 为其解码值"),
  ("过滤后订单数", "code/95_numbers.py:12", "trips.parquet bbox 过滤后 3,213,310", OK, ""),
  ("用户数", "code/95_numbers.py:13", "trips.parquet:userid.nunique=349,241", OK, ""))
R("sec3_data.tex", 21,
  ("单车数", "code/95_numbers.py:13", "trips.parquet:bikeid.nunique=484,754", OK, ""))
R("sec3_data.tex", 27,
  ("窗口内日订单量下界", "code/01_recon.py:23-26", "逐日 trips 表；5/13=225,181", OK, ""),
  ("窗口内日订单量上界", "code/01_recon.py:23-26", "5/18=322,141", OK, ""),
  ("窗口起始日（10 日）", "code/lib.py:21", "ALL_DAYS 从 10 日起", OK, "日期标签"))
R("sec3_data.tex", 31,
  ("5/20 后日订单量下界", "code/01_recon.py:23-26", "逐日 trips 表", OK, ""),
  ("5/20 后日订单量上界", "code/01_recon.py:23-26", "逐日 trips 表", OK, ""))
R("sec3_data.tex", 59,
  ("geohash 单元尺度（m）", "—", "同 153 m 推导", OK, ""))
R("sec3_data.tex", 61,
  ("相邻订单精确匹配率（%）", "code/04_behaviour.py:22", "match.mean()=0.1283", OK, "仅打印，未落盘"))
R("sec3_data.tex", 65,
  ("geohash 单元尺度（m）", "—", "同 153 m 推导", OK, ""),
  ("匹配率（%）", "code/04_behaviour.py:22", "match.mean()=0.1283", OK, ""))
R("sec3_data.tex", 68,
  ("采用的采样率 p", "code/04_behaviour.py:22", "match.mean()=0.128 → p=0.128", OK, ""))
R("sec3_data.tex", 76,
  ("1−e^{−μ} 中的 1", "—", "formula constant", OK, ""))
R("sec3_data.tex", 79,
  ("1−e^{−μ} 中的 1", "—", "formula constant", OK, ""))
R("sec3_data.tex", 83,
  ("5/16 的 T/D", "code/60_sensitivity.py:27", "sensitivity.json:fleet_days[6].ratio=1.478", OK, ""),
  ("5/16 的 μ", "code/60_sensitivity.py:27", "sensitivity.json:fleet_days[6].mu=0.840", OK, ""))
R("sec3_data.tex", 85,
  ("车队估计 F=T/μ", "code/02_recon_spatial.py:56",
   "data/runs/T001/fleet_estimate.json:reproduction.value=343,638", CF,
   "与 sec9:18 的 343,605（同为 5/16）相差 33；见冲突清单 C1"))
R("sec3_data.tex", 90,
  ("1−e^{−pμ} 中的 1", "—", "formula constant", OK, ""))
R("sec3_data.tex", 91,
  ("1−e^{−pμ} 中的 1", "—", "formula constant", OK, ""))
R("sec3_data.tex", 92,
  ("重采样下界（%）", "code/60_sensitivity.py:40", "p=0.05", OK, ""))
R("sec3_data.tex", 93,
  ("重采样上界（%）", "code/60_sensitivity.py:40", "p=0.50", OK, ""),
  ("最大偏差（%）", "code/60_sensitivity.py:44", "1−330,586/343,638=3.8%", OK, "论文写 4%"))
R("sec3_data.tex", 96,
  ("车队规模", "code/02_recon_spatial.py:56", "fleet estimate 343,638", CF,
   "与 sec9:18 的 343,605 同一量两值（C1）"),
  ("缩放后日出行量", "code/95_numbers.py:27-28", "q1_params.daily_trips/0.128=2,143,414", OK, "→2.14e6"))
R("sec3_data.tex", 97,
  ("每车每日骑行次数", "code/95_numbers.py:27-28", "2,143,414/343,638=6.24", OK, "→6.2"))
R("sec3_data.tex", 105,
  ("格点边长（°）", "code/lib.py:15", "STEP=0.005", OK, ""),
  ("格点纬度边长（m）", "code/lib.py:18", "0.005×111.2=0.556 km", OK, "→555 m"),
  ("格点经度边长（m）", "code/lib.py:18", "0.005×85.2=0.426 km", OK, "→426 m"))
R("sec3_data.tex", 106,
  ("格点最小样本量", "code/lib.py:16", "THR=50", OK, ""),
  ("研究格点数", "code/10_q1_model.py:164", "q1_params.json:cells=5,263", OK, ""))
R("sec3_data.tex", 107,
  ("格点覆盖典型日出行份额（%）", "code/10_q1_model.py:28", "打印 both-in share（全 14 天）=95.6%", NC,
   "无典型日口径的落盘字段；全 14 天 95.6%、典型日复算 94.8%，均非论文 95.8%"))
R("sec3_data.tex", 113,
  ("小区日需求下界（样本）", "code/11_q1_extra.py:126-128", "d_hour.sum(1).min()=604", OK, ""),
  ("小区日需求上界（样本）", "code/11_q1_extra.py:126-128", "d_hour.sum(1).max()=12,119", OK, ""),
  ("小区尺度（km）", "—", "无脚本产出", NC, "由 1,247 km²/50 得 ≈5 km；论文写 4 km"))
R("sec3_data.tex", 128,
  ("过滤后订单数", "code/95_numbers.py:12", "trips.parquet:3,213,310", OK, ""),
  ("剔除记录数", "—", "3,214,096−3,213,310=786", OK, "表内注"))
R("sec3_data.tex", 129,
  ("用户数", "code/95_numbers.py:13", "trips.parquet:userid.nunique=349,241", OK, ""))
R("sec3_data.tex", 130,
  ("单车数", "code/95_numbers.py:13", "trips.parquet:bikeid.nunique=484,754", OK, ""),
  ("数据覆盖天数", "code/01_recon.py:15", "df.date.nunique()=14", OK, ""))
R("sec3_data.tex", 131,
  ("有数据日期数", "code/01_recon.py:15", "date.nunique()=14（17 日缺失）", OK, ""))
R("sec3_data.tex", 132,
  ("典型日数", "code/lib.py:19-20", "TYP_DAYS=7", OK, ""))
R("sec3_data.tex", 133,
  ("典型日样本出行量", "code/10_q1_model.py:156,164", "q1_params.json:daily_trips=274,357.43", OK, ""))
R("sec3_data.tex", 134,
  ("缩放后日出行量", "code/95_numbers.py:27-28", "274,357/0.128=2,143,414", OK, ""),
  ("采样率 p", "code/04_behaviour.py:22", "p=0.128", OK, "表内注"))
R("sec3_data.tex", 135,
  ("出行距离中位数（km）", "code/01_recon.py:40", "dist_km.describe 50%=0.70", OK, ""),
  ("对数正态 σ", "code/11_q1_extra.py:95-96", "lognorm shape=0.65", OK, "表内注"))
R("sec3_data.tex", 136,
  ("出行距离 P90（km）", "code/01_recon.py:40", "dist_km 90%=1.41", OK, ""))
R("sec3_data.tex", 137,
  ("研究格点数", "code/10_q1_model.py:164", "q1_params.json:cells=5,263", OK, ""),
  ("格点边长（°）", "code/lib.py:15", "STEP=0.005", OK, "表内注"))
R("sec3_data.tex", 138,
  ("交通小区数", "code/11_q1_extra.py:99", "NZ=50", OK, ""))
R("sec3_data.tex", 139,
  ("车队估计", "code/02_recon_spatial.py:56", "343,638", CF, "与 sec9:18 的 343,605 冲突（C1）"))
R("sec3_data.tex", 140,
  ("每车每日骑行次数", "code/95_numbers.py:27-28", "2,143,414/343,638=6.24", OK, "→6.2"))
R("sec3_data.tex", 146,
  ("经验中位数（km）", "code/01_recon.py:40", "median=0.70", OK, ""),
  ("经验 P90（km）", "code/01_recon.py:40", "p90=1.41", OK, ""))
R("sec3_data.tex", 147,
  ("拟合 σ", "code/11_q1_extra.py:96", "shape=0.65", OK, ""))
R("sec3_data.tex", 148,
  ("拟合中位数（km）", "code/11_q1_extra.py:96", "scale=0.66", OK, ""),
  ("中位数差（km）", "—", "0.70−0.66=0.04", OK, "由两值相减"))
R("sec3_data.tex", 149,
  ("可分辨距离下限（m）", "—", "geohash 单元 117–153 m", OK, "量级假设"))

# ------------------------------------------------------------------ sec4
R("sec4_q1.tex", 11,
  ("15 分钟时间分辨率", "code/lib.py:32", "qtr 定义", OK, ""))
R("sec4_q1.tex", 14,
  ("式(6) 的 1/7 因子", "—", "formula constant", OK, ""),
  ("7 个典型日", "code/lib.py:19-20", "TYP_DAYS=7", OK, ""))
R("sec4_q1.tex", 17,
  ("研究格点数", "code/10_q1_model.py:164", "q1_params.json:cells=5,263", OK, ""),
  ("每日时段数", "code/10_q1_model.py:154", "96 个 15 分钟时段", OK, ""),
  ("格点×时段单元数", "—", "5,263×96=505,248", OK, "→5.05e5"))
R("sec4_q1.tex", 19,
  ("单元平均出行数", "—", "274,357/5,263/96=0.543", OK, "paper-side arithmetic"))
R("sec4_q1.tex", 33,
  ("β>0 的下界", "—", "constraint bound", OK, "公式常数"))
R("sec4_q1.tex", 53,
  ("Gaussian 核 σ（格）", "code/10_q1_model.py:50", "smooth_attraction sigma=1.2", OK, "假设"),
  ("σ 对应距离（m）", "—", "1.2×~500 m≈600 m", OK, "近似换算"))
R("sec4_q1.tex", 57,
  ("A_j=0 的定义值", "—", "formula constant", OK, ""),
  ("最忙时段零吸引格点份额（%）", "—", "无脚本产出", NC, "疑似 no-code（任务单线索）"),
  ("夜间零吸引格点份额（%）", "—", "无脚本产出", NC, "疑似 no-code（任务单线索）"))
R("sec4_q1.tex", 78,
  ("β_h=0（仅吸引基准）", "—", "model variant definition", OK, "表注"))
R("sec4_q1.tex", 84,
  ("Top-5 的 k", "code/10_q1_model.py:105", "k=5", OK, "表头"))
R("sec4_q1.tex", 99,
  ("候选目的地格点数", "code/10_q1_model.py:164", "q1_params.json:cells=5,263", OK, ""))
R("sec4_q1.tex", 100,
  ("候选目的地格点数", "code/10_q1_model.py:164", "5,263（式内）", OK, ""),
  ("均匀基准 log-loss = log 5263", "—", "ln(5263)=8.569", OK, "→8.57"))
R("sec4_q1.tex", 102,
  ("仅距离 log-loss 下界", "code/10_q1_model.py:142-145", "q1_validation.ll_distance.min=3.279", OK, ""),
  ("仅距离 log-loss 上界", "code/10_q1_model.py:142-145", "q1_validation.ll_distance.max=3.646", OK, ""))
R("sec4_q1.tex", 105,
  ("引力相对仅距离的增益下限（nats）", "—",
   "q1_validation: ll_distance−ll_model 最小 0.035", OK, "paper-side arithmetic"),
  ("增益上限（nats）", "—", "最大 0.175", OK, ""),
  ("Top-5 增益下限（百分点）", "—", "top5_model−top5_dist 最小 1.66→1.7", OK, ""),
  ("Top-5 增益上限（百分点）", "—",
   "按表内四舍五入 0.476−0.401=7.5；精确值 7.41", OK,
   "与精确值差 0.09 个百分点，属舍入传播"))
R("sec4_q1.tex", 111,
  ("早间 β", "code/10_q1_model.py:146", "q1_params.bands.morning 6-9.beta=2.014", OK, ""))
R("sec4_q1.tex", 112,
  ("午间 β", "code/10_q1_model.py:146", "q1_params.bands.midday 10-15.beta=2.517", OK, ""))
R("sec4_q1.tex", 121,
  ("路网/直线距离系数", "code/11_q1_extra.py:87", "dur=1.35*dist/v", OK, "假设"))
R("sec4_q1.tex", 123,
  ("骑行速度均值（km/h）", "code/11_q1_extra.py:86", "rng.normal(11.5,2.5)", OK, "假设"),
  ("速度标准差", "code/11_q1_extra.py:86", "2.5", OK, ""),
  ("速度裁剪下界", "code/11_q1_extra.py:86", "clip(6,20)", OK, ""),
  ("速度裁剪上界", "code/11_q1_extra.py:86", "20", OK, ""))
R("sec4_q1.tex", 126,
  ("路网系数", "code/11_q1_extra.py:87", "1.35", OK, "公式常数"),
  ("分钟换算", "—", "×60", OK, "公式常数"),
  ("开关锁附加分钟", "code/11_q1_extra.py:87", "+1.0", OK, ""))
R("sec4_q1.tex", 130,
  ("时长中位数（min）", "code/11_q1_extra.py:88-90", "np.median(dur)", OK, "仅打印"),
  ("时长 P10（min）", "code/11_q1_extra.py:88-90", "np.percentile(dur,10)", OK, "仅打印"),
  ("时长 P90（min）", "code/11_q1_extra.py:88-90", "np.percentile(dur,90)", OK, "仅打印"))
R("sec4_q1.tex", 140,
  ("KS 统计量", "code/90_figs.py:155-157",
   "figs/fig3_q1_validation.pdf 面板(a) 文本标注 KS=0.196", OK,
   "仅图中标注（fig3a），未落盘到 data/"))
R("sec4_q1.tex", 161,
  ("聚类数 k", "code/11_q1_extra.py:63-68", "sil 最大者 k=5", OK, ""),
  ("轮廓系数", "code/11_q1_extra.py:63-67", "sil[5]=0.256", OK, "仅打印"))
R("sec4_q1.tex", 162,
  ("备选 k", "code/11_q1_extra.py:63", "k=3", OK, ""),
  ("备选 k", "code/11_q1_extra.py:63", "k=4", OK, ""),
  ("轮廓系数 k=3", "code/11_q1_extra.py:63-67", "sil[3]=0.239", OK, "仅打印"),
  ("轮廓系数 k=4", "code/11_q1_extra.py:63-67", "sil[4]=0.237", OK, "仅打印"))
R("sec4_q1.tex", 187,
  ("多用途类“每日”骑行次数", "—", "q1_segments.seg0.trips=9.18（10 天窗口总量）", CF,
   "表 5 表头为 Trips per day，实为 10 天窗口内总次数；每日应为 0.92（C2）"))
R("sec4_q1.tex", 189,
  ("多用途类出行份额（%）", "code/11_q1_extra.py:77", "q1_segments.seg0.share_trips=0.5177", OK, ""))
R("sec4_q1.tex", 190,
  ("重短途类“每日”骑行次数", "—", "q1_segments.seg3.trips=13.98（10 天窗口）", CF,
   "每日应为 1.40；同 C2"),
  ("重短途类平均距离（km）", "code/11_q1_extra.py:72", "q1_segments.seg3.dist=0.6725", OK, ""))
R("sec4_q1.tex", 192,
  ("通勤类平均距离（km）", "code/11_q1_extra.py:72", "q1_segments.seg2.dist=1.1693", OK, ""),
  ("通勤类高峰出行份额（%）", "code/11_q1_extra.py:74", "q1_segments.seg2.peak=0.9104", OK, ""))
R("sec4_q1.tex", 193,
  ("通勤类出行份额（%）", "code/11_q1_extra.py:77", "q1_segments.seg2.share_trips=0.1061", OK, ""))
R("sec4_q1.tex", 197,
  ("通勤类出行份额（%）", "code/11_q1_extra.py:77", "q1_segments.seg2.share_trips=0.1061", OK, ""))

# ------------------------------------------------------------------ sec5
R("sec5_q2.tex", 14,
  ("1/p 放大因子", "—", "formula constant", OK, ""))
R("sec5_q2.tex", 17,
  ("最小小区午夜存量", "code/20_q2_diag.py:26", "q2_state.npz:I0.min=1,986", OK, ""))
R("sec5_q2.tex", 18,
  ("小区午夜存量中位数", "code/20_q2_diag.py:26", "q2_state.npz:I0.median=6,912", OK, ""))
R("sec5_q2.tex", 19,
  ("最大累计缺口（辆）", "code/20_q2_diag.py:33-38", "q2_state.npz 推导 cum.min=−348.57", OK,
   "论文写 −349（取绝对值）"),
  ("5 分钟分辨率", "code/24_cell_curve.py:20", "step5 定义", OK, ""))
R("sec5_q2.tex", 20,
  ("5 分钟/格点下干涸格点数", "—", "无脚本产出", NC, "疑似 no-code（任务单线索）"),
  ("研究格点数", "code/10_q1_model.py:164", "cells=5,263", OK, ""))
R("sec5_q2.tex", 24,
  ("交通小区数", "code/11_q1_extra.py:99", "NZ=50", OK, ""))
R("sec5_q2.tex", 25,
  ("日终总漂移", "code/21_q2_opt.py:123", "q2_results[alpha=1,K=0].dev=45,118.9", OK, ""))
R("sec5_q2.tex", 36,
  ("邻域半径上限（km）", "code/21_q2_opt.py:49", "dist<=9.0", OK, ""),
  ("弧数", "code/21_q2_opt.py:53", "len(arcs)=290", OK, ""),
  ("平均弧长（km）", "code/21_q2_opt.py:53", "np.mean(dist)=5.97", OK, "仅打印"))
R("sec5_q2.tex", 58,
  ("非负约束 0", "—", "formula constant", OK, ""))
R("sec5_q2.tex", 61,
  ("c_km（CNY/km）", "code/21_q2_opt.py:21", "COST_KM=6.0", OK, ""))
R("sec5_q2.tex", 62,
  ("c_load（CNY/辆）", "code/21_q2_opt.py:22", "COST_LOAD=0.5", OK, ""))
R("sec5_q2.tex", 63,
  ("未服务惩罚 w（CNY）", "code/21_q2_opt.py:23", "PENALTY=6.0", OK, ""))
R("sec5_q2.tex", 65,
  ("c_dev（CNY/辆）", "code/21_q2_opt.py:24", "COST_DEV=3.0", OK, ""))
R("sec5_q2.tex", 67,
  ("卡车日固定成本（CNY）", "code/21_q2_opt.py:150", "fixed_K_cost=600.0", OK, ""))
R("sec5_q2.tex", 76,
  ("卡车容量 Q", "code/21_q2_opt.py:20", "Q_TRUCK=40", OK, ""))
R("sec5_q2.tex", 93,
  ("固定成本 600K 中的 600", "code/21_q2_opt.py:150", "600×K", OK, "表注"))
R("sec5_q2.tex", 116,
  ("α 参数", "code/21_q2_opt.py:145-148", "alpha=1.0", OK, ""),
  ("K 参数", "code/21_q2_opt.py:145-148", "K=60", OK, ""),
  ("搬移量", "code/21_q2_opt.py:121", "q2_results[alpha=1,K=60].moved=23,959", OK, ""),
  ("卡车公里", "code/21_q2_opt.py:120", "q2_results[alpha=1,K=60].truck_km=5,441", OK, ""))
R("sec5_q2.tex", 117,
  ("方案日成本（CNY）", "code/21_q2_opt.py:116", "q2_results[alpha=1,K=60].obj=83,325.24", OK, ""))
R("sec5_q2.tex", 118,
  ("单位搬移成本（CNY）", "code/95_numbers.py:45-47", "obj/moved=3.478", OK, ""))
R("sec5_q2.tex", 124,
  ("搬移量", "code/21_q2_opt.py:121", "23,959", OK, ""))
R("sec5_q2.tex", 125,
  ("日终总漂移", "code/21_q2_opt.py:123", "45,119", OK, ""))
R("sec5_q2.tex", 129,
  ("无卡车时未服务出行", "code/21_q2_opt.py:119", "q2_results[alpha=1,K=0].unmet=1,276.18", OK, ""),
  ("未服务占比（%）", "—", "1,276.18/2,143,414=0.0595%", OK, "→0.06"))
R("sec5_q2.tex", 130,
  ("日终总漂移", "code/21_q2_opt.py:123", "45,119", OK, ""))
R("sec5_q2.tex", 132,
  ("对比方案 K", "code/21_q2_opt.py:149", "K=60", OK, ""),
  ("对比方案 K", "code/21_q2_opt.py:149", "K=150", OK, ""))
R("sec5_q2.tex", 133,
  ("60→150 辆卡车的成本增量（CNY）", "—",
   "q2_results[alpha=1,K=150].obj−[alpha=1,K=60].obj=50,620.7", OK,
   "论文写 50,600（截断到百位），差 21"))
R("sec5_q2.tex", 135,
  ("α 参数", "code/21_q2_opt.py:148", "alpha=0.8", OK, ""),
  ("K 参数", "code/21_q2_opt.py:149", "K=80", OK, ""))
R("sec5_q2.tex", 136,
  ("服务率（%）", "—", "1−816.42/2,143,414=99.962%", OK, "→99.96"),
  ("α 参数", "code/21_q2_opt.py:148", "alpha=0.6", OK, ""),
  ("服务率（%）", "—", "1−34,492.5/2,143,414=98.391%", OK, "→98.4"))
R("sec5_q2.tex", 137,
  ("车队削减幅度（%）", "code/21_q2_opt.py:148", "1−0.6=0.4", OK, ""))
R("sec5_q2.tex", 144,
  ("单车日资本成本（CNY）", "code/90_figs.py:252", "c_bike=1.0", OK, "假设 A9"))
R("sec5_q2.tex", 146,
  ("最优 α", "code/90_figs.py:254-258", "argmin total", OK, ""),
  ("最优车队（辆）", "—", "343,638×0.8=274,910", OK, "→275,000"),
  ("最优卡车数", "code/90_figs.py:254-258", "K=80", OK, ""),
  ("最优日总成本（CNY）", "code/90_figs.py:256-258", "274,910+115,145=390,055", OK, "→390,000"))
R("sec5_q2.tex", 147,
  ("观测车队日总成本（CNY）", "code/90_figs.py:256-258", "343,638+83,325=426,963", OK, "→427,000"))
R("sec5_q2.tex", 148,
  ("α 参数", "code/21_q2_opt.py:148", "0.6", OK, ""),
  ("日总成本（CNY）", "code/90_figs.py:256-258", "206,183+412,379=618,562", OK, "→619,000"))
R("sec5_q2.tex", 151,
  ("α 参数", "code/90_figs.py:254", "0.9", OK, ""),
  ("日总成本（CNY）", "code/90_figs.py:256-258", "309,274+84,628=393,902", OK, "→394,000"),
  ("与最优的差距（%）", "—", "393,902/390,055−1=0.99%", OK, ""))
R("sec5_q2.tex", 152,
  ("车队削减幅度（%）", "—", "1−0.9=0.1", OK, ""),
  ("车队削减幅度（%）", "—", "1−0.6=0.4", OK, ""))

# ------------------------------------------------------------------ sec6
R("sec6_q3.tex", 21,
  ("max(0,·) 的 0", "—", "formula constant", OK, ""))
R("sec6_q3.tex", 23,
  ("缺口总量 N（辆/日）", "code/30_q3_incentive.py:154", "q3_summary.json:need_total=216,343.75", OK, ""))
R("sec6_q3.tex", 28,
  ("研究格点数", "code/10_q1_model.py:164", "cells=5,263", OK, ""),
  ("Δ_c>0 的格点数", "code/30_q3_incentive.py:42", "打印 (need>0).sum()=5,192", OK, ""),
  ("Δ_c>0 的下界 0", "—", "formula constant", OK, ""))
R("sec6_q3.tex", 29,
  ("缺口中位数", "code/30_q3_incentive.py:44", "np.percentile(need,50)=17.9", OK, "仅打印"),
  ("缺口 P90", "code/30_q3_incentive.py:44", "np.percentile(need,90)=98.2", OK, "仅打印"))
R("sec6_q3.tex", 30,
  ("缺口 P99", "code/30_q3_incentive.py:44", "np.percentile(need,99)=323.3", OK, "仅打印"))
R("sec6_q3.tex", 38,
  ("分钟换算 60", "—", "formula constant", OK, ""),
  ("米/公里换算 1000", "—", "formula constant", OK, ""))
R("sec6_q3.tex", 39,
  ("骑行速度 v（km/h）", "code/30_q3_incentive.py:63", "V_BIKE=10.0", OK, "假设 A7"),
  ("步行速度 w（m/min）", "code/30_q3_incentive.py:63", "WALK_M_MIN=80.0", OK, ""))
R("sec6_q3.tex", 44,
  ("logit 分母 1", "—", "formula constant", OK, ""),
  ("logit 分子 1", "—", "formula constant", OK, ""))
R("sec6_q3.tex", 45,
  ("κ", "code/30_q3_incentive.py:64", "KAPPA=2.0", OK, "假设 A7"),
  ("ρ", "code/30_q3_incentive.py:64", "RHO=0.35", OK, ""),
  ("c0", "code/30_q3_incentive.py:64", "C0=0.8", OK, ""))
R("sec6_q3.tex", 49,
  ("ρ 复述", "code/30_q3_incentive.py:64", "RHO=0.35", OK, ""),
  ("c0 复述", "code/30_q3_incentive.py:64", "C0=0.8", OK, ""))
R("sec6_q3.tex", 53,
  ("最优奖励 r*", "code/30_q3_incentive.py:158", "q3_summary.opt.reward=1.20", OK, ""),
  ("50 m 处接受率（%）", "code/30_q3_incentive.py:71-72", "p_acc(1.2,0.05)=0.782", OK,
   "公式可复算，未落盘"))
R("sec6_q3.tex", 54,
  ("50 m 拒绝距离", "code/30_q3_incentive.py:71-72", "r_km=0.05", OK, ""),
  ("225 m 处接受率（%）", "code/30_q3_incentive.py:71-72", "p_acc(1.2,0.225)=0.536", OK, ""),
  ("225 m 拒绝距离", "code/30_q3_incentive.py:71-72", "r_km=0.225", OK, ""),
  ("450 m 处接受率（%）", "code/30_q3_incentive.py:71-72", "p_acc(1.2,0.45)=0.212", OK, ""),
  ("450 m 拒绝距离", "code/30_q3_incentive.py:71-72", "r_km=0.45", OK, ""))
R("sec6_q3.tex", 60,
  ("距离区间下界 0", "—", "formula constant", OK, ""))
R("sec6_q3.tex", 76,
  ("供给曲线半径（m）", "code/30_q3_incentive.py:90", "R0=0.45", OK, "表注"),
  ("供给曲线半径（m）", "code/30_q3_incentive.py:90", "R0=0.45", OK, "表注"))
R("sec6_q3.tex", 78,
  ("卡车单位成本（CNY）", "code/30_q3_incentive.py:109", "TRUCK_UNIT=83,325/23,959=3.478", OK, "表注"))
R("sec6_q3.tex", 101,
  ("“前 10 万次”中的 10 万", "—", "q(1.0)=104,409 的约数", OK, "叙述取整"),
  ("r=1.0 时账单（CNY）", "code/30_q3_incentive.py:163", "q3_summary.supply[r=1.0].bill=104,409", OK, ""))
R("sec6_q3.tex", 102,
  ("平均成本（CNY/次）", "—", "104,409/104,409=1.0", OK, "paper-side arithmetic"),
  ("边际成本（CNY/次）", "—", "q3_summary.supply 差分=2.49（跳过 r=2.5）", OK,
   "paper-side arithmetic；仅打印值无落盘字段"))
R("sec6_q3.tex", 103,
  ("卡车单位成本", "code/30_q3_incentive.py:109", "3.478", OK, ""),
  ("第一档距离带（m）", "code/30_q3_incentive.py:91", "EDGES 0.10 km", OK, ""))
R("sec6_q3.tex", 106,
  ("“10 万次”阈值", "—", "q≈104,409", OK, "叙述取整"))
R("sec6_q3.tex", 109,
  ("r=1.5 边际成本", "—", "q3_summary.supply 差分=3.63", OK, "paper-side arithmetic"),
  ("r=1.5", "code/30_q3_incentive.py:112", "rs 含 1.5", OK, ""),
  ("r=2.0 边际成本", "—", "q3_summary.supply 差分=6.37", OK, "paper-side arithmetic"),
  ("r=2.0", "code/30_q3_incentive.py:112", "rs 含 2.0", OK, ""),
  ("r=3.0 边际成本", "—", "q3_summary.supply 差分=17.31（跳过 r=2.5）", OK,
   "paper-side arithmetic"))
R("sec6_q3.tex", 110,
  ("r=3.0", "code/30_q3_incentive.py:112", "rs 含 3.0", OK, ""),
  ("r=3.5 覆盖率（%）", "code/30_q3_incentive.py:163", "154,846/216,344=71.57%", OK, ""))
R("sec6_q3.tex", 119,
  ("c_truck（CNY）", "code/30_q3_incentive.py:109", "TRUCK_UNIT=3.478", OK, ""))
R("sec6_q3.tex", 123,
  ("区间下界 r", "code/30_q3_incentive.py:112", "1.0", OK, ""),
  ("区间上界 r", "code/30_q3_incentive.py:112", "1.5", OK, ""))
R("sec6_q3.tex", 126,
  ("最优奖励", "code/30_q3_incentive.py:158", "opt.reward=1.20", OK, ""),
  ("最优投放量", "code/30_q3_incentive.py:158", "opt.q=115,384.96", OK, ""),
  ("占缺口份额（%）", "code/30_q3_incentive.py:158", "115,385/216,344=53.33%", OK, ""))
R("sec6_q3.tex", 127,
  ("激励账单（CNY）", "code/30_q3_incentive.py:159", "opt.bill=138,461.95", OK, ""))
R("sec6_q3.tex", 129,
  ("最优处边际成本", "code/30_q3_incentive.py:160", "opt.mc=3.4618", OK, ""),
  ("与卡车成本差（%）", "—", "3.4618/3.4778−1=−0.46%", OK, "论文写 within 1%"),
  ("剩余需卡车搬运量", "—", "216,343.75−115,384.96=100,958.79", OK, "→100,959"))
R("sec6_q3.tex", 130,
  ("卡车搬运成本（CNY）", "—", "3.4778×100,958.79=351,116", OK, "paper-side arithmetic"),
  ("混合方案日成本（CNY）", "code/30_q3_incentive.py:160", "opt.total=489,578.08", OK, ""))
R("sec6_q3.tex", 131,
  ("全卡车基准（CNY）", "—", "3.4778×216,343.75=752,404", OK, "paper-side arithmetic"),
  ("节省率（%）", "code/30_q3_incentive.py:160", "1−489,578/752,404=34.93%", OK, ""))
R("sec6_q3.tex", 140,
  ("r=3.5", "code/30_q3_incentive.py:112", "3.5", OK, ""),
  ("该奖励覆盖率（%）", "code/30_q3_incentive.py:163", "71.57%", OK, ""),
  ("剩余缺口（%）", "—", "100−71.57=28.4", OK, "→28"))
R("sec6_q3.tex", 143,
  ("半径 450 m", "code/30_q3_incentive.py:51", "Rs[0]=0.45", OK, ""),
  ("450 m 覆盖（%）", "code/30_q3_incentive.py:156", "pot450/need=71.91%", OK, ""),
  ("半径 600 m", "code/30_q3_incentive.py:51", "Rs[1]=0.60", OK, ""),
  ("600 m 覆盖（%）", "code/30_q3_incentive.py:161", "pot600/need=83.28%", OK, ""),
  ("半径 850 m", "code/30_q3_incentive.py:51", "Rs[2]=0.85", OK, ""),
  ("850 m 覆盖（%）", "code/30_q3_incentive.py:161", "pot850/need=92.18%", OK, ""))
R("sec6_q3.tex", 144,
  ("目标覆盖率（%）", "code/30_q3_incentive.py:150", "curve>=0.95", OK, ""),
  ("所需半径（m）", "code/30_q3_incentive.py:150-152", "radii[argmax(curve>=0.95)]=1,025", OK, "仅打印"))
R("sec6_q3.tex", 158,
  ("统一奖励（CNY）", "code/30_q3_incentive.py:158", "opt.reward=1.20", OK, "方案"))
R("sec6_q3.tex", 170,
  ("每日预算（CNY）", "—", "方案假设，非模型输出", OK, "约等于 opt.bill 138,462"),
  ("奖励 r", "code/30_q3_incentive.py:158", "1.20", OK, ""))

# ------------------------------------------------------------------ sec7
R("sec7_q4.tex", 12,
  ("15 分钟网格", "code/40_q4_efence.py:24", "q = hour*4+minute//15", OK, ""))
R("sec7_q4.tex", 14,
  ("max{0,·} 的 0", "—", "formula constant", OK, ""))
R("sec7_q4.tex", 18,
  ("每日时段数", "code/40_q4_efence.py:79", "range(96)", OK, ""),
  ("Monte Carlo 重复次数", "code/40_q4_efence.py:93", "reps=300", OK, ""))
R("sec7_q4.tex", 22,
  ("1/p 放大因子", "—", "formula constant", OK, ""))
R("sec7_q4.tex", 31,
  ("(1−1/e) 中的 1", "—", "formula constant", OK, ""),
  ("(1−1/e) 中的 1", "—", "formula constant", OK, ""),
  ("贪心近似保证（%）", "—", "1−1/e=0.632", OK, "理论常数"))
R("sec7_q4.tex", 32,
  ("候选格点数", "code/10_q1_model.py:164", "cells=5,263", OK, ""),
  ("候选站点数", "code/40_q4_efence.py:37", "cand_top=2500", OK, ""))
R("sec7_q4.tex", 53,
  ("推荐半径（m）", "code/40_q4_efence.py:59", "R=0.45", OK, ""))
R("sec7_q4.tex", 55,
  ("围栏数", "code/40_q4_efence.py:121", "n_fences=626", OK, ""),
  ("半径（m）", "code/40_q4_efence.py:59", "R=0.25", OK, ""),
  ("覆盖率（%）", "code/40_q4_efence.py:121", "by_radius.0.25.coverage=38.99%", OK, ""),
  ("围栏数", "code/40_q4_efence.py:121", "by_radius.0.25.fences=505", OK, ""))
R("sec7_q4.tex", 57,
  ("半径（m）", "code/40_q4_efence.py:59", "R=0.65", OK, ""),
  ("覆盖率（%）", "code/40_q4_efence.py:121", "by_radius.0.65.coverage=83.0%", OK, ""),
  ("围栏数", "code/40_q4_efence.py:121", "by_radius.0.65.fences=537", OK, ""))
R("sec7_q4.tex", 58,
  ("可接受的骑行距离（m）", "—", "叙述性假设", NC, "无来源，修辞性距离"))
R("sec7_q4.tex", 61,
  ("边际收益转折点（围栏数）", "code/40_q4_efence.py:118", "q4_frontier c450 曲线", OK, "叙述取整"),
  ("前 200 座围栏", "code/q4_frontier.npz", "c450[199]", OK, ""),
  ("前 200 座覆盖率（%）", "code/40_q4_efence.py:118", "q4_frontier.npz:c450[199]=39.1%", OK, ""))
R("sec7_q4.tex", 62,
  ("第 201–400 座（叙述分组）", "—", "200 座/组；曲线值见下一条", OK, ""),
  ("增量（百分点）", "code/40_q4_efence.py:118", "c450[399]−c450[199]=21", OK, ""),
  ("第 401–600 座（叙述分组）", "—", "200 座/组；曲线值见下一条", OK, ""),
  ("增量（百分点）", "code/40_q4_efence.py:118", "c450[599]−c450[399]=14", OK, ""))
R("sec7_q4.tex", 63,
  ("推荐半径（m）", "code/40_q4_efence.py:59", "R=0.45", OK, ""))
R("sec7_q4.tex", 69,
  ("围栏数", "code/40_q4_efence.py:121", "626", OK, "图注"),
  ("围栏数", "code/40_q4_efence.py:121", "626", OK, "图注"),
  ("半径（m）", "code/40_q4_efence.py:59", "450", OK, "图注"))
R("sec7_q4.tex", 71,
  ("实际容量上限（槽）", "—", "设计假设", OK, "图注；非数据"))
R("sec7_q4.tex", 72,
  ("OSM 半径（m）", "code/50_q5_platform.py:58", "d_ent<=0.3", OK, "图注"))
R("sec7_q4.tex", 73,
  ("OSM 半径（m）", "code/50_q5_platform.py:58", "0.3 km", OK, "图注"))
R("sec7_q4.tex", 81,
  ("容量中位数（槽）", "code/40_q4_efence.py:123", "median_capacity=416.625", OK, ""),
  ("容量最大值（槽）", "code/40_q4_efence.py:125-131", "by_radius.0.45.max=4,827", OK, ""))
R("sec7_q4.tex", 82,
  ("单车占地（m²）", "code/40_q4_efence.py:98", "1.4 m²/slot", OK, "假设"))
R("sec7_q4.tex", 83,
  ("总用地（公顷）", "code/40_q4_efence.py:124", "q4_summary.land_ha=48.22", OK, ""),
  ("足球场数", "—", "48.22/0.72≈67", NC, "无脚本；依赖足球场面积假设"))
R("sec7_q4.tex", 84,
  ("最大围栏用地（公顷）", "—", "4,827×1.4/10,000=0.676", NC, "无脚本，paper-side arithmetic"))
R("sec7_q4.tex", 88,
  ("总容量（槽）", "code/40_q4_efence.py:122", "total_capacity=344,462.65", OK, ""),
  ("车队规模", "code/02_recon_spatial.py:56", "343,638", CF, "与 sec9:18 的 343,605 冲突（C1 的引用）"),
  ("容量冗余（%）", "—", "344,463/343,638−1=0.24%", OK, "→0.2"))
R("sec7_q4.tex", 91,
  ("半径（m）", "code/40_q4_efence.py:59", "0.65", OK, ""),
  ("总容量", "code/40_q4_efence.py:125-131", "by_radius.0.65.total=342,606", OK, ""),
  ("半径（m）", "code/40_q4_efence.py:59", "0.45", OK, ""),
  ("总容量", "code/40_q4_efence.py:125-131", "by_radius.0.45.total=344,463", OK, ""),
  ("两方案差异（%）", "—", "344,463/342,606−1=0.54%", OK, "→0.6"))
R("sec7_q4.tex", 92,
  ("半径（m）", "code/40_q4_efence.py:59", "0.25", OK, ""))
R("sec7_q4.tex", 93,
  ("未覆盖份额（%）", "—", "1−0.39=61%", OK, ""))
R("sec7_q4.tex", 94,
  ("单点集中容量（槽）", "—", "叙述性假设", NC, "无来源，修辞值"))
R("sec7_q4.tex", 96,
  ("最忙格点份额（%）", "—", "无脚本产出", NC, "疑似 no-code（任务单线索）"))
R("sec7_q4.tex", 97,
  ("最忙 10% 格点承接份额（%）", "—", "无脚本产出", NC, "疑似 no-code（任务单线索）"))
R("sec7_q4.tex", 100,
  ("单围栏容量上限（槽）", "—", "设计假设", OK, "假设"),
  ("占地（m²）", "—", "150×1.4=210", OK, "由假设换算"))
R("sec7_q4.tex", 102,
  ("超过 150 槽的围栏份额（%）", "—", "无脚本产出", NC,
   "可由 q4_frontier.npz:caps 复算，但无代码行"))
R("sec7_q4.tex", 115,
  ("OSM 地铁口数", "code/50_q5_platform.py:109", "q5_summary.osm_entrances=1,097", OK, ""))
R("sec7_q4.tex", 116,
  ("全部格点临近份额（%）", "code/50_q5_platform.py:112", "baseline_near_entrance=0.08246", OK, ""),
  ("半径（m）", "code/50_q5_platform.py:58", "0.3 km", OK, ""),
  ("Top-300 份额（%）", "code/50_q5_platform.py:111", "top300_near_entrance=0.15667", OK, ""),
  ("半径（m）", "code/50_q5_platform.py:58", "0.3 km", OK, ""))
R("sec7_q4.tex", 118,
  ("Top-100 格点数", "code/50_q5_platform.py:113", "argsort(-Ad)[:100]", OK, ""),
  ("Top-100 到地铁口中位距离（km）", "code/50_q5_platform.py:113", "median_dist_top100=0.496", OK, ""),
  ("全部格点中位距离（km）", "code/50_q5_platform.py:114", "median_dist_all=1.215", OK, ""))
R("sec7_q4.tex", 119,
  ("半径（m）", "code/50_q5_platform.py:58", "0.3 km", OK, ""))

# ------------------------------------------------------------------ sec8
R("sec8_q5.tex", 18,
  ("平台份额 s", "code/50_q5_platform.py:21", "share=1.0", OK, ""),
  ("平台份额 s", "code/50_q5_platform.py:21", "share=0.7", OK, ""),
  ("平台份额 s", "code/50_q5_platform.py:21", "share=0.5", OK, ""),
  ("平台份额 s", "code/50_q5_platform.py:21", "share=0.35", OK, ""))
R("sec8_q5.tex", 38,
  ("1/n 中的 1", "—", "formula constant", OK, ""))
R("sec8_q5.tex", 43,
  ("平台数 1", "—", "表头", OK, ""),
  ("平台数 2", "—", "表头", OK, ""),
  ("平台数 3", "—", "表头", OK, ""))
R("sec8_q5.tex", 45,
  ("单平台容量中位数（槽）", "code/50_q5_platform.py:109", "capacity_duplication.median.one=417", OK, ""),
  ("两平台总容量（槽）", "code/50_q5_platform.py:108-115", "2×230.05=460.1", OK,
   "标签歧义（非数值冲突）：单元格是该地点跨平台总槽位，q5_summary 存单平台 230.05；建议行标签改 Total slots per fence location"),
  ("增量（%）", "code/50_q5_platform.py:102", "2×230.05/417.05−1=10.3%", OK, ""),
  ("三平台总容量（槽）", "code/50_q5_platform.py:108-115", "3×172.0=516.0", OK,
   "标签歧义（非数值冲突）：跨平台总槽位=3×172.0，q5_summary 存单平台 172.0"),
  ("增量（%）", "code/50_q5_platform.py:102", "3×172.0/417.05−1=23.7%", OK, ""))
R("sec8_q5.tex", 46,
  ("单平台最大容量（槽）", "code/50_q5_platform.py:109", "largest.one=4,827", OK, ""),
  ("两平台总容量（槽）", "code/50_q5_platform.py:108-115", "2×2,451=4,902", OK,
   "标签歧义（非数值冲突）：跨平台总槽位=2×2,451，q5_summary 存单平台 2,451"),
  ("增量（%）", "code/50_q5_platform.py:102", "+1.6%", OK, ""),
  ("三平台总容量（槽）", "code/50_q5_platform.py:108-115", "3×1,656=4,968", OK,
   "标签歧义（非数值冲突）：跨平台总槽位=3×1,656.05，q5_summary 存单平台 1,656.05"),
  ("增量（%）", "code/50_q5_platform.py:102", "+2.9%", OK, ""))
R("sec8_q5.tex", 50,
  ("三平台容量增量（%）", "code/50_q5_platform.py:102", "23.7%→24%", OK, ""))
R("sec8_q5.tex", 56,
  ("弧数", "code/21_q2_opt.py:53", "len(arcs)=290", OK, ""))
R("sec8_q5.tex", 58,
  ("卡车日固定成本（CNY）", "code/21_q2_opt.py:150", "600", OK, ""),
  ("固定成本占日成本（%）", "—", "600×60/83,325=43.2%", OK, "paper-side arithmetic"))
R("sec8_q5.tex", 59,
  ("K=60", "code/21_q2_opt.py:145-148", "K=60", OK, ""))
R("sec8_q5.tex", 64,
  ("√3 的 3", "—", "formula constant", OK, ""))
R("sec8_q5.tex", 66,
  ("缺口中位数", "code/30_q3_incentive.py:44", "17.9", OK, ""),
  ("中位格点标准误（辆）", "—", "sqrt(17.9)=4.23", NC, "解析推导，无脚本产出"))
R("sec8_q5.tex", 67,
  ("1/3 份额标准误（辆）", "—", "sqrt(3×17.9)=7.33", NC, "解析推导，无脚本产出"))
R("sec8_q5.tex", 78,
  ("1/n 中的 1", "—", "formula constant", OK, ""))
R("sec8_q5.tex", 101,
  ("三平台容量增量（%）", "code/50_q5_platform.py:102", "23.7%→24%", OK, ""))
R("sec8_q5.tex", 109,
  ("α 参数", "code/90_figs.py:254", "0.8", OK, ""))
R("sec8_q5.tex", 116,
  ("最忙格点份额（%）", "—", "无脚本产出", NC, "疑似 no-code（任务单线索）"),
  ("承接出行份额（%）", "—", "无脚本产出", NC, "疑似 no-code（任务单线索）"))

# ------------------------------------------------------------------ sec9
R("sec9_sensitivity.tex", 17,
  ("日期 10 日", "code/60_sensitivity.py:33", "fleet_days[0].date=2017-05-10", OK, ""),
  ("日期 11 日", "code/60_sensitivity.py:33", "fleet_days[1].date", OK, ""),
  ("日期 12 日", "code/60_sensitivity.py:33", "fleet_days[2].date", OK, ""),
  ("日期 15 日", "code/60_sensitivity.py:33", "fleet_days[5].date", OK, ""),
  ("日期 16 日", "code/60_sensitivity.py:33", "fleet_days[6].date", OK, ""))
R("sec9_sensitivity.tex", 18,
  ("5/10 车队估计", "code/60_sensitivity.py:29", "sensitivity.json:fleet_days[0].fleet=313,181", OK, ""),
  ("5/11 车队估计", "code/60_sensitivity.py:29", "fleet_days[1].fleet=332,398", OK, ""),
  ("5/12 车队估计", "code/60_sensitivity.py:29", "fleet_days[2].fleet=325,991", OK, ""),
  ("5/15 车队估计", "code/60_sensitivity.py:29", "fleet_days[5].fleet=336,788", OK, ""),
  ("5/16 车队估计", "code/60_sensitivity.py:29", "fleet_days[6].fleet=343,604.83", CF,
   "与 sec3:85/表3 的 343,638 相差 33（C1）"))
R("sec9_sensitivity.tex", 20,
  ("日期 13 日", "code/60_sensitivity.py:33", "fleet_days[3].date", OK, ""),
  ("日期 14 日", "code/60_sensitivity.py:33", "fleet_days[4].date", OK, ""),
  ("日期 18 日", "code/60_sensitivity.py:33", "fleet_days[7].date", OK, ""),
  ("日期 19 日", "code/60_sensitivity.py:33", "fleet_days[8].date", OK, ""),
  ("日期 21 日", "code/60_sensitivity.py:33", "fleet_days[9].date", OK, ""))
R("sec9_sensitivity.tex", 21,
  ("5/13 车队估计", "code/60_sensitivity.py:29", "fleet_days[3].fleet=326,046", OK, ""),
  ("5/14 车队估计", "code/60_sensitivity.py:29", "fleet_days[4].fleet=326,690", OK, ""),
  ("5/18 车队估计", "code/60_sensitivity.py:29", "fleet_days[7].fleet=292,831", OK, ""),
  ("5/19 车队估计", "code/60_sensitivity.py:29", "fleet_days[8].fleet=300,887", OK, ""),
  ("5/21 车队估计", "code/60_sensitivity.py:29", "fleet_days[9].fleet=306,947", OK, ""))
R("sec9_sensitivity.tex", 26,
  ("最小值", "code/60_sensitivity.py:31", "fleet.min()=292,831", OK, ""),
  ("最大值", "code/60_sensitivity.py:31", "fleet.max()=343,605", OK, ""),
  ("均值", "code/60_sensitivity.py:31", "fleet.mean()=320,536", OK, ""))
R("sec9_sensitivity.tex", 27,
  ("标准差", "code/60_sensitivity.py:31", "fleet.std()=15,580", OK, "总体标准差"),
  ("变异系数（%）", "—", "15,580/320,536=4.86%", OK, "→4.9"),
  ("日期 18 日", "code/60_sensitivity.py:33", "fleet_days[7].date", OK, ""),
  ("日期 19 日", "code/60_sensitivity.py:33", "fleet_days[8].date", OK, ""))
R("sec9_sensitivity.tex", 28,
  ("5/18 的 T/D", "code/60_sensitivity.py:27", "fleet_days[7].ratio=1.649", OK, ""))
R("sec9_sensitivity.tex", 29,
  ("参照 T/D", "code/60_sensitivity.py:27", "fleet_days[2].ratio=1.461", OK,
   "1.46 为 5/12；5/16 为 1.478"))
R("sec9_sensitivity.tex", 31,
  ("采用的周中车队值", "code/02_recon_spatial.py:56", "343,638", CF,
   "与本节表 5/16 的 343,605 相差 33（C1）"))
R("sec9_sensitivity.tex", 36,
  ("重采样下界（%）", "code/60_sensitivity.py:40", "p=0.05", OK, ""),
  ("重采样上界（%）", "code/60_sensitivity.py:40", "p=0.50", OK, ""),
  ("重采样最小值", "code/60_sensitivity.py:44", "sensitivity.json:sampling_invariance[1][1]=330,586（p=12.8%）", OK, ""),
  ("重采样最大值", "code/60_sensitivity.py:44", "sensitivity.json:sampling_invariance[0][1]=338,401（p=5%）", OK, ""),
  ("最大偏差（%）", "—", "1−330,586/343,638=3.8%", OK, "→4"))
R("sec9_sensitivity.tex", 42,
  ("κ 基线", "code/30_q3_incentive.py:64", "2.0", OK, ""),
  ("ρ 基线", "code/30_q3_incentive.py:64", "0.35", OK, ""),
  ("c0 基线", "code/30_q3_incentive.py:64", "0.8", OK, ""))
R("sec9_sensitivity.tex", 46,
  ("全卡车基准（CNY/日）", "code/30_q3_incentive.py:160", "752,404", OK, "表注"))
R("sec9_sensitivity.tex", 55,
  ("最优奖励", "code/30_q3_incentive.py:158", "1.20", OK, ""),
  ("投放量", "code/30_q3_incentive.py:158", "115,385", OK, ""),
  ("总成本", "code/30_q3_incentive.py:160", "489,578", OK, ""),
  ("节省率（%）", "code/30_q3_incentive.py:160", "34.93", OK, ""))
R("sec9_sensitivity.tex", 56,
  ("κ 变体", "code/60_sensitivity.py:77", "kappa low κ=1.0", OK, ""),
  ("最优奖励", "code/60_sensitivity.py:83", "q3_sensitivity[1].reward=1.20", OK, ""),
  ("投放量", "code/60_sensitivity.py:83", "q3_sensitivity[1].q=78,709", OK, ""),
  ("总成本", "code/60_sensitivity.py:83", "q3_sensitivity[1].total=573,116", OK, ""),
  ("节省率（%）", "—", "1−573,116/752,404=23.83%", OK, ""))
R("sec9_sensitivity.tex", 57,
  ("κ 变体", "code/60_sensitivity.py:77", "kappa high κ=3.0", OK, ""),
  ("最优奖励", "code/60_sensitivity.py:83", "q3_sensitivity[2].reward=1.05", OK, ""),
  ("投放量", "code/60_sensitivity.py:83", "q3_sensitivity[2].q=131,671", OK, ""),
  ("总成本", "code/60_sensitivity.py:83", "q3_sensitivity[2].total=432,730", OK, ""),
  ("节省率（%）", "—", "1−432,730/752,404=42.49%", OK, ""))
R("sec9_sensitivity.tex", 58,
  ("ρ 变体", "code/60_sensitivity.py:77", "rho low ρ=0.20", OK, ""),
  ("最优奖励", "code/60_sensitivity.py:83", "q3_sensitivity[3].reward=1.00", OK, ""),
  ("投放量", "code/60_sensitivity.py:83", "q3_sensitivity[3].q=123,685", OK, ""),
  ("总成本", "code/60_sensitivity.py:83", "q3_sensitivity[3].total=445,935", OK, ""),
  ("节省率（%）", "—", "1−445,935/752,404=40.73%", OK, ""))
R("sec9_sensitivity.tex", 59,
  ("ρ 变体", "code/60_sensitivity.py:77", "rho high ρ=0.50", OK, ""),
  ("最优奖励", "code/60_sensitivity.py:83", "q3_sensitivity[4].reward=1.30", OK, ""),
  ("投放量", "code/60_sensitivity.py:83", "q3_sensitivity[4].q=102,666", OK, ""),
  ("总成本", "code/60_sensitivity.py:83", "q3_sensitivity[4].total=528,814", OK, ""),
  ("节省率（%）", "—", "1−528,814/752,404=29.72%", OK, ""))
R("sec9_sensitivity.tex", 60,
  ("c0 变体", "code/60_sensitivity.py:77", "inertia low c0=0.4", OK, ""),
  ("最优奖励", "code/60_sensitivity.py:83", "q3_sensitivity[5].reward=1.05", OK, ""),
  ("投放量", "code/60_sensitivity.py:83", "q3_sensitivity[5].q=117,884", OK, ""),
  ("总成本", "code/60_sensitivity.py:83", "q3_sensitivity[5].total=466,202", OK, ""),
  ("节省率（%）", "—", "1−466,202/752,404=38.04%", OK, ""))
R("sec9_sensitivity.tex", 61,
  ("c0 变体", "code/60_sensitivity.py:77", "inertia high c0=1.2", OK, ""),
  ("最优奖励", "code/60_sensitivity.py:83", "q3_sensitivity[6].reward=1.35", OK, ""),
  ("投放量", "code/60_sensitivity.py:83", "q3_sensitivity[6].q=112,779", OK, ""),
  ("总成本", "code/60_sensitivity.py:83", "q3_sensitivity[6].total=512,429", OK, ""),
  ("节省率（%）", "—", "1−512,429/752,404=31.89%", OK, ""))
R("sec9_sensitivity.tex", 67,
  ("最优奖励下界", "code/60_sensitivity.py:83", "min reward=1.00", OK, ""),
  ("最优奖励上界", "code/60_sensitivity.py:83", "max reward=1.35", OK, ""))
R("sec9_sensitivity.tex", 68,
  ("节省率下界（%）", "code/60_sensitivity.py:83", "min saving=23.8", OK, ""),
  ("节省率上界（%）", "code/60_sensitivity.py:83", "max saving=42.5", OK, ""))
R("sec9_sensitivity.tex", 70,
  ("投放量最大/最小比", "—", "131,671/78,709=1.67", OK, "→1.7"))
R("sec9_sensitivity.tex", 75,
  ("α", "code/60_sensitivity.py:91", "alpha=1.0", OK, ""),
  ("K", "code/60_sensitivity.py:91", "K=60", OK, ""),
  ("c_km 下界", "code/60_sensitivity.py:89", "3.0", OK, ""),
  ("c_km 中值", "code/60_sensitivity.py:89", "6.0", OK, ""))
R("sec9_sensitivity.tex", 76,
  ("c_km 上界", "code/60_sensitivity.py:89", "12.0", OK, ""))
R("sec9_sensitivity.tex", 84,
  ("c_km=3", "code/60_sensitivity.py:93", "q2_cost_km[0].cost_km=3", OK, ""),
  ("搬移量", "code/60_sensitivity.py:93", "q2_cost_km[0].moved=25,587", OK, ""),
  ("卡车公里", "code/60_sensitivity.py:93", "q2_cost_km[0].truck_km=5,814", OK, ""),
  ("日成本", "code/60_sensitivity.py:93", "q2_cost_km[0].obj=66,242", OK, ""))
R("sec9_sensitivity.tex", 85,
  ("c_km=6", "code/60_sensitivity.py:93", "q2_cost_km[1].cost_km=6", OK, ""),
  ("搬移量", "code/60_sensitivity.py:93", "q2_cost_km[1].moved=23,959", OK, ""),
  ("卡车公里", "code/60_sensitivity.py:93", "q2_cost_km[1].truck_km=5,441", OK, ""),
  ("日成本", "code/60_sensitivity.py:93", "q2_cost_km[1].obj=83,325", OK, ""))
R("sec9_sensitivity.tex", 86,
  ("c_km=12", "code/60_sensitivity.py:93", "q2_cost_km[2].cost_km=12", OK, ""),
  ("搬移量", "code/60_sensitivity.py:93", "q2_cost_km[2].moved=21,835", OK, ""),
  ("卡车公里", "code/60_sensitivity.py:93", "q2_cost_km[2].truck_km=4,906", OK, ""),
  ("日成本", "code/60_sensitivity.py:93", "q2_cost_km[2].obj=113,964", OK, ""))
R("sec9_sensitivity.tex", 91,
  ("四倍价格下搬移量降幅（%）", "—", "1−21,835/25,587=14.7%", OK, "→15"))
R("sec9_sensitivity.tex", 103,
  ("研究格点数", "code/10_q1_model.py:164", "cells=5,263", OK, ""),
  ("车队规模", "code/02_recon_spatial.py:56", "343,638", CF, "与 343,605 冲突（C1）"))
R("sec9_sensitivity.tex", 108,
  ("cell 模拟步行半径（m）", "code/24_cell_curve.py:47", "300 m 半径（sim_cell）", OK, ""),
  ("α=1 未服务率（%）", "code/24_cell_curve.py:53", "cell_fleet_curve.npy[0,1]=13.43%", OK, ""))
R("sec9_sensitivity.tex", 109,
  ("α=1", "code/24_cell_curve.py:44", "a=1.0", OK, ""),
  ("α=0.2 未服务率（%）", "code/24_cell_curve.py:53", "cell_fleet_curve.npy[6,1]=40.28%", OK, ""),
  ("α=0.2", "code/24_cell_curve.py:44", "a=0.2", OK, ""))
R("sec9_sensitivity.tex", 122,
  ("研究区面积（km²）", "—", "5,263×0.556×0.426≈1,247", NC,
   "无脚本产出；由格点数与格点尺寸可复算"))
R("sec9_sensitivity.tex", 123,
  ("每车每日骑行次数", "code/95_numbers.py:27-28", "6.24", OK, "→6.2"))

# ------------------------------------------------------------------ sec10
R("sec10_evaluation.tex", 9,
  ("卡车单位成本（CNY）", "code/30_q3_incentive.py:109", "TRUCK_UNIT=3.478", OK, ""),
  ("奖励（CNY）", "code/30_q3_incentive.py:158", "1.20", OK, ""))
R("sec10_evaluation.tex", 23,
  ("OSM 地铁口数", "code/50_q5_platform.py:109", "1,097", OK, ""))
R("sec10_evaluation.tex", 38,
  ("推荐奖励（CNY）", "code/30_q3_incentive.py:158", "1.20", OK, ""))
R("sec10_evaluation.tex", 44,
  ("K=60 的卡车数", "code/21_q2_opt.py:145-148", "60", OK, ""))
R("sec10_evaluation.tex", 61,
  ("假设失效时的奖励阈值（CNY）", "—", "外推情景，无脚本产出", NC,
   "κ<1.0 未在表 9 扫描，无法核对"))
R("sec10_evaluation.tex", 64,
  ("四倍价格下搬移量降幅（%）", "—", "1−21,835/25,587=14.7%", OK, "→15"))
R("sec10_evaluation.tex", 69,
  ("最大累计缺口（辆）", "code/20_q2_diag.py:33-38", "|cum.min|=348.57", OK, "论文写 349"),
  ("最小小区午夜存量", "code/20_q2_diag.py:26", "q2_state.npz:I0.min=1,986", OK, ""))
R("sec10_evaluation.tex", 82,
  ("夜间搬移量（约）", "code/21_q2_opt.py:121", "23,959→24,000", OK, "取整"))
R("sec10_evaluation.tex", 83,
  ("占日出行（%）", "—", "23,959/2,143,414=1.118%", OK, "→1.1"))
R("sec10_evaluation.tex", 85,
  ("奖励（CNY）", "code/30_q3_incentive.py:158", "1.20", OK, ""))
R("sec10_evaluation.tex", 87,
  ("围栏数", "code/40_q4_efence.py:121", "626", OK, ""))

# ------------------------------------------------------------------ ai_report
R("ai_report.tex", 15,
  ("原始文件大小（MB）", "code/00_prepare.py:9",
   "raw CSV 191,497,836 B = 191.5 MB (10^6)", OK, "按 10^6 换算"))
R("ai_report.tex", 44,
  ("OSM 地铁口数", "code/50_q5_platform.py:109", "1,097", OK, ""))

# ------------------------------------------------------- table 4: tab:q1
_Q1_TAB = [
    (87, "night 0-5", 18364, 2.0963, -3.312718, -3.4878013, -7.912348,
     -8.568457, 0.47550535, 0.40142688),
    (88, "morning 6-9", 386102, 2.0144, -3.511614, -3.6455083, -8.082624,
     -8.568457, 0.39864895, 0.33920097),
    (89, "midday 10-15", 366212, 2.5175, -3.2389507, -3.2790344, -8.091006,
     -8.568456, 0.45753106, 0.4406289),
    (90, "evening 16-19", 428478, 2.1297, -3.5171754, -3.5624664, -8.139999,
     -8.568457, 0.40834293, 0.38337475),
    (91, "late 20-23", 165671, 2.3640, -3.3600483, -3.3950362, -8.184953,
     -8.568456, 0.4547078, 0.43812925),
]
for _ln, _band, _ntr, _b, _lm, _ld, _la, _lu, _tm, _td in _Q1_TAB:
    R("sec4_q1.tex", _ln,
      ("训练集出行数", "code/10_q1_model.py:146",
       f"q1_params.json:bands.{_band}.n_train={_ntr:,}", OK, "表4"),
      ("距离衰减 β_h（km⁻¹）", "code/10_q1_model.py:146",
       f"q1_params.json:bands.{_band}.beta={_b:.4f}", OK, "表4"),
      ("引力模型 log-loss", "code/10_q1_model.py:142-145,169",
       f"q1_validation.csv:band={_band}.ll_model={_lm}", OK, "表4"),
      ("仅距离模型 log-loss", "code/10_q1_model.py:142-145,169",
       f"q1_validation.csv:band={_band}.ll_distance={_ld}", OK, "表4"),
      ("仅吸引模型 log-loss", "code/10_q1_model.py:142-145,169",
       f"q1_validation.csv:band={_band}.ll_attraction={_la}", OK, "表4"),
      ("均匀基准 log-loss", "code/10_q1_model.py:139-145,169",
       f"q1_validation.csv:band={_band}.ll_uniform={_lu}", OK, "表4"),
      ("Top-5 命中率（引力）", "code/10_q1_model.py:138-145,169",
       f"q1_validation.csv:band={_band}.top5_model={_tm}", OK, "表4"),
      ("Top-5 命中率（仅距离）", "code/10_q1_model.py:141-145,169",
       f"q1_validation.csv:band={_band}.top5_dist={_td}", OK, "表4"))

# ------------------------------------------------------- table 5: tab:seg
_SEG_TAB = [
    (177, 0, "Regular multi-purpose", 132984, 40.4, 51.8, 9.18, 9.179683,
     0.84, 0.19),
    (178, 3, "Heavy short-trip", 49931, 15.2, 29.6, 13.98, 13.979151,
     0.67, 0.20),
    (179, 2, "Commuter", 67528, 20.5, 10.6, 3.71, 3.705500, 1.17, 0.07),
    (180, 1, "Low frequency", 49396, 15.0, 5.2, 2.48, 2.481254, 0.89, 0.03),
    (181, 4, "Weekend / leisure", 29410, 8.9, 2.8, 2.25, 2.254097,
     0.92, 0.79),
]
for _ln, _s, _name, _u, _su, _st, _tp, _traw, _dist, _we in _SEG_TAB:
    R("sec4_q1.tex", _ln,
      ("用户数", "code/11_q1_extra.py:72-78",
       f"q1_segments.csv:seg={_s}.users={_u:,}", OK, "表5"),
      ("用户份额（%）", "code/11_q1_extra.py:79",
       f"q1_segments.csv:seg={_s}.share_users={_su / 100:.4f}", OK, "表5"),
      ("出行份额（%）", "code/11_q1_extra.py:79",
       f"q1_segments.csv:seg={_s}.share_trips={_st / 100:.4f}", OK, "表5"),
      ("每用户每日出行次数", "code/11_q1_extra.py:72-78,80",
       f"q1_segments.csv:seg={_s}.trips={_traw:.4f}（10 天窗口）", CF,
       f"表头为 Trips per day，字段实为 10 天窗口总次数；每日={_tp / 10:.2f}（C2）"),
      ("平均距离（km）", "code/11_q1_extra.py:72-78",
       f"q1_segments.csv:seg={_s}.dist={_dist}", OK, "表5"),
      ("周末份额", "code/11_q1_extra.py:74-78",
       f"q1_segments.csv:seg={_s}.weekend={_we}", OK, "表5"))

# ------------------------------------------------------- table 6: tab:q2
_Q2_TAB = [
    (101, 1.0, 0, 0, 0, 1276, 45119, 143014),
    (102, 1.0, 20, 9600, 1739, 0, 27195, 108820),
    (103, 1.0, 40, 19200, 4125, 0, 7995, 82333),
    (104, 1.0, 60, 23959, 5441, 0, None, None),
    (105, None, None, None, None, None, 900, 83325),
    (106, 1.0, 150, 24409, 4990, 0, 600, 133946),
    (107, 0.9, 60, 25403, 5688, 0, 600, 84628),
    (108, 0.8, 80, 31632, 7330, 816, 816, 115145),
    (109, 0.7, 100, 34271, 7860, 10430, 10430, 218167),
    (110, 0.6, 80, 28794, 6592, 34493, 34493, 412379),
]


def _q2row(a, k):
    return f"q2_results.json:alpha={a},K={k}"


R("sec5_q2.tex", 101,
  ("α（车队乘数）", "code/21_q2_opt.py:145-148", _q2row(1.0, 0) + ".alpha=1.0", OK, "表6"),
  ("卡车数 K", "code/21_q2_opt.py:145-148", _q2row(1.0, 0) + ".K=0", OK, "表6"),
  ("搬移量（辆）", "code/21_q2_opt.py:121", _q2row(1.0, 0) + ".moved=0", OK, "表6"),
  ("卡车公里", "code/21_q2_opt.py:120", _q2row(1.0, 0) + ".truck_km=0", OK, "表6"),
  ("未服务出行", "code/21_q2_opt.py:119", _q2row(1.0, 0) + ".unmet=1,276.18", OK, "表6"),
  ("日终漂移", "code/21_q2_opt.py:123", _q2row(1.0, 0) + ".dev=45,118.91", OK, "表6"),
  ("日成本（CNY）", "code/21_q2_opt.py:116", _q2row(1.0, 0) + ".obj=143,013.79", OK, "表6"))
R("sec5_q2.tex", 102,
  ("α", "code/21_q2_opt.py:145-148", _q2row(1.0, 20) + ".alpha=1.0", OK, "表6"),
  ("K", "code/21_q2_opt.py:145-148", _q2row(1.0, 20) + ".K=20", OK, "表6"),
  ("搬移量", "code/21_q2_opt.py:121", _q2row(1.0, 20) + ".moved=9,600", OK, "表6"),
  ("卡车公里", "code/21_q2_opt.py:120", _q2row(1.0, 20) + ".truck_km=1,739.20", OK, "表6"),
  ("未服务出行", "code/21_q2_opt.py:119", _q2row(1.0, 20) + ".unmet=0", OK, "表6"),
  ("日终漂移", "code/21_q2_opt.py:123", _q2row(1.0, 20) + ".dev=27,195.09", OK, "表6"),
  ("日成本", "code/21_q2_opt.py:116", _q2row(1.0, 20) + ".obj=108,820.44", OK, "表6"))
R("sec5_q2.tex", 103,
  ("α", "code/21_q2_opt.py:145-148", _q2row(1.0, 40) + ".alpha=1.0", OK, "表6"),
  ("K", "code/21_q2_opt.py:145-148", _q2row(1.0, 40) + ".K=40", OK, "表6"),
  ("搬移量", "code/21_q2_opt.py:121", _q2row(1.0, 40) + ".moved=19,200", OK, "表6"),
  ("卡车公里", "code/21_q2_opt.py:120", _q2row(1.0, 40) + ".truck_km=4,124.61", OK, "表6"),
  ("未服务出行", "code/21_q2_opt.py:119", _q2row(1.0, 40) + ".unmet=0", OK, "表6"),
  ("日终漂移", "code/21_q2_opt.py:123", _q2row(1.0, 40) + ".dev=7,995.09", OK, "表6"),
  ("日成本", "code/21_q2_opt.py:116", _q2row(1.0, 40) + ".obj=82,332.94", OK, "表6"))
R("sec5_q2.tex", 104,
  ("α", "code/21_q2_opt.py:145-148", _q2row(1.0, 60) + ".alpha=1.0", OK, "表6"),
  ("K", "code/21_q2_opt.py:145-148", _q2row(1.0, 60) + ".K=60", OK, "表6"),
  ("搬移量", "code/21_q2_opt.py:121", _q2row(1.0, 60) + ".moved=23,959.15", OK, "表6"),
  ("卡车公里", "code/21_q2_opt.py:120", _q2row(1.0, 60) + ".truck_km=5,440.94", OK, "表6"),
  ("未服务出行", "code/21_q2_opt.py:119", _q2row(1.0, 60) + ".unmet=0", OK, "表6"))
R("sec5_q2.tex", 105,
  ("日终漂移", "code/21_q2_opt.py:123", _q2row(1.0, 60) + ".dev=900.45", OK, "表6 续行"),
  ("日成本", "code/21_q2_opt.py:116", _q2row(1.0, 60) + ".obj=83,325.24", OK, "表6 续行"))
R("sec5_q2.tex", 106,
  ("α", "code/21_q2_opt.py:145-148", _q2row(1.0, 150) + ".alpha=1.0", OK, "表6"),
  ("K", "code/21_q2_opt.py:145-148", _q2row(1.0, 150) + ".K=150", OK, "表6"),
  ("搬移量", "code/21_q2_opt.py:121", _q2row(1.0, 150) + ".moved=24,408.48", OK, "表6"),
  ("卡车公里", "code/21_q2_opt.py:120", _q2row(1.0, 150) + ".truck_km=4,990.06", OK, "表6"),
  ("未服务出行", "code/21_q2_opt.py:119", _q2row(1.0, 150) + ".unmet=0", OK, "表6"),
  ("日终漂移", "code/21_q2_opt.py:123", _q2row(1.0, 150) + ".dev=600.45", OK, "表6"),
  ("日成本", "code/21_q2_opt.py:116", _q2row(1.0, 150) + ".obj=133,945.94", OK, "表6"))
R("sec5_q2.tex", 107,
  ("α", "code/21_q2_opt.py:145-148", _q2row(0.9, 60) + ".alpha=0.9", OK, "表6"),
  ("K", "code/21_q2_opt.py:145-148", _q2row(0.9, 60) + ".K=60", OK, "表6"),
  ("搬移量", "code/21_q2_opt.py:121", _q2row(0.9, 60) + ".moved=25,403.41", OK, "表6"),
  ("卡车公里", "code/21_q2_opt.py:120", _q2row(0.9, 60) + ".truck_km=5,687.51", OK, "表6"),
  ("未服务出行", "code/21_q2_opt.py:119", _q2row(0.9, 60) + ".unmet=0", OK, "表6"),
  ("日终漂移", "code/21_q2_opt.py:123", _q2row(0.9, 60) + ".dev=600.45", OK, "表6"),
  ("日成本", "code/21_q2_opt.py:116", _q2row(0.9, 60) + ".obj=84,628.11", OK, "表6"))
R("sec5_q2.tex", 108,
  ("α", "code/21_q2_opt.py:145-148", _q2row(0.8, 80) + ".alpha=0.8", OK, "表6"),
  ("K", "code/21_q2_opt.py:145-148", _q2row(0.8, 80) + ".K=80", OK, "表6"),
  ("搬移量", "code/21_q2_opt.py:121", _q2row(0.8, 80) + ".moved=31,631.47", OK, "表6"),
  ("卡车公里", "code/21_q2_opt.py:120", _q2row(0.8, 80) + ".truck_km=7,330.26", OK, "表6"),
  ("未服务出行", "code/21_q2_opt.py:119", _q2row(0.8, 80) + ".unmet=816.42", OK, "表6"),
  ("日终漂移", "code/21_q2_opt.py:123", _q2row(0.8, 80) + ".dev=816.42", OK, "表6"),
  ("日成本", "code/21_q2_opt.py:116", _q2row(0.8, 80) + ".obj=115,145.08", OK, "表6"))
R("sec5_q2.tex", 109,
  ("α", "code/21_q2_opt.py:145-148", _q2row(0.7, 100) + ".alpha=0.7", OK, "表6"),
  ("K", "code/21_q2_opt.py:145-148", _q2row(0.7, 100) + ".K=100", OK, "表6"),
  ("搬移量", "code/21_q2_opt.py:121", _q2row(0.7, 100) + ".moved=34,271.22", OK, "表6"),
  ("卡车公里", "code/21_q2_opt.py:120", _q2row(0.7, 100) + ".truck_km=7,860.27", OK, "表6"),
  ("未服务出行", "code/21_q2_opt.py:119", _q2row(0.7, 100) + ".unmet=10,429.92", OK, "表6"),
  ("日终漂移", "code/21_q2_opt.py:123", _q2row(0.7, 100) + ".dev=10,429.92", OK, "表6"),
  ("日成本", "code/21_q2_opt.py:116", _q2row(0.7, 100) + ".obj=218,166.48", OK, "表6"))
R("sec5_q2.tex", 110,
  ("α", "code/21_q2_opt.py:145-148", _q2row(0.6, 80) + ".alpha=0.6", OK, "表6"),
  ("K", "code/21_q2_opt.py:145-148", _q2row(0.6, 80) + ".K=80", OK, "表6"),
  ("搬移量", "code/21_q2_opt.py:121", _q2row(0.6, 80) + ".moved=28,794.02", OK, "表6"),
  ("卡车公里", "code/21_q2_opt.py:120", _q2row(0.6, 80) + ".truck_km=6,591.58", OK, "表6"),
  ("未服务出行", "code/21_q2_opt.py:119", _q2row(0.6, 80) + ".unmet=34,492.53", OK, "表6"),
  ("日终漂移", "code/21_q2_opt.py:123", _q2row(0.6, 80) + ".dev=34,492.53", OK, "表6"),
  ("日成本", "code/21_q2_opt.py:116", _q2row(0.6, 80) + ".obj=412,379.29", OK, "表6"))

# ------------------------------------------------------- table 7: tab:supply
_SUP_TAB = [
    (86, 0.2, 0, 51380, 23.7, 10276, None),
    (87, 0.4, 1, 64938, 30.0, 25975, 1.16),
    (88, 0.6, 2, 78709, 36.4, 47226, 1.54),
    (89, 0.8, 3, 92036, 42.5, 73629, 1.98),
    (90, 1.0, 4, 104409, 48.3, 104409, 2.49),
    (91, 1.5, 5, 128873, 59.6, 193310, 3.63),
    (92, 2.0, 6, 143615, 66.4, 287229, 6.37),
    (93, 3.0, 8, 153654, 71.0, 460961, 17.31),
    (94, 3.5, 9, 154846, 71.6, 541961, 67.95),
]
for _ln, _r, _i, _q, _sh, _bill, _mc in _SUP_TAB:
    _e = [
        ("奖励 r（CNY）", "code/30_q3_incentive.py:112", f"rs[{_i}]={_r}", OK, "表7"),
        ("投放量 q(r)", "code/30_q3_incentive.py:163",
         f"q3_summary.supply[{_i}][1]={_q:,}", OK, "表7"),
        ("占缺口份额（%）", "code/30_q3_incentive.py:163",
         f"supply[{_i}][1]/need_total={_sh}%", OK, "表7"),
        ("激励账单（CNY）", "code/30_q3_incentive.py:163",
         f"q3_summary.supply[{_i}][2]={_bill:,}", OK, "表7"),
    ]
    if _mc is not None:
        _e.append(("边际成本（CNY/次）", "—",
                   f"q3_summary.supply 差分={_mc}（跳过 r=2.5）", OK,
                   "表7；paper-side arithmetic，无落盘字段"))
    R("sec6_q3.tex", _ln, *_e)

# ------------------------------------------------------- table 8: tab:fence
_FEN_TAB = [
    (46, 250, "0.25", 505, 39.0, 249860, 397, 3261),
    (47, 450, "0.45", 626, 75.3, 344463, 417, 4827),
    (48, 650, "0.65", 537, 83.0, 342606, 489, 6038),
]
for _ln, _R, _key, _f, _cov, _tot, _med, _mx in _FEN_TAB:
    R("sec7_q4.tex", _ln,
      ("步行半径 R（m）", "code/40_q4_efence.py:59", f"R={_R / 1000:.2f} km", OK, "表8"),
      ("围栏数", "code/40_q4_efence.py:121",
       f"q4_summary.by_radius.{_key}.fences={_f}", OK, "表8"),
      ("覆盖率（%）", "code/40_q4_efence.py:121",
       f"q4_summary.by_radius.{_key}.coverage={_cov}%", OK, "表8"),
      ("总容量（槽）", "code/40_q4_efence.py:125-131",
       f"q4_summary.by_radius.{_key}.total={_tot:,}", OK, "表8"),
      ("容量中位数（槽）", "code/40_q4_efence.py:125-131",
       f"q4_summary.by_radius.{_key}.median={_med}", OK, "表8"),
      ("容量最大值（槽）", "code/40_q4_efence.py:125-131",
       f"q4_summary.by_radius.{_key}.max={_mx:,}", OK, "表8"))
