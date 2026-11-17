"""T001-R2 step 2: closed-fleet central value and interval.

Method A (normalised reference): the 9-day window is close to uniform coverage,
so its unseen share 1 - D9/N9_hat = 1 - 462,893/483,787 = 4.319% is applied to
the 14-day observed union: F_A = 484,754 / (1 - 0.04319) ~ 506.6k.  This
OVERSTATES the 14-day unseen share (14 days give each bicycle more chances to
appear), so it is an upper reference.

Method B (independent): two-window capture-recapture, window 1 = 10-19 May,
window 2 = 20-24 May.  Chapman N = (n1+1)(n2+1)/(m+1) - 1, with the standard
sampling variance.  Heterogeneity (heavy users more likely captured) biases
Chapman down; the truncated second window (54.5% volume) lowers n2 and m, which
biases it up.  The two act in opposite directions.

Method C (sensitivity only): constant per-day capture extrapolated from 9 to 14
days, (1-q)^14 with (1-q)^9 = 4.319%; homogeneous assumption makes it a lower
reference.
"""
import io
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = os.path.join(ROOT, "data", "runs", "T001R2")
os.makedirs(RUN, exist_ok=True)

# observed unions from code/audit/t001r2_truncation.py
D9 = 462893
D14 = 484754
n1, n2, m = 462893, 331822, 309961
# T001 9-day estimates (delivery/T001_delivery.json)
N9_bb = 483787.2060108598
F_old = 343637.9864744583
# two-window union check
assert n1 + n2 - m == D14

# Method A
unseen9 = 1.0 - D9 / N9_bb
F_A = D14 / (1.0 - unseen9)

# Method B: Chapman + sampling variance
F_B = (n1 + 1) * (n2 + 1) / (m + 1) - 1
var_B = ((n1 + 1) * (n2 + 1) * (n1 - m) * (n2 - m)
         / ((m + 1) ** 2 * (m + 2)))
se_B = var_B ** 0.5
lo_sampling, hi_sampling = F_B - 1.96 * se_B, F_B + 1.96 * se_B

# Method C: homogeneous per-day capture extrapolated to 14 days
# P(never in 9 days) = (1-q)^9 = unseen9, so 1-q = unseen9^(1/9)
notseen_day = unseen9 ** (1.0 / 9.0)
unseen14_hom = notseen_day ** 14
F_C = D14 / (1.0 - unseen14_hom)

# Method A multiplicative form (same as F_A, shown for audit)
F_A2 = N9_bb * D14 / D9

# central and model-uncertainty interval
F_center = F_B
lo_model = D14
hi_model = F_A

# conversions
rides_day = 274357.0 / 0.128           # 2,143,414 (paper's scaled daily rides)
P_star = 274910.0                       # Q2 physical fleet at old alpha=0.8
out = dict(
    inputs=dict(D9=D9, D14=D14, n1=n1, n2=n2, overlap=m,
                N9_beta_binomial=N9_bb, unseen9=unseen9,
                F_old=F_old),
    methods={
        "A_9day_unseen_on_14day_union": dict(
            F=F_A, multiplicative_check=F_A2,
            unseen9=unseen9, formula=("unseen9 = 1 - 462,893/483,787 = 4.319%; "
                                      "F_A = 484,754/(1 - unseen9) = 506,635"),
            note="upper reference; overstates the 14-day unseen share"),
        "B_chapman_9day_vs_5day": dict(
            F=F_B, se=se_B, ci95_sampling=[lo_sampling, hi_sampling],
            note="central candidate; heterogeneity biases down, truncated "
                 "second window biases up"),
        "C_homogeneous_day_extrapolation": dict(
            F=F_C, notseen_per_day=notseen_day, unseen14=unseen14_hom,
            formula=("(1-q)^9 = 4.319% -> 1-q = 0.7053; (1-q)^14 = 0.754%; "
                     "F_C = 484,754/(1 - 0.754%) = 488,436"),
            note="lower reference; ignores heterogeneity"),
    },
    central=dict(F=F_center, method="B_chapman_9day_vs_5day",
                 model_interval=[lo_model, hi_model],
                 model_interval_note=("lower = observed 14-day union (hard "
                                      "bound); upper = method A (normalised "
                                      "reference); not a sampling CI")),
    excluded_methods=[
        dict(name="beta-binomial 9-day", F=483787.206, why="9-day window; below 14-day union"),
        dict(name="zero-truncated NB 9-day", F=478337.265, why="9-day window; below 14-day union"),
        dict(name="homogeneous 9-day", F=465212.226, why="9-day window; below 14-day union"),
    ],
    exclusion_statement=("9 天点估计低于 14 天并集下界，故不作为中心值；"
                         "被排除方法：β-二项 483,787、零截断 NB 478,337、"
                         "同质 9 天 465,212。"),
    conversions={
        "rides_per_day": rides_day,
        "rides_per_bike_per_day": {
            "central": rides_day / F_center,
            "lower": rides_day / lo_model,
            "upper": rides_day / hi_model,
            "old": rides_day / F_old,
            "formula": "2,143,414 / F",
        },
        "alpha_star": {
            "central": P_star / F_center,
            "lower": P_star / lo_model,
            "upper": P_star / hi_model,
            "old_alpha": 0.8,
            "formula": "274,910 / F",
        },
        "I0_multiplier_vs_343638": {
            "central": F_center / F_old,
            "lower": lo_model / F_old,
            "upper": hi_model / F_old,
            "formula": "F / 343,638",
        },
    },
    affected_paper=[
        dict(coord="paper/summary.tex:23-24",
             old="3.4e5 bicycles serving 6.2 rides each per day",
             new=("4.96e5 bicycles (interval 4.85e5-5.07e5) serving "
                  "4.3 rides each per day (4.23-4.42)")),
        dict(coord="paper/sec3_data.tex:85",
             old="F = T/mu = 343,638",
             new="F = 495,540 (interval 484,754-506,640) under closed fleet"),
        dict(coord="paper/sec3_data.tex:139-140 (table 3)",
             old="Fleet estimate 343,638; rides per bicycle per day 6.2",
             new="Fleet estimate 495,540 (484,754-506,640); rides per bicycle per day 4.3"),
        dict(coord="paper/sec3_data.tex:96-97",
             old="At 343,638 bicycles and 2.14e6 rides per day, 6.2 rides per bicycle per day",
             new="At 495,540 bicycles and 2.14e6 rides per day, 4.3 rides per bicycle per day"),
        dict(coord="paper/sec5_q2.tex:14-15",
             old="I_z0 is the zone's share of the fleet",
             new=("fleet level 495,540; every I_z0 scales by 495,540/343,638 "
                  "= 1.442 versus the current numbers")),
        dict(coord="paper/sec5_q2.tex:17-19",
             old="smallest zone 1,986, median 6,912 bicycles at midnight",
             new="scale by 1.442: 2,864 and 9,968 (excursion -349 unchanged)"),
        dict(coord="paper/sec5_q2.tex:143-149",
             old="alpha in {0.4,...,1.0}; alpha=0.8 = 275,000 bicycles",
             new=("same physical grid is alpha* in {0.277,...,0.693}; the "
                  "275,000-bicycle optimum is alpha*=0.555 (cost 390k unchanged); "
                  "the old alpha=1 (343,638) is alpha*=0.693")),
        dict(coord="paper/sec5_q2.tex:159-163 (fig4_q2 panel d)",
             old="total daily cost against fleet size, alpha grid 0.4-1.0",
             new=("multiply the fleet axis by 1.442 or relabel alpha as "
                  "alpha*=alpha_old*343,638/495,540; optimum stays at 274,910 "
                  "bicycles")),
        dict(coord="paper/sec9_sensitivity.tex:17-22",
             old="table of daily fleet estimates, adopted value 343,638",
             new=("relabel as same-day occupancy estimates; keep the daily table, "
                  "but it no longer defines the fleet")),
        dict(coord="paper/sec9_sensitivity.tex:31",
             old="We adopt the weekday value 343,638",
             new=("closed-fleet centre 495,540 (484,754-506,640); 343,638 is "
                  "demoted to the 5/16 same-day occupancy estimate")),
        dict(coord="paper/sec9_sensitivity.tex:103",
             old="5,263 cells are served by 343,638 bicycles",
             new="5,263 cells are served by 495,540 bicycles"),
        dict(coord="paper/sec5_q2.tex:19-21",
             old="172 of 5,263 cells run dry",
             new="affected; exact count requires the Q2 rerun in phase B (higher fleet, fewer dry cells)"),
    ],
    notes=[
        "不重跑 Q2-Q5；Q2 的成本/卡车数只做 alpha 标签换算，物理车队点 274,910 不变。",
        "区间下端是 14 天观测并集（硬下界），上端是方法 A（高估未见份额），"
        "不是抽样置信区间；Chapman 抽样 95% CI 仅 ±260 辆，模型不确定性主导。",
        "方法 B 的第二窗口 5/20-5/24 是截断样本（单量 54.5%、单车数 63.1%、"
        "13 个零订单小时），这会降低 n2 与重叠 m、抬高 Chapman；异质性则压低 "
        "Chapman，两向部分抵消。",
        "未测分支：按日分批投放的开放模型；若车队可跨城市调动，14 天并集不再是下界。",
    ],
)
json.dump(out, io.open(os.path.join(RUN, "fleet_closed.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
