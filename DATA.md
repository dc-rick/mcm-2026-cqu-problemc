# 数据说明

本仓库不包含原始订单数据与派生的大文件。`data/` 被 `.gitignore` 排除，原因有二：原始数据按赛题条款分发；`dist_matrix.npy` 为 106 MB，超过 GitHub 单文件 100 MB 的硬限制。

## 原始数据

- **来源**：2017 摩拜算法挑战赛（Mobike Cup / 2017 Mobike Algorithm Challenge）公开订单数据，北京，2017-05-10 至 2017-05-24。
- **获取**：按赛题官方发布渠道下载后，将订单文件放入 `data/raw/`。本仓库不再分发该文件。
- **规模**：原始记录 3,214,096 条（清洗后），484,754 辆不同单车，349,241 名用户。
- **注意**：根目录的 `paper_snapshot_2026-11-11.pdf` 是 2026-11-11 的论文快照（25 页），不是赛题；原题 `2026_CQU_MCM_Problem_C.pdf`（2 页）由用户另行保管，**原题不随仓库分发**。题面第 2 问要求 *based on the travel behavior model built above*，第 4 问要求 *using the travel behaviour model from Question (1) to predict parking demand*。

原始订单没有结束时间，因此骑行速度、绕行系数与时长都是**假设**（11.5 km/h、1.35、+1 分钟），不是观测；同车相邻订单的间隔只能作为时长的上界。相关敏感性见 T011。

## 清洗后表 `data/trips.parquet`

由 `code/00_prepare.py` 从原始 CSV 生成，主要字段：

| 字段 | 含义 |
|---|---|
| `orderid` | 订单号 |
| `userid` / `bikeid` | 用户 / 单车标识 |
| `biketype` | 车型 |
| `geohashed_start_loc` / `geohashed_end_loc` | 起终点 geohash |
| `start_lat` / `start_lng` / `end_lat` / `end_lng` | 起终点经纬度 |
| `starttime` | 订单开始时间；原始 CSV 只有开始时间，没有结束时间 |

## 派生文件

| 文件 | 生成方式 | 说明 |
|---|---|---|
| `cells.parquet` | 空间重建脚本 | 5,263 个 0.005° 格子的索引与中心坐标 |
| `dist_matrix.npy` | `code/build_dist_matrix.py` | 格子中心距离矩阵，106 MB；投影 km 坐标下的欧氏距离，对角线为 0.35/√2 km |
| `q1_params.json` | `code/10_q1_model.py` | Q1 五带 β 与对数似然 |
| `q2_state.npz` / `q2_solution.npz` | `code/20_q2_diag.py` / `code/21_q2_opt.py` | Q2 初值与 LP 解 |
| `user_features.parquet` / `user_seg.parquet` | `code/11_q1_extra.py` | 用户特征与 k=5 分段 |
| `runs/T00x/` | 各任务脚本 | 每个任务的中间结果与复算证据 |

说明：`dist_matrix.npy` 由 `code/build_dist_matrix.py` 重建，2026-11-18 已核对与仓库外原文件逐位相同（SHA-256 `4e3259f0…bceb97`）。`00_prepare.py` 与 `01`–`04` 已去除本机绝对路径（T015）。

## 重建顺序

```bash
python -I code/00_prepare.py --raw data/raw/<orders>.csv   # -> data/trips.parquet
python -I code/10_q1_model.py     # -> q1_params.json
python -I code/11_q1_extra.py     # -> user_features.parquet, user_seg.parquet
python -I code/20_q2_diag.py      # -> q2_state.npz
python -I code/21_q2_opt.py       # -> q2_solution.npz
# ……按编号继续至 90_figs.py、95_numbers.py
```

`code/lib.py` 以自身文件位置解析项目根目录，因此从任意当前目录运行都可以，但脚本之间仍有先后依赖。
