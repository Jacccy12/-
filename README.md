# 城市道路网络多目标通行路径规划

研究生数模集训题：在 DIMACS5 风格城市路网（**NY / BAY / COL**）上，完成五目标单源最短路、三目标精确 Pareto 前沿、高维近似前沿，以及偏好推荐与道路中断重规划。

| 问题 | 任务 | 主程序 | 典型输出 |
|------|------|--------|----------|
| 一 | 五目标各自最短路 | `problem1.py` | `results/result1.csv` |
| 二 | 三目标精确 Pareto（\(c_1,c_2,c_3\)） | `problem2.py` / `run_problem2_*.py` | `results/result2.csv` |
| 三 | \(m=2,3,5\) 维 Pareto（精确 / ε 近似） | `problem3.py` / `run_problem3_production.py` | `results/result3.csv` |
| 四 | 多偏好推荐 + 封路重规划 | `problem4.py` | `results/result4.csv` |

路网边权为五维：`distance`、`travel_time`、`elevation`、`avg_degree`、`hop_count`。

---

## 环境要求

- Python 3.10+（推荐）
- Windows / Linux；问题二、三的高性能路径依赖 **g++** 编译的本地 DLL/SO（可选，缺失时部分模块会回退或报错）
- 依赖：

```bash
pip install -r requirements.txt
```

主要依赖：`numpy`、`pandas`、`matplotlib`、`numba`。

---

## 数据准备

1. 将题目数据包中的边表放到可被程序发现的位置，例如：

```text
data/
├─ edges/
│  ├─ edges_NY_5obj.txt
│  ├─ edges_BAY_5obj.txt
│  └─ edges_COL_5obj.txt
└─ dimacs5_{ny,bay,col}/
   ├─ queries_problem1.csv
   ├─ queries_problem2.csv
   ├─ queries_problem34.csv
   └─ closed_edges_problem4.csv   # 仅问题四
```

2. 或在项目根目录写 `DATA_ROOT.txt`，内容为数据根目录的绝对路径（该目录下需有 `edges/edges_NY_5obj.txt`）。

3. 首次读边表会生成 `cache/{NY,BAY,COL}_graph.npz`，之后直接读缓存。

> 边表与完整 `result2.csv` 体积很大，**不建议**整库推上 GitHub。可用 [Git LFS](https://git-lfs.com/) 或自行说明数据获取方式。

---

## 快速开始

```bash
# 克隆后进入工程
cd mop_routing
pip install -r requirements.txt

# 问题一冒烟（每城前 2 组 OD）
python problem1.py --limit 2

# 问题二单查询冒烟
python problem2.py --datasets NY --limit 1

# 问题三单查询冒烟
python problem3.py --datasets NY --limit 1 --objectives 2,5

# 问题四单查询冒烟（默认可读问题三候选）
python problem4.py --datasets NY --limit 1
```

---

## 问题说明与运行

### 问题一：五目标单目标最短路

对每城约 100 组 OD，分别最小化五个目标，输出路径与五维累计成本。

```bash
python problem1.py --out results/result1.csv
python problem1.py --validate --out results/result1.csv
python analyze_problem1.py --result results/result1.csv
```

### 问题二：三目标精确 Pareto

目标：距离、时间、地形起伏。主算法为 **T-MDA**（C++：`t_mda_core.cpp`，封装：`t_mda.py`），配合静态精确走廊剪枝（`corridor.py`）与种子集（`seed_bank.py`）。仅当队列清空（\(Q=\emptyset\)）时写入 **EXACT** 分片。

```bash
# 单进程 / 小规模
python problem2.py --datasets NY --limit 1

# 批量 + 断点续算
python run_problem2_batch.py

# 分层并行调度（生产）
python run_problem2_scheduler.py

# 校验（全量建议用快速版）
python validate_problem2.py
python analysis/problem2/validate_result2_fast.py

# 论文插图
python analysis/problem2/gen_paper_figures.py
```

正式合并结果示例规模：约 **280 万** 行非支配向量（以本机跑通为准）。

### 问题三：高维 Pareto（\(m=2/3/5\)）

- \(m=2,3\)：可走精确 T-MDA 或加权采样 + 非支配过滤  
- \(m=5\)：加权候选 + ε-网格近似  

```bash
python problem3.py --datasets NY,BAY,COL --objectives 2,3,5
python run_problem3_production.py --approx-only
python analyze_problem3.py
```

### 问题四：偏好推荐与扰动重规划

从问题三候选集做极差标准化与多偏好推荐；删除 `closed_edges_problem4.csv` 中的有向边后，用加权 Dijkstra 重规划并对比。

```bash
# 四偏好 × original/disrupted
python problem4.py --datasets NY,BAY,COL

# 竞赛常用 5 方案格式
python problem4.py --official --output results/result4_official.csv

python analyze_problem4.py
python validate_problem4.py results/result4.csv --expect-per-query 8
```

偏好示例：`time_priority`、`distance_priority`、`stable_priority`、`balanced`。

---

## 项目结构

```text
mop_routing/
├─ config.py / graph.py / shortest_path.py   # 配置、CSR 路网、单目标最短路
├─ problem1.py … problem4.py                 # 四问入口
├─ corridor.py / seed_bank.py                # 走廊剪枝、Pareto 种子
├─ multiobjective_exact.py                   # 精确多目标工具与校验
├─ t_mda.py / t_mda_core.cpp                 # T-MDA 高性能核心
├─ run_problem2_batch.py / run_problem2_scheduler.py
├─ run_problem3_production.py
├─ solver/                                   # 问题三：ε-grid、加权前沿、质量指标
├─ problem4/                                 # 推荐、删边、重规划、评估
├─ data/                                     # 查询与封闭边（边表见 DATA_ROOT）
├─ cache/                                    # 图缓存
├─ results/                                  # result*.csv 与分片
├─ analysis/                                 # 统计分析、论文插图与笔记
├─ tests/                                    # 单元测试
└─ requirements.txt
```

---

## 结果与校验

| 文件 | 含义 |
|------|------|
| `results/result1.csv` | 五目标最短路 |
| `results/result2.csv` | 三目标精确 Pareto |
| `results/result3.csv` | \(m=2,3,5\) 候选前沿 |
| `results/result4.csv` | 推荐与重规划方案 |

校验脚本：`validate_problem2.py`、`validate_problem4.py`、`analysis/problem2/validate_result2_fast.py` 等。分片目录支持断点续跑（如 `results/problem2_parts/`、`results/problem3_parts/`、`results/problem4_parts/`）。

---

## 方法概要

1. **问题一**：复用 CSR + 双向 Dijkstra / BFS，五目标分别求解。  
2. **问题二**：可采纳反向启发式 + 静态精确走廊 + T-MDA；永久前沿懒索引加速支配检验，差分门控保证与优化前集合一致。  
3. **问题三**：低维可精确，五维用加权采样与 ε-支配近似，并统计 HV / 覆盖等质量指标。  
4. **问题四**：不在扰动后重求全 Pareto；在候选集上按偏好评分推荐，删边后对标量边权做 Dijkstra 重规划。

---

## 注意事项

- 问题二全量 EXACT 耗时长、内存高，请用调度器并监控 RSS。  
- C++ 扩展需本机可调用 `g++`；Windows 上请保证 DLL 与 Python 位数一致。  
- 推送 GitHub 前请排除或 LFS 管理：`cache/`、`results/*.csv`、大型 `*.dll`、原始 `edges_*.txt`。  
- 题面 PDF、官方数据包版权归题目方，请勿违规传播。

---

## 许可

本仓库为课程 / 集训解题代码，仅供学习交流。题目数据与 PDF 的使用遵循原发放方规定。
