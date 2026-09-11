# 城市道路网络多目标通行路径规划 — 求解项目

从原始压缩包（文件名可能因 ZIP 编码显示为乱码）整理出的可运行工程。
题面 PDF、三城市合并边表、查询与示例结果均已对齐。

## 目录

```text
mop_routing/
├─ 城市道路网络多目标通行路径规划_研究生版.pdf
├─ config.py                 # 数据路径自动发现
├─ graph.py                  # CSR 五维有向路网 + .npz 缓存
├─ shortest_path.py          # 双向 Dijkstra / BFS
├─ validator.py              # 路径与 result CSV 校验
├─ analyze_problem1.py       # 冲突矩阵 / Jaccard / 热力图
├─ problem1.py               # 问题一主程序
├─ problem2.py ~ problem4.py # 后续问题占位
├─ data/dimacs5_{ny,bay,col}/ # 查询与封闭边（小文件已拷贝）
├─ examples/                 # 官方结果示例
├─ cache/                    # NY/BAY/COL_graph.npz
├─ results/                  # result1.csv 等
└─ analysis/                 # 问题一统计分析
```

边表体积较大，程序通过 `DATA_ROOT.txt` / 自动搜索定位原始 `data/edges/`，不重复拷贝。

## 环境

```bash
pip install -r requirements.txt
```

## 问题一

对 NY / BAY / COL 各 100 组 OD，分别按 5 个目标求最优路径，输出严格格式 CSV。

```bash
# 冒烟测试（每城前 2 组）
python problem1.py --limit 2

# 正式全量（1500 行）
python problem1.py --out results/result1.csv

# 校验
python problem1.py --validate --out results/result1.csv

# 冲突与路径重合分析
python analyze_problem1.py --result results/result1.csv
```

首次运行会解析 `edges_*_5obj.txt` 并写入 `cache/*_graph.npz`，之后直接读缓存。

## 输出格式（问题一）

```text
dataset,query_id,source,target,objective,c1,c2,c3,c4,c5,path
```

`objective ∈ {distance, travel_time, elevation, avg_degree, hop_count}`  
`query_id` 保持 `0001` 字符串格式；`path` 为完整 `u->v->...` 节点序列。
