# -*- coding: utf-8 -*-
"""
Problem 4: preference-based recommendation from Problem-3 candidates,
then disruption replanning on the closed-edge network.

Flow
  result3 candidates → normalize → preference score → recommend
                     → delete closed edges → weighted Dijkstra → compare

Usage
  python problem4.py --limit 1 --datasets NY
  python problem4.py --datasets NY,BAY,COL
  python problem4.py --official   # 5-scheme competition layout
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from config import ANALYSIS_DIR, DATASETS, PROJECT_ROOT, RESULTS_DIR, queries_path
from graph import RoadGraph, load_dataset, load_queries
from problem4.disruption import disrupt_graph, load_closed_edges
from problem4.recommendation import build_candidate_set, recommend_scheme
from problem4.replanning import (
    candidate_span_scales,
    custom_replan_with_lambda,
    median_edge_scales,
    time_shortest,
    weighted_dijkstra,
)
from problem4.weight_scheme import (
    OFFICIAL_ORIGINAL,
    PREFERENCE_SCHEMES,
)

# Prefer jsy dense formal result3 when available
_JSY_RESULT3 = (
    PROJECT_ROOT
    / "analysis"
    / "problem3jsy"
    / "problem3"
    / "output_final"
    / "result3_研064.csv"
)
RESULT3_DEFAULT = _JSY_RESULT3 if _JSY_RESULT3.exists() else (RESULTS_DIR / "result3.csv")
RESULT4 = RESULTS_DIR / "result4.csv"
PARTS = RESULTS_DIR / "problem4_parts"
PERF = ANALYSIS_DIR / "problem4" / "performance.csv"

FIELDS = [
    "dataset",
    "query_id",
    "source",
    "target",
    "scheme",
    "network_state",
    "feasible",
    "c1",
    "c2",
    "c3",
    "c4",
    "c5",
    "path",
]


def qid_str(x) -> str:
    return f"{int(x):04d}"


def path_to_str(path: Optional[Sequence[int]]) -> str:
    if not path:
        return ""
    return "->".join(str(int(v)) for v in path)


def blank_row(ds, qid, source, target, scheme, network_state) -> dict:
    return {
        "dataset": ds,
        "query_id": qid,
        "source": int(source),
        "target": int(target),
        "scheme": scheme,
        "network_state": network_state,
        "feasible": False,
        "c1": "",
        "c2": "",
        "c3": "",
        "c4": "",
        "c5": "",
        "path": "",
    }


def filled_row(
    ds, qid, source, target, scheme, network_state, cost, path
) -> dict:
    return {
        "dataset": ds,
        "query_id": qid,
        "source": int(source),
        "target": int(target),
        "scheme": scheme,
        "network_state": network_state,
        "feasible": True,
        "c1": int(cost[0]),
        "c2": int(cost[1]),
        "c3": int(cost[2]),
        "c4": int(cost[3]),
        "c5": int(cost[4]),
        "path": path_to_str(path),
    }


def load_result3_groups(
    path: Path,
    datasets: Sequence[str],
    *,
    merge_objectives: bool = True,
) -> Dict[Tuple[str, str], Tuple[List[List[int]], List[List[int]], int, int]]:
    """
    Returns map (dataset, query_id) -> (costs, paths, source, target).
    By default merges m=2/3/5 candidates for the same OD.
    """
    usecols = [
        "dataset",
        "query_id",
        "source",
        "target",
        "objective_count",
        "c1",
        "c2",
        "c3",
        "c4",
        "c5",
        "path",
    ]
    df = pd.read_csv(path, dtype={"query_id": str, "dataset": str}, usecols=usecols)
    df["query_id"] = df["query_id"].map(qid_str)
    df = df[df["dataset"].isin(list(datasets))]
    if not merge_objectives:
        df = df[df["objective_count"].astype(int) == 5]

    groups: Dict[Tuple[str, str], Tuple[List, List, int, int]] = {}
    for (ds, qid), g in df.groupby(["dataset", "query_id"], sort=True):
        costs, paths = [], []
        source = int(g.iloc[0]["source"])
        target = int(g.iloc[0]["target"])
        for r in g.itertuples(index=False):
            costs.append(
                [int(r.c1), int(r.c2), int(r.c3), int(r.c4), int(r.c5)]
            )
            paths.append([int(x) for x in str(r.path).split("->") if x])
        groups[(ds, qid)] = (costs, paths, source, target)
    return groups


def solve_query(
    ds: str,
    qid: str,
    source: int,
    target: int,
    costs: List[List[int]],
    paths: List[List[int]],
    graph: RoadGraph,
    graph_d: RoadGraph,
    closed: set,
    *,
    official: bool,
    scale_mode: str,
    n_dirichlet: int,
) -> Tuple[List[dict], dict]:
    """Return result rows + timing stats for one OD."""
    cand = build_candidate_set(costs, paths)
    if scale_mode == "span" and len(cand) > 0:
        scales = candidate_span_scales(cand.c_min, cand.c_max)
    else:
        scales = median_edge_scales(graph)

    rows: List[dict] = []
    stats = {
        "dataset": ds,
        "query_id": qid,
        "n_candidates": len(cand),
        "t_recommend": 0.0,
        "t_replan": 0.0,
        "lambda_star": "",
    }

    # --- original: scheme-aware recommendation from Problem-3 candidates ---
    t0 = time.perf_counter()
    schemes = list(OFFICIAL_ORIGINAL) if official else list(PREFERENCE_SCHEMES.keys())
    rec_paths = {}
    for name in schemes:
        cost, path, _ = recommend_scheme(
            cand, name, closed=closed, n_dirichlet=n_dirichlet
        )
        rec_paths[name] = path
        if cost is None:
            rows.append(blank_row(ds, qid, source, target, name, "original"))
        else:
            rows.append(
                filled_row(ds, qid, source, target, name, "original", cost, path)
            )
    stats["t_recommend"] = time.perf_counter() - t0

    # --- disrupted: replanning on G' ---
    t1 = time.perf_counter()
    if official:
        p_time, c_time, _ = time_shortest(graph_d, source, target)
        if c_time is None:
            rows.append(
                blank_row(
                    ds, qid, source, target, "replanning_time_shortest", "disrupted"
                )
            )
        else:
            rows.append(
                filled_row(
                    ds,
                    qid,
                    source,
                    target,
                    "replanning_time_shortest",
                    "disrupted",
                    c_time,
                    p_time,
                )
            )

        # continuity-aware custom replan anchored at balanced recommendation
        p, c, _, lam = custom_replan_with_lambda(
            graph_d,
            source,
            target,
            scales,
            rec_paths.get("balanced"),
            None if c_time is None else float(c_time[1]),
        )
        stats["lambda_star"] = lam
        if c is None:
            rows.append(
                blank_row(ds, qid, source, target, "replanning_custom", "disrupted")
            )
        else:
            rows.append(
                filled_row(
                    ds, qid, source, target, "replanning_custom", "disrupted", c, p
                )
            )
    else:
        for name, w in PREFERENCE_SCHEMES.items():
            p, c, _ = weighted_dijkstra(graph_d, source, target, w, scales)
            if c is None:
                rows.append(blank_row(ds, qid, source, target, name, "disrupted"))
            else:
                rows.append(
                    filled_row(ds, qid, source, target, name, "disrupted", c, p)
                )
    stats["t_replan"] = time.perf_counter() - t1
    return rows, stats


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    tmp.replace(path)


def merge_parts(parts_dir: Path, out: Path) -> int:
    files = sorted(parts_dir.glob("*_*.csv"))
    all_rows: List[dict] = []
    for p in files:
        with open(p, encoding="utf-8") as f:
            all_rows.extend(csv.DictReader(f))
    # stable order: dataset, query, scheme order
    order_ds = {d: i for i, d in enumerate(DATASETS)}
    state_order = {"original": 0, "disrupted": 1}
    scheme_order = {
        "time_priority": 0,
        "distance_priority": 1,
        "stable_priority": 2,
        "balanced": 3,
        "replanning_time_shortest": 4,
        "replanning_custom": 5,
    }
    all_rows.sort(
        key=lambda r: (
            order_ds.get(r["dataset"], 99),
            r["query_id"],
            state_order.get(r["network_state"], 9),
            scheme_order.get(r["scheme"], 9),
        )
    )
    write_csv(out, all_rows)
    return len(all_rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Problem 4 recommendation + replanning")
    ap.add_argument("--result3", type=Path, default=RESULT3_DEFAULT)
    ap.add_argument("--output", type=Path, default=RESULT4)
    ap.add_argument("--datasets", type=str, default="NY,BAY,COL")
    ap.add_argument("--limit", type=int, default=0, help="max queries per city (0=all)")
    ap.add_argument(
        "--official",
        action="store_true",
        help="emit 5-scheme competition layout (time/stable/balanced + 2 replans)",
    )
    ap.add_argument(
        "--m5-only",
        action="store_true",
        help="use only objective_count=5 candidates (default merges m=2/3/5)",
    )
    ap.add_argument(
        "--scale-mode",
        choices=("median", "span"),
        default="span",
        help="edge-cost scales for disrupted weighted Dijkstra (default: candidate span)",
    )
    ap.add_argument(
        "--n-dirichlet",
        type=int,
        default=500,
        help="Dirichlet samples for robust balanced recommendation",
    )
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args(argv)

    PARTS.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "problem4").mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        n = merge_parts(PARTS, args.output)
        print(f"[merge] wrote {n} rows -> {args.output}")
        return 0

    if not args.result3.exists():
        raise SystemExit(f"missing result3: {args.result3}")

    datasets = [d.strip().upper() for d in args.datasets.split(",") if d.strip()]
    print(f"[p4] loading candidates from {args.result3}", flush=True)
    groups = load_result3_groups(
        args.result3, datasets, merge_objectives=not args.m5_only
    )
    print(f"[p4] OD groups: {len(groups)}  official={args.official}", flush=True)

    # align with queries_problem34 order
    query_lists = {}
    for ds in datasets:
        qdf = load_queries(queries_path(ds, "problem34"))
        qdf["query_id"] = qdf["query_id"].map(qid_str)
        if args.limit > 0:
            qdf = qdf.head(args.limit)
        query_lists[ds] = qdf

    perf_rows = []
    for ds in datasets:
        print(f"[p4] load graph {ds}", flush=True)
        g = load_dataset(ds)
        closed = load_closed_edges(ds)
        print(f"[p4] {ds}: closed directed edges = {len(closed)}", flush=True)
        gd = disrupt_graph(g, closed)
        print(
            f"[p4] {ds}: edges {g.n_edges} -> {gd.n_edges} "
            f"(removed {g.n_edges - gd.n_edges})",
            flush=True,
        )

        for r in query_lists[ds].itertuples(index=False):
            qid = qid_str(r.query_id)
            key = (ds, qid)
            frag = PARTS / f"{ds}_{qid}.csv"
            if frag.exists():
                print(f"  [skip] {ds} {qid} (checkpoint)", flush=True)
                continue
            if key not in groups:
                print(f"  [warn] no candidates for {ds} {qid}", flush=True)
                source, target = int(r.source), int(r.target)
                costs, paths = [], []
            else:
                costs, paths, source, target = groups[key]

            t_all = time.perf_counter()
            rows, stats = solve_query(
                ds,
                qid,
                source,
                target,
                costs,
                paths,
                g,
                gd,
                closed,
                official=args.official,
                scale_mode=args.scale_mode,
                n_dirichlet=args.n_dirichlet,
            )
            write_csv(frag, rows)
            stats["wall_sec"] = time.perf_counter() - t_all
            perf_rows.append(stats)
            print(
                f"  [{ds} {qid}] cand={stats['n_candidates']} "
                f"rec={stats['t_recommend']:.3f}s replan={stats['t_replan']:.3f}s "
                f"wall={stats['wall_sec']:.3f}s",
                flush=True,
            )

    if perf_rows:
        pd.DataFrame(perf_rows).to_csv(PERF, index=False)
        print(f"[p4] performance -> {PERF}", flush=True)

    n = merge_parts(PARTS, args.output)
    print(f"[p4] done: {n} rows -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
