# -*- coding: utf-8 -*-
"""
Problem 2: exact 3-objective Pareto fronts (distance, travel_time, elevation).

Usage:
  python problem2.py --datasets NY --limit 1
  python problem2.py --datasets NY --compare-mode
  python problem2.py
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from config import ANALYSIS_DIR, DATASETS, RESULTS_DIR, queries_path
from corridor import build_corridor, lexmin_all_orders
from graph import load_dataset, load_queries
from multiobjective_exact import (
    load_or_compute_heuristics,
    namoa_star_exact,
    update_skyline,
    verify_anchor_mins,
    verify_internal_nondominated,
)
from seed_bank import build_seed_bank, suffix_cost3_from_trees


PARTS_DIR = RESULTS_DIR / "problem2_parts"


def write_query_fragment(ds: str, qid: str, source: int, target: int, solutions, parts_dir: Path):
    parts_dir.mkdir(parents=True, exist_ok=True)
    path = parts_dir / f"{ds}_{qid}.csv"
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "query_id",
                "source",
                "target",
                "solution_id",
                "c1",
                "c2",
                "c3",
            ],
        )
        w.writeheader()
        for sid, (c1, c2, c3) in enumerate(solutions, start=1):
            w.writerow(
                {
                    "dataset": ds,
                    "query_id": qid,
                    "source": source,
                    "target": target,
                    "solution_id": sid,
                    "c1": c1,
                    "c2": c2,
                    "c3": c3,
                }
            )
    tmp.replace(path)
    return path


def merge_parts(parts_dir: Path, out_path: Path) -> int:
    files = sorted(parts_dir.glob("*_*.csv"))
    n = 0
    with open(out_path, "w", newline="", encoding="utf-8") as fout:
        writer = None
        for fp in files:
            with open(fp, newline="", encoding="utf-8") as fin:
                reader = csv.DictReader(fin)
                if writer is None:
                    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)
                    n += 1
    return n


def enrich_seeds(graph, source: int, target: int, base_seeds):
    seeds = list(base_seeds)
    for y in lexmin_all_orders(graph, source - 1, target - 1):
        update_skyline(seeds, y)
    return seeds


def run_problem2(
    datasets: List[str],
    limit: int | None,
    out_path: Path,
    stats_path: Path,
    use_cache: bool = True,
    lex_order=(0, 1, 2),
    max_labels: int = 30_000_000,
) -> Path:
    stat_fields = [
        "dataset",
        "query_id",
        "source",
        "target",
        "pareto_size",
        "runtime_sec",
        "h_runtime_sec",
        "search_runtime_sec",
        "labels_generated",
        "labels_expanded",
        "labels_peak",
        "node_dominance_pruned",
        "goal_bound_pruned",
        "duplicate_pruned",
        "max_labels_at_one_node",
        "nodes_pruned_corridor",
        "edges_pruned_corridor",
        "n_seeds",
        "exact",
        "mode",
    ]
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with open(stats_path, "w", newline="", encoding="utf-8") as fstat:
        sw = csv.DictWriter(fstat, fieldnames=stat_fields)
        sw.writeheader()

        for ds in datasets:
            graph = load_dataset(ds)
            queries = load_queries(queries_path(ds, "problem2"))
            if limit is not None:
                queries = queries.iloc[:limit]

            by_target: Dict[int, list] = defaultdict(list)
            for _, row in queries.iterrows():
                by_target[int(row["target"])].append(row)

            print(
                f"[problem2] {ds}: {len(queries)} queries, "
                f"{len(by_target)} distinct targets",
                flush=True,
            )

            for target, rows in by_target.items():
                th = time.time()
                hb = load_or_compute_heuristics(graph, ds, target, use_cache=use_cache)
                h_time = time.time() - th
                sources = [int(r["source"]) for r in rows]
                bank = build_seed_bank(graph, sources, target, hb, k=10)
                print(
                    f"  [{ds}] heuristics+weighted-seeds target={target} "
                    f"({len(rows)} sources) in {time.time()-th:.2f}s",
                    flush=True,
                )

                for row in rows:
                    qid = str(row["query_id"])
                    source = int(row["source"])
                    frag = PARTS_DIR / f"{ds}_{qid}.csv"
                    if frag.exists():
                        print(f"    skip existing {frag.name}", flush=True)
                        continue

                    tq = time.time()
                    seeds = enrich_seeds(graph, source, target, bank.get(source, []))
                    corr = build_corridor(graph, source - 1, hb, seeds)
                    # Mainline: static corridor ON; suffix UB OFF; dyn corridor OFF.
                    print(
                        f"    [{ds}] {qid}: seeds={len(seeds)} "
                        f"corridor nodes_pruned={corr.n_nodes_pruned} "
                        f"edges_pruned={corr.n_edges_pruned} "
                        f"kept_edges={corr.n_edges_kept} "
                        f"(suffix=OFF dyn=OFF cached-blockmeta)",
                        flush=True,
                    )

                    res = namoa_star_exact(
                        graph,
                        source,
                        target,
                        heuristics=hb,
                        use_heuristic=True,
                        lex_order=lex_order,
                        h_runtime_sec=h_time / max(len(rows), 1),
                        extra_seeds=seeds,
                        edge_allowed=corr.edge_allowed,
                        max_labels=max_labels,
                        suffix_q=None,
                        d_s=None,
                    )

                    s0 = source - 1
                    h_at_s = (
                        int(hb.h[s0, 0]),
                        int(hb.h[s0, 1]),
                        int(hb.h[s0, 2]),
                    )
                    st = res.stats
                    is_exact = bool(st.exact_finished)
                    if is_exact:
                        verify_internal_nondominated(res.solutions)
                        verify_anchor_mins(res.solutions, h_at_s)
                        write_query_fragment(
                            ds, qid, source, target, res.solutions, PARTS_DIR
                        )

                    sw.writerow(
                        {
                            "dataset": ds,
                            "query_id": qid,
                            "source": source,
                            "target": target,
                            "pareto_size": st.pareto_size,
                            "runtime_sec": f"{time.time()-tq:.4f}",
                            "h_runtime_sec": f"{st.h_runtime_sec:.4f}",
                            "search_runtime_sec": f"{st.search_runtime_sec:.4f}",
                            "labels_generated": st.labels_generated,
                            "labels_expanded": st.labels_expanded,
                            "labels_peak": st.labels_peak,
                            "node_dominance_pruned": st.node_dominance_pruned,
                            "goal_bound_pruned": st.goal_bound_pruned,
                            "duplicate_pruned": st.duplicate_pruned,
                            "max_labels_at_one_node": st.max_labels_at_one_node,
                            "nodes_pruned_corridor": corr.n_nodes_pruned,
                            "edges_pruned_corridor": corr.n_edges_pruned,
                            "n_seeds": len(seeds),
                            "exact": is_exact,
                            "mode": st.mode,
                        }
                    )
                    fstat.flush()
                    status = (
                        "EXACT"
                        if is_exact
                        else ("EARLY_STOP" if st.early_stop else "OVERFLOW/INCOMPLETE")
                    )
                    print(
                        f"    query {qid}: |S|={st.pareto_size} "
                        f"gen={st.labels_generated} exp={st.labels_expanded} "
                        f"goal_pr={st.goal_bound_pruned} "
                        f"search={st.search_runtime_sec:.1f}s "
                        f"wall={time.time()-tq:.1f}s {status}",
                        flush=True,
                    )

    n = merge_parts(PARTS_DIR, out_path)
    print(f"[problem2] merged {n} rows -> {out_path} ({time.time()-t0:.1f}s)", flush=True)
    return out_path


def compare_one(dataset: str, query_idx: int = 0) -> None:
    graph = load_dataset(dataset)
    queries = load_queries(queries_path(dataset, "problem2"))
    row = queries.iloc[query_idx]
    qid, s, t = str(row["query_id"]), int(row["source"]), int(row["target"])
    print(f"[compare] {dataset} {qid} {s}->{t}", flush=True)
    hb = load_or_compute_heuristics(graph, dataset, t, use_cache=True)
    bank = build_seed_bank(graph, [s], t, hb, k=10)
    seeds = enrich_seeds(graph, s, t, bank[s])
    corr = build_corridor(graph, s - 1, hb, seeds)
    suffix_q = suffix_cost3_from_trees(graph, hb)
    print(
        f"  seeds={len(seeds)} edges_pruned={corr.n_edges_pruned}/{graph.n_edges}",
        flush=True,
    )

    r1 = namoa_star_exact(
        graph, s, t, heuristics=hb, use_heuristic=True,
        extra_seeds=seeds, edge_allowed=corr.edge_allowed,
        suffix_q=suffix_q, d_s=corr.d_s,
    )
    print(
        f"  NAMOA*  |S|={r1.stats.pareto_size} gen={r1.stats.labels_generated} "
        f"search={r1.stats.search_runtime_sec:.2f}s",
        flush=True,
    )
    r0 = namoa_star_exact(
        graph, s, t, heuristics=hb, use_heuristic=False,
        extra_seeds=seeds, edge_allowed=corr.edge_allowed,
        suffix_q=suffix_q, d_s=corr.d_s,
    )
    print(
        f"  h=0     |S|={r0.stats.pareto_size} gen={r0.stats.labels_generated} "
        f"search={r0.stats.search_runtime_sec:.2f}s",
        flush=True,
    )
    if r1.solutions != r0.solutions:
        raise SystemExit("MISMATCH: NAMOA* vs multidijkstra")
    print("[compare] OK", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Solve problem 2")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=str, default=str(RESULTS_DIR / "result2.csv"))
    parser.add_argument(
        "--stats",
        type=str,
        default=str(ANALYSIS_DIR / "problem2" / "query_stats.csv"),
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--compare-mode", action="store_true")
    parser.add_argument("--compare-query", type=int, default=0)
    parser.add_argument("--max-labels", type=int, default=30_000_000)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()

    if args.merge_only:
        n = merge_parts(PARTS_DIR, Path(args.out))
        print(f"merged {n} rows -> {args.out}")
        return

    if args.compare_mode:
        compare_one(args.datasets[0], args.compare_query)
        return

    run_problem2(
        args.datasets,
        args.limit,
        Path(args.out),
        Path(args.stats),
        use_cache=not args.no_cache,
        max_labels=args.max_labels,
    )


if __name__ == "__main__":
    main()
