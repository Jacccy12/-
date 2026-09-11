# -*- coding: utf-8 -*-
"""
Problem 1: five single-objective shortest paths on NY / BAY / COL.

Usage:
  python problem1.py                  # all cities, all 100 queries
  python problem1.py --datasets NY    # one city
  python problem1.py --limit 3        # smoke test first 3 queries
  python problem1.py --validate       # validate existing result CSV
  python problem1.py --validate --full-optimality
  python problem1.py --compare-baseline
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from config import ANALYSIS_DIR, DATASETS, OBJECTIVES, RESULTS_DIR, queries_path
from graph import load_dataset, load_queries
from shortest_path import path_to_string, shortest_path
from validator import validate_path


def solve_query(graph, query_id: str, source: int, target: int) -> List[dict]:
    rows = []
    for obj in OBJECTIVES:
        path0, costs = shortest_path(graph, source, target, obj)
        validate_path(graph, path0, source, target, costs)
        rows.append(
            {
                "dataset": graph.dataset,
                "query_id": query_id,
                "source": int(source),
                "target": int(target),
                "objective": obj,
                "c1": int(costs[0]),
                "c2": int(costs[1]),
                "c3": int(costs[2]),
                "c4": int(costs[3]),
                "c5": int(costs[4]),
                "path": path_to_string(path0),
            }
        )
    return rows


def run_problem1(
    datasets: List[str],
    limit: Optional[int],
    out_path: Path,
    *,
    timing_path: Optional[Path] = None,
) -> Path:
    fieldnames = [
        "dataset",
        "query_id",
        "source",
        "target",
        "objective",
        "c1",
        "c2",
        "c3",
        "c4",
        "c5",
        "path",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: avoid corrupting formal CSV if the process is killed mid-run.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    n_written = 0
    t0 = time.perf_counter()
    city_times: Dict[str, float] = {}

    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ds in datasets:
            graph = load_dataset(ds)
            queries = load_queries(queries_path(ds, "problem1"))
            if limit is not None:
                queries = queries.iloc[:limit]
            print(f"[problem1] {ds}: {len(queries)} queries x 5 objectives", flush=True)
            td0 = time.perf_counter()

            for idx, (_, row) in enumerate(queries.iterrows(), start=1):
                qid = str(row["query_id"])
                s, t = int(row["source"]), int(row["target"])
                tq = time.perf_counter()
                for r in solve_query(graph, qid, s, t):
                    writer.writerow(r)
                    n_written += 1
                f.flush()
                print(
                    f"  [{ds}] query {qid} ({idx}/{len(queries)}) "
                    f"done in {time.perf_counter() - tq:.2f}s  total_rows={n_written}",
                    flush=True,
                )

            city_times[ds] = time.perf_counter() - td0
            print(f"[problem1] {ds} finished in {city_times[ds]:.1f}s", flush=True)

    tmp_path.replace(out_path)
    total = time.perf_counter() - t0
    print(f"[problem1] wrote {n_written} rows -> {out_path}  ({total:.1f}s)", flush=True)

    if timing_path is not None:
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "after_total_sec": total,
            "cities": {ds: {"after_sec": city_times.get(ds)} for ds in datasets},
            "n_rows": n_written,
            "limit": limit,
        }
        timing_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[problem1] wrote timing -> {timing_path}")

    return out_path


def _merge_performance(
    before: Optional[dict],
    after: dict,
    out_path: Path,
) -> None:
    """Write performance_comparison.json; before fields may be null."""
    cities = {}
    for ds in DATASETS:
        b = (before or {}).get("cities", {}).get(ds, {})
        a = after.get("cities", {}).get(ds, {})
        b_sec = b.get("before_sec", b.get("after_sec"))
        a_sec = a.get("after_sec")
        entry = {"before_sec": b_sec if before else None, "after_sec": a_sec}
        if entry["before_sec"] and entry["after_sec"] and entry["before_sec"] > 0:
            entry["speedup"] = entry["before_sec"] / entry["after_sec"]
        else:
            entry["speedup"] = None
        cities[ds] = entry

    before_total = (before or {}).get("before_total_sec") or (before or {}).get("after_total_sec")
    after_total = after.get("after_total_sec")
    speedup = None
    if before and before_total and after_total and before_total > 0:
        speedup = before_total / after_total

    payload = {
        "before_total_sec": before_total if before else None,
        "after_total_sec": after_total,
        "speedup": speedup,
        "user_reported_before_sec": 390.0,  # ~6.5 min informal; not used as official before
        "NY": cities.get("NY"),
        "BAY": cities.get("BAY"),
        "COL": cities.get("COL"),
        "note": (
            "before_* is null unless measured with --bench-before / legacy timing file; "
            "do not treat user_reported_before_sec as validated."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[problem1] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Solve problem 1")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--limit", type=int, default=None, help="Only first N queries per city")
    parser.add_argument("--out", type=str, default=str(RESULTS_DIR / "result1.csv"))
    parser.add_argument("--validate", action="store_true")
    parser.add_argument(
        "--full-optimality",
        action="store_true",
        help="With --validate: independent uni Dijkstra/BFS optimality for all 1500 rows",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare optimal objective values vs results/baseline_result1.csv",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=str(RESULTS_DIR / "baseline_result1.csv"),
    )
    parser.add_argument(
        "--bench-before",
        action="store_true",
        help="Time legacy shortest_path on same queries (writes before timing JSON)",
    )
    args = parser.parse_args()

    analysis_dir = ANALYSIS_DIR / "problem1"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    if args.rebuild_cache:
        for ds in args.datasets:
            load_dataset(ds, rebuild_cache=True)

    if args.validate:
        from validator import validate_result1_csv

        graphs = {ds: load_dataset(ds) for ds in DATASETS}
        report = validate_result1_csv(
            args.out,
            graphs,
            full_optimality=args.full_optimality,
            report_path=str(analysis_dir / "validation_report.json"),
        )
        if not report.passed:
            raise SystemExit(1)
        return

    if args.compare_baseline:
        from validator import compare_optimal_objectives

        summary = compare_optimal_objectives(args.baseline, args.out)
        out = analysis_dir / "baseline_regression.json"
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        if not summary["passed"]:
            raise SystemExit("Baseline optimal-objective regression FAILED")
        print("[compare] all optimal objective values identical")
        return

    if args.bench_before:
        # Time legacy implementation without writing formal results.
        import shortest_path_legacy as spl

        t0 = time.perf_counter()
        city_times = {}
        for ds in args.datasets:
            graph = load_dataset(ds)
            queries = load_queries(queries_path(ds, "problem1"))
            if args.limit is not None:
                queries = queries.iloc[: args.limit]
            td0 = time.perf_counter()
            for _, row in queries.iterrows():
                s, t = int(row["source"]), int(row["target"])
                for obj in OBJECTIVES:
                    spl.shortest_path(graph, s, t, obj)
            city_times[ds] = time.perf_counter() - td0
            print(f"[bench-before] {ds}: {city_times[ds]:.1f}s")
        total = time.perf_counter() - t0
        before = {
            "before_total_sec": total,
            "cities": {ds: {"before_sec": city_times[ds]} for ds in args.datasets},
        }
        path = analysis_dir / "timing_before.json"
        path.write_text(json.dumps(before, indent=2), encoding="utf-8")
        print(f"[bench-before] total {total:.1f}s -> {path}")
        return

    timing_path = analysis_dir / "timing_after.json"
    run_problem1(args.datasets, args.limit, Path(args.out), timing_path=timing_path)

    after = json.loads(timing_path.read_text(encoding="utf-8"))
    before_path = analysis_dir / "timing_before.json"
    before = None
    if before_path.exists():
        before = json.loads(before_path.read_text(encoding="utf-8"))
    _merge_performance(before, after, analysis_dir / "performance_comparison.json")


if __name__ == "__main__":
    main()
