# -*- coding: utf-8 -*-
"""
Problem-3 analysis: dimension effect + ε sensitivity + HV / ε-coverage.

Examples
  python analyze_problem3.py --dim-effect --datasets NY --limit 1
  python analyze_problem3.py --eps-sweep --datasets NY --limit 1
  python analyze_problem3.py --from-result results/result3.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from config import ANALYSIS_DIR, DATASETS, RESULTS_DIR, queries_path
from graph import load_dataset, load_queries
from solver.epsilon_grid import build_epsilon_front, filter_nondominated
from solver.quality import evaluate_front
from solver.weighted_front import generate_weighted_candidates

OUT = ANALYSIS_DIR / "problem3"
OUT.mkdir(parents=True, exist_ok=True)


def qid_str(x) -> str:
    return f"{int(x):04d}"


def run_dim_effect(datasets: List[str], limit: int, samples: int, eps: float) -> Path:
    """
    Compare m=2/3/5 under a common weighted+filter pipeline for fair timing,
    plus report intended production modes (exact vs ε-grid).
    """
    rows = []
    for ds in datasets:
        g = load_dataset(ds)
        qdf = load_queries(queries_path(ds, "problem34"))
        if limit:
            qdf = qdf.iloc[:limit]
        for _, row in qdf.iterrows():
            qid = qid_str(row["query_id"])
            s, t = int(row["source"]), int(row["target"])
            for m, mode in ((2, "weighted_nd"), (3, "weighted_nd"), (5, "eps_grid")):
                t0 = time.time()
                costs, paths, zmin, zmax = generate_weighted_candidates(
                    g, s, t, m=m, n_samples=samples
                )
                if m == 5:
                    front, _, _, _ = build_epsilon_front(
                        costs, paths, m=5, eps=eps, z_min=zmin, z_max=zmax
                    )
                else:
                    front, _ = filter_nondominated(costs, m, paths)
                wall = time.time() - t0
                q = evaluate_front(front, m, eps=eps)
                rows.append(
                    {
                        "dataset": ds,
                        "query_id": qid,
                        "objective_count": m,
                        "mode": mode,
                        "samples": samples,
                        "eps": eps if m == 5 else "",
                        "n_solutions": len(front),
                        "n_candidates": len(costs),
                        "wall_sec": round(wall, 3),
                        "hv": round(q["hv"], 6),
                        "eps_coverage": round(q["eps_coverage"], 6),
                    }
                )
                print(
                    f"[dim] {ds} {qid} m={m} |S|={len(front)} wall={wall:.2f}s HV={q['hv']:.4f}",
                    flush=True,
                )
    path = OUT / "dim_effect.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def run_eps_sweep(
    datasets: List[str],
    limit: int,
    samples: int,
    eps_list: List[float],
) -> Path:
    rows = []
    for ds in datasets:
        g = load_dataset(ds)
        qdf = load_queries(queries_path(ds, "problem34"))
        if limit:
            qdf = qdf.iloc[:limit]
        for _, row in qdf.iterrows():
            qid = qid_str(row["query_id"])
            s, t = int(row["source"]), int(row["target"])
            costs, paths, zmin, zmax = generate_weighted_candidates(
                g, s, t, m=5, n_samples=samples
            )
            # reference = full ND of candidates
            ref, _ = filter_nondominated(costs, 5, paths)
            for eps in eps_list:
                t0 = time.time()
                front, _, _, _ = build_epsilon_front(
                    costs, paths, m=5, eps=eps, z_min=zmin, z_max=zmax
                )
                wall = time.time() - t0
                q = evaluate_front(front, 5, ref_costs=ref, eps=eps)
                rows.append(
                    {
                        "dataset": ds,
                        "query_id": qid,
                        "eps": eps,
                        "samples": samples,
                        "n_ref": len(ref),
                        "n_solutions": len(front),
                        "wall_sec": round(wall, 3),
                        "hv": round(q["hv"], 6),
                        "eps_coverage": round(q["eps_coverage"], 6),
                    }
                )
                print(
                    f"[eps] {ds} {qid} eps={eps} |S|={len(front)} "
                    f"cov={q['eps_coverage']:.3f} HV={q['hv']:.4f}",
                    flush=True,
                )
    path = OUT / "eps_sensitivity.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def summarize_result(result_csv: Path) -> Path:
    by = defaultdict(list)
    with open(result_csv, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["dataset"], row["query_id"], int(row["objective_count"]))
            cost = tuple(int(row[f"c{i}"]) for i in range(1, 6))
            by[key].append(cost)
    rows = []
    for (ds, qid, m), costs in sorted(by.items()):
        q = evaluate_front(costs, m)
        rows.append(
            {
                "dataset": ds,
                "query_id": qid,
                "objective_count": m,
                "n_solutions": len(costs),
                "hv": round(q["hv"], 6),
            }
        )
    path = OUT / "result3_quality_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["dataset"])
        w.writeheader()
        w.writerows(rows)
    # aggregate
    agg = defaultdict(list)
    for r in rows:
        agg[r["objective_count"]].append(r["n_solutions"])
    summary = {
        "groups": len(rows),
        "by_m": {
            str(m): {
                "groups": len(v),
                "mean_|S|": float(np.mean(v)),
                "min_|S|": int(np.min(v)),
                "max_|S|": int(np.max(v)),
            }
            for m, v in sorted(agg.items())
        },
    }
    (OUT / "result3_quality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="NY")
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--eps-list", default="0.01,0.03,0.05,0.1")
    ap.add_argument("--dim-effect", action="store_true")
    ap.add_argument("--eps-sweep", action="store_true")
    ap.add_argument("--from-result", type=Path, default=None)
    args = ap.parse_args()
    datasets = [d.strip().upper() for d in args.datasets.split(",") if d.strip()]

    if args.from_result:
        p = summarize_result(args.from_result)
        print(f"[ok] {p}")
        return 0
    if args.dim_effect:
        p = run_dim_effect(datasets, args.limit, args.samples, args.eps)
        print(f"[ok] {p}")
    if args.eps_sweep:
        eps_list = [float(x) for x in args.eps_list.split(",")]
        p = run_eps_sweep(datasets, args.limit, args.samples, eps_list)
        print(f"[ok] {p}")
    if not args.dim_effect and not args.eps_sweep and not args.from_result:
        ap.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
