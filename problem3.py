# -*- coding: utf-8 -*-
"""
Problem 3: multi-objective Pareto fronts for m=2,3,5 on queries_problem34.

Route
  m=2,3  → exact T-MDA (m=2 zeroes elevation so ND collapses to 2D)
  m=5    → weighted candidates + ε-grid approximate front

Usage
  python problem3.py --datasets NY --limit 1 --objectives 2,5
  python problem3.py --datasets NY,BAY,COL --objectives 2,3,5 --eps 0.05
  python problem3.py --merge-only
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from config import ANALYSIS_DIR, DATASETS, RESULTS_DIR, queries_path
from graph import load_dataset, load_queries
from solver.epsilon_grid import build_epsilon_front
from solver.exact_tmda import solve_tmda_exact
from solver.weighted_front import generate_weighted_candidates

PARTS = RESULTS_DIR / "problem3_parts"
RESULT3 = RESULTS_DIR / "result3.csv"
STATS = ANALYSIS_DIR / "problem3" / "problem3_stats.csv"
FIELDS = [
    "dataset",
    "query_id",
    "source",
    "target",
    "objective_count",
    "solution_id",
    "c1",
    "c2",
    "c3",
    "c4",
    "c5",
    "path",
]


def qid_str(x) -> str:
    return f"{int(x):04d}"


def frag_path(ds: str, qid: str, m: int) -> Path:
    return PARTS / f"{ds}_{qid}_m{m}.csv"


def path_to_str(path0: Optional[Sequence[int]]) -> str:
    if not path0:
        return ""
    return "->".join(str(int(v) + 1) for v in path0)


def write_fragment(
    ds: str,
    qid: str,
    source: int,
    target: int,
    m: int,
    solutions: Sequence[Sequence[int]],
    paths: Sequence[Optional[Sequence[int]]],
) -> Path:
    PARTS.mkdir(parents=True, exist_ok=True)
    path = frag_path(ds, qid, m)
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for sid, (cost, pth) in enumerate(zip(solutions, paths), start=1):
            c = list(cost) + [0] * (5 - len(cost))
            w.writerow(
                {
                    "dataset": ds,
                    "query_id": qid,
                    "source": source,
                    "target": target,
                    "objective_count": m,
                    "solution_id": sid,
                    "c1": int(c[0]),
                    "c2": int(c[1]),
                    "c3": int(c[2]),
                    "c4": int(c[3]),
                    "c5": int(c[4]),
                    "path": path_to_str(pth),
                }
            )
    tmp.replace(path)
    return path


def merge_parts(parts_dir: Path, out_path: Path) -> int:
    files = sorted(parts_dir.glob("*_m*.csv"))
    n = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as fout:
        writer = None
        for fp in files:
            with open(fp, newline="", encoding="utf-8") as fin:
                reader = csv.DictReader(fin)
                if writer is None:
                    writer = csv.DictWriter(fout, fieldnames=FIELDS)
                    writer.writeheader()
                for row in reader:
                    writer.writerow({k: row.get(k, "") for k in FIELDS})
                    n += 1
    return n


def solve_m5_epsilon(
    graph,
    source: int,
    target: int,
    *,
    eps: float,
    samples: int,
) -> Dict:
    t0 = time.time()
    costs, paths, z_min, z_max = generate_weighted_candidates(
        graph, source, target, m=5, n_samples=samples
    )
    front_c, front_p, _, _ = build_epsilon_front(
        costs, paths, m=5, eps=eps, z_min=z_min, z_max=z_max
    )
    wall = time.time() - t0
    return {
        "solutions": front_c,
        "paths": front_p,
        "exact": False,
        "wall_sec": wall,
        "pareto_size": len(front_c),
        "candidates": len(costs),
        "expanded": 0,
        "generated": len(costs),
        "mode": f"eps_grid_m5_eps{eps}_n{samples}",
        "status": "APPROX",
    }


def solve_one_group(
    ds: str,
    qid: str,
    source: int,
    target: int,
    graph,
    m: int,
    args,
) -> Dict:
    out_frag = frag_path(ds, qid, m)
    if out_frag.exists() and not args.force:
        n = sum(1 for _ in open(out_frag, encoding="utf-8")) - 1
        print(f"[skip] {ds} {qid} m={m} exists |S|≈{n}", flush=True)
        return {
            "dataset": ds,
            "query_id": qid,
            "objective_count": m,
            "status": "SKIP",
            "pareto_size": max(n, 0),
            "wall_sec": 0.0,
            "mode": "skip",
            "exact": int(m in (2, 3)),
        }

    print(f"[solve] {ds} {qid} m={m} ...", flush=True)
    if m in (2, 3):
        if args.approx_only:
            # fallback: dense weighted + ND (supported points only)
            from solver.epsilon_grid import filter_nondominated
            from solver.weighted_front import generate_weighted_candidates

            t0 = time.time()
            costs, paths, _, _ = generate_weighted_candidates(
                graph, source, target, m=m, n_samples=args.samples3 if m == 3 else args.samples2
            )
            fc, fp = filter_nondominated(costs, m, paths)
            order = sorted(range(len(fc)), key=lambda i: fc[i][:m])
            fc = [fc[i] for i in order]
            fp = [fp[i] for i in order]
            res = {
                "solutions": fc,
                "paths": fp,
                "exact": False,
                "wall_sec": time.time() - t0,
                "pareto_size": len(fc),
                "expanded": 0,
                "generated": len(costs),
                "mode": f"weighted_nd_m{m}",
                "status": "APPROX",
            }
        else:
            res = solve_tmda_exact(
                graph,
                ds,
                source,
                target,
                m=m,
                use_corridor=not args.no_corridor,
                max_rss_mb=args.max_rss_mb,
                max_wall_sec=args.max_wall_sec,
                resume=not args.no_resume,
            )
    elif m == 5:
        res = solve_m5_epsilon(
            graph,
            source,
            target,
            eps=args.eps,
            samples=args.samples5,
        )
    else:
        raise ValueError(f"unsupported m={m}")

    write_fragment(ds, qid, source, target, m, res["solutions"], res["paths"])
    print(
        f"[done] {ds} {qid} m={m} |S|={res['pareto_size']} "
        f"status={res['status']} wall={res['wall_sec']:.1f}s mode={res['mode']}",
        flush=True,
    )
    return {
        "dataset": ds,
        "query_id": qid,
        "source": source,
        "target": target,
        "objective_count": m,
        "status": res["status"],
        "exact": int(bool(res.get("exact"))),
        "pareto_size": res["pareto_size"],
        "wall_sec": round(float(res["wall_sec"]), 3),
        "mode": res["mode"],
        "expanded": res.get("expanded", 0),
        "generated": res.get("generated", 0),
    }


def parse_objectives(s: str) -> List[int]:
    ms = [int(x) for x in s.split(",") if x.strip()]
    for m in ms:
        if m not in (2, 3, 5):
            raise ValueError(f"objective_count must be 2,3,5 got {m}")
    return ms


def _worker_job(job: Tuple) -> Dict:
    """Process-pool worker: load graph once per task (simple, robust)."""
    ds, qid, source, target, m, args_dict = job
    ns = argparse.Namespace(**args_dict)
    g = load_dataset(ds)
    return solve_one_group(ds, qid, source, target, g, m, ns)


def main() -> int:
    ap = argparse.ArgumentParser(description="Problem 3: 2/3 exact T-MDA + 5 ε-grid")
    ap.add_argument("--datasets", default="NY,BAY,COL")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--objectives", default="2,3,5", help="e.g. 2,3,5 or 5")
    ap.add_argument("--eps", type=float, default=0.05, help="ε for 5-obj grid")
    ap.add_argument("--samples2", type=int, default=256)
    ap.add_argument("--samples3", type=int, default=1024)
    ap.add_argument("--samples5", type=int, default=4096)
    ap.add_argument("--max-rss-mb", type=int, default=12000)
    ap.add_argument("--max-wall-sec", type=int, default=0, help="0=unlimited for exact")
    ap.add_argument("--workers", type=int, default=1, help="parallel queries (m=5 recommended 2-4)")
    ap.add_argument("--no-corridor", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute existing fragments")
    ap.add_argument(
        "--approx-only",
        action="store_true",
        help="use weighted ND for m=2/3 (smoke / fallback, not exact)",
    )
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    ANALYSIS_DIR.joinpath("problem3").mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        n = merge_parts(PARTS, RESULT3)
        print(f"[merge] {n} rows -> {RESULT3}", flush=True)
        return 0

    datasets = [d.strip().upper() for d in args.datasets.split(",") if d.strip()]
    for d in datasets:
        if d not in DATASETS:
            raise SystemExit(f"unknown dataset {d}")
    objectives = parse_objectives(args.objectives)

    jobs: List[Tuple] = []
    for ds in datasets:
        qdf = load_queries(queries_path(ds, "problem34"))
        if args.limit is not None:
            qdf = qdf.iloc[: args.limit]
        for _, row in qdf.iterrows():
            qid = qid_str(row["query_id"])
            source, target = int(row["source"]), int(row["target"])
            for m in objectives:
                jobs.append((ds, qid, source, target, m, vars(args)))

    print(f"[p3] jobs={len(jobs)} workers={args.workers} objectives={objectives}", flush=True)
    stats_rows: List[Dict] = []
    n_workers = max(1, int(args.workers))
    if n_workers == 1:
        graphs = {}
        for ds, qid, source, target, m, _ in jobs:
            if ds not in graphs:
                graphs[ds] = load_dataset(ds)
            stats_rows.append(solve_one_group(ds, qid, source, target, graphs[ds], m, args))
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_worker_job, job) for job in jobs]
            for fut in as_completed(futs):
                st = fut.result()
                stats_rows.append(st)
                done += 1
                if done % 5 == 0 or done == len(jobs):
                    print(f"[p3] progress {done}/{len(jobs)}", flush=True)

    # append stats
    write_header = not STATS.exists()
    with open(STATS, "a", newline="", encoding="utf-8") as f:
        fields = [
            "dataset",
            "query_id",
            "source",
            "target",
            "objective_count",
            "status",
            "exact",
            "pareto_size",
            "wall_sec",
            "mode",
            "expanded",
            "generated",
        ]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in stats_rows:
            w.writerow(r)

    n = merge_parts(PARTS, RESULT3)
    meta = {
        "rows": n,
        "fragments": len(list(PARTS.glob("*_m*.csv"))),
        "objectives": objectives,
        "eps": args.eps,
        "result": str(RESULT3),
    }
    (ANALYSIS_DIR / "problem3" / "last_run.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[merge] {n} rows -> {RESULT3}", flush=True)
    print(json.dumps(meta, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
