# -*- coding: utf-8 -*-
"""Aggregate T-MDA throughput vs worker count (exact-preserving, NO fragments).

Runs N parallel short probes (max_expanded) on distinct hard queries and reports:
  aggregate_exp_s = sum_k (expanded_k / wall_k)

Pick the largest N where aggregate still rises meaningfully and RSS fits.
Does not touch results/problem2_parts.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from config import ANALYSIS_DIR, queries_path
from corridor import build_corridor, lexmin_all_orders
from graph import load_dataset, load_queries
from multiobjective_exact import load_or_compute_heuristics, namoa_star_exact, update_skyline
from seed_bank import build_seed_bank
from t_mda import ensure_dll, set_nqp_sweep, set_resource_limits, clear_resource_limits

OUT = ANALYSIS_DIR / "problem2" / "worker_throughput"
SCAN = ANALYSIS_DIR / "problem2" / "difficulty_scan.csv"


def pick_jobs(n: int) -> list[dict]:
    """Prefer hard BAY/COL rows not yet fragmented; fall back to scan order."""
    parts = Path(__file__).resolve().parent / "results" / "problem2_parts"
    jobs = []
    with SCAN.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("bucket") not in ("hard", "extreme", "medium"):
                continue
            ds, qid = r["dataset"], str(r["query_id"]).zfill(4)
            if (parts / f"{ds}_{qid}.csv").exists():
                continue
            # skip BAY0002 if currently the production target (optional)
            jobs.append({
                "dataset": ds,
                "query_id": qid,
                "source": int(r["source"]),
                "target": int(r["target"]),
            })
            if len(jobs) >= n + 8:
                break
    # Prefer spread across datasets
    picked = []
    used = set()
    for ds in ("COL", "BAY", "NY"):
        for j in jobs:
            if j["dataset"] == ds and j["query_id"] not in used:
                picked.append(j)
                used.add((j["dataset"], j["query_id"]))
                if len(picked) >= n:
                    return picked
    return jobs[:n]


def _probe(job: dict) -> dict:
    ds = job["dataset"]
    qid = job["query_id"]
    s, t = int(job["source"]), int(job["target"])
    max_exp = int(job["max_expanded"])
    rss = int(job["max_rss_mb"])
    ensure_dll()
    set_nqp_sweep(False)
    set_resource_limits(max_rss_mb=rss, max_nqp_live=40_000_000, max_live=0, max_wall_sec=0)
    g = load_dataset(ds)
    hb = load_or_compute_heuristics(g, ds, t, use_cache=True)
    bank = build_seed_bank(g, [s], t, hb, k=10)
    seeds = list(bank[s])
    for y in lexmin_all_orders(g, s - 1, t - 1):
        update_skyline(seeds, y)
    corr = build_corridor(g, s - 1, hb, seeds)
    t0 = time.time()
    try:
        res = namoa_star_exact(
            g, s, t, heuristics=hb, use_heuristic=True,
            extra_seeds=seeds, edge_allowed=corr.edge_allowed.copy(),
            max_labels=0, max_expanded=max_exp, backend="t_mda",
        )
    finally:
        clear_resource_limits()
    wall = max(time.time() - t0, 1e-6)
    st = res.stats
    exp = int(st.labels_expanded)
    return {
        "dataset": ds,
        "query_id": qid,
        "pid": os.getpid(),
        "expanded": exp,
        "generated": int(st.labels_generated),
        "wall_sec": round(wall, 3),
        "exp_s": round(exp / wall, 1),
        "peak_nqp": int(st.u_bound_size),
        "rss_cap": rss,
    }


def run_config(n_workers: int, max_expanded: int, total_rss: int) -> dict:
    jobs = pick_jobs(n_workers)
    if len(jobs) < n_workers:
        raise SystemExit(f"need {n_workers} free queries, got {len(jobs)}")
    per_rss = max(1536, total_rss // n_workers)
    payload = []
    for j in jobs[:n_workers]:
        d = dict(j)
        d["max_expanded"] = max_expanded
        d["max_rss_mb"] = per_rss
        payload.append(d)
    print(f"[thru] workers={n_workers} stop@{max_expanded} per_rss={per_rss}", flush=True)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_probe, job) for job in payload]
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            print(
                f"  {r['dataset']} {r['query_id']} exp_s={r['exp_s']} "
                f"wall={r['wall_sec']}s exp={r['expanded']}",
                flush=True,
            )
    wall = time.time() - t0
    agg = sum(r["exp_s"] for r in results)
    return {
        "n_workers": n_workers,
        "max_expanded": max_expanded,
        "per_rss_mb": per_rss,
        "wall_batch_sec": round(wall, 2),
        "aggregate_exp_s": round(agg, 1),
        "mean_exp_s": round(agg / n_workers, 1),
        "results": results,
    }


def recommend(cfgs: list[dict], min_gain: float = 0.08) -> int:
    """Largest N with >=min_gain aggregate uplift vs previous."""
    best = cfgs[0]["n_workers"]
    prev = cfgs[0]["aggregate_exp_s"]
    for c in cfgs[1:]:
        gain = (c["aggregate_exp_s"] - prev) / max(prev, 1.0)
        if gain >= min_gain:
            best = c["n_workers"]
            prev = c["aggregate_exp_s"]
        else:
            break
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", default="1,2,3,4")
    ap.add_argument("--max-expanded", type=int, default=2_000_000)
    ap.add_argument("--total-rss-mb", type=int, default=9000)
    args = ap.parse_args()
    ensure_dll()
    OUT.mkdir(parents=True, exist_ok=True)
    ns = [int(x) for x in args.workers.split(",") if x.strip()]
    cfgs = []
    for n in ns:
        cfgs.append(run_config(n, args.max_expanded, args.total_rss_mb))
    rec = recommend(cfgs)
    report = {"configs": cfgs, "recommended_workers": rec}
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Worker throughput benchmark",
        "",
        f"stop@{args.max_expanded}  total_rss≈{args.total_rss_mb}MB",
        "",
        "| workers | aggregate exp/s | mean exp/s | batch wall |",
        "|--------:|----------------:|-----------:|-----------:|",
    ]
    for c in cfgs:
        lines.append(
            f"| {c['n_workers']} | {c['aggregate_exp_s']} | "
            f"{c['mean_exp_s']} | {c['wall_batch_sec']} |"
        )
    lines += ["", f"**recommended_workers = {rec}**", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[thru] recommended_workers={rec} -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
