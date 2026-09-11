# -*- coding: utf-8 -*-
"""
Problem 2 batch EXACT with frozen T-MDA + resource guards.

Pipeline:
  1) Optional difficulty scan (max_expanded budget) → easy/medium/hard/extreme
  2) Solve remaining queries in easy→hard order (skip existing EXACT fragments)
  3) Fragment only if status=EXACT (Q=∅)
  4) When 90/90 present, merge → results/result2.csv

Usage:
  python run_problem2_batch.py --scan-only --scan-expanded 500000
  python run_problem2_batch.py                 # full batch
  python run_problem2_batch.py --merge-only
  python run_problem2_batch.py --datasets NY --limit 5
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import psutil
except ImportError:
    psutil = None

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
from problem2 import merge_parts, write_query_fragment
from seed_bank import build_seed_bank
from t_mda import (
    clear_checkpoint,
    clear_resource_limits,
    clear_resume_path,
    ensure_dll,
    set_checkpoint,
    set_checkpoint_state_every,
    set_nqp_sweep,
    set_resource_limits,
    set_resume_path,
)

PARTS = RESULTS_DIR / "problem2_parts"
OUT = ANALYSIS_DIR / "problem2"
OUT.mkdir(parents=True, exist_ok=True)
PARTS.mkdir(parents=True, exist_ok=True)

RESULT2 = RESULTS_DIR / "result2.csv"
STATS = OUT / "batch_query_stats.csv"
SCAN = OUT / "difficulty_scan.csv"
SCAN_MD = OUT / "difficulty_scan.md"
PROGRESS = OUT / "batch_progress.json"


def qid_str(qid) -> str:
    return str(qid).zfill(4)


def frag_path(ds: str, qid) -> Path:
    return PARTS / f"{ds}_{qid_str(qid)}.csv"


def default_max_rss_mb() -> int:
    if not psutil:
        return 10000
    return max(2048, int(psutil.virtual_memory().total * 0.65 / (1024 * 1024)))


def enrich_seeds(graph, source: int, target: int, base):
    seeds = list(base)
    for y in lexmin_all_orders(graph, source - 1, target - 1):
        update_skyline(seeds, y)
    return seeds


def classify(expanded: int, open_end: int, nqp_end: int, exact: bool) -> str:
    if exact:
        return "easy"
    # budget hit
    if open_end <= 2000 and nqp_end <= 200_000:
        return "medium"
    if open_end <= 8000 and nqp_end <= 1_000_000:
        return "hard"
    return "extreme"


def list_all_queries(datasets: List[str], limit: Optional[int]) -> List[Tuple[str, object]]:
    items = []
    for ds in datasets:
        q = load_queries(queries_path(ds, "problem2"))
        if limit is not None:
            q = q.iloc[:limit]
        for _, row in q.iterrows():
            items.append((ds, row))
    return items


def existing_exact_count(datasets: List[str], limit: Optional[int]) -> Tuple[int, int]:
    items = list_all_queries(datasets, limit)
    have = sum(1 for ds, row in items if frag_path(ds, row["query_id"]).exists())
    return have, len(items)


def run_scan(datasets: List[str], limit: Optional[int], scan_expanded: int, use_cache: bool):
    ensure_dll()
    set_nqp_sweep(False)
    clear_resource_limits()
    clear_checkpoint()
    clear_resume_path()

    rows = []
    for ds in datasets:
        g = load_dataset(ds)
        queries = load_queries(queries_path(ds, "problem2"))
        if limit is not None:
            queries = queries.iloc[:limit]
        by_target: Dict[int, list] = defaultdict(list)
        for _, row in queries.iterrows():
            by_target[int(row["target"])].append(row)

        for target, trows in by_target.items():
            hb = load_or_compute_heuristics(g, ds, target, use_cache=use_cache)
            sources = [int(r["source"]) for r in trows]
            bank = build_seed_bank(g, sources, target, hb, k=10)
            for row in trows:
                qid = qid_str(row["query_id"])
                s = int(row["source"])
                fp = frag_path(ds, qid)
                if fp.exists():
                    rows.append({
                        "dataset": ds, "query_id": qid, "source": s, "target": target,
                        "status": "EXACT_EXISTING", "bucket": "easy",
                        "expanded": "", "generated": "", "open_peak": "", "nqp_live": "",
                        "pareto_size": "", "wall_sec": 0, "exact": 1,
                    })
                    print(f"[scan] {ds} {qid} skip (fragment exists)", flush=True)
                    continue
                seeds = enrich_seeds(g, s, target, bank.get(s, []))
                corr = build_corridor(g, s - 1, hb, seeds)
                t0 = time.time()
                res = namoa_star_exact(
                    g, s, target, heuristics=hb, use_heuristic=True,
                    extra_seeds=seeds, edge_allowed=corr.edge_allowed.copy(),
                    max_labels=0, max_expanded=scan_expanded, backend="t_mda",
                )
                wall = time.time() - t0
                st = res.stats
                exact = bool(st.exact_finished)
                open_end = int(st.open_peak)  # peak during; for scan use live if available
                # Prefer live open from labels_live / stale for nqp
                nqp_end = int(st.stale_skipped)  # nqp_live end for t_mda
                # open at end not directly exposed; use open_peak as proxy for scan
                open_proxy = int(st.open_peak)
                bucket = classify(st.labels_expanded, open_proxy, nqp_end, exact)
                status = "EXACT" if exact else "BUDGET"
                row_out = {
                    "dataset": ds, "query_id": qid, "source": s, "target": target,
                    "status": status, "bucket": bucket,
                    "expanded": st.labels_expanded, "generated": st.labels_generated,
                    "open_peak": open_proxy, "nqp_live": nqp_end,
                    "pareto_size": st.pareto_size, "wall_sec": round(wall, 2),
                    "exact": int(exact),
                }
                rows.append(row_out)
                print(
                    f"[scan] {ds} {qid} {status} bucket={bucket} "
                    f"exp={st.labels_expanded} open_peak={open_proxy} "
                    f"nqp={nqp_end} |S|={st.pareto_size} wall={wall:.1f}s",
                    flush=True,
                )
                if exact:
                    s0 = s - 1
                    h_at_s = (int(hb.h[s0, 0]), int(hb.h[s0, 1]), int(hb.h[s0, 2]))
                    verify_internal_nondominated(res.solutions)
                    verify_anchor_mins(res.solutions, h_at_s)
                    # normalize qid in fragment to zero-padded string via write_query_fragment
                    write_query_fragment(ds, qid, s, target, sorted(set(res.solutions)), PARTS)

    # write scan csv
    if rows:
        fields = list(rows[0].keys())
        with SCAN.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    buckets = defaultdict(int)
    for r in rows:
        buckets[r["bucket"]] += 1
    md = [
        f"# Problem 2 Difficulty Scan @ {scan_expanded:,} expansions",
        "",
        f"Total scanned/skipped: {len(rows)}",
        "",
        "| bucket | count |",
        "|--------|------:|",
    ]
    for b in ("easy", "medium", "hard", "extreme"):
        md.append(f"| {b} | {buckets.get(b, 0)} |")
    md.append("")
    md.append("| dataset | query_id | bucket | status | exp | open_peak | nqp | |S| | wall |")
    md.append("|---------|----------|--------|--------|----:|----------:|----:|---:|-----:|")
    for r in sorted(rows, key=lambda x: ({"easy": 0, "medium": 1, "hard": 2, "extreme": 3}.get(x["bucket"], 9), x["dataset"], x["query_id"])):
        md.append(
            f"| {r['dataset']} | {r['query_id']} | {r['bucket']} | {r['status']} | "
            f"{r.get('expanded','')} | {r.get('open_peak', r.get('open',''))} | "
            f"{r.get('nqp_live', r.get('nqp',''))} | {r.get('pareto_size','')} | {r.get('wall_sec','')} |"
        )
    SCAN_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[scan] wrote {SCAN} and {SCAN_MD}", flush=True)
    print(f"[scan] buckets={dict(buckets)}", flush=True)
    return rows


def bucket_order(bucket: str) -> int:
    return {"easy": 0, "medium": 1, "hard": 2, "extreme": 3}.get(bucket, 9)


def load_scan_order() -> Dict[Tuple[str, str], str]:
    if not SCAN.exists():
        return {}
    out = {}
    with SCAN.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["dataset"], qid_str(r["query_id"]))] = r.get("bucket", "medium")
    return out


def solve_one(
    ds: str,
    qid: str,
    source: int,
    target: int,
    g,
    hb,
    seeds,
    edge_mask,
    max_rss_mb: int,
    max_nqp_live: int,
    max_wall_sec: int,
    ckpt_every: int,
    ckpt_state_every: int,
    resume: bool = True,
) -> dict:
    ckpt = Path(tempfile.gettempdir()) / f"tmda_p2_{ds}_{qid}"
    ckpt.mkdir(parents=True, exist_ok=True)
    state_bin = ckpt / "state.bin"

    set_nqp_sweep(False)
    set_resource_limits(
        max_rss_mb=max_rss_mb,
        max_nqp_live=max_nqp_live,
        max_live=0,
        max_wall_sec=max_wall_sec,
    )
    set_checkpoint(str(ckpt), ckpt_every)
    set_checkpoint_state_every(ckpt_state_every)
    if resume and state_bin.exists():
        set_resume_path(str(state_bin))
        print(f"[resume] {ds} {qid} from {state_bin}", flush=True)
    else:
        clear_resume_path()

    t0 = time.time()
    try:
        res = namoa_star_exact(
            g, source, target, heuristics=hb, use_heuristic=True,
            extra_seeds=seeds, edge_allowed=edge_mask.copy(),
            max_labels=0, max_expanded=0, backend="t_mda",
        )
    finally:
        clear_resume_path()
        clear_checkpoint()
        clear_resource_limits()

    wall = time.time() - t0
    st = res.stats
    exact = bool(st.exact_finished)
    if "RESOURCE" in (st.mode or ""):
        status = "RESOURCE_LIMIT"
    elif exact:
        status = "EXACT"
    elif st.early_stop:
        status = "EARLY_STOP"
    else:
        status = "INCOMPLETE"

    out = {
        "dataset": ds,
        "query_id": qid,
        "source": source,
        "target": target,
        "status": status,
        "exact": int(exact),
        "pareto_size": len(res.solutions),
        "generated": st.labels_generated,
        "expanded": st.labels_expanded,
        "peak_open": st.open_peak,
        "peak_nqp": st.u_bound_size,
        "nqp_live": st.stale_skipped,
        "peak_live": st.labels_peak,
        "wall_sec": round(wall, 2),
        "mode": st.mode,
    }

    if exact:
        s0 = source - 1
        h_at_s = (int(hb.h[s0, 0]), int(hb.h[s0, 1]), int(hb.h[s0, 2]))
        verify_internal_nondominated(res.solutions)
        verify_anchor_mins(res.solutions, h_at_s)
        sols = sorted(set(res.solutions))
        write_query_fragment(ds, qid, source, target, sols, PARTS)
        h = hashlib.sha256(
            ("\n".join(f"{a},{b},{c}" for a, b, c in sols)).encode()
        ).hexdigest()[:16]
        out["hash16"] = h
        out["fragment"] = str(frag_path(ds, qid))
    return out


def run_batch(args):
    ensure_dll()
    max_rss = args.max_rss_mb or default_max_rss_mb()
    scan_map = load_scan_order()

    # Build worklist
    work = []
    for ds in args.datasets:
        q = load_queries(queries_path(ds, "problem2"))
        if args.limit is not None:
            q = q.iloc[: args.limit]
        for _, row in q.iterrows():
            qid = qid_str(row["query_id"])
            bucket = scan_map.get((ds, qid), "medium")
            work.append((bucket_order(bucket), bucket, ds, qid, int(row["source"]), int(row["target"])))
    work.sort()

    have, total = existing_exact_count(args.datasets, args.limit)
    print(f"[batch] {have}/{total} fragments already present; max_rss={max_rss}MB", flush=True)

    # stats file append/create
    stat_fields = [
        "dataset", "query_id", "source", "target", "status", "exact",
        "pareto_size", "generated", "expanded", "peak_open", "peak_nqp",
        "nqp_live", "peak_live", "wall_sec", "mode", "bucket", "hash16",
    ]
    write_header = not STATS.exists()
    fstat = STATS.open("a", newline="", encoding="utf-8")
    sw = csv.DictWriter(fstat, fieldnames=stat_fields, extrasaction="ignore")
    if write_header:
        sw.writeheader()

    # group by (ds, target) for heuristics reuse
    by_ds_target: Dict[Tuple[str, int], list] = defaultdict(list)
    for order_i, bucket, ds, qid, s, t in work:
        by_ds_target[(ds, t)].append((order_i, bucket, qid, s))

    # But we want global easy-first; so iterate sorted work, caching graphs/hb/banks
    graph_cache = {}
    hb_cache = {}
    bank_cache = {}

    done_exact = have
    results_meta = []

    try:
        for order_i, bucket, ds, qid, source, target in work:
            fp = frag_path(ds, qid)
            if fp.exists():
                print(f"[batch] skip {ds} {qid} (exists)", flush=True)
                continue

            if ds not in graph_cache:
                graph_cache[ds] = load_dataset(ds)
            g = graph_cache[ds]
            key = (ds, target)
            if key not in hb_cache:
                print(f"[batch] heuristics {ds} target={target}", flush=True)
                hb_cache[key] = load_or_compute_heuristics(
                    g, ds, target, use_cache=not args.no_cache
                )
                # build bank for all sources of this target still in work
                sources = [s for _, _, d, _, s, t in work if d == ds and t == target]
                sources = sorted(set(sources))
                bank_cache[key] = build_seed_bank(g, sources, target, hb_cache[key], k=10)

            hb = hb_cache[key]
            bank = bank_cache[key]
            seeds = enrich_seeds(g, source, target, bank.get(source, []))
            corr = build_corridor(g, source - 1, hb, seeds)
            print(
                f"[batch] SOLVE {ds} {qid} bucket={bucket} "
                f"seeds={len(seeds)} edges_kept={corr.n_edges_kept}",
                flush=True,
            )
            meta = solve_one(
                ds, qid, source, target, g, hb, seeds, corr.edge_allowed,
                max_rss_mb=max_rss,
                max_nqp_live=args.max_nqp_live,
                max_wall_sec=args.max_wall_sec,
                ckpt_every=args.ckpt_every_sec,
                ckpt_state_every=args.ckpt_state_every_sec,
            )
            meta["bucket"] = bucket
            results_meta.append(meta)
            sw.writerow(meta)
            fstat.flush()

            if meta["exact"]:
                done_exact += 1
            print(
                f"[batch] {ds} {qid} -> {meta['status']} |S|={meta['pareto_size']} "
                f"exp={meta['expanded']} wall={meta['wall_sec']}s "
                f"progress={done_exact}/{total}",
                flush=True,
            )
            PROGRESS.write_text(
                json.dumps(
                    {
                        "exact_done": done_exact,
                        "total": total,
                        "last": meta,
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    finally:
        fstat.close()

    have2, total2 = existing_exact_count(args.datasets, args.limit)
    print(f"[batch] fragments now {have2}/{total2}", flush=True)
    if have2 == total2 and total2 > 0:
        n = merge_parts(PARTS, RESULT2)
        print(f"[batch] MERGED {n} solution rows -> {RESULT2}", flush=True)
        return 0
    print(
        f"[batch] NOT merging yet ({have2}/{total2}). "
        f"Re-run to continue; RESOURCE_LIMIT queries need more RAM/time.",
        flush=True,
    )
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--scan-only", action="store_true")
    ap.add_argument("--scan-expanded", type=int, default=500_000)
    ap.add_argument("--skip-scan", action="store_true", help="Go straight to full EXACT")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--max-rss-mb", type=int, default=0)
    ap.add_argument("--max-nqp-live", type=int, default=40_000_000)
    ap.add_argument("--max-wall-sec", type=int, default=0, help="0=unlimited per query")
    ap.add_argument("--ckpt-every-sec", type=int, default=300)
    ap.add_argument("--ckpt-state-every-sec", type=int, default=1800)
    args = ap.parse_args()

    if args.merge_only:
        n = merge_parts(PARTS, RESULT2)
        print(f"merged {n} rows -> {RESULT2}")
        return

    if args.scan_only:
        run_scan(args.datasets, args.limit, args.scan_expanded, use_cache=not args.no_cache)
        return

    if not args.skip_scan and not SCAN.exists():
        print("[batch] no scan file; running difficulty scan first", flush=True)
        run_scan(args.datasets, args.limit, args.scan_expanded, use_cache=not args.no_cache)

    raise SystemExit(run_batch(args))


if __name__ == "__main__":
    main()
