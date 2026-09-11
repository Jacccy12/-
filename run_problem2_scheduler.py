# -*- coding: utf-8 -*-
"""
Problem 2 production pipeline helpers:
  - short difficulty scan (wall-capped)
  - stratified parallel scheduler (easy/medium high concurrency, hard low)
  - watchdog: keep legacy BAY0001 serial job, then hand off to scheduler

Does NOT interrupt an in-flight BAY0001 solve.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import psutil
except ImportError:
    psutil = None

from config import ANALYSIS_DIR, DATASETS, RESULTS_DIR, queries_path
from corridor import build_corridor
from graph import load_dataset, load_queries
from multiobjective_exact import load_or_compute_heuristics
from problem2 import merge_parts
from run_problem2_batch import (
    PARTS,
    RESULT2,
    SCAN,
    SCAN_MD,
    STATS,
    bucket_order,
    enrich_seeds,
    existing_exact_count,
    frag_path,
    qid_str,
    solve_one,
    default_max_rss_mb,
)
from seed_bank import build_seed_bank
from t_mda import (
    clear_checkpoint,
    clear_resource_limits,
    clear_resume_path,
    ensure_dll,
    set_nqp_sweep,
    set_resource_limits,
)

OUT = ANALYSIS_DIR / "problem2"
OUT.mkdir(parents=True, exist_ok=True)
SCHED_LOG = OUT / "scheduler.log"
WATCH_LOG = OUT / "watch_bay0001.log"
HOLD_FILE = OUT / "hold_queries.json"  # queries owned by another process


def classify_scan(
    exact: bool,
    open_peak: int,
    nqp_live: int,
    expanded: int,
    wall_sec: float,
) -> str:
    if exact:
        return "easy"
    # short probe did not finish
    if open_peak <= 1500 and nqp_live <= 150_000:
        return "medium"
    if open_peak <= 6000 and nqp_live <= 800_000:
        return "hard"
    return "extreme"


def parse_exclude(items: List[str]) -> Set[Tuple[str, str]]:
    out = set()
    for it in items or []:
        if ":" in it:
            ds, qid = it.split(":", 1)
        elif "_" in it:
            ds, qid = it.split("_", 1)
        else:
            continue
        out.add((ds.upper(), qid_str(qid)))
    return out


def run_scan(
    datasets: List[str],
    limit: Optional[int],
    scan_expanded: int,
    scan_wall_sec: int,
    exclude: Set[Tuple[str, str]],
    use_cache: bool,
    append: bool,
):
    from multiobjective_exact import (
        namoa_star_exact,
        verify_anchor_mins,
        verify_internal_nondominated,
    )
    from problem2 import write_query_fragment

    ensure_dll()
    set_nqp_sweep(False)
    clear_checkpoint()
    clear_resume_path()

    existing_rows: Dict[Tuple[str, str], dict] = {}
    if append and SCAN.exists():
        with SCAN.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                existing_rows[(r["dataset"], qid_str(r["query_id"]))] = r

    rows: List[dict] = []
    # keep prior rows for excluded / already scanned
    for k, r in existing_rows.items():
        if k in exclude or frag_path(k[0], k[1]).exists():
            rows.append(r)

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
                key = (ds, qid)
                s = int(row["source"])
                fp = frag_path(ds, qid)

                if fp.exists():
                    rec = {
                        "dataset": ds, "query_id": qid, "source": s, "target": target,
                        "status": "EXACT_EXISTING", "bucket": "easy",
                        "expanded": "", "generated": "", "open_peak": "",
                        "nqp_live": "", "pareto_size": "", "wall_sec": 0,
                        "speed_exp_s": "", "rss_mb": "", "exact": 1,
                    }
                    rows.append(rec)
                    print(f"[scan] {ds} {qid} EXACT_EXISTING -> easy", flush=True)
                    continue

                if key in exclude:
                    rec = {
                        "dataset": ds, "query_id": qid, "source": s, "target": target,
                        "status": "IN_FLIGHT", "bucket": "extreme",
                        "expanded": "", "generated": "", "open_peak": "",
                        "nqp_live": "", "pareto_size": "", "wall_sec": "",
                        "speed_exp_s": "", "rss_mb": "", "exact": 0,
                    }
                    rows.append(rec)
                    print(f"[scan] {ds} {qid} EXCLUDE (in-flight) -> extreme placeholder", flush=True)
                    continue

                if append and key in existing_rows and existing_rows[key].get("status") not in (
                    "IN_FLIGHT", ""
                ):
                    rows.append(existing_rows[key])
                    print(f"[scan] {ds} {qid} reuse prior scan row", flush=True)
                    continue

                seeds = enrich_seeds(g, s, target, bank.get(s, []))
                corr = build_corridor(g, s - 1, hb, seeds)

                # Wall-capped probe; generated unlimited
                set_resource_limits(
                    max_rss_mb=default_max_rss_mb(),
                    max_nqp_live=40_000_000,
                    max_live=0,
                    max_wall_sec=scan_wall_sec if scan_wall_sec > 0 else 0,
                )
                t0 = time.time()
                rss0 = psutil.Process().memory_info().rss if psutil else 0
                try:
                    res = namoa_star_exact(
                        g, s, target, heuristics=hb, use_heuristic=True,
                        extra_seeds=seeds, edge_allowed=corr.edge_allowed.copy(),
                        max_labels=0,
                        max_expanded=scan_expanded if scan_expanded > 0 else 0,
                        backend="t_mda",
                    )
                finally:
                    clear_resource_limits()
                wall = time.time() - t0
                rss1 = psutil.Process().memory_info().rss if psutil else 0
                st = res.stats
                exact = bool(st.exact_finished)
                open_peak = int(st.open_peak)
                nqp_live = int(st.stale_skipped)
                speed = (st.labels_expanded / wall) if wall > 0 else 0.0
                # RESOURCE_LIMIT from wall counts as BUDGET
                if exact:
                    status = "EXACT"
                elif "RESOURCE" in (st.mode or ""):
                    status = "WALL_BUDGET"
                elif st.early_stop:
                    status = "EXP_BUDGET"
                else:
                    status = "BUDGET"
                bucket = classify_scan(exact, open_peak, nqp_live, st.labels_expanded, wall)
                rec = {
                    "dataset": ds, "query_id": qid, "source": s, "target": target,
                    "status": status, "bucket": bucket,
                    "expanded": st.labels_expanded,
                    "generated": st.labels_generated,
                    "open_peak": open_peak,
                    "nqp_live": nqp_live,
                    "pareto_size": st.pareto_size,
                    "wall_sec": round(wall, 2),
                    "speed_exp_s": round(speed, 1),
                    "rss_mb": round((rss1 - rss0) / (1024 * 1024), 1) if psutil else "",
                    "exact": int(exact),
                }
                rows.append(rec)
                print(
                    f"[scan] {ds} {qid} {status} bucket={bucket} "
                    f"exp={st.labels_expanded} open={open_peak} nqp={nqp_live} "
                    f"|S|={st.pareto_size} speed={speed:.0f}/s wall={wall:.1f}s",
                    flush=True,
                )
                if exact:
                    s0 = s - 1
                    h_at_s = (int(hb.h[s0, 0]), int(hb.h[s0, 1]), int(hb.h[s0, 2]))
                    verify_internal_nondominated(res.solutions)
                    verify_anchor_mins(res.solutions, h_at_s)
                    write_query_fragment(
                        ds, qid, s, target, sorted(set(res.solutions)), PARTS
                    )

    # dedupe by (ds,qid) last wins
    uniq: Dict[Tuple[str, str], dict] = {}
    for r in rows:
        uniq[(r["dataset"], qid_str(r["query_id"]))] = r
    rows = list(uniq.values())

    fields = [
        "dataset", "query_id", "source", "target", "status", "bucket",
        "expanded", "generated", "open_peak", "nqp_live", "pareto_size",
        "wall_sec", "speed_exp_s", "rss_mb", "exact",
    ]
    with SCAN.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (bucket_order(x["bucket"]), x["dataset"], x["query_id"])):
            w.writerow(r)

    buckets = defaultdict(int)
    for r in rows:
        buckets[r["bucket"]] += 1
    md = [
        f"# Problem 2 Difficulty Scan",
        "",
        f"- scan_expanded={scan_expanded}",
        f"- scan_wall_sec={scan_wall_sec}",
        f"- exclude={sorted(exclude)}",
        f"- n={len(rows)}",
        "",
        "| bucket | count |",
        "|--------|------:|",
    ]
    for b in ("easy", "medium", "hard", "extreme"):
        md.append(f"| {b} | {buckets.get(b, 0)} |")
    md += [
        "",
        "| dataset | query_id | bucket | status | exp | open | nqp | |S| | speed | wall |",
        "|---------|----------|--------|--------|----:|-----:|----:|---:|------:|-----:|",
    ]
    for r in sorted(rows, key=lambda x: (bucket_order(x["bucket"]), x["dataset"], x["query_id"])):
        md.append(
            f"| {r['dataset']} | {r['query_id']} | {r['bucket']} | {r['status']} | "
            f"{r.get('expanded','')} | {r.get('open_peak','')} | {r.get('nqp_live','')} | "
            f"{r.get('pareto_size','')} | {r.get('speed_exp_s','')} | {r.get('wall_sec','')} |"
        )
    SCAN_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[scan] wrote {SCAN}", flush=True)
    print(f"[scan] buckets={dict(buckets)}", flush=True)
    return rows


def _worker_solve(job: dict) -> dict:
    """Process entry: isolate ctypes / DLL globals per worker."""
    ds = job["dataset"]
    qid = job["query_id"]
    source = int(job["source"])
    target = int(job["target"])
    bucket = job["bucket"]
    max_rss = int(job["max_rss_mb"])
    # Per-worker RSS share when parallel. Floor 5500MB: NY needs ~3GB+ to finish.
    workers = max(1, int(job.get("n_workers_hint", 1)))
    per_rss = max(5500, max_rss // workers)

    ensure_dll()
    g = load_dataset(ds)
    hb = load_or_compute_heuristics(g, ds, target, use_cache=True)
    bank = build_seed_bank(g, [source], target, hb, k=10)
    seeds = enrich_seeds(g, source, target, bank.get(source, []))
    corr = build_corridor(g, source - 1, hb, seeds)
    print(
        f"[worker {os.getpid()}] SOLVE {ds} {qid} bucket={bucket} rss_cap={per_rss}",
        flush=True,
    )
    meta = solve_one(
        ds, qid, source, target, g, hb, seeds, corr.edge_allowed,
        max_rss_mb=per_rss,
        max_nqp_live=int(job.get("max_nqp_live", 40_000_000)),
        max_wall_sec=int(job.get("max_wall_sec", 0)),
        ckpt_every=int(job.get("ckpt_every", 300)),
        ckpt_state_every=int(job.get("ckpt_state_every", 1800)),
        resume=True,
    )
    meta["bucket"] = bucket
    print(
        f"[worker {os.getpid()}] {ds} {qid} -> {meta['status']} "
        f"|S|={meta['pareto_size']} wall={meta['wall_sec']}s",
        flush=True,
    )
    return meta


def load_hold() -> Set[Tuple[str, str]]:
    if not HOLD_FILE.exists():
        return set()
    data = json.loads(HOLD_FILE.read_text(encoding="utf-8"))
    return {(x["dataset"], qid_str(x["query_id"])) for x in data.get("hold", [])}


def save_hold(pairs: Set[Tuple[str, str]], note: str = ""):
    HOLD_FILE.write_text(
        json.dumps(
            {
                "note": note,
                "hold": [{"dataset": d, "query_id": q} for d, q in sorted(pairs)],
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def build_jobs_from_scan(
    skip_hold: bool,
    buckets: Optional[List[str]] = None,
) -> List[dict]:
    if not SCAN.exists():
        raise SystemExit(f"missing scan file {SCAN}; run --scan-only first")
    hold = load_hold() if skip_hold else set()
    want = set(buckets) if buckets else {"easy", "medium", "hard", "extreme"}
    jobs = []
    with SCAN.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ds, qid = r["dataset"], qid_str(r["query_id"])
            if frag_path(ds, qid).exists():
                continue
            if (ds, qid) in hold:
                continue
            b = r.get("bucket", "medium")
            if b not in want:
                continue
            jobs.append({
                "dataset": ds,
                "query_id": qid,
                "source": int(r["source"]),
                "target": int(r["target"]),
                "bucket": b,
            })
    jobs.sort(key=lambda j: (bucket_order(j["bucket"]), j["dataset"], j["query_id"]))
    return jobs


def append_stats(meta: dict):
    fields = [
        "dataset", "query_id", "source", "target", "status", "exact",
        "pareto_size", "generated", "expanded", "peak_open", "peak_nqp",
        "nqp_live", "peak_live", "wall_sec", "mode", "bucket", "hash16",
    ]
    write_header = not STATS.exists()
    with STATS.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(meta)


def run_wave(jobs: List[dict], n_workers: int, args) -> int:
    if not jobs:
        return 0
    max_rss = args.max_rss_mb or default_max_rss_mb()
    payload = []
    for j in jobs:
        d = dict(j)
        d["max_rss_mb"] = max_rss
        d["max_nqp_live"] = args.max_nqp_live
        d["max_wall_sec"] = args.max_wall_sec
        d["ckpt_every"] = args.ckpt_every_sec
        d["ckpt_state_every"] = args.ckpt_state_every_sec
        d["n_workers_hint"] = n_workers
        payload.append(d)

    print(f"[sched] wave n={len(payload)} workers={n_workers}", flush=True)
    done = 0
    # Windows: need spawn; ProcessPoolExecutor handles it
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_worker_solve, job): job for job in payload}
        for fut in as_completed(futs):
            job = futs[fut]
            try:
                meta = fut.result()
            except Exception as e:
                print(f"[sched] FAIL {job['dataset']} {job['query_id']}: {e}", flush=True)
                continue
            append_stats(meta)
            if meta.get("exact"):
                done += 1
            have, total = existing_exact_count(list(DATASETS), None)
            print(f"[sched] progress fragments={have}/{total}", flush=True)
    return done


def run_scheduler(args):
    ensure_dll()
    # Phase 1: easy + medium parallel
    easy_med = build_jobs_from_scan(skip_hold=True, buckets=["easy", "medium"])
    print(f"[sched] easy/medium pending={len(easy_med)}", flush=True)
    run_wave(easy_med, n_workers=max(1, args.easy_workers), args=args)

    # Phase 2: hard / extreme low concurrency
    hard = build_jobs_from_scan(skip_hold=True, buckets=["hard", "extreme"])
    print(f"[sched] hard/extreme pending={len(hard)}", flush=True)
    run_wave(hard, n_workers=max(1, args.hard_workers), args=args)

    # Any leftovers (e.g. hold released)
    rest = build_jobs_from_scan(skip_hold=True)
    if rest:
        print(f"[sched] leftovers={len(rest)}", flush=True)
        run_wave(rest, n_workers=1, args=args)

    have, total = existing_exact_count(list(DATASETS), None)
    print(f"[sched] done fragments={have}/{total}", flush=True)
    if have == total and total > 0:
        n = merge_parts(PARTS, RESULT2)
        print(f"[sched] MERGED {n} rows -> {RESULT2}", flush=True)
        return 0
    return 1


def find_batch_pids() -> List[int]:
    if not psutil:
        return []
    pids = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(p.info.get("cmdline") or [])
            if "run_problem2_batch.py" in cmd and "--skip-scan" in cmd:
                pids.append(p.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return pids


def watch_bay_then_schedule(args):
    """Keep BAY0001 serial job; after its fragment appears, stop legacy batch and schedule."""
    bay_frag = frag_path("BAY", "0001")
    save_hold({("BAY", "0001")}, note="legacy skip-scan owns BAY0001")
    print(f"[watch] holding BAY:0001; waiting for {bay_frag.name}", flush=True)
    print(f"[watch] legacy pids={find_batch_pids()}", flush=True)

    while not bay_frag.exists():
        # still running?
        pids = find_batch_pids()
        if not pids:
            # maybe finished without us noticing, or crashed
            log = OUT / "batch_exact_run.log"
            if log.exists():
                tail = log.read_text(encoding="utf-8", errors="ignore")[-2000:]
                if "BAY 0001 -> EXACT" in tail or bay_frag.exists():
                    break
            print("[watch] legacy batch gone; BAY fragment missing — will schedule without it", flush=True)
            break
        print(
            f"[watch] waiting BAY0001... pids={pids} frags={existing_exact_count(list(DATASETS), None)}",
            flush=True,
        )
        time.sleep(60)

    if bay_frag.exists():
        print("[watch] BAY_0001.csv present", flush=True)

    # Stop legacy batch so it does not start BAY0002 serial
    for pid in find_batch_pids():
        try:
            if psutil:
                p = psutil.Process(pid)
                print(f"[watch] terminating legacy batch pid={pid}", flush=True)
                p.terminate()
                try:
                    p.wait(timeout=30)
                except psutil.TimeoutExpired:
                    p.kill()
        except Exception as e:
            print(f"[watch] terminate failed {pid}: {e}", flush=True)

    # release hold
    save_hold(set(), note="cleared after BAY0001")
    # update scan row for BAY0001 if exact
    if bay_frag.exists() and SCAN.exists():
        rows = list(csv.DictReader(SCAN.open(encoding="utf-8")))
        fields = rows[0].keys() if rows else []
        changed = False
        for r in rows:
            if r["dataset"] == "BAY" and qid_str(r["query_id"]) == "0001":
                r["status"] = "EXACT_EXISTING"
                r["bucket"] = "easy"
                r["exact"] = "1"
                changed = True
        if changed:
            with SCAN.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(fields))
                w.writeheader()
                w.writerows(rows)

    print("[watch] launching scheduler", flush=True)
    return run_scheduler(args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--scan-only", action="store_true")
    ap.add_argument("--scan-expanded", type=int, default=400_000,
                    help="expansion budget for probe (0=wall only)")
    ap.add_argument("--scan-wall-sec", type=int, default=120,
                    help="wall-time budget per probe query (default 2 min)")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="ds:qid pairs to skip in scan (e.g. BAY:0001)")
    ap.add_argument("--append-scan", action="store_true")
    ap.add_argument("--schedule", action="store_true",
                    help="run stratified parallel scheduler from scan file")
    ap.add_argument("--watch-bay-then-schedule", action="store_true",
                    help="wait for BAY0001 fragment, stop legacy batch, then schedule")
    ap.add_argument("--set-hold", nargs="*", default=None,
                    help="set hold list ds:qid")
    ap.add_argument("--easy-workers", type=int, default=2)
    ap.add_argument("--hard-workers", type=int, default=1)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--max-rss-mb", type=int, default=0)
    ap.add_argument("--max-nqp-live", type=int, default=40_000_000)
    ap.add_argument("--max-wall-sec", type=int, default=0)
    ap.add_argument("--ckpt-every-sec", type=int, default=300)
    ap.add_argument("--ckpt-state-every-sec", type=int, default=1800)
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    if args.merge_only:
        n = merge_parts(PARTS, RESULT2)
        print(f"merged {n} rows -> {RESULT2}")
        return

    if args.set_hold is not None:
        save_hold(parse_exclude(args.set_hold), note="manual hold")
        print(f"hold={load_hold()}")
        return

    if args.scan_only:
        excl = parse_exclude(args.exclude)
        # always exclude holds
        excl |= load_hold()
        run_scan(
            args.datasets, args.limit,
            scan_expanded=args.scan_expanded,
            scan_wall_sec=args.scan_wall_sec,
            exclude=excl,
            use_cache=not args.no_cache,
            append=args.append_scan,
        )
        return

    if args.watch_bay_then_schedule:
        raise SystemExit(watch_bay_then_schedule(args))

    if args.schedule:
        raise SystemExit(run_scheduler(args))

    ap.print_help()


if __name__ == "__main__":
    # Windows ProcessPoolExecutor needs freeze_support
    import multiprocessing as mp
    mp.freeze_support()
    main()
