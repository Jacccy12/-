# -*- coding: utf-8 -*-
"""
NY0001 resource-based EXACT with frozen T-MDA.

- max_labels=0 (generated is recorded only; not a stop condition)
- Guards: max_rss / max_nqp_live / optional wall
- Checkpoint every 5 min to ASCII path (resume via state.bin)
- Fragment written ONLY if status=EXACT (Q empty)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

from config import ANALYSIS_DIR, RESULTS_DIR, queries_path
from corridor import build_corridor, lexmin_all_orders
from graph import load_dataset, load_queries
from multiobjective_exact import load_or_compute_heuristics, namoa_star_exact, update_skyline
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
    DLL_PATH,
    SRC_PATH,
)

PARTS = RESULTS_DIR / "problem2_parts"
PARTS.mkdir(parents=True, exist_ok=True)
OUT = ANALYSIS_DIR / "problem2"
OUT.mkdir(parents=True, exist_ok=True)

# Frozen algorithm knobs (do not change mid-run)
BACKEND = "t_mda"
NQP_SWEEP = False
STATIC_CORRIDOR = True
SUFFIX_UB = False
DYNAMIC_CORRIDOR = False
PENDING_SKYLINE = False


def default_max_rss_mb() -> int:
    if not psutil:
        return 10000
    total = psutil.virtual_memory().total
    return max(2048, int(total * 0.65 / (1024 * 1024)))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true", help="Resume from checkpoint state.bin")
    ap.add_argument("--max-rss-mb", type=int, default=0, help="0=auto 65%% of RAM")
    ap.add_argument("--max-nqp-live", type=int, default=40_000_000)
    ap.add_argument("--max-live", type=int, default=0, help="0=disabled")
    ap.add_argument("--max-wall-sec", type=int, default=0, help="0=disabled")
    ap.add_argument("--ckpt-every-sec", type=int, default=300, help="status.json interval")
    ap.add_argument("--ckpt-state-every-sec", type=int, default=1800,
                    help="state.bin interval (full resume snapshot; expensive)")
    args = ap.parse_args()

    ensure_dll()
    max_rss = args.max_rss_mb or default_max_rss_mb()

    # ASCII checkpoint dir (Windows fopen UTF-8 path issues)
    ckpt = Path(tempfile.gettempdir()) / "tmda_exact_NY0001"
    ckpt.mkdir(parents=True, exist_ok=True)
    analysis_ckpt = OUT / "exact_NY_0001_t_mda_ckpt"
    analysis_ckpt.mkdir(parents=True, exist_ok=True)

    g = load_dataset("NY")
    q = load_queries(queries_path("NY", "problem2")).iloc[0]
    s, t = int(q["source"]), int(q["target"])
    qid = str(q["query_id"]).zfill(4)

    freeze = {
        "backend": BACKEND,
        "nqp_sweep": NQP_SWEEP,
        "static_corridor": STATIC_CORRIDOR,
        "suffix_ub": SUFFIX_UB,
        "dynamic_corridor": DYNAMIC_CORRIDOR,
        "pending_skyline": PENDING_SKYLINE,
        "max_labels_generated": 0,
        "max_rss_mb": max_rss,
        "max_nqp_live": args.max_nqp_live,
        "max_live": args.max_live,
        "max_wall_sec": args.max_wall_sec,
        "ckpt_every_sec": args.ckpt_every_sec,
        "ckpt_state_every_sec": args.ckpt_state_every_sec,
        "perf_v2": "PermFront block-index + state.bin every 30min",
        "dll": str(DLL_PATH),
        "dll_sha256": file_sha256(DLL_PATH) if DLL_PATH.exists() else None,
        "src_sha256": file_sha256(SRC_PATH),
        "query": {"dataset": "NY", "query_id": qid, "source": s, "target": t},
        "checkpoint_dir": str(ckpt),
    }
    (OUT / "T_MDA_FROZEN.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    (OUT / "T_MDA_FROZEN.md").write_text(
        f"""# T-MDA Frozen — Problem 2 resource EXACT

## Algorithm (DO NOT CHANGE mid-run)
- Backend: **T-MDA** (`t_mda.dll`)
- Static corridor: ON
- NQP sweep: **OFF**
- PendingSkyline: **OFF**
- Suffix UB / dynamic corridor / LB patches: **OFF**

## Resource guards
- `max_labels` (generated) = **0** (unlimited; recorded only)
- `max_rss_mb` = {max_rss} (~65% RAM)
- `max_nqp_live` = {args.max_nqp_live}
- `max_live` = {args.max_live or 'OFF'}
- `max_wall_sec` = {args.max_wall_sec or 'OFF'}

## Checkpoint
- Dir: `{ckpt}`
- Every: {args.ckpt_every_sec}s → `status.json` + `state.bin`
- Resume: `python run_t_mda_ny0001_exact.py --resume`

## Hashes
- src: `{freeze['src_sha256']}`
- dll: `{freeze['dll_sha256']}`

Fragment `NY_0001.csv` only if `status=EXACT` (Q=∅).
""",
        encoding="utf-8",
    )

    print(
        f"[exact] NY {qid} {s}->{t} T-MDA resource EXACT "
        f"rss_cap={max_rss}MB nqp_cap={args.max_nqp_live} "
        f"ckpt={ckpt} resume={args.resume}",
        flush=True,
    )

    hb = load_or_compute_heuristics(g, "NY", t, use_cache=True)
    bank = build_seed_bank(g, [s], t, hb, k=10)
    seeds = list(bank[s])
    for y in lexmin_all_orders(g, s - 1, t - 1):
        update_skyline(seeds, y)
    corr = build_corridor(g, s - 1, hb, seeds)
    print(f"[exact] seeds={len(seeds)} edge_keep={corr.n_edges_kept}", flush=True)

    set_nqp_sweep(False)
    set_resource_limits(
        max_rss_mb=max_rss,
        max_nqp_live=args.max_nqp_live,
        max_live=args.max_live,
        max_wall_sec=args.max_wall_sec,
    )
    set_checkpoint(str(ckpt), args.ckpt_every_sec)
    set_checkpoint_state_every(args.ckpt_state_every_sec)
    state_bin = ckpt / "state.bin"
    if args.resume and state_bin.exists():
        set_resume_path(str(state_bin))
        print(f"[exact] resuming from {state_bin}", flush=True)
    else:
        clear_resume_path()
        if args.resume:
            print("[exact] --resume set but state.bin missing; starting fresh", flush=True)

    proc = psutil.Process() if psutil else None
    rss0 = proc.memory_info().rss if proc else 0
    t0 = time.time()
    try:
        res = namoa_star_exact(
            g, s, t, heuristics=hb, use_heuristic=True,
            extra_seeds=seeds, edge_allowed=corr.edge_allowed.copy(),
            max_labels=0,  # unlimited generated
            max_expanded=0,
            backend="t_mda",
        )
    finally:
        clear_resume_path()
        clear_checkpoint()
        clear_resource_limits()
        set_nqp_sweep(False)

    wall = time.time() - t0
    rss1 = proc.memory_info().rss if proc else 0
    st = res.stats

    # Mirror checkpoint artifacts into analysis (may fail if Chinese path — try)
    for name in ("status.json", "status_history.csv", "state.bin"):
        src = ckpt / name
        if src.exists():
            try:
                shutil.copy2(src, analysis_ckpt / name)
            except OSError:
                pass

    if st.exact_finished:
        status = "EXACT"
    elif "RESOURCE" in (st.mode or ""):
        status = "RESOURCE_LIMIT"
    elif st.early_stop:
        status = "BENCH"
    else:
        status = "LABEL_LIMIT"

    meta = {
        "dataset": "NY",
        "query_id": qid,
        "source": s,
        "target": t,
        "backend": BACKEND,
        "frozen": freeze,
        "exact": bool(st.exact_finished),
        "status": status,
        "pareto_size": len(res.solutions),
        "runtime_sec": round(wall, 3),
        "generated": st.labels_generated,
        "expanded": st.labels_expanded,
        "peak_live": st.labels_peak,
        "peak_open": st.open_peak,
        "peak_nqp": st.u_bound_size,
        "nqp_live_end": st.stale_skipped,
        "rss_start_mb": round(rss0 / (1024 * 1024), 1) if proc else None,
        "rss_end_mb": round(rss1 / (1024 * 1024), 1) if proc else None,
        "mode": st.mode,
        "checkpoint_dir": str(ckpt),
    }
    meta_path = OUT / f"exact_NY_{qid}_t_mda.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"[exact] status={status} |S|={len(res.solutions)} gen={st.labels_generated} "
        f"exp={st.labels_expanded} open_peak={st.open_peak} nqp_peak={st.u_bound_size} "
        f"nqp_live={st.stale_skipped} wall={wall:.1f}s "
        f"rss_end={meta['rss_end_mb']}MB",
        flush=True,
    )

    if not st.exact_finished:
        print(
            "[exact] NOT writing fragment (not EXACT). "
            f"Resume with: python run_t_mda_ny0001_exact.py --resume",
            flush=True,
        )
        return 1

    sols = sorted(set(res.solutions))
    lines = ["dataset,query_id,source,target,solution_id,c1,c2,c3\n"]
    for i, (c1, c2, c3) in enumerate(sols, 1):
        lines.append(f"NY,{qid},{s},{t},{i},{c1},{c2},{c3}\n")
    frag = PARTS / f"NY_{qid}.csv"
    tmp = frag.with_suffix(".csv.tmp")
    tmp.write_text("".join(lines), encoding="utf-8")
    tmp.replace(frag)
    h = hashlib.sha256("".join(lines).encode()).hexdigest()[:16]
    meta["fragment"] = str(frag)
    meta["hash16"] = h
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (PARTS / f"NY_{qid}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[exact] wrote {frag} hash={h}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
