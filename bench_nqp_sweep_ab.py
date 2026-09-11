# -*- coding: utf-8 -*-
"""
NY0001 @ 3M: T-MDA baseline vs periodic NQP stale sweep + pending-dom diagnostic.

Does NOT write EXACT fragments. Does NOT full-rerun NY0001.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

from config import ANALYSIS_DIR, queries_path
from corridor import build_corridor, lexmin_all_orders
from graph import load_dataset, load_queries
from multiobjective_exact import load_or_compute_heuristics, namoa_star_exact, update_skyline
from seed_bank import build_seed_bank
from t_mda import (
    clear_profile_dir,
    ensure_dll,
    set_nqp_sweep,
    set_profile_dir,
)

OUT = ANALYSIS_DIR / "problem2" / "nqp_sweep_ab_NY_0001"
MAX_EXP = 3_000_000
SWEEP_EVERY = 500_000


def make_seeds(g, s, t, hb):
    bank = build_seed_bank(g, [s], t, hb, k=10)
    seeds = list(bank[s])
    for y in lexmin_all_orders(g, s - 1, t - 1):
        update_skyline(seeds, y)
    return seeds


def read_summary(dir_path: Path) -> dict:
    p = dir_path / "nqp_summary.csv"
    out = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines()[1:]:
        if "," not in line:
            continue
        k, v = line.split(",", 1)
        out[k] = v
    return out


def read_pending(dir_path: Path) -> list[dict]:
    p = dir_path / "pending_dom_checkpoints.csv"
    rows = []
    if not p.exists():
        return rows
    lines = p.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return rows
    hdr = lines[0].split(",")
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) != len(hdr):
            continue
        rows.append(dict(zip(hdr, parts)))
    return rows


def run_one(tag: str, sweep: bool, g, s, t, hb, seeds, edge_mask, tmp_root: Path):
    prof = tmp_root / tag
    if prof.exists():
        shutil.rmtree(prof)
    prof.mkdir(parents=True)

    proc = psutil.Process() if psutil else None
    rss0 = proc.memory_info().rss if proc else 0
    set_profile_dir(str(prof))
    set_nqp_sweep(sweep, SWEEP_EVERY)
    t0 = time.time()
    try:
        res = namoa_star_exact(
            g, s, t, heuristics=hb, use_heuristic=True,
            extra_seeds=seeds, edge_allowed=edge_mask.copy(),
            max_labels=80_000_000, max_expanded=MAX_EXP,
            backend="t_mda",
        )
    finally:
        set_nqp_sweep(False)
        clear_profile_dir()
    wall = time.time() - t0
    rss1 = proc.memory_info().rss if proc else 0
    st = res.stats
    summary = read_summary(prof)
    pending = read_pending(prof)

    dest = OUT / tag
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(prof, dest)

    return {
        "tag": tag,
        "sweep": sweep,
        "wall_sec": round(wall, 2),
        "rss_end_mb": round(rss1 / (1024 * 1024), 1) if proc else None,
        "rss_delta_mb": round((rss1 - rss0) / (1024 * 1024), 1) if proc else None,
        "generated": st.labels_generated,
        "expanded": st.labels_expanded,
        "peak_open": st.open_peak,
        "peak_live": st.labels_peak,
        "peak_nqp": st.u_bound_size,
        "nqp_live_end": st.stale_skipped,
        "pareto_size": st.pareto_size,
        "pareto_head": res.solutions[:5],
        "dll_summary": summary,
        "pending_checkpoints": pending,
    }


def pct(new, old):
    if old in (None, 0):
        return None
    return round(100.0 * (new - old) / old, 2)


def main():
    ensure_dll()
    OUT.mkdir(parents=True, exist_ok=True)

    g = load_dataset("NY")
    q = load_queries(queries_path("NY", "problem2")).iloc[0]
    s, t = int(q["source"]), int(q["target"])
    print(f"[sweep-ab] NY0001 {s}->{t} stop@{MAX_EXP} every={SWEEP_EVERY}", flush=True)

    hb = load_or_compute_heuristics(g, "NY", t, use_cache=True)
    seeds = make_seeds(g, s, t, hb)
    corr = build_corridor(g, s - 1, hb, seeds)
    edge_mask = corr.edge_allowed.copy()
    print(f"[sweep-ab] seeds={len(seeds)} edge_keep={corr.n_edges_kept}", flush=True)

    tmp_root = Path(tempfile.gettempdir()) / "tmda_nqp_sweep_ab_NY0001"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True)

    rows = []
    for tag, sweep in [("A_baseline", False), ("B_sweep", True)]:
        print(f"\n===== {tag} sweep={sweep} =====", flush=True)
        row = run_one(tag, sweep, g, s, t, hb, seeds, edge_mask, tmp_root)
        print(
            f"  wall={row['wall_sec']}s gen={row['generated']} "
            f"peak_nqp={row['peak_nqp']} live_nqp={row['nqp_live_end']} "
            f"pareto={row['pareto_size']}",
            flush=True,
        )
        rows.append(row)

    a, b = rows
    sum_a, sum_b = a["dll_summary"], b["dll_summary"]

    def fnum(d, k, default=None):
        try:
            return float(d.get(k, default))
        except (TypeError, ValueError):
            return default

    peak_a = a["peak_nqp"] or 0
    peak_b = b["peak_nqp"] or 0
    live_a = a["nqp_live_end"] or 0
    live_b = b["nqp_live_end"] or 0
    nqp_drop_peak = pct(peak_b, peak_a)
    nqp_drop_live = pct(live_b, live_a)
    wall_overhead = pct(b["wall_sec"], a["wall_sec"])

    # Pareto identity: same size + same head (bench may truncate; compare full lists if small)
    pareto_ok = a["pareto_size"] == b["pareto_size"] and a["pareto_head"] == b["pareto_head"]

    gate_nqp = (nqp_drop_peak is not None and nqp_drop_peak <= -10.0) or (
        nqp_drop_live is not None and nqp_drop_live <= -10.0
    )
    gate_wall = wall_overhead is None or wall_overhead <= 8.0
    gate_pass = bool(pareto_ok and gate_nqp and gate_wall)

    # Pending diagnosis from B (sweep on) — also dump from A if present
    pending_rows = b["pending_checkpoints"] or a["pending_checkpoints"]
    R_at_3M = None
    for pr in pending_rows:
        if pr.get("expanded") in ("3000000", "3e+06"):
            try:
                R_at_3M = float(pr["R_pending"])
            except (KeyError, ValueError):
                pass

    verdict_pending = "unknown"
    if R_at_3M is not None:
        if R_at_3M >= 0.40:
            verdict_pending = "IMPLEMENT_PendingSkyline_HIGH"
        elif R_at_3M >= 0.20:
            verdict_pending = "IMPLEMENT_PendingSkyline_WORTH"
        elif R_at_3M < 0.10:
            verdict_pending = "STOP_NQP_CLEANUP_WIDE_FRONTIER"
        else:
            verdict_pending = "MARGINAL_PendingSkyline"

    report = {
        "query": "NY_0001",
        "max_expanded": MAX_EXP,
        "sweep_every": SWEEP_EVERY,
        "rows": [
            {k: v for k, v in r.items() if k != "dll_summary"}
            | {"dll_summary": r["dll_summary"]}
            for r in rows
        ],
        "delta_B_vs_A": {
            "peak_nqp_pct": nqp_drop_peak,
            "nqp_live_pct": nqp_drop_live,
            "wall_pct": wall_overhead,
            "generated_pct": pct(b["generated"], a["generated"]),
            "sweep_removed": fnum(sum_b, "sweep_removed"),
            "sweep_count": fnum(sum_b, "sweep_count"),
            "sweep_ms": fnum(sum_b, "sweep_ms"),
        },
        "gate_sweep": {
            "pareto_identical_head": pareto_ok,
            "nqp_drop_ge_10pct": gate_nqp,
            "wall_overhead_le_8pct": gate_wall,
            "PASS": gate_pass,
            "keep_sweep": gate_pass,
        },
        "pending_diag": {
            "checkpoints": pending_rows,
            "R_pending_3M": R_at_3M,
            "verdict": verdict_pending,
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = f"""# NQP Stale Sweep A/B + Pending Dom — NY0001 @ 3M

## Status
**DIAGNOSTIC only — NOT EXACT.** No fragment. No full NY0001 rerun.

## Sweep A/B

| metric | A baseline | B sweep@{SWEEP_EVERY // 1000}k | Δ% |
|--------|----------:|-------------------------------:|---:|
| wall_sec | {a['wall_sec']} | {b['wall_sec']} | {wall_overhead} |
| generated | {a['generated']} | {b['generated']} | {pct(b['generated'], a['generated'])} |
| peak_NQP | {a['peak_nqp']} | {b['peak_nqp']} | {nqp_drop_peak} |
| nqp_live | {a['nqp_live_end']} | {b['nqp_live_end']} | {nqp_drop_live} |
| peak_OPEN | {a['peak_open']} | {b['peak_open']} | {pct(b['peak_open'], a['peak_open'])} |
| pareto_size | {a['pareto_size']} | {b['pareto_size']} | |
| sweep_removed | — | {sum_b.get('sweep_removed', '?')} | |
| sweep_count | — | {sum_b.get('sweep_count', '?')} | |
| sweep_ms | — | {sum_b.get('sweep_ms', '?')} | |

**Sweep Gate = {'PASS' if gate_pass else 'FAIL'}** (keep={gate_pass})
- Pareto head identical: {pareto_ok}
- NQP peak/live ↓ ≥10%: {gate_nqp}
- Wall overhead ≤8%: {gate_wall}

## Pending same-node dominance (fresh NQP)

| expanded | fresh_nqp | fresh_dom | R_pending | p50 | p90 | p99 | max |
|---------:|----------:|----------:|----------:|----:|----:|----:|----:|
"""
    for pr in pending_rows:
        md += (
            f"| {pr.get('expanded','?')} | {pr.get('fresh_nqp','?')} | "
            f"{pr.get('fresh_dom_same_node','?')} | {pr.get('R_pending','?')} | "
            f"{pr.get('p50_pending','?')} | {pr.get('p90_pending','?')} | "
            f"{pr.get('p99_pending','?')} | {pr.get('max_pending','?')} |\n"
        )
    md += f"""
**R_pending @ 3M = {R_at_3M}**

### Next-cut verdict
`{verdict_pending}`

- ≥40% → PendingSkyline is the breakthrough candidate
- ≥20% → worth implementing PendingSkyline[v]
- <10% → stop NQP-cleanup chasing; admit wide exact frontier
"""
    (OUT / "REPORT.md").write_text(md, encoding="utf-8")
    print("\n" + md, flush=True)
    print(f"[sweep-ab] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
