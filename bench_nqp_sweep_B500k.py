# -*- coding: utf-8 -*-
"""Retune B: interval-only sweep every 500k; merge into existing A baseline report."""
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

from bench_nqp_sweep_ab import (
    MAX_EXP,
    OUT,
    make_seeds,
    pct,
    read_pending,
    read_summary,
)
from config import queries_path
from corridor import build_corridor
from graph import load_dataset, load_queries
from multiobjective_exact import load_or_compute_heuristics, namoa_star_exact
from t_mda import clear_profile_dir, ensure_dll, set_nqp_sweep, set_profile_dir


def main():
    ensure_dll()
    OUT.mkdir(parents=True, exist_ok=True)
    g = load_dataset("NY")
    q = load_queries(queries_path("NY", "problem2")).iloc[0]
    s, t = int(q["source"]), int(q["target"])
    hb = load_or_compute_heuristics(g, "NY", t, use_cache=True)
    seeds = make_seeds(g, s, t, hb)
    corr = build_corridor(g, s - 1, hb, seeds)
    edge_mask = corr.edge_allowed.copy()

    tmp = Path(tempfile.gettempdir()) / "tmda_nqp_sweep_B500k"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()

    set_profile_dir(str(tmp))
    set_nqp_sweep(True, 500_000)
    proc = psutil.Process() if psutil else None
    rss0 = proc.memory_info().rss if proc else 0
    t0 = time.time()
    try:
        res = namoa_star_exact(
            g, s, t, heuristics=hb, use_heuristic=True,
            extra_seeds=seeds, edge_allowed=edge_mask.copy(),
            max_labels=80_000_000, max_expanded=MAX_EXP, backend="t_mda",
        )
    finally:
        set_nqp_sweep(False)
        clear_profile_dir()
    wall = time.time() - t0
    rss1 = proc.memory_info().rss if proc else 0
    st = res.stats
    summary = read_summary(tmp)
    pending = read_pending(tmp)

    dest = OUT / "B_sweep_500k"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(tmp, dest)

    print(
        f"wall={wall:.2f} gen={st.labels_generated} peak_nqp={st.u_bound_size} "
        f"live={st.stale_skipped} pareto={st.pareto_size} "
        f"sweep_removed={summary.get('sweep_removed')} "
        f"sweep_ms={summary.get('sweep_ms')}",
        flush=True,
    )

    old = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    a = next(r for r in old["rows"] if r["tag"] == "A_baseline")
    prior_b = next((r for r in old["rows"] if r["tag"] == "B_sweep"), None)
    b = {
        "tag": "B_sweep_500k",
        "sweep": True,
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

    peak_a, peak_b = a["peak_nqp"] or 0, b["peak_nqp"] or 0
    live_a, live_b = a["nqp_live_end"] or 0, b["nqp_live_end"] or 0
    nqp_drop_peak, nqp_drop_live = pct(peak_b, peak_a), pct(live_b, live_a)
    wall_overhead = pct(b["wall_sec"], a["wall_sec"])
    pareto_ok = a["pareto_size"] == b["pareto_size"] and a["pareto_head"] == b["pareto_head"]
    gate_nqp = (nqp_drop_peak is not None and nqp_drop_peak <= -10.0) or (
        nqp_drop_live is not None and nqp_drop_live <= -10.0
    )
    gate_wall = wall_overhead is None or wall_overhead <= 8.0
    gate_pass = bool(pareto_ok and gate_nqp and gate_wall)

    R = None
    for pr in pending:
        if pr.get("expanded") == "3000000":
            R = float(pr["R_pending"])
    verdict = (
        "STOP_NQP_CLEANUP_WIDE_FRONTIER"
        if (R is not None and R < 0.10)
        else "CHECK"
    )

    report = {
        "query": "NY_0001",
        "max_expanded": MAX_EXP,
        "sweep_every": 500000,
        "note": "B retuned: interval-only every 500k. A from prior run.",
        "rows": [a, b],
        "prior_B_250k_growth": prior_b,
        "delta_B_vs_A": {
            "peak_nqp_pct": nqp_drop_peak,
            "nqp_live_pct": nqp_drop_live,
            "wall_pct": wall_overhead,
            "generated_pct": pct(b["generated"], a["generated"]),
            "sweep_removed": float(summary.get("sweep_removed", 0) or 0),
            "sweep_count": float(summary.get("sweep_count", 0) or 0),
            "sweep_ms": float(summary.get("sweep_ms", 0) or 0),
        },
        "gate_sweep": {
            "pareto_identical_head": pareto_ok,
            "nqp_drop_ge_10pct": gate_nqp,
            "wall_overhead_le_8pct": gate_wall,
            "PASS": gate_pass,
            "keep_sweep": gate_pass,
        },
        "pending_diag": {
            "checkpoints": pending,
            "R_pending_3M": R,
            "verdict": verdict,
        },
        "prior_gate_250k_growth": old.get("gate_sweep"),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# NQP Stale Sweep A/B + Pending Dom — NY0001 @ 3M",
        "",
        "## Status",
        "**DIAGNOSTIC only — NOT EXACT.** No fragment. No full NY0001 rerun.",
        "",
        "## Sweep A/B (B = interval-only @ 500k)",
        "",
        "| metric | A baseline | B sweep@500k | Δ% |",
        "|--------|----------:|-------------:|---:|",
        f"| wall_sec | {a['wall_sec']} | {b['wall_sec']} | {wall_overhead} |",
        f"| generated | {a['generated']} | {b['generated']} | {pct(b['generated'], a['generated'])} |",
        f"| peak_NQP | {a['peak_nqp']} | {b['peak_nqp']} | {nqp_drop_peak} |",
        f"| nqp_live | {a['nqp_live_end']} | {b['nqp_live_end']} | {nqp_drop_live} |",
        f"| peak_OPEN | {a['peak_open']} | {b['peak_open']} | {pct(b['peak_open'], a['peak_open'])} |",
        f"| pareto_size | {a['pareto_size']} | {b['pareto_size']} | |",
        f"| sweep_removed | — | {summary.get('sweep_removed', '?')} | |",
        f"| sweep_count | — | {summary.get('sweep_count', '?')} | |",
        f"| sweep_ms | — | {summary.get('sweep_ms', '?')} | |",
        "",
        f"**Sweep Gate = {'PASS' if gate_pass else 'FAIL'}** (keep={gate_pass})",
        f"- Pareto identical: {pareto_ok}",
        f"- NQP peak/live drop >=10%: {gate_nqp}",
        f"- Wall overhead <=8%: {gate_wall}",
        "",
        "Prior B (250k + growth): wall +37.9%, 30 sweeps / 22.6s — FAIL wall.",
        "",
        "## Pending same-node dominance (fresh NQP)",
        "",
        "| expanded | fresh_nqp | fresh_dom | R_pending | p50 | p90 | p99 | max |",
        "|---------:|----------:|----------:|----------:|----:|----:|----:|----:|",
    ]
    for pr in pending:
        lines.append(
            f"| {pr.get('expanded')} | {pr.get('fresh_nqp')} | "
            f"{pr.get('fresh_dom_same_node')} | {pr.get('R_pending')} | "
            f"{pr.get('p50_pending')} | {pr.get('p90_pending')} | "
            f"{pr.get('p99_pending')} | {pr.get('max_pending')} |"
        )
    lines += [
        "",
        f"**R_pending @ 3M = {R}**",
        "",
        "### Next-cut verdict",
        f"`{verdict}`",
        "",
        "Fresh NQP same-node pending dominance is only ~5%. The bulk of fresh NQP",
        "is mutually non-dominated across in-arcs. Do **not** implement",
        "PendingSkyline as the next bet. Admit NY0001 has a wide exact",
        "multiobjective frontier; further NQP-cleanup patches have low expected return.",
        "",
    ]
    md = "\n".join(lines)
    (OUT / "REPORT.md").write_text(md, encoding="utf-8")
    print(md, flush=True)


if __name__ == "__main__":
    main()
