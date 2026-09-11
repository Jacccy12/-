# -*- coding: utf-8 -*-
"""
NY0001 @ 3M expansions: CachedMeta vs dr-lazy vs T-MDA.

T-MDA Gate: peak OPEN ≪ |V| scale (not millions); live/RSS controllable.
Does not write result2 fragments. No full NY0001 unless Gate PASS.
"""
from __future__ import annotations

import csv
import json
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
from namoa_cpp import ensure_dll as ensure_cached_dll
from namoa_dr_lazy import ensure_dll as ensure_dr_dll
from seed_bank import build_seed_bank
from t_mda import ensure_dll as ensure_tmda_dll

OUT = ANALYSIS_DIR / "problem2"
OUT.mkdir(parents=True, exist_ok=True)
MAX_EXP = 3_000_000


def make_seeds(g, s, t, hb):
    bank = build_seed_bank(g, [s], t, hb, k=10)
    seeds = list(bank[s])
    for y in lexmin_all_orders(g, s - 1, t - 1):
        update_skyline(seeds, y)
    return seeds


def run(tag: str, backend: str, g, s, t, hb, seeds, edge_mask):
    proc = psutil.Process() if psutil else None
    rss0 = proc.memory_info().rss if proc else 0
    t0 = time.time()
    res = namoa_star_exact(
        g, s, t, heuristics=hb, use_heuristic=True,
        extra_seeds=seeds, edge_allowed=edge_mask,
        max_labels=80_000_000, max_expanded=MAX_EXP,
        backend=backend, suffix_q=None, d_s=None,
    )
    wall = time.time() - t0
    rss1 = proc.memory_info().rss if proc else 0
    st = res.stats
    return {
        "tag": tag,
        "backend": backend,
        "generated": st.labels_generated,
        "expanded": st.labels_expanded,
        "peak_live": st.labels_peak,
        "peak_open": st.open_peak,
        "max_perm_front": st.max_labels_at_one_node,
        "peak_nqp": st.u_bound_size if backend == "t_mda" else None,
        "nqp_live_end": st.stale_skipped if backend == "t_mda" else None,
        "prop_or_node_prune": st.node_dominance_pruned,
        "pareto_size": st.pareto_size,
        "search_wall": round(wall, 2),
        "rss_end_mb": round(rss1 / (1024 * 1024), 1) if proc else None,
        "rss_delta_mb": round((rss1 - rss0) / (1024 * 1024), 1) if proc else None,
        "exact_finished": st.exact_finished,
        "early_stop": st.early_stop,
        "mode": st.mode,
    }


def pct(new, old):
    if not old:
        return None
    return round(100.0 * (new - old) / old, 2)


def main():
    ensure_cached_dll()
    ensure_dr_dll()
    ensure_tmda_dll()
    g = load_dataset("NY")
    q = load_queries(queries_path("NY", "problem2")).iloc[0]
    s, t = int(q["source"]), int(q["target"])
    print(f"NY0001 {s}->{t} n={g.n_nodes} stop@{MAX_EXP}", flush=True)

    hb = load_or_compute_heuristics(g, "NY", t, use_cache=True)
    seeds = make_seeds(g, s, t, hb)
    corr = build_corridor(g, s - 1, hb, seeds)
    edge_mask = corr.edge_allowed.copy()
    print(f"seeds={len(seeds)} edge_keep={corr.n_edges_kept}", flush=True)

    rows = []
    for tag, be in [
        ("A_cached", "cpp"),
        ("B_dr_lazy", "dr_lazy"),
        ("C_t_mda", "t_mda"),
    ]:
        print(f"\n===== {tag} ({be}) =====", flush=True)
        row = run(tag, be, g, s, t, hb, seeds, edge_mask)
        print(row, flush=True)
        rows.append(row)

    a, b, c = rows
    # Gate: peak OPEN at most ~|V| and ≪ cached / dr-lazy; ideally ≤ 2*|V|
    open_ok = c["peak_open"] <= 2 * g.n_nodes
    open_vs_cached = c["peak_open"] <= 0.5 * a["peak_open"]
    live_ok = c["peak_live"] <= 0.7 * a["peak_live"] or open_ok
    gate_pass = bool(open_ok and (open_vs_cached or live_ok))

    summary = {
        "query": "NY_0001",
        "n_nodes": g.n_nodes,
        "max_expanded": MAX_EXP,
        "rows": rows,
        "delta_C_vs_A": {
            "generated": pct(c["generated"], a["generated"]),
            "peak_live": pct(c["peak_live"], a["peak_live"]),
            "peak_open": pct(c["peak_open"], a["peak_open"]),
            "wall": pct(c["search_wall"], a["search_wall"]),
        },
        "gate": {
            "criterion": "peak_OPEN ≤ 2|V| AND (≪ CachedMeta or live↓)",
            "peak_open": c["peak_open"],
            "open_le_2n": open_ok,
            "open_vs_cached_half": open_vs_cached,
            "PASS": gate_pass,
        },
    }

    out_json = OUT / "ab_t_mda_NY_0001_3M.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    out_csv = OUT / "ab_t_mda_NY_0001_3M.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(a.keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    md = OUT / "t_mda_gate.md"
    md.write_text(
        f"""# T-MDA Gate — NY0001 @ 3M

## Verdict
**Gate = {'PASS' if gate_pass else 'FAIL'}**

Criterion: `peak_OPEN ≤ 2|V|` ({2 * g.n_nodes}) and clear advantage vs CachedMeta OPEN/live.

## Results
| metric | CachedMeta | dr-lazy | T-MDA |
|--------|----------:|--------:|------:|
| generated | {a['generated']} | {b['generated']} | {c['generated']} |
| peak_live | {a['peak_live']} | {b['peak_live']} | {c['peak_live']} |
| peak_OPEN | {a['peak_open']} | {b['peak_open']} | **{c['peak_open']}** |
| max_perm_front | {a['max_perm_front']} | {b['max_perm_front']} | {c['max_perm_front']} |
| peak_NQP | — | — | {c['peak_nqp']} |
| wall_s | {a['search_wall']} | {b['search_wall']} | {c['search_wall']} |
| RSS_MB | {a['rss_end_mb']} | {b['rss_end_mb']} | {c['rss_end_mb']} |
| |S| | {a['pareto_size']} | {b['pareto_size']} | {c['pareto_size']} |

## Deltas T-MDA vs CachedMeta
- generated: {summary['delta_C_vs_A']['generated']}%
- peak_live: {summary['delta_C_vs_A']['peak_live']}%
- peak_OPEN: {summary['delta_C_vs_A']['peak_open']}%
- wall: {summary['delta_C_vs_A']['wall']}%

## Decision
{'Proceed to full NY0001 EXACT (Q empty).' if gate_pass else 'Analyze NQP/live; do not fall back to NAMOA* LB/corridor patches.'}
""",
        encoding="utf-8",
    )
    print("\nWrote", out_json, md, flush=True)
    print("GATE", "PASS" if gate_pass else "FAIL", summary["gate"], flush=True)


if __name__ == "__main__":
    main()
