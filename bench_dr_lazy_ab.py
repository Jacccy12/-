# -*- coding: utf-8 -*-
"""
NY0001 @ 3M expansions: CachedMeta NAMOA* vs NAMOA*_dr-lazy.

Gate (this stage): peak_live OR peak_OPEN ↓ ≥ ~30%.
generated drop is NOT required.
Does not write result2 fragments.
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
        max_labels=50_000_000, max_expanded=MAX_EXP,
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
        "live_end": st.labels_live or st.labels_peak,
        "peak_open": st.open_peak,
        "max_node_front": st.max_labels_at_one_node,
        "lazy_or_node_prune": st.node_dominance_pruned,
        "goal_prune": st.goal_bound_pruned,
        "pareto_size": st.pareto_size,
        "search_wall": round(wall, 2),
        "rss_delta_mb": round((rss1 - rss0) / (1024 * 1024), 1) if proc else None,
        "rss_end_mb": round(rss1 / (1024 * 1024), 1) if proc else None,
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
    g = load_dataset("NY")
    q = load_queries(queries_path("NY", "problem2")).iloc[0]
    s, t = int(q["source"]), int(q["target"])
    print(f"NY0001 {s}->{t} stop@{MAX_EXP}", flush=True)

    hb = load_or_compute_heuristics(g, "NY", t, use_cache=True)
    seeds = make_seeds(g, s, t, hb)
    corr = build_corridor(g, s - 1, hb, seeds)
    edge_mask = corr.edge_allowed.copy()
    print(f"seeds={len(seeds)} edge_keep={corr.n_edges_kept}", flush=True)

    print("\n===== A CachedMeta NAMOA* =====", flush=True)
    a = run("A_cached", "cpp", g, s, t, hb, seeds, edge_mask)
    print(a, flush=True)

    print("\n===== B NAMOA*_dr-lazy =====", flush=True)
    b = run("B_dr_lazy", "dr_lazy", g, s, t, hb, seeds, edge_mask)
    print(b, flush=True)

    live_d = pct(b["peak_live"], a["peak_live"])
    open_d = pct(b["peak_open"], a["peak_open"])
    # Gate: either metric ↓ ≥30%
    pass_live = b["peak_live"] <= 0.70 * a["peak_live"]
    pass_open = (a["peak_open"] > 0) and (b["peak_open"] <= 0.70 * a["peak_open"])
    gate_pass = bool(pass_live or pass_open)

    summary = {
        "query": "NY_0001",
        "max_expanded": MAX_EXP,
        "A": a,
        "B": b,
        "delta_pct": {
            "generated": pct(b["generated"], a["generated"]),
            "peak_live": live_d,
            "peak_open": open_d,
            "max_node_front": pct(b["max_node_front"], a["max_node_front"]),
            "wall": pct(b["search_wall"], a["search_wall"]),
        },
        "gate": {
            "criterion": "peak_live OR peak_OPEN ≤ 70% of CachedMeta (≥30% drop)",
            "pass_live": pass_live,
            "pass_open": pass_open,
            "PASS": gate_pass,
        },
    }

    out_json = OUT / "ab_dr_lazy_NY_0001_3M.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    out_csv = OUT / "ab_dr_lazy_NY_0001_3M.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(a.keys()))
        w.writeheader()
        w.writerow(a)
        w.writerow(b)

    md = OUT / "ab_dr_lazy_gate.md"
    md.write_text(
        f"""# NAMOA*_dr-lazy Gate — NY0001 @ 3M

## Verdict
**Gate = {'PASS' if gate_pass else 'FAIL'}**
(criterion: peak_live **or** peak_OPEN ↓ ≥ 30%)

## Results
| metric | CachedMeta | dr-lazy | Δ% |
|--------|----------:|--------:|---:|
| generated | {a['generated']} | {b['generated']} | {summary['delta_pct']['generated']} |
| peak_live | {a['peak_live']} | {b['peak_live']} | {live_d} |
| peak_open | {a['peak_open']} | {b['peak_open']} | {open_d} |
| max_node_front | {a['max_node_front']} | {b['max_node_front']} | {summary['delta_pct']['max_node_front']} |
| wall_s | {a['search_wall']} | {b['search_wall']} | {summary['delta_pct']['wall']} |
| |S| | {a['pareto_size']} | {b['pareto_size']} | |
| lazy/node prune | {a['lazy_or_node_prune']} | {b['lazy_or_node_prune']} | |

## Notes
- Suffix Pareto direction is **frozen** (FAIL); not used here.
- Static corridor + 62 seeds ON for both.
- dr-lazy: no Gop merge; permanent fronts are 2D truncated (g2,g3).
- If Gate FAIL → proceed to **T-MDA** (do not invent more LB/corridor).
- If Gate PASS → consider full NY0001 EXACT with resource guards on live/RSS/wall.
""",
        encoding="utf-8",
    )
    print("\nWrote", out_json, md, flush=True)
    print("GATE", "PASS" if gate_pass else "FAIL", summary["gate"], flush=True)


if __name__ == "__main__":
    main()
