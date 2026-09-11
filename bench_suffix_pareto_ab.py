# -*- coding: utf-8 -*-
"""
NY0001 @ 3M expanded A/B: baseline CachedMeta vs target-side suffix Pareto.

Variants:
  A baseline: no suffix Pareto
  B diagnostic_forall: induced band P_B with exact flags (OPTIMISTIC / not
    formally exact on NY — every band node can leave band). If this fails the
    generated Gate, true complete P_t cannot pass either (weaker prune).
  C safe_join: inject g+p into S for induced suffixes (exact feasible); no ∀.

Does NOT write result2 fragments.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from config import ANALYSIS_DIR, queries_path
from corridor import build_corridor, lexmin_all_orders
from graph import load_dataset, load_queries
from multiobjective_exact import load_or_compute_heuristics, namoa_star_exact, update_skyline
from namoa_cpp import clear_suffix_pareto, ensure_dll, set_suffix_pareto
from seed_bank import build_seed_bank, weight_grid
from suffix_pareto import reverse_namoa_suffix, select_target_band

OUT = ANALYSIS_DIR / "problem2"
OUT.mkdir(parents=True, exist_ok=True)
MAX_EXP = 3_000_000
P1_MIN = 0.93


def make_seeds(g, s, t, hb):
    bank = build_seed_bank(g, [s], t, hb, k=10)
    seeds = list(bank[s])
    for y in lexmin_all_orders(g, s - 1, t - 1):
        update_skyline(seeds, y)
    return seeds


def run_one(tag: str, g, s, t, hb, seeds, edge_mask, sp_cache=None, join=False):
    clear_suffix_pareto()
    if sp_cache is not None:
        set_suffix_pareto(
            sp_cache.off, sp_cache.c0, sp_cache.c1, sp_cache.c2, sp_cache.exact,
            enable_join=join,
        )
    t0 = time.time()
    res = namoa_star_exact(
        g, s, t, heuristics=hb, use_heuristic=True,
        extra_seeds=seeds, edge_allowed=edge_mask,
        max_labels=50_000_000, max_expanded=MAX_EXP,
        suffix_q=None, d_s=None,
    )
    wall = time.time() - t0
    clear_suffix_pareto()
    st = res.stats
    return {
        "tag": tag,
        "generated": st.labels_generated,
        "expanded": st.labels_expanded,
        "live": st.labels_live or st.labels_peak,
        "peak_live": st.labels_peak,
        "max_node": st.max_labels_at_one_node,
        "node_prune": st.node_dominance_pruned,
        "goal_prune": st.goal_bound_pruned,
        "pareto_size": st.pareto_size,
        "sp_prune_hits": int(st.sp_prune_hits),
        "sp_join_updates": int(st.sp_join_updates),
        "search_wall": round(wall, 2),
        "exact_finished": st.exact_finished,
        "early_stop": st.early_stop,
    }


def main():
    ensure_dll()
    g = load_dataset("NY")
    q = load_queries(queries_path("NY", "problem2")).iloc[0]
    s, t = int(q["source"]), int(q["target"])
    print(f"NY0001 {s}->{t} stop@{MAX_EXP} p1_min={P1_MIN}", flush=True)

    hb = load_or_compute_heuristics(g, "NY", t, use_cache=True)
    seeds = make_seeds(g, s, t, hb)
    corr = build_corridor(g, s - 1, hb, seeds)
    edge_mask = corr.edge_allowed.copy()
    band = select_target_band(
        corr.d_s, hb.h, p1_min=P1_MIN, source_0=s - 1, target_0=t - 1
    )
    print(f"seeds={len(seeds)} band={int(band.sum())} edge_keep={corr.n_edges_kept}", flush=True)

    t_sp = time.time()
    sp = reverse_namoa_suffix(
        g, t - 1,
        edge_allowed=edge_mask,
        band=band,
        d_s=corr.d_s,
        incumbents=seeds,
        max_labels=5_000_000,
        store_band_only=True,
        expand_band_only=True,
        certify_induced=True,  # diagnostic: optimistic exact flags
    )
    sp_wall = time.time() - t_sp
    print("suffix reverse:", sp.stats, f"wall={sp_wall:.2f}", flush=True)

    rows = []
    # A
    print("\n===== A baseline =====", flush=True)
    rows.append(run_one("A_baseline", g, s, t, hb, seeds, edge_mask))
    print(rows[-1], flush=True)

    # B diagnostic forall
    print("\n===== B diagnostic forall (induced P_B, NOT formally exact) =====", flush=True)
    rows.append(run_one("B_diag_forall", g, s, t, hb, seeds, edge_mask, sp_cache=sp, join=False))
    print(rows[-1], flush=True)

    # C safe join only
    sp_join = reverse_namoa_suffix(
        g, t - 1,
        edge_allowed=edge_mask,
        band=band,
        d_s=corr.d_s,
        incumbents=seeds,
        max_labels=5_000_000,
        store_band_only=True,
        expand_band_only=True,
        certify_induced=False,  # exact flags off
    )
    # still need frontiers packed — re-run with certify to get pts but zero exact
    # Actually certify_induced=False leaves empty pack. Use sp but zero exact:
    import numpy as np
    sp_join_cache = sp
    sp_join_cache.exact = np.zeros_like(sp.exact)
    print("\n===== C safe join only =====", flush=True)
    rows.append(
        run_one("C_safe_join", g, s, t, hb, seeds, edge_mask, sp_cache=sp_join_cache, join=True)
    )
    print(rows[-1], flush=True)

    # Pull sp hits from a one-off by re-reading — wire via namoa stats in multiobjective
    # Patch: re-run B quickly not needed; document from C++ stderr if any.

    base = rows[0]
    def pct(a, b):
        return round(100.0 * (a - b) / b, 2) if b else None

    summary = {
        "query": "NY_0001",
        "max_expanded": MAX_EXP,
        "p1_min": P1_MIN,
        "suffix_preprocess": sp.stats,
        "suffix_preprocess_wall": round(sp_wall, 2),
        "rows": rows,
        "gate": {
            "B_generated_delta_pct": pct(rows[1]["generated"], base["generated"]),
            "C_generated_delta_pct": pct(rows[2]["generated"], base["generated"]),
            "B_live_delta_pct": pct(rows[1]["live"], base["live"]),
            "C_live_delta_pct": pct(rows[2]["live"], base["live"]),
            "B_max_node_delta_pct": pct(rows[1]["max_node"], base["max_node"]),
            "pass_15pct_generated": bool(
                rows[1]["generated"] <= 0.85 * base["generated"]
            ),
            "note": (
                "B uses induced-band P_B with optimistic exact flags; "
                "NY band nodes can all reach outside, so B is NOT submission-safe. "
                "If B fails Gate, true complete P_t cannot pass either."
            ),
        },
    }

    out_json = OUT / "ab_suffix_pareto_NY_0001_3M.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    out_csv = OUT / "ab_suffix_pareto_NY_0001_3M.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md = OUT / "ab_suffix_pareto_gate.md"
    gpass = summary["gate"]["pass_15pct_generated"]
    md.write_text(
        f"""# Target-side Suffix Pareto Gate — NY0001 @ 3M

## Verdict
**Gate = {'PASS' if gpass else 'FAIL'}** (need generated ↓ ≥15% on searchable exact prune)

## Preprocess
- band p1≥{P1_MIN}: {int(band.sum())} nodes
- reverse induced: gen={sp.stats.get('generated')} complete_open={sp.stats.get('raw_open_empty')}
- max_front={sp.stats.get('max_front')} avg_front={sp.stats.get('avg_front')}
- preprocess wall={sp_wall:.2f}s

## Results
| tag | generated | live | max_node | goal_prune | wall |
|-----|-----------|------|----------|------------|------|
| A baseline | {rows[0]['generated']} | {rows[0]['live']} | {rows[0]['max_node']} | {rows[0]['goal_prune']} | {rows[0]['search_wall']} |
| B diag ∀ | {rows[1]['generated']} | {rows[1]['live']} | {rows[1]['max_node']} | {rows[1]['goal_prune']} | {rows[1]['search_wall']} |
| C safe join | {rows[2]['generated']} | {rows[2]['live']} | {rows[2]['max_node']} | {rows[2]['goal_prune']} | {rows[2]['search_wall']} |

## Deltas vs A
- B generated: {summary['gate']['B_generated_delta_pct']}%
- C generated: {summary['gate']['C_generated_delta_pct']}%

## Exactness
- Full residual reverse NAMOA* does **not** finish within 2–5M labels (same explosion as forward).
- Induced band reverse finishes easily, but **all** band nodes can reach outside the band on the residual → induced P_B is **incomplete** → ∀ prune with P_B is **not** submission-exact.
- B is diagnostic only. C (join) is exact but is an incumbent-strengthening method, not the ∀ LB.

## Decision
{'Worth pursuing full P_t / absorbing construction.' if gpass else 'STOP full NY0001 re-run. Seek other exact LB / bidirectional.'}
""",
        encoding="utf-8",
    )
    print("\nWrote", out_json, out_csv, md, flush=True)
    print("GATE", "PASS" if gpass else "FAIL", summary["gate"], flush=True)


if __name__ == "__main__":
    main()
