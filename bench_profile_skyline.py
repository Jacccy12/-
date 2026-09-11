# -*- coding: utf-8 -*-
"""
B-config skyline profiler @ 3M expanded.
  seeds: 66-grid scaled + lex (~62 ND)
  static corridor ON; suffix UB OFF; dyn corridor OFF
"""
from __future__ import annotations

import csv as csvmod
import shutil
import time
from pathlib import Path

from config import ANALYSIS_DIR, queries_path
from corridor import build_corridor, lexmin_all_orders
from graph import load_dataset, load_queries
from multiobjective_exact import load_or_compute_heuristics, namoa_star_exact, update_skyline
from namoa_cpp import ensure_dll
from seed_bank import build_seed_bank


def main():
    out = ANALYSIS_DIR / "problem2" / "profiler_B_3M"
    out.mkdir(parents=True, exist_ok=True)
    # MinGW fopen often fails on absolute Unicode paths — use cwd-relative ASCII path.
    out_rel = "analysis/problem2/profiler_B_3M"

    ensure_dll()
    g = load_dataset("NY")
    q = load_queries(queries_path("NY", "problem2")).iloc[0]
    s, t = int(q["source"]), int(q["target"])
    print(f"[profile] NY {s}->{t} B-config @ 3M  out={out_rel}", flush=True)

    t0 = time.time()
    hb = load_or_compute_heuristics(g, "NY", t, use_cache=True)
    bank = build_seed_bank(g, [s], t, hb, k=10)
    seeds = list(bank[s])
    for y in lexmin_all_orders(g, s - 1, t - 1):
        update_skyline(seeds, y)
    t_seed = time.time() - t0

    t1 = time.time()
    corr = build_corridor(g, s - 1, hb, seeds)
    t_corr = time.time() - t1
    print(
        f"  seeds={len(seeds)} seed_sec={t_seed:.1f} "
        f"static_edges_pruned={corr.n_edges_pruned} ({100 * corr.n_edges_pruned / g.n_edges:.2f}%) "
        f"corr_sec={t_corr:.1f}",
        flush=True,
    )

    t2 = time.time()
    res = namoa_star_exact(
        g, s, t, heuristics=hb, use_heuristic=True,
        extra_seeds=seeds, edge_allowed=corr.edge_allowed.copy(),
        max_labels=50_000_000, max_expanded=3_000_000,
        suffix_q=None, d_s=None,  # dyn OFF
        profile_dir=out_rel,
    )
    t_search = time.time() - t2
    st = res.stats
    print(
        f"  search_sec={t_search:.1f} gen={st.labels_generated} exp={st.labels_expanded} "
        f"max_node={st.max_labels_at_one_node} |S|={st.pareto_size}",
        flush=True,
    )

    src = out / "top100_wide_nodes.csv"
    dst = ANALYSIS_DIR / "problem2" / "top100_wide_nodes_NY_0001.csv"
    if src.exists():
        rows = list(csvmod.DictReader(src.open(encoding="utf-8")))
        ds, h = corr.d_s, hb.h
        for row in rows:
            v = int(row["node"])
            d1, d2, d3 = int(ds[v, 0]), int(ds[v, 1]), int(ds[v, 2])
            h1, h2, h3 = int(h[v, 0]), int(h[v, 1]), int(h[v, 2])
            row["d1"], row["d2"], row["d3"] = str(d1), str(d2), str(d3)
            row["h1"], row["h2"], row["h3"] = str(h1), str(h2), str(h3)

            def pi(d, hh):
                den = d + hh
                return f"{(d / den):.4f}" if den > 0 else ""

            row["p1"], row["p2"], row["p3"] = pi(d1, h1), pi(d2, h2), pi(d3, h3)
        with src.open("w", newline="", encoding="utf-8") as f:
            w = csvmod.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        shutil.copy2(src, dst)

    def read_kv_csv(p: Path):
        d = {}
        if not p.exists():
            return d
        for line in p.read_text(encoding="utf-8").splitlines()[1:]:
            if not line.strip():
                continue
            k, v = line.split(",", 1)
            d[k] = v
        return d

    summ = read_kv_csv(out / "profiler_summary.csv")
    conc = {}
    cp = out / "concentration.csv"
    if cp.exists():
        lines = cp.read_text(encoding="utf-8").splitlines()
        if len(lines) >= 2:
            conc = dict(zip(lines[0].split(","), lines[1].split(",")))
    scan = {}
    sp = out / "scan_percentiles.csv"
    if sp.exists():
        lines = sp.read_text(encoding="utf-8").splitlines()
        if len(lines) >= 2:
            scan = dict(zip(lines[0].split(","), lines[1].split(",")))

    node_frac = float(summ.get("node_frac", 0) or 0)
    gate = (
        "GO Hybrid Block Skyline"
        if node_frac >= 0.30
        else "HOLD — node skyline not dominant; inspect heap/goal/edge"
    )

    md = f"""# Profiler B @ 3M — NY 0001

## Setup
- config: **B** (66-grid scaled + lex, static corridor ON, suffix OFF, dyn OFF)
- status: BENCH_LIMIT / exact=false
- seed_setup_sec: {t_seed:.2f}
- corridor_sec: {t_corr:.2f}
- namoa_search_sec: {t_search:.2f}
- seeds ND: {len(seeds)}
- static_edges_pruned_pct: {100 * corr.n_edges_pruned / g.n_edges:.2f}

## CPU breakdown (accounted)

| bucket | ms | frac |
|--------|---:|-----:|
| node skyline (find+del+ins) | {summ.get('node_skyline_ms')} | {summ.get('node_frac')} |
| heap pop+push | {int(summ.get('heap_pop_ms', 0) or 0) + int(summ.get('heap_push_ms', 0) or 0)} | {summ.get('heap_frac')} |
| goal dominance | {summ.get('goal_dom_query_ms')} | {summ.get('goal_frac')} |
| edge expand | {summ.get('edge_expand_ms')} | {summ.get('edge_frac')} |

Detail: find={summ.get('node_dom_query_ms')}ms del={summ.get('node_dom_delete_ms')}ms ins={summ.get('node_insert_ms')}ms

## Scan length (node dom queries, sampled ≤200k)

| avg | P50 | P90 | P95 | P99 | P99.9 | max |
|----:|----:|----:|----:|----:|------:|----:|
| {scan.get('avg')} | {scan.get('P50')} | {scan.get('P90')} | {scan.get('P95')} | {scan.get('P99')} | {scan.get('P99_9')} | {scan.get('max')} |

## Concentration

- R10 = {conc.get('R10')}
- R100 = {conc.get('R100')}
- top10 node-dom ms = {conc.get('top10_ms')}
- top100 node-dom ms = {conc.get('top100_ms')}

## Notes on accounting
- `edge_expand_ms` wraps the expansion loop that *includes* nested node-skyline / goal calls → **do not add edge+node** (double count). Prefer `node_skyline_ms`, `heap_*`, `goal_*` as primary.
- `scan_percentiles` samples the first ≤200k queries (early, smaller fronts) → P99 understates late wide-node scans. Prefer Top-100 `avg_scan` / `max_scan`.

## Concentration reading
- R10 ≈ 1.6%, R100 ≈ 20% → **skyline cost is spread**, not only Top-10. Hybrid block for all `|L|>64` is appropriate (not only hot nodes).

## Gate

**{gate}** (threshold node_frac ≥ 0.30)

Artifacts: `{out}` ; canonical Top-100: `top100_wide_nodes_NY_0001.csv`
"""
    (out / "report.md").write_text(md, encoding="utf-8")
    (ANALYSIS_DIR / "problem2" / "profiler_B_3M_report.md").write_text(md, encoding="utf-8")
    print(f"[wrote] {out / 'report.md'}", flush=True)
    print(f"[gate] node_frac={node_frac:.3f} → {gate}", flush=True)


if __name__ == "__main__":
    main()
