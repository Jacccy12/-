# -*- coding: utf-8 -*-
"""
Controlled A/B/C/D ablation @ expanded=3e6 — sequential only.
Same namoa_core_v3.dll; feature flags via suffix_q / d_s / seed density.
status=BENCH_LIMIT, exact=false. Never writes result2 fragments.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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
from namoa_cpp import ensure_dll
from seed_bank import build_seed_bank, suffix_cost3_from_trees, weight_grid


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def make_seeds(g, s, t, hb, k_grid: int, use_scales: bool = True):
    """
    k_grid<=0: anchors + lex only.
    k_grid>0: weighted simplex (+ optional median scales) + lex.
    For A we use k=5 without scales to match v2 ~8 ND seeds.
    """
    t0 = time.time()
    if k_grid <= 0:
        from multiobjective_exact import seed_anchor_solutions
        seeds = seed_anchor_solutions(g, s - 1, hb)
        n_weighted_nd = len(seeds)
        n_grid = 0
        for y in lexmin_all_orders(g, s - 1, t - 1):
            update_skyline(seeds, y)
        return seeds, n_weighted_nd, n_grid, time.time() - t0

    # patch scales off via temporary monkey on objective_scales
    import seed_bank as sb
    scales_fn = sb.objective_scales
    if not use_scales:
        sb.objective_scales = lambda graph: (1, 1, 1)
    try:
        bank = build_seed_bank(g, [s], t, hb, k=k_grid)
    finally:
        sb.objective_scales = scales_fn
    seeds = list(bank[s])
    n_weighted_nd = len(seeds)
    for y in lexmin_all_orders(g, s - 1, t - 1):
        update_skyline(seeds, y)
    return seeds, n_weighted_nd, len(weight_grid(k_grid)), time.time() - t0


def run_variant(tag: str, k_grid: int, use_suffix: bool, use_dyn: bool, max_expanded: int,
                use_scales: bool = True):
    g = load_dataset("NY")
    q = load_queries(queries_path("NY", "problem2")).iloc[0]
    s, t = int(q["source"]), int(q["target"])
    print(
        f"\n===== {tag}: k={k_grid} scales={int(use_scales)} suffix={int(use_suffix)} "
        f"dyn={int(use_dyn)} stop@{max_expanded} =====",
        flush=True,
    )

    hb = load_or_compute_heuristics(g, "NY", t, use_cache=True)
    seeds, n_weighted_nd, n_grid, t_seed = make_seeds(
        g, s, t, hb, k_grid, use_scales=use_scales
    )

    t1 = time.time()
    corr = build_corridor(g, s - 1, hb, seeds)
    t_corr = time.time() - t1

    # Always pass static corridor mask. Dyn refresh needs a mutable copy + d_s.
    edge_mask = corr.edge_allowed.copy()
    suffix_q = suffix_cost3_from_trees(g, hb) if use_suffix else None
    d_s = corr.d_s if use_dyn else None

    proc = psutil.Process() if psutil else None
    peak_rss = proc.memory_info().rss if proc else 0

    t_search0 = time.time()
    res = namoa_star_exact(
        g, s, t, heuristics=hb, use_heuristic=True,
        extra_seeds=seeds, edge_allowed=edge_mask,
        max_labels=50_000_000, max_expanded=max_expanded,
        suffix_q=suffix_q, d_s=d_s,
    )
    wall = time.time() - t_search0
    if proc:
        peak_rss = max(peak_rss, proc.memory_info().rss)

    st = res.stats
    exp = max(st.labels_expanded, 1)
    live = st.labels_live or st.labels_peak
    row = {
        "tag": tag,
        "status": "BENCH_LIMIT",
        "exact": False,
        "k_grid": k_grid,
        "n_weight_grid": n_grid,
        "seed_build_sec": round(t_seed, 2),
        "seed_raw_count": n_grid,  # weighted attempts (grid size); ND after
        "seed_weighted_nd": n_weighted_nd,
        "seed_nd_count": len(seeds),
        "static_nodes_pruned": corr.n_nodes_pruned,
        "static_edges_pruned": corr.n_edges_pruned,
        "static_nodes_pruned_pct": round(100.0 * corr.n_nodes_pruned / g.n_nodes, 2),
        "static_edges_pruned_pct": round(100.0 * corr.n_edges_pruned / g.n_edges, 2),
        "suffix_ub": int(use_suffix),
        "dyn_corridor": int(use_dyn),
        "expanded": st.labels_expanded,
        "generated": st.labels_generated,
        "rg": round(st.labels_generated / exp, 4),
        "live": live,
        "peak_live": st.labels_peak,
        "rl": round(live / exp, 4),
        "rl_peak": round(st.labels_peak / exp, 4),
        "peak_open": st.open_peak,
        "node_prune": st.node_dominance_pruned,
        "goal_prune": st.goal_bound_pruned,
        "stale": st.stale_skipped,
        "dup_prune": st.duplicate_pruned,
        "current_S": st.pareto_size,
        "current_U_bound": st.u_bound_size,
        "max_node": st.max_labels_at_one_node,
        "corridor_refresh_count": st.corridor_refresh_count,
        "dynamic_edges_pruned": st.edges_killed_dyn,
        "wall_sec": round(wall, 2),
        "wall_sec_per_1M_exp": round(wall / (exp / 1e6), 2),
        "t_corr_sec": round(t_corr, 2),
        "peak_RSS_MB": round(peak_rss / (1024 * 1024), 1) if peak_rss else "",
        "dll": ensure_dll().name,
        "dll_sha16": file_sha256(ensure_dll()),
    }
    keys = (
        "seed_nd_count", "static_edges_pruned_pct", "generated", "rg", "live",
        "goal_prune", "current_U_bound", "max_node", "dynamic_edges_pruned",
        "wall_sec", "wall_sec_per_1M_exp",
    )
    print(" ", {k: row[k] for k in keys}, flush=True)
    return row


def write_md(rows, path: Path, dll_sha: str):
    by = {r["tag"]: r for r in rows}
    a, b, c, d = by.get("A"), by.get("B"), by.get("C"), by.get("D")

    def pct(new, old):
        if not old:
            return "n/a"
        return f"{100.0 * (new - old) / old:+.1f}%"

    lines = [
        "# NY 0001 A/B/C/D ablation @ 3M expanded",
        "",
        f"- DLL: `namoa_core_v3.dll` sha16=`{dll_sha}`",
        "- status: **BENCH_LIMIT** / exact=false (no result2 fragments)",
        "- static corridor: ON for all; blocked goal index + g1-sorted skyline: ON for all",
        "",
        "| metric | A | B | C | D |",
        "|---|---:|---:|---:|---:|",
    ]
    metrics = [
        ("seed_nd_count", "seed_nd"),
        ("static_edges_pruned_pct", "static_edge_prune_%"),
        ("generated", "generated"),
        ("rg", "rg"),
        ("live", "live"),
        ("peak_live", "peak_live"),
        ("peak_open", "peak_open"),
        ("node_prune", "node_prune"),
        ("goal_prune", "goal_prune"),
        ("stale", "stale"),
        ("current_S", "current_S"),
        ("current_U_bound", "current_U"),
        ("max_node", "max_node"),
        ("corridor_refresh_count", "refresh_count"),
        ("dynamic_edges_pruned", "dyn_edges_pruned"),
        ("wall_sec", "wall_sec"),
        ("wall_sec_per_1M_exp", "wall/1M"),
        ("peak_RSS_MB", "peak_RSS_MB"),
    ]
    for key, label in metrics:
        cells = [label]
        for tag in "ABCD":
            r = by.get(tag)
            cells.append(str(r[key]) if r else "")
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## Deltas (search space)", ""]
    if a and b:
        lines.append(
            f"- A→B (seeds): gen {pct(b['generated'], a['generated'])}, "
            f"live {pct(b['live'], a['live'])}, max_node {pct(b['max_node'], a['max_node'])}, "
            f"goal_pr {pct(b['goal_prune'], a['goal_prune'])}"
        )
    if b and c:
        lines.append(
            f"- B→C (suffix UB): gen {pct(c['generated'], b['generated'])}, "
            f"live {pct(c['live'], b['live'])}, max_node {pct(c['max_node'], b['max_node'])}, "
            f"goal_pr {pct(c['goal_prune'], b['goal_prune'])}, |U|={c['current_U_bound']}"
        )
    if c and d:
        lines.append(
            f"- C→D (dyn corridor): gen {pct(d['generated'], c['generated'])}, "
            f"live {pct(d['live'], c['live'])}, max_node {pct(d['max_node'], c['max_node'])}, "
            f"dyn_edges={d['dynamic_edges_pruned']}, refreshes={d['corridor_refresh_count']}"
        )
    if a and d:
        lines.append(
            f"- A→D (full): gen {pct(d['generated'], a['generated'])}, "
            f"live {pct(d['live'], a['live'])}, max_node {pct(d['max_node'], a['max_node'])}"
        )
        gen_drop = (a["generated"] - d["generated"]) / max(a["generated"], 1)
        live_drop = (a["live"] - d["live"]) / max(a["live"], 1)
        lines += [
            "",
            "## Gate recommendation",
            "",
        ]
        if gen_drop >= 0.15 and live_drop >= 0.15 and d["max_node"] < a["max_node"]:
            lines.append(
                f"**GO full NY0001 with D** (gen↓{100*gen_drop:.1f}%, live↓{100*live_drop:.1f}%). "
                "Keep label cap=30M as resource guard."
            )
        elif abs(gen_drop) < 0.05 and abs(live_drop) < 0.05:
            lines.append(
                "**HOLD full re-run** — search space barely changed; prioritize node-skyline DS / profiler / Top-100 wide nodes."
            )
        else:
            lines.append(
                f"**MARGINAL** (gen↓{100*gen_drop:.1f}%, live↓{100*live_drop:.1f}%) — inspect component deltas before full re-run."
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-expanded", type=int, default=3_000_000)
    ap.add_argument("--out", type=str, default=str(ANALYSIS_DIR / "problem2" / "ab_NY_0001_3M.csv"))
    ap.add_argument("--only", type=str, default="")
    args = ap.parse_args()

    dll = ensure_dll()
    sha = file_sha256(dll)
    print(f"[ab] DLL={dll} sha16={sha}", flush=True)

    variants = [
        # A: v2-like sparse seeds (k=5, no scale norm) → historically ~8 ND
        ("A", 5, False, False, False),
        # B/C/D: dense normalized 66-grid
        ("B", 10, False, False, True),
        ("C", 10, True, False, True),
        ("D", 10, True, True, True),
    ]
    allow = {x.strip().upper() for x in args.only.split(",") if x.strip()} or None

    rows = []
    for tag, k, suf, dyn, scales in variants:
        if allow and tag not in allow:
            continue
        rows.append(run_variant(tag, k, suf, dyn, args.max_expanded, use_scales=scales))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    md = out.with_suffix(".md")
    write_md(rows, md, sha)
    print(f"[wrote] {out}", flush=True)
    print(f"[wrote] {md}", flush=True)


if __name__ == "__main__":
    main()
