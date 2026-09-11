# -*- coding: utf-8 -*-
"""Exact T-MDA adapters for problem-3 (m=2 with c3≡0, m=3 standard)."""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from corridor import build_corridor, lexmin_all_orders
from graph import RoadGraph
from multiobjective_exact import (
    load_or_compute_heuristics,
    namoa_star_exact,
    update_skyline,
)
from seed_bank import build_seed_bank
from t_mda import (
    clear_checkpoint,
    clear_resource_limits,
    clear_resume_path,
    set_checkpoint,
    set_checkpoint_state_every,
    set_nqp_sweep,
    set_resource_limits,
    set_resume_path,
)

Cost3 = Tuple[int, int, int]


def _enrich_seeds(graph: RoadGraph, source: int, target: int, base: List[Cost3]) -> List[Cost3]:
    seeds = list(base)
    for y in lexmin_all_orders(graph, source - 1, target - 1):
        update_skyline(seeds, y)
    return seeds


def solve_tmda_exact(
    graph: RoadGraph,
    dataset: str,
    source_1: int,
    target_1: int,
    *,
    m: int = 3,
    use_corridor: bool = True,
    max_rss_mb: int = 12000,
    max_wall_sec: int = 0,
    ckpt_dir: Optional[Path] = None,
    resume: bool = True,
) -> Dict:
    """
    Exact T-MDA on first m∈{2,3} objectives.
    For m=2, elevation weights are zeroed so 3-obj ND ≡ 2-obj ND.
    """
    if m not in (2, 3):
        raise ValueError("exact T-MDA adapter only supports m=2 or m=3")

    g = graph
    if m == 2:
        g = RoadGraph(
            n_nodes=graph.n_nodes,
            n_edges=graph.n_edges,
            offset=graph.offset,
            to=graph.to,
            w=graph.w.copy(),
            rev_offset=graph.rev_offset,
            rev_to=graph.rev_to,
            rev_w=graph.rev_w.copy(),
            dataset=graph.dataset,
        )
        g.w[:, 2] = 0
        g.rev_w[:, 2] = 0

    hb = load_or_compute_heuristics(g, dataset, target_1, use_cache=(m == 3))
    if m == 2:
        hb.h = hb.h.copy()
        hb.h[:, 2] = 0

    edge_mask = np.ones(g.n_edges, dtype=np.uint8)
    seeds: List[Cost3] = []
    if use_corridor:
        bank = build_seed_bank(g, [source_1], target_1, hb, k=10)
        seeds = _enrich_seeds(g, source_1, target_1, bank.get(source_1, []))
        corr = build_corridor(g, source_1 - 1, hb, seeds)
        edge_mask = corr.edge_allowed.astype(np.uint8)

    tag = f"p3_m{m}_{dataset}_{source_1}_{target_1}"
    ckpt = Path(ckpt_dir) if ckpt_dir else Path(tempfile.gettempdir()) / f"tmda_{tag}"
    ckpt.mkdir(parents=True, exist_ok=True)
    state_bin = ckpt / "state.bin"

    set_nqp_sweep(False)
    set_resource_limits(
        max_rss_mb=max_rss_mb,
        max_nqp_live=0,
        max_live=0,
        max_wall_sec=max_wall_sec,
    )
    set_checkpoint(str(ckpt), 300)
    set_checkpoint_state_every(7200)
    if resume and state_bin.exists():
        set_resume_path(str(state_bin))
    else:
        clear_resume_path()

    t0 = time.time()
    try:
        res = namoa_star_exact(
            g,
            source_1,
            target_1,
            heuristics=hb,
            use_heuristic=True,
            extra_seeds=seeds or None,
            edge_allowed=edge_mask.copy(),
            max_labels=0,
            max_expanded=0,
            backend="t_mda",
        )
    finally:
        clear_resume_path()
        clear_checkpoint()
        clear_resource_limits()

    wall = time.time() - t0
    exact = bool(res.stats.exact_finished)
    mode = res.stats.mode or ""
    if "RESOURCE" in mode:
        status = "RESOURCE_LIMIT"
    elif exact:
        status = "EXACT"
    else:
        status = "INCOMPLETE"

    if m == 2:
        uniq = sorted({(int(a), int(b)) for a, b, _ in res.solutions})
        sols = [(a, b, 0, 0, 0) for a, b in uniq]
    else:
        sols = [(int(a), int(b), int(c), 0, 0) for a, b, c in sorted(set(res.solutions))]

    return {
        "solutions": sols,
        "paths": [None] * len(sols),
        "exact": exact,
        "wall_sec": wall,
        "pareto_size": len(sols),
        "expanded": int(res.stats.labels_expanded),
        "generated": int(res.stats.labels_generated),
        "mode": f"t_mda_exact_m{m}",
        "status": status,
    }
