# -*- coding: utf-8 -*-
"""
Exact 3-objective NAMOA* (distance, travel_time, elevation).

Heuristics h_i from reverse single-objective Dijkstra; dual dominance pruning
(node skyline + goal-bound). Integer costs only — exact Pareto front.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import CACHE_DIR
from graph import RoadGraph
from shortest_path import INF

Cost3 = Tuple[int, int, int]
LabelKey = Tuple[int, int, int, int, int, int, int]


def dominates_or_equal(a: Cost3, b: Cost3) -> bool:
    return a[0] <= b[0] and a[1] <= b[1] and a[2] <= b[2]


def strictly_dominates(a: Cost3, b: Cost3) -> bool:
    return dominates_or_equal(a, b) and a != b


def update_skyline(front: List[Cost3], y: Cost3) -> bool:
    for z in front:
        if dominates_or_equal(z, y):
            return False
    front[:] = [z for z in front if not dominates_or_equal(y, z)]
    front.append(y)
    return True


def is_dominated_by_set(y: Cost3, front: Sequence[Cost3]) -> bool:
    for z in front:
        if dominates_or_equal(z, y):
            return True
    return False


@dataclass
class HeuristicBundle:
    target: int  # 0-index
    h: np.ndarray  # (n, 3) int64
    nxt: np.ndarray  # (n, 3) int32

    def path_cost3(self, graph: RoadGraph, source: int, obj: int) -> Cost3:
        if source == self.target:
            return (0, 0, 0)
        if int(self.h[source, obj]) >= INF:
            raise RuntimeError(f"Unreachable under obj {obj}: {source+1}->{self.target+1}")
        costs = np.zeros(3, dtype=np.int64)
        v = source
        guard = 0
        n = graph.n_nodes
        while v != self.target:
            w = int(self.nxt[v, obj])
            if w < 0:
                raise RuntimeError(f"Broken SP tree at {v+1} obj={obj}")
            c = graph.edge_cost(v, w)
            if c is None:
                raise RuntimeError(f"Missing edge {v+1}->{w+1} on SP tree")
            costs += c[:3]
            v = w
            guard += 1
            if guard > n + 5:
                raise RuntimeError("Cycle in SP tree reconstruction")
        return (int(costs[0]), int(costs[1]), int(costs[2]))


def reverse_dijkstra_one(
    graph: RoadGraph, target: int, obj_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    n = graph.n_nodes
    dist = np.full(n, INF, dtype=np.int64)
    nxt = np.full(n, -1, dtype=np.int32)
    dist[target] = 0
    pq: List[Tuple[int, int]] = [(0, target)]
    roff, rto, rw = graph.rev_offset, graph.rev_to, graph.rev_w
    while pq:
        d_u, u = heapq.heappop(pq)
        if d_u != dist[u]:
            continue
        a, b = int(roff[u]), int(roff[u + 1])
        for e in range(a, b):
            v = int(rto[e])
            nd = d_u + int(rw[e, obj_idx])
            if nd < dist[v]:
                dist[v] = nd
                nxt[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, nxt


def compute_heuristics(graph: RoadGraph, target_0: int) -> HeuristicBundle:
    h = np.empty((graph.n_nodes, 3), dtype=np.int64)
    nxt = np.empty((graph.n_nodes, 3), dtype=np.int32)
    for k in range(3):
        dist, nxt_k = reverse_dijkstra_one(graph, target_0, k)
        h[:, k] = dist
        nxt[:, k] = nxt_k
    return HeuristicBundle(target=target_0, h=h, nxt=nxt)


def heuristic_cache_path(dataset: str, target_1: int) -> Path:
    d = CACHE_DIR / "problem2"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{dataset}_target_{target_1}_h.npz"


def load_or_compute_heuristics(
    graph: RoadGraph, dataset: str, target_1: int, use_cache: bool = True
) -> HeuristicBundle:
    path = heuristic_cache_path(dataset, target_1)
    if use_cache and path.exists():
        z = np.load(path)
        return HeuristicBundle(target=int(target_1) - 1, h=z["h"], nxt=z["nxt"])
    hb = compute_heuristics(graph, int(target_1) - 1)
    if use_cache:
        np.savez_compressed(path, h=hb.h, nxt=hb.nxt)
    return hb


@dataclass
class NamoaStats:
    pareto_size: int = 0
    runtime_sec: float = 0.0
    h_runtime_sec: float = 0.0
    search_runtime_sec: float = 0.0
    labels_generated: int = 0
    labels_expanded: int = 0
    labels_peak: int = 0
    node_dominance_pruned: int = 0
    goal_bound_pruned: int = 0
    duplicate_pruned: int = 0
    max_labels_at_one_node: int = 0
    mode: str = "namoa"
    exact_finished: int = 0
    early_stop: int = 0
    stale_skipped: int = 0
    open_peak: int = 0
    u_bound_size: int = 0
    edges_killed_dyn: int = 0
    corridor_refresh_count: int = 0
    labels_live: int = 0
    sp_prune_hits: int = 0
    sp_join_updates: int = 0


@dataclass
class NamoaResult:
    solutions: List[Cost3]
    stats: NamoaStats


def seed_anchor_solutions(graph: RoadGraph, source_0: int, hb: HeuristicBundle) -> List[Cost3]:
    front: List[Cost3] = []
    for k in range(3):
        if int(hb.h[source_0, k]) >= INF:
            continue
        update_skyline(front, hb.path_cost3(graph, source_0, k))
    return front


def namoa_star_exact(
    graph: RoadGraph,
    source_1: int,
    target_1: int,
    heuristics: Optional[HeuristicBundle] = None,
    *,
    use_heuristic: bool = True,
    lex_order: Tuple[int, int, int] = (0, 1, 2),
    h_runtime_sec: float = 0.0,
    max_labels: int = 30_000_000,
    backend: str = "cpp",
    extra_seeds: Optional[List[Cost3]] = None,
    edge_allowed: Optional[np.ndarray] = None,
    max_expanded: int = 0,
    suffix_q: Optional[np.ndarray] = None,
    d_s: Optional[np.ndarray] = None,
    profile_dir: Optional[str] = None,
) -> NamoaResult:
    """
    Exact 3-objective search.
    max_expanded>0 stops early for A/B benches (NOT exact).
    For backend=t_mda resource EXACT runs: pass max_labels=0 (unlimited generated)
    and configure set_resource_limits / set_checkpoint on the t_mda module.
    suffix_q: (n,3,3) feasible suffix vectors; d_s: (n,3) for dynamic corridor.
    """
    t_all = time.time()
    s = int(source_1) - 1
    t = int(target_1) - 1
    stats = NamoaStats(
        mode="namoa" if use_heuristic else "multidijkstra",
        h_runtime_sec=h_runtime_sec,
    )

    t_h = time.time()
    if heuristics is None:
        heuristics = compute_heuristics(graph, t)
        stats.h_runtime_sec += time.time() - t_h

    if int(heuristics.h[s, 0]) >= INF:
        stats.runtime_sec = time.time() - t_all
        return NamoaResult(solutions=[], stats=stats)

    seeds = seed_anchor_solutions(graph, s, heuristics)
    if extra_seeds:
        for y in extra_seeds:
            update_skyline(seeds, y)
    seed0 = np.array([y[0] for y in seeds], dtype=np.int64)
    seed1 = np.array([y[1] for y in seeds], dtype=np.int64)
    seed2 = np.array([y[2] for y in seeds], dtype=np.int64)

    t_search = time.time()
    if backend == "t_mda":
        from t_mda import t_mda

        w = graph.w
        h = heuristics.h
        if use_heuristic:
            h0, h1, h2 = h[:, 0], h[:, 1], h[:, 2]
        else:
            z = np.zeros(graph.n_nodes, dtype=np.int64)
            h0, h1, h2 = z, z, z
        solutions, st, status = t_mda(
            graph.offset,
            graph.to,
            w[:, 0],
            w[:, 1],
            w[:, 2],
            h0,
            h1,
            h2,
            s,
            t,
            use_heuristic,
            seed0,
            seed1,
            seed2,
            edge_allowed=edge_allowed,
            max_labels=max_labels,
            max_expanded=max_expanded,
        )
        stats.pareto_size = int(st[0])
        stats.labels_generated = int(st[1])
        stats.labels_expanded = int(st[2])
        stats.labels_peak = int(st[3])
        stats.node_dominance_pruned = int(st[4])  # prop_prune
        stats.goal_bound_pruned = int(st[5])  # nqp_reject
        stats.duplicate_pruned = int(st[6])  # peak_nqp
        stats.max_labels_at_one_node = int(st[7])
        stats.stale_skipped = int(st[9]) if len(st) > 9 else 0  # nqp_live end
        stats.open_peak = int(st[10]) if len(st) > 10 else 0
        stats.exact_finished = 1 if status == 1 else 0
        stats.early_stop = 1 if status in (2, 3) else 0
        stats.labels_live = int(st[15]) if len(st) > 15 else int(st[3])
        stats.u_bound_size = int(st[6])  # reuse: peak_nqp
        stats.mode = "t_mda"
        if status == 3:
            stats.mode = "t_mda_RESOURCE_LIMIT"
    elif backend == "dr_lazy":
        from namoa_dr_lazy import namoa_dr_lazy

        w = graph.w
        h = heuristics.h
        if use_heuristic:
            h0, h1, h2 = h[:, 0], h[:, 1], h[:, 2]
        else:
            z = np.zeros(graph.n_nodes, dtype=np.int64)
            h0, h1, h2 = z, z, z
        solutions, st, status = namoa_dr_lazy(
            graph.offset,
            graph.to,
            w[:, 0],
            w[:, 1],
            w[:, 2],
            h0,
            h1,
            h2,
            s,
            t,
            use_heuristic,
            seed0,
            seed1,
            seed2,
            edge_allowed=edge_allowed,
            max_labels=max_labels,
            max_expanded=max_expanded,
        )
        stats.pareto_size = int(st[0])
        stats.labels_generated = int(st[1])
        stats.labels_expanded = int(st[2])
        stats.labels_peak = int(st[3])
        stats.node_dominance_pruned = int(st[4])  # lazy_reject
        stats.goal_bound_pruned = int(st[5])
        stats.duplicate_pruned = int(st[6])  # gen_goal_prune
        stats.max_labels_at_one_node = int(st[7])
        stats.stale_skipped = int(st[9]) if len(st) > 9 else 0
        stats.open_peak = int(st[10]) if len(st) > 10 else 0
        stats.exact_finished = 1 if status == 1 else 0
        stats.early_stop = 1 if status == 2 else 0
        stats.labels_live = int(st[15]) if len(st) > 15 else int(st[3])
        stats.mode = "namoa_dr_lazy"
    elif backend == "cpp":
        from namoa_cpp import namoa_cpp

        w = graph.w
        h = heuristics.h
        if use_heuristic:
            h0, h1, h2 = h[:, 0], h[:, 1], h[:, 2]
        else:
            z = np.zeros(graph.n_nodes, dtype=np.int64)
            h0, h1, h2 = z, z, z
        solutions, st, status = namoa_cpp(
            graph.offset,
            graph.to,
            w[:, 0],
            w[:, 1],
            w[:, 2],
            h0,
            h1,
            h2,
            s,
            t,
            use_heuristic,
            seed0,
            seed1,
            seed2,
            edge_allowed=edge_allowed,
            max_labels=max_labels,
            max_expanded=max_expanded,
            suffix_q=suffix_q,
            ds0=None if d_s is None else d_s[:, 0],
            ds1=None if d_s is None else d_s[:, 1],
            ds2=None if d_s is None else d_s[:, 2],
            profile_dir=profile_dir,
        )
        stats.pareto_size = int(st[0])
        stats.labels_generated = int(st[1])
        stats.labels_expanded = int(st[2])
        stats.labels_peak = int(st[3])
        stats.node_dominance_pruned = int(st[4])
        stats.goal_bound_pruned = int(st[5])
        stats.duplicate_pruned = int(st[6])
        stats.max_labels_at_one_node = int(st[7])
        stats.stale_skipped = int(st[9]) if len(st) > 9 else 0
        stats.open_peak = int(st[10]) if len(st) > 10 else 0
        stats.exact_finished = 1 if status == 1 else 0
        stats.early_stop = 1 if status == 2 else 0
        stats.u_bound_size = int(st[12]) if len(st) > 12 else 0
        stats.edges_killed_dyn = int(st[13]) if len(st) > 13 else 0
        stats.corridor_refresh_count = int(st[14]) if len(st) > 14 else 0
        stats.labels_live = int(st[15]) if len(st) > 15 else 0
        stats.sp_prune_hits = int(st[26]) if len(st) > 26 else 0
        stats.sp_join_updates = int(st[27]) if len(st) > 27 else 0
    elif backend == "numba":
        raise RuntimeError("use cpp backend")
    else:
        solutions = _namoa_python(
            graph, s, t, heuristics, use_heuristic, lex_order, stats, seeds
        )

    stats.search_runtime_sec = time.time() - t_search
    stats.runtime_sec = time.time() - t_all
    stats.pareto_size = len(solutions)
    return NamoaResult(solutions=solutions, stats=stats)


def _namoa_python(graph, s, t, heuristics, use_heuristic, lex_order, stats, seeds):
    """Original Python NAMOA* (kept for cross-checks on tiny graphs)."""
    solutions: List[Cost3] = list(seeds)
    hmat = heuristics.h if use_heuristic else np.zeros((graph.n_nodes, 3), dtype=np.int64)
    off, to, w = graph.offset, graph.to, graph.w

    def f_of(node: int, g: Cost3) -> Cost3:
        return (g[0] + int(hmat[node, 0]), g[1] + int(hmat[node, 1]), g[2] + int(hmat[node, 2]))

    def heap_key(f: Cost3, g: Cost3, lid: int) -> LabelKey:
        i, j, k = lex_order
        return (f[i], f[j], f[k], g[0], g[1], g[2], lid)

    frontier: Dict[int, List[Tuple[int, int, int, int]]] = {}
    active: List[bool] = []
    label_node: List[int] = []
    label_g: List[Cost3] = []
    open_heap: List[LabelKey] = []
    live_labels = 0

    def add_label(node: int, g: Cost3) -> Optional[int]:
        nonlocal live_labels
        g1, g2, g3 = g
        lst = frontier.setdefault(node, [])
        for og1, og2, og3, oid in lst:
            if og1 <= g1 and og2 <= g2 and og3 <= g3:
                if (og1, og2, og3) == (g1, g2, g3):
                    stats.duplicate_pruned += 1
                else:
                    stats.node_dominance_pruned += 1
                return None
        kept = []
        for og1, og2, og3, oid in lst:
            if g1 <= og1 and g2 <= og2 and g3 <= og3 and (g1, g2, g3) != (og1, og2, og3):
                active[oid] = False
                live_labels -= 1
            else:
                kept.append((og1, og2, og3, oid))
        lid = len(active)
        active.append(True)
        kept.append((g1, g2, g3, lid))
        frontier[node] = kept
        while len(label_node) <= lid:
            label_node.append(-1)
            label_g.append((0, 0, 0))
        label_node[lid] = node
        label_g[lid] = g
        stats.labels_generated += 1
        live_labels += 1
        if len(kept) > stats.max_labels_at_one_node:
            stats.max_labels_at_one_node = len(kept)
        if live_labels > stats.labels_peak:
            stats.labels_peak = live_labels
        return lid

    g0: Cost3 = (0, 0, 0)
    lid0 = add_label(s, g0)
    if lid0 is not None:
        heapq.heappush(open_heap, heap_key(f_of(s, g0), g0, lid0))

    while open_heap:
        key = heapq.heappop(open_heap)
        lid = key[6]
        if not active[lid]:
            continue
        v = label_node[lid]
        g = label_g[lid]
        stats.labels_expanded += 1
        f = f_of(v, g)
        if is_dominated_by_set(f, solutions):
            stats.goal_bound_pruned += 1
            continue
        if v == t:
            update_skyline(solutions, g)
            continue
        for e in range(int(off[v]), int(off[v + 1])):
            u = int(to[e])
            g_new = (g[0] + int(w[e, 0]), g[1] + int(w[e, 1]), g[2] + int(w[e, 2]))
            if u == t:
                if is_dominated_by_set(g_new, solutions):
                    stats.goal_bound_pruned += 1
                    continue
                update_skyline(solutions, g_new)
                add_label(t, g_new)
                continue
            lst = frontier.get(u)
            if lst is not None:
                dominated = False
                for og1, og2, og3, oid in lst:
                    if og1 <= g_new[0] and og2 <= g_new[1] and og3 <= g_new[2]:
                        dominated = True
                        if (og1, og2, og3) == g_new:
                            stats.duplicate_pruned += 1
                        else:
                            stats.node_dominance_pruned += 1
                        break
                if dominated:
                    continue
            f_new = f_of(u, g_new)
            if is_dominated_by_set(f_new, solutions):
                stats.goal_bound_pruned += 1
                continue
            new_id = add_label(u, g_new)
            if new_id is None:
                continue
            heapq.heappush(open_heap, heap_key(f_new, g_new, new_id))

    solutions.sort()
    uniq: List[Cost3] = []
    for y in solutions:
        if not uniq or uniq[-1] != y:
            uniq.append(y)
    return uniq


def verify_anchor_mins(solutions: Sequence[Cost3], h_at_source: Cost3) -> None:
    if not solutions:
        raise AssertionError("Empty Pareto set")
    mins = (
        min(y[0] for y in solutions),
        min(y[1] for y in solutions),
        min(y[2] for y in solutions),
    )
    for i in range(3):
        if mins[i] != h_at_source[i]:
            raise AssertionError(
                f"Pareto min c{i+1}={mins[i]} != Dijkstra h={h_at_source[i]}"
            )


def verify_internal_nondominated(solutions: Sequence[Cost3]) -> None:
    n = len(solutions)
    if len(set(solutions)) != n:
        raise AssertionError("Duplicate vectors in Pareto set")
    for i in range(n):
        for j in range(n):
            if i != j and strictly_dominates(solutions[i], solutions[j]):
                raise AssertionError(f"{solutions[i]} dominates {solutions[j]}")
