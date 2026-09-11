# -*- coding: utf-8 -*-
"""
Exact query-specific corridor pruning for NAMOA*.

For any s-t path through v:
  C_i >= d_i^s(v) + h_i(v) =: L_i(v)
If some incumbent y dominates L(v), no Pareto path can go through v.

For edge e=(u,v):
  L_i(e) = d_i^s(u) + c_i(e) + h_i(v)
Same argument — edge can be excluded safely.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from graph import RoadGraph
from multiobjective_exact import Cost3, HeuristicBundle, dominates_or_equal
from shortest_path import INF


def forward_dijkstra(graph: RoadGraph, source: int, obj_idx: int) -> np.ndarray:
    """dist[v] = min cost source->v under obj_idx. 0-indexed source."""
    n = graph.n_nodes
    dist = np.full(n, INF, dtype=np.int64)
    dist[source] = 0
    pq: List[Tuple[int, int]] = [(0, source)]
    off, to, w = graph.offset, graph.to, graph.w
    while pq:
        d_u, u = heapq.heappop(pq)
        if d_u != dist[u]:
            continue
        for e in range(int(off[u]), int(off[u + 1])):
            v = int(to[e])
            nd = d_u + int(w[e, obj_idx])
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


@dataclass
class Corridor:
    d_s: np.ndarray  # (n, 3) forward single-obj distances from s
    h: np.ndarray  # (n, 3) reverse heuristics to t
    node_allowed: np.ndarray  # bool (n,)
    edge_allowed: np.ndarray  # bool (n_edges,)
    n_nodes_kept: int = 0
    n_edges_kept: int = 0
    n_nodes_pruned: int = 0
    n_edges_pruned: int = 0


def build_corridor(
    graph: RoadGraph,
    source_0: int,
    hb: HeuristicBundle,
    incumbents: Sequence[Cost3],
) -> Corridor:
    """Build exact node/edge masks given current feasible Pareto seeds."""
    d_s = np.empty((graph.n_nodes, 3), dtype=np.int64)
    for k in range(3):
        d_s[:, k] = forward_dijkstra(graph, source_0, k)
    h = hb.h

    node_allowed = np.ones(graph.n_nodes, dtype=np.bool_)
    # source and target always kept
    node_allowed[:] = True
    if incumbents:
        for v in range(graph.n_nodes):
            if v == source_0 or v == hb.target:
                continue
            # unreachable under any single obj -> keep (or prune if all INF)
            if int(d_s[v, 0]) >= INF or int(h[v, 0]) >= INF:
                node_allowed[v] = False
                continue
            L = (
                int(d_s[v, 0]) + int(h[v, 0]),
                int(d_s[v, 1]) + int(h[v, 1]),
                int(d_s[v, 2]) + int(h[v, 2]),
            )
            # if any incumbent weakly dominates L, no improving path via v
            for y in incumbents:
                if dominates_or_equal(y, L):
                    node_allowed[v] = False
                    break

    edge_allowed = np.ones(graph.n_edges, dtype=np.bool_)
    off, to, w = graph.offset, graph.to, graph.w
    if incumbents:
        for u in range(graph.n_nodes):
            a, b = int(off[u]), int(off[u + 1])
            if not node_allowed[u]:
                edge_allowed[a:b] = False
                continue
            for e in range(a, b):
                v = int(to[e])
                if not node_allowed[v]:
                    edge_allowed[e] = False
                    continue
                if int(d_s[u, 0]) >= INF or int(h[v, 0]) >= INF:
                    edge_allowed[e] = False
                    continue
                Le = (
                    int(d_s[u, 0]) + int(w[e, 0]) + int(h[v, 0]),
                    int(d_s[u, 1]) + int(w[e, 1]) + int(h[v, 1]),
                    int(d_s[u, 2]) + int(w[e, 2]) + int(h[v, 2]),
                )
                for y in incumbents:
                    if dominates_or_equal(y, Le):
                        edge_allowed[e] = False
                        break

    n_nodes_kept = int(node_allowed.sum())
    n_edges_kept = int(edge_allowed.sum())
    return Corridor(
        d_s=d_s,
        h=h,
        node_allowed=node_allowed,
        edge_allowed=edge_allowed,
        n_nodes_kept=n_nodes_kept,
        n_edges_kept=n_edges_kept,
        n_nodes_pruned=graph.n_nodes - n_nodes_kept,
        n_edges_pruned=graph.n_edges - n_edges_kept,
    )


def lexmin_path_cost3(
    graph: RoadGraph,
    source_0: int,
    target_0: int,
    order: Tuple[int, int, int],
) -> Cost3:
    """
    Lexicographic shortest path under (c[order[0]], c[order[1]], c[order[2]]).
    One Dijkstra with tuple keys — exact, no scaling tricks.
    """
    i, j, k = order
    n = graph.n_nodes
    BIG = int(INF)
    dist: List[Cost3] = [(BIG, BIG, BIG)] * n
    dist[source_0] = (0, 0, 0)

    def key_of(full: Cost3) -> Cost3:
        return (full[i], full[j], full[k])

    pq: List[Tuple[Cost3, int]] = [(key_of(dist[source_0]), source_0)]
    off, to, w = graph.offset, graph.to, graph.w
    while pq:
        key_u, u = heapq.heappop(pq)
        if key_u != key_of(dist[u]):
            continue
        if u == target_0:
            break
        g0, g1, g2 = dist[u]
        for e in range(int(off[u]), int(off[u + 1])):
            v = int(to[e])
            full = (g0 + int(w[e, 0]), g1 + int(w[e, 1]), g2 + int(w[e, 2]))
            nk = key_of(full)
            if nk < key_of(dist[v]):
                dist[v] = full
                heapq.heappush(pq, (nk, v))
    if dist[target_0][0] >= BIG:
        raise RuntimeError("lexmin unreachable")
    return dist[target_0]


def lexmin_all_orders(graph: RoadGraph, source_0: int, target_0: int) -> List[Cost3]:
    from itertools import permutations

    out: List[Cost3] = []
    for order in permutations((0, 1, 2)):
        try:
            out.append(lexmin_path_cost3(graph, source_0, target_0, order))
        except RuntimeError:
            continue
    return out
