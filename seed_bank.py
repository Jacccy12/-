# -*- coding: utf-8 -*-
"""Weighted / dense simplex seeds + suffix 3D cost vectors (exact)."""
from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from graph import RoadGraph
from multiobjective_exact import Cost3, HeuristicBundle, update_skyline
from shortest_path import INF


def weight_grid(k: int = 5) -> List[Tuple[int, int, int]]:
    """Nonnegative integer weights (a,b,c) with a+b+c = k."""
    ws = set()
    for a in range(k + 1):
        for b in range(k + 1 - a):
            c = k - a - b
            ws.add((a, b, c))
    return sorted(ws)


def objective_scales(graph: RoadGraph) -> Tuple[int, int, int]:
    """
    Fixed positive scales for weighted seeds: use median positive edge cost per obj.
    Weighted Dijkstra minimizes sum w_i * C_i / s_i (via integer w_i' = w_i * LCM/s_i approx).
    Seeds remain true feasible paths — scales only affect which supported points we hit.
    """
    w = graph.w[:, :3]
    scales = []
    for k in range(3):
        col = w[:, k]
        pos = col[col > 0]
        if pos.size == 0:
            scales.append(1)
        else:
            scales.append(max(1, int(np.median(pos))))
    return int(scales[0]), int(scales[1]), int(scales[2])


def reverse_weighted_dijkstra(
    graph: RoadGraph,
    target: int,
    wa: int,
    wb: int,
    wc: int,
    scales: Optional[Tuple[int, int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reverse Dijkstra on normalized scalarized cost:
        wa*(c0/s0) + wb*(c1/s1) + wc*(c2/s2)
    Float distances (seed discovery only). Returned nxt yields true 3D path costs.
    """
    n = graph.n_nodes
    dist = np.full(n, np.inf, dtype=np.float64)
    nxt = np.full(n, -1, dtype=np.int32)
    dist[target] = 0.0
    pq: List[Tuple[float, int]] = [(0.0, target)]
    roff, rto, rw = graph.rev_offset, graph.rev_to, graph.rev_w
    if scales is None:
        s0, s1, s2 = 1.0, 1.0, 1.0
    else:
        s0, s1, s2 = float(scales[0]), float(scales[1]), float(scales[2])
    while pq:
        d_u, u = heapq.heappop(pq)
        if d_u != dist[u]:
            continue
        for e in range(int(roff[u]), int(roff[u + 1])):
            v = int(rto[e])
            c0, c1, c2 = float(rw[e, 0]), float(rw[e, 1]), float(rw[e, 2])
            add = wa * (c0 / s0) + wb * (c1 / s1) + wc * (c2 / s2)
            nd = d_u + add
            if nd < dist[v]:
                dist[v] = nd
                nxt[v] = u
                heapq.heappush(pq, (nd, v))
    out_dist = np.full(n, INF, dtype=np.int64)
    finite = np.isfinite(dist)
    out_dist[finite] = np.minimum(dist[finite], float(INF - 1)).astype(np.int64)
    return out_dist, nxt


def path_cost3_from_next(graph: RoadGraph, source: int, target: int, nxt: np.ndarray) -> Cost3:
    if source == target:
        return (0, 0, 0)
    costs = np.zeros(3, dtype=np.int64)
    v = source
    guard = 0
    while v != target:
        w = int(nxt[v])
        if w < 0:
            raise RuntimeError(f"Broken weighted tree at {v+1}")
        c = graph.edge_cost(v, w)
        if c is None:
            raise RuntimeError(f"Missing edge {v+1}->{w+1}")
        costs += c[:3]
        v = w
        guard += 1
        if guard > graph.n_nodes + 5:
            raise RuntimeError("Cycle in weighted tree")
    return (int(costs[0]), int(costs[1]), int(costs[2]))


def suffix_cost3_from_trees(graph: RoadGraph, hb: HeuristicBundle) -> np.ndarray:
    """
    q[v, k, :] = full 3D cost of following objective-k reverse SP tree from v to t.
    Safe feasible suffix; g+q[v,k] is a valid upper-bound incumbent (after de-cycling).
    """
    n = graph.n_nodes
    t = hb.target
    q = np.zeros((n, 3, 3), dtype=np.int64)
    # memo per tree
    for k in range(3):
        done = np.zeros(n, dtype=np.bool_)
        done[t] = True
        stack: List[int] = []

        def resolve(v: int) -> None:
            if done[v]:
                return
            path = []
            x = v
            while not done[x]:
                path.append(x)
                nxt = int(hb.nxt[x, k])
                if nxt < 0:
                    # unreachable under this tree
                    for y in path:
                        q[y, k, :] = INF // 8
                        done[y] = True
                    return
                if done[nxt]:
                    # finish chain
                    cur = nxt
                    for y in reversed(path):
                        c = graph.edge_cost(y, int(hb.nxt[y, k]))
                        if c is None:
                            q[y, k, :] = INF // 8
                        else:
                            q[y, k, :] = c[:3] + q[int(hb.nxt[y, k]), k, :]
                        done[y] = True
                    return
                x = nxt
                if len(path) > n + 5:
                    for y in path:
                        q[y, k, :] = INF // 8
                        done[y] = True
                    return

        for v in range(n):
            if not done[v]:
                resolve(v)
    return q


def build_seed_bank(
    graph: RoadGraph,
    sources_1: Sequence[int],
    target_1: int,
    hb: HeuristicBundle,
    k: int = 10,
) -> Dict[int, List[Cost3]]:
    """
    Dense simplex weighted seeds: for k=10 there are C(12,2)=66 weights.
    Still exact — only adds true feasible incumbents.
    """
    target = int(target_1) - 1
    sources0 = [int(s) - 1 for s in sources_1]
    bank: Dict[int, List[Cost3]] = {int(s): [] for s in sources_1}

    for s1, s0 in zip(sources_1, sources0):
        front: List[Cost3] = []
        for obj in range(3):
            if int(hb.h[s0, obj]) < INF:
                update_skyline(front, hb.path_cost3(graph, s0, obj))
        bank[int(s1)] = front

    weights = weight_grid(k)
    scales = objective_scales(graph)
    print(
        f"    [seeds] target={target_1}: {len(weights)} weighted reverse Dijkstras "
        f"(k={k}, scales={scales})",
        flush=True,
    )

    for wi, (wa, wb, wc) in enumerate(weights, start=1):
        if wa == 0 and wb == 0 and wc == 0:
            continue
        _, nxt = reverse_weighted_dijkstra(graph, target, wa, wb, wc, scales=scales)
        for s1, s0 in zip(sources_1, sources0):
            if int(nxt[s0]) < 0 and s0 != target:
                continue
            try:
                y = path_cost3_from_next(graph, s0, target, nxt)
            except RuntimeError:
                continue
            update_skyline(bank[int(s1)], y)
        if wi % 15 == 0 or wi == len(weights):
            avg = float(np.mean([len(v) for v in bank.values()]))
            print(f"      weight {wi}/{len(weights)} avg_seeds={avg:.1f}", flush=True)

    return bank
