# -*- coding: utf-8 -*-
"""Weighted-sum Dijkstra candidate generation for m-objective routing."""
from __future__ import annotations

import heapq
from typing import List, Optional, Sequence, Tuple

import numpy as np

from graph import RoadGraph

Cost5 = Tuple[int, int, int, int, int]


def radical_inverse(n: int, base: int) -> float:
    f = 1.0
    res = 0.0
    while n:
        f /= base
        res += f * (n % base)
        n //= base
    return res


def sample_weights(k: int, m: int) -> np.ndarray:
    """k-th low-discrepancy weight on the simplex (exponential of Halton)."""
    primes = (2, 3, 5, 7, 11)
    w = np.empty(m, dtype=float)
    for i in range(m):
        u = max(radical_inverse(k + 1, primes[i]), 1e-12)
        w[i] = -np.log(u)
    w /= w.sum()
    return w


def simplex_weights(n_samples: int, m: int) -> List[np.ndarray]:
    """Include unit / equal weights plus low-discrepancy interior samples."""
    out: List[np.ndarray] = []
    for i in range(m):
        e = np.zeros(m, dtype=float)
        e[i] = 1.0
        out.append(e)
    eq = np.ones(m, dtype=float) / m
    out.append(eq)
    if m == 2:
        for k in range(max(n_samples - len(out), 0)):
            a = k / max(n_samples - 1, 1)
            out.append(np.array([a, 1.0 - a], dtype=float))
    else:
        for k in range(max(n_samples - len(out), 0)):
            out.append(sample_weights(k, m))
    return out


def weighted_dijkstra(
    graph: RoadGraph,
    source0: int,
    target0: int,
    coef: Sequence[float],
    m: int,
) -> Tuple[Optional[List[int]], Optional[Cost5]]:
    """
    Minimize sum_{i<m} coef[i]*c_i. Returns 0-indexed path and full 5-cost.
    coef should already include normalization (weight / scale).
    """
    n = graph.n_nodes
    if source0 == target0:
        return [source0], (0, 0, 0, 0, 0)
    dist = np.full(n, np.inf, dtype=float)
    pred = np.full(n, -1, dtype=np.int32)
    dist[source0] = 0.0
    pq: List[Tuple[float, int]] = [(0.0, source0)]
    off, to, w = graph.offset, graph.to, graph.w
    c = np.asarray(coef[:m], dtype=float)
    while pq:
        d_u, u = heapq.heappop(pq)
        if d_u != dist[u]:
            continue
        if u == target0:
            break
        a, b = int(off[u]), int(off[u + 1])
        for e in range(a, b):
            v = int(to[e])
            ew = 0.0
            for i in range(m):
                ew += c[i] * float(w[e, i])
            nd = d_u + ew
            if nd < dist[v]:
                dist[v] = nd
                pred[v] = u
                heapq.heappush(pq, (nd, v))
    if not np.isfinite(dist[target0]):
        return None, None
    rev: List[int] = []
    v = target0
    while v != source0:
        rev.append(v)
        v = int(pred[v])
        if v < 0:
            return None, None
    rev.append(source0)
    path = list(reversed(rev))
    costs = graph.evaluate_path(path)
    return path, tuple(int(x) for x in costs)  # type: ignore[return-value]


def generate_weighted_candidates(
    graph: RoadGraph,
    source_1: int,
    target_1: int,
    m: int,
    n_samples: int,
    *,
    prefer_cpp: bool = True,
) -> Tuple[List[Cost5], List[List[int]], np.ndarray, np.ndarray]:
    """
    Anchors + weighted samples → unique paths with 5-costs.
    Prefer C++ backend (problem3jsy-speed); fall back to Python if unavailable.
    """
    if prefer_cpp:
        try:
            from solver.p3_weighted_cpp import generate_weighted_candidates_cpp

            return generate_weighted_candidates_cpp(graph, source_1, target_1, m, n_samples)
        except Exception as exc:  # noqa: BLE001
            print(f"[weighted] C++ backend unavailable ({exc}); using Python", flush=True)

    s = int(source_1) - 1
    t = int(target_1) - 1
    from shortest_path import bidirectional_dijkstra

    anchors: List[Cost5] = []
    anchor_paths: List[List[int]] = []
    for i in range(m):
        path0, _ = bidirectional_dijkstra(graph, s, t, i)
        if path0 is None:
            raise RuntimeError(f"unreachable on objective {i}")
        c = tuple(int(x) for x in graph.evaluate_path(path0))
        anchors.append(c)  # type: ignore[arg-type]
        anchor_paths.append(path0)
    arr = np.asarray(anchors, dtype=float)
    z_min = arr.min(axis=0)[:m]
    z_max = arr.max(axis=0)[:m]
    scale = np.maximum(z_max - z_min, 1.0)

    seen = {}
    for p, c in zip(anchor_paths, anchors):
        key = c[:m]
        if key not in seen or len(p) < len(seen[key][1]):
            seen[key] = (c, p)

    for w in simplex_weights(n_samples, m):
        coef = w / scale
        path0, cost = weighted_dijkstra(graph, s, t, coef, m)
        if path0 is None or cost is None:
            continue
        key = cost[:m]
        if key not in seen or len(path0) < len(seen[key][1]):
            seen[key] = (cost, path0)

    costs = [v[0] for v in seen.values()]
    paths = [v[1] for v in seen.values()]
    return costs, paths, z_min, z_max
