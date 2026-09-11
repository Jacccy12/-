# -*- coding: utf-8 -*-
"""Single-objective bidirectional Dijkstra / BFS on CSR road graphs.

Work arrays are reused across queries via generation stamps (no O(|V|) fill).
Hot search buffers use Python lists to avoid NumPy scalar boxing in tight loops.
"""
from __future__ import annotations

import heapq
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from graph import RoadGraph

INF = np.iinfo(np.int64).max // 4

# stamp overflow guard
_STAMP_RESET = 2_000_000_000


class SearchWorkspace:
    """Preallocated search buffers shared across queries on one graph."""

    __slots__ = (
        "n",
        "search_id",
        "dist_f",
        "dist_b",
        "stamp_f",
        "stamp_b",
        "pred_f",
        "pred_b",
        "pred_stamp_f",
        "pred_stamp_b",
        "settled_f",
        "settled_b",
    )

    def __init__(self, n: int):
        self.n = int(n)
        self.search_id = 0
        # Python lists: faster than ndarray in pure-Python Dijkstra loops
        self.dist_f = [INF] * n
        self.dist_b = [INF] * n
        self.stamp_f = [0] * n
        self.stamp_b = [0] * n
        self.pred_f = [-1] * n
        self.pred_b = [-1] * n
        self.pred_stamp_f = [0] * n
        self.pred_stamp_b = [0] * n
        self.settled_f = [0] * n
        self.settled_b = [0] * n

    def begin(self) -> int:
        self.search_id += 1
        if self.search_id >= _STAMP_RESET:
            n = self.n
            self.stamp_f = [0] * n
            self.stamp_b = [0] * n
            self.pred_stamp_f = [0] * n
            self.pred_stamp_b = [0] * n
            self.settled_f = [0] * n
            self.settled_b = [0] * n
            self.search_id = 1
        return self.search_id


def get_workspace(graph: RoadGraph) -> SearchWorkspace:
    ws = getattr(graph, "_search_ws", None)
    if ws is None or ws.n != graph.n_nodes:
        ws = SearchWorkspace(graph.n_nodes)
        graph._search_ws = ws  # type: ignore[attr-defined]
    return ws


def _reconstruct(
    meet: int,
    pred_f: List[int],
    pred_stamp_f: List[int],
    pred_b: List[int],
    pred_stamp_b: List[int],
    sid: int,
) -> List[int]:
    forward: List[int] = []
    v = meet
    while v != -1:
        forward.append(v)
        if pred_stamp_f[v] != sid:
            break
        v = pred_f[v]
    forward.reverse()

    backward: List[int] = []
    v = pred_b[meet] if pred_stamp_b[meet] == sid else -1
    while v != -1:
        backward.append(v)
        if pred_stamp_b[v] != sid:
            break
        v = pred_b[v]
    return forward + backward


def bidirectional_dijkstra(
    graph: RoadGraph,
    source: int,
    target: int,
    obj_idx: int,
    workspace: Optional[SearchWorkspace] = None,
) -> Tuple[Optional[List[int]], int]:
    """
    Bidirectional Dijkstra for a single non-negative cost dimension.
    Tie-break: heap key (distance, node_id). Predecessor only on strict improvement.
    """
    if source == target:
        return [source], 0

    ws = workspace or get_workspace(graph)
    sid = ws.begin()

    dist_f = ws.dist_f
    dist_b = ws.dist_b
    stamp_f = ws.stamp_f
    stamp_b = ws.stamp_b
    pred_f = ws.pred_f
    pred_b = ws.pred_b
    pred_stamp_f = ws.pred_stamp_f
    pred_stamp_b = ws.pred_stamp_b
    settled_f = ws.settled_f
    settled_b = ws.settled_b

    stamp_f[source] = sid
    dist_f[source] = 0
    stamp_b[target] = sid
    dist_b[target] = 0

    pq_f: List[Tuple[int, int]] = [(0, source)]
    pq_b: List[Tuple[int, int]] = [(0, target)]

    mu = INF
    meet = -1

    off = graph.offset
    to = graph.to
    w = graph.w
    roff = graph.rev_offset
    rto = graph.rev_to
    rw = graph.rev_w

    while pq_f or pq_b:
        if pq_f:
            while pq_f and settled_f[pq_f[0][1]] == sid:
                heapq.heappop(pq_f)
        if pq_b:
            while pq_b and settled_b[pq_b[0][1]] == sid:
                heapq.heappop(pq_b)

        df = pq_f[0][0] if pq_f else INF
        db = pq_b[0][0] if pq_b else INF
        if df == INF and db == INF:
            break
        if df + db >= mu:
            break

        if df <= db:
            d_u, u = heapq.heappop(pq_f)
            if settled_f[u] == sid:
                continue
            if stamp_f[u] != sid or d_u != dist_f[u]:
                continue
            settled_f[u] = sid
            if settled_b[u] == sid:
                cand = d_u + dist_b[u]
                if cand < mu:
                    mu = cand
                    meet = u
            a = int(off[u])
            b = int(off[u + 1])
            for e in range(a, b):
                v = int(to[e])
                nd = d_u + int(w[e, obj_idx])
                if stamp_f[v] != sid or nd < dist_f[v]:
                    stamp_f[v] = sid
                    dist_f[v] = nd
                    pred_stamp_f[v] = sid
                    pred_f[v] = u
                    heapq.heappush(pq_f, (nd, v))
                    if settled_b[v] == sid:
                        cand = nd + dist_b[v]
                        if cand < mu:
                            mu = cand
                            meet = v
        else:
            d_u, u = heapq.heappop(pq_b)
            if settled_b[u] == sid:
                continue
            if stamp_b[u] != sid or d_u != dist_b[u]:
                continue
            settled_b[u] = sid
            if settled_f[u] == sid:
                cand = d_u + dist_f[u]
                if cand < mu:
                    mu = cand
                    meet = u
            a = int(roff[u])
            b = int(roff[u + 1])
            for e in range(a, b):
                v = int(rto[e])
                nd = d_u + int(rw[e, obj_idx])
                if stamp_b[v] != sid or nd < dist_b[v]:
                    stamp_b[v] = sid
                    dist_b[v] = nd
                    pred_stamp_b[v] = sid
                    pred_b[v] = u
                    heapq.heappush(pq_b, (nd, v))
                    if settled_f[v] == sid:
                        cand = nd + dist_f[v]
                        if cand < mu:
                            mu = cand
                            meet = v

    if meet < 0 or mu >= INF:
        return None, INF

    path = _reconstruct(meet, pred_f, pred_stamp_f, pred_b, pred_stamp_b, sid)
    costs = graph.evaluate_path(path)
    if int(costs[obj_idx]) != int(mu):
        return dijkstra(graph, source, target, obj_idx, workspace=ws)
    return path, int(mu)


def dijkstra(
    graph: RoadGraph,
    source: int,
    target: int,
    obj_idx: int,
    workspace: Optional[SearchWorkspace] = None,
) -> Tuple[Optional[List[int]], int]:
    """Unidirectional Dijkstra (fallback / independent optimality check)."""
    if source == target:
        return [source], 0

    ws = workspace or get_workspace(graph)
    sid = ws.begin()
    dist_f = ws.dist_f
    stamp_f = ws.stamp_f
    pred_f = ws.pred_f
    pred_stamp_f = ws.pred_stamp_f

    stamp_f[source] = sid
    dist_f[source] = 0
    pq: List[Tuple[int, int]] = [(0, source)]
    off, to, w = graph.offset, graph.to, graph.w

    while pq:
        d_u, u = heapq.heappop(pq)
        if stamp_f[u] != sid or d_u != dist_f[u]:
            continue
        if u == target:
            break
        a = int(off[u])
        b = int(off[u + 1])
        for e in range(a, b):
            v = int(to[e])
            nd = d_u + int(w[e, obj_idx])
            if stamp_f[v] != sid or nd < dist_f[v]:
                stamp_f[v] = sid
                dist_f[v] = nd
                pred_stamp_f[v] = sid
                pred_f[v] = u
                heapq.heappush(pq, (nd, v))

    if stamp_f[target] != sid:
        return None, INF

    path: List[int] = []
    v = target
    while True:
        path.append(v)
        if pred_stamp_f[v] != sid:
            break
        v = pred_f[v]
        if v < 0:
            break
    path.reverse()
    return path, dist_f[target]


def single_source_dijkstra(
    graph: RoadGraph,
    source: int,
    obj_idx: int,
    workspace: Optional[SearchWorkspace] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Dijkstra from source to all nodes. Returns (dist[n], pred[n])."""
    ws = workspace or get_workspace(graph)
    sid = ws.begin()
    dist_f = ws.dist_f
    stamp_f = ws.stamp_f
    pred_f = ws.pred_f
    pred_stamp_f = ws.pred_stamp_f

    stamp_f[source] = sid
    dist_f[source] = 0
    pq: List[Tuple[int, int]] = [(0, source)]
    off, to, w = graph.offset, graph.to, graph.w
    n = graph.n_nodes

    while pq:
        d_u, u = heapq.heappop(pq)
        if stamp_f[u] != sid or d_u != dist_f[u]:
            continue
        a = int(off[u])
        b = int(off[u + 1])
        for e in range(a, b):
            v = int(to[e])
            nd = d_u + int(w[e, obj_idx])
            if stamp_f[v] != sid or nd < dist_f[v]:
                stamp_f[v] = sid
                dist_f[v] = nd
                pred_stamp_f[v] = sid
                pred_f[v] = u
                heapq.heappush(pq, (nd, v))

    dist = np.full(n, INF, dtype=np.int64)
    pred = np.full(n, -1, dtype=np.int32)
    for v in range(n):
        if stamp_f[v] == sid:
            dist[v] = dist_f[v]
            if pred_stamp_f[v] == sid:
                pred[v] = pred_f[v]
    return dist, pred


def reverse_dijkstra_all_nodes(
    graph: RoadGraph,
    target: int,
    objective,
    workspace: Optional[SearchWorkspace] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reverse Dijkstra from target on reverse CSR.
    dist[v] = cost v→target; nxt[v] = next hop toward target.
    """
    from config import OBJECTIVE_INDEX

    if isinstance(objective, str):
        obj_idx = OBJECTIVE_INDEX[objective]
    else:
        obj_idx = int(objective)

    ws = workspace or get_workspace(graph)
    sid = ws.begin()
    dist_b = ws.dist_b
    stamp_b = ws.stamp_b
    pred_b = ws.pred_b
    pred_stamp_b = ws.pred_stamp_b

    stamp_b[target] = sid
    dist_b[target] = 0
    pq: List[Tuple[int, int]] = [(0, target)]
    roff, rto, rw = graph.rev_offset, graph.rev_to, graph.rev_w
    n = graph.n_nodes

    while pq:
        d_u, u = heapq.heappop(pq)
        if stamp_b[u] != sid or d_u != dist_b[u]:
            continue
        a = int(roff[u])
        b = int(roff[u + 1])
        for e in range(a, b):
            v = int(rto[e])
            nd = d_u + int(rw[e, obj_idx])
            if stamp_b[v] != sid or nd < dist_b[v]:
                stamp_b[v] = sid
                dist_b[v] = nd
                pred_stamp_b[v] = sid
                pred_b[v] = u
                heapq.heappush(pq, (nd, v))

    dist = np.full(n, INF, dtype=np.int64)
    nxt = np.full(n, -1, dtype=np.int32)
    for v in range(n):
        if stamp_b[v] == sid:
            dist[v] = dist_b[v]
            if pred_stamp_b[v] == sid:
                nxt[v] = pred_b[v]
    return dist, nxt


def bfs(
    graph: RoadGraph,
    source: int,
    target: int,
    workspace: Optional[SearchWorkspace] = None,
) -> Tuple[Optional[List[int]], int]:
    """Unidirectional fewest-hops BFS (independent optimality check)."""
    if source == target:
        return [source], 0

    ws = workspace or get_workspace(graph)
    sid = ws.begin()
    dist_f = ws.dist_f
    stamp_f = ws.stamp_f
    pred_f = ws.pred_f
    pred_stamp_f = ws.pred_stamp_f

    stamp_f[source] = sid
    dist_f[source] = 0
    q: deque = deque([source])
    off, to = graph.offset, graph.to

    while q:
        u = q.popleft()
        if u == target:
            break
        du = dist_f[u]
        a = int(off[u])
        b = int(off[u + 1])
        for e in range(a, b):
            v = int(to[e])
            if stamp_f[v] != sid:
                stamp_f[v] = sid
                dist_f[v] = du + 1
                pred_stamp_f[v] = sid
                pred_f[v] = u
                q.append(v)

    if stamp_f[target] != sid:
        return None, INF

    path: List[int] = []
    v = target
    while True:
        path.append(v)
        if pred_stamp_f[v] != sid:
            break
        v = pred_f[v]
        if v < 0:
            break
    path.reverse()
    return path, dist_f[target]


def bidirectional_bfs(
    graph: RoadGraph,
    source: int,
    target: int,
    workspace: Optional[SearchWorkspace] = None,
) -> Tuple[Optional[List[int]], int]:
    """Bidirectional BFS for fewest edges (c5). Forward CSR + reverse CSR."""
    if source == target:
        return [source], 0

    ws = workspace or get_workspace(graph)
    sid = ws.begin()
    dist_f = ws.dist_f
    dist_b = ws.dist_b
    stamp_f = ws.stamp_f
    stamp_b = ws.stamp_b
    pred_f = ws.pred_f
    pred_b = ws.pred_b
    pred_stamp_f = ws.pred_stamp_f
    pred_stamp_b = ws.pred_stamp_b

    stamp_f[source] = sid
    dist_f[source] = 0
    stamp_b[target] = sid
    dist_b[target] = 0
    q_f: deque = deque([source])
    q_b: deque = deque([target])

    mu = INF
    meet = -1

    off, to = graph.offset, graph.to
    roff, rto = graph.rev_offset, graph.rev_to

    while q_f or q_b:
        df = dist_f[q_f[0]] if q_f else INF
        db = dist_b[q_b[0]] if q_b else INF
        if df == INF and db == INF:
            break
        if df + db >= mu:
            break

        expand_forward = True
        if q_f and q_b:
            expand_forward = df <= db
        elif not q_f:
            expand_forward = False

        if expand_forward:
            u = q_f.popleft()
            du = dist_f[u]
            a = int(off[u])
            b = int(off[u + 1])
            for e in range(a, b):
                v = int(to[e])
                if stamp_f[v] != sid:
                    stamp_f[v] = sid
                    dist_f[v] = du + 1
                    pred_stamp_f[v] = sid
                    pred_f[v] = u
                    q_f.append(v)
                    if stamp_b[v] == sid:
                        cand = du + 1 + dist_b[v]
                        if cand < mu or (cand == mu and (meet < 0 or v < meet)):
                            mu = cand
                            meet = v
        else:
            u = q_b.popleft()
            du = dist_b[u]
            a = int(roff[u])
            b = int(roff[u + 1])
            for e in range(a, b):
                v = int(rto[e])
                if stamp_b[v] != sid:
                    stamp_b[v] = sid
                    dist_b[v] = du + 1
                    pred_stamp_b[v] = sid
                    pred_b[v] = u
                    q_b.append(v)
                    if stamp_f[v] == sid:
                        cand = dist_f[v] + du + 1
                        if cand < mu or (cand == mu and (meet < 0 or v < meet)):
                            mu = cand
                            meet = v

    if meet < 0 or mu >= INF:
        return None, INF

    path = _reconstruct(meet, pred_f, pred_stamp_f, pred_b, pred_stamp_b, sid)
    if len(path) - 1 != mu or len(path) != len(set(path)):
        return bfs(graph, source, target, workspace=ws)
    return path, mu


def shortest_path(
    graph: RoadGraph,
    source_1: int,
    target_1: int,
    objective: str,
    use_bidirectional: bool = True,
) -> Tuple[Optional[List[int]], np.ndarray]:
    """
    Compute single-objective optimal path.
    c1–c4 → bidirectional Dijkstra; hop_count → bidirectional BFS.
    """
    from config import OBJECTIVE_INDEX

    s = int(source_1) - 1
    t = int(target_1) - 1
    if s < 0 or t < 0 or s >= graph.n_nodes or t >= graph.n_nodes:
        raise ValueError(f"Node out of range: {source_1}->{target_1}")

    obj_idx = OBJECTIVE_INDEX[objective]
    ws = get_workspace(graph)

    if objective == "hop_count":
        if use_bidirectional:
            path, _ = bidirectional_bfs(graph, s, t, workspace=ws)
        else:
            path, _ = bfs(graph, s, t, workspace=ws)
    elif use_bidirectional:
        path, _ = bidirectional_dijkstra(graph, s, t, obj_idx, workspace=ws)
    else:
        path, _ = dijkstra(graph, s, t, obj_idx, workspace=ws)

    if path is None:
        raise RuntimeError(f"Unreachable: {graph.dataset} {source_1}->{target_1} ({objective})")

    costs = graph.evaluate_path(path)
    return path, costs


def path_to_string(path_0: List[int]) -> str:
    return "->".join(str(v + 1) for v in path_0)


def path_edges(path_0: List[int]) -> set:
    return {(path_0[i], path_0[i + 1]) for i in range(len(path_0) - 1)}
