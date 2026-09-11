# -*- coding: utf-8 -*-
"""ε-grid / ε-dominance utilities for high-dimensional Pareto approximation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Cost = Tuple[int, ...]


def strict_dominates(a: Sequence[int], b: Sequence[int], m: int) -> bool:
    """a strictly dominates b on first m objectives."""
    le = True
    lt = False
    for i in range(m):
        if a[i] > b[i]:
            return False
        if a[i] < b[i]:
            lt = True
    return le and lt


def weak_dominates(a: Sequence[int], b: Sequence[int], m: int) -> bool:
    return all(a[i] <= b[i] for i in range(m))


def filter_nondominated(
    costs: Sequence[Sequence[int]],
    m: int,
    meta: Optional[Sequence] = None,
) -> Tuple[List[Cost], List]:
    """Keep strict nondominated subset on first m dims. meta aligned with costs."""
    n = len(costs)
    if n == 0:
        return [], []
    arr = [tuple(int(c[i]) for i in range(m)) + tuple(int(x) for x in c[m:]) for c in costs]
    keep = [True] * n
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            if strict_dominates(arr[i], arr[j], m):
                keep[j] = False
            elif strict_dominates(arr[j], arr[i], m):
                keep[i] = False
                break
    out_c = [arr[i] for i in range(n) if keep[i]]
    if meta is None:
        return out_c, []
    out_m = [meta[i] for i in range(n) if keep[i]]
    return out_c, out_m


def normalize_costs(
    costs: Sequence[Sequence[int]],
    m: int,
    z_min: Optional[Sequence[float]] = None,
    z_max: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Z in [0,1]^m, z_min, z_max)."""
    arr = np.asarray([[c[i] for i in range(m)] for c in costs], dtype=float)
    if arr.size == 0:
        return arr, np.zeros(m), np.ones(m)
    lo = np.asarray(z_min, dtype=float) if z_min is not None else arr.min(axis=0)
    hi = np.asarray(z_max, dtype=float) if z_max is not None else arr.max(axis=0)
    span = np.maximum(hi - lo, 1.0)
    z = (arr - lo) / span
    return z, lo, hi


def grid_key(z_row: Sequence[float], eps: float) -> Tuple[int, ...]:
    e = max(float(eps), 1e-12)
    return tuple(int(np.floor(float(v) / e)) for v in z_row)


@dataclass
class GridCell:
    cost: Cost
    path: Optional[List[int]]
    z: Tuple[float, ...]
    score: float


def representative_score(cost: Sequence[int], z: Sequence[float]) -> float:
    """Smaller is better: prefer closer to ideal in normalized L1, then smaller c1."""
    return float(sum(z)) + 1e-9 * float(cost[0])


class EpsilonGrid:
    """Keep at most one representative per ε-box (ε-approximate front compression)."""

    def __init__(self, m: int, eps: float, z_min: Sequence[float], z_max: Sequence[float]):
        self.m = int(m)
        self.eps = float(eps)
        self.z_min = np.asarray(z_min, dtype=float)
        self.z_max = np.asarray(z_max, dtype=float)
        self.span = np.maximum(self.z_max - self.z_min, 1.0)
        self.cells: Dict[Tuple[int, ...], GridCell] = {}

    def _normalize_one(self, cost: Sequence[int]) -> Tuple[float, ...]:
        return tuple(float((cost[i] - self.z_min[i]) / self.span[i]) for i in range(self.m))

    def insert(self, cost: Sequence[int], path: Optional[List[int]] = None) -> bool:
        """Insert/replace cell representative. Returns True if cell content changed."""
        full = tuple(int(x) for x in cost)
        z = self._normalize_one(full)
        key = grid_key(z, self.eps)
        score = representative_score(full, z)
        old = self.cells.get(key)
        if old is None or score < old.score:
            self.cells[key] = GridCell(cost=full, path=path, z=z, score=score)
            return True
        return False

    def insert_many(
        self,
        costs: Sequence[Sequence[int]],
        paths: Optional[Sequence[Optional[List[int]]]] = None,
    ) -> int:
        n = 0
        for i, c in enumerate(costs):
            p = None if paths is None else paths[i]
            if self.insert(c, p):
                n += 1
        return n

    def export(self) -> Tuple[List[Cost], List[Optional[List[int]]]]:
        items = list(self.cells.values())
        items.sort(key=lambda g: g.cost[: self.m])
        return [g.cost for g in items], [g.path for g in items]


def epsilon_covers(approx: Sequence[Sequence[int]], ref: Sequence[Sequence[int]], m: int, eps: float,
                   z_min: Sequence[float], z_max: Sequence[float]) -> float:
    """
    Fraction of reference points that are ε-covered by approx:
    exists a in approx s.t. a_i <= (1+ε)*r_i  (on raw costs; additive on normalized z).
    Here we use additive coverage in normalized space: a_z_i <= r_z_i + eps for all i.
    """
    if not ref:
        return 1.0
    span = np.maximum(np.asarray(z_max, float) - np.asarray(z_min, float), 1.0)
    lo = np.asarray(z_min, float)

    def nz(c):
        return (np.asarray(c[:m], float) - lo) / span

    A = [nz(a) for a in approx]
    covered = 0
    for r in ref:
        rz = nz(r)
        ok = False
        for az in A:
            if np.all(az <= rz + eps):
                ok = True
                break
        if ok:
            covered += 1
    return covered / len(ref)


def build_epsilon_front(
    costs: Sequence[Sequence[int]],
    paths: Sequence[Optional[List[int]]],
    m: int,
    eps: float,
    z_min: Optional[Sequence[float]] = None,
    z_max: Optional[Sequence[float]] = None,
) -> Tuple[List[Cost], List[Optional[List[int]]], np.ndarray, np.ndarray]:
    """ND filter → ε-grid compression → ND filter again."""
    nd_c, nd_p = filter_nondominated(costs, m, paths)
    if not nd_c:
        return [], [], np.zeros(m), np.ones(m)
    _, lo, hi = normalize_costs(nd_c, m, z_min, z_max)
    grid = EpsilonGrid(m, eps, lo, hi)
    grid.insert_many(nd_c, nd_p)
    gc, gp = grid.export()
    out_c, out_p = filter_nondominated(gc, m, gp)
    order = sorted(range(len(out_c)), key=lambda i: out_c[i][:m])
    return [out_c[i] for i in order], [out_p[i] for i in order], lo, hi
