# -*- coding: utf-8 -*-
"""Pareto quality metrics: hypervolume (MC) and ε-coverage."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from solver.epsilon_grid import epsilon_covers, normalize_costs


def hypervolume_monte_carlo(
    costs: Sequence[Sequence[int]],
    m: int,
    z_min: Sequence[float],
    z_max: Sequence[float],
    samples: int = 20000,
    seed: int = 0,
) -> float:
    """
    Estimate normalized HV in [0,1]^m w.r.t. reference point (1,...,1)
    after mapping costs with (z_min, z_max). Minimization: dominated region
    toward the ideal (0,...,0).
    """
    if not costs:
        return 0.0
    z, _, _ = normalize_costs(costs, m, z_min, z_max)
    z = np.clip(z, 0.0, 1.0)
    rng = np.random.default_rng(seed)
    pts = rng.random((samples, m))
    # point p is dominated by front if exists s with s <= p componentwise
    # (s closer to 0). Count fraction of unit cube dominated.
    dominated = np.zeros(samples, dtype=bool)
    for row in z:
        dominated |= np.all(pts >= row[None, :], axis=1)
    return float(dominated.mean())


def evaluate_front(
    costs: Sequence[Sequence[int]],
    m: int,
    *,
    ref_costs: Optional[Sequence[Sequence[int]]] = None,
    eps: float = 0.05,
    hv_samples: int = 20000,
    seed: int = 0,
) -> Dict[str, float]:
    """Return quality dict for one front."""
    if not costs:
        return {"n": 0, "hv": 0.0, "eps_coverage": 0.0}
    pool = list(costs) if ref_costs is None else list(costs) + list(ref_costs)
    _, lo, hi = normalize_costs(pool, m)
    hv = hypervolume_monte_carlo(costs, m, lo, hi, samples=hv_samples, seed=seed)
    cov = 1.0
    if ref_costs:
        cov = epsilon_covers(costs, ref_costs, m, eps, lo, hi)
    return {
        "n": float(len(costs)),
        "hv": hv,
        "eps_coverage": cov,
        "z_min_sum": float(np.sum(lo)),
        "z_max_sum": float(np.sum(hi)),
    }
