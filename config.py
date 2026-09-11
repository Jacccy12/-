# -*- coding: utf-8 -*-
"""Project paths and constants for multi-objective road routing."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
EXAMPLES_DIR = PROJECT_ROOT / "examples"

for _d in (CACHE_DIR, RESULTS_DIR, ANALYSIS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATASETS = ("NY", "BAY", "COL")
OBJECTIVES = ("distance", "travel_time", "elevation", "avg_degree", "hop_count")
OBJECTIVE_INDEX = {name: i for i, name in enumerate(OBJECTIVES)}


def resolve_data_root() -> Path:
    """Locate original package data folder (handles ZIP filename mojibake)."""
    pointer = PROJECT_ROOT / "DATA_ROOT.txt"
    if pointer.exists():
        p = Path(pointer.read_text(encoding="utf-8").strip())
        if (p / "edges" / "edges_NY_5obj.txt").exists():
            return p

    local = PROJECT_ROOT / "data"
    if (local / "edges" / "edges_NY_5obj.txt").exists():
        return local

    # Search siblings / parents for the edges file.
    search_roots = [PROJECT_ROOT.parent, PROJECT_ROOT]
    for root in search_roots:
        for hit in root.rglob("edges_NY_5obj.txt"):
            return hit.parent.parent
    raise FileNotFoundError("Cannot locate edges_NY_5obj.txt / data root")


DATA_ROOT = resolve_data_root()


def edges_path(dataset: str) -> Path:
    return DATA_ROOT / "edges" / f"edges_{dataset}_5obj.txt"


def queries_path(dataset: str, problem: str) -> Path:
    """problem in {'problem1','problem2','problem34'}."""
    city = dataset.lower()
    # Prefer project-local copies, fall back to package data.
    local = PROJECT_ROOT / "data" / f"dimacs5_{city}" / f"queries_{problem}.csv"
    if local.exists():
        return local
    return DATA_ROOT / f"dimacs5_{city}" / f"queries_{problem}.csv"


def closed_edges_path(dataset: str) -> Path:
    city = dataset.lower()
    local = PROJECT_ROOT / "data" / f"dimacs5_{city}" / "closed_edges_problem4.csv"
    if local.exists():
        return local
    return DATA_ROOT / f"dimacs5_{city}" / "closed_edges_problem4.csv"


def cache_graph_path(dataset: str) -> Path:
    return CACHE_DIR / f"{dataset}_graph.npz"
