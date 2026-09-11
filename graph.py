# -*- coding: utf-8 -*-
"""CSR directed road network with five edge costs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from config import cache_graph_path, edges_path


@dataclass
class RoadGraph:
    """0-indexed CSR graph. External node IDs are 1..n_nodes."""

    n_nodes: int
    n_edges: int
    offset: np.ndarray
    to: np.ndarray
    w: np.ndarray
    rev_offset: np.ndarray
    rev_to: np.ndarray
    rev_w: np.ndarray
    dataset: str = ""

    def out_neighbors(self, u: int) -> Tuple[np.ndarray, np.ndarray]:
        a, b = int(self.offset[u]), int(self.offset[u + 1])
        return self.to[a:b], self.w[a:b]

    def edge_cost(self, u: int, v: int) -> Optional[np.ndarray]:
        tos, ws = self.out_neighbors(u)
        # linear scan; degrees are tiny on road networks
        for i in range(tos.shape[0]):
            if int(tos[i]) == v:
                return ws[i]
        return None

    def evaluate_path(self, path_nodes_0: Sequence[int]) -> np.ndarray:
        costs = np.zeros(5, dtype=np.int64)
        if len(path_nodes_0) < 2:
            return costs
        for i in range(len(path_nodes_0) - 1):
            u, v = int(path_nodes_0[i]), int(path_nodes_0[i + 1])
            c = self.edge_cost(u, v)
            if c is None:
                raise ValueError(f"Missing directed edge {u + 1}->{v + 1}")
            costs += c
        return costs

    def has_edge(self, u: int, v: int) -> bool:
        return self.edge_cost(u, v) is not None

    def without_edges(self, closed: Iterable[Tuple[int, int]]) -> "RoadGraph":
        closed_set = {(int(u), int(v)) for u, v in closed}
        if not closed_set:
            return self
        mask = np.ones(self.n_edges, dtype=bool)
        # rebuild src array from offset
        src = np.empty(self.n_edges, dtype=np.int32)
        for u in range(self.n_nodes):
            a, b = int(self.offset[u]), int(self.offset[u + 1])
            src[a:b] = u
            for e in range(a, b):
                if (u, int(self.to[e])) in closed_set:
                    mask[e] = False
        return build_csr(
            self.n_nodes,
            src[mask],
            self.to[mask],
            self.w[mask],
            dataset=self.dataset,
        )


def build_csr(
    n_nodes: int,
    src0: np.ndarray,
    dst0: np.ndarray,
    w: np.ndarray,
    dataset: str = "",
) -> RoadGraph:
    """Build forward+reverse CSR from 0-indexed edge arrays."""
    src0 = np.asarray(src0, dtype=np.int32)
    dst0 = np.asarray(dst0, dtype=np.int32)
    w = np.asarray(w, dtype=np.int64)
    n_edges = int(src0.shape[0])

    order = np.argsort(src0, kind="mergesort")
    src_s = src0[order]
    dst_s = dst0[order]
    w_s = w[order]

    offset = np.zeros(n_nodes + 1, dtype=np.int64)
    np.add.at(offset, src_s + 1, 1)
    np.cumsum(offset, out=offset)

    # reverse: original u->v becomes rev v->u with same weights
    rev_src = dst0
    rev_dst = src0
    rev_order = np.argsort(rev_src, kind="mergesort")
    rev_src_s = rev_src[rev_order]
    rev_dst_s = rev_dst[rev_order]
    rev_w_s = w[rev_order]

    rev_offset = np.zeros(n_nodes + 1, dtype=np.int64)
    np.add.at(rev_offset, rev_src_s + 1, 1)
    np.cumsum(rev_offset, out=rev_offset)

    return RoadGraph(
        n_nodes=n_nodes,
        n_edges=n_edges,
        offset=offset,
        to=dst_s,
        w=w_s,
        rev_offset=rev_offset,
        rev_to=rev_dst_s,
        rev_w=rev_w_s,
        dataset=dataset,
    )


def load_edges_txt(path: Path) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Parse merged 5-obj edge file. Returns n_nodes, src, dst, w (1-indexed)."""
    import pandas as pd

    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=["src", "dst", "c1", "c2", "c3", "c4", "c5"],
        dtype=np.int64,
        engine="c",
    )
    src = df["src"].to_numpy(dtype=np.int32)
    dst = df["dst"].to_numpy(dtype=np.int32)
    w = df[["c1", "c2", "c3", "c4", "c5"]].to_numpy(dtype=np.int64)
    n_nodes = int(max(int(src.max()), int(dst.max())))
    return n_nodes, src, dst, w


def save_graph_cache(graph: RoadGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        n_nodes=np.int64(graph.n_nodes),
        n_edges=np.int64(graph.n_edges),
        offset=graph.offset,
        to=graph.to,
        w=graph.w,
        rev_offset=graph.rev_offset,
        rev_to=graph.rev_to,
        rev_w=graph.rev_w,
        dataset=np.asarray(graph.dataset),
    )


def load_graph_cache(path: Path) -> RoadGraph:
    z = np.load(path, allow_pickle=True)
    dataset = str(z["dataset"])
    return RoadGraph(
        n_nodes=int(z["n_nodes"]),
        n_edges=int(z["n_edges"]),
        offset=z["offset"],
        to=z["to"],
        w=z["w"],
        rev_offset=z["rev_offset"],
        rev_to=z["rev_to"],
        rev_w=z["rev_w"],
        dataset=dataset,
    )


def load_dataset(dataset: str, use_cache: bool = True, rebuild_cache: bool = False) -> RoadGraph:
    cache = cache_graph_path(dataset)
    if use_cache and cache.exists() and not rebuild_cache:
        g = load_graph_cache(cache)
        g.dataset = dataset
        print(f"[load] cache hit {dataset}: nodes={g.n_nodes}, edges={g.n_edges}")
        return g

    path = edges_path(dataset)
    print(f"[load] reading {path}")
    n_nodes, src, dst, w = load_edges_txt(path)
    print(f"[load] {dataset}: nodes={n_nodes}, edges={len(src)}")
    g = build_csr(n_nodes, src - 1, dst - 1, w, dataset=dataset)
    if use_cache:
        print(f"[load] writing cache {cache}")
        save_graph_cache(g, cache)
    return g


def load_queries(path: Path):
    import pandas as pd

    df = pd.read_csv(path, dtype={"query_id": str})
    df["source"] = df["source"].astype(np.int64)
    df["target"] = df["target"].astype(np.int64)
    return df
