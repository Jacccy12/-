# -*- coding: utf-8 -*-
"""ctypes wrapper for p3_weighted_core.dll (fast candidate generation)."""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

from graph import RoadGraph

ROOT = Path(__file__).resolve().parent
DLL = ROOT / ("p3_weighted_core.dll" if sys.platform == "win32" else "p3_weighted_core.so")
SRC = ROOT / "p3_weighted_core.cpp"

Cost5 = Tuple[int, int, int, int, int]
_LIB = None


def ensure_dll() -> Path:
    if DLL.exists() and DLL.stat().st_mtime >= SRC.stat().st_mtime:
        return DLL
    cmd = [
        "g++",
        "-O3",
        "-march=native",
        "-shared",
        "-std=c++17",
        "-static-libgcc",
        "-static-libstdc++",
        str(SRC),
        "-o",
        str(DLL),
    ]
    print("[build]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))
    return DLL


def _load():
    global _LIB
    if _LIB is not None:
        return _LIB
    path = ensure_dll()
    if sys.platform == "win32":
        mingw = Path(r"D:\mingw64\bin")
        if mingw.exists():
            os.add_dll_directory(str(mingw))
    lib = ctypes.CDLL(str(path))
    lib.p3_weighted_generate.argtypes = [
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.p3_weighted_generate.restype = ctypes.c_int
    _LIB = lib
    return _LIB


def generate_weighted_candidates_cpp(
    graph: RoadGraph,
    source_1: int,
    target_1: int,
    m: int,
    n_samples: int,
    max_out: int = 50000,
) -> Tuple[List[Cost5], List[List[int]], np.ndarray, np.ndarray]:
    lib = _load()
    s = int(source_1) - 1
    t = int(target_1) - 1
    offset = np.ascontiguousarray(graph.offset, dtype=np.int64)
    to = np.ascontiguousarray(graph.to, dtype=np.int32)
    w5 = np.ascontiguousarray(graph.w.reshape(-1), dtype=np.int64)
    out_costs = np.zeros(max_out * 5, dtype=np.int64)
    out_len = np.zeros(max_out, dtype=np.int32)
    path_cap = max(max_out * 2048, 2_000_000)
    out_paths = np.zeros(path_cap, dtype=np.int32)
    path_used = np.zeros(1, dtype=np.int32)

    nsol = int(
        lib.p3_weighted_generate(
            ctypes.c_int32(graph.n_nodes),
            offset.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            to.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            w5.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ctypes.c_int32(s),
            ctypes.c_int32(t),
            ctypes.c_int32(int(m)),
            ctypes.c_int32(int(n_samples)),
            out_costs.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            out_len.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            out_paths.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(max_out),
            ctypes.c_int32(path_cap),
            path_used.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
    )
    if nsol < 0:
        raise RuntimeError(f"p3_weighted_generate failed: {nsol}")

    costs: List[Cost5] = []
    paths: List[List[int]] = []
    cursor = 0
    for i in range(nsol):
        c = (
            int(out_costs[i * 5 + 0]),
            int(out_costs[i * 5 + 1]),
            int(out_costs[i * 5 + 2]),
            int(out_costs[i * 5 + 3]),
            int(out_costs[i * 5 + 4]),
        )
        L = int(out_len[i])
        p = [int(x) for x in out_paths[cursor : cursor + L]]
        cursor += L
        costs.append(c)
        paths.append(p)

    arr = np.asarray(costs, dtype=float)
    z_min = arr[:, :m].min(axis=0)
    z_max = arr[:, :m].max(axis=0)
    return costs, paths, z_min, z_max
