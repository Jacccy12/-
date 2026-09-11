# -*- coding: utf-8 -*-
"""ctypes wrapper for namoa_dr_lazy_core (NAMOA*_dr-lazy)."""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
DLL_PATH = ROOT / (
    "namoa_dr_lazy.dll" if sys.platform == "win32" else "namoa_dr_lazy.so"
)
SRC_PATH = ROOT / "namoa_dr_lazy_core.cpp"


def ensure_dll() -> Path:
    if DLL_PATH.exists() and DLL_PATH.stat().st_mtime >= SRC_PATH.stat().st_mtime:
        return DLL_PATH
    gpp = "g++"
    cmd = [
        gpp, "-O3", "-shared", "-std=c++17",
        "-static-libgcc", "-static-libstdc++",
        str(SRC_PATH), "-o", str(DLL_PATH),
    ]
    print("[build]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))
    return DLL_PATH


def _load_lib():
    path = ensure_dll()
    if sys.platform == "win32":
        mingw = Path(r"D:\mingw64\bin")
        if mingw.exists():
            os.add_dll_directory(str(mingw))
    lib = ctypes.CDLL(str(path))
    lib.namoa_dr_lazy_search.argtypes = [
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int32,
        ctypes.c_int64,
    ]
    lib.namoa_dr_lazy_search.restype = ctypes.c_int
    return lib


_LIB = None


def namoa_dr_lazy(
    offset: np.ndarray,
    to: np.ndarray,
    w0: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    h0: np.ndarray,
    h1: np.ndarray,
    h2: np.ndarray,
    source: int,
    target: int,
    use_h: bool,
    seed0: np.ndarray,
    seed1: np.ndarray,
    seed2: np.ndarray,
    edge_allowed: Optional[np.ndarray] = None,
    max_out: int = 200000,
    max_labels: int = 50_000_000,
    max_expanded: int = 0,
):
    global _LIB
    if _LIB is None:
        _LIB = _load_lib()

    n_seed = int(seed0.shape[0])
    out_c1 = np.zeros(max_out, dtype=np.int64)
    out_c2 = np.zeros(max_out, dtype=np.int64)
    out_c3 = np.zeros(max_out, dtype=np.int64)
    stats = np.zeros(32, dtype=np.int64)

    def p64(a):
        return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))

    def p32(a):
        return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    if edge_allowed is None:
        edge_ptr = ctypes.POINTER(ctypes.c_uint8)()
        ea_keep = None
    else:
        ea_keep = np.ascontiguousarray(edge_allowed.astype(np.uint8))
        edge_ptr = ea_keep.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    nsol = _LIB.namoa_dr_lazy_search(
        ctypes.c_int32(int(offset.shape[0] - 1)),
        p64(np.ascontiguousarray(offset, dtype=np.int64)),
        p32(np.ascontiguousarray(to, dtype=np.int32)),
        p64(np.ascontiguousarray(w0, dtype=np.int64)),
        p64(np.ascontiguousarray(w1, dtype=np.int64)),
        p64(np.ascontiguousarray(w2, dtype=np.int64)),
        p64(np.ascontiguousarray(h0, dtype=np.int64)),
        p64(np.ascontiguousarray(h1, dtype=np.int64)),
        p64(np.ascontiguousarray(h2, dtype=np.int64)),
        ctypes.c_int32(source),
        ctypes.c_int32(target),
        ctypes.c_int32(1 if use_h else 0),
        p64(np.ascontiguousarray(seed0, dtype=np.int64)),
        p64(np.ascontiguousarray(seed1, dtype=np.int64)),
        p64(np.ascontiguousarray(seed2, dtype=np.int64)),
        ctypes.c_int32(n_seed),
        edge_ptr,
        p64(out_c1),
        p64(out_c2),
        p64(out_c3),
        ctypes.c_int32(max_out),
        p64(stats),
        ctypes.c_int32(max_labels),
        ctypes.c_int64(int(max_expanded)),
    )
    status = int(stats[11])
    if nsol < 0 or status == 0 and int(stats[0]) < 0:
        return [], stats, 0
    n = int(stats[0])
    if n > max_out:
        n = max_out
    sols = list(zip(out_c1[:n].tolist(), out_c2[:n].tolist(), out_c3[:n].tolist()))
    return sols, stats, status
