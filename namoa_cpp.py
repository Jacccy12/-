# -*- coding: utf-8 -*-
"""ctypes wrapper for namoa_core (v3 API with suffix UB / max_expanded)."""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
# Prefer v3 (new API); fall back to rebuilding namoa_core.dll when unlocked.
DLL_V3 = ROOT / "namoa_core_v3.dll"
DLL_PATH = ROOT / ("namoa_core.dll" if sys.platform == "win32" else "namoa_core.so")
SRC_PATH = ROOT / "namoa_core.cpp"


def ensure_dll() -> Path:
    if DLL_V3.exists() and DLL_V3.stat().st_mtime >= SRC_PATH.stat().st_mtime:
        return DLL_V3
    if DLL_PATH.exists() and DLL_PATH.stat().st_mtime >= SRC_PATH.stat().st_mtime:
        # old API only — rebuild v3 beside it
        pass
    gpp = "g++"
    out = DLL_V3 if sys.platform == "win32" else ROOT / "namoa_core_v3.so"
    cmd = [
        gpp, "-O3", "-shared", "-std=c++17",
        "-static-libgcc", "-static-libstdc++",
        str(SRC_PATH), "-o", str(out),
    ]
    print("[build]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))
    return out


def _load_lib():
    path = ensure_dll()
    if sys.platform == "win32":
        mingw = Path(r"D:\mingw64\bin")
        if mingw.exists():
            os.add_dll_directory(str(mingw))
    lib = ctypes.CDLL(str(path))
    lib.namoa_search.argtypes = [
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
        ctypes.c_int64,  # max_expanded
        ctypes.POINTER(ctypes.c_int64),  # suffix_q
        ctypes.POINTER(ctypes.c_int64),  # ds0
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
    ]
    lib.namoa_search.restype = ctypes.c_int
    lib.namoa_search_profile.argtypes = list(lib.namoa_search.argtypes) + [
        ctypes.c_char_p,
    ]
    lib.namoa_search_profile.restype = ctypes.c_int
    lib.namoa_set_suffix_pareto.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int32,
    ]
    lib.namoa_set_suffix_pareto.restype = None
    lib.namoa_clear_suffix_pareto.argtypes = []
    lib.namoa_clear_suffix_pareto.restype = None
    return lib


_LIB = None
# Keep numpy arrays alive while DLL holds raw pointers
_SP_KEEP = None


def set_suffix_pareto(
    off: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    exact: np.ndarray,
    enable_join: bool = False,
) -> None:
    global _LIB, _SP_KEEP
    if _LIB is None:
        _LIB = _load_lib()
    _SP_KEEP = (
        np.ascontiguousarray(off, dtype=np.int64),
        np.ascontiguousarray(c0, dtype=np.int64),
        np.ascontiguousarray(c1, dtype=np.int64),
        np.ascontiguousarray(c2, dtype=np.int64),
        np.ascontiguousarray(exact, dtype=np.uint8),
    )
    off_a, c0_a, c1_a, c2_a, ex_a = _SP_KEEP

    def p64(a):
        return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))

    _LIB.namoa_set_suffix_pareto(
        p64(off_a),
        p64(c0_a),
        p64(c1_a),
        p64(c2_a),
        ex_a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int32(1 if enable_join else 0),
    )


def clear_suffix_pareto() -> None:
    global _LIB, _SP_KEEP
    if _LIB is None:
        _LIB = _load_lib()
    _LIB.namoa_clear_suffix_pareto()
    _SP_KEEP = None


def namoa_cpp(
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
    max_labels: int = 30_000_000,
    max_expanded: int = 0,
    suffix_q: Optional[np.ndarray] = None,
    ds0: Optional[np.ndarray] = None,
    ds1: Optional[np.ndarray] = None,
    ds2: Optional[np.ndarray] = None,
    profile_dir: Optional[str] = None,
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

    if suffix_q is None:
        sq_ptr = ctypes.POINTER(ctypes.c_int64)()
        sq_keep = None
    else:
        sq_keep = np.ascontiguousarray(suffix_q.reshape(-1).astype(np.int64))
        sq_ptr = sq_keep.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))

    def opt64(a):
        if a is None:
            return ctypes.POINTER(ctypes.c_int64)(), None
        arr = np.ascontiguousarray(a.astype(np.int64))
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), arr

    ds0_ptr, ds0_keep = opt64(ds0)
    ds1_ptr, ds1_keep = opt64(ds1)
    ds2_ptr, ds2_keep = opt64(ds2)

    args = [
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
        sq_ptr,
        ds0_ptr,
        ds1_ptr,
        ds2_ptr,
    ]
    if profile_dir:
        nsol = _LIB.namoa_search_profile(
            *args, profile_dir.encode("utf-8")
        )
    else:
        nsol = _LIB.namoa_search(*args)
    status = int(stats[11])
    if nsol < 0 or int(stats[0]) < 0:
        stats[0] = max(0, int(stats[0]))
        if int(stats[0]) < 0:
            stats[0] = 0
        status = 0
        return [], stats, status
    n = int(stats[0])
    if n > max_out:
        n = max_out
    sols = list(zip(out_c1[:n].tolist(), out_c2[:n].tolist(), out_c3[:n].tolist()))
    return sols, stats, status
