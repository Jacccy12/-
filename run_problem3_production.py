# -*- coding: utf-8 -*-
"""
Fast production runner (C++ weighted sampling + ε-grid on m=5).

Defaults mirror problem3jsy dense budgets: 512 / 2048 / 4096.
Exact T-MDA is optional and much slower — not used here.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
LOG = ROOT / "analysis" / "problem3" / "production_run_fast.log"


def run(cmd: list[str]) -> int:
    print("\n===", " ".join(cmd), "===\n", flush=True)
    with LOG.open("a", encoding="utf-8") as log:
        log.write("\n=== " + " ".join(cmd) + " ===\n")
        log.flush()
        p = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return p.wait()


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    # warm-build C++ DLL once in parent before workers spawn
    from solver.p3_weighted_cpp import ensure_dll

    ensure_dll()

    parts = ROOT / "results" / "problem3_parts"
    parts.mkdir(parents=True, exist_ok=True)

    common = [
        PY,
        "-u",
        "problem3.py",
        "--datasets",
        "NY,BAY,COL",
        "--approx-only",  # C++ weighted ND (m=2/3) / ε-grid (m=5)
        "--workers",
        "3",
    ]

    phases = [
        ["--objectives", "2", "--samples2", "512"],
        ["--objectives", "3", "--samples3", "2048"],
        ["--objectives", "5", "--samples5", "4096", "--eps", "0.05"],
    ]
    for extra in phases:
        rc = run(common + extra)
        if rc != 0:
            return rc

    rc = run([PY, "-u", "problem3.py", "--merge-only"])
    print("[production-fast] ALL DONE", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
