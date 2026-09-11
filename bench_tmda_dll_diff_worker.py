# -*- coding: utf-8 -*-
"""Subprocess worker for bench_tmda_dll_diff (TMDA_DLL must be set before import)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multiobjective_exact import namoa_star_exact  # noqa: E402
from tests.test_t_mda import (  # noqa: E402
    random_dag,
    toy_chain_diamond,
    toy_multi_in,
    toy_zero_elev,
)


def main() -> int:
    case = json.loads(sys.stdin.read())
    if case["kind"] == "toy":
        g = {
            "diamond": toy_chain_diamond,
            "zero": toy_zero_elev,
            "multi": toy_multi_in,
        }[case["graph"]]()
    else:
        g = random_dag(case["n"], seed=case["seed"])
    res = namoa_star_exact(g, case["s"], case["t"], backend="t_mda")
    sols = sorted(set(res.solutions))
    st = res.stats
    out = {
        "name": case["name"],
        "exact": bool(st.exact_finished),
        "sols": sols,
        "expanded": int(st.labels_expanded),
        "generated": int(st.labels_generated),
        "pareto_size": len(sols),
        "min0": min(x[0] for x in sols) if sols else None,
        "min1": min(x[1] for x in sols) if sols else None,
        "min2": min(x[2] for x in sols) if sols else None,
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
