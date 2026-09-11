# -*- coding: utf-8 -*-
"""Hard-query T-MDA profiler (exact-preserving timers). Does not write fragments."""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

from config import ANALYSIS_DIR, queries_path
from corridor import build_corridor, lexmin_all_orders
from graph import load_dataset, load_queries
from multiobjective_exact import load_or_compute_heuristics, namoa_star_exact, update_skyline
from seed_bank import build_seed_bank
from t_mda import clear_profile_dir, ensure_dll, set_nqp_sweep, set_profile_dir

OUT = ANALYSIS_DIR / "problem2" / "profiler_NY_0002"
MAX_EXP = 5_000_000


def main():
    ensure_dll()
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.gettempdir()) / "tmda_prof_NY0002"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    g = load_dataset("NY")
    q = load_queries(queries_path("NY", "problem2")).iloc[1]  # 0002 hard
    s, t = int(q["source"]), int(q["target"])
    qid = str(q["query_id"]).zfill(4)
    print(f"[prof] NY {qid} {s}->{t} stop@{MAX_EXP}", flush=True)

    hb = load_or_compute_heuristics(g, "NY", t, use_cache=True)
    bank = build_seed_bank(g, [s], t, hb, k=10)
    seeds = list(bank[s])
    for y in lexmin_all_orders(g, s - 1, t - 1):
        update_skyline(seeds, y)
    corr = build_corridor(g, s - 1, hb, seeds)

    set_nqp_sweep(False)
    set_profile_dir(str(tmp))
    t0 = time.time()
    try:
        res = namoa_star_exact(
            g, s, t, heuristics=hb, use_heuristic=True,
            extra_seeds=seeds, edge_allowed=corr.edge_allowed.copy(),
            max_labels=0, max_expanded=MAX_EXP, backend="t_mda",
        )
    finally:
        clear_profile_dir()
    wall = time.time() - t0
    st = res.stats

    for f in tmp.glob("*"):
        shutil.copy2(f, OUT / f.name)

    timing = {}
    tp = OUT / "timing_breakdown.csv"
    if tp.exists():
        for line in tp.read_text(encoding="utf-8").splitlines()[1:]:
            if not line or line.startswith("note"):
                continue
            parts = line.split(",")
            if len(parts) >= 4:
                timing[parts[0]] = {
                    "ms": float(parts[2]),
                    "pct": float(parts[3]),
                    "count": parts[4] if len(parts) > 4 else "",
                }

    report = {
        "query": f"NY_{qid}",
        "max_expanded": MAX_EXP,
        "wall_sec": round(wall, 2),
        "expanded": st.labels_expanded,
        "generated": st.labels_generated,
        "peak_open": st.open_peak,
        "peak_nqp": st.u_bound_size,
        "nqp_live": st.stale_skipped,
        "pareto_size": st.pareto_size,
        "exact": bool(st.exact_finished),
        "timing": timing,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        f"# T-MDA Profiler — NY_{qid} @{MAX_EXP // 1_000_000}M",
        "",
        "## Status",
        "Diagnostic only — NOT EXACT fragment. Exact-preserving timers.",
        "",
        f"- wall={wall:.2f}s expanded={st.labels_expanded} generated={st.labels_generated}",
        f"- peak_open={st.open_peak} peak_nqp={st.u_bound_size} |S|={st.pareto_size}",
        "",
        "## Time breakdown (% of solver wall)",
        "",
        "| component | ms | % wall | count |",
        "|-----------|---:|-------:|------:|",
    ]
    order = [
        "dominance_total", "dom_goal", "dom_perm", "nqp_next_queue",
        "propagate", "heap", "extract_perm", "wall_total",
    ]
    for k in order:
        if k not in timing:
            continue
        v = timing[k]
        md.append(f"| {k} | {v['ms']:.1f} | {v['pct']:.1f} | {v.get('count','')} |")
    md += [
        "",
        "Note: `nqp_next_queue` and `propagate` **include** nested dominance time.",
        "",
        "## Next cut",
    ]
    dom = timing.get("dominance_total", {}).get("pct", 0)
    nqp = timing.get("nqp_next_queue", {}).get("pct", 0)
    prop = timing.get("propagate", {}).get("pct", 0)
    heap = timing.get("heap", {}).get("pct", 0)
    if dom >= 40:
        md.append(f"- **Primary: dominance ({dom:.0f}%)** → stronger skyline / Pst index.")
    if nqp >= 30:
        md.append(f"- **NQP path ({nqp:.0f}%)** → list layout / fewer copies.")
    if prop >= 50:
        md.append(f"- **propagate ({prop:.0f}%)** → dominated by nested work; see dom vs rest.")
    if heap >= 20:
        md.append(f"- **heap ({heap:.0f}%)** → decrease-key / fewer stale pushes.")
    if max(dom, nqp, heap) < 25:
        md.append("- No single >25% bucket; consider multi-query parallel or alt exact engine.")
    (OUT / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md), flush=True)
    print(f"[prof] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
