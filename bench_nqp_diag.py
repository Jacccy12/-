# -*- coding: utf-8 -*-
"""
NQP / permanent diagnostic for T-MDA on NY0001 (controlled budget).

Does NOT write EXACT fragments. Produces analysis under
analysis/problem2/nqp_diag_NY_0001/
"""
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
from t_mda import clear_profile_dir, ensure_dll, set_profile_dir

OUT = ANALYSIS_DIR / "problem2" / "nqp_diag_NY_0001"
MAX_EXP = 3_000_000


def main():
    ensure_dll()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    # DLL fopen on Windows cannot open UTF-8 Chinese paths reliably.
    tmp_prof = Path(tempfile.gettempdir()) / "tmda_nqp_diag_NY0001"
    if tmp_prof.exists():
        shutil.rmtree(tmp_prof)
    tmp_prof.mkdir(parents=True)

    g = load_dataset("NY")
    q = load_queries(queries_path("NY", "problem2")).iloc[0]
    s, t = int(q["source"]), int(q["target"])
    print(f"[nqp-diag] NY0001 {s}->{t} stop@{MAX_EXP}", flush=True)
    print(f"[nqp-diag] profile_dir={tmp_prof}", flush=True)

    hb = load_or_compute_heuristics(g, "NY", t, use_cache=True)
    bank = build_seed_bank(g, [s], t, hb, k=10)
    seeds = list(bank[s])
    for y in lexmin_all_orders(g, s - 1, t - 1):
        update_skyline(seeds, y)
    corr = build_corridor(g, s - 1, hb, seeds)

    set_profile_dir(str(tmp_prof))
    t0 = time.time()
    try:
        res = namoa_star_exact(
            g, s, t, heuristics=hb, use_heuristic=True,
            extra_seeds=seeds, edge_allowed=corr.edge_allowed.copy(),
            max_labels=80_000_000, max_expanded=MAX_EXP,
            backend="t_mda",
        )
    finally:
        clear_profile_dir()
    wall = time.time() - t0
    st = res.stats

    for f in tmp_prof.glob("*"):
        shutil.copy2(f, OUT / f.name)
    print(f"[nqp-diag] copied profile files -> {OUT}", flush=True)

    summary_path = OUT / "nqp_summary.csv"
    summary = {}
    if summary_path.exists():
        for line in summary_path.read_text(encoding="utf-8").splitlines()[1:]:
            if "," not in line:
                continue
            k, v = line.split(",", 1)
            summary[k] = v
    else:
        print("[nqp-diag] WARNING: nqp_summary.csv missing", flush=True)

    report = {
        "status": "DIAGNOSTIC_BENCH_LIMIT",
        "exact": False,
        "max_expanded": MAX_EXP,
        "wall_sec": round(wall, 2),
        "generated": st.labels_generated,
        "expanded": st.labels_expanded,
        "peak_open": st.open_peak,
        "peak_live": st.labels_peak,
        "peak_nqp": st.u_bound_size,
        "nqp_live_end": st.stale_skipped,
        "dll_summary": summary,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    stale_frac = summary.get("nqp_stale_frac", "?")
    md = f"""# T-MDA NQP Diagnostic — NY0001 @{MAX_EXP // 1_000_000}M

## Status
**DIAGNOSTIC only — NOT EXACT.** No fragment written.

Full-run abort: `exact_NY_0001_t_mda_DIAGNOSTIC_STOP.md` (~20M exp, OPEN rising, NQP≈2.69M).

## This controlled run
| metric | value |
|--------|------:|
| expanded | {st.labels_expanded} |
| generated | {st.labels_generated} |
| peak_OPEN | {st.open_peak} |
| peak_NQP | {st.u_bound_size} |
| nqp_live (end) | {summary.get('nqp_live', st.stale_skipped)} |
| nqp_enter | {summary.get('nqp_enter', '?')} |
| nqp_reject | {summary.get('nqp_reject', '?')} |
| nqp_promote | {summary.get('nqp_promote', '?')} |
| reject/(enter) | {summary.get('nqp_reject_rate', '?')} |
| avg wait reject (exp) | {summary.get('avg_wait_reject_exp', '?')} |
| avg wait promote (exp) | {summary.get('avg_wait_promote_exp', '?')} |
| avg wait live (exp) | {summary.get('avg_wait_live_exp', '?')} |
| **NQP already stale (perm)** | {summary.get('nqp_stale_perm', '?')} |
| **NQP already stale (goal)** | {summary.get('nqp_stale_goal', '?')} |
| NQP still fresh | {summary.get('nqp_fresh', '?')} |
| **stale fraction of live NQP** | **{stale_frac}** |
| perm_total | {summary.get('perm_total', '?')} |
| max_perm_node | {summary.get('max_perm_node', '?')} |
| wall_s | {wall:.1f} |

## Files
- `nqp_growth.csv`, `nqp_len_hist.csv`, `nqp_top100.csv`, `perm_front_hist.csv`, `nqp_summary.csv`

## Interpretation
High `nqp_stale_frac` → safe early NQP cleanup is next.
Low → NQP mostly still viable → need indexing / other exact filters.
"""
    (OUT / "nqp_diag_report.md").write_text(md, encoding="utf-8")
    print(md, flush=True)
    print("[nqp-diag] wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
