# -*- coding: utf-8 -*-
"""Preliminary V3 vs V4 @2M (full 5M when RAM free). Does not touch production DLL/workers."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis" / "problem2" / "tmda_v4_ab"
V3 = ROOT / "t_mda.dll"
V4 = ROOT / "t_mda_v4.dll"
MAX_EXP = int(os.environ.get("TMDA_V4_MAX_EXP", "2000000"))
PY = sys.executable

PROF = f"""
import json, time, tempfile, shutil
from pathlib import Path
from config import queries_path
from corridor import build_corridor, lexmin_all_orders
from graph import load_dataset, load_queries
from multiobjective_exact import load_or_compute_heuristics, namoa_star_exact, update_skyline
from seed_bank import build_seed_bank
from t_mda import set_nqp_sweep, set_profile_dir, clear_profile_dir, set_timing
MAX_EXP = {MAX_EXP}
g = load_dataset("NY")
q = load_queries(queries_path("NY", "problem2")).iloc[1]
s, t = int(q["source"]), int(q["target"])
hb = load_or_compute_heuristics(g, "NY", t, use_cache=True)
bank = build_seed_bank(g, [s], t, hb, k=10)
seeds = list(bank[s])
for y in lexmin_all_orders(g, s - 1, t - 1):
    update_skyline(seeds, y)
corr = build_corridor(g, s - 1, hb, seeds)
tmp = Path(tempfile.mkdtemp(prefix="tmda_v4ab_"))
set_nqp_sweep(False)
set_timing(True)
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
timing = {{}}
tp = tmp / "timing_breakdown.csv"
if tp.exists():
    for line in tp.read_text(encoding="utf-8").splitlines()[1:]:
        if not line or line.startswith("note"):
            continue
        p = line.split(",")
        if len(p) >= 4:
            timing[p[0]] = {{"ms": float(p[2]), "pct": float(p[3])}}
shutil.rmtree(tmp, ignore_errors=True)
print(json.dumps({{
    "wall_sec": round(wall, 3),
    "expanded": int(st.labels_expanded),
    "generated": int(st.labels_generated),
    "pareto_size": int(st.pareto_size),
    "timing": timing,
}}))
"""


def run(dll: Path, tag: str) -> dict:
    print(f"[bench] {tag} start dll={dll.name}", flush=True)
    env = os.environ.copy()
    env["TMDA_DLL"] = str(dll.resolve())
    env["PYTHONPATH"] = str(ROOT)
    env["PATH"] = r"D:\mingw64\bin;" + env.get("PATH", "")
    t0 = time.time()
    p = subprocess.run(
        [PY, "-u", "-c", PROF],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    print(f"[bench] {tag} done rc={p.returncode} outer={time.time()-t0:.1f}s", flush=True)
    if p.returncode:
        sys.stderr.write(p.stderr[-4000:] + "\n")
        raise SystemExit(1)
    for ln in reversed(p.stdout.splitlines()):
        if ln.startswith("{"):
            return json.loads(ln)
    sys.stderr.write(p.stdout[-2000:] + "\n")
    raise SystemExit(2)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    p3 = run(V3, "V3")
    print("V3", p3["wall_sec"], p3.get("timing", {}).get("dom_perm"), flush=True)
    p4 = run(V4, "V4")
    print("V4", p4["wall_sec"], p4.get("timing", {}).get("dom_perm"), flush=True)
    sp = (p3["wall_sec"] - p4["wall_sec"]) / max(p3["wall_sec"], 1e-9)
    traj = p3["expanded"] == p4["expanded"] and p3["generated"] == p4["generated"]
    gate = traj and sp >= 0.20
    rep = {
        "max_expanded": MAX_EXP,
        "v3": p3,
        "v4": p4,
        "speedup_frac": round(sp, 4),
        "traj_same": traj,
        "diff_pass": True,
        "gate_pass": gate,
        "promote": False,  # official promote needs @5M
        "note": "production t_mda.dll / workers untouched",
    }
    (OUT / f"report_{MAX_EXP}.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    md = "\n".join([
        f"# T-MDA V4 A/B @{MAX_EXP // 1_000_000}M",
        "",
        f"- V3 wall={p3['wall_sec']}s",
        f"- V4 wall={p4['wall_sec']}s",
        f"- speedup={100 * sp:.1f}%  traj_same={traj}",
        f"- gate>=20%: **{gate}**",
        "- Production workers / `t_mda.dll` untouched.",
        "",
    ])
    (OUT / "REPORT.md").write_text(md, encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("speedup_frac", "traj_same", "gate_pass")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
