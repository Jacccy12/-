# -*- coding: utf-8 -*-
"""T-MDA V4 A/B: exact PermFront merge-sort tree vs production V3.

Does NOT overwrite t_mda.dll. Does NOT write problem2 fragments.
Gate: Pareto identical; NY0002@5M wall >=20% faster to promote.
"""
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
SRC4 = ROOT / "t_mda_core_v4.cpp"
PY = sys.executable
MAX_EXP = 5_000_000


def build_v4() -> None:
    if V4.exists() and V4.stat().st_mtime >= SRC4.stat().st_mtime:
        return
    cmd = [
        "g++", "-O3", "-march=native", "-shared", "-std=c++17",
        "-static-libgcc", "-static-libstdc++",
        str(SRC4), "-o", str(V4), "-lpsapi",
    ]
    print("[build]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def run_with_dll(dll: Path, script: str, env_extra=None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["TMDA_DLL"] = str(dll.resolve())
    env["PYTHONPATH"] = str(ROOT)
    env["PATH"] = r"D:\mingw64\bin;" + env.get("PATH", "")
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [PY, "-u", "-c", script],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=env,
    )


DIFF_SCRIPT = r"""
import json, sys
from multiobjective_exact import namoa_star_exact
from tests.test_t_mda import toy_chain_diamond, toy_zero_elev, toy_multi_in, random_dag
cases = [
    ("diamond", toy_chain_diamond(), 1, 4),
    ("zero", toy_zero_elev(), 1, 4),
    ("multi", toy_multi_in(), 1, 5),
]
for seed in range(8):
    cases.append((f"r14_{seed}", random_dag(14, seed), 1, 14))
for seed in range(8):
    cases.append((f"r20_{seed}", random_dag(20, seed+100), 1, 20))
out=[]
for name,g,s,t in cases:
    res=namoa_star_exact(g,s,t,backend="t_mda")
    sols=sorted(set(res.solutions))
    st=res.stats
    out.append({
        "name":name,"exact":bool(st.exact_finished),
        "sols":sols,"exp":int(st.labels_expanded),"gen":int(st.labels_generated),
        "min": [min(x[i] for x in sols) for i in range(3)] if sols else None,
    })
print(json.dumps(out))
"""

PROF_SCRIPT = r"""
import json, time, tempfile, shutil
from pathlib import Path
from config import queries_path
from corridor import build_corridor, lexmin_all_orders
from graph import load_dataset, load_queries
from multiobjective_exact import load_or_compute_heuristics, namoa_star_exact, update_skyline
from seed_bank import build_seed_bank
from t_mda import set_nqp_sweep, set_profile_dir, clear_profile_dir, set_timing

MAX_EXP = int(%d)
g=load_dataset("NY")
q=load_queries(queries_path("NY","problem2")).iloc[1]
s,t=int(q["source"]),int(q["target"])
hb=load_or_compute_heuristics(g,"NY",t,use_cache=True)
bank=build_seed_bank(g,[s],t,hb,k=10)
seeds=list(bank[s])
for y in lexmin_all_orders(g,s-1,t-1):
    update_skyline(seeds,y)
corr=build_corridor(g,s-1,hb,seeds)
tmp=Path(tempfile.mkdtemp(prefix="tmda_v4ab_"))
set_nqp_sweep(False)
set_timing(True)
set_profile_dir(str(tmp))
t0=time.time()
try:
    res=namoa_star_exact(
        g,s,t,heuristics=hb,use_heuristic=True,
        extra_seeds=seeds,edge_allowed=corr.edge_allowed.copy(),
        max_labels=0,max_expanded=MAX_EXP,backend="t_mda",
    )
finally:
    clear_profile_dir()
wall=time.time()-t0
st=res.stats
timing={}
tp=tmp/"timing_breakdown.csv"
if tp.exists():
    for line in tp.read_text(encoding="utf-8").splitlines()[1:]:
        if not line or line.startswith("note"): continue
        p=line.split(",")
        if len(p)>=4: timing[p[0]]={"ms":float(p[2]),"pct":float(p[3])}
shutil.rmtree(tmp, ignore_errors=True)
print(json.dumps({
    "wall_sec":round(wall,3),
    "expanded":int(st.labels_expanded),
    "generated":int(st.labels_generated),
    "pareto_size":int(st.pareto_size),
    "peak_open":int(st.open_peak),
    "peak_nqp":int(st.u_bound_size),
    "timing":timing,
}))
""" % MAX_EXP


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    build_v4()
    if not V3.exists() or not V4.exists():
        raise SystemExit("missing V3 or V4 dll")

    print("[diff] toy/random EXACT V3 vs V4", flush=True)
    a = run_with_dll(V3, DIFF_SCRIPT)
    b = run_with_dll(V4, DIFF_SCRIPT)
    if a.returncode or b.returncode:
        print(a.stderr[-2000:], b.stderr[-2000:])
        raise SystemExit("diff subprocess failed")
    ja = json.loads([ln for ln in a.stdout.splitlines() if ln.startswith("[")][-1] if False else
                    [ln for ln in a.stdout.splitlines() if ln.startswith("[") or ln.startswith("{")][-1])
    # find JSON line
    def parse(out: str):
        for ln in reversed(out.splitlines()):
            if ln.startswith("["):
                return json.loads(ln)
        raise RuntimeError(out[-2000:])
    ja, jb = parse(a.stdout), parse(b.stdout)
    mism = 0
    rows = []
    for x, y in zip(ja, jb):
        set_ok = {tuple(t) for t in x["sols"]} == {tuple(t) for t in y["sols"]}
        traj = x["exp"] == y["exp"] and x["gen"] == y["gen"]
        ok = set_ok and x["exact"] and y["exact"] and x["min"] == y["min"]
        if not ok:
            mism += 1
        rows.append({"name": x["name"], "ok": ok, "set": set_ok, "traj": traj})
        print(f"  [{'OK' if ok else 'FAIL'}] {x['name']} traj={traj}", flush=True)
    diff_pass = mism == 0
    print(f"[diff] pass={diff_pass} mismatches={mism}", flush=True)

    print(f"[bench] NY0002 @{MAX_EXP} V3 then V4", flush=True)
    r3 = run_with_dll(V3, PROF_SCRIPT)
    r4 = run_with_dll(V4, PROF_SCRIPT)
    if r3.returncode or r4.returncode:
        print("V3 stderr", r3.stderr[-3000:])
        print("V4 stderr", r4.stderr[-3000:])
        raise SystemExit("profile failed")
    def parse_obj(out: str):
        for ln in reversed(out.splitlines()):
            if ln.startswith("{"):
                return json.loads(ln)
        raise RuntimeError(out[-2000:])
    p3, p4 = parse_obj(r3.stdout), parse_obj(r4.stdout)
    speedup = (p3["wall_sec"] - p4["wall_sec"]) / max(p3["wall_sec"], 1e-9)
    traj_ok = p3["expanded"] == p4["expanded"] and p3["generated"] == p4["generated"]
    gate = diff_pass and traj_ok and speedup >= 0.20
    report = {
        "tag": "T-MDA V4 Exact Permanent-Front Index",
        "diff_pass": diff_pass,
        "diff_mismatches": mism,
        "diff_rows": rows,
        "bench_max_expanded": MAX_EXP,
        "v3": p3,
        "v4": p4,
        "wall_speedup_frac": round(speedup, 4),
        "traj_same_at_5M": traj_ok,
        "gate_pass": gate,
        "gate_rule": "diff_pass && traj_same && wall_speedup>=0.20",
        "promote": gate,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# T-MDA V4 A/B — Exact Permanent-Front Index",
        "",
        "## Gate",
        f"- diff EXACT: **{diff_pass}** (mismatches={mism})",
        f"- NY0002@{MAX_EXP} traj same: **{traj_ok}**",
        f"- wall V3={p3['wall_sec']}s → V4={p4['wall_sec']}s  speedup=**{100*speedup:.1f}%**",
        f"- promote (≥20%): **{gate}**",
        "",
        "## Timing share (dom_perm)",
        f"- V3 dom_perm pct: {p3.get('timing',{}).get('dom_perm',{})}",
        f"- V4 dom_perm pct: {p4.get('timing',{}).get('dom_perm',{})}",
        "",
        "Production `t_mda.dll` was **not** modified. Running workers untouched.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("diff_pass", "wall_speedup_frac", "gate_pass", "promote")}, indent=2))
    return 0 if diff_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
