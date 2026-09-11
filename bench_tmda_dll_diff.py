# -*- coding: utf-8 -*-
"""Differential EXACT gate: two T-MDA DLLs must agree on Pareto sets.

Each case runs in a fresh subprocess with TMDA_DLL pinned.
Default: t_mda.dll vs latest t_mda.dll.pre_lazy_*.
Does NOT write problem2 fragments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis" / "problem2" / "dll_diff_lazy_v3"
WORKER = ROOT / "bench_tmda_dll_diff_worker.py"


def find_pre_lazy() -> Path:
    cands = sorted(ROOT.glob("t_mda.dll.pre_lazy_*"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise SystemExit("no t_mda.dll.pre_lazy_* found")
    return cands[-1]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_cases(n_random: int) -> list:
    cases = [
        {"name": "diamond", "kind": "toy", "s": 1, "t": 4, "graph": "diamond"},
        {"name": "zero_elev", "kind": "toy", "s": 1, "t": 4, "graph": "zero"},
        {"name": "multi_in", "kind": "toy", "s": 1, "t": 5, "graph": "multi"},
    ]
    for seed in range(n_random):
        cases.append({
            "name": f"rand14_{seed}",
            "kind": "rand",
            "s": 1,
            "t": 14,
            "n": 14,
            "seed": seed,
        })
    for seed in range(n_random):
        cases.append({
            "name": f"rand20_{seed}",
            "kind": "rand",
            "s": 1,
            "t": 20,
            "n": 20,
            "seed": seed + 100,
        })
    return cases


def run_case(dll: Path, case: dict) -> dict:
    env = os.environ.copy()
    env["TMDA_DLL"] = str(dll.resolve())
    env["PYTHONPATH"] = str(ROOT)
    t0 = time.time()
    p = subprocess.run(
        [sys.executable, str(WORKER)],
        cwd=str(ROOT),
        input=json.dumps(case),
        capture_output=True,
        text=True,
        env=env,
    )
    wall = time.time() - t0
    if p.returncode != 0:
        raise RuntimeError(
            f"case={case['name']} dll={dll.name} rc={p.returncode}\n"
            f"stdout={p.stdout[-2000:]}\nstderr={p.stderr[-2000:]}"
        )
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("{")][-1]
    out = json.loads(line)
    out["wall_sec"] = round(wall, 4)
    out["dll"] = dll.name
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-dll", type=Path, default=ROOT / "t_mda.dll")
    ap.add_argument("--old-dll", type=Path, default=None)
    ap.add_argument("--n-random", type=int, default=8)
    args = ap.parse_args()
    old = args.old_dll or find_pre_lazy()
    new = args.new_dll
    if not new.exists():
        raise SystemExit(f"missing new dll {new}")

    OUT.mkdir(parents=True, exist_ok=True)
    cases = build_cases(args.n_random)
    rows = []
    mismatches = 0
    print(f"[diff] old={old.name} sha={sha256_file(old)[:16]}", flush=True)
    print(f"[diff] new={new.name} sha={sha256_file(new)[:16]}", flush=True)
    print(f"[diff] cases={len(cases)}", flush=True)

    for case in cases:
        a = run_case(old, case)
        b = run_case(new, case)
        sa = {tuple(x) for x in a["sols"]}
        sb = {tuple(x) for x in b["sols"]}
        set_ok = sa == sb
        mins_ok = (a["min0"], a["min1"], a["min2"]) == (b["min0"], b["min1"], b["min2"])
        exact_ok = a["exact"] and b["exact"]
        traj_same = (
            a["expanded"] == b["expanded"] and a["generated"] == b["generated"]
        )
        ok = set_ok and mins_ok and exact_ok
        if not ok:
            mismatches += 1
        row = {
            "name": case["name"],
            "ok": ok,
            "set_equal": set_ok,
            "mins_equal": mins_ok,
            "both_exact": exact_ok,
            "traj_same": traj_same,
            "old_exp": a["expanded"],
            "new_exp": b["expanded"],
            "old_gen": a["generated"],
            "new_gen": b["generated"],
            "pareto_size": a["pareto_size"],
            "old_wall": a["wall_sec"],
            "new_wall": b["wall_sec"],
        }
        rows.append(row)
        flag = "OK" if ok else "FAIL"
        traj = "traj=same" if traj_same else f"traj={a['expanded']}->{b['expanded']}"
        print(f"  [{flag}] {case['name']} |S|={a['pareto_size']} {traj}", flush=True)

    report = {
        "old_dll": str(old),
        "new_dll": str(new),
        "old_sha256": sha256_file(old),
        "new_sha256": sha256_file(new),
        "n_cases": len(cases),
        "mismatches": mismatches,
        "pass": mismatches == 0,
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# T-MDA DLL differential — lazy index vs pre_lazy",
        "",
        f"- old: `{old.name}` sha256={report['old_sha256'][:16]}…",
        f"- new: `{new.name}` sha256={report['new_sha256'][:16]}…",
        f"- cases: {len(cases)}  mismatches: **{mismatches}**  pass: **{report['pass']}**",
        "",
        "| case | ok | set | mins | exact | traj | |S| |",
        "|------|:--:|:---:|:----:|:-----:|:----:|----:|",
    ]
    for r in rows:
        md.append(
            f"| {r['name']} | {r['ok']} | {r['set_equal']} | {r['mins_equal']} | "
            f"{r['both_exact']} | {r['traj_same']} | {r['pareto_size']} |"
        )
    (OUT / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[diff] pass={report['pass']} mismatches={mismatches} -> {OUT}", flush=True)
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
