# -*- coding: utf-8 -*-
"""
Freeze the finished (or live) NY0001 v2 corridor run into:
  analysis/problem2/benchmark_NY_0001_v2.md
  analysis/problem2/benchmark_NY_0001_v2_slopes.csv

Does NOT modify the running process. Parse progress from a terminal log or --log file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from config import ANALYSIS_DIR

LINE_RE = re.compile(
    r"\[namoa\] exp=(?P<exp>\d+) gen=(?P<gen>\d+) live=(?P<live>\d+) open=(?P<open>\d+) "
    r"\|S\|=(?P<S>\d+) node_pr=(?P<node_pr>\d+) goal_pr=(?P<goal_pr>\d+) "
    r"dup=(?P<dup>\d+) stale=(?P<stale>\d+) max_node=(?P<max_node>\d+)"
)


def sha16(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def parse_log(text: str):
    rows = []
    for m in LINE_RE.finditer(text):
        d = {k: int(v) for k, v in m.groupdict().items()}
        exp = max(d["exp"], 1)
        d["rg"] = d["gen"] / exp
        d["rl"] = d["live"] / exp
        rows.append(d)
    # dedupe by exp keep last
    by = {}
    for r in rows:
        by[r["exp"]] = r
    return [by[k] for k in sorted(by)]


def detect_status(text: str, last) -> str:
    if re.search(r"\bEXACT\b", text) and "OVERFLOW" not in text[-2000:]:
        # final print from problem2
        if "OVERFLOW" in text or "LABEL" in text:
            pass
    if "OVERFLOW" in text or "LABEL_LIMIT" in text or "overflow" in text.lower():
        return "LABEL_LIMIT"
    if re.search(r"wall=.*EXACT", text):
        return "EXACT"
    if last and last["exp"] >= 30_000_000:
        return "LABEL_LIMIT"
    return "RUNNING"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, required=True)
    ap.add_argument("--wall-sec", type=float, default=None)
    ap.add_argument("--peak-rss-mb", type=float, default=None)
    ap.add_argument("--status", type=str, default=None)
    args = ap.parse_args()

    log_path = Path(args.log)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    slopes = parse_log(text)
    if not slopes:
        raise SystemExit("no [namoa] progress lines found")

    last = slopes[-1]
    status = args.status or detect_status(text, last)

    out_dir = ANALYSIS_DIR / "problem2"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "benchmark_NY_0001_v2_slopes.csv"
    md_path = out_dir / "benchmark_NY_0001_v2.md"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fields = [
            "exp", "gen", "live", "open", "S", "rg", "rl",
            "node_pr", "goal_pr", "dup", "stale", "max_node",
            "node_pr_per_exp", "goal_pr_per_exp", "S_per_exp", "max_node_per_exp",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in slopes:
            exp = r["exp"]
            w.writerow({
                "exp": r["exp"], "gen": r["gen"], "live": r["live"], "open": r["open"],
                "S": r["S"], "rg": f"{r['rg']:.6f}", "rl": f"{r['rl']:.6f}",
                "node_pr": r["node_pr"], "goal_pr": r["goal_pr"],
                "dup": r["dup"], "stale": r["stale"], "max_node": r["max_node"],
                "node_pr_per_exp": f"{r['node_pr']/exp:.6f}",
                "goal_pr_per_exp": f"{r['goal_pr']/exp:.6f}",
                "S_per_exp": f"{r['S']/exp:.8f}",
                "max_node_per_exp": f"{r['max_node']/exp:.8f}",
            })

    root = Path(__file__).resolve().parent
    dll_v2 = root / "namoa_core.dll"
    cpp = root / "namoa_core.cpp"

    # static corridor from log if present
    m_corr = re.search(
        r"seeds=(\d+) corridor nodes_pruned=(\d+) edges_pruned=(\d+) kept_edges=(\d+)",
        text,
    )
    seeds = nodes_pr = edges_pr = kept = "?"
    if m_corr:
        seeds, nodes_pr, edges_pr, kept = m_corr.groups()

    cmd_m = re.search(r"command: \"(.*)\"", text)
    cmdline = cmd_m.group(1) if cmd_m else "problem2.py --datasets NY --limit 1 --max-labels 30000000"

    md = f"""# NY 0001 benchmark v2 (corridor + 8 seeds) — FROZEN

> Do **not** re-run this configuration. Use v3 A/B/C/D ablation next.

## Identity
- frozen_at_utc: {datetime.now(timezone.utc).isoformat()}
- status: **{status}**
- command: `{cmdline}`
- DLL: `namoa_core.dll` sha16={sha16(dll_v2)} (legacy API; no suffix-UB / no dyn corridor)
- source: `namoa_core.cpp` sha16={sha16(cpp)} *(note: current tree may have advanced past the DLL this run loaded)*
- compiler flags (historical build): `-O3 -shared -std=c++17 -static-libgcc -static-libstdc++`

## Static setup
- seeds (ND after lex enrich): {seeds}
- static nodes_pruned: {nodes_pr}
- static edges_pruned: {edges_pr} / kept_edges={kept}
- label cap: 30_000_000

## Final / last snapshot
| metric | value |
|--------|------:|
| status | {status} |
| wall_sec | {args.wall_sec if args.wall_sec is not None else "see process / fill manually"} |
| peak_rss_mb | {args.peak_rss_mb if args.peak_rss_mb is not None else "fill manually"} |
| generated | {last['gen']} |
| expanded | {last['exp']} |
| live (last) | {last['live']} |
| peak_open (last open) | {last['open']} |
| node_prune | {last['node_pr']} |
| goal_prune | {last['goal_pr']} |
| stale | {last['stale']} |
| dup | {last['dup']} |
| current/final \|S\| | {last['S']} |
| max_node | {last['max_node']} |
| rg | {last['rg']:.4f} |
| rl | {last['rl']:.4f} |

## Slopes
See `benchmark_NY_0001_v2_slopes.csv` ({len(slopes)} checkpoints).

## Note on \|S\|
If status ≠ EXACT, `|S|` is the **incumbent** skyline during search, not the final exact Pareto front.
"""
    md_path.write_text(md, encoding="utf-8")
    print(f"[wrote] {csv_path}")
    print(f"[wrote] {md_path}")
    print(f"status={status} last_exp={last['exp']} |S|={last['S']} max_node={last['max_node']}")


if __name__ == "__main__":
    main()
