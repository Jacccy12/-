# -*- coding: utf-8 -*-
"""
Problem 4 analysis: preference sensitivity, disruption impact, city summary.

Usage
  python analyze_problem4.py
  python analyze_problem4.py --result4 results/result4.csv --limit-sensitivity 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import ANALYSIS_DIR, RESULTS_DIR
from problem4.evaluate import (
    build_comparison,
    reachability_summary,
    sensitivity_sweep,
)
from problem4.recommendation import build_candidate_set, recommend_row
from problem4.weight_scheme import PREFERENCE_SCHEMES


def qid_str(x) -> str:
    return f"{int(x):04d}"


def run_sensitivity(
    result3: Path,
    out_csv: Path,
    *,
    datasets,
    limit_per_city: int,
    focus_idx: int,
) -> pd.DataFrame:
    usecols = [
        "dataset",
        "query_id",
        "c1",
        "c2",
        "c3",
        "c4",
        "c5",
        "path",
        "objective_count",
    ]
    df = pd.read_csv(result3, dtype={"query_id": str}, usecols=usecols)
    df["query_id"] = df["query_id"].map(qid_str)
    df = df[df["dataset"].isin(datasets)]

    rows = []
    for ds in datasets:
        qids = sorted(df[df["dataset"] == ds]["query_id"].unique())
        if limit_per_city > 0:
            qids = qids[:limit_per_city]
        for qid in qids:
            g = df[(df["dataset"] == ds) & (df["query_id"] == qid)]
            costs = g[["c1", "c2", "c3", "c4", "c5"]].astype(int).to_numpy()
            paths = [
                [int(x) for x in str(p).split("->") if x] for p in g["path"].tolist()
            ]
            # sweep time weight (idx=1) around time_priority base
            base = PREFERENCE_SCHEMES["time_priority"]
            sens = sensitivity_sweep(costs, paths, base, focus_idx=focus_idx)
            sens.insert(0, "dataset", ds)
            sens.insert(1, "query_id", qid)
            _, default_path, _ = recommend_row(
                build_candidate_set(costs, paths), base
            )
            default_key = (
                "" if default_path is None else "->".join(map(str, default_path))
            )
            sens["changed_vs_default"] = sens["path"] != default_key
            rows.append(sens)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def preference_diversity(result4: pd.DataFrame) -> pd.DataFrame:
    """How often different schemes pick different paths (original only)."""
    orig = result4[result4["network_state"] == "original"].copy()
    rows = []
    for (ds, qid), g in orig.groupby(["dataset", "query_id"]):
        paths = g.set_index("scheme")["path"].to_dict()
        names = list(paths.keys())
        n_unique = len(set(paths.values()))
        rows.append(
            {
                "dataset": ds,
                "query_id": qid,
                "n_schemes": len(names),
                "n_unique_paths": n_unique,
                "all_same": n_unique <= 1,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result3", type=Path, default=RESULTS_DIR / "result3.csv")
    ap.add_argument("--result4", type=Path, default=RESULTS_DIR / "result4.csv")
    ap.add_argument("--output-dir", type=Path, default=ANALYSIS_DIR / "problem4")
    ap.add_argument("--datasets", type=str, default="NY,BAY,COL")
    ap.add_argument("--limit-sensitivity", type=int, default=5)
    ap.add_argument(
        "--focus-idx",
        type=int,
        default=1,
        help="objective index to sweep (0=dist .. 1=time ..)",
    )
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [d.strip().upper() for d in args.datasets.split(",") if d.strip()]

    if not args.result4.exists():
        raise SystemExit(f"missing result4: {args.result4} (run problem4.py first)")

    r4 = pd.read_csv(args.result4, dtype={"query_id": str})
    r4["query_id"] = r4["query_id"].map(qid_str)

    reach = reachability_summary(r4)
    reach.to_csv(args.output_dir / "reachability.csv", index=False)
    print("[analyze] reachability")
    print(reach.to_string(index=False))

    comp = build_comparison(r4)
    comp.to_csv(args.output_dir / "replanning_comparison.csv", index=False)
    if not comp.empty:
        print("\n[analyze] mean cost deltas (disrupted vs original):")
        for i in range(1, 6):
            col = f"delta_c{i}"
            if col in comp.columns:
                print(f"  {col}: {comp[col].mean(skipna=True):+.3%}")
        if "path_change" in comp.columns:
            print(f"  path_change(Jaccard dist): {comp['path_change'].mean(skipna=True):.3f}")

    div = preference_diversity(r4)
    div.to_csv(args.output_dir / "preference_diversity.csv", index=False)
    print(
        f"\n[analyze] preference diversity: "
        f"{(~div['all_same']).mean():.1%} queries have scheme disagreement"
    )

    city = (
        comp.groupby("dataset", sort=True)
        .agg(
            n=("scheme", "count"),
            disrupted_ok=("disrupted_feasible", "mean"),
            mean_path_change=("path_change", "mean"),
            mean_delta_c1=("delta_c1", "mean"),
            mean_delta_c2=("delta_c2", "mean"),
        )
        .reset_index()
    )
    city.to_csv(args.output_dir / "city_summary.csv", index=False)
    print("\n[analyze] city summary:")
    print(city.to_string(index=False))

    if args.result3.exists() and args.limit_sensitivity != 0:
        sens = run_sensitivity(
            args.result3,
            args.output_dir / "preference_sensitivity.csv",
            datasets=datasets,
            limit_per_city=args.limit_sensitivity,
            focus_idx=args.focus_idx,
        )
        if not sens.empty:
            # aggregate: as w_time increases, mean c2 should drop
            agg = (
                sens.groupby("w_focus")[["c1", "c2", "c3", "c4", "c5"]]
                .mean(numeric_only=True)
                .reset_index()
            )
            agg.to_csv(args.output_dir / "sensitivity_mean_costs.csv", index=False)
            print("\n[analyze] sensitivity mean costs vs w_focus:")
            print(agg.to_string(index=False))

    print(f"\n[analyze] wrote files under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
