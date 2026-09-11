# -*- coding: utf-8 -*-
"""Summarize problem-2 Pareto sizes / runtimes; optional 3D scatter for one query."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import ANALYSIS_DIR, DATASETS


def analyze(result_csv: Path, stats_csv: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(result_csv, dtype={"query_id": str})
    stats = pd.read_csv(stats_csv, dtype={"query_id": str}) if stats_csv.exists() else None

    summary = {}
    rows = []
    for ds in DATASETS:
        g = df[df.dataset == ds]
        if g.empty:
            continue
        sizes = g.groupby("query_id").size()
        entry = {
            "n_queries": int(sizes.shape[0]),
            "n_solutions": int(len(g)),
            "pareto_mean": float(sizes.mean()),
            "pareto_median": float(sizes.median()),
            "pareto_max": int(sizes.max()),
            "pareto_min": int(sizes.min()),
        }
        if stats is not None:
            st = stats[stats.dataset == ds]
            if not st.empty:
                entry["runtime_mean_sec"] = float(st["runtime_sec"].astype(float).mean())
                entry["runtime_max_sec"] = float(st["runtime_sec"].astype(float).max())
                entry["labels_gen_mean"] = float(st["labels_generated"].mean())
                entry["goal_prune_mean"] = float(st["goal_bound_pruned"].mean())
        summary[ds] = entry
        rows.append({"dataset": ds, **entry})

    pd.DataFrame(rows).to_csv(out_dir / "city_summary.csv", index=False)
    (out_dir / "problem2_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))

    # plot one mid-size front per city
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        for ds in DATASETS:
            g = df[df.dataset == ds]
            if g.empty:
                continue
            sizes = g.groupby("query_id").size().sort_values()
            qid = sizes.index[len(sizes) // 2]
            pts = g[g.query_id == qid][["c1", "c2", "c3"]].to_numpy()
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=20, c="steelblue")
            ax.set_xlabel("c1 distance")
            ax.set_ylabel("c2 time")
            ax.set_zlabel("c3 elevation")
            ax.set_title(f"{ds} query {qid} |S|={len(pts)}")
            fig.tight_layout()
            fig.savefig(out_dir / f"pareto3d_{ds}_{qid}.png", dpi=140)
            plt.close(fig)

            fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
            pairs = [(0, 1, "c1", "c2"), (0, 2, "c1", "c3"), (1, 2, "c2", "c3")]
            for ax, (i, j, xi, yi) in zip(axes, pairs):
                ax.scatter(pts[:, i], pts[:, j], s=18, c="darkorange")
                ax.set_xlabel(xi)
                ax.set_ylabel(yi)
            fig.suptitle(f"{ds} {qid} projections")
            fig.tight_layout()
            fig.savefig(out_dir / f"pareto2d_{ds}_{qid}.png", dpi=140)
            plt.close(fig)
    except Exception as exc:
        print(f"[analyze2] plot skipped: {exc}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result", default="results/result2.csv")
    p.add_argument("--stats", default=str(ANALYSIS_DIR / "problem2" / "query_stats.csv"))
    p.add_argument("--out", default=str(ANALYSIS_DIR / "problem2"))
    args = p.parse_args()
    analyze(Path(args.result), Path(args.stats), Path(args.out))


if __name__ == "__main__":
    main()
