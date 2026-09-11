# -*- coding: utf-8 -*-
"""
Problem-1 post analysis: relative regret conflict, range-normalized conflict,
edge/node Jaccard, path identical rate.

Usage:
  python analyze_problem1.py --result results/result1.csv
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import ANALYSIS_DIR, DATASETS, OBJECTIVES
from validator import parse_path_string


def path_edge_set(path_str: str) -> set:
    """Directed edge set of a path."""
    nodes = parse_path_string(path_str)
    return {(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)}


def path_node_set(path_str: str) -> set:
    return set(parse_path_string(path_str))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def range_normalized_loss(cost_mat: np.ndarray) -> np.ndarray:
    """
    L[i,j] = (C_j(π_i) - C_j*) / (C_j^max - C_j*)
    Legacy range-normalized conflict (kept as supplement).
    """
    n = cost_mat.shape[0]
    L = np.zeros((n, n), dtype=float)
    opt = np.diag(cost_mat).astype(float)
    for j in range(n):
        cmax = float(cost_mat[:, j].max())
        denom = cmax - opt[j]
        if denom <= 0:
            continue
        for i in range(n):
            L[i, j] = (float(cost_mat[i, j]) - opt[j]) / denom
    return L


def relative_regret(cost_mat: np.ndarray) -> np.ndarray:
    """
    R[i,j] = (C_j(π_i) - C_j*) / max(C_j*, 1)
    Paper primary conflict metric.
    """
    n = cost_mat.shape[0]
    R = np.zeros((n, n), dtype=float)
    opt = np.diag(cost_mat).astype(float)
    for j in range(n):
        denom = max(opt[j], 1.0)
        for i in range(n):
            R[i, j] = (float(cost_mat[i, j]) - opt[j]) / denom
    return R


def symmetric_conflict(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def _quantile_stats(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "Q25": float(np.percentile(values, 25)),
        "Q75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
    }


def _pair_name(a: str, b: str) -> str:
    return f"{a}-{b}"


def analyze(result_csv: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(result_csv, dtype={"query_id": str})

    obj_pairs: List[Tuple[str, str]] = list(combinations(OBJECTIVES, 2))

    # query-level rows
    query_rows: List[dict] = []

    # per city stacks of directional relative regret R[i,j]
    city_R: Dict[str, List[np.ndarray]] = {ds: [] for ds in DATASETS}
    city_L: Dict[str, List[np.ndarray]] = {ds: [] for ds in DATASETS}
    city_Je: Dict[str, List[np.ndarray]] = {ds: [] for ds in DATASETS}
    city_Jn: Dict[str, List[np.ndarray]] = {ds: [] for ds in DATASETS}
    city_identical: Dict[str, Dict[Tuple[str, str], int]] = {
        ds: {p: 0 for p in obj_pairs} for ds in DATASETS
    }
    city_n = {ds: 0 for ds in DATASETS}
    mean_opt = {ds: np.zeros(5, dtype=float) for ds in DATASETS}

    # per-city pair samples for jaccard / regret stats
    pair_edge_j: Dict[str, Dict[Tuple[str, str], List[float]]] = {
        ds: {p: [] for p in obj_pairs} for ds in DATASETS
    }
    pair_node_j: Dict[str, Dict[Tuple[str, str], List[float]]] = {
        ds: {p: [] for p in obj_pairs} for ds in DATASETS
    }
    pair_regret_ij: Dict[str, Dict[Tuple[str, str], List[float]]] = {
        ds: {p: [] for p in obj_pairs} for ds in DATASETS
    }
    pair_regret_ji: Dict[str, Dict[Tuple[str, str], List[float]]] = {
        ds: {p: [] for p in obj_pairs} for ds in DATASETS
    }

    for ds, gdf in df.groupby("dataset"):
        for qid, qdf in gdf.groupby("query_id"):
            if len(qdf) != 5:
                raise ValueError(f"{ds} {qid}: expected 5 rows, got {len(qdf)}")
            qdf = qdf.set_index("objective").loc[list(OBJECTIVES)]
            cost_mat = qdf[["c1", "c2", "c3", "c4", "c5"]].to_numpy(dtype=float)
            R = relative_regret(cost_mat)
            L = range_normalized_loss(cost_mat)
            city_R[ds].append(R)
            city_L[ds].append(L)
            city_n[ds] += 1
            mean_opt[ds] += np.diag(cost_mat)

            edges = {obj: path_edge_set(qdf.loc[obj, "path"]) for obj in OBJECTIVES}
            nodes = {obj: path_node_set(qdf.loc[obj, "path"]) for obj in OBJECTIVES}
            paths = {obj: str(qdf.loc[obj, "path"]) for obj in OBJECTIVES}

            Je = np.zeros((5, 5), dtype=float)
            Jn = np.zeros((5, 5), dtype=float)
            qrow = {
                "dataset": ds,
                "query_id": qid,
                "source": int(qdf.iloc[0]["source"]),
                "target": int(qdf.iloc[0]["target"]),
            }
            for i, oi in enumerate(OBJECTIVES):
                for j, oj in enumerate(OBJECTIVES):
                    Je[i, j] = jaccard(edges[oi], edges[oj])
                    Jn[i, j] = jaccard(nodes[oi], nodes[oj])
            city_Je[ds].append(Je)
            city_Jn[ds].append(Jn)

            for oi, oj in obj_pairs:
                i, j = OBJECTIVES.index(oi), OBJECTIVES.index(oj)
                ej = float(Je[i, j])
                nj = float(Jn[i, j])
                rij = float(R[i, j])
                rji = float(R[j, i])
                pair_edge_j[ds][(oi, oj)].append(ej)
                pair_node_j[ds][(oi, oj)].append(nj)
                pair_regret_ij[ds][(oi, oj)].append(rij)
                pair_regret_ji[ds][(oi, oj)].append(rji)
                same = int(paths[oi] == paths[oj])
                city_identical[ds][(oi, oj)] += same
                qrow[f"rel_regret_{oi}_to_{oj}"] = rij
                qrow[f"rel_regret_{oj}_to_{oi}"] = rji
                qrow[f"edge_jaccard_{oi}_{oj}"] = ej
                qrow[f"node_jaccard_{oi}_{oj}"] = nj
                qrow[f"path_identical_{oi}_{oj}"] = same
            query_rows.append(qrow)

    # --- write legacy mean conflict/jaccard (do not delete old style) ---
    summary = {
        "primary_metric": "relative_regret_median",
        "supplement_metric": "range_normalized_conflict",
        "cities": {},
    }

    dir_rows = []
    sym_rows = []
    range_rows = []
    jaccard_edge_rows = []
    jaccard_node_rows = []
    identical_rows = []

    for ds in DATASETS:
        if city_n[ds] == 0:
            continue
        n = city_n[ds]
        R_stack = np.stack(city_R[ds], axis=0)  # (n,5,5)
        L_stack = np.stack(city_L[ds], axis=0)
        Je_stack = np.stack(city_Je[ds], axis=0)

        # city-level directional: median over queries (paper primary)
        R_median = np.median(R_stack, axis=0)
        R_mean = np.mean(R_stack, axis=0)
        R_q25 = np.percentile(R_stack, 25, axis=0)
        R_q75 = np.percentile(R_stack, 75, axis=0)
        R_max = np.max(R_stack, axis=0)

        K_sym = symmetric_conflict(R_median)
        # zero diagonal
        np.fill_diagonal(K_sym, 0.0)
        np.fill_diagonal(R_median, 0.0)

        L_mean = np.mean(L_stack, axis=0)
        L_sym = symmetric_conflict(L_mean)
        np.fill_diagonal(L_sym, 0.0)

        # keep old-style files (mean range-normalized conflict) for continuity
        pd.DataFrame(L_sym, index=OBJECTIVES, columns=OBJECTIVES).to_csv(
            out_dir / f"conflict_{ds}.csv", float_format="%.6f"
        )
        Jebar = np.mean(Je_stack, axis=0)
        pd.DataFrame(Jebar, index=OBJECTIVES, columns=OBJECTIVES).to_csv(
            out_dir / f"jaccard_{ds}.csv", float_format="%.6f"
        )

        # directional relative regret stats for all i→j
        for i, oi in enumerate(OBJECTIVES):
            for j, oj in enumerate(OBJECTIVES):
                dir_rows.append(
                    {
                        "dataset": ds,
                        "from_objective": oi,
                        "to_objective": oj,
                        "mean": float(R_mean[i, j]),
                        "median": float(R_median[i, j]),
                        "Q25": float(R_q25[i, j]),
                        "Q75": float(R_q75[i, j]),
                        "max": float(R_max[i, j]),
                    }
                )
                range_rows.append(
                    {
                        "dataset": ds,
                        "from_objective": oi,
                        "to_objective": oj,
                        "mean_range_normalized": float(L_mean[i, j]),
                        "symmetric_mean": float(L_sym[i, j]),
                    }
                )

        for i, oi in enumerate(OBJECTIVES):
            for j, oj in enumerate(OBJECTIVES):
                if i > j:
                    continue
                sym_rows.append(
                    {
                        "dataset": ds,
                        "objective_i": oi,
                        "objective_j": oj,
                        "K_ij_median": float(K_sym[i, j]),
                        "K_i_to_j_median": float(R_median[i, j]),
                        "K_j_to_i_median": float(R_median[j, i]),
                    }
                )

        for oi, oj in obj_pairs:
            ej = np.asarray(pair_edge_j[ds][(oi, oj)], dtype=float)
            nj = np.asarray(pair_node_j[ds][(oi, oj)], dtype=float)
            est = _quantile_stats(ej)
            nst = _quantile_stats(nj)
            jaccard_edge_rows.append({"dataset": ds, "pair": _pair_name(oi, oj), **est})
            jaccard_node_rows.append({"dataset": ds, "pair": _pair_name(oi, oj), **nst})
            identical_rows.append(
                {
                    "dataset": ds,
                    "pair": _pair_name(oi, oj),
                    "identical_count": city_identical[ds][(oi, oj)],
                    "n_queries": n,
                    "P_same": city_identical[ds][(oi, oj)] / n,
                }
            )

        # paper heatmaps: symmetric relative-regret median
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6.2, 5.2))
            im = ax.imshow(K_sym, cmap="YlOrRd", vmin=0, vmax=max(0.01, float(K_sym.max())))
            ax.set_xticks(range(5), OBJECTIVES, rotation=45, ha="right")
            ax.set_yticks(range(5), OBJECTIVES)
            ax.set_title(f"{ds} relative regret conflict (median)")
            for i in range(5):
                for j in range(5):
                    ax.text(j, i, f"{K_sym[i, j]:.2f}", ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            fig.savefig(out_dir / f"conflict_relative_{ds}.png", dpi=160)
            plt.close(fig)

            # also keep old heatmap name for continuity
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(L_sym, cmap="YlOrRd", vmin=0, vmax=1)
            ax.set_xticks(range(5), OBJECTIVES, rotation=45, ha="right")
            ax.set_yticks(range(5), OBJECTIVES)
            ax.set_title(f"{ds} range-normalized conflict (mean)")
            for i in range(5):
                for j in range(5):
                    ax.text(j, i, f"{L_sym[i, j]:.2f}", ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            fig.savefig(out_dir / f"conflict_heatmap_{ds}.png", dpi=160)
            plt.close(fig)
        except Exception as exc:
            print(f"[analyze] skip heatmap for {ds}: {exc}")

        summary["cities"][ds] = {
            "n_queries": n,
            "mean_optimal_costs": {
                OBJECTIVES[i]: float(mean_opt[ds][i] / n) for i in range(5)
            },
            "relative_regret": {
                "directional_median": R_median.tolist(),
                "symmetric_median": K_sym.tolist(),
                "directional_mean": R_mean.tolist(),
            },
            "range_normalized_conflict": {
                "directional_mean": L_mean.tolist(),
                "symmetric_mean": L_sym.tolist(),
            },
            "edge_jaccard_median_by_pair": {
                _pair_name(a, b): float(np.median(pair_edge_j[ds][(a, b)]))
                for a, b in obj_pairs
            },
            "path_identical_rate": {
                _pair_name(a, b): city_identical[ds][(a, b)] / n for a, b in obj_pairs
            },
            # legacy keys for older readers
            "conflict_matrix": L_sym.tolist(),
            "mean_jaccard": Jebar.tolist(),
        }

    # combined edge jaccard figure
    try:
        import matplotlib.pyplot as plt

        # median edge Jaccard matrix per city in one figure
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        for ax, ds in zip(axes, DATASETS):
            if city_n[ds] == 0:
                continue
            Je_med = np.median(np.stack(city_Je[ds], axis=0), axis=0)
            im = ax.imshow(Je_med, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(5), OBJECTIVES, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(5), OBJECTIVES, fontsize=7)
            ax.set_title(f"{ds} median edge Jaccard")
            for i in range(5):
                for j in range(5):
                    ax.text(j, i, f"{Je_med[i, j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02)
        fig.tight_layout()
        fig.savefig(out_dir / "jaccard_edge.png", dpi=160)
        plt.close(fig)
    except Exception as exc:
        print(f"[analyze] skip jaccard_edge.png: {exc}")

    pd.DataFrame(dir_rows).to_csv(out_dir / "conflict_relative_directional.csv", index=False)
    pd.DataFrame(sym_rows).to_csv(out_dir / "conflict_relative_symmetric.csv", index=False)
    pd.DataFrame(range_rows).to_csv(out_dir / "conflict_range_normalized.csv", index=False)
    pd.DataFrame(jaccard_edge_rows).to_csv(out_dir / "jaccard_edge_summary.csv", index=False)
    pd.DataFrame(jaccard_node_rows).to_csv(out_dir / "jaccard_node_summary.csv", index=False)
    pd.DataFrame(identical_rows).to_csv(out_dir / "path_identical_rate.csv", index=False)
    pd.DataFrame(query_rows).to_csv(out_dir / "query_level_metrics.csv", index=False)

    (out_dir / "problem1_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[analyze] wrote analysis -> {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=str, default="results/result1.csv")
    parser.add_argument("--out", type=str, default=str(ANALYSIS_DIR / "problem1"))
    args = parser.parse_args()
    analyze(Path(args.result), Path(args.out))


if __name__ == "__main__":
    main()
