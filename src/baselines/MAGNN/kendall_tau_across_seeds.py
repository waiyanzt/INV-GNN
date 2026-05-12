#!/usr/bin/env python3
"""
Pairwise Kendall tau comparison across seeds and graph variants.

Merges score CSVs on (head, tail) pairs and computes:
  - Overall Kendall tau on all merged scores
  - Per-paper (per-head) macro Kendall tau
  - Top-1 match rate (fraction of heads where both variants agree on rank-1 tail)
  - Top-k Kendall tau on the top-k ranked tails per head

Usage (from MAGNN directory):
  python kendall_tau_across_seeds.py \
    --variant var1=csv_seed1.csv,csv_seed2.csv,csv_seed3.csv \
    --variant var2=csv_seed1.csv,csv_seed2.csv,csv_seed3.csv \
    --per-paper --hit1 --hitk --topk 3
"""
from __future__ import annotations

import argparse
import itertools
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def load_scores(path: str, head_col: str, tail_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {}
    if head_col != "paper_id" and head_col in df.columns:
        rename[head_col] = "paper_id"
    if tail_col != "conf_id" and tail_col in df.columns:
        rename[tail_col] = "conf_id"
    if rename:
        df = df.rename(columns=rename)
    need = {"paper_id", "conf_id", "score"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} must have columns {need}, got {list(df.columns)}")
    return df[["paper_id", "conf_id", "score"]]


def pairwise_tau(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    tau, p = kendalltau(a, b, nan_policy="omit")
    return (float(tau) if np.isfinite(tau) else float("nan"),
            float(p) if np.isfinite(p) else float("nan"))


def overall_tau(df_a: pd.DataFrame, df_b: pd.DataFrame) -> float:
    merged = df_a.merge(df_b, on=["paper_id", "conf_id"], suffixes=("_a", "_b"))
    if merged.empty:
        return float("nan")
    tau, _ = pairwise_tau(merged["score_a"].values, merged["score_b"].values)
    return tau


def per_paper_tau(df_a: pd.DataFrame, df_b: pd.DataFrame) -> float:
    merged = df_a.merge(df_b, on=["paper_id", "conf_id"], suffixes=("_a", "_b"))
    if merged.empty:
        return float("nan")
    taus = []
    for _, grp in merged.groupby("paper_id"):
        if len(grp) < 2:
            continue
        t, _ = pairwise_tau(grp["score_a"].values, grp["score_b"].values)
        if np.isfinite(t):
            taus.append(t)
    return float(np.mean(taus)) if taus else float("nan")


def top1_match(df_a: pd.DataFrame, df_b: pd.DataFrame) -> float:
    merged = df_a.merge(df_b, on=["paper_id", "conf_id"], suffixes=("_a", "_b"))
    if merged.empty:
        return float("nan")
    matches = 0
    total = 0
    for pid, grp in merged.groupby("paper_id"):
        if grp.empty:
            continue
        best_a = grp.loc[grp["score_a"].idxmax(), "conf_id"]
        best_b = grp.loc[grp["score_b"].idxmax(), "conf_id"]
        if best_a == best_b:
            matches += 1
        total += 1
    return matches / total if total > 0 else float("nan")


def topk_tau(df_a: pd.DataFrame, df_b: pd.DataFrame, k: int) -> float:
    merged = df_a.merge(df_b, on=["paper_id", "conf_id"], suffixes=("_a", "_b"))
    if merged.empty:
        return float("nan")
    taus = []
    for _, grp in merged.groupby("paper_id"):
        if len(grp) < 2:
            continue
        sorted_a = grp.nlargest(k, "score_a")["conf_id"].values
        sorted_b = grp.nlargest(k, "score_b")["conf_id"].values
        union = list(set(sorted_a) | set(sorted_b))
        if len(union) < 2:
            continue
        rank_a = grp.set_index("conf_id")["score_a"].reindex(union).rank(ascending=False).values
        rank_b = grp.set_index("conf_id")["score_b"].reindex(union).rank(ascending=False).values
        t, _ = pairwise_tau(rank_a, rank_b)
        if np.isfinite(t):
            taus.append(t)
    return float(np.mean(taus)) if taus else float("nan")


def fmt(vals: List[float]) -> str:
    finite = [v for v in vals if np.isfinite(v)]
    if not finite:
        return "nan"
    a = np.array(finite)
    if len(a) == 1:
        return f"{a[0]:.4f}"
    return f"{a.mean():.4f} \u00b1 {a.std(ddof=0):.4f}"


def print_matrix(label: str, names: List[str], matrix: Dict[Tuple[str, str], List[float]]):
    print(f"\n=== Pairwise {label} ===")
    pairs = list(itertools.combinations(range(len(names)), 2))
    header = "".ljust(12) + "".join(n.ljust(22) for n in names)
    print(header)
    for i, ni in enumerate(names):
        row = ni.ljust(12)
        for j, nj in enumerate(names):
            if i == j:
                row += "---".ljust(22)
            elif i < j:
                row += fmt(matrix[(ni, nj)]).ljust(22)
            else:
                row += fmt(matrix[(nj, ni)]).ljust(22)
        print(row)


def main():
    ap = argparse.ArgumentParser(
        description="Pairwise Kendall tau between graph variants across seeds.")
    ap.add_argument(
        "--variant", action="append", required=True,
        help="name=csv1,csv2,csv3  (repeat for each variant)")
    ap.add_argument("--per-paper", action="store_true",
                    help="Compute per-paper (macro) Kendall tau")
    ap.add_argument("--hit1", action="store_true",
                    help="Compute top-1 match rate")
    ap.add_argument("--hitk", action="store_true",
                    help="Compute top-k Kendall tau")
    ap.add_argument("--topk", type=int, default=3,
                    help="k for top-k tau (default: 3)")
    ap.add_argument("--head-col", default="paper_id",
                    help="Column name for head entity (default: paper_id)")
    ap.add_argument("--tail-col", default="conf_id",
                    help="Column name for tail entity (default: conf_id)")
    args = ap.parse_args()

    variants: Dict[str, List[str]] = {}
    for spec in args.variant:
        name, paths_str = spec.split("=", 1)
        variants[name] = [p.strip() for p in paths_str.split(",")]

    names = list(variants.keys())
    n_seeds = len(next(iter(variants.values())))
    for vn, csvs in variants.items():
        if len(csvs) != n_seeds:
            raise ValueError(
                f"Variant {vn} has {len(csvs)} CSVs but expected {n_seeds}")

    mat_overall: Dict[Tuple[str, str], List[float]] = {}
    mat_pp: Dict[Tuple[str, str], List[float]] = {}
    mat_top1: Dict[Tuple[str, str], List[float]] = {}
    mat_topk: Dict[Tuple[str, str], List[float]] = {}

    for na, nb in itertools.combinations(names, 2):
        mat_overall[(na, nb)] = []
        mat_pp[(na, nb)] = []
        mat_top1[(na, nb)] = []
        mat_topk[(na, nb)] = []

    for seed_idx in range(n_seeds):
        dfs = {}
        for vn in names:
            path = variants[vn][seed_idx]
            dfs[vn] = load_scores(path, args.head_col, args.tail_col)

        print(f"\n--- Seed index {seed_idx} ---")
        for na, nb in itertools.combinations(names, 2):
            da, db = dfs[na], dfs[nb]
            t = overall_tau(da, db)
            mat_overall[(na, nb)].append(t)
            print(f"  {na} vs {nb}: overall tau = {t:.4f}")

            if args.per_paper:
                pp = per_paper_tau(da, db)
                mat_pp[(na, nb)].append(pp)
                print(f"    per-paper tau = {pp:.4f}")

            if args.hit1:
                h1 = top1_match(da, db)
                mat_top1[(na, nb)].append(h1)
                print(f"    top-1 match   = {h1:.4f}")

            if args.hitk:
                hk = topk_tau(da, db, args.topk)
                mat_topk[(na, nb)].append(hk)
                print(f"    top-{args.topk} tau     = {hk:.4f}")

    print_matrix("Overall Kendall tau", names, mat_overall)
    if args.per_paper:
        print_matrix("Per-paper Kendall tau", names, mat_pp)
    if args.hit1:
        print_matrix(f"Top-1 match rate", names, mat_top1)
    if args.hitk:
        print_matrix(f"Top-{args.topk} Kendall tau", names, mat_topk)

    print("\n=== Detailed per-pair (mean +/- std over seeds) ===")
    header = f"{'Pair':<20} {'Overall tau':<22}"
    if args.per_paper:
        header += f"{'Per-paper tau':<22}"
    if args.hit1:
        header += f"{'Top-1 match':<22}"
    if args.hitk:
        header += f"{'Top-{0} tau'.format(args.topk):<22}"
    print(header)
    print("-" * len(header))
    for na, nb in itertools.combinations(names, 2):
        row = f"{na} vs {nb:<12} {fmt(mat_overall[(na, nb)]):<22}"
        if args.per_paper:
            row += f"{fmt(mat_pp[(na, nb)]):<22}"
        if args.hit1:
            row += f"{fmt(mat_top1[(na, nb)]):<22}"
        if args.hitk:
            row += f"{fmt(mat_topk[(na, nb)]):<22}"
        print(row)


if __name__ == "__main__":
    main()
