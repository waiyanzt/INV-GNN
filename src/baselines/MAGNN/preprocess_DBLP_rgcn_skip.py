#!/usr/bin/env python3
"""
DBLP RGCN preprocessing — universal skip

Examples::
    python preprocess_DBLP_rgcn_skip.py --variant v1,v2,v3
    python preprocess_DBLP_rgcn_skip.py --variant v1,v2,v3 --universal-area-channels paper_conf
"""

import argparse
import os
import random
import shutil

import numpy as np
import pandas as pd
import torch

SEED = 1566911444


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _parse_variants(s):
    vals = [x.strip().lower() for x in str(s).split(",") if x.strip()]
    good = {"v1", "v2", "v3"}
    bad = [v for v in vals if v not in good]
    if bad:
        raise SystemExit(f"Unknown variant(s): {bad}; expected subset of {sorted(good)}")
    return vals


def load_tables(raw_dir):
    al = pd.read_csv(
        os.path.join(raw_dir, "author_label.txt"),
        sep="\t",
        names=["author_id", "label", "author_name"],
        header=None,
        encoding="utf-8",
    )
    pa = pd.read_csv(
        os.path.join(raw_dir, "paper_author.txt"),
        sep="\t",
        names=["paper_id", "author_id"],
        header=None,
        encoding="utf-8",
    )
    pc = pd.read_csv(
        os.path.join(raw_dir, "paper_conf.txt"),
        sep="\t",
        names=["paper_id", "conf_id"],
        header=None,
        encoding="utf-8",
    )
    pt = pd.read_csv(
        os.path.join(raw_dir, "paper_term.txt"),
        sep="\t",
        names=["paper_id", "term_id"],
        header=None,
        encoding="utf-8",
    )
    return al, pa, pc, pt


def map_pairs_npz(shared_npz, Pa_map, Co_map):
    Z = np.load(shared_npz)

    def M(X, name):
        df = pd.DataFrame(X, columns=["paper_id", "conf_id"])
        df["paper_id"] = df["paper_id"].map(Pa_map)
        df["conf_id"] = df["conf_id"].map(Co_map)
        out = df.dropna().to_numpy(dtype=np.int64)
        dropped = len(df) - len(out)
        if dropped:
            print(f"[warn] {name}: dropped {dropped}", flush=True)
        return out

    out = {
        "train_pos": M(Z["train_pos"], "train_pos"),
        "val_pos": M(Z["val_pos"], "val_pos"),
        "test_pos": M(Z["test_pos"], "test_pos"),
        "train_neg": M(Z["train_neg"], "train_neg"),
        "val_neg": M(Z["val_neg"], "val_neg"),
        "test_neg": M(Z["test_neg"], "test_neg"),
    }
    paper_subset = Z["paper_subset"].tolist() if "paper_subset" in Z.files else None
    return out, paper_subset


def filter_tables(al, pa, pc, pt, paper_subset=None, min_conf=0):
    valid_authors = set(al["author_id"])
    pa = pa[pa["author_id"].isin(valid_authors)].reset_index(drop=True)

    valid_papers = set(pa["paper_id"])
    pc = pc[pc["paper_id"].isin(valid_papers)].reset_index(drop=True)
    pt = pt[pt["paper_id"].isin(valid_papers)].reset_index(drop=True)
    pa = pa[pa["paper_id"].isin(valid_papers)].reset_index(drop=True)

    if min_conf > 0:
        big = pc["conf_id"].value_counts().loc[lambda x: x >= min_conf].index
        pc = pc[pc["conf_id"].isin(big)].reset_index(drop=True)
        valid_papers = set(pc["paper_id"])
        pt = pt[pt["paper_id"].isin(valid_papers)].reset_index(drop=True)
        pa = pa[pa["paper_id"].isin(valid_papers)].reset_index(drop=True)

    if paper_subset is not None:
        keep = set(paper_subset) & set(pc["paper_id"])
        pc = pc[pc["paper_id"].isin(keep)].reset_index(drop=True)
        pa = pa[pa["paper_id"].isin(keep)].reset_index(drop=True)
        pt = pt[pt["paper_id"].isin(keep)].reset_index(drop=True)

    return al, pa, pc, pt


def build_canonical_paper_area(al, pa):
    a2r = al.set_index("author_id")["label"]
    pr_raw = (
        pa.assign(area_id=pa["author_id"].map(a2r))[["paper_id", "area_id"]]
        .drop_duplicates()
        .dropna()
        .reset_index(drop=True)
    )
    area_counts = pr_raw.groupby("paper_id")["area_id"].nunique()
    bad_papers = set(area_counts[area_counts != 1].index.tolist())
    return pr_raw, bad_papers


def build_maps(pa, pc, pt, pr):
    Au_ids = sorted(pa["author_id"].unique())
    Pa_ids = sorted(pc["paper_id"].unique())
    Te_ids = sorted(pt["term_id"].unique())
    Co_ids = sorted(pc["conf_id"].unique())
    Ar_ids = sorted(pr["area_id"].unique())
    Au_map = {a: i for i, a in enumerate(Au_ids)}
    Pa_map = {p: i for i, p in enumerate(Pa_ids)}
    Te_map = {t: i for i, t in enumerate(Te_ids)}
    Co_map = {c: i for i, c in enumerate(Co_ids)}
    Ar_map = {r: i for i, r in enumerate(Ar_ids)}
    return Au_map, Pa_map, Te_map, Co_map, Ar_map


def build_universal_graph_data(
    al, pa, pc, pt, pr, Pa_map, Au_map, Te_map, Co_map, Ar_map, splits, universal_area_channels: str = "all"
):
    """
    Heterograph tensors; P–C from train only.

    universal_area_channels:
      ``all`` — paper–area, conference–area, author–area (default).
      ``paper_conf`` — paper–area and conference–area only (paper–venue LP ablation).
    """
    if universal_area_channels not in ("all", "paper_conf"):
        raise ValueError(f"universal_area_channels must be 'all' or 'paper_conf', got {universal_area_channels!r}")
    train_pos = splits["train_pos"]

    A2P_src, A2P_dst = [], []
    for p, a in pa[["paper_id", "author_id"]].itertuples(index=False):
        if p in Pa_map and a in Au_map:
            A2P_src.append(Au_map[a])
            A2P_dst.append(Pa_map[p])

    P2T_src, P2T_dst = [], []
    for p, t in pt[["paper_id", "term_id"]].itertuples(index=False):
        if p in Pa_map and t in Te_map:
            P2T_src.append(Pa_map[p])
            P2T_dst.append(Te_map[t])

    P2R_src, P2R_dst = [], []
    for p, r in pr[["paper_id", "area_id"]].itertuples(index=False):
        if p in Pa_map and r in Ar_map:
            P2R_src.append(Pa_map[p])
            P2R_dst.append(Ar_map[r])

    cr = (
        pc[["paper_id", "conf_id"]]
        .drop_duplicates()
        .merge(pr[["paper_id", "area_id"]].drop_duplicates(), on="paper_id")
        [["conf_id", "area_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    C2R_src, C2R_dst = [], []
    for c, r in cr.itertuples(index=False):
        if c in Co_map and r in Ar_map:
            C2R_src.append(Co_map[c])
            C2R_dst.append(Ar_map[r])

    graph_data = {
        "author-paper": (np.array(A2P_src, dtype=np.int64), np.array(A2P_dst, dtype=np.int64)),
        "paper-term": (np.array(P2T_src, dtype=np.int64), np.array(P2T_dst, dtype=np.int64)),
        "paper-area": (np.array(P2R_src, dtype=np.int64), np.array(P2R_dst, dtype=np.int64)),
        "conference-area": (np.array(C2R_src, dtype=np.int64), np.array(C2R_dst, dtype=np.int64)),
        "paper-conference": (
            train_pos[:, 0].astype(np.int64),
            train_pos[:, 1].astype(np.int64),
        ),
    }
    if universal_area_channels == "all":
        ar = (
            pa[["paper_id", "author_id"]]
            .drop_duplicates()
            .merge(pr[["paper_id", "area_id"]].drop_duplicates(), on="paper_id")
            [["author_id", "area_id"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        A2R_src, A2R_dst = [], []
        for a, r in ar.itertuples(index=False):
            if a in Au_map and r in Ar_map:
                A2R_src.append(Au_map[a])
                A2R_dst.append(Ar_map[r])
        graph_data["author-area"] = (np.array(A2R_src, dtype=np.int64), np.array(A2R_dst, dtype=np.int64))
    return graph_data


def preprocess_one(variant, out_dir_base, graph_data, meta):
    out_dir = os.path.join(out_dir_base, f"DBLP_rgcn_skip_{variant}")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(graph_data, os.path.join(out_dir, "graph_data.pt"))
    torch.save(meta, os.path.join(out_dir, "meta.pt"))
    print(f"Saved universal graph to {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Preprocess DBLP RGCN universal skip graph")
    ap.add_argument(
        "--variant",
        default="v1,v2,v3",
        help="Comma list v1,v2,v3 (each output dir gets the same graph for Kendall τ=1).",
    )
    ap.add_argument("--raw-dir", default="data/raw/DBLP/")
    ap.add_argument(
        "--shared-npz",
        default="data/preprocessed/DBLP_shared_splits/DBLP_pc_shared_splits.npz",
    )
    ap.add_argument("--min-conf", type=int, default=0)
    ap.add_argument("--out-dir", default="data/preprocessed")
    ap.add_argument(
        "--universal-area-channels",
        choices=("all", "paper_conf"),
        default="all",
        help="'all' = paper+conf+author area edges; 'paper_conf' = paper+conf area only (drop author–area for LP ablation).",
    )
    args = ap.parse_args()

    set_seed()
    variants = _parse_variants(args.variant)

    al, pa, pc, pt = load_tables(args.raw_dir)
    tmp_Pa_map = {p: i for i, p in enumerate(sorted(pa["paper_id"].unique()))}
    tmp_Co_map = {c: i for i, c in enumerate(sorted(pc["conf_id"].unique()))}
    _, paper_subset = map_pairs_npz(args.shared_npz, tmp_Pa_map, tmp_Co_map)

    al, pa, pc, pt = filter_tables(al, pa, pc, pt, paper_subset=paper_subset, min_conf=args.min_conf)

    pr_raw, bad_papers = build_canonical_paper_area(al, pa)
    if bad_papers:
        print(f"Removing {len(bad_papers)} papers violating 1 paper -> 1 area", flush=True)
        pa = pa[~pa["paper_id"].isin(bad_papers)].reset_index(drop=True)
        pc = pc[~pc["paper_id"].isin(bad_papers)].reset_index(drop=True)
        pt = pt[~pt["paper_id"].isin(bad_papers)].reset_index(drop=True)
        pr_raw, bad2 = build_canonical_paper_area(al, pa)
        assert len(bad2) == 0

    pr = pr_raw.drop_duplicates(subset=["paper_id"]).reset_index(drop=True)
    Au_map, Pa_map, Te_map, Co_map, Ar_map = build_maps(pa, pc, pt, pr)
    splits, _ = map_pairs_npz(args.shared_npz, Pa_map, Co_map)

    print(
        f"Authors={len(Au_map)} Papers={len(Pa_map)} Terms={len(Te_map)} "
        f"Confs={len(Co_map)} Areas={len(Ar_map)}",
        flush=True,
    )

    graph_data = build_universal_graph_data(
        al,
        pa,
        pc,
        pt,
        pr,
        Pa_map,
        Au_map,
        Te_map,
        Co_map,
        Ar_map,
        splits,
        universal_area_channels=args.universal_area_channels,
    )

    meta = {
        "num_nodes": {
            "author": len(Au_map),
            "paper": len(Pa_map),
            "term": len(Te_map),
            "conference": len(Co_map),
            "area": len(Ar_map),
        },
        "splits": {k: torch.tensor(v, dtype=torch.long) for k, v in splits.items()},
        "variant": "universal",
        "paper_conf_edges": "train_only",
        "universal_area_channels": args.universal_area_channels,
    }

    edge_msg = (
        f"Universal edges ({args.universal_area_channels}): "
        f"|P–C|={len(graph_data['paper-conference'][0])} (train) "
        f"|P–R|={len(graph_data['paper-area'][0])} |C–R|={len(graph_data['conference-area'][0])}"
    )
    if "author-area" in graph_data:
        edge_msg += f" |Auth–R|={len(graph_data['author-area'][0])}"
    print(edge_msg, flush=True)

    for v in variants:
        print(f"=== write DBLP_rgcn_skip_{v} (same graph) ===", flush=True)
        preprocess_one(v, args.out_dir, graph_data, meta)

    print("Done. v1/v2/v3 directories are identical — use any variant path or all for τ=1 checks.")


if __name__ == "__main__":
    main()
