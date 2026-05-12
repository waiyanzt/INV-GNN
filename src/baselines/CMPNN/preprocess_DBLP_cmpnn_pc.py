#!/usr/bin/env python3
"""
Examples::

    python preprocess_DBLP_cmpnn_pc.py --variant v1,v2,v3
"""
import argparse
import os
import random
import shutil
from pathlib import Path

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


def map_pairs_npz(shared_npz, pa_map, co_map):
    z = np.load(shared_npz)

    def M(X, name):
        df = pd.DataFrame(X, columns=["paper_id", "conf_id"])
        df["paper_id"] = df["paper_id"].map(pa_map)
        df["conf_id"] = df["conf_id"].map(co_map)
        out = df.dropna().to_numpy(dtype=np.int64)
        dropped = len(df) - len(out)
        if dropped:
            print(f"[warn] {name}: dropped {dropped}", flush=True)
        return out

    out = {
        "train_pos": M(z["train_pos"], "train_pos"),
        "val_pos": M(z["val_pos"], "val_pos"),
        "test_pos": M(z["test_pos"], "test_pos"),
        "train_neg": M(z["train_neg"], "train_neg"),
        "val_neg": M(z["val_neg"], "val_neg"),
        "test_neg": M(z["test_neg"], "test_neg"),
    }
    paper_subset = z["paper_subset"].tolist() if "paper_subset" in z.files else None
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
    au_ids = sorted(pa["author_id"].unique())
    pa_ids = sorted(pc["paper_id"].unique())
    te_ids = sorted(pt["term_id"].unique())
    co_ids = sorted(pc["conf_id"].unique())
    ar_ids = sorted(pr["area_id"].unique())

    au_map = {a: i for i, a in enumerate(au_ids)}
    pa_map = {p: i for i, p in enumerate(pa_ids)}
    te_map = {t: i for i, t in enumerate(te_ids)}
    co_map = {c: i for i, c in enumerate(co_ids)}
    ar_map = {r: i for i, r in enumerate(ar_ids)}
    return au_map, pa_map, te_map, co_map, ar_map


def build_edge_list_and_meta(
    variant: str,
    al,
    pa,
    pc,
    pt,
    pr,
    au_map,
    pa_map,
    te_map,
    co_map,
    ar_map,
    splits,
):
    """Four relations: 0=PA, 1=PT, 2=area-channel (variant), 3=PC. P–C = train only."""
    REL_PA = 0
    REL_PT = 1
    REL_AREA = 2
    REL_PC = 3

    off_author = 0
    off_paper = off_author + len(au_map)
    off_term = off_paper + len(pa_map)
    off_conf = off_term + len(te_map)
    off_area = off_conf + len(co_map)
    total_nodes = off_area + len(ar_map)

    edge_list = []

    for p, a in pa[["paper_id", "author_id"]].itertuples(index=False):
        if p in pa_map and a in au_map:
            edge_list.append([off_paper + pa_map[p], off_author + au_map[a], REL_PA])

    for p, t in pt[["paper_id", "term_id"]].itertuples(index=False):
        if p in pa_map and t in te_map:
            edge_list.append([off_paper + pa_map[p], off_term + te_map[t], REL_PT])

    train_pc = splits["train_pos"]
    for row in train_pc:
        p_local, c_local = int(row[0]), int(row[1])
        edge_list.append([off_paper + p_local, off_conf + c_local, REL_PC])

    if variant == "v1":
        for p, r in pr[["paper_id", "area_id"]].itertuples(index=False):
            if p in pa_map and r in ar_map:
                edge_list.append([off_paper + pa_map[p], off_area + ar_map[r], REL_AREA])
        area_rel_name = "paper-area"
    elif variant == "v2":
        cr = (
            pc[["paper_id", "conf_id"]]
            .drop_duplicates()
            .merge(pr[["paper_id", "area_id"]].drop_duplicates(), on="paper_id")
            [["conf_id", "area_id"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        for c, r in cr.itertuples(index=False):
            if c in co_map and r in ar_map:
                edge_list.append([off_conf + co_map[c], off_area + ar_map[r], REL_AREA])
        area_rel_name = "conference-area"
    elif variant == "v3":
        ar = (
            pa[["paper_id", "author_id"]]
            .drop_duplicates()
            .merge(pr[["paper_id", "area_id"]].drop_duplicates(), on="paper_id")
            [["author_id", "area_id"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        for a, r in ar.itertuples(index=False):
            if a in au_map and r in ar_map:
                edge_list.append([off_author + au_map[a], off_area + ar_map[r], REL_AREA])
        area_rel_name = "author-area"
    else:
        raise ValueError(variant)

    edge_arr = np.array(edge_list, dtype=np.int64)
    edge_arr = np.unique(edge_arr, axis=0)
    print(f"  Unique directed edges: {len(edge_arr)}", flush=True)

    meta = {
        "num_nodes": {
            "author": len(au_map),
            "paper": len(pa_map),
            "term": len(te_map),
            "conference": len(co_map),
            "area": len(ar_map),
            "total": total_nodes,
        },
        "offsets": {
            "author": off_author,
            "paper": off_paper,
            "term": off_term,
            "conference": off_conf,
            "area": off_area,
        },
        "relation_map": {
            "paper-author": REL_PA,
            "paper-term": REL_PT,
            area_rel_name: REL_AREA,
            "paper-conference": REL_PC,
        },
        "splits": {k: torch.tensor(v, dtype=torch.long) for k, v in splits.items()},
        "variant": variant,
        "graph_kind": f"dblp_cmpnn_pc_{variant}_train_pc_only",
        "paper_conf_edges": "train_only",
    }

    return torch.tensor(edge_arr, dtype=torch.long), meta


def main():
    ap = argparse.ArgumentParser(description="DBLP CMPNN non-skip v1/v2/v3 (train-only P–C)")
    ap.add_argument("--variant", default="v1,v2,v3", help="Comma list: v1, v2, v3")
    ap.add_argument("--raw-dir", default="../MAGNN/data/raw/DBLP")
    ap.add_argument(
        "--shared-npz",
        default="../MAGNN/data/preprocessed/DBLP_shared_splits/DBLP_pc_shared_splits.npz",
    )
    ap.add_argument("--min-conf", type=int, default=0)
    ap.add_argument("--out-dir", default="data/preprocessed")
    args = ap.parse_args()

    set_seed()

    al, pa, pc, pt = load_tables(args.raw_dir)
    tmp_pa_map = {p: i for i, p in enumerate(sorted(pa["paper_id"].unique()))}
    tmp_co_map = {c: i for i, c in enumerate(sorted(pc["conf_id"].unique()))}
    _, paper_subset = map_pairs_npz(args.shared_npz, tmp_pa_map, tmp_co_map)

    al, pa, pc, pt = filter_tables(al, pa, pc, pt, paper_subset=paper_subset, min_conf=args.min_conf)

    pr_raw, bad_papers = build_canonical_paper_area(al, pa)
    if bad_papers:
        print(f"Removing {len(bad_papers)} papers with ambiguous multi-area pr", flush=True)
        pa = pa[~pa["paper_id"].isin(bad_papers)].reset_index(drop=True)
        pc = pc[~pc["paper_id"].isin(bad_papers)].reset_index(drop=True)
        pt = pt[~pt["paper_id"].isin(bad_papers)].reset_index(drop=True)
        pr_raw, bad2 = build_canonical_paper_area(al, pa)
        assert len(bad2) == 0

    pr = pr_raw.drop_duplicates(subset=["paper_id"]).reset_index(drop=True)
    au_map, pa_map, te_map, co_map, ar_map = build_maps(pa, pc, pt, pr)
    splits, _ = map_pairs_npz(args.shared_npz, pa_map, co_map)

    print(
        f"Authors={len(au_map)} Papers={len(pa_map)} Terms={len(te_map)} "
        f"Confs={len(co_map)} Areas={len(ar_map)}",
        flush=True,
    )

    out_root = Path(args.out_dir)
    for variant in _parse_variants(args.variant):
        print(f"=== DBLP CMPNN pc {variant} (train-only paper–conf in KG) ===", flush=True)
        edge_list, meta = build_edge_list_and_meta(
            variant, al, pa, pc, pt, pr, au_map, pa_map, te_map, co_map, ar_map, splits
        )
        sub = out_root / f"DBLP_cmpnn_pc_{variant}"
        if sub.exists():
            shutil.rmtree(sub)
        sub.mkdir(parents=True, exist_ok=True)
        torch.save(edge_list, sub / "edge_list.pt")
        torch.save(meta, sub / "meta.pt")
        print(f"Saved {sub}", flush=True)


if __name__ == "__main__":
    main()
