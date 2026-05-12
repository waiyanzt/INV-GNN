#!/usr/bin/env python3
"""
DBLP CMPNN **universal skip** graph

Examples::

    python preprocess_DBLP_cmpnn_skip.py --variant v1,v2,v3
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


def build_universal_edge_list_and_meta(args):
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

    canonical_pa = {}
    for p, r in pr[["paper_id", "area_id"]].itertuples(index=False):
        if p in pa_map and r in ar_map:
            canonical_pa[pa_map[p]] = ar_map[r]

    conf_area = set()
    for p, c in pc[["paper_id", "conf_id"]].itertuples(index=False):
        if p in pa_map and c in co_map:
            pl = pa_map[p]
            if pl in canonical_pa:
                conf_area.add((co_map[c], canonical_pa[pl]))

    off_author = 0
    off_paper = off_author + len(au_map)
    off_term = off_paper + len(pa_map)
    off_conf = off_term + len(te_map)
    off_area = off_conf + len(co_map)
    total_nodes = off_area + len(ar_map)

    REL_PAPER_AUTHOR = 0
    REL_PAPER_TERM = 1
    REL_PAPER_CONF = 2
    REL_PAPER_AREA = 3
    REL_CONF_AREA = 4

    edge_list = []

    for p, a in pa[["paper_id", "author_id"]].itertuples(index=False):
        if p in pa_map and a in au_map:
            edge_list.append(
                [off_paper + pa_map[p], off_author + au_map[a], REL_PAPER_AUTHOR]
            )

    for p, t in pt[["paper_id", "term_id"]].itertuples(index=False):
        if p in pa_map and t in te_map:
            edge_list.append([off_paper + pa_map[p], off_term + te_map[t], REL_PAPER_TERM])

    train_pc = splits["train_pos"]
    for row in train_pc:
        p_local, c_local = int(row[0]), int(row[1])
        edge_list.append([off_paper + p_local, off_conf + c_local, REL_PAPER_CONF])

    for p_local, r_local in sorted(canonical_pa.items()):
        edge_list.append([off_paper + int(p_local), off_area + int(r_local), REL_PAPER_AREA])

    for c_local, r_local in sorted(conf_area):
        edge_list.append([off_conf + int(c_local), off_area + int(r_local), REL_CONF_AREA])

    edge_arr = np.array(edge_list, dtype=np.int64)
    edge_arr = np.unique(edge_arr, axis=0)
    print(f"Unique directed edges: {len(edge_arr)}", flush=True)

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
            "paper-author": REL_PAPER_AUTHOR,
            "paper-term": REL_PAPER_TERM,
            "paper-conference": REL_PAPER_CONF,
            "paper-area": REL_PAPER_AREA,
            "conference-area": REL_CONF_AREA,
        },
        "splits": {k: torch.tensor(v, dtype=torch.long) for k, v in splits.items()},
        "graph_kind": "dblp_cmpnn_skip_universal_paper_conf_area_train_pc_only",
        "paper_conf_edges": "train_only",
        "universal_area_channels": "paper_conf",
    }

    return torch.tensor(edge_arr, dtype=torch.long), meta


def save_variant(out_root: Path, variant: str, edge_list: torch.Tensor, meta_base: dict):
    out_dir = out_root / f"DBLP_cmpnn_skip_{variant}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(meta_base)
    meta["variant"] = variant
    torch.save(edge_list, out_dir / "edge_list.pt")
    torch.save(meta, out_dir / "meta.pt")
    print(f"Wrote {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="DBLP CMPNN skip: paper+conf area only, universal graph → v1/v2/v3"
    )
    ap.add_argument("--variant", default="v1,v2,v3", help="Comma list; same graph in each folder")
    ap.add_argument("--raw-dir", default="../MAGNN/data/raw/DBLP")
    ap.add_argument(
        "--shared-npz",
        default="../MAGNN/data/preprocessed/DBLP_shared_splits/DBLP_pc_shared_splits.npz",
    )
    ap.add_argument("--min-conf", type=int, default=0)
    ap.add_argument("--out-dir", default="data/preprocessed")
    args = ap.parse_args()

    set_seed()
    print("=== DBLP CMPNN skip: universal (P-area + C-area, train P–C only) ===", flush=True)

    edge_list, meta = build_universal_edge_list_and_meta(args)
    out_root = Path(args.out_dir)

    for v in _parse_variants(args.variant):
        save_variant(out_root, v, edge_list, meta)

    print("Done. v1/v2/v3 identical → τ≈1 across variants for the same seed.", flush=True)


if __name__ == "__main__":
    main()
