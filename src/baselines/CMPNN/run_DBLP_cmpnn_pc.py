#!/usr/bin/env python3
"""
  python preprocess_DBLP_cmpnn_pc.py --variant v1,v2,v3
  python run_DBLP_cmpnn_pc.py --variants v1,v2,v3 --save-postfix DBLP_cmpnn_pc
  python run_DBLP_cmpnn_pc.py --compare-only --variants v1,v2,v3 --save-postfix DBLP_cmpnn_pc
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from run_DBLP_cmpnn_skip import (
    _parse_variants,
    _variant_pairs_from_csv_spec,
    kendall_csv_with_hits,
    run_one_variant,
    summarize,
)

# After preprocess: ``DBLP_cmpnn_pc_v1``, ``_v2``, ``_v3`` (area channel differs by variant).
BASE_PC = {
    "v1": "data/preprocessed/DBLP_cmpnn_pc_v1",
    "v2": "data/preprocessed/DBLP_cmpnn_pc_v2",
    "v3": "data/preprocessed/DBLP_cmpnn_pc_v3",
}


def _parse_variants_pc(s):
    out = [x.strip().lower() for x in str(s).split(",") if x.strip()]
    for v in out:
        if v not in BASE_PC:
            raise SystemExit(f"Unknown variant {v!r}; expected one of {','.join(BASE_PC)}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DBLP CMPNN paper–conf (non-skip variants)")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--variants", default="v1,v2,v3")
    ap.add_argument("--input-dim", type=int, default=32)
    ap.add_argument("--hidden-dims", default="32,32")
    ap.add_argument("--message-func", default="distmult")
    ap.add_argument("--aggregate-func", default="pna")
    ap.add_argument("--short-cut", action="store_true")
    ap.add_argument("--layer-norm", action="store_true")
    ap.add_argument("--num-mlp-layer", type=int, default=2)
    ap.add_argument("--rgcn", action="store_true")
    ap.add_argument("--num-bases", type=int, default=None)
    ap.add_argument("--initialization", default="Query")
    ap.add_argument("--has-readout", action="store_true")
    ap.add_argument("--readout-type", default="mean")
    ap.add_argument("--query-specific-readout", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="LP often needs more epochs than small defaults; tune with val F1.",
    )
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--neg-k",
        type=int,
        default=3,
        help="Max negatives per paper (train/val/test). Small k (e.g. 3) ⇒ few candidates ⇒ MRR/Hits@1 can saturate.",
    )
    ap.add_argument("--ckpt", default="checkpoint/dblp_cmpnn_pc.pt")
    ap.add_argument("--seeds", default="1566911444,20241017,20251017")
    ap.add_argument("--save-postfix", default="DBLP_cmpnn_pc")
    ap.add_argument(
        "--compare",
        default="",
        help="Comma-separated variants for Kendall after training, e.g. v1,v2,v3 (all pairs) or v1,v3.",
    )
    ap.add_argument(
        "--compare-only",
        action="store_true",
        help="Only Kendall from existing CSVs; use --compare or default --variants for variant list.",
    )
    ap.add_argument("--score-csv-a", default="")
    ap.add_argument("--score-csv-b", default="")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    merge_keys = ["paper_id", "conf_id"]
    query_col = "paper_id"

    def _print_kendall_block(va: str, vb: str):
        o_taus, h1_taus, h3_taus = [], [], []
        for sd in seeds:
            pa = f"{args.save_postfix}_{va}_seed{sd}_scores.csv"
            pb = f"{args.save_postfix}_{vb}_seed{sd}_scores.csv"
            if not os.path.isfile(pa) or not os.path.isfile(pb):
                print(f"  seed {sd}: missing {pa} or {pb}", flush=True)
                continue
            kk = kendall_csv_with_hits(pa, pb, merge_keys, query_col)
            print(
                f"  seed {sd} | pair_rows={kk['overall_n']} | overall_τ={kk['overall_tau']:.6f} | "
                f"papers={kk['hits_n']} | Hits@1_τ={kk['h1_tau']:.6f} | Hits@3_τ={kk['h3_tau']:.6f}",
                flush=True,
            )
            if np.isfinite(kk["overall_tau"]):
                o_taus.append(float(kk["overall_tau"]))
            if np.isfinite(kk["h1_tau"]):
                h1_taus.append(float(kk["h1_tau"]))
            if np.isfinite(kk["h3_tau"]):
                h3_taus.append(float(kk["h3_tau"]))
        if o_taus:
            oa = np.array(o_taus, dtype=float)
            msg = f"  Mean overall τ: {oa.mean():.6f} ± {oa.std(ddof=0):.6f}"
            if h1_taus:
                a1 = np.array(h1_taus, dtype=float)
                msg += f" | Hits@1 τ: {a1.mean():.6f} ± {a1.std(ddof=0):.6f}"
            if h3_taus:
                a3 = np.array(h3_taus, dtype=float)
                msg += f" | Hits@3 τ: {a3.mean():.6f} ± {a3.std(ddof=0):.6f}"
            print(msg, flush=True)

    if args.compare_only:
        if args.score_csv_a and args.score_csv_b:
            kk = kendall_csv_with_hits(args.score_csv_a, args.score_csv_b, merge_keys, query_col)
            print(
                f"pair_rows={kk['overall_n']} | overall_τ={kk['overall_tau']:.6f} | "
                f"papers={kk['hits_n']} | Hits@1_τ={kk['h1_tau']:.6f} | Hits@3_τ={kk['h3_tau']:.6f}",
                flush=True,
            )
            sys.exit(0)

        spec = args.compare.strip() if (args.compare and str(args.compare).strip()) else args.variants
        _parse_variants_pc(spec)
        pairs = _variant_pairs_from_csv_spec(spec)
        print(f"\n########## Kendall τ (overall + per-paper Hits@1/Hits@3) | spec={spec!r} ##########")
        for va, vb in pairs:
            print(f"\n--- {va} vs {vb} ---", flush=True)
            _print_kendall_block(va, vb)
        sys.exit(0)

    variants = _parse_variants_pc(args.variant) if args.variant else _parse_variants_pc(args.variants)

    by_v = {}
    for v in variants:
        print(f"\n########## DBLP CMPNN pc (non-skip) variant {v} ##########", flush=True)
        metrics_runs = []
        for seed in seeds:
            stats = run_one_variant(args, v, seed, preprocessed_base=BASE_PC[v])
            metrics_runs.append(stats)
        by_v[v] = metrics_runs
        summarize(metrics_runs)

    print("\nDBLP CMPNN pc (non-skip) summary | mean ± std over seeds")
    print("Variant | Precision | Recall | F1 | Hits@1 | Hits@3 | MRR | Train Time (s) | Epochs")
    for v in variants:
        mets = by_v[v]

        def mstd(key):
            arr = np.array([x[key] for x in mets], dtype=float)
            return float(arr.mean()), float(arr.std(ddof=0))

        p_m, p_s = mstd("Precision")
        r_m, r_s = mstd("Recall")
        f_m, f_s = mstd("F1")
        h1_m, h1_s = mstd("Hits@1")
        h3_m, h3_s = mstd("Hits@3")
        m_m, m_s = mstd("MRR")
        tw_m, tw_s = mstd("Train Time (s)")
        e_m, e_s = mstd("Epochs")
        print(
            f"{v:>7} | {p_m:.4f} ± {p_s:.4f} | {r_m:.4f} ± {r_s:.4f} | {f_m:.4f} ± {f_s:.4f} | "
            f"{h1_m:.4f} ± {h1_s:.4f} | {h3_m:.4f} ± {h3_s:.4f} | {m_m:.4f} ± {m_s:.4f} | "
            f"{tw_m:.2f} ± {tw_s:.2f} | {e_m:.2f} ± {e_s:.2f}",
            flush=True,
        )

    if args.compare.strip():
        _parse_variants_pc(args.compare.strip())
        pairs = _variant_pairs_from_csv_spec(args.compare.strip())
        print(f"\n########## Kendall τ after training | compare={args.compare!r} ##########")
        for va, vb in pairs:
            print(f"\n--- {va} vs {vb} ---", flush=True)
            _print_kendall_block(va, vb)
