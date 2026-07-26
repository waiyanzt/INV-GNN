"""
RGCN link prediction on WordNet graph variants (wordnet_3hops_augmented_full).

Trains one model per variant (no_changes / all_inverse_edges / transitive_edges /
universal_edges) with shared validation/test splits. Uses WordNet-specific hyperparameters: 200-dim
embeddings, single RGCN layer, basis decomposition (2 bases), c_i
normalization, edge dropout, full-batch training, one negative per positive.

Usage:
  python run_wordnet_lp.py --variant no_changes --epochs 5
  python run_wordnet_lp.py
"""

import argparse
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from model_RGCN_lp_wordnet import WordNetRGCNLinkPredictor
from wordnet_lp import (
    CANONICAL_VARIANTS,
    VARIANT_ALIASES,
    WordNetLPDataset,
    canonicalize_variant,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_SPLITS_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "wordnet_3hops_augmented_full",
    "wordnet_splits.npz",
)
DEFAULT_CHECKPOINT_DIR = os.path.join(
    os.path.dirname(__file__), "checkpoint", "wordnet"
)

VARIANTS = list(CANONICAL_VARIANTS)
SEEDS = [1566911444, 20241017, 20251017]

# ---------------------------------------------------------------------------
# WordNet RGCN hyperparameters (paper-style)
# ---------------------------------------------------------------------------

HIDDEN_DIM = 200
# The four variants use one shared relation vocabulary. A larger basis count
# avoids forcing every base, inverse, and shortcut relation into an extremely
# small coefficient subspace. The exact vocabulary size is read from the NPZ.
NUM_BASES = 30
LR = 0.01
NEG_PER_POS = 1
PATIENCE = 30
MAX_EPOCHS = 3000
EVAL_INTERVAL = 1
BINARY_K = 50
WEIGHT_DECAY = 0.01


# ---------------------------------------------------------------------------
# Negative sampling (training)
# ---------------------------------------------------------------------------


def sample_neg_train(pos_array: np.ndarray, num_entities: int, k: int, rng) -> np.ndarray:
    """Online negative sampling with 50/50 corrupt-head/corrupt-tail. Returns (N*k, 3) int32."""
    n = len(pos_array)
    total = n * k

    heads = np.repeat(pos_array[:, 0], k)
    rels = np.repeat(pos_array[:, 1], k)
    tails = np.repeat(pos_array[:, 2], k)

    corrupt_values = rng.integers(0, num_entities, size=total, dtype=np.int32)
    corrupt_head = rng.random(total) < 0.5

    neg_heads = np.where(corrupt_head, corrupt_values, heads)
    neg_tails = np.where(corrupt_head, tails, corrupt_values)

    return np.stack([neg_heads, rels, neg_tails], axis=1).astype(np.int32)


# ---------------------------------------------------------------------------
# Filter dicts
# ---------------------------------------------------------------------------


def build_filter_dicts(*triple_arrays):
    tail_raw = defaultdict(set)
    head_raw = defaultdict(set)
    for arr in triple_arrays:
        for h, r, t in arr.tolist():
            tail_raw[(h, r)].add(t)
            head_raw[(r, t)].add(h)
    tail_filters = {k: frozenset(v) for k, v in tail_raw.items()}
    head_filters = {k: frozenset(v) for k, v in head_raw.items()}
    return tail_filters, head_filters


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model,
    entity_embs: torch.Tensor,
    pos_array: np.ndarray,
    device,
    tail_filters: dict,
    head_filters: dict,
    num_entities: int,
    eval_batch_size: int = 512,
    binary_k: int = BINARY_K,
):
    model.eval()

    binary_rng = np.random.default_rng(42)

    all_entity_embs = entity_embs

    raw_ranks_tail = []
    filt_ranks_tail = []
    raw_ranks_head = []
    filt_ranks_head = []

    all_logits = []
    all_labels = []

    pos_t = torch.from_numpy(pos_array).long().to(device)
    n = len(pos_array)

    for start in range(0, n, eval_batch_size):
        batch = pos_t[start : start + eval_batch_size]
        h_idx = batch[:, 0]
        r_idx = batch[:, 1]
        t_idx = batch[:, 2]

        h_emb = all_entity_embs[h_idx]
        r_emb = model.rel_emb(r_idx)
        t_emb = all_entity_embs[t_idx]

        hr = h_emb * r_emb
        scores_tail = hr @ all_entity_embs.T

        rt = r_emb * t_emb
        scores_head = rt @ all_entity_embs.T

        scores_tail_cpu = scores_tail.cpu()
        scores_head_cpu = scores_head.cpu()

        for i in range(len(batch)):
            h = h_idx[i].item()
            r = r_idx[i].item()
            t = t_idx[i].item()

            s_tail = scores_tail_cpu[i]
            s_head = scores_head_cpu[i]

            true_tail_score = s_tail[t].item()
            true_head_score = s_head[h].item()

            raw_rank_t = int((s_tail >= true_tail_score).sum().item())
            raw_rank_h = int((s_head >= true_head_score).sum().item())
            raw_ranks_tail.append(raw_rank_t)
            raw_ranks_head.append(raw_rank_h)

            s_tail_f = s_tail.clone()
            for other_t in tail_filters.get((h, r), frozenset()):
                if other_t != t:
                    s_tail_f[other_t] = float("-inf")
            filt_rank_t = int((s_tail_f >= true_tail_score).sum().item())
            filt_ranks_tail.append(filt_rank_t)

            s_head_f = s_head.clone()
            for other_h in head_filters.get((r, t), frozenset()):
                if other_h != h:
                    s_head_f[other_h] = float("-inf")
            filt_rank_h = int((s_head_f >= true_head_score).sum().item())
            filt_ranks_head.append(filt_rank_h)

            known_tails = tail_filters.get((h, r), frozenset())
            negs = []
            candidates = binary_rng.integers(0, num_entities, size=binary_k * 4)
            for c in candidates:
                if c not in known_tails and len(negs) < binary_k:
                    negs.append(int(c))
            while len(negs) < binary_k:
                negs.append(int(binary_rng.integers(0, num_entities)))

            neg_scores = s_tail[negs].numpy()
            all_logits.append(true_tail_score)
            all_logits.extend(neg_scores.tolist())
            all_labels.append(1)
            all_labels.extend([0] * binary_k)

    raw_ranks = np.array(raw_ranks_tail + raw_ranks_head, dtype=np.float32)
    filt_ranks = np.array(filt_ranks_tail + filt_ranks_head, dtype=np.float32)

    logits = np.array(all_logits, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int32)
    preds = (logits > 0).astype(np.int32)

    return {
        "raw_MRR": float(np.mean(1.0 / raw_ranks)),
        "filtered_MRR": float(np.mean(1.0 / filt_ranks)),
        "Hits@1": float(np.mean(filt_ranks <= 1)),
        "Hits@3": float(np.mean(filt_ranks <= 3)),
        "Hits@10": float(np.mean(filt_ranks <= 10)),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_and_eval(variant: str, seed: int, device, args):
    print(f"\n{'='*60}")
    print(f"  Variant: {variant} | Seed: {seed}")
    print(f"{'='*60}")

    dataset = WordNetLPDataset(variant, args.splits_path)
    print(dataset)

    # Make both sampling and model initialization reproducible for each seed.
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tail_filters, head_filters = build_filter_dicts(
        dataset.train_pos, dataset.val_pos, dataset.test_pos
    )

    edge_index, edge_type = dataset.get_train_graph(device=device)

    model = WordNetRGCNLinkPredictor(
        num_entities=dataset.num_entities,
        num_relations=dataset.num_relations,
        hidden_dim=args.hidden_dim,
        num_bases=args.num_bases,
        edge_dropout_other=args.edge_dropout_other,
        root_dropout_loop=args.root_dropout_loop,
    ).to(device)

    optimizer = torch.optim.Adam(
        [
            {
                "params": [p for n, p in model.named_parameters() if "rel_emb" not in n],
                "weight_decay": 0.0,
            },
            {
                "params": model.rel_emb.parameters(),
                "weight_decay": args.weight_decay,
            },
        ],
        lr=args.lr,
    )

    train_pos = dataset.train_pos
    batch_size = args.batch_size if args.batch_size > 0 else len(train_pos)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(
        args.checkpoint_dir,
        f"best_model_R{dataset.num_relations}_seed_{seed}_variant_{variant}.pt",
    )

    best_val_filt_mrr = -1.0
    patience_counter = 0
    epochs_trained = 0
    checkpoint_saved_this_run = False

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        epochs_trained = epoch
        model.train()

        perm = rng.permutation(len(train_pos))
        train_shuffled = train_pos[perm]

        total_loss = 0.0
        num_batches = 0

        for start in range(0, len(train_shuffled), batch_size):
            batch_pos = train_shuffled[start : start + batch_size]
            batch_neg = sample_neg_train(
                batch_pos, dataset.num_entities, args.neg_per_pos, rng
            )

            pos_t = torch.from_numpy(batch_pos).long().to(device)
            neg_t = torch.from_numpy(batch_neg).long().to(device)

            entity_embs = model.encode(edge_index, edge_type, training=True)

            pos_scores = model.score(entity_embs, pos_t[:, 0], pos_t[:, 1], pos_t[:, 2])
            neg_scores = model.score(entity_embs, neg_t[:, 0], neg_t[:, 1], neg_t[:, 2])

            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat(
                [
                    torch.ones(len(pos_scores), device=device),
                    torch.zeros(len(neg_scores), device=device),
                ]
            )
            loss = F.binary_cross_entropy_with_logits(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        if epoch % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                entity_embs = model.encode(edge_index, edge_type, training=False)
            val_metrics = evaluate(
                model,
                entity_embs,
                dataset.val_pos,
                device,
                tail_filters,
                head_filters,
                dataset.num_entities,
            )
            val_filt_mrr = val_metrics["filtered_MRR"]

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:4d} | loss {avg_loss:.4f} | "
                f"val filtered_MRR {val_filt_mrr:.4f} | "
                f"raw_MRR {val_metrics['raw_MRR']:.4f} | "
                f"H@1 {val_metrics['Hits@1']:.4f} | "
                f"H@3 {val_metrics['Hits@3']:.4f} | "
                f"H@10 {val_metrics['Hits@10']:.4f} | "
                f"{elapsed:.0f}s"
            )

            if val_filt_mrr > best_val_filt_mrr:
                best_val_filt_mrr = val_filt_mrr
                torch.save(model.state_dict(), ckpt_path)
                checkpoint_saved_this_run = True
                patience_counter = 0
                print(f"  --> New best ({val_filt_mrr:.4f}), checkpoint saved.")
            else:
                patience_counter += args.eval_interval
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                    break

    run_time = time.time() - t0

    if checkpoint_saved_this_run:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"\nLoaded best checkpoint: {os.path.basename(ckpt_path)}")
    else:
        print(
            "\nNo validation checkpoint was created in this run "
            "(epochs may be smaller than eval_interval); using final model weights."
        )
    model.to(device)

    model.eval()
    with torch.no_grad():
        entity_embs = model.encode(edge_index, edge_type, training=False)

    val_metrics = evaluate(
        model,
        entity_embs,
        dataset.val_pos,
        device,
        tail_filters,
        head_filters,
        dataset.num_entities,
    )
    test_metrics = evaluate(
        model,
        entity_embs,
        dataset.test_pos,
        device,
        tail_filters,
        head_filters,
        dataset.num_entities,
    )

    def fmt(m):
        return (
            f"filtered_MRR={m['filtered_MRR']:.4f}  raw_MRR={m['raw_MRR']:.4f}  "
            f"H@1={m['Hits@1']:.4f}  H@3={m['Hits@3']:.4f}  H@10={m['Hits@10']:.4f}  "
            f"Acc={m['accuracy']:.4f}  Prec={m['precision']:.4f}  "
            f"Rec={m['recall']:.4f}  F1={m['macro_f1']:.4f}"
        )

    print(f"\n  Epochs trained: {epochs_trained}  Run time: {run_time:.1f}s")
    print(f"  Final val:  {fmt(val_metrics)}")
    print(f"  Final test: {fmt(test_metrics)}")

    return val_metrics, test_metrics, run_time, epochs_trained


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANT_ALIASES) + ["all"],
        default="all",
        help=(
            "Which graph variant to train on. Canonical names and paper aliases "
            "(unchanged/inverse/transitive/universal) are accepted."
        ),
    )
    parser.add_argument(
        "--splits-path",
        "--splits_path",
        dest="splits_path",
        default=DEFAULT_SPLITS_PATH,
        help="Path to the four-variant wordnet_splits.npz",
    )
    parser.add_argument(
        "--checkpoint-dir",
        "--checkpoint_dir",
        dest="checkpoint_dir",
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory for per-variant checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--num_bases", type=int, default=NUM_BASES)
    parser.add_argument(
        "--edge_dropout_other",
        type=float,
        default=0.4,
        help="Drop probability for neighbor (non-root) edges during training",
    )
    parser.add_argument(
        "--root_dropout_loop",
        type=float,
        default=0.2,
        help="Drop probability for the root (self) connection during training",
    )
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=WEIGHT_DECAY,
        help="L2 penalty on DistMult decoder (rel_emb) only",
    )
    parser.add_argument("--neg_per_pos", type=int, default=NEG_PER_POS)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Mini-batch size; 0 = full batch (all training triples per step)",
    )
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--eval_interval", type=int, default=EVAL_INTERVAL)
    args = parser.parse_args()

    args.splits_path = os.path.abspath(os.path.expanduser(args.splits_path))
    args.checkpoint_dir = os.path.abspath(os.path.expanduser(args.checkpoint_dir))
    if not os.path.isfile(args.splits_path):
        raise FileNotFoundError(f"Split NPZ not found: {args.splits_path}")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.eval_interval <= 0:
        raise ValueError("--eval_interval must be positive")
    if args.neg_per_pos <= 0:
        raise ValueError("--neg_per_pos must be positive")

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Splits: {args.splits_path}")
    print(f"Checkpoints: {args.checkpoint_dir}")

    variants_to_run = (
        VARIANTS if args.variant == "all" else [canonicalize_variant(args.variant)]
    )

    results = {v: [] for v in variants_to_run}

    for variant in variants_to_run:
        for seed in args.seeds:
            val_m, test_m, rt, ep = train_and_eval(variant, seed, device, args)
            results[variant].append((val_m, test_m, rt, ep))

    METRICS = [
        "filtered_MRR",
        "raw_MRR",
        "Hits@1",
        "Hits@3",
        "Hits@10",
        "accuracy",
        "precision",
        "recall",
        "macro_f1",
    ]

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    for variant, seed_results in results.items():
        run_times = [r[2] for r in seed_results]
        epochs_list = [r[3] for r in seed_results]
        per_seed_ep = ", ".join(
            f"seed={args.seeds[i]}: {seed_results[i][3]}"
            for i in range(len(seed_results))
        )
        print(f"\n  Variant: {variant}")
        print(f"    RunTime(s): {np.mean(run_times):.1f} ± {np.std(run_times):.1f}")
        print(
            f"    Epochs:     {np.mean(epochs_list):.1f} ± {np.std(epochs_list):.1f}  ({per_seed_ep})"
        )
        for split_name, idx in [("val", 0), ("test", 1)]:
            all_m = [r[idx] for r in seed_results]
            parts = []
            for key in METRICS:
                vals = [m[key] for m in all_m]
                parts.append(f"{key}={np.mean(vals):.4f}±{np.std(vals):.4f}")
            print(f"    {split_name}: " + "  ".join(parts))


if __name__ == "__main__":
    main()
