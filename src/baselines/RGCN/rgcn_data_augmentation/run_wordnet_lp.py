"""
RGCN link prediction on WordNet graph variants (wordnet_3hops_augmented_full).

Trains one model per variant (no_changes / all_inverse_edges / transitive_edges /
universal_edges) with shared validation/test splits. Uses WordNet-specific hyperparameters: 200-dim
embeddings, single RGCN layer, basis decomposition (2 bases), c_i
normalization, edge dropout, full-batch training, one negative per positive.

Usage:
  python run_wordnet_lp.py --variant no_changes --epochs 5
  python run_wordnet_lp.py
  python run_wordnet_lp.py --variant no_changes --checkpoint-only
"""

import argparse
import csv
import gc
import os
import time
from collections import defaultdict
from pathlib import Path

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
from rgcn_aug_common import (
    checkpoint_size_bytes,
    cuda_memory_stats,
    model_memory_bytes,
    process_peak_rss_bytes,
    reset_cuda_peak,
    write_json,
)
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
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "results", "rgcn_baseline", "WORDNET"
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
# Telemetry and result serialization
# ---------------------------------------------------------------------------


def checkpoint_path(args, dataset, seed: int, variant: str) -> Path:
    return Path(args.checkpoint_dir) / (
        f"best_model_R{dataset.num_relations}_seed_{seed}_variant_{variant}.pt"
    )


def seed_output_dir(args, variant: str, seed: int) -> Path:
    path = Path(args.output_dir) / variant / f"seed_{seed}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_checkpoint_state(model, path: Path, device) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=device)
    load_result = model.load_state_dict(state, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint incompatibility for {path}: "
            f"missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )


def print_memory_summary(memory: dict) -> None:
    mib = 1024 ** 2
    print("  Memory telemetry:")
    print(f"    parameters: {memory['parameter_bytes'] / mib:.2f} MiB")
    print(f"    buffers: {memory['buffer_bytes'] / mib:.2f} MiB")
    print(f"    static model: {memory['static_model_bytes'] / mib:.2f} MiB")
    print(f"    checkpoint: {memory['checkpoint_bytes'] / mib:.2f} MiB")
    print(f"    process peak RSS: {memory['process_peak_rss_bytes'] / mib:.2f} MiB")
    for phase_key in ("training_gpu", "representative_training_step_gpu", "inference_gpu"):
        if phase_key not in memory:
            continue
        phase = memory[phase_key]
        print(
            f"    {phase_key}: peak allocated="
            f"{phase['gpu_peak_allocated_bytes'] / mib:.2f} MiB, "
            f"peak reserved={phase['gpu_peak_reserved_bytes'] / mib:.2f} MiB"
        )


def flatten_summary(summary: dict) -> dict:
    row = {
        "seed": summary["seed"],
        "variant": summary["variant"],
        "mode": summary["mode"],
        "epochs_trained": summary.get("epochs_trained"),
        "training_seconds": summary.get("training_seconds"),
        "representative_training_step_seconds": summary.get(
            "representative_training_step_seconds"
        ),
        "checkpoint": summary["checkpoint"],
    }
    for split in ("val", "test"):
        for key, value in summary[f"{split}_metrics"].items():
            row[f"{split}_{key}"] = value
    memory = summary["memory"]
    for key in (
        "parameter_bytes",
        "buffer_bytes",
        "static_model_bytes",
        "checkpoint_bytes",
        "process_peak_rss_bytes",
    ):
        row[key] = memory[key]
    for phase_key in (
        "training_gpu",
        "representative_training_step_gpu",
        "inference_gpu",
    ):
        for key, value in memory.get(phase_key, {}).items():
            row[f"{phase_key}_{key}"] = value
    return row


def write_variant_results(args, variant: str, summaries: list[dict]) -> None:
    variant_dir = Path(args.output_dir) / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    profile_only = all(summary["mode"] == "checkpoint_profile" for summary in summaries)
    stem = "checkpoint_profile" if profile_only else "baseline"
    rows = [flatten_summary(summary) for summary in summaries]
    csv_path = variant_dir / f"{stem}_seed_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        variant_dir / f"{stem}_all_seed_summaries.json",
        {"runs": summaries},
    )


def format_metrics(metrics: dict) -> str:
    return (
        f"filtered_MRR={metrics['filtered_MRR']:.4f}  "
        f"raw_MRR={metrics['raw_MRR']:.4f}  "
        f"H@1={metrics['Hits@1']:.4f}  "
        f"H@3={metrics['Hits@3']:.4f}  "
        f"H@10={metrics['Hits@10']:.4f}  "
        f"Acc={metrics['accuracy']:.4f}  "
        f"Prec={metrics['precision']:.4f}  "
        f"Rec={metrics['recall']:.4f}  "
        f"F1={metrics['macro_f1']:.4f}"
    )


def profile_checkpoint_and_eval(
    variant,
    seed,
    device,
    args,
    dataset,
    model,
    optimizer,
    edge_index,
    edge_type,
    tail_filters,
    head_filters,
    train_pos,
    batch_size,
    ckpt_path,
):
    """Profile a saved model without claiming to recover historical peaks."""
    load_checkpoint_state(model, ckpt_path, device)
    rng = np.random.default_rng(seed)
    batch_pos = train_pos[rng.permutation(len(train_pos))[:batch_size]]
    batch_neg = sample_neg_train(
        batch_pos, dataset.num_entities, args.neg_per_pos, rng
    )
    pos_t = torch.from_numpy(batch_pos).long().to(device)
    neg_t = torch.from_numpy(batch_neg).long().to(device)

    reset_cuda_peak(device)
    step_start = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    entity_embs = model.encode(edge_index, edge_type, training=True)
    pos_scores = model.score(
        entity_embs, pos_t[:, 0], pos_t[:, 1], pos_t[:, 2]
    )
    neg_scores = model.score(
        entity_embs, neg_t[:, 0], neg_t[:, 1], neg_t[:, 2]
    )
    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat(
        [
            torch.ones(len(pos_scores), device=device),
            torch.zeros(len(neg_scores), device=device),
        ]
    )
    loss = F.binary_cross_entropy_with_logits(scores, labels)
    loss.backward()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    step_seconds = time.perf_counter() - step_start
    representative_training_gpu = cuda_memory_stats(device)

    # The profiling step mutates only the in-memory model. Restore the persisted
    # checkpoint and release optimizer/activation tensors before inference.
    del (
        batch_pos,
        batch_neg,
        pos_t,
        neg_t,
        entity_embs,
        pos_scores,
        neg_scores,
        scores,
        labels,
        loss,
        optimizer,
    )
    load_checkpoint_state(model, ckpt_path, device)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    reset_cuda_peak(device)
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
    inference_gpu = cuda_memory_stats(device)

    memory = {
        **model_memory_bytes(model),
        "checkpoint_bytes": checkpoint_size_bytes(ckpt_path),
        "process_peak_rss_bytes": process_peak_rss_bytes(),
        "representative_training_step_gpu": representative_training_gpu,
        "inference_gpu": inference_gpu,
    }
    summary = {
        "dataset": "WORDNET",
        "model": "legacy_WordNetRGCNLinkPredictor",
        "variant": variant,
        "seed": seed,
        "mode": "checkpoint_profile",
        "checkpoint": str(ckpt_path.resolve()),
        "splits_npz": str(Path(args.splits_path).resolve()),
        "representative_training_step_seconds": step_seconds,
        "telemetry_semantics": {
            "representative_training_step_gpu": (
                "Fresh replay of one forward/backward/Adam step from the saved "
                "weights on the current device; not the historical training peak."
            ),
            "inference_gpu": (
                "Fresh full validation-and-test evaluation on the current device."
            ),
            "process_peak_rss_bytes": (
                "Process-wide ru_maxrss observed during this profiling process."
            ),
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "memory": memory,
    }
    seed_dir = seed_output_dir(args, variant, seed)
    write_json(seed_dir / "checkpoint_profile.json", summary)

    print(f"\n  Checkpoint profile: {ckpt_path.name}")
    print(f"  Representative training step: {step_seconds:.3f}s")
    print(f"  Final val:  {format_metrics(val_metrics)}")
    print(f"  Final test: {format_metrics(test_metrics)}")
    print_memory_summary(memory)
    return summary


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
    ckpt_path = checkpoint_path(args, dataset, seed, variant)

    if args.checkpoint_only:
        return profile_checkpoint_and_eval(
            variant,
            seed,
            device,
            args,
            dataset,
            model,
            optimizer,
            edge_index,
            edge_type,
            tail_filters,
            head_filters,
            train_pos,
            batch_size,
            ckpt_path,
        )

    reset_cuda_peak(device)
    training_start_peak_rss = process_peak_rss_bytes()

    best_val_filt_mrr = -1.0
    patience_counter = 0
    epochs_trained = 0
    checkpoint_saved_this_run = False

    t0 = time.perf_counter()
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

            elapsed = time.perf_counter() - t0
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

    run_time = time.perf_counter() - t0
    training_gpu = cuda_memory_stats(device)
    training_peak_rss = max(training_start_peak_rss, process_peak_rss_bytes())

    if checkpoint_saved_this_run:
        load_checkpoint_state(model, ckpt_path, device)
        print(f"\nLoaded best checkpoint: {ckpt_path.name}")
    else:
        print(
            "\nNo validation checkpoint was created in this run "
            "(epochs may be smaller than eval_interval); using final model weights."
        )
    model.to(device)

    # Exclude optimizer state and retained training tensors from the separately
    # reported inference peak.
    del (
        optimizer,
        pos_t,
        neg_t,
        entity_embs,
        pos_scores,
        neg_scores,
        scores,
        labels,
        loss,
        batch_neg,
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    reset_cuda_peak(device)
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
    inference_gpu = cuda_memory_stats(device)

    memory = {
        **model_memory_bytes(model),
        "checkpoint_bytes": checkpoint_size_bytes(ckpt_path),
        "process_peak_rss_bytes": max(
            training_peak_rss, process_peak_rss_bytes()
        ),
        "training_gpu": training_gpu,
        "inference_gpu": inference_gpu,
    }
    summary = {
        "dataset": "WORDNET",
        "model": "legacy_WordNetRGCNLinkPredictor",
        "variant": variant,
        "seed": seed,
        "mode": "full_training",
        "checkpoint": str(ckpt_path.resolve()),
        "splits_npz": str(Path(args.splits_path).resolve()),
        "epochs_trained": epochs_trained,
        "training_seconds": run_time,
        "best_val_filtered_MRR": best_val_filt_mrr,
        "telemetry_semantics": {
            "training_gpu": (
                "Peak CUDA allocator statistics across training and periodic "
                "validation after model/graph construction."
            ),
            "inference_gpu": (
                "Peak CUDA allocator statistics for final validation-and-test "
                "evaluation after loading the best checkpoint."
            ),
            "process_peak_rss_bytes": (
                "Process-wide ru_maxrss; this value is cumulative when multiple "
                "seeds run in one Python process."
            ),
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "memory": memory,
    }
    seed_dir = seed_output_dir(args, variant, seed)
    write_json(seed_dir / "summary.json", summary)

    print(f"\n  Epochs trained: {epochs_trained}  Run time: {run_time:.1f}s")
    print(f"  Final val:  {format_metrics(val_metrics)}")
    print(f"  Final test: {format_metrics(test_metrics)}")
    print_memory_summary(memory)

    return summary


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
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for structured per-seed metrics and telemetry",
    )
    parser.add_argument(
        "--checkpoint-only",
        action="store_true",
        help=(
            "Do not retrain. Load each existing checkpoint, reproduce evaluation "
            "metrics, and measure fresh inference plus representative-step memory."
        ),
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
    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
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
    print(f"Results: {args.output_dir}")
    if args.checkpoint_only:
        print(
            "Mode: checkpoint-only (fresh representative-step and inference "
            "telemetry; historical training peaks are not recoverable)"
        )

    variants_to_run = (
        VARIANTS if args.variant == "all" else [canonicalize_variant(args.variant)]
    )

    results = {v: [] for v in variants_to_run}

    for variant in variants_to_run:
        for seed in args.seeds:
            summary = train_and_eval(variant, seed, device, args)
            results[variant].append(summary)
        write_variant_results(args, variant, results[variant])

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
    print("CHECKPOINT PROFILE SUMMARY" if args.checkpoint_only else "SUMMARY")
    print(f"{'='*80}")

    for variant, seed_results in results.items():
        print(f"\n  Variant: {variant}")
        if args.checkpoint_only:
            step_times = [
                result["representative_training_step_seconds"]
                for result in seed_results
            ]
            print(
                f"    RepresentativeStep(s): "
                f"{np.mean(step_times):.3f} ± {np.std(step_times):.3f}"
            )
        else:
            run_times = [result["training_seconds"] for result in seed_results]
            epochs_list = [result["epochs_trained"] for result in seed_results]
            per_seed_ep = ", ".join(
                f"seed={result['seed']}: {result['epochs_trained']}"
                for result in seed_results
            )
            print(
                f"    RunTime(s): "
                f"{np.mean(run_times):.1f} ± {np.std(run_times):.1f}"
            )
            print(
                f"    Epochs:     "
                f"{np.mean(epochs_list):.1f} ± {np.std(epochs_list):.1f}  "
                f"({per_seed_ep})"
            )
        for split_name in ("val", "test"):
            all_m = [result[f"{split_name}_metrics"] for result in seed_results]
            parts = []
            for key in METRICS:
                vals = [m[key] for m in all_m]
                parts.append(f"{key}={np.mean(vals):.4f}±{np.std(vals):.4f}")
            print(f"    {split_name}: " + "  ".join(parts))

        memory_phase = (
            "representative_training_step_gpu"
            if args.checkpoint_only
            else "training_gpu"
        )
        peak_allocated = [
            result["memory"][memory_phase]["gpu_peak_allocated_bytes"]
            / (1024 ** 2)
            for result in seed_results
        ]
        peak_reserved = [
            result["memory"][memory_phase]["gpu_peak_reserved_bytes"]
            / (1024 ** 2)
            for result in seed_results
        ]
        inference_allocated = [
            result["memory"]["inference_gpu"]["gpu_peak_allocated_bytes"]
            / (1024 ** 2)
            for result in seed_results
        ]
        print(
            f"    {memory_phase} peak allocated (MiB): "
            f"{np.mean(peak_allocated):.2f} ± {np.std(peak_allocated):.2f}"
        )
        print(
            f"    {memory_phase} peak reserved (MiB): "
            f"{np.mean(peak_reserved):.2f} ± {np.std(peak_reserved):.2f}"
        )
        print(
            f"    inference peak allocated (MiB): "
            f"{np.mean(inference_allocated):.2f} ± "
            f"{np.std(inference_allocated):.2f}"
        )

    print(f"\n[OK] Structured results written under {args.output_dir}")


if __name__ == "__main__":
    main()
