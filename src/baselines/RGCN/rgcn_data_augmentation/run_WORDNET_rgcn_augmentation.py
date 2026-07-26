#!/usr/bin/env python3
"""Joint graph-variant augmentation for the legacy WordNet RGCN model.

The encoder/decoder, c_i normalization, edge dropout, root dropout, optimizer
parameter groups, negative sampling, filtered ranking, and checkpoint selection
match run_wordnet_lp.py/model_RGCN_lp_wordnet.py. One model/optimizer is shared
across variants.

Two evaluation families are reported:
1. legacy-compatible per-variant metrics, for comparison with old runs;
2. shared fixed-candidate metrics, used only for cross-variant invariance.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from model_RGCN_lp_wordnet import WordNetRGCNLinkPredictor
from wordnet_lp import (
    CANONICAL_VARIANTS,
    VARIANT_ALIASES,
    WordNetLPDataset,
    canonicalize_variant,
)
from rgcn_aug_common import (
    EarlyStopper,
    atomic_torch_save,
    atomic_write_csv,
    checkpoint_size_bytes,
    cuda_memory_stats,
    link_binary_metrics,
    load_latest_training_state,
    mean_dict,
    merge_cuda_memory_stats,
    model_memory_bytes,
    process_peak_rss_bytes,
    reset_cuda_peak,
    resolve_device,
    save_latest_training_state,
    set_determinism,
    torch_load_full,
    triple_link_invariance_rows,
    write_json,
)

BINARY_K = 50


def parse_csv(value: str) -> List[str]:
    out = [item.strip() for item in value.split(",") if item.strip()]
    if not out or len(out) != len(set(out)):
        raise SystemExit("Expected a nonempty comma-separated list without duplicates")
    return out


DEFAULT_VARIANTS = list(CANONICAL_VARIANTS)


def pack_keys(triples: np.ndarray, num_entities: int, num_relations: int) -> np.ndarray:
    triples = np.asarray(triples, dtype=np.int64)
    return (
        triples[:, 0] * (num_relations * num_entities)
        + triples[:, 1] * num_entities
        + triples[:, 2]
    )


def fixed_tail_candidates(
    positives: np.ndarray,
    known_keys: set[int],
    num_entities: int,
    num_relations: int,
    negatives_per_positive: int,
    seed: int,
):
    """Create one deterministic shared candidate table for invariance metrics."""
    rng = np.random.default_rng(seed)
    triples = []
    labels = []
    query_ids = []
    for query_id, (head, relation, tail) in enumerate(
        np.asarray(positives, dtype=np.int64).tolist()
    ):
        triples.append((head, relation, tail))
        labels.append(1)
        query_ids.append(query_id)
        selected = set()
        attempts = 0
        while len(selected) < negatives_per_positive:
            candidate = int(rng.integers(0, num_entities))
            attempts += 1
            if attempts > max(1000, negatives_per_positive * 1000):
                raise RuntimeError(
                    f"Unable to sample {negatives_per_positive} shared negatives "
                    f"for query {query_id}. Reduce --fixed-negatives."
                )
            if candidate == tail or candidate in selected:
                continue
            key = head * (num_relations * num_entities) + relation * num_entities + candidate
            if key in known_keys:
                continue
            selected.add(candidate)
        for candidate in sorted(selected):
            triples.append((head, relation, candidate))
            labels.append(0)
            query_ids.append(query_id)
    return (
        np.asarray(triples, dtype=np.int64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(query_ids, dtype=np.int64),
    )


def resolve_splits_path(data_root: Path, splits_npz: str | None) -> Path:
    path = Path(splits_npz) if splits_npz else data_root / "wordnet_splits.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run preprocess_WORDNET_rgcn_augmentation.py first."
        )
    return path


def load_wordnet_splits(path: Path) -> Dict[str, np.ndarray]:
    """Load the exact NPZ schema used by the updated single-variant runner."""
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    required = {
        "val_pos",
        "test_pos",
        "entity_vocab",
        "relation_vocab",
        "num_entities",
        "num_relations",
        "base_relation_ids",
        *{f"train_pos_{variant}" for variant in DEFAULT_VARIANTS},
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{path} is missing required arrays: {missing}")

    if "variant_names" in payload:
        archive_variants = tuple(str(v) for v in np.asarray(payload["variant_names"]).tolist())
        if archive_variants != tuple(CANONICAL_VARIANTS):
            raise ValueError(
                f"Unexpected NPZ variant_names {archive_variants}; "
                f"expected {tuple(CANONICAL_VARIANTS)}"
            )
    if "format_version" in payload:
        format_version = str(np.asarray(payload["format_version"]).item())
        if format_version != "wordnet_lp_four_variants_v1":
            raise ValueError(
                f"Unsupported WordNet NPZ format_version {format_version!r}; "
                "expected 'wordnet_lp_four_variants_v1'"
            )
    return payload


def _triple_set(array: np.ndarray) -> set[tuple[int, int, int]]:
    return {tuple(map(int, row)) for row in np.asarray(array)}


def validate_leakage_free_archive(shared: Mapping[str, np.ndarray]) -> None:
    """Repeat the critical leakage checks before any training begins."""
    num_entities = int(shared["num_entities"])
    num_relations = int(shared["num_relations"])
    relation_vocab = [str(item) for item in shared["relation_vocab"].tolist()]
    if len(relation_vocab) != num_relations:
        raise ValueError("relation_vocab length does not match num_relations")

    val_pos = np.asarray(shared["val_pos"], dtype=np.int64)
    test_pos = np.asarray(shared["test_pos"], dtype=np.int64)
    val_set = _triple_set(val_pos)
    test_set = _triple_set(test_pos)
    if val_set & test_set:
        raise ValueError("WordNet validation and test positives overlap")
    heldout = val_set | test_set

    for variant in DEFAULT_VARIANTS:
        train = np.asarray(shared[f"train_pos_{variant}"], dtype=np.int64)
        if train.ndim != 2 or train.shape[1] != 3:
            raise ValueError(f"train_pos_{variant} must have shape (N, 3)")
        if len(train) == 0:
            raise ValueError(f"train_pos_{variant} is empty")
        if train[:, [0, 2]].min() < 0 or train[:, [0, 2]].max() >= num_entities:
            raise ValueError(f"train_pos_{variant} contains an out-of-range entity ID")
        if train[:, 1].min() < 0 or train[:, 1].max() >= num_relations:
            raise ValueError(f"train_pos_{variant} contains an out-of-range relation ID")
        direct_overlap = _triple_set(train) & heldout
        if direct_overlap:
            raise ValueError(
                f"train_pos_{variant} contains {len(direct_overlap)} held-out positives"
            )

    relation_to_id = {name: idx for idx, name in enumerate(relation_vocab)}
    base_relation_ids = {int(value) for value in shared["base_relation_ids"].tolist()}
    inverse_id_by_base_id = {}
    for base_id in base_relation_ids:
        inverse_name = f"{relation_vocab[base_id]}__inv"
        if inverse_name in relation_to_id:
            inverse_id_by_base_id[base_id] = relation_to_id[inverse_name]
    heldout_inverse = {
        (int(tail), inverse_id_by_base_id[int(relation)], int(head))
        for head, relation, tail in np.concatenate([val_pos, test_pos], axis=0)
        if int(relation) in inverse_id_by_base_id
    }
    for variant in ("all_inverse_edges", "universal_edges"):
        overlap = _triple_set(shared[f"train_pos_{variant}"]) & heldout_inverse
        if overlap:
            raise ValueError(
                f"train_pos_{variant} contains {len(overlap)} inverses of held-out triples"
            )


def prepare_bundles(
    splits_path: Path,
    variants: List[str],
    fixed_negatives: int,
    candidate_seed: int,
):
    """Load variants through the same WordNetLPDataset used by single-variant runs."""
    shared = load_wordnet_splits(splits_path)
    validate_leakage_free_archive(shared)

    canonical_variants = [canonicalize_variant(v) for v in variants]
    if len(canonical_variants) != len(set(canonical_variants)):
        raise ValueError(
            "The requested WordNet variants collapse to duplicates after alias "
            f"canonicalization: {variants} -> {canonical_variants}"
        )

    datasets = {
        variant: WordNetLPDataset(variant, splits_path)
        for variant in canonical_variants
    }
    reference = datasets[canonical_variants[0]]
    bundles: Dict[str, Dict[str, Any]] = {}
    for variant, dataset in datasets.items():
        if dataset.num_entities != reference.num_entities:
            raise ValueError(f"num_entities differs for {variant}")
        if dataset.num_relations != reference.num_relations:
            raise ValueError(f"num_relations differs for {variant}")
        if not np.array_equal(dataset.entity_vocab, reference.entity_vocab):
            raise ValueError(f"entity_vocab differs for {variant}")
        if not np.array_equal(dataset.relation_vocab, reference.relation_vocab):
            raise ValueError(f"relation_vocab differs for {variant}")
        if not np.array_equal(dataset.val_pos, reference.val_pos):
            raise ValueError(f"validation positives differ for {variant}")
        if not np.array_equal(dataset.test_pos, reference.test_pos):
            raise ValueError(f"test positives differ for {variant}")

        edge_index, edge_type = dataset.get_train_graph()
        train_pos = np.asarray(dataset.train_pos, dtype=np.int64)
        bundles[variant] = {
            "variant": variant,
            "edge_index": edge_index,
            "edge_type": edge_type,
            "train_pos": train_pos,
            "num_entities": dataset.num_entities,
            "num_relations": dataset.num_relations,
            "edge_count": int(edge_index.shape[1]),
            "format_version": dataset.format_version,
        }

    # Use every available graph plus validation/test positives when excluding
    # false negatives. This keeps invariance candidates identical regardless of
    # whether an optional subset of variants is selected for an ablation.
    all_known = [
        np.asarray(shared[f"train_pos_{variant}"], dtype=np.int64)
        for variant in DEFAULT_VARIANTS
    ] + [
        np.asarray(shared["val_pos"], dtype=np.int64),
        np.asarray(shared["test_pos"], dtype=np.int64),
    ]
    known_union = np.unique(np.concatenate(all_known, axis=0), axis=0)
    known_keys = set(
        pack_keys(known_union, reference.num_entities, reference.num_relations).tolist()
    )
    val_candidates, val_labels, val_query_ids = fixed_tail_candidates(
        reference.val_pos,
        known_keys,
        reference.num_entities,
        reference.num_relations,
        fixed_negatives,
        candidate_seed + 101,
    )
    test_candidates, test_labels, test_query_ids = fixed_tail_candidates(
        reference.test_pos,
        known_keys,
        reference.num_entities,
        reference.num_relations,
        fixed_negatives,
        candidate_seed + 103,
    )
    shared = dict(shared)
    shared.update(
        {
            "val_pos": np.asarray(reference.val_pos, dtype=np.int64),
            "test_pos": np.asarray(reference.test_pos, dtype=np.int64),
            "known_union": known_union,
            "val_candidate_triples": val_candidates,
            "val_candidate_labels": val_labels,
            "val_candidate_query_ids": val_query_ids,
            "test_candidate_triples": test_candidates,
            "test_candidate_labels": test_labels,
            "test_candidate_query_ids": test_query_ids,
        }
    )
    return bundles, shared, canonical_variants

def sample_neg_train(
    pos_array: np.ndarray,
    num_entities: int,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Exact legacy 50/50 head/tail corruption."""
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


def build_filter_dicts(*triple_arrays):
    tail_raw = defaultdict(set)
    head_raw = defaultdict(set)
    for array in triple_arrays:
        for head, relation, tail in np.asarray(array).tolist():
            tail_raw[(head, relation)].add(tail)
            head_raw[(relation, tail)].add(head)
    return (
        {key: frozenset(value) for key, value in tail_raw.items()},
        {key: frozenset(value) for key, value in head_raw.items()},
    )


@torch.no_grad()
def legacy_evaluate(
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
    """Evaluation copied from the legacy runner, including binary sampling."""
    model.eval()
    binary_rng = np.random.default_rng(42)
    raw_ranks_tail = []
    filtered_ranks_tail = []
    raw_ranks_head = []
    filtered_ranks_head = []
    all_logits = []
    all_labels = []

    pos_t = torch.from_numpy(pos_array).long().to(device)
    for start in range(0, len(pos_array), eval_batch_size):
        batch = pos_t[start : start + eval_batch_size]
        h_idx = batch[:, 0]
        r_idx = batch[:, 1]
        t_idx = batch[:, 2]
        h_emb = entity_embs[h_idx]
        r_emb = model.rel_emb(r_idx)
        t_emb = entity_embs[t_idx]
        scores_tail = (h_emb * r_emb) @ entity_embs.T
        scores_head = (r_emb * t_emb) @ entity_embs.T
        scores_tail_cpu = scores_tail.cpu()
        scores_head_cpu = scores_head.cpu()

        for row in range(len(batch)):
            head = h_idx[row].item()
            relation = r_idx[row].item()
            tail = t_idx[row].item()
            tail_scores = scores_tail_cpu[row]
            head_scores = scores_head_cpu[row]
            true_tail_score = tail_scores[tail].item()
            true_head_score = head_scores[head].item()
            raw_ranks_tail.append(int((tail_scores >= true_tail_score).sum().item()))
            raw_ranks_head.append(int((head_scores >= true_head_score).sum().item()))

            filtered_tail = tail_scores.clone()
            for other_tail in tail_filters.get((head, relation), frozenset()):
                if other_tail != tail:
                    filtered_tail[other_tail] = float("-inf")
            filtered_ranks_tail.append(
                int((filtered_tail >= true_tail_score).sum().item())
            )

            filtered_head = head_scores.clone()
            for other_head in head_filters.get((relation, tail), frozenset()):
                if other_head != head:
                    filtered_head[other_head] = float("-inf")
            filtered_ranks_head.append(
                int((filtered_head >= true_head_score).sum().item())
            )

            known_tails = tail_filters.get((head, relation), frozenset())
            negatives = []
            candidates = binary_rng.integers(0, num_entities, size=binary_k * 4)
            for candidate in candidates:
                if candidate not in known_tails and len(negatives) < binary_k:
                    negatives.append(int(candidate))
            while len(negatives) < binary_k:
                negatives.append(int(binary_rng.integers(0, num_entities)))
            negative_scores = tail_scores[negatives].numpy()
            all_logits.append(true_tail_score)
            all_logits.extend(negative_scores.tolist())
            all_labels.append(1)
            all_labels.extend([0] * binary_k)

    raw_ranks = np.asarray(raw_ranks_tail + raw_ranks_head, dtype=np.float32)
    filtered_ranks = np.asarray(
        filtered_ranks_tail + filtered_ranks_head, dtype=np.float32
    )
    logits = np.asarray(all_logits, dtype=np.float32)
    labels = np.asarray(all_labels, dtype=np.int32)
    predictions = (logits > 0).astype(np.int32)
    return {
        "raw_MRR": float(np.mean(1.0 / raw_ranks)),
        "filtered_MRR": float(np.mean(1.0 / filtered_ranks)),
        "Hits@1": float(np.mean(filtered_ranks <= 1)),
        "Hits@3": float(np.mean(filtered_ranks <= 3)),
        "Hits@10": float(np.mean(filtered_ranks <= 10)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "recall": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
    }


def score_numpy_triples(model, embeddings, triples, device, batch_size):
    scores = []
    for start in range(0, len(triples), batch_size):
        batch = torch.from_numpy(triples[start : start + batch_size]).long().to(device)
        scores.append(
            model.score(
                embeddings, batch[:, 0], batch[:, 1], batch[:, 2]
            ).detach().cpu()
        )
    return torch.cat(scores).numpy() if scores else np.empty(0, dtype=np.float32)


def shared_candidate_frame(
    model,
    embeddings,
    triples,
    labels,
    query_ids,
    device,
    score_batch_size,
):
    logits = score_numpy_triples(
        model, embeddings, triples, device, score_batch_size
    )
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    frame = pd.DataFrame(
        {
            "query_id": query_ids.astype(np.int64),
            "head": triples[:, 0].astype(np.int64),
            "relation": triples[:, 1].astype(np.int64),
            "tail": triples[:, 2].astype(np.int64),
            "label": labels.astype(np.int64),
            "logit": logits,
            "score": probabilities,
        }
    )
    bce = float(
        F.binary_cross_entropy_with_logits(
            torch.from_numpy(logits),
            torch.from_numpy(labels.astype(np.float32)),
        )
    )
    metrics = link_binary_metrics(labels.astype(np.int64), probabilities, 0.5)
    return bce, metrics, frame


def run_seed(args, variants: List[str], seed: int, output_root: Path) -> Dict[str, Any]:
    set_determinism(seed)
    device = resolve_device(args.device)
    splits_path = resolve_splits_path(Path(args.data_root), args.splits_npz)
    bundles, shared, variants = prepare_bundles(
        splits_path, variants, args.fixed_negatives, args.candidate_seed
    )
    reference = bundles[variants[0]]

    model = WordNetRGCNLinkPredictor(
        num_entities=reference["num_entities"],
        num_relations=reference["num_relations"],
        hidden_dim=args.hidden_dim,
        num_bases=args.num_bases,
        edge_dropout_other=args.edge_dropout_other,
        root_dropout_loop=args.root_dropout_loop,
    ).to(device)
    optimizer = torch.optim.Adam(
        [
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if "rel_emb" not in name
                ],
                "weight_decay": 0.0,
            },
            {
                "params": model.rel_emb.parameters(),
                "weight_decay": args.weight_decay,
            },
        ],
        lr=args.lr,
    )

    batch_sizes = {
        variant: len(bundle["train_pos"])
        if args.batch_size <= 0
        else args.batch_size
        for variant, bundle in bundles.items()
    }
    batches_per_variant = {
        variant: int(np.ceil(len(bundle["train_pos"]) / batch_sizes[variant]))
        for variant, bundle in bundles.items()
    }
    updates_per_super_epoch = int(sum(batches_per_variant.values()))

    patience_checks = (
        args.patience_evals
        if args.patience_evals is not None
        else int(np.ceil(args.patience / args.eval_interval))
    )
    seed_dir = output_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = seed_dir / "shared_checkpoint.pt"
    latest_state_path = seed_dir / "latest_training_state.pt"
    # The single-variant runner treats any strict MRR increase as an improvement.
    early_stopper = EarlyStopper(
        mode="max", patience=patience_checks, min_delta=0.0
    )
    rng = np.random.default_rng(seed)

    filters_by_variant = {
        variant: build_filter_dicts(
            bundle["train_pos"], shared["val_pos"], shared["test_pos"]
        )
        for variant, bundle in bundles.items()
    }
    run_config = {
        "dataset": "WORDNET",
        "model": "legacy_WordNetRGCNLinkPredictor",
        "seed": seed,
        "variants": variants,
        "hidden_dim": args.hidden_dim,
        "num_bases": args.num_bases,
        "edge_dropout_other": args.edge_dropout_other,
        "root_dropout_loop": args.root_dropout_loop,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "batch_size": args.batch_size,
        "neg_per_pos": args.neg_per_pos,
        "patience": args.patience,
        "patience_checks": patience_checks,
        "eval_interval": args.eval_interval,
        "eval_batch_size": args.eval_batch_size,
        "binary_k": args.binary_k,
        "score_batch_size": args.score_batch_size,
        "splits_npz": str(splits_path.resolve()),
        "fixed_negatives": args.fixed_negatives,
        "candidate_seed": args.candidate_seed,
        "npz_format_version": str(
            np.asarray(shared.get("format_version", "legacy")).item()
        ),
        "npz_variant_names": [
            str(v) for v in np.asarray(
                shared.get("variant_names", DEFAULT_VARIANTS)
            ).tolist()
        ],
    }

    history: List[Dict[str, Any]] = []
    super_epochs_ran = variant_epochs = optimizer_steps = 0
    train_graph_forwards = validation_graph_forwards = validation_checks = 0
    prior_training_seconds = 0.0
    prior_peak_rss = 0
    prior_training_gpu: Dict[str, int] = {}

    resume_state = load_latest_training_state(
        latest_state_path,
        resume=args.resume,
        run_config=run_config,
        modules={"model": model},
        optimizer=optimizer,
        early_stopper=early_stopper,
        rng=rng,
        device=device,
    )
    if resume_state is not None:
        history = list(resume_state["history"])
        counters = resume_state["counters"]
        optimizer_steps = int(counters["optimizer_steps"])
        variant_epochs = int(counters["variant_epochs"])
        train_graph_forwards = int(counters["train_graph_forwards"])
        validation_graph_forwards = int(counters["validation_graph_forwards"])
        validation_checks = int(counters["validation_checks"])
        super_epochs_ran = int(resume_state["completed_super_epoch"])
        prior_training_seconds = float(resume_state["training_seconds_elapsed"])
        prior_peak_rss = int(resume_state.get("process_peak_rss_bytes", 0))
        prior_training_gpu = dict(resume_state.get("training_gpu", {}))

    device_graphs = {
        variant: (
            bundle["edge_index"].to(device),
            bundle["edge_type"].to(device),
        )
        for variant, bundle in bundles.items()
    }

    reset_cuda_peak(device)
    start_time = time.perf_counter()
    for super_epoch in range(super_epochs_ran, args.super_epochs):
        if early_stopper.should_stop:
            print("[resume] Early stopping was already reached; skipping training.", flush=True)
            break
        cycle_start = time.perf_counter()
        order = [variants[i] for i in rng.permutation(len(variants))]
        train_losses: Dict[str, float] = {}

        for variant in order:
            bundle = bundles[variant]
            edge_index, edge_type = device_graphs[variant]
            shuffled = bundle["train_pos"][rng.permutation(len(bundle["train_pos"]))]
            losses = []
            for start in range(0, len(shuffled), batch_sizes[variant]):
                positive_np = shuffled[start : start + batch_sizes[variant]]
                negative_np = sample_neg_train(
                    positive_np,
                    bundle["num_entities"],
                    args.neg_per_pos,
                    rng,
                )
                positive = torch.from_numpy(positive_np).long().to(device)
                negative = torch.from_numpy(negative_np).long().to(device)

                model.train()
                optimizer.zero_grad(set_to_none=True)
                # The model performs edge dropout, c_i recomputation, and root
                # dropout internally exactly as the legacy code did.
                entity_embs = model.encode(edge_index, edge_type, training=True)
                train_graph_forwards += 1
                positive_scores = model.score(
                    entity_embs, positive[:, 0], positive[:, 1], positive[:, 2]
                )
                negative_scores = model.score(
                    entity_embs, negative[:, 0], negative[:, 1], negative[:, 2]
                )
                scores = torch.cat([positive_scores, negative_scores])
                labels = torch.cat(
                    [
                        torch.ones(len(positive_scores), device=device),
                        torch.zeros(len(negative_scores), device=device),
                    ]
                )
                loss = F.binary_cross_entropy_with_logits(scores, labels)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer_steps += 1
                losses.append(float(loss.detach().cpu()))
            variant_epochs += 1
            train_losses[variant] = float(np.mean(losses))

        super_epochs_ran = super_epoch + 1
        # Match the single-variant runner exactly: validate only at explicit
        # eval_interval boundaries. A non-multiple final epoch is not evaluated.
        do_eval = super_epochs_ran % args.eval_interval == 0
        val_metrics: Dict[str, Dict[str, float]] = {
            variant: {} for variant in variants
        }
        mean_val_filtered_mrr = float("nan")
        if do_eval:
            validation_checks += 1
            for variant in variants:
                edge_index, edge_type = device_graphs[variant]
                model.eval()
                with torch.no_grad():
                    embeddings = model.encode(edge_index, edge_type, training=False)
                validation_graph_forwards += 1
                tail_filters, head_filters = filters_by_variant[variant]
                val_metrics[variant] = legacy_evaluate(
                    model,
                    embeddings,
                    shared["val_pos"],
                    device,
                    tail_filters,
                    head_filters,
                    bundles[variant]["num_entities"],
                    eval_batch_size=args.eval_batch_size,
                    binary_k=args.binary_k,
                )
            mean_val_filtered_mrr = float(
                np.mean([val_metrics[v]["filtered_MRR"] for v in variants])
            )
            improved = early_stopper.update(mean_val_filtered_mrr)
            if improved:
                atomic_torch_save(
                    {
                        "model": model.state_dict(),
                        "metadata": {
                            "seed": seed,
                            "variants": variants,
                            "best_super_epoch": super_epochs_ran,
                            "optimizer_steps": optimizer_steps,
                            "selection_metric": "mean_legacy_filtered_MRR",
                            "splits_npz": str(splits_path.resolve()),
                            "split_protocol": (
                                "official_leakage_free_four_variant_wordnet_splits"
                            ),
                        },
                    },
                    checkpoint_path,
                )

        row = {
            "super_epoch": super_epochs_ran,
            "variant_order": ",".join(order),
            "variant_epochs_cumulative": variant_epochs,
            "optimizer_steps_cumulative": optimizer_steps,
            "updates_per_super_epoch": updates_per_super_epoch,
            "mean_train_loss": float(np.mean(list(train_losses.values()))),
            "validation_check": int(do_eval),
            "validation_checks_cumulative": validation_checks,
            "mean_val_filtered_MRR": mean_val_filtered_mrr,
            "best_mean_val_filtered_MRR": (
                early_stopper.best if np.isfinite(early_stopper.best) else None
            ),
            "cycle_seconds": time.perf_counter() - cycle_start,
        }
        for variant in variants:
            row[f"batches_{variant}"] = batches_per_variant[variant]
            row[f"train_loss_{variant}"] = train_losses[variant]
            if do_eval:
                for metric_name, metric_value in val_metrics[variant].items():
                    row[f"val_{metric_name}_{variant}"] = metric_value
        history.append(row)
        atomic_write_csv(pd.DataFrame(history), seed_dir / "training_history.csv")

        segment_seconds = time.perf_counter() - start_time
        current_training_gpu = merge_cuda_memory_stats(
            prior_training_gpu, cuda_memory_stats(device)
        )
        current_peak_rss = max(prior_peak_rss, process_peak_rss_bytes())
        save_latest_training_state(
            latest_state_path,
            dataset="WORDNET",
            run_config=run_config,
            modules={"model": model},
            optimizer=optimizer,
            early_stopper=early_stopper,
            rng=rng,
            completed_super_epoch=super_epochs_ran,
            counters={
                "optimizer_steps": optimizer_steps,
                "variant_epochs": variant_epochs,
                "train_graph_forwards": train_graph_forwards,
                "validation_graph_forwards": validation_graph_forwards,
                "validation_checks": validation_checks,
            },
            history=history,
            training_seconds_elapsed=prior_training_seconds + segment_seconds,
            process_peak_rss_bytes_value=current_peak_rss,
            training_gpu=current_training_gpu,
        )
        print(
            f"seed={seed} super_epoch={super_epochs_ran:04d} "
            f"optimizer_steps={optimizer_steps} train_loss={row['mean_train_loss']:.6f} "
            f"val_filtered_MRR={mean_val_filtered_mrr:.6f} order={order}",
            flush=True,
        )
        if do_eval and early_stopper.should_stop:
            break

    training_seconds = prior_training_seconds + (time.perf_counter() - start_time)
    training_memory = merge_cuda_memory_stats(
        prior_training_gpu, cuda_memory_stats(device)
    )
    cumulative_peak_rss = max(prior_peak_rss, process_peak_rss_bytes())
    if np.isfinite(early_stopper.best):
        if not checkpoint_path.exists():
            raise RuntimeError(
                "Validation produced a best score but the best checkpoint is missing."
            )
        model.load_state_dict(
            torch_load_full(checkpoint_path, map_location=device)["model"]
        )
        checkpoint_kind = "best_validation_filtered_MRR"
    else:
        # Match the single-variant runner for short jobs that never reach an
        # evaluation boundary: evaluate the final model weights rather than a
        # stale or nonexistent validation checkpoint.
        atomic_torch_save(
            {
                "model": model.state_dict(),
                "metadata": {
                    "seed": seed,
                    "variants": variants,
                    "best_super_epoch": None,
                    "optimizer_steps": optimizer_steps,
                    "selection_metric": None,
                    "checkpoint_kind": "final_weights_no_validation",
                    "splits_npz": str(splits_path.resolve()),
                },
            },
            checkpoint_path,
        )
        checkpoint_kind = "final_weights_no_validation"

    reset_cuda_peak(device)
    legacy_metrics_by_variant: Dict[str, Dict[str, float]] = {}
    shared_metrics_by_variant: Dict[str, Dict[str, float]] = {}
    shared_frames: Dict[str, pd.DataFrame] = {}
    test_graph_forwards = 0
    for variant in variants:
        edge_index, edge_type = device_graphs[variant]
        model.eval()
        with torch.no_grad():
            embeddings = model.encode(edge_index, edge_type, training=False)
        test_graph_forwards += 1
        tail_filters, head_filters = filters_by_variant[variant]
        legacy_metrics = legacy_evaluate(
            model,
            embeddings,
            shared["test_pos"],
            device,
            tail_filters,
            head_filters,
            bundles[variant]["num_entities"],
            eval_batch_size=args.eval_batch_size,
            binary_k=args.binary_k,
        )
        legacy_metrics["edge_count"] = float(bundles[variant]["edge_count"])
        legacy_metrics["training_triples"] = float(len(bundles[variant]["train_pos"]))
        legacy_metrics_by_variant[variant] = legacy_metrics

        candidate_bce, candidate_metrics, frame = shared_candidate_frame(
            model,
            embeddings,
            shared["test_candidate_triples"],
            shared["test_candidate_labels"],
            shared["test_candidate_query_ids"],
            device,
            args.score_batch_size,
        )
        shared_metrics_by_variant[variant] = {
            "candidate_BCE": candidate_bce,
            **candidate_metrics,
        }
        shared_frames[variant] = frame
        frame.to_csv(seed_dir / f"shared_candidate_test_scores_{variant}.csv", index=False)

    inference_memory = cuda_memory_stats(device)
    invariance_rows = triple_link_invariance_rows(shared_frames, threshold=0.5)
    pd.DataFrame(invariance_rows).to_csv(seed_dir / "pairwise_invariance.csv", index=False)
    pd.DataFrame(
        [
            {"variant": variant, **metrics}
            for variant, metrics in legacy_metrics_by_variant.items()
        ]
    ).to_csv(seed_dir / "legacy_test_metrics_by_variant.csv", index=False)
    pd.DataFrame(
        [
            {"variant": variant, **metrics}
            for variant, metrics in shared_metrics_by_variant.items()
        ]
    ).to_csv(seed_dir / "shared_candidate_metrics_by_variant.csv", index=False)

    expected_steps = super_epochs_ran * updates_per_super_epoch
    expected_variant_epochs = super_epochs_ran * len(variants)
    if optimizer_steps != expected_steps or variant_epochs != expected_variant_epochs:
        raise AssertionError(
            f"Epoch accounting mismatch: steps {optimizer_steps}/{expected_steps}, "
            f"variant epochs {variant_epochs}/{expected_variant_epochs}"
        )

    summary = {
        "dataset": "WORDNET",
        "model": "legacy_WordNetRGCNLinkPredictor",
        "seed": seed,
        "variants": variants,
        "epoch_accounting": {
            "definition": "one super-epoch visits every variant and every supervised triple batch in that variant",
            "super_epochs_ran": super_epochs_ran,
            "variant_epochs_ran": variant_epochs,
            "batches_per_variant": batches_per_variant,
            "updates_per_super_epoch": updates_per_super_epoch,
            "optimizer_steps": optimizer_steps,
            "expected_optimizer_steps": expected_steps,
            "validation_checks": validation_checks,
            "train_graph_forwards": train_graph_forwards,
            "validation_graph_forwards": validation_graph_forwards,
            "test_graph_forwards": test_graph_forwards,
        },
        "training_seconds": training_seconds,
        "selection_metric": "mean_legacy_filtered_MRR",
        "splits_npz": str(splits_path.resolve()),
        "split_protocol": "official_leakage_free_four_variant_wordnet_splits",
        "best_mean_val_filtered_MRR": (
            early_stopper.best if np.isfinite(early_stopper.best) else None
        ),
        "checkpoint_kind": checkpoint_kind,
        "evaluation_protocols": {
            "legacy": "legacy metric implementation with variant-specific filters and deterministic binary negatives; compare with per-variant runs made from the same leakage-free NPZ",
            "shared_candidate": "identical fixed candidate triples across variants; used for invariance only",
        },
        "mean_legacy_test_metrics": mean_dict(
            list(legacy_metrics_by_variant.values())
        ),
        "per_variant_legacy_test_metrics": legacy_metrics_by_variant,
        "mean_shared_candidate_metrics": mean_dict(
            list(shared_metrics_by_variant.values())
        ),
        "per_variant_shared_candidate_metrics": shared_metrics_by_variant,
        "pairwise_invariance": invariance_rows,
        "memory": {
            **model_memory_bytes(model),
            "checkpoint_bytes": checkpoint_size_bytes(checkpoint_path),
            "process_peak_rss_bytes": cumulative_peak_rss,
            "training_gpu": training_memory,
            "inference_gpu": inference_memory,
        },
    }
    write_json(seed_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WordNet joint augmentation using the legacy RGCN model"
    )
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--seeds", default="1566911444,20241017,20251017")
    parser.add_argument("--data-root", default="data/wordnet_3hops_augmented_full")
    parser.add_argument(
        "--splits-npz", "--splits-path", "--splits_path",
        dest="splits_npz", default=None,
        help="Leakage-free NPZ; defaults to <data-root>/wordnet_splits.npz",
    )
    parser.add_argument("--output-dir", default="results/rgcn_augmentation/WORDNET")
    parser.add_argument(
        "--super-epochs", "--epochs", dest="super_epochs", type=int, default=3000,
        help="Total joint super-epochs; --epochs is a compatibility alias",
    )
    parser.add_argument(
        "--eval-interval", "--eval_interval", dest="eval_interval",
        type=int, default=1,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Patience measured in super-epochs (30 with eval_interval=1 = 30 checks)",
    )
    parser.add_argument(
        "--patience-evals",
        type=int,
        default=None,
        help="Optional direct override for number of failed evaluation checks",
    )
    parser.add_argument(
        "--batch-size", "--batch_size", dest="batch_size", type=int, default=0
    )
    parser.add_argument(
        "--neg-per-pos", "--neg_per_pos", dest="neg_per_pos", type=int, default=1
    )
    parser.add_argument(
        "--hidden-dim", "--hidden_dim", dest="hidden_dim", type=int, default=200
    )
    parser.add_argument(
        "--num-bases", "--num_bases", dest="num_bases", type=int, default=30
    )
    parser.add_argument(
        "--edge-dropout-other", "--edge_dropout_other",
        dest="edge_dropout_other", type=float, default=0.4,
    )
    parser.add_argument(
        "--root-dropout-loop", "--root_dropout_loop",
        dest="root_dropout_loop", type=float, default=0.2,
    )
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--weight-decay", "--weight_decay",
        dest="weight_decay", type=float, default=0.01,
    )
    parser.add_argument(
        "--grad-clip", type=float, default=0.0, help="0 disables clipping (legacy default)"
    )
    parser.add_argument(
        "--eval-batch-size", "--eval_batch_size",
        dest="eval_batch_size", type=int, default=512,
    )
    parser.add_argument("--binary-k", type=int, default=50)
    parser.add_argument("--score-batch-size", type=int, default=65536)
    parser.add_argument("--fixed-negatives", type=int, default=50)
    parser.add_argument("--candidate-seed", type=int, default=1566911444)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    requested_variants = parse_csv(args.variants)
    try:
        variants = [canonicalize_variant(v) for v in requested_variants]
    except ValueError as exc:
        parser.error(str(exc))
    if len(variants) != len(set(variants)):
        parser.error(
            "Duplicate WordNet variants after alias canonicalization: "
            f"{requested_variants} -> {variants}"
        )
    seeds = [int(value) for value in parse_csv(args.seeds)]
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = [run_seed(args, variants, seed, output_root) for seed in seeds]
    rows = []
    for summary in summaries:
        rows.append(
            {
                "seed": summary["seed"],
                "super_epochs_ran": summary["epoch_accounting"]["super_epochs_ran"],
                "variant_epochs_ran": summary["epoch_accounting"]["variant_epochs_ran"],
                "updates_per_super_epoch": summary["epoch_accounting"]["updates_per_super_epoch"],
                "optimizer_steps": summary["epoch_accounting"]["optimizer_steps"],
                "training_seconds": summary["training_seconds"],
                "best_mean_val_filtered_MRR": summary["best_mean_val_filtered_MRR"],
                **{
                    f"mean_legacy_test_{key}": value
                    for key, value in summary["mean_legacy_test_metrics"].items()
                },
                **{
                    f"mean_shared_candidate_{key}": value
                    for key, value in summary["mean_shared_candidate_metrics"].items()
                },
            }
        )
    pd.DataFrame(rows).to_csv(output_root / "seed_summary.csv", index=False)
    write_json(output_root / "all_seed_summaries.json", {"runs": summaries})
    print(f"[OK] Results written under {output_root}")


if __name__ == "__main__":
    main()
