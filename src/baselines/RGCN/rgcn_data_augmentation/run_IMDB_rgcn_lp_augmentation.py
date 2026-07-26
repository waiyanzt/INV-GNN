#!/usr/bin/env python3
"""Joint graph-variant augmentation for IMDb RGCN link prediction.

This runner follows the repository's ``run_IMDB_rgcn_lp.py`` modeling and
metrics while replacing independent per-variant training with one shared
encoder, optimizer, and best checkpoint.

A *super-epoch* visits every selected graph variant exactly once.  Within each
variant, all positive training rows are processed in minibatches and each batch
recomputes full-graph RGCN embeddings, matching the legacy runner.  Early
stopping is checked only after the complete balanced super-epoch using mean
validation BCE across variants.

Documented tasks:
  md: movie-director link prediction, variants v1 and v3
  ml: movie-link link prediction, variants v1-v4

The upstream runner also parses ``mg`` (movie-genre); it is supported here when
matching preprocessed directories exist.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import kendalltau

from rgcn_aug_common import (
    IMDB_LP_RELATIONS,
    EarlyStopper,
    RGCNEncoder,
    assert_same_indexers,
    assert_same_tensor,
    atomic_torch_save,
    atomic_write_csv,
    checkpoint_size_bytes,
    cuda_memory_stats,
    link_binary_metrics,
    load_latest_training_state,
    mean_dict,
    merge_cuda_memory_stats,
    model_memory_bytes,
    pairwise_loss,
    process_peak_rss_bytes,
    relation_to_id,
    reset_cuda_peak,
    resolve_device,
    save_latest_training_state,
    set_determinism,
    to_homogeneous_with_global_relations,
    torch_load_full,
    write_json,
)


VALID_VARIANTS: Mapping[str, Tuple[str, ...]] = {
    "md": ("v1", "v3"),
    "mg": ("v1", "v2", "v3", "v4"),
    "ml": ("v1", "v2", "v3", "v4"),
}

TAIL_NODE_TYPE = {"md": "director", "mg": "genre", "ml": "link"}
TAIL_SCORE_COLUMN = {
    "md": "director_local",
    "mg": "genre_id",
    "ml": "link_local",
}

KNOWN_GRAPH_KEYS = {
    "movie-actor",
    "movie-director",
    "movie-link",
    "movie-genre",
    "link-director",
    "link-actor",
}


def parse_task(value: str) -> str:
    task = str(value).strip().lower()
    if task not in VALID_VARIANTS:
        raise argparse.ArgumentTypeError("task must be md, mg, or ml")
    return task


def parse_variants(spec: str, task: str) -> List[str]:
    if not spec.strip():
        return list(VALID_VARIANTS[task])
    variants = [item.strip().lower() for item in spec.split(",") if item.strip()]
    if not variants or len(set(variants)) != len(variants):
        raise SystemExit("--variants must contain distinct variant names")
    invalid = set(variants) - set(VALID_VARIANTS[task])
    if invalid:
        raise SystemExit(
            f"Invalid variants for task={task}: {sorted(invalid)}; "
            f"expected a subset of {list(VALID_VARIANTS[task])}"
        )
    return variants


def base_dir(data_root: Path, task: str, variant: str) -> Path:
    return data_root / f"IMDB_rgcn_lp_{task}_{variant}"


def load_preprocessed(path: Path):
    graph_path = path / "graph_data.pt"
    meta_path = path / "meta.pt"
    if not graph_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"Expected graph_data.pt and meta.pt under {path}")
    graph_data = torch.load(graph_path, map_location="cpu")
    meta = torch.load(meta_path, map_location="cpu")
    return graph_data, meta


def build_heterograph(
    graph_data: Mapping[str, Any], num_nodes: Mapping[str, int]
) -> dgl.DGLHeteroGraph:
    """Reproduce the upstream IMDb-LP heterograph construction."""
    unknown = set(graph_data) - KNOWN_GRAPH_KEYS
    if unknown:
        raise ValueError(f"Unknown IMDb-LP graph_data keys: {sorted(unknown)}")

    data: Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]] = {}

    def add_bidirectional(
        source_type: str,
        forward_name: str,
        destination_type: str,
        reverse_name: str,
        key: str,
    ) -> None:
        if key not in graph_data:
            return
        source, destination = graph_data[key]
        data[(source_type, forward_name, destination_type)] = (source, destination)
        data[(destination_type, reverse_name, source_type)] = (destination, source)

    add_bidirectional("movie", "movie-actor", "actor", "actor-movie", "movie-actor")
    add_bidirectional(
        "movie", "movie-director", "director", "director-movie", "movie-director"
    )
    add_bidirectional("movie", "movie-link", "link", "link-movie", "movie-link")
    add_bidirectional("movie", "movie-genre", "genre", "genre-movie", "movie-genre")
    add_bidirectional(
        "link", "link-director", "director", "director-link", "link-director"
    )
    add_bidirectional("link", "link-actor", "actor", "actor-link", "link-actor")

    if not data:
        raise ValueError("IMDb-LP graph contains no recognized edge types")
    return dgl.heterograph(
        data, num_nodes_dict={key: int(value) for key, value in num_nodes.items()}
    )


def prepare_bundles(task: str, variants: Sequence[str], data_root: Path):
    relation_ids = relation_to_id(IMDB_LP_RELATIONS)
    bundles: Dict[str, Dict[str, Any]] = {}
    num_nodes_by_variant: Dict[str, Dict[str, int]] = {}
    indexers_by_variant: Dict[str, Mapping[str, torch.Tensor]] = {}
    splits_by_name: Dict[str, Dict[str, torch.Tensor]] = {}

    for variant in variants:
        graph_data, meta = load_preprocessed(base_dir(data_root, task, variant))
        num_nodes = {key: int(value) for key, value in meta["num_nodes"].items()}
        graph = build_heterograph(graph_data, num_nodes)
        homogeneous, edge_types, indexers = to_homogeneous_with_global_relations(
            graph, relation_ids
        )
        tail_type = TAIL_NODE_TYPE[task]
        for required_type in ("movie", tail_type):
            if required_type not in indexers:
                raise ValueError(
                    f"{task}/{variant} lacks node type {required_type!r}; "
                    "the requested prediction task cannot be scored"
                )

        raw_splits = meta.get("splits")
        if not isinstance(raw_splits, Mapping):
            raise ValueError(f"{task}/{variant} meta.pt lacks a splits mapping")
        required_splits = (
            "train_pos",
            "train_neg",
            "val_pos",
            "val_neg",
            "test_pos",
            "test_neg",
        )
        missing_splits = set(required_splits) - set(raw_splits)
        if missing_splits:
            raise ValueError(
                f"{task}/{variant} is missing splits: {sorted(missing_splits)}"
            )
        splits = {name: raw_splits[name].long().cpu() for name in required_splits}
        for name in ("train_pos", "val_pos", "test_pos"):
            if splits[name].ndim != 2 or splits[name].shape[1] != 2:
                raise ValueError(f"{task}/{variant} {name} must have shape (N, 2)")
        for name in ("train_neg", "val_neg", "test_neg"):
            if splits[name].ndim != 2:
                raise ValueError(f"{task}/{variant} {name} must have shape (N, K)")
        for prefix in ("train", "val", "test"):
            if splits[f"{prefix}_pos"].shape[0] != splits[f"{prefix}_neg"].shape[0]:
                raise ValueError(
                    f"{task}/{variant} {prefix} positives and negative rows differ"
                )

        bundles[variant] = {
            "variant": variant,
            "graph": homogeneous,
            "edge_types": edge_types,
            "indexers": indexers,
            "movie_indexer": indexers["movie"],
            "tail_indexer": indexers[tail_type],
            "edge_count": int(homogeneous.num_edges()),
            "graph_keys": sorted(graph_data),
            "splits": splits,
        }
        num_nodes_by_variant[variant] = num_nodes
        indexers_by_variant[variant] = indexers
        for name, value in splits.items():
            splits_by_name.setdefault(name, {})[variant] = value

    reference_variant = variants[0]
    reference_num_nodes = num_nodes_by_variant[reference_variant]
    for variant in variants[1:]:
        if num_nodes_by_variant[variant] != reference_num_nodes:
            raise ValueError(
                f"num_nodes differs between {reference_variant} and {variant}: "
                f"{reference_num_nodes} vs {num_nodes_by_variant[variant]}"
            )
        if bundles[variant]["graph"].num_nodes() != bundles[reference_variant]["graph"].num_nodes():
            raise ValueError("Homogeneous node counts differ across IMDb-LP variants")

    assert_same_indexers(indexers_by_variant)
    for split_name, values in splits_by_name.items():
        assert_same_tensor(f"splits.{split_name}", values)
    return bundles


def device_bundle(cpu_bundle: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        **cpu_bundle,
        "graph": cpu_bundle["graph"].to(device),
        "edge_types": cpu_bundle["edge_types"].to(device),
        "movie_indexer": cpu_bundle["movie_indexer"].to(device),
        "tail_indexer": cpu_bundle["tail_indexer"].to(device),
    }


def node_type_embeddings(encoder: RGCNEncoder, bundle: Mapping[str, Any]):
    all_embeddings = encoder(bundle["graph"], bundle["edge_types"])
    return (
        all_embeddings[bundle["movie_indexer"]],
        all_embeddings[bundle["tail_indexer"]],
    )


def score_edges(
    movie_embeddings: torch.Tensor,
    tail_embeddings: torch.Tensor,
    edges: np.ndarray,
) -> torch.Tensor:
    edge_tensor = torch.as_tensor(edges, dtype=torch.long, device=movie_embeddings.device)
    return (
        movie_embeddings[edge_tensor[:, 0]] * tail_embeddings[edge_tensor[:, 1]]
    ).sum(dim=-1)


def flatten_negative_matrix(positive_edges: np.ndarray, negative_tails: np.ndarray) -> np.ndarray:
    positive_edges = np.asarray(positive_edges, dtype=np.int64)
    negative_tails = np.asarray(negative_tails, dtype=np.int64)
    if len(positive_edges) != len(negative_tails):
        raise ValueError("Positive rows and negative-tail rows must align")
    k = negative_tails.shape[1]
    return np.column_stack(
        [np.repeat(positive_edges[:, 0], k), negative_tails.reshape(-1)]
    ).astype(np.int64, copy=False)


def evaluate_validation(
    encoder: RGCNEncoder,
    cpu_bundle: Mapping[str, Any],
    shared_splits: Mapping[str, np.ndarray],
    device: torch.device,
) -> float:
    bundle = device_bundle(cpu_bundle, device)
    encoder.eval()
    bce = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        movie_embeddings, tail_embeddings = node_type_embeddings(encoder, bundle)
        positive_logits = score_edges(
            movie_embeddings, tail_embeddings, shared_splits["val_pos"]
        )
        negative_edges = flatten_negative_matrix(
            shared_splits["val_pos"], shared_splits["val_neg"]
        )
        negative_logits = score_edges(movie_embeddings, tail_embeddings, negative_edges)
        labels = torch.cat(
            [torch.ones_like(positive_logits), torch.zeros_like(negative_logits)]
        )
        logits = torch.cat([positive_logits, negative_logits])
        validation_loss = float(bce(logits, labels).cpu())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return validation_loss


def rowwise_ranking_metrics(
    positive_scores: np.ndarray, negative_scores: np.ndarray
) -> Tuple[Dict[str, float], np.ndarray]:
    positive_scores = np.asarray(positive_scores, dtype=float)
    negative_scores = np.asarray(negative_scores, dtype=float)
    if negative_scores.ndim != 2 or len(positive_scores) != len(negative_scores):
        raise ValueError("Ranking scores must have shapes (N,) and (N, K)")

    ranks = np.empty(len(positive_scores), dtype=np.int64)
    for row_index, true_score in enumerate(positive_scores):
        # Match the upstream stable Python sort: the positive item is inserted
        # first, so exact score ties keep the positive ahead of later negatives.
        items = [(float(true_score), 1)] + [
            (float(score), 0) for score in negative_scores[row_index]
        ]
        items.sort(key=lambda item: item[0], reverse=True)
        ranks[row_index] = next(
            rank + 1 for rank, (_score, is_positive) in enumerate(items) if is_positive
        )

    denominator = max(1, len(ranks))
    return (
        {
            "Hits@1": float(np.sum(ranks <= 1) / denominator),
            "Hits@3": float(np.sum(ranks <= 3) / denominator),
            "Hits@5": float(np.sum(ranks <= 5) / denominator),
            "MRR": float(np.sum(1.0 / ranks) / denominator),
            "ranking_queries": float(len(ranks)),
        },
        ranks,
    )


def evaluate_test(
    task: str,
    encoder: RGCNEncoder,
    cpu_bundle: Mapping[str, Any],
    shared_splits: Mapping[str, np.ndarray],
    device: torch.device,
    threshold: float,
):
    bundle = device_bundle(cpu_bundle, device)
    encoder.eval()
    with torch.no_grad():
        movie_embeddings, tail_embeddings = node_type_embeddings(encoder, bundle)
        positive_probabilities = torch.sigmoid(
            score_edges(movie_embeddings, tail_embeddings, shared_splits["test_pos"])
        ).cpu().numpy()
        negative_edges = flatten_negative_matrix(
            shared_splits["test_pos"], shared_splits["test_neg"]
        )
        negative_probabilities_flat = torch.sigmoid(
            score_edges(movie_embeddings, tail_embeddings, negative_edges)
        ).cpu().numpy()

    negative_probabilities = negative_probabilities_flat.reshape(
        len(shared_splits["test_pos"]), shared_splits["test_neg"].shape[1]
    )
    y_true = np.concatenate(
        [np.ones(len(positive_probabilities)), np.zeros(len(negative_probabilities_flat))]
    )
    y_score = np.concatenate([positive_probabilities, negative_probabilities_flat])
    metrics = link_binary_metrics(y_true, y_score, threshold)
    ranking_metrics, ranks = rowwise_ranking_metrics(
        positive_probabilities, negative_probabilities
    )
    metrics.update(ranking_metrics)

    movie_column = "movie_local"
    tail_column = TAIL_SCORE_COLUMN[task]
    rows: List[Dict[str, Any]] = []
    candidate_id = 0
    for row_index, positive in enumerate(shared_splits["test_pos"]):
        rows.append(
            {
                "candidate_id": candidate_id,
                "query_row": row_index,
                movie_column: int(positive[0]),
                tail_column: int(positive[1]),
                "score": float(positive_probabilities[row_index]),
                "label": 1,
                "rank_of_positive": int(ranks[row_index]),
            }
        )
        candidate_id += 1
        for negative_index, negative_tail in enumerate(shared_splits["test_neg"][row_index]):
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "query_row": row_index,
                    movie_column: int(positive[0]),
                    tail_column: int(negative_tail),
                    "score": float(negative_probabilities[row_index, negative_index]),
                    "label": 0,
                    "rank_of_positive": int(ranks[row_index]),
                }
            )
            candidate_id += 1

    return metrics, pd.DataFrame(rows), ranks


def _safe_tau(a: np.ndarray, b: np.ndarray) -> float:
    statistic = kendalltau(np.asarray(a), np.asarray(b), nan_policy="omit").statistic
    return float(statistic) if statistic is not None and np.isfinite(statistic) else float("nan")


def imdb_lp_invariance_rows(
    outputs: Mapping[str, pd.DataFrame],
    ranks_by_variant: Mapping[str, np.ndarray],
    threshold: float,
) -> List[Dict[str, Any]]:
    variants = list(outputs)
    rows: List[Dict[str, Any]] = []
    for index, variant_a in enumerate(variants):
        for variant_b in variants[index + 1 :]:
            frame_a = outputs[variant_a].sort_values("candidate_id")
            frame_b = outputs[variant_b].sort_values("candidate_id")
            identity_columns = ["candidate_id", "query_row", "label"]
            if len(frame_a) != len(frame_b):
                raise ValueError(
                    f"Candidate counts differ between {variant_a} and {variant_b}"
                )
            for column in identity_columns:
                if not np.array_equal(
                    frame_a[column].to_numpy(), frame_b[column].to_numpy()
                ):
                    raise ValueError(
                        f"Candidate identity column {column!r} differs between "
                        f"{variant_a} and {variant_b}"
                    )
            score_a = frame_a["score"].to_numpy()
            score_b = frame_b["score"].to_numpy()
            difference = score_a - score_b
            ranks_a = np.asarray(ranks_by_variant[variant_a])
            ranks_b = np.asarray(ranks_by_variant[variant_b])
            rows.append(
                {
                    "variant_a": variant_a,
                    "variant_b": variant_b,
                    "candidate_count": int(len(score_a)),
                    "query_count": int(len(ranks_a)),
                    "kendall_tau_scores": _safe_tau(score_a, score_b),
                    "prediction_agreement": float(
                        np.mean((score_a >= threshold) == (score_b >= threshold))
                    ),
                    "rank_agreement": float(np.mean(ranks_a == ranks_b)),
                    "kendall_tau_positive_ranks": _safe_tau(ranks_a, ranks_b),
                    "hits_at_1_agreement": float(
                        np.mean((ranks_a <= 1) == (ranks_b <= 1))
                    ),
                    "hits_at_3_agreement": float(
                        np.mean((ranks_a <= 3) == (ranks_b <= 3))
                    ),
                    "max_abs_score_diff": float(np.max(np.abs(difference))),
                    "mean_abs_score_diff": float(np.mean(np.abs(difference))),
                }
            )
    return rows


def run_seed(
    args: argparse.Namespace,
    task: str,
    variants: List[str],
    seed: int,
    output_root: Path,
) -> Dict[str, Any]:
    set_determinism(seed)
    device = resolve_device(args.device)
    bundles = prepare_bundles(task, variants, Path(args.data_root))
    reference = bundles[variants[0]]
    shared_splits = {
        key: value.numpy() for key, value in reference["splits"].items()
    }

    train_positive = shared_splits["train_pos"]
    train_negative = shared_splits["train_neg"]
    batches_per_variant = int(np.ceil(len(train_positive) / args.batch_size))
    if batches_per_variant <= 0:
        raise ValueError("Training split is empty")

    encoder = RGCNEncoder(
        num_nodes=int(reference["graph"].num_nodes()),
        num_rels=len(IMDB_LP_RELATIONS),
        in_dim=args.in_dim,
        hid_dim=args.hid_dim,
        out_dim=args.out_dim,
        num_layers=args.layers,
        num_bases=args.num_bases,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    seed_dir = output_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = seed_dir / "shared_checkpoint.pt"
    latest_state_path = seed_dir / "latest_training_state.pt"
    early_stopper = EarlyStopper(mode="min", patience=args.patience)
    rng = np.random.RandomState(seed)
    run_config = {
        "dataset": "IMDB_LP",
        "task": task,
        "seed": seed,
        "variants": variants,
        "in_dim": args.in_dim,
        "hid_dim": args.hid_dim,
        "out_dim": args.out_dim,
        "layers": args.layers,
        "num_bases": args.num_bases,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "emb_reg": args.emb_reg,
        "grad_clip": args.grad_clip,
        "batch_size": args.batch_size,
        "threshold": args.threshold,
        "patience": args.patience,
        "global_relations": list(IMDB_LP_RELATIONS),
    }

    history: List[Dict[str, Any]] = []
    optimizer_steps = 0
    variant_epochs = 0
    train_graph_forwards = 0
    validation_graph_forwards = 0
    super_epochs_ran = 0
    prior_training_seconds = 0.0
    prior_peak_rss = 0
    prior_training_gpu: Dict[str, int] = {}

    resume_state = load_latest_training_state(
        latest_state_path,
        resume=args.resume,
        run_config=run_config,
        modules={"encoder": encoder},
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
        super_epochs_ran = int(resume_state["completed_super_epoch"])
        prior_training_seconds = float(resume_state["training_seconds_elapsed"])
        prior_peak_rss = int(resume_state.get("process_peak_rss_bytes", 0))
        prior_training_gpu = dict(resume_state.get("training_gpu", {}))

    reset_cuda_peak(device)
    training_start = time.perf_counter()

    for super_epoch in range(super_epochs_ran, args.super_epochs):
        if early_stopper.should_stop:
            print(
                "[resume] This seed already satisfied early stopping; skipping further training.",
                flush=True,
            )
            break

        cycle_start = time.perf_counter()
        variant_order = [variants[index] for index in rng.permutation(len(variants))]
        train_losses: Dict[str, float] = {}

        for variant in variant_order:
            bundle = device_bundle(bundles[variant], device)
            positive_order = rng.permutation(len(train_positive))
            batch_losses: List[float] = []
            encoder.train()

            for batch_index in range(batches_per_variant):
                selected = positive_order[
                    batch_index * args.batch_size : (batch_index + 1) * args.batch_size
                ]
                if len(selected) == 0:
                    continue
                positive_batch = train_positive[selected]
                negative_matrix = train_negative[selected]
                negative_edges = flatten_negative_matrix(
                    positive_batch, negative_matrix
                )

                optimizer.zero_grad(set_to_none=True)
                movie_embeddings, tail_embeddings = node_type_embeddings(encoder, bundle)
                train_graph_forwards += 1
                positive_logits = score_edges(
                    movie_embeddings, tail_embeddings, positive_batch
                )
                negative_logits = score_edges(
                    movie_embeddings, tail_embeddings, negative_edges
                )
                loss = pairwise_loss(positive_logits, negative_logits)
                loss = loss + args.emb_reg * encoder.emb.weight.pow(2).mean()
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        encoder.parameters(), args.grad_clip
                    )
                optimizer.step()

                optimizer_steps += 1
                batch_losses.append(float(loss.detach().cpu()))
                del (
                    movie_embeddings,
                    tail_embeddings,
                    positive_logits,
                    negative_logits,
                    loss,
                )

            variant_epochs += 1
            train_losses[variant] = float(np.mean(batch_losses))
            del bundle

        validation_losses: Dict[str, float] = {}
        for variant in variants:
            validation_losses[variant] = evaluate_validation(
                encoder, bundles[variant], shared_splits, device
            )
            validation_graph_forwards += 1
        mean_validation_loss = float(np.mean(list(validation_losses.values())))
        super_epochs_ran = super_epoch + 1
        improved = early_stopper.update(mean_validation_loss)
        if improved:
            atomic_torch_save(
                {
                    "encoder": encoder.state_dict(),
                    "metadata": {
                        "dataset": "IMDB_LP",
                        "task": task,
                        "seed": seed,
                        "variants": variants,
                        "best_super_epoch": super_epochs_ran,
                        "optimizer_steps": optimizer_steps,
                        "variant_epochs": variant_epochs,
                        "batches_per_variant": batches_per_variant,
                        "global_relations": list(IMDB_LP_RELATIONS),
                    },
                },
                checkpoint_path,
            )

        row: Dict[str, Any] = {
            "super_epoch": super_epochs_ran,
            "variant_order": ",".join(variant_order),
            "variant_epochs_cumulative": variant_epochs,
            "optimizer_steps_cumulative": optimizer_steps,
            "batches_per_variant": batches_per_variant,
            "train_graph_forwards_cumulative": train_graph_forwards,
            "validation_graph_forwards_cumulative": validation_graph_forwards,
            "mean_train_loss": float(np.mean(list(train_losses.values()))),
            "mean_val_loss": mean_validation_loss,
            "best_mean_val_loss": early_stopper.best,
            "cycle_seconds": time.perf_counter() - cycle_start,
        }
        for variant in variants:
            row[f"train_loss_{variant}"] = train_losses[variant]
            row[f"val_loss_{variant}"] = validation_losses[variant]
        history.append(row)
        atomic_write_csv(pd.DataFrame(history), seed_dir / "training_history.csv")

        segment_seconds = time.perf_counter() - training_start
        current_training_gpu = merge_cuda_memory_stats(
            prior_training_gpu, cuda_memory_stats(device)
        )
        current_peak_rss = max(prior_peak_rss, process_peak_rss_bytes())
        save_latest_training_state(
            latest_state_path,
            dataset=f"IMDB_LP_{task}",
            run_config=run_config,
            modules={"encoder": encoder},
            optimizer=optimizer,
            early_stopper=early_stopper,
            rng=rng,
            completed_super_epoch=super_epochs_ran,
            counters={
                "optimizer_steps": optimizer_steps,
                "variant_epochs": variant_epochs,
                "train_graph_forwards": train_graph_forwards,
                "validation_graph_forwards": validation_graph_forwards,
            },
            history=history,
            training_seconds_elapsed=prior_training_seconds + segment_seconds,
            process_peak_rss_bytes_value=current_peak_rss,
            training_gpu=current_training_gpu,
        )

        print(
            f"task={task} seed={seed} super_epoch={super_epochs_ran:03d} "
            f"variant_epochs={variant_epochs} optimizer_steps={optimizer_steps} "
            f"batches_per_variant={batches_per_variant} "
            f"mean_train_loss={row['mean_train_loss']:.6f} "
            f"mean_val_loss={mean_validation_loss:.6f} "
            f"best={early_stopper.best:.6f} order={variant_order}",
            flush=True,
        )
        if early_stopper.should_stop:
            print(
                "Early stopping after a complete balanced super-epoch.", flush=True
            )
            break

    training_seconds = prior_training_seconds + (time.perf_counter() - training_start)
    training_memory = merge_cuda_memory_stats(
        prior_training_gpu, cuda_memory_stats(device)
    )
    cumulative_peak_rss = max(prior_peak_rss, process_peak_rss_bytes())

    if not checkpoint_path.exists():
        raise RuntimeError("No best checkpoint was saved")
    best_state = torch_load_full(checkpoint_path, map_location=device)
    encoder.load_state_dict(best_state["encoder"])

    reset_cuda_peak(device)
    per_variant_metrics: Dict[str, Dict[str, float]] = {}
    score_frames: Dict[str, pd.DataFrame] = {}
    ranks_by_variant: Dict[str, np.ndarray] = {}
    test_graph_forwards = 0
    for variant in variants:
        metrics, score_frame, ranks = evaluate_test(
            task,
            encoder,
            bundles[variant],
            shared_splits,
            device,
            args.threshold,
        )
        test_graph_forwards += 1
        metrics["edge_count"] = float(bundles[variant]["edge_count"])
        per_variant_metrics[variant] = metrics
        score_frames[variant] = score_frame
        ranks_by_variant[variant] = ranks
        score_frame.to_csv(seed_dir / f"test_scores_{variant}.csv", index=False)
        score_frame.to_csv(
            seed_dir
            / f"IMDB_rgcn_lp_augmentation_{task}_{variant}_seed{seed}_scores.csv",
            index=False,
        )

    inference_memory = cuda_memory_stats(device)
    invariance_rows = imdb_lp_invariance_rows(
        score_frames, ranks_by_variant, args.threshold
    )
    pd.DataFrame(invariance_rows).to_csv(
        seed_dir / "pairwise_invariance.csv", index=False
    )
    pd.DataFrame(
        [
            {"task": task, "variant": variant, **metrics}
            for variant, metrics in per_variant_metrics.items()
        ]
    ).to_csv(seed_dir / "test_metrics_by_variant.csv", index=False)

    expected_optimizer_steps = (
        super_epochs_ran * len(variants) * batches_per_variant
    )
    expected_variant_epochs = super_epochs_ran * len(variants)
    if (
        optimizer_steps != expected_optimizer_steps
        or variant_epochs != expected_variant_epochs
    ):
        raise AssertionError(
            "Incorrect epoch/update accounting: "
            f"actual steps={optimizer_steps}, expected={expected_optimizer_steps}; "
            f"actual variant epochs={variant_epochs}, "
            f"expected={expected_variant_epochs}"
        )

    memory = model_memory_bytes(encoder)
    summary = {
        "dataset": "IMDB_LP",
        "task": task,
        "seed": seed,
        "variants": variants,
        "epoch_accounting": {
            "definition": (
                "one super-epoch visits every selected graph variant and all "
                "positive-row minibatches within each variant"
            ),
            "super_epochs_ran": super_epochs_ran,
            "variant_epochs_ran": variant_epochs,
            "batches_per_variant": batches_per_variant,
            "optimizer_steps": optimizer_steps,
            "expected_optimizer_steps": expected_optimizer_steps,
            "train_graph_forwards": train_graph_forwards,
            "validation_graph_forwards": validation_graph_forwards,
            "test_graph_forwards": test_graph_forwards,
        },
        "shared_split_shapes": {
            key: list(value.shape) for key, value in shared_splits.items()
        },
        "negative_candidates_per_positive": {
            split: int(shared_splits[f"{split}_neg"].shape[1])
            for split in ("train", "val", "test")
        },
        "training_seconds": training_seconds,
        "best_mean_val_loss": early_stopper.best,
        "mean_test_metrics": mean_dict(list(per_variant_metrics.values())),
        "per_variant_test_metrics": per_variant_metrics,
        "pairwise_invariance": invariance_rows,
        "memory": {
            **memory,
            "checkpoint_bytes": checkpoint_size_bytes(checkpoint_path),
            "process_peak_rss_bytes": cumulative_peak_rss,
            "training_gpu": training_memory,
            "inference_gpu": inference_memory,
        },
        "global_native_relations": list(IMDB_LP_RELATIONS),
        "graph_keys_by_variant": {
            variant: bundles[variant]["graph_keys"] for variant in variants
        },
    }
    write_json(seed_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IMDb RGCN link-prediction joint graph-variant augmentation"
    )
    parser.add_argument("--task", type=parse_task, required=True)
    parser.add_argument(
        "--variants",
        default="",
        help=(
            "Comma-separated variants. Defaults to v1,v3 for md and "
            "v1,v2,v3,v4 for ml/mg."
        ),
    )
    parser.add_argument("--data-root", default="data/preprocessed")
    parser.add_argument("--seeds", default="1566911444,20241017,20251017")
    parser.add_argument("--super-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--threshold", "--th", dest="threshold", type=float, default=0.5)
    parser.add_argument("--in-dim", type=int, default=128)
    parser.add_argument("--hid-dim", type=int, default=256)
    parser.add_argument("--out-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--num-bases", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--emb-reg", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest_training_state.pt if present",
    )
    parser.add_argument(
        "--output-dir", default="results/rgcn_augmentation/IMDB_LP"
    )
    args = parser.parse_args()

    task = args.task
    variants = parse_variants(args.variants, task)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    output_root = Path(args.output_dir) / task
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = [
        run_seed(args, task, variants, seed, output_root) for seed in seeds
    ]
    rows: List[Dict[str, Any]] = []
    for summary in summaries:
        rows.append(
            {
                "task": task,
                "seed": summary["seed"],
                "super_epochs_ran": summary["epoch_accounting"]["super_epochs_ran"],
                "variant_epochs_ran": summary["epoch_accounting"]["variant_epochs_ran"],
                "batches_per_variant": summary["epoch_accounting"]["batches_per_variant"],
                "optimizer_steps": summary["epoch_accounting"]["optimizer_steps"],
                "training_seconds": summary["training_seconds"],
                "best_mean_val_loss": summary["best_mean_val_loss"],
                **{
                    f"mean_test_{key}": value
                    for key, value in summary["mean_test_metrics"].items()
                },
            }
        )
    pd.DataFrame(rows).to_csv(output_root / "seed_summary.csv", index=False)
    write_json(output_root / "all_seed_summaries.json", {"runs": summaries})
    print(f"[OK] IMDb-LP {task} results written under {output_root}")


if __name__ == "__main__":
    main()
