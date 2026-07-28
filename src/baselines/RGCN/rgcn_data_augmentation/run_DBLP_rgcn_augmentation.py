#!/usr/bin/env python3
"""Joint graph-variant data augmentation for DBLP RGCN or SlotGAT LP.

One encoder, optimizer, and checkpoint are shared across v1-v3.  A super-epoch
visits every variant once and processes all positive training edges for each.
For B=ceil(num_train_positive/batch_size) batches per variant:

    variant_epochs = super_epochs * number_of_variants
    optimizer_steps = super_epochs * number_of_variants * B

The encoder performs full-graph propagation inside every supervised edge
batch, matching the original repository runner. A nonpositive batch size uses
one full positive-edge batch. Early stopping is checked only
after a complete balanced super-epoch using mean validation BCE across variants.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rgcn_aug_common import (
    DBLP_RELATIONS,
    EarlyStopper,
    torch_load_full,
    save_latest_training_state,
    merge_cuda_memory_stats,
    load_latest_training_state,
    atomic_write_csv,
    atomic_torch_save,
    RGCNEncoder,
    assert_same_indexers,
    assert_same_tensor,
    checkpoint_size_bytes,
    cuda_memory_stats,
    group_negatives_by_query,
    link_binary_metrics,
    link_invariance_rows,
    link_ranking_metrics,
    mean_dict,
    model_memory_bytes,
    pairwise_loss,
    process_peak_rss_bytes,
    relation_to_id,
    reset_cuda_peak,
    resolve_device,
    set_determinism,
    subsample_negatives_per_query,
    to_homogeneous_with_global_relations,
    write_json,
)


SLOTGAT_ROOT = Path(__file__).resolve().parents[2] / "SlotGAT"
VALID_VARIANTS = ("v1", "v2", "v3")


def parse_variants(spec: str) -> List[str]:
    variants = [item.strip().lower() for item in spec.split(",") if item.strip()]
    if not variants or len(set(variants)) != len(variants):
        raise SystemExit("--variants must contain distinct variant names")
    unknown = set(variants) - set(VALID_VARIANTS)
    if unknown:
        raise SystemExit(f"Unknown variants: {sorted(unknown)}")
    return variants


def load_preprocessed(base_dir: str):
    graph_data = torch.load(os.path.join(base_dir, "graph_data.pt"), map_location="cpu")
    meta = torch.load(os.path.join(base_dir, "meta.pt"), map_location="cpu")
    return graph_data, meta


def build_graph(graph_data: Mapping[str, Any], num_nodes: Mapping[str, int], variant: str):
    data = {
        ("author", "author-paper", "paper"): graph_data["author-paper"],
        ("paper", "paper-author", "author"): (graph_data["author-paper"][1], graph_data["author-paper"][0]),
        ("paper", "paper-conference", "conference"): graph_data["paper-conference"],
        ("conference", "conference-paper", "paper"): (graph_data["paper-conference"][1], graph_data["paper-conference"][0]),
        ("paper", "paper-term", "term"): graph_data["paper-term"],
        ("term", "term-paper", "paper"): (graph_data["paper-term"][1], graph_data["paper-term"][0]),
    }
    if variant == "v1":
        data[("paper", "paper-area", "area")] = graph_data["paper-area"]
        data[("area", "area-paper", "paper")] = (graph_data["paper-area"][1], graph_data["paper-area"][0])
    elif variant == "v2":
        data[("conference", "conference-area", "area")] = graph_data["conference-area"]
        data[("area", "area-conference", "conference")] = (graph_data["conference-area"][1], graph_data["conference-area"][0])
    elif variant == "v3":
        data[("author", "author-area", "area")] = graph_data["author-area"]
        data[("area", "area-author", "author")] = (graph_data["author-area"][1], graph_data["author-area"][0])
    else:
        raise ValueError(variant)
    return dgl.heterograph(data, num_nodes_dict={key: int(value) for key, value in num_nodes.items()})


def prepare_bundles(variants: List[str], data_root: Path):
    relation_ids = relation_to_id(DBLP_RELATIONS)
    bundles: Dict[str, Dict[str, Any]] = {}
    num_nodes_dicts = {}
    indexers_by_variant = {}
    splits_by_name: Dict[str, Dict[str, torch.Tensor]] = {}
    node_types_by_variant: Dict[str, torch.Tensor] = {}

    for variant in variants:
        graph_data, meta = load_preprocessed(
            str(data_root / f"DBLP_rgcn_{variant}")
        )
        num_nodes = {key: int(value) for key, value in meta["num_nodes"].items()}
        graph = build_graph(graph_data, num_nodes, variant)
        homogeneous, edge_types, indexers = to_homogeneous_with_global_relations(graph, relation_ids)
        bundles[variant] = {
            "variant": variant,
            "graph": homogeneous,
            "edge_types": edge_types,
            "node_type": homogeneous.ndata[dgl.NTYPE].long().cpu(),
            "indexers": indexers,
            "paper_indexer": indexers["paper"],
            "conference_indexer": indexers["conference"],
            "edge_count": int(homogeneous.num_edges()),
            "splits": {key: value.long().cpu() for key, value in meta["splits"].items()},
        }
        num_nodes_dicts[variant] = num_nodes
        indexers_by_variant[variant] = indexers
        node_types_by_variant[variant] = bundles[variant]["node_type"]
        for key, value in bundles[variant]["splits"].items():
            splits_by_name.setdefault(key, {})[variant] = value

    reference_variant = variants[0]
    for variant in variants[1:]:
        if num_nodes_dicts[variant] != num_nodes_dicts[reference_variant]:
            raise ValueError(f"num_nodes differs between {reference_variant} and {variant}")
        if bundles[variant]["graph"].num_nodes() != bundles[reference_variant]["graph"].num_nodes():
            raise ValueError("Homogeneous node counts differ across variants")
    assert_same_indexers(indexers_by_variant)
    assert_same_tensor("homogeneous node types", node_types_by_variant)
    for split_name, values in splits_by_name.items():
        assert_same_tensor(f"splits.{split_name}", values)
    return bundles


def device_bundle(cpu_bundle: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        **cpu_bundle,
        "graph": cpu_bundle["graph"].to(device),
        "edge_types": cpu_bundle["edge_types"].to(device),
        "paper_indexer": cpu_bundle["paper_indexer"].to(device),
        "conference_indexer": cpu_bundle["conference_indexer"].to(device),
    }


def node_type_embeddings(encoder: nn.Module, bundle: Mapping[str, Any]):
    all_embeddings = encoder(bundle["graph"], bundle["edge_types"])
    return (
        all_embeddings[bundle["paper_indexer"]],
        all_embeddings[bundle["conference_indexer"]],
    )


def score_edges(paper_embeddings, conference_embeddings, edges: np.ndarray) -> torch.Tensor:
    edge_tensor = torch.as_tensor(edges, dtype=torch.long, device=paper_embeddings.device)
    return (
        paper_embeddings[edge_tensor[:, 0]]
        * conference_embeddings[edge_tensor[:, 1]]
    ).sum(dim=-1)


def evaluate_validation(encoder, cpu_bundle, shared_splits, device):
    bundle = device_bundle(cpu_bundle, device)
    encoder.eval()
    bce = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        paper_embeddings, conference_embeddings = node_type_embeddings(encoder, bundle)
        positive_logits = score_edges(paper_embeddings, conference_embeddings, shared_splits["val_pos"])
        negative_logits = score_edges(paper_embeddings, conference_embeddings, shared_splits["val_neg"])
        labels = torch.cat([torch.ones_like(positive_logits), torch.zeros_like(negative_logits)])
        logits = torch.cat([positive_logits, negative_logits])
        validation_loss = float(bce(logits, labels).cpu())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return validation_loss


def evaluate_test(encoder, cpu_bundle, shared_splits, device, threshold):
    bundle = device_bundle(cpu_bundle, device)
    encoder.eval()
    with torch.no_grad():
        paper_embeddings, conference_embeddings = node_type_embeddings(encoder, bundle)
        positive_probabilities = torch.sigmoid(
            score_edges(paper_embeddings, conference_embeddings, shared_splits["test_pos"])
        ).cpu().numpy()
        negative_probabilities = torch.sigmoid(
            score_edges(paper_embeddings, conference_embeddings, shared_splits["test_neg"])
        ).cpu().numpy()
    y_true = np.concatenate(
        [np.ones_like(positive_probabilities), np.zeros_like(negative_probabilities)]
    )
    y_score = np.concatenate([positive_probabilities, negative_probabilities])
    metrics = link_binary_metrics(y_true, y_score, threshold)
    metrics.update(
        link_ranking_metrics(
            shared_splits["test_pos"],
            shared_splits["test_neg"],
            positive_probabilities,
            negative_probabilities,
        )
    )
    positive_frame = pd.DataFrame(
        {
            "paper_id": shared_splits["test_pos"][:, 0],
            "conf_id": shared_splits["test_pos"][:, 1],
            "score": positive_probabilities,
            "label": 1,
        }
    )
    negative_frame = pd.DataFrame(
        {
            "paper_id": shared_splits["test_neg"][:, 0],
            "conf_id": shared_splits["test_neg"][:, 1],
            "score": negative_probabilities,
            "label": 0,
        }
    )
    return metrics, pd.concat([positive_frame, negative_frame], ignore_index=True)


def build_encoder(args, reference: Mapping[str, Any], device: torch.device):
    if args.encoder == "rgcn":
        return RGCNEncoder(
            num_nodes=int(reference["graph"].num_nodes()),
            num_rels=len(DBLP_RELATIONS),
            in_dim=args.in_dim,
            hid_dim=args.hid_dim,
            out_dim=args.out_dim,
            num_layers=args.layers,
            num_bases=args.num_bases,
            dropout=args.dropout,
        ).to(device)

    slotgat_root = str(SLOTGAT_ROOT)
    if slotgat_root not in sys.path:
        sys.path.insert(0, slotgat_root)
    from model_slotgat_lp_heterogeneous import HeterogeneousSlotGATEncoder

    return HeterogeneousSlotGATEncoder(
        node_type=reference["node_type"],
        num_relations=len(DBLP_RELATIONS),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        edge_feats=args.edge_feats,
        dropout_feat=args.dropout_feat,
        dropout_attn=args.dropout_attn,
        negative_slope=args.slope,
        alpha=args.alpha,
        aggregator=args.aggregator,
        sa_att_dim=args.sa_att_dim,
        edge_chunk_size=args.slotgat_edge_chunk_size,
        decomposed_layers=args.slotgat_decomposed_layers,
    ).to(device)


def encoder_name(encoder: str) -> str:
    return (
        "HeterogeneousSlotGATEncoder"
        if encoder == "slotgat"
        else "legacy_RGCNEncoder"
    )


def run_seed(args, variants: List[str], seed: int, output_root: Path) -> Dict[str, Any]:
    set_determinism(seed)
    if args.encoder == "slotgat":
        torch.use_deterministic_algorithms(True, warn_only=True)
    device = resolve_device(args.device)
    bundles = prepare_bundles(variants, Path(args.data_root))
    reference = bundles[variants[0]]
    shared_splits = {key: value.numpy() for key, value in reference["splits"].items()}

    if args.neg_per_paper > 0:
        shared_splits["train_neg"] = subsample_negatives_per_query(
            shared_splits["train_neg"], args.neg_per_paper, seed + 11
        )
        shared_splits["val_neg"] = subsample_negatives_per_query(
            shared_splits["val_neg"], args.neg_per_paper, seed + 13
        )
        shared_splits["test_neg"] = subsample_negatives_per_query(
            shared_splits["test_neg"], args.neg_per_paper, seed + 17
        )

    train_positive = shared_splits["train_pos"]
    negative_by_paper = group_negatives_by_query(shared_splits["train_neg"])
    if len(train_positive) == 0:
        raise ValueError("Training split is empty")
    effective_batch_size = (
        len(train_positive) if args.batch_size <= 0 else args.batch_size
    )
    batches_per_variant = int(
        np.ceil(len(train_positive) / effective_batch_size)
    )
    if batches_per_variant <= 0:
        raise AssertionError("No DBLP training batches were created")

    encoder = build_encoder(args, reference, device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    seed_dir = output_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = seed_dir / "shared_checkpoint.pt"
    latest_state_path = seed_dir / "latest_training_state.pt"
    early_stopper = EarlyStopper(mode="min", patience=args.patience)
    rng = np.random.RandomState(seed)
    run_config: Dict[str, Any] = {
        "dataset": "DBLP", "seed": seed, "variants": variants,
        "in_dim": args.in_dim, "hid_dim": args.hid_dim, "out_dim": args.out_dim,
        "layers": args.layers, "num_bases": args.num_bases, "dropout": args.dropout,
        "lr": args.lr, "weight_decay": args.weight_decay, "grad_clip": args.grad_clip,
        "emb_reg": args.emb_reg, "batch_size": args.batch_size,
        "neg_per_paper": args.neg_per_paper, "patience": args.patience,
    }
    if args.encoder == "slotgat":
        # Preserve the historical RGCN resume fingerprint while fully
        # fingerprinting every SlotGAT architecture and batching choice.
        run_config.update(
            {
                "encoder": args.encoder,
                "data_root": str(Path(args.data_root).resolve()),
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "edge_feats": args.edge_feats,
                "dropout_feat": args.dropout_feat,
                "dropout_attn": args.dropout_attn,
                "slope": args.slope,
                "alpha": args.alpha,
                "aggregator": args.aggregator,
                "sa_att_dim": args.sa_att_dim,
                "slotgat_edge_chunk_size": (
                    args.slotgat_edge_chunk_size
                ),
                "slotgat_decomposed_layers": (
                    args.slotgat_decomposed_layers
                ),
                "effective_batch_size": effective_batch_size,
                "global_relations": list(DBLP_RELATIONS),
            }
        )

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
        latest_state_path, resume=args.resume, run_config=run_config,
        modules={"encoder": encoder}, optimizer=optimizer, early_stopper=early_stopper,
        rng=rng, device=device,
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
        order = [variants[index] for index in rng.permutation(len(variants))]
        train_losses: Dict[str, float] = {}

        for variant in order:
            bundle = device_bundle(bundles[variant], device)
            positive_order = rng.permutation(len(train_positive))
            batch_losses: List[float] = []
            encoder.train()

            for batch_index in range(batches_per_variant):
                selected = positive_order[
                    batch_index
                    * effective_batch_size : (batch_index + 1)
                    * effective_batch_size
                ]
                positive_batch = train_positive[selected]
                negative_blocks = [
                    negative_by_paper[int(paper_id)]
                    for paper_id in positive_batch[:, 0]
                    if int(paper_id) in negative_by_paper
                ]
                if not negative_blocks:
                    raise ValueError("No negative examples found for the current positive batch")
                negative_batch = np.concatenate(negative_blocks, axis=0)

                optimizer.zero_grad(set_to_none=True)
                paper_embeddings, conference_embeddings = node_type_embeddings(encoder, bundle)
                train_graph_forwards += 1
                positive_logits = score_edges(paper_embeddings, conference_embeddings, positive_batch)
                negative_logits = score_edges(paper_embeddings, conference_embeddings, negative_batch)
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
                del paper_embeddings, conference_embeddings, positive_logits, negative_logits, loss

            variant_epochs += 1
            train_losses[variant] = float(np.mean(batch_losses))
            del bundle

        val_losses = {}
        for variant in variants:
            val_losses[variant] = evaluate_validation(
                encoder, bundles[variant], shared_splits, device
            )
            validation_graph_forwards += 1
        mean_val_loss = float(np.mean(list(val_losses.values())))
        super_epochs_ran = super_epoch + 1
        improved = early_stopper.update(mean_val_loss)
        if improved:
            atomic_torch_save(
                {
                    "encoder": encoder.state_dict(),
                    "metadata": {
                        "encoder": args.encoder,
                        "model": encoder_name(args.encoder),
                        "seed": seed,
                        "variants": variants,
                        "best_super_epoch": super_epochs_ran,
                        "optimizer_steps": optimizer_steps,
                        "variant_epochs": variant_epochs,
                        "batches_per_variant": batches_per_variant,
                        "effective_batch_size": effective_batch_size,
                        "global_relations": list(DBLP_RELATIONS),
                    },
                },
                checkpoint_path,
            )

        row: Dict[str, Any] = {
            "super_epoch": super_epochs_ran,
            "variant_order": ",".join(order),
            "variant_epochs_cumulative": variant_epochs,
            "optimizer_steps_cumulative": optimizer_steps,
            "batches_per_variant": batches_per_variant,
            "train_graph_forwards_cumulative": train_graph_forwards,
            "validation_graph_forwards_cumulative": validation_graph_forwards,
            "mean_train_loss": float(np.mean(list(train_losses.values()))),
            "mean_val_loss": mean_val_loss,
            "best_mean_val_loss": early_stopper.best,
            "cycle_seconds": time.perf_counter() - cycle_start,
        }
        for variant in variants:
            row[f"train_loss_{variant}"] = train_losses[variant]
            row[f"val_loss_{variant}"] = val_losses[variant]
        history.append(row)
        atomic_write_csv(pd.DataFrame(history), seed_dir / "training_history.csv")
        segment_seconds = time.perf_counter() - training_start
        current_training_gpu = merge_cuda_memory_stats(
            prior_training_gpu, cuda_memory_stats(device)
        )
        current_peak_rss = max(prior_peak_rss, process_peak_rss_bytes())
        save_latest_training_state(
            latest_state_path, dataset="DBLP", run_config=run_config,
            modules={"encoder": encoder}, optimizer=optimizer, early_stopper=early_stopper,
            rng=rng, completed_super_epoch=super_epochs_ran,
            counters={
                "optimizer_steps": optimizer_steps, "variant_epochs": variant_epochs,
                "train_graph_forwards": train_graph_forwards,
                "validation_graph_forwards": validation_graph_forwards,
            },
            history=history,
            training_seconds_elapsed=prior_training_seconds + segment_seconds,
            process_peak_rss_bytes_value=current_peak_rss, training_gpu=current_training_gpu,
        )

        print(
            f"seed={seed} super_epoch={super_epochs_ran:03d} "
            f"variant_epochs={variant_epochs} optimizer_steps={optimizer_steps} "
            f"batches_per_variant={batches_per_variant} "
            f"mean_train_loss={row['mean_train_loss']:.6f} "
            f"mean_val_loss={mean_val_loss:.6f} best={early_stopper.best:.6f} "
            f"order={order}",
            flush=True,
        )
        if early_stopper.should_stop:
            print("Early stopping after a complete balanced super-epoch.", flush=True)
            break

    training_seconds = prior_training_seconds + (time.perf_counter() - training_start)
    training_memory = merge_cuda_memory_stats(prior_training_gpu, cuda_memory_stats(device))
    cumulative_peak_rss = max(prior_peak_rss, process_peak_rss_bytes())
    if not checkpoint_path.exists():
        raise RuntimeError("No checkpoint was saved")
    state = torch_load_full(checkpoint_path, map_location=device)
    encoder.load_state_dict(state["encoder"])

    reset_cuda_peak(device)
    per_variant_metrics: Dict[str, Dict[str, float]] = {}
    score_frames: Dict[str, pd.DataFrame] = {}
    test_graph_forwards = 0
    for variant in variants:
        metrics, score_frame = evaluate_test(
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
        score_frame.to_csv(seed_dir / f"test_scores_{variant}.csv", index=False)

    inference_memory = cuda_memory_stats(device)
    invariance_rows = link_invariance_rows(score_frames, args.threshold)
    pd.DataFrame(invariance_rows).to_csv(seed_dir / "pairwise_invariance.csv", index=False)
    pd.DataFrame(
        [{"variant": variant, **metrics} for variant, metrics in per_variant_metrics.items()]
    ).to_csv(seed_dir / "test_metrics_by_variant.csv", index=False)

    memory = model_memory_bytes(encoder)
    expected_optimizer_steps = super_epochs_ran * len(variants) * batches_per_variant
    expected_variant_epochs = super_epochs_ran * len(variants)
    if optimizer_steps != expected_optimizer_steps or variant_epochs != expected_variant_epochs:
        raise AssertionError(
            "Incorrect epoch/update accounting: "
            f"actual steps={optimizer_steps}, expected={expected_optimizer_steps}; "
            f"actual variant epochs={variant_epochs}, expected={expected_variant_epochs}"
        )

    summary = {
        "dataset": "DBLP",
        "encoder": args.encoder,
        "model": encoder_name(args.encoder),
        "seed": seed,
        "variants": variants,
        "epoch_accounting": {
            "definition": "one super-epoch visits every variant and all positive-edge batches within each variant",
            "super_epochs_ran": super_epochs_ran,
            "variant_epochs_ran": variant_epochs,
            "batches_per_variant": batches_per_variant,
            "effective_batch_size": effective_batch_size,
            "optimizer_steps": optimizer_steps,
            "expected_optimizer_steps": expected_optimizer_steps,
            "single_variant_epoch_equivalents": variant_epochs,
            "train_graph_forwards": train_graph_forwards,
            "validation_graph_forwards": validation_graph_forwards,
            "test_graph_forwards": test_graph_forwards,
        },
        "shared_split_sizes": {key: int(len(value)) for key, value in shared_splits.items()},
        "negative_candidates_per_paper_cap": args.neg_per_paper,
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
        "global_native_relations": list(DBLP_RELATIONS),
        "structural_self_loop_edge_type": (
            len(DBLP_RELATIONS)
            if args.encoder == "slotgat"
            else None
        ),
        "slotgat": (
            {
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "edge_feats": args.edge_feats,
                "dropout_feat": args.dropout_feat,
                "dropout_attn": args.dropout_attn,
                "slope": args.slope,
                "alpha": args.alpha,
                "aggregator": args.aggregator,
                "sa_att_dim": args.sa_att_dim,
                "edge_chunk_size": args.slotgat_edge_chunk_size,
                "decomposed_layers": args.slotgat_decomposed_layers,
            }
            if args.encoder == "slotgat"
            else None
        ),
    }
    write_json(seed_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DBLP RGCN or SlotGAT joint graph-variant augmentation"
    )
    parser.add_argument(
        "--encoder",
        choices=("rgcn", "slotgat"),
        default="rgcn",
    )
    parser.add_argument("--variants", default="v1,v2,v3")
    parser.add_argument("--data-root", default="data/preprocessed")
    parser.add_argument("--seeds", default="1566911444,20241017,20251017")
    parser.add_argument("--super-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Positive edges per graph encoding; 0 selects one full batch",
    )
    parser.add_argument("--neg-per-paper", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--in-dim", type=int, default=128)
    parser.add_argument("--hid-dim", type=int, default=256)
    parser.add_argument("--out-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--num-bases", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--edge-feats", type=int, default=64)
    parser.add_argument("--dropout-feat", type=float, default=0.5)
    parser.add_argument("--dropout-attn", type=float, default=0.2)
    parser.add_argument("--slope", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--aggregator",
        choices=("SA", "average", "max"),
        default="SA",
    )
    parser.add_argument("--sa-att-dim", type=int, default=3)
    parser.add_argument(
        "--slotgat-edge-chunk-size",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--slotgat-decomposed-layers",
        type=int,
        default=1,
    )
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--emb-reg", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true", help="Resume from latest_training_state.pt if present")
    parser.add_argument("--output-dir", default="results/rgcn_augmentation/DBLP")
    args = parser.parse_args()
    if args.batch_size < 0:
        raise SystemExit("--batch-size must be zero or a positive integer")
    if args.encoder == "slotgat" and (
        args.hidden_dim <= 0
        or args.num_layers <= 0
        or args.num_heads <= 0
        or args.edge_feats < 0
        or args.sa_att_dim <= 0
        or args.slotgat_edge_chunk_size < 0
        or args.slotgat_decomposed_layers <= 0
    ):
        raise SystemExit(
            "SlotGAT requires positive hidden/layer/head/SA dimensions and "
            "decomposed layers, plus nonnegative edge dimensions/chunk size"
        )

    variants = parse_variants(args.variants)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = [run_seed(args, variants, seed, output_root) for seed in seeds]
    rows = []
    for summary in summaries:
        row = {
            "encoder": summary["encoder"],
            "seed": summary["seed"],
            "super_epochs_ran": summary["epoch_accounting"]["super_epochs_ran"],
            "variant_epochs_ran": summary["epoch_accounting"]["variant_epochs_ran"],
            "batches_per_variant": summary["epoch_accounting"]["batches_per_variant"],
            "optimizer_steps": summary["epoch_accounting"]["optimizer_steps"],
            "training_seconds": summary["training_seconds"],
            "best_mean_val_loss": summary["best_mean_val_loss"],
            **{f"mean_test_{key}": value for key, value in summary["mean_test_metrics"].items()},
        }
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_root / "seed_summary.csv", index=False)
    write_json(output_root / "all_seed_summaries.json", {"runs": summaries})
    print(f"[OK] Results written under {output_root}")


if __name__ == "__main__":
    main()
