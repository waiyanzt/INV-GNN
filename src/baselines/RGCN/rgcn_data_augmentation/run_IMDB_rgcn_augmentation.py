#!/usr/bin/env python3
"""Joint graph-variant data augmentation for IMDb RGCN node classification.

One encoder, classifier, optimizer, and checkpoint are shared across v1-v4.
A *super-epoch* is one balanced visit to every selected graph variant.  Because
IMDb training is full batch, each variant visit performs exactly one optimizer
step.  Therefore:

    variant_epochs = super_epochs * number_of_variants
    optimizer_steps = variant_epochs

Early stopping is checked only after a complete super-epoch using mean
validation Macro-F1 over variants.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from rgcn_aug_common import (
    IMDB_RELATIONS,
    EarlyStopper,
    torch_load_full,
    save_latest_training_state,
    merge_cuda_memory_stats,
    load_latest_training_state,
    atomic_write_csv,
    atomic_torch_save,
    NodeClassifier,
    RGCNEncoder,
    assert_same_indexers,
    assert_same_tensor,
    checkpoint_size_bytes,
    classification_invariance_rows,
    classification_metrics,
    cuda_memory_stats,
    mean_dict,
    model_memory_bytes,
    process_peak_rss_bytes,
    relation_to_id,
    reset_cuda_peak,
    resolve_device,
    set_determinism,
    to_homogeneous_with_global_relations,
    write_json,
)


BASE = {
    "v1": "data/preprocessed/IMDB_rgcn_v1",
    "v2": "data/preprocessed/IMDB_rgcn_v2",
    "v3": "data/preprocessed/IMDB_rgcn_v3",
    "v4": "data/preprocessed/IMDB_rgcn_v4",
}


def parse_variants(spec: str) -> List[str]:
    variants = [item.strip().lower() for item in spec.split(",") if item.strip()]
    if not variants or len(set(variants)) != len(variants):
        raise SystemExit("--variants must contain distinct variant names")
    unknown = set(variants) - set(BASE)
    if unknown:
        raise SystemExit(f"Unknown variants: {sorted(unknown)}")
    return variants


def load_preprocessed(base_dir: str):
    graph_data = torch.load(os.path.join(base_dir, "graph_data.pt"), map_location="cpu")
    meta = torch.load(os.path.join(base_dir, "meta.pt"), map_location="cpu")
    return graph_data, meta


def build_graph(graph_data: Mapping[str, Any], num_nodes: Mapping[str, int], variant: str):
    data = {}
    if variant == "v1":
        data[("actor", "actor-movie", "movie")] = graph_data["actor-movie"]
        data[("movie", "movie-actor", "actor")] = (graph_data["actor-movie"][1], graph_data["actor-movie"][0])
        data[("movie", "movie-link", "link")] = graph_data["movie-link"]
        data[("link", "link-movie", "movie")] = (graph_data["movie-link"][1], graph_data["movie-link"][0])
        data[("movie", "movie-director", "director")] = graph_data["movie-director"]
        data[("director", "director-movie", "movie")] = (graph_data["movie-director"][1], graph_data["movie-director"][0])
    elif variant == "v2":
        data[("actor", "actor-link", "link")] = graph_data["actor-link"]
        data[("link", "link-actor", "actor")] = (graph_data["actor-link"][1], graph_data["actor-link"][0])
        data[("link", "link-movie", "movie")] = graph_data["link-movie"]
        data[("movie", "movie-link", "link")] = (graph_data["link-movie"][1], graph_data["link-movie"][0])
        data[("link", "link-director", "director")] = graph_data["link-director"]
        data[("director", "director-link", "link")] = (graph_data["link-director"][1], graph_data["link-director"][0])
    elif variant == "v3":
        data[("actor", "actor-link", "link")] = graph_data["actor-link"]
        data[("link", "link-actor", "actor")] = (graph_data["actor-link"][1], graph_data["actor-link"][0])
        data[("link", "link-movie", "movie")] = graph_data["link-movie"]
        data[("movie", "movie-link", "link")] = (graph_data["link-movie"][1], graph_data["link-movie"][0])
        data[("movie", "movie-director", "director")] = graph_data["movie-director"]
        data[("director", "director-movie", "movie")] = (graph_data["movie-director"][1], graph_data["movie-director"][0])
    elif variant == "v4":
        data[("actor", "actor-movie", "movie")] = graph_data["actor-movie"]
        data[("movie", "movie-actor", "actor")] = (graph_data["actor-movie"][1], graph_data["actor-movie"][0])
        data[("movie", "movie-link", "link")] = graph_data["movie-link"]
        data[("link", "link-movie", "movie")] = (graph_data["movie-link"][1], graph_data["movie-link"][0])
        data[("link", "link-director", "director")] = graph_data["link-director"]
        data[("director", "director-link", "link")] = (graph_data["link-director"][1], graph_data["link-director"][0])
    else:
        raise ValueError(variant)
    return dgl.heterograph(data, num_nodes_dict={key: int(value) for key, value in num_nodes.items()})


def prepare_bundles(variants: List[str]):
    relation_ids = relation_to_id(IMDB_RELATIONS)
    bundles: Dict[str, Dict[str, Any]] = {}
    num_nodes_dicts = {}
    labels = {}
    train_indices = {}
    val_indices = {}
    test_indices = {}
    indexers_by_variant = {}

    for variant in variants:
        graph_data, meta = load_preprocessed(BASE[variant])
        num_nodes = {key: int(value) for key, value in meta["num_nodes"].items()}
        graph = build_graph(graph_data, num_nodes, variant)
        homogeneous, edge_types, indexers = to_homogeneous_with_global_relations(graph, relation_ids)
        bundle = {
            "variant": variant,
            "graph": homogeneous,
            "edge_types": edge_types,
            "indexers": indexers,
            "movie_indexer": indexers["movie"],
            "labels": meta["labels"].long(),
            "train_idx": meta["train_idx"].long(),
            "val_idx": meta["val_idx"].long(),
            "test_idx": meta["test_idx"].long(),
            "edge_count": int(homogeneous.num_edges()),
        }
        bundles[variant] = bundle
        num_nodes_dicts[variant] = num_nodes
        labels[variant] = bundle["labels"]
        train_indices[variant] = bundle["train_idx"]
        val_indices[variant] = bundle["val_idx"]
        test_indices[variant] = bundle["test_idx"]
        indexers_by_variant[variant] = indexers

    reference_variant = variants[0]
    for variant in variants[1:]:
        if num_nodes_dicts[variant] != num_nodes_dicts[reference_variant]:
            raise ValueError(f"num_nodes differs between {reference_variant} and {variant}")
        if bundles[variant]["graph"].num_nodes() != bundles[reference_variant]["graph"].num_nodes():
            raise ValueError("Homogeneous node counts differ across variants")
    assert_same_tensor("labels", labels)
    assert_same_tensor("train_idx", train_indices)
    assert_same_tensor("val_idx", val_indices)
    assert_same_tensor("test_idx", test_indices)
    assert_same_indexers(indexers_by_variant)
    return bundles


def device_bundle(cpu_bundle: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        **cpu_bundle,
        "graph": cpu_bundle["graph"].to(device),
        "edge_types": cpu_bundle["edge_types"].to(device),
        "movie_indexer": cpu_bundle["movie_indexer"].to(device),
        "labels": cpu_bundle["labels"].to(device),
        "train_idx": cpu_bundle["train_idx"].to(device),
        "val_idx": cpu_bundle["val_idx"].to(device),
        "test_idx": cpu_bundle["test_idx"].to(device),
    }


def forward_logits(encoder, classifier, bundle):
    all_embeddings = encoder(bundle["graph"], bundle["edge_types"])
    movie_embeddings = all_embeddings[bundle["movie_indexer"]]
    return classifier(movie_embeddings)


def evaluate_variant(encoder, classifier, cpu_bundle, device, split: str):
    bundle = device_bundle(cpu_bundle, device)
    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        logits = forward_logits(encoder, classifier, bundle)
        indices = bundle[f"{split}_idx"]
        loss = F.cross_entropy(logits[indices], bundle["labels"][indices]).item()
        metrics = classification_metrics(logits, bundle["labels"], indices)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return loss, metrics, logits.detach().cpu(), cpu_bundle[f"{split}_idx"].cpu()


def run_seed(args, variants: List[str], seed: int, output_root: Path) -> Dict[str, Any]:
    set_determinism(seed)
    device = resolve_device(args.device)
    bundles = prepare_bundles(variants)
    reference = bundles[variants[0]]
    num_nodes = int(reference["graph"].num_nodes())
    num_classes = int(reference["labels"].max().item()) + 1

    encoder = RGCNEncoder(
        num_nodes=num_nodes,
        num_rels=len(IMDB_RELATIONS),
        in_dim=args.in_dim,
        hid_dim=args.hid_dim,
        out_dim=args.out_dim,
        num_layers=args.layers,
        num_bases=args.num_bases,
        dropout=args.dropout,
    ).to(device)
    classifier = NodeClassifier(args.out_dim, num_classes).to(device)
    parameters = list(encoder.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)

    seed_dir = output_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = seed_dir / "shared_checkpoint.pt"
    latest_state_path = seed_dir / "latest_training_state.pt"
    early_stopper = EarlyStopper(mode="max", patience=args.patience)
    rng = np.random.RandomState(seed)
    run_config = {
        "dataset": "IMDB", "seed": seed, "variants": variants,
        "in_dim": args.in_dim, "hid_dim": args.hid_dim, "out_dim": args.out_dim,
        "layers": args.layers, "num_bases": args.num_bases, "dropout": args.dropout,
        "lr": args.lr, "weight_decay": args.weight_decay, "grad_clip": args.grad_clip,
        "patience": args.patience,
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
        latest_state_path, resume=args.resume, run_config=run_config,
        modules={"encoder": encoder, "classifier": classifier}, optimizer=optimizer,
        early_stopper=early_stopper, rng=rng, device=device,
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
            encoder.train()
            classifier.train()
            optimizer.zero_grad(set_to_none=True)

            logits = forward_logits(encoder, classifier, bundle)
            train_graph_forwards += 1
            loss = F.cross_entropy(
                logits[bundle["train_idx"]],
                bundle["labels"][bundle["train_idx"]],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            optimizer.step()

            optimizer_steps += 1
            variant_epochs += 1
            train_losses[variant] = float(loss.detach().cpu())
            del logits, loss, bundle

        val_losses: Dict[str, float] = {}
        val_metrics: Dict[str, Dict[str, float]] = {}
        for variant in variants:
            loss, metrics, _, _ = evaluate_variant(encoder, classifier, bundles[variant], device, "val")
            validation_graph_forwards += 1
            val_losses[variant] = loss
            val_metrics[variant] = metrics

        mean_val_macro_f1 = float(np.mean([val_metrics[v]["Macro_F1"] for v in variants]))
        mean_val_loss = float(np.mean(list(val_losses.values())))
        super_epochs_ran = super_epoch + 1
        improved = early_stopper.update(mean_val_macro_f1)
        if improved:
            atomic_torch_save(
                {
                    "encoder": encoder.state_dict(),
                    "classifier": classifier.state_dict(),
                    "metadata": {
                        "seed": seed,
                        "variants": variants,
                        "best_super_epoch": super_epochs_ran,
                        "optimizer_steps": optimizer_steps,
                        "variant_epochs": variant_epochs,
                        "global_relations": list(IMDB_RELATIONS),
                    },
                },
                checkpoint_path,
            )

        row: Dict[str, Any] = {
            "super_epoch": super_epochs_ran,
            "variant_order": ",".join(order),
            "variant_epochs_cumulative": variant_epochs,
            "optimizer_steps_cumulative": optimizer_steps,
            "train_graph_forwards_cumulative": train_graph_forwards,
            "validation_graph_forwards_cumulative": validation_graph_forwards,
            "mean_train_loss": float(np.mean(list(train_losses.values()))),
            "mean_val_loss": mean_val_loss,
            "mean_val_macro_f1": mean_val_macro_f1,
            "best_mean_val_macro_f1": early_stopper.best,
            "cycle_seconds": time.perf_counter() - cycle_start,
        }
        for variant in variants:
            row[f"train_loss_{variant}"] = train_losses[variant]
            row[f"val_loss_{variant}"] = val_losses[variant]
            row[f"val_macro_f1_{variant}"] = val_metrics[variant]["Macro_F1"]
            row[f"val_accuracy_{variant}"] = val_metrics[variant]["Accuracy"]
        history.append(row)
        atomic_write_csv(pd.DataFrame(history), seed_dir / "training_history.csv")
        segment_seconds = time.perf_counter() - training_start
        current_training_gpu = merge_cuda_memory_stats(
            prior_training_gpu, cuda_memory_stats(device)
        )
        current_peak_rss = max(prior_peak_rss, process_peak_rss_bytes())
        save_latest_training_state(
            latest_state_path, dataset="IMDB", run_config=run_config,
            modules={"encoder": encoder, "classifier": classifier}, optimizer=optimizer,
            early_stopper=early_stopper, rng=rng, completed_super_epoch=super_epochs_ran,
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
            f"mean_train_loss={row['mean_train_loss']:.6f} "
            f"mean_val_macro_f1={mean_val_macro_f1:.6f} "
            f"best={early_stopper.best:.6f} order={order}",
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
    classifier.load_state_dict(state["classifier"])

    reset_cuda_peak(device)
    per_variant_metrics: Dict[str, Dict[str, float]] = {}
    outputs: Dict[str, Dict[str, np.ndarray]] = {}
    test_graph_forwards = 0
    for variant in variants:
        _, metrics, logits, test_idx = evaluate_variant(encoder, classifier, bundles[variant], device, "test")
        test_graph_forwards += 1
        selected_logits = logits[test_idx]
        probabilities = torch.softmax(selected_logits, dim=1).numpy()
        predictions = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)
        item_ids = test_idx.numpy().astype(np.int64)
        labels = bundles[variant]["labels"][test_idx].numpy().astype(np.int64)

        score_frame = pd.DataFrame({"movie_id": item_ids, "label": labels, "prediction": predictions, "confidence": confidence})
        for class_id in range(probabilities.shape[1]):
            score_frame[f"prob_class_{class_id}"] = probabilities[:, class_id]
            score_frame[f"logit_class_{class_id}"] = selected_logits[:, class_id].numpy()
        score_frame.to_csv(seed_dir / f"test_scores_{variant}.csv", index=False)

        metrics.update({"edge_count": float(bundles[variant]["edge_count"])})
        per_variant_metrics[variant] = metrics
        outputs[variant] = {
            "item_id": item_ids,
            "logits": selected_logits.numpy(),
            "probabilities": probabilities,
            "prediction": predictions,
            "confidence": confidence,
        }

    inference_memory = cuda_memory_stats(device)
    invariance_rows = classification_invariance_rows(outputs)
    pd.DataFrame(invariance_rows).to_csv(seed_dir / "pairwise_invariance.csv", index=False)
    pd.DataFrame(
        [{"variant": variant, **metrics} for variant, metrics in per_variant_metrics.items()]
    ).to_csv(seed_dir / "test_metrics_by_variant.csv", index=False)

    combined_model = torch.nn.ModuleList([encoder, classifier])
    memory = model_memory_bytes(combined_model)
    summary = {
        "dataset": "IMDB",
        "seed": seed,
        "variants": variants,
        "epoch_accounting": {
            "definition": "one super-epoch is one full-batch optimizer update on every variant",
            "super_epochs_ran": super_epochs_ran,
            "variant_epochs_ran": variant_epochs,
            "optimizer_steps": optimizer_steps,
            "expected_optimizer_steps": super_epochs_ran * len(variants),
            "train_graph_forwards": train_graph_forwards,
            "validation_graph_forwards": validation_graph_forwards,
            "test_graph_forwards": test_graph_forwards,
        },
        "training_seconds": training_seconds,
        "best_mean_val_macro_f1": early_stopper.best,
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
        "global_native_relations": list(IMDB_RELATIONS),
    }
    if optimizer_steps != super_epochs_ran * len(variants):
        raise AssertionError("Incorrect full-batch epoch/update accounting")
    write_json(seed_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="IMDb RGCN joint graph-variant augmentation")
    parser.add_argument("--variants", default="v1,v2,v3,v4")
    parser.add_argument("--seeds", default="1566911444,20241017,20251017")
    parser.add_argument("--super-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--in-dim", type=int, default=128)
    parser.add_argument("--hid-dim", type=int, default=128)
    parser.add_argument("--out-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--num-bases", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true", help="Resume from latest_training_state.pt if present")
    parser.add_argument("--output-dir", default="results/rgcn_augmentation/IMDB")
    args = parser.parse_args()

    variants = parse_variants(args.variants)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = [run_seed(args, variants, seed, output_root) for seed in seeds]
    rows = []
    for summary in summaries:
        row = {
            "seed": summary["seed"],
            "super_epochs_ran": summary["epoch_accounting"]["super_epochs_ran"],
            "variant_epochs_ran": summary["epoch_accounting"]["variant_epochs_ran"],
            "optimizer_steps": summary["epoch_accounting"]["optimizer_steps"],
            "training_seconds": summary["training_seconds"],
            "best_mean_val_macro_f1": summary["best_mean_val_macro_f1"],
            **{f"mean_test_{key}": value for key, value in summary["mean_test_metrics"].items()},
        }
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_root / "seed_summary.csv", index=False)
    write_json(output_root / "all_seed_summaries.json", {"runs": summaries})
    print(f"[OK] Results written under {output_root}")


if __name__ == "__main__":
    main()
