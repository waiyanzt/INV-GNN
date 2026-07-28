#!/usr/bin/env python3
"""Joint graph-variant data augmentation for IMDb SlotGAT node classification.

One SlotGAT network, optimizer, and checkpoint are shared across IMDb v1-v4.
A super-epoch is one full-batch optimizer update on every selected graph
variant, in randomized order. Validation and early stopping happen only after
the complete balanced super-epoch.

The HGB preprocessing uses compact relation IDs whose meanings differ between
variants. This runner therefore remaps every directed relation to one global
semantic vocabulary before passing edge types to the shared SlotGAT model.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import dgl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
SLOTGAT_ROOT = SCRIPT_DIR.parent
RGCN_AUGMENTATION_DIR = (
    SLOTGAT_ROOT.parent / "RGCN" / "rgcn_data_augmentation"
)
sys.path.insert(0, str(SLOTGAT_ROOT))
sys.path.insert(0, str(RGCN_AUGMENTATION_DIR))

from model import slotGAT  # noqa: E402
from rgcn_aug_common import (  # noqa: E402
    IMDB_RELATIONS,
    EarlyStopper,
    atomic_torch_save,
    atomic_write_csv,
    checkpoint_size_bytes,
    classification_invariance_rows,
    classification_metrics,
    cuda_memory_stats,
    load_latest_training_state,
    mean_dict,
    merge_cuda_memory_stats,
    model_memory_bytes,
    process_peak_rss_bytes,
    relation_to_id,
    reset_cuda_peak,
    resolve_device,
    save_latest_training_state,
    torch_load_full,
    write_json,
)
from utils.data import load_data  # noqa: E402
from utils.tools import set_seed  # noqa: E402


NUM_CLASSES = 3
MOVIE_TYPE = 0
VARIANT_PREFIXES = {
    "v1": "IMDB_var1",
    "v2": "IMDB_var2",
    "v3": "IMDB_var3",
    "v4": "IMDB_var4",
}
NODE_TYPE_NAMES = {
    0: "movie",
    1: "director",
    2: "actor",
    3: "link",
}


def parse_variants(spec: str) -> List[str]:
    variants = []
    for raw_value in spec.split(","):
        value = raw_value.strip().lower()
        if not value:
            continue
        if value.isdigit():
            value = f"v{value}"
        variants.append(value)
    if not variants or len(set(variants)) != len(variants):
        raise SystemExit("--variants must contain distinct IMDb variants")
    unknown = set(variants) - set(VARIANT_PREFIXES)
    if unknown:
        raise SystemExit(
            f"Unknown variants: {sorted(unknown)}; choose from "
            f"{sorted(VARIANT_PREFIXES)}"
        )
    return variants


def parse_seeds(spec: str) -> List[int]:
    seeds = [int(value.strip()) for value in spec.split(",") if value.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise SystemExit("--seeds must contain distinct integer seeds")
    return seeds


def matrix_to_tensor(matrix) -> torch.Tensor:
    if isinstance(matrix, np.ndarray):
        return torch.from_numpy(matrix).float()
    coo = matrix.tocoo()
    indices = torch.from_numpy(
        np.vstack((coo.row, coo.col)).astype(np.int64, copy=False)
    )
    values = torch.from_numpy(coo.data.astype(np.float32, copy=False))
    return torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()


def apply_feats_type(
    features: Sequence[torch.Tensor], feats_type: int
) -> Tuple[List[torch.Tensor], List[int]]:
    copied = list(features)
    if feats_type == 0:
        return copied, [int(feature.shape[1]) for feature in copied]
    if feats_type == 1:
        for type_id in range(1, len(copied)):
            copied[type_id] = torch.zeros((copied[type_id].shape[0], 10))
        return (
            copied,
            [int(copied[0].shape[1])] + [10] * (len(copied) - 1),
        )
    raise ValueError(f"Unsupported --feats-type {feats_type}; use 0 or 1")


def matrices_equal(left, right) -> bool:
    if left.shape != right.shape:
        return False
    if sp.issparse(left) or sp.issparse(right):
        left_sparse = sp.csr_matrix(left)
        right_sparse = sp.csr_matrix(right)
        difference = left_sparse != right_sparse
        return difference.nnz == 0
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def semantic_relation_name(source_type: int, target_type: int) -> str:
    try:
        relation = (
            f"{NODE_TYPE_NAMES[int(source_type)]}-"
            f"{NODE_TYPE_NAMES[int(target_type)]}"
        )
    except KeyError as error:
        raise ValueError(
            f"Unknown IMDb node-type pair ({source_type}, {target_type})"
        ) from error
    if relation not in IMDB_RELATIONS:
        raise ValueError(
            f"IMDb relation {relation!r} is absent from the global vocabulary"
        )
    return relation


def build_graph(dl) -> Tuple[dgl.DGLGraph, torch.Tensor]:
    """Build the original SlotGAT topology with globally aligned edge types."""
    global_relation_ids = relation_to_id(IMDB_RELATIONS)
    source_parts: List[torch.Tensor] = []
    target_parts: List[torch.Tensor] = []
    type_parts: List[torch.Tensor] = []

    for local_relation_id in sorted(dl.links["data"]):
        source, target = dl.links["data"][local_relation_id].nonzero()
        source_tensor = torch.from_numpy(
            np.asarray(source, dtype=np.int64)
        )
        target_tensor = torch.from_numpy(
            np.asarray(target, dtype=np.int64)
        )
        source_type, target_type = dl.links["meta"][local_relation_id]
        relation_name = semantic_relation_name(source_type, target_type)
        global_id = global_relation_ids[relation_name]
        source_parts.append(source_tensor)
        target_parts.append(target_tensor)
        type_parts.append(
            torch.full(
                (source_tensor.numel(),),
                global_id,
                dtype=torch.long,
            )
        )

    if not source_parts:
        raise ValueError("IMDb graph contains no relation edges")
    source = torch.cat(source_parts)
    target = torch.cat(target_parts)
    edge_type = torch.cat(type_parts)

    # Match run_IMDB_nc.py: remove any input self-loop and add exactly one
    # structural self-loop per node. IMDb preprocessing already materializes
    # both directions of every native relation, so adjM + adjM.T does not add
    # any additional topology beyond these directed edges.
    non_self = source != target
    source = source[non_self]
    target = target[non_self]
    edge_type = edge_type[non_self]
    num_nodes = int(dl.nodes["total"])
    native_order = torch.argsort(source * num_nodes + target)
    source = source[native_order]
    target = target[native_order]
    edge_type = edge_type[native_order]
    node_ids = torch.arange(int(dl.nodes["total"]), dtype=torch.long)
    source = torch.cat((source, node_ids))
    target = torch.cat((target, node_ids))
    edge_type = torch.cat(
        (
            edge_type,
            torch.full(
                (node_ids.numel(),),
                len(IMDB_RELATIONS),
                dtype=torch.long,
            ),
        )
    )

    graph = dgl.graph(
        (source, target),
        num_nodes=num_nodes,
    )
    graph.num_ntypes = len(dl.nodes["count"])
    return graph, edge_type


def prepare_bundles(
    variants: Sequence[str], feats_type: int
) -> Tuple[Dict[str, Dict[str, Any]], List[torch.Tensor], List[int]]:
    bundles: Dict[str, Dict[str, Any]] = {}
    raw_features_by_variant: Dict[str, Sequence[Any]] = {}

    for variant in variants:
        features, _adjacency, labels, split, dl = load_data(
            prefix=VARIANT_PREFIXES[variant],
            multi_labels=False,
        )
        graph, edge_type = build_graph(dl)
        train_idx = np.sort(split["train_idx"]).astype(np.int64, copy=False)
        val_idx = np.sort(split["val_idx"]).astype(np.int64, copy=False)
        test_idx = np.sort(split["test_idx"]).astype(np.int64, copy=False)
        n_movies = int(dl.nodes["count"][MOVIE_TYPE])
        if any(
            indices.size == 0 or int(indices.max()) >= n_movies
            for indices in (train_idx, val_idx, test_idx)
        ):
            raise ValueError(
                f"{variant} contains an empty split or a non-movie split index"
            )
        if (
            np.intersect1d(train_idx, val_idx).size
            or np.intersect1d(train_idx, test_idx).size
            or np.intersect1d(val_idx, test_idx).size
        ):
            raise ValueError(f"{variant} train/validation/test splits overlap")
        labels_array = np.asarray(labels, dtype=np.int64)
        if labels_array.shape != (n_movies,):
            raise ValueError(
                f"{variant} labels have shape {labels_array.shape}; "
                f"expected ({n_movies},)"
            )
        if np.unique(labels_array).tolist() != list(range(NUM_CLASSES)):
            raise ValueError(
                f"{variant} labels are not the expected classes "
                f"0..{NUM_CLASSES - 1}"
            )

        raw_features_by_variant[variant] = features
        bundles[variant] = {
            "variant": variant,
            "graph": graph,
            "edge_type": edge_type,
            "labels": torch.from_numpy(labels_array).long(),
            "train_idx": torch.from_numpy(train_idx).long(),
            "val_idx": torch.from_numpy(val_idx).long(),
            "test_idx": torch.from_numpy(test_idx).long(),
            "node_counts": tuple(
                int(dl.nodes["count"][type_id])
                for type_id in range(len(dl.nodes["count"]))
            ),
            "edge_count": int(graph.num_edges()),
            "native_edge_count": int(dl.links["total"]),
        }

    reference_variant = variants[0]
    reference = bundles[reference_variant]
    for variant in variants[1:]:
        current = bundles[variant]
        for key in ("labels", "train_idx", "val_idx", "test_idx"):
            if not torch.equal(reference[key], current[key]):
                raise ValueError(
                    f"{key} differs between {reference_variant} and {variant}"
                )
        if reference["node_counts"] != current["node_counts"]:
            raise ValueError(
                f"Node counts differ between {reference_variant} and {variant}"
            )
        reference_features = raw_features_by_variant[reference_variant]
        current_features = raw_features_by_variant[variant]
        if len(reference_features) != len(current_features):
            raise ValueError(
                f"Feature-type count differs between {reference_variant} "
                f"and {variant}"
            )
        for type_id, (left, right) in enumerate(
            zip(reference_features, current_features)
        ):
            if not matrices_equal(left, right):
                raise ValueError(
                    f"Node features for type {type_id} differ between "
                    f"{reference_variant} and {variant}. A shared SlotGAT "
                    "requires aligned nodes and features."
                )

    features = [
        matrix_to_tensor(feature)
        for feature in raw_features_by_variant[reference_variant]
    ]
    features, in_dims = apply_feats_type(features, feats_type)
    return bundles, features, in_dims


def bundle_to_device(
    cpu_bundle: Mapping[str, Any], device: torch.device
) -> Dict[str, Any]:
    graph = cpu_bundle["graph"].to(device)
    graph.num_ntypes = cpu_bundle["graph"].num_ntypes
    return {
        **cpu_bundle,
        "graph": graph,
        "edge_type": cpu_bundle["edge_type"].to(device),
        "labels": cpu_bundle["labels"].to(device),
        "train_idx": cpu_bundle["train_idx"].to(device),
        "val_idx": cpu_bundle["val_idx"].to(device),
        "test_idx": cpu_bundle["test_idx"].to(device),
    }


def make_network(
    args,
    graph: dgl.DGLGraph,
    in_dims: Sequence[int],
    num_ntypes: int,
):
    recorder = {
        "meta": {
            "getSAAttentionScore": "False",
            "retainLayerAttention": False,
        },
        "data": {},
        "status": "None",
    }
    heads = [args.num_heads] * args.num_layers + [1]
    network = slotGAT(
        graph,
        args.edge_feats,
        len(IMDB_RELATIONS) + 1,
        list(in_dims),
        args.hidden_dim,
        NUM_CLASSES,
        args.num_layers,
        heads,
        F.elu,
        args.dropout_feat,
        args.dropout_attn,
        args.slope,
        True,
        args.alpha,
        num_ntype=num_ntypes,
        eindexer=None,
        aggregator=args.aggregator,
        SAattDim=args.sa_att_dim,
        dataRecorder=recorder,
        vis_data_saver=None,
    )
    return network, recorder


def forward_logits(
    network,
    features: Sequence[torch.Tensor],
    bundle: Mapping[str, Any],
    recorder: Dict[str, Any],
    status: str,
) -> torch.Tensor:
    network.g = bundle["graph"]
    recorder["status"] = status
    logits, _ = network(features, bundle["edge_type"])
    recorder["status"] = "None"
    return logits


def ranking_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
) -> Dict[str, float]:
    selected = logits[indices]
    truth = labels[indices]
    sorted_indices = torch.argsort(selected, dim=1, descending=True)
    ranks = (
        (sorted_indices == truth.unsqueeze(1))
        .nonzero(as_tuple=False)[:, 1]
        .float()
        + 1.0
    )
    return {
        "Hit_1": float((ranks <= 1).float().mean().item()),
        "Hit_3": float((ranks <= min(3, NUM_CLASSES)).float().mean().item()),
        "MRR": float((1.0 / ranks).mean().item()),
    }


def evaluate_variant(
    network,
    features: Sequence[torch.Tensor],
    bundle: Mapping[str, Any],
    recorder: Dict[str, Any],
    split: str,
):
    network.eval()
    with torch.no_grad():
        logits = forward_logits(
            network, features, bundle, recorder, status="Evaluation"
        )
        indices = bundle[f"{split}_idx"]
        loss = F.cross_entropy(
            logits[indices],
            bundle["labels"][indices],
        ).item()
        metrics = classification_metrics(
            logits, bundle["labels"], indices
        )
        metrics.update(
            ranking_metrics(logits, bundle["labels"], indices)
        )
    if logits.device.type == "cuda":
        torch.cuda.synchronize(logits.device)
    return (
        loss,
        metrics,
        logits.detach().cpu(),
        indices.detach().cpu(),
    )


def run_seed(
    args,
    variants: Sequence[str],
    seed: int,
    output_root: Path,
) -> Dict[str, Any]:
    set_seed(seed)
    device = resolve_device(args.device)
    cpu_bundles, cpu_features, in_dims = prepare_bundles(
        variants, args.feats_type
    )
    bundles = {
        variant: bundle_to_device(cpu_bundles[variant], device)
        for variant in variants
    }
    features = [feature.to(device) for feature in cpu_features]
    reference = bundles[variants[0]]
    network, recorder = make_network(
        args,
        reference["graph"],
        in_dims,
        len(reference["node_counts"]),
    )
    network = network.to(device)
    parameters = list(network.parameters())
    optimizer = torch.optim.Adam(
        parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    seed_dir = output_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = seed_dir / "shared_checkpoint.pt"
    latest_state_path = seed_dir / "latest_training_state.pt"
    early_stopper = EarlyStopper(mode="min", patience=args.patience)
    rng = np.random.RandomState(seed)
    run_config = {
        "dataset": "IMDB",
        "encoder": "slotgat",
        "seed": seed,
        "variants": list(variants),
        "feats_type": args.feats_type,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "edge_feats": args.edge_feats,
        "dropout_feat": args.dropout_feat,
        "dropout_attn": args.dropout_attn,
        "slope": args.slope,
        "alpha": args.alpha,
        "aggregator": args.aggregator,
        "sa_att_dim": args.sa_att_dim,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "patience": args.patience,
        "global_relations": list(IMDB_RELATIONS),
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
        modules={"network": network},
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
        validation_graph_forwards = int(
            counters["validation_graph_forwards"]
        )
        super_epochs_ran = int(resume_state["completed_super_epoch"])
        prior_training_seconds = float(
            resume_state["training_seconds_elapsed"]
        )
        prior_peak_rss = int(
            resume_state.get("process_peak_rss_bytes", 0)
        )
        prior_training_gpu = dict(
            resume_state.get("training_gpu", {})
        )

    reset_cuda_peak(device)
    training_start = time.perf_counter()

    for super_epoch in range(super_epochs_ran, args.super_epochs):
        if early_stopper.should_stop:
            print(
                "[resume] This seed already satisfied early stopping; "
                "skipping further training.",
                flush=True,
            )
            break
        cycle_start = time.perf_counter()
        order = [
            variants[index]
            for index in rng.permutation(len(variants))
        ]
        train_losses: Dict[str, float] = {}

        for variant in order:
            bundle = bundles[variant]
            network.train()
            optimizer.zero_grad(set_to_none=True)
            logits = forward_logits(
                network,
                features,
                bundle,
                recorder,
                status="Training",
            )
            train_graph_forwards += 1
            loss = F.cross_entropy(
                logits[bundle["train_idx"]],
                bundle["labels"][bundle["train_idx"]],
            )
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    parameters, args.grad_clip
                )
            optimizer.step()

            optimizer_steps += 1
            variant_epochs += 1
            train_losses[variant] = float(loss.detach().cpu())
            del logits, loss

        val_losses: Dict[str, float] = {}
        val_metrics: Dict[str, Dict[str, float]] = {}
        for variant in variants:
            loss, metrics, _, _ = evaluate_variant(
                network,
                features,
                bundles[variant],
                recorder,
                "val",
            )
            validation_graph_forwards += 1
            val_losses[variant] = loss
            val_metrics[variant] = metrics

        mean_val_loss = float(np.mean(list(val_losses.values())))
        mean_val_macro_f1 = float(
            np.mean(
                [
                    val_metrics[variant]["Macro_F1"]
                    for variant in variants
                ]
            )
        )
        super_epochs_ran = super_epoch + 1
        improved = early_stopper.update(mean_val_loss)
        if improved:
            atomic_torch_save(
                {
                    "network": network.state_dict(),
                    "metadata": {
                        "seed": seed,
                        "variants": list(variants),
                        "best_super_epoch": super_epochs_ran,
                        "optimizer_steps": optimizer_steps,
                        "variant_epochs": variant_epochs,
                        "best_mean_val_loss": early_stopper.best,
                        "global_relations": list(IMDB_RELATIONS),
                        "structural_self_loop_edge_type": len(
                            IMDB_RELATIONS
                        ),
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
            "validation_graph_forwards_cumulative": (
                validation_graph_forwards
            ),
            "mean_train_loss": float(
                np.mean(list(train_losses.values()))
            ),
            "mean_val_loss": mean_val_loss,
            "mean_val_macro_f1": mean_val_macro_f1,
            "best_mean_val_loss": early_stopper.best,
            "cycle_seconds": time.perf_counter() - cycle_start,
        }
        for variant in variants:
            row[f"train_loss_{variant}"] = train_losses[variant]
            row[f"val_loss_{variant}"] = val_losses[variant]
            row[f"val_macro_f1_{variant}"] = (
                val_metrics[variant]["Macro_F1"]
            )
            row[f"val_accuracy_{variant}"] = (
                val_metrics[variant]["Accuracy"]
            )
        history.append(row)
        atomic_write_csv(
            pd.DataFrame(history),
            seed_dir / "training_history.csv",
        )
        segment_seconds = time.perf_counter() - training_start
        current_training_gpu = merge_cuda_memory_stats(
            prior_training_gpu,
            cuda_memory_stats(device),
        )
        current_peak_rss = max(
            prior_peak_rss, process_peak_rss_bytes()
        )
        save_latest_training_state(
            latest_state_path,
            dataset="IMDB",
            run_config=run_config,
            modules={"network": network},
            optimizer=optimizer,
            early_stopper=early_stopper,
            rng=rng,
            completed_super_epoch=super_epochs_ran,
            counters={
                "optimizer_steps": optimizer_steps,
                "variant_epochs": variant_epochs,
                "train_graph_forwards": train_graph_forwards,
                "validation_graph_forwards": (
                    validation_graph_forwards
                ),
            },
            history=history,
            training_seconds_elapsed=(
                prior_training_seconds + segment_seconds
            ),
            process_peak_rss_bytes_value=current_peak_rss,
            training_gpu=current_training_gpu,
        )

        print(
            f"seed={seed} super_epoch={super_epochs_ran:03d} "
            f"variant_epochs={variant_epochs} "
            f"optimizer_steps={optimizer_steps} "
            f"mean_train_loss={row['mean_train_loss']:.6f} "
            f"mean_val_loss={mean_val_loss:.6f} "
            f"best={early_stopper.best:.6f} order={order}",
            flush=True,
        )
        if early_stopper.should_stop:
            print(
                "Early stopping after a complete balanced super-epoch.",
                flush=True,
            )
            break

    training_seconds = prior_training_seconds + (
        time.perf_counter() - training_start
    )
    training_memory = merge_cuda_memory_stats(
        prior_training_gpu, cuda_memory_stats(device)
    )
    cumulative_peak_rss = max(
        prior_peak_rss, process_peak_rss_bytes()
    )
    if not checkpoint_path.exists():
        raise RuntimeError("No shared checkpoint was saved")
    checkpoint = torch_load_full(
        checkpoint_path, map_location=device
    )
    network.load_state_dict(checkpoint["network"])

    reset_cuda_peak(device)
    per_variant_metrics: Dict[str, Dict[str, float]] = {}
    outputs: Dict[str, Dict[str, np.ndarray]] = {}
    test_graph_forwards = 0
    for variant in variants:
        _, metrics, logits, test_idx = evaluate_variant(
            network,
            features,
            bundles[variant],
            recorder,
            "test",
        )
        test_graph_forwards += 1
        selected_logits = logits[test_idx]
        probabilities = torch.softmax(
            selected_logits, dim=1
        ).numpy()
        predictions = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)
        item_ids = test_idx.numpy().astype(np.int64)
        labels = (
            cpu_bundles[variant]["labels"][test_idx]
            .numpy()
            .astype(np.int64)
        )

        score_frame = pd.DataFrame(
            {
                "movie_id": item_ids,
                "label": labels,
                "prediction": predictions,
                "confidence": confidence,
            }
        )
        for class_id in range(probabilities.shape[1]):
            score_frame[f"prob_class_{class_id}"] = (
                probabilities[:, class_id]
            )
            score_frame[f"logit_class_{class_id}"] = (
                selected_logits[:, class_id].numpy()
            )
        atomic_write_csv(
            score_frame,
            seed_dir / f"test_scores_{variant}.csv",
        )

        metrics.update(
            {
                "edge_count": float(
                    bundles[variant]["edge_count"]
                ),
                "native_edge_count": float(
                    bundles[variant]["native_edge_count"]
                ),
            }
        )
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
    atomic_write_csv(
        pd.DataFrame(invariance_rows),
        seed_dir / "pairwise_invariance.csv",
    )
    atomic_write_csv(
        pd.DataFrame(
            [
                {"variant": variant, **metrics}
                for variant, metrics in per_variant_metrics.items()
            ]
        ),
        seed_dir / "test_metrics_by_variant.csv",
    )

    summary = {
        "dataset": "IMDB",
        "encoder": "SlotGAT",
        "seed": seed,
        "variants": list(variants),
        "epoch_accounting": {
            "definition": (
                "one super-epoch is one full-batch optimizer update "
                "on every selected variant"
            ),
            "super_epochs_ran": super_epochs_ran,
            "variant_epochs_ran": variant_epochs,
            "optimizer_steps": optimizer_steps,
            "expected_optimizer_steps": (
                super_epochs_ran * len(variants)
            ),
            "train_graph_forwards": train_graph_forwards,
            "validation_graph_forwards": (
                validation_graph_forwards
            ),
            "test_graph_forwards": test_graph_forwards,
        },
        "training_seconds": training_seconds,
        "best_mean_val_loss": early_stopper.best,
        "mean_test_metrics": mean_dict(
            list(per_variant_metrics.values())
        ),
        "per_variant_test_metrics": per_variant_metrics,
        "pairwise_invariance": invariance_rows,
        "memory": {
            **model_memory_bytes(network),
            "checkpoint_bytes": checkpoint_size_bytes(
                checkpoint_path
            ),
            "process_peak_rss_bytes": cumulative_peak_rss,
            "training_gpu": training_memory,
            "inference_gpu": inference_memory,
        },
        "global_native_relations": list(IMDB_RELATIONS),
        "structural_self_loop_edge_type": len(IMDB_RELATIONS),
        "features_type": args.feats_type,
        "input_dimensions": in_dims,
    }
    if optimizer_steps != super_epochs_ran * len(variants):
        raise AssertionError(
            "Incorrect full-batch super-epoch/update accounting"
        )
    write_json(seed_dir / "summary.json", summary)

    del network, optimizer, bundles, features, cpu_bundles, cpu_features
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "IMDb SlotGAT joint graph-variant augmentation "
            "(movie node classification)"
        )
    )
    parser.add_argument("--variants", default="v1,v2,v3,v4")
    parser.add_argument(
        "--seeds",
        default="1566911444,20241017,20251017",
    )
    parser.add_argument("--super-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--feats-type", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--edge-feats", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--dropout-feat", type=float, default=0.5)
    parser.add_argument("--dropout-attn", type=float, default=0.2)
    parser.add_argument("--slope", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--aggregator",
        default="SA",
        choices=[
            "SA",
            "average",
            "last_fc",
            "max",
            "onedimconv",
        ],
    )
    parser.add_argument("--sa-att-dim", type=int, default=3)
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=0.0,
        help="Maximum gradient norm; zero disables clipping",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume exactly from latest_training_state.pt if present"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "results/slotgat_augmentation/IMDB"
        ),
    )
    return parser


def validate_args(args) -> None:
    positive_integer_fields = (
        "super_epochs",
        "patience",
        "hidden_dim",
        "num_heads",
        "num_layers",
        "edge_feats",
        "sa_att_dim",
    )
    for name in positive_integer_fields:
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.feats_type not in (0, 1):
        raise SystemExit("--feats-type must be 0 or 1")
    if args.lr <= 0:
        raise SystemExit("--lr must be positive")
    if args.weight_decay < 0 or args.grad_clip < 0:
        raise SystemExit(
            "--weight-decay and --grad-clip must be nonnegative"
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    variants = parse_variants(args.variants)
    seeds = parse_seeds(args.seeds)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = [
        run_seed(args, variants, seed, output_root)
        for seed in seeds
    ]
    rows = []
    for summary in summaries:
        rows.append(
            {
                "seed": summary["seed"],
                "super_epochs_ran": (
                    summary["epoch_accounting"]["super_epochs_ran"]
                ),
                "variant_epochs_ran": (
                    summary["epoch_accounting"]["variant_epochs_ran"]
                ),
                "optimizer_steps": (
                    summary["epoch_accounting"]["optimizer_steps"]
                ),
                "training_seconds": summary["training_seconds"],
                "best_mean_val_loss": (
                    summary["best_mean_val_loss"]
                ),
                **{
                    f"mean_test_{key}": value
                    for key, value in summary[
                        "mean_test_metrics"
                    ].items()
                },
            }
        )
    atomic_write_csv(
        pd.DataFrame(rows),
        output_root / "seed_summary.csv",
    )
    write_json(
        output_root / "all_seed_summaries.json",
        {"runs": summaries},
    )
    print(f"[OK] Results written under {output_root}")


if __name__ == "__main__":
    main()
