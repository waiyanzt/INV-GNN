#!/usr/bin/env python3
"""Joint graph-variant augmentation for Freebase RGCN and SlotGAT encoders.

One model and optimizer are shared across graph variants. A super-epoch visits
every selected variant once, and mean validation NLL selects the checkpoint.
The RGCN defaults retain the legacy model exactly; SlotGAT uses the supplied
Freebase node-classification architecture with globally aligned edge types.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from rgcn_aug_common import (
    EarlyStopper,
    assert_same_tensor,
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
    reset_cuda_peak,
    resolve_device,
    save_latest_training_state,
    set_determinism,
    torch_load_full,
    write_json,
)

SLOTGAT_ROOT = Path(__file__).resolve().parents[2] / "SlotGAT"


def parse_csv(value: str) -> List[str]:
    out = [item.strip() for item in value.split(",") if item.strip()]
    if not out or len(out) != len(set(out)):
        raise SystemExit("Expected a nonempty comma-separated list without duplicates")
    return out


def load_bundle(root: Path, variant: str) -> Dict[str, Any]:
    path = root / variant / "rgcn_data.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run preprocess_FREEBASE_rgcn_augmentation.py first."
        )
    payload = torch.load(path, map_location="cpu")
    return {
        "variant": variant,
        "edge_index": payload["edge_index"].long(),
        "edge_type": payload["edge_type"].long(),
        "labels": payload["y"].long(),
        "train_idx": payload["train_mask"].nonzero(as_tuple=False).view(-1).long(),
        "val_idx": payload["val_mask"].nonzero(as_tuple=False).view(-1).long(),
        "test_idx": payload["test_mask"].nonzero(as_tuple=False).view(-1).long(),
        "node_type": payload["node_type"].long(),
        "num_nodes": int(payload["num_nodes"]),
        "num_relations": int(payload["num_relations"]),
        "num_classes": int(payload["num_classes"]),
        "edge_count": int(payload["edge_index"].shape[1]),
    }


def prepare_bundles(data_root: Path, variants: List[str]) -> Dict[str, Dict[str, Any]]:
    bundles = {variant: load_bundle(data_root, variant) for variant in variants}
    for key in ("labels", "train_idx", "val_idx", "test_idx", "node_type"):
        assert_same_tensor(key, {variant: bundle[key] for variant, bundle in bundles.items()})
    reference = bundles[variants[0]]
    for variant, bundle in bundles.items():
        for key in ("num_nodes", "num_relations", "num_classes"):
            if bundle[key] != reference[key]:
                raise ValueError(
                    f"{key} differs between {variants[0]} and {variant}: "
                    f"{reference[key]} vs {bundle[key]}"
                )
    return bundles


def device_bundle(bundle: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        **bundle,
        "edge_index": bundle["edge_index"].to(device),
        "edge_type": bundle["edge_type"].to(device),
        "labels": bundle["labels"].to(device),
        "train_idx": bundle["train_idx"].to(device),
        "val_idx": bundle["val_idx"].to(device),
        "test_idx": bundle["test_idx"].to(device),
    }


def index_batches(indices: torch.Tensor, batch_size: int, rng: np.random.RandomState):
    order = indices.cpu().numpy().copy()
    rng.shuffle(order)
    if batch_size <= 0:
        yield torch.from_numpy(order).long()
        return
    for start in range(0, len(order), batch_size):
        yield torch.from_numpy(order[start : start + batch_size]).long()


def classification_ranking_metrics(
    log_probabilities: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
) -> Dict[str, float]:
    selected = log_probabilities[indices]
    true_labels = labels[indices].long()
    top_k = min(3, selected.shape[1])
    top = torch.topk(selected, k=top_k, dim=1).indices
    sorted_classes = torch.argsort(selected, dim=1, descending=True)
    ranks = (
        (sorted_classes == true_labels.unsqueeze(1))
        .nonzero(as_tuple=False)[:, 1]
        .float()
        + 1.0
    )
    return {
        "Hits@1": float((top[:, 0] == true_labels).float().mean().cpu()),
        "Hits@3": float(
            (top == true_labels.unsqueeze(1)).any(dim=1).float().mean().cpu()
        ),
        "MRR": float((1.0 / ranks).mean().cpu()),
    }


@torch.no_grad()
def evaluate_variant(model, cpu_bundle, device, split: str, prepared_bundle=None):
    bundle = (
        prepared_bundle
        if prepared_bundle is not None
        else device_bundle(cpu_bundle, device)
    )
    model.eval()
    log_probabilities = model(bundle["edge_index"], bundle["edge_type"])
    indices = bundle[f"{split}_idx"]
    loss = float(F.nll_loss(log_probabilities[indices], bundle["labels"][indices]).cpu())
    metrics = classification_metrics(log_probabilities, bundle["labels"], indices)
    metrics.update(
        classification_ranking_metrics(
            log_probabilities, bundle["labels"], indices
        )
    )
    return loss, metrics, log_probabilities.detach().cpu(), cpu_bundle[f"{split}_idx"].cpu()


def build_model(args, reference: Mapping[str, Any], device: torch.device):
    if args.encoder == "rgcn":
        from model_RGCN_freebase_nc import RGCNFeatureless

        return RGCNFeatureless(
            num_nodes=reference["num_nodes"],
            num_relations=reference["num_relations"],
            hidden_dim=args.hidden_dim,
            num_classes=reference["num_classes"],
            num_bases=args.num_bases,
            dropout=args.dropout,
            edge_chunk_size=args.edge_chunk_size,
        ).to(device)

    slotgat_root = str(SLOTGAT_ROOT)
    if slotgat_root not in sys.path:
        sys.path.insert(0, slotgat_root)
    from model_slotgat_nc_freebase import FreebaseSlotGATClassifier

    return FreebaseSlotGATClassifier(
        node_type=reference["node_type"],
        num_relations=reference["num_relations"],
        num_classes=reference["num_classes"],
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


def model_name(args) -> str:
    if args.encoder == "slotgat":
        return "FreebaseSlotGATClassifier"
    return (
        "legacy_RGCNFeatureless_chunked_recompute_v1"
        if args.edge_chunk_size > 0
        else "legacy_RGCNFeatureless_pyg"
    )


def run_seed(args, variants: List[str], seed: int, output_root: Path) -> Dict[str, Any]:
    set_determinism(seed)
    if args.encoder == "slotgat":
        torch.use_deterministic_algorithms(True, warn_only=True)
    device = resolve_device(args.device)
    bundles = prepare_bundles(Path(args.data_root), variants)
    reference = bundles[variants[0]]

    model = build_model(args, reference, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    seed_dir = output_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = seed_dir / "shared_checkpoint.pt"
    latest_state_path = seed_dir / "latest_training_state.pt"
    # The legacy runner selected the minimum validation NLL.
    early_stopper = EarlyStopper(mode="min", patience=args.patience)
    rng = np.random.RandomState(seed)

    n_train = len(reference["train_idx"])
    batches_per_variant = 1 if args.label_batch_size <= 0 else int(
        np.ceil(n_train / args.label_batch_size)
    )
    updates_per_super_epoch = len(variants) * batches_per_variant
    run_config = {
        "dataset": "FREEBASE",
        "model": model_name(args),
        "seed": seed,
        "variants": variants,
        "hidden_dim": args.hidden_dim,
        "num_bases": args.num_bases,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "label_batch_size": args.label_batch_size,
        "edge_chunk_size": args.edge_chunk_size,
        "chunked_backend_version": (
            "recompute_v1" if args.edge_chunk_size > 0 else None
        ),
        "patience": args.patience,
        "data_root": str(Path(args.data_root).resolve()),
    }
    if args.encoder == "slotgat":
        # Preserve the historical RGCN resume configuration when the default
        # encoder is used.
        for key in (
            "num_bases",
            "dropout",
            "edge_chunk_size",
            "chunked_backend_version",
        ):
            run_config.pop(key)
        run_config.update(
            {
                "encoder": "slotgat",
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "edge_feats": args.edge_feats,
                "dropout_feat": args.dropout_feat,
                "dropout_attn": args.dropout_attn,
                "slope": args.slope,
                "alpha": args.alpha,
                "aggregator": args.aggregator,
                "sa_att_dim": args.sa_att_dim,
                "slotgat_edge_chunk_size": args.slotgat_edge_chunk_size,
                "slotgat_decomposed_layers": args.slotgat_decomposed_layers,
                "relation_layout": "global_forward_reverse_plus_structural_self",
            }
        )
    history: List[Dict[str, Any]] = []
    super_epochs_ran = variant_epochs = optimizer_steps = 0
    train_graph_forwards = validation_graph_forwards = 0
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
        super_epochs_ran = int(resume_state["completed_super_epoch"])
        prior_training_seconds = float(resume_state["training_seconds_elapsed"])
        prior_peak_rss = int(resume_state.get("process_peak_rss_bytes", 0))
        prior_training_gpu = dict(resume_state.get("training_gpu", {}))

    # SlotGAT caches DGL graphs by tensor identity, so keep one stable device
    # tensor bundle per variant. RGCN retains its legacy on-demand transfers.
    prepared_bundles = (
        {
            variant: device_bundle(bundle, device)
            for variant, bundle in bundles.items()
        }
        if args.encoder == "slotgat"
        else {}
    )

    reset_cuda_peak(device)
    training_start = time.perf_counter()
    for super_epoch in range(super_epochs_ran, args.super_epochs):
        if early_stopper.should_stop:
            print("[resume] Early stopping was already reached; skipping training.", flush=True)
            break

        order = [variants[i] for i in rng.permutation(len(variants))]
        cycle_start = time.perf_counter()
        train_losses: Dict[str, float] = {}

        for variant in order:
            bundle = (
                prepared_bundles[variant]
                if args.encoder == "slotgat"
                else device_bundle(bundles[variant], device)
            )
            losses = []
            for batch_cpu in index_batches(
                bundle["train_idx"].cpu(), args.label_batch_size, rng
            ):
                batch = batch_cpu.to(device)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                log_probabilities = model(bundle["edge_index"], bundle["edge_type"])
                train_graph_forwards += 1
                loss = F.nll_loss(
                    log_probabilities[batch], bundle["labels"][batch]
                )
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer_steps += 1
                losses.append(float(loss.detach().cpu()))
            variant_epochs += 1
            train_losses[variant] = float(np.mean(losses))

        val_losses: Dict[str, float] = {}
        val_metrics: Dict[str, Dict[str, float]] = {}
        for variant in variants:
            val_loss, metrics, _, _ = evaluate_variant(
                model,
                bundles[variant],
                device,
                "val",
                prepared_bundles.get(variant),
            )
            validation_graph_forwards += 1
            val_losses[variant] = val_loss
            val_metrics[variant] = metrics

        super_epochs_ran = super_epoch + 1
        mean_val_loss = float(np.mean(list(val_losses.values())))
        improved = early_stopper.update(mean_val_loss)
        if improved:
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "metadata": {
                        "seed": seed,
                        "variants": variants,
                        "best_super_epoch": super_epochs_ran,
                        "optimizer_steps": optimizer_steps,
                        "selection_metric": "mean_validation_nll",
                    },
                },
                checkpoint_path,
            )

        row = {
            "super_epoch": super_epochs_ran,
            "variant_order": ",".join(order),
            "variant_epochs_cumulative": variant_epochs,
            "optimizer_steps_cumulative": optimizer_steps,
            "batches_per_variant": batches_per_variant,
            "updates_per_super_epoch": updates_per_super_epoch,
            "mean_train_loss": float(np.mean(list(train_losses.values()))),
            "mean_val_loss": mean_val_loss,
            "best_mean_val_loss": early_stopper.best,
            "mean_val_macro_f1": float(
                np.mean([val_metrics[v]["Macro_F1"] for v in variants])
            ),
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
            latest_state_path,
            dataset="FREEBASE",
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
            },
            history=history,
            training_seconds_elapsed=prior_training_seconds + segment_seconds,
            process_peak_rss_bytes_value=current_peak_rss,
            training_gpu=current_training_gpu,
        )
        print(
            f"seed={seed} super_epoch={super_epochs_ran:03d} "
            f"optimizer_steps={optimizer_steps} mean_val_loss={mean_val_loss:.6f} "
            f"mean_val_macro_f1={row['mean_val_macro_f1']:.6f} order={order}",
            flush=True,
        )
        if early_stopper.should_stop:
            break

    training_seconds = prior_training_seconds + (time.perf_counter() - training_start)
    training_memory = merge_cuda_memory_stats(
        prior_training_gpu, cuda_memory_stats(device)
    )
    cumulative_peak_rss = max(prior_peak_rss, process_peak_rss_bytes())
    if not checkpoint_path.exists():
        raise RuntimeError("No best checkpoint was saved")
    model.load_state_dict(torch_load_full(checkpoint_path, map_location=device)["model"])

    reset_cuda_peak(device)
    per_variant_metrics: Dict[str, Dict[str, float]] = {}
    outputs: Dict[str, Dict[str, np.ndarray]] = {}
    test_graph_forwards = 0
    for variant in variants:
        test_loss, metrics, log_probabilities, test_idx = evaluate_variant(
            model,
            bundles[variant],
            device,
            "test",
            prepared_bundles.get(variant),
        )
        test_graph_forwards += 1
        selected = log_probabilities[test_idx]
        probabilities = selected.exp().numpy()
        predictions = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)
        item_ids = test_idx.numpy().astype(np.int64)
        labels = bundles[variant]["labels"][test_idx].numpy().astype(np.int64)
        frame = pd.DataFrame(
            {
                "node_id": item_ids,
                "label": labels,
                "prediction": predictions,
                "confidence": confidence,
            }
        )
        for class_id in range(probabilities.shape[1]):
            frame[f"prob_class_{class_id}"] = probabilities[:, class_id]
            frame[f"log_probability_class_{class_id}"] = selected[:, class_id].numpy()
        frame.to_csv(seed_dir / f"test_scores_{variant}.csv", index=False)
        metrics["NLL"] = test_loss
        metrics["edge_count"] = float(bundles[variant]["edge_count"])
        per_variant_metrics[variant] = metrics
        outputs[variant] = {
            "item_id": item_ids,
            "logits": selected.numpy(),
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

    expected_steps = super_epochs_ran * updates_per_super_epoch
    expected_variant_epochs = super_epochs_ran * len(variants)
    if optimizer_steps != expected_steps or variant_epochs != expected_variant_epochs:
        raise AssertionError(
            f"Epoch accounting mismatch: steps {optimizer_steps}/{expected_steps}, "
            f"variant epochs {variant_epochs}/{expected_variant_epochs}"
        )

    summary = {
        "dataset": "FREEBASE",
        "model": run_config["model"],
        "aggregation_backend": (
            "chunked_recompute_exact_slotgat_attention"
            if args.encoder == "slotgat"
            else (
                "chunked_recompute_exact_mean"
                if args.edge_chunk_size > 0
                else "pyg_rgcn_conv"
            )
        ),
        "edge_chunk_size": (
            args.slotgat_edge_chunk_size
            if args.encoder == "slotgat"
            else args.edge_chunk_size
        ),
        "decomposed_layers": (
            args.slotgat_decomposed_layers
            if args.encoder == "slotgat"
            else None
        ),
        "encoder": args.encoder,
        "seed": seed,
        "variants": variants,
        "epoch_accounting": {
            "definition": "one super-epoch visits every variant; each label batch performs one complete graph forward and one optimizer update",
            "super_epochs_ran": super_epochs_ran,
            "variant_epochs_ran": variant_epochs,
            "batches_per_variant": batches_per_variant,
            "updates_per_super_epoch": updates_per_super_epoch,
            "optimizer_steps": optimizer_steps,
            "expected_optimizer_steps": expected_steps,
            "train_graph_forwards": train_graph_forwards,
            "validation_graph_forwards": validation_graph_forwards,
            "test_graph_forwards": test_graph_forwards,
        },
        "training_seconds": training_seconds,
        "selection_metric": "mean_validation_nll",
        "best_mean_val_loss": early_stopper.best,
        "mean_test_metrics": mean_dict(list(per_variant_metrics.values())),
        "per_variant_test_metrics": per_variant_metrics,
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
        description="Freebase joint graph-variant augmentation for RGCN or SlotGAT"
    )
    parser.add_argument("--encoder", choices=("rgcn", "slotgat"), default="rgcn")
    parser.add_argument("--variants", default="unchanged,exact_2")
    parser.add_argument("--seeds", default="1566911444,20241017,20251017")
    parser.add_argument("--data-root", default="data/rgcn_augmentation/freebase")
    parser.add_argument("--output-dir", default="results/rgcn_augmentation/FREEBASE")
    parser.add_argument("--super-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument(
        "--label-batch-size", type=int, default=0, help="0 = legacy full-batch loss"
    )
    parser.add_argument(
        "--edge-chunk-size",
        type=int,
        default=0,
        help=(
            "Maximum relation edges gathered at once; 0 uses the original PyG "
            "RGCNConv, while a positive value uses exact chunked mean aggregation "
            "with a memory-bounded recomputing backward pass"
        ),
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-bases", type=int, default=30)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--edge-feats", type=int, default=64)
    parser.add_argument("--dropout-feat", type=float, default=0.5)
    parser.add_argument("--dropout-attn", type=float, default=0.2)
    parser.add_argument("--slope", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--aggregator",
        choices=("SA", "average", "last_fc", "max", "onedimconv"),
        default="SA",
    )
    parser.add_argument("--sa-att-dim", type=int, default=3)
    parser.add_argument(
        "--slotgat-edge-chunk-size",
        type=int,
        default=250000,
        help=(
            "Maximum edges processed at once by exact recomputing SlotGAT "
            "attention; only used with --encoder slotgat"
        ),
    )
    parser.add_argument(
        "--slotgat-decomposed-layers",
        type=int,
        default=4,
        help=(
            "Number of feature slices used inside each chunked SlotGAT "
            "message aggregation"
        ),
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument(
        "--grad-clip", type=float, default=0.0, help="0 disables clipping (legacy default)"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.edge_chunk_size < 0:
        raise SystemExit("--edge-chunk-size must be 0 or a positive integer")
    if args.encoder == "slotgat" and (
        args.num_layers <= 0
        or args.num_heads <= 0
        or args.edge_feats < 0
        or args.slotgat_edge_chunk_size <= 0
        or args.slotgat_decomposed_layers <= 0
    ):
        raise SystemExit(
            "SlotGAT requires positive --num-layers/--num-heads and "
            "--slotgat-edge-chunk-size/--slotgat-decomposed-layers, plus "
            "nonnegative --edge-feats"
        )

    variants = parse_csv(args.variants)
    seeds = [int(value) for value in parse_csv(args.seeds)]
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = [run_seed(args, variants, seed, output_root) for seed in seeds]
    rows = []
    for summary in summaries:
        rows.append(
            {
                "seed": summary["seed"],
                "aggregation_backend": summary["aggregation_backend"],
                "edge_chunk_size": summary["edge_chunk_size"],
                "decomposed_layers": summary["decomposed_layers"],
                "super_epochs_ran": summary["epoch_accounting"]["super_epochs_ran"],
                "variant_epochs_ran": summary["epoch_accounting"]["variant_epochs_ran"],
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
    print(f"[OK] Results written under {output_root}")


if __name__ == "__main__":
    main()
