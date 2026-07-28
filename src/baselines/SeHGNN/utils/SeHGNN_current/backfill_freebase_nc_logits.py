#!/usr/bin/env python3
"""Backfill SeHGNN Freebase logits and memory metadata from saved checkpoints.

The utility is designed for result JSON files created by both older and newer
versions of ``run_freebase_magnn_channels.py``.  It can recover, without full
retraining:

* raw train/validation/test logits;
* checkpoint file size;
* exact reconstructed-model parameter and buffer memory;
* peak inference GPU allocation by replaying inference; and
* a representative peak training GPU allocation by replaying one full-size
  optimizer step followed by one validation batch.

The replayed training peak is not the historical peak from the original run.
It is a new measurement under the current hardware/software environment using
the saved best checkpoint and the original batch-size hyperparameters.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

# Import as a module so older runners that do not export newly-added helper names
# remain usable through compatibility fallbacks below.
import run_freebase_magnn_channels as runner


def resolve_existing_path(raw: str, result_json: Path, data_dir: Path) -> Path:
    path = Path(raw).expanduser()
    candidates = [
        path,
        result_json.parent / path,
        result_json.parent / path.name,
        data_dir / path,
        data_dir / path.name,
    ]
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve(strict=False)
        if str(candidate) in seen:
            continue
        seen.add(str(candidate))
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not locate {raw!r}; checked: " + ", ".join(str(x) for x in candidates)
    )


def bytes_to_mib(num_bytes: int | float) -> float:
    return float(num_bytes) / (1024.0 ** 2)


def clear_cuda(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def _config_get(config: Mapping[str, Any], name: str, default: Any) -> Any:
    return config.get(name, default)


def build_sehgnn_model_compat(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    feature_keys: Sequence[str],
) -> torch.nn.Module:
    """Reconstruct the exact training architecture with old/new-runner support."""
    helper = getattr(runner, "build_sehgnn_model", None)
    if callable(helper):
        return helper(config, manifest, feature_keys)

    sehgnn_cls = getattr(runner, "SeHGNN", None)
    if sehgnn_cls is None:
        importer = getattr(runner, "import_sehgnn", None)
        if callable(importer):
            sehgnn_cls = importer()
    if sehgnn_cls is None:
        raise ImportError(
            "The installed run_freebase_magnn_channels.py exposes neither "
            "build_sehgnn_model nor SeHGNN/import_sehgnn. Replace it with the "
            "updated runner or run this file from src/baselines/SeHGNN."
        )

    target_type = int(manifest["target_type"])
    node_counts = {int(k): int(v) for k, v in manifest["node_counts"].items()}
    return sehgnn_cls(
        dataset="Freebase",
        nfeat=int(_config_get(config, "embed_size", 512)),
        hidden=int(_config_get(config, "hidden", 512)),
        nclass=int(manifest["num_classes"]),
        feat_keys=feature_keys,
        label_feat_keys=[],
        tgt_type=str(target_type),
        dropout=float(_config_get(config, "dropout", 0.5)),
        input_drop=float(_config_get(config, "input_drop", 0.0)),
        att_drop=float(_config_get(config, "att_drop", 0.0)),
        n_fp_layers=int(_config_get(config, "n_fp_layers", 2)),
        n_task_layers=int(_config_get(config, "n_task_layers", 4)),
        act=str(_config_get(config, "act", "none")),
        residual=bool(_config_get(config, "residual", False)),
        data_size=node_counts,
        num_heads=int(_config_get(config, "num_heads", 1)),
    )


def extract_model_state_compat(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    helper = getattr(runner, "extract_model_state", None)
    if callable(helper):
        return helper(checkpoint)
    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "state_dict", "model", "net", "network"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise KeyError("Could not locate a model state_dict in checkpoint")


def batch_feature_dict_compat(
    features: Mapping[str, Any], batch_cpu: torch.Tensor, device: torch.device
) -> Mapping[str, Any]:
    helper = getattr(runner, "batch_feature_dict", None)
    if not callable(helper):
        raise ImportError(
            "The installed runner does not expose batch_feature_dict, which is "
            "required to replay SeHGNN batches. Replace the runner with the updated version."
        )
    return helper(features, batch_cpu, device)


def forward_model(
    model: torch.nn.Module,
    features: Mapping[str, Any],
    batch_cpu: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    batch_feats = batch_feature_dict_compat(features, batch_cpu, device)
    output = model(batch_cpu.to(device), batch_feats, {}, None)
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not torch.is_tensor(output):
        raise TypeError(f"Expected tensor logits, got {type(output)!r}")
    return output


def predict_indices_compat(
    model: torch.nn.Module,
    features: Mapping[str, Any],
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    helper = getattr(runner, "predict_indices", None)
    if callable(helper):
        return helper(model, features, indices, batch_size, device)

    outputs = []
    loader = DataLoader(
        torch.as_tensor(indices, dtype=torch.long),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    model.eval()
    with torch.no_grad():
        for batch_cpu in loader:
            outputs.append(forward_model(model, features, batch_cpu, device).detach().cpu())
    if not outputs:
        return torch.empty((0, int(getattr(model, "nclass", 0))))
    return torch.cat(outputs, dim=0)


def metrics_from_logits_compat(
    logits: torch.Tensor, labels: torch.Tensor
) -> Dict[str, float]:
    helper = getattr(runner, "metrics_from_logits", None)
    if callable(helper):
        return helper(logits, labels)

    logits_cpu = logits.detach().cpu()
    labels_cpu = labels.detach().cpu()
    preds = logits_cpu.argmax(dim=-1).numpy()
    truth = labels_cpu.numpy()
    order = torch.argsort(logits_cpu, dim=1, descending=True)
    matches = order.eq(labels_cpu.view(-1, 1))
    ranks = matches.float().argmax(dim=1) + 1
    return {
        "accuracy": float(accuracy_score(truth, preds)),
        "micro_f1": float(f1_score(truth, preds, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(truth, preds, average="macro", zero_division=0)),
        "macro_precision": float(
            precision_score(truth, preds, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(truth, preds, average="macro", zero_division=0)
        ),
        "hit_at_1": float((ranks <= 1).float().mean().item()),
        "hit_at_3": float((ranks <= min(3, logits_cpu.shape[1])).float().mean().item()),
        "mrr": float((1.0 / ranks.float()).mean().item()),
    }


def architecture_config(payload: Mapping[str, Any]) -> Dict[str, Any]:
    stored = dict(payload.get("hyperparameters", {}))
    defaults = {
        "embed_size": 512,
        "hidden": 512,
        "n_fp_layers": 2,
        "n_task_layers": 4,
        "num_heads": 1,
        "dropout": 0.5,
        "input_drop": 0.0,
        "att_drop": 0.0,
        "act": "none",
        "residual": False,
        "lr": 0.005,
        "weight_decay": 0.0001,
        "batch_size": 10000,
        "eval_batch_size": 20000,
        "grad_clip": 0.0,
    }
    defaults.update(stored)
    return defaults


def static_model_memory(model: torch.nn.Module) -> Dict[str, float]:
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    buffer_bytes = sum(
        buffer.numel() * buffer.element_size() for buffer in model.buffers()
    )
    return {
        "parameter_mib": bytes_to_mib(parameter_bytes),
        "buffer_mib": bytes_to_mib(buffer_bytes),
        "static_model_mib": bytes_to_mib(parameter_bytes + buffer_bytes),
    }


def profile_inference_peak(
    model: torch.nn.Module,
    features: Mapping[str, Any],
    split_indices: Mapping[str, np.ndarray],
    eval_batch_size: int,
    device: torch.device,
) -> float | None:
    if device.type != "cuda":
        return None
    model.eval()
    clear_cuda(device)
    torch.cuda.reset_peak_memory_stats(device)
    for indices in split_indices.values():
        output = predict_indices_compat(model, features, indices, eval_batch_size, device)
        del output
    torch.cuda.synchronize(device)
    return bytes_to_mib(torch.cuda.max_memory_allocated(device))


def profile_training_peak(
    *,
    model: torch.nn.Module,
    features: Mapping[str, Any],
    labels: torch.Tensor,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    config: Mapping[str, Any],
    device: torch.device,
) -> float | None:
    """Replay one optimizer step plus one validation batch.

    The original runner resets the CUDA peak counter before training and reads it
    after the training/validation loop.  A full training batch, Adam state
    allocation, backward pass, optimizer step, and one full-size validation
    batch therefore reproduce the allocation categories that can determine that
    peak without replaying all epochs.
    """
    if device.type != "cuda":
        return None
    if len(train_idx) == 0:
        raise ValueError("Cannot profile training memory with an empty train split")

    train_batch_size = int(config.get("batch_size", 10000))
    eval_batch_size = int(config.get("eval_batch_size", 20000))
    train_batch = torch.as_tensor(train_idx[:train_batch_size], dtype=torch.long)
    val_batch = torch.as_tensor(val_idx[:eval_batch_size], dtype=torch.long)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("lr", 0.005)),
        weight_decay=float(config.get("weight_decay", 0.0001)),
    )

    model.train()
    clear_cuda(device)
    torch.cuda.reset_peak_memory_stats(device)

    optimizer.zero_grad(set_to_none=True)
    logits = forward_model(model, features, train_batch, device)
    target = labels[train_batch].to(device)
    loss = F.cross_entropy(logits, target)
    loss.backward()
    grad_clip = float(config.get("grad_clip", 0.0))
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    # Validation is performed while the optimizer and its state remain live in
    # the real training loop.  Include one maximum-size validation batch.
    if len(val_batch) > 0:
        model.eval()
        with torch.no_grad():
            val_logits = forward_model(model, features, val_batch, device)
        del val_logits

    torch.cuda.synchronize(device)
    peak = bytes_to_mib(torch.cuda.max_memory_allocated(device))

    optimizer.zero_grad(set_to_none=True)
    del logits, target, loss, optimizer
    clear_cuda(device)
    return peak


def needs_memory_profile(run: Mapping[str, Any], overwrite_memory: bool) -> bool:
    if overwrite_memory:
        return True
    memory = run.get("memory")
    if not isinstance(memory, Mapping):
        return True
    required = (
        "checkpoint_mib",
        "parameter_mib",
        "peak_training_gpu_mib",
        "peak_inference_gpu_mib",
    )
    return any(memory.get(key) is None for key in required)


def backfill_one(result_json: Path, args: argparse.Namespace) -> None:
    result_json = result_json.expanduser().resolve()
    if not result_json.is_file():
        raise FileNotFoundError(result_json)
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    data_dir = (
        args.data_dir.expanduser().resolve()
        if args.data_dir is not None
        else Path(str(payload["data_dir"])).expanduser().resolve()
    )

    manifest, dataset, features, _ = runner.load_preprocessed(data_dir)
    config = architecture_config(payload)
    if args.batch_size is not None:
        config["batch_size"] = int(args.batch_size)
    if args.eval_batch_size is not None:
        config["eval_batch_size"] = int(args.eval_batch_size)
    eval_batch_size = int(config["eval_batch_size"])
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}"
    )

    labels = torch.from_numpy(dataset["labels"].astype(np.int64, copy=False))
    split_indices = {
        "train": dataset["train_idx"].astype(np.int64, copy=False),
        "val": dataset["val_idx"].astype(np.int64, copy=False),
        "test": dataset["test_idx"].astype(np.int64, copy=False),
    }
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else result_json.parent / "logits"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    changed = False
    for run in payload.get("runs", []):
        seed = int(run["seed"])
        checkpoint_raw = run.get("checkpoint") or run.get("checkpoint_path")
        if not checkpoint_raw:
            raise KeyError(f"seed={seed}: run has neither 'checkpoint' nor 'checkpoint_path'")
        checkpoint_path = resolve_existing_path(str(checkpoint_raw), result_json, data_dir)

        existing_logits_path: Path | None = None
        existing_logits = run.get("logits_file")
        if existing_logits:
            try:
                existing_logits_path = resolve_existing_path(
                    str(existing_logits), result_json, data_dir
                )
            except FileNotFoundError:
                existing_logits_path = None
        need_logits = args.overwrite_logits or existing_logits_path is None
        need_memory = args.profile_memory and needs_memory_profile(run, args.overwrite_memory)

        if not need_logits and not need_memory:
            print(f"seed={seed}: logits and memory metadata already complete; skipping")
            continue

        # One exact architecture reconstruction provides static memory and
        # inference/logits.  A second reconstruction is used for the training
        # replay so optimizer updates cannot affect saved logits.
        model = build_sehgnn_model_compat(config, manifest, list(features.keys())).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state = extract_model_state_compat(checkpoint)
        model.load_state_dict(state, strict=True)
        model.eval()

        if need_logits:
            arrays: Dict[str, np.ndarray] = {}
            for split_name, indices in split_indices.items():
                logits = predict_indices_compat(
                    model, features, indices, eval_batch_size, device
                )
                arrays[f"{split_name}_logits"] = logits.numpy().astype(
                    np.float32, copy=False
                )
                arrays[f"{split_name}_labels"] = labels[indices].numpy().astype(
                    np.int64, copy=False
                )
                arrays[f"{split_name}_idx"] = indices
                run.setdefault("splits", {})[split_name] = metrics_from_logits_compat(
                    logits, labels[indices]
                )

            logits_path = output_dir / f"{checkpoint_path.stem}_logits.npz"
            if logits_path.exists() and not args.overwrite_logits:
                # Reuse an already-created artifact even when the old JSON did
                # not contain its path.
                print(f"seed={seed}: reusing existing logits artifact {logits_path}")
            else:
                np.savez_compressed(logits_path, **arrays)
                print(f"seed={seed}: wrote {logits_path}")
            run["logits_file"] = str(logits_path)
            changed = True

        if need_memory:
            memory: MutableMapping[str, Any] = dict(run.get("memory", {}))
            static = static_model_memory(model)
            memory["checkpoint_mib"] = bytes_to_mib(checkpoint_path.stat().st_size)
            memory["parameter_mib"] = static["parameter_mib"]
            memory["buffer_mib"] = static["buffer_mib"]
            memory["static_model_mib"] = static["static_model_mib"]

            if device.type == "cuda":
                memory["peak_inference_gpu_mib"] = profile_inference_peak(
                    model, features, split_indices, eval_batch_size, device
                )

                del model
                clear_cuda(device)
                training_model = build_sehgnn_model_compat(
                    config, manifest, list(features.keys())
                ).to(device)
                training_model.load_state_dict(state, strict=True)
                memory["peak_training_gpu_mib"] = profile_training_peak(
                    model=training_model,
                    features=features,
                    labels=labels,
                    train_idx=split_indices["train"],
                    val_idx=split_indices["val"],
                    config=config,
                    device=device,
                )
                del training_model
                clear_cuda(device)
                model = None
            else:
                memory.setdefault("peak_inference_gpu_mib", None)
                memory.setdefault("peak_training_gpu_mib", None)
                print(
                    f"seed={seed}: CPU mode fills checkpoint/parameter memory only; "
                    "GPU peak fields remain unavailable"
                )

            memory["profile_metadata"] = {
                "profiled_at_utc": datetime.now(timezone.utc).isoformat(),
                "device": str(device),
                "checkpoint_mib_source": "checkpoint file size",
                "parameter_mib_source": "reconstructed model parameters",
                "peak_inference_gpu_mib_source": (
                    "checkpoint replay over train/val/test" if device.type == "cuda" else None
                ),
                "peak_training_gpu_mib_source": (
                    "representative checkpoint replay: one optimizer step plus one validation batch"
                    if device.type == "cuda"
                    else None
                ),
                "peak_training_is_historical": False,
                "batch_size": int(config["batch_size"]),
                "eval_batch_size": int(config["eval_batch_size"]),
            }
            run["memory"] = memory
            changed = True
            print(
                f"seed={seed}: checkpoint={memory['checkpoint_mib']:.2f} MiB, "
                f"parameters={memory['parameter_mib']:.2f} MiB, "
                f"train_peak={memory.get('peak_training_gpu_mib')}, "
                f"inference_peak={memory.get('peak_inference_gpu_mib')}"
            )

        if model is not None:
            del model
        del checkpoint, state
        clear_cuda(device)

    if changed:
        backup = result_json.with_suffix(result_json.suffix + ".before_artifact_backfill")
        if not backup.exists():
            shutil.copy2(result_json, backup)
        result_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Updated {result_json}")
        print(f"Backup:  {backup}")
    else:
        print(f"No changes required for {result_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate missing logits and memory metadata from already-trained "
            "SeHGNN Freebase checkpoints"
        )
    )
    parser.add_argument("--result-json", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional override; normally read from each result JSON",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument(
        "--no-profile-memory",
        dest="profile_memory",
        action="store_false",
        help="Backfill logits only",
    )
    parser.set_defaults(profile_memory=True)
    parser.add_argument("--overwrite", dest="overwrite_logits", action="store_true")
    parser.add_argument("--overwrite-logits", action="store_true")
    parser.add_argument("--overwrite-memory", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner_path = Path(getattr(runner, "__file__", "<unknown>")).resolve(strict=False)
    compatibility_mode = not callable(getattr(runner, "build_sehgnn_model", None))
    print(f"Using runner module: {runner_path}")
    if compatibility_mode:
        print(
            "Runner compatibility mode: reconstructing SeHGNN from the legacy inline constructor"
        )
    if args.data_dir is not None and len(args.result_json) > 1:
        raise ValueError("--data-dir can only be used with one --result-json")
    if args.cpu and args.profile_memory:
        print(
            "Warning: --cpu cannot produce peak GPU memory. Static checkpoint and "
            "parameter memory will still be filled."
        )
    for result_json in args.result_json:
        backfill_one(result_json, args)


if __name__ == "__main__":
    main()