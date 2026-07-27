#!/usr/bin/env python3
import os
import sys
import gc
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import dgl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import kendalltau

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from model import slotGAT
from utils.tools import set_seed, func_args_parse, single_feat_net
from utils.data import load_data
from generic_gnn_memory_profiler import print_memory_stats, profile_checkpoint_training_memory

NUM_CLASSES = 8


def sp_to_spt(mat):
    """Convert a SciPy sparse matrix to a coalesced float32 PyTorch sparse tensor."""
    coo = mat.tocoo(copy=False)
    indices = torch.from_numpy(
        np.vstack((coo.row, coo.col)).astype(np.int64, copy=False)
    )
    values = torch.from_numpy(
        coo.data.astype(np.float32, copy=False)
    )
    return torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=coo.shape,
        dtype=torch.float32,
    ).coalesce()


def mat2tensor(mat):
    if isinstance(mat, np.ndarray):
        return torch.as_tensor(mat, dtype=torch.float32)
    return sp_to_spt(mat)


def build_graph(features_list_cpu, adjM, dl, device):
    """
    Build the graph using the same semantics and edge ordering as the training runner,
    while avoiding unnecessary GPU->CPU copies and per-node CUDA assignments.

    Preserved behavior:
      * edge2type[(u, v)] = relation id
      * missing self loops use len(dl.links['count'])
      * missing reverse edges use the same relation-id counting rule
      * graph = adjM + adjM.T
      * remove all self loops, then add one self loop per node
      * e_feat follows DGL EID order
      * raw relation types are remapped by first appearance to contiguous ids
      * num_etype remains the same upper bound used during training
    """
    edge2type = {}

    # Original typed directed edges.
    for relation_id, relation_adj in dl.links["data"].items():
        rows, cols = relation_adj.nonzero()
        for u, v in zip(rows, cols):
            edge2type[(int(u), int(v))] = relation_id

    # Self-loop relation type.
    self_loop_type = len(dl.links["count"])
    for node_id in range(dl.nodes["total"]):
        edge2type.setdefault((node_id, node_id), self_loop_type)

    # Missing reverse-edge relation types, preserving the training runner's rule.
    count_reverse = 0
    for _, relation_adj in dl.links["data"].items():
        rows, cols = relation_adj.nonzero()
        added_reverse_for_relation = False
        reverse_type = len(dl.links["count"]) + 1 + count_reverse

        for u, v in zip(rows, cols):
            u = int(u)
            v = int(v)
            if (v, u) not in edge2type:
                edge2type[(v, u)] = reverse_type
                added_reverse_for_relation = True

        if added_reverse_for_relation:
            count_reverse += 1

    num_etype = len(dl.links["count"]) + 1 + count_reverse

    # Construct and enumerate the graph on CPU. This preserves DGL edge ordering
    # while avoiding a full GPU->CPU graph copy.
    g_cpu = dgl.DGLGraph(adjM + adjM.T)
    g_cpu = dgl.remove_self_loop(g_cpu)
    g_cpu = dgl.add_self_loop(g_cpu)

    src, dst = g_cpu.edges(order="eid")
    src_np = src.numpy()
    dst_np = dst.numpy()

    # Preserve the original first-seen contiguous relation remapping.
    count_mappings = {}
    next_type_id = 0
    remapped_edge_types = np.empty(len(src_np), dtype=np.int64)

    for edge_index, (u, v) in enumerate(zip(src_np, dst_np)):
        raw_type = edge2type[(int(u), int(v))]
        mapped_type = count_mappings.get(raw_type)
        if mapped_type is None:
            mapped_type = next_type_id
            count_mappings[raw_type] = mapped_type
            next_type_id += 1
        remapped_edge_types[edge_index] = mapped_type

    assert next_type_id <= num_etype, (
        f"Observed {next_type_id} remapped edge types, but num_etype={num_etype}"
    )

    e_feat_cpu = torch.from_numpy(remapped_edge_types).long()

    # Build the same dense node-type one-hot tensor expected by SlotGAT, but
    # vectorize the writes on CPU rather than performing one CUDA write per node.
    num_ntypes = len(features_list_cpu)
    num_nodes = dl.nodes["total"]
    node_ntype_indexer_cpu = torch.zeros(
        (num_nodes, num_ntypes),
        dtype=torch.float32,
    )
    node_idx_by_ntype = []

    start = 0
    for ntype, feature in enumerate(features_list_cpu):
        node_count = int(feature.shape[0])
        end = start + node_count
        node_idx_by_ntype.append(list(range(start, end)))
        node_ntype_indexer_cpu[start:end, ntype] = 1.0
        start = end

    assert start == num_nodes, (
        f"Feature rows cover {start} nodes, but dl reports {num_nodes} nodes"
    )

    # Transfer the completed graph metadata exactly once.
    g = g_cpu.to(device)
    e_feat = e_feat_cpu.to(device, non_blocking=True)

    g.num_ntypes = num_ntypes
    g.node_idx_by_ntype = node_idx_by_ntype
    g.node_ntype_indexer = node_ntype_indexer_cpu.to(
        device, non_blocking=True
    )

    del g_cpu, e_feat_cpu, node_ntype_indexer_cpu
    return g, e_feat, num_etype


def load_variant_once(variant, device):
    """
    Load and construct one variant once. The returned graph/features are reused
    across all checkpoint seeds for that variant.
    """
    variant_name = f"Freebase_{variant}"
    print(f"\nLoading variant data once: {variant_name}", flush=True)

    raw_features, adjM, raw_labels, split_dict, dl = load_data(
        prefix=variant_name,
        multi_labels=False,
    )

    # Keep features on CPU while constructing graph metadata.
    features_list_cpu = [mat2tensor(feature) for feature in raw_features]
    in_dims = [int(feature.shape[1]) for feature in features_list_cpu]

    train_idx = np.sort(split_dict["train_idx"])
    test_idx = np.sort(split_dict["test_idx"])

    g, e_feat, num_etype = build_graph(
        features_list_cpu,
        adjM,
        dl,
        device,
    )

    # Transfer feature tensors only after graph construction has finished.
    features_list = [
        feature.to(device, non_blocking=True)
        for feature in features_list_cpu
    ]
    labels = torch.as_tensor(
        raw_labels,
        dtype=torch.long,
        device=device,
    )

    del raw_features, raw_labels, adjM, split_dict, dl, features_list_cpu
    gc.collect()

    return {
        "features_list": features_list,
        "in_dims": in_dims,
        "labels": labels,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "g": g,
        "e_feat": e_feat,
        "num_etype": num_etype,
        "num_ntypes": len(features_list),
    }






def full_logit_kendall_tau(logits_a, logits_b):
    """
    Mean Kendall tau-b across test nodes, comparing the complete class-logit
    rankings for two models.

    Each test node contributes one Kendall tau computed across all class logits.
    The returned value is the mean over nodes, ignoring undefined node-level
    correlations.
    """
    if logits_a.shape != logits_b.shape:
        raise ValueError(
            f"Logit shapes differ: {tuple(logits_a.shape)} vs "
            f"{tuple(logits_b.shape)}"
        )

    a = logits_a.detach().cpu().numpy()
    b = logits_b.detach().cpu().numpy()

    taus = []
    for row_a, row_b in zip(a, b):
        tau, _ = kendalltau(
            row_a,
            row_b,
            variant="b",
            nan_policy="omit",
        )
        if np.isfinite(tau):
            taus.append(float(tau))

    return float(np.mean(taus)) if taus else float("nan")


def nanmean_or_nan(values):
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def aggregate_output_fusion_memory_stats(seed_stats):
    """
    Aggregate numeric memory fields across variants for one seed.

    Hybrid rule:
      * additive footprint/storage fields are SUMMED across variants:
          - parameter count / parameter memory
          - buffer memory
          - total model memory
          - checkpoint size
      * runtime peak-memory fields are MAXIMIZED across variants because
        constituent models are executed sequentially.

    Fields must be present and numeric for every variant to be reported.
    """
    if not seed_stats:
        return {}, {}

    numeric_by_variant = [
        extract_numeric_stats(stats)
        for stats in seed_stats
    ]

    common_keys = set(numeric_by_variant[0])
    for numeric in numeric_by_variant[1:]:
        common_keys &= set(numeric)

    additive_tokens = (
        "parameter",
        "param_",
        "paramcount",
        "param_count",
        "buffer",
        "checkpoint",
        "model_memory",
        "model_bytes",
        "static_model",
        "total_model",
    )

    summed = {}
    maximized = {}

    for key in sorted(common_keys):
        key_lower = key.lower()
        values = [numeric[key] for numeric in numeric_by_variant]

        if any(token in key_lower for token in additive_tokens):
            summed[key] = float(sum(values))
        else:
            maximized[key] = float(max(values))

    return summed, maximized


def print_output_fusion_memory_summary(memory_stats_by_seed, variants):
    """
    Report Output Fusion memory using the hybrid aggregation rule:

      * parameter/checkpoint/model-footprint fields:
            sum across variants within each seed
      * runtime peak-memory fields:
            max across variants within each seed
      * final report:
            mean +/- population std across seeds
    """
    print("\n=== Output Fusion Memory Metrics ===")
    print(
        "Footprint/storage fields: sum across variants per seed. "
        "Runtime peak fields: max across variants per seed. "
        "Then mean +/- std across seeds."
    )
    print(f"Variants: {', '.join(variants)}")

    summed_values_by_key = defaultdict(list)
    max_values_by_key = defaultdict(list)

    for seed, stats_list in memory_stats_by_seed.items():
        summed, maximized = aggregate_output_fusion_memory_stats(
            stats_list
        )

        print(f"\n  Seed {seed} Output Fusion memory:")

        if summed:
            print("    Additive footprint/storage fields:")
            for key, value in sorted(summed.items()):
                print(f"      {key}: {value:.4f}")
                summed_values_by_key[key].append(value)

        if maximized:
            print("    Sequential runtime peak fields:")
            for key, value in sorted(maximized.items()):
                print(f"      {key}: {value:.4f}")
                max_values_by_key[key].append(value)

        if not summed and not maximized:
            print("    No common numeric profiler fields were available.")

    print("\n  Across-seed Output Fusion summary:")

    if summed_values_by_key:
        print("    Additive footprint/storage fields:")
        for key, values in sorted(summed_values_by_key.items()):
            arr = np.asarray(values, dtype=np.float64)
            print(
                f"      {key}: {arr.mean():.4f} +/- {arr.std():.4f}"
            )

    if max_values_by_key:
        print("    Sequential runtime peak fields:")
        for key, values in sorted(max_values_by_key.items()):
            arr = np.asarray(values, dtype=np.float64)
            print(
                f"      {key}: {arr.mean():.4f} +/- {arr.std():.4f}"
            )

    if not summed_values_by_key and not max_values_by_key:
        print("    No common numeric profiler fields were available.")



def extract_numeric_stats(stats):
    """
    Return numeric entries from a flat profiler result dictionary.
    Non-numeric metadata is ignored when averaging across seeds.
    """
    if not isinstance(stats, dict):
        return {}

    numeric = {}
    for key, value in stats.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric[key] = float(value)
    return numeric


def print_average_memory_stats(memory_stats_by_variant):
    """Print mean +/- population std for numeric profiler fields across seeds."""
    print("\n=== Training Memory Profile Summary Across Seeds ===")

    for variant, seed_stats in memory_stats_by_variant.items():
        print(f"\nVariant: {variant}")
        values_by_key = defaultdict(list)

        for stats in seed_stats:
            for key, value in extract_numeric_stats(stats).items():
                values_by_key[key].append(value)

        if not values_by_key:
            print("  No numeric profiler fields were available to aggregate.")
            continue

        for key, values in sorted(values_by_key.items()):
            arr = np.asarray(values, dtype=np.float64)
            print(
                f"  {key}: {arr.mean():.4f} +/- {arr.std():.4f}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        nargs="+",
        type=str,
        default=["unchanged", "exact_2"],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1566911444, 20241017, 20251017],
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", default=False)

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--edge-feats", type=int, default=64)
    parser.add_argument("--dropout-feat", type=float, default=0.5)
    parser.add_argument("--dropout-attn", type=float, default=0.2)
    parser.add_argument("--slope", type=float, default=0.05)
    parser.add_argument("--aggregator", type=str, default="SA")
    parser.add_argument("--SAattDim", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.gpu}"
        if not args.cpu and torch.cuda.is_available()
        else "cpu"
    )

    # Keep the original metric logic unchanged.
    seed_results = {
        metric: []
        for metric in [
            "Accuracy",
            "Precision",
            "Recall",
            "Macro-F1",
            "Micro-F1",
            "FullKendallTau",
        ]
    }

    # Store small CPU test logits by seed. Each seed receives one tensor per
    # variant; fusion happens only after all variants have been evaluated.
    variant_logits_by_seed = {
        seed: []
        for seed in args.seeds
    }
    true_labels_by_seed = {}
    test_idx_by_seed = {}

    # One memory-profile result per (variant, seed), then averaged across seeds.
    memory_stats_by_variant = {
        variant: []
        for variant in args.variants
    }
    memory_stats_by_seed = {
        seed: []
        for seed in args.seeds
    }

    # Variant-outer loop: build each potentially huge graph once, reuse it across
    # all seeds, then free it before loading the next variant.
    for variant in args.variants:
        print(f"\n{'=' * 60}")
        print(f"Processing Variant: {variant}")
        print(f"{'=' * 60}")

        data = load_variant_once(variant, device)

        features_list = data["features_list"]
        in_dims = data["in_dims"]
        labels = data["labels"]
        train_idx = data["train_idx"]
        test_idx = data["test_idx"]
        g = data["g"]
        e_feat = data["e_feat"]
        num_etype = data["num_etype"]
        num_ntypes = data["num_ntypes"]

        heads = [args.num_heads] * args.num_layers + [1]

        for seed in args.seeds:
            print(f"\n--- Variant {variant}, Seed {seed} ---", flush=True)
            set_seed(seed)

            dataRecorder = {
                "meta": {"getSAAttentionScore": "False"},
                "data": {},
                "status": "None",
            }

            fargs, fkargs = func_args_parse(
                g,
                args.edge_feats,
                num_etype,
                in_dims,
                args.hidden_dim,
                NUM_CLASSES,
                args.num_layers,
                heads,
                F.elu,
                args.dropout_feat,
                args.dropout_attn,
                args.slope,
                True,
                0.05,
                num_ntype=num_ntypes,
                eindexer=None,
                aggregator=args.aggregator,
                SAattDim=args.SAattDim,
                dataRecorder=dataRecorder,
                vis_data_saver=None,
            )

            ckp_fname = (
                f"checkpoint/"
                f"slotgat_freebase_nc_{variant}_seed{seed}.pt"
            )
            if not os.path.isfile(ckp_fname):
                raise FileNotFoundError(
                    f"Checkpoint not found: {ckp_fname}"
                )

            def build_model():
                return single_feat_net(slotGAT, *fargs, **fkargs)

            def load_batch():
                return {
                    "features": features_list,
                    "edge_feats": e_feat,
                    "train_idx": train_idx,
                    "labels": labels,
                }

            def forward_loss(model, batch):
                model.train()
                dataRecorder["status"] = "Training"
                logits, _ = model(
                    batch["features"],
                    batch["edge_feats"],
                )
                dataRecorder["status"] = "None"
                logp = F.log_softmax(logits, 1)
                return F.nll_loss(
                    logp[batch["train_idx"]],
                    batch["labels"][batch["train_idx"]],
                )

            def build_optimizer(model):
                return torch.optim.Adam(
                    model.parameters(),
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                )

            # Exactly one profiler call for this variant/seed pair.
            print("--- Training Step Memory Profile ---", flush=True)
            train_stats = profile_checkpoint_training_memory(
                checkpoint_path=ckp_fname,
                build_model_fn=build_model,
                load_batch_fn=load_batch,
                forward_loss_fn=forward_loss,
                build_optimizer_fn=build_optimizer,
                device=device,
            )

            # Ensure the additive footprint/storage fields are always available,
            # independent of which fields the generic profiler returns.
            footprint_model = build_model().to(device)
            parameter_count = sum(
                parameter.numel()
                for parameter in footprint_model.parameters()
            )
            parameter_memory_bytes = sum(
                parameter.numel() * parameter.element_size()
                for parameter in footprint_model.parameters()
            )
            buffer_memory_bytes = sum(
                buffer.numel() * buffer.element_size()
                for buffer in footprint_model.buffers()
            )
            checkpoint_size_bytes = os.path.getsize(ckp_fname)

            if not isinstance(train_stats, dict):
                raise TypeError(
                    "profile_checkpoint_training_memory must return a dictionary "
                    "so memory fields can be aggregated."
                )

            train_stats = dict(train_stats)
            train_stats.update({
                "parameter_count": float(parameter_count),
                "parameter_memory_bytes": float(parameter_memory_bytes),
                "buffer_memory_bytes": float(buffer_memory_bytes),
                "total_model_memory_bytes": float(
                    parameter_memory_bytes + buffer_memory_bytes
                ),
                "checkpoint_size_bytes": float(checkpoint_size_bytes),
            })

            del footprint_model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            print_memory_stats(train_stats)
            memory_stats_by_variant[variant].append(train_stats)
            memory_stats_by_seed[seed].append(train_stats)

            # Re-instantiate a fresh model for inference so profiler updates cannot
            # affect the logits used for output fusion.
            net = build_model().to(device)
            state_dict = torch.load(
                ckp_fname,
                map_location=device,
            )
            load_result = net.load_state_dict(
                state_dict,
                strict=False,
            )
            if load_result.missing_keys or load_result.unexpected_keys:
                raise RuntimeError(
                    f"Checkpoint incompatibility for {ckp_fname}\n"
                    f"Missing keys: {load_result.missing_keys}\n"
                    f"Unexpected keys: {load_result.unexpected_keys}"
                )
            del state_dict, load_result

            net.eval()

            print("--- Inference Memory Profile ---", flush=True)
            if device.type == "cuda":
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)

                with torch.inference_mode():
                    dataRecorder["status"] = "FinalTesting"
                    logits, _ = net(features_list, e_feat)
                    dataRecorder["status"] = "None"
                    test_out = logits[test_idx]

                torch.cuda.synchronize(device)
                peak_allocated = torch.cuda.max_memory_allocated(device)
                print(
                    "Peak Inference Allocated Memory: "
                    f"{peak_allocated / (1024 ** 2):.2f} MiB"
                )
            else:
                with torch.inference_mode():
                    dataRecorder["status"] = "FinalTesting"
                    logits, _ = net(features_list, e_feat)
                    dataRecorder["status"] = "None"
                    test_out = logits[test_idx]

            if test_out.shape != (len(test_idx), NUM_CLASSES):
                raise RuntimeError(
                    f"Unexpected test logit shape for variant={variant}, seed={seed}: "
                    f"{tuple(test_out.shape)}"
                )
            if not torch.isfinite(test_out).all():
                raise RuntimeError(
                    f"Non-finite test logits for variant={variant}, seed={seed}"
                )

            # Save only the small test tensor on CPU.
            variant_logits_by_seed[seed].append(
                test_out.detach().cpu()
            )

            current_labels = (
                labels[test_idx]
                .detach()
                .cpu()
                .numpy()
            )

            # Preserve the original label check, and also ensure test-node alignment.
            if seed not in true_labels_by_seed:
                true_labels_by_seed[seed] = current_labels.copy()
                test_idx_by_seed[seed] = test_idx.copy()
            else:
                assert np.array_equal(
                    test_idx_by_seed[seed],
                    test_idx,
                ), (
                    "Test node indices do not align across variants for "
                    f"seed={seed}, variant={variant}"
                )
                assert np.array_equal(
                    true_labels_by_seed[seed],
                    current_labels,
                ), (
                    "Test labels do not align across variants for "
                    f"seed={seed}, variant={variant}"
                )

            del net, logits, test_out, train_stats, fargs, fkargs
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Release this entire variant before constructing the next variant.
        del data
        del features_list, in_dims, labels, train_idx, test_idx
        del g, e_feat, num_etype, num_ntypes
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)

    # Output fusion: average variant logits independently within each seed,
    # compute that seed's metrics, then summarize metrics across seeds.
    for seed in args.seeds:
        print(f"\n{'=' * 40}")
        print(f"--- Aggregating Ensemble for Seed {seed} ---")
        print(f"{'=' * 40}")

        avg_logits = torch.mean(
            torch.stack(variant_logits_by_seed[seed]),
            dim=0,
        )
        pred_labels = avg_logits.argmax(dim=1).numpy()
        true_labels = true_labels_by_seed[seed]

        # Full Kendall tau: average per-node tau-b over all class logits,
        # comparing fusion against each constituent variant, then average those
        # fusion-vs-variant values within this seed.
        full_tau_against_variants = [
            full_logit_kendall_tau(avg_logits, variant_logits)
            for variant_logits in variant_logits_by_seed[seed]
        ]
        mean_full_tau = nanmean_or_nan(full_tau_against_variants)

        acc = accuracy_score(true_labels, pred_labels)
        prec = precision_score(
            true_labels,
            pred_labels,
            average="macro",
            zero_division=0,
        )
        rec = recall_score(
            true_labels,
            pred_labels,
            average="macro",
            zero_division=0,
        )
        f1_macro = f1_score(
            true_labels,
            pred_labels,
            average="macro",
            zero_division=0,
        )
        f1_micro = f1_score(
            true_labels,
            pred_labels,
            average="micro",
            zero_division=0,
        )

        seed_results["Accuracy"].append(acc)
        seed_results["Precision"].append(prec)
        seed_results["Recall"].append(rec)
        seed_results["Macro-F1"].append(f1_macro)
        seed_results["Micro-F1"].append(f1_micro)
        seed_results["FullKendallTau"].append(mean_full_tau)

        print(
            f"  Seed {seed}: full fusion-vs-variant "
            f"Kendall tau={mean_full_tau:.4f}"
        )

    print(
        f"\n=== Final SlotGAT Freebase Node Classification Output Fusion Metrics "
        f"across {len(args.seeds)} seeds ==="
    )
    for metric, values in seed_results.items():
        print(
            f"{metric:<10}: "
            f"{np.mean(values):.4f} +/- {np.std(values):.4f}"
        )

    print_average_memory_stats(memory_stats_by_variant)
    print_output_fusion_memory_summary(
        memory_stats_by_seed,
        args.variants,
    )


if __name__ == "__main__":
    main()
