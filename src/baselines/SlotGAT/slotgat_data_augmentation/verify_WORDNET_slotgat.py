#!/usr/bin/env python3
"""Tiny forward/backward smoke test for the WordNet SlotGAT link predictor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


SLOTGAT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SLOTGAT_ROOT))

from model_slotgat_lp_wordnet import WordNetSlotGATLinkPredictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    torch.manual_seed(7)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 1, 4],
            [1, 2, 0, 4, 5, 3, 4, 2],
        ],
        dtype=torch.long,
        device=device,
    )
    edge_type = torch.tensor(
        [0, 1, 2, 0, 1, 2, 1, 0],
        dtype=torch.long,
        device=device,
    )
    positive = torch.tensor(
        [[0, 0, 1], [1, 1, 2], [4, 1, 5]],
        dtype=torch.long,
        device=device,
    )
    negative = torch.tensor(
        [[0, 0, 5], [1, 1, 4], [4, 1, 0]],
        dtype=torch.long,
        device=device,
    )

    model = WordNetSlotGATLinkPredictor(
        num_entities=6,
        num_relations=3,
        hidden_dim=8,
        num_layers=2,
        num_heads=2,
        edge_feats=4,
        dropout_feat=0.0,
        dropout_attn=0.0,
    ).to(device)
    model.train()
    embeddings = model.encode(edge_index, edge_type, training=True)
    if embeddings.shape != (6, 8):
        raise AssertionError(f"Unexpected embedding shape: {embeddings.shape}")
    positive_scores = model.score(
        embeddings, positive[:, 0], positive[:, 1], positive[:, 2]
    )
    negative_scores = model.score(
        embeddings, negative[:, 0], negative[:, 1], negative[:, 2]
    )
    scores = torch.cat((positive_scores, negative_scores))
    labels = torch.cat(
        (
            torch.ones(len(positive_scores), device=device),
            torch.zeros(len(negative_scores), device=device),
        )
    )
    loss = F.binary_cross_entropy_with_logits(scores, labels)
    loss.backward()

    missing_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if missing_gradients:
        raise AssertionError(f"Parameters without gradients: {missing_gradients}")
    nonfinite_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
        and not torch.isfinite(parameter.grad).all().item()
    ]
    if nonfinite_gradients:
        raise AssertionError(f"Non-finite gradients: {nonfinite_gradients}")

    model.eval()
    with torch.no_grad():
        first = model.encode(edge_index, edge_type, training=False)
        second = model.encode(edge_index, edge_type, training=False)
    torch.testing.assert_close(first, second)
    if len(model._graph_cache) != 1:
        raise AssertionError("The same variant graph was not reused from cache")

    graph, graph_edge_types = next(iter(model._graph_cache.values()))
    if graph.num_edges() != edge_index.shape[1] + model.num_entities:
        raise AssertionError("Exactly one structural self-loop per entity is required")
    if not torch.all(
        graph_edge_types[-model.num_entities :] == model.self_loop_edge_type
    ):
        raise AssertionError("Structural self-loops use the wrong edge type")

    print(
        "[OK] WordNet SlotGAT forward/backward smoke test passed on "
        f"{device}; loss={loss.item():.6f}"
    )


if __name__ == "__main__":
    main()
