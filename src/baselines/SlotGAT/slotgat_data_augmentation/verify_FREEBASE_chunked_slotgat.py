#!/usr/bin/env python3
"""Verify exact chunked SlotGAT attention against the standard DGL path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


SLOTGAT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SLOTGAT_ROOT))

from model.conv import slotGATConv


def recorder():
    return {
        "meta": {
            "getSAAttentionScore": "False",
            "retainLayerAttention": False,
        },
        "data": {},
        "status": "None",
    }


def make_layers(edge_chunk_size: int):
    first = slotGATConv(
        edge_feats=4,
        num_etypes=5,
        in_feats=4,
        out_feats=3,
        num_heads=2,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.05,
        residual=False,
        activation=F.elu,
        alpha=0.05,
        num_ntype=2,
        eindexer=None,
        inputhead=True,
        dataRecorder=recorder(),
        edge_chunk_size=edge_chunk_size,
        decomposed_layers=2 if edge_chunk_size > 0 else 1,
    )
    second = slotGATConv(
        edge_feats=4,
        num_etypes=5,
        in_feats=6,
        out_feats=3,
        num_heads=2,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.05,
        residual=True,
        activation=F.elu,
        alpha=0.05,
        num_ntype=2,
        eindexer=None,
        dataRecorder=recorder(),
        edge_chunk_size=edge_chunk_size,
        decomposed_layers=2 if edge_chunk_size > 0 else 1,
    )
    return nn.ModuleList((first, second))


def forward(layers, graph, features, edge_type):
    hidden, attention = layers[0](graph, features, edge_type)
    hidden = hidden.flatten(1)
    hidden, _ = layers[1](
        graph,
        hidden,
        edge_type,
        res_attn=attention,
    )
    return hidden


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(17)

    source = torch.tensor(
        [0, 1, 2, 3, 4, 5, 0, 2, 4, 1, 3, 5],
        device=device,
    )
    target = torch.tensor(
        [1, 2, 0, 4, 5, 3, 0, 2, 4, 1, 3, 5],
        device=device,
    )
    edge_type = torch.tensor(
        [0, 1, 2, 3, 0, 1, 4, 4, 4, 4, 4, 4],
        device=device,
    )
    graph = dgl.graph((source, target), num_nodes=6, device=device)
    graph.slotgat_edge_index = torch.stack((source, target), dim=0)

    standard = make_layers(edge_chunk_size=0).to(device)
    chunked = make_layers(edge_chunk_size=3).to(device)
    chunked.load_state_dict(standard.state_dict(), strict=True)

    standard_input = torch.randn(6, 8, device=device, requires_grad=True)
    chunked_input = standard_input.detach().clone().requires_grad_(True)
    standard_output = forward(standard, graph, standard_input, edge_type)
    chunked_output = forward(chunked, graph, chunked_input, edge_type)
    torch.testing.assert_close(
        standard_output,
        chunked_output,
        rtol=2e-5,
        atol=2e-6,
    )

    probe = torch.randn_like(standard_output)
    (standard_output * probe).sum().backward()
    (chunked_output * probe).sum().backward()
    torch.testing.assert_close(
        standard_input.grad,
        chunked_input.grad,
        rtol=5e-5,
        atol=5e-6,
    )
    for (standard_name, standard_parameter), (
        chunked_name,
        chunked_parameter,
    ) in zip(standard.named_parameters(), chunked.named_parameters()):
        if standard_name != chunked_name:
            raise AssertionError(
                f"Parameter order differs: {standard_name} vs {chunked_name}"
            )
        torch.testing.assert_close(
            standard_parameter.grad,
            chunked_parameter.grad,
            rtol=8e-5,
            atol=8e-6,
            msg=lambda message, name=standard_name: f"{name}: {message}",
        )

    print(
        "[OK] Chunked SlotGAT matches standard DGL SlotGAT forward and "
        f"backward on {device}"
    )


if __name__ == "__main__":
    main()
