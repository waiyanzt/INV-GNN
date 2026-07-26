#!/usr/bin/env python3
"""Verify recomputing chunked Freebase RGCN equivalence to PyG."""

from __future__ import annotations

import argparse

import torch
from torch_geometric.nn import RGCNConv

from chunked_rgcn_conv import ChunkedRGCNConv
from model_RGCN_freebase_nc import RGCNFeatureless


def maximum_absolute_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(1566911444)

    num_nodes = 11
    in_channels = 7
    out_channels = 5
    num_relations = 4
    num_bases = 3

    # Interleaved relation IDs, repeated destinations, and more than one chunk
    # per populated relation exercise the exact path used by Freebase.
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 3, 5, 7, 9, 0, 2, 4, 6],
            [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7, 7],
        ],
        dtype=torch.long,
        device=device,
    )
    edge_type = torch.tensor(
        [0, 1, 0, 2, 1, 2, 3, 0, 3, 1, 0, 2, 2, 1, 3, 0, 1, 3, 1, 0],
        dtype=torch.long,
        device=device,
    )

    standard = RGCNConv(
        in_channels,
        out_channels,
        num_relations,
        num_bases=num_bases,
    ).to(device)
    chunked = ChunkedRGCNConv(
        in_channels,
        out_channels,
        num_relations,
        num_bases=num_bases,
        edge_chunk_size=3,
    ).to(device)
    chunked.load_state_dict(standard.state_dict(), strict=True)

    if tuple(standard.state_dict()) != tuple(chunked.state_dict()):
        raise AssertionError("Standard and chunked state-dict keys differ")
    for name, tensor in standard.state_dict().items():
        torch.testing.assert_close(
            tensor,
            chunked.state_dict()[name],
            atol=0,
            rtol=0,
        )

    standard_x = torch.randn(
        num_nodes, in_channels, device=device, requires_grad=True
    )
    chunked_x = standard_x.detach().clone().requires_grad_(True)
    probe = torch.randn(num_nodes, out_channels, device=device)

    standard_output = standard(standard_x, edge_index, edge_type)
    chunked_output = chunked(chunked_x, edge_index, edge_type)
    torch.testing.assert_close(
        standard_output,
        chunked_output,
        atol=args.atol,
        rtol=args.rtol,
    )

    (standard_output * probe).sum().backward()
    (chunked_output * probe).sum().backward()
    torch.testing.assert_close(
        standard_x.grad,
        chunked_x.grad,
        atol=args.atol,
        rtol=args.rtol,
    )

    standard_parameters = dict(standard.named_parameters())
    chunked_parameters = dict(chunked.named_parameters())
    if standard_parameters.keys() != chunked_parameters.keys():
        raise AssertionError("Standard and chunked parameter names differ")
    parameter_gradient_differences = {}
    for name in standard_parameters:
        standard_gradient = standard_parameters[name].grad
        chunked_gradient = chunked_parameters[name].grad
        if standard_gradient is None or chunked_gradient is None:
            raise AssertionError(f"Missing gradient for parameter {name}")
        torch.testing.assert_close(
            standard_gradient,
            chunked_gradient,
            atol=args.atol,
            rtol=args.rtol,
        )
        parameter_gradient_differences[name] = maximum_absolute_difference(
            standard_gradient, chunked_gradient
        )

    # One identical optimizer step confirms that the custom backward feeds the
    # unchanged optimizer/parameter layout exactly as the PyG layer does.
    standard_optimizer = torch.optim.Adam(standard.parameters(), lr=0.001)
    chunked_optimizer = torch.optim.Adam(chunked.parameters(), lr=0.001)
    standard_optimizer.step()
    chunked_optimizer.step()
    parameter_step_differences = {}
    for name in standard_parameters:
        torch.testing.assert_close(
            standard_parameters[name],
            chunked_parameters[name],
            atol=args.atol,
            rtol=args.rtol,
        )
        parameter_step_differences[name] = maximum_absolute_difference(
            standard_parameters[name], chunked_parameters[name]
        )

    standard_model = RGCNFeatureless(
        num_nodes=num_nodes,
        num_relations=num_relations,
        hidden_dim=in_channels,
        num_classes=out_channels,
        num_bases=num_bases,
        edge_chunk_size=0,
    )
    chunked_model = RGCNFeatureless(
        num_nodes=num_nodes,
        num_relations=num_relations,
        hidden_dim=in_channels,
        num_classes=out_channels,
        num_bases=num_bases,
        edge_chunk_size=3,
    )
    chunked_model.load_state_dict(standard_model.state_dict(), strict=True)
    if tuple(standard_model.state_dict()) != tuple(chunked_model.state_dict()):
        raise AssertionError("Standard and chunked model state-dict keys differ")

    print("[OK] Recomputing chunked RGCN matches PyG RGCNConv")
    print(
        "maximum_forward_absolute_difference:",
        maximum_absolute_difference(standard_output, chunked_output),
    )
    print(
        "maximum_input_gradient_absolute_difference:",
        maximum_absolute_difference(standard_x.grad, chunked_x.grad),
    )
    print("parameter_gradient_absolute_differences:", parameter_gradient_differences)
    print("parameter_step_absolute_differences:", parameter_step_differences)


if __name__ == "__main__":
    main()
