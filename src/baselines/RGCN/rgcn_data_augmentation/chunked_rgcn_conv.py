"""Memory-bounded, exact RGCN aggregation for very large edge sets.

The layer subclasses PyG's :class:`RGCNConv`, so its parameters and state-dict
layout are unchanged. Forward aggregation gathers edges in bounded chunks. Its
custom backward recomputes those aggregates instead of retaining every chunk's
autograd bookkeeping between the two RGCN layers.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor
from torch_geometric.nn import RGCNConv


def _relation_mean(
    x: Tensor,
    edge_index: Tensor,
    relation_edge_ids: Tensor,
    edge_chunk_size: int,
) -> Tuple[Tensor, Tensor]:
    """Return exact per-target relation mean and its full-graph degree."""
    num_nodes, in_channels = x.shape
    relation_sum = x.new_zeros((num_nodes, in_channels))
    relation_degree = x.new_zeros((num_nodes, 1))

    for start in range(0, relation_edge_ids.numel(), edge_chunk_size):
        chunk_ids = relation_edge_ids[start : start + edge_chunk_size]
        source = edge_index[0, chunk_ids]
        target = edge_index[1, chunk_ids]
        relation_sum.index_add_(0, target, x[source])
        relation_degree.index_add_(
            0,
            target,
            torch.ones((target.numel(), 1), dtype=x.dtype, device=x.device),
        )

    return relation_sum / relation_degree.clamp_min(1.0), relation_degree


class _ChunkedRGCNFunction(torch.autograd.Function):
    """Exact basis-decomposed RGCN with a recomputing backward pass."""

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        basis_weight: Tensor,
        comp: Tensor,
        root: Tensor,
        bias: Tensor,
        edge_chunk_size: int,
    ) -> Tensor:
        num_relations = comp.shape[0]
        in_channels = x.shape[1]
        out_channels = basis_weight.shape[2]
        weight = (comp @ basis_weight.reshape(basis_weight.shape[0], -1)).reshape(
            num_relations, in_channels, out_channels
        )
        out = x.new_zeros((x.shape[0], out_channels))

        for relation_id in range(num_relations):
            relation_edge_ids = torch.nonzero(
                edge_type == relation_id, as_tuple=False
            ).flatten()
            if relation_edge_ids.numel() == 0:
                continue
            relation_mean, _ = _relation_mean(
                x, edge_index, relation_edge_ids, edge_chunk_size
            )
            out.addmm_(relation_mean, weight[relation_id])

        out.addmm_(x, root)
        out.add_(bias)

        # Save only graph/model inputs. In particular, do not retain per-edge
        # gathers, relation aggregates, or their autograd graphs.
        ctx.save_for_backward(x, edge_index, edge_type, basis_weight, comp, root)
        ctx.edge_chunk_size = edge_chunk_size
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x, edge_index, edge_type, basis_weight, comp, root = ctx.saved_tensors
        edge_chunk_size = ctx.edge_chunk_size
        num_relations = comp.shape[0]
        in_channels = x.shape[1]
        out_channels = basis_weight.shape[2]
        weight = (comp @ basis_weight.reshape(basis_weight.shape[0], -1)).reshape(
            num_relations, in_channels, out_channels
        )

        grad_x = torch.zeros_like(x)
        grad_weight = torch.zeros_like(weight)

        for relation_id in range(num_relations):
            relation_edge_ids = torch.nonzero(
                edge_type == relation_id, as_tuple=False
            ).flatten()
            if relation_edge_ids.numel() == 0:
                continue

            relation_mean, relation_degree = _relation_mean(
                x, edge_index, relation_edge_ids, edge_chunk_size
            )
            grad_weight[relation_id] = relation_mean.transpose(0, 1) @ grad_output
            grad_relation_mean = grad_output @ weight[relation_id].transpose(0, 1)

            for start in range(0, relation_edge_ids.numel(), edge_chunk_size):
                chunk_ids = relation_edge_ids[start : start + edge_chunk_size]
                source = edge_index[0, chunk_ids]
                target = edge_index[1, chunk_ids]
                grad_x.index_add_(
                    0,
                    source,
                    grad_relation_mean[target] / relation_degree[target],
                )

        grad_x.addmm_(grad_output, root.transpose(0, 1))
        grad_basis_weight = torch.einsum("rb,rio->bio", comp, grad_weight)
        grad_comp = torch.einsum("rio,bio->rb", grad_weight, basis_weight)
        grad_root = x.transpose(0, 1) @ grad_output
        grad_bias = grad_output.sum(dim=0)

        return (
            grad_x,
            None,
            None,
            grad_basis_weight,
            grad_comp,
            grad_root,
            grad_bias,
            None,
        )


class ChunkedRGCNConv(RGCNConv):
    """PyG-compatible RGCNConv with exact, recomputing chunked aggregation.

    This intentionally supports the dense floating-point, basis-decomposed path
    used by Freebase node classification. All graph edges contribute and
    relation degrees are calculated over the complete graph.
    """

    def __init__(self, *args, edge_chunk_size: int, **kwargs) -> None:
        if edge_chunk_size <= 0:
            raise ValueError("edge_chunk_size must be positive")
        super().__init__(*args, **kwargs)
        if self.aggr != "mean":
            raise ValueError("ChunkedRGCNConv currently requires aggr='mean'")
        if self.num_bases is None:
            raise ValueError("ChunkedRGCNConv currently requires num_bases")
        if self.num_blocks is not None:
            raise ValueError("ChunkedRGCNConv does not support num_blocks")
        if self.root is None or self.bias is None:
            raise ValueError("ChunkedRGCNConv currently requires root and bias")
        self.edge_chunk_size = int(edge_chunk_size)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
    ) -> Tensor:
        if not isinstance(x, Tensor) or not torch.is_floating_point(x):
            raise TypeError("ChunkedRGCNConv requires floating-point Tensor features")
        if not isinstance(edge_index, Tensor) or edge_index.ndim != 2:
            raise TypeError("edge_index must be a rank-2 Tensor")
        if edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if not isinstance(edge_type, Tensor) or edge_type.ndim != 1:
            raise TypeError("edge_type must be a rank-1 Tensor")
        if edge_index.shape[1] != edge_type.shape[0]:
            raise ValueError("edge_index and edge_type contain different edge counts")
        if x.ndim != 2 or x.shape[1] != self.in_channels_l:
            raise ValueError(
                f"x must have shape [num_nodes, {self.in_channels_l}]"
            )

        return _ChunkedRGCNFunction.apply(
            x,
            edge_index,
            edge_type,
            self.weight,
            self.comp,
            self.root,
            self.bias,
            self.edge_chunk_size,
        )
