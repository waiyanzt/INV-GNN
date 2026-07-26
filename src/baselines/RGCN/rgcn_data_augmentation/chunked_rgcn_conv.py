"""Memory-bounded RGCN aggregation for very large relation edge sets.

The layer subclasses PyG's :class:`RGCNConv`, so its parameters and state-dict
layout are identical. It changes only how relation-specific mean aggregation is
evaluated: all edges are retained, but source features are gathered in bounded
chunks instead of one tensor for an entire relation.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.nn import RGCNConv


class ChunkedRGCNConv(RGCNConv):
    """PyG-compatible RGCNConv with chunked, exact mean aggregation.

    This implementation intentionally supports the tensor/float/basis path used
    by the Freebase node-classification model. Relation degrees are accumulated
    across the complete relation before normalization, so chunks do not alter
    the mathematical RGCN operation.
    """

    def __init__(self, *args, edge_chunk_size: int, **kwargs) -> None:
        if edge_chunk_size <= 0:
            raise ValueError("edge_chunk_size must be positive")
        super().__init__(*args, **kwargs)
        if self.aggr != "mean":
            raise ValueError("ChunkedRGCNConv currently requires aggr='mean'")
        if self.num_blocks is not None:
            raise ValueError("ChunkedRGCNConv does not support num_blocks")
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
        # PyG also supports bipartite features and node-index tensors. The
        # Freebase path uses one dense feature row per graph node, which is the
        # only path this memory-bounded implementation intentionally supports.
        num_nodes = x.shape[0]

        weight = self.weight
        if self.num_bases is not None:
            weight = (
                self.comp @ weight.view(self.num_bases, -1)
            ).view(self.num_relations, self.in_channels_l, self.out_channels)

        out = x.new_zeros((num_nodes, self.out_channels))

        for relation_id in range(self.num_relations):
            relation_edge_ids = torch.nonzero(
                edge_type == relation_id, as_tuple=False
            ).flatten()
            if relation_edge_ids.numel() == 0:
                continue

            relation_sum = x.new_zeros((num_nodes, self.in_channels_l))
            relation_degree = x.new_zeros((num_nodes, 1))

            for start in range(0, relation_edge_ids.numel(), self.edge_chunk_size):
                chunk_ids = relation_edge_ids[start : start + self.edge_chunk_size]
                source = edge_index[0, chunk_ids]
                target = edge_index[1, chunk_ids]
                relation_sum.index_add_(0, target, x[source])
                relation_degree.index_add_(
                    0,
                    target,
                    torch.ones(
                        (target.numel(), 1),
                        dtype=x.dtype,
                        device=x.device,
                    ),
                )

            relation_mean = relation_sum / relation_degree.clamp_min_(1.0)
            out = out + relation_mean @ weight[relation_id]

        if self.root is not None:
            out = out + x @ self.root
        if self.bias is not None:
            out = out + self.bias
        return out
