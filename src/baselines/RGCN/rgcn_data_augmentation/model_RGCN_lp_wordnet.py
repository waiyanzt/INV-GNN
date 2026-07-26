"""
WordNet RGCN link prediction (Schlichtkrull et al. 2018–style settings).

Single encoding layer, 200-dim embeddings, basis decomposition (2 bases),
normalization by c_i = sum_r |N_i^r| (same scalar for all relations at node i),
edge dropout on neighbor edges, separate stochastic scaling of the root (self)
connection. DistMult decoder; L2 on relation embeddings is applied in the
optimizer (weight_decay on rel_emb only), not in this file.

Neighbor aggregation:
  out_i = (1 / c_i) * sum_{(j,r) -> i} W_r x_j  +  root_scale * (x_i @ Theta_root)
where c_i is the number of incoming edges to i in the current (dropout) graph.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import scatter


class WordNetRGCNLayer(nn.Module):
    """One RGCN layer with basis decomposition and node-wise c_i normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        num_bases: int = 2,
        bias: bool = True,
    ):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("RGCN channel sizes must be positive")
        if num_relations <= 0:
            raise ValueError("num_relations must be positive")
        if num_bases <= 0:
            raise ValueError("num_bases must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.weight = nn.Parameter(torch.empty(num_bases, in_channels, out_channels))
        self.comp = nn.Parameter(torch.empty(num_relations, num_bases))
        self.root = nn.Parameter(torch.empty(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.comp)
        nn.init.xavier_uniform_(self.root)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        c_i: torch.Tensor | None = None,
        root_scale: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, in_channels)
            edge_index: (2, E) with edge_index[0]=source j, edge_index[1]=target i
            edge_type: (E,)
            c_i: (N,) optional; if None, computed as incoming edge counts per target
            root_scale: scalar or (1,) tensor multiplier on the root connection
        """
        N = x.size(0)
        device, dtype = x.device, x.dtype

        row, col = edge_index
        if row.numel() == 0:
            neigh = x.new_zeros(N, self.out_channels)
        else:
            # Vectorized basis-decomposition message passing.
            #
            # The original formulation iterates over relations:
            #     for r: neigh[i] += sum_{e: type=r, col=i} x[row[e]] @ W[r]
            # with W[r] = sum_b comp[r, b] * weight[b]. With many relations,
            # that loop is dominated by Python and CUDA-launch overhead, not
            # FLOPs.
            #
            # Substituting the basis decomposition and swapping the sum order:
            #     neigh[i] = sum_b ( sum_{e: col=i} comp[type[e], b] * x[row[e]] ) @ weight[b]
            # so we can iterate over the (small) basis dimension B instead.
            # For each basis we (1) scale source embeddings by comp[type[e], b],
            # (2) scatter-sum into target nodes (still in input space), and
            # (3) project once by weight[b]. This trades R kernel launches
            # for num_bases of them, and avoids materializing either the full
            # (R, in, out) weight tensor or any (E, in, out) edge-wise tensor.
            coeffs = self.comp[edge_type]                          # (E, B)
            src_x = x[row]                                         # (E, in)
            neigh = x.new_zeros(N, self.out_channels)
            for b in range(self.num_bases):
                weighted = coeffs[:, b : b + 1] * src_x            # (E, in)
                agg_b = scatter(
                    weighted, col, dim=0, dim_size=N, reduce="sum"
                )                                                  # (N, in)
                neigh = neigh + agg_b @ self.weight[b]             # (N, out)

        if c_i is None:
            ones = torch.ones(col.size(0), device=device, dtype=dtype)
            deg = scatter(ones, col, dim=0, dim_size=N, reduce="sum")
            c_i = deg.clamp(min=1.0)
        else:
            c_i = c_i.clamp(min=1.0)

        out = neigh / c_i.unsqueeze(-1)

        if isinstance(root_scale, torch.Tensor):
            rs = root_scale.to(device=device, dtype=dtype)
        else:
            rs = torch.tensor(root_scale, device=device, dtype=dtype)
        if rs.dim() == 0:
            rs = rs.view(1)
        out = out + rs.view(-1, 1) * (x @ self.root)

        if self.bias is not None:
            out = out + self.bias
        return out


class WordNetRGCNLinkPredictor(nn.Module):
    """Single-layer RGCN encoder + DistMult scores (logits)."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        hidden_dim: int = 200,
        num_bases: int = 2,
        edge_dropout_other: float = 0.4,
        root_dropout_loop: float = 0.2,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.edge_dropout_other = edge_dropout_other
        self.root_dropout_loop = root_dropout_loop

        self.entity_emb = nn.Embedding(num_entities, hidden_dim)
        self.rel_emb = nn.Embedding(num_relations, hidden_dim)
        self.conv = WordNetRGCNLayer(
            hidden_dim, hidden_dim, num_relations, num_bases=num_bases
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def _drop_edges(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Edge dropout on neighbor edges (keep prob = 1 - edge_dropout_other)."""
        if not training or self.edge_dropout_other <= 0:
            row, col = edge_index
            device = edge_index.device
            dtype = edge_index.dtype
            ones = torch.ones(col.size(0), device=device, dtype=torch.float32)
            c_i = scatter(ones, col, dim=0, dim_size=self.num_entities, reduce="sum")
            return edge_index, edge_type, c_i.clamp(min=1.0)

        E = edge_index.size(1)
        device = edge_index.device
        keep_prob = 1.0 - self.edge_dropout_other
        mask = torch.rand(E, device=device) < keep_prob
        ei = edge_index[:, mask]
        et = edge_type[mask]
        row, col = ei
        if col.numel() == 0:
            c_i = torch.ones(self.num_entities, device=device)
        else:
            ones = torch.ones(col.size(0), device=device)
            c_i = scatter(ones, col, dim=0, dim_size=self.num_entities, reduce="sum")
            c_i = c_i.clamp(min=1.0)
        return ei, et, c_i

    def encode(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        training: bool = False,
    ) -> torch.Tensor:
        x = self.entity_emb.weight
        ei, et, c_i = self._drop_edges(edge_index, edge_type, training)

        if training and self.root_dropout_loop > 0:
            root_scale = torch.bernoulli(
                torch.full(
                    (1,),
                    1.0 - self.root_dropout_loop,
                    device=x.device,
                    dtype=x.dtype,
                )
            )
        else:
            root_scale = torch.tensor(1.0, device=x.device, dtype=x.dtype)

        return self.conv(x, ei, et, c_i=c_i, root_scale=root_scale)

    def score(
        self,
        entity_embs: torch.Tensor,
        h_idx: torch.Tensor,
        r_idx: torch.Tensor,
        t_idx: torch.Tensor,
    ) -> torch.Tensor:
        h = entity_embs[h_idx]
        r = self.rel_emb(r_idx)
        t = entity_embs[t_idx]
        return (h * r * t).sum(dim=-1)

    def forward(self, edge_index, edge_type, h_idx, r_idx, t_idx, training: bool = False):
        entity_embs = self.encode(edge_index, edge_type, training=training)
        return self.score(entity_embs, h_idx, r_idx, t_idx)
