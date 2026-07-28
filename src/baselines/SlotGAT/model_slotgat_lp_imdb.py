"""Multi-node-type SlotGAT encoder for IMDb link prediction.

The shared IMDb-LP protocol supplies a homogeneous DGL graph with globally
aligned native relation IDs. This module adds a dedicated structural self-loop
edge type, encodes every node in its own semantic type slot, and returns one
embedding per node for the protocol's movie-tail dot-product decoder.
"""

from __future__ import annotations

import math
from typing import Tuple

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.conv import slotGATConv


class IMDbSlotGATEncoder(nn.Module):
    """Featureless, multi-type SlotGAT encoder for IMDb md/ml prediction."""

    def __init__(
        self,
        node_type: torch.Tensor,
        num_relations: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 8,
        edge_feats: int = 64,
        dropout_feat: float = 0.5,
        dropout_attn: float = 0.2,
        negative_slope: float = 0.05,
        alpha: float = 0.05,
        aggregator: str = "SA",
        sa_att_dim: int = 3,
        edge_chunk_size: int = 0,
        decomposed_layers: int = 1,
    ) -> None:
        super().__init__()
        node_type = node_type.detach().cpu().long()
        if node_type.ndim != 1 or node_type.numel() == 0:
            raise ValueError("node_type must be a nonempty one-dimensional tensor")
        if num_relations <= 0:
            raise ValueError("num_relations must be positive")
        if hidden_dim <= 0 or num_layers <= 0 or num_heads <= 0:
            raise ValueError(
                "hidden_dim, num_layers, and num_heads must be positive"
            )
        if edge_feats < 0 or edge_chunk_size < 0:
            raise ValueError("edge_feats and edge_chunk_size must be nonnegative")
        if decomposed_layers <= 0 or sa_att_dim <= 0:
            raise ValueError("decomposed_layers and sa_att_dim must be positive")
        if aggregator not in {"SA", "average", "max"}:
            raise ValueError("aggregator must be SA, average, or max")

        unique_types = torch.unique(node_type, sorted=True)
        expected_types = torch.arange(len(unique_types))
        if not torch.equal(unique_types, expected_types):
            raise ValueError("IMDb homogeneous node type IDs must be dense from zero")

        self.num_nodes = int(node_type.numel())
        self.num_ntypes = int(unique_types.numel())
        self.num_relations = int(num_relations)
        self.self_loop_edge_type = int(num_relations)
        self.hidden_dim = int(hidden_dim)
        self.aggregator = aggregator
        self.edge_chunk_size = int(edge_chunk_size)
        self.decomposed_layers = int(decomposed_layers)
        self.register_buffer("node_type", node_type)
        self.emb = nn.Embedding(self.num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.emb.weight)

        recorder = {
            "meta": {
                "getSAAttentionScore": "False",
                "retainLayerAttention": False,
            },
            "data": {},
            "status": "None",
        }
        self.data_recorder = recorder
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(
            slotGATConv(
                edge_feats,
                num_relations + 1,
                hidden_dim,
                hidden_dim,
                num_heads,
                dropout_feat,
                dropout_attn,
                negative_slope,
                False,
                F.elu,
                alpha=alpha,
                num_ntype=self.num_ntypes,
                eindexer=None,
                inputhead=True,
                dataRecorder=recorder,
                edge_chunk_size=edge_chunk_size,
                decomposed_layers=decomposed_layers,
            )
        )
        for _ in range(1, num_layers):
            self.gat_layers.append(
                slotGATConv(
                    edge_feats,
                    num_relations + 1,
                    hidden_dim * num_heads,
                    hidden_dim,
                    num_heads,
                    dropout_feat,
                    dropout_attn,
                    negative_slope,
                    True,
                    F.elu,
                    alpha=alpha,
                    num_ntype=self.num_ntypes,
                    eindexer=None,
                    dataRecorder=recorder,
                    edge_chunk_size=edge_chunk_size,
                    decomposed_layers=decomposed_layers,
                )
            )
        self.output_layer = slotGATConv(
            edge_feats,
            num_relations + 1,
            hidden_dim * num_heads,
            hidden_dim,
            1,
            dropout_feat,
            dropout_attn,
            negative_slope,
            True,
            None,
            alpha=alpha,
            num_ntype=self.num_ntypes,
            eindexer=None,
            dataRecorder=recorder,
            edge_chunk_size=edge_chunk_size,
            decomposed_layers=decomposed_layers,
        )

        if aggregator == "SA":
            self.macro_linear = nn.Linear(hidden_dim, sa_att_dim, bias=True)
            nn.init.xavier_normal_(self.macro_linear.weight, gain=1.414)
            nn.init.normal_(
                self.macro_linear.bias,
                std=1.414 * math.sqrt(1.0 / sa_att_dim),
            )
            self.macro_semantic_vec = nn.Parameter(
                torch.empty(sa_att_dim, 1)
            )
            nn.init.normal_(self.macro_semantic_vec, std=1.0)

    def _graph_with_structural_loops(
        self,
        graph: dgl.DGLGraph,
        edge_type: torch.Tensor,
    ) -> Tuple[dgl.DGLGraph, torch.Tensor]:
        if graph.num_nodes() != self.num_nodes:
            raise ValueError(
                f"Graph has {graph.num_nodes()} nodes; expected {self.num_nodes}"
            )
        if edge_type.ndim != 1 or edge_type.numel() != graph.num_edges():
            raise ValueError("edge_type must have one value per graph edge")
        if edge_type.numel() and (
            int(edge_type.min()) < 0
            or int(edge_type.max()) >= self.num_relations
        ):
            raise ValueError("edge_type contains an ID outside the global vocabulary")

        source, target = graph.edges(order="eid")
        non_self = source != target
        source = source[non_self]
        target = target[non_self]
        native_types = edge_type[non_self]
        node_ids = torch.arange(self.num_nodes, device=source.device)
        source = torch.cat((source, node_ids))
        target = torch.cat((target, node_ids))
        structural_types = torch.full(
            (self.num_nodes,),
            self.self_loop_edge_type,
            dtype=edge_type.dtype,
            device=edge_type.device,
        )
        graph_edge_types = torch.cat((native_types, structural_types))
        slotgat_graph = dgl.graph(
            (source, target),
            num_nodes=self.num_nodes,
            device=source.device,
        )
        slotgat_graph.num_ntypes = self.num_ntypes
        if self.edge_chunk_size > 0:
            slotgat_graph.slotgat_edge_index = torch.stack(
                (source, target), dim=0
            )
        return slotgat_graph, graph_edge_types

    def _typed_input(self) -> torch.Tensor:
        """Place each learned node feature only in its native type slot."""
        node_embeddings = self.emb.weight
        typed = node_embeddings.new_zeros(
            (self.num_nodes, self.num_ntypes, self.hidden_dim)
        )
        node_ids = torch.arange(
            self.num_nodes, device=node_embeddings.device
        )
        typed[node_ids, self.node_type] = node_embeddings
        return typed.flatten(1)

    def _aggregate_slots(self, slots: torch.Tensor) -> torch.Tensor:
        if self.aggregator == "average":
            return slots.mean(dim=1)
        if self.aggregator == "max":
            return slots.max(dim=1).values

        slot_scores = (
            torch.tanh(self.macro_linear(slots))
            @ self.macro_semantic_vec
        ).mean(dim=0, keepdim=True)
        attention = F.softmax(slot_scores, dim=1)
        self.slot_scores = attention
        return (slots * attention).sum(dim=1)

    def forward(
        self,
        graph: dgl.DGLGraph,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        slotgat_graph, graph_edge_types = self._graph_with_structural_loops(
            graph, edge_type
        )
        hidden = self._typed_input()
        residual_attention = None
        for layer in self.gat_layers:
            hidden, residual_attention = layer(
                slotgat_graph,
                hidden,
                graph_edge_types,
                res_attn=residual_attention,
            )
            hidden = hidden.flatten(1)
        hidden, _ = self.output_layer(
            slotgat_graph,
            hidden,
            graph_edge_types,
            res_attn=None,
        )
        slots = hidden.squeeze(1).view(
            self.num_nodes, self.num_ntypes, self.hidden_dim
        )
        return self._aggregate_slots(slots)
