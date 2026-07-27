"""SlotGAT node classifier for joint Freebase graph-variant training."""

from __future__ import annotations

from typing import Dict, List, Tuple

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import slotGAT


class FreebaseSlotGATClassifier(nn.Module):
    """Featureless SlotGAT classifier with lazily constructed variant graphs."""

    def __init__(
        self,
        node_type: torch.Tensor,
        num_relations: int,
        num_classes: int = 8,
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
    ) -> None:
        super().__init__()
        node_type = node_type.detach().cpu().long()
        if node_type.ndim != 1 or node_type.numel() == 0:
            raise ValueError("node_type must be a nonempty one-dimensional tensor")
        if num_relations <= 0 or num_classes <= 0:
            raise ValueError("num_relations and num_classes must be positive")

        unique_types = torch.unique(node_type, sorted=True)
        expected_types = torch.arange(len(unique_types))
        if not torch.equal(unique_types, expected_types):
            raise ValueError("Freebase node type IDs must be dense from zero")

        self.num_nodes = int(node_type.numel())
        self.num_ntypes = int(len(unique_types))
        self.num_relations = int(num_relations)
        self.self_loop_edge_type = int(num_relations)
        self.node_counts: List[int] = []
        cursor = 0
        for type_id in range(self.num_ntypes):
            indices = (node_type == type_id).nonzero(as_tuple=False).view(-1)
            count = int(indices.numel())
            expected = torch.arange(cursor, cursor + count)
            if not torch.equal(indices, expected):
                raise ValueError(
                    "SlotGAT requires Freebase nodes to be contiguous by node type"
                )
            self.node_counts.append(count)
            cursor += count

        recorder = {
            "meta": {
                "getSAAttentionScore": "False",
                "retainLayerAttention": False,
            },
            "data": {},
            "status": "None",
        }
        dummy_graph = dgl.graph(
            (
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
            ),
            num_nodes=self.num_nodes,
        )
        dummy_graph.num_ntypes = self.num_ntypes
        heads = [num_heads] * num_layers + [1]
        self.network = slotGAT(
            dummy_graph,
            edge_feats,
            num_relations + 1,
            self.node_counts,
            hidden_dim,
            num_classes,
            num_layers,
            heads,
            F.elu,
            dropout_feat,
            dropout_attn,
            negative_slope,
            True,
            alpha,
            num_ntype=self.num_ntypes,
            eindexer=None,
            aggregator=aggregator,
            SAattDim=sa_att_dim,
            dataRecorder=recorder,
            vis_data_saver=None,
        )
        self.data_recorder = recorder
        self._feature_cache: Dict[str, List[torch.Tensor]] = {}
        self._graph_cache: Dict[
            Tuple[int, int, int, str], Tuple[dgl.DGLGraph, torch.Tensor]
        ] = {}

    def _identity_features(self, device: torch.device) -> List[torch.Tensor]:
        key = str(device)
        cached = self._feature_cache.get(key)
        if cached is not None:
            return cached
        features = []
        for count in self.node_counts:
            diagonal = torch.arange(count, device=device)
            indices = torch.stack((diagonal, diagonal))
            values = torch.ones(count, device=device)
            features.append(
                torch.sparse_coo_tensor(
                    indices,
                    values,
                    (count, count),
                    device=device,
                ).coalesce()
            )
        self._feature_cache[key] = features
        return features

    def _graph_and_edge_features(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> Tuple[dgl.DGLGraph, torch.Tensor]:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E)")
        if edge_type.ndim != 1 or edge_type.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_type must have shape (E,)")
        cache_key = (
            int(edge_index.data_ptr()),
            int(edge_type.data_ptr()),
            int(edge_index.shape[1]),
            str(edge_index.device),
        )
        cached = self._graph_cache.get(cache_key)
        if cached is not None:
            return cached

        # The reference runner removes graph self-loops and then adds one
        # structural self-loop per node. Preserve that behavior while retaining
        # all non-self directed relation edges from the globally aligned
        # augmentation preprocessing.
        non_self = edge_index[0] != edge_index[1]
        src = edge_index[0, non_self]
        dst = edge_index[1, non_self]
        relation_types = edge_type[non_self]
        node_ids = torch.arange(self.num_nodes, device=edge_index.device)
        src = torch.cat((src, node_ids))
        dst = torch.cat((dst, node_ids))
        structural_types = torch.full(
            (self.num_nodes,),
            self.self_loop_edge_type,
            dtype=edge_type.dtype,
            device=edge_type.device,
        )
        graph_edge_types = torch.cat((relation_types, structural_types))
        graph = dgl.graph(
            (src, dst),
            num_nodes=self.num_nodes,
            device=edge_index.device,
        )
        graph.num_ntypes = self.num_ntypes
        self._graph_cache[cache_key] = (graph, graph_edge_types)
        return graph, graph_edge_types

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        graph, graph_edge_types = self._graph_and_edge_features(
            edge_index, edge_type
        )
        self.network.g = graph
        logits, _ = self.network(
            self._identity_features(edge_index.device),
            graph_edge_types,
        )
        return F.log_softmax(logits, dim=1)
