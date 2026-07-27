"""SlotGAT encoder with a DistMult decoder for WordNet link prediction.

WordNet has a single node type, so SlotGAT uses one semantic slot.  The graph
variants still differ through their directed edges and globally shared edge
relation IDs.  A dedicated final edge type is reserved for structural
self-loops; it is not part of the DistMult relation vocabulary.
"""

from __future__ import annotations

from typing import Dict, Tuple

import dgl
import torch
import torch.nn as nn

from model.conv import slotGATConv


class WordNetSlotGATLinkPredictor(nn.Module):
    """Featureless SlotGAT encoder plus DistMult relation scoring."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        hidden_dim: int = 200,
        num_layers: int = 2,
        num_heads: int = 8,
        edge_feats: int = 64,
        dropout_feat: float = 0.5,
        dropout_attn: float = 0.2,
        negative_slope: float = 0.05,
        alpha: float = 0.05,
    ) -> None:
        super().__init__()
        if num_entities <= 0 or num_relations <= 0:
            raise ValueError("num_entities and num_relations must be positive")
        if hidden_dim <= 0 or num_layers <= 0 or num_heads <= 0:
            raise ValueError("hidden_dim, num_layers, and num_heads must be positive")
        if edge_feats < 0:
            raise ValueError("edge_feats must be nonnegative")

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_edge_types = num_relations + 1
        self.self_loop_edge_type = num_relations
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.entity_emb = nn.Embedding(num_entities, hidden_dim)
        self.rel_emb = nn.Embedding(num_relations, hidden_dim)
        self.gat_layers = nn.ModuleList()

        recorder = {
            "meta": {
                "getSAAttentionScore": "False",
                "retainLayerAttention": False,
            },
            "data": {},
            "status": "None",
        }
        self.data_recorder = recorder

        # Match the SlotGAT node-classification encoder: `num_layers` hidden
        # multi-head layers followed by a one-head output projection.
        self.gat_layers.append(
            slotGATConv(
                edge_feats,
                self.num_edge_types,
                hidden_dim,
                hidden_dim,
                num_heads,
                dropout_feat,
                dropout_attn,
                negative_slope,
                False,
                torch.nn.functional.elu,
                alpha=alpha,
                num_ntype=1,
                eindexer=None,
                inputhead=True,
                dataRecorder=recorder,
            )
        )
        for _ in range(1, num_layers):
            self.gat_layers.append(
                slotGATConv(
                    edge_feats,
                    self.num_edge_types,
                    hidden_dim * num_heads,
                    hidden_dim,
                    num_heads,
                    dropout_feat,
                    dropout_attn,
                    negative_slope,
                    True,
                    torch.nn.functional.elu,
                    alpha=alpha,
                    num_ntype=1,
                    eindexer=None,
                    dataRecorder=recorder,
                )
            )
        self.output_layer = slotGATConv(
            edge_feats,
            self.num_edge_types,
            hidden_dim * num_heads,
            hidden_dim,
            1,
            dropout_feat,
            dropout_attn,
            negative_slope,
            True,
            None,
            alpha=alpha,
            num_ntype=1,
            eindexer=None,
            dataRecorder=recorder,
        )

        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

        # DGL graphs are execution artifacts, not learned state.  They are
        # created lazily per input tensor pair and intentionally excluded from
        # checkpoints/state_dicts.
        self._graph_cache: Dict[
            Tuple[int, int, int, str], Tuple[dgl.DGLGraph, torch.Tensor]
        ] = {}

    def _graph_and_edge_features(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> Tuple[dgl.DGLGraph, torch.Tensor]:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E)")
        if edge_type.ndim != 1 or edge_type.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_type must have shape (E,)")
        if edge_type.numel() and (
            int(edge_type.min()) < 0 or int(edge_type.max()) >= self.num_relations
        ):
            raise ValueError("edge_type contains an ID outside the decoder vocabulary")

        device_key = str(edge_index.device)
        cache_key = (
            int(edge_index.data_ptr()),
            int(edge_type.data_ptr()),
            int(edge_index.shape[1]),
            device_key,
        )
        cached = self._graph_cache.get(cache_key)
        if cached is not None:
            return cached

        node_ids = torch.arange(self.num_entities, device=edge_index.device)
        src = torch.cat((edge_index[0], node_ids))
        dst = torch.cat((edge_index[1], node_ids))
        self_types = torch.full(
            (self.num_entities,),
            self.self_loop_edge_type,
            dtype=edge_type.dtype,
            device=edge_type.device,
        )
        edge_features = torch.cat((edge_type, self_types))
        graph = dgl.graph(
            (src, dst),
            num_nodes=self.num_entities,
            device=edge_index.device,
        )
        graph.num_ntypes = 1
        self._graph_cache[cache_key] = (graph, edge_features)
        return graph, edge_features

    def encode(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        training: bool = False,
    ) -> torch.Tensor:
        # `training` is retained for interface compatibility with the shared
        # WordNet protocol. Dropout behavior follows module.train()/eval().
        del training
        graph, edge_features = self._graph_and_edge_features(edge_index, edge_type)
        hidden = self.entity_emb.weight
        residual_attention = None
        for layer in self.gat_layers:
            hidden, residual_attention = layer(
                graph,
                hidden,
                edge_features,
                res_attn=residual_attention,
            )
            hidden = hidden.flatten(1)
        hidden, _ = self.output_layer(
            graph,
            hidden,
            edge_features,
            res_attn=None,
        )
        return hidden.mean(dim=1)

    def score(
        self,
        entity_embs: torch.Tensor,
        h_idx: torch.Tensor,
        r_idx: torch.Tensor,
        t_idx: torch.Tensor,
    ) -> torch.Tensor:
        heads = entity_embs[h_idx]
        relations = self.rel_emb(r_idx)
        tails = entity_embs[t_idx]
        return (heads * relations * tails).sum(dim=-1)

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        h_idx: torch.Tensor,
        r_idx: torch.Tensor,
        t_idx: torch.Tensor,
        training: bool = False,
    ) -> torch.Tensor:
        embeddings = self.encode(edge_index, edge_type, training=training)
        return self.score(embeddings, h_idx, r_idx, t_idx)
