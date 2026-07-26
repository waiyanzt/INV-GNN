"""Featureless 2-layer RGCN for Freebase node classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

try:
    from .chunked_rgcn_conv import ChunkedRGCNConv
except ImportError:
    # Direct script execution places this directory on sys.path.
    from chunked_rgcn_conv import ChunkedRGCNConv


class RGCNFeatureless(nn.Module):
    def __init__(
        self,
        num_nodes,
        num_relations,
        hidden_dim,
        num_classes,
        num_bases=30,
        dropout=0.0,
        edge_chunk_size=0,
    ):
        super().__init__()
        self.embed = nn.Parameter(torch.empty(num_nodes, hidden_dim))
        nn.init.xavier_uniform_(self.embed)
        if edge_chunk_size > 0:
            self.conv1 = ChunkedRGCNConv(
                hidden_dim,
                hidden_dim,
                num_relations,
                num_bases=num_bases,
                edge_chunk_size=edge_chunk_size,
            )
            self.conv2 = ChunkedRGCNConv(
                hidden_dim,
                num_classes,
                num_relations,
                num_bases=num_bases,
                edge_chunk_size=edge_chunk_size,
            )
        else:
            self.conv1 = RGCNConv(
                hidden_dim,
                hidden_dim,
                num_relations,
                num_bases=num_bases,
            )
            self.conv2 = RGCNConv(
                hidden_dim,
                num_classes,
                num_relations,
                num_bases=num_bases,
            )
        self.dropout = dropout
        self.edge_chunk_size = int(edge_chunk_size)

    def forward(self, edge_index, edge_type):
        x = F.relu(self.conv1(self.embed, edge_index, edge_type))
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return F.log_softmax(x, dim=1)
