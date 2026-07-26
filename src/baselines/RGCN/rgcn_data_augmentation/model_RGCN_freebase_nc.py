"""Featureless 2-layer RGCN for Freebase node classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class RGCNFeatureless(nn.Module):
    def __init__(self, num_nodes, num_relations, hidden_dim, num_classes, num_bases=30, dropout=0.0):
        super().__init__()
        self.embed = nn.Parameter(torch.empty(num_nodes, hidden_dim))
        nn.init.xavier_uniform_(self.embed)
        self.conv1 = RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_dim, num_classes, num_relations, num_bases=num_bases)
        self.dropout = dropout

    def forward(self, edge_index, edge_type):
        x = F.relu(self.conv1(self.embed, edge_index, edge_type))
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return F.log_softmax(x, dim=1)
