import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class RGCNLinkPredictor(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_relations, num_bases=30):
        super(RGCNLinkPredictor, self).__init__()
        self.conv1 = RGCNConv(in_dim, hidden_dim, num_relations, num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_dim, out_dim, num_relations, num_bases=num_bases)

    def forward(self, x, edge_index, edge_type):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        return x  # node embeddings


def dot_product_score(x_src, x_dst):
    return (x_src * x_dst).sum(dim=1)
