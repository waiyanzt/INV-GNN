import numpy as np
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import scipy.io as sio
#from dblp import DBLP4057Dataset
#from dblp_transformed import DBLP4057DatasetTransformed
import torch
#from torch_geometric.data import Data
#from torch_geometric.data import HeteroData

import os
import dgl
from dgl.data import DGLDataset
from dgl.data.utils import save_graphs, load_graphs, generate_mask_tensor, idx2mask

dataset = dgl.data.CSVDataset('./data/csv/DBLP_area')
g = dataset[0]  # only one graph
print(g)

dataset = dgl.data.CSVDataset('./data/csv/DBLP_area_transformed')
graph = dataset[0]  # only one graph
print(graph)

node_features = graph.ndata['feat']
node_labels = graph.ndata['label']
train_mask = graph.ndata['train_mask']
valid_mask = graph.ndata['val_mask']
test_mask = graph.ndata['test_mask']
#n_features = [ node_features[nf].shape[1] for nf in node_features]
#n_labels = int(node_labels.max().item() + 1)



"""
def load_data_RGCN(dataset):
    if dataset == "DBLP":
        data = DBLP4057Dataset()
    elif dataset == "DBLP_Transformed":
        data = DBLP4057DatasetTransformed()
    else:
        return NotImplementedError("Unsupported dataset {}".format(dataset))
    data.process()
    graphs = data[0]
    # Merge graphs and prepare data for PyTorch Geometric
    x, edge_index, edge_type, train_mask, val_mask, test_mask = merge_dgl_graphs(graphs)

    # Create PyTorch Geometric Data object
    pyg_data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    pyg_data.train_mask = train_mask
    pyg_data.val_mask = val_mask
    pyg_data.test_mask = test_mask
    pyg_data.y = graphs[0].ndata['label']

    return pyg_data
"""

"""def merge_dgl_graphs(graphs):
    # Create empty lists to hold node and edge data
    edge_src, edge_dst, edge_type = [], [], []
    x = []
    train_mask, val_mask, test_mask = [], [], []
    for graph in graphs:
        print(graph)
    for etype, graph in enumerate(graphs):
        print(graph)
        # Append node and edge data to lists
        edge_src += graph.edges()[0].tolist()
        edge_dst += graph.edges()[1].tolist()
        edge_type += [etype] * graph.number_of_edges()

        # Assume node features and masks are the same for all graphs
        if etype == 0:
            x = graph.ndata['feat']
            train_mask = graph.ndata['train_mask']
            val_mask = graph.ndata['val_mask']
            test_mask = graph.ndata['test_mask']

    # Convert lists to PyTorch tensors
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)

    return x, edge_index, edge_type, train_mask, val_mask, test_mask

# Merge graphs and prepare data for PyTorch Geometric
x, edge_index, edge_type, train_mask, val_mask, test_mask = merge_dgl_graphs([graph])"""