import numpy as np
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import scipy.io as sio
from dblp import DBLP4057Dataset
from dblp_transformed import DBLP4057DatasetTransformed
import torch
from torch_geometric.data import Data
from torch_geometric.data import HeteroData
from dblp_area import DBLP_areaDataset
from dblp_area_transformed import DBLP_areaTransformedDataset


def load_data_RGCN(dataset):
    if dataset == "DBLP":
        data = DBLP4057Dataset()
    elif dataset == "DBLP_area":
        data = DBLP_areaDataset()
    elif dataset == "DBLP_area_transformed":
        data = DBLP_areaTransformedDataset()
    else:
        return NotImplementedError("Unsupported dataset {}".format(dataset))
    data.process()
    graphs = data[0]

    x, edge_index, edge_type, train_mask, val_mask, test_mask = merge_dgl_graphs(graphs)
    pyg_data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    pyg_data.train_mask = train_mask
    pyg_data.val_mask = val_mask
    pyg_data.test_mask = test_mask
    pyg_data.y = graphs[0].ndata['label']

    # Attach statistics
    pyg_data.node_counts = {
        'paper': data.num_paper,
        'author': data.num_author,
        'term': data.num_term,
        'area': data.num_area
    }
    pyg_data.edge_counts = data.num_edges
    if hasattr(data, 'label_counts'):
        pyg_data.label_counts = data.label_counts


    return pyg_data






def merge_dgl_graphs(graphs):
    # Create empty lists to hold node and edge data
    edge_src, edge_dst, edge_type = [], [], []
    x = []
    train_mask, val_mask, test_mask = [], [], []

    for etype, graph in enumerate(graphs):
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


def load_data_HAN(dataset):
    if dataset == "DBLP":
        return load_dblp()
    else:
        return NotImplementedError("Unsupported dataset {}".format(dataset))


def load_dblp(path='../MAGNN/data/raw/DBLP/DBLP4057_GAT_with_idx.mat'):
    # Load the preprocessed DBLP the author provided.
    data = sio.loadmat(path)

    features, labels = data['features'], data['label']
    # Three adjacency matrix for three meta paths the dataset contains.
    net_APTPA, net_APCPA, net_APA = data['net_APTPA'], data['net_APCPA'], data['net_APA']
    idx_train, idx_val, idx_test = data['train_idx'], data['val_idx'], data['test_idx']

    net_APTPA = torch.LongTensor(net_APTPA)
    net_APCPA = torch.LongTensor(net_APCPA)
    net_APA = torch.LongTensor(net_APA)
    meta_path_list = [net_APTPA, net_APCPA, net_APA]

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train).squeeze()
    idx_val = torch.LongTensor(idx_val).squeeze()
    idx_test = torch.LongTensor(idx_test).squeeze()

    return features, meta_path_list, labels, idx_train, idx_val, idx_test


def accuracy(preds, labels):
    _, indices = torch.max(preds, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    if len(prediction) == 0:
        return 0.0

    correct = (prediction == labels).sum()
    return correct / max(len(prediction), 1)



def knn_classifier(X, y, seed=42, k=5, time=10):
    """
    Adapted from Jhy1993/HAN
    A typical KNN classifier
    """
    for split in [0.2, 0.4, 0.6, 0.8]:
        micro_list = []
        macro_list = []
        for i in range(time):
            # Shuffle the test set to make sure each time difference sets of data are selected.
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - split, random_state=seed)

            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            f1_micro = f1_score(y_test, y_pred, average='micro')
            macro_list.append(f1_macro)
            micro_list.append(f1_micro)

        print('KNN({}avg, split:{}, k={}) f1_macro: {:.4f}, f1_micro: {:.4f}'.format(
            time, split, k, sum(macro_list) / len(macro_list), sum(micro_list) / len(micro_list)))

