import os
import dgl
import torch
import pandas as pd
from dgl.data import DGLDataset

class DBLP_areaDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='DBLP_area')

    def process(self):
        path = '../MAGNN/data/csv/DBLP_area/'

        paper = pd.read_csv(os.path.join(path, 'paper.csv'))
        paper = paper[paper['label'] != -1].reset_index(drop=True)
        author = pd.read_csv(os.path.join(path, 'author.csv'))
        term = pd.read_csv(os.path.join(path, 'term.csv'))
        area = pd.read_csv(os.path.join(path, 'area.csv'))

        paper_author = pd.read_csv(os.path.join(path, 'paper_author.csv'))
        paper_term = pd.read_csv(os.path.join(path, 'paper_term.csv'))
        paper_area = pd.read_csv(os.path.join(path, 'paper_area.csv'))

        num_paper = len(paper)
        num_author = len(author)
        num_term = len(term)
        num_area = len(area)
        num_nodes = num_paper + num_author + num_term + num_area

        src = []
        dst = []

        for idx, row in paper_author.iterrows():
            src.append(row['src_id'])
            dst.append(row['dst_id'])
        for idx, row in paper_term.iterrows():
            src.append(row['src_id'])
            dst.append(row['dst_id'])
        for idx, row in paper_area.iterrows():
            src.append(row['src_id'])
            dst.append(row['dst_id'])

        g = dgl.graph((src, dst), num_nodes=num_nodes)

        feat_dim = len(paper['feat'].iloc[0].split(','))
        features = torch.zeros((num_nodes, feat_dim))


        # Load paper features
        paper_features = torch.tensor(
            [list(map(float, f.split(','))) for f in paper['feat']],
            dtype=torch.float32
        )
        features[0:num_paper] = paper_features

        g.ndata['feat'] = features

        # Labels (papers only)
        labels = torch.full((num_nodes,), -1, dtype=torch.long)
        paper_labels = torch.tensor(paper['label'].values, dtype=torch.long)
        labels[0:num_paper] = paper_labels
        g.ndata['label'] = labels

        # Masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[0:num_paper] = torch.tensor(paper['train_mask'], dtype=torch.bool)
        val_mask[0:num_paper] = torch.tensor(paper['val_mask'], dtype=torch.bool)
        test_mask[0:num_paper] = torch.tensor(paper['test_mask'], dtype=torch.bool)

        g.ndata['train_mask'] = train_mask
        g.ndata['val_mask'] = val_mask
        g.ndata['test_mask'] = test_mask

        self.gs = [g]
        self.num_paper = num_paper
        self.num_author = num_author
        self.num_term = num_term
        self.num_area = num_area
        self.num_edges = {
            'paper_author': len(paper_author),
            'paper_term': len(paper_term),
            'paper_area': len(paper_area)
        }
        self.label_counts = paper['label'].value_counts().to_dict()


        

    def __getitem__(self, idx):
        return self.gs

    def __len__(self):
        return 1
