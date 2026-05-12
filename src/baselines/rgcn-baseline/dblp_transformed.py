import os
import torch
import dgl
import scipy.io as sio
from collections import defaultdict
from dgl.data import DGLDataset
from dgl.data.utils import save_graphs, load_graphs, generate_mask_tensor, idx2mask

def create_transformed_dblp():
    def read_txt(filename):
        with open(filename, 'r', errors='ignore') as f:
            lines = f.read().strip().splitlines()
        data = [line.split('\t') for line in lines if line.strip()]
        return data

    # read authors
    authors = read_txt("data/author.txt")
    author_ids = [int(x[0]) for x in authors]
    unique_authors = sorted(set(author_ids))
    author2idx = {aid: idx for idx, aid in enumerate(unique_authors)}

    # read areas
    confs = read_txt("data/conf.txt")
    conf_ids = [int(x[0]) for x in confs]
    unique_confs = sorted(set(conf_ids))
    conf2idx = {cid: idx for idx, cid in enumerate(unique_confs)}

    # read papers
    papers = read_txt("data/paper.txt")
    paper_ids = [int(x[0]) for x in papers]
    unique_papers = sorted(set(paper_ids))
    paper2idx = {pid: idx for idx, pid in enumerate(unique_papers)}

    # add edges
    pa_data = read_txt("data/paper_author.txt")
    pa_src, pa_dst = [], []
    for row in pa_data:
        paper_id = int(row[0])
        author_id = int(row[1])
        if paper_id in paper2idx and author_id in author2idx:
            pa_src.append(paper2idx[paper_id])
            pa_dst.append(author2idx[author_id])
    pa_src = torch.tensor(pa_src)
    pa_dst = torch.tensor(pa_dst)

    # add paper_conf edges
    pc_data = read_txt("data/paper_conf.txt")
    pc_src, pc_dst = [], []
    for row in pc_data:
        paper_id = int(row[0])
        conf_id = int(row[1])
        if paper_id in paper2idx and conf_id in conf2idx:
            pc_src.append(paper2idx[paper_id])
            pc_dst.append(conf2idx[conf_id])
    pc_src = torch.tensor(pc_src)
    pc_dst = torch.tensor(pc_dst)

    # add transformed paper->area edges
    paper_to_conf = {}
    for i in range(len(pc_src)):
        pid = int(pc_src[i])
        conf_idx = int(pc_dst[i])
        paper_to_conf[pid] = conf_idx

    paper_to_authors = defaultdict(list)
    for i in range(len(pa_src)):
        pid = int(pa_src[i])
        author_idx = int(pa_dst[i])
        paper_to_authors[pid].append(author_idx)

    ac_src, ac_dst = [], []
    for pid, conf_idx in paper_to_conf.items():
        if pid in paper_to_authors:
            for author_idx in paper_to_authors[pid]:
                ac_src.append(author_idx)
                ac_dst.append(conf_idx)
    ac_src = torch.tensor(ac_src)
    ac_dst = torch.tensor(ac_dst)

    data_dict = {
        ('paper', 'written_by', 'author'): (pa_src, pa_dst),
        ('author', 'writes', 'paper'): (pa_dst, pa_src),
        ('paper', 'published_in', 'conf'): (pc_src, pc_dst),
        ('conf', 'publishes', 'paper'): (pc_dst, pc_src),
        ('author', 'affiliated_with', 'conf'): (ac_src, ac_dst),
        ('conf', 'affiliates', 'author'): (ac_dst, ac_src)
    }
    hg = dgl.heterograph(data_dict)
    return hg

class DBLP4057DatasetTransformed(DGLDataset):
    """
    The DBLP dataset processed into a heterogeneous graph using text files.
    This dataset now includes derived direct Author -> Conf (area) edges.
    """
    def __init__(self):
        super().__init__('DBLP4057', url='https://pan.baidu.com/s/1Qr2e97MofXsBhUvQqgJqDg')

    def save(self):
        save_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'), self.gs)

    def load(self):
        self.gs, _ = load_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'))
        for g in self.gs:
            for key in ('train_mask', 'val_mask', 'test_mask'):
                for ntype in g.ntypes:
                    if key in g.nodes[ntype].data:
                        g.nodes[ntype].data[key] = g.nodes[ntype].data[key].bool()

    def process(self):
        hg = create_transformed_dblp()

        for ntype in hg.ntypes:
            num_nodes = hg.num_nodes(ntype)
            hg.nodes[ntype].data['feat'] = torch.randn(num_nodes, 10)

        if 'author' in hg.ntypes:
            hg.nodes['author'].data['author_features'] = hg.nodes['author'].data['feat']

        if 'author' in hg.ntypes:
            num_authors = hg.num_nodes('author')
            hg.nodes['author'].data['label'] = torch.randint(0, 4, (num_authors,))
            perm = torch.randperm(num_authors)
            train_mask = torch.zeros(num_authors, dtype=torch.bool)
            val_mask = torch.zeros(num_authors, dtype=torch.bool)
            test_mask = torch.zeros(num_authors, dtype=torch.bool)
            train_mask[perm[:int(0.6 * num_authors)]] = True
            val_mask[perm[int(0.6 * num_authors):int(0.8 * num_authors)]] = True
            test_mask[perm[int(0.8 * num_authors):]] = True
            hg.nodes['author'].data['train_mask'] = train_mask
            hg.nodes['author'].data['val_mask'] = val_mask
            hg.nodes['author'].data['test_mask'] = test_mask

        self.gs = [hg]


    def has_cache(self):
        return os.path.exists(os.path.join(self.save_path, self.name + '_dgl_graph.bin'))

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError('This dataset has only one graph')
        return self.gs

    def __len__(self):
        return 1

    @property
    def num_classes(self):
        return 4