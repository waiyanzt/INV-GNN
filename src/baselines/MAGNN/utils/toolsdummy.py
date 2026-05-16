import sys
import numpy as np


def parse_adjlist(adj_rows, edge_idx_rows, samples=None):
    """
    Given:
      adj_rows       : list of strings "u v1 v2 …"
      edge_idx_rows  : list of np.ndarray of shape (#neighbors, path_length)
      samples        : int or None

    Returns:
      remapped_edges : list of (u_local, v_local)
      stacked        : np.ndarray stacked path‐indices
      num_nodes      : size of the induced subgraph
      mapping        : global→local dict
    """
    edges = []
    idx_chunks = []
    nodes = set()

    for row_str, idx_arr in zip(adj_rows, edge_idx_rows):
        parts = list(map(int, row_str.split()))
        src = parts[0]
        dsts = parts[1:]

        nodes.add(src)

        if samples is not None and len(dsts) > samples:
            try:
                choice = np.random.choice(len(dsts), samples, replace=False)
                dsts = [dsts[i] for i in choice]
                idx_arr = idx_arr[choice]
            except Exception:
                print(f"[WARN] cannot sample {samples} from {len(dsts)} neighbors of {src}",
                      file=sys.stderr)

        for dst in dsts:
            nodes.add(dst)
            edges.append((src, dst))

        idx_chunks.append(idx_arr)

    sorted_nodes = sorted(nodes)
    mapping = {n: i for i, n in enumerate(sorted_nodes)}
    edges = [(mapping[s], mapping[d]) for s, d in edges]

    if idx_chunks:
        stacked = np.vstack(idx_chunks)
    else:
        path_len = idx_chunks[0].shape[1] if idx_chunks else 0
        stacked = np.empty((0, path_len), dtype=int)

    return edges, stacked, len(nodes), mapping


def parse_minibatch_DBLP(adj_rows_list, edge_idx_rows_list, samples=None):
    """
    Original signature:
      adj_rows_list       : list of lists of "u v1 v2 …" strings, one per metapath
      edge_idx_rows_list  : list of lists of np.ndarray, one per metapath
      samples             : int or None

    Calls parse_adjlist per metapath and collects results.
    """
    edges_list = []
    stacked_list = []
    num_nodes_list = []
    mapping_list = []

    for adj_rows, edge_idx_rows in zip(adj_rows_list, edge_idx_rows_list):
        edges, stacked, num_nodes, mapping = parse_adjlist(adj_rows, edge_idx_rows, samples)
        edges_list.append(edges)
        stacked_list.append(stacked)
        num_nodes_list.append(num_nodes)
        mapping_list.append(mapping)

    return edges_list, stacked_list, num_nodes_list, mapping_list


class index_generator:
    def __init__(self, batch_size, num_data=None, indices=None, shuffle=True):
        if num_data is not None:
            self.indices = np.arange(num_data)
        if indices is not None:
            self.indices = np.array(indices)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.reset()

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.ptr = 0

    def next(self):
        if self.ptr >= len(self.indices):
            self.reset()
        end = min(self.ptr + self.batch_size, len(self.indices))
        batch = self.indices[self.ptr:end]
        self.ptr = end
        return batch

    def num_iterations(self):
        return int(np.ceil(len(self.indices) / self.batch_size))
