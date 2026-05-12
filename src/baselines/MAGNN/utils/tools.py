import torch
import dgl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC

def idx_to_one_hot(idx_arr):
    one_hot = np.zeros((idx_arr.shape[0], idx_arr.max() + 1))
    one_hot[np.arange(idx_arr.shape[0]), idx_arr] = 1
    return one_hot


def kmeans_test(X, y, n_clusters, repeat=10):
    nmi_list = []
    ari_list = []
    for _ in range(repeat):
        kmeans = KMeans(n_clusters=n_clusters)
        y_pred = kmeans.fit_predict(X)
        nmi_score = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        ari_score = adjusted_rand_score(y, y_pred)
        nmi_list.append(nmi_score)
        ari_list.append(ari_score)
    return np.mean(nmi_list), np.std(nmi_list), np.mean(ari_list), np.std(ari_list)


def svm_test(X, y, test_sizes=(0.2, 0.4, 0.6, 0.8), repeat=10):
    random_states = [182318 + i for i in range(repeat)]
    result_macro_f1_list = []
    result_micro_f1_list = []
    for test_size in test_sizes:
        macro_f1_list = []
        micro_f1_list = []
        for i in range(repeat):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=True, random_state=random_states[i])
            svm = LinearSVC(dual=False)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            micro_f1 = f1_score(y_test, y_pred, average='micro')
            macro_f1_list.append(macro_f1)
            micro_f1_list.append(micro_f1)
        result_macro_f1_list.append((np.mean(macro_f1_list), np.std(macro_f1_list)))
        result_micro_f1_list.append((np.mean(micro_f1_list), np.std(micro_f1_list)))
    return result_macro_f1_list, result_micro_f1_list


def evaluate_results_nc(embeddings, labels, num_classes):
    print('SVM test')
    svm_macro_f1_list, svm_micro_f1_list = svm_test(embeddings, labels)
    print('Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(macro_f1_mean, macro_f1_std, train_size) for
                                    (macro_f1_mean, macro_f1_std), train_size in
                                    zip(svm_macro_f1_list, [0.8, 0.6, 0.4, 0.2])]))
    print('Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(micro_f1_mean, micro_f1_std, train_size) for
                                    (micro_f1_mean, micro_f1_std), train_size in
                                    zip(svm_micro_f1_list, [0.8, 0.6, 0.4, 0.2])]))
    print('K-means test')
    nmi_mean, nmi_std, ari_mean, ari_std = kmeans_test(embeddings, labels, num_classes)
    print('NMI: {:.6f}~{:.6f}'.format(nmi_mean, nmi_std))
    print('ARI: {:.6f}~{:.6f}'.format(ari_mean, ari_std))

    return svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std


def parse_adjlist(adjlist, edge_metapath_indices, samples=None):
    edges = []
    nodes = set()
    result_indices = []
    for row, indices in zip(adjlist, edge_metapath_indices):
        row_parsed = list(map(int, row.split(' ')))
        nodes.add(row_parsed[0])
        if len(row_parsed) > 1:
            # sampling neighbors
            if samples is None:
                neighbors = row_parsed[1:]
                result_indices.append(indices)
            else:
                # undersampling frequent neighbors
                unique, counts = np.unique(row_parsed[1:], return_counts=True)
                p = []
                for count in counts:
                    p += [(count ** (3 / 4)) / count] * count
                p = np.array(p)
                p = p / p.sum()
                samples = min(samples, len(row_parsed) - 1)
                sampled_idx = np.sort(np.random.choice(len(row_parsed) - 1, samples, replace=False, p=p))
                neighbors = [row_parsed[i + 1] for i in sampled_idx]
                result_indices.append(indices[sampled_idx])
        else:
            neighbors = []
            result_indices.append(indices)
            #print("row_parsed <= 1")
        for dst in neighbors:
            nodes.add(dst)
            edges.append((row_parsed[0], dst))
    mapping = {map_from: map_to for map_to, map_from in enumerate(sorted(nodes))}
    edges = list(map(lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges))
    result_indices = np.vstack(result_indices)
    return edges, result_indices, len(nodes), mapping


def parse_minibatch(adjlists, edge_metapath_indices_list, idx_batch, device, samples=None):
    g_list = []
    result_indices_list = []
    idx_batch_mapped_list = []
    for adjlist, indices in zip(adjlists, edge_metapath_indices_list):
        edges, result_indices, num_nodes, mapping = parse_adjlist(
            [adjlist[i] for i in idx_batch], [indices[i] for i in idx_batch], samples)

        g = dgl.DGLGraph()
        #g = dgl.graph()
        g.add_nodes(num_nodes)
        ##print("num nodes:", g.num_nodes())
        if len(edges) > 0:
            sorted_index = sorted(range(len(edges)), key=lambda i : edges[i])
            
            '''
            for i in sorted_index:
                t = (edges[i][1], edges[i][0])
                #print(list(zip(*[(edges[i][1], edges[i][0]))
                g.add_edges(edges[i][1], edges[i][1])
            '''
            #el = list(zip(*[(edges[i][1], edges[i][0]) for i in sorted_index])) 
            #g.add_edges(torch.tensor(list(el[1])), torch.tensor(list(el[0])))
            g.add_edges(*list(zip(*[(edges[i][1], edges[i][0]) for i in sorted_index])))
            ##print("num_edges:", g.num_edges())
            result_indices = torch.LongTensor(result_indices[sorted_index]).to(device)
            #print("result indicies at sorted: ", result_indices)
        else:
            result_indices = torch.LongTensor(result_indices).to(device)
        #g.add_edges(*list(zip(*[(dst, src) for src, dst in sorted(edges)])))
        #result_indices = torch.LongTensor(result_indices).to(device)
        g_list.append(g)
        result_indices_list.append(result_indices)
        idx_batch_mapped_list.append(np.array([mapping[idx] for idx in idx_batch]))

    return g_list, result_indices_list, idx_batch_mapped_list


def parse_adjlist_LastFM(adjlist, edge_metapath_indices, samples=None, exclude=None, offset=None, mode=None):
    edges = []
    nodes = set()
    result_indices = []
    for row, indices in zip(adjlist, edge_metapath_indices):
        row_parsed = list(map(int, row.split(' ')))
        nodes.add(row_parsed[0])
        if len(row_parsed) > 1:
            # sampling neighbors
            if samples is None:
                if exclude is not None:
                    if mode == 0:
                        mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True for u1, a1, u2, a2 in indices[:, [0, 1, -1, -2]]]
                    else:
                        mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True for a1, u1, a2, u2 in indices[:, [0, 1, -1, -2]]]
                    neighbors = np.array(row_parsed[1:])[mask]
                    result_indices.append(indices[mask])
                else:
                    neighbors = row_parsed[1:]
                    result_indices.append(indices)
            else:
                # undersampling frequent neighbors
                unique, counts = np.unique(row_parsed[1:], return_counts=True)
                p = []
                for count in counts:
                    p += [(count ** (3 / 4)) / count] * count
                p = np.array(p)
                p = p / p.sum()
                samples = min(samples, len(row_parsed) - 1)
                sampled_idx = np.sort(np.random.choice(len(row_parsed) - 1, samples, replace=False, p=p))
                if exclude is not None:
                    if mode == 0:
                        mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True for u1, a1, u2, a2 in indices[sampled_idx][:, [0, 1, -1, -2]]]
                    else:
                        mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True for a1, u1, a2, u2 in indices[sampled_idx][:, [0, 1, -1, -2]]]
                    neighbors = np.array([row_parsed[i + 1] for i in sampled_idx])[mask]
                    result_indices.append(indices[sampled_idx][mask])
                else:
                    neighbors = [row_parsed[i + 1] for i in sampled_idx]
                    result_indices.append(indices[sampled_idx])
        else:
            neighbors = [row_parsed[0]]
            indices = np.array([[row_parsed[0]] * indices.shape[1]])
            if mode == 1:
                indices += offset
            result_indices.append(indices)
        for dst in neighbors:
            nodes.add(dst)
            edges.append((row_parsed[0], dst))
    mapping = {map_from: map_to for map_to, map_from in enumerate(sorted(nodes))}
    edges = list(map(lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges))
    result_indices = np.vstack(result_indices)
    return edges, result_indices, len(nodes), mapping


def parse_minibatch_LastFM(adjlists_ua, edge_metapath_indices_list_ua, user_artist_batch, device, samples=None, use_masks=None, offset=None):
    g_lists = [[], []]
    result_indices_lists = [[], []]
    idx_batch_mapped_lists = [[], []]
    for mode, (adjlists, edge_metapath_indices_list) in enumerate(zip(adjlists_ua, edge_metapath_indices_list_ua)):
        for adjlist, indices, use_mask in zip(adjlists, edge_metapath_indices_list, use_masks[mode]):
            if use_mask:
                edges, result_indices, num_nodes, mapping = parse_adjlist_LastFM(
                    [adjlist[row[mode]] for row in user_artist_batch], [indices[row[mode]] for row in user_artist_batch], samples, user_artist_batch, offset, mode)
            else:
                edges, result_indices, num_nodes, mapping = parse_adjlist_LastFM(
                    [adjlist[row[mode]] for row in user_artist_batch], [indices[row[mode]] for row in user_artist_batch], samples, offset=offset, mode=mode)

            g = dgl.DGLGraph()
            g.add_nodes(num_nodes)
            if len(edges) > 0:
                sorted_index = sorted(range(len(edges)), key=lambda i : edges[i])
                g.add_edges(*list(zip(*[(edges[i][1], edges[i][0]) for i in sorted_index])))
                result_indices = torch.LongTensor(result_indices[sorted_index]).to(device)
            else:
                result_indices = torch.LongTensor(result_indices).to(device)
            g_lists[mode].append(g)
            result_indices_lists[mode].append(result_indices)
            idx_batch_mapped_lists[mode].append(np.array([mapping[row[mode]] for row in user_artist_batch]))

    return g_lists, result_indices_lists, idx_batch_mapped_lists


class index_generator:
    def __init__(self, batch_size, num_data=None, indices=None, shuffle=True):
        if num_data is not None:
            self.num_data = num_data
            self.indices = np.arange(num_data)
        if indices is not None:
            self.num_data = len(indices)
            self.indices = np.copy(indices)
        self.batch_size = batch_size
        self.iter_counter = 0
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.indices)

    def next(self):
        if self.num_iterations_left() <= 0:
            self.reset()
        self.iter_counter += 1
        return np.copy(self.indices[(self.iter_counter - 1) * self.batch_size:self.iter_counter * self.batch_size])

    def num_iterations(self):
        return int(np.ceil(self.num_data / self.batch_size))

    def num_iterations_left(self):
        return self.num_iterations() - self.iter_counter

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.iter_counter = 0


def parse_lp_minibatch(adjlists_pa, mp_dicts_pa, batch_pairs, device, samples=None):
    """
    Build per-meta-path DGLGraphs & index tensors directly from the per-node path dicts.
    """
    g_lists, idx_lists, map_lists = [], [], []

    for mode in (0, 1):
        g_meta, idx_meta, map_meta = [], [], []

        # for each meta-path in this mode
        for mp_dict in mp_dicts_pa[mode]:
            # 1) Which nodes are we expanding? (paper IDs or area IDs)
            batch_nodes = [pair[mode] for pair in batch_pairs]

            edges = []
            all_paths = []
            nodes = set()

            # 2) Gather & (optionally) sample paths for each node
            for src in batch_nodes:
                paths = mp_dict.get(src, np.empty((0, len(next(iter(mp_dict.values()))[0])), dtype=int))
                nodes.add(src)

                if samples is not None and paths.shape[0] > samples:
                    # uniform sample indices
                    idxs = np.random.choice(paths.shape[0], samples, replace=False)
                    paths = paths[idxs]

                for path in paths:
                    dst = path[-1]
                    nodes.add(dst)
                    edges.append((src, dst))
                    all_paths.append(path[::-1])  # reverse so loader-consistent

            # 3) Create a local node→idx mapping
            mapping = {n: i for i, n in enumerate(sorted(nodes))}

            # 4) Remap edges and stack result indices
            remapped_edges = [(mapping[s], mapping[d]) for s, d in edges]
            order = sorted(range(len(remapped_edges)), key=lambda i: remapped_edges[i])
            if remapped_edges:
                srcs, dsts = zip(*(remapped_edges[i] for i in order))
            else:
                srcs, dsts = [], []
            result_indices = np.stack([all_paths[i] for i in order], axis=0) if all_paths else np.empty((0, len(all_paths[0])), dtype=int)

            # 5) Build the DGLGraph
            g = dgl.DGLGraph(multigraph=True)
            g.add_nodes(len(mapping))
            if srcs:
                g.add_edges(list(srcs), list(dsts))

            # 6) Convert paths to a torch.LongTensor
            idx_tensor = torch.LongTensor(result_indices).to(device)

            # 7) Map our batch nodes into the graph’s node-space
            remapped_batch = np.array([mapping[n] for n in batch_nodes], dtype=np.int64)

            # Collect
            g_meta.append(g)
            idx_meta.append(idx_tensor)
            map_meta.append(remapped_batch)

        g_lists.append(g_meta)
        idx_lists.append(idx_meta)
        map_lists.append(map_meta)

    return g_lists, idx_lists, map_lists


def parse_lp_minibatch_pa(adjlists_pa, mp_dicts_pa, batch_pairs, device, samples=None):
    """
    Build per-meta-path DGLGraphs & index tensors directly from the per-node path dicts,
    and guard against “no paths” by using the meta‐path length to size empty tensors.
    """
    g_lists, idx_lists, map_lists = [], [], []

    # For each of the two modes (0=paper→area, 1=area→paper)
    for mode in (0, 1):
        g_meta, idx_meta, map_meta = [], [], []

        # Loop over each meta-path dictionary in this mode
        for mp_dict in mp_dicts_pa[mode]:
            batch_nodes = [pair[mode] for pair in batch_pairs]
            edges, all_paths, nodes = [], [], set()

            # 1) Figure out path length for this meta-path
            if mp_dict:
                # grab the first array value, if any
                first_paths = next(iter(mp_dict.values()))
                path_len = first_paths.shape[1] if isinstance(first_paths, np.ndarray) else 0
            else:
                path_len = 0

            # 2) Gather & (optionally) sample paths for each batch node
            for src in batch_nodes:
                paths = mp_dict.get(src, np.empty((0, path_len), dtype=int))
                nodes.add(src)
                if samples is not None and paths.shape[0] > samples:
                    idxs = np.random.choice(paths.shape[0], samples, replace=False)
                    paths = paths[idxs]
                for path in paths:
                    dst = path[-1]
                    nodes.add(dst)
                    edges.append((src, dst))
                    all_paths.append(path[::-1])  # reverse for consistency

            # 3) Build local node→index mapping
            mapping = {n: i for i, n in enumerate(sorted(nodes))}

            # 4) Remap & sort edges
            remapped = [(mapping[s], mapping[d]) for s, d in edges]
            order = sorted(range(len(remapped)), key=lambda i: remapped[i])
            if remapped:
                srcs, dsts = zip(*(remapped[i] for i in order))
            else:
                srcs, dsts = [], []

            # 5) Build the result_indices array
            if all_paths:
                result_indices = np.stack([all_paths[i] for i in order], axis=0)
            else:
                result_indices = np.empty((0, path_len), dtype=int)

            # 6) Build the DGLGraph
            g = dgl.DGLGraph(multigraph=True)
            g.add_nodes(len(mapping))
            if srcs:
                g.add_edges(list(srcs), list(dsts))

            # 7) Convert paths to a torch tensor
            idx_tensor = torch.LongTensor(result_indices).to(device)

            # 8) Map batch nodes into this graph’s node space
            remapped_batch = np.array([mapping[n] for n in batch_nodes], dtype=np.int64)

            # Collect
            g_meta.append(g)
            idx_meta.append(idx_tensor)
            map_meta.append(remapped_batch)

        g_lists.append(g_meta)
        idx_lists.append(idx_meta)
        map_lists.append(map_meta)

    return g_lists, idx_lists, map_lists


def parse_lp_minibatch_pc(adjlists_pc, mp_dicts_pc, batch_pairs, device, samples=None):
    """
    Build per-meta-path DGLGraphs & index tensors directly from the per-node path dicts,
    for the Paper–Venue (PC) link prediction variant.
    """
    g_lists, idx_lists, map_lists = [], [], []

    # For each of the two modes (0=paper→conf, 1=conf→paper)
    for mode in (0, 1):
        g_meta, idx_meta, map_meta = [], [], []

        # Loop over each meta-path dictionary in this mode
        for mp_dict in mp_dicts_pc[mode]:
            batch_nodes = [pair[mode] for pair in batch_pairs]
            edges, all_paths, nodes = [], [], set()

            # determine this meta-path’s length (number of hops)
            example = next(iter(mp_dict.values()), None)
            if example is not None and example.size > 0:
                # example.shape == (num_paths, path_length)
                path_len = example.shape[1]
            else:
                path_len = 0

            # 1) Gather & optionally sample paths for each batch node
            for src in batch_nodes:
                paths = mp_dict.get(src, np.empty((0, path_len), dtype=int))
                nodes.add(src)
                if samples is not None and paths.shape[0] > samples:
                    idxs = np.random.choice(paths.shape[0], samples, replace=False)
                    paths = paths[idxs]
                for path in paths:
                    dst = path[-1]
                    nodes.add(dst)
                    edges.append((src, dst))
                    all_paths.append(path[::-1])  # reverse for consistency

            # 2) Build local node→index mapping
            mapping = {n: i for i, n in enumerate(sorted(nodes))}

            # 3) Remap edges to local indices & sort them
            remapped = [(mapping[s], mapping[d]) for s, d in edges]
            order = sorted(range(len(remapped)), key=lambda i: remapped[i])
            if remapped:
                srcs, dsts = zip(*(remapped[i] for i in order))
            else:
                srcs, dsts = [], []

            # 4) Build the result_indices array, guarded against empty all_paths
            if all_paths:
                result_indices = np.stack([all_paths[i] for i in order], axis=0)
            else:
                result_indices = np.empty((0, path_len), dtype=int)

            # 5) Build a DGLGraph for this meta-path
            g = dgl.DGLGraph(multigraph=True)
            g.add_nodes(len(mapping))
            if srcs:
                g.add_edges(list(srcs), list(dsts))

            # 6) Convert the path indices to a torch tensor
            idx_tensor = torch.LongTensor(result_indices).to(device)

            # 7) Map our batch nodes into this graph’s node space
            remapped_batch = np.array([mapping[n] for n in batch_nodes], dtype=np.int64)

            g_meta.append(g)
            idx_meta.append(idx_tensor)
            map_meta.append(remapped_batch)

        g_lists.append(g_meta)
        idx_lists.append(idx_meta)
        map_lists.append(map_meta)

    return g_lists, idx_lists, map_lists

def parse_lp_minibatch_pc_v2(adjlists_pc, mp_dicts_pc, batch_pairs, device, samples=None):
    """
    Build per-meta-path DGLGraphs & index tensors for paper–venue V2.
    """
    g_lists, idx_lists, map_lists = [], [], []

    for mode in (0, 1):
        g_meta, idx_meta, map_meta = [], [], []
        for mp_dict in mp_dicts_pc[mode]:
            # batch nodes (paper IDs if mode=0, venue IDs if mode=1)
            batch_nodes = [pair[mode] for pair in batch_pairs]

            # figure out the path-length for this meta-path
            # (all values in mp_dict are arrays of shape [#paths, path_len])
            if mp_dict:
                example = next(iter(mp_dict.values()))
                path_len = example.shape[1]
            else:
                path_len = 0

            edges, all_paths, nodes = [], [], set()

            # gather (and sample) paths
            for src in batch_nodes:
                paths = mp_dict.get(src, np.empty((0, path_len), dtype=int))
                nodes.add(src)
                if samples is not None and paths.shape[0] > samples:
                    idxs = np.random.choice(paths.shape[0], samples, replace=False)
                    paths = paths[idxs]
                for path in paths:
                    dst = path[-1]
                    nodes.add(dst)
                    edges.append((src, dst))
                    all_paths.append(path[::-1])

            # build local mapping
            mapping = {n: i for i, n in enumerate(sorted(nodes))}
            remapped = [(mapping[s], mapping[d]) for s, d in edges]
            order = sorted(range(len(remapped)), key=lambda i: remapped[i])
            if remapped:
                srcs, dsts = zip(*(remapped[i] for i in order))
            else:
                srcs, dsts = [], []

            # stack the paths (or make empty array)
            if all_paths:
                result_indices = np.stack([all_paths[i] for i in order], axis=0)
            else:
                result_indices = np.empty((0, path_len), dtype=int)

            # build DGLGraph
            g = dgl.DGLGraph(multigraph=True)
            g.add_nodes(len(mapping))
            if srcs:
                g.add_edges(list(srcs), list(dsts))

            # tensorize
            idx_tensor = torch.LongTensor(result_indices).to(device)
            remapped_batch = torch.LongTensor([mapping[n] for n in batch_nodes]).to(device)

            g_meta.append(g)
            idx_meta.append(idx_tensor)
            map_meta.append(remapped_batch)

        g_lists.append(g_meta)
        idx_lists.append(idx_meta)
        map_lists.append(map_meta)

    return g_lists, idx_lists, map_lists


def parse_lp_minibatch_ca(adjlists_ca, mp_dicts_ca, batch_pairs, device, samples=None):
    """
    Build per‐meta‐path DGLGraphs & index tensors for conf–area prediction,
    falling back to the direct (conf→area) or (area→conf) link if no paths exist.
    """
    g_lists, idx_lists, map_lists = [], [], []

    for mode in (0, 1):
        # mode=0: expand conf→area; mode=1: expand area→conf
        g_meta, idx_meta, map_meta = [], [], []

        for mp_dict in mp_dicts_ca[mode]:
            # prepare
            batch_src   = [pair[mode] for pair in batch_pairs]
            batch_tgt   = [pair[1-mode] for pair in batch_pairs]
            nodes       = set()
            edges       = []
            all_paths   = []

            # determine target width if mp_dict nonempty
            if mp_dict:
                first = next(iter(mp_dict.values()))
                width = first.shape[1]
            else:
                width = 2  # fallback to simple 1‐hop sequences

            # collect paths (or fallback)
            for src, tgt in zip(batch_src, batch_tgt):
                paths = mp_dict.get(src, None)
                if paths is None or len(paths)==0:
                    # no long metapaths: use simple direct path [src, tgt]
                    paths = np.array([[src, tgt]], dtype=int)
                else:
                    # optionally sample
                    if samples is not None and paths.shape[0] > samples:
                        idxs = np.random.choice(len(paths), samples, replace=False)
                        paths = paths[idxs]
                nodes.add(src)
                for path in paths:
                    dst = path[-1]
                    nodes.add(dst)
                    edges.append((src, dst))
                    all_paths.append(path[::-1])  # reverse for MAGNN loader consistency

            # build mapping
            mapping = {n:i for i,n in enumerate(sorted(nodes))}

            # remap edges & paths
            remapped = [(mapping[s], mapping[d]) for s,d in edges]
            if remapped:
                order = sorted(range(len(remapped)), key=lambda i: remapped[i])
                srcs, dsts = zip(*(remapped[i] for i in order))
                result_indices = np.stack([all_paths[i] for i in order], axis=0)
            else:
                srcs = dsts = ()
                result_indices = np.empty((0, width), dtype=int)

            # build DGLGraph
            g = dgl.DGLGraph(multigraph=True)
            g.add_nodes(len(mapping))
            if srcs:
                g.add_edges(list(srcs), list(dsts))

            # to tensor
            idx_tensor = torch.LongTensor(result_indices).to(device)

            # remap batch indices
            remapped_batch = np.array([mapping[s] for s in batch_src], dtype=np.int64)

            g_meta.append(g)
            idx_meta.append(idx_tensor)
            map_meta.append(remapped_batch)

        g_lists.append(g_meta)
        idx_lists.append(idx_meta)
        map_lists.append(map_meta)

    return g_lists, idx_lists, map_lists


def parse_lp_minibatch_pa_var2(adjlists_pa, mp_dicts_pa, batch_pairs, device, samples=None):
    """
    For each mode (paper‑centric, area‑centric) and each meta‐path dict,
    builds a DGLGraph with edges for the sampled paths, a tensor of path indices,
    and a tensor mapping your batch nodes into the graph’s node space.
    """
    g_lists, idx_lists, map_lists = [], [], []

    for mode in (0, 1):
        g_meta, idx_meta, map_meta = [], [], []

        for mp_dict in mp_dicts_pa[mode]:
            # which src nodes are we expanding this batch?
            batch_nodes = [pair[mode] for pair in batch_pairs]

            # determine expected path length
            if mp_dict:
                path_len = next(iter(mp_dict.values())).shape[1]
            else:
                path_len = 0

            edges, all_paths, nodes = [], [], set()

            # collect (src→dst) edges from each path, plus the path itself
            for src in batch_nodes:
                nodes.add(src)
                paths = mp_dict.get(src, np.empty((0, path_len), int))
                if samples and len(paths) > samples:
                    idxs = np.random.choice(len(paths), samples, replace=False)
                    paths = paths[idxs]
                for p in paths:
                    dst = p[-1]
                    nodes.add(dst)
                    edges.append((src, dst))
                    all_paths.append(p[::-1])  # reverse to match loader expectation

            # build mapping src global→local
            mapping = {nid: i for i, nid in enumerate(sorted(nodes))}

            # remap edges & stack path indices
            if edges:
                remapped = [(mapping[s], mapping[d]) for s, d in edges]
                order = sorted(range(len(remapped)), key=lambda i: remapped[i])
                srcs, dsts = zip(*(remapped[i] for i in order))
                paths_arr = np.stack([all_paths[i] for i in order], axis=0)
            else:
                srcs, dsts = [], []
                paths_arr = np.empty((0, path_len), dtype=int)

            # build the DGLGraph
            g = dgl.DGLGraph(multigraph=True)
            g.add_nodes(len(mapping))
            if srcs:
                g.add_edges(list(srcs), list(dsts))

            # to device
            idx_tensor   = torch.LongTensor(paths_arr).to(device)
            batch_tensor = torch.LongTensor([mapping[n] for n in batch_nodes]).to(device)

            g_meta.append(g)
            idx_meta.append(idx_tensor)
            map_meta.append(batch_tensor)

        g_lists.append(g_meta)
        idx_lists.append(idx_meta)
        map_lists.append(map_meta)

    return g_lists, idx_lists, map_lists
