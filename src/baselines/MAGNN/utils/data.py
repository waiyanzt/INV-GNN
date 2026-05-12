import networkx as nx
import numpy as np
import scipy
import pickle
import scipy.sparse
import os

def load_IMDB_data(prefix='data/preprocessed/IMDB_processed'):
    G00 = nx.read_adjlist(prefix + '/0/0-1-0.adjlist', create_using=nx.MultiDiGraph)
    G01 = nx.read_adjlist(prefix + '/0/0-2-0.adjlist', create_using=nx.MultiDiGraph)
    G10 = nx.read_adjlist(prefix + '/1/1-0-1.adjlist', create_using=nx.MultiDiGraph)
    G11 = nx.read_adjlist(prefix + '/1/1-0-2-0-1.adjlist', create_using=nx.MultiDiGraph)
    G20 = nx.read_adjlist(prefix + '/2/2-0-2.adjlist', create_using=nx.MultiDiGraph)
    G21 = nx.read_adjlist(prefix + '/2/2-0-1-0-2.adjlist', create_using=nx.MultiDiGraph)
    idx00 = np.load(prefix + '/0/0-1-0_idx.npy')
    idx01 = np.load(prefix + '/0/0-2-0_idx.npy')
    idx10 = np.load(prefix + '/1/1-0-1_idx.npy')
    idx11 = np.load(prefix + '/1/1-0-2-0-1_idx.npy')
    idx20 = np.load(prefix + '/2/2-0-2_idx.npy')
    idx21 = np.load(prefix + '/2/2-0-1-0-2_idx.npy')
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz')
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz')
    features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz')
    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
    return [[G00, G01], [G10, G11], [G20, G21]], \
           [[idx00, idx01], [idx10, idx11], [idx20, idx21]], \
           [features_0, features_1, features_2],\
           adjM, \
           type_mask,\
           labels,\
           train_val_test_idx


def load_DBLP_data(prefix='data/preprocessed/DBLP_processed'):
    if prefix == 'data/preprocessed/DBLP_preprocessed_area':
        print('loading dblp_preprocessed_area')
        return load_DBLP_area_data(prefix)
    elif prefix == 'data/preprocessed/DBLP_preprocessed_area_transformed':
        print('loading dblp_preprocessed_area_transformed')
        return load_DBLP_area_transformed_data(prefix)
    in_file = open(prefix + '/0/0-1-0.adjlist', 'r')
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00[3:]
    in_file.close()
    in_file = open(prefix + '/0/0-1-2-1-0.adjlist', 'r')
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01[3:]
    in_file.close()
    in_file = open(prefix + '/0/0-1-3-1-0.adjlist', 'r')
    adjlist02 = [line.strip() for line in in_file]
    adjlist02 = adjlist02[3:]
    in_file.close()

    in_file = open(prefix + '/0/0-1-0_idx.pickle', 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-1-2-1-0_idx.pickle', 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-1-3-1-0_idx.pickle', 'rb')
    idx02 = pickle.load(in_file)
    in_file.close()

    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = np.load(prefix + '/features_2.npy')
    #features_3 = np.eye(20, dtype=np.float32)
    features_3 = np.eye(4, dtype=np.float32)

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')

    return [adjlist00, adjlist01, adjlist02], \
           [idx00, idx01, idx02], \
           [features_0, features_1, features_2, features_3],\
           adjM, \
           type_mask,\
           labels,\
           train_val_test_idx

def load_DBLP_area_data(prefix='data/preprocessed/DBLP_preprocessed_area'):
    in_file = open(prefix + '/0/0-1-0.adjlist', 'r')
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00[3:]
    in_file.close()
    in_file = open(prefix + '/0/0-2-0.adjlist', 'r')
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01[3:]
    in_file.close()
    in_file = open(prefix + '/0/0-3-0.adjlist', 'r')
    adjlist02 = [line.strip() for line in in_file]
    adjlist02 = adjlist02[3:]
    in_file.close()

    in_file = open(prefix + '/0/0-1-0_idx.pickle', 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-2-0_idx.pickle', 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-3-0_idx.pickle', 'rb')
    idx02 = pickle.load(in_file)
    in_file.close()
    


    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = np.load(prefix + '/features_2.npy')
    #features_3 = np.eye(20, dtype=np.float32)
    features_3 = np.eye(4, dtype=np.float32)

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')

    return [adjlist00, adjlist01, adjlist02], \
           [idx00, idx01, idx02], \
           [features_0, features_1, features_2, features_3],\
           adjM, \
           type_mask,\
           labels,\
           train_val_test_idx

def load_DBLP_area_transformed_data(prefix='data/preprocessed/DBLP_preprocessed_area_transformed'):
    in_file = open(prefix + '/0/0-1-0.adjlist', 'r')
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00[3:]
    in_file.close()
    in_file = open(prefix + '/0/0-2-0.adjlist', 'r')
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01[3:]
    in_file.close()
    in_file = open(prefix + '/0/0-1-3-1-0.adjlist', 'r')
    adjlist02 = [line.strip() for line in in_file]
    adjlist02 = adjlist02[3:]
    in_file.close()

    in_file = open(prefix + '/0/0-1-0_idx.pickle', 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-2-0_idx.pickle', 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-1-3-1-0_idx.pickle', 'rb')
    idx02 = pickle.load(in_file)
    in_file.close()

    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = np.load(prefix + '/features_2.npy')
    #features_3 = np.eye(20, dtype=np.float32)
    features_3 = np.eye(4, dtype=np.float32)

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')

    return [adjlist00, adjlist01, adjlist02], \
           [idx00, idx01, idx02], \
           [features_0, features_1, features_2, features_3],\
           adjM, \
           type_mask,\
           labels,\
           train_val_test_idx


def load_LastFM_data(prefix='data/preprocessed/LastFM_processed'):
    in_file = open(prefix + '/0/0-1-0.adjlist', 'r')
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00
    in_file.close()
    in_file = open(prefix + '/0/0-1-2-1-0.adjlist', 'r')
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01
    in_file.close()
    in_file = open(prefix + '/0/0-0.adjlist', 'r')
    adjlist02 = [line.strip() for line in in_file]
    adjlist02 = adjlist02
    in_file.close()
    in_file = open(prefix + '/1/1-0-1.adjlist', 'r')
    adjlist10 = [line.strip() for line in in_file]
    adjlist10 = adjlist10
    in_file.close()
    in_file = open(prefix + '/1/1-2-1.adjlist', 'r')
    adjlist11 = [line.strip() for line in in_file]
    adjlist11 = adjlist11
    in_file.close()
    in_file = open(prefix + '/1/1-0-0-1.adjlist', 'r')
    adjlist12 = [line.strip() for line in in_file]
    adjlist12 = adjlist12
    in_file.close()

    in_file = open(prefix + '/0/0-1-0_idx.pickle', 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-1-2-1-0_idx.pickle', 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-0_idx.pickle', 'rb')
    idx02 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/1/1-0-1_idx.pickle', 'rb')
    idx10 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/1/1-2-1_idx.pickle', 'rb')
    idx11 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/1/1-0-0-1_idx.pickle', 'rb')
    idx12 = pickle.load(in_file)
    in_file.close()

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    train_val_test_pos_user_artist = np.load(prefix + '/train_val_test_pos_user_artist.npz')
    train_val_test_neg_user_artist = np.load(prefix + '/train_val_test_neg_user_artist.npz')

    return [[adjlist00, adjlist01, adjlist02],[adjlist10, adjlist11, adjlist12]],\
           [[idx00, idx01, idx02], [idx10, idx11, idx12]],\
           adjM, type_mask, train_val_test_pos_user_artist, train_val_test_neg_user_artist


# load skipgram-format embeddings, treat missing node embeddings as zero vectors
def load_skipgram_embedding(path, num_embeddings):
    count = 0
    with open(path, 'r') as infile:
        _, dim = list(map(int, infile.readline().strip().split(' ')))
        embeddings = np.zeros((num_embeddings, dim))
        for line in infile.readlines():
            count += 1
            line = line.strip().split(' ')
            embeddings[int(line[0])] = np.array(list(map(float, line[1:])))
    print('{} out of {} nodes have non-zero embeddings'.format(count, num_embeddings))
    return embeddings


# load metapath2vec embeddings
def load_metapath2vec_embedding(path, type_list, num_embeddings_list, offset_list):
    count = 0
    with open(path, 'r') as infile:
        _, dim = list(map(int, infile.readline().strip().split(' ')))
        embeddings_dict = {type: np.zeros((num_embeddings, dim)) for type, num_embeddings in zip(type_list, num_embeddings_list)}
        offset_dict = {type: offset for type, offset in zip(type_list, offset_list)}
        for line in infile.readlines():
            line = line.strip().split(' ')
            # drop </s> token
            if line[0] == '</s>':
                continue
            count += 1
            embeddings_dict[line[0][0]][int(line[0][1:]) - offset_dict[line[0][0]]] = np.array(list(map(float, line[1:])))
    print('{} node embeddings loaded'.format(count))
    return embeddings_dict


def load_glove_vectors(dim=50):
    print('Loading GloVe pretrained word vectors')
    file_paths = {
        50: 'data/wordvec/GloVe/glove.6B.50d.txt',
        100: 'data/wordvec/GloVe/glove.6B.100d.txt',
        200: 'data/wordvec/GloVe/glove.6B.200d.txt',
        300: 'data/wordvec/GloVe/glove.6B.300d.txt'
    }
    f = open(file_paths[dim], 'r', encoding='utf-8')
    wordvecs = {}
    for line in f.readlines():
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        wordvecs[word] = embedding
    print('Done.', len(wordvecs), 'words loaded!')
    return wordvecs

def load_DBLP_lp_pa_data(prefix='data/preprocessed/DBLP_lp_pa'):
    """
    Loads:
      - adjlists_pa:       [[adj0_meta0, …], [adj1_meta0, …]]
      - edge_mp_indices_pa:[[dict for meta0], [dict for meta1]]
      - type_mask:         np.array of length N
      - train_pos, val_pos, test_pos
      - train_neg, val_neg, test_neg
    """

    # 1) Define your meta-path names (must match the folder & file names)
    meta0 = ['1-0-1', '1-2-1', '1-3-1', '1-4-1']        # paper→area
    meta1 = ['4-1-4', '4-1-0-1-4', '4-1-2-1-4', '4-1-3-1-4']  # area→paper

    adjlists_pa = []
    edge_mp_indices_pa = []

    # 2) Load each mode’s .adjlist & its corresponding _idx.pickle
    for mode, metas in enumerate((meta0, meta1)):
        mode_adjs = []
        mode_dicts = []
        for m in metas:
            # read adjacency-list lines
            with open(f"{prefix}/{mode}/{m}.adjlist", 'r') as f:
                lines = [line.strip() for line in f]
            mode_adjs.append(lines)

            # load the per-node dict
            with open(f"{prefix}/{mode}/{m}_idx.pickle", 'rb') as pf:
                node2paths = pickle.load(pf)
            mode_dicts.append(node2paths)

        adjlists_pa.append(mode_adjs)
        edge_mp_indices_pa.append(mode_dicts)

    # 3) Global graph + types
    adjM      = scipy.sparse.load_npz(f"{prefix}/adjM.npz")
    type_mask = np.load(f"{prefix}/node_types.npy")

    # 4) Pos/neg splits
    pos = np.load(f"{prefix}/train_val_test_pos_paper_area.npz")
    neg = np.load(f"{prefix}/train_val_test_neg_paper_area.npz")
    train_pos = pos['train_pos_paper_area']
    val_pos   = pos['val_pos_paper_area']
    test_pos  = pos['test_pos_paper_area']
    train_neg = neg['train_neg_paper_area']
    val_neg   = neg['val_neg_paper_area']
    test_neg  = neg['test_neg_paper_area']

    return (
        adjlists_pa,
        edge_mp_indices_pa,
        type_mask,
        train_pos, val_pos, test_pos,
        train_neg, val_neg, test_neg
    )

def load_DBLP_lp_pat_data(prefix='data/preprocessed/DBLP_lp_pa_t'):
    """
    Loads:
      - adjlists_pa:       [[adj0_meta0, …], [adj1_meta0, …]]
      - edge_mp_indices_pa:[[dict for meta0], [dict for meta1]]
      - type_mask:         np.array of length N
      - train_pos, val_pos, test_pos
      - train_neg, val_neg, test_neg
    """

    # 1) Define your meta-path names (must match the folder & file names)
    meta0 = ['1-0-1', '1-2-1', '1-3-1', '1-4-1']        # paper→area
    meta1 = ['4-1-4', '4-1-0-1-4', '4-1-2-1-4', '4-1-3-1-4', '4-3-4', '4-3-1-3-4', '4-3-1-0-1-3-4', '4-3-1-2-1-3-4']  # area→paper

    adjlists_pa = []
    edge_mp_indices_pa = []

    # 2) Load each mode’s .adjlist & its corresponding _idx.pickle
    for mode, metas in enumerate((meta0, meta1)):
        mode_adjs = []
        mode_dicts = []
        for m in metas:
            # read adjacency-list lines
            with open(f"{prefix}/{mode}/{m}.adjlist", 'r') as f:
                lines = [line.strip() for line in f]
            mode_adjs.append(lines)

            # load the per-node dict
            with open(f"{prefix}/{mode}/{m}_idx.pickle", 'rb') as pf:
                node2paths = pickle.load(pf)
            mode_dicts.append(node2paths)

        adjlists_pa.append(mode_adjs)
        edge_mp_indices_pa.append(mode_dicts)

    # 3) Global graph + types
    adjM      = scipy.sparse.load_npz(f"{prefix}/adjM.npz")
    type_mask = np.load(f"{prefix}/node_types.npy")

    # 4) Pos/neg splits
    pos = np.load(f"{prefix}/train_val_test_pos_paper_area.npz")
    neg = np.load(f"{prefix}/train_val_test_neg_paper_area.npz")
    train_pos = pos['train_pos_paper_area']
    val_pos   = pos['val_pos_paper_area']
    test_pos  = pos['test_pos_paper_area']
    train_neg = neg['train_neg_paper_area']
    val_neg   = neg['val_neg_paper_area']
    test_neg  = neg['test_neg_paper_area']

    return (
        adjlists_pa,
        edge_mp_indices_pa,
        type_mask,
        train_pos, val_pos, test_pos,
        train_neg, val_neg, test_neg
    )


def load_DBLP_lp_pc_data(prefix='data/preprocessed/DBLP_lp_pc_var1'):
    """
    Loads everything for paper–venue variation 1:
      - adjlists_pc:       [[adj0_meta0, …], [adj1_meta0, …]]
      - edge_mp_indices_pc:[[dict for meta0], [dict for meta1]]
      - type_mask:         np.array of length N
      - train_pos, val_pos, test_pos
      - train_neg, val_neg, test_neg
    """
    # 1) define the two folders’ meta‐path names (must exactly match your dirs)
    meta0 = ['1-0-1','1-2-1','1-3-1','1-4-1']      # paper‐centric
    meta1 = ['3-1-3','3-1-0-1-3','3-1-2-1-3','3-1-4-1-3']  # venue‐centric

    adjlists_pc = []
    edge_mp_indices_pc = []

    for mode, metas in enumerate((meta0, meta1)):
        mode_adjs, mode_dicts = [], []
        for m in metas:
            # load .adjlist
            with open(f"{prefix}/{mode}/{m}.adjlist", 'r') as f:
                mode_adjs.append([line.strip() for line in f])
            # load the per-node pickle
            with open(f"{prefix}/{mode}/{m}_idx.pickle", 'rb') as pf:
                mode_dicts.append(pickle.load(pf))
        adjlists_pc.append(mode_adjs)
        edge_mp_indices_pc.append(mode_dicts)

    # 2) global graph + types
    adjM      = scipy.sparse.load_npz(f"{prefix}/adjM.npz")
    type_mask = np.load(f"{prefix}/node_types.npy")

    # 3) train / val / test splits
    pos = np.load(f"{prefix}/train_val_test_pos_paper_conf.npz")
    neg = np.load(f"{prefix}/train_val_test_neg_paper_conf.npz")
    train_pos = pos['train_pos']
    val_pos   = pos['val_pos']
    test_pos  = pos['test_pos']
    train_neg = neg['train_neg']
    val_neg   = neg['val_neg']
    test_neg  = neg['test_neg']

    return (
        adjlists_pc,
        edge_mp_indices_pc,
        type_mask,
        train_pos, val_pos, test_pos,
        train_neg, val_neg, test_neg
    )

def load_DBLP_lp_pc_v2_data(prefix='data/preprocessed/DBLP_lp_pc_v2'):
    """
    Loads everything for paper–venue (conf) Variation 2:
      - adjlists_pc:       [[adj0_meta0, …], [adj1_meta0, …]]
      - edge_mp_indices_pc:[[dict for meta0], [dict for meta1]]
      - type_mask:         np.array of length N
      - train_pos, val_pos, test_pos
      - train_neg, val_neg, test_neg
    """

    # 1) meta-path names must match your folder names under prefix/0/ and prefix/1/
    meta0 = ['1-0-1', '1-2-1', '1-3-1', '1-3-4-3-1']        # paper→paper via author/term/conf/venue→area→venue→paper
    meta1 = ['3-1-3', '3-1-0-1-3', '3-1-2-1-3', '3-4-3']  # venue→paper→venue via the four venues-centric paths

    adjlists_pc      = []
    edge_mp_indices_pc = []

    for mode, metas in enumerate((meta0, meta1)):
        mode_adjs = []
        mode_dicts = []
        for m in metas:
            # read adjacency-list
            with open(f"{prefix}/{mode}/{m}.adjlist", 'r') as f:
                mode_adjs.append([line.strip() for line in f])
            # load per-node path dict
            with open(f"{prefix}/{mode}/{m}_idx.pickle", 'rb') as pf:
                mode_dicts.append(pickle.load(pf))
        adjlists_pc.append(mode_adjs)
        edge_mp_indices_pc.append(mode_dicts)

    # 2) globals
    adjM      = scipy.sparse.load_npz(f"{prefix}/adjM.npz")
    type_mask = np.load(f"{prefix}/node_types.npy")

    # 3) splits
    pos = np.load(f"{prefix}/train_val_test_pos_paper_conf.npz")
    neg = np.load(f"{prefix}/train_val_test_neg_paper_conf.npz")
    train_pos = pos['train_pos']
    val_pos   = pos['val_pos']
    test_pos  = pos['test_pos']
    train_neg = neg['train_neg']
    val_neg   = neg['val_neg']
    test_neg  = neg['test_neg']

    return (
        adjlists_pc,
        edge_mp_indices_pc,
        type_mask,
        train_pos, val_pos, test_pos,
        train_neg, val_neg, test_neg
    )


def load_DBLP_lp_ca_v1_data(prefix='data/preprocessed/DBLP_lp_ca_var1'):
    """
    Returns:
      adjlists_ca, mp_dicts_ca, type_mask,
      train_pos, val_pos, test_pos,
      train_neg, val_neg, test_neg
    """

    # 1) meta-path names (must match your folders)
    meta0 = ['3-1-3','3-1-0-1-3','3-1-2-1-3','3-4-3']
    meta1 = ['4-3-4','4-3-1-3-4','4-3-1-0-1-3-4','4-3-1-2-1-3-4']

    adjlists_ca, mp_dicts_ca = [], []
    for mode, metas in enumerate((meta0, meta1)):
        mode_adjs, mode_dicts = [], []
        for m in metas:
            with open(f"{prefix}/{mode}/{m}.adjlist") as f:
                mode_adjs.append([l.strip() for l in f])
            with open(f"{prefix}/{mode}/{m}_idx.pickle","rb") as pf:
                mode_dicts.append(pickle.load(pf))
        adjlists_ca.append(mode_adjs)
        mp_dicts_ca.append(mode_dicts)

    # 2) globals
    adjM      = scipy.sparse.load_npz(f"{prefix}/adjM.npz")
    type_mask = np.load(f"{prefix}/node_types.npy")

    # 3) splits
    pos = np.load(f"{prefix}/train_val_test_pos_conf_area.npz")
    neg = np.load(f"{prefix}/train_val_test_neg_conf_area.npz")
    train_pos, val_pos, test_pos = pos['train_pos'], pos['val_pos'], pos['test_pos']
    train_neg, val_neg, test_neg = neg['train_neg'], neg['val_neg'], neg['test_neg']

    return adjlists_ca, mp_dicts_ca, type_mask, \
           train_pos, val_pos, test_pos, \
           train_neg, val_neg, test_neg


def load_DBLP_lp_ca_v2_data(prefix='data/preprocessed/DBLP_lp_ca_var2'):
    """
    Loads the preprocessed meta-path adjlists & per-node pickles,
    prunes any completely-empty meta-paths, and returns:

      adjlists [2][K_mode]      : raw adjlist lines (strings)
      mp_dicts  [2][K_mode]     : dict[src_node] -> np.array(paths)
      type_mask  np.array N
      train_pos, val_pos, test_pos
      train_neg, val_neg, test_neg
    """
    # 1) your original meta‐path filename lists (must match your dirs)
    meta0 = ['3-1-3','3-1-0-1-3','3-1-2-1-3','3-4-3','3-1-4-1-3']
    meta1 = [
      '4-1-4','4-1-0-1-4','4-1-2-1-4','4-1-3-1-4',
      '4-3-4','4-3-1-3-4','4-3-1-0-1-3-4','4-3-1-2-1-3-4'
    ]

    raw_adjlists = [[], []]
    raw_mpdicts  = [[], []]
    for mode, metas in enumerate((meta0, meta1)):
        for m in metas:
            # load adjlist lines
            with open(os.path.join(prefix, str(mode), f"{m}.adjlist")) as f:
                adj_lines = [line.strip() for line in f]
            # load per-node path dict
            with open(os.path.join(prefix, str(mode), f"{m}_idx.pickle"), 'rb') as pf:
                node2paths = pickle.load(pf)
            raw_adjlists[mode].append((m, adj_lines))
            raw_mpdicts[mode].append((m, node2paths))

    # 2) PRUNE ANY META-PATH WITHOUT A SINGLE PATH
    adjlists, mpdicts = [[],[]], [[],[]]
    for mode in (0,1):
        for (m, adj_lines), (_, node2paths) in zip(raw_adjlists[mode], raw_mpdicts[mode]):
            # keep if at least one src has ≥1 path
            if any(paths.shape[0] for paths in node2paths.values()):
                adjlists[mode].append(adj_lines)
                mpdicts[mode].append(node2paths)

    # 3) load globals
    adjM      = scipy.sparse.load_npz(os.path.join(prefix, 'adjM.npz'))
    type_mask = np.load(os.path.join(prefix, 'node_types.npy'))

    # 4) splits
    P = np.load(os.path.join(prefix, 'train_val_test_pos_conf_area.npz'))
    N = np.load(os.path.join(prefix, 'train_val_test_neg_conf_area.npz'))
    train_pos, val_pos, test_pos = P['train_pos'], P['val_pos'], P['test_pos']
    train_neg, val_neg, test_neg = N['train_neg'], N['val_neg'], N['test_neg']

    return adjlists, mpdicts, type_mask, \
           train_pos, val_pos, test_pos, train_neg, val_neg, test_neg

import os, pickle
import numpy as np
import scipy.sparse as sp

def _read_adjlist_raw(path):
    """Return list of STRING lines exactly as in the .adjlist file."""
    if not os.path.exists(path):
        return []
    with open(path, 'r') as f:
        # keep each row as a single string; strip trailing newline
        lines = [ln.rstrip('\n') for ln in f]
    return lines

def _read_idx_pickle(path, L, target_count):
    """Return dict[target_idx] -> np.ndarray of paths (possibly empty)."""
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    # synthesize empties if missing
    import numpy as np
    empty = np.empty((0, L), dtype=np.int32)
    return {t: empty for t in range(target_count)}

def load_DBLP_lp_pc_var1_data(expected_metapaths,
                              base='data/preprocessed/DBLP_lp_pc_var1_noAuthor/'):

    type_mask = np.load(os.path.join(base, 'node_types.npy'))
    adjM = sp.load_npz(os.path.join(base, 'adjM.npz'))

    num_paper = int((type_mask == 1).sum())
    num_conf  = int((type_mask == 3).sum())

    posz = np.load(os.path.join(base, 'train_val_test_pos_paper_conf.npz'))
    negz = np.load(os.path.join(base, 'train_val_test_neg_paper_conf.npz'))
    pos = {
        'train_pos_paper_conf': posz['train_pos'],
        'val_pos_paper_conf':   posz['val_pos'],
        'test_pos_paper_conf':  posz['test_pos'],
    }
    neg = {
        'train_neg_paper_conf': negz['train_neg'],
        'val_neg_paper_conf':   negz['val_neg'],
        'test_neg_paper_conf':  negz['test_neg'],
    }

    adjlists = [[], []]
    edge_metapath_indices_list = [[], []]
    target_counts = [num_paper, num_conf]

    for mode in (0, 1):
        for meta in expected_metapaths[mode]:
            name = '-'.join(map(str, meta))
            L = len(meta)
            adj_path = os.path.join(base, f'{mode}', f'{name}.adjlist')
            idx_path = os.path.join(base, f'{mode}', f'{name}_idx.pickle')

            # IMPORTANT: return raw STRING lines (not parsed ints)
            adj = _read_adjlist_raw(adj_path)
            # safety: ensure we have a row per target (our writer already does this)
            if len(adj) < target_counts[mode]:
                # pad missing rows with just the index (no neighbors)
                start = len(adj)
                adj += [f"{i}" for i in range(start, target_counts[mode])]
            edge_idx = _read_idx_pickle(idx_path, L, target_counts[mode])

            adjlists[mode].append(adj)
            edge_metapath_indices_list[mode].append(edge_idx)

    return adjlists, edge_metapath_indices_list, adjM, type_mask, pos, neg, num_paper, num_conf


# --- DBLP paper–venue LP: Variation 2 loader (Area - Venue) ---

def load_DBLP_lp_pc_var2_data(expected_metapaths,
                              base='data/preprocessed/DBLP_lp_pc_var2_noAuthor/'):

    type_mask = np.load(os.path.join(base, 'node_types.npy'))
    adjM = sp.load_npz(os.path.join(base, 'adjM.npz'))

    num_paper = int((type_mask == 1).sum())
    num_conf  = int((type_mask == 3).sum())

    posz = np.load(os.path.join(base, 'train_val_test_pos_paper_conf.npz'))
    negz = np.load(os.path.join(base, 'train_val_test_neg_paper_conf.npz'))
    pos = {
        'train_pos_paper_conf': posz['train_pos'],
        'val_pos_paper_conf':   posz['val_pos'],
        'test_pos_paper_conf':  posz['test_pos'],
    }
    neg = {
        'train_neg_paper_conf': negz['train_neg'],
        'val_neg_paper_conf':   negz['val_neg'],
        'test_neg_paper_conf':  negz['test_neg'],
    }

    adjlists = [[], []]
    edge_metapath_indices_list = [[], []]
    target_counts = [num_paper, num_conf]

    for mode in (0, 1):
        for meta in expected_metapaths[mode]:
            name = '-'.join(map(str, meta))
            L = len(meta)
            adj_path = os.path.join(base, f'{mode}', f'{name}.adjlist')
            idx_path = os.path.join(base, f'{mode}', f'{name}_idx.pickle')

            # Read raw STRING lines (parser expects strings)
            if os.path.exists(adj_path):
                with open(adj_path, 'r') as f:
                    adj = [ln.rstrip('\n') for ln in f]
            else:
                adj = []
            # Pad to a line per target if needed
            if len(adj) < target_counts[mode]:
                start = len(adj)
                adj += [f"{i}" for i in range(start, target_counts[mode])]

            # Load per-target metapath arrays or synthesize empties
            if os.path.exists(idx_path):
                with open(idx_path, 'rb') as f:
                    edge_idx = pickle.load(f)
            else:
                empty = np.empty((0, L), dtype=np.int32)
                edge_idx = {t: empty for t in range(target_counts[mode])}

            adjlists[mode].append(adj)
            edge_metapath_indices_list[mode].append(edge_idx)

    return adjlists, edge_metapath_indices_list, adjM, type_mask, pos, neg, num_paper, num_conf

DEFAULT_EXPECTED_METAPATHS_VAR3 = [
    [(1,0,1), (1,2,1), (1,3,1), (1,0,4,0,1)],
    [(3,1,0,1,3), (3,1,2,1,3), (3,1,0,4,0,1,3)]
]

def load_DBLP_lp_pc_var3_data(expected_metapaths=DEFAULT_EXPECTED_METAPATHS_VAR3,
                              base='data/preprocessed/DBLP_lp_pc_var3/'):
    type_mask = np.load(os.path.join(base, 'node_types.npy'))
    adjM = sp.load_npz(os.path.join(base, 'adjM.npz'))

    # 1: paper, 3: conf (unchanged type ids)
    num_paper = int((type_mask == 1).sum())
    num_conf  = int((type_mask == 3).sum())

    posz = np.load(os.path.join(base, 'train_val_test_pos_paper_conf.npz'))
    negz = np.load(os.path.join(base, 'train_val_test_neg_paper_conf.npz'))
    pos = {
        'train_pos_paper_conf': posz['train_pos'],
        'val_pos_paper_conf':   posz['val_pos'],
        'test_pos_paper_conf':  posz['test_pos'],
    }
    neg = {
        'train_neg_paper_conf': negz['train_neg'],
        'val_neg_paper_conf':   negz['val_neg'],
        'test_neg_paper_conf':  negz['test_neg'],
    }

    adjlists = [[], []]
    edge_metapath_indices_list = [[], []]
    target_counts = [num_paper, num_conf]

    for mode in (0, 1):
        for meta in expected_metapaths[mode]:
            name = '-'.join(map(str, meta))
            L = len(meta)
            adj_path = os.path.join(base, f'{mode}', f'{name}.adjlist')
            idx_path = os.path.join(base, f'{mode}', f'{name}_idx.pickle')

            # raw STRING lines (MAGNN parser expects strings)
            adj = _read_adjlist_raw(adj_path)
            # safety: ensure a row per target
            if len(adj) < target_counts[mode]:
                start = len(adj)
                adj += [f"{i}" for i in range(start, target_counts[mode])]

            edge_idx = _read_idx_pickle(idx_path, L, target_counts[mode])

            adjlists[mode].append(adj)
            edge_metapath_indices_list[mode].append(edge_idx)

    return adjlists, edge_metapath_indices_list, adjM, type_mask, pos, neg, num_paper, num_conf
