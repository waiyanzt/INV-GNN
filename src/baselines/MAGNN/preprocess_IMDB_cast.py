import pathlib

import numpy as np
import scipy.sparse
import scipy.io
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
import nltk
'''
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt_tab')
'''


def get_metapath_adjacency_matrix(adjM, type_mask, metapath):
    """
    :param M: the raw adjacency matrix
    :param type_mask: an array of types of all node
    :param metapath
    :return: a list of metapath-based adjacency matrices
    """
    out_adjM = scipy.sparse.csr_matrix(adjM[np.ix_(type_mask == metapath[0], type_mask == metapath[1])])
    for i in range(1, len(metapath) - 1):
        out_adjM = out_adjM.dot(scipy.sparse.csr_matrix(adjM[np.ix_(type_mask == metapath[i], type_mask == metapath[i + 1])]))
    return out_adjM.toarray()


# networkx.has_path may search too
def get_metapath_neighbor_pairs(M, type_mask, expected_metapaths):
    """
    :param M: the raw adjacency matrix
    :param type_mask: an array of types of all node
    :param expected_metapaths: a list of expected metapaths
    :return: a list of python dictionaries, consisting of metapath-based neighbor pairs and intermediate paths
    """
    outs = []
    for metapath in expected_metapaths:
        # consider only the edges relevant to the expected metapath
        print("Getting neighbor pairs for metapath: ", metapath)
        mask = np.zeros(M.shape, dtype=bool)
        for i in range((len(metapath) - 1) // 2):
            temp = np.zeros(M.shape, dtype=bool)
            temp[np.ix_(type_mask == metapath[i], type_mask == metapath[i + 1])] = True
            temp[np.ix_(type_mask == metapath[i + 1], type_mask == metapath[i])] = True
            mask = np.logical_or(mask, temp)
        partial_g_nx = nx.from_numpy_array((M * mask).astype(int))
        print("Finished getting pairs")

        # only need to consider the former half of the metapath
        # e.g., we only need to consider 0-1-2 for the metapath 0-1-2-1-0
        metapath_to_target = {}
        for source in (type_mask == metapath[0]).nonzero()[0]:
            for target in (type_mask == metapath[(len(metapath) - 1) // 2]).nonzero()[0]:
                # check if there is a possible valid path from source to target node
                has_path = False
                single_source_paths = nx.single_source_shortest_path(
                    partial_g_nx, source, cutoff=(len(metapath) + 1) // 2 - 1)
                if target in single_source_paths:
                    has_path = True

                #if nx.has_path(partial_g_nx, source, target):
                if has_path:
                    shortests = [p for p in nx.all_shortest_paths(partial_g_nx, source, target) if
                                 len(p) == (len(metapath) + 1) // 2]
                    if len(shortests) > 0:
                        metapath_to_target[target] = metapath_to_target.get(target, []) + shortests
        print("finished considering half of the metapath")
        metapath_neighbor_paris = {}
        for key, value in metapath_to_target.items():
            for p1 in value:
                for p2 in value:
                    metapath_neighbor_paris[(p1[0], p2[0])] = metapath_neighbor_paris.get((p1[0], p2[0]), []) + [
                        p1 + p2[-2::-1]]
        outs.append(metapath_neighbor_paris)
    return outs


def get_networkx_graph(neighbor_pairs, type_mask, ctr_ntype):
    indices = np.where(type_mask == ctr_ntype)[0]
    idx_mapping = {}
    for i, idx in enumerate(indices):
        idx_mapping[idx] = i
    G_list = []
    for metapaths in neighbor_pairs:
        edge_count = 0
        sorted_metapaths = sorted(metapaths.items())
        G = nx.MultiDiGraph()
        G.add_nodes_from(range(len(indices)))
        for (src, dst), paths in sorted_metapaths:
            for _ in range(len(paths)):
                G.add_edge(idx_mapping[src], idx_mapping[dst])
                edge_count += 1
        G_list.append(G)
    return G_list


def get_edge_metapath_idx_array(neighbor_pairs):
    all_edge_metapath_idx_array = []
    for metapath_neighbor_pairs in neighbor_pairs:
        sorted_metapath_neighbor_pairs = sorted(metapath_neighbor_pairs.items())
        edge_metapath_idx_array = []
        for _, paths in sorted_metapath_neighbor_pairs:
            edge_metapath_idx_array.extend(paths)
        edge_metapath_idx_array = np.array(edge_metapath_idx_array, dtype=int)
        all_edge_metapath_idx_array.append(edge_metapath_idx_array)
        print(edge_metapath_idx_array.shape)
    return all_edge_metapath_idx_array


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

save_prefix = 'data/preprocessed/IMDB_preprocessed_cast/'
num_ntypes = 4

movies = pd.read_csv('data/raw/IMDB/movie_metadata.csv', encoding='utf-8').dropna(
    axis=0, subset=['actor_1_name', 'director_name']).drop_duplicates(subset=['movie_title', 'title_year'], keep='last').reset_index(drop=True)

# extract labels, and delete movies with unwanted genres
# 0 for action, 1 for comedy, 2 for drama, -1 for others
labels = np.zeros((len(movies)), dtype=int)
#for movie_idx, genres in movies['genres'].iteritems():
for movie_idx, genres in movies['genres'].items():
    labels[movie_idx] = -1
    for genre in genres.split('|'):
        if genre == 'Action':
            labels[movie_idx] = 0
            break
        elif genre == 'Comedy':
            labels[movie_idx] = 1
            break
        elif genre == 'Drama':
            labels[movie_idx] = 2
            break
unwanted_idx = np.where(labels == -1)[0]
movies = movies.drop(unwanted_idx).reset_index(drop=True)
labels = np.delete(labels, unwanted_idx, 0)

directors = list(set(movies['director_name'].dropna()))
directors.sort()
actors = list(set(movies['actor_1_name'].dropna().to_list() +
                  movies['actor_2_name'].dropna().to_list() +
                  movies['actor_3_name'].dropna().to_list()))
actors.sort()


actors_2 = list()
actors_3 = list()

for movie_idx, movie in movies.iterrows():
    acts = list()
    acts_2 = list()
    if movie['actor_1_name'] == movie['actor_1_name']:
        acts.append(movie['actor_1_name'])
    if movie['actor_3_name'] == movie['actor_3_name']:
        acts.append(movie['actor_2_name'])
    if movie['actor_3_name'] == movie['actor_3_name']:
        acts.append(movie['actor_3_name'])
    if movie['movie_title'] == movie['movie_title']:
        acts_2.append(movie['movie_title'])
        #acts.append(movie['movie_title'])
    if movie['title_year'] == movie['title_year']:
        acts_2.append(str(movie['title_year']))
        #acts.append(str(movie['title_year']))
    #acts.sort()
    actor_string = acts[0]
    for act in acts[1:]:
        actor_string += ' '
        actor_string += act
    #actor_string.strip()
    actors_2.append(actor_string)

    actor_string = ''
    for act in acts_2:
        actor_string += act
        actor_string += ' '
    actor_string.strip()
    actors_3.append(actor_string)

vectorizer = CountVectorizer(min_df=2)
movie_X = vectorizer.fit_transform(movies['plot_keywords'].fillna('').values)
#print(movie_X)

'''for actor in actors:
    print(actor)
    actors_2.append(actor.sort())
'''

print(len(actors_2))
print(len(set(actors_2)))
print([item for item in set(actors_2) if actors_2.count(item) > 1])
print([item for item in set(list(movies['plot_keywords'])) if list(movies['plot_keywords']).count(item) > 1])

exit()
"""
author_label = pd.read_csv('data/raw/DBLP/author_label.txt', sep='\t', header=None, names=['author_id', 'label', 'author_name'], keep_default_na=False, encoding='utf-8')
paper_area = pd.read_csv('data/raw/DBLP/paper_label.txt', sep='\t', header=None, names=['paper_id', 'area_id', 'paper_name'], keep_default_na=False, encoding='utf-8')
paper_author = pd.read_csv('data/raw/DBLP/paper_author.txt', sep='\t', header=None, names=['paper_id', 'author_id'], keep_default_na=False, encoding='utf-8')
paper_conf = pd.read_csv('data/raw/DBLP/paper_conf.txt', sep='\t', header=None, names=['paper_id', 'conf_id'], keep_default_na=False, encoding='utf-8')
paper_term = pd.read_csv('data/raw/DBLP/paper_term.txt', sep='\t', header=None, names=['paper_id', 'term_id'], keep_default_na=False, encoding='utf-8')
papers = pd.read_csv('data/raw/DBLP/paper.txt', sep='\t', header=None, names=['paper_id', 'paper_title'], keep_default_na=False, encoding='cp1252')
terms = pd.read_csv('data/raw/DBLP/term.txt', sep='\t', header=None, names=['term_id', 'term'], keep_default_na=False, encoding='utf-8')
confs = pd.read_csv('data/raw/DBLP/conf.txt', sep='\t', header=None, names=['conf_id', 'conf'], keep_default_na=False, encoding='utf-8')
areas = pd.read_csv('data/raw/DBLP/label.txt', sep='\t', header=None, names=['area_id', 'area_name'], keep_default_na=False, encoding='utf-8')


glove_dim = 50
glove_vectors = load_glove_vectors(dim=glove_dim)

# filter out all nodes which does not associated with labeled papers
labeled_authors = author_label['author_id'].to_list()
paper_author = paper_author[paper_author['author_id'].isin(labeled_authors)].reset_index(drop=True)
valid_papers = paper_author['paper_id'].unique()
papers = papers[papers['paper_id'].isin(valid_papers)].reset_index(drop=True)
paper_area = paper_area[paper_area['paper_id'].isin(valid_papers)].reset_index(drop=True)
paper_conf = paper_conf[paper_conf['paper_id'].isin(valid_papers)].reset_index(drop=True)
paper_term = paper_term[paper_term['paper_id'].isin(valid_papers)].reset_index(drop=True)
valid_terms = paper_term['term_id'].unique()
valid_areas = paper_area['area_id'].unique()
terms = terms[terms['term_id'].isin(valid_terms)].reset_index(drop=True)
areas = areas[areas['area_id'].isin(valid_areas)].reset_index(drop=True)




# term lemmatization and grouping
lemmatizer = WordNetLemmatizer()
lemma_id_mapping = {}
lemma_list = []
lemma_id_list = []
i = 0
for _, row in terms.iterrows():
    i += 1
    lemma = lemmatizer.lemmatize(row['term'])
    lemma_list.append(lemma)
    if lemma not in lemma_id_mapping:
        lemma_id_mapping[lemma] = row['term_id']
    lemma_id_list.append(lemma_id_mapping[lemma])
terms['lemma'] = lemma_list
terms['lemma_id'] = lemma_id_list

term_lemma_mapping = {row['term_id']: row['lemma_id'] for _, row in terms.iterrows()}
lemma_id_list = []
for _, row in paper_term.iterrows():
    lemma_id_list.append(term_lemma_mapping[row['term_id']])
paper_term['lemma_id'] = lemma_id_list

paper_term = paper_term[['paper_id', 'lemma_id']]
paper_term.columns = ['paper_id', 'term_id']
paper_term = paper_term.drop_duplicates()
terms = terms[['lemma_id', 'lemma']]
terms.columns = ['term_id', 'term']
terms = terms.drop_duplicates()

# filter out stopwords from terms
stopwords = sklearn_stopwords.union(set(nltk_stopwords.words('english')))
stopword_id_list = terms[terms['term'].isin(stopwords)]['term_id'].to_list()
paper_term = paper_term[~(paper_term['term_id'].isin(stopword_id_list))].reset_index(drop=True)
terms = terms[~(terms['term'].isin(stopwords))].reset_index(drop=True)

author_label = author_label.sort_values('author_id').reset_index(drop=True)
papers = papers.sort_values('paper_id').reset_index(drop=True)
terms = terms.sort_values('term_id').reset_index(drop=True)
confs = confs.sort_values('conf_id').reset_index(drop=True)
areas = areas.sort_values('area_id').reset_index(drop=True)

# extract labels of authors
labels = author_label['label'].to_numpy()
labels = paper_conf['conf_id'].to_numpy()

# build the adjacency matrix for the graph consisting of authors, papers, terms and conferences
# 0 for authors, 1 for papers, 2 for terms, 3 for conferences, 4 for areas
dim = len(papers) + len(author_label) + len(terms) + len(areas)
type_mask = np.zeros((dim), dtype=int)
type_mask[len(papers):len(papers)+len(author_label)] = 1
type_mask[len(author_label)+len(papers):len(author_label)+len(papers)+len(terms)] = 2
type_mask[len(author_label)+len(papers)+len(terms):] = 3
#type_mask[len(author_label)+len(papers)+len(terms)+len(confs):] = 4

paper_id_mapping = {row['paper_id']: i for i, row in papers.iterrows()}
author_id_mapping = {row['author_id']: i + len(papers) for i, row in author_label.iterrows()}
term_id_mapping = {row['term_id']: i + len(author_label) + len(papers) for i, row in terms.iterrows()}
#conf_id_mapping = {row['conf_id']: i + len(author_label) + len(papers) + len(terms) for i, row in confs.iterrows()}
area_id_mapping = {row['area_id']: i + len(author_label) + len(papers) + len(terms) for i, row in areas.iterrows()}

adjM = np.zeros((dim, dim), dtype=int)
for _, row in paper_author.iterrows():
    idx1 = paper_id_mapping[row['paper_id']]
    idx2 = author_id_mapping[row['author_id']]
    adjM[idx1, idx2] = 1
    adjM[idx2, idx1] = 1
for _, row in paper_term.iterrows():
    idx1 = paper_id_mapping[row['paper_id']]
    idx2 = term_id_mapping[row['term_id']]
    adjM[idx1, idx2] = 1
    adjM[idx2, idx1] = 1
'''for _, row in paper_conf.iterrows():
    idx1 = paper_id_mapping[row['paper_id']]
    idx2 = conf_id_mapping[row['conf_id']]
    adjM[idx1, idx2] = 1
    adjM[idx2, idx1] = 1
'''
for _, row in paper_area.iterrows():
    idx1 = paper_id_mapping[row['paper_id']]
    idx2 = area_id_mapping[row['area_id']]
    adjM[idx1, idx2] = 1
    adjM[idx2, idx1] = 1

# use HAN paper's preprocessed data as the features of authors (https://github.com/Jhy1993/HAN)
mat = scipy.io.loadmat('data/raw/DBLP/DBLP4057_GAT_with_idx.mat')
features_author = np.array(list(zip(*sorted(zip(labeled_authors, mat['features']), key=lambda tup: tup[0])))[1])
features_author = scipy.sparse.csr_matrix(features_author)

# use bag-of-words representation of paper titles as the features of papers
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
vectorizer = CountVectorizer(min_df=2, stop_words='english', tokenizer=LemmaTokenizer())#, token_pattern=None)
features_paper = vectorizer.fit_transform(papers['paper_title'].values)

# use pretrained GloVe vectors as the features of terms
features_term = np.zeros((len(terms), glove_dim))
for i, row in terms.iterrows():
    features_term[i] = glove_vectors.get(row['term'], glove_vectors['the'])


expected_metapaths = [
    [(0, 1, 0), (0, 2, 0), (0, 3, 0)],
    [(1, 0, 1), (1, 0, 2, 0, 1), (1, 0, 3, 0, 1)],
    [(2, 0, 2), (2, 0, 1, 0, 2), (2, 0, 3, 0, 2)],
    [(3, 0, 3), (3, 0, 1, 0, 3), (3, 0, 2, 0, 3)]
]
# create the directories if they do not exist
for i in range(1):
    pathlib.Path(save_prefix + '{}'.format(i)).mkdir(parents=True, exist_ok=True)
for i in range(1):
    # get metapath based neighbor pairs
    neighbor_pairs = get_metapath_neighbor_pairs(adjM, type_mask, expected_metapaths[i])
    #print("neighbor pairs: ", neighbor_pairs)
    # construct and save metapath-based networks
    G_list = get_networkx_graph(neighbor_pairs, type_mask, i)
    print("constructed and saveed metapath-based networks")
    
    # save data
    # networkx graph (metapath specific)
    for G, metapath in zip(G_list, expected_metapaths[i]):
        nx.write_adjlist(G, save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '.adjlist')
    print("finished networkx graph (metapath specific)")
    # node indices of edge metapaths
    all_edge_metapath_idx_array = get_edge_metapath_idx_array(neighbor_pairs)
    print("all edge metapath idx array: ", all_edge_metapath_idx_array)
    for metapath, edge_metapath_idx_array in zip(expected_metapaths[i], all_edge_metapath_idx_array):
        np.save(save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '_idx.npy', edge_metapath_idx_array)
    print("saved node & edge indicies of edge metapaths")
# save data
# all nodes adjacency matrix
scipy.sparse.save_npz(save_prefix + 'adjM.npz', scipy.sparse.csr_matrix(adjM))
# all nodes (authors, papers, terms and conferences) features
# currently only have features of authors, papers and terms
scipy.sparse.save_npz(save_prefix + 'features_{}.npz'.format(0), features_author)
scipy.sparse.save_npz(save_prefix + 'features_{}.npz'.format(1), features_paper)
np.save(save_prefix + 'features_{}.npy'.format(2), features_term)
# all nodes (authors, papers, terms, and areas) type labels
np.save(save_prefix + 'node_types.npy', type_mask)
# author labels
np.save(save_prefix + 'labels.npy', labels)
# area labels
np.save(save_prefix + 'areas.npy', areas)
# author train/validation/test splits
rand_seed = 1566911444
train_idx, val_idx = train_test_split(np.arange(len(labels)), test_size=int(len(labels)*.1), random_state=rand_seed)
train_idx, test_idx = train_test_split(train_idx, test_size=len(labels)-(int(len(labels)*.1)*2), random_state=rand_seed)
train_idx.sort()
val_idx.sort()
test_idx.sort()
np.savez(save_prefix + 'train_val_test_idx.npz',
         val_idx=val_idx,
         train_idx=train_idx,
         test_idx=test_idx)


# post-processing for mini-batched training
target_idx_list = np.arange(len(labels))
for metapath in [(0, 1, 0), (0, 2, 0), (0, 3, 0)]:
    edge_metapath_idx_array = np.load(save_prefix + '{}/'.format(0) + '-'.join(map(str, metapath)) + '_idx.npy')
    target_metapaths_mapping = {}
    for target_idx in target_idx_list:
        target_metapaths_mapping[target_idx] = edge_metapath_idx_array[edge_metapath_idx_array[:, 0] == target_idx][:, ::-1]
    out_file = open(save_prefix + '{}/'.format(0) + '-'.join(map(str, metapath)) + '_idx.pickle', 'wb')
    pickle.dump(target_metapaths_mapping, out_file)
    out_file.close()"""