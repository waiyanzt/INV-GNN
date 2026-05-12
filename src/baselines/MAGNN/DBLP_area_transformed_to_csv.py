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
from nltk import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import os


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

save_prefix = 'data/csv/DBLP_area_transformed/'
os.makedirs(save_prefix, exist_ok=True)  
num_ntypes = 4

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

conf_counts = paper_conf['conf_id'].value_counts()
keep_confs = conf_counts[conf_counts >= 1400].index.tolist()

paper_conf = paper_conf[paper_conf['conf_id'].isin(keep_confs)].reset_index(drop=True)
valid_papers = paper_conf['paper_id'].unique()

papers = papers[papers['paper_id'].isin(valid_papers)].reset_index(drop=True)
paper_term = paper_term[paper_term['paper_id'].isin(valid_papers)].reset_index(drop=True)
paper_author = paper_author[paper_author['paper_id'].isin(valid_papers)].reset_index(drop=True)

valid_authors = paper_author['author_id'].unique()
author_label = author_label[author_label['author_id'].isin(valid_authors)].reset_index(drop=True)

labels = paper_conf['conf_id'].to_numpy()
mapped_conf = {}
for label in labels:
    mapped_conf[label] = 0
new_label = 0
for label in mapped_conf:
    mapped_conf[label] = new_label
    paper_conf.loc[paper_conf['conf_id'] == label, 'conf_id'] = new_label
    confs.loc[confs['conf_id'] == label, 'conf_id'] = new_label
    new_label += 1



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
# 0 for papers, 1 for authors, 2 for terms, 3 for conferences, 4 for areas
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

"""edge_paper_author = {"src_id":[], "dst_id": [], "label": []}
edge_paper_term = {"src_id":[], "dst_id": [], "label": []}
edge_author_area = {"src_id":[], "dst_id": [], "label": []}"""
edge_paper_author = {"src_id":[], "dst_id": []}
edge_paper_term = {"src_id":[], "dst_id": []}
edge_author_area = {"src_id":[], "dst_id": []}

edge_author_paper = {"src_id":[], "dst_id": []}
edge_term_paper = {"src_id":[], "dst_id": []}
edge_area_author = {"src_id":[], "dst_id": []}

adjM = np.zeros((dim, dim), dtype=int)
for _, row in paper_author.iterrows():
    idx1 = paper_id_mapping[row['paper_id']]
    idx2 = author_id_mapping[row['author_id']]
    adjM[idx1, idx2] = 1
    adjM[idx2, idx1] = 1
    edge_paper_author["src_id"].append(idx1)
    edge_paper_author["dst_id"].append(idx2)
    #edge_paper_author["label"].append(0)

    edge_author_paper["src_id"].append(idx2)
    edge_author_paper["dst_id"].append(idx1)
    #edge_author_paper["label"].append(1)

for _, row in paper_term.iterrows():
    idx1 = paper_id_mapping[row['paper_id']]
    idx2 = term_id_mapping[row['term_id']]
    adjM[idx1, idx2] = 1
    adjM[idx2, idx1] = 1
    edge_paper_term["src_id"].append(idx1)
    edge_paper_term["dst_id"].append(idx2)
    #edge_paper_term["label"].append(2)

    edge_term_paper["src_id"].append(idx2)
    edge_term_paper["dst_id"].append(idx1)
    #edge_term_paper["label"].append(3)
'''for _, row in paper_conf.iterrows():
    idx1 = paper_id_mapping[row['paper_id']]
    idx2 = conf_id_mapping[row['conf_id']]
    adjM[idx1, idx2] = 1
    adjM[idx2, idx1] = 1
'''
for _, row in author_label.iterrows():
    idx1 = author_id_mapping[row['author_id']]
    idx2 = area_id_mapping[row['label']]
    adjM[idx1, idx2] = 1
    adjM[idx2, idx1] = 1
    edge_author_area["src_id"].append(idx1)
    edge_author_area["dst_id"].append(idx2)
    #edge_author_area["label"].append(4)

    edge_area_author["src_id"].append(idx2)
    edge_area_author["dst_id"].append(idx1)
    #edge_area_author["label"].append(5)

# use HAN paper's preprocessed data as the features of authors (https://github.com/Jhy1993/HAN)
mat = scipy.io.loadmat('data/raw/DBLP/DBLP4057_GAT_with_idx.mat')
mat_feat, lab_authors = [], []
for l, m in zip(labeled_authors, mat['features']):
    if l in valid_authors:
        lab_authors.append(l)
        mat_feat.append(m)
#features_author = np.array(list(zip(*sorted(zip(labeled_authors, mat['features']), key=lambda tup: tup[0])))[1])
features_author = np.array(list(zip(*sorted(zip(lab_authors, mat_feat), key=lambda tup: tup[0])))[1])
features_author = scipy.sparse.csr_matrix(features_author)
node_author = {"node_id": lab_authors, "feat": [','.join(str(item) for item in innerlist) for innerlist in mat_feat]}
for nid, node_id in enumerate(node_author["node_id"]):
    node_author["node_id"][nid] = author_id_mapping[node_id]
# use bag-of-words representation of paper titles as the features of papers
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
vectorizer = CountVectorizer(min_df=2, stop_words='english', tokenizer=LemmaTokenizer())#, token_pattern=None)
features_paper = vectorizer.fit_transform(papers['paper_title'].values)
node_paper = {"node_id": papers['paper_id'].to_list(), "feat": [','.join(str(item) for item in innerlist) for innerlist in features_paper.toarray()], "label": labels}
for nid, node_id in enumerate(node_paper["node_id"]):
    node_paper["node_id"][nid] = paper_id_mapping[node_id]

# use pretrained GloVe vectors as the features of terms
features_term = np.zeros((len(terms), glove_dim))
feat_term = []
for i, row in terms.iterrows():
    features_term[i] = glove_vectors.get(row['term'], glove_vectors['the'])
    feat_term.append(glove_vectors.get(row['term'], glove_vectors['the']))
node_term = {"node_id": terms["term_id"].to_list(), "feat": [','.join(str(item) for item in innerlist) for innerlist in feat_term]}
for nid, node_id in enumerate(node_term["node_id"]):
    node_term["node_id"][nid] = term_id_mapping[node_id]

node_area = {"node_id": areas["area_id"].to_list()}
for nid, node_id in enumerate(node_area["node_id"]):
    node_area["node_id"][nid] = area_id_mapping[node_id]

# author train/validation/test splits
rand_seed = 1566911444
train_idx, val_idx = train_test_split(np.arange(len(labels)), test_size=int(len(labels)*.1), random_state=rand_seed)
train_idx, test_idx = train_test_split(train_idx, test_size=len(labels)-(int(len(labels)*.1)*2), random_state=rand_seed)
train_idx.sort()
val_idx.sort()
test_idx.sort()

train_mask, test_mask, val_mask  = [], [], []
for l in np.arange(len(labels)):
    if l in train_idx:
        train_mask.append(True)
        test_mask.append(False)
        val_mask.append(False)
    elif l in test_idx:
        test_mask.append(True)
        train_mask.append(False)
        val_mask.append(False)
    elif l in val_idx:
        val_mask.append(True)
        train_mask.append(False)
        test_mask.append(False)
    else:
        print("something is wrong")

node_paper["train_mask"] = train_mask
node_paper["test_mask"] = test_mask
node_paper["val_mask"] = val_mask



edge_author_area = pd.DataFrame.from_dict(edge_author_area)
edge_paper_term = pd.DataFrame.from_dict(edge_paper_term)
edge_paper_author = pd.DataFrame.from_dict(edge_paper_author)

edge_area_author = pd.DataFrame.from_dict(edge_area_author)
edge_term_paper = pd.DataFrame.from_dict(edge_term_paper)
edge_author_paper = pd.DataFrame.from_dict(edge_author_paper)

node_author = pd.DataFrame.from_dict(node_author)
node_term = pd.DataFrame.from_dict(node_term)
node_area = pd.DataFrame.from_dict(node_area)
node_paper = pd.DataFrame.from_dict(node_paper)

edge_author_area.to_csv(save_prefix + "author_area.csv", index=False)
edge_paper_term.to_csv(save_prefix + "paper_term.csv", index=False)
edge_paper_author.to_csv(save_prefix + "paper_author.csv", index=False)

edge_area_author.to_csv(save_prefix + "area_author.csv", index=False)
edge_term_paper.to_csv(save_prefix + "term_paper.csv", index=False)
edge_author_paper.to_csv(save_prefix + "author_paper.csv", index=False)

node_author.to_csv(save_prefix + "author.csv", index=False)
node_term.to_csv(save_prefix + "term.csv", index=False)
node_area.to_csv(save_prefix + "area.csv", index=False)
node_paper.to_csv(save_prefix + "paper.csv", index=False)