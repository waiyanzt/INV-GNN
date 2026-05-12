# DBLP paper–venue LP preprocessing (Var2: area-venue)

import os, pickle, pathlib, random
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

OUT = 'data/preprocessed/DBLP_lp_pc_var2_ct_from_pt/'
RAW = 'data/raw/DBLP/'
SHARED = 'data/preprocessed/DBLP_shared_splits/DBLP_pc_shared_splits.npz'
SEED = 1566911444
MIN_CONF = 0

MAX_PAPERS_PER_AUTHOR   = 100
MAX_PAPERS_PER_TERM     = 50
MAX_PAPERS_PER_AREA     = 100
MAX_PAPERS_PER_CONF     = 200
MAX_PAPERS_PER_AUTHOR_V = 30
MAX_PAPERS_PER_TERM_V   = 30
MAX_PAPERS_PER_AREA_V   = 30

random.seed(SEED); np.random.seed(SEED)
os.makedirs(OUT, exist_ok=True)

print('1) load raw tables')
author = pd.read_csv(os.path.join(RAW,'author_label.txt'), sep='\t',
                     names=['author_id','label','author_name'],
                     header=None, encoding='utf-8')
pa = pd.read_csv(os.path.join(RAW,'paper_author.txt'), sep='\t',
                 names=['paper_id','author_id'], header=None, encoding='utf-8')
pc = pd.read_csv(os.path.join(RAW,'paper_conf.txt'), sep='\t',
                 names=['paper_id','conf_id'], header=None, encoding='utf-8')
pt = pd.read_csv(os.path.join(RAW,'paper_term.txt'), sep='\t',
                 names=['paper_id','term_id'], header=None, encoding='utf-8')


"""
max_pav = pc[['paper_id','conf_id']].drop_duplicates().to_numpy(dtype=np.int64)
max_ptv = pc[['paper_id','conf_id']].drop_duplicates().to_numpy(dtype=np.int64)
max_parv = pc[['paper_id','conf_id']].drop_duplicates().to_numpy(dtype=np.int64)
"""

print('2) filter to labeled authors + papers with venues')
valid_authors = set(author['author_id'])
pa = pa[pa['author_id'].isin(valid_authors)].reset_index(drop=True)
valid_papers = set(pa['paper_id'])
pc = pc[pc['paper_id'].isin(valid_papers)].reset_index(drop=True)
pt = pt[pt['paper_id'].isin(valid_papers)].reset_index(drop=True)
pa = pa[pa['paper_id'].isin(valid_papers)].reset_index(drop=True)

big_confs = pc['conf_id'].value_counts().loc[lambda x: x>=MIN_CONF].index
pc = pc[pc['conf_id'].isin(big_confs)].reset_index(drop=True)

valid_papers = set(pc['paper_id'])
pt = pt[pt['paper_id'].isin(valid_papers)].reset_index(drop=True)
pa = pa[pa['paper_id'].isin(valid_papers)].reset_index(drop=True)

# --- Enforce the exact 10% subset used when creating shared splits ---
shared = np.load(SHARED)
if 'paper_subset' in shared.files:
    keep_papers = set(shared['paper_subset'].tolist())
    before = len(valid_papers)
    # keep only papers that survived the earlier filters AND are in the subset
    keep_papers = keep_papers & valid_papers
    pc = pc[pc['paper_id'].isin(keep_papers)].reset_index(drop=True)
    pa = pa[pa['paper_id'].isin(keep_papers)].reset_index(drop=True)
    pt = pt[pt['paper_id'].isin(keep_papers)].reset_index(drop=True)
    valid_papers = set(pc['paper_id'])
    print(f"→ Enforced paper subset from SHARED: kept {len(valid_papers)} / {before} papers")
else:
    print("→ [warn] SHARED file has no 'paper_subset'; using full paper universe")


print('3) paper→area via author labels (keep P–R edges)')
a2r = author.set_index('author_id')['label']
pr = (pa.assign(area_id=pa['author_id'].map(a2r))
        [['paper_id','area_id']].drop_duplicates().dropna().reset_index(drop=True))
print(f"   papers={pc['paper_id'].nunique()} authorships={len(pa)} "
      f"confs={pc['conf_id'].nunique()} terms={pt['term_id'].nunique()} areas={pr['area_id'].nunique()}")


max_pa = pa[['paper_id','author_id']].drop_duplicates().to_numpy(dtype=np.int64)
max_pt = pt[['paper_id','term_id']].drop_duplicates().to_numpy(dtype=np.int64)
max_pr = pr[['paper_id','area_id']].drop_duplicates().to_numpy(dtype=np.int64)
max_pc = pc[['paper_id','conf_id']].drop_duplicates().to_numpy(dtype=np.int64)
print("MAX_PAPERS_PER_AUTHOR", len(max_pa))
print("MAX_PAPERS_PER_TERM", len(max_pt))
print("MAX_PAPERS_PER_AREA", len(max_pr))
print("MAX_PAPERS_PER_CONF", len(max_pc))




print('4) reindex per type + offsets')
Au_ids = sorted(author['author_id'].unique())
Pa_ids = sorted(valid_papers)
Te_ids = sorted(pt['term_id'].unique())
Co_ids = sorted(pc['conf_id'].unique())
Ar_ids = sorted(pr['area_id'].unique())
Au_map = {a:i for i,a in enumerate(Au_ids)}
Pa_map = {p:i for i,p in enumerate(Pa_ids)}
Te_map = {t:i for i,t in enumerate(Te_ids)}
Co_map = {c:i for i,c in enumerate(Co_ids)}
Ar_map = {r:i for i,r in enumerate(Ar_ids)}
A,P,T,C,R = len(Au_map), len(Pa_map), len(Te_map), len(Co_map), len(Ar_map)
offA,offP,offT,offC,offR = 0, A, A+P, A+P+T, A+P+T+C
N = A+P+T+C+R
print(f"Num Authors: {A}, Num Papers: {P}, Num Terms: {T}, Num Confs: {C}, Num Areas: {R}")


print('5) load SHARED splits and map to local ids')
shared = np.load(SHARED)

def map_pairs(arr, Pmap, Cmap, name):
    df = pd.DataFrame(arr, columns=['paper_id','conf_id'])
    df['paper_id'] = df['paper_id'].map(Pmap)
    df['conf_id']  = df['conf_id'].map(Cmap)
    out = df.dropna().to_numpy(dtype=np.int32)
    dropped = len(arr) - len(out)
    if dropped:
        print(f'   [warn] {name}: dropped {dropped} pairs not present after filtering')
    return out

train_pos = map_pairs(shared['train_pos'], Pa_map, Co_map, 'train_pos')
val_pos   = map_pairs(shared['val_pos'],   Pa_map, Co_map, 'val_pos')
test_pos  = map_pairs(shared['test_pos'],  Pa_map, Co_map, 'test_pos')
train_neg = map_pairs(shared['train_neg'], Pa_map, Co_map, 'train_neg')
val_neg   = map_pairs(shared['val_neg'],   Pa_map, Co_map, 'val_neg')
test_neg  = map_pairs(shared['test_neg'],  Pa_map, Co_map, 'test_neg')

print(f"   splits: train_pos={len(train_pos)}, val_pos={len(val_pos)}, test_pos={len(test_pos)}")
print(f"           train_neg={len(train_neg)}, val_neg={len(val_neg)}, test_neg={len(test_neg)}")

print('6) adjacency (TRAIN P–C only; all P–A, P–T, P–R)')
type_mask = np.zeros(N, dtype=np.int32)
"""type_mask[offP:offP+P] = 1
type_mask[offT:offT+T] = 2
type_mask[offC:offC+C] = 3
type_mask[offR:offR+R] = 4"""
type_mask[A:A+P] = 1
type_mask[A+P:A+P+T] = 2
type_mask[A+P+T:A+P+T+C] = 3
type_mask[A+P+T+C:] = 4

adjM = np.zeros((N,N), dtype=np.int32)
#A-P / 0-1
for p,a in pa[['paper_id','author_id']].itertuples(index=False):
    if a in Au_map and p in Pa_map:
        ai,pi = 0+Au_map[a], A+Pa_map[p]
        adjM[ai,pi] = 1
        adjM[pi,ai] = 1
# P–T / 1-2
for p,t in pt[['paper_id','term_id']].itertuples(index=False):
    if p in Pa_map and t in Te_map:
        pi,ti = A+Pa_map[p], A+P+Te_map[t]
        adjM[pi,ti] = 1
        adjM[ti,pi] = 1
# P–C / 1-3
for p,c in pc[['paper_id','conf_id']].itertuples(index=False):
    if p in Pa_map and c in Co_map:
        pi,ci = A+Pa_map[p], A+P+T+Co_map[c]
        adjM[pi,ci] = 1
        adjM[ci,pi] = 1
# C–R / 3-4
cr = (pc[['paper_id','conf_id']]
        .drop_duplicates()
        .merge(pr[['paper_id','area_id']].drop_duplicates(), on='paper_id')
        [['conf_id','area_id']].drop_duplicates().reset_index(drop=True))

for c, r in cr.itertuples(index=False):
    if c in Co_map and r in Ar_map:
        ci, ri = A+P+T+Co_map[c], A+P+T+C+Ar_map[r]
        adjM[ci,ri] = 1
        adjM[ri,ci] = 1

# 8 edge types
author_paper_list = {i: adjM[i, A:A+P].nonzero()[0] for i in range(A)}
paper_author_list = {i: adjM[A + i, 0:A].nonzero()[0] for i in range(P)}
term_paper_list = {i: adjM[A + P + i, A:A+P].nonzero()[0] for i in range(T)}
paper_term_list = {i: adjM[A + i, A+P:A+P+T].nonzero()[0] for i in range(P)}
conf_paper_list = {i: adjM[A+P+T+i, A:A+P].nonzero()[0] for i in range(C)}
paper_conf_list = {i: adjM[A + i, A+P+T:A+P+T+C].nonzero()[0] for i in range(P)}
# area_author_list = {i: adjM[A+P+T+C+i, 0:A].nonzero()[0] for i in range(R)}
# author_area_list = {i: adjM[i, A+P+T+C:].nonzero()[0] for i in range(A)}
area_conf_list    = {i: adjM[A+P+T+C+i, A+P+T:A+P+T+C].nonzero()[0] for i in range(R)}
conf_area_list    = {i: adjM[A+P+T+i,     A+P+T+C:        ].nonzero()[0] for i in range(C)}


#1-0-1
print("Starting Metapath: 101")
p_a_p = []
for a, p_list in author_paper_list.items():
    p_a_p.extend([(p1, a, p2) for p1 in p_list for p2 in p_list])
p_a_p = np.array(p_a_p)
p_a_p[:, [0,2]] += A
sorted_index = sorted(list(range(len(p_a_p))), key=lambda i : p_a_p[i, [0, 2, 1]].tolist())
p_a_p = p_a_p[sorted_index]

#1-2-1
print("Starting Metapath: 121")
p_t_p = []
for t, p_list in term_paper_list.items():
    p_t_p.extend([(p1, t, p2) for p1 in p_list for p2 in p_list])
p_t_p = np.array(p_t_p)
p_t_p[:, [0,2]] += A
p_t_p[:, 1] += A+P
sorted_index = sorted(list(range(len(p_t_p))), key=lambda i : p_t_p[i, [0, 2, 1]].tolist())
p_t_p = p_t_p[sorted_index]

#1-3-1
print("Starting Metapath: 131")
p_c_p = []
for c, p_list in conf_paper_list.items():
    p_c_p.extend([(p1, c, p2) for p1 in p_list for p2 in p_list])
p_c_p = np.array(p_c_p)
p_c_p[:, [0,2]] += A
p_c_p[:, 1] += A+P+T
sorted_index = sorted(list(range(len(p_c_p))), key=lambda i : p_c_p[i, [0, 2, 1]].tolist())
p_c_p = p_c_p[sorted_index]

#0-4-0
# print("Starting Metapath: 040")
# a_r_a = []
# for r, a_list in area_author_list.items():
#     a_r_a.extend([(a1, r, a2) for a1 in a_list for a2 in a_list])
# a_r_a = np.array(a_r_a)
# a_r_a[:, 1] += A+P+T
# sorted_index = sorted(list(range(len(a_r_a))), key=lambda i : a_r_a[i, [0, 2, 1]].tolist())
# a_r_a = a_r_a[sorted_index]

#3-4-3
print("Starting Metapath: 343")
c_r_c = []
for r, c_list in area_conf_list.items():
    c_r_c.extend([(c1, r, c2) for c1 in c_list for c2 in c_list])
c_r_c = np.array(c_r_c)
# add offsets: C (ends), R (middle)
if c_r_c.size:
    c_r_c[:, [0,2]] += A+P+T
    c_r_c[:, 1]     += A+P+T+C
    sorted_index = sorted(list(range(len(c_r_c))), key=lambda i : c_r_c[i, [0, 2, 1]].tolist())
    c_r_c = c_r_c[sorted_index]


#1-0-4-0-1
# print("Starting Metapath: 10401")
# p_a_r_a_p = []
# for a1, r, a2 in a_r_a:
#     if len(author_paper_list[a1]) == 0 or len(author_paper_list[a2]) == 0:
#         continue
#     candidate_p1_list = np.random.choice(len(author_paper_list[a1]), max(1,int(0.2 * len(author_paper_list[a1]))), replace=False)
#     candidate_p1_list = author_paper_list[a1][candidate_p1_list]
#     candidate_p2_list = np.random.choice(len(author_paper_list[a2]), max(1,int(0.2 * len(author_paper_list[a2]))), replace=False)
#     candidate_p2_list = author_paper_list[a2][candidate_p2_list]
#     p_a_r_a_p.extend([(p1, a1, r, a2, p2) for p1 in candidate_p1_list for p2 in candidate_p2_list])
# p_a_r_a_p = np.array(p_a_r_a_p)
# p_a_r_a_p[:, [0,4]] += A
# sorted_index = sorted(list(range(len(p_a_r_a_p))), key=lambda i : p_a_r_a_p[i, [0, 4, 1, 2, 3]].tolist())
# p_a_r_a_p = p_a_r_a_p[sorted_index]

#1-3-4-3-1
print("Starting Metapath: 13431")
p_c_r_c_p = []
for c1, r, c2 in c_r_c:
    c1i = c1 - (A+P+T)
    c2i = c2 - (A+P+T)
    if len(conf_paper_list[c1i]) == 0 or len(conf_paper_list[c2i]) == 0:
        continue
    cand_p1_idx = np.random.choice(len(conf_paper_list[c1i]), max(1, int(0.2*len(conf_paper_list[c1i]))), replace=False)
    cand_p2_idx = np.random.choice(len(conf_paper_list[c2i]), max(1, int(0.2*len(conf_paper_list[c2i]))), replace=False)
    p1_list = conf_paper_list[c1i][cand_p1_idx]
    p2_list = conf_paper_list[c2i][cand_p2_idx]
    p_c_r_c_p.extend([(p1, c1, r, c2, p2) for p1 in p1_list for p2 in p2_list])
p_c_r_c_p = np.array(p_c_r_c_p)
if p_c_r_c_p.size:
    # add offsets only to papers (c1,c2,r already global)
    p_c_r_c_p[:, [0,4]] += A
    sorted_index = sorted(list(range(len(p_c_r_c_p))), key=lambda i : p_c_r_c_p[i, [0, 4, 1, 2, 3]].tolist())
    p_c_r_c_p = p_c_r_c_p[sorted_index]


#3-1-0-1-3
print("Starting Metapath: 31013")
c_p_a_p_c = []
for p1, a, p2 in p_a_p:
    if len(paper_conf_list[p1 - A]) == 0 or len(paper_conf_list[p2 - A]) == 0:
        continue
    candidate_c1_list = np.random.choice(len(paper_conf_list[p1 - A]), max(1,int(0.2 * len(paper_conf_list[p1 - A]))), replace=False)
    candidate_c1_list = paper_conf_list[p1 - A][candidate_c1_list]
    candidate_c2_list = np.random.choice(len(paper_conf_list[p2 - A]), max(1,int(0.2 * len(paper_conf_list[p2 - A]))), replace=False)
    candidate_c2_list = paper_conf_list[p2 - A][candidate_c2_list]
    c_p_a_p_c.extend([(c1, p1, a, p2, c2) for c1 in candidate_c1_list for c2 in candidate_c2_list])
c_p_a_p_c = np.array(c_p_a_p_c)
c_p_a_p_c[:, [0,4]] += A+P+T
sorted_index = sorted(list(range(len(c_p_a_p_c))), key=lambda i : c_p_a_p_c[i, [0, 4, 1, 2, 3]].tolist())
c_p_a_p_c = c_p_a_p_c[sorted_index]

#3-1-2-1-3
# print("Starting Metapath: 31213")
# c_p_t_p_c = []
# for p1, t, p2 in p_t_p:
#     if len(paper_conf_list[p1 - A]) == 0 or len(paper_conf_list[p2 - A]) == 0:
#         continue
#     candidate_c1_list = np.random.choice(len(paper_conf_list[p1 - A]), max(1,int(0.2 * len(paper_conf_list[p1 - A]))), replace=False)
#     candidate_c1_list = paper_conf_list[p1 - A][candidate_c1_list]
#     candidate_c2_list = np.random.choice(len(paper_conf_list[p2 - A]), max(1,int(0.2 * len(paper_conf_list[p2 - A]))), replace=False)
#     candidate_c2_list = paper_conf_list[p2 - A][candidate_c2_list]
#     c_p_t_p_c.extend([(c1, p1, t, p2, c2) for c1 in candidate_c1_list for c2 in candidate_c2_list])
# c_p_t_p_c = np.array(c_p_t_p_c)
# c_p_t_p_c[:, [0,4]] += A+P+T
# sorted_index = sorted(list(range(len(c_p_t_p_c))), key=lambda i : c_p_t_p_c[i, [0, 4, 1, 2, 3]].tolist())
# c_p_t_p_c = c_p_t_p_c[sorted_index]

#3-1-3
print("Starting Metapath: 313")
c_p_c = []
for p, c_list in paper_conf_list.items():
    c_p_c.extend([(c1, p, c2) for c1 in c_list for c2 in c_list])
c_p_c = np.array(c_p_c)
c_p_c[:, [0,2]] += A+P+T
c_p_c[:, 1] += A
sorted_index = sorted(list(range(len(c_p_c))), key=lambda i : c_p_c[i, [0, 2, 1]].tolist())
c_p_c = c_p_c[sorted_index]

#3-1-0-4-0-1-3
# print("Starting Metapath: 3104013")
# c_p_a_r_a_p_c = []
# for p1, a1, r, a2, p2 in p_a_r_a_p:
#     if len(paper_conf_list[p1 - A]) == 0 or len(paper_conf_list[p2 - A]) == 0:
#         continue
#     candidate_c1_list = np.random.choice(len(paper_conf_list[p1 - A]), max(1,int(0.2 * len(paper_conf_list[p1 - A]))), replace=False)
#     candidate_c1_list = paper_conf_list[p1 - A][candidate_c1_list]
#     candidate_c2_list = np.random.choice(len(paper_conf_list[p2 - A]), max(1,int(0.2 * len(paper_conf_list[p2 - A]))), replace=False)
#     candidate_c2_list = paper_conf_list[p2 - A][candidate_c2_list]
#     c_p_a_r_a_p_c.extend([(c1, p1, a1, r, a2, p2, c2) for c1 in candidate_c1_list for c2 in candidate_c2_list])
# c_p_a_r_a_p_c = np.array(c_p_a_r_a_p_c)
# c_p_a_r_a_p_c[:, [0,6]] += A+P+T
# sorted_index = sorted(list(range(len(c_p_a_r_a_p_c))), key=lambda i : c_p_a_r_a_p_c[i, [0, 6, 1, 2, 3, 4, 5]].tolist())
# c_p_a_r_a_p_c = c_p_a_r_a_p_c[sorted_index]

"""
rows, cols, data = [], [], []
# A–P
for p,a in pa[['paper_id','author_id']].itertuples(index=False):
    if a in Au_map and p in Pa_map:
        ai,pi = offA+Au_map[a], offP+Pa_map[p]
        rows += [ai,pi]; cols += [pi,ai]; data += [1,1]
# P–T
for p,t in pt[['paper_id','term_id']].itertuples(index=False):
    if p in Pa_map and t in Te_map:
        pi,ti = offP+Pa_map[p], offT+Te_map[t]
        rows += [pi,ti]; cols += [ti,pi]; data += [1,1]
# P–R
for p,r in pr[['paper_id','area_id']].itertuples(index=False):
    if p in Pa_map and r in Ar_map:
        pi,ri = offP+Pa_map[p], offR+Ar_map[r]
        rows += [pi,ri]; cols += [ri,pi]; data += [1,1]
# P–C (TRAIN ONLY)
for pL,cL in train_pos:
    pi,ci = offP+pL, offC+cL
    rows += [pi,ci]; cols += [ci,pi]; data += [1,1]

adjM = sp.csr_matrix((data,(rows,cols)), shape=(N,N), dtype=np.int8)
"""



with open(os.path.join(OUT,'node_maps.pkl'),'wb') as f:
    pickle.dump({'author_idx':Au_map,'paper_idx':Pa_map,'term_idx':Te_map,'conf_idx':Co_map,'area_idx':Ar_map,
                 'offsets':{'A':offA,'P':offP,'T':offT,'C':offC,'R':offR}}, f)

"""
print('7) neighbor dicts (local ids)')
P_by_A, P_by_T, P_by_R, P_by_C = {}, {}, {}, {}
for p,a in pa[['paper_id','author_id']].itertuples(index=False):
    if a in Au_map and p in Pa_map:
        P_by_A.setdefault(Au_map[a], []).append(Pa_map[p])
for p,t in pt[['paper_id','term_id']].itertuples(index=False):
    if t in Te_map and p in Pa_map:
        P_by_T.setdefault(Te_map[t], []).append(Pa_map[p])
for p,r in pr[['paper_id','area_id']].itertuples(index=False):
    if r in Ar_map and p in Pa_map:
        P_by_R.setdefault(Ar_map[r], []).append(Pa_map[p])
for pL,cL in train_pos:  # TRAIN ONLY
    P_by_C.setdefault(cL, []).append(pL)
P_by_A = {k: np.array(v, dtype=np.int32) for k,v in P_by_A.items()}
P_by_T = {k: np.array(v, dtype=np.int32) for k,v in P_by_T.items()}
P_by_R = {k: np.array(v, dtype=np.int32) for k,v in P_by_R.items()}
P_by_C = {k: np.array(v, dtype=np.int32) for k,v in P_by_C.items()}
C_of_P_train = {pL: cL for pL,cL in train_pos}

print('8) metapaths (TRAIN P–C where relevant)')
def build_pap(P_by_X, off_mid, cap):
    rows=[]
    for mid, plist in tqdm(P_by_X.items(), total=len(P_by_X)):
        sub = plist if len(plist)<=cap else plist[np.random.choice(len(plist), cap, replace=False)]
        n=len(sub)
        if n:
            p1 = np.repeat(sub, n); p2 = np.tile(sub, n)
            rows.append(np.stack([offP+p1, off_mid+np.full(n*n, mid, dtype=np.int32), offP+p2],1))
    return np.concatenate(rows,0) if rows else np.empty((0,3),dtype=np.int32)

def build_cpxpc(P_by_X, off_mid, cap):
    rows=[]
    for mid, plist in tqdm(P_by_X.items(), total=len(P_by_X)):
        sub = plist if len(plist)<=cap else plist[np.random.choice(len(plist), cap, replace=False)]
        sub = np.array([p for p in sub if p in C_of_P_train], dtype=np.int32)
        n=len(sub)
        if n:
            p1 = np.repeat(sub, n); p2 = np.tile(sub, n)
            c1 = np.array([C_of_P_train[p] for p in p1], dtype=np.int32)
            c2 = np.array([C_of_P_train[p] for p in p2], dtype=np.int32)
            rows.append(np.stack([offC+c1, offP+p1, off_mid+np.full(n*n, mid, dtype=np.int32), offP+p2, offC+c2],1))
    return np.concatenate(rows,0) if rows else np.empty((0,5),dtype=np.int32)

arr_101   = build_pap(P_by_A, offA, MAX_PAPERS_PER_AUTHOR)   # (1,0,1)
arr_121   = build_pap(P_by_T, offT, MAX_PAPERS_PER_TERM)     # (1,2,1)
arr_141   = build_pap(P_by_R, offR, MAX_PAPERS_PER_AREA)     # (1,4,1)
arr_131 = []
for c, plist in tqdm(P_by_C.items(), total=len(P_by_C)):     # (1,3,1) TRAIN
    sub = plist if len(plist)<=MAX_PAPERS_PER_CONF else plist[np.random.choice(len(plist), MAX_PAPERS_PER_CONF, replace=False)]
    n=len(sub)
    if n:
        p1 = np.repeat(sub, n); p2 = np.tile(sub, n)
        arr_131.append(np.stack([offP+p1, offC+np.full(n*n, c, dtype=np.int32), offP+p2],1))
arr_131 = np.concatenate(arr_131,0) if len(arr_131) else np.empty((0,3),dtype=np.int32)

arr_31013 = build_cpxpc(P_by_A, offA, MAX_PAPERS_PER_AUTHOR_V)  # (3,1,0,1,3)
arr_31213 = build_cpxpc(P_by_T, offT, MAX_PAPERS_PER_TERM_V)    # (3,1,2,1,3)
arr_31413 = build_cpxpc(P_by_R, offR, MAX_PAPERS_PER_AREA_V)    # (3,1,4,1,3)

def sort_meta(arr, L):
    if arr.size == 0: return arr
    idx = ( [0,2,1] if L==3 else [0,4,1,2,3] )
    idx = sorted(range(len(arr)), key=lambda i: arr[i, idx].tolist())
    return arr[idx]

arr_101   = sort_meta(arr_101,   3)
arr_121   = sort_meta(arr_121,   3)
arr_141   = sort_meta(arr_141,   3)
arr_131   = sort_meta(arr_131,   3)
arr_31013 = sort_meta(arr_31013, 5)
arr_31213 = sort_meta(arr_31213, 5)
arr_31413 = sort_meta(arr_31413, 5)

expected_metapaths = [
    [(1,0,1), (1,2,1), (1,4,1), (1,3,1)],
    [(3,1,0,1,3), (3,1,2,1,3), (3,1,4,1,3)]
]

expected_metapaths = [
    [(1,0,1), (1,2,1), (1,0,4,0,1), (1,3,1)],
    [(3,1,0,1,3), (3,1,2,1,3), (3,1,3), (3,1,0,4,0,1,3)]
]
"""


# expected_metapaths = [
#     [(1,0,1), (1,2,1), (1,3,1), (1,3,4,3,1)],
#     [(3,1,0,1,3), (3,1,2,1,3), (3,1,3,), (3,4,3)]
# ]
expected_metapaths = [
    [(1,0,1), (1,2,1), (1,3,1), (1,3,4,3,1)],
    [(3,1,0,1,3), (3,1,3,)]
]
print('9) write *_idx.pickle and .adjlist')
for i in range(2):
    pathlib.Path(os.path.join(OUT, f'{i}')).mkdir(parents=True, exist_ok=True)
"""    
mp = {
    (1,0,1): arr_101, (1,2,1): arr_121, (1,4,1): arr_141, (1,3,1): arr_131,
    (3,1,0,1,3): arr_31013, (3,1,2,1,3): arr_31213, (3,1,4,1,3): arr_31413
}
"""
mp = {
    (1,0,1): p_a_p, 
    (1,2,1): p_t_p, 
    (1,3,1): p_c_p,
    # (1,0,4,0,1): p_a_r_a_p, 
    (1,3,4,3,1): p_c_r_c_p,
    (3,1,0,1,3): c_p_a_p_c, 
    # (3,1,2,1,3): c_p_t_p_c,
    (3,1,3): c_p_c,
    # (3,1,0,4,0,1,3): c_p_a_r_a_p_c
    # (3,4,3): c_r_c
}

target_idx_lists = [np.arange(P), np.arange(C)]
offset_list      = [offP, offC]
offset_list = [A, A+P+T]
for i, metas in enumerate(expected_metapaths):
    for metapath in metas:
        edge_arr = mp.get(tuple(metapath), np.empty((0,len(metapath)), dtype=np.int32))
        name = '-'.join(map(str, metapath))
        with open(os.path.join(OUT, f'{i}', f'{name}_idx.pickle'), 'wb') as f:
            per_target, left, right = {}, 0, 0
            for tgt in target_idx_lists[i]:
                while right < len(edge_arr) and edge_arr[right, 0] == tgt + offset_list[i]:
                    right += 1
                per_target[tgt] = edge_arr[left:right, ::-1]
                left = right
            pickle.dump(per_target, f)
        with open(os.path.join(OUT, f'{i}', f'{name}.adjlist'), 'w') as f:
            left, right = 0, 0
            for tgt in target_idx_lists[i]:
                while right < len(edge_arr) and edge_arr[right, 0] == tgt + offset_list[i]:
                    right += 1
                nbrs = edge_arr[left:right, -1] - offset_list[i]
                f.write(f'{tgt} ' + ' '.join(map(str, nbrs)) + '\n' if len(nbrs)>0 else f'{tgt}\n')
                left = right

print('10) save adjM + node_types.npy')
sp.save_npz(os.path.join(OUT,'adjM.npz'), sp.csr_matrix(adjM))
np.save(os.path.join(OUT,'node_types.npy'), type_mask)

print('11) save SHARED splits (already mapped)')
np.savez(os.path.join(OUT,'train_val_test_pos_paper_conf.npz'),
         train_pos=train_pos, val_pos=val_pos, test_pos=test_pos)
np.savez(os.path.join(OUT,'train_val_test_neg_paper_conf.npz'),
         train_neg=train_neg, val_neg=val_neg, test_neg=test_neg)

print('done:', OUT)