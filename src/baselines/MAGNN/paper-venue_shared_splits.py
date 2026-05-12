import os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

RAW = 'data/raw/DBLP/'
OUT = 'data/preprocessed/DBLP_shared_splits/'
SEED = 1566911444
TEST_PCT = 0.20
VAL_PCT  = 0.10
MIN_CONF = 0           # conference filter
PAPER_KEEP_FRAC = 1.0 #0.10 # keep 10% of papers

os.makedirs(OUT, exist_ok=True)
rng = np.random.RandomState(SEED)

print("→ Loading raw tables")
al = pd.read_csv(os.path.join(RAW,'author_label.txt'), sep='\t',
                 names=['author_id','label','author_name'], header=None, encoding='utf-8')
pa = pd.read_csv(os.path.join(RAW,'paper_author.txt'), sep='\t',
                 names=['paper_id','author_id'], header=None, encoding='utf-8')
pc = pd.read_csv(os.path.join(RAW,'paper_conf.txt'), sep='\t',
                 names=['paper_id','conf_id'], header=None, encoding='utf-8')
pt = pd.read_csv(os.path.join(RAW,'paper_term.txt'), sep='\t',
                 names=['paper_id','term_id'], header=None, encoding='utf-8')

# 1) Same base filtering
valid_authors = set(al['author_id'])
pa = pa[pa['author_id'].isin(valid_authors)].reset_index(drop=True)

valid_papers = set(pa['paper_id'])
pc = pc[pc['paper_id'].isin(valid_papers)].reset_index(drop=True)
pt = pt[pt['paper_id'].isin(valid_papers)].reset_index(drop=True)
pa = pa[pa['paper_id'].isin(valid_papers)].reset_index(drop=True)

big_confs = pc['conf_id'].value_counts().loc[lambda x: x >= MIN_CONF].index
pc = pc[pc['conf_id'].isin(big_confs)].reset_index(drop=True)

# 2) Universe after conf filter
valid_papers = set(pc['paper_id'])
pa = pa[pa['paper_id'].isin(valid_papers)].reset_index(drop=True)
pt = pt[pt['paper_id'].isin(valid_papers)].reset_index(drop=True)

# 3) Downsample papers to 10% (reproducible)
paper_list = np.array(sorted(valid_papers))
k = max(1, int(len(paper_list) * PAPER_KEEP_FRAC))
keep_papers = set(rng.choice(paper_list, size=k, replace=False))
before = len(valid_papers)

# Filter all tables to kept papers
pc = pc[pc['paper_id'].isin(keep_papers)].reset_index(drop=True)
pa = pa[pa['paper_id'].isin(keep_papers)].reset_index(drop=True)
pt = pt[pt['paper_id'].isin(keep_papers)].reset_index(drop=True)
valid_papers = set(keep_papers)

print(f"→ Kept papers: {len(valid_papers)} / {before} ({PAPER_KEEP_FRAC*100:.1f}%)")

# 4) PAPER-DISJOINT positives (only for kept papers)
pc_uni = pc[['paper_id','conf_id']].drop_duplicates().to_numpy(dtype=np.int64)

papers = np.array(sorted(valid_papers))
# split papers into train / (val+test)
p_train, p_tmp = train_test_split(
    papers, test_size=TEST_PCT + VAL_PCT, random_state=SEED, shuffle=True
)
# split (val+test) into val / test with the same ratio you used before
rel = TEST_PCT / (TEST_PCT + VAL_PCT) if (TEST_PCT + VAL_PCT) > 0 else 0.5
p_val, p_test = train_test_split(p_tmp, test_size=rel, random_state=SEED, shuffle=True)

P_train = set(p_train.tolist())
P_val   = set(p_val.tolist())
P_test  = set(p_test.tolist())

def keep_pos_for(P_set):
    if len(P_set) == 0:
        return np.empty((0, 2), dtype=np.int64)
    mask = np.isin(pc_uni[:, 0], list(P_set))
    return pc_uni[mask]

train_pos = keep_pos_for(P_train)
val_pos   = keep_pos_for(P_val)
test_pos  = keep_pos_for(P_test)

print(f"→ Paper-disjoint splits:")
print(f"   papers: train={len(P_train)}  val={len(P_val)}  test={len(P_test)}")
print(f"   pos:    train={len(train_pos)} val={len(val_pos)} test={len(test_pos)}")

# 5) Build per-split negative universes from split-specific papers only (ALL remaining venues)
C = sorted(pc['conf_id'].unique())

def make_all_negs(pos_arr, P_set, C_all):
    if len(P_set) == 0:
        return np.empty((0, 2), dtype=np.int64)
    true = set(map(tuple, pos_arr.tolist()))
    # all (paper, conf) pairs for this split, excluding true ones
    comp = np.array([(p, c) for p in P_set for c in C_all if (p, c) not in true], dtype=np.int64)
    return comp

rng = np.random.RandomState(SEED)  # re-seed for consistency (not used here, but keep)

train_neg = make_all_negs(train_pos, P_train, C)
val_neg   = make_all_negs(val_pos,   P_val,   C)
test_neg  = make_all_negs(test_pos,  P_test,  C)

# Informative print: expected ≈ (|C|-1) per positive if each paper has exactly one venue
neg_per_pos_train = (len(train_neg) / len(train_pos)) if len(train_pos) else 0.0
neg_per_pos_val   = (len(val_neg)   / len(val_pos))   if len(val_pos)   else 0.0
neg_per_pos_test  = (len(test_neg)  / len(test_pos))  if len(test_pos)  else 0.0
print(f"→ Negatives: train={len(train_neg)}  val={len(val_neg)}  test={len(test_neg)} (~{len(C)-1}× pos expected)")
print(f"   per-pos:  train={neg_per_pos_train:.1f}×  val={neg_per_pos_val:.1f}×  test={neg_per_pos_test:.1f}×")


# 6) Save splits + exact subset for preprocessing to reuse
np.savez(os.path.join(OUT, 'DBLP_pc_shared_splits.npz'),
         train_pos=train_pos, val_pos=val_pos, test_pos=test_pos,
         train_neg=train_neg, val_neg=val_neg, test_neg=test_neg,
         paper_subset=np.array(sorted(keep_papers), dtype=np.int64),
         papers_train=np.array(sorted(P_train), dtype=np.int64),
         papers_val=np.array(sorted(P_val), dtype=np.int64),
         papers_test=np.array(sorted(P_test), dtype=np.int64))

print("Saved:", os.path.join(OUT, 'DBLP_pc_shared_splits.npz'))

