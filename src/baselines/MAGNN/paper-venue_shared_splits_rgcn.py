# import os, numpy as np, pandas as pd
# from sklearn.model_selection import train_test_split

# RAW = 'data/raw/DBLP/'
# OUT = 'data/preprocessed/DBLP_shared_splits/'
# SEED = 1566911444
# TEST_PCT = 0.20
# VAL_PCT  = 0.10
# MIN_CONF = 0
# PAPER_KEEP_FRAC = 1.0

# os.makedirs(OUT, exist_ok=True)
# rng = np.random.RandomState(SEED)

# print("→ Loading raw tables")
# al = pd.read_csv(os.path.join(RAW,'author_label.txt'), sep='\t',
#                  names=['author_id','label','author_name'], header=None, encoding='utf-8')
# pa = pd.read_csv(os.path.join(RAW,'paper_author.txt'), sep='\t',
#                  names=['paper_id','author_id'], header=None, encoding='utf-8')
# pc = pd.read_csv(os.path.join(RAW,'paper_conf.txt'), sep='\t',
#                  names=['paper_id','conf_id'], header=None, encoding='utf-8')
# pt = pd.read_csv(os.path.join(RAW,'paper_term.txt'), sep='\t',
#                  names=['paper_id','term_id'], header=None, encoding='utf-8')

# # 1) Base filtering (keep only papers that have valid authors)
# valid_authors = set(al['author_id'])
# pa = pa[pa['author_id'].isin(valid_authors)].reset_index(drop=True)

# valid_papers = set(pa['paper_id'])
# pc = pc[pc['paper_id'].isin(valid_papers)].reset_index(drop=True)
# pt = pt[pt['paper_id'].isin(valid_papers)].reset_index(drop=True)
# pa = pa[pa['paper_id'].isin(valid_papers)].reset_index(drop=True)

# # Optional conf filter
# big_confs = pc['conf_id'].value_counts().loc[lambda x: x >= MIN_CONF].index
# pc = pc[pc['conf_id'].isin(big_confs)].reset_index(drop=True)

# # Recompute valid papers after conf filter
# valid_papers = set(pc['paper_id'])
# pa = pa[pa['paper_id'].isin(valid_papers)].reset_index(drop=True)
# pt = pt[pt['paper_id'].isin(valid_papers)].reset_index(drop=True)

# # 2) Downsample papers if needed
# paper_list = np.array(sorted(valid_papers))
# k = max(1, int(len(paper_list) * PAPER_KEEP_FRAC))
# keep_papers = set(rng.choice(paper_list, size=k, replace=False))
# before = len(valid_papers)

# pc = pc[pc['paper_id'].isin(keep_papers)].reset_index(drop=True)
# pa = pa[pa['paper_id'].isin(keep_papers)].reset_index(drop=True)
# pt = pt[pt['paper_id'].isin(keep_papers)].reset_index(drop=True)
# valid_papers = set(keep_papers)

# print(f"→ Kept papers: {len(valid_papers)} / {before} ({PAPER_KEEP_FRAC*100:.1f}%)")

# # 3) Paper-disjoint positive pairs (PC tuples)
# pc_uni = pc[['paper_id','conf_id']].drop_duplicates().to_numpy(dtype=np.int64)

# papers = np.array(sorted(valid_papers))
# p_train, p_tmp = train_test_split(
#     papers, test_size=TEST_PCT + VAL_PCT, random_state=SEED, shuffle=True
# )
# rel = TEST_PCT / (TEST_PCT + VAL_PCT) if (TEST_PCT + VAL_PCT) > 0 else 0.5
# p_val, p_test = train_test_split(p_tmp, test_size=rel, random_state=SEED, shuffle=True)

# P_train = set(p_train.tolist())
# P_val   = set(p_val.tolist())
# P_test  = set(p_test.tolist())

# def keep_pos_for(P_set):
#     if len(P_set) == 0:
#         return np.empty((0, 2), dtype=np.int64)
#     mask = np.isin(pc_uni[:, 0], list(P_set))
#     return pc_uni[mask]

# train_pos_pc = keep_pos_for(P_train)  # shape: (p, c)
# val_pos_pc   = keep_pos_for(P_val)
# test_pos_pc  = keep_pos_for(P_test)

# print("→ Paper-disjoint splits:")
# print(f"   papers: train={len(P_train)}  val={len(P_val)}  test={len(P_test)}")
# print(f"   pos (PC): train={len(train_pos_pc)} val={len(val_pos_pc)} test={len(test_pos_pc)}")

# # 4) Negatives for PC: 1 random conf per positive paper
# C_all = sorted(pc['conf_id'].unique())

# def sample_one_neg_conf_per_pos_PC(pos_arr, C_all, rng):
#     if len(pos_arr) == 0:
#         return np.empty((0, 2), dtype=np.int64)
#     true_set = set(map(tuple, pos_arr.tolist()))
#     by_paper = {}
#     for p, c in pos_arr:
#         by_paper.setdefault(int(p), set()).add(int(c))
#     neg_rows = np.empty((len(pos_arr), 2), dtype=np.int64)
#     for i, (p, c_true) in enumerate(pos_arr):
#         true_confs = by_paper[int(p)]
#         candidates = [c for c in C_all if c not in true_confs]
#         if not candidates:
#             c_neg = int(c_true)
#         else:
#             c_neg = int(rng.choice(candidates))
#         if (int(p), c_neg) in true_set:
#             alt = [cc for cc in candidates if (int(p), int(cc)) not in true_set]
#             if alt:
#                 c_neg = int(rng.choice(alt))
#         neg_rows[i, 0] = int(p)
#         neg_rows[i, 1] = c_neg
#     return neg_rows

# rng = np.random.RandomState(SEED)
# train_neg_pc = sample_one_neg_conf_per_pos_PC(train_pos_pc, C_all, rng)
# val_neg_pc   = sample_one_neg_conf_per_pos_PC(val_pos_pc,   C_all, rng)
# test_neg_pc  = sample_one_neg_conf_per_pos_PC(test_pos_pc,  C_all, rng)

# print(f"→ PC negatives: train={len(train_neg_pc)}  val={len(val_neg_pc)}  test={len(test_neg_pc)} (1× pos)")

# # 5) Build CP positives by swapping columns (c, p)
# train_pos_cp = train_pos_pc[:, [1, 0]]
# val_pos_cp   = val_pos_pc[:, [1, 0]]
# test_pos_cp  = test_pos_pc[:, [1, 0]]

# # 6) Negatives for CP: 1 random paper per positive conf
# #    For each (c_true, p_true), sample p_neg not in true_papers(c_true)
# P_all = sorted(valid_papers)

# def sample_one_neg_paper_per_pos_CP(pos_cp, P_all, all_pos_cp, rng):
#     """
#     pos_cp: (c, p) positives for the split
#     P_all:  list of all candidate papers (after filtering/downsampling)
#     all_pos_cp: (c, p) positives across ALL splits to avoid accidental leakage
#     """
#     if len(pos_cp) == 0:
#         return np.empty((0, 2), dtype=np.int64)
#     true_set = set(map(tuple, all_pos_cp.tolist()))
#     by_conf = {}
#     for c, p in all_pos_cp:
#         by_conf.setdefault(int(c), set()).add(int(p))
#     neg_rows = np.empty((len(pos_cp), 2), dtype=np.int64)
#     for i, (c, p_true) in enumerate(pos_cp):
#         true_papers = by_conf.get(int(c), set())
#         candidates = [p for p in P_all if p not in true_papers]
#         if not candidates:
#             p_neg = int(p_true)
#         else:
#             p_neg = int(rng.choice(candidates))
#         if (int(c), p_neg) in true_set:
#             alt = [pp for pp in candidates if (int(c), int(pp)) not in true_set]
#             if alt:
#                 p_neg = int(rng.choice(alt))
#         neg_rows[i, 0] = int(c)
#         neg_rows[i, 1] = p_neg
#     return neg_rows

# # Build ALL positives in CP space to guard negatives for each split
# all_pos_cp = np.vstack([train_pos_cp, val_pos_cp, test_pos_cp])

# rng = np.random.RandomState(SEED)
# train_neg_cp = sample_one_neg_paper_per_pos_CP(train_pos_cp, P_all, all_pos_cp, rng)
# val_neg_cp   = sample_one_neg_paper_per_pos_CP(val_pos_cp,   P_all, all_pos_cp, rng)
# test_neg_cp  = sample_one_neg_paper_per_pos_CP(test_pos_cp,  P_all, all_pos_cp, rng)

# print(f"→ CP negatives: train={len(train_neg_cp)}  val={len(val_neg_cp)}  test={len(test_neg_cp)} (1× pos)")

# # 7) Save two files (PC and CP), each with its own tuple order and 1× negatives
# pc_path = os.path.join(OUT, 'DBLP_pc_shared_splits.npz')
# cp_path = os.path.join(OUT, 'DBLP_cp_shared_splits.npz')

# np.savez(pc_path,
#          train_pos=train_pos_pc, val_pos=val_pos_pc, test_pos=test_pos_pc,
#          train_neg=train_neg_pc, val_neg=val_neg_pc, test_neg=test_neg_pc,
#          paper_subset=np.array(sorted(keep_papers), dtype=np.int64),
#          papers_train=np.array(sorted(P_train), dtype=np.int64),
#          papers_val=np.array(sorted(P_val), dtype=np.int64),
#          papers_test=np.array(sorted(P_test), dtype=np.int64))
# print("Saved:", pc_path)

# np.savez(cp_path,
#          train_pos=train_pos_cp, val_pos=val_pos_cp, test_pos=test_pos_cp,
#          train_neg=train_neg_cp, val_neg=val_neg_cp, test_neg=test_neg_cp,
#          paper_subset=np.array(sorted(keep_papers), dtype=np.int64),
#          papers_train=np.array(sorted(P_train), dtype=np.int64),
#          papers_val=np.array(sorted(P_val), dtype=np.int64),
#          papers_test=np.array(sorted(P_test), dtype=np.int64))
# print("Saved:", cp_path)


# Build shared splits for DBLP (Paper–Conference only)
# For every positive (p, c_true), negatives are (p, c) for all c != c_true

import os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

RAW = 'data/raw/DBLP/'
OUT = 'data/preprocessed/DBLP_shared_splits/'
SEED = 1566911444
TEST_PCT = 0.20
VAL_PCT  = 0.10
MIN_CONF = 0
PAPER_KEEP_FRAC = 1.0

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

# 1) Base filtering (keep only papers that have valid authors)
valid_authors = set(al['author_id'])
pa = pa[pa['author_id'].isin(valid_authors)].reset_index(drop=True)

valid_papers = set(pa['paper_id'])
pc = pc[pc['paper_id'].isin(valid_papers)].reset_index(drop=True)
pt = pt[pt['paper_id'].isin(valid_papers)].reset_index(drop=True)
pa = pa[pa['paper_id'].isin(valid_papers)].reset_index(drop=True)

# Optional conf filter
big_confs = pc['conf_id'].value_counts().loc[lambda x: x >= MIN_CONF].index
pc = pc[pc['conf_id'].isin(big_confs)].reset_index(drop=True)

# Recompute valid papers after conf filter
valid_papers = set(pc['paper_id'])
pa = pa[pa['paper_id'].isin(valid_papers)].reset_index(drop=True)
pt = pt[pt['paper_id'].isin(valid_papers)].reset_index(drop=True)

# 2) Downsample papers if needed
paper_list = np.array(sorted(valid_papers))
k = max(1, int(len(paper_list) * PAPER_KEEP_FRAC))
keep_papers = set(rng.choice(paper_list, size=k, replace=False))
before = len(valid_papers)

pc = pc[pc['paper_id'].isin(keep_papers)].reset_index(drop=True)
pa = pa[pa['paper_id'].isin(keep_papers)].reset_index(drop=True)
pt = pt[pt['paper_id'].isin(keep_papers)].reset_index(drop=True)
valid_papers = set(keep_papers)

print(f"→ Kept papers: {len(valid_papers)} / {before} ({PAPER_KEEP_FRAC*100:.1f}%)")

# 3) Deduplicate PC (each paper should have exactly one conf)
pc_uni = pc[['paper_id','conf_id']].drop_duplicates().to_numpy(dtype=np.int64)

# Sanity: enforce 1-to-1 (paper -> conf) for this task
# If your source guarantees this, the assert will pass. If not, it will fail to warn you.
dup_counts = pd.DataFrame(pc_uni, columns=['paper_id','conf_id']).groupby('paper_id')['conf_id'].nunique()
assert (dup_counts <= 1).all(), "Found papers with multiple conferences — PC should be 1:1."

# 4) Paper-disjoint split of PAPERS
papers = np.array(sorted(valid_papers))
p_train, p_tmp = train_test_split(
    papers, test_size=TEST_PCT + VAL_PCT, random_state=SEED, shuffle=True
)
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

train_pos_pc = keep_pos_for(P_train)  # shape: (n_train, 2) as (paper, conf)
val_pos_pc   = keep_pos_for(P_val)
test_pos_pc  = keep_pos_for(P_test)

print("→ Paper-disjoint splits:")
print(f"   papers: train={len(P_train)}  val={len(P_val)}  test={len(P_test)}")
print(f"   pos (PC): train={len(train_pos_pc)} val={len(val_pos_pc)} test={len(test_pos_pc)}")

# 5) Build ALL-NEGATIVE sets: for each positive (p, c_true),
#    create (p, c) for every c != c_true, using ALL conferences observed.
C_all = np.array(sorted(pc['conf_id'].unique()), dtype=np.int64)
C = len(C_all)
print(f"→ Total conferences: {C}")

def all_negatives_for_pos(pos_arr, C_all):
    """
    pos_arr: (N,2) of (paper, conf_true)
    returns: (N*(|C|-1), 2) negatives covering all other conferences for each paper
    """
    if len(pos_arr) == 0:
        return np.empty((0, 2), dtype=np.int64)

    # list-comprehension version (clear & deterministic):
    rows = []
    for p, c_true in pos_arr:
        # all conferences except the true one
        confs = C_all[C_all != c_true]
        if len(confs) == 0:
            continue
        rows.append(np.column_stack([np.full_like(confs, p), confs]))
    return np.vstack(rows) if rows else np.empty((0, 2), dtype=np.int64)

train_neg_pc = all_negatives_for_pos(train_pos_pc, C_all)
val_neg_pc   = all_negatives_for_pos(val_pos_pc,   C_all)
test_neg_pc  = all_negatives_for_pos(test_pos_pc,  C_all)

def ratio_str(pos_arr, neg_arr):
    r = (len(neg_arr) / max(1, len(pos_arr))) if len(pos_arr) else 0.0
    return f"{len(neg_arr)} ({r:.1f}× pos)"

print("→ PC negatives (all other confs per positive):")
print(f"   train={ratio_str(train_pos_pc, train_neg_pc)}  "
      f"val={ratio_str(val_pos_pc, val_neg_pc)}  "
      f"test={ratio_str(test_pos_pc, test_neg_pc)}")

# 6) Save ONE npz (PC only), same filename as before so downstream stays unchanged
pc_path = os.path.join(OUT, 'DBLP_pc_shared_splits.npz')

np.savez(pc_path,
         train_pos=train_pos_pc, val_pos=val_pos_pc, test_pos=test_pos_pc,
         train_neg=train_neg_pc, val_neg=val_neg_pc, test_neg=test_neg_pc,
         paper_subset=np.array(sorted(valid_papers), dtype=np.int64),
         papers_train=np.array(sorted(P_train), dtype=np.int64),
         papers_val=np.array(sorted(P_val), dtype=np.int64),
         papers_test=np.array(sorted(P_test), dtype=np.int64),
         conf_list=C_all)

print("Saved:", pc_path)
print("Done.")
