import os, pickle, argparse
import numpy as np
import torch
from tqdm import tqdm

from utils.data import load_DBLP_lp_pc_var1_data, load_DBLP_lp_pc_var2_data
from utils.tools import index_generator, parse_minibatch_LastFM
from model import MAGNN_lp


# deterministic seeding
SEED = 1566911444
def set_determinism(seed=SEED):
    import os, random, numpy as np, torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    # must be set before CUDA context is created
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


# Variant-specific settings
expected_metapaths_v1 = [
    [(1,0,1), (1,2,1), (1,3,1), (1,4,1)],
    [(3,1,0,1,3), (3,1,2,1,3), (3,1,4,1,3)]
]
etypes_lists_v1 = [
    [[1,0], [2,3], [4,5], [6,7]],
    [[5,1,0,4], [5,2,3,4], [5,6,7,4]],
]
expected_metapaths_v2 = [
    [(1,0,1), (1,2,1), (1,3,1), (1,3,4,3,1)],
    [(3,1,0,1,3), (3,1,2,1,3), (3,4,3)]
]
etypes_lists_v2 = [
    [[1,0], [2,3], [4,5], [4,6,7,5]],
    [[5,1,0,4], [5,2,3,4], [6,7]],
]
NUM_ETYPES = 8
NUM_NTYPE  = 5  # 0=author,1=paper,2=term,3=conf,4=area

def build_features(type_mask, feats_type, device):
    feats, in_dims = [], []
    if feats_type == 0:
        for i in range(NUM_NTYPE):
            dim = int((type_mask == i).sum())
            in_dims.append(dim)
            idx = np.vstack((np.arange(dim), np.arange(dim)))
            feats.append(
                torch.sparse_coo_tensor(
                    torch.LongTensor(idx),
                    torch.FloatTensor(np.ones(dim)),
                    torch.Size([dim, dim]),
                    device=device
                )
            )
    else:
        for i in range(NUM_NTYPE):
            dim = 10
            in_dims.append(dim)
            feats.append(torch.zeros(((type_mask == i).sum(), 10), device=device))
    return feats, in_dims

def invert_map(d):
    return {v:k for k,v in d.items()}

def map_pairs_local_to_orig(pairs_local, paper_inv, conf_inv):
    po = [paper_inv[int(p)] for p in pairs_local[:,0]]
    co = [conf_inv[int(c)]  for c in pairs_local[:,1]]
    return list(zip(po, co))

def map_pairs_orig_to_local(pairs_orig, paper_map, conf_map):
    out = []
    for (po,co) in pairs_orig:
        if (po in paper_map) and (co in conf_map):
            out.append([paper_map[po], conf_map[co]])
    return np.array(out, dtype=np.int32)

def get_common_test_pairs(base_v1, base_v2):
    pos1 = np.load(os.path.join(base_v1, 'train_val_test_pos_paper_conf.npz'))
    neg1 = np.load(os.path.join(base_v1, 'train_val_test_neg_paper_conf.npz'))
    pos2 = np.load(os.path.join(base_v2, 'train_val_test_pos_paper_conf.npz'))
    neg2 = np.load(os.path.join(base_v2, 'train_val_test_neg_paper_conf.npz'))

    test1 = np.vstack([pos1['test_pos'], neg1['test_neg']]).astype(np.int32)
    test2 = np.vstack([pos2['test_pos'], neg2['test_neg']]).astype(np.int32)

    with open(os.path.join(base_v1, 'node_maps.pkl'), 'rb') as f:
        maps1 = pickle.load(f)
    with open(os.path.join(base_v2, 'node_maps.pkl'), 'rb') as f:
        maps2 = pickle.load(f)

    p_inv1 = invert_map(maps1['paper_idx']); c_inv1 = invert_map(maps1['conf_idx'])
    p_inv2 = invert_map(maps2['paper_idx']); c_inv2 = invert_map(maps2['conf_idx'])

    orig1 = set(map_pairs_local_to_orig(test1, p_inv1, c_inv1))
    orig2 = set(map_pairs_local_to_orig(test2, p_inv2, c_inv2))

    common_orig = sorted(list(orig1 & orig2))
    if len(common_orig) == 0:
        raise RuntimeError("No common test pairs found between var1 and var2!")

    test1_local = map_pairs_orig_to_local(common_orig, maps1['paper_idx'], maps1['conf_idx'])
    test2_local = map_pairs_orig_to_local(common_orig, maps2['paper_idx'], maps2['conf_idx'])
    return common_orig, test1_local, test2_local

@torch.no_grad()
def predict_probs(adjlists, edge_metapath_indices_list, type_mask_np, pairs_local, num_paper,
                  feats_type, hidden_dim, num_heads, attn_vec_dim, rnn_type, dropout_rate,
                  etypes_lists, checkpoint, device, neighbor_samples):

    features_list, in_dims = build_features(type_mask_np, feats_type, device)

    net = MAGNN_lp([len(adjlists[0]), len(adjlists[1])], NUM_ETYPES,
                   etypes_lists, in_dims,
                   hidden_dim, hidden_dim, num_heads, attn_vec_dim, rnn_type, dropout_rate).to(device)
    net.load_state_dict(torch.load(checkpoint, weights_only=True))
    net.eval()

    probs = []
    bs = 1024
    it = index_generator(batch_size=bs, num_data=len(pairs_local), shuffle=False)
    # NEW: base seed for per-batch reseed (keeps neighbor sampling identical)
    base = SEED
    for b in tqdm(range(it.num_iterations()), desc="predict", leave=False):
        # NEW: reseed each batch to remove sampler noise
        np.random.seed(base + b)
        import random; random.seed(base + b)
        torch.manual_seed(base + b); torch.cuda.manual_seed_all(base + b)

        idx = it.next()
        batch_pairs = pairs_local[idx].tolist()

        g_lists, indices_lists, idx_batch_mapped_lists = parse_minibatch_LastFM(
            adjlists, edge_metapath_indices_list, batch_pairs, device,
            neighbor_samples,
            [[False]*len(adjlists[0]), [False]*len(adjlists[1])],
            num_paper
        )
        [L, R], _ = net((g_lists, features_list, type_mask_np, indices_lists, idx_batch_mapped_lists))
        L = L.view(-1, 1, L.shape[1]); R = R.view(-1, R.shape[1], 1)
        out = torch.bmm(L, R).flatten()
        probs.append(torch.sigmoid(out).cpu().numpy())
    return np.concatenate(probs, axis=0)

def main():
    set_determinism(SEED)

    ap = argparse.ArgumentParser(description="Robustness: agreement between Var1 and Var2 predictions")
    ap.add_argument('--var1-dir', default='data/preprocessed/DBLP_lp_pc_var1/')
    ap.add_argument('--var2-dir', default='data/preprocessed/DBLP_lp_pc_var2/')
    ap.add_argument('--ckpt-v1', default='checkpoint/checkpoint_DBLP_pv.pt')
    ap.add_argument('--ckpt-v2', default='checkpoint/checkpoint_DBLP_pv_var2.pt')
    ap.add_argument('--feats-type', type=int, default=0)
    ap.add_argument('--hidden-dim', type=int, default=64)
    ap.add_argument('--num-heads', type=int, default=8)
    ap.add_argument('--attn-vec-dim', type=int, default=128)
    ap.add_argument('--rnn-type', default='RotatE0')
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--samples', type=int, default=100)
    ap.add_argument('--threshold', type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("→ Loading var1 ...")
    (adj_v1, epi_v1, _, type_v1, pos_v1, neg_v1, numP1, numC1) = load_DBLP_lp_pc_var1_data(expected_metapaths_v1, base=args.var1_dir)
    print("→ Loading var2 ...")
    (adj_v2, epi_v2, _, type_v2, pos_v2, neg_v2, numP2, numC2) = load_DBLP_lp_pc_var2_data(expected_metapaths_v2, base=args.var2_dir)

    print("→ Aligning common test pairs ...")
    common_orig, test1_local, test2_local = get_common_test_pairs(args.var1_dir, args.var2_dir)
    print(f"   common pairs: {len(common_orig)}")

    print("→ Predicting with Var1 ...")
    y1 = predict_probs(adj_v1, epi_v1, type_v1, test1_local, numP1,
                       args.feats_type, args.hidden_dim, args.num_heads, args.attn_vec_dim,
                       args.rnn_type, args.dropout, etypes_lists_v1, args.ckpt_v1, device, args.samples)
    print("→ Predicting with Var2 ...")
    y2 = predict_probs(adj_v2, epi_v2, type_v2, test2_local, numP2,
                       args.feats_type, args.hidden_dim, args.num_heads, args.attn_vec_dim,
                       args.rnn_type, args.dropout, etypes_lists_v2, args.ckpt_v2, device, args.samples)

    th = args.threshold
    p1 = (y1 >= th).astype(int)
    p2 = (y2 >= th).astype(int)
    agree = (p1 == p2).astype(int)
    robustness = agree.mean()

    both_pos = np.mean((p1 == 1) & (p2 == 1))
    both_neg = np.mean((p1 == 0) & (p2 == 0))
    v1_pos_v2_neg = np.mean((p1 == 1) & (p2 == 0))
    v1_neg_v2_pos = np.mean((p1 == 0) & (p2 == 1))

    print("\n===== Robustness (V1 vs V2) =====")
    print(f"Threshold        : {th:.2f}")
    print(f"Pairs compared   : {len(common_orig)}")
    print(f"Agreement        : {robustness*100:.2f}%")
    print(f"  both 1         : {both_pos*100:.2f}%")
    print(f"  both 0         : {both_neg*100:.2f}%")
    print(f"  V1=1, V2=0     : {v1_pos_v2_neg*100:.2f}%")
    print(f"  V1=0, V2=1     : {v1_neg_v2_pos*100:.2f}%")

if __name__ == "__main__":
    main()
