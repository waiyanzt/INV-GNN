#!/usr/bin/env python3
import os
import time
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from utils.pytorchtools import EarlyStopping
from utils.data import load_DBLP_lp_pc_var2_data
from utils.dblp_lp_eval import subsample_test_neg_per_paper
from utils.tools import index_generator, parse_minibatch_LastFM
from model import MAGNN_lp

# Params
num_ntype = 5  # 0=author,1=paper,2=term,3=conf,4=area
dropout_rate = 0.5
lr = 0.005
weight_decay = 0.001

# Relation ids for VAR-2:
# 0:A→P, 1:P→A, 2:P→T, 3:T→P, 4:P→C, 5:C→P, 6:A→R, 7:R→A
NUM_ETYPES = 8

expected_metapaths = [
    [(1,0,1), (1,2,1), (1,3,1), (1,3,4,3,1)],
    [(3,1,0,1,3), (3,1,3,)]
]

# Edge-type sequences aligned with the above metapaths
etypes_lists = [
    [[1,0], [2,3], [4,5], [4,6,7,5]],              # paper-centric
    [[5,1,0,4], [5,4]]                              # venue-centric
]
use_masks = [[True]*4, [True]*2]
no_masks  = [[False]*4, [False]*2]

def set_seed(seed):
    import os, random, numpy as np, torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try: torch.use_deterministic_algorithms(True)
    except Exception: pass

def run_model_DBLP_pv_var2(feats_type, hidden_dim, num_heads, attn_vec_dim, rnn_type,
                           num_epochs, patience, batch_size, neighbor_samples, repeat, save_postfix,
                           threshold, K, neg_mult, base_seed=None, test_neg_per_paper=3):

    print("→ Loading preprocessed DBLP paper–venue data (var2: area↔venue) ...")
    t0 = time.time()
    (adjlists, edge_metapath_indices_list, _, type_mask,
     pos_splits, neg_splits, num_paper, num_conf) = load_DBLP_lp_pc_var2_data(expected_metapaths)
    print(f"   loaded in {time.time()-t0:.2f}s")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"→ Device: {device}")
    print(f"→ Negatives per positive (train/val): {neg_mult}")

    # per-run RNG for all NumPy sampling
    rng = np.random.default_rng(base_seed if base_seed is not None else 0)

    # Features
    print("→ Building features ...")
    features_list, in_dims = [], []
    if feats_type == 0:
        for i in range(num_ntype):
            dim = int((type_mask == i).sum())
            in_dims.append(dim)
            idx = np.vstack((np.arange(dim), np.arange(dim)))
            features_list.append(
                torch.sparse_coo_tensor(
                    torch.LongTensor(idx),
                    torch.FloatTensor(np.ones(dim)),
                    torch.Size([dim, dim]),
                    device=device
                )
            )
    else:
        for i in range(num_ntype):
            dim = 10
            in_dims.append(dim)
            features_list.append(torch.zeros(((type_mask == i).sum(), 10)).to(device))

    train_pos = pos_splits['train_pos_paper_conf']
    val_pos   = pos_splits['val_pos_paper_conf']
    test_pos  = pos_splits['test_pos_paper_conf']
    train_neg = neg_splits['train_neg_paper_conf']
    val_neg   = neg_splits['val_neg_paper_conf']
    test_neg  = neg_splits['test_neg_paper_conf']
    if test_neg_per_paper and test_neg_per_paper > 0:
        _nf = len(test_neg)
        test_neg = subsample_test_neg_per_paper(test_neg, test_neg_per_paper, base_seed if base_seed is not None else 0)
        print(f"→ Test negatives subsampled: {_nf} -> {len(test_neg)} (max {test_neg_per_paper} per paper)")

    print(f"→ Targets: papers={num_paper}, confs={num_conf}")
    print(f"→ Splits: train_pos={len(train_pos)}, val_pos={len(val_pos)}, test_pos={len(test_pos)}")
    print(f"           train_neg={len(train_neg)}, val_neg={len(val_neg)}, test_neg(eval)={len(test_neg)}")
    print(f"→ Metapaths per mode: {[len(m) for m in expected_metapaths]} (must match etypes_lists: {[len(e) for e in etypes_lists]})")
    print(f"→ GNN layers (K): {K}")

    # fix validation negatives once per run (deterministic order)
    val_neg_fixed = val_neg.copy()
    if len(val_neg_fixed) > 0:
        perm = rng.permutation(len(val_neg_fixed))
        val_neg_fixed = val_neg_fixed[perm]
    val_neg_cursor = 0

    os.makedirs('checkpoint', exist_ok=True)

    auc_list, ap_list = [], []
    prec_list, rec_list, f1_list, acc_list = [], [], [], []
    hits1_list, hits3_list, hits5_list, mrr_list = [], [], [], []

    for rep in range(repeat):
        print(f"\n===== Run {rep+1}/{repeat} =====")
        net = MAGNN_lp([4, 2], NUM_ETYPES, etypes_lists, in_dims,
                       hidden_dim, hidden_dim, num_heads, attn_vec_dim, rnn_type,
                       dropout_rate, num_layers=K).to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"→ Model params: {n_params:,}")

        # training loop
        net.train()
        ckpt_path = f'checkpoint/checkpoint_{save_postfix}_K{K}.pt'
        early_stopping = EarlyStopping(patience=patience, verbose=True, save_path=ckpt_path)
        train_pos_idx_generator = index_generator(batch_size=batch_size, num_data=len(train_pos))
        val_idx_generator = index_generator(batch_size=batch_size, num_data=len(val_pos), shuffle=False)

        for epoch in range(num_epochs):
            epoch_t0 = time.time()
            net.train()
            iters = train_pos_idx_generator.num_iterations()
            pbar = tqdm(range(iters), desc=f"Epoch {epoch:03d} [train]", leave=False)
            running_loss = 0.0

            for it in pbar:
                pos_idx = train_pos_idx_generator.next()
                pos_idx.sort()
                train_pos_batch = train_pos[pos_idx].tolist()

                neg_bs = neg_mult * len(pos_idx)
                replace_flag = neg_bs > len(train_neg)
                neg_idx = rng.choice(len(train_neg), neg_bs, replace=replace_flag)
                neg_idx.sort()
                train_neg_batch = train_neg[neg_idx].tolist()

                pos_g_lists, pos_indices_lists, pos_idx_batch_mapped_lists = parse_minibatch_LastFM(
                    adjlists, edge_metapath_indices_list, train_pos_batch, device,
                    neighbor_samples, use_masks, num_paper
                )
                neg_g_lists, neg_indices_lists, neg_idx_batch_mapped_lists = parse_minibatch_LastFM(
                    adjlists, edge_metapath_indices_list, train_neg_batch, device,
                    neighbor_samples, no_masks, num_paper
                )

                [pos_emb_left, pos_emb_right], _ = net(
                    (pos_g_lists, features_list, type_mask, pos_indices_lists, pos_idx_batch_mapped_lists)
                )
                [neg_emb_left, neg_emb_right], _ = net(
                    (neg_g_lists, features_list, type_mask, neg_indices_lists, neg_idx_batch_mapped_lists)
                )

                pos_out = torch.bmm(pos_emb_left.view(-1,1,pos_emb_left.shape[1]),
                                    pos_emb_right.view(-1,pos_emb_right.shape[1],1))
                neg_out = -torch.bmm(neg_emb_left.view(-1,1,neg_emb_left.shape[1]),
                                     neg_emb_right.view(-1,neg_emb_right.shape[1],1))
                train_loss = -(F.logsigmoid(pos_out).mean() + F.logsigmoid(neg_out).mean())

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                running_loss += train_loss.item()
                if (it + 1) % max(1, iters // 5) == 0:
                    pbar.set_postfix(loss=f"{running_loss / (it+1):.4f}")

            # validation
            net.eval()
            val_loss_vals = []
            v_iters = val_idx_generator.num_iterations()
            vbar = tqdm(range(v_iters), desc=f"Epoch {epoch:03d} [val]  ", leave=False)
            with torch.no_grad():
                for _ in vbar:
                    val_idx = val_idx_generator.next()
                    val_pos_batch = val_pos[val_idx].tolist()

                    neg_bs = neg_mult * len(val_idx)
                    if len(val_neg_fixed) == 0:
                        val_neg_batch = []
                    else:
                        end = val_neg_cursor + neg_bs
                        if end <= len(val_neg_fixed):
                            val_neg_batch = val_neg_fixed[val_neg_cursor:end]
                            val_neg_cursor = end % len(val_neg_fixed)
                        else:
                            take1 = len(val_neg_fixed) - val_neg_cursor
                            take2 = neg_bs - take1
                            part1 = val_neg_fixed[val_neg_cursor:]
                            part2 = val_neg_fixed[:take2]
                            val_neg_batch = np.concatenate([part1, part2], axis=0)
                            val_neg_cursor = take2 % len(val_neg_fixed)
                    val_neg_batch = val_neg_batch.tolist() if len(val_neg_fixed) > 0 else []

                    val_pos_g, val_pos_i, val_pos_m = parse_minibatch_LastFM(
                        adjlists, edge_metapath_indices_list, val_pos_batch, device,
                        neighbor_samples, no_masks, num_paper
                    )
                    val_neg_g, val_neg_i, val_neg_m = parse_minibatch_LastFM(
                        adjlists, edge_metapath_indices_list, val_neg_batch, device,
                        neighbor_samples, no_masks, num_paper
                    )
                    [pos_L, pos_R], _ = net((val_pos_g, features_list, type_mask, val_pos_i, val_pos_m))
                    [neg_L, neg_R], _ = net((val_neg_g, features_list, type_mask, val_neg_i, val_neg_m))

                    pos_out = torch.bmm(pos_L.view(-1,1,pos_L.shape[1]), pos_R.view(-1,pos_R.shape[1],1))
                    neg_out = -torch.bmm(neg_L.view(-1,1,neg_L.shape[1]), neg_R.view(-1,neg_R.shape[1],1))
                    vloss = -(F.logsigmoid(pos_out).mean() + F.logsigmoid(neg_out).mean())
                    val_loss_vals.append(vloss.item())
                    vbar.set_postfix(loss=f"{np.mean(val_loss_vals):.4f}")

            val_loss = float(np.mean(val_loss_vals)) if val_loss_vals else 0.0
            print(f"Epoch {epoch:03d} | Val_Loss {val_loss:.4f} | Time(s) {time.time()-epoch_t0:.2f}")
            early_stopping(torch.tensor(val_loss), net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        # Test on best checkpoint
        print("→ Testing best checkpoint ...")
        net.load_state_dict(torch.load(ckpt_path, map_location=device))
        net.eval()

        pos_prob, neg_prob = [], []

        # Positives
        pos_gen = index_generator(batch_size=batch_size, num_data=len(test_pos), shuffle=False)
        pbar = tqdm(range(pos_gen.num_iterations()), desc="[test:pos] ", leave=False)
        with torch.no_grad():
            for _ in pbar:
                idx = pos_gen.next()
                batch = test_pos[idx].tolist()
                g,i,m = parse_minibatch_LastFM(adjlists, edge_metapath_indices_list, batch, device,
                                               neighbor_samples, no_masks, num_paper)
                [L,R], _ = net((g, features_list, type_mask, i, m))
                out = torch.bmm(L.view(-1,1,L.shape[1]), R.view(-1,R.shape[1],1)).flatten()
                pos_prob.append(torch.sigmoid(out))

        # Negatives
        neg_gen = index_generator(batch_size=batch_size, num_data=len(test_neg), shuffle=False)
        pbar = tqdm(range(neg_gen.num_iterations()), desc="[test:neg] ", leave=True)
        with torch.no_grad():
            for _ in pbar:
                idx = neg_gen.next()
                batch = test_neg[idx].tolist()
                g,i,m = parse_minibatch_LastFM(adjlists, edge_metapath_indices_list, batch, device,
                                               neighbor_samples, no_masks, num_paper)
                [L,R], _ = net((g, features_list, type_mask, i, m))
                out = torch.bmm(L.view(-1,1,L.shape[1]), R.view(-1,R.shape[1],1)).flatten()
                neg_prob.append(torch.sigmoid(out))

        y_proba_test = torch.cat(pos_prob + neg_prob).cpu().numpy()
        y_true_test  = np.array([1]*len(test_pos) + [0]*len(test_neg))
        auc = roc_auc_score(y_true_test, y_proba_test)
        ap  = average_precision_score(y_true_test, y_proba_test)

        # Ranking metrics
        from collections import defaultdict
        y_pos = torch.cat(pos_prob).cpu().numpy()
        y_neg = torch.cat(neg_prob).cpu().numpy()
        pairs_pos = test_pos
        pairs_neg = test_neg

        # --- Save per-pair scores for Kendall τ ---
        import pandas as pd
        pairs_all = np.vstack((pairs_pos, pairs_neg))
        scores_all = np.concatenate((y_pos, y_neg))
        df_out = pd.DataFrame({'paper_id': pairs_all[:,0].astype(int),
                               'conf_id':  pairs_all[:,1].astype(int),
                               'score':    scores_all})
        df_out.to_csv(f"{save_postfix}_scores.csv", index=False)
        
        cand = defaultdict(list)
        for (p, c), s in zip(pairs_pos, y_pos):
            cand[int(p)].append((float(s), int(c), 1))
        for (p, c), s in zip(pairs_neg, y_neg):
            cand[int(p)].append((float(s), int(c), 0))

        hits1 = hits3 = hits5 = 0
        rr_sum = 0.0
        num_papers = 0
        for p, items in cand.items():
            if not items: 
                continue
            items.sort(key=lambda x: x[0], reverse=True)
            ranks = [i+1 for i, (_, _, t) in enumerate(items) if t == 1]
            if not ranks:
                continue
            r = min(ranks)
            num_papers += 1
            rr_sum += 1.0 / r
            if r <= 1: hits1 += 1
            if r <= 3: hits3 += 1
            if r <= 5: hits5 += 1

        print("Argmax (ranking) on test:")
        _h1 = hits1/num_papers
        _h3 = hits3/num_papers
        _h5 = hits5/num_papers
        _mr = rr_sum/num_papers
        if test_neg_per_paper and test_neg_per_paper <= 4:
            print(
                f"Hits@1 = {_h1:.6f}  Hits@3 = {_h3:.6f}  MRR = {_mr:.6f}  "
                f"(Hits@5 omitted: ≤{1+test_neg_per_paper} candidates/paper)"
            )
        else:
            print(f"Hits@1 = {_h1:.6f}  Hits@3 = {_h3:.6f}  Hits@5 = {_h5:.6f}  MRR = {_mr:.6f}")

        hits1 = _h1
        hits3 = _h3
        hits5 = _h5
        mrr   = _mr

        hits1_list.append(hits1)
        hits3_list.append(hits3)
        hits5_list.append(hits5)
        mrr_list.append(mrr)

        # Thresholded metrics
        th = threshold
        y_pred_test = (y_proba_test >= th).astype(int)
        TP = np.sum((y_pred_test == 1) & (y_true_test == 1))
        TN = np.sum((y_pred_test == 0) & (y_true_test == 0))
        FP = np.sum((y_pred_test == 1) & (y_true_test == 0))
        FN = np.sum((y_pred_test == 0) & (y_true_test == 1))
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        acc  = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
        print(f"Confusion matrix @ threshold {th:.2f}: TP={TP}  TN={TN}  FP={FP}  FN={FN} (P={TP+FN}, N={TN+FP})")

        print('Link Prediction Test (var2)')
        print(f'AUC = {auc:.6f}')
        print(f'AP  = {ap:.6f}')
        print(f'Precision = {prec:.6f}')
        print(f'Recall    = {rec:.6f}')
        print(f'F1        = {f1:.6f}')
        print(f'Accuracy  = {acc:.6f}')

        auc_list.append(auc); ap_list.append(ap)
        prec_list.append(prec); rec_list.append(rec); f1_list.append(f1); acc_list.append(acc)

    # Return stats for the caller (seed loop)
    return {
        "auc_mean": float(np.mean(auc_list)), "auc_std": float(np.std(auc_list)),
        "ap_mean": float(np.mean(ap_list)),   "ap_std": float(np.std(ap_list)),
        "precision_mean": float(np.mean(prec_list)), "precision_std": float(np.std(prec_list)),
        "recall_mean": float(np.mean(rec_list)),     "recall_std": float(np.std(rec_list)),
        "f1_mean": float(np.mean(f1_list)),          "f1_std": float(np.std(f1_list)),
        "accuracy_mean": float(np.mean(acc_list)),   "accuracy_std": float(np.std(acc_list)),
        "hits1_mean": float(np.mean(hits1_list)), "hits1_std": float(np.std(hits1_list)),
        "hits3_mean": float(np.mean(hits3_list)), "hits3_std": float(np.std(hits3_list)),
        "hits5_mean": float(np.mean(hits5_list)), "hits5_std": float(np.std(hits5_list)),
        "mrr_mean":  float(np.mean(mrr_list)),  "mrr_std":  float(np.std(mrr_list)),
    }

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MAGNN-LP on DBLP paper–venue (var2: area↔venue)')
    ap.add_argument('--feats-type', type=int, default=0)
    ap.add_argument('--hidden-dim', type=int, default=64)
    ap.add_argument('--num-heads', type=int, default=8)
    ap.add_argument('--attn-vec-dim', type=int, default=128)
    ap.add_argument('--rnn-type', default='RotatE0')
    ap.add_argument('--epoch', type=int, default=100)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--samples', type=int, default=100)
    ap.add_argument('--repeat', type=int, default=1)  # keep 1; we loop seeds ourselves
    ap.add_argument('--save-postfix', default='DBLP_pv_var2')
    ap.add_argument('--threshold', type=float, default=0.5,
                    help='Decision threshold for precision/recall/F1/accuracy (default=0.5).')
    ap.add_argument('--K', type=int, default=3, help='Number of GNN layers.')
    ap.add_argument('--seeds', default='1566911444,20241017,20251017',
                    help='Comma-separated list of seeds for N runs')
    ap.add_argument('--neg-mult', type=int, default=3,
                    help='Negatives per positive for train/val sampling')
    ap.add_argument(
        '--test-neg-per-paper',
        type=int,
        default=3,
        help='Subsample test negatives per paper for ranking/AUC (default=3). 0 = full split.',
    )
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]

    all_stats = []
    for seed in seeds:
        print(f"\n Seed {seed} \n")
        set_seed(seed)
        sp = f"{args.save_postfix}_seed{seed}"
        stats = run_model_DBLP_pv_var2(args.feats_type, args.hidden_dim, args.num_heads, args.attn_vec_dim,
                                       args.rnn_type, args.epoch, args.patience, args.batch_size,
                                       args.samples, 1, sp, args.threshold, args.K, args.neg_mult,
                                       base_seed=seed, test_neg_per_paper=args.test_neg_per_paper)
        all_stats.append((seed, stats))

    print("\n Summary over seeds")
    for seed, st in all_stats:
        print(f"Seed {seed}: "
            f"AUC={st['auc_mean']:.4f}±{st['auc_std']:.4f}  "
            f"AP={st['ap_mean']:.4f}±{st['ap_std']:.4f}  "
            f"Precision={st['precision_mean']:.4f}±{st['precision_std']:.4f}  "
            f"Recall={st['recall_mean']:.4f}±{st['recall_std']:.4f}  "
            f"F1={st['f1_mean']:.4f}±{st['f1_std']:.4f}  "
            f"Acc={st['accuracy_mean']:.4f}±{st['accuracy_std']:.4f}  "
            f"Hits@1={st['hits1_mean']:.4f}±{st['hits1_std']:.4f}  "
            f"Hits@3={st['hits3_mean']:.4f}±{st['hits3_std']:.4f}  "
            f"Hits@5={st['hits5_mean']:.4f}±{st['hits5_std']:.4f}  "
            f"MRR={st['mrr_mean']:.4f}±{st['mrr_std']:.4f}")

    print("\nFinal mean ± std across all seeds:")
    metric_keys = [
        "auc_mean", "ap_mean",
        "precision_mean", "recall_mean", "f1_mean", "accuracy_mean",
        "hits1_mean", "hits3_mean", "hits5_mean", "mrr_mean"
    ]
    metric_labels = [
        "AUC", "AP", "Precision", "Recall", "F1", "Accuracy",
        "Hits@1", "Hits@3", "Hits@5", "MRR"
    ]

    for k, label in zip(metric_keys, metric_labels):
        vals = np.array([st[k] for _, st in all_stats])
        print(f"{label:<10}: {vals.mean():.4f} ± {vals.std():.4f}")
