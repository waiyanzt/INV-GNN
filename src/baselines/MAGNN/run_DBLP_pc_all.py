#!/usr/bin/env python3
import os
import sys
import types
import warnings
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange, tqdm
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from utils.pytorchtools import EarlyStopping
from model import MAGNN_lp
from utils.tools import index_generator

# loaders + parsers
from utils.data   import load_DBLP_lp_pc_data    as load_pc_v1_data
from utils.tools  import parse_lp_minibatch_pc   as parse_pc_v1

from utils.data   import load_DBLP_lp_pc_v2_data as load_pc_v2_data
from utils.tools  import parse_lp_minibatch_pc_v2 as parse_pc_v2

def run_pc(prefix, load_fn, parse_fn, etypes_lists, tag,
           hidden_dim, num_heads, attn_vec_dim, rnn_type,
           epochs, patience, batch_size, samp, repeat):
    # 1) load
    if 'prefix' in load_fn.__code__.co_varnames:
        adj, mp_idx, type_mask, \
        train_pos, val_pos, test_pos, \
        train_neg, val_neg, test_neg = load_fn(prefix=prefix)
    else:
        adj, mp_idx, type_mask, \
        train_pos, val_pos, test_pos, \
        train_neg, val_neg, test_neg = load_fn()

    device = torch.device('cpu')

    # 2) one‐hot features
    num_ntype = int(type_mask.max())+1
    features_list, in_dims = [], []
    for t in range(num_ntype):
        dim = int((type_mask==t).sum())
        in_dims.append(dim)
        idx = np.vstack((np.arange(dim), np.arange(dim)))
        features_list.append(
            torch.sparse.FloatTensor(
                torch.LongTensor(idx),
                torch.FloatTensor(np.ones(dim)),
                torch.Size([dim,dim])
            ).to(device)
        )

    # prepare ground truth for test
    y_test_true = np.hstack([np.ones(len(test_pos)), np.zeros(len(test_neg))]).astype(int)

    print(f"\n=== {tag} (epochs={epochs}, bs={batch_size}, samp={samp}) ===")

    # will store last-run split preds for robustness
    last_run_preds = {'train':None, 'val':None, 'test':None}

    for run in range(repeat):
        print(f"\n--- RUN {run+1}/{repeat} ---")
        net = MAGNN_lp(
            [len(adj[0]), len(adj[1])],
            10,
            etypes_lists,
            in_dims,
            hidden_dim, hidden_dim,
            num_heads, attn_vec_dim,
            rnn_type, dropout_rate=0.5
        ).to(device)

        opt = torch.optim.Adam(net.parameters(), lr=0.005, weight_decay=0.001)
        stopper = EarlyStopping(patience=patience, verbose=True,
                                save_path=f'checkpoint/pc_{tag}.pt')

        train_gen = index_generator(batch_size, len(train_pos))
        val_gen   = index_generator(batch_size, len(val_pos), shuffle=False)

        # training
        for epoch in trange(epochs, desc=" Epochs"):
            net.train(); train_gen.reset()
            for _ in tqdm(range(train_gen.num_iterations()),
                          desc="  Batches", leave=False):
                pidx = np.sort(train_gen.next())
                nidx = np.sort(np.random.choice(len(train_neg), len(pidx), replace=False))
                p_batch = [tuple(x) for x in train_pos[pidx]]
                n_batch = [tuple(x) for x in train_neg[nidx]]

                pg, pi, pm = parse_fn(adj, mp_idx, p_batch, device, samp)
                ng, ni, nm = parse_fn(adj, mp_idx, n_batch, device, samp)

                (pu, pv), _ = net((pg, features_list, type_mask, pi, pm))
                (nu, nv), _ = net((ng, features_list, type_mask, ni, nm))

                pos_score = torch.bmm(pu.unsqueeze(1), pv.unsqueeze(2))
                neg_score = -torch.bmm(nu.unsqueeze(1), nv.unsqueeze(2))
                loss = -torch.mean(F.logsigmoid(pos_score) + F.logsigmoid(neg_score))

                opt.zero_grad(); loss.backward(); opt.step()

            # validation
            net.eval()
            v_losses = []
            with torch.no_grad():
                for _ in range(val_gen.num_iterations()):
                    vidx = np.sort(val_gen.next())
                    pb = [tuple(x) for x in val_pos[vidx]]
                    nb = [tuple(x) for x in val_neg[vidx]]

                    pg, pi, pm = parse_fn(adj, mp_idx, pb, device, samp)
                    ng, ni, nm = parse_fn(adj, mp_idx, nb, device, samp)

                    (pu, pv), _ = net((pg, features_list, type_mask, pi, pm))
                    (nu, nv), _ = net((ng, features_list, type_mask, ni, nm))

                    ps = torch.bmm(pu.unsqueeze(1), pv.unsqueeze(2))
                    ns = -torch.bmm(nu.unsqueeze(1), nv.unsqueeze(2))
                    v_losses.append((-torch.mean(F.logsigmoid(ps) + F.logsigmoid(ns))).item())

            stopper(np.mean(v_losses), net)
            if stopper.early_stop:
                break

        # inference on train, val, test
        def infer(split_pos, split_neg):
            gen = index_generator(batch_size, len(split_pos), shuffle=False)
            scores, bins = [], []
            with torch.no_grad():
                while gen.num_iterations_left()>0:
                    idx = np.sort(gen.next())
                    pb = [tuple(x) for x in split_pos[idx]]
                    nb = [tuple(x) for x in split_neg[idx]]

                    pg, pi, pm = parse_fn(adj, mp_idx, pb, device, samp)
                    ng, ni, nm = parse_fn(adj, mp_idx, nb, device, samp)

                    (pu, pv), _ = net((pg, features_list, type_mask, pi, pm))
                    (nu, nv), _ = net((ng, features_list, type_mask, ni, nm))

                    p_scores = torch.sigmoid(torch.bmm(pu.unsqueeze(1), pv.unsqueeze(2)).flatten())
                    n_scores = torch.sigmoid(torch.bmm(nu.unsqueeze(1), nv.unsqueeze(2)).flatten())
                    batch_scores = torch.cat([p_scores, n_scores]).cpu().numpy()
                    scores.append(batch_scores)
            all_scores = np.hstack(scores)
            all_bins   = (all_scores >= 0.5).astype(int)
            return all_scores, all_bins

        net.load_state_dict(torch.load(f'checkpoint/pc_{tag}.pt'))

        # test metrics
        test_scores, test_bins = infer(test_pos, test_neg)
        auc = roc_auc_score(y_test_true, test_scores)
        ap  = average_precision_score(y_test_true, test_scores)
        prec = precision_score(y_test_true, test_bins)
        rec  = recall_score(y_test_true, test_bins)
        f1   = f1_score(y_test_true, test_bins)

        print(f"== RUN {run+1} TEST == AUC={auc:.4f} AP={ap:.4f} "
              f"P={prec:.4f} R={rec:.4f} F1={f1:.4f}")

        if run == repeat-1:
            _, train_bins = infer(train_pos, train_neg)
            _,  val_bins  = infer(val_pos,   val_neg)
            last_run_preds['train'] = train_bins
            last_run_preds['val']   = val_bins
            last_run_preds['test']  = test_bins

    return last_run_preds

def robustness(p1, p2):
    return [
        accuracy_score(p1['train'], p2['train']),
        accuracy_score(p1['val'],   p2['val']),
        accuracy_score(p1['test'],  p2['test']),
    ]

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--hidden-dim',   type=int,   default=64)
    p.add_argument('--num-heads',    type=int,   default=8)
    p.add_argument('--attn-vec-dim', type=int,   default=128)
    p.add_argument('--rnn-type',     type=str,   default='RotatE0')
    p.add_argument('--epoch',        type=int,   default=100)
    p.add_argument('--patience',     type=int,   default=10)
    p.add_argument('--batch-size',   type=int,   default=64)
    p.add_argument('--samples',      type=int,   default=20)
    p.add_argument('--repeat',       type=int,   default=1)
    args = p.parse_args()

    # Variant 1
    et1 = [
      [[1,0],[2,3],[4,5],[6,7]],        # paper‐centric
      [[5,4],[5,1,0,4],[5,2,3,4],[5,6,7,4]]  # venue‐centric
    ]
    preds1 = run_pc(
        prefix='data/preprocessed/DBLP_lp_pc_var1',
        load_fn=load_pc_v1_data,
        parse_fn=parse_pc_v1,
        etypes_lists=et1,
        tag='PC_V1',
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        attn_vec_dim=args.attn_vec_dim,
        rnn_type=args.rnn_type,
        epochs=args.epoch,
        patience=args.patience,
        batch_size=args.batch_size,
        samp=args.samples,
        repeat=args.repeat,
    )

    # Variant 2
    et2 = [
      [[1,0],      [2,3],      [4,5],      [4,8,9,5]],
      [[5,4],      [5,1,0,4],  [5,2,3,4],  [8,9]]
    ]

    preds2 = run_pc(
        prefix='data/preprocessed/DBLP_lp_pc_v2',
        load_fn=load_pc_v2_data,
        parse_fn=parse_pc_v2,
        etypes_lists=et2,
        tag='PC_V2',
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        attn_vec_dim=args.attn_vec_dim,
        rnn_type=args.rnn_type,
        epochs=args.epoch,
        patience=args.patience,
        batch_size=args.batch_size,
        samp=args.samples,
        repeat=args.repeat,
    )

    tr, va, te = robustness(preds1, preds2)
    print("\n=== Robustness ===")
    print(f" Train_Robust: {tr:.4f}% | Val_Robust: {va:.4f}% | Test_Robust: {te:.4f}%")
