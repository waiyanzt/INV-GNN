"""
SlotGAT node classification on 4 IMDB graph variants.
Target: movie -> genre (3 classes: Action / Comedy / Drama).

Variant connectivity (all share movie-link edges):
  1: actor-movie,  director-movie
  2: actor-link,   director-movie
  3: actor-link,   director-link
  4: actor-movie,  director-link

Usage:
    python run_IMDB_nc.py                              # all 4 variants
    python run_IMDB_nc.py --variant 1                  # variant 1 only
    python run_IMDB_nc.py --variant 1 2 --seeds 42,123
"""

import os
import sys
import gc
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score as sklearn_f1,
)
from tqdm import tqdm
import dgl
import scipy.sparse as sp

from model import slotGAT
from utils.pytorchtools import EarlyStopping
from utils.tools import set_seed, func_args_parse, single_feat_net, blank_profile, vis_data_collector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from utils.data import load_data

NUM_CLASSES = 3

VARIANT_NAMES = {
    1: 'IMDB_var1',
    2: 'IMDB_var2',
    3: 'IMDB_var3',
    4: 'IMDB_var4',
    5: 'IMDB_var5',
}


def ranking_metrics_from_logits(split_logits, true_labels):
    """Compute Hit@1, Hit@3, and MRR from class logits."""
    if split_logits.numel() == 0:
        return 0.0, 0.0, 0.0

    true_labels = true_labels.to(split_logits.device).long()
    top1 = split_logits.argmax(dim=1)
    hit1 = (top1 == true_labels).float().mean().item()

    k = min(3, split_logits.size(1))
    topk = torch.topk(split_logits, k=k, dim=1).indices
    hit3 = (topk == true_labels.unsqueeze(1)).any(dim=1).float().mean().item()

    sorted_idx = torch.argsort(split_logits, dim=1, descending=True)
    ranks = (sorted_idx == true_labels.unsqueeze(1)).nonzero(as_tuple=False)[:, 1].float() + 1.0
    mrr = (1.0 / ranks).mean().item()
    return hit1, hit3, mrr


def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)


def apply_feats_type(features_list, feats_type):
    """MAGNN-style feature modes for IMDB."""
    if feats_type == 0:
        in_dims = [feat.shape[1] for feat in features_list]
        return features_list, in_dims
    if feats_type == 1:
        in_dims = [features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(1, len(features_list)):
            features_list[i] = torch.zeros(
                (features_list[i].shape[0], 10),
                device=features_list[i].device,
            )
        return features_list, in_dims
    raise ValueError(f'Unsupported feats_type: {feats_type}')


def build_graph(features_list, adjM, dl, device):
    """Build DGL homogeneous graph with edge type features, following SlotGAT's approach."""
    edge2type = {}
    for k in dl.links['data']:
        for u, v in zip(*dl.links['data'][k].nonzero()):
            edge2type[(u, v)] = k

    for i in range(dl.nodes['total']):
        if (i, i) not in edge2type:
            edge2type[(i, i)] = len(dl.links['count'])

    count_reverse = 0
    for k in dl.links['data']:
        FLAG = 0
        for u, v in zip(*dl.links['data'][k].nonzero()):
            if (v, u) not in edge2type:
                edge2type[(v, u)] = count_reverse + 1 + len(dl.links['count'])
                FLAG = 1
        count_reverse += FLAG

    num_etype = len(dl.links['count']) + 1 + count_reverse

    g = dgl.DGLGraph(adjM + (adjM.T))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)

    e_feat = []
    count = 0
    count_mappings = {}
    counted_dict = {}
    g_ = g.cpu()
    for u, v in zip(*g_.edges()):
        u = u.item()
        v = v.item()
        if not counted_dict.setdefault(edge2type[(u, v)], False):
            count_mappings[edge2type[(u, v)]] = count
            counted_dict[edge2type[(u, v)]] = True
            count += 1
        e_feat.append(count_mappings[edge2type[(u, v)]])
    e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)

    num_ntypes = len(features_list)
    num_nodes = dl.nodes['total']
    g.num_ntypes = num_ntypes
    g.node_idx_by_ntype = []
    g.node_ntype_indexer = torch.zeros(num_nodes, num_ntypes).to(device)
    g.edge_type_indexer = F.one_hot(e_feat).to(device)

    idx_count = 0
    ntype_count = 0
    for feature in features_list:
        temp = []
        for _ in feature:
            temp.append(idx_count)
            g.node_ntype_indexer[idx_count][ntype_count] = 1
            idx_count += 1
        g.node_idx_by_ntype.append(temp)
        ntype_count += 1

    return g, e_feat, num_etype


def run_nc(args, variant, variant_name):
    print(f'\n{"="*60}')
    print(f'  IMDB Node Classification (SlotGAT) — Variant {variant}')
    print(f'  Dataset: {variant_name}')
    print(f'{"="*60}\n')

    device = torch.device(
        f'cuda:{args.gpu}' if not args.cpu and torch.cuda.is_available() else 'cpu'
    )
    print(f'Device: {device}')

    all_results = []

    for seed in args.seeds:
        print(f'\n--- seed={seed} ---')
        set_seed(seed)

        features_list, adjM, labels, train_val_test_idx, dl = load_data(
            prefix=variant_name, multi_labels=False
        )

        num_classes = dl.labels_train['num_classes']
        features_list = [mat2tensor(f).to(device) for f in features_list]
        features_list, in_dims = apply_feats_type(features_list, args.feats_type)
        labels = torch.LongTensor(labels).to(device)

        train_idx = np.sort(train_val_test_idx['train_idx'])
        val_idx = np.sort(train_val_test_idx['val_idx'])
        test_idx = np.sort(train_val_test_idx['test_idx'])

        n_movies = dl.nodes['count'][0]
        assert len(set(train_idx) & set(val_idx)) == 0, "train/val overlap"
        assert len(set(train_idx) & set(test_idx)) == 0, "train/test overlap"
        assert len(set(val_idx) & set(test_idx)) == 0, "val/test overlap"
        assert train_idx.max() < n_movies and val_idx.max() < n_movies and test_idx.max() < n_movies, \
            "split indices exceed movie node count"
        labels_np_check = labels.detach().cpu().numpy()
        assert len(np.unique(labels_np_check)) == num_classes, \
            f"Expected {num_classes} classes in labels, got {len(np.unique(labels_np_check))}"

        print(f'  Nodes: {dl.nodes["total"]}, Movie nodes: {n_movies}, Node types: {len(features_list)}')
        print(f'  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}, '
              f'Total labeled: {len(train_idx)+len(val_idx)+len(test_idx)}')
        print(f'  Classes: {num_classes}, Features dims: {in_dims}')
        print(f'  Test label distribution: {dict(zip(*np.unique(labels[test_idx].cpu().numpy(), return_counts=True)))}')

        g, e_feat, num_etype = build_graph(features_list, adjM, dl, device)
        print(f'  Graph edges: {g.num_edges()}, Edge types: {num_etype}')

        num_ntypes = len(features_list)
        heads = [args.num_heads] * args.num_layers + [1]

        getSAAttentionScore = "False"
        dataRecorder = {
            "meta": {"getSAAttentionScore": getSAAttentionScore},
            "data": {},
            "status": "None",
        }

        fargs, fkargs = func_args_parse(
            g, args.edge_feats, num_etype, in_dims,
            args.hidden_dim, num_classes, args.num_layers, heads,
            F.elu, args.dropout_feat, args.dropout_attn,
            args.slope, True, 0.05,
            num_ntype=num_ntypes, eindexer=None,
            aggregator=args.aggregator, SAattDim=args.SAattDim,
            dataRecorder=dataRecorder, vis_data_saver=None,
        )

        net = single_feat_net(slotGAT, *fargs, **fkargs)
        net.to(device)

        if seed == args.seeds[0]:
            n_params = sum(p.numel() for p in net.parameters())
            print(f'  Model params: {n_params}')

        optimizer = torch.optim.Adam(
            net.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        os.makedirs('checkpoint', exist_ok=True)
        ckp_fname = f'checkpoint/slotgat_imdb_nc_var{variant}_seed{seed}.pt'
        early_stopping = EarlyStopping(
            patience=args.patience, verbose=False, save_path=ckp_fname
        )

        loss_fn = F.nll_loss
        train_start_time = time.time()
        epochs_trained = 0

        for epoch in tqdm(range(args.epoch), desc=f'var{variant} seed={seed}'):
            epochs_trained = epoch + 1
            net.train()
            dataRecorder["status"] = "Training"
            logits, _ = net(features_list, e_feat)
            dataRecorder["status"] = "None"
            logp = F.log_softmax(logits, 1)
            train_loss = loss_fn(logp[train_idx], labels[train_idx])

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            net.eval()
            with torch.no_grad():
                dataRecorder["status"] = "Validation"
                logits, _ = net(features_list, e_feat)
                dataRecorder["status"] = "None"
                logp = F.log_softmax(logits, 1)
                val_loss = loss_fn(logp[val_idx], labels[val_idx])

            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

            if epoch > 0 and epoch % 20 == 0:
                preds_all = logits.argmax(dim=1)
                val_acc = accuracy_score(
                    labels[val_idx].cpu().numpy(),
                    preds_all[val_idx].cpu().numpy()
                )
                print(f'  Epoch {epoch}: train_loss={train_loss.item():.4f} '
                      f'val_loss={val_loss.item():.4f} val_acc={val_acc:.4f}')

        net.load_state_dict(torch.load(ckp_fname), strict=False)
        net.eval()
        with torch.no_grad():
            dataRecorder["status"] = "FinalTesting"
            logits, _ = net(features_list, e_feat)
            dataRecorder["status"] = "None"
        train_time_sec = time.time() - train_start_time

        final_preds = logits.argmax(dim=1).cpu().numpy()
        labels_np = labels.cpu().numpy()

        splits = [
            ('Train', train_idx),
            ('Val', val_idx),
            ('Test', test_idx),
        ]
        train_acc = val_acc = test_acc = 0.0
        train_prec_macro = val_prec_macro = test_prec_macro = 0.0
        train_rec_macro = val_rec_macro = test_rec_macro = 0.0
        train_micro_f1 = val_micro_f1 = test_micro_f1 = 0.0
        train_macro_f1 = val_macro_f1 = test_macro_f1 = 0.0
        test_hit1 = 0.0
        test_hit3 = 0.0
        test_mrr = 0.0

        for split_name, nids in splits:
            true = labels_np[nids]
            pred = final_preds[nids]
            split_logits = logits[nids]
            split_true_t = labels[nids]
            hit1, hit3, mrr = ranking_metrics_from_logits(split_logits, split_true_t)
            acc = accuracy_score(true, pred)
            prec_macro = precision_score(true, pred, average='macro', zero_division=0)
            rec_macro = recall_score(true, pred, average='macro', zero_division=0)
            f1_macro = sklearn_f1(true, pred, average='macro', zero_division=0)
            prec_micro = precision_score(true, pred, average='micro', zero_division=0)
            rec_micro = recall_score(true, pred, average='micro', zero_division=0)
            f1_micro = sklearn_f1(true, pred, average='micro', zero_division=0)
            print(f'  {split_name:5s}: Acc={acc:.4f}  '
                  f'Prec_macro={prec_macro:.4f}  Rec_macro={rec_macro:.4f}  F1_macro={f1_macro:.4f}  '
                  f'Prec_micro={prec_micro:.4f}  Rec_micro={rec_micro:.4f}  F1_micro={f1_micro:.4f}  '
                  f'Hit@1={hit1:.4f}  Hit@3={hit3:.4f}  MRR={mrr:.4f}')
            if split_name == 'Train':
                train_acc, train_prec_macro, train_rec_macro = acc, prec_macro, rec_macro
                train_micro_f1, train_macro_f1 = f1_micro, f1_macro
            elif split_name == 'Val':
                val_acc, val_prec_macro, val_rec_macro = acc, prec_macro, rec_macro
                val_micro_f1, val_macro_f1 = f1_micro, f1_macro
            elif split_name == 'Test':
                test_acc, test_prec_macro, test_rec_macro = acc, prec_macro, rec_macro
                test_micro_f1, test_macro_f1 = f1_micro, f1_macro
                test_hit1 = hit1
                test_hit3 = hit3
                test_mrr = mrr

        all_results.append({
            'seed': seed,
            'train_acc': train_acc,
            'train_prec_macro': train_prec_macro,
            'train_rec_macro': train_rec_macro,
            'train_micro_f1': train_micro_f1,
            'train_macro_f1': train_macro_f1,
            'val_acc': val_acc,
            'val_prec_macro': val_prec_macro,
            'val_rec_macro': val_rec_macro,
            'val_micro_f1': val_micro_f1,
            'val_macro_f1': val_macro_f1,
            'test_acc': test_acc,
            'test_prec_macro': test_prec_macro,
            'test_rec_macro': test_rec_macro,
            'test_micro_f1': test_micro_f1,
            'test_macro_f1': test_macro_f1,
            'test_hit1': test_hit1,
            'test_hit3': test_hit3,
            'test_mrr': test_mrr,
            'train_time_sec': train_time_sec,
            'epochs_trained': epochs_trained,
        })
        print(f'  Epochs trained: {epochs_trained} (cap={args.epoch}, patience={args.patience})')

        del net, g
        if not args.cpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print(f'\n--- Summary for Variant {variant} ---')
    for split_label, ak, pk, rk, mk, mak in (
        ('Train', 'train_acc', 'train_prec_macro', 'train_rec_macro', 'train_micro_f1', 'train_macro_f1'),
        ('Val', 'val_acc', 'val_prec_macro', 'val_rec_macro', 'val_micro_f1', 'val_macro_f1'),
        ('Test', 'test_acc', 'test_prec_macro', 'test_rec_macro', 'test_micro_f1', 'test_macro_f1'),
    ):
        av = [r[ak] for r in all_results]
        pv = [r[pk] for r in all_results]
        rv = [r[rk] for r in all_results]
        mv = [r[mk] for r in all_results]
        mav = [r[mak] for r in all_results]
        print(
            f'  {split_label}: Acc={np.mean(av)*100:.2f} +/- {np.std(av)*100:.2f}  '
            f'Precision={np.mean(pv)*100:.2f} +/- {np.std(pv)*100:.2f}  '
            f'Recall={np.mean(rv)*100:.2f} +/- {np.std(rv)*100:.2f}  '
            f'Micro-F1={np.mean(mv)*100:.2f} +/- {np.std(mv)*100:.2f}  '
            f'Macro-F1={np.mean(mav)*100:.2f} +/- {np.std(mav)*100:.2f}'
        )
    train_time_sec = [r['train_time_sec'] for r in all_results]
    epochs_trained_list = [r['epochs_trained'] for r in all_results]
    test_hit1 = [r['test_hit1'] for r in all_results]
    test_hit3 = [r['test_hit3'] for r in all_results]
    test_mrr = [r['test_mrr'] for r in all_results]
    per_seed_epochs = ', '.join(
        f'seed={r["seed"]}: {r["epochs_trained"]}' for r in all_results
    )
    print(f'  TrainTime(s): {np.mean(train_time_sec):.2f} +/- {np.std(train_time_sec):.2f}')
    print(f'  Epochs:        {np.mean(epochs_trained_list):.1f} +/- {np.std(epochs_trained_list):.1f}  ({per_seed_epochs})')
    print(f'  Test Hit@1:    {np.mean(test_hit1)*100:.2f} +/- {np.std(test_hit1)*100:.2f}')
    print(f'  Test Hit@3:    {np.mean(test_hit3)*100:.2f} +/- {np.std(test_hit3)*100:.2f}')
    print(f'  Test MRR:      {np.mean(test_mrr)*100:.2f} +/- {np.std(test_mrr)*100:.2f}')

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(
        description='SlotGAT NC on IMDB variants (movie genre, 3 classes)'
    )
    parser.add_argument('--variant', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                        choices=[1, 2, 3, 4, 5])
    parser.add_argument('--feats-type', type=int, default=0, choices=[0, 1],
                        help='0: use original features, 1: only target node type keeps features; others use zero vectors (dim=10)')
    parser.add_argument('--seeds', default='1566911444,20241017,20251017')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--patience', type=int, default=40)

    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--edge-feats', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('--dropout-feat', type=float, default=0.5)
    parser.add_argument('--dropout-attn', type=float, default=0.2)
    parser.add_argument('--slope', type=float, default=0.05)
    parser.add_argument('--aggregator', type=str, default='SA',
                        choices=['SA', 'average', 'last_fc', 'max', 'onedimconv'])
    parser.add_argument('--SAattDim', type=int, default=3)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    print(args)

    all_variant_results = {}
    for v in args.variant:
        variant_name = VARIANT_NAMES[v]
        results = run_nc(args, v, variant_name)
        all_variant_results[v] = results

    if len(args.variant) > 1:
        print(f'\n{"="*60}')
        print('  Cross-Variant IMDB NC Summary (SlotGAT)')
        print(f'{"="*60}')
        for v, results in all_variant_results.items():
            print(f'  Variant {v}:')
            for split_label, ak, pk, rk, mk, mak in (
                ('Train', 'train_acc', 'train_prec_macro', 'train_rec_macro', 'train_micro_f1', 'train_macro_f1'),
                ('Val', 'val_acc', 'val_prec_macro', 'val_rec_macro', 'val_micro_f1', 'val_macro_f1'),
                ('Test', 'test_acc', 'test_prec_macro', 'test_rec_macro', 'test_micro_f1', 'test_macro_f1'),
            ):
                av = [r[ak] for r in results]
                pv = [r[pk] for r in results]
                rv = [r[rk] for r in results]
                mv = [r[mk] for r in results]
                mav = [r[mak] for r in results]
                print(
                    f'    {split_label}: Acc={np.mean(av)*100:.2f} +/- {np.std(av)*100:.2f}  '
                    f'Precision={np.mean(pv)*100:.2f} +/- {np.std(pv)*100:.2f}  '
                    f'Recall={np.mean(rv)*100:.2f} +/- {np.std(rv)*100:.2f}  '
                    f'Micro-F1={np.mean(mv)*100:.2f} +/- {np.std(mv)*100:.2f}  '
                    f'Macro-F1={np.mean(mav)*100:.2f} +/- {np.std(mav)*100:.2f}'
                )
            train_time_sec = [r['train_time_sec'] for r in results]
            epochs_trained_list = [r['epochs_trained'] for r in results]
            test_hit1 = [r['test_hit1'] for r in results]
            test_hit3 = [r['test_hit3'] for r in results]
            test_mrr = [r['test_mrr'] for r in results]
            per_seed_epochs = ', '.join(
                f'seed={r["seed"]}: {r["epochs_trained"]}' for r in results
            )
            print(
                f'    TrainTime(s): {np.mean(train_time_sec):.2f} +/- {np.std(train_time_sec):.2f}  '
                f'Epochs={np.mean(epochs_trained_list):.1f} +/- {np.std(epochs_trained_list):.1f}  '
                f'Hit@1={np.mean(test_hit1)*100:.2f} +/- {np.std(test_hit1)*100:.2f}  '
                f'Hit@3={np.mean(test_hit3)*100:.2f} +/- {np.std(test_hit3)*100:.2f}  '
                f'MRR={np.mean(test_mrr)*100:.2f} +/- {np.std(test_mrr)*100:.2f}'
            )
            print(f'    Per-seed epochs: {per_seed_epochs}')
