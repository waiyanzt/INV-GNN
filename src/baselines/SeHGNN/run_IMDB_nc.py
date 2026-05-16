"""
SeHGNN node classification on IMDB graph variants.
Target: movie -> genre (3 classes: Action / Comedy / Drama).

Variant connectivity (all share movie-link edges):
  1: actor-movie,  director-movie
  2: actor-link,   director-link
  3: actor-link,   director-movie
  4: actor-movie,  director-link
  5: actor-movie,  actor-link,  director-movie,  director-link  (pruned to union of v1..v4)
  6: actor-movie,  actor-link,  director-movie,  director-link  (all 77 metapaths, no filtering)

Usage:
    python run_IMDB_nc.py                              # all variants
    python run_IMDB_nc.py --variant 1                  # variant 1 only
    python run_IMDB_nc.py --variant 1 2 --seeds 42,123

MAGNN-aligned skip bundles (same adjacency / labels / splits as MAGNN skip; semantic movie channels):
    From baselines/SeHGNN:  python preprocess_IMDB_skip.py --variant v1   # or --all
    From baselines/SeHGNN: python run_IMDB_nc.py --imdb-skip-data --variant 1

Optional: ``--imdb-data-root DIR`` for legacy MAGNN folders ``DIR/IMDB_preprocessed_skip_v*``.
Default skip data: ``data/preprocessed_IMDB_skip_vN`` under SeHGNN.
"""

import os
import gc
import argparse
import datetime
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score as sklearn_f1,
)
from tqdm import tqdm

from model import SeHGNN
from utils.pytorchtools import EarlyStopping
from utils.tools import (
    set_random_seed, evaluator, get_n_params, hg_propagate_feat_dgl, train,
    universal_allowed_metapath_keys,
)
from utils.data import load_imdb_graph, load_imdb_nc_labels
from utils.imdb_skip_semantic_feats import build_imdb_skip_semantic_movie_feats
from utils.imdb_variant_dir import add_imdb_skip_args, resolve_imdb_nc_dir

NUM_CLASSES = 3
TGT_TYPE = 'M'

# Default relative dirs (use resolve_imdb_nc_dir for cwd-independent paths).
VARIANT_DIRS = {
    1: 'data/IMDB_var1/',
    2: 'data/IMDB_var2/',
    3: 'data/IMDB_var3/',
    4: 'data/IMDB_var4/',
    5: 'data/IMDB_var5',
    6: 'data/IMDB_var6',  # universal graph, all 77 metapaths (no filtering)
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


def run_nc(args, variant, base_dir):
    print(f'\n{"="*60}')
    print(f'  IMDB Node Classification — Variant {variant}')
    print(f'  Data dir: {base_dir}')
    print(f'{"="*60}\n')

    device = torch.device(
        f'cuda:{args.gpu}' if not args.cpu and torch.cuda.is_available() else 'cpu'
    )
    print(f'Device: {device}')

    g, node_counts, type_mask = load_imdb_graph(base_dir)
    print(f'Graph: {g}')
    print(f'Node counts: {node_counts}')

    labels, train_nid, val_nid, test_nid, num_classes = \
        load_imdb_nc_labels(base_dir)
    num_nodes = node_counts[TGT_TYPE]
    print(f'NC labels: {num_classes} classes, '
          f'train={len(train_nid)}, val={len(val_nid)}, test={len(test_nid)}')

    all_results = []

    feats_shared = None
    if args.imdb_skip_data:
        if args.num_hops != 4:
            print(
                'Note: --imdb-skip-data uses fixed MAGNN skip semantic movie channels; '
                '--num-hops is ignored.',
            )
        movie_feat = g.nodes['M'].data['M']
        prop_tic = datetime.datetime.now()
        feats_shared = build_imdb_skip_semantic_movie_feats(
            base_dir, variant, movie_feat, device=device,
        )
        prop_toc = datetime.datetime.now()
        print(f'Semantic skip feature build time: {prop_toc - prop_tic}')
        print(f'Feature keys: {sorted(feats_shared.keys())}')

    for seed in args.seeds:
        print(f'\n--- seed={seed} ---')
        set_random_seed(seed)

        if args.imdb_skip_data:
            feats = feats_shared
        else:
            g_copy, _, _ = load_imdb_graph(base_dir)

            extra_metapath = []
            max_length = args.num_hops + 1

            prop_tic = datetime.datetime.now()
            g_copy = hg_propagate_feat_dgl(
                g_copy, TGT_TYPE, args.num_hops, max_length, extra_metapath, echo=False,
            )
            prop_toc = datetime.datetime.now()
            print(f'Feature propagation time: {prop_toc - prop_tic}')

            feats = {}
            keys = list(g_copy.nodes[TGT_TYPE].data.keys())
            for k in keys:
                feats[k] = g_copy.nodes[TGT_TYPE].data.pop(k)

            if variant == 5:
                allowed = universal_allowed_metapath_keys(args.num_hops, TGT_TYPE)
                pruned = {k: v for k, v in feats.items() if k in allowed}
                print(
                    f'Variant 5: pruning to union of v1..v4 metapaths '
                    f'({len(feats)} -> {len(pruned)} keys)'
                )
                feats = pruned
            elif variant == 6:
                print(f'Variant 6: retaining all {len(feats)} metapaths (no filtering)')

            print(f'Feature keys: {sorted(feats.keys())}')

        data_size = {k: v.shape[-1] for k, v in feats.items()}
        label_feats = {}
        if args.label_num_hops > 0:
            g_label, _, _ = load_imdb_graph(base_dir)
            for ntype in g_label.ntypes:
                for k in list(g_label.nodes[ntype].data.keys()):
                    g_label.nodes[ntype].data.pop(k)

            label_seed = torch.zeros((num_nodes, num_classes), dtype=torch.float32)
            label_seed[train_nid, labels[train_nid].long()] = 1.0
            g_label.nodes[TGT_TYPE].data['Y'] = label_seed

            label_max_length = args.label_num_hops + 1
            g_label = hg_propagate_feat_dgl(
                g_label, TGT_TYPE, args.label_num_hops, label_max_length, [], echo=False,
            )
            for k in list(g_label.nodes[TGT_TYPE].data.keys()):
                label_feats[k] = g_label.nodes[TGT_TYPE].data[k]
            print(f'Label feature keys: {sorted(label_feats.keys())}')

        labels_cuda = labels.to(device)

        trainval_point = len(train_nid)
        valtest_point = trainval_point + len(val_nid)
        labeled_nid = np.concatenate([train_nid, val_nid, test_nid])
        labeled_num_nodes = len(labeled_nid)

        train_loader = torch.utils.data.DataLoader(
            torch.LongTensor(train_nid), batch_size=args.batch_size,
            shuffle=True, drop_last=False,
        )

        eval_batch_size = 2 * args.batch_size
        eval_loader = []
        for batch_idx in range((labeled_num_nodes - 1) // eval_batch_size + 1):
            batch_start = batch_idx * eval_batch_size
            batch_end = min(labeled_num_nodes, (batch_idx + 1) * eval_batch_size)
            batch = torch.LongTensor(labeled_nid[batch_start:batch_end])
            batch_feats = {k: x[batch] for k, x in feats.items()}
            batch_labels_feats = {k: x[batch] for k, x in label_feats.items()}
            eval_loader.append((batch, batch_feats, batch_labels_feats, None))

        model = SeHGNN(
            dataset='IMDB',
            nfeat=args.embed_size,
            hidden=args.hidden,
            nclass=num_classes,
            feat_keys=feats.keys(),
            label_feat_keys=label_feats.keys(),
            tgt_type=TGT_TYPE,
            dropout=args.dropout,
            input_drop=args.input_drop,
            att_drop=args.att_drop,
            n_fp_layers=args.n_fp_layers,
            n_task_layers=args.n_task_layers,
            act=args.act,
            residual=args.residual,
            data_size=data_size,
        )
        model = model.to(device)
        if seed == args.seeds[0]:
            print(model)
            print(f'# Params: {get_n_params(model)}')

        loss_fcn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )

        os.makedirs('checkpoint', exist_ok=True)
        skip_tag = '_skip' if args.imdb_skip_data else ''
        checkpt_file = f'checkpoint/sehgnn_imdb_nc_var{variant}{skip_tag}_seed{seed}.pkl'
        early_stopping = EarlyStopping(
            patience=args.patience, verbose=False, save_path=checkpt_file,
        )

        running_best_val = float('inf')
        best_epoch = 0
        epochs_run = 0
        train_start_time = time.time()

        for epoch in tqdm(range(args.epoch), desc=f'var{variant} seed={seed}'):
            epochs_run = epoch + 1
            gc.collect()
            if not args.cpu and torch.cuda.is_available():
                torch.cuda.synchronize()

            feats_relabeled = {k: v for k, v in feats.items()}
            loss, acc = train(
                model, feats_relabeled, label_feats, labels_cuda,
                loss_fcn, optimizer, train_loader, evaluator,
            )

            with torch.no_grad():
                model.eval()
                raw_preds = []
                for batch, batch_feats, batch_labels_feats, batch_mask in eval_loader:
                    batch = batch.to(device)
                    batch_feats = {k: x.to(device) for k, x in batch_feats.items()}
                    batch_labels_feats = {k: x.to(device) for k, x in batch_labels_feats.items()}
                    raw_preds.append(
                        model(batch, batch_feats, batch_labels_feats, None).cpu()
                    )
                raw_preds = torch.cat(raw_preds, dim=0)

                loss_train = loss_fcn(
                    raw_preds[:trainval_point], labels[train_nid]
                ).item()
                loss_val = loss_fcn(
                    raw_preds[trainval_point:valtest_point], labels[val_nid]
                )
                loss_test = loss_fcn(
                    raw_preds[valtest_point:], labels[test_nid]
                ).item()

            preds = raw_preds.argmax(dim=-1)
            train_acc = evaluator(preds[:trainval_point], labels[train_nid])
            val_acc = evaluator(preds[trainval_point:valtest_point], labels[val_nid])
            test_acc = evaluator(preds[valtest_point:], labels[test_nid])

            lv = loss_val.item()
            if lv < running_best_val - 1e-10:
                running_best_val = lv
                best_epoch = epoch

            early_stopping(loss_val, model)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

            if epoch > 0 and epoch % 20 == 0:
                print(
                    f'  Epoch {epoch}: train_loss={loss:.4f} '
                    f'val_loss={lv:.4f} '
                    f'val_f1=({val_acc[0]*100:.2f},{val_acc[1]*100:.2f}) '
                    f'test_f1=({test_acc[0]*100:.2f},{test_acc[1]*100:.2f})'
                )

        model.load_state_dict(torch.load(checkpt_file), strict=False)
        model.eval()
        with torch.no_grad():
            raw_preds = []
            for batch, batch_feats, batch_labels_feats, batch_mask in eval_loader:
                batch = batch.to(device)
                batch_feats = {k: x.to(device) for k, x in batch_feats.items()}
                batch_labels_feats = {k: x.to(device) for k, x in batch_labels_feats.items()}
                raw_preds.append(
                    model(batch, batch_feats, batch_labels_feats, None).cpu()
                )
            best_pred = torch.cat(raw_preds, dim=0)
        train_time_sec = time.time() - train_start_time

        preds = best_pred.argmax(dim=-1)
        best_val = evaluator(preds[trainval_point:valtest_point], labels[val_nid])
        best_test = evaluator(preds[valtest_point:], labels[test_nid])

        print(f'\nBest epoch {best_epoch}: '
              f'Val ({best_val[0]*100:.2f}, {best_val[1]*100:.2f})  '
              f'Test ({best_test[0]*100:.2f}, {best_test[1]*100:.2f})  '
              f'Epochs run: {epochs_run}')

        test_hit1, test_hit3, test_mrr = ranking_metrics_from_logits(
            best_pred[valtest_point:], labels[test_nid]
        )
        final_preds = best_pred.argmax(dim=-1)
        splits = [
            ('Train', train_nid, slice(0, trainval_point)),
            ('Val',   val_nid,   slice(trainval_point, valtest_point)),
            ('Test',  test_nid,  slice(valtest_point, None)),
        ]
        train_prec_macro = val_prec_macro = test_prec_macro = 0.0
        train_rec_macro = val_rec_macro = test_rec_macro = 0.0
        train_acc = val_acc = test_acc = 0.0
        train_micro_f1 = val_micro_f1 = test_micro_f1 = 0.0
        train_macro_f1 = val_macro_f1 = test_macro_f1 = 0.0
        for split_name, nids, sl in splits:
            true = labels[nids].numpy()
            pred = final_preds[sl].numpy()
            split_logits = best_pred[sl]
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
                train_acc, train_micro_f1, train_macro_f1 = acc, f1_micro, f1_macro
                train_prec_macro, train_rec_macro = prec_macro, rec_macro
            elif split_name == 'Val':
                val_acc, val_micro_f1, val_macro_f1 = acc, f1_micro, f1_macro
                val_prec_macro, val_rec_macro = prec_macro, rec_macro
            elif split_name == 'Test':
                test_acc, test_micro_f1, test_macro_f1 = acc, f1_micro, f1_macro
                test_prec_macro = prec_macro
                test_rec_macro = rec_macro

        all_results.append({
            'seed': seed,
            'best_epoch': best_epoch,
            'val_micro_f1_eval': best_val[0],
            'val_macro_f1_eval': best_val[1],
            'test_micro_f1_eval': best_test[0],
            'test_macro_f1_eval': best_test[1],
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
            'test_micro_f1': test_micro_f1,
            'test_macro_f1': test_macro_f1,
            'test_prec_macro': test_prec_macro,
            'test_rec_macro': test_rec_macro,
            'test_hit1': test_hit1,
            'test_hit3': test_hit3,
            'test_mrr': test_mrr,
            'train_time_sec': train_time_sec,
            'epochs_run': epochs_run,
        })

        del model
        if not args.imdb_skip_data:
            del g_copy
        if args.label_num_hops > 0:
            del g_label
        if not args.cpu:
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
    test_prec_macro = [r['test_prec_macro'] for r in all_results]
    test_rec_macro = [r['test_rec_macro'] for r in all_results]
    test_hit1 = [r['test_hit1'] for r in all_results]
    test_hit3 = [r['test_hit3'] for r in all_results]
    test_mrr = [r['test_mrr'] for r in all_results]
    epochs_run_list = [r['epochs_run'] for r in all_results]
    print(f'  TrainTime(s): {np.mean(train_time_sec):.2f} +/- {np.std(train_time_sec):.2f}')
    print(f'  Epochs Run:   {np.mean(epochs_run_list):.2f} +/- {np.std(epochs_run_list):.2f}')
    print(f'  Test Prec_macro: {np.mean(test_prec_macro)*100:.2f} +/- {np.std(test_prec_macro)*100:.2f}')
    print(f'  Test Rec_macro:  {np.mean(test_rec_macro)*100:.2f} +/- {np.std(test_rec_macro)*100:.2f}')
    print(f'  Test Hit@1:    {np.mean(test_hit1)*100:.2f} +/- {np.std(test_hit1)*100:.2f}')
    print(f'  Test Hit@3:    {np.mean(test_hit3)*100:.2f} +/- {np.std(test_hit3)*100:.2f}')
    print(f'  Test MRR:      {np.mean(test_mrr)*100:.2f} +/- {np.std(test_mrr)*100:.2f}')

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(
        description='SeHGNN NC on IMDB variants (movie genre, 3 classes)'
    )
    parser.add_argument('--variant', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6],
                        choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument('--seeds', default='1566911444,20241017,20251017')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--embed-size', type=int, default=512)
    parser.add_argument(
        '--num-hops', type=int, default=4,
        help='DGL propagation depth (ignored with --imdb-skip-data; skip mode uses M/MAM/MDM/MLM only).',
    )
    parser.add_argument(
        '--label-num-hops', type=int, default=4,
        help='Label propagation depth using train labels only (0 disables label propagation).',
    )
    parser.add_argument('--n-fp-layers', type=int, default=2)
    parser.add_argument('--n-task-layers', type=int, default=4)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--input-drop', type=float, default=0.0)
    parser.add_argument('--att-drop', type=float, default=0.)
    parser.add_argument('--act', type=str, default='none',
                        choices=['none', 'relu', 'leaky_relu', 'sigmoid'])
    parser.add_argument('--residual', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--batch-size', type=int, default=10000)
    parser.add_argument('--patience', type=int, default=50)
    add_imdb_skip_args(parser)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    print(args)

    all_variant_results = {}
    for v in args.variant:
        base_dir = resolve_imdb_nc_dir(
            v,
            imdb_skip_data=args.imdb_skip_data,
            imdb_data_root=args.imdb_data_root,
        )
        results = run_nc(args, v, base_dir)
        all_variant_results[v] = results

    if len(args.variant) > 1:
        print(f'\n{"="*60}')
        print('  Cross-Variant IMDB NC Summary')
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
            test_prec_macro = [r['test_prec_macro'] for r in results]
            test_rec_macro = [r['test_rec_macro'] for r in results]
            test_hit1 = [r['test_hit1'] for r in results]
            test_hit3 = [r['test_hit3'] for r in results]
            test_mrr = [r['test_mrr'] for r in results]
            epochs_run_list = [r['epochs_run'] for r in results]
            print(
                f'    TrainTime(s): {np.mean(train_time_sec):.2f} +/- {np.std(train_time_sec):.2f}  '
                f'    Epochs Run: {np.mean(epochs_run_list):.2f} +/- {np.std(epochs_run_list):.2f}  '
                f'    Test (extra): Prec_macro={np.mean(test_prec_macro)*100:.2f}'
                f' +/- {np.std(test_prec_macro)*100:.2f}  '
                f'Rec_macro={np.mean(test_rec_macro)*100:.2f}'
                f' +/- {np.std(test_rec_macro)*100:.2f}  '
                f'Hit@1={np.mean(test_hit1)*100:.2f} +/- {np.std(test_hit1)*100:.2f}  '
                f'Hit@3={np.mean(test_hit3)*100:.2f} +/- {np.std(test_hit3)*100:.2f}  '
                f'MRR={np.mean(test_mrr)*100:.2f} +/- {np.std(test_mrr)*100:.2f}'
            )
