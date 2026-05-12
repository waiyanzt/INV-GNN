"""CMPNN link prediction: IMDB movie->genre (v1-v4), metrics aligned with ``run_CMPNN_IMDB_ml``.

Graph topology is defined in ``imdb_mg_data.build_triplets_mg`` (movie-hub v1, link-hub v2, etc.);
``num_relation=4`` with ``rel_mg=3`` always.
"""
import os
import sys
import time
import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils import data as torch_data
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

import torchdrug

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'MAGNN'))

from cmpnn.model import CMPNN
from imdb_mg_data import MAX_DISTINCT_WRONG_GENRES, preprocess_imdb_mg
from utils.pytorchtools import EarlyStopping


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


class IMDB_MG_Query(torch_data.Dataset):
    def __init__(self, pos, neg, off_g):
        self.m = pos[:, 0].astype(np.int64)
        self.g_true = pos[:, 1].astype(np.int64)
        self.g_neg = neg.astype(np.int64)
        self.off_g = off_g

    def __len__(self):
        return len(self.m)

    def __getitem__(self, i):
        h = int(self.m[i])
        t_true = self.off_g + self.g_true[i]
        t_neg = self.off_g + self.g_neg[i]
        return h, t_true, t_neg


def collate_query_mg(batch, rel_mg):
    b = list(zip(*batch))
    h = torch.tensor(b[0], dtype=torch.long)
    t_true = torch.tensor(b[1], dtype=torch.long)
    t_neg = torch.stack([torch.tensor(x, dtype=torch.long) for x in b[2]])
    B, K = t_neg.size()
    h_index = h.repeat(1 + K)
    t_index = torch.cat([t_true.unsqueeze(1), t_neg], dim=1).view(-1)
    r_index = torch.full((B * (1 + K),), rel_mg, dtype=torch.long)
    y = torch.cat([torch.ones(B), torch.zeros(B * K)]).float()
    return h_index, t_index, r_index, y


def neg_logsigmoid_loss(scores):
    pos = scores[..., 0]
    neg = scores[..., 1:]
    return -(F.logsigmoid(pos).mean() + F.logsigmoid(-neg).mean())


def _cmpnn_forward_lp(model, graph, h_index, t_index, r_index, device):
    score = model(graph, h_index.to(device), t_index.to(device),
                  r_index.to(device), all_loss=None, metric=None)
    return score


@torch.no_grad()
def evaluate(model, graph, loader, device):
    model.eval()
    all_y, all_p = [], []
    hits1 = hits3 = hits5 = 0
    rr_sum = 0.0
    n_q = 0
    for h_index, t_index, r_index, y in loader:
        scores = _cmpnn_forward_lp(model, graph, h_index, t_index, r_index, device)
        B = y.sum().int().item()
        K = scores.dim() // B - 1 if B > 0 else 0
        probs = torch.sigmoid(scores).view(B, 1 + K) if B > 0 else torch.sigmoid(scores)
        if B > 0 and K > 0:
            s_true = probs[:, 0]
            rank = (probs[:, 1:] >= s_true.unsqueeze(1)).sum(dim=1) + 1
            hits1 += (rank <= 1).sum().item()
            hits3 += (rank <= 3).sum().item()
            hits5 += (rank <= 5).sum().item()
            rr_sum += (1.0 / rank.float()).sum().item()
            n_q += B
        all_y.append(y)
        all_p.append(torch.sigmoid(scores).flatten().detach().cpu().numpy())

    y_all = torch.cat(all_y).numpy()
    p_all = np.concatenate(all_p)
    return {
        'auc': roc_auc_score(y_all, p_all),
        'ap': average_precision_score(y_all, p_all),
        'hits1': hits1 / max(n_q, 1), 'hits3': hits3 / max(n_q, 1),
        'hits5': hits5 / max(n_q, 1), 'mrr': rr_sum / max(n_q, 1),
    }


@torch.no_grad()
def evaluate_full(model, graph, test_pos, test_neg, off_g, rel_mg,
                  batch_size, device, threshold):
    model.eval()
    K = test_neg.shape[1]
    all_scores, all_labels, all_pairs = [], [], []
    hits1 = hits3 = hits5 = 0
    rr_sum = 0.0
    n_q = 0

    for start in range(0, len(test_pos), batch_size):
        pos_batch = test_pos[start:start + batch_size]
        neg_batch = test_neg[start:start + batch_size]
        B = len(pos_batch)

        h_list, t_list = [], []
        for j in range(B):
            m = int(pos_batch[j, 0])
            g_true = int(pos_batch[j, 1])
            h_idx = np.full(1 + K, m, dtype=np.int64)
            t_idx = np.concatenate([[off_g + g_true], off_g + neg_batch[j].astype(np.int64)])
            h_list.append(h_idx)
            t_list.append(t_idx)

        h_index = torch.tensor(np.concatenate(h_list), dtype=torch.long)
        r_index = torch.full_like(h_index, rel_mg)
        t_index = torch.tensor(np.concatenate(t_list), dtype=torch.long)

        scores = _cmpnn_forward_lp(model, graph, h_index, t_index, r_index, device)
        probs = torch.sigmoid(scores).view(B, 1 + K)

        rank = (probs[:, 1:] >= probs[:, 0:1]).sum(dim=1) + 1
        hits1 += (rank <= 1).sum().item()
        hits3 += (rank <= 3).sum().item()
        hits5 += (rank <= 5).sum().item()
        rr_sum += (1.0 / rank.float()).sum().item()
        n_q += B

        probs_np = probs.cpu().numpy()
        for ki in range(B):
            all_scores.append(probs_np[ki, 0])
            all_labels.append(1)
            all_pairs.append((int(pos_batch[ki, 0]), int(pos_batch[ki, 1])))
            for jj in range(K):
                all_scores.append(probs_np[ki, 1 + jj])
                all_labels.append(0)
                all_pairs.append((int(pos_batch[ki, 0]), int(neg_batch[ki, jj])))

    y_true = np.array(all_labels)
    y_proba = np.array(all_scores)
    auc = roc_auc_score(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    y_pred = (y_proba >= threshold).astype(int)
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = TP / max(TP + FP, 1)
    rec = TP / max(TP + FN, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    acc = (TP + TN) / max(TP + TN + FP + FN, 1)

    pairs_arr = np.array(all_pairs)
    scores_df = pd.DataFrame({
        'movie_local': pairs_arr[:, 0], 'genre_id': pairs_arr[:, 1],
        'label': y_true, 'prob': y_proba,
    })

    return {
        'auc': auc, 'ap': ap,
        'hits1': hits1 / max(n_q, 1), 'hits3': hits3 / max(n_q, 1),
        'hits5': hits5 / max(n_q, 1), 'mrr': rr_sum / max(n_q, 1),
        'top1_accuracy': hits1 / max(n_q, 1),
        'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc,
        'confusion': (TP, TN, FP, FN),
        'scores_df': scores_df,
    }


def run_model_imdb_mg(csv_path, shared_npz, variant, neg_k, input_dim,
                      hidden_dim, num_layers, num_epochs, patience, batch_size,
                      threshold, save_postfix, base_seed, gpu, use_cpu,
                      eval_only, checkpoint_path):
    t0 = time.time()
    triplets, meta, splits = preprocess_imdb_mg(csv_path, shared_npz, variant, neg_k, base_seed)
    print(f'-> Preprocess {time.time()-t0:.1f}s | entities={meta["num_entity"]} '
          f'rel={meta["num_relation"]} triplets={len(triplets)}')

    off_g = meta['off_g']
    rel_mg = meta['rel_mg']
    print(f'-> Genre tails: {meta.get("num_genres", 3)} nodes at off_g={off_g}; '
          f'rel_mg={rel_mg} (M->G); num_relation={meta["num_relation"]}')
    print(f'  {meta.get("neg_note", "")}')

    if use_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cuda:0')

    edge_list = torch.as_tensor(triplets, dtype=torch.long)
    graph = torchdrug.data.Graph(edge_list, num_node=int(meta['num_entity']),
                                 num_relation=int(meta['num_relation']))
    graph = graph.to(device)

    train_ds = IMDB_MG_Query(splits['train_pos'], splits['train_neg'], off_g)
    val_ds = IMDB_MG_Query(splits['val_pos'], splits['val_neg'], off_g)
    train_loader = torch_data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                         collate_fn=lambda b: collate_query_mg(b, rel_mg=rel_mg))
    val_loader = torch_data.DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                       collate_fn=lambda b: collate_query_mg(b, rel_mg=rel_mg))

    if hidden_dim != input_dim:
        raise ValueError('hidden_dim must equal input_dim')

    model = CMPNN(input_dim=input_dim,
                  hidden_dims=[hidden_dim] * num_layers,
                  message_func='distmult', aggregate_func='pna',
                  short_cut=True, layer_norm=True, dependent=False,
                  set_boundary=True, remove_one_hop=True,
                  activation='relu', initialization='Query')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters())
    os.makedirs('checkpoint', exist_ok=True)
    ckpt_path = f'checkpoint/checkpoint_{save_postfix}.pt'
    early_stopping = EarlyStopping(patience=patience, verbose=False, save_path=ckpt_path)

    if eval_only:
        print('-> eval-only, skip training')
    else:
        for epoch in range(num_epochs):
            model.train()
            losses = []
            for h_index, t_index, r_index, _y in tqdm(train_loader, desc=f'Epoch {epoch}'):
                scores = _cmpnn_forward_lp(model, graph, h_index, t_index, r_index, device)
                B = _y.sum().int().item()
                K = scores.dim() // B - 1 if B > 0 else 0
                loss = neg_logsigmoid_loss(scores.view(B, 1 + K) if B > 0 else scores)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            vm = evaluate(model, graph, val_loader, device)
            print(f'Epoch {epoch} loss={np.mean(losses):.4f} '
                  f'val_AP={vm["ap"]:.4f} val_AUC={vm["auc"]:.4f}')
            early_stopping(-vm['ap'], model)
            if early_stopping.early_stop:
                print('Early stopping')
                break

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'Missing checkpoint: {ckpt_path}')
    model.load_state_dict(torch.load(ckpt_path))
    print('-> Testing best checkpoint ...')

    results = evaluate_full(model, graph, splits['test_pos'], splits['test_neg'],
                            off_g, rel_mg, batch_size, device, threshold)
    cm = results['confusion']

    print(f"-> Primary (ranking): argmax over 1+K genre tails "
          f"(K={meta['neg_k']}; 3 genres total, max 2 distinct wrong / query).")
    print(f"    Top-1 acc = Hits@1 = {results['hits1']:.4f}  |  MRR = {results['mrr']:.4f}")
    print(f"    AUC = {results['auc']:.4f}  AP = {results['ap']:.4f}")
    print(f"    Hits@3 = {results['hits3']:.4f}  Hits@5 = {results['hits5']:.4f}")
    print(f"  Secondary (sigmoid threshold {threshold}; optional if scores shift):")
    print(f"    Confusion: TP={cm[0]} TN={cm[1]} FP={cm[2]} FN={cm[3]}")
    print(f"    Precision = {results['precision']:.4f}  Recall = {results['recall']:.4f}")
    print(f"    F1 = {results['f1']:.4f}  Accuracy = {results['accuracy']:.4f}")

    results['scores_df'].to_csv(f'{save_postfix}_scores.csv', index=False)
    return {k: v for k, v in results.items() if k != 'scores_df'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='CMPNN IMDB movie->genre (v1-v4)')
    ap.add_argument('--csv', default='../MAGNN/data/raw/IMDB/movie_metadata.csv')
    ap.add_argument('--shared-npz', default='IMDB_mg_shared_splits.npz')
    ap.add_argument('--variant', default='v1')
    ap.add_argument('--input-dim', type=int, default=32)
    ap.add_argument('--hidden-dim', type=int, default=32)
    ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--epoch', type=int, default=20)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--neg-k', type=int, default=2,
                    help='Distinct wrong genre tails per query; max 2 (three genres).')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--seeds', default='1566911444,20241017,20251017')
    ap.add_argument('--save-postfix', default='IMDB_cmpnn_mg_v1')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--checkpoint', default=None)
    args = ap.parse_args()

    if args.neg_k < 1 or args.neg_k > MAX_DISTINCT_WRONG_GENRES:
        ap.error(f'--neg-k must be in [1, {MAX_DISTINCT_WRONG_GENRES}] '
                 f'(three genres -> at most two distinct wrong tails).')
    if args.eval_only and args.checkpoint is None:
        ap.error('--eval-only requires --checkpoint')

    seeds = [int(s) for s in args.seeds.split(',')]
    all_results = []

    for s in seeds:
        print('=' * 60)
        print(f'\n Seed {s}')
        set_seed(s)
        _ = run_model_imdb_mg(
            csv_path=args.csv, shared_npz=args.shared_npz,
            variant=args.variant, neg_k=args.neg_k,
            input_dim=args.input_dim, hidden_dim=args.hidden_dim,
            num_layers=args.layers, num_epochs=args.epoch,
            patience=args.patience, batch_size=args.batch_size,
            threshold=args.threshold,
            save_postfix=f'{args.save_postfix}_seed{s}',
            base_seed=s, gpu=args.gpu, use_cpu=args.cpu,
            eval_only=args.eval_only, checkpoint_path=args.checkpoint,
        )
        all_results.append(_)

    print('\n Summary over seeds\n')
    for s, st in zip(seeds, all_results):
        print(f"Seed {s}: AUC={st['auc']:.4f}  AP={st['ap']:.4f}  "
              f"Hits@1={st['hits1']:.4f}  MRR={st['mrr']:.4f}  "
              f"Hits@3={st['hits3']:.4f}  Hits@5={st['hits5']:.4f}  "
              f"Prec={st['precision']:.4f}  Rec={st['recall']:.4f}  "
              f"F1={st['f1']:.4f}  Acc={st['accuracy']:.4f}")

    print('\nFinal mean +/- std across all seeds:')
    for key in ['auc', 'ap', 'hits1', 'mrr', 'hits3', 'hits5',
                'precision', 'recall', 'f1', 'accuracy']:
        vals = [r[key] for r in all_results]
        print(f"  {key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
