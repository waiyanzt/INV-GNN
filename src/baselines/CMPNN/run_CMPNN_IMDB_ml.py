"""
CMPNN link prediction on IMDB (movie -> link )

Shared splits: build_IMDB_ml_shared_splits.py
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


def read_imdb_frame(csv_path):
    movies = pd.read_csv(csv_path, encoding='utf-8')
    movies = movies.drop_duplicates(subset='movie_imdb_link').dropna(
        subset=['movie_imdb_link', 'actor_1_name', 'director_name', 'genres']
    ).reset_index(drop=True)

    labels = np.full(len(movies), -1, dtype=np.int64)
    for i, genres in movies['genres'].astype(str).items():
        for g in genres.split('|'):
            g = g.strip()
            if g == 'Action':
                labels[i] = 0
                break
            elif g == 'Comedy':
                labels[i] = 1
                break
            elif g == 'Drama':
                labels[i] = 2
                break

    keep = np.where(labels >= 0)[0]
    movies = movies.iloc[keep].reset_index(drop=True)
    return movies


def _map_ml_pairs(arr):
    """Shared npz rows are movie indices; link local id == row (row-aligned)."""
    out = arr.astype(np.int64)
    return out


def _sample_k_negs(pos_pairs, L_all, true_set, k, rng):
    N = len(pos_pairs)
    neg = np.zeros((N, k), dtype=np.int64)
    for i, (mL, l_true) in enumerate(pos_pairs):
        cand = [x for x in L_all if (mL, x) not in true_set]
        chosen = rng.choice(cand, size=k, replace=(k > len(cand)))
        neg[i] = chosen
    return neg


def _offsets(movies):
    directors = sorted(set(movies['director_name'].dropna().tolist()))
    actors = sorted(set(movies['actor_1_name'].dropna().tolist()))
    M = len(movies)
    Dn = len(directors)
    An = len(actors)
    off_d = M
    off_a = M + Dn
    off_l = M + Dn + An
    N = off_l + M
    return M, Dn, An, off_d, off_a, off_l, N


def build_triplets(movies, variant, train_pos):
    """Structure edges + train movie->link only (DBLP-style). Directed triplets."""
    M, _Dn, _An, off_d, off_a, off_l, N = _offsets(movies)

    directors = sorted(set(movies['director_name'].dropna().tolist()))
    actors = sorted(set(movies['actor_1_name'].dropna().tolist()))

    def d_idx(name):
        return off_d + directors.index(name)

    def a_idx(name):
        return off_a + actors.index(name)

    triplets = []
    train_ml = set(map(tuple, train_pos.tolist()))

    if variant == 'v1':
        r_md = 0
        r_ma = 1
        r_ml = 2
        rel_ml = r_ml
        for i, row in movies.iterrows():
            i = int(i)
            di = d_idx(row['director_name'])
            triplets.append([i, di, r_md])
            for col in ['actor_1_name']:
                ai = a_idx(row[col])
                triplets.append([i, ai, r_ma])
            li = off_l + i
            triplets.append([i, li, r_ml])
    elif variant == 'v2':
        r_ld = 0
        r_la = 1
        r_ml = 2
        rel_ml = r_ml
        for i, row in movies.iterrows():
            i = int(i)
            li = off_l + i
            di = d_idx(row['director_name'])
            triplets.append([li, di, r_ld])
            for col in ['actor_1_name']:
                ai = a_idx(row[col])
                triplets.append([li, ai, r_la])
            triplets.append([i, li, r_ml])
    elif variant == 'v3':
        r_ml = 0
        r_la = 1
        r_md = 2
        rel_ml = r_ml
        for i, row in movies.iterrows():
            i = int(i)
            li = off_l + i
            triplets.append([i, li, r_ml])
            for col in ['actor_1_name']:
                ai = a_idx(row[col])
                triplets.append([li, ai, r_la])
            di = d_idx(row['director_name'])
            triplets.append([i, di, r_md])
    elif variant == 'v4':
        r_ma = 0
        r_ml = 1
        r_ld = 2
        rel_ml = r_ml
        for i, row in movies.iterrows():
            i = int(i)
            for col in ['actor_1_name']:
                ai = a_idx(row[col])
                triplets.append([i, ai, r_ma])
            li = off_l + i
            triplets.append([i, li, r_ml])
            di = d_idx(row['director_name'])
            triplets.append([li, di, r_ld])
    else:
        raise ValueError(f"variant must be v1-v4, got {variant}")

    for m, l in train_ml:
        triplets.append([int(m), off_l + int(l), rel_ml])

    triplets = np.asarray(triplets, dtype=np.int64)
    num_relation = rel_ml + 1
    meta = {
        'num_entity': N,
        'num_relation': num_relation,
        'rel_ml': rel_ml,
        'off_l': off_l,
        'counts': {'train_pos': len(train_ml)},
    }
    return triplets, meta


def preprocess_imdb_ml(csv_path, shared_npz, variant, neg_k, seed):
    movies = read_imdb_frame(csv_path)
    shared = np.load(shared_npz)
    train_pos = _map_ml_pairs(shared['train_pos'])
    val_pos = _map_ml_pairs(shared['val_pos'])
    test_pos = _map_ml_pairs(shared['test_pos'])

    triplets, meta = build_triplets(movies, variant, train_pos)

    rng = np.random.RandomState(seed)
    L_all = np.arange(len(movies), dtype=np.int64)
    train_true = set(map(tuple, train_pos.tolist()))
    val_true = set(map(tuple, val_pos.tolist()))
    test_true = set(map(tuple, test_pos.tolist()))
    all_true = train_true | val_true | test_true

    k = min(int(neg_k), max(1, len(L_all) - 1))
    train_neg = _sample_k_negs(train_pos, L_all, all_true, k, rng)
    val_neg = _sample_k_negs(val_pos, L_all, all_true, k, rng)
    test_neg = _sample_k_negs(test_pos, L_all, all_true, k, rng)

    meta['neg_k'] = k
    meta['counts'].update({
        'val_pos': len(val_pos),
        'test_pos': len(test_pos),
    })

    splits = dict(
        train_pos=train_pos, train_neg=train_neg,
        val_pos=val_pos, val_neg=val_neg,
        test_pos=test_pos, test_neg=test_neg,
    )
    return triplets, meta, splits


class IMDB_ML_Query(torch_data.Dataset):
    def __init__(self, pos, neg, off_l):
        self.mL = pos[:, 0].astype(np.int64)
        self.l_true = pos[:, 1].astype(np.int64)
        self.l_neg = neg.astype(np.int64)
        self.off_l = off_l

    def __len__(self):
        return len(self.mL)

    def __getitem__(self, i):
        h = int(self.mL[i])
        t_true = self.off_l + self.l_true[i]
        t_neg = self.off_l + self.l_neg[i]
        return h, t_true, t_neg


def collate_query(batch, rel_ml):
    """(B, 1+K) tensors — required when graph.num_relation > 0 (CMPNN KG forward)."""
    b = list(zip(*batch))
    h = torch.tensor(b[0], dtype=torch.long)
    t_true = torch.tensor(b[1], dtype=torch.long)
    t_neg = torch.stack([torch.tensor(x, dtype=torch.long) for x in b[2]])
    B, K = t_neg.size()
    h_index = h.unsqueeze(1).expand(-1, 1 + K)
    t_index = torch.cat([t_true.unsqueeze(1), t_neg], dim=1)
    r_index = torch.full_like(h_index, rel_ml, dtype=torch.long)
    y = torch.zeros_like(h_index, dtype=torch.float)
    y[:, 0] = 1.0
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
        probs = torch.sigmoid(scores)
        B, L = probs.shape
        K = L - 1
        if B > 0 and K > 0:
            s_true = probs[:, 0]
            better = (probs[:, 1:] >= s_true.unsqueeze(1)).sum(dim=1)
            rank = better + 1
            hits1 += (rank <= 1).sum().item()
            hits3 += (rank <= 3).sum().item()
            hits5 += (rank <= 5).sum().item()
            rr_sum += (1.0 / rank.float()).sum().item()
            n_q += B
        all_y.append(y.flatten())
        all_p.append(probs.flatten().detach().cpu().numpy())

    y_all = torch.cat(all_y).numpy()
    p_all = np.concatenate(all_p)
    auc = roc_auc_score(y_all, p_all)
    ap = average_precision_score(y_all, p_all)
    return {
        'auc': auc, 'ap': ap,
        'hits1': hits1 / max(n_q, 1), 'hits3': hits3 / max(n_q, 1),
        'hits5': hits5 / max(n_q, 1), 'mrr': rr_sum / max(n_q, 1),
    }


@torch.no_grad()
def evaluate_full(model, graph, test_pos, test_neg, off_l, rel_ml,
                  batch_size, device, threshold):
    model.eval()
    K = test_neg.shape[1]
    all_scores, all_labels, all_pairs = [], [], []
    hits1 = hits3 = hits5 = 0
    rr_sum = 0.0
    n_q = 0
    n_pos = len(test_pos)

    for start in range(0, n_pos, batch_size):
        end = min(start + batch_size, n_pos)
        pos_batch = test_pos[start:end]
        neg_batch = test_neg[start:end]
        B = len(pos_batch)

        h_list, t_list = [], []
        for j in range(B):
            mL = int(pos_batch[j, 0])
            l_true = int(pos_batch[j, 1])
            h_global = mL
            t_true_global = off_l + l_true
            t_neg_global = off_l + neg_batch[j].astype(np.int64)
            h_idx = np.full(1 + K, h_global, dtype=np.int64)
            t_idx = np.concatenate([[t_true_global], t_neg_global])
            h_list.append(h_idx)
            t_list.append(t_idx)

        h_index = torch.tensor(np.stack(h_list), dtype=torch.long)
        t_index = torch.tensor(np.stack(t_list), dtype=torch.long)
        r_index = torch.full_like(h_index, rel_ml)

        scores = _cmpnn_forward_lp(model, graph, h_index, t_index, r_index, device)
        probs = torch.sigmoid(scores)

        s_true = probs[:, 0]
        better = (probs[:, 1:] >= s_true.unsqueeze(1)).sum(dim=1)
        rank = better + 1
        hits1 += (rank <= 1).sum().item()
        hits3 += (rank <= 3).sum().item()
        hits5 += (rank <= 5).sum().item()
        rr_sum += (1.0 / rank.float()).sum().item()
        n_q += B

        probs_np = probs.cpu().numpy()
        for ki in range(B):
            l_neg = neg_batch[ki]
            all_scores.append(probs_np[ki, 0])
            all_labels.append(1)
            all_pairs.append((int(pos_batch[ki, 0]), int(pos_batch[ki, 1])))
            for ln in l_neg:
                all_scores.append(probs_np[ki, 1 + np.where(neg_batch[ki] == ln)[0][0]])
                all_labels.append(0)
                all_pairs.append((int(pos_batch[ki, 0]), int(ln)))

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

    h1 = hits1 / max(n_q, 1)
    h3 = hits3 / max(n_q, 1)
    h5 = hits5 / max(n_q, 1)
    mrr = rr_sum / max(n_q, 1)

    pairs_arr = np.array(all_pairs)
    scores_df = pd.DataFrame({
        'movie_local': pairs_arr[:, 0], 'link_local': pairs_arr[:, 1],
        'label': y_true, 'prob': y_proba,
    })

    return {
        'auc': auc, 'ap': ap,
        'hits1': h1, 'hits3': h3, 'hits5': h5, 'mrr': mrr,
        'top1_accuracy': h1,
        'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc,
        'confusion': (TP, TN, FP, FN),
        'scores_df': scores_df,
    }


def run_model_imdb_ml(csv_path, shared_npz, variant, neg_k, input_dim,
                      hidden_dim, num_layers, num_epochs, patience, batch_size,
                      threshold, save_postfix, base_seed, gpu, use_cpu,
                      eval_only, checkpoint_path):
    t0 = time.time()
    print('-> Preprocessing IMDB for CMPNN ...')
    triplets, meta, splits = preprocess_imdb_ml(csv_path, shared_npz, variant, neg_k, base_seed)
    print(f'   done in {time.time() - t0:.1f}s')

    off_l = meta['off_l']
    rel_ml = meta.get('rel_ml', 2)

    print(f"-> KG: {meta['num_entity']} entities, {meta['num_relation']} relations, "
          f"{len(triplets)} triplets  variant={variant}")
    print(f"  rel_ml={rel_ml}  remove_one_hop=True")
    print(f"-> Splits: train_pos={meta['counts']['train_pos']}, "
          f"val_pos={meta['counts']['val_pos']}, test_pos={meta['counts']['test_pos']}")
    print(f"-> neg_k={meta['neg_k']}")

    if use_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cuda:0')
    print(f'-> Device: {device}')

    edge_list = torch.as_tensor(triplets, dtype=torch.long)
    graph = torchdrug.data.Graph(edge_list, num_node=int(meta['num_entity']),
                                 num_relation=int(meta['num_relation']))
    graph = graph.to(device)

    train_ds = IMDB_ML_Query(splits['train_pos'], splits['train_neg'], off_l)
    val_ds = IMDB_ML_Query(splits['val_pos'], splits['val_neg'], off_l)
    train_loader = torch_data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                         collate_fn=lambda b: collate_query(b, rel_ml=rel_ml))
    val_loader = torch_data.DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                       collate_fn=lambda b: collate_query(b, rel_ml=rel_ml))

    if hidden_dim != input_dim:
        raise ValueError('For current CMPNN impl, set hidden_dim == input_dim '
                         '(boundary/message dim must match).')

    model = CMPNN(input_dim=input_dim,
                  hidden_dims=[hidden_dim] * num_layers,
                  num_relation=int(meta['num_relation']),
                  message_func='distmult', aggregate_func='pna',
                  short_cut=True, layer_norm=True, dependent=False,
                  set_boundary=True, remove_one_hop=True,
                  activation='relu', initialization='Query')
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'-> Model params: {n_params}')

    optimizer = torch.optim.Adam(model.parameters())
    os.makedirs('checkpoint', exist_ok=True)
    ckpt_path = f'checkpoint/checkpoint_{save_postfix}.pt'
    early_stopping = EarlyStopping(patience=patience, verbose=False, save_path=ckpt_path)

    train_wall_sec = 0.0
    epochs_ran = 0
    if not eval_only:
        train_t0 = time.perf_counter()
        for epoch in range(num_epochs):
            epoch_t0 = time.time()
            model.train()
            losses = []
            pbar = tqdm(train_loader, desc=f'Epoch {epoch} [train]')
            for h_index, t_index, r_index, _y in pbar:
                scores = _cmpnn_forward_lp(model, graph, h_index, t_index, r_index, device)
                loss = neg_logsigmoid_loss(scores)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                pbar.set_postfix(loss=f'{np.mean(losses):.4f}')

            val_metrics = evaluate(model, graph, val_loader, device)
            print(f'Epoch {epoch} | val_AP={val_metrics["ap"]:.4f} '
                  f'val_AUC={val_metrics["auc"]:.4f} | Time(s) {time.time() - epoch_t0:.1f}')
            early_stopping(-val_metrics['ap'], model)
            epochs_ran = epoch + 1
            if early_stopping.early_stop:
                print('Early stopping!')
                break
        train_wall_sec = float(time.perf_counter() - train_t0)
    else:
        print('-> Eval-only: loading checkpoint ...')

    model.load_state_dict(torch.load(ckpt_path))
    print('-> Testing best checkpoint ...')

    results = evaluate_full(model, graph, splits['test_pos'], splits['test_neg'],
                            off_l, rel_ml, batch_size, device, threshold)
    cm = results['confusion']

    print("-> Primary (ranking): predict tail by **argmax** over 1+K scores "
          "(same spirit as NC argmax over class logits).")
    print(f"    Top-1 acc = Hits@1 = {results['hits1']:.4f}  |  MRR = {results['mrr']:.4f}")
    print(f"    AUC = {results['auc']:.4f}  AP = {results['ap']:.4f}")
    print(f"    Hits@3 = {results['hits3']:.4f}  Hits@5 = {results['hits5']:.4f}")
    print(f"  Secondary (sigmoid threshold {threshold}; optional if scores shift):")
    print(f"    Confusion: TP={cm[0]} TN={cm[1]} FP={cm[2]} FN={cm[3]}")
    print(f"    Precision = {results['precision']:.4f}  Recall = {results['recall']:.4f}")
    print(f"    F1 = {results['f1']:.4f}  Accuracy = {results['accuracy']:.4f}")
    print(f"    Train wall (s)={train_wall_sec:.2f}  Epochs={epochs_ran}")

    results['scores_df'].to_csv(f'{save_postfix}_scores.csv', index=False)

    out = {k: v for k, v in results.items() if k != 'scores_df'}
    out['train_wall_sec'] = train_wall_sec
    out['epochs_ran'] = float(epochs_ran)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='CMPNN-LP on IMDB movie-link (MAGNN graph variants)')
    ap.add_argument('--csv', default='../MAGNN/data/raw/IMDB/movie_metadata.csv',
                    help='movie_metadata.csv (MAGNN IMDB raw)')
    ap.add_argument('--shared-npz', default='IMDB_ml_shared_splits.npz',
                    help='IMDB_ml_shared_splits.npz from build script')
    ap.add_argument('--variant', default='v1',
                    help='Graph layout matching preprocess_IMDB_star*.py')
    ap.add_argument('--input-dim', type=int, default=32)
    ap.add_argument('--hidden-dim', type=int, default=32)
    ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--epoch', type=int, default=20)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--neg-k', type=int, default=19)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--seeds', default='1566911444,20241017,20251017')
    ap.add_argument('--save-postfix', default='IMDB_cmpnn_ml_v1')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--checkpoint', default=None)
    args = ap.parse_args()

    if args.eval_only and args.checkpoint is None:
        ap.error('--eval-only requires --checkpoint')

    seeds = [int(s) for s in args.seeds.split(',')]
    all_results = []

    for s in seeds:
        print('=' * 60)
        print(f'\n Seed {s}')
        set_seed(s)
        _ = run_model_imdb_ml(
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
              f"F1={st['f1']:.4f}  Acc={st['accuracy']:.4f}  "
              f"train_s={st['train_wall_sec']:.1f}  epochs={int(st['epochs_ran'])}")

    print('\nFinal mean +/- std across all seeds:')
    for key in ['auc', 'ap', 'hits1', 'mrr', 'hits3', 'hits5',
                'precision', 'recall', 'f1', 'accuracy',
                'train_wall_sec', 'epochs_ran']:
        vals = [r[key] for r in all_results]
        print(f"  {key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
