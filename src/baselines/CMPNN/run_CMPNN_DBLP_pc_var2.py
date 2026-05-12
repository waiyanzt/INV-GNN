"""
CMPNN link prediction on DBLP paper→conference, variant 2: area connected
to Conference (venue) rather than to Paper.

Shared splits loaded from ``DBLP_pc_shared_splits.npz``.
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


def _read_tsv(path, names):
    return pd.read_csv(path, sep='\t', header=None, names=names, encoding='utf-8')


def _map_pairs(arr, p_map, c_map):
    df = pd.DataFrame(arr, columns=['paper_id', 'conf_id'])
    df['paper_id'] = df['paper_id'].map(p_map)
    df['conf_id'] = df['conf_id'].map(c_map)
    df = df.dropna()
    return df.to_numpy().astype(np.int64)


def _sample_k_negs(pos_pairs, C_all, true_set, k, rng):
    N = len(pos_pairs)
    neg = np.zeros((N, k), dtype=np.int64)
    for i, (pL, c_true) in enumerate(pos_pairs):
        cand = [c for c in C_all if (pL, c) not in true_set]
        chosen = rng.choice(cand, size=k, replace=(k > len(cand)))
        neg[i] = chosen
    return neg


def preprocess_dblp_var2(raw_dir, shared_npz, neg_k, seed):
    """Build KG triplets for var2: Area connected to Conference (venue)."""
    rng = np.random.RandomState(seed)

    author = _read_tsv(os.path.join(raw_dir, 'author_label.txt'),
                       ['author_id', 'label'])
    pa = _read_tsv(os.path.join(raw_dir, 'paper_author.txt'),
                   ['paper_id', 'author_id'])
    pc = _read_tsv(os.path.join(raw_dir, 'paper_conf.txt'),
                   ['paper_id', 'conf_id'])
    pt = _read_tsv(os.path.join(raw_dir, 'paper_term.txt'),
                   ['paper_id', 'term_id'])

    pa = pa.astype(np.int64)
    pc = pc.astype(np.int64)
    pt = pt.astype(np.int64)
    author = author.astype(np.int64)

    valid_authors = set(author['author_id'].tolist())
    valid_papers = set(pa[pa['author_id'].isin(valid_authors)]['paper_id'].tolist())

    shared = np.load(shared_npz)

    a2r = author.set_index('author_id')['label']
    paper_area = pa.assign(area_id=pa['author_id'].map(a2r)).dropna()
    paper_area = paper_area[['paper_id', 'area_id']].drop_duplicates().astype(np.int64)

    # var2: area connects to conference, so build conference→area via paper
    cr = pc.merge(paper_area, on='paper_id')[['conf_id', 'area_id']].drop_duplicates()

    Au_ids = sorted(pa['author_id'].unique())
    Pa_ids = sorted(set(pa['paper_id'].unique()) | set(pc['paper_id'].unique()) | set(pt['paper_id'].unique()))
    Te_ids = sorted(pt['term_id'].unique())
    Co_ids = sorted(pc['conf_id'].unique())
    Ar_ids = sorted(paper_area['area_id'].unique())

    Au_map = {a: i for i, a in enumerate(Au_ids)}
    Pa_map = {p: i for i, p in enumerate(Pa_ids)}
    Te_map = {t: i for i, t in enumerate(Te_ids)}
    Co_map = {c: i for i, c in enumerate(Co_ids)}
    Ar_map = {r: i for i, r in enumerate(Ar_ids)}

    A = len(Au_ids)
    P = len(Pa_ids)
    T = len(Te_ids)
    C = len(Co_ids)
    R = len(Ar_ids)

    offA = 0
    offP = A
    offT = A + P
    offC = A + P + T
    offR = A + P + T + C
    N = A + P + T + C + R

    train_pos = _map_pairs(shared['train_pos'], Pa_map, Co_map)
    val_pos = _map_pairs(shared['val_pos'], Pa_map, Co_map)
    test_pos = _map_pairs(shared['test_pos'], Pa_map, Co_map)

    C_all = np.arange(C, dtype=np.int64)
    train_true = set(map(tuple, train_pos.tolist()))
    val_true = set(map(tuple, val_pos.tolist()))
    test_true = set(map(tuple, test_pos.tolist()))
    all_true = train_true | val_true | test_true

    k = min(int(neg_k), max(1, C - 1))
    train_neg = _sample_k_negs(train_pos, C_all, all_true, k, rng)
    val_neg = _sample_k_negs(val_pos, C_all, all_true, k, rng)
    test_neg = _sample_k_negs(test_pos, C_all, all_true, k, rng)

    REL = {'P-A': 0, 'P-T': 1, 'C-R': 2, 'P-C': 3}
    num_relation = len(REL)
    rel_pc = REL['P-C']

    triplets = []
    for _, row in pa.iterrows():
        pL = Pa_map.get(int(row['paper_id']))
        aL = Au_map.get(int(row['author_id']))
        if pL is not None and aL is not None:
            triplets.append([offP + pL, offA + aL, REL['P-A']])

    for _, row in pt.iterrows():
        pL = Pa_map.get(int(row['paper_id']))
        tL = Te_map.get(int(row['term_id']))
        if pL is not None and tL is not None:
            triplets.append([offP + pL, offT + tL, REL['P-T']])

    # var2: area→conference (C-R relation)
    for _, row in cr.iterrows():
        c0 = Co_map.get(int(row['conf_id']))
        r0 = Ar_map.get(int(row['area_id']))
        if c0 is not None and r0 is not None:
            triplets.append([offC + c0, offR + r0, REL['C-R']])

    for pL, cL in train_pos:
        triplets.append([offP + int(pL), offC + int(cL), rel_pc])

    triplets = np.asarray(triplets, dtype=np.int64)

    meta = {
        'num_entity': N,
        'num_relation': num_relation,
        'rel_pc': rel_pc,
        'offsets': {'A': offA, 'P': offP, 'T': offT, 'C': offC, 'R': offR},
        'counts': {
            'train_pos': len(train_pos),
            'val_pos': len(val_pos),
            'test_pos': len(test_pos),
        },
        'neg_k': k,
        'offP': offP,
        'offC': offC,
    }

    splits = {
        'train_pos': train_pos, 'train_neg': train_neg,
        'val_pos': val_pos, 'val_neg': val_neg,
        'test_pos': test_pos, 'test_neg': test_neg,
    }
    return triplets, meta, splits


class DBLP_PC_Query(torch_data.Dataset):
    def __init__(self, pos, neg, offP, offC):
        self.pL = pos[:, 0].astype(np.int64)
        self.c_true = pos[:, 1].astype(np.int64)
        self.c_neg = neg.astype(np.int64)
        self.offP = offP
        self.offC = offC

    def __len__(self):
        return len(self.pL)

    def __getitem__(self, i):
        h = self.offP + int(self.pL[i])
        t_true = self.offC + self.c_true[i]
        t_neg = self.offC + self.c_neg[i]
        return h, t_true, t_neg


def collate_query(batch, rel_pc):
    b = list(zip(*batch))
    h = torch.tensor(b[0], dtype=torch.long)
    t_true = torch.tensor(b[1], dtype=torch.long)
    t_neg = torch.stack([torch.tensor(x, dtype=torch.long) for x in b[2]])
    B, K = t_neg.size()
    h_index = h.repeat(1 + K)
    t_index = torch.cat([t_true.unsqueeze(1), t_neg], dim=1).view(-1)
    r_index = torch.full((B * (1 + K),), rel_pc, dtype=torch.long)
    y = torch.cat([torch.ones(B), torch.zeros(B * K)]).float()
    return h_index, t_index, r_index, y


def neg_logsigmoid_loss(scores):
    pos = scores[..., 0]
    neg = scores[..., 1:]
    return -(F.logsigmoid(pos).mean() + F.logsigmoid(-neg).mean())


def _cmpnn_forward_lp(model, graph, h_index, t_index, r_index, device):
    """CMPNN masks (h,t) one-hop edges only when ``all_loss`` is not ``None``
    (see task.py)."""
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
            better = (probs[:, 1:] >= s_true.unsqueeze(1)).sum(dim=1)
            rank = better + 1
            hits1 += (rank <= 1).sum().item()
            hits3 += (rank <= 3).sum().item()
            hits5 += (rank <= 5).sum().item()
            rr_sum += (1.0 / rank.float()).sum().item()
            n_q += B
        all_y.append(y)
        all_p.append(torch.sigmoid(scores).flatten().detach().cpu().numpy())

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
def evaluate_full(model, graph, test_pos, test_neg, offP, offC, rel_pc,
                  batch_size, device, threshold):
    """Test evaluation: ranking metrics (primary) and optional threshold
    binarization (secondary)."""
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
            pL = int(pos_batch[j, 0])
            c_true = int(pos_batch[j, 1])
            h_global = offP + pL
            t_true_global = offC + c_true
            t_neg_global = offC + neg_batch[j].astype(np.int64)
            h_idx = np.full(1 + K, h_global, dtype=np.int64)
            t_idx = np.concatenate([[t_true_global], t_neg_global])
            h_list.append(h_idx)
            t_list.append(t_idx)

        h_index = torch.tensor(np.concatenate(h_list), dtype=torch.long)
        r_index = torch.full_like(h_index, rel_pc)
        t_index = torch.tensor(np.concatenate(t_list), dtype=torch.long)

        scores = _cmpnn_forward_lp(model, graph, h_index, t_index, r_index, device)
        probs = torch.sigmoid(scores).view(B, 1 + K)

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
            c_neg = neg_batch[ki]
            all_scores.append(probs_np[ki, 0])
            all_labels.append(1)
            all_pairs.append((int(pos_batch[ki, 0]), int(pos_batch[ki, 1])))
            for c_n in c_neg:
                all_scores.append(probs_np[ki, 1 + np.where(neg_batch[ki] == c_n)[0][0]])
                all_labels.append(0)
                all_pairs.append((int(pos_batch[ki, 0]), int(c_n)))

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
    top1_accuracy = h1

    pairs_arr = np.array(all_pairs)
    scores_df = pd.DataFrame({
        'paper_local': pairs_arr[:, 0], 'conf_local': pairs_arr[:, 1],
        'label': y_true, 'prob': y_proba,
    })

    return {
        'auc': auc, 'ap': ap,
        'hits1': h1, 'hits3': h3, 'hits5': h5, 'mrr': mrr,
        'top1_accuracy': top1_accuracy,
        'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc,
        'confusion': (TP, TN, FP, FN),
        'scores_df': scores_df,
    }


def run_model_DBLP_pc(raw_dir, shared_npz, neg_k, input_dim, hidden_dim,
                      num_layers, num_epochs, patience, batch_size, threshold,
                      save_postfix, base_seed, gpu, use_cpu, eval_only,
                      checkpoint_path):
    t0 = time.time()
    print('-> Preprocessing DBLP raw data for CMPNN (var2: area↔venue) ...')
    triplets, meta, splits = preprocess_dblp_var2(raw_dir, shared_npz, neg_k, base_seed)
    print(f'   done in {time.time() - t0:.1f}s')

    offP = meta['offP']
    offC = meta['offC']

    print(f"-> KG: {meta['num_entity']} entities, {meta['num_relation']} relations, "
          f"{len(triplets)} triplets")
    print(f"-> Splits: train_pos={meta['counts']['train_pos']}, "
          f"val_pos={meta['counts']['val_pos']}, test_pos={meta['counts']['test_pos']}")
    print(f"-> neg_k={meta['neg_k']} (same k sampled negatives per query for train/val/test)")
    print(f"-> layers={num_layers}")

    if use_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cuda:0')
    print(f'-> Device: {device}')

    edge_list = torch.as_tensor(triplets, dtype=torch.long)
    graph = torchdrug.data.Graph(edge_list, num_node=int(meta['num_entity']),
                                 num_relation=int(meta['num_relation']))
    graph = graph.to(device)
    rel_pc = int(meta['rel_pc'])

    train_ds = DBLP_PC_Query(splits['train_pos'], splits['train_neg'], offP, offC)
    val_ds = DBLP_PC_Query(splits['val_pos'], splits['val_neg'], offP, offC)
    train_loader = torch_data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                         collate_fn=lambda b: collate_query(b, rel_pc=rel_pc))
    val_loader = torch_data.DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                       collate_fn=lambda b: collate_query(b, rel_pc=rel_pc))

    if hidden_dim != input_dim:
        raise ValueError('For current CMPNN impl, set hidden_dim == input_dim '
                         '(boundary/message dim must match).')

    model = CMPNN(input_dim=input_dim,
                  hidden_dims=[hidden_dim] * num_layers,
                  message_func='distmult',
                  aggregate_func='pna',
                  short_cut=True, layer_norm=True, dependent=False,
                  set_boundary=True, remove_one_hop=True,
                  activation='relu', initialization='Query')
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'-> Model params: {n_params}')

    optimizer = torch.optim.Adam(model.parameters())
    os.makedirs('checkpoint', exist_ok=True)
    ckpt_path = f'checkpoint/checkpoint_{save_postfix}_seed{base_seed}.pt'
    early_stopping = EarlyStopping(patience=patience, verbose=False, save_path=ckpt_path)

    if not eval_only:
        for epoch in range(num_epochs):
            epoch_t0 = time.time()
            model.train()
            losses = []
            pbar = tqdm(train_loader, desc=f'Epoch {epoch} [train]')
            for h_index, t_index, r_index, _y in pbar:
                scores = _cmpnn_forward_lp(model, graph, h_index, t_index, r_index, device)
                B = _y.sum().int().item()
                K = scores.dim() // B - 1 if B > 0 else 0
                loss = neg_logsigmoid_loss(scores.view(B, 1 + K) if B > 0 else scores)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                pbar.set_postfix(loss=f'{np.mean(losses):.4f}')

            val_metrics = evaluate(model, graph, val_loader, device)
            val_loss = -val_metrics['ap']
            print(f'Epoch {epoch} | val_AP={val_metrics["ap"]:.4f} '
                  f'val_AUC={val_metrics["auc"]:.4f} | Time(s) {time.time() - epoch_t0:.1f}')
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print('Early stopping!')
                break
    else:
        print('-> Eval-only: loading checkpoint for causal diagnosis ...')

    model.load_state_dict(torch.load(ckpt_path))
    print('-> Testing best checkpoint ...')

    results = evaluate_full(model, graph, splits['test_pos'], splits['test_neg'],
                            offP, offC, rel_pc, batch_size, device, threshold)
    cm = results['confusion']

    print(f"-> Primary (ranking): predict tail by **argmax** over 1+K scores "
          f"(same spirit as NC argmax over class logits).")
    print(f"    Top-1 acc = Hits@1 = {results['hits1']:.4f}  |  MRR = {results['mrr']:.4f}")
    print(f"    AUC = {results['auc']:.4f}  AP = {results['ap']:.4f}")
    print(f"    Hits@3 = {results['hits3']:.4f}  Hits@5 = {results['hits5']:.4f}")
    print(f"  Secondary (sigmoid threshold {threshold}; optional if scores shift):")
    print(f"    Confusion: TP={cm[0]} TN={cm[1]} FP={cm[2]} FN={cm[3]}")
    print(f"    Precision = {results['precision']:.4f}  Recall = {results['recall']:.4f}")
    print(f"    F1 = {results['f1']:.4f}  Accuracy = {results['accuracy']:.4f}")

    results['scores_df'].to_csv(f'{save_postfix}_seed{base_seed}_scores.csv', index=False)
    return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='CMPNN-LP on DBLP paper-conf (var2: area↔venue)')
    ap.add_argument('--raw-dir', default='../MAGNN/data/raw/DBLP',
                    help='Path to MAGNN/data/raw/DBLP')
    ap.add_argument('--shared-npz',
                    default='../MAGNN/data/preprocessed/DBLP_shared_splits/DBLP_pc_shared_splits.npz',
                    help='Path to DBLP_pc_shared_splits.npz')
    ap.add_argument('--input-dim', type=int, default=32)
    ap.add_argument('--hidden-dim', type=int, default=32)
    ap.add_argument('--layers', type=int, default=6, help='Number of CMPNN GNN layers')
    ap.add_argument('--epoch', type=int, default=20)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--neg-k', type=int, default=19,
                    help='Same k sampled negatives per positive for train/val/test (capped at C-1)')
    ap.add_argument('--threshold', type=float, default=0.5,
                    help='Sigmoid threshold for secondary P/R/F1/acc (primary: AUC, AP, Hits@k, MRR)')
    ap.add_argument('--seeds', default='1566911444,20241017,20251017',
                    help='Comma-separated list of seeds')
    ap.add_argument('--save-postfix', default='DBLP_cmpnn_pc_var2')
    ap.add_argument('--gpu', type=int, default=0, help='GPU id (e.g. 1 for cuda:1)')
    ap.add_argument('--cpu', action='store_true',
                    help='Force CPU (use when CUDA fails or no GPU visible)')
    ap.add_argument('--eval-only', action='store_true',
                    help='Skip training; load checkpoint and run test only')
    ap.add_argument('--checkpoint', default=None, help='Checkpoint path for --eval-only')
    args = ap.parse_args()

    if args.eval_only and args.checkpoint is None:
        ap.error('--eval-only requires --checkpoint')

    seeds = [int(s) for s in args.seeds.split(',')]
    all_results = []

    for s in seeds:
        print('=' * 60)
        print(f'\n Seed {s}')
        set_seed(s)
        _ = run_model_DBLP_pc(
            raw_dir=args.raw_dir, shared_npz=args.shared_npz,
            neg_k=args.neg_k, input_dim=args.input_dim,
            hidden_dim=args.hidden_dim, num_layers=args.layers,
            num_epochs=args.epoch, patience=args.patience,
            batch_size=args.batch_size, threshold=args.threshold,
            save_postfix=f'{args.save_postfix}_seed{s}',
            base_seed=s, gpu=args.gpu, use_cpu=args.cpu,
            eval_only=args.eval_only, checkpoint_path=args.checkpoint,
        )
        all_results.append(_)

    print('\n Summary over seeds')
    print('(Primary: ranking / Top-1 = Hits@1; secondary: threshold metrics)')
    for s, st in zip(seeds, all_results):
        print(f"Seed {s}: AUC={st['auc']:.4f}  AP={st['ap']:.4f}  "
              f"Hits@1={st['hits1']:.4f}  MRR={st['mrr']:.4f}  "
              f"Hits@3={st['hits3']:.4f}  Hits@5={st['hits5']:.4f}  "
              f"Prec={st['precision']:.4f}  Rec={st['recall']:.4f}  "
              f"F1={st['f1']:.4f}  Acc={st['accuracy']:.4f}")

    print('\nFinal mean +/- std across all seeds:')
    for key in ['auc', 'ap', 'hits1', 'mrr', 'hits3', 'hits5',
                'precision', 'recall', 'f1', 'accuracy', 'top1_accuracy']:
        vals = [r.get(key, r.get('hits1', 0)) for r in all_results]
        print(f"  {key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
