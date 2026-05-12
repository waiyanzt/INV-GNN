"""CMPNN: IMDB movie to director LP (v1, v3)."""
import os, sys, time, argparse, random
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
from imdb_md_data import preprocess_imdb_md
from utils.pytorchtools import EarlyStopping

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try: torch.use_deterministic_algorithms(True)
    except Exception: pass

class IMDB_MD_Query(torch_data.Dataset):
    def __init__(self, pos, neg, off_d):
        self.m = pos[:, 0].astype(np.int64)
        self.d_true = pos[:, 1].astype(np.int64)
        self.d_neg = neg.astype(np.int64)
        self.off_d = off_d
    def __len__(self): return len(self.m)
    def __getitem__(self, i):
        h = int(self.m[i])
        t_true = self.off_d + self.d_true[i]
        t_neg = self.off_d + self.d_neg[i]
        return h, t_true, t_neg

def collate_query_md(batch, rel_md):
    """(B, 1+K) tensors — required when graph.num_relation > 0 (CMPNN KG forward)."""
    b = list(zip(*batch))
    h = torch.tensor(b[0], dtype=torch.long)
    t_true = torch.tensor(b[1], dtype=torch.long)
    t_neg = torch.stack([torch.tensor(x, dtype=torch.long) for x in b[2]])
    B, K = t_neg.size()
    h_index = h.unsqueeze(1).expand(-1, 1 + K)
    t_index = torch.cat([t_true.unsqueeze(1), t_neg], dim=1)
    r_index = torch.full_like(h_index, rel_md, dtype=torch.long)
    y = torch.zeros_like(h_index, dtype=torch.float)
    y[:, 0] = 1.0
    return h_index, t_index, r_index, y

def neg_logsigmoid_loss(scores):
    pos = scores[..., 0]; neg = scores[..., 1:]
    return -(F.logsigmoid(pos).mean() + F.logsigmoid(-neg).mean())

def _cmpnn_forward_lp(model, graph, h_index, t_index, r_index, device):
    return model(graph, h_index.to(device), t_index.to(device),
                 r_index.to(device), all_loss=None, metric=None)

@torch.no_grad()
def evaluate(model, graph, loader, device):
    model.eval()
    all_y, all_p = [], []
    hits1 = hits3 = hits5 = 0; rr_sum = 0.0; n_q = 0
    for h_index, t_index, r_index, y in loader:
        scores = _cmpnn_forward_lp(model, graph, h_index, t_index, r_index, device)
        probs = torch.sigmoid(scores)
        B, L = probs.shape
        K = L - 1
        if B > 0 and K > 0:
            s_true = probs[:, 0]
            rank = (probs[:, 1:] >= s_true.unsqueeze(1)).sum(dim=1) + 1
            hits1 += (rank <= 1).sum().item()
            hits3 += (rank <= 3).sum().item()
            hits5 += (rank <= 5).sum().item()
            rr_sum += (1.0 / rank.float()).sum().item()
            n_q += B
        all_y.append(y.flatten())
        all_p.append(probs.flatten().detach().cpu().numpy())
    y_all = torch.cat(all_y).numpy(); p_all = np.concatenate(all_p)
    return {'auc': roc_auc_score(y_all, p_all), 'ap': average_precision_score(y_all, p_all),
            'hits1': hits1/max(n_q,1), 'hits3': hits3/max(n_q,1),
            'hits5': hits5/max(n_q,1), 'mrr': rr_sum/max(n_q,1)}

@torch.no_grad()
def evaluate_full(model, graph, test_pos, test_neg, off_d, rel_md, batch_size, device, threshold):
    model.eval(); K = test_neg.shape[1]
    all_scores, all_labels, all_pairs = [], [], []
    hits1 = hits3 = hits5 = 0; rr_sum = 0.0; n_q = 0
    for start in range(0, len(test_pos), batch_size):
        pos_batch = test_pos[start:start+batch_size]
        neg_batch = test_neg[start:start+batch_size]; B = len(pos_batch)
        h_list, t_list = [], []
        for j in range(B):
            m = int(pos_batch[j,0]); d_true = int(pos_batch[j,1])
            h_list.append(np.full(1+K, m, dtype=np.int64))
            t_list.append(np.concatenate([[off_d+d_true], off_d+neg_batch[j].astype(np.int64)]))
        h_index = torch.tensor(np.stack(h_list), dtype=torch.long)
        t_index = torch.tensor(np.stack(t_list), dtype=torch.long)
        r_index = torch.full_like(h_index, rel_md)
        scores = _cmpnn_forward_lp(model, graph, h_index, t_index, r_index, device)
        probs = torch.sigmoid(scores)
        rank = (probs[:,1:] >= probs[:,0:1]).sum(dim=1) + 1
        hits1 += (rank<=1).sum().item(); hits3 += (rank<=3).sum().item()
        hits5 += (rank<=5).sum().item(); rr_sum += (1.0/rank.float()).sum().item(); n_q += B
        probs_np = probs.cpu().numpy()
        for ki in range(B):
            all_scores.append(probs_np[ki,0]); all_labels.append(1)
            all_pairs.append((int(pos_batch[ki,0]), int(pos_batch[ki,1])))
            for jj in range(K):
                all_scores.append(probs_np[ki,1+jj]); all_labels.append(0)
                all_pairs.append((int(pos_batch[ki,0]), int(neg_batch[ki,jj])))
    y_true = np.array(all_labels); y_proba = np.array(all_scores)
    auc = roc_auc_score(y_true, y_proba); ap = average_precision_score(y_true, y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    TP = int(((y_pred==1)&(y_true==1)).sum()); TN = int(((y_pred==0)&(y_true==0)).sum())
    FP = int(((y_pred==1)&(y_true==0)).sum()); FN = int(((y_pred==0)&(y_true==1)).sum())
    prec = TP/max(TP+FP,1); rec = TP/max(TP+FN,1)
    f1 = 2*prec*rec/max(prec+rec,1e-9); acc = (TP+TN)/max(TP+TN+FP+FN,1)
    pairs_arr = np.array(all_pairs)
    scores_df = pd.DataFrame({'movie_local': pairs_arr[:,0], 'director_local': pairs_arr[:,1],
                               'label': y_true, 'prob': y_proba})
    return {'auc': auc, 'ap': ap, 'hits1': hits1/max(n_q,1), 'hits3': hits3/max(n_q,1),
            'hits5': hits5/max(n_q,1), 'mrr': rr_sum/max(n_q,1), 'top1_accuracy': hits1/max(n_q,1),
            'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc,
            'confusion': (TP,TN,FP,FN), 'scores_df': scores_df}

def run_model_imdb_md(csv_path, shared_npz, variant, neg_k, input_dim,
                      hidden_dim, num_layers, num_epochs, patience, batch_size,
                      threshold, save_postfix, base_seed, gpu, use_cpu,
                      eval_only, checkpoint_path):
    t0 = time.time()
    triplets, meta, splits = preprocess_imdb_md(csv_path, shared_npz, variant, neg_k, base_seed)
    print(f'-> Preprocess {time.time()-t0:.1f}s | entities={meta["num_entity"]} '
          f'rel={meta["num_relation"]} triplets={len(triplets)}')
    off_d = meta['off_d']; rel_md = meta['rel_md']
    print(f'-> off_d={off_d} rel_md={rel_md} variant={variant} neg_k={meta.get("neg_k",neg_k)}')
    print(f'  {meta.get("sizes", {})}')
    if use_cpu: device = torch.device('cpu')
    else: device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cuda:0')
    graph = torchdrug.data.Graph(torch.as_tensor(triplets, dtype=torch.long),
                                 num_node=int(meta['num_entity']),
                                 num_relation=int(meta['num_relation']))
    graph = graph.to(device)
    train_ds = IMDB_MD_Query(splits['train_pos'], splits['train_neg'], off_d)
    val_ds = IMDB_MD_Query(splits['val_pos'], splits['val_neg'], off_d)
    collate_fn = lambda b: collate_query_md(b, rel_md=rel_md)
    train_loader = torch_data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = torch_data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    if hidden_dim != input_dim:
        raise ValueError('hidden_dim must equal input_dim')
    model = CMPNN(input_dim=input_dim, hidden_dims=[hidden_dim]*num_layers,
                  num_relation=int(meta['num_relation']),
                  message_func='distmult', aggregate_func='pna',
                  short_cut=True, layer_norm=True, dependent=False,
                  set_boundary=True, remove_one_hop=True,
                  activation='relu', initialization='Query')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    os.makedirs('checkpoint', exist_ok=True)
    ckpt_path = f'checkpoint/checkpoint_{save_postfix}.pt'
    early_stopping = EarlyStopping(patience=patience, verbose=False, save_path=ckpt_path)
    train_wall_sec = 0.0
    epochs_ran = 0
    if eval_only:
        print('-> eval-only, skip training')
    else:
        train_t0 = time.perf_counter()
        for epoch in range(num_epochs):
            model.train(); losses = []
            for h_index, t_index, r_index, _y in tqdm(train_loader, desc=f'Epoch {epoch}'):
                scores = _cmpnn_forward_lp(model, graph, h_index, t_index, r_index, device)
                loss = neg_logsigmoid_loss(scores)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                losses.append(loss.item())
            vm = evaluate(model, graph, val_loader, device)
            print(f'Epoch {epoch} loss={np.mean(losses):.4f} val_AP={vm["ap"]:.4f} val_AUC={vm["auc"]:.4f}')
            early_stopping(-vm['ap'], model)
            epochs_ran = epoch + 1
            if early_stopping.early_stop: print('Early stopping'); break
        train_wall_sec = float(time.perf_counter() - train_t0)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'Missing checkpoint: {ckpt_path}')
    model.load_state_dict(torch.load(ckpt_path))
    print('-> Testing best checkpoint ...')
    results = evaluate_full(model, graph, splits['test_pos'], splits['test_neg'],
                            off_d, rel_md, batch_size, device, threshold)
    print(f"    Hits@1={results['hits1']:.4f} MRR={results['mrr']:.4f} "
          f"AUC={results['auc']:.4f} AP={results['ap']:.4f}")
    print(f"    Hits@3={results['hits3']:.4f} Hits@5={results['hits5']:.4f}")
    print(f"    P/R/F1/Acc @ {threshold}: {results['precision']:.4f}/"
          f"{results['recall']:.4f}/{results['f1']:.4f}/{results['accuracy']:.4f}")
    print(f"    Train wall (s)={train_wall_sec:.2f}  Epochs={epochs_ran}")
    results['scores_df'].to_csv(f'{save_postfix}_scores.csv', index=False)
    out = {k: v for k, v in results.items() if k != 'scores_df'}
    out['train_wall_sec'] = train_wall_sec
    out['epochs_ran'] = float(epochs_ran)
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='CMPNN IMDB movie to director (v1, v3)')
    ap.add_argument('--csv', default='../MAGNN/data/raw/IMDB/movie_metadata.csv')
    ap.add_argument('--shared-npz', default='IMDB_md_shared_splits.npz')
    ap.add_argument('--variant', default='v1')
    ap.add_argument('--input-dim', type=int, default=32)
    ap.add_argument('--hidden-dim', type=int, default=32)
    ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--epoch', type=int, default=20)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--neg-k', type=int, default=19)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--seeds', default='1566911444,20241017,20251017')
    ap.add_argument('--save-postfix', default='IMDB_cmpnn_md_v1')
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
        print('=' * 60); print(f'\n Seed {s}')
        set_seed(s)
        _ = run_model_imdb_md(
            csv_path=args.csv, shared_npz=args.shared_npz,
            variant=args.variant, neg_k=args.neg_k,
            input_dim=args.input_dim, hidden_dim=args.hidden_dim,
            num_layers=args.layers, num_epochs=args.epoch,
            patience=args.patience, batch_size=args.batch_size,
            threshold=args.threshold,
            save_postfix=f'{args.save_postfix}_seed{s}',
            base_seed=s, gpu=args.gpu, use_cpu=args.cpu,
            eval_only=args.eval_only, checkpoint_path=args.checkpoint)
        all_results.append(_)
    print('\n Summary over seeds\n')
    for s, st in zip(seeds, all_results):
        print(f"Seed {s}: AUC={st['auc']:.4f} AP={st['ap']:.4f} "
              f"Hits@1={st['hits1']:.4f} MRR={st['mrr']:.4f} "
              f"H3={st['hits3']:.4f} H5={st['hits5']:.4f} Acc={st['accuracy']:.4f} "
              f"train_s={st['train_wall_sec']:.1f} epochs={int(st['epochs_ran'])}")
    print('\nFinal mean +/- std across all seeds:')
    for key in ['auc','ap','hits1','mrr','hits3','hits5','precision','recall','f1','accuracy',
                'train_wall_sec', 'epochs_ran']:
        vals = [r[key] for r in all_results]
        print(f"  {key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
