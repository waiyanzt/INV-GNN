import gc
import os
import random
from collections import defaultdict

import dgl
import dgl.function as fn

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm


VARIANT_EDGE_TYPES = {
    1: [('M', 'A'), ('M', 'D'), ('M', 'L')],
    2: [('L', 'A'), ('L', 'D'), ('M', 'L')],
    3: [('L', 'A'), ('M', 'D'), ('M', 'L')],
    4: [('M', 'A'), ('L', 'D'), ('M', 'L')],
}


def _enumerate_propagation_keys(edges_undirected, num_hops, tgt_type):
    """Mirror hg_propagate_feat_dgl key naming for a given undirected edge set.

    Returns the set of keys that would land on tgt_type after `num_hops` of
    propagation: keys are prepended with destination type each hop, on the
    final hop only target-type destinations are kept, and non-target nodes
    drop keys of length <= current hop after each step.
    """
    adj = defaultdict(set)
    all_types = {tgt_type}
    for s, d in edges_undirected:
        adj[s].add(d)
        adj[d].add(s)
        all_types.add(s)
        all_types.add(d)

    keys_at = {nt: {nt} for nt in all_types}
    for hop in range(1, num_hops + 1):
        new_keys = defaultdict(set)
        for src in all_types:
            for dst in adj[src]:
                for k in keys_at[src]:
                    if len(k) != hop:
                        continue
                    if hop == num_hops and dst != tgt_type:
                        continue
                    new_keys[dst].add(dst + k)
        for nt, ks in new_keys.items():
            keys_at[nt] |= ks
        for nt in all_types:
            if nt == tgt_type:
                continue
            keys_at[nt] = {k for k in keys_at[nt] if len(k) > hop}
    return keys_at[tgt_type]


def universal_allowed_metapath_keys(num_hops, tgt_type='M'):
    """Union of metapath keys across IMDB variants 1..4 — the legal v5 set."""
    allowed = set()
    for v in (1, 2, 3, 4):
        allowed |= _enumerate_propagation_keys(VARIANT_EDGE_TYPES[v], num_hops, tgt_type)
    return allowed

import warnings
warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")
warnings.filterwarnings("ignore", message="Setting attributes on ParameterDict is not supported.")


def set_random_seed(seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def evaluator(gt, pred):
    gt = gt.cpu().squeeze()
    pred = pred.cpu().squeeze()
    return f1_score(gt, pred, average='micro'), f1_score(gt, pred, average='macro')


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def hg_propagate_feat_dgl(g, tgt_type, num_hops, max_length, extra_metapath, echo=False):
    for hop in range(1, max_length):
        reserve_heads = [ele[:hop] for ele in extra_metapath if len(ele) > hop]
        for etype in g.etypes:
            stype, _, dtype = g.to_canonical_etype(etype)
            for k in list(g.nodes[stype].data.keys()):
                if len(k) == hop:
                    current_dst_name = f'{dtype}{k}'
                    if (hop == num_hops and dtype != tgt_type and k not in reserve_heads) \
                      or (hop > num_hops and k not in reserve_heads):
                        continue
                    if echo:
                        print(k, etype, current_dst_name)
                    g[etype].update_all(
                        fn.copy_u(k, 'm'),
                        fn.mean('m', current_dst_name), etype=etype)

        for ntype in g.ntypes:
            if ntype == tgt_type:
                continue
            removes = []
            for k in g.nodes[ntype].data.keys():
                if len(k) <= hop:
                    removes.append(k)
            for k in removes:
                g.nodes[ntype].data.pop(k)
            if echo and len(removes):
                print('remove', removes)
        gc.collect()

        if echo:
            print(f'-- hop={hop} ---')
            for ntype in g.ntypes:
                for k, v in g.nodes[ntype].data.items():
                    print(f'{ntype} {k} {v.shape}', v[:, -1].max(), v[:, -1].mean())
            print(f'------\n')
    return g


def check_acc(preds_dict, condition, init_labels, train_nid, val_nid, test_nid,
              show_test=True, loss_type='ce'):
    mask_train, mask_val, mask_test = [], [], []
    remove_label_keys = []
    k = list(preds_dict.keys())[0]
    v = preds_dict[k]
    if loss_type == 'ce':
        na, nb, nc = len(train_nid), len(val_nid), len(test_nid)
    elif loss_type == 'bce':
        na, nb, nc = (len(train_nid) * v.size(1),
                      len(val_nid) * v.size(1),
                      len(test_nid) * v.size(1))

    for k, v in preds_dict.items():
        if loss_type == 'ce':
            pred = v.argmax(1)
        elif loss_type == 'bce':
            pred = (v > 0).int()

        a = pred[train_nid] == init_labels[train_nid]
        b = pred[val_nid] == init_labels[val_nid]
        c = pred[test_nid] == init_labels[test_nid]
        ra, rb, rc = a.sum() / na, b.sum() / nb, c.sum() / nc

        if loss_type == 'ce':
            vv = torch.log(v / (v.sum(1, keepdim=True) + 1e-6) + 1e-6)
            la = F.nll_loss(vv[train_nid], init_labels[train_nid])
            lb = F.nll_loss(vv[val_nid], init_labels[val_nid])
            lc = F.nll_loss(vv[test_nid], init_labels[test_nid])
        else:
            vv = (v / 2. + 0.5).clamp(1e-6, 1 - 1e-6)
            la = F.binary_cross_entropy(vv[train_nid], init_labels[train_nid].float())
            lb = F.binary_cross_entropy(vv[val_nid], init_labels[val_nid].float())
            lc = F.binary_cross_entropy(vv[test_nid], init_labels[test_nid].float())

        if condition(ra, rb, rc, k):
            mask_train.append(a)
            mask_val.append(b)
            mask_test.append(c)
        else:
            remove_label_keys.append(k)
        if show_test:
            print(k, ra, rb, rc, la, lb, lc,
                  (ra / rb - 1) * 100, (ra / rc - 1) * 100,
                  (1 - la / lb) * 100, (1 - la / lc) * 100)
        else:
            print(k, ra, rb, la, lb, (ra / rb - 1) * 100, (1 - la / lb) * 100)
    print(set(list(preds_dict.keys())) - set(remove_label_keys))

    print((torch.stack(mask_train, dim=0).sum(0) > 0).sum() / na)
    print((torch.stack(mask_val, dim=0).sum(0) > 0).sum() / nb)
    if show_test:
        print((torch.stack(mask_test, dim=0).sum(0) > 0).sum() / nc)


def train(model, feats, label_feats, labels_cuda, loss_fcn, optimizer,
          train_loader, evaluator, mask=None, scalar=None):
    model.train()
    device = labels_cuda.device
    total_loss = 0
    iter_num = 0
    y_true, y_pred = [], []

    for batch in train_loader:
        if isinstance(feats, list):
            batch_feats = [x[batch].to(device) for x in feats]
        elif isinstance(feats, dict):
            batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
        else:
            assert 0
        batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}
        if mask is not None:
            batch_mask = {k: x[batch].to(device) for k, x in mask.items()}
        else:
            batch_mask = None
        batch_y = labels_cuda[batch]

        optimizer.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                output_att = model(batch, batch_feats, batch_labels_feats, batch_mask)
                loss_train = loss_fcn(output_att, batch_y)
            scalar.scale(loss_train).backward()
            scalar.step(optimizer)
            scalar.update()
        else:
            output_att = model(batch, batch_feats, batch_labels_feats, batch_mask)
            loss_train = loss_fcn(output_att, batch_y)
            loss_train.backward()
            optimizer.step()

        y_true.append(batch_y.cpu().to(torch.long))
        if isinstance(loss_fcn, nn.BCEWithLogitsLoss):
            y_pred.append((output_att.data.cpu() > 0.).int())
        else:
            y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss_train.item()
        iter_num += 1
    loss = total_loss / iter_num
    acc = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
    return loss, acc
