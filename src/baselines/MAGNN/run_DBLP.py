import time
import argparse

import torch
import torch.nn.functional as F
import numpy as np

from utils.pytorchtools import EarlyStopping
from utils.data import load_DBLP_data
from utils.tools import index_generator, evaluate_results_nc, parse_minibatch
from model import MAGNN_nc_mb
import os 
from sklearn.metrics import accuracy_score, precision_score, recall_score
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Params
out_dim = 4 #areas
out_dim = 2 #venues/conferences
dropout_rate = 0.5
lr = 0.005
weight_decay = 0.001
etypes_list = [[0, 1], [0, 2, 3, 1], [0, 4, 5, 1]]

def run_model_DBLP(feats_type, hidden_dim, num_heads, attn_vec_dim, rnn_type,
                   num_epochs, patience, batch_size, neighbor_samples, repeat, save_postfix, data_prefix):
    adjlists, edge_metapath_indices_list, features_list, adjM, type_mask, labels, train_val_test_idx = load_DBLP_data(data_prefix)
    
    """Print detailed graph statistics"""
    print("\n=== Graph Statistics ===")
    
    # Node counts
    paper_count = np.sum(type_mask == 0)
    author_count = np.sum(type_mask == 1)
    term_count = np.sum(type_mask == 2)
    area_count = np.sum(type_mask == 3)
    
    print("\nNode Counts:")
    print(f"Papers: {paper_count:,}")
    print(f"Authors: {author_count:,}") 
    print(f"Terms: {term_count:,}")
    print(f"Areas: {area_count:,}")
    
    # Edge counts
    paper_author_edges = np.sum(adjM[:paper_count, paper_count:paper_count+author_count])
    paper_term_edges = np.sum(adjM[:paper_count, paper_count+author_count:paper_count+author_count+term_count])
    paper_area_edges = np.sum(adjM[:paper_count, paper_count+author_count+term_count:])
    
    print("\nEdge Counts:")
    print(f"Paper-Author: {paper_author_edges:,}")
    print(f"Paper-Term: {paper_term_edges:,}")
    print(f"Paper-Area: {paper_area_edges:,}")
    
    # Label distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    print("\nLabel Distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"Class {label}: {count:,} samples")

    #print(labels)
    unique_labels = {}
    new_label = 0
    '''for l in labels:
        if l not in unique_labels:
            unique_labels[l] = 0
    for ul in unique_labels:
        unique_labels[ul] = new_label
        new_label += 1
    for i in range(len(labels)):
        labels[i] = unique_labels[labels[i]]'''
    
    #print(edge_metapath_indices_list)
    #print(labels)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    features_list = [torch.FloatTensor(features).to(device) for features in features_list]
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1:
        in_dims = [features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(1, len(features_list)):
            features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2:
        in_dims = [features.shape[0] for features in features_list]
        in_dims[0] = features_list[0].shape[1]
        for i in range(1, len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse_coo_tensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse_coo_tensor(indices, values, torch.Size([dim, dim])).to(device)
    labels = torch.LongTensor(labels).to(device)
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)

    svm_macro_f1_lists = []
    svm_micro_f1_lists = []
    nmi_mean_list = []
    nmi_std_list = []
    ari_mean_list = []
    ari_std_list = []
    for _ in range(repeat):
        net = MAGNN_nc_mb(3, 6, etypes_list, in_dims, hidden_dim, out_dim, num_heads, attn_vec_dim, rnn_type, dropout_rate)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        # training loop
        net.train()
        if etypes_list[2][0] == 0:
            early_stopping = EarlyStopping(patience=patience, verbose=True, save_path='checkpoint/checkpoint_transformed{}.pt'.format(save_postfix))
        else:
            early_stopping = EarlyStopping(patience=patience, verbose=True, save_path='checkpoint/checkpoint_{}.pt'.format(save_postfix))
        dur1 = []
        dur2 = []
        dur3 = []
        train_idx_generator = index_generator(batch_size=batch_size, indices=train_idx)
        val_idx_generator = index_generator(batch_size=batch_size, indices=val_idx, shuffle=False)
        train_logits = []
        train_label_idx = []
        for epoch in range(num_epochs):
            t_start = time.time()
            # training
            net.train()
            for iteration in range(train_idx_generator.num_iterations()):
                # forward
                t0 = time.time()

                train_idx_batch = train_idx_generator.next()
                train_idx_batch.sort()
                train_g_list, train_indices_list, train_idx_batch_mapped_list = parse_minibatch(
                    adjlists, edge_metapath_indices_list, train_idx_batch, device, neighbor_samples)
                t1 = time.time()
                dur1.append(t1 - t0)

                logits, embeddings = net(
                    (train_g_list, features_list, type_mask, train_indices_list, train_idx_batch_mapped_list))
                logp = F.log_softmax(logits, 1)
                train_loss = F.nll_loss(logp, labels[train_idx_batch])
                train_logits.append(logits)
                #print(list(train_idx_batch))
                train_label_idx = train_label_idx + list(train_idx_batch)

                #print(np.argmax(emebeddings.detach()))
                #acc = accuracy_score(labels[train_idx_batch].cpu().numpy(), np.argmax(torch.cat([embeddings], 0).cpu().numpy(), axis=1))

                #exit()

                t2 = time.time()
                dur2.append(t2 - t1)

                # autograd
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                t3 = time.time()
                dur3.append(t3 - t2)

                # print training info
                if iteration % 50 == 0:
                    train_pred = torch.cat(train_logits, 0)
                    scores = []
                    #train_label = np.cat(train_label_idx, 0)
                    #print(torch.argmax(train_pred, dim=1).shape, len(train_label_idx))
                    for y_label, train_pred in zip(train_label_idx, torch.argmax(train_pred, dim=1)):
                        if(labels[y_label].item() == train_pred.item()):
                            scores.append(1)
                        else:
                            scores.append(0)
                    print(
                        'Epoch {:05d} | Iteration {:05d} | Train_Loss {:.4f} | Train_Acc {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f} | Time3(s) {:.4f}'.format(
                            epoch, iteration, train_loss.item(), sum(scores)/len(scores), np.mean(dur1), np.mean(dur2), np.mean(dur3)))

            # validation
            net.eval()
            val_logp = []
            val_logits = []
            val_label_idx = []
            with torch.no_grad():
                for iteration in range(val_idx_generator.num_iterations()):
                    # forward
                    val_idx_batch = val_idx_generator.next()
                    val_g_list, val_indices_list, val_idx_batch_mapped_list = parse_minibatch(
                        adjlists, edge_metapath_indices_list, val_idx_batch, device, neighbor_samples)
                    logits, embeddings = net(
                        (val_g_list, features_list, type_mask, val_indices_list, val_idx_batch_mapped_list))
                    logp = F.log_softmax(logits, 1)
                    val_logp.append(logp)
                    val_logits.append(logits)
                    #val_label_idx = val_label_idx + list(val_idx_batch)
                val_logits = torch.cat(val_logits, 0)
                '''scores = []
                for y_label, val_pred in zip(val_label_idx, torch.argmax(val_logits, dim=1)):
                    if(labels[y_label].item() == val_pred.item()):
                        scores.append(1)
                    else:
                        scores.append(0)
                print("Val Acc:", sum(scores)/len(scores))'''
                acc = accuracy_score(labels[val_idx].cpu().numpy(), np.argmax(val_logits.cpu().numpy(), axis=1))
                val_loss = F.nll_loss(torch.cat(val_logp, 0), labels[val_idx])
            t_end = time.time()
            # print validation info
            print('Epoch {:05d} | Val_Loss {:.4f} | Val_Acc {:.4f} | Time(s) {:.4f}'.format(
                epoch, val_loss.item(), acc, t_end - t_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        # testing with evaluate_results_nc
        test_idx_generator = index_generator(batch_size=batch_size, indices=test_idx, shuffle=False)
        if etypes_list[2][0] == 0:
            net.load_state_dict(torch.load('checkpoint/checkpoint_transformed{}.pt'.format(save_postfix)))
        else:
            net.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(save_postfix)))
        net.eval()
        test_logits = []
        test_embeddings = []
        val_logits = []
        train_logits = []
        with torch.no_grad():
            #Train
            train_idx_generator = index_generator(batch_size=batch_size, indices=train_idx, shuffle=False)
            for iteration in range(train_idx_generator.num_iterations()):
                # forward
                train_idx_batch = train_idx_generator.next()
                train_g_list, train_indices_list, train_idx_batch_mapped_list = parse_minibatch(
                    adjlists, edge_metapath_indices_list, train_idx_batch, device, neighbor_samples)
                logits, embeddings = net(
                    (train_g_list, features_list, type_mask, train_indices_list, train_idx_batch_mapped_list))
                train_logits.append(logits)
            train_logits = torch.cat(train_logits, 0)
            train_preds = np.argmax(train_logits.cpu().numpy(), axis=1)
            true_train_labels = labels[train_idx].cpu().numpy()

            train_acc = accuracy_score(true_train_labels, train_preds)
            train_prec = precision_score(true_train_labels, train_preds, average='macro', zero_division=0)
            train_rec = recall_score(true_train_labels, train_preds, average='macro', zero_division=0)

            print(f"Train_Acc: {train_acc:.4f} | Train_Precision: {train_prec:.4f} | Train_Recall: {train_rec:.4f}")
            # acc = accuracy_score(labels[train_idx].cpu().numpy(), np.argmax(train_logits.cpu().numpy(), axis=1))
            # print("Train_Acc", acc)
            #Val
            for iteration in range(val_idx_generator.num_iterations()):
                # forward
                val_idx_batch = val_idx_generator.next()
                val_g_list, val_indices_list, val_idx_batch_mapped_list = parse_minibatch(
                    adjlists, edge_metapath_indices_list, val_idx_batch, device, neighbor_samples)
                logits, embeddings = net(
                    (val_g_list, features_list, type_mask, val_indices_list, val_idx_batch_mapped_list))
                val_logits.append(logits)
            val_logits = torch.cat(val_logits, 0)
            val_preds = np.argmax(val_logits.cpu().numpy(), axis=1)
            true_val_labels = labels[val_idx].cpu().numpy()

            val_acc = accuracy_score(true_val_labels, val_preds)
            val_prec = precision_score(true_val_labels, val_preds, average='macro', zero_division=0)
            val_rec = recall_score(true_val_labels, val_preds, average='macro', zero_division=0)

            print(f"Val_Acc: {val_acc:.4f} | Val_Precision: {val_prec:.4f} | Val_Recall: {val_rec:.4f}")
            # acc = accuracy_score(labels[val_idx].cpu().numpy(), np.argmax(val_logits.cpu().numpy(), axis=1))
            # print("Val_Acc", acc)
            #Test
            for iteration in range(test_idx_generator.num_iterations()):
                # forward
                test_idx_batch = test_idx_generator.next()
                test_g_list, test_indices_list, test_idx_batch_mapped_list = parse_minibatch(adjlists,
                                                                                             edge_metapath_indices_list,
                                                                                             test_idx_batch,
                                                                                             device, neighbor_samples)
                logits, embeddings = net((test_g_list, features_list, type_mask, test_indices_list, test_idx_batch_mapped_list))
                test_logits.append(logits)
                test_embeddings.append(embeddings)
            test_logits = torch.cat(test_logits, 0)
            test_embeddings = torch.cat(test_embeddings,0)
            test_preds = np.argmax(test_logits.cpu().numpy(), axis=1)
            true_test_labels = labels[test_idx].cpu().numpy()

            test_acc = accuracy_score(true_test_labels, test_preds)
            test_prec = precision_score(true_test_labels, test_preds, average='macro', zero_division=0)
            test_rec = recall_score(true_test_labels, test_preds, average='macro', zero_division=0)

            print(f"Test_Acc: {test_acc:.4f} | Test_Precision: {test_prec:.4f} | Test_Recall: {test_rec:.4f}")
            # acc = accuracy_score(labels[test_idx].cpu().numpy(), np.argmax(test_logits.cpu().numpy(), axis=1))
            # print("Test_Acc", acc)
            svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std = evaluate_results_nc(
                test_embeddings.cpu().numpy(), labels[test_idx].cpu().numpy(), num_classes=out_dim)
        svm_macro_f1_lists.append(svm_macro_f1_list)
        svm_micro_f1_lists.append(svm_micro_f1_list)
        nmi_mean_list.append(nmi_mean)
        nmi_std_list.append(nmi_std)
        ari_mean_list.append(ari_mean)
        ari_std_list.append(ari_std)

    # print out a summary of the evaluations
    svm_macro_f1_lists = np.transpose(np.array(svm_macro_f1_lists), (1, 0, 2))
    svm_micro_f1_lists = np.transpose(np.array(svm_micro_f1_lists), (1, 0, 2))
    nmi_mean_list = np.array(nmi_mean_list)
    nmi_std_list = np.array(nmi_std_list)
    ari_mean_list = np.array(ari_mean_list)
    ari_std_list = np.array(ari_std_list)
    print('----------------------------------------------------------------')
    print('SVM tests summary')
    print('Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
        macro_f1[:, 0].mean(), macro_f1[:, 1].mean(), train_size) for macro_f1, train_size in
        zip(svm_macro_f1_lists, [0.8, 0.6, 0.4, 0.2])]))
    print('Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
        micro_f1[:, 0].mean(), micro_f1[:, 1].mean(), train_size) for micro_f1, train_size in
        zip(svm_micro_f1_lists, [0.8, 0.6, 0.4, 0.2])]))
    print('K-means tests summary')
    print('NMI: {:.6f}~{:.6f}'.format(nmi_mean_list.mean(), nmi_std_list.mean()))
    print('ARI: {:.6f}~{:.6f}'.format(ari_mean_list.mean(), ari_std_list.mean()))

    
    return [
        np.argmax(train_logits.cpu().numpy(), axis=1), 
        np.argmax(val_logits.cpu().numpy(), axis=1), 
        np.argmax(test_logits.cpu().numpy(), axis=1)] # return classifications for robustness calculation


def robustness(pred1, pred2):
    '''assert len(pred1) == len(pred2)
    matches = np.sum(pred1==pred2)
    return (matches / (len(pred1))) * 100'''
    train_robust_score = accuracy_score(pred1[0], pred2[0])
    val_robust_score = accuracy_score(pred1[1], pred2[1])
    test_robust_score = accuracy_score(pred1[2], pred2[2])
    return [train_robust_score, val_robust_score, test_robust_score]

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
    ap.add_argument('--feats-type', type=int, default=2,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2.')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--attn-vec-dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
    ap.add_argument('--rnn-type', default='RotatE0', help='Type of the aggregator. Default is RotatE0.')
    ap.add_argument('--epoch', type=int, default=100, help='Number of epochs. Default is 100.')
    ap.add_argument('--patience', type=int, default=5, help='Patience. Default is 5.')
    ap.add_argument('--batch-size', type=int, default=8, help='Batch size. Default is 8.')
    ap.add_argument('--samples', type=int, default=100, help='Number of neighbors sampled. Default is 100.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--save-postfix', default='DBLP', help='Postfix for the saved model and result. Default is DBLP.')

    args = ap.parse_args()

    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')

    labels = []
    for i in range(2):
        data_prefix = 'data/preprocessed/DBLP_preprocessed_area' if i == 0 else 'data/preprocessed/DBLP_preprocessed_area_transformed'
        etypes_list = [[0,1],[2,3],[4,5]] if i == 0 else [[0,1],[2,3],[0,4,5,1]]
        save_postfix = 'dblp_preprocessed' if i == 0 else 'dblp_preprocessed_transformed'
        print('----------------------------------------------------------------')
        print("Training/testing using dataset {}".format(data_prefix))
        print('----------------------------------------------------------------')
        labels.append(run_model_DBLP(args.feats_type, args.hidden_dim, args.num_heads, args.attn_vec_dim, args.rnn_type,
                    args.epoch, args.patience, args.batch_size, args.samples, args.repeat, save_postfix, data_prefix))

    print('----------------------------------------------------------------')
    print("Evaluating Robustness")
    print('----------------------------------------------------------------')
    robust = robustness(labels[0], labels[1])
    print('Train_Robust {:.4f} | Val_Robust {:.4f} | Test_Robust {:.4f}'.format(
        robust[0], robust[1], robust[2]))