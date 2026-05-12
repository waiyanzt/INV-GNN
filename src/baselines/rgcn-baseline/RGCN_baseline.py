import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from model_RGCN import RGCN
from utils import load_data_RGCN, accuracy, knn_classifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def robustness(preds1, preds2):
    """
    preds1 and preds2 are tuples/lists of numpy arrays:
      ( y_train_pred_1, y_val_pred_1, y_test_pred_1 )
      ( y_train_pred_2, y_val_pred_2, y_test_pred_2 )
    Returns three floats: [ train_acc_overlap, val_acc_overlap, test_acc_overlap ].
    """
    y1_train, y1_val, y1_test = preds1
    y2_train, y2_val, y2_test = preds2

    train_robust = accuracy_score(y1_train, y2_train)
    val_robust   = accuracy_score(y1_val,   y2_val)
    test_robust  = accuracy_score(y1_test,  y2_test)

    return [train_robust, val_robust, test_robust]

def train_and_eval(dataset_name, args):
    print(f"\n================== Running on {dataset_name} ==================\n")
    pyg_data = load_data_RGCN(dataset_name)
    labels = pyg_data.y 

    print("\n=== Graph Statistics ===")
    print("Node Counts:")
    for k, v in pyg_data.node_counts.items():
        print(f"{k.capitalize()}s: {v}")
    
    print("\nEdge Counts:")
    for k, v in pyg_data.edge_counts.items():
        name = k.replace('_', '-').capitalize()
        print(f"{name}: {v}")
    
    print("\nLabel Distribution:")
    label_counts = getattr(pyg_data, 'label_counts', {})
    for label in sorted(label_counts):
        print(f"Class {label}: {label_counts[label]} samples")


    model = RGCN(
        num_nodes=pyg_data.x.size(1),
        num_relations=pyg_data.edge_type.max().item() + 1,
        hidden_dim=args.hidden_dim,
        num_classes=labels.unique().size(0)
    )

    # if args.cuda:
    #     model.cuda()
    #     pyg_data = pyg_data.cuda()
    #     idx_train = pyg_data.train_mask
    #     idx_val = pyg_data.val_mask
    #     idx_test = pyg_data.test_mask
    #     labels = labels.cuda()


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_loss_val = float('inf')
    patience = args.patience
    counter = 0
    idx_train = pyg_data.train_mask
    idx_val = pyg_data.val_mask
    idx_test = pyg_data.test_mask
    best_val_acc = 0.0
    best_val_epoch = -1


    for epoch in range(args.epochs):
        model.train()
        output = model(pyg_data)
        loss_train = criterion(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            output = model(pyg_data)
            loss_val = criterion(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            if acc_val > best_val_acc:
                best_val_acc = acc_val
                best_val_epoch = epoch + 1  # +1 to match printed epoch


            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val))

            if loss_val < best_loss_val:
                best_loss_val = loss_val
                torch.save(model.state_dict(), f'best_model_{dataset_name}.pth')
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break
    
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f} at Epoch {best_val_epoch}")

    # Load best model weights before final evaluation
    model.load_state_dict(torch.load(f"best_model_{dataset_name}.pth"))
    model.eval()
    output = model(pyg_data)

    # Gather final predictions (as raw class‐indices) on each split
    y_train_pred = output[idx_train].argmax(dim=1).cpu().numpy()
    y_val_pred   = output[idx_val].argmax(dim=1).cpu().numpy()
    y_test_pred  = output[idx_test].argmax(dim=1).cpu().numpy()

    # Compute metrics on test
    acc       = accuracy(output[idx_test], labels[idx_test])
    precision = precision_score(
        labels[idx_test].cpu().numpy(), y_test_pred,
        average='macro', zero_division=0
    )
    recall    = recall_score(
        labels[idx_test].cpu().numpy(), y_test_pred,
        average='macro', zero_division=0
    )
    f1        = f1_score(
        labels[idx_test].cpu().numpy(), y_test_pred,
        average='macro', zero_division=0
    )

    print(f"\n=== Final Evaluation on {dataset_name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Macro-F1: {f1:.4f}\n")

    # <<< CHANGED: return both (metrics_tuple) and (predictions_tuple)
    metrics = (acc, precision, recall, f1)
    preds   = (y_train_pred, y_val_pred, y_test_pred)
    return metrics, preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=88,
                        help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Number of hidden dimension.')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads for node-level attention.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='Alpha for the leaky_relu.')
    parser.add_argument('--q_vector', type=int, default=128,
                        help='The dimension for the semantic attention embedding.')
    parser.add_argument('--patience', type=int, default=100,
                        help='Number of epochs with no improvement to wait before stopping')
    parser.add_argument('--dataset', type=str, default='DBLP',
                        help='The dataset to use for the model.')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # List of datasets to train sequentially
    datasets = ['DBLP_area', 'DBLP_area_transformed']

    all_metrics = []
    all_preds   = []

    for dataset in datasets:
        metrics_i, preds_i = train_and_eval(dataset, args)
        all_metrics.append(metrics_i)
        all_preds.append(preds_i)

    # all_preds[0] is (y_train_pred_area, y_val_pred_area, y_test_pred_area)
    # all_preds[1] is (y_train_pred_area_trans, ...)

    print('----------------------------------------------------------------')
    print("Evaluating Robustness (overlap between original vs. transformed predictions)")
    print('----------------------------------------------------------------')

    train_robust, val_robust, test_robust = robustness(all_preds[0], all_preds[1])
    print(f"Train Robustness: {train_robust:.4f}")
    print(f"Val   Robustness: {val_robust:.4f}")
    print(f"Test  Robustness: {test_robust:.4f}\n")
        
    
