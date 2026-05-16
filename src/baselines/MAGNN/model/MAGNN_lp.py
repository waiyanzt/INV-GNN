import torch
import torch.nn as nn
import numpy as np

from model.base_MAGNN import MAGNN_ctr_ntype_specific


# ------------ One MAGNN block used for link prediction ------------
class MAGNN_lp_layer(nn.Module):
    def __init__(self,
                 num_metapaths_list,
                 num_edge_type,
                 etypes_lists,
                 in_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 attn_drop=0.5):
        super(MAGNN_lp_layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        # etype-specific parameters
        r_vec = None
        if rnn_type == 'TransE0':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type // 2, in_dim)))
        elif rnn_type == 'TransE1':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type, in_dim)))
        elif rnn_type == 'RotatE0':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type // 2, in_dim // 2, 2)))
        elif rnn_type == 'RotatE1':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type, in_dim // 2, 2)))
        if r_vec is not None:
            nn.init.xavier_normal_(r_vec.data, gain=1.414)

        # ctr_ntype-specific layers
        self.user_layer = MAGNN_ctr_ntype_specific(
            num_metapaths_list[0], etypes_lists[0], in_dim,
            num_heads, attn_vec_dim, rnn_type, r_vec, attn_drop, use_minibatch=True
        )
        self.item_layer = MAGNN_ctr_ntype_specific(
            num_metapaths_list[1], etypes_lists[1], in_dim,
            num_heads, attn_vec_dim, rnn_type, r_vec, attn_drop, use_minibatch=True
        )

        # multiple heads are concatenated => in_dim * num_heads
        self.fc_user = nn.Linear(in_dim * num_heads, out_dim, bias=True)
        self.fc_item = nn.Linear(in_dim * num_heads, out_dim, bias=True)
        nn.init.xavier_normal_(self.fc_user.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc_item.weight, gain=1.414)

    def forward(self, inputs):
        g_lists, features, type_mask, edge_metapath_indices_lists, target_idx_lists = inputs

        # ctr_ntype-specific target embeddings (pre-head)
        h_user = self.user_layer(
            (g_lists[0], features, type_mask, edge_metapath_indices_lists[0], target_idx_lists[0])
        )
        h_item = self.item_layer(
            (g_lists[1], features, type_mask, edge_metapath_indices_lists[1], target_idx_lists[1])
        )

        # project to out_dim for this layer
        logits_user = self.fc_user(h_user)
        logits_item = self.fc_item(h_item)
        return [logits_user, logits_item], [h_user, h_item]


# ------------ Stacked MAGNN for link prediction (supports K layers) ------------
class MAGNN_lp(nn.Module):
    def __init__(self,
                 num_metapaths_list,
                 num_edge_type,
                 etypes_lists,
                 feats_dim_list,
                 hidden_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 dropout_rate=0.5,
                 num_layers=3):  # default to 3 layers as requested
        super(MAGNN_lp, self).__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = int(num_layers)

        # ntype-specific transformation -> hidden_dim (no activation; matches original)
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True)
                                      for feats_dim in feats_dim_list])
        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(dropout_rate)
        else:
            self.feat_drop = nn.Identity()
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        if self.num_layers == 1:
            # exact original behavior: one block outputs out_dim directly
            self.layer1 = MAGNN_lp_layer(
                num_metapaths_list, num_edge_type, etypes_lists,
                hidden_dim, out_dim, num_heads, attn_vec_dim, rnn_type, attn_drop=dropout_rate
            )
        else:
            # K blocks: each outputs hidden_dim; then final heads map hidden_dim -> out_dim
            self.layers = nn.ModuleList([
                MAGNN_lp_layer(
                    num_metapaths_list, num_edge_type, etypes_lists,
                    hidden_dim, hidden_dim, num_heads, attn_vec_dim, rnn_type, attn_drop=dropout_rate
                )
                for _ in range(self.num_layers)
            ])
            self.out_user = nn.Linear(hidden_dim, out_dim, bias=True)
            self.out_item = nn.Linear(hidden_dim, out_dim, bias=True)
            nn.init.xavier_normal_(self.out_user.weight, gain=1.414)
            nn.init.xavier_normal_(self.out_item.weight, gain=1.414)

    def forward(self, inputs):
        g_lists, features_list, type_mask, edge_metapath_indices_lists, target_idx_lists = inputs

        # ensure graphs are on feature device
        dev = torch.device(features_list[0].device)
        for g_list in g_lists:
            for i in range(len(g_list)):
                g_list[i] = g_list[i].to(dev)

        # ntype-specific transformation -> global node features
        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=dev)
        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            transformed_features[node_indices] = fc(features_list[i])
        transformed_features = self.feat_drop(transformed_features)

        if self.num_layers == 1:
            # single-layer path (original)
            [logits_user, logits_item], [h_user, h_item] = self.layer1(
                (g_lists, transformed_features, type_mask, edge_metapath_indices_lists, target_idx_lists)
            )
            return [logits_user, logits_item], [h_user, h_item]

        # K-layer path: aggregate per-layer logits (hidden_dim) and also keep raw h_* for inspection
        logits_user_sum = None
        logits_item_sum = None
        h_user_sum = None
        h_item_sum = None

        for layer in self.layers:
            [logits_user_k, logits_item_k], [h_user_k, h_item_k] = layer(
                (g_lists, transformed_features, type_mask, edge_metapath_indices_lists, target_idx_lists)
            )
            logits_user_sum = logits_user_k if logits_user_sum is None else (logits_user_sum + logits_user_k)
            logits_item_sum = logits_item_k if logits_item_sum is None else (logits_item_sum + logits_item_k)
            h_user_sum = h_user_k if h_user_sum is None else (h_user_sum + h_user_k)
            h_item_sum = h_item_k if h_item_sum is None else (h_item_sum + h_item_k)

        # average across layers to stabilize scale
        logits_user_mean = logits_user_sum / self.num_layers
        logits_item_mean = logits_item_sum / self.num_layers
        h_user_mean = h_user_sum / self.num_layers
        h_item_mean = h_item_sum / self.num_layers

        # final projection to out_dim
        logits_user = self.out_user(self.feat_drop(logits_user_mean))
        logits_item = self.out_item(self.feat_drop(logits_item_mean))
        return [logits_user, logits_item], [h_user_mean, h_item_mean]
