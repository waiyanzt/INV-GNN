"""SlotGAT model class.

Adapted from SlotGAT_ICML23/NC/methods/SlotGAT/GNN.py.
Only the slotGAT class is retained; other model classes (myGAT, changedGAT,
GAT, GCN, MLP, LabelPropagation) and torch_geometric / torch_scatter /
torch.profiler dependencies have been removed.
"""

import torch
import torch as th
import torch.nn.functional as F
import torch.nn as nn
import math
from .conv import slotGATConv


class slotGAT(nn.Module):
    def __init__(self,
                 g,
                 edge_dim,
                 num_etypes,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 alpha,
                 num_ntype,
                 eindexer, aggregator="SA", predicted_by_slot="None",
                 addLogitsTrain="None", SAattDim=32, dataRecorder=None,
                 targetTypeAttention="False", vis_data_saver=None):
        super(slotGAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        self.num_ntype = num_ntype
        self.num_classes = num_classes
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.predicted_by_slot = predicted_by_slot
        self.addLogitsTrain = addLogitsTrain
        self.SAattDim = SAattDim
        self.vis_data_saver = vis_data_saver
        self.dataRecorder = dataRecorder

        if aggregator == "SA":
            last_dim = num_classes

            self.macroLinear = nn.Linear(last_dim, self.SAattDim, bias=True)
            nn.init.xavier_normal_(self.macroLinear.weight, gain=1.414)
            nn.init.normal_(self.macroLinear.bias, std=1.414 * math.sqrt(1 / (self.macroLinear.bias.flatten().shape[0])))
            self.macroSemanticVec = nn.Parameter(torch.FloatTensor(self.SAattDim, 1))
            nn.init.normal_(self.macroSemanticVec, std=1)

        self.last_fc = nn.Parameter(th.FloatTensor(size=(num_classes * self.num_ntype, num_classes)))
        nn.init.xavier_normal_(self.last_fc, gain=1.414)
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input projection (no residual)
        self.gat_layers.append(slotGATConv(edge_dim, num_etypes,
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation,
            alpha=alpha, num_ntype=num_ntype, eindexer=eindexer,
            inputhead=True, dataRecorder=dataRecorder))
        # hidden layers
        for l in range(1, num_layers):
            self.gat_layers.append(slotGATConv(edge_dim, num_etypes,
                num_hidden * heads[l - 1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation,
                alpha=alpha, num_ntype=num_ntype, eindexer=eindexer,
                dataRecorder=dataRecorder))
        # output projection
        self.gat_layers.append(slotGATConv(edge_dim, num_etypes,
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None,
            alpha=alpha, num_ntype=num_ntype, eindexer=eindexer,
            dataRecorder=dataRecorder))
        self.aggregator = aggregator
        self.by_slot = [f"by_slot_{nt}" for nt in range(g.num_ntypes)]
        assert aggregator in (["onedimconv", "average", "last_fc", "max", "SA"] + self.by_slot)
        if self.aggregator == "onedimconv":
            self.nt_aggr = nn.Parameter(torch.FloatTensor(1, 1, self.num_ntype, 1))
            nn.init.normal_(self.nt_aggr, std=1)
        self.register_buffer('epsilon', torch.FloatTensor([1e-12]))

    def l2byslot(self, x):
        x = x.view(-1, self.num_ntype, int(x.shape[1] / self.num_ntype))
        x = x / (torch.max(torch.norm(x, dim=2, keepdim=True), self.epsilon))
        x = x.flatten(1)
        return x

    def forward(self, features_list, e_feat, get_out="False"):
        encoded_embeddings = None
        h = []
        for nt_id, (fc, feature) in enumerate(zip(self.fc_list, features_list)):
            nt_ft = fc(feature)
            emsen_ft = torch.zeros([nt_ft.shape[0], nt_ft.shape[1] * self.num_ntype]).to(feature.device)
            emsen_ft[:, nt_ft.shape[1] * nt_id:nt_ft.shape[1] * (nt_id + 1)] = nt_ft
            h.append(emsen_ft)
        h = torch.cat(h, 0)
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](self.g, h, e_feat, get_out=get_out, res_attn=res_attn)
            h = h.flatten(1)
            encoded_embeddings = h
        logits, _ = self.gat_layers[-1](self.g, h, e_feat, get_out=get_out, res_attn=None)

        if self.aggregator == "SA":
            logits = logits.squeeze(1)
            logits = self.l2byslot(logits)
            logits = logits.view(-1, self.num_ntype, int(logits.shape[1] / self.num_ntype))

            if "getSlots" in get_out:
                self.logits = logits.detach()

            slot_scores = (F.tanh(self.macroLinear(logits)) @ self.macroSemanticVec).mean(0, keepdim=True)
            self.slot_scores = F.softmax(slot_scores, dim=1)
            logits = (logits * self.slot_scores).sum(1)
            if self.dataRecorder["meta"]["getSAAttentionScore"] == "True":
                self.dataRecorder["data"][f"{self.dataRecorder['status']}_SAAttentionScore"] = self.slot_scores.flatten().tolist()

        if self.predicted_by_slot != "None" and self.training == False:
            logits = logits.view(-1, 1, self.num_ntype, self.num_classes)
            if self.predicted_by_slot == "max":
                if "getMaxSlot" in get_out:
                    maxSlotIndexesWithLabels = logits.max(2)[1].squeeze(1)
                    logits_indexer = logits.max(2)[0].max(2)[1]
                    self.maxSlotIndexes = torch.gather(maxSlotIndexesWithLabels, 1, logits_indexer)
                logits = logits.max(2)[0]
            elif self.predicted_by_slot == "all":
                if "getSlots" in get_out:
                    self.logits = logits.detach()
                logits = logits.view(-1, 1, self.num_ntype, self.num_classes).mean(2)
            else:
                target_slot = int(self.predicted_by_slot)
                logits = logits[:, :, target_slot, :].squeeze(2)
        else:
            if self.aggregator == "average":
                logits = logits.view(-1, 1, self.num_ntype, self.num_classes).mean(2)
            elif self.aggregator == "onedimconv":
                logits = (logits.view(-1, 1, self.num_ntype, self.num_classes) * F.softmax(self.leaky_relu(self.nt_aggr), dim=2)).sum(2)
            elif self.aggregator == "last_fc":
                logits = logits.view(-1, 1, self.num_ntype, self.num_classes)
                logits = logits.flatten(1)
                logits = logits.matmul(self.last_fc).unsqueeze(1)
            elif self.aggregator == "max":
                logits = logits.view(-1, 1, self.num_ntype, self.num_classes).max(2)[0]
            elif self.aggregator == "None":
                logits = logits.view(-1, 1, self.num_ntype, self.num_classes).flatten(2)
            elif self.aggregator == "SA":
                logits = logits.view(-1, 1, 1, self.num_classes).flatten(2)
            else:
                raise NotImplementedError()

        self.logits_mean = logits.flatten().mean()
        logits = logits.mean(1)

        logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
        return logits, encoded_embeddings
