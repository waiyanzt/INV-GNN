"""SlotGAT convolution layer (slotGATConv).

Adapted from SlotGAT_ICML23/NC/methods/SlotGAT/conv.py.
Only slotGATConv is retained; other conv classes and torch.profiler
instrumentation have been removed.
"""

import torch as th
from torch import nn
import torch
from dgl import function as fn
from dgl.nn.pytorch import edge_softmax
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
import torch.nn.functional as F
import numpy as np

from .chunked_attention import (
    ChunkedAttentionState,
    chunked_attention_aggregate,
)


class slotGATConv(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """
    def __init__(self,
                 edge_feats,
                 num_etypes,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=False,
                 alpha=0.,
                 num_ntype=None, eindexer=None, inputhead=False,
                 dataRecorder=None, edge_chunk_size=0,
                 decomposed_layers=1):
        super(slotGATConv, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.num_etypes = int(num_etypes)
        self.edge_emb = nn.Embedding(num_etypes, edge_feats) if edge_feats else None
        self.eindexer = eindexer
        self.num_ntype = num_ntype

        self.attentions = None
        self.dataRecorder = dataRecorder

        if isinstance(in_feats, tuple):
            raise NotImplementedError()
        else:
            self.fc = nn.Parameter(th.FloatTensor(size=(self.num_ntype, self._in_src_feats, out_feats * num_heads)))
        self.fc_e = nn.Linear(edge_feats, edge_feats * num_heads, bias=False) if edge_feats else None
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats * self.num_ntype)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats * self.num_ntype)))
        self.attn_e = nn.Parameter(th.FloatTensor(size=(1, num_heads, edge_feats))) if edge_feats else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Parameter(th.FloatTensor(size=(self.num_ntype, self._in_src_feats, out_feats * num_heads)))
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        self.alpha = alpha
        self.inputhead = inputhead
        self.edge_chunk_size = int(edge_chunk_size)
        if self.edge_chunk_size < 0:
            raise ValueError("edge_chunk_size must be nonnegative")
        self.decomposed_layers = int(decomposed_layers)
        if self.decomposed_layers <= 0:
            raise ValueError("decomposed_layers must be positive")

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc, gain=gain)
        else:
            raise NotImplementedError()
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self._edge_feats:
            nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        elif isinstance(self.res_fc, Identity):
            pass
        elif isinstance(self.res_fc, nn.Parameter):
            nn.init.xavier_normal_(self.res_fc, gain=gain)
        if self._edge_feats:
            nn.init.xavier_normal_(self.fc_e.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat, get_out=[""], res_attn=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                raise NotImplementedError()
            else:
                h_src = h_dst = self.feat_drop(feat)

                if self.inputhead:
                    h_src = h_src.view(-1, 1, self.num_ntype, self._in_src_feats)
                else:
                    h_src = h_src.view(-1, self._num_heads, self.num_ntype, int(self._in_src_feats / self._num_heads))
                h_dst = h_src = h_src.permute(2, 0, 1, 3).flatten(2)
                if "getEmb" in get_out:
                    self.emb = h_dst.cpu().detach()
                feat_dst = torch.bmm(h_src, self.fc)
                feat_src = feat_dst = feat_dst.permute(1, 0, 2).view(
                    -1, self.num_ntype, self._num_heads, self._out_feats).permute(0, 2, 1, 3).flatten(2)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                relation_edge_scores = None
                if self._edge_feats:
                    # Algebraically fuse edge embedding, projection, and the
                    # final attention-vector dot product per relation type:
                    #
                    #   dot(fc_e(edge_emb[e]), attn_e)
                    #
                    # The original implementation materialized an
                    # E x heads x edge_feats tensor. Freebase exact_2 has about
                    # 94.7M directed edges, making that intermediate roughly
                    # 181 GiB for 8 heads and 64 edge dimensions. Computing the
                    # same scalar table for each relation first reduces the
                    # edge-sized tensor to E x heads without changing the
                    # learned function or gradients.
                    relation_edge_features = self.fc_e(
                        self.edge_emb.weight
                    ).view(-1, self._num_heads, self._edge_feats)
                    relation_edge_scores = (
                        relation_edge_features * self.attn_e
                    ).sum(dim=-1)
                el = (feat_src * self.attn_l).sum(dim=-1)
                er = (feat_dst * self.attn_r).sum(dim=-1)

            if self.edge_chunk_size > 0:
                if graph.is_block:
                    raise NotImplementedError(
                        "Chunked SlotGAT does not support DGL blocks"
                    )
                if res_attn is not None and not isinstance(
                    res_attn, ChunkedAttentionState
                ):
                    raise TypeError(
                        "Chunked SlotGAT residual attention requires a "
                        "ChunkedAttentionState"
                    )
                if relation_edge_scores is None:
                    relation_edge_scores = feat_src.new_zeros(
                        (self.num_etypes, self._num_heads)
                    )
                edge_index = getattr(graph, "slotgat_edge_index", None)
                if edge_index is None:
                    source, target = graph.edges(order="eid")
                    edge_index = torch.stack((source, target), dim=0)
                active_attn_dropout = (
                    float(self.attn_drop.p) if self.training else 0.0
                )
                rst, attention = chunked_attention_aggregate(
                    feat_src=feat_src,
                    el=el,
                    er=er,
                    relation_scores=relation_edge_scores,
                    edge_index=edge_index,
                    edge_type=e_feat,
                    negative_slope=float(self.leaky_relu.negative_slope),
                    alpha=float(self.alpha),
                    dropout_p=active_attn_dropout,
                    edge_chunk_size=self.edge_chunk_size,
                    decomposed_layers=self.decomposed_layers,
                    previous_state=res_attn,
                )
                self.attentions = None
            else:
                if isinstance(res_attn, ChunkedAttentionState):
                    raise TypeError(
                        "Standard SlotGAT cannot consume chunked attention state"
                    )
                ee = (
                    relation_edge_scores[e_feat].unsqueeze(-1)
                    if self._edge_feats
                    else 0
                )
                el_dgl = el.unsqueeze(-1)
                er_dgl = er.unsqueeze(-1)
                graph.srcdata.update({'ft': feat_src, 'el': el_dgl})
                graph.dstdata.update({'er': er_dgl})
                if self._edge_feats:
                    graph.edata.update({'ee': ee})
                graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
                e_ = graph.edata.pop('e')
                ee = graph.edata.pop('ee') if self._edge_feats else 0
                e = self.leaky_relu(e_ + ee)
                a = self.attn_drop(edge_softmax(graph, e))
                if res_attn is not None:
                    a = a * (1 - self.alpha) + res_attn * self.alpha
                if self.dataRecorder["status"] == "FinalTesting":
                    if "attention" not in self.dataRecorder["data"]:
                        self.dataRecorder["data"]["attention"] = []
                    self.dataRecorder["data"]["attention"].append(a)
                graph.edata['a'] = a
                graph.update_all(
                    fn.u_mul_e('ft', 'a', 'm'),
                    fn.sum('m', 'ft'),
                )
                rst = graph.dstdata['ft']
                attention = graph.edata.pop('a').detach()
                retain_attention = (
                    self.dataRecorder is not None
                    and self.dataRecorder.get("meta", {}).get(
                        "retainLayerAttention", False
                    )
                )
                # Keeping every layer's detached E x H attention tensor is
                # unnecessary for ordinary training and is especially costly
                # for augmented WordNet/Freebase graphs.
                self.attentions = attention if retain_attention else None

            if self.res_fc is not None:
                if self._in_dst_feats != self._out_feats:
                    resval = torch.bmm(h_src, self.res_fc)
                    resval = resval.permute(1, 0, 2).view(
                        -1, self.num_ntype, self._num_heads, self._out_feats).permute(0, 2, 1, 3).flatten(2)
                else:
                    resval = self.res_fc(h_src).view(h_dst.shape[0], -1, self._out_feats * self.num_ntype)
                rst = rst + resval
            if self.bias:
                rst = rst + self.bias_param
            if self.activation:
                rst = self.activation(rst)
            torch.cuda.empty_cache()
            return rst, attention
