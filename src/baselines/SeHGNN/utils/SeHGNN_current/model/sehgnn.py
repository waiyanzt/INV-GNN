import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor


def xavier_uniform_(tensor, gain=1.):
    fan_in, fan_out = tensor.size()[-2:]
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return torch.nn.init._no_grad_uniform_(tensor, -a, a)


class Transformer(nn.Module):
    '''
        The transformer-based semantic fusion in SeHGNN.
    '''
    def __init__(self, n_channels, num_heads=1, att_drop=0., act='none'):
        super(Transformer, self).__init__()
        self.n_channels = n_channels
        self.num_heads = num_heads
        assert self.n_channels % (self.num_heads * 4) == 0

        self.query = nn.Linear(self.n_channels, self.n_channels//4)
        self.key   = nn.Linear(self.n_channels, self.n_channels//4)
        self.value = nn.Linear(self.n_channels, self.n_channels)

        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.att_drop = nn.Dropout(att_drop)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        elif act == 'none':
            self.act = lambda x: x
        else:
            assert 0, f'Unrecognized activation function {act} for class Transformer'

        self.reset_parameters()

    def reset_parameters(self):
        for k, v in self._modules.items():
            if hasattr(v, 'reset_parameters'):
                v.reset_parameters()
        nn.init.zeros_(self.gamma)

    def forward(self, x, mask=None):
        B, M, C = x.size() # batchsize, num_metapaths, channels
        H = self.num_heads
        if mask is not None:
            # A one-dimensional mask is shared by the whole minibatch.  This is
            # used by graph-variant augmentation so absent semantic channels can
            # retain parameters without contributing messages.
            if mask.dim() == 1:
                mask = mask.view(1, M).expand(B, M)
            if mask.size() != torch.Size((B, M)):
                raise ValueError(f'Expected channel mask {(B, M)}, got {tuple(mask.size())}')
            mask = mask.to(device=x.device, dtype=x.dtype)

        f = self.query(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]
        g = self.key(x).view(B, M, H, -1).permute(0,2,3,1)   # [B, H, -1, M]
        h = self.value(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]

        attention = self.act(f @ g / math.sqrt(f.size(-1)))
        if mask is not None:
            # Exclude absent key/value channels before softmax.  With an all-one
            # mask this is exactly the original attention computation, including
            # its attention-dropout behavior.
            active_keys = mask.to(dtype=torch.bool).view(B, 1, 1, M)
            attention = attention.masked_fill(~active_keys, torch.finfo(attention.dtype).min)
        beta = F.softmax(attention, dim=-1) # [B, H, M, M(normalized)]
        beta = self.att_drop(beta)

        o = self.gamma * (beta @ h) # [B, H, M, -1]
        return o.permute(0,2,1,3).reshape((B, M, C)) + x


class LinearPerMetapath(nn.Module):
    '''
        Linear projection per metapath for feature projection in SeHGNN.
    '''
    def __init__(self, cin, cout, num_metapaths):
        super(LinearPerMetapath, self).__init__()
        self.cin = cin
        self.cout = cout
        self.num_metapaths = num_metapaths

        self.W = nn.Parameter(torch.randn(self.num_metapaths, self.cin, self.cout))
        self.bias = nn.Parameter(torch.zeros(self.num_metapaths, self.cout))

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.W, gain=gain)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return torch.einsum('bcm,cmn->bcn', x, self.W) + self.bias.unsqueeze(0)


unfold_nested_list = lambda x: sum(x, [])

class SeHGNN(nn.Module):
    '''
        The SeHGNN model.
    '''
    def __init__(self, dataset, nfeat, hidden, nclass, feat_keys, label_feat_keys, tgt_type,
                 dropout, input_drop, att_drop, n_fp_layers, n_task_layers, act,
                 residual=False, data_size=None, drop_metapath=0., num_heads=1):
        super(SeHGNN, self).__init__()
        self.dataset = dataset
        self.feat_keys = sorted(feat_keys)
        self.label_feat_keys = sorted(label_feat_keys)
        self.num_channels = num_channels = len(self.feat_keys) + len(self.label_feat_keys)
        self.tgt_type = tgt_type
        self.residual = residual

        self.input_drop = nn.Dropout(input_drop)

        self.data_size = data_size
        self.embeding = nn.ParameterDict({})
        for k, v in data_size.items():
            self.embeding[str(k)] = nn.Parameter(torch.Tensor(v, nfeat))

        if len(self.label_feat_keys):
            self.labels_embeding = nn.ParameterDict({})
            for k in self.label_feat_keys:
                self.labels_embeding[k] = nn.Parameter(torch.Tensor(nclass, nfeat))
        else:
            self.labels_embeding = {}

        self.feature_projection = nn.Sequential(
            *([LinearPerMetapath(nfeat, hidden, num_channels),
               nn.LayerNorm([num_channels, hidden]),
               nn.PReLU(),
               nn.Dropout(dropout),]
            + unfold_nested_list([[
               LinearPerMetapath(hidden, hidden, num_channels),
               nn.LayerNorm([num_channels, hidden]),
               nn.PReLU(),
               nn.Dropout(dropout),] for _ in range(n_fp_layers - 1)])
            )
        )

        self.semantic_fusion = Transformer(hidden, num_heads=num_heads, att_drop=att_drop, act=act)
        self.fc_after_concat = nn.Linear(num_channels * hidden, hidden)

        if self.residual:
            self.res_fc = nn.Linear(nfeat, hidden)

        if self.dataset not in ['IMDB', 'Freebase']:
            self.task_mlp = nn.Sequential(
                *([nn.PReLU(),
                   nn.Dropout(dropout),]
                + unfold_nested_list([[
                   nn.Linear(hidden, hidden),
                   nn.BatchNorm1d(hidden, affine=False),
                   nn.PReLU(),
                   nn.Dropout(dropout),] for _ in range(n_task_layers - 1)])
                + [nn.Linear(hidden, nclass),
                   nn.BatchNorm1d(nclass, affine=False, track_running_stats=False)]
                )
            )
        else:
            self.task_mlp = nn.ModuleList(
                [nn.Sequential(
                    nn.PReLU(),
                    nn.Dropout(dropout))]
                + [nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.BatchNorm1d(hidden, affine=False),
                    nn.PReLU(),
                    nn.Dropout(dropout)) for _ in range(n_task_layers - 1)]
                + [nn.Sequential(
                    nn.Linear(hidden, nclass),
                    nn.LayerNorm(nclass, elementwise_affine=False),
                    )]
            )

        self.reset_parameters()

    def reset_parameters(self):
        for k, v in self._modules.items():
            if isinstance(v, nn.ParameterDict):
                for _k, _v in v.items():
                    _v.data.uniform_(-0.5, 0.5)
            elif isinstance(v, nn.ModuleList):
                for block in v:
                    if isinstance(block, nn.Sequential):
                        for layer in block:
                            if hasattr(layer, 'reset_parameters'):
                                layer.reset_parameters()
                    elif hasattr(block, 'reset_parameters'):
                        block.reset_parameters()
            elif isinstance(v, nn.Sequential):
                for layer in v:
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
            elif hasattr(v, 'reset_parameters'):
                v.reset_parameters()

    @staticmethod
    def _masked_layer_norm(x, channel_mask, layer):
        """LayerNorm over active channel/hidden entries only.

        The original SeHGNN LayerNorm uses normalized_shape=[channels, hidden],
        so ordinary normalization would let structurally absent channels change
        the mean and variance of active channels.  With an all-one mask this is
        algebraically the same operation as the original nn.LayerNorm.
        """
        expanded = channel_mask.unsqueeze(-1)
        count = expanded.sum(dim=(1, 2), keepdim=True) * x.size(-1)
        count = count.clamp_min(1.0)
        mean = (x * expanded).sum(dim=(1, 2), keepdim=True) / count
        variance = ((x - mean).pow(2) * expanded).sum(dim=(1, 2), keepdim=True) / count
        x = (x - mean) * torch.rsqrt(variance + layer.eps)
        if layer.elementwise_affine:
            x = x * layer.weight.unsqueeze(0) + layer.bias.unsqueeze(0)
        return x * expanded

    def _project_features(self, x, channel_mask):
        """Run the existing projection stack with structural channel masks."""
        for layer in self.feature_projection:
            if isinstance(layer, nn.LayerNorm) and channel_mask is not None:
                x = self._masked_layer_norm(x, channel_mask, layer)
            else:
                x = layer(x)
                if channel_mask is not None:
                    # This is required immediately after LinearPerMetapath because
                    # it has a learnable per-channel bias. Reapplying after every
                    # pointwise layer also prevents dropout/activation artifacts.
                    x = x * channel_mask.unsqueeze(-1)
        return x

    def forward(self, batch, feature_dict, label_dict={}, mask=None):
        if isinstance(feature_dict[self.tgt_type], torch.Tensor):
            features = {k: self.input_drop(x @ self.embeding[k]) for k, x in feature_dict.items()}
        elif isinstance(feature_dict[self.tgt_type], SparseTensor):
            # Freebase has so many metapaths that we use feature projection per target node type instead of per metapath
            features = {k: self.input_drop(x @ self.embeding[k[-1]]) for k, x in feature_dict.items()}
        else:
            assert 0

        B = num_node = features[self.tgt_type].shape[0]
        C = self.num_channels
        D = features[self.tgt_type].shape[1]

        labels = {k: self.input_drop(x @ self.labels_embeding[k]) for k, x in label_dict.items()}

        x = [features[k] for k in self.feat_keys] + [labels[k] for k in self.label_feat_keys]
        x = torch.stack(x, dim=1) # [B, C, D]

        channel_mask = None
        if mask is not None:
            channel_mask = mask
            if channel_mask.dim() == 1:
                channel_mask = channel_mask.view(1, C).expand(B, C)
            if channel_mask.size() != torch.Size((B, C)):
                raise ValueError(
                    f'Expected channel mask {(B, C)}, got {tuple(channel_mask.size())}'
                )
            channel_mask = channel_mask.to(device=x.device, dtype=x.dtype)

        x = self._project_features(x, channel_mask)
        x = self.semantic_fusion(x, mask=channel_mask)
        if channel_mask is not None:
            x = x * channel_mask.unsqueeze(-1)
        x = x.transpose(1,2)

        x = self.fc_after_concat(x.reshape(B, -1))
        if self.residual:
            x = x + self.res_fc(features[self.tgt_type])

        if self.dataset not in ['IMDB', 'Freebase']:
            return self.task_mlp(x)
        else:
            x = self.task_mlp[0](x)
            for i in range(1, len(self.task_mlp)-1):
                x = self.task_mlp[i](x) + x
            x = self.task_mlp[-1](x)
            return x
