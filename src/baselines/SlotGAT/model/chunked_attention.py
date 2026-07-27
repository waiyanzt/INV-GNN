"""Exact memory-bounded attention aggregation for very large SlotGAT graphs.

The implementation performs destination-wise softmax over the complete graph
while visiting edges in bounded chunks. Its custom backward recomputes chunk
attention and uses the closed-form softmax gradient, so no E x heads attention
autograd graph is retained between SlotGAT layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ChunkedAttentionState:
    """Detached recipe for reproducing one layer's final edge attention."""

    el: Tensor
    er: Tensor
    relation_scores: Tensor
    edge_index: Tensor
    edge_type: Tensor
    maximum: Tensor
    denominator: Tensor
    negative_slope: float
    alpha: float
    dropout_p: float
    dropout_seed: int
    edge_chunk_size: int
    previous: Optional["ChunkedAttentionState"]


def _edge_logits(
    el: Tensor,
    er: Tensor,
    relation_scores: Tensor,
    edge_index: Tensor,
    edge_type: Tensor,
    start: int,
    end: int,
    negative_slope: float,
) -> Tensor:
    source = edge_index[0, start:end]
    target = edge_index[1, start:end]
    pre_activation = (
        el[source] + er[target] + relation_scores[edge_type[start:end]]
    )
    return F.leaky_relu(pre_activation, negative_slope=negative_slope)


@torch.no_grad()
def softmax_statistics(
    el: Tensor,
    er: Tensor,
    relation_scores: Tensor,
    edge_index: Tensor,
    edge_type: Tensor,
    negative_slope: float,
    edge_chunk_size: int,
) -> Tuple[Tensor, Tensor]:
    """Compute exact global destination softmax maximum and denominator."""
    num_nodes, num_heads = el.shape
    maximum = el.new_full((num_nodes, num_heads), float("-inf"))
    num_edges = edge_type.numel()
    for start in range(0, num_edges, edge_chunk_size):
        end = min(start + edge_chunk_size, num_edges)
        target = edge_index[1, start:end]
        logits = _edge_logits(
            el,
            er,
            relation_scores,
            edge_index,
            edge_type,
            start,
            end,
            negative_slope,
        )
        maximum.scatter_reduce_(
            0,
            target.unsqueeze(1).expand(-1, num_heads),
            logits,
            reduce="amax",
            include_self=True,
        )

    denominator = el.new_zeros((num_nodes, num_heads))
    for start in range(0, num_edges, edge_chunk_size):
        end = min(start + edge_chunk_size, num_edges)
        target = edge_index[1, start:end]
        logits = _edge_logits(
            el,
            er,
            relation_scores,
            edge_index,
            edge_type,
            start,
            end,
            negative_slope,
        )
        denominator.index_add_(0, target, torch.exp(logits - maximum[target]))
    return maximum, denominator.clamp_min_(torch.finfo(el.dtype).tiny)


def _dropout_scale(
    shape,
    *,
    dropout_p: float,
    dropout_seed: int,
    chunk_index: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor | float:
    if dropout_p <= 0:
        return 1.0
    generator = torch.Generator(device=device)
    generator.manual_seed(dropout_seed + chunk_index)
    keep = torch.rand(shape, device=device, generator=generator) >= dropout_p
    return keep.to(dtype=dtype) / (1.0 - dropout_p)


def _base_attention_chunk(
    *,
    el: Tensor,
    er: Tensor,
    relation_scores: Tensor,
    edge_index: Tensor,
    edge_type: Tensor,
    maximum: Tensor,
    denominator: Tensor,
    negative_slope: float,
    dropout_p: float,
    dropout_seed: int,
    edge_chunk_size: int,
    start: int,
    end: int,
) -> Tuple[Tensor, Tensor, Tensor | float]:
    target = edge_index[1, start:end]
    logits = _edge_logits(
        el,
        er,
        relation_scores,
        edge_index,
        edge_type,
        start,
        end,
        negative_slope,
    )
    attention = torch.exp(logits - maximum[target]) / denominator[target]
    scale = _dropout_scale(
        attention.shape,
        dropout_p=dropout_p,
        dropout_seed=dropout_seed,
        chunk_index=start // edge_chunk_size,
        device=attention.device,
        dtype=attention.dtype,
    )
    return attention, attention * scale, scale


@torch.no_grad()
def state_attention_chunk(
    state: ChunkedAttentionState,
    start: int,
    end: int,
) -> Tensor:
    """Reproduce final detached attention for one edge interval."""
    base, dropped, _ = _base_attention_chunk(
        el=state.el,
        er=state.er,
        relation_scores=state.relation_scores,
        edge_index=state.edge_index,
        edge_type=state.edge_type,
        maximum=state.maximum,
        denominator=state.denominator,
        negative_slope=state.negative_slope,
        dropout_p=state.dropout_p,
        dropout_seed=state.dropout_seed,
        edge_chunk_size=state.edge_chunk_size,
        start=start,
        end=end,
    )
    del base
    if state.previous is None or state.alpha <= 0:
        return dropped
    previous = state_attention_chunk(state.previous, start, end)
    return dropped * (1.0 - state.alpha) + previous * state.alpha


class _ChunkedAttentionAggregate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        feat_src: Tensor,
        el: Tensor,
        er: Tensor,
        relation_scores: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        maximum: Tensor,
        denominator: Tensor,
        negative_slope: float,
        alpha: float,
        dropout_p: float,
        dropout_seed: int,
        edge_chunk_size: int,
        decomposed_layers: int,
        previous_state: Optional[ChunkedAttentionState],
    ) -> Tensor:
        num_nodes, num_heads, feature_dim = feat_src.shape
        output = feat_src.new_zeros((num_nodes, num_heads, feature_dim))
        feature_chunk_size = (
            feature_dim + decomposed_layers - 1
        ) // decomposed_layers
        num_edges = edge_type.numel()
        for start in range(0, num_edges, edge_chunk_size):
            end = min(start + edge_chunk_size, num_edges)
            source = edge_index[0, start:end]
            target = edge_index[1, start:end]
            _, dropped, _ = _base_attention_chunk(
                el=el,
                er=er,
                relation_scores=relation_scores,
                edge_index=edge_index,
                edge_type=edge_type,
                maximum=maximum,
                denominator=denominator,
                negative_slope=negative_slope,
                dropout_p=dropout_p,
                dropout_seed=dropout_seed,
                edge_chunk_size=edge_chunk_size,
                start=start,
                end=end,
            )
            attention = dropped
            if previous_state is not None and alpha > 0:
                previous = state_attention_chunk(previous_state, start, end)
                attention = dropped * (1.0 - alpha) + previous * alpha
            for feature_start in range(0, feature_dim, feature_chunk_size):
                feature_end = min(
                    feature_start + feature_chunk_size, feature_dim
                )
                output[:, :, feature_start:feature_end].index_add_(
                    0,
                    target,
                    attention.unsqueeze(-1)
                    * feat_src[source, :, feature_start:feature_end],
                )

        ctx.save_for_backward(
            feat_src,
            el,
            er,
            relation_scores,
            edge_index,
            edge_type,
            maximum,
            denominator,
        )
        ctx.negative_slope = float(negative_slope)
        ctx.alpha = float(alpha)
        ctx.dropout_p = float(dropout_p)
        ctx.dropout_seed = int(dropout_seed)
        ctx.edge_chunk_size = int(edge_chunk_size)
        ctx.decomposed_layers = int(decomposed_layers)
        ctx.previous_state = previous_state
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (
            feat_src,
            el,
            er,
            relation_scores,
            edge_index,
            edge_type,
            maximum,
            denominator,
        ) = ctx.saved_tensors
        negative_slope = ctx.negative_slope
        alpha = ctx.alpha
        dropout_p = ctx.dropout_p
        dropout_seed = ctx.dropout_seed
        edge_chunk_size = ctx.edge_chunk_size
        decomposed_layers = ctx.decomposed_layers
        previous_state = ctx.previous_state

        grad_feat_src = torch.zeros_like(feat_src)
        weighted_q_sum = el.new_zeros(el.shape)
        feature_dim = feat_src.shape[2]
        feature_chunk_size = (
            feature_dim + decomposed_layers - 1
        ) // decomposed_layers
        num_edges = edge_type.numel()

        # First pass: direct value-path gradient and the destination-wise term
        # required by the exact softmax Jacobian.
        for start in range(0, num_edges, edge_chunk_size):
            end = min(start + edge_chunk_size, num_edges)
            source = edge_index[0, start:end]
            target = edge_index[1, start:end]
            _base, dropped, _scale = _base_attention_chunk(
                el=el,
                er=er,
                relation_scores=relation_scores,
                edge_index=edge_index,
                edge_type=edge_type,
                maximum=maximum,
                denominator=denominator,
                negative_slope=negative_slope,
                dropout_p=dropout_p,
                dropout_seed=dropout_seed,
                edge_chunk_size=edge_chunk_size,
                start=start,
                end=end,
            )
            final_attention = dropped
            if previous_state is not None and alpha > 0:
                previous = state_attention_chunk(previous_state, start, end)
                final_attention = (
                    dropped * (1.0 - alpha) + previous * alpha
                )
            q = el.new_zeros((end - start, el.shape[1]))
            for feature_start in range(0, feature_dim, feature_chunk_size):
                feature_end = min(
                    feature_start + feature_chunk_size, feature_dim
                )
                grad_feat_src[:, :, feature_start:feature_end].index_add_(
                    0,
                    source,
                    final_attention.unsqueeze(-1)
                    * grad_output[target, :, feature_start:feature_end],
                )
                q.add_(
                    (
                        grad_output[target, :, feature_start:feature_end]
                        * feat_src[source, :, feature_start:feature_end]
                    ).sum(dim=-1)
                )
            weighted_q_sum.index_add_(0, target, dropped * q)

        grad_el = torch.zeros_like(el)
        grad_er = torch.zeros_like(er)
        grad_relation_scores = torch.zeros_like(relation_scores)

        # Second pass: attention-logit gradients. Previous residual attention
        # is intentionally detached, matching the original SlotGAT layer.
        current_factor = 1.0 - alpha if previous_state is not None else 1.0
        for start in range(0, num_edges, edge_chunk_size):
            end = min(start + edge_chunk_size, num_edges)
            source = edge_index[0, start:end]
            target = edge_index[1, start:end]
            base, _dropped, scale = _base_attention_chunk(
                el=el,
                er=er,
                relation_scores=relation_scores,
                edge_index=edge_index,
                edge_type=edge_type,
                maximum=maximum,
                denominator=denominator,
                negative_slope=negative_slope,
                dropout_p=dropout_p,
                dropout_seed=dropout_seed,
                edge_chunk_size=edge_chunk_size,
                start=start,
                end=end,
            )
            q = el.new_zeros((end - start, el.shape[1]))
            for feature_start in range(0, feature_dim, feature_chunk_size):
                feature_end = min(
                    feature_start + feature_chunk_size, feature_dim
                )
                q.add_(
                    (
                        grad_output[target, :, feature_start:feature_end]
                        * feat_src[source, :, feature_start:feature_end]
                    ).sum(dim=-1)
                )
            grad_logits = current_factor * base * (
                scale * q - weighted_q_sum[target]
            )
            pre_activation = (
                el[source]
                + er[target]
                + relation_scores[edge_type[start:end]]
            )
            grad_pre_activation = grad_logits * torch.where(
                pre_activation > 0,
                torch.ones((), device=pre_activation.device, dtype=pre_activation.dtype),
                torch.full(
                    (),
                    negative_slope,
                    device=pre_activation.device,
                    dtype=pre_activation.dtype,
                ),
            )
            grad_el.index_add_(0, source, grad_pre_activation)
            grad_er.index_add_(0, target, grad_pre_activation)
            grad_relation_scores.index_add_(
                0, edge_type[start:end], grad_pre_activation
            )

        return (
            grad_feat_src,
            grad_el,
            grad_er,
            grad_relation_scores,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def chunked_attention_aggregate(
    *,
    feat_src: Tensor,
    el: Tensor,
    er: Tensor,
    relation_scores: Tensor,
    edge_index: Tensor,
    edge_type: Tensor,
    negative_slope: float,
    alpha: float,
    dropout_p: float,
    edge_chunk_size: int,
    decomposed_layers: int,
    previous_state: Optional[ChunkedAttentionState],
) -> Tuple[Tensor, ChunkedAttentionState]:
    if edge_chunk_size <= 0:
        raise ValueError("edge_chunk_size must be positive")
    if decomposed_layers <= 0:
        raise ValueError("decomposed_layers must be positive")
    if previous_state is not None:
        if previous_state.edge_chunk_size != edge_chunk_size:
            raise ValueError("Residual attention states must use one chunk size")
        if previous_state.edge_type.shape != edge_type.shape:
            raise ValueError("Residual attention edge layouts differ")

    maximum, denominator = softmax_statistics(
        el.detach(),
        er.detach(),
        relation_scores.detach(),
        edge_index,
        edge_type,
        negative_slope,
        edge_chunk_size,
    )
    active_dropout = float(dropout_p)
    dropout_seed = (
        int(
            torch.randint(
                0,
                2**31 - 1,
                (1,),
                device=feat_src.device,
            ).item()
        )
        if active_dropout > 0
        else 0
    )
    output = _ChunkedAttentionAggregate.apply(
        feat_src,
        el,
        er,
        relation_scores,
        edge_index,
        edge_type,
        maximum,
        denominator,
        negative_slope,
        alpha,
        active_dropout,
        dropout_seed,
        edge_chunk_size,
        decomposed_layers,
        previous_state,
    )
    state = ChunkedAttentionState(
        el=el.detach(),
        er=er.detach(),
        relation_scores=relation_scores.detach(),
        edge_index=edge_index,
        edge_type=edge_type,
        maximum=maximum,
        denominator=denominator,
        negative_slope=float(negative_slope),
        alpha=float(alpha),
        dropout_p=active_dropout,
        dropout_seed=dropout_seed,
        edge_chunk_size=int(edge_chunk_size),
        previous=previous_state,
    )
    return output, state
