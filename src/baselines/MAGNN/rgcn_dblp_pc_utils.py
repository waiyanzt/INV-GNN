"""Shared helpers for RGCN DBLP paper–conference link prediction scripts."""
from __future__ import annotations

import numpy as np


def subsample_negs_per_paper(neg: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
    """
    Keep at most k negative (paper_id, conf_id) rows per paper.
    Papers are processed in sorted paper order; within each paper, k rows are chosen
    uniformly without replacement when possible.
    """
    if k <= 0 or len(neg) == 0:
        return neg
    order = np.lexsort((neg[:, 1], neg[:, 0]))
    neg = neg[order]
    chunks: list[np.ndarray] = []
    i = 0
    while i < len(neg):
        j = i
        p = int(neg[i, 0])
        while j < len(neg) and int(neg[j, 0]) == p:
            j += 1
        block = neg[i:j]
        if len(block) <= k:
            chunks.append(block)
        else:
            pick = rng.choice(len(block), size=k, replace=False)
            chunks.append(block[np.sort(pick)])
        i = j
    return np.vstack(chunks) if chunks else neg[:0]
