"""
DBLP paper–venue link prediction — shared **evaluation** helpers.

Training still uses the full ``train_neg`` / ``val_neg`` from preprocessed splits.
For **test**, we optionally subsample ``test_neg`` so ranking metrics match a
small closed-world pool (e.g. 1 true + 3 negatives per paper), comparable to CMPNN.

**Hits@k semantics** (same as standard information retrieval):

- Sort all scored (paper, conference) pairs for a paper by **descending** model score.
- Let **r** be the **1-based rank** of the **true** test positive (``is_true == 1``).
  If multiple true flags existed, we use the **best** (smallest) rank.
- **Hits@1**: ``r == 1`` (the correct conference is top-1).
- **Hits@3**: ``r <= 3`` (correct is in positions 1, 2, or 3).
- **Hits@5**: ``r <= 5``. If there are only **4** candidates total (1 pos + 3 negs),
  then ``r <= 5`` is always true whenever the true label is present, so Hits@5 is **uninformative**
  and callers should omit or interpret it accordingly.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def subsample_test_neg_per_paper(
    neg_edges: np.ndarray,
    k: int,
    seed: int,
) -> np.ndarray:
    """
    Keep at most ``k`` negative (paper_id, conf_id) rows **per paper** (column 0).

    ``neg_edges`` must be shape (N, 2), int-like. Papers with fewer than ``k``
    negatives keep all of them.

    Deterministic given ``seed`` (uses ``numpy.random.RandomState``).
    """
    if k <= 0:
        return np.asarray(neg_edges, dtype=np.int64)
    rng = np.random.RandomState(int(seed) + 90210)
    by_paper: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in neg_edges:
        p, c = int(row[0]), int(row[1])
        by_paper[p].append((p, c))
    out: list[tuple[int, int]] = []
    for _p, rows in by_paper.items():
        if len(rows) <= k:
            out.extend(rows)
        else:
            idx = rng.choice(len(rows), size=k, replace=False)
            out.extend(rows[i] for i in idx)
    return np.asarray(out, dtype=np.int64)
