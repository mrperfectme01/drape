"""Session co-visitation ("people who viewed this also viewed/bought") as a sparse top-K graph."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .als import EVENT_WEIGHTS


def build_covisit(sessions, items, etypes, n_items: int, topk: int = 50) -> sp.csr_matrix:
    _, s_idx = np.unique(sessions, return_inverse=True)
    B = sp.csr_matrix((EVENT_WEIGHTS[etypes], (s_idx, items)), shape=(s_idx.max() + 1, n_items))
    B.sum_duplicates()
    B.data = np.minimum(B.data, 10.0)                 # cap per-session weight
    B.data = np.log1p(B.data)
    M = (B.T @ B).tocsr()
    M.setdiag(0)
    M.eliminate_zeros()
    deg = np.sqrt(np.asarray(B.multiply(B).sum(0)).ravel()) + 1e-6
    M = sp.diags(1 / deg) @ M @ sp.diags(1 / deg)       # cosine-normalised co-occurrence
    M = M.tocsr()
    rows, cols, vals = [], [], []
    for r in range(n_items):
        lo, hi = M.indptr[r], M.indptr[r + 1]
        if lo == hi:
            continue
        d, c = M.data[lo:hi], M.indices[lo:hi]
        if len(d) > topk:
            top = np.argpartition(-d, topk)[:topk]
            d, c = d[top], c[top]
        rows.extend([r] * len(d)); cols.extend(c.tolist()); vals.extend(d.tolist())
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)


def covisit_scores(G: sp.csr_matrix, recent: list[int], decay: float = 0.8) -> np.ndarray:
    """Score every item by weighted similarity to a recency-ordered list (most recent last)."""
    n = G.shape[0]
    if not recent:
        return np.zeros(n, np.float32)
    w = np.zeros(n, np.float32)
    for pos, it in enumerate(reversed(recent[-20:])):
        w[it] += decay ** pos
    return np.asarray(G.T @ w).ravel()
