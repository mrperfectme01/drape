"""Ranking metrics. `rels` is the relevance grade list in *ranked* order."""
from __future__ import annotations

import numpy as np


def dcg_at_k(rels, k: int) -> float:
    r = np.asarray(rels, float)[:k]
    return float(((2 ** r - 1) / np.log2(np.arange(2, len(r) + 2))).sum())


def ndcg_at_k(rels, k: int, all_rels=None) -> float:
    """all_rels: grades of *every* relevant item (incl. ones not retrieved) for the ideal DCG."""
    ideal_src = np.asarray(all_rels if all_rels is not None else rels, float)
    idcg = dcg_at_k(np.sort(ideal_src)[::-1], k)
    return dcg_at_k(rels, k) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(list(ranked)[:k]) & relevant) / len(relevant)


def hit_at_k(ranked, relevant: set, k: int) -> float:
    return float(bool(set(list(ranked)[:k]) & relevant))


def mrr(ranked, relevant: set, k: int = 100) -> float:
    for i, x in enumerate(list(ranked)[:k]):
        if x in relevant:
            return 1.0 / (i + 1)
    return 0.0


def auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    s = np.concatenate([scores_pos, scores_neg])
    ranks = s.argsort().argsort() + 1
    n_p, n_n = len(scores_pos), len(scores_neg)
    return float((ranks[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))
