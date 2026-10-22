"""Feature builders shared by offline training and online serving (one code path = no skew)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SEARCH_FEATURES = [
    "bm25", "semantic_cos", "rrf", "category_prob", "color_match", "style_hint_match",
    "als_score", "user_style_aff", "user_color_aff", "price_fit", "log_popularity",
    "conversion_rate", "rating", "freshness", "session_covisit", "user_known",
]
REC_FEATURES = [
    "als_score", "als_rank", "covisit", "log_popularity", "conversion_rate", "user_style_aff",
    "user_color_aff", "user_slot_aff", "price_fit", "rating", "freshness", "seen_count",
    "in_popular_pool", "user_known",
]


@dataclass
class UserContext:
    als_vec: np.ndarray | None = None
    style_dist: np.ndarray = field(default_factory=lambda: np.full(6, 1 / 6))
    color_dist: np.ndarray = field(default_factory=lambda: np.full(16, 1 / 16))
    slot_dist: np.ndarray = field(default_factory=lambda: np.full(6, 1 / 6))
    rel_price: float | None = None       # mean log(price / subcategory base price)
    n_events: int = 0
    gender: str | None = None
    recent: list[int] = field(default_factory=list)
    seen: dict[int, int] = field(default_factory=dict)
    purchased: set = field(default_factory=set)


@dataclass
class ItemTable:
    """Column-oriented item attributes for vectorised feature computation."""
    style: np.ndarray
    color: np.ndarray
    slot: np.ndarray
    subcat: np.ndarray            # str
    gender: np.ndarray            # str
    price: np.ndarray
    rel_price: np.ndarray
    rating: np.ndarray
    added: np.ndarray
    log_pop: np.ndarray
    conv: np.ndarray


def _price_fit(it: ItemTable, cands: np.ndarray, u: UserContext) -> np.ndarray:
    if u.rel_price is None:
        return np.zeros(len(cands))
    return -np.abs(it.rel_price[cands] - u.rel_price)


def search_features(it: ItemTable, cands: np.ndarray, *, bm25: np.ndarray, sem: np.ndarray,
                    rrf: np.ndarray, subcat_probs: dict, color: str | None, color_ix: int | None,
                    style_hints_ix: list[int], u: UserContext, item_factors: np.ndarray,
                    covisit: np.ndarray, now: float) -> np.ndarray:
    n = len(cands)
    cat_p = np.array([subcat_probs.get(s, 0.0) for s in it.subcat[cands]]) if subcat_probs else np.zeros(n)
    color_match = (it.color[cands] == color_ix).astype(float) if color_ix is not None else np.zeros(n)
    style_hint = np.isin(it.style[cands], style_hints_ix).astype(float) if style_hints_ix else np.zeros(n)
    als = item_factors[cands] @ u.als_vec if u.als_vec is not None else np.zeros(n)
    return np.column_stack([
        bm25, sem, rrf, cat_p, color_match, style_hint, als,
        np.log(u.style_dist[it.style[cands]] + 1e-3),
        np.log(u.color_dist[it.color[cands]] + 1e-3),
        _price_fit(it, cands, u),
        it.log_pop[cands], it.conv[cands], it.rating[cands],
        -np.log1p(np.maximum(now - it.added[cands], 0)),
        covisit[cands], np.full(n, float(np.log1p(u.n_events))),
    ]).astype(np.float32)


def rec_features(it: ItemTable, cands: np.ndarray, *, als_scores: np.ndarray,
                 covisit: np.ndarray, popular_mask: np.ndarray, u: UserContext,
                 now: float) -> np.ndarray:
    n = len(cands)
    als = als_scores[cands]
    rank = np.empty(n)
    rank[np.argsort(-als)] = np.arange(n)
    seen = np.array([u.seen.get(int(c), 0) for c in cands], float)
    slot_ix = it.slot[cands]
    return np.column_stack([
        als, np.log1p(rank), covisit[cands], it.log_pop[cands], it.conv[cands],
        np.log(u.style_dist[it.style[cands]] + 1e-3),
        np.log(u.color_dist[it.color[cands]] + 1e-3),
        np.log(u.slot_dist[slot_ix] + 1e-3),
        _price_fit(it, cands, u), it.rating[cands],
        -np.log1p(np.maximum(now - it.added[cands], 0)),
        np.log1p(seen), popular_mask[cands].astype(float),
        np.full(n, float(np.log1p(u.n_events))),
    ]).astype(np.float32)
