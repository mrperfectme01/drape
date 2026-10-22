"""'Complete the look': a learned outfit-compatibility model.

From co-purchases across complementary slots (shirt bought with chinos and loafers in one
session) it learns:
  * a 16x16 colour-pair PMI matrix  -> rediscovers colour harmony (olive+mustard, maroon+beige...)
  * a 6x6 style-pair PMI matrix     -> formal goes with formal, sporty with streetwear...
and combines them with price coherence and behavioural similarity (ALS) in a logistic
regression trained against same-slot random negatives. Nobody hand-codes fashion rules.
"""
from __future__ import annotations

import numpy as np

from ..data import schema as S

FEATURES = ["color_pmi", "style_pmi", "same_style", "abs_log_price_ratio", "tier_gap",
            "als_cos", "same_brand"]


def complementary_pairs(items, sessions, item_ids, etypes, min_etype: int = 2):
    """(anchor, complement) pairs bought in the same session across complementary slots."""
    mask = etypes >= min_etype
    by_sess: dict[int, list[int]] = {}
    for s, i in zip(sessions[mask], item_ids[mask]):
        by_sess.setdefault(int(s), []).append(int(i))
    pairs = []
    for lst in by_sess.values():
        lst = list(dict.fromkeys(lst))
        for a in lst:
            for b in lst:
                if a != b and items[b]["slot"] in S.COMPLEMENTS[items[a]["slot"]]:
                    pairs.append((a, b))
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


class OutfitModel:
    def __init__(self, items: list[dict], item_factors: np.ndarray | None = None):
        self.items = items
        self.color = np.array([S.COLORS.index(it["color"]) for it in items])
        self.style = np.array([S.STYLES.index(it["style"]) for it in items])
        self.logp = np.log([it["price"] for it in items])
        self.tier = np.array([S.PRICE_TIERS.index(it["tier"]) for it in items])
        self.brand = np.array([hash(it["brand"]) % 10_000 for it in items])
        self.slot = np.array([it["slot"] for it in items])
        self.gender = np.array([it["gender"] for it in items])
        F = item_factors if item_factors is not None else np.zeros((len(items), 1))
        self.F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
        self.color_pmi = np.zeros((len(S.COLORS), len(S.COLORS)))
        self.style_pmi = np.zeros((len(S.STYLES), len(S.STYLES)))
        self.w = np.zeros(len(FEATURES))
        self.b = 0.0
        self.mu = np.zeros(len(FEATURES))
        self.sd = np.ones(len(FEATURES))

    @staticmethod
    def _pmi(a: np.ndarray, b: np.ndarray, base_a: np.ndarray, base_b: np.ndarray, n: int):
        """PMI of observed pairs vs. what random same-slot pairing would give (catalog base rates)."""
        C = np.zeros((n, n))
        np.add.at(C, (a, b), 1)
        C = C + C.T + 1.0
        expect = np.outer(np.bincount(base_a, minlength=n) + 1, np.bincount(base_b, minlength=n) + 1)
        P = C / C.sum()
        E = expect / expect.sum()
        return np.log(P / E)

    def features(self, a: int, cands: np.ndarray) -> np.ndarray:
        return np.column_stack([
            self.color_pmi[self.color[a], self.color[cands]],
            self.style_pmi[self.style[a], self.style[cands]],
            (self.style[cands] == self.style[a]).astype(float),
            np.abs(self.logp[cands] - self.logp[a]),
            np.abs(self.tier[cands] - self.tier[a]),
            self.F[cands] @ self.F[a],
            (self.brand[cands] == self.brand[a]).astype(float),
        ])

    def _negatives(self, rng, pairs: np.ndarray, k: int) -> np.ndarray:
        slots = {s: np.where(self.slot == s)[0] for s in S.SLOTS}
        neg = []
        for a, b in pairs:
            pool = slots[self.slot[b]]
            neg.extend((a, int(x)) for x in rng.choice(pool, size=k))
        return np.array(neg)

    def fit(self, pairs: np.ndarray, neg_per_pos: int = 4, epochs: int = 300, seed: int = 0):
        rng = np.random.default_rng(seed)
        n_c, n_s = len(S.COLORS), len(S.STYLES)
        self.color_pmi = self._pmi(self.color[pairs[:, 0]], self.color[pairs[:, 1]],
                                   self.color, self.color, n_c)
        self.style_pmi = self._pmi(self.style[pairs[:, 0]], self.style[pairs[:, 1]],
                                   self.style, self.style, n_s)
        neg = self._negatives(rng, pairs, neg_per_pos)
        X = np.vstack([self.features(a, np.array([b]))[0] for a, b in pairs]
                      + [self.features(a, np.array([b]))[0] for a, b in neg])
        y = np.concatenate([np.ones(len(pairs)), np.zeros(len(neg))])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xn = (X - self.mu) / self.sd
        w, b = np.zeros(X.shape[1]), 0.0
        for _ in range(epochs):                         # full-batch logistic regression
            p = 1 / (1 + np.exp(-(Xn @ w + b)))
            g = p - y
            w -= 0.5 * (Xn.T @ g / len(y) + 1e-3 * w)
            b -= 0.5 * g.mean()
        self.w, self.b = w, b
        return self

    def score(self, a: int, cands: np.ndarray) -> np.ndarray:
        return ((self.features(a, cands) - self.mu) / self.sd) @ self.w + self.b

    def complete_look(self, a: int, per_slot: int = 3, exclude: set | None = None) -> dict[str, list]:
        exclude = exclude or set()
        g = self.gender[a]
        gmask = np.ones(len(self.items), bool) if g == "unisex" else (
            (self.gender == g) | (self.gender == "unisex"))
        out = {}
        for slot in S.COMPLEMENTS[self.slot[a]]:
            cands = np.where(gmask & (self.slot == slot))[0]
            if not len(cands):
                continue
            sc = self.score(a, cands)
            order = np.argsort(-sc)
            picks, seen_sub = [], set()
            for j in order:                           # one per subcategory → a varied look
                c = int(cands[j])
                sub = self.items[c]["subcategory"]
                if c in exclude or sub in seen_sub:
                    continue
                picks.append((c, float(sc[j])))
                seen_sub.add(sub)
                if len(picks) == per_slot:
                    break
            out[slot] = picks
        return out

    def learned_color_harmony(self, top: int = 8) -> list[tuple[str, str, float]]:
        pairs = []
        n = len(S.COLORS)
        for i in range(n):
            for j in range(i + 1, n):
                if S.COLORS[i] in S.NEUTRALS or S.COLORS[j] in S.NEUTRALS:
                    continue
                pairs.append((S.COLORS[i], S.COLORS[j], float(self.color_pmi[i, j])))
        return sorted(pairs, key=lambda t: -t[2])[:top]
