"""Query understanding: turn "navy linen shirt for office under 2k for men" into structure.

Two parts:
1. Deterministic slot extraction (price ranges, colours + synonyms, gender, occasion -> style)
2. A *learned* product-type classifier: multinomial Naive Bayes over query tokens, trained on
   search logs where the label is the subcategory of the item the user carted/bought after the
   query. It learns e.g. "sweatshirt" -> hoodie, "khakis" -> chinos, and that "bag" is split
   between handbag and backpack, with no hand-written synonym table.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..data import schema as S
from ..text.tokenize import normalize, tokenize

_NUM = r"(\d+(?:\.\d+)?)\s*(k)?"
PRICE_PATTERNS = [
    (re.compile(rf"between\s+(?:rs\.?\s*)?{_NUM}\s+(?:and|to|-)\s+(?:rs\.?\s*)?{_NUM}"), "range"),
    (re.compile(rf"(?:rs\.?\s*)?{_NUM}\s*(?:-|to)\s*(?:rs\.?\s*)?{_NUM}"), "range"),
    (re.compile(rf"(?:under|below|less than|within|upto|up to|max|<)\s*(?:rs\.?\s*)?{_NUM}"), "max"),
    (re.compile(rf"(?:above|over|more than|min|>)\s*(?:rs\.?\s*)?{_NUM}"), "min"),
]
GENDER_WORDS = {"men": "men", "man": "men", "mens": "men", "male": "men", "boys": "men",
                "boy": "men", "gents": "men", "him": "men", "women": "women", "woman": "women",
                "womens": "women", "female": "women", "ladies": "women", "girls": "women",
                "girl": "women", "her": "women"}
OCCASION_TO_STYLE = {occ: st for st, occs in S.OCCASIONS.items() for occ in occs}
OCCASION_TO_STYLE.update({"interview": "formal", "college": "casual", "weekend": "casual",
                          "wedding": "ethnic", "festival": "ethnic", "sport": "sporty",
                          "yoga": "sporty", "street": "streetwear", "clubbing": "party"})


def _num(v: str, k: str | None) -> float:
    return float(v) * (1000 if k else 1)


@dataclass
class ParsedQuery:
    raw: str
    tokens: list[str]
    color: str | None = None
    min_price: float | None = None
    max_price: float | None = None
    gender: str | None = None
    style_hints: list[str] = field(default_factory=list)
    subcat_probs: dict[str, float] = field(default_factory=dict)

    @property
    def top_subcat(self) -> tuple[str | None, float]:
        if not self.subcat_probs:
            return None, 0.0
        k = max(self.subcat_probs, key=self.subcat_probs.get)
        return k, self.subcat_probs[k]

    def as_dict(self) -> dict:
        sc, p = self.top_subcat
        return {"color": self.color, "min_price": self.min_price, "max_price": self.max_price,
                "gender": self.gender, "style_hints": self.style_hints,
                "category": sc, "category_confidence": round(p, 3),
                "category_dist": {k: round(v, 3) for k, v in sorted(
                    self.subcat_probs.items(), key=lambda kv: -kv[1])[:3]},
                "keywords": self.tokens}


class QueryClassifier:
    """Multinomial NB: P(subcat | query tokens + bigrams)."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.classes: list[str] = []
        self.log_prior: np.ndarray | None = None
        self.log_lik: dict[str, np.ndarray] = {}
        self.unk: np.ndarray | None = None

    @staticmethod
    def feats(text: str) -> list[str]:
        t = tokenize(text)
        return t + [f"{a}_{b}" for a, b in zip(t, t[1:])]

    def fit(self, texts: list[str], labels: list[str]) -> "QueryClassifier":
        self.classes = sorted(set(labels))
        ci = {c: i for i, c in enumerate(self.classes)}
        C = len(self.classes)
        counts: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(C))
        prior = np.zeros(C)
        for t, y in zip(texts, labels):
            prior[ci[y]] += 1
            for f in self.feats(t):
                counts[f][ci[y]] += 1
        totals = sum(counts.values())
        V = len(counts)
        denom = totals + self.alpha * V
        self.log_prior = np.log(prior / prior.sum())
        self.log_lik = {f: np.log((c + self.alpha) / denom) for f, c in counts.items()}
        self.unk = np.log(self.alpha / denom)
        return self

    def predict_proba(self, text: str) -> dict[str, float]:
        fs = [f for f in self.feats(text) if f in self.log_lik]
        if not fs:
            return {}
        lp = self.log_prior + sum(self.log_lik[f] for f in fs)
        p = np.exp(lp - lp.max())
        p /= p.sum()
        return {c: float(v) for c, v in zip(self.classes, p) if v > 1e-3}


class QueryParser:
    def __init__(self, classifier: QueryClassifier | None = None):
        self.clf = classifier
        self.colors = sorted(list(S.COLORS) + list(S.COLOR_SYNONYMS), key=len, reverse=True)

    def parse(self, q: str) -> ParsedQuery:
        text = q.lower().replace(",", "").replace("₹", "rs ")
        out = ParsedQuery(raw=q, tokens=[])
        for pat, kind in PRICE_PATTERNS:
            m = pat.search(text)
            if not m:
                continue
            g = m.groups()
            if kind == "range":
                lo, hi = sorted((_num(g[0], g[1]), _num(g[2], g[3])))
                if hi >= 100:
                    out.min_price, out.max_price = lo, hi
            elif kind == "max":
                out.max_price = _num(*g)
            else:
                out.min_price = _num(*g)
            text = text[: m.start()] + " " + text[m.end():]
        text = normalize(text)
        for c in self.colors:
            if re.search(rf"\b{re.escape(c)}\b", text):
                out.color = S.COLOR_SYNONYMS.get(c, c)
                text = re.sub(rf"\b{re.escape(c)}\b", " ", text, count=1)
                break
        words = re.findall(r"[a-z0-9]+", text)
        for w in words:
            if w in GENDER_WORDS:
                out.gender = GENDER_WORDS[w]
        joined = " ".join(words)
        for occ, st in OCCASION_TO_STYLE.items():
            if re.search(rf"\b{occ}\b", joined) and st not in out.style_hints:
                out.style_hints.append(st)
        for st in S.STYLES:
            if st in words and st not in out.style_hints:
                out.style_hints.append(st)
        out.tokens = [t for t in tokenize(" ".join(w for w in words if w not in GENDER_WORDS))]
        if self.clf is not None:
            out.subcat_probs = self.clf.predict_proba(" ".join(out.tokens))
        return out
