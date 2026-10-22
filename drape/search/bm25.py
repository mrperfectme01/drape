"""BM25F-lite inverted index on a sparse matrix.

Fields are weighted at indexing time (title x2, attributes x1.5, description x1), and every
(term, doc) BM25 weight is precomputed into a CSR matrix, so query scoring is a sparse
row-sum over the query's terms: O(postings), no Python loop over documents.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import scipy.sparse as sp

from ..text.tokenize import tokenize

FIELD_WEIGHTS = {"title": 2.0, "attrs": 1.5, "description": 1.0}


def item_fields(it: dict) -> dict[str, str]:
    attrs = " ".join([it["subcategory"], it["color"], it["material"], it["style"], it["gender"],
                      it["pattern"], it["fit"], it["season"], it["brand"], it["slot"]])
    return {"title": it["title"], "attrs": attrs, "description": it["description"]}


class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.vocab: dict[str, int] = {}
        self.W: sp.csr_matrix | None = None   # (n_terms, n_docs) BM25 weights
        self.idf: np.ndarray | None = None

    def fit(self, items: list[dict]) -> "BM25Index":
        rows, cols, vals, doclen = [], [], [], np.zeros(len(items))
        for d, it in enumerate(items):
            tf: Counter = Counter()
            for field, text in item_fields(it).items():
                for tok in tokenize(text):
                    tf[tok] += FIELD_WEIGHTS[field]
            doclen[d] = sum(tf.values())
            for tok, f in tf.items():
                rows.append(self.vocab.setdefault(tok, len(self.vocab)))
                cols.append(d)
                vals.append(f)
        n_docs = len(items)
        tf_m = sp.csr_matrix((vals, (rows, cols)), shape=(len(self.vocab), n_docs))
        df = np.diff(tf_m.indptr)
        self.idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        norm = self.k1 * (1 - self.b + self.b * doclen / doclen.mean())
        tf_m = tf_m.tocoo()
        w = tf_m.data * (self.k1 + 1) / (tf_m.data + norm[tf_m.col]) * self.idf[tf_m.row]
        self.W = sp.csr_matrix((w, (tf_m.row, tf_m.col)), shape=tf_m.shape)
        return self

    def term_ids(self, tokens: list[str]) -> list[int]:
        return [self.vocab[t] for t in tokens if t in self.vocab]

    def score(self, tokens: list[str], weights: list[float] | None = None) -> np.ndarray:
        """Dense score vector over all docs (fast: sparse slice + sum)."""
        n = self.W.shape[1]
        ids = [(self.vocab[t], (weights[i] if weights else 1.0))
               for i, t in enumerate(tokens) if t in self.vocab]
        if not ids:
            return np.zeros(n, dtype=np.float32)
        rows = np.array([i for i, _ in ids])
        wq = np.array([w for _, w in ids])
        return np.asarray(self.W[rows].T @ wq).ravel().astype(np.float32)

    def search(self, query: str, k: int = 100) -> tuple[np.ndarray, np.ndarray]:
        s = self.score(tokenize(query))
        k = min(k, len(s))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        top = top[s[top] > 0]
        return top, s[top]
