"""Dense text retrieval: TF-IDF over words + char trigrams, compressed by truncated SVD (LSA).

Char trigrams make retrieval typo-tolerant ("sneekers" still lands near "sneakers"); the SVD
projection folds co-occurring terms together so "office" pulls up "formal"/"blazer" items.
Vectors are L2-normalised so inner product == cosine and they can go straight into HNSW.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from ..text.tokenize import tokenize
from .bm25 import item_fields


def _features(text: str) -> Counter:
    toks = tokenize(text)
    feats: Counter = Counter(toks)
    feats.update(f"{a}_{b}" for a, b in zip(toks, toks[1:]))
    for t in toks:
        w = f"#{t}#"
        feats.update("c:" + w[i:i + 3] for i in range(len(w) - 2))
    return feats


class LSAEncoder:
    def __init__(self, dim: int = 96, min_df: int = 2):
        self.dim, self.min_df = dim, min_df
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.proj: np.ndarray | None = None      # (n_feats, dim)

    def _matrix(self, texts: list[str], grow: bool) -> sp.csr_matrix:
        rows, cols, vals = [], [], []
        for r, t in enumerate(texts):
            for f, c in _features(t).items():
                j = self.vocab.get(f)
                if j is None and grow:
                    j = self.vocab[f] = len(self.vocab)
                if j is not None:
                    rows.append(r); cols.append(j); vals.append(1 + np.log(c))
        return sp.csr_matrix((vals, (rows, cols)), shape=(len(texts), len(self.vocab)))

    def fit_transform(self, items: list[dict]) -> np.ndarray:
        texts = [" ".join(item_fields(it).values()) for it in items]
        X = self._matrix(texts, grow=True)
        df = np.bincount(X.indices, minlength=X.shape[1])
        keep = np.where(df >= self.min_df)[0]
        remap = {f: int(np.searchsorted(keep, j)) for f, j in self.vocab.items() if df[j] >= self.min_df}
        X = X[:, keep]
        self.vocab = remap
        df = df[keep]
        self.idf = np.log((1 + X.shape[0]) / (1 + df)) + 1
        X = self._l2(X.multiply(self.idf).tocsr())
        _, _, vt = svds(X.astype(np.float64), k=self.dim, random_state=0)
        self.proj = vt.T.astype(np.float32)
        return self._norm(np.asarray(X @ self.proj, dtype=np.float32))

    def encode(self, texts: list[str]) -> np.ndarray:
        X = self._l2(self._matrix(texts, grow=False).multiply(self.idf).tocsr())
        return self._norm(np.asarray(X @ self.proj, dtype=np.float32))

    @staticmethod
    def _l2(X: sp.csr_matrix) -> sp.csr_matrix:
        n = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel()) + 1e-9
        return sp.diags(1 / n) @ X

    @staticmethod
    def _norm(V: np.ndarray) -> np.ndarray:
        return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
