"""Hierarchical Navigable Small World graph (Malkov & Yashunin, 2016), written from scratch.

- Multi-layer proximity graph; level ~ floor(-ln(U) * mL), mL = 1/ln(M)
- Greedy descent on upper layers, beam search (ef) on the target layer
- Neighbour selection uses the paper's diversity heuristic (Alg. 4) with keepPrunedConnections,
  which matters a lot on clustered data like product catalogs (many near-duplicate items)
- Metrics: 'cosine' (vectors pre-normalised) and 'l2'. Maximum-inner-product search is reduced to
  L2 search with the Bachrach et al. (2014) augmentation, see `mips_transform`.
- `self.n_dist` counts distance evaluations: a hardware-independent cost metric for benchmarks.
"""
from __future__ import annotations

import heapq
import math
import pickle

import numpy as np


def mips_transform(items: np.ndarray, queries: np.ndarray | None = None):
    """Append one coordinate so that argmax <q, x>  ==  argmin ||q' - x'||^2."""
    norms = np.linalg.norm(items, axis=1)
    big = norms.max()
    x_aug = np.hstack([items, np.sqrt(np.maximum(big ** 2 - norms ** 2, 0))[:, None]])
    if queries is None:
        return x_aug.astype(np.float32)
    q_aug = np.hstack([queries, np.zeros((len(queries), 1))])
    return x_aug.astype(np.float32), q_aug.astype(np.float32)


class HNSW:
    def __init__(self, dim: int, M: int = 16, ef_construction: int = 120, metric: str = "cosine",
                 seed: int = 0):
        assert metric in ("cosine", "l2")
        self.dim, self.M, self.M0 = dim, M, 2 * M
        self.efc, self.metric = ef_construction, metric
        self.mL = 1 / math.log(M)
        self.rng = np.random.default_rng(seed)
        self.X = np.empty((0, dim), np.float32)
        self.links: list[dict[int, list[int]]] = []   # links[level][node] -> neighbours
        self.levels: list[int] = []
        self.entry: int | None = None
        self.n_dist = 0

    # ---------- distances ----------
    def _d(self, q: np.ndarray, ids) -> np.ndarray:
        self.n_dist += len(ids)
        V = self.X[ids]
        if self.metric == "cosine":
            return 1.0 - V @ q
        diff = V - q
        return np.einsum("ij,ij->i", diff, diff)

    # ---------- core search on one layer ----------
    def _search_layer(self, q, eps: list[int], ef: int, level: int) -> list[tuple[float, int]]:
        d0 = self._d(q, eps)
        visited = set(eps)
        cand = [(float(d), e) for d, e in zip(d0, eps)]
        heapq.heapify(cand)
        res = [(-float(d), e) for d, e in zip(d0, eps)]
        heapq.heapify(res)
        while len(res) > ef:
            heapq.heappop(res)
        layer = self.links[level]
        while cand:
            d, c = heapq.heappop(cand)
            if d > -res[0][0] and len(res) >= ef:
                break
            nbrs = [n for n in layer.get(c, ()) if n not in visited]
            if not nbrs:
                continue
            visited.update(nbrs)
            for dn, n in zip(self._d(q, nbrs), nbrs):
                dn = float(dn)
                if len(res) < ef or dn < -res[0][0]:
                    heapq.heappush(cand, (dn, n))
                    heapq.heappush(res, (-dn, n))
                    if len(res) > ef:
                        heapq.heappop(res)
        return sorted((-d, n) for d, n in res)

    def _select(self, cands: list[tuple[float, int]], m: int) -> list[int]:
        """Diversity heuristic: keep e only if it is closer to the base than to any kept r."""
        kept: list[int] = []
        pruned: list[int] = []
        for d, e in cands:
            if len(kept) >= m:
                break
            if not kept or np.all(d < self._d(self.X[e], kept)):
                kept.append(e)
            else:
                pruned.append(e)
        for e in pruned:                       # keepPrunedConnections
            if len(kept) >= m:
                break
            kept.append(e)
        return kept

    # ---------- build ----------
    def add(self, vecs: np.ndarray) -> "HNSW":
        vecs = np.asarray(vecs, np.float32)
        start = len(self.X)
        self.X = np.vstack([self.X, vecs]) if start else vecs.copy()
        for i in range(start, start + len(vecs)):
            self._insert(i)
        return self

    def _insert(self, i: int) -> None:
        lvl = int(-math.log(1 - self.rng.random()) * self.mL)
        self.levels.append(lvl)
        while len(self.links) <= lvl:
            self.links.append({})
        if self.entry is None:
            for l in range(lvl + 1):
                self.links[l][i] = []
            self.entry = i
            return
        q = self.X[i]
        ep = [self.entry]
        top = self.levels[self.entry]
        for l in range(top, lvl, -1):
            ep = [self._search_layer(q, ep, 1, l)[0][1]]
        for l in range(min(lvl, top), -1, -1):
            W = self._search_layer(q, ep, self.efc, l)
            mmax = self.M0 if l == 0 else self.M
            nbrs = self._select(W, self.M)
            self.links[l][i] = nbrs
            for n in nbrs:
                ln = self.links[l][n]
                ln.append(i)
                if len(ln) > mmax:
                    dd = self._d(self.X[n], ln)
                    self.links[l][n] = self._select(sorted(zip(dd.tolist(), ln)), mmax)
            ep = [n for _, n in W]
        for l in range(top + 1, lvl + 1):
            self.links[l][i] = []
        if lvl > top:
            self.entry = i

    # ---------- query ----------
    def query(self, q: np.ndarray, k: int = 10, ef: int = 64) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q, np.float32)
        ep = [self.entry]
        for l in range(self.levels[self.entry], 0, -1):
            ep = [self._search_layer(q, ep, 1, l)[0][1]]
        W = self._search_layer(q, ep, max(ef, k), 0)[:k]
        return np.array([n for _, n in W]), np.array([d for d, _ in W], np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "HNSW":
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            obj.__dict__.update(pickle.load(f))
        return obj
