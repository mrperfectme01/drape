"""Implicit-feedback matrix factorisation (Hu, Koren & Volinsky, 2008) with a CG solver.

Confidence c_ui = 1 + alpha * log(1 + w_ui / eps), where w_ui sums event weights
(view=1, cart=4, purchase=10). Each half-step solves, per user,
    (YᵀY + Yᵤᵀ(Cᵤ - I)Yᵤ + λI) xᵤ = Yᵤᵀ Cᵤ pᵤ
with a few conjugate-gradient iterations warm-started from the previous solution
(Takács et al., 2011), so cost is O(nnz·k + (m+n)·k²) per epoch instead of O((m+n)·k³).

`fold_in` computes a brand-new user's vector from a handful of live events in ~1 ms with
item factors frozen; that's what makes recommendations react inside a session.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

EVENT_WEIGHTS = np.array([1.0, 4.0, 10.0], np.float32)   # view, cart, purchase


def interaction_matrix(users, items, etypes, n_users, n_items) -> sp.csr_matrix:
    w = EVENT_WEIGHTS[etypes]
    return sp.csr_matrix((w, (users, items)), shape=(n_users, n_items), dtype=np.float32)


class ImplicitALS:
    def __init__(self, factors: int = 64, reg: float = 0.05, alpha: float = 8.0, eps: float = 1.0,
                 iterations: int = 15, cg_steps: int = 3, seed: int = 0):
        self.k, self.reg, self.alpha, self.eps = factors, reg, alpha, eps
        self.iterations, self.cg_steps, self.seed = iterations, cg_steps, seed
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.loss_history: list[float] = []

    def confidence(self, W: sp.csr_matrix) -> sp.csr_matrix:
        C = W.copy()
        C.data = 1.0 + self.alpha * np.log1p(C.data / self.eps)
        return C

    def _half_step(self, C: sp.csr_matrix, X: np.ndarray, Y: np.ndarray) -> None:
        YtY = Y.T @ Y + self.reg * np.eye(self.k, dtype=Y.dtype)
        indptr, indices, data = C.indptr, C.indices, C.data
        for u in range(X.shape[0]):
            lo, hi = indptr[u], indptr[u + 1]
            if lo == hi:
                X[u] = 0
                continue
            idx, c = indices[lo:hi], data[lo:hi]
            Yu = Y[idx]
            x = X[u]
            # r = b - A x ; b = Yu^T c ; A x = YtY x + Yu^T ((c-1) * (Yu x))
            r = Yu.T @ c - (YtY @ x + Yu.T @ ((c - 1) * (Yu @ x)))
            p = r.copy()
            rs = r @ r
            for _ in range(self.cg_steps):
                if rs < 1e-10:
                    break
                Ap = YtY @ p + Yu.T @ ((c - 1) * (Yu @ p))
                a = rs / (p @ Ap)
                x = x + a * p
                r = r - a * Ap
                rs_new = r @ r
                p = r + (rs_new / rs) * p
                rs = rs_new
            X[u] = x

    def fit(self, W: sp.csr_matrix, verbose: bool = False) -> "ImplicitALS":
        rng = np.random.default_rng(self.seed)
        n_u, n_i = W.shape
        C = self.confidence(W).astype(np.float32)
        Ct = C.T.tocsr()
        self.user_factors = (rng.standard_normal((n_u, self.k)) * 0.01).astype(np.float32)
        self.item_factors = (rng.standard_normal((n_i, self.k)) * 0.01).astype(np.float32)
        for it in range(self.iterations):
            self._half_step(C, self.user_factors, self.item_factors)
            self._half_step(Ct, self.item_factors, self.user_factors)
            if verbose:
                self.loss_history.append(self.loss(C))
                print(f"  ALS iter {it + 1:2d} loss={self.loss_history[-1]:.4f}")
        self._YtY = None
        return self

    def loss(self, C: sp.csr_matrix) -> float:
        """Weighted squared loss over observed cells + regularisation (unobserved cells omitted)."""
        X, Y = self.user_factors, self.item_factors
        coo = C.tocoo()
        pred = np.einsum("ij,ij->i", X[coo.row], Y[coo.col])
        obs = (coo.data * (1 - pred) ** 2).sum()
        return float((obs + self.reg * ((X ** 2).sum() + (Y ** 2).sum())) / C.nnz)

    def fold_in(self, items: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Exact least-squares user vector for an unseen/updated user, item factors frozen."""
        Y = self.item_factors
        if len(items) == 0:
            return np.zeros(self.k, np.float32)
        agg: dict[int, float] = {}
        for i, w in zip(items, weights):
            agg[int(i)] = agg.get(int(i), 0.0) + float(w)
        idx = np.fromiter(agg.keys(), int)
        c = 1.0 + self.alpha * np.log1p(np.fromiter(agg.values(), float) / self.eps)
        Yu = Y[idx]
        if getattr(self, "_YtY", None) is None:
            self._YtY = Y.T @ Y
        A = self._YtY + (Yu.T * (c - 1)) @ Yu + self.reg * np.eye(self.k)
        b = Yu.T @ c
        return np.linalg.solve(A, b).astype(np.float32)
