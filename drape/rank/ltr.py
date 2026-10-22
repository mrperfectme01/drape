"""LambdaRank with a small MLP scorer, implemented in NumPy (manual backprop + Adam).

For each query group, every pair (i, j) with rel_i > rel_j contributes
    lambda_ij = -|ΔNDCG_ij| * sigmoid(-(s_i - s_j))
i.e. the RankNet gradient re-weighted by how much swapping the pair would change NDCG@k
(Burges, 2010). Gradients are summed per document and back-propagated through the MLP.
"""
from __future__ import annotations

import pickle

import numpy as np


def dcg_gains(rel: np.ndarray) -> np.ndarray:
    return 2.0 ** rel - 1


class LambdaRankMLP:
    def __init__(self, hidden: int = 32, lr: float = 3e-3, epochs: int = 12, k: int = 10,
                 l2: float = 1e-4, seed: int = 0):
        self.hidden, self.lr, self.epochs, self.k, self.l2 = hidden, lr, epochs, k, l2
        self.seed = seed
        self.mu = self.sd = None
        self.params: dict[str, np.ndarray] = {}
        self.feature_names: list[str] = []
        self.history: list[float] = []

    # ----- model -----
    def _init(self, d: int) -> None:
        rng = np.random.default_rng(self.seed)
        self.params = {
            "W1": rng.standard_normal((d, self.hidden)) * np.sqrt(2 / d),
            "b1": np.zeros(self.hidden),
            "W2": rng.standard_normal((self.hidden, 1)) * np.sqrt(1 / self.hidden),
            "b2": np.zeros(1),
            "Wl": np.zeros((d, 1)),   # linear skip connection: keeps a strong linear baseline
        }

    def _forward(self, X: np.ndarray):
        P = self.params
        h_pre = X @ P["W1"] + P["b1"]
        h = np.maximum(h_pre, 0)
        s = (h @ P["W2"] + P["b2"] + X @ P["Wl"]).ravel()
        return s, (X, h_pre, h)

    def _backward(self, cache, g: np.ndarray) -> dict[str, np.ndarray]:
        X, h_pre, h = cache
        P = self.params
        g = g[:, None]
        grads = {"W2": h.T @ g, "b2": g.sum(0), "Wl": X.T @ g}
        dh = (g @ P["W2"].T) * (h_pre > 0)
        grads["W1"] = X.T @ dh
        grads["b1"] = dh.sum(0)
        for k in ("W1", "W2", "Wl"):
            grads[k] += self.l2 * P[k]
        return grads

    # ----- lambda gradients for one group -----
    def _lambdas(self, s: np.ndarray, rel: np.ndarray) -> np.ndarray:
        n = len(s)
        order = np.argsort(-s)
        rank = np.empty(n, int)
        rank[order] = np.arange(n)
        disc = 1 / np.log2(rank + 2)
        disc[rank >= self.k] = 0.0     # NDCG@k: positions past k contribute nothing
        ideal = np.sort(dcg_gains(rel))[::-1][: self.k]
        idcg = (ideal / np.log2(np.arange(2, len(ideal) + 2))).sum()
        if idcg == 0:
            return np.zeros(n)
        gains = dcg_gains(rel)
        dG = gains[:, None] - gains[None, :]
        dD = disc[:, None] - disc[None, :]
        delta = np.abs(dG * dD) / idcg
        mask = rel[:, None] > rel[None, :]
        sdiff = s[:, None] - s[None, :]
        rho = 1 / (1 + np.exp(np.clip(sdiff, -30, 30)))     # sigmoid(-(si - sj))
        lam = np.where(mask, -rho * (delta + 1e-3), 0.0)     # small floor keeps far-pairs learning
        return lam.sum(1) - lam.sum(0)                       # dLoss/ds per document

    def fit(self, X: np.ndarray, rel: np.ndarray, groups: np.ndarray,
            feature_names: list[str] | None = None, verbose: bool = False) -> "LambdaRankMLP":
        """groups: array of group sizes (rows of X are contiguous per group)."""
        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xn = (X - self.mu) / self.sd
        self._init(X.shape[1])
        bounds = np.concatenate([[0], np.cumsum(groups)])
        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v = {k: np.zeros_like(v) for k, v in self.params.items()}
        rng = np.random.default_rng(self.seed)
        step, batch = 0, 32
        for ep in range(self.epochs):
            perm = rng.permutation(len(groups))
            for b0 in range(0, len(perm), batch):
                gs = perm[b0:b0 + batch]
                rows = np.concatenate([np.arange(bounds[g], bounds[g + 1]) for g in gs])
                s, cache = self._forward(Xn[rows])
                gvec = np.empty(len(rows))
                off = 0
                for g in gs:
                    n = bounds[g + 1] - bounds[g]
                    gvec[off:off + n] = self._lambdas(s[off:off + n], rel[bounds[g]:bounds[g + 1]])
                    off += n
                grads = self._backward(cache, gvec / len(gs))
                step += 1
                for k in self.params:
                    m[k] = 0.9 * m[k] + 0.1 * grads[k]
                    v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
                    mh, vh = m[k] / (1 - 0.9 ** step), v[k] / (1 - 0.999 ** step)
                    self.params[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)
            if verbose:
                nd = self.evaluate(X, rel, groups)
                self.history.append(nd)
                print(f"  LTR epoch {ep + 1:2d} train NDCG@{self.k}={nd:.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._forward((X - self.mu) / self.sd)[0]

    def evaluate(self, X, rel, groups) -> float:
        from ..eval.metrics import ndcg_at_k
        s = self.predict(X)
        bounds = np.concatenate([[0], np.cumsum(groups)])
        vals = []
        for g in range(len(groups)):
            a, b = bounds[g], bounds[g + 1]
            order = np.argsort(-s[a:b])
            vals.append(ndcg_at_k(rel[a:b][order], self.k))
        return float(np.mean(vals))

    def feature_importance(self, X: np.ndarray) -> dict[str, float]:
        """Permutation-free sensitivity: mean |∂score/∂x| * std(x) over a sample."""
        Xn = (X - self.mu) / self.sd
        _, (_, h_pre, _) = self._forward(Xn)
        P = self.params
        gate = (h_pre > 0).astype(float)                           # (n, H)
        grad = (gate * P["W2"].ravel()) @ P["W1"].T + P["Wl"].ravel()  # (n, d) wrt normalised x
        imp = np.abs(grad).mean(0) * Xn.std(0)
        imp = imp / imp.sum()
        return dict(sorted(zip(self.feature_names, imp.round(4).tolist()), key=lambda kv: -kv[1]))

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: str) -> "LambdaRankMLP":
        o = cls.__new__(cls)
        with open(path, "rb") as f:
            o.__dict__.update(pickle.load(f))
        return o
