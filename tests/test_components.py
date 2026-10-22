import numpy as np
import pytest

from drape.ann.hnsw import HNSW, mips_transform
from drape.eval.metrics import auc, mrr, ndcg_at_k, recall_at_k
from drape.rank.ltr import LambdaRankMLP
from drape.recs.als import ImplicitALS
from drape.search.query_parser import QueryParser
from drape.text.tokenize import tokenize

import scipy.sparse as sp


# ---------------------------------------------------------------- text & query understanding
def test_tokenize_normalises_and_stems():
    assert tokenize("T-Shirts for Men") == ["tshirt", "men"]
    assert tokenize("dresses") == ["dress"]
    assert tokenize("jeans") == tokenize("jean")


@pytest.mark.parametrize("q,lo,hi", [
    ("shirt under 2k", None, 2000), ("shoes below rs 1500", None, 1500),
    ("kurta between 1000 and 3000", 1000, 3000), ("jeans 800-1500", 800, 1500),
    ("watch above 5000", 5000, None),
])
def test_price_extraction(q, lo, hi):
    p = QueryParser().parse(q)
    assert (p.min_price, p.max_price) == (lo, hi)


def test_colour_synonyms_gender_and_occasion():
    p = QueryParser().parse("burgundy blazer for women for office")
    assert p.color == "maroon" and p.gender == "women" and "formal" in p.style_hints
    assert "burgundy" not in p.tokens and "blazer" in p.tokens


# ---------------------------------------------------------------- metrics
def test_metrics():
    assert ndcg_at_k([3, 2, 1, 0], 4) == pytest.approx(1.0)
    assert ndcg_at_k([0, 0, 3], 3) < ndcg_at_k([3, 0, 0], 3)
    assert ndcg_at_k([1], 10, all_rels=[1, 1]) < 1.0
    assert recall_at_k([1, 2, 3], {2, 9}, 3) == 0.5
    assert mrr([5, 6, 7], {7}) == pytest.approx(1 / 3)
    assert auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0


# ---------------------------------------------------------------- HNSW
def _clustered(n=3000, d=24, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(30, d))
    X = centers[rng.integers(30, size=n)] + 0.3 * rng.normal(size=(n, d))
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)


def test_hnsw_recall_and_work():
    X = _clustered()
    h = HNSW(X.shape[1], M=12, ef_construction=80).add(X)
    rng = np.random.default_rng(1)
    Q = X[rng.integers(len(X), size=100)] + 0.1 * rng.normal(size=(100, X.shape[1]))
    Q = (Q / np.linalg.norm(Q, axis=1, keepdims=True)).astype(np.float32)
    gt = np.argsort(-(Q @ X.T), axis=1)[:, :10]
    h.n_dist = 0
    rec = np.mean([len(set(h.query(q, 10, 64)[0]) & set(g)) / 10 for q, g in zip(Q, gt)])
    assert rec >= 0.95
    assert h.n_dist / len(Q) < 0.5 * len(X)            # far fewer distance evals than a scan


def test_mips_reduction_is_exact():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 8)) * rng.uniform(0.2, 3, size=(500, 1))
    q = rng.normal(size=(1, 8))
    Xa, qa = mips_transform(X, q)
    by_ip = np.argsort(-(X @ q[0]))[:5]
    by_l2 = np.argsort(((Xa - qa[0]) ** 2).sum(1))[:5]
    assert list(by_ip) == list(by_l2)


def test_hnsw_save_load(tmp_path):
    X = _clustered(500)
    h = HNSW(X.shape[1]).add(X)
    h.save(tmp_path / "h.pkl")
    h2 = HNSW.load(tmp_path / "h.pkl")
    assert list(h.query(X[3], 5)[0]) == list(h2.query(X[3], 5)[0])


# ---------------------------------------------------------------- ALS
def test_als_recovers_structure_and_fold_in():
    rng = np.random.default_rng(0)
    U, I = rng.normal(size=(300, 4)), rng.normal(size=(200, 4))
    P = (U @ I.T) > 1.5
    users, items = np.nonzero(P)
    W = sp.csr_matrix((np.ones(len(users)), (users, items)), shape=P.shape)
    m = ImplicitALS(factors=8, iterations=10, reg=0.1).fit(W)
    S = m.user_factors @ m.item_factors.T
    pos, neg = S[P], S[~P]
    assert auc(pos[:5000], neg[:5000]) > 0.9
    u = 7
    v = m.fold_in(np.nonzero(P[u])[0], np.full(P[u].sum(), 1.0))
    cos = v @ m.user_factors[u] / (np.linalg.norm(v) * np.linalg.norm(m.user_factors[u]))
    assert cos > 0.9


# ---------------------------------------------------------------- LambdaRank
def test_lambdarank_learns_nonlinear_relevance():
    rng = np.random.default_rng(0)
    G, n = 200, 20
    X = rng.normal(size=(G * n, 5))
    rel = np.clip(np.round(1.5 * X[:, 0] * X[:, 1] + X[:, 2]), 0, 3)   # interaction term
    groups = np.full(G, n)
    r = LambdaRankMLP(hidden=16, epochs=25, lr=1e-2).fit(X, rel, groups)
    learned = r.evaluate(X, rel, groups)
    rnd = LambdaRankMLP(); rnd.mu, rnd.sd = X.mean(0), X.std(0); rnd._init(5)
    rnd.params["Wl"] = rng.normal(size=(5, 1))
    assert learned > 0.85 and learned > rnd.evaluate(X, rel, groups) + 0.1
    imp = r.feature_importance(X)
    assert set(list(imp)[:3]) == {"f0", "f1", "f2"}
