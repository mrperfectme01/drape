"""Offline training-data construction and evaluation with a strict temporal protocol.

    day 0 ───────────── T1=90 ──────── T2=105 ──────── 120
         fit models (E1)     │ rank-train │
                     fit models (E2)      │   test     │

Rankers learn on [T1, T2) using features from models frozen at T1 (no leakage: ALS never saw
the labels). Test metrics are on [T2, 120) with features from models frozen at T2 and the
rankers trained before. Real-time session context only ever uses events *before* the query.
"""
from __future__ import annotations

import copy
from collections import defaultdict

import numpy as np

from ..data import schema as S
from ..data.generator import ETYPE_BUY, ETYPE_CART
from ..engine import Engine
from ..rank.features import REC_FEATURES, SEARCH_FEATURES, rec_features, search_features
from ..recs.covisit import covisit_scores
from ..recs.outfit import complementary_pairs
from .metrics import auc, hit_at_k, mrr, ndcg_at_k, recall_at_k


class EventLog:
    """Per-user, time-sorted access to the raw event stream."""

    def __init__(self, events: dict):
        o = np.lexsort((events["ts"], events["user"]))
        self.u, self.i = events["user"][o], events["item"][o]
        self.e, self.t = events["etype"][o], events["ts"][o]
        self.s = events["session"][o]
        self.bounds = np.searchsorted(self.u, np.arange(self.u.max() + 2))
        self.by_session: dict[int, dict[int, int]] = defaultdict(dict)
        for s, i, e in zip(events["session"], events["item"], events["etype"]):
            d = self.by_session[int(s)]
            d[int(i)] = max(d.get(int(i), -1), int(e) + 1)       # grade: view1 cart2 buy3

    def context_slice(self, user: int, ts: float, n: int = 20) -> slice:
        a, b = self.bounds[user], self.bounds[user + 1]
        # events are float32: a session's first view can round to just below the query ts,
        # so look strictly before a small guard band (sessions are hours apart)
        j = a + np.searchsorted(self.t[a:b], ts - 1e-3)
        return slice(max(a, j - n), j)

    def recent_before(self, user: int, ts: float, n: int = 20) -> list[int]:
        return [int(x) for x in self.i[self.context_slice(user, ts, n)]]

    def window(self, user: int, lo: float, hi: float):
        a, b = self.bounds[user], self.bounds[user + 1]
        t = self.t[a:b]
        m = (t >= lo) & (t < hi)
        return self.i[a:b][m], self.e[a:b][m]


def _ctx_at(engine: Engine, log: EventLog, user: int, ts: float):
    ctx = copy.copy(engine.user(user))
    ctx.recent = log.recent_before(user, ts)
    return ctx


# --------------------------------------------------------------------------- search
def search_groups(engine: Engine, log: EventLog, queries, lo, hi, max_neg=60, seed=0,
                  max_queries=None):
    rng = np.random.default_rng(seed)
    Xs, ys, gs = [], [], []
    qs = [q for q in queries if lo <= q["ts"] < hi]
    if max_queries:
        qs = [qs[i] for i in rng.choice(len(qs), min(max_queries, len(qs)), replace=False)]
    for q in qs:
        grades = log.by_session.get(q["session"], {})
        if not grades:
            continue
        pq = engine.parser.parse(q["text"])
        cands, f = engine.search_candidates(pq, now=q["ts"])
        if not len(cands):
            continue
        y = np.array([grades.get(int(c), 0) for c in cands], float)
        if y.max() == 0:
            continue
        pos, neg = np.flatnonzero(y > 0), np.flatnonzero(y == 0)
        if len(neg) > max_neg:
            neg = rng.choice(neg, max_neg, replace=False)
        sel = np.concatenate([pos, neg])
        ctx = _ctx_at(engine, log, q["user"], q["ts"])
        cov = covisit_scores(engine.covisit, ctx.recent)
        X = search_features(engine.it, cands, bm25=f["bm25"], sem=f["sem"], rrf=f["rrf"],
                            subcat_probs=pq.subcat_probs, color=pq.color, color_ix=f["color_ix"],
                            style_hints_ix=[S.STYLES.index(s) for s in pq.style_hints], u=ctx,
                            item_factors=engine.als.item_factors, covisit=cov, now=q["ts"])
        Xs.append(X[sel]); ys.append(y[sel]); gs.append(len(sel))
    return np.vstack(Xs), np.concatenate(ys), np.array(gs)


def eval_search(engine: Engine, log: EventLog, queries, lo, hi, modes=("bm25", "hybrid", "ltr"),
                k=10, max_queries=3000, seed=1) -> dict:
    rng = np.random.default_rng(seed)
    qs = [q for q in queries if lo <= q["ts"] < hi and log.by_session.get(q["session"])]
    qs = [qs[i] for i in rng.choice(len(qs), min(max_queries, len(qs)), replace=False)]
    res = {m: defaultdict(list) for m in modes}
    for q in qs:
        grades = log.by_session[q["session"]]
        rel_all = list(grades.values())
        strong = {i for i, g in grades.items() if g >= 2}
        ctx = _ctx_at(engine, log, q["user"], q["ts"])
        for m in modes:
            r = engine.search(q["text"], now=q["ts"], ctx=ctx, mode=m, k=k)
            ranked = r["ids"]
            rels = [grades.get(i, 0) for i in ranked]
            res[m]["ndcg@10"].append(ndcg_at_k(rels, k, rel_all))
            res[m]["recall@10"].append(recall_at_k(ranked, set(grades), k))
            if strong:
                res[m]["mrr_purchase_or_cart"].append(mrr(ranked, strong, k))
            if m != "bm25":
                res[m]["latency_ms"].append(r["timings_ms"]["total"])
    out = {}
    for m, d in res.items():
        out[m] = {key: float(np.mean(v)) for key, v in d.items() if key != "latency_ms"}
        if d.get("latency_ms"):
            out[m]["p50_ms"] = float(np.percentile(d["latency_ms"], 50))
            out[m]["p99_ms"] = float(np.percentile(d["latency_ms"], 99))
    out["n_queries"] = len(qs)
    return out


def eval_query_classifier(engine: Engine, log: EventLog, queries, lo, hi) -> dict:
    hit1 = hit3 = n = 0
    for q in queries:
        if not (lo <= q["ts"] < hi):
            continue
        g = log.by_session.get(q["session"])
        if not g:
            continue
        label = engine.items[max(g, key=g.get)]["subcategory"]
        probs = engine.parser.parse(q["text"]).subcat_probs
        top = sorted(probs, key=probs.get, reverse=True)
        hit1 += bool(top[:1] == [label]); hit3 += label in top[:3]; n += 1
    return {"top1_acc": hit1 / n, "top3_acc": hit3 / n, "n": n}


# --------------------------------------------------------------------------- recommendations
def rec_groups(engine: Engine, log: EventLog, users, lo, hi, max_neg=80, seed=0):
    rng = np.random.default_rng(seed)
    Xs, ys, gs = [], [], []
    for u in users:
        it, et = log.window(u, lo, hi)
        if not len(it):
            continue
        grades: dict[int, int] = {}
        for i, e in zip(it, et):
            grades[int(i)] = max(grades.get(int(i), 0), int(e) + 1)
        ctx = engine.user(u)
        cands, als_s, cov, pop_mask = engine.rec_candidates(ctx, lo)
        y = np.array([grades.get(int(c), 0) for c in cands], float)
        if y.max() == 0:
            continue
        pos, neg = np.flatnonzero(y > 0), np.flatnonzero(y == 0)
        if len(neg) > max_neg:
            neg = rng.choice(neg, max_neg, replace=False)
        sel = np.concatenate([pos, neg])
        X = rec_features(engine.it, cands, als_scores=als_s, covisit=cov, popular_mask=pop_mask,
                         u=ctx, now=lo)
        Xs.append(X[sel]); ys.append(y[sel]); gs.append(len(sel))
    return np.vstack(Xs), np.concatenate(ys), np.array(gs)


def eval_recs(engine: Engine, log: EventLog, users, lo, hi,
              modes=("popular", "covisit", "als", "ltr"), k=20) -> dict:
    res = {m: defaultdict(list) for m in modes}
    n = 0
    for u in users:
        it, et = log.window(u, lo, hi)
        strong = {int(i) for i, e in zip(it, et) if e >= ETYPE_CART}
        ctx = engine.user(u)
        strong -= ctx.purchased
        if not strong:
            continue
        n += 1
        for m in modes:
            r = engine.recommend(u, k=k, now=lo, mode=m)
            ranked = r["ids"]
            rels = [1.0 if i in strong else 0.0 for i in ranked]
            res[m]["recall@20"].append(recall_at_k(ranked, strong, k))
            res[m]["ndcg@20"].append(ndcg_at_k(rels, k, [1.0] * len(strong)))
            res[m]["hit@10"].append(hit_at_k(ranked, strong, 10))
            res[m]["latency_ms"].append(r["timings_ms"]["total"])
    out = {}
    for m, d in res.items():
        out[m] = {key: float(np.mean(v)) for key, v in d.items() if key != "latency_ms"}
        out[m]["p50_ms"] = float(np.percentile(d["latency_ms"], 50))
        out[m]["p99_ms"] = float(np.percentile(d["latency_ms"], 99))
    out["n_users"] = n
    return out


# --------------------------------------------------------------------------- outfits
def eval_outfit(engine: Engine, events, lo, hi, n_pairs=2000, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    m = (events["ts"] >= lo) & (events["ts"] < hi)
    pairs = complementary_pairs(engine.items, events["session"][m], events["item"][m],
                                events["etype"][m])
    if len(pairs) > n_pairs:
        pairs = pairs[rng.choice(len(pairs), n_pairs, replace=False)]
    om = engine.outfit
    pos, neg, hits, rand_hits = [], [], [], []
    for a, b in pairs:
        g = om.gender[a]
        gm = np.ones(len(om.items), bool) if g == "unisex" else (om.gender == g) | (om.gender == "unisex")
        pool = np.where(gm & (om.slot == om.slot[b]))[0]
        if len(pool) < 10 or not gm[b]:            # cross-gender pair from an 'any' shopper
            continue
        sc = om.score(a, pool)
        sb = om.score(a, np.array([b]))[0]
        pos.append(sb)
        neg.append(sc[rng.integers(len(sc))])
        rank = (sc > sb).sum()
        hits.append(rank < 50)
        rand_hits.append(50 / len(pool))
    planted = {tuple(sorted(p)) for p in S.HARMONY                  # neutral pairs are excluded
               if p[0] not in S.NEUTRALS and p[1] not in S.NEUTRALS}  # from the learned ranking
    learned = om.learned_color_harmony(top=len(planted))
    found = sum(tuple(sorted((a, b))) in planted for a, b, _ in learned)
    return {"auc": auc(np.array(pos), np.array(neg)), "hit@50": float(np.mean(hits)),
            "random_hit@50": float(np.mean(rand_hits)), "n_pairs": len(pos),
            "color_harmony_precision": found / len(planted),
            "learned_top_pairs": [(a, b, round(v, 3)) for a, b, v in learned]}


# --------------------------------------------------------------------------- ANN
def eval_ann(engine: Engine, queries, n=300, seed=0) -> dict:
    import time
    rng = np.random.default_rng(seed)
    qs = [queries[i]["text"] for i in rng.choice(len(queries), n, replace=False)]
    Q = engine.text.lsa.encode(qs)
    E = engine.text.emb
    h = engine.text.hnsw
    gt = np.argsort(-(Q @ E.T), axis=1)[:, :10]
    t = time.perf_counter()
    for q in Q:
        s = E @ q
        np.argpartition(-s, 10)[:10]
    brute_ms = (time.perf_counter() - t) / n * 1e3
    rows = []
    for ef in (10, 16, 32, 64, 128, 256):
        h.n_dist = 0
        rec = 0.0
        t = time.perf_counter()
        for qi, q in enumerate(Q):
            ids, _ = h.query(q, 10, ef)
            rec += len(set(ids.tolist()) & set(gt[qi].tolist())) / 10
        rows.append({"ef": ef, "recall@10": rec / n, "ms_per_query": (time.perf_counter() - t) / n * 1e3,
                     "dist_evals": h.n_dist / n, "speedup_in_dist_evals": len(E) / (h.n_dist / n)})
    # MIPS index on ALS factors
    from ..ann.hnsw import mips_transform
    users = [u for u, c in engine.users.items() if c.als_vec is not None][:n]
    U = np.array([engine.users[u].als_vec for u in users])
    F = engine.als.item_factors
    gt50 = np.argsort(-(U @ F.T), axis=1)[:, :50]
    _, Uq = mips_transform(F, U)
    rec50 = np.mean([len(set(engine.als_index.query(q, 50, 200)[0].tolist()) & set(g.tolist())) / 50
                     for q, g in zip(Uq, gt50)])
    return {"text_hnsw": rows, "brute_force_ms_numpy": brute_ms, "n_items": len(E),
            "als_mips_recall@50_ef200": float(rec50)}
