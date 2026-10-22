"""Drape engine: wires retrieval, personalisation and ranking into one servable object.

            ┌──────────── query understanding (slots + learned category NB) ────────────┐
 query ───► │  BM25F (sparse)  ∪  LSA dense ⟶ HNSW   ⟶  RRF fusion  ⟶  hard filters     │
            └───────────────────────────────────────────────┬───────────────────────────┘
                                                            ▼  ~300 candidates
       user ctx (ALS vector, style/colour/price profile, live session) ──► LambdaRank MLP ──► top-K

 home feed:  ALS (MIPS→HNSW) ∪ co-visitation(session) ∪ popular  ──► LambdaRank MLP ──► top-K
 product page: similar items (behaviour + text) and complete-the-look (learned compatibility)
"""
from __future__ import annotations

import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .ann.hnsw import HNSW, mips_transform
from .data import schema as S
from .data.generator import ETYPE_BUY, ETYPE_CART, ETYPE_VIEW
from .rank.features import (REC_FEATURES, SEARCH_FEATURES, ItemTable, UserContext,
                            rec_features, search_features)
from .rank.ltr import LambdaRankMLP
from .recs.als import EVENT_WEIGHTS, ImplicitALS, interaction_matrix
from .recs.covisit import build_covisit, covisit_scores
from .recs.outfit import OutfitModel, complementary_pairs
from .search.bm25 import BM25Index
from .search.query_parser import QueryClassifier, QueryParser
from .search.semantic import LSAEncoder
from .text.tokenize import tokenize

RRF_K = 60


class TextIndex:
    """Event-independent indices (built once per catalog)."""

    def __init__(self, items: list[dict], hnsw_m: int = 16, hnsw_efc: int = 100, verbose=False):
        t = time.time()
        self.bm25 = BM25Index().fit(items)
        self.lsa = LSAEncoder(dim=96)
        self.emb = self.lsa.fit_transform(items)
        if verbose:
            print(f"  BM25 + LSA built in {time.time() - t:.1f}s; building text HNSW...")
        t = time.time()
        self.hnsw = HNSW(self.emb.shape[1], M=hnsw_m, ef_construction=hnsw_efc, metric="cosine")
        self.hnsw.add(self.emb)
        if verbose:
            print(f"  text HNSW built in {time.time() - t:.1f}s")


class Engine:
    def __init__(self, items: list[dict], text: TextIndex):
        self.items = items
        self.text = text
        n = len(items)
        self.n_items = n
        base = np.array([S.SUBCATEGORIES[it["subcategory"]][2] for it in items], float)
        price = np.array([it["price"] for it in items], float)
        self.it = ItemTable(
            style=np.array([S.STYLES.index(it["style"]) for it in items]),
            color=np.array([S.COLORS.index(it["color"]) for it in items]),
            slot=np.array([S.SLOTS.index(it["slot"]) for it in items]),
            subcat=np.array([it["subcategory"] for it in items]),
            gender=np.array([it["gender"] for it in items]),
            price=price, rel_price=np.log(price / base),
            rating=np.array([it["rating"] for it in items], float),
            added=np.array([it["added_day"] for it in items], float),
            log_pop=np.zeros(n), conv=np.zeros(n),
        )
        self.parser = QueryParser()
        self.search_ranker: LambdaRankMLP | None = None
        self.rec_ranker: LambdaRankMLP | None = None
        self.users: dict[int, UserContext] = {}
        self.now = 0.0

    # ======================================================================= fitting
    def fit(self, events: dict, queries: list[dict], cutoff: float, als_iters: int = 15,
            use_ann: bool = True, verbose: bool = False) -> "Engine":
        self.now = cutoff
        m = events["ts"] < cutoff
        ev = {k: v[m] for k, v in events.items()}
        n_users = int(events["user"].max()) + 1
        t0 = time.time()

        # --- popularity & conversion (time-decayed, half-life 14 days)
        decay = 0.5 ** ((cutoff - ev["ts"]) / 14.0)
        pop = np.bincount(ev["item"], weights=decay * EVENT_WEIGHTS[ev["etype"]], minlength=self.n_items)
        views = np.bincount(ev["item"][ev["etype"] == ETYPE_VIEW], minlength=self.n_items)
        buys = np.bincount(ev["item"][ev["etype"] == ETYPE_BUY], minlength=self.n_items)
        self.it.log_pop = np.log1p(pop)
        self.it.conv = (buys + 0.5) / (views + 10.0)

        # --- collaborative filtering
        W = interaction_matrix(ev["user"], ev["item"], ev["etype"], n_users, self.n_items)
        self.als = ImplicitALS(factors=64, iterations=als_iters).fit(W, verbose=False)
        if verbose:
            print(f"  ALS fitted in {time.time() - t0:.1f}s")
        t = time.time()
        self.use_ann = use_ann
        if use_ann:
            Xa = mips_transform(self.als.item_factors)
            self.als_index = HNSW(Xa.shape[1], M=12, ef_construction=80, metric="l2")
            self.als_index.add(Xa)
            if verbose:
                print(f"  ALS MIPS-HNSW built in {time.time() - t:.1f}s")

        # --- co-visitation graph
        self.covisit = build_covisit(ev["session"], ev["item"], ev["etype"], self.n_items)

        # --- user profiles
        self.users = {}
        self._build_profiles(ev, n_users)

        # --- query classifier from search logs (label = subcategory the user engaged with)
        eng = defaultdict(list)
        for s, i, e in zip(ev["session"], ev["item"], ev["etype"]):
            eng[int(s)].append((int(e), int(i)))
        texts, labels = [], []
        for q in queries:
            if q["ts"] >= cutoff or q["session"] not in eng:
                continue
            e, i = max(eng[q["session"]])
            texts.append(" ".join(self.parser.parse(q["text"]).tokens))
            labels.append(self.items[i]["subcategory"])
        self.parser = QueryParser(QueryClassifier().fit(texts, labels))

        # --- outfit compatibility
        pairs = complementary_pairs(self.items, ev["session"], ev["item"], ev["etype"])
        self.outfit = OutfitModel(self.items, self.als.item_factors).fit(pairs)
        if verbose:
            print(f"  behaviour models fitted in {time.time() - t0:.1f}s "
                  f"({len(texts)} labelled queries, {len(pairs)} outfit pairs)")
        return self

    def _build_profiles(self, ev: dict, n_users: int) -> None:
        order = np.lexsort((ev["ts"], ev["user"]))
        u_arr, i_arr, e_arr = ev["user"][order], ev["item"][order], ev["etype"][order]
        bounds = np.searchsorted(u_arr, np.arange(n_users + 1))
        for u in range(n_users):
            a, b = bounds[u], bounds[u + 1]
            ctx = UserContext()
            if a < b:
                self._fill_profile(ctx, i_arr[a:b], e_arr[a:b])
                ctx.als_vec = self.als.user_factors[u].copy()
            self.users[u] = ctx

    def _fill_profile(self, ctx: UserContext, items: np.ndarray, etypes: np.ndarray) -> None:
        w = EVENT_WEIGHTS[etypes]
        it = self.it
        ctx.style_dist = (np.bincount(it.style[items], w, 6) + 0.3) / (w.sum() + 1.8)
        ctx.color_dist = (np.bincount(it.color[items], w, 16) + 0.15) / (w.sum() + 2.4)
        ctx.slot_dist = (np.bincount(it.slot[items], w, 6) + 0.3) / (w.sum() + 1.8)
        ctx.rel_price = float(np.average(it.rel_price[items], weights=w))
        ctx.n_events = len(items)
        g = Counter(x for x in it.gender[items] if x != "unisex")
        if g:
            top, cnt = g.most_common(1)[0]
            ctx.gender = top if cnt / sum(g.values()) > 0.8 else None
        ctx.recent = [int(x) for x in items[-20:]]
        ctx.seen = dict(Counter(int(x) for x in items))
        ctx.purchased = {int(x) for x, e in zip(items, etypes) if e == ETYPE_BUY}

    # ======================================================================= live updates
    def user(self, user_id: int | None) -> UserContext:
        if user_id is None:
            return UserContext()
        return self.users.setdefault(int(user_id), UserContext())

    def log_event(self, user_id: int, item_id: int, etype: int) -> dict:
        """Online update: session memory + ALS fold-in (no retraining)."""
        t = time.perf_counter()
        ctx = self.user(user_id)
        if not hasattr(ctx, "_items"):
            ctx._items = list(ctx.recent)
            ctx._etypes = [ETYPE_VIEW] * len(ctx._items)
        ctx._items.append(int(item_id))
        ctx._etypes.append(int(etype))
        items, etypes = np.array(ctx._items), np.array(ctx._etypes)
        self._fill_profile(ctx, items, etypes)
        ctx.als_vec = self.als.fold_in(items, EVENT_WEIGHTS[etypes])
        return {"ok": True, "update_ms": round((time.perf_counter() - t) * 1e3, 3)}

    # ======================================================================= search
    def _live_mask(self, now: float) -> np.ndarray:
        return self.it.added <= now

    def search_candidates(self, pq, now: float, n_cand: int = 300):
        it = self.it
        mask = self._live_mask(now)
        if pq.max_price is not None:
            mask &= it.price <= pq.max_price
        if pq.min_price is not None:
            mask &= it.price >= pq.min_price
        if pq.gender:
            mask &= (it.gender == pq.gender) | (it.gender == "unisex")
        color_ix = S.COLORS.index(pq.color) if pq.color else None
        if color_ix is not None and (mask & (it.color == color_ix)).sum() >= 5:
            mask &= it.color == color_ix
        sc, conf = pq.top_subcat
        if sc is not None and conf >= 0.5:
            keep = [s for s, p in pq.subcat_probs.items() if p >= 0.1]
            cmask = mask & np.isin(it.subcat, keep)
            if cmask.sum() >= 10:
                mask = cmask
        # lexical
        toks = list(pq.tokens)
        wts = [1.0] * len(toks)
        for s, p in pq.subcat_probs.items():           # query expansion with learned category
            if p >= 0.1:
                st = tokenize(s)
                toks += st
                wts += [p] * len(st)
        bm = self.text.bm25.score(toks, wts)
        bm_m = np.where(mask, bm, -1.0)
        k = min(n_cand, int(mask.sum()))
        if k == 0:
            return np.array([], int), {}
        lex = np.argpartition(-bm_m, k - 1)[:k]
        lex = lex[bm_m[lex] > 0]
        # dense
        qtext = " ".join(pq.tokens + ([pq.color] if pq.color else []))
        qv = self.text.lsa.encode([qtext or pq.raw])[0]
        dense_ids, dd = self.text.hnsw.query(qv, k=n_cand, ef=max(n_cand, 128))
        keep = mask[dense_ids]
        dense_ids = dense_ids[keep]
        cands = np.unique(np.concatenate([lex, dense_ids]))
        if len(cands) < 50:                              # tiny result set → fill from mask
            extra = np.where(mask)[0]
            extra = extra[np.argsort(-it.log_pop[extra])][:200]
            cands = np.unique(np.concatenate([cands, extra]))
        sem = self.text.emb[cands] @ qv
        r_lex = np.full(len(cands), 1e6)
        order = np.argsort(-bm[cands])
        r_lex[order] = np.arange(len(cands))
        r_sem = np.empty(len(cands))
        r_sem[np.argsort(-sem)] = np.arange(len(cands))
        rrf = 1 / (RRF_K + r_lex + 1) + 1 / (RRF_K + r_sem + 1)
        return cands, {"bm25": bm[cands], "sem": sem, "rrf": rrf, "color_ix": color_ix}

    def search(self, query: str, user_id: int | None = None, k: int = 20, now: float | None = None,
               ctx: UserContext | None = None, mode: str = "ltr") -> dict:
        """mode: 'bm25' | 'hybrid' | 'ltr' (personalised)."""
        t0 = time.perf_counter()
        now = self.now if now is None else now
        pq = self.parser.parse(query)
        t1 = time.perf_counter()
        if mode == "bm25":
            s = self.text.bm25.score(pq.tokens + ([pq.color] if pq.color else []))
            s = np.where(self._live_mask(now), s, -1)
            top = np.argsort(-s)[:k]
            return {"ids": top.tolist(), "parsed": pq.as_dict(), "timings_ms": {}}
        cands, f = self.search_candidates(pq, now)
        t2 = time.perf_counter()
        if not len(cands):
            return {"ids": [], "scores": [], "parsed": pq.as_dict(), "timings_ms": {}}
        u = ctx if ctx is not None else self.user(user_id)
        if mode == "hybrid" or self.search_ranker is None:
            cat_p = np.array([pq.subcat_probs.get(s, 0) for s in self.it.subcat[cands]])
            scores = f["rrf"] * (1 + cat_p)
        else:
            cov = covisit_scores(self.covisit, u.recent)
            X = search_features(
                self.it, cands, bm25=f["bm25"], sem=f["sem"], rrf=f["rrf"],
                subcat_probs=pq.subcat_probs, color=pq.color, color_ix=f["color_ix"],
                style_hints_ix=[S.STYLES.index(s) for s in pq.style_hints], u=u,
                item_factors=self.als.item_factors, covisit=cov, now=now)
            scores = self.search_ranker.predict(X)
        order = np.argsort(-scores)[:k]
        t3 = time.perf_counter()
        return {
            "ids": cands[order].tolist(), "scores": scores[order].round(4).tolist(),
            "parsed": pq.as_dict(), "n_candidates": int(len(cands)),
            "timings_ms": {"parse": round((t1 - t0) * 1e3, 2), "retrieve": round((t2 - t1) * 1e3, 2),
                           "rank": round((t3 - t2) * 1e3, 2), "total": round((t3 - t0) * 1e3, 2)},
        }

    # ======================================================================= recommendations
    def _gender_mask(self, g: str | None) -> np.ndarray:
        if g is None:
            return np.ones(self.n_items, bool)
        return (self.it.gender == g) | (self.it.gender == "unisex")

    def rec_candidates(self, u: UserContext, now: float, n_als: int = 200):
        live = self._live_mask(now) & self._gender_mask(u.gender)
        if u.purchased:
            live[list(u.purchased)] = False
        als_scores = (self.als.item_factors @ u.als_vec if u.als_vec is not None
                      else np.zeros(self.n_items, np.float32))
        parts = []
        if u.als_vec is not None:
            if self.use_ann:
                _, q = mips_transform(self.als.item_factors, u.als_vec[None])
                ids, _ = self.als_index.query(q[0], k=n_als * 2, ef=n_als * 2)
                parts.append(ids[live[ids]][:n_als])
            else:
                s = np.where(live, als_scores, -np.inf)
                parts.append(np.argpartition(-s, n_als)[:n_als])
        cov = covisit_scores(self.covisit, u.recent)
        cm = np.where(live, cov, 0)
        nz = np.flatnonzero(cm)
        if len(nz):
            parts.append(nz[np.argsort(-cm[nz])][:100])
        pop = np.where(live, self.it.log_pop, -1)
        pop_ids = np.argpartition(-pop, 60)[:60]
        parts.append(pop_ids)
        cands = np.unique(np.concatenate(parts))
        pop_mask = np.zeros(self.n_items, bool)
        pop_mask[pop_ids] = True
        return cands, als_scores, cov, pop_mask

    def recommend(self, user_id: int | None, k: int = 20, now: float | None = None,
                  ctx: UserContext | None = None, mode: str = "ltr") -> dict:
        """mode: 'popular' | 'covisit' | 'als' | 'ltr'."""
        t0 = time.perf_counter()
        now = self.now if now is None else now
        u = ctx if ctx is not None else self.user(user_id)
        cands, als_s, cov, pop_mask = self.rec_candidates(u, now)
        if mode == "popular" or (u.n_events == 0 and mode != "ltr"):
            s = self.it.log_pop[cands]
        elif mode == "covisit":
            s = cov[cands] + 1e-3 * self.it.log_pop[cands]
        elif mode == "als" or self.rec_ranker is None:
            s = als_s[cands]
        else:
            X = rec_features(self.it, cands, als_scores=als_s, covisit=cov, popular_mask=pop_mask,
                             u=u, now=now)
            s = self.rec_ranker.predict(X)
        order = np.argsort(-s)
        out = self._diversify(cands[order], k)
        return {"ids": out, "timings_ms": {"total": round((time.perf_counter() - t0) * 1e3, 2)},
                "n_candidates": int(len(cands))}

    def _diversify(self, ranked: np.ndarray, k: int, max_per_subcat: int = 4) -> list[int]:
        """Light business rule: cap repeats of one subcategory in the visible slate."""
        out, cnt = [], Counter()
        for i in ranked:
            sc = self.it.subcat[i]
            if cnt[sc] < max_per_subcat:
                out.append(int(i))
                cnt[sc] += 1
            if len(out) == k:
                break
        return out

    # ======================================================================= product page
    def similar(self, item_id: int, k: int = 12) -> list[int]:
        """Blend 'looks like' (text embedding) with 'shopped like' (ALS + co-visitation)."""
        it = self.it
        same = (it.subcat == it.subcat[item_id]) & self._gender_mask(
            None if it.gender[item_id] == "unisex" else it.gender[item_id])
        same[item_id] = False
        ids, d = self.text.hnsw.query(self.text.emb[item_id], k=200, ef=200)
        text_s = np.zeros(self.n_items)
        text_s[ids] = 1 - d
        F = self.als.item_factors
        nf = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
        beh = nf @ nf[item_id]
        cov = np.asarray(self.covisit[item_id].todense()).ravel()
        s = 0.5 * text_s + 0.35 * beh + 2.0 * cov
        s = np.where(same, s, -np.inf)
        return np.argsort(-s)[:k].tolist()

    def complete_look(self, item_id: int, per_slot: int = 3) -> dict[str, list[int]]:
        look = self.outfit.complete_look(item_id, per_slot=per_slot)
        return {slot: [i for i, _ in picks] for slot, picks in look.items()}

    # ======================================================================= persistence
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str | Path) -> "Engine":
        with open(path, "rb") as f:
            return pickle.load(f)
