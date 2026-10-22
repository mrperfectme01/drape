"""HTTP serving layer.

    uvicorn drape.api.server:app --port 8000

Endpoints
  GET  /search?q=...&user_id=..&k=..     personalised search (+ parsed query, stage timings)
  GET  /recommend/{user_id}?k=..          "For you" home feed
  GET  /items/{item_id}?user_id=..        product page: item, similar items, complete-the-look
  POST /events                             {user_id, item_id, type: view|cart|purchase}
  GET  /users/{user_id}                   learned taste profile
  GET  /stats                             per-endpoint request counts and p50/p95/p99 latency
"""
from __future__ import annotations

import os
import time
from collections import OrderedDict, defaultdict, deque
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..data import schema as S
from ..engine import Engine

ENGINE_PATH = os.environ.get("DRAPE_ENGINE", "artifacts/engine.pkl")
STATIC = Path(__file__).parent / "static"
ETYPES = {"view": 0, "cart": 1, "purchase": 2}

# Handlers are `async def` on purpose: the work is CPU-bound Python/NumPy holding the GIL, so
# running it on FastAPI's threadpool only adds context-switch thrash. Scale out with processes
# (`--workers N`) behind a shared event store, not threads.
app = FastAPI(title="Drape", version="1.0",
              description="Personalised fashion search, recommendations and outfit completion.")
_engine: Engine | None = None
_lat: dict[str, deque] = defaultdict(lambda: deque(maxlen=5000))
_count: dict[str, int] = defaultdict(int)


class LRU:
    def __init__(self, cap: int = 2048):
        self.cap, self.d = cap, OrderedDict()
        self.hits = self.misses = 0

    def get(self, k):
        if k in self.d:
            self.d.move_to_end(k)
            self.hits += 1
            return self.d[k]
        self.misses += 1
        return None

    def put(self, k, v):
        self.d[k] = v
        self.d.move_to_end(k)
        if len(self.d) > self.cap:
            self.d.popitem(last=False)


_cache = LRU()
_user_version: dict[int, int] = defaultdict(int)   # bumped on every event → cache invalidation


def engine() -> Engine:
    global _engine
    if _engine is None:
        if not Path(ENGINE_PATH).exists():
            raise HTTPException(503, f"engine not built: run `python -m scripts.train` ({ENGINE_PATH})")
        _engine = Engine.load(ENGINE_PATH)
    return _engine


def set_engine(e: Engine) -> None:          # used by tests
    global _engine
    _engine = e
    _cache.d.clear()


@app.middleware("http")
async def timing(request: Request, call_next):
    t = time.perf_counter()
    resp = await call_next(request)
    route = request.scope.get("route")
    name = getattr(route, "path", request.url.path)
    ms = (time.perf_counter() - t) * 1e3
    _lat[name].append(ms)
    _count[name] += 1
    resp.headers["X-Response-Time-ms"] = f"{ms:.2f}"
    return resp


def card(e: Engine, i: int, score: float | None = None) -> dict:
    it = e.items[i]
    out = {k: it[k] for k in ("id", "title", "brand", "subcategory", "slot", "gender", "style",
                              "color", "material", "price", "rating", "pattern")}
    if score is not None:
        out["score"] = score
    return out


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
async def health():
    e = engine()
    return {"status": "ok", "items": e.n_items, "users": len(e.users),
            "ranker": e.search_ranker is not None}


@app.get("/search")
async def search(q: str = Query(..., min_length=1, max_length=200), user_id: int | None = None,
           k: int = Query(24, ge=1, le=100), mode: str = Query("ltr", pattern="^(bm25|hybrid|ltr)$")):
    e = engine()
    key = ("s", q.strip().lower(), user_id, _user_version[user_id] if user_id is not None else 0, k, mode)
    hit = _cache.get(key)
    if hit is not None:
        return {**hit, "cached": True}
    r = e.search(q, user_id=user_id, k=k, mode=mode)
    scores = r.get("scores") or [None] * len(r["ids"])
    out = {"query": q, "parsed": r["parsed"], "n_candidates": r.get("n_candidates"),
           "timings_ms": r["timings_ms"],
           "results": [card(e, i, s) for i, s in zip(r["ids"], scores)]}
    _cache.put(key, out)
    return {**out, "cached": False}


@app.get("/trending")
async def trending(k: int = Query(24, ge=1, le=100)):
    from ..rank.features import UserContext
    e = engine()
    r = e.recommend(None, k=k, ctx=UserContext(), mode="popular")
    return {"timings_ms": r["timings_ms"], "results": [card(e, i) for i in r["ids"]]}


@app.get("/recommend/{user_id}")
async def recommend(user_id: int, k: int = Query(24, ge=1, le=100),
              mode: str = Query("ltr", pattern="^(popular|covisit|als|ltr)$")):
    e = engine()
    r = e.recommend(user_id, k=k, mode=mode)
    return {"user_id": user_id, "timings_ms": r["timings_ms"], "n_candidates": r["n_candidates"],
            "results": [card(e, i) for i in r["ids"]]}


@app.get("/items/{item_id}")
async def item(item_id: int, user_id: int | None = None):
    e = engine()
    if not 0 <= item_id < e.n_items:
        raise HTTPException(404, "no such item")
    t = time.perf_counter()
    look = e.complete_look(item_id)
    sim = e.similar(item_id)
    return {"item": {**card(e, item_id), "description": e.items[item_id]["description"]},
            "similar": [card(e, i) for i in sim],
            "complete_the_look": {slot: [card(e, i) for i in ids] for slot, ids in look.items()},
            "timings_ms": {"total": round((time.perf_counter() - t) * 1e3, 2)}}


class Event(BaseModel):
    user_id: int = Field(ge=0)
    item_id: int = Field(ge=0)
    type: str = Field(pattern="^(view|cart|purchase)$")


@app.post("/events")
async def log_event(ev: Event):
    e = engine()
    if ev.item_id >= e.n_items:
        raise HTTPException(404, "no such item")
    r = e.log_event(ev.user_id, ev.item_id, ETYPES[ev.type])
    _user_version[ev.user_id] += 1
    return r


@app.get("/users/{user_id}")
async def user(user_id: int):
    e = engine()
    u = e.user(user_id)
    top = lambda dist, names: [{"name": names[i], "share": round(float(dist[i]), 3)}  # noqa: E731
                               for i in np.argsort(-dist)[:3]]
    return {"user_id": user_id, "known": u.n_events > 0, "events": u.n_events,
            "gender": u.gender, "top_styles": top(u.style_dist, S.STYLES),
            "top_colors": top(u.color_dist, S.COLORS),
            "price_position": None if u.rel_price is None else round(float(np.exp(u.rel_price)), 2),
            "recent": [card(e, i) for i in u.recent[-6:][::-1]]}


@app.get("/stats")
async def stats():
    out = {}
    for name, v in _lat.items():
        a = np.array(v)
        out[name] = {"count": _count[name], "p50_ms": round(float(np.percentile(a, 50)), 2),
                     "p95_ms": round(float(np.percentile(a, 95)), 2),
                     "p99_ms": round(float(np.percentile(a, 99)), 2)}
    return {"endpoints": out, "cache": {"hits": _cache.hits, "misses": _cache.misses,
                                        "size": len(_cache.d)}}
