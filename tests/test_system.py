"""End-to-end tests on a small simulated world (built once per session, ~30 s)."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from drape.api import server
from drape.data import schema as S
from drape.data.generator import generate
from drape.engine import Engine, TextIndex


@pytest.fixture(scope="session")
def world():
    d = generate(n_items=1500, n_users=400, seed=3)
    text = TextIndex(d.items, hnsw_m=8, hnsw_efc=40)
    e = Engine(d.items, text).fit(d.events, d.queries, cutoff=120, als_iters=6)
    return d, e


def test_dataset_is_consistent(world):
    d, _ = world
    ev = d.events
    assert np.all(np.diff(ev["ts"]) >= 0)
    assert ev["item"].max() < len(d.items) and ev["user"].max() < d.n_users
    assert (ev["etype"] == 2).sum() < (ev["etype"] == 1).sum() < (ev["etype"] == 0).sum()


def test_search_respects_hard_filters(world):
    _, e = world
    r = e.search("black t-shirt for men under 1500", k=15)
    assert r["parsed"]["color"] == "black" and r["parsed"]["max_price"] == 1500
    for i in r["ids"]:
        it = e.items[i]
        assert it["price"] <= 1500 and it["gender"] in ("men", "unisex") and it["color"] == "black"


def test_learned_category_understanding(world):
    _, e = world
    r = e.search("sweatshirt", k=10)
    assert r["parsed"]["category"] == "hoodie"
    assert sum(e.items[i]["subcategory"] == "hoodie" for i in r["ids"]) >= 8


def test_typo_tolerance(world):
    _, e = world
    r = e.search("snekers", k=10, mode="hybrid")
    assert sum(e.items[i]["subcategory"] == "sneakers" for i in r["ids"]) >= 5


def test_recommendations_exclude_purchases_and_respect_gender(world):
    _, e = world
    u = next(u for u, c in e.users.items() if c.purchased and c.gender)
    ids = e.recommend(u, k=20, mode="als")["ids"]
    ctx = e.users[u]
    assert not set(ids) & ctx.purchased
    assert all(e.items[i]["gender"] in (ctx.gender, "unisex") for i in ids)


def test_live_event_moves_user_vector(world):
    _, e = world
    u = 10_000                                   # cold-start user
    item = next(i for i, it in enumerate(e.items) if it["subcategory"] == "saree")
    e.log_event(u, item, 2)
    ctx = e.users[u]
    assert ctx.als_vec is not None and np.linalg.norm(ctx.als_vec) > 0
    assert ctx.style_dist.argmax() == S.STYLES.index(e.items[item]["style"])


def test_complete_the_look_returns_complementary_slots(world):
    _, e = world
    top = next(i for i, it in enumerate(e.items) if it["slot"] == "top" and it["gender"] == "men")
    look = e.complete_look(top)
    assert set(look) <= set(S.COMPLEMENTS["top"]) and "bottom" in look
    for ids in look.values():
        for i in ids:
            assert e.items[i]["gender"] in ("men", "unisex")


def test_api_end_to_end(world):
    _, e = world
    server.set_engine(e)
    c = TestClient(server.app)
    assert c.get("/health").json()["status"] == "ok"
    r = c.get("/search", params={"q": "navy shirt", "user_id": 5}).json()
    assert r["results"] and r["cached"] is False
    assert c.get("/search", params={"q": "navy shirt", "user_id": 5}).json()["cached"] is True
    assert c.post("/events", json={"user_id": 5, "item_id": 1, "type": "cart"}).json()["ok"]
    assert c.get("/search", params={"q": "navy shirt", "user_id": 5}).json()["cached"] is False
    assert c.get("/recommend/5").status_code == 200
    assert "complete_the_look" in c.get("/items/3").json()
    assert c.get("/users/5").json()["known"]
    assert c.get("/trending").status_code == 200
    assert c.get("/items/999999").status_code == 404
    assert c.post("/events", json={"user_id": 5, "item_id": 1, "type": "steal"}).status_code == 422
    assert "/search" in c.get("/stats").json()["endpoints"]


def test_offline_context_never_leaks_the_query_session(world):
    """Regression: float32 timestamps once let a session's first click into its own context."""
    from drape.eval.offline import EventLog
    d, _ = world
    log = EventLog(d.events)
    for q in d.queries[::7]:
        assert not (log.s[log.context_slice(q["user"], q["ts"])] == q["session"]).any()
