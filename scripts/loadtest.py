"""Concurrent load test against a running server: mixed search / feed / product-page / event traffic.

    python -m scripts.loadtest --url http://localhost:8000 --n 2000 --concurrency 16
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time

import httpx
import numpy as np

QUERIES = ["black sneakers", "navy linen shirt for office under 2k", "sweatshirt for women",
           "wedding kurta for men", "khakis", "red dress for party", "bag for college",
           "running shoes under 3000", "olive jacket", "white tshirt", "formal trousers for men",
           "silk saree for diwali", "leather boots", "maroon blouse", "sunglasses", "denims"]


async def one(client: httpx.AsyncClient, rng: random.Random, lat: dict):
    r = rng.random()
    u = rng.randrange(6000)
    if r < 0.5:
        q = rng.choice(QUERIES)
        name, req = "search", client.get("/search", params={"q": q, "user_id": u})
    elif r < 0.75:
        name, req = "recommend", client.get(f"/recommend/{u}")
    elif r < 0.9:
        name, req = "item", client.get(f"/items/{rng.randrange(20000)}")
    else:
        name, req = "event", client.post("/events", json={"user_id": u, "item_id": rng.randrange(20000),
                                                          "type": rng.choice(["view", "cart"])})
    t = time.perf_counter()
    resp = await req
    resp.raise_for_status()
    lat.setdefault(name, []).append((time.perf_counter() - t) * 1e3)


async def main(url: str, n: int, concurrency: int, seed: int = 0):
    rng = random.Random(seed)
    lat: dict[str, list] = {}
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=url, timeout=30) as client:
        await client.get("/health")

        async def task():
            async with sem:
                await one(client, rng, lat)

        t = time.perf_counter()
        await asyncio.gather(*(task() for _ in range(n)))
        wall = time.perf_counter() - t
    out = {"requests": n, "concurrency": concurrency, "throughput_rps": round(n / wall, 1),
           "endpoints": {k: {"count": len(v), "p50_ms": round(float(np.percentile(v, 50)), 1),
                             "p99_ms": round(float(np.percentile(v, 99)), 1)} for k, v in lat.items()}}
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=16)
    a = ap.parse_args()
    asyncio.run(main(a.url, a.n, a.concurrency))
