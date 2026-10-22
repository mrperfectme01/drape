"""End-to-end: simulate data → fit models → train rankers → evaluate → export serving engine.

    python -m scripts.train            # full run (~15 min on 2 CPU cores)
    python -m scripts.train --quick    # smaller sample for CI / smoke tests
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from drape.data.generator import Dataset, generate
from drape.engine import Engine, TextIndex
from drape.eval import offline as O
from drape.rank.features import REC_FEATURES, SEARCH_FEATURES
from drape.rank.ltr import LambdaRankMLP

ART = Path("artifacts")
REP = Path("reports")
T1, T2, T_END = 90.0, 105.0, 120.0


def stage(msg: str):
    print(f"\n== {msg}", flush=True)
    return time.time()


def main(quick: bool = False) -> None:
    global ART, REP
    if quick:
        ART, REP = Path("artifacts_quick"), Path("artifacts_quick/reports")
    ART.mkdir(exist_ok=True)
    REP.mkdir(exist_ok=True)
    t_all = time.time()

    t = stage("data")
    if (ART / "data" / "meta.json").exists():
        data = Dataset.load(ART / "data")
    else:
        data = generate(n_items=4000, n_users=1500) if quick else generate()
        data.save(ART / "data")
    ev = data.events
    print(f"  {len(data.items)} items, {data.n_users} users, {len(ev['user'])} events, "
          f"{len(data.queries)} search sessions ({time.time() - t:.0f}s)")

    t = stage("text indices (BM25F, LSA, HNSW)")
    tpath = ART / "text_index.pkl"
    if tpath.exists():
        text = pickle.load(open(tpath, "rb"))
    else:
        text = TextIndex(data.items, verbose=True)
        pickle.dump(text, open(tpath, "wb"))

    log = O.EventLog(ev)
    results: dict = {"dataset": {"items": len(data.items), "users": data.n_users,
                                 "events": int(len(ev["user"])), "queries": len(data.queries)}}

    t = stage(f"E1: fit behaviour models on days [0, {T1})")
    e1 = Engine(data.items, text).fit(ev, data.queries, cutoff=T1, verbose=True)

    t = stage(f"build ranking data from days [{T1}, {T2})")
    Xs, ys, gs = O.search_groups(e1, log, data.queries, T1, T2,
                                 max_queries=1500 if quick else 6000)
    print(f"  search: {len(gs)} query groups, {len(ys)} rows, {(ys > 0).mean():.1%} positive")
    users = np.unique(ev["user"][(ev["ts"] >= T1) & (ev["ts"] < T2)])
    Xr, yr, gr = O.rec_groups(e1, log, users, T1, T2)
    print(f"  recs:   {len(gr)} user groups, {len(yr)} rows ({time.time() - t:.0f}s)")

    t = stage("train LambdaRank MLPs")
    search_ranker = LambdaRankMLP(hidden=32, epochs=10, k=10).fit(
        Xs, ys, gs, SEARCH_FEATURES, verbose=True)
    rec_ranker = LambdaRankMLP(hidden=32, epochs=10, k=20).fit(Xr, yr, gr, REC_FEATURES, verbose=True)
    results["search_feature_importance"] = search_ranker.feature_importance(Xs[:20000])
    results["rec_feature_importance"] = rec_ranker.feature_importance(Xr[:20000])
    results["search_ltr_curve"] = search_ranker.history
    results["rec_ltr_curve"] = rec_ranker.history

    t = stage(f"E2: refit behaviour models on days [0, {T2}) and evaluate on [{T2}, {T_END})")
    e2 = Engine(data.items, text).fit(ev, data.queries, cutoff=T2, verbose=True)
    e2.search_ranker, e2.rec_ranker = search_ranker, rec_ranker

    t = stage("evaluate: search")
    results["search"] = O.eval_search(e2, log, data.queries, T2, T_END,
                                      max_queries=800 if quick else 3000)
    print(json.dumps(results["search"], indent=1))
    results["query_classifier"] = O.eval_query_classifier(e2, log, data.queries, T2, T_END)
    print(" ", results["query_classifier"])

    t = stage("evaluate: recommendations")
    test_users = np.unique(ev["user"][(ev["ts"] >= T2)])
    results["recs"] = O.eval_recs(e2, log, test_users, T2, T_END)
    print(json.dumps(results["recs"], indent=1))

    t = stage("evaluate: complete-the-look")
    results["outfit"] = O.eval_outfit(e2, ev, T2, T_END)
    print(json.dumps(results["outfit"], indent=1))

    t = stage("evaluate: ANN indices")
    results["ann"] = O.eval_ann(e2, data.queries)
    print(json.dumps(results["ann"], indent=1))

    t = stage(f"final serving engine: fit on all days [0, {T_END})")
    final = Engine(data.items, text).fit(ev, data.queries, cutoff=T_END, verbose=True)
    final.search_ranker, final.rec_ranker = search_ranker, rec_ranker
    final.save(ART / "engine.pkl")
    results["build_seconds"] = round(time.time() - t_all, 1)
    (ART / "results.json").write_text(json.dumps(results, indent=2, default=float))
    from scripts.plots import make_all
    make_all(results, final, REP)
    print(f"\nDone in {results['build_seconds']:.0f}s -> {ART}/engine.pkl, {ART}/results.json, {REP}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
