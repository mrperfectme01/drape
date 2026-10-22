"""README figures from artifacts/results.json (static PNGs, light surface)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from drape.data import schema as S  # noqa: E402

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, BASE = "#2a78d6", "#b9b8b1"
DIV_NEG, DIV_MID, DIV_POS = "#e34948", "#f0efec", "#2a78d6"


def _style(ax, title):
    ax.set_facecolor(SURF)
    ax.figure.set_facecolor(SURF)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=12)


def _bars(ax, labels, vals, best_last=True, fmt="{:.3f}"):
    colors = [BASE] * (len(vals) - 1) + [BLUE] if best_last else [BLUE] * len(vals)
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors, height=0.55)
    ax.set_yticks(y, labels, color=INK)
    ax.invert_yaxis()
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.01, yi, fmt.format(v), va="center", color=INK, fontsize=9)
    ax.set_xlim(0, max(vals) * 1.18)


def make_all(res: dict, engine, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    # 1. search quality
    s = res["search"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8))
    names = ["BM25 baseline", "Hybrid retrieval\n+ query understanding", "Personalised\nLambdaRank"]
    _bars(axes[0], names, [s[m]["ndcg@10"] for m in ("bm25", "hybrid", "ltr")])
    _style(axes[0], "Search NDCG@10")
    _bars(axes[1], names, [s[m]["mrr_purchase_or_cart"] for m in ("bm25", "hybrid", "ltr")])
    _style(axes[1], "MRR of first cart / purchase")
    axes[1].set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "search_quality.png", dpi=160)
    plt.close(fig)

    # 2. recommendation quality
    r = res["recs"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8))
    names = ["Popularity", "Session co-visitation", "Implicit ALS", "Hybrid + LambdaRank"]
    modes = ("popular", "covisit", "als", "ltr")
    _bars(axes[0], names, [r[m]["recall@20"] for m in modes])
    _style(axes[0], "Home feed Recall@20")
    _bars(axes[1], names, [r[m]["hit@10"] for m in modes])
    _style(axes[1], "Hit rate @10")
    axes[1].set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "rec_quality.png", dpi=160)
    plt.close(fig)

    # 3. HNSW trade-off
    rows = res["ann"]["text_hnsw"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = [row["dist_evals"] for row in rows]
    y = [row["recall@10"] for row in rows]
    ax.plot(x, y, color=BLUE, lw=2, marker="o", ms=7, mec=SURF, mew=2)
    for row in rows:
        ax.annotate(f"ef={row['ef']}", (row["dist_evals"], row["recall@10"]),
                    textcoords="offset points", xytext=(6, -12), fontsize=8, color=INK2)
    n = res["ann"]["n_items"]
    ax.set_xlabel(f"distance evaluations per query (exact scan = {n:,})", color=INK2)
    ax.set_ylabel("recall@10 vs exact", color=INK2)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _style(ax, "HNSW (from scratch): recall vs work")
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "hnsw_tradeoff.png", dpi=160)
    plt.close(fig)

    # 4. learned colour harmony
    cols = [c for c in S.COLORS if c not in S.NEUTRALS]
    ix = [S.COLORS.index(c) for c in cols]
    M = engine.outfit.color_pmi[np.ix_(ix, ix)].copy()
    np.fill_diagonal(M, np.nan)
    M -= np.nanmedian(M)                      # centre on the typical colour pair
    lim = np.nanmax(np.abs(M))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", [DIV_NEG, DIV_MID, DIV_POS])
    cmap.set_bad(SURF)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(M, cmap=cmap, vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", color=INK, fontsize=9)
    ax.set_yticks(range(len(cols)), cols, color=INK, fontsize=9)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if (a, b) in S.HARMONY:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=INK, lw=1.6))
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK2, length=0)
    cb.set_label("learned PMI vs. typical pair", color=INK2)
    _style(ax, "Colour harmony learned from co-purchases\n(boxed = pairs planted in the simulator)")
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "color_harmony.png", dpi=160)
    plt.close(fig)

    # 5. ranker feature importance
    imp = res["search_feature_importance"]
    top = list(imp.items())[:10]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    _bars(ax, [k for k, _ in top], [v for _, v in top], best_last=False, fmt="{:.1%}")
    _style(ax, "Search ranker: feature sensitivity (top 10)")
    fig.tight_layout()
    fig.savefig(out / "search_feature_importance.png", dpi=160)
    plt.close(fig)
