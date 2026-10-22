"""Behavioural simulator: builds a fashion catalog, a user population and ~0.5M events.

Why simulate instead of scraping? Public fashion datasets (H&M, Myntra dumps) lack one of
(text, sessions, search queries, outfit co-purchases). The simulator produces *all* of them
from a hidden preference model, which means every offline metric has a known ground truth.
Swap `generate()` for a real loader with the same schema to run on production data.

Hidden model per user u and item i:
    util(u, i) = A*log p_u(style_i) + B*log p_u(color_i) - C*|log(price_i / budget_u)| + D*quality_i
Sessions start from a search query, the user browses a filtered pool sampled by softmax(util/T),
adds some to cart, buys a few, and sometimes continues to "complete the look" in a complementary
slot where colour harmony and style match add to utility.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import schema as S

ETYPE_VIEW, ETYPE_CART, ETYPE_BUY = 0, 1, 2
ETYPE_NAMES = ("view", "cart", "purchase")

STYLE_COLOR_BIAS = {
    "casual": {"blue": 2, "white": 2, "olive": 1.5, "beige": 1.5, "grey": 1.5},
    "formal": {"white": 3, "navy": 3, "grey": 3, "black": 2.5, "beige": 1.5, "lavender": 1.2},
    "ethnic": {"maroon": 3, "mustard": 3, "red": 2.5, "teal": 2, "pink": 2, "green": 1.5},
    "sporty": {"black": 3, "grey": 2.5, "navy": 2, "teal": 1.5, "red": 1.5},
    "streetwear": {"black": 3, "white": 2, "olive": 2, "grey": 2, "mustard": 1.2},
    "party": {"black": 3, "red": 2.5, "maroon": 2, "navy": 1.5, "pink": 1.3},
}

QUERY_ALIASES = {
    "t-shirt": ("t-shirt", "tshirt", "tee", "t shirt"), "shirt": ("shirt",),
    "polo": ("polo", "polo tshirt"), "blouse": ("blouse", "top"),
    "sweater": ("sweater", "pullover", "jumper"), "hoodie": ("hoodie", "sweatshirt"),
    "kurta": ("kurta",), "jeans": ("jeans", "denims"), "chinos": ("chinos", "khakis"),
    "trousers": ("trousers", "pants", "formal pants"), "joggers": ("joggers", "track pants"),
    "shorts": ("shorts",), "skirt": ("skirt",), "dress": ("dress", "frock"),
    "saree": ("saree", "sari"), "jacket": ("jacket",), "blazer": ("blazer", "suit jacket"),
    "coat": ("coat", "overcoat"), "sneakers": ("sneakers", "casual shoes", "trainers"),
    "loafers": ("loafers",), "boots": ("boots",), "heels": ("heels", "pumps"),
    "sandals": ("sandals", "slippers"), "watch": ("watch",), "belt": ("belt",),
    "handbag": ("handbag", "purse", "bag"), "backpack": ("backpack", "bag"),
    "sunglasses": ("sunglasses", "shades"),
}


@dataclass
class Dataset:
    items: list[dict]
    events: dict[str, np.ndarray]           # user, item, etype, ts, session
    queries: list[dict]                     # session, user, ts, text, filters
    n_users: int
    user_gender: np.ndarray
    meta: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "items.json").write_text(json.dumps(self.items))
        (p / "queries.json").write_text(json.dumps(self.queries))
        np.savez_compressed(p / "events.npz", user_gender=self.user_gender, **self.events)
        (p / "meta.json").write_text(json.dumps({**self.meta, "n_users": self.n_users}))

    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        p = Path(path)
        z = np.load(p / "events.npz")
        meta = json.loads((p / "meta.json").read_text())
        ev = {k: z[k] for k in ("user", "item", "etype", "ts", "session")}
        return cls(json.loads((p / "items.json").read_text()), ev,
                   json.loads((p / "queries.json").read_text()), meta["n_users"],
                   z["user_gender"], meta)


def _round_price(x: float) -> int:
    return int(max(199, round(x / 100) * 100 - 1))


def _make_brands(rng: np.random.Generator) -> list[dict]:
    brands = []
    for root in S.BRAND_ROOTS:
        suffix = S.BRAND_SUFFIX[rng.integers(len(S.BRAND_SUFFIX))]
        tier = S.PRICE_TIERS[rng.choice(4, p=[0.3, 0.4, 0.2, 0.1])]
        styles = list(rng.choice(S.STYLES, size=2, replace=False))
        brands.append({"name": f"{root} {suffix}", "tier": tier, "styles": styles})
    return brands


def _make_items(rng: np.random.Generator, n_items: int, horizon: float) -> tuple[list[dict], dict]:
    brands = _make_brands(rng)
    style_p = np.array([0.28, 0.2, 0.12, 0.14, 0.14, 0.12])
    items, hidden_q = [], np.empty(n_items)
    for i in range(n_items):
        style = S.STYLES[rng.choice(6, p=style_p)]
        subcat = rng.choice(sorted(S.STYLE_SUBCATS[style]))
        slot, genders, base, mats = S.SUBCATEGORIES[subcat]
        gender = genders[rng.integers(len(genders))]
        cand = [b for b in brands if style in b["styles"]] or brands
        brand = cand[rng.integers(len(cand))]
        cw = np.ones(len(S.COLORS))
        for c, w in STYLE_COLOR_BIAS[style].items():
            cw[S.COLORS.index(c)] = w
        color = S.COLORS[rng.choice(len(S.COLORS), p=cw / cw.sum())]
        material = mats[rng.integers(len(mats))]
        fit = S.FITS[rng.integers(len(S.FITS))] if slot in ("top", "bottom", "outerwear", "onepiece") else ""
        pattern = S.PATTERNS[rng.choice(6, p=[.45, .12, .1, .15, .1, .08])] if slot != "footwear" else "solid"
        if style == "ethnic" and pattern in ("striped", "checked"):
            pattern = "embroidered"
        price = _round_price(base * S.TIER_MULT[brand["tier"]] * rng.lognormal(0, 0.25))
        occ = list(rng.choice(S.OCCASIONS[style], size=2, replace=False))
        season = S.MATERIAL_SEASON[material]
        title = " ".join(w for w in [brand["name"], fit.title(), color.title(),
                                     pattern.title() if pattern != "solid" else "",
                                     material.title(), subcat.title()] if w)
        desc = (f"{pattern} {color} {subcat} in {material} for {gender}. "
                f"A {style} pick, perfect for {occ[0]} and {occ[1]}. "
                f"{'Breathable for ' + season if season != 'all-season' else 'Wear it all year'}.")
        hidden_q[i] = rng.normal()
        rating = float(np.clip(3.9 + 0.35 * hidden_q[i] + rng.normal(0, 0.2), 2.5, 5.0))
        items.append({
            "id": i, "title": title, "description": desc, "brand": brand["name"],
            "tier": brand["tier"], "subcategory": subcat, "slot": slot, "gender": gender,
            "style": style, "color": color, "material": material, "fit": fit, "pattern": pattern,
            "season": season, "price": price, "rating": round(rating, 1),
            "added_day": float(rng.uniform(-60, horizon * 0.9)),
        })
    return items, {"quality": hidden_q}


def generate(n_items: int = 20_000, n_users: int = 6_000, horizon_days: float = 120.0,
             seed: int = 7) -> Dataset:
    rng = np.random.default_rng(seed)
    items, hidden = _make_items(rng, n_items, horizon_days)

    # ---- vectorised item attributes ----
    style_ix = np.array([S.STYLES.index(it["style"]) for it in items])
    color_ix = np.array([S.COLORS.index(it["color"]) for it in items])
    log_price = np.log([it["price"] for it in items])
    gender_arr = np.array([it["gender"] for it in items])
    slot_arr = np.array([it["slot"] for it in items])
    subcat_arr = np.array([it["subcategory"] for it in items])
    added = np.array([it["added_day"] for it in items])
    quality = hidden["quality"]
    neutral = np.array([c in S.NEUTRALS for c in S.COLORS])
    harmony = np.array([[S.colors_harmonize(a, b) for b in S.COLORS] for a in S.COLORS])

    # ---- users ----
    u_gender = rng.choice(np.array(["men", "women", "any"]), size=n_users, p=[0.45, 0.45, 0.10])
    u_style = rng.dirichlet(np.full(6, 0.35), size=n_users)
    u_color = rng.dirichlet(np.full(len(S.COLORS), 0.25), size=n_users)
    u_color = 0.75 * u_color + 0.25 * (neutral / neutral.sum())
    u_budget = rng.normal(0.0, 0.45, size=n_users)          # log-multiplier vs base price
    u_sessions = np.clip(rng.pareto(1.6, size=n_users) * 6 + 2, 2, 80).astype(int)

    base_lp = np.log([S.SUBCATEGORIES[s][2] for s in subcat_arr])
    rel_lp = log_price - base_lp                            # tier/price position within subcat

    def gender_ok(ug: str) -> np.ndarray:
        if ug == "any":
            return np.ones(n_items, bool)
        return (gender_arr == ug) | (gender_arr == "unisex")

    g_masks = {g: gender_ok(g) for g in ("men", "women", "any")}
    pools: dict[tuple[str, str], np.ndarray] = {}
    for g, m in g_masks.items():
        for sc in S.SUBCATEGORIES:
            pools[(g, sc)] = np.where(m & (subcat_arr == sc))[0]
        for sl in S.SLOTS:
            pools[(g, "slot:" + sl)] = np.where(m & (slot_arr == sl))[0]

    def utility(u: int, idx: np.ndarray) -> np.ndarray:
        return (2.2 * np.log(u_style[u, style_ix[idx]] + 1e-3)
                + 1.6 * np.log(u_color[u, color_ix[idx]] + 1e-3)
                - 2.5 * np.abs(rel_lp[idx] - u_budget[u])
                + 0.8 * quality[idx])

    E = {k: [] for k in ("user", "item", "etype", "ts", "session")}
    queries: list[dict] = []
    sid = 0
    for u in range(n_users):
        ug = u_gender[u]
        times = np.sort(rng.uniform(0, horizon_days, size=u_sessions[u]))
        top_colors = np.argsort(-u_color[u])[:3]
        for t in times:
            style = rng.choice(6, p=u_style[u])
            subs = [s for s in sorted(S.STYLE_SUBCATS[S.STYLES[style]]) if len(pools[(ug, s)])]
            subcat = subs[rng.integers(len(subs))]
            pool = pools[(ug, subcat)]
            pool = pool[added[pool] <= t]                      # only items live at time t
            if len(pool) < 5:
                continue
            # ---- search query the user types ----
            words, filt = [], {"subcategory": subcat}
            if rng.random() < 0.45:
                c = int(top_colors[rng.choice(3, p=[.6, .25, .15])])
                cpool = pool[color_ix[pool] == c]
                if len(cpool) >= 3:
                    pool, filt["color"] = cpool, S.COLORS[c]
                    words.append(S.COLORS[c])
            alias = QUERY_ALIASES[subcat][rng.integers(len(QUERY_ALIASES[subcat]))]
            words.append(alias)
            if ug != "any" and rng.random() < 0.3:
                words.append(f"for {ug}")
            if rng.random() < 0.35:
                occ = S.OCCASIONS[S.STYLES[style]]
                words.append(f"for {occ[rng.integers(len(occ))]}")
            if rng.random() < 0.25:
                cap = int(math.exp(base_lp[pool[0]] + u_budget[u] + 0.35) // 100 * 100)
                ppool = pool[log_price[pool] <= math.log(max(cap, 1))]
                if len(ppool) >= 3:
                    pool, filt["max_price"] = ppool, cap
                    words.append(f"under {cap}")
            queries.append({"session": sid, "user": u, "ts": float(t), "text": " ".join(words),
                            "filters": filt})
            # ---- browse / cart / buy ----
            util = utility(u, pool)
            p = np.exp((util - util.max()) / 0.9)
            p /= p.sum()
            k = min(len(pool), int(rng.geometric(0.22)) + 1)
            viewed = rng.choice(pool, size=k, replace=False, p=p)
            vu = utility(u, viewed)
            bought_any = None
            for j, (it, uv) in enumerate(zip(viewed, vu)):
                ts = t + j * 0.002
                E["user"].append(u); E["item"].append(it); E["etype"].append(ETYPE_VIEW)
                E["ts"].append(ts); E["session"].append(sid)
                if rng.random() < 1 / (1 + math.exp(-(uv - util.max() + 0.6))):
                    E["user"].append(u); E["item"].append(it); E["etype"].append(ETYPE_CART)
                    E["ts"].append(ts + 0.0005); E["session"].append(sid)
                    if rng.random() < 0.45:
                        E["user"].append(u); E["item"].append(it); E["etype"].append(ETYPE_BUY)
                        E["ts"].append(ts + 0.001); E["session"].append(sid)
                        bought_any = int(it)
            # ---- complete the look ----
            if bought_any is not None and rng.random() < 0.5:
                anchor = bought_any
                for comp_slot in rng.permutation(S.COMPLEMENTS[slot_arr[anchor]])[:2]:
                    cpool = pools[(ug, "slot:" + comp_slot)]
                    cpool = cpool[added[cpool] <= t]
                    if len(cpool) < 5:
                        continue
                    cu = (utility(u, cpool)
                          + 2.5 * (style_ix[cpool] == style_ix[anchor])
                          + 2.0 * harmony[color_ix[anchor], color_ix[cpool]]
                          - 2.0 * (~harmony[color_ix[anchor], color_ix[cpool]]))
                    cp = np.exp((cu - cu.max()) / 0.8)
                    cp /= cp.sum()
                    m = min(len(cpool), int(rng.integers(2, 5)))
                    cv = rng.choice(cpool, size=m, replace=False, p=cp)
                    best = cv[np.argmax(cu[np.searchsorted(cpool, cv)])]
                    for j, it in enumerate(cv):
                        ts = t + 0.05 + j * 0.002
                        E["user"].append(u); E["item"].append(it); E["etype"].append(ETYPE_VIEW)
                        E["ts"].append(ts); E["session"].append(sid)
                    if rng.random() < 0.55:
                        for et in (ETYPE_CART, ETYPE_BUY):
                            E["user"].append(u); E["item"].append(int(best)); E["etype"].append(et)
                            E["ts"].append(t + 0.09); E["session"].append(sid)
            sid += 1

    events = {
        "user": np.array(E["user"], np.int32), "item": np.array(E["item"], np.int32),
        "etype": np.array(E["etype"], np.int8), "ts": np.array(E["ts"], np.float32),
        "session": np.array(E["session"], np.int32),
    }
    order = np.argsort(events["ts"], kind="stable")
    events = {k: v[order] for k, v in events.items()}
    meta = {"n_items": n_items, "horizon_days": horizon_days, "seed": seed, "n_sessions": sid}
    return Dataset(items, events, queries, n_users, u_gender, meta)
