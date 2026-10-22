"""Catalog vocabulary: categories, attributes, and fashion rules.

These rules double as the *hidden ground truth* the simulator uses to generate
user behaviour. The models never read them directly; they have to learn the
structure back from clicks, carts and purchases.
"""
from __future__ import annotations

# subcategory -> (slot, allowed genders, base price INR, typical materials)
SUBCATEGORIES: dict[str, tuple[str, tuple[str, ...], int, tuple[str, ...]]] = {
    "t-shirt":    ("top", ("men", "women", "unisex"), 699, ("cotton", "polyester", "linen")),
    "shirt":      ("top", ("men", "women"), 1299, ("cotton", "linen", "silk")),
    "polo":       ("top", ("men",), 999, ("cotton", "polyester")),
    "blouse":     ("top", ("women",), 1199, ("silk", "cotton", "polyester")),
    "sweater":    ("top", ("men", "women", "unisex"), 1799, ("wool", "cotton")),
    "hoodie":     ("top", ("men", "women", "unisex"), 1599, ("cotton", "polyester")),
    "kurta":      ("top", ("men", "women"), 1399, ("cotton", "silk", "linen")),
    "jeans":      ("bottom", ("men", "women"), 1899, ("denim",)),
    "chinos":     ("bottom", ("men", "women"), 1599, ("cotton",)),
    "trousers":   ("bottom", ("men", "women"), 1799, ("polyester", "wool", "linen")),
    "joggers":    ("bottom", ("men", "women", "unisex"), 1199, ("cotton", "polyester")),
    "shorts":     ("bottom", ("men", "women", "unisex"), 799, ("cotton", "denim", "polyester")),
    "skirt":      ("bottom", ("women",), 1299, ("cotton", "denim", "silk")),
    "dress":      ("onepiece", ("women",), 2299, ("cotton", "silk", "polyester", "linen")),
    "saree":      ("onepiece", ("women",), 3499, ("silk", "cotton")),
    "jacket":     ("outerwear", ("men", "women", "unisex"), 2999, ("denim", "leather", "polyester")),
    "blazer":     ("outerwear", ("men", "women"), 3999, ("wool", "linen", "polyester")),
    "coat":       ("outerwear", ("men", "women"), 4999, ("wool",)),
    "sneakers":   ("footwear", ("men", "women", "unisex"), 2999, ("leather", "polyester")),
    "loafers":    ("footwear", ("men", "women"), 2799, ("leather",)),
    "boots":      ("footwear", ("men", "women"), 3999, ("leather",)),
    "heels":      ("footwear", ("women",), 2499, ("leather", "polyester")),
    "sandals":    ("footwear", ("men", "women", "unisex"), 999, ("leather", "polyester")),
    "watch":      ("accessory", ("men", "women", "unisex"), 3499, ("leather",)),
    "belt":       ("accessory", ("men", "women"), 799, ("leather",)),
    "handbag":    ("accessory", ("women",), 2799, ("leather", "polyester")),
    "backpack":   ("accessory", ("unisex",), 1999, ("polyester",)),
    "sunglasses": ("accessory", ("men", "women", "unisex"), 1499, ("polyester",)),
}

SLOTS = ("top", "bottom", "onepiece", "outerwear", "footwear", "accessory")

# Which slots complete an outfit for a given anchor slot.
COMPLEMENTS = {
    "top": ("bottom", "footwear", "outerwear", "accessory"),
    "bottom": ("top", "footwear", "accessory"),
    "onepiece": ("footwear", "accessory", "outerwear"),
    "outerwear": ("top", "bottom", "footwear"),
    "footwear": ("top", "bottom", "onepiece"),
    "accessory": ("top", "bottom", "onepiece"),
}

STYLES = ("casual", "formal", "ethnic", "sporty", "streetwear", "party")

# subcategories that plausibly exist in each style (gives the catalog structure)
STYLE_SUBCATS = {
    "casual": {"t-shirt", "shirt", "polo", "sweater", "hoodie", "jeans", "chinos", "shorts",
               "skirt", "dress", "jacket", "sneakers", "sandals", "watch", "belt", "backpack",
               "sunglasses", "handbag"},
    "formal": {"shirt", "blouse", "sweater", "trousers", "chinos", "skirt", "dress", "blazer",
               "coat", "loafers", "heels", "boots", "watch", "belt", "handbag"},
    "ethnic": {"kurta", "saree", "trousers", "sandals", "loafers", "heels", "watch", "handbag"},
    "sporty": {"t-shirt", "polo", "hoodie", "joggers", "shorts", "jacket", "sneakers", "backpack",
               "watch", "sunglasses"},
    "streetwear": {"t-shirt", "hoodie", "jeans", "joggers", "shorts", "jacket", "sneakers", "boots",
                   "backpack", "sunglasses"},
    "party": {"shirt", "blouse", "dress", "skirt", "jeans", "trousers", "blazer", "jacket", "heels",
              "boots", "loafers", "watch", "handbag", "sunglasses"},
}

OCCASIONS = {
    "casual": ("weekend", "college", "travel", "everyday"),
    "formal": ("office", "interview", "meeting", "work"),
    "ethnic": ("wedding", "festive", "diwali", "puja"),
    "sporty": ("gym", "running", "workout", "training"),
    "streetwear": ("college", "concert", "hangout", "weekend"),
    "party": ("party", "date", "club", "night out"),
}

COLORS = ("black", "white", "navy", "grey", "beige", "olive", "maroon", "red", "blue",
          "pink", "green", "yellow", "brown", "mustard", "lavender", "teal")
NEUTRALS = {"black", "white", "navy", "grey", "beige", "brown"}

# Explicit harmonious non-neutral pairs (symmetric)
_HARMONY_PAIRS = {
    ("olive", "mustard"), ("maroon", "beige"), ("blue", "yellow"), ("pink", "lavender"),
    ("teal", "mustard"), ("green", "brown"), ("red", "navy"), ("lavender", "teal"),
    ("blue", "pink"), ("olive", "maroon"),
}
HARMONY = _HARMONY_PAIRS | {(b, a) for a, b in _HARMONY_PAIRS}

COLOR_SYNONYMS = {
    "dark blue": "navy", "navy blue": "navy", "off white": "beige", "cream": "beige",
    "khaki": "beige", "tan": "brown", "charcoal": "grey", "gray": "grey", "wine": "maroon",
    "burgundy": "maroon", "army green": "olive", "sky blue": "blue", "purple": "lavender",
    "turquoise": "teal", "golden": "mustard",
}

FITS = ("slim fit", "regular fit", "relaxed fit", "oversized", "tailored")
PATTERNS = ("solid", "striped", "checked", "printed", "textured", "embroidered")
SEASONS = ("summer", "winter", "monsoon", "all-season")

# Material -> natural season
MATERIAL_SEASON = {"linen": "summer", "cotton": "all-season", "wool": "winter",
                   "denim": "all-season", "silk": "all-season", "polyester": "monsoon",
                   "leather": "winter"}

BRAND_ROOTS = ("Aster", "Banyan", "Cobalt", "Driftwood", "Ember", "Fable", "Gully", "Hemline",
               "Indigo", "Juniper", "Kaveri", "Loom", "Monsoon", "Nomad", "Orbit", "Paisley",
               "Quill", "Rivet", "Saffron", "Thread", "Urban", "Vellum", "Weave", "Yarn",
               "Zephyr", "Alder", "Bodhi", "Cinder", "Dhaga", "Ecru", "Flint", "Garnet",
               "Hazel", "Ivory", "Jharokha", "Khadi")
BRAND_SUFFIX = ("Co.", "Studio", "Label", "Atelier", "Supply", "& Sons")
PRICE_TIERS = ("value", "mid", "premium", "luxury")
TIER_MULT = {"value": 0.55, "mid": 1.0, "premium": 1.8, "luxury": 3.5}


def colors_harmonize(a: str, b: str) -> bool:
    return a in NEUTRALS or b in NEUTRALS or (a, b) in HARMONY


def gender_compatible(a: str, b: str) -> bool:
    return a == b or "unisex" in (a, b)
