"""Tokenizer + light stemmer shared by indexing and query time (they must agree)."""
from __future__ import annotations

import re

_SPLIT = re.compile(r"[a-z0-9]+")
STOP = {"a", "an", "the", "for", "in", "and", "of", "with", "to", "it", "is", "on", "my", "me",
        "i", "want", "need", "some", "show", "looking", "buy", "pick", "perfect", "wear",
        "all", "year", "under", "below", "less", "than", "above", "over", "rs", "inr", "best"}


def normalize(text: str) -> str:
    t = text.lower().replace("t-shirt", "tshirt").replace("t shirt", "tshirt")
    return t.replace("&", " and ").replace("-", " ")


def stem(tok: str) -> str:
    if len(tok) <= 3 or tok.isdigit():
        return tok
    for suf, rep in (("ies", "y"), ("sses", "ss"), ("shes", "sh"), ("ches", "ch")):
        if tok.endswith(suf):
            return tok[: -len(suf)] + rep
    if tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def tokenize(text: str, drop_stop: bool = True) -> list[str]:
    toks = _SPLIT.findall(normalize(text))
    return [stem(t) for t in toks if not (drop_stop and t in STOP)]
