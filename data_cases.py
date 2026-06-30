"""nano-case data — code-generated (messy identifier -> target case style).

Pure / standalone (numpy only): the same generator used for training and eval, so
the data path is byte-identical. The recipe: sample a word list from a small fixed
vocabulary, pick a target case, render the gold target canonically, and corrupt a
copy into a messy *input* — ~45% of the time into a boundary-DESTROYED form (no
separators, single global case) that a regex cannot segment. The label is correct
by construction (it's the canonical render of the very words used).

    <case> | <messy identifier> => <canonical render>
"""

from __future__ import annotations

import numpy as np

# A small, fixed identifier vocabulary (~90 words). This set IS the prior the model
# uses to segment boundary-free inputs; kept small so a 1M model can learn it.
_WORDS = [
    "user", "name", "id", "key", "value", "count", "index", "list", "map", "item",
    "node", "token", "buffer", "stream", "file", "path", "data", "model", "view",
    "config", "field", "form", "input", "output", "error", "status", "code",
    "message", "header", "body", "param", "query", "result", "cache", "queue",
    "worker", "task", "job", "event", "state", "store", "action", "handler",
    "manager", "service", "client", "server", "request", "response", "session",
    "account", "profile", "address", "number", "label", "title", "content",
    "image", "video", "audio", "color", "width", "height", "offset", "size",
    "length", "parser", "builder", "factory", "adapter", "filter", "render",
    "update", "create", "delete", "fetch", "load", "save", "send", "parse",
    "connection", "timeout", "retry", "limit", "page", "row", "column", "table",
]
_ACRONYMS = [
    "http", "https", "json", "html", "xml", "api", "url", "uri", "db", "io",
    "ui", "ux", "css", "sql", "jwt", "utf", "ascii", "gpu", "cpu", "ssl", "tcp",
    "udp", "dns", "uuid", "md5", "sha", "ip", "os", "sdk", "cli", "gui",
]
_DIGITS = ["2", "3", "4", "8", "16", "32", "64", "v1", "v2", "x"]
_CASES = ["snake", "kebab", "camel", "pascal", "const"]


def _title(w: str) -> str:
    return w[:1].upper() + w[1:]


def render(words: list[str], case: str) -> str:
    """Deterministic canonical rendering of a word list in a target case."""
    if case == "snake":
        return "_".join(words)
    if case == "kebab":
        return "-".join(words)
    if case == "const":
        return "_".join(w.upper() for w in words)
    if case == "camel":
        return words[0] + "".join(_title(w) for w in words[1:])
    if case == "pascal":
        return "".join(_title(w) for w in words)
    raise ValueError(case)


def _corrupt_input(rng, words: list[str]) -> str:
    mode = rng.random()
    if mode < 0.45:
        # DESTROYED: no separators, single global case — regex can't segment.
        glued = "".join(words)
        roll = rng.random()
        if roll < 0.45:
            return glued.lower()
        if roll < 0.9:
            return glued.upper()
        return "".join(c.upper() if rng.random() < 0.5 else c for c in glued)
    if mode < 0.7:
        return render(words, _CASES[int(rng.integers(len(_CASES)))])  # clean other-style
    if mode < 0.85:
        sep = rng.choice([" ", ".", "/"])
        return sep.join(w.upper() if rng.random() < 0.3 else
                        (_title(w) if rng.random() < 0.3 else w) for w in words)
    seps = ["_", "-", ".", "__", "-_"]  # garbage separators
    out = []
    for i, w in enumerate(words):
        if i:
            out.append(rng.choice(seps))
        out.append(w.upper() if rng.random() < 0.25 else
                   (_title(w) if rng.random() < 0.35 else w))
    return "".join(out)


def _sample_words(rng) -> list[str]:
    n = int(rng.choice([1, 2, 3, 4], p=[0.12, 0.45, 0.33, 0.10]))
    words = []
    for _ in range(n):
        roll = rng.random()
        if roll < 0.22:
            words.append(_ACRONYMS[int(rng.integers(len(_ACRONYMS)))])
        elif roll < 0.30:
            words.append(_DIGITS[int(rng.integers(len(_DIGITS)))])
        else:
            words.append(_WORDS[int(rng.integers(len(_WORDS)))])
    return words


def case_pairs(seed: int, n: int) -> list[tuple[str, str]]:
    """`n` deterministic (prompt, target) pairs from `seed`."""
    rng = np.random.default_rng(seed)
    out: list[tuple[str, str]] = []
    for _ in range(n):
        words = _sample_words(rng)
        target_case = _CASES[int(rng.integers(len(_CASES)))]
        inp = _corrupt_input(rng, words)
        target = render(words, target_case)
        out.append((f"{target_case} | {inp} => ", target))
    return out
