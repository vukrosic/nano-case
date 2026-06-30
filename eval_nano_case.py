"""Benchmark nano-case against a naive regex script — the whole point of the model.

Reports exact-match accuracy for the model AND for a standard regex case-converter,
overall and on the regex-killer "smushed" slice (boundary-destroyed inputs: no
separators, single global case, multi-word — where a splitter has nothing to go on).
Uses batched greedy decode (fast and exact); the data path is data_cases.case_pairs,
identical to training.

    python eval_nano_case.py [--n 4000] [--weights model.safetensors] [--config config.json]
"""

from __future__ import annotations

import argparse
import re

import torch

from data_cases import case_pairs, render
from modeling_nano_case import load

_HOLDOUT_SEED = 987_654_321
_EOS = 10


def _split(s: str) -> list[str]:
    words: list[str] = []
    for part in re.split(r"[\s_\-./]+", s):
        if not part:
            continue
        part = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", part)
        part = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", part)
        part = re.sub(r"(?<=[a-zA-Z])(?=[0-9])", " ", part)
        part = re.sub(r"(?<=[0-9])(?=[a-zA-Z])", " ", part)
        words += part.split()
    return [w.lower() for w in words if w]


def regex_convert(inp: str, case: str) -> str:
    """The script the model must beat: split on separators/humps, lowercase, render."""
    words = _split(inp)
    return render(words, case) if words else inp


def is_hard(inp: str, target: str) -> bool:
    """Boundary-destroyed (no separator, single global case) AND multi-word."""
    if re.search(r"[\s_\-./]", inp):
        return False
    letters = [c for c in inp if c.isalpha()]
    if not (letters and (inp.islower() or inp.isupper())):
        return False
    return len(_split(target)) >= 2


@torch.no_grad()
def greedy_batch(model, prompts, n_new, device="cpu"):
    """Batched greedy decode, bucketed by prompt length (exact, padding-free)."""
    enc = [list(p.encode("utf-8")) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(enc[i]))
    max_seq = model.cfg["max_seq_len"]
    preds = [None] * len(prompts)
    i = 0
    while i < len(order):
        L = len(enc[order[i]])
        j = i
        while j < len(order) and len(enc[order[j]]) == L:
            j += 1
        rows = order[i:j]
        toks = torch.tensor([enc[k] for k in rows], dtype=torch.long, device=device)
        for _ in range(n_new):
            nxt = model(toks[:, -max_seq:])[:, -1, :].argmax(-1, keepdim=True)
            toks = torch.cat([toks, nxt], dim=1)
        tail = toks[:, L:L + n_new]
        for r, k in enumerate(rows):
            s = bytes(b & 0xFF for b in tail[r].tolist())
            preds[k] = s.split(b"\n", 1)[0].decode("utf-8", "replace")
        i = j
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--weights", default="model.safetensors")
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load(args.weights, args.config, device=device)
    pairs = case_pairs(_HOLDOUT_SEED, args.n)
    prompts = [p for p, _ in pairs]
    preds = greedy_batch(model, prompts, 40, device=device)

    m_ok = b_ok = 0
    sm = sm_m = sm_b = 0
    for (prompt, target), pred in zip(pairs, preds):
        case, rest = prompt.split(" | ", 1)
        inp = rest.rsplit(" => ", 1)[0]
        base = regex_convert(inp, case)
        mo, bo = pred == target, base == target
        m_ok += mo
        b_ok += bo
        if is_hard(inp, target):
            sm += 1
            sm_m += mo
            sm_b += bo
    n = len(pairs)
    params = sum(p.numel() for p in model.parameters())
    print(f"params     : {params:,}")
    print(f"held-out   : {n} examples (seed {_HOLDOUT_SEED})\n")
    print(f"{'':14}{'model':>10}{'regex script':>16}")
    print(f"  {'overall':12}{m_ok/n:>9.1%}{b_ok/n:>16.1%}   (N={n})")
    if sm:
        print(f"  {'smushed':12}{sm_m/sm:>9.1%}{sm_b/sm:>16.1%}   "
              f"(N={sm} — no separators, single case: the regex-killer slice)")


if __name__ == "__main__":
    main()
