"""Tests that gate the science: correct labels, no leakage, determinism, and a
regression test that the PUBLISHED weights reproduce the headline behaviour.

    pytest test_nano_case.py -q
"""

from __future__ import annotations

import re

from data_cases import case_pairs, render


def _ref_render(words, case):
    t = [w[:1].upper() + w[1:] for w in words]
    return {
        "snake": "_".join(words), "kebab": "-".join(words),
        "const": "_".join(w.upper() for w in words),
        "camel": words[0] + "".join(t[1:]), "pascal": "".join(t),
    }[case]


def test_render_matches_independent_reference():
    for words, case, expect in [
        (["sdk", "model"], "const", "SDK_MODEL"),
        (["user", "table", "handler"], "camel", "userTableHandler"),
        (["sql", "query", "name"], "kebab", "sql-query-name"),
        (["render", "token", "error"], "pascal", "RenderTokenError"),
        (["md5", "cache"], "snake", "md5_cache"),
    ]:
        assert render(words, case) == expect == _ref_render(words, case)


def test_labels_correct_by_construction():
    sig = {
        "snake": re.compile(r"^[a-z0-9]+(_[a-z0-9]+)*$"),
        "kebab": re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$"),
        "const": re.compile(r"^[A-Z0-9]+(_[A-Z0-9]+)*$"),
        "camel": re.compile(r"^[a-z0-9][a-zA-Z0-9]*$"),
        "pascal": re.compile(r"^[A-Z0-9][a-zA-Z0-9]*$"),
    }
    for prompt, target in case_pairs(1, 4000):
        assert sig[prompt.split(" | ", 1)[0]].match(target), (prompt, target)


def test_deterministic():
    assert case_pairs(0, 200) == case_pairs(0, 200)


def test_holdout_low_overlap_with_train():
    train = set(case_pairs(0, 100000))
    holdout = case_pairs(987_654_321, 4000)
    assert sum(1 for p in holdout if p in train) / len(holdout) < 0.25


def test_published_weights_reproduce():
    # The shipped weights must still do the hard, regex-defeating conversions.
    import os
    if not (os.path.exists("model.safetensors") and os.path.exists("config.json")):
        import pytest
        pytest.skip("weights not present in this checkout")
    from modeling_nano_case import load, to_case
    m = load()
    for case, ident, gold in [
        ("const", "sdkmodel", "SDK_MODEL"),
        ("camel", "usertablehandler", "userTableHandler"),
        ("kebab", "sqlqueryname", "sql-query-name"),
        ("snake", "md5cache", "md5_cache"),
        ("pascal", "rendertokenerror", "RenderTokenError"),
    ]:
        assert to_case(m, case, ident) == gold, (case, ident)
