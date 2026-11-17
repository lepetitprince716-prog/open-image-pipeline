#!/usr/bin/env python3
"""Tests for the wildcard prompt engine.

Runs under pytest (`pytest scripts/test_wildcard.py`) or as a plain script
(`python3 scripts/test_wildcard.py`) so it can be verified without pytest installed.
File-wildcard tests build a throwaway directory with tempfile, so they do not depend on
the shipped wildcards/ content.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wildcard import WildcardEngine, WildcardError, expand  # noqa: E402


def test_plain_text_passes_through() -> None:
    assert WildcardEngine().expand("1girl, blue dress, rooftop") == "1girl, blue dress, rooftop"


def test_choice_is_deterministic_for_a_seed() -> None:
    eng = WildcardEngine()
    assert eng.expand("{red|green|blue}", seed=1) == eng.expand("{red|green|blue}", seed=1)


def test_choice_only_yields_listed_options() -> None:
    eng = WildcardEngine()
    seen = {eng.expand("{red|green|blue}", seed=s) for s in range(40)}
    assert seen <= {"red", "green", "blue"}
    assert len(seen) >= 2  # variety across seeds


def test_literal_braces_without_pipe_are_kept() -> None:
    assert WildcardEngine().expand("weight {single} here") == "weight {single} here"


def test_zero_weight_option_is_never_picked() -> None:
    eng = WildcardEngine()
    assert all(eng.expand("{1::keep|0::drop}", seed=s) == "keep" for s in range(30))


def test_multi_select_picks_exact_count_distinct() -> None:
    eng = WildcardEngine()
    for s in range(20):
        parts = eng.expand("{2$$a|b|c}", seed=s).split(", ")
        assert len(parts) == 2
        assert len(set(parts)) == 2
        assert set(parts) <= {"a", "b", "c"}


def test_multi_select_range_count() -> None:
    eng = WildcardEngine()
    counts = {len(eng.expand("{1-3$$a|b|c}", seed=s).split(", ")) for s in range(40)}
    assert counts <= {1, 2, 3}
    assert len(counts) >= 2


def test_nested_groups_resolve() -> None:
    eng = WildcardEngine()
    seen = {eng.expand("{a {b|b}|c}", seed=s) for s in range(30)}
    assert seen <= {"a b", "c"}


def test_file_wildcard_reads_a_line() -> None:
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "pose.txt").write_text("standing\nsitting\n# comment\n\nkneeling\n", encoding="utf-8")
        eng = WildcardEngine(d)
        seen = {eng.expand("pose: __pose__", seed=s) for s in range(40)}
        assert seen <= {"pose: standing", "pose: sitting", "pose: kneeling"}
        assert len(seen) >= 2


def test_inline_metadata_marker_is_stripped() -> None:
    """N3: pool lines carry ' # motion:..., register:...' metadata; the engine must strip it."""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "pose.txt").write_text(
            "standing contrapposto # motion:static\n"
            "twirling in place, skirt flaring # motion:dance, register:bubbly-cute\n",
            encoding="utf-8",
        )
        eng = WildcardEngine(d)
        seen = {eng.expand("__pose__", seed=s) for s in range(30)}
        assert seen <= {"standing contrapposto", "twirling in place, skirt flaring"}
        assert all(" # " not in v and "motion:" not in v and "register:" not in v for v in seen)


def test_file_wildcard_line_can_nest() -> None:
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "mood.txt").write_text("{calm|calm} light\n", encoding="utf-8")
        assert WildcardEngine(d).expand("__mood__", seed=3) == "calm light"


def test_missing_file_raises() -> None:
    with tempfile.TemporaryDirectory() as d:
        try:
            WildcardEngine(d).expand("__nope__")
        except WildcardError:
            return
        raise AssertionError("expected WildcardError for a missing wildcard file")


def test_path_traversal_is_blocked() -> None:
    with tempfile.TemporaryDirectory() as d:
        try:
            WildcardEngine(d).expand("__../secret__")
        except WildcardError:
            return
        raise AssertionError("expected WildcardError for a traversal escape")


def test_recursion_cycle_is_guarded() -> None:
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "loop.txt").write_text("__loop__\n", encoding="utf-8")
        try:
            WildcardEngine(d).expand("__loop__")
        except WildcardError:
            return
        raise AssertionError("expected WildcardError for a self-referential cycle")


def test_expand_many_is_reproducible_and_varies() -> None:
    eng = WildcardEngine()
    first = eng.expand_many("{a|b|c|d|e}", n=8, seed=99)
    second = eng.expand_many("{a|b|c|d|e}", n=8, seed=99)
    assert first == second
    assert len(set(first)) >= 2


def test_module_level_expand() -> None:
    assert expand("plain") == "plain"


def _run_all() -> int:
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in funcs:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001 - test harness reports any failure
            failed += 1
            print(f"FAIL {fn.__name__}: {exc!r}")
    print(f"\n{len(funcs) - failed}/{len(funcs)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
