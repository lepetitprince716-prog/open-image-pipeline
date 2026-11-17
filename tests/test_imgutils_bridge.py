"""Unit tests for scripts/imgutils_bridge.py failure-reason detection (PR #169 follow-up)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import imgutils_bridge as bridge  # noqa: E402

CACHE_MISS_HINT = "model not in local HF cache"


def _make_exc_class(name: str, module: str, base=Exception):
    cls = type(name, (base,), {})
    cls.__module__ = module
    return cls


def test_op_reason_matches_hf_cache_miss_via_mro_subclass():
    # huggingface_hub renamed LocalEntryNotFoundError -> EntryNotFoundError in newer releases;
    # a subclass of either must still be recognized via the MRO walk.
    base = _make_exc_class("EntryNotFoundError", "huggingface_hub.errors")
    subclass = _make_exc_class("RenamedCacheMissError", "huggingface_hub.errors", base=base)

    reason = bridge._op_reason("wd14 error", subclass("nope"))

    assert reason.startswith("wd14 error: ")
    assert CACHE_MISS_HINT in reason


def test_op_reason_matches_cache_miss_wrapped_in_cause_chain():
    inner = _make_exc_class("LocalEntryNotFoundError", "huggingface_hub.utils._errors")("missing")
    wrapper = RuntimeError("tagger load failed")
    wrapper.__cause__ = inner

    reason = bridge._op_reason("qc error", wrapper)

    assert CACHE_MISS_HINT in reason


def test_op_reason_matches_cache_miss_via_context_chain():
    inner_cls = _make_exc_class("EntryNotFoundError", "huggingface_hub")
    try:
        raise inner_cls("missing")
    except inner_cls:
        try:
            raise ValueError("outer")
        except ValueError as outer:
            reason = bridge._op_reason("wd14 error", outer)

    assert CACHE_MISS_HINT in reason


def test_op_reason_falls_back_to_plain_name_check():
    # Legacy behavior preserved: a class literally named LocalEntryNotFoundError is treated
    # as a cache miss even without a huggingface_hub module path (e.g. a stub in tests/venvs).
    stub = _make_exc_class("LocalEntryNotFoundError", "some_test_stub")

    assert CACHE_MISS_HINT in bridge._op_reason("wd14 error", stub("x"))


def test_op_reason_generic_error_stays_generic():
    class EntryNotFoundError(Exception):
        pass  # right name, wrong module — not an HF cache miss, no fallback match either

    EntryNotFoundError.__module__ = "unrelated_pkg.errors"

    assert bridge._op_reason("ccip error", EntryNotFoundError("x")) == "ccip error: EntryNotFoundError"
    assert bridge._op_reason("ccip error", ValueError("x")) == "ccip error: ValueError"


def test_op_reason_handles_cyclic_exception_chains():
    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__context__ = b
    b.__context__ = a  # pathological cycle must terminate, not hang

    assert bridge._op_reason("qc error", a) == "qc error: RuntimeError"
