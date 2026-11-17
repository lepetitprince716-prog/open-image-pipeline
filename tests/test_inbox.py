"""Tests for loramake.inbox, the question store behind the gallery inbox page."""
from pathlib import Path

import pytest

from loramake import inbox


def _ask(data: dict, qid: str = "q1") -> dict:
    return inbox.ask(data, qid=qid, title="Spend 204 Buzz?", detail="12 renders",
                     options=["approve", "hold"])


def test_ask_answer_roundtrip_through_the_store(tmp_path: Path):
    store = tmp_path / "inbox.json"
    data = inbox.load(store)
    _ask(data)
    inbox.save(store, data)

    loaded = inbox.load(store)
    assert [question["id"] for question in inbox.pending(loaded)] == ["q1"]
    inbox.answer(loaded, qid="q1", value="approve")
    inbox.save(store, loaded)

    again = inbox.load(store)
    assert again["questions"][0]["answer"] == "approve"
    assert again["questions"][0]["answered"]
    assert inbox.pending(again) == []


def test_ask_rejects_duplicate_id_unless_replace():
    data = inbox.empty()
    _ask(data)
    with pytest.raises(inbox.InboxError):
        _ask(data)
    inbox.ask(data, qid="q1", title="again", options=["a"], replace=True)
    assert [question["title"] for question in data["questions"]] == ["again"]


@pytest.mark.parametrize("bad", [
    {"qid": "Bad Id"},
    {"title": ""},
    {"title": "t" * (inbox.TITLE_MAX + 1)},
    {"options": []},
    {"options": ["a", "a"]},
    {"options": ["x"] * (inbox.OPTIONS_MAX + 1)},
])
def test_ask_validates_fields(bad: dict):
    kwargs = {"qid": "q1", "title": "t", "options": ["a", "b"]}
    kwargs.update(bad)
    with pytest.raises(inbox.InboxError):
        inbox.ask(inbox.empty(), **kwargs)


def test_answer_rejects_unknown_empty_and_oversized_values():
    data = inbox.empty()
    _ask(data)
    with pytest.raises(inbox.InboxError):
        inbox.answer(data, qid="missing", value="x")
    with pytest.raises(inbox.InboxError):
        inbox.answer(data, qid="q1", value="   ")
    with pytest.raises(inbox.InboxError):
        inbox.answer(data, qid="q1", value="x" * (inbox.ANSWER_MAX + 1))


def test_reopen_and_drop():
    data = inbox.empty()
    _ask(data)
    inbox.answer(data, qid="q1", value="hold")
    inbox.reopen(data, qid="q1")
    assert data["questions"][0]["answer"] is None
    assert data["questions"][0]["answered"] is None
    assert inbox.drop(data, qid="q1") is True
    assert inbox.drop(data, qid="q1") is False


def test_load_rejects_malformed_store(tmp_path: Path):
    store = tmp_path / "inbox.json"
    store.write_text('{"v": 1, "questions": [{"id": "q1"}]}', encoding="utf-8")
    with pytest.raises(inbox.InboxError):
        inbox.load(store)
    store.write_text("not json", encoding="utf-8")
    with pytest.raises(inbox.InboxError):
        inbox.load(store)
