"""Tests for reading one batch's decisions back out."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import review_batch  # noqa: E402
import review_result  # noqa: E402


@pytest.fixture
def batches(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    outputs = repo / "outputs"
    outputs.mkdir(parents=True)
    for module in (review_batch, review_result):
        monkeypatch.setattr(module, "OUTPUTS", outputs)
    monkeypatch.setattr(review_batch, "ROOT", repo)
    return repo


def _register(repo: Path, batch: str, names: list[str], captions: list[str] | None = None,
              mode: str = "review") -> None:
    paths = []
    for name in names:
        target = repo / name
        target.write_bytes(name.encode())
        paths.append(str(target))
    review_batch.register_batch(paths, batch, captions=captions, mode=mode)


def _write_result(repo: Path, batch: str, payload: dict) -> None:
    path = repo / "outputs" / "_gallery" / "review_batches" / batch / "result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_untouched_review_batch_reports_everything_pending(batches):
    _register(batches, "fresh", ["a.png", "b.png", "c.mp4"])

    report = review_result.read_result("fresh")

    assert report["mode"] == "review"
    assert report["total"] == 3
    assert report["reviewed"] == 0
    assert [item["i"] for item in report["pending"]] == [0, 1, 2]
    assert report["love"] == report["yes"] == report["no"] == []
    assert report["comments"] == {}


def test_decisions_are_joined_with_names_captions_and_media(batches):
    _register(batches, "done", ["a.png", "b.png", "c.mp4"],
              captions=["方案 A", "方案 B", "片段"])
    _write_result(batches, "done", {
        "v": 1, "batch": "done", "y": [2], "n": [0], "c": {"1": "手指再修一下"},
    })

    report = review_result.read_result("done")

    assert report["reviewed"] == 3
    assert report["pending"] == []
    assert report["yes"] == [
        {"i": 2, "name": "c.mp4", "media": "video", "caption": "片段"}
    ]
    assert report["no"] == [
        {"i": 0, "name": "a.png", "media": "image", "caption": "方案 A"}
    ]
    assert report["comments"] == {
        "1": {"i": 1, "name": "b.png", "media": "image", "caption": "方案 B",
              "comment": "手指再修一下"}
    }


def test_love_is_reported_as_its_own_tier_without_duplicating_yes(batches):
    _register(batches, "loved", ["a.png", "b.png", "c.mp4"], captions=["", "", "片段"])
    _write_result(batches, "loved", {
        "v": 1, "batch": "loved", "y": [0], "l": [2], "n": [1], "c": {},
    })

    report = review_result.read_result("loved")

    assert report["love"] == [
        {"i": 2, "name": "c.mp4", "media": "video", "caption": "片段"}
    ]
    assert [item["i"] for item in report["yes"]] == [0]
    assert report["reviewed"] == 3
    assert report["pending"] == []


def test_results_written_before_the_love_tier_read_back_with_an_empty_love(batches):
    _register(batches, "legacy", ["a.png", "b.png"])
    _write_result(batches, "legacy", {"v": 1, "batch": "legacy", "y": [0], "n": [], "c": {}})

    report = review_result.read_result("legacy")

    assert report["love"] == []
    assert [item["i"] for item in report["yes"]] == [0]
    assert report["reviewed"] == 1
    assert [item["i"] for item in report["pending"]] == [1]


def test_partial_review_keeps_the_undecided_items_in_pending(batches):
    _register(batches, "half", ["a.png", "b.png", "c.png", "d.png"])
    _write_result(batches, "half", {
        "v": 1, "batch": "half", "y": [0], "n": [], "c": {"2": "偏色"},
    })

    report = review_result.read_result("half")

    assert report["reviewed"] == 2
    assert [item["i"] for item in report["pending"]] == [1, 3]


def test_view_batches_report_no_decisions_rather_than_all_pending(batches):
    _register(batches, "looker", ["a.png", "b.png"], mode="view")

    report = review_result.read_result("looker")

    assert report["mode"] == "view"
    assert report["total"] == 2
    assert report["reviewed"] == 0
    assert report["pending"] == []
    assert report["love"] == []


def test_indexes_absent_from_the_manifest_are_ignored(batches):
    _register(batches, "stale", ["a.png"])
    _write_result(batches, "stale", {
        "v": 1, "batch": "stale", "y": [0, 99], "n": [], "c": {"42": "ghost"},
    })

    report = review_result.read_result("stale")

    assert [item["i"] for item in report["yes"]] == [0]
    assert report["comments"] == {}
    assert report["reviewed"] == 1


@pytest.mark.parametrize("batch", ["..", "../escape", "bad/slash", "bad space", ""])
def test_invalid_batch_ids_are_refused(batches, batch):
    with pytest.raises(review_result.ReviewResultError, match="invalid batch ID"):
        review_result.read_result(batch)


def test_unknown_batch_is_reported(batches):
    with pytest.raises(review_result.ReviewResultError, match="unknown batch"):
        review_result.read_result("nope")


def test_cli_json_output_is_one_line(batches, capsys):
    _register(batches, "cli", ["a.png"])
    _write_result(batches, "cli", {"v": 1, "batch": "cli", "y": [0], "n": [], "c": {}})

    assert review_result.main(["cli", "--json"]) == 0

    output = capsys.readouterr().out
    assert output.count("\n") == 1
    assert json.loads(output)["yes"][0]["name"] == "a.png"


def test_cli_text_output_lists_each_bucket(batches, capsys):
    _register(batches, "text", ["a.png", "b.png", "c.png", "d.png"])
    _write_result(batches, "text", {
        "v": 1, "batch": "text", "y": [0], "l": [3], "n": [1], "c": {},
    })

    assert review_result.main(["text"]) == 0

    output = capsys.readouterr().out
    assert "batch text (review) — 3/4 decided" in output
    assert "love (1):\n  [3] d.png" in output
    assert "yes (1):\n  [0] a.png" in output
    assert "no (1):\n  [1] b.png" in output
    assert "pending (1):\n  [2] c.png" in output
