"""Tests for loramake.rounds: the round ledger derived from review batches and their traces."""
import json
from pathlib import Path

from loramake import rounds


def _tree(tmp_path: Path) -> dict:
    repo = tmp_path / "repo"
    review_root = repo / "outputs" / "_gallery" / "review_batches"
    render = repo / "outputs" / "civitai" / "probe_a" / "x" / "img.jpg"
    render.parent.mkdir(parents=True)
    render.write_bytes(b"jpg")
    (render.parent.parent / "gates_report.json").write_text(json.dumps(
        {"summary": {"images": 1, "overall": {"pass": 1, "fail": 0, "unverified": 0}}}))
    deck_render = repo / "outputs" / "nai" / "b" / "0.png"
    deck_render.parent.mkdir(parents=True)
    (deck_render.parent / "gate_report.json").write_text(json.dumps({"summary": {"overall": {"pass": 2, "fail": 0}}}))
    decks = repo / "configs" / "concept_decks"
    decks.mkdir(parents=True)
    (decks / "b_nai_2026.json").write_text(json.dumps({"deck": "b_nai_2026", "cards": []}))
    for batch, items, result in (
        ("probe_a", [{"i": 0, "path": str(render.relative_to(repo)), "caption": "[x] a"}], {"v": 1, "l": [0], "y": [], "n": [], "c": {}}),
        ("deck_b", [{"i": 0, "path": "outputs/nai/b/0.png"}, {"i": 1, "path": "outputs/nai/b/1.png"}], None),
    ):
        (review_root / batch).mkdir(parents=True)
        (review_root / batch / "manifest.json").write_text(json.dumps(
            {"v": 2, "batch": batch, "mode": "review", "created_at": f"2026-09-0{1 if batch == 'probe_a' else 2}T00:00:00Z", "items": items}))
        if result is not None:
            (review_root / batch / "result.json").write_text(json.dumps(dict(result, batch=batch)))
    ledger = repo / "outputs" / "civitai" / "buzz_ledger.jsonl"
    ledger.write_text('{"batch": "probe_a", "id": "x", "cost_buzz": 19}\n{"batch": "probe_a", "id": "x", "cost_buzz": 19}\nnot json\n')
    specs = repo / "configs" / "civitai_probes"
    specs.mkdir(parents=True)
    (specs / "a.json").write_text(json.dumps({"batch": "probe_a", "items": []}))
    cards = repo / "catalogs" / "style"
    cards.mkdir(parents=True)
    (cards / "c.json").write_text(json.dumps({"id": "c", "realizations": {"civitai": [
        {"id": "civitai:1@2", "evidence": {"ref": "outputs/_gallery/review_batches/probe_a"}}]}}))
    library = repo / "scripts" / "nai_artist_strings.json"
    library.parent.mkdir(parents=True)
    library.write_text(json.dumps({"cards": {"k": {"note": "R2 bare probe (yes, batch deck_b)"}}}))
    return {"repo": repo, "review_root": review_root, "ledger": ledger, "cards": cards, "library": library}


def test_collect_joins_verdicts_spend_spec_gates_and_writebacks(tmp_path: Path):
    tree = _tree(tmp_path)

    rows = rounds.collect(review_root=tree["review_root"], repo=tree["repo"], ledger_path=tree["ledger"],
                          cards_dir=tree["cards"], library_path=tree["library"])

    assert [row["batch"] for row in rows] == ["deck_b", "probe_a"]
    probe = rows[1]
    assert probe["verdicts"] == {"love": 1, "yes": 0, "no": 0, "comment": 0}
    assert probe["pending"] is False
    assert probe["spend"] == {"buzz": 38, "renders": 2}
    assert probe["spec"] == "configs/civitai_probes/a.json"
    assert probe["gates"] == {"pass": 1, "fail": 0, "unverified": 0}
    assert probe["writeback"] == {"registry": 1, "library": 0}
    deck = rows[0]
    assert deck["pending"] is True
    assert deck["decided"] == 0
    assert deck["spend"] is None
    assert deck["writeback"] == {"registry": 0, "library": 1}
    assert deck["gates"] == {"pass": 2, "fail": 0}
    assert deck["spec"] == "configs/concept_decks/b_nai_2026.json"


def test_collect_without_review_root_is_empty(tmp_path: Path):
    assert rounds.collect(review_root=tmp_path / "missing", repo=tmp_path, ledger_path=tmp_path / "l",
                          cards_dir=tmp_path / "c", library_path=tmp_path / "lib") == []
