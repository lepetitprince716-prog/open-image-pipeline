"""Tests for scripts/csd_style.py: staging from review batches and the embedding report math."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import csd_style  # noqa: E402


def _review(root: Path, repo: Path, batch: str, count: int) -> None:
    items = []
    for i in range(count):
        image = repo / "outputs" / batch / f"img{i}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"jpg" + bytes([i]))
        items.append({"i": i, "kind": "repo", "path": str(image.relative_to(repo)), "name": image.name,
                      "media": "image", "caption": f"{batch} item {i}"})
    (root / batch).mkdir(parents=True)
    (root / batch / "manifest.json").write_text(json.dumps({"v": 2, "batch": batch, "mode": "review", "items": items}))


def _embeddings(csd_root: Path, batch: str, vectors: dict[str, list[float]]) -> None:
    payload = {"v": 1, "model": "test", "count": len(vectors),
               "items": [{"name": name, "style": vec, "content": vec} for name, vec in vectors.items()]}
    (csd_root / f"{batch}.json").write_text(json.dumps(payload))


def test_stage_copies_images_and_writes_index(tmp_path: Path):
    repo = tmp_path / "repo"
    review_root = repo / "review"
    csd_root = repo / "csd"
    _review(review_root, repo, "alpha", 2)
    _review(review_root, repo, "beta", 1)
    script = tmp_path / "boxa_csd_embed.py"
    script.write_text("print('embed')")

    index = csd_style.stage("r1", ["alpha", "beta"], csd_root=csd_root, review_root=review_root,
                            repo=repo, embed_script=script)

    stage_dir = csd_root / "stage" / "r1"
    assert [entry["name"] for entry in index["items"]] == ["alpha__000.jpg", "alpha__001.jpg", "beta__000.jpg"]
    assert (stage_dir / "alpha__001.jpg").read_bytes() == b"jpg\x01"
    assert (stage_dir / "boxa_csd_embed.py").is_file()
    assert json.loads((stage_dir / "index.json").read_text())["reviews"] == ["alpha", "beta"]
    with pytest.raises(csd_style.CsdError):
        csd_style.stage("r1", ["alpha"], csd_root=csd_root, review_root=review_root, repo=repo, embed_script=script)


def test_report_and_compare_use_style_cosine(tmp_path: Path):
    repo = tmp_path / "repo"
    review_root = repo / "review"
    csd_root = repo / "csd"
    _review(review_root, repo, "alpha", 2)
    _review(review_root, repo, "beta", 2)
    csd_style.stage("r1", ["alpha", "beta"], csd_root=csd_root, review_root=review_root,
                    repo=repo, embed_script=tmp_path / "missing.py")
    _embeddings(csd_root, "r1", {
        "alpha__000.jpg": [1.0, 0.0, 0.0],
        "alpha__001.jpg": [0.0, 1.0, 0.0],
        "beta__000.jpg": [0.99, 0.1, 0.0],
        "beta__001.jpg": [0.0, 0.0, 1.0],
    })

    result = csd_style.report("r1", top=1, csd_root=csd_root)
    assert result["groups"]["alpha"]["mean"] == 0.0
    assert result["groups"]["alpha"]["spread"] == 1.0
    assert result["duplicates"] == [{"a": "alpha__000.jpg", "b": "beta__000.jpg", "sim": 0.9949}]
    first = result["neighbours"][0]
    assert first["name"] == "alpha__000.jpg"
    assert first["neighbours"][0]["name"] == "beta__000.jpg"

    cross = csd_style.compare("r1", "alpha", "beta", csd_root=csd_root)
    assert [m["b"] for m in cross["matches"]] == ["beta__000.jpg", "beta__000.jpg"]
    assert cross["matches"][0]["sim"] == 0.9949
    with pytest.raises(csd_style.CsdError):
        csd_style.compare("r1", "alpha", "gamma", csd_root=csd_root)


def test_spread_handles_single_item():
    assert csd_style.spread(np.ones((1, 1)))["mean"] is None


def _corpus(tmp_path: Path):
    """Two staged batches: `new` holds a twin of a `corpus` item plus a genuinely new one."""
    repo = tmp_path / "repo"
    review_root = repo / "review"
    csd_root = repo / "csd"
    _review(review_root, repo, "old", 2)
    _review(review_root, repo, "fresh", 2)
    csd_style.stage("corpus", ["old"], csd_root=csd_root, review_root=review_root,
                    repo=repo, embed_script=tmp_path / "missing.py")
    csd_style.stage("new", ["fresh"], csd_root=csd_root, review_root=review_root,
                    repo=repo, embed_script=tmp_path / "missing.py")
    _embeddings(csd_root, "corpus", {"old__000.jpg": [1.0, 0.0, 0.0], "old__001.jpg": [0.0, 1.0, 0.0]})
    _embeddings(csd_root, "new", {"fresh__000.jpg": [1.0, 0.001, 0.0],   # twin of old__000
                                  "fresh__001.jpg": [0.0, 0.0, 1.0]})    # genuinely new
    return csd_root


def test_dedup_skips_only_the_candidate_with_a_near_twin_in_the_corpus(tmp_path: Path):
    csd_root = _corpus(tmp_path)
    result = csd_style.dedup("new", ["corpus"], csd_root=csd_root)
    assert [row["name"] for row in result["skip"]] == ["fresh__000.jpg"]
    assert [row["name"] for row in result["keep"]] == ["fresh__001.jpg"]
    assert result["skip"][0]["nearest"] == "old__000.jpg"
    assert result["skip"][0]["nearest_batch"] == "corpus"
    assert result["n"] == 2


def test_dedup_threshold_is_adjustable(tmp_path: Path):
    csd_root = _corpus(tmp_path)
    assert csd_style.dedup("new", ["corpus"], threshold=1.01, csd_root=csd_root)["skip"] == []


def test_dedup_never_matches_an_item_against_itself(tmp_path: Path):
    csd_root = _corpus(tmp_path)
    result = csd_style.dedup("new", ["new"], csd_root=csd_root)
    assert result["skip"] == []           # the only near-twin of each item is itself


def test_dedup_fails_closed_without_embeddings(tmp_path: Path):
    csd_root = _corpus(tmp_path)
    with pytest.raises(csd_style.CsdError):
        csd_style.dedup("new", ["never_embedded"], csd_root=csd_root)


def test_dedup_still_catches_the_same_image_staged_into_another_batch(tmp_path: Path):
    """The self-match guard must not excuse a genuine restage: same name, different batch."""
    repo = tmp_path / "repo"
    review_root = repo / "review"
    csd_root = repo / "csd"
    _review(review_root, repo, "fresh", 2)
    # the same review batch staged twice, so both CSD batches hold the same item names
    csd_style.stage("corpus", ["fresh"], csd_root=csd_root, review_root=review_root,
                    repo=repo, embed_script=tmp_path / "missing.py")
    csd_style.stage("new", ["fresh"], csd_root=csd_root, review_root=review_root,
                    repo=repo, embed_script=tmp_path / "missing.py")
    vectors = {"fresh__000.jpg": [1.0, 0.0, 0.0], "fresh__001.jpg": [0.0, 0.0, 1.0]}
    _embeddings(csd_root, "corpus", vectors)
    _embeddings(csd_root, "new", vectors)

    result = csd_style.dedup("new", ["corpus"], csd_root=csd_root)
    assert sorted(row["name"] for row in result["skip"]) == ["fresh__000.jpg", "fresh__001.jpg"]
    assert result["keep"] == []
