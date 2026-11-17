from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import review_batch  # noqa: E402


@pytest.fixture
def isolated_repo(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    outputs = repo / "outputs"
    outputs.mkdir(parents=True)
    monkeypatch.setattr(review_batch, "ROOT", repo)
    monkeypatch.setattr(review_batch, "OUTPUTS", outputs)
    return repo


def _batch_dir(repo: Path, batch: str) -> Path:
    return repo / "outputs" / "_gallery" / "review_batches" / batch


def _read_manifest(repo: Path, batch: str) -> dict:
    path = _batch_dir(repo, batch) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _read_batch(repo: Path, batch: str) -> tuple[dict, dict]:
    """Manifest + result. Only valid for review batches; view batches have no result."""
    directory = _batch_dir(repo, batch)
    return (
        _read_manifest(repo, batch),
        json.loads((directory / "result.json").read_text(encoding="utf-8")),
    )


def test_repo_image_is_referenced_without_copy(isolated_repo):
    image = isolated_repo / "images" / "inside.png"
    image.parent.mkdir()
    image.write_bytes(b"inside-image")

    registration = review_batch.register_batch([str(image)], "inside")
    manifest, result = _read_batch(isolated_repo, "inside")

    digest = hashlib.sha256(b"inside-image").hexdigest()
    assert manifest["items"] == [
        {
            "i": 0,
            "kind": "repo",
            "path": "images/inside.png",
            "name": "inside.png",
            "media": "image",
            "caption": "",
            "sha256": digest,
            "source_path": image.as_posix(),
        }
    ]
    assert result == {"v": 1, "batch": "inside", "y": [], "l": [], "n": [], "c": {}}
    assert registration["manifest"] == (
        "outputs/_gallery/review_batches/inside/manifest.json"
    )
    assert not (_batch_dir(isolated_repo, "inside") / "assets").exists()


def test_external_image_is_snapshotted_and_symlink_cannot_escape(isolated_repo, tmp_path):
    external = tmp_path / "external.webp"
    external.write_bytes(b"external-image")
    link = isolated_repo / "looks-internal.webp"
    link.symlink_to(external)

    review_batch.register_batch([str(link)], "external")
    manifest, _ = _read_batch(isolated_repo, "external")

    item = manifest["items"][0]
    snapshot = _batch_dir(isolated_repo, "external") / item["path"]
    assert item["kind"] == "batch_asset"
    assert item["source_path"] == link.as_posix()
    assert item["sha256"] == hashlib.sha256(b"external-image").hexdigest()
    assert snapshot.parent == _batch_dir(isolated_repo, "external") / "assets"
    assert snapshot.read_bytes() == b"external-image"
    assert not snapshot.is_symlink()


def test_multiple_images_keep_input_order(isolated_repo, tmp_path):
    inside = isolated_repo / "one.jpg"
    outside = tmp_path / "two.jpeg"
    inside.write_bytes(b"one")
    outside.write_bytes(b"two")

    review_batch.register_batch([str(inside), str(outside)], "many")
    manifest, _ = _read_batch(isolated_repo, "many")

    assert [item["i"] for item in manifest["items"]] == [0, 1]
    assert [item["name"] for item in manifest["items"]] == ["one.jpg", "two.jpeg"]
    assert manifest["items"][0]["path"] == "one.jpg"
    assert manifest["items"][0]["kind"] == "repo"
    assert manifest["items"][1]["kind"] == "batch_asset"
    assert manifest["items"][1]["path"].startswith("assets/")


def test_external_output_root_keeps_snapshot_batch_relative(
    isolated_repo, tmp_path, monkeypatch
):
    outputs = tmp_path / "detached-outputs"
    external = tmp_path / "source.png"
    external.write_bytes(b"external")
    monkeypatch.setattr(review_batch, "OUTPUTS", outputs)

    review_batch.register_batch([str(external)], "detached")
    batch_dir = outputs / "_gallery" / "review_batches" / "detached"
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    item = manifest["items"][0]

    assert item["kind"] == "batch_asset"
    assert item["path"].startswith("assets/")
    assert not Path(item["path"]).is_absolute()
    assert (batch_dir / item["path"]).read_bytes() == b"external"


@pytest.mark.parametrize(
    ("name", "kind", "message"),
    [
        ("bad.gif", "file", "unsupported media extension"),
        ("missing.png", "missing", "does not exist"),
        ("folder.png", "directory", "not a file"),
    ],
)
def test_invalid_inputs_leave_no_batch(isolated_repo, name, kind, message):
    path = isolated_repo / name
    if kind == "file":
        path.write_bytes(b"gif")
    elif kind == "directory":
        path.mkdir()

    with pytest.raises(review_batch.ReviewBatchError, match=message):
        review_batch.register_batch([str(path)], "invalid")

    assert not _batch_dir(isolated_repo, "invalid").exists()


@pytest.mark.parametrize(
    "batch",
    ["-bad", ".bad", "_bad", "bad/slash", "bad space", "é", "a" * 129],
)
def test_batch_id_validation_is_strict(isolated_repo, batch):
    image = isolated_repo / "valid.png"
    image.write_bytes(b"png")

    with pytest.raises(review_batch.ReviewBatchError, match="batch ID"):
        review_batch.register_batch([str(image)], batch)


def test_existing_batch_is_never_overwritten(isolated_repo):
    image = isolated_repo / "valid.png"
    image.write_bytes(b"first")
    review_batch.register_batch([str(image)], "same")
    manifest_path = _batch_dir(isolated_repo, "same") / "manifest.json"
    before = manifest_path.read_bytes()
    image.write_bytes(b"second")

    with pytest.raises(review_batch.ReviewBatchError, match="already exists"):
        review_batch.register_batch([str(image)], "same")

    assert manifest_path.read_bytes() == before


def test_cli_json_output_url_and_atomic_file_layout(
    isolated_repo, monkeypatch, capsys
):
    image = isolated_repo / "valid.png"
    image.write_bytes(b"png")
    monkeypatch.setenv("LORAMAKE_GALLERY_PORT", "9001")

    assert review_batch.main(["--batch", "url.test", "--json", str(image)]) == 0

    output = capsys.readouterr().out
    assert output.count("\n") == 1
    registration = json.loads(output)
    assert registration["url"] == (
        "http://127.0.0.1:9001/review?batch=url.test"
    )
    directory = _batch_dir(isolated_repo, "url.test")
    assert sorted(path.name for path in directory.iterdir()) == [
        "manifest.json",
        "result.json",
    ]
    assert not list(directory.glob(".*.tmp"))


def test_generated_batch_id_is_valid_and_unique(isolated_repo):
    first = isolated_repo / "first.png"
    second = isolated_repo / "second.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    one = review_batch.register_batch([str(first)])
    two = review_batch.register_batch([str(second)])

    assert review_batch.BATCH_RE.fullmatch(one["batch"])
    assert review_batch.BATCH_RE.fullmatch(two["batch"])
    assert one["batch"] != two["batch"]


# --- video, captions, spec files and modes ------------------------------------------

def test_video_is_registered_with_media_kind_and_caption(isolated_repo):
    clip = isolated_repo / "shot.mp4"
    clip.write_bytes(b"clip-bytes")

    review_batch.register_batch(
        [str(clip)], "clips", captions=["Wan 2.2, 49 帧"], mode="review"
    )
    manifest, _ = _read_batch(isolated_repo, "clips")

    assert manifest["v"] == 2
    assert manifest["mode"] == "review"
    assert manifest["items"][0]["media"] == "video"
    assert manifest["items"][0]["caption"] == "Wan 2.2, 49 帧"


def test_view_mode_writes_no_result_file_and_returns_the_view_url(isolated_repo):
    image = isolated_repo / "look.png"
    image.write_bytes(b"look")

    registration = review_batch.register_batch([str(image)], "looker", mode="view")

    assert registration["mode"] == "view"
    assert registration["count"] == 1
    assert registration["url"].endswith("/view?batch=looker")
    assert not (_batch_dir(isolated_repo, "looker") / "result.json").exists()


def test_review_mode_is_the_default_and_returns_the_review_url(isolated_repo):
    image = isolated_repo / "decide.png"
    image.write_bytes(b"decide")

    registration = review_batch.register_batch([str(image)], "decider")

    assert registration["mode"] == "review"
    assert registration["url"].endswith("/review?batch=decider")
    assert (_batch_dir(isolated_repo, "decider") / "result.json").exists()


def test_spec_file_supplies_paths_and_captions(isolated_repo, tmp_path):
    first = isolated_repo / "one.png"
    second = isolated_repo / "two.mp4"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps([
        {"path": str(first), "caption": "方案 A"},
        str(second),
    ]), encoding="utf-8")

    assert review_batch.main(
        ["--spec", str(spec), "--batch", "spec-batch", "--mode", "view", "--json"]
    ) == 0
    manifest = _read_manifest(isolated_repo, "spec-batch")

    assert [item["caption"] for item in manifest["items"]] == ["方案 A", ""]
    assert [item["media"] for item in manifest["items"]] == ["image", "video"]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("[]", "non-empty JSON array"),
        ("{\"items\": []}", "non-empty JSON array"),
        ("[{\"caption\": \"no path\"}]", 'missing a non-empty "path"'),
        ("[{\"path\": 5}]", 'missing a non-empty "path"'),
        ("not json", "not valid JSON"),
    ],
)
def test_malformed_specs_are_rejected(tmp_path, payload, message):
    spec = tmp_path / "bad.json"
    spec.write_text(payload, encoding="utf-8")

    with pytest.raises(review_batch.ReviewBatchError, match=message):
        review_batch._spec_entries(str(spec))


def test_overlong_caption_is_rejected(isolated_repo):
    image = isolated_repo / "wordy.png"
    image.write_bytes(b"wordy")

    with pytest.raises(review_batch.ReviewBatchError, match="caption longer than"):
        review_batch.register_batch(
            [str(image)], "wordy", captions=["x" * (review_batch.CAPTION_MAX + 1)]
        )

    assert not _batch_dir(isolated_repo, "wordy").exists()


def test_mismatched_caption_count_is_rejected(isolated_repo):
    image = isolated_repo / "solo.png"
    image.write_bytes(b"solo")

    with pytest.raises(review_batch.ReviewBatchError, match="one-to-one"):
        review_batch.register_batch([str(image)], "solo", captions=["a", "b"])


def test_oversized_external_media_is_refused_before_copying(
    isolated_repo, tmp_path, monkeypatch
):
    external = tmp_path / "huge.mp4"
    external.write_bytes(b"huge")
    monkeypatch.setattr(review_batch, "SNAPSHOT_MAX_BYTES", 2)

    with pytest.raises(review_batch.ReviewBatchError, match="larger than"):
        review_batch.register_batch([str(external)], "huge")

    assert not _batch_dir(isolated_repo, "huge").exists()


def test_oversized_repo_media_is_allowed_because_it_is_never_copied(
    isolated_repo, monkeypatch
):
    inside = isolated_repo / "big.mp4"
    inside.write_bytes(b"big-but-inside")
    monkeypatch.setattr(review_batch, "SNAPSHOT_MAX_BYTES", 2)

    review_batch.register_batch([str(inside)], "big", mode="view")
    manifest = _read_manifest(isolated_repo, "big")

    assert manifest["items"][0]["kind"] == "repo"


def test_purely_local_batches_never_load_the_remote_transport(isolated_repo, monkeypatch):
    image = isolated_repo / "local.png"
    image.write_bytes(b"local")

    def fail_import(name, *args, **kwargs):
        raise AssertionError(f"unexpected import of {name}")

    monkeypatch.setattr(
        review_batch, "_localise", lambda paths, alias=None: fail_import("remote_media")
    )
    with pytest.raises(AssertionError):
        review_batch.register_batch([str(image)], "local")


def _stub_gallery(monkeypatch, health, legacy=False):
    """Install a fake gallery_app so serving_tree_warning has something to probe."""
    import types

    module = types.SimpleNamespace(
        probe_health=lambda port=None, timeout=2.0: health,
        is_legacy_gallery=lambda port=None, timeout=3.0: legacy,
    )
    monkeypatch.setitem(sys.modules, "gallery_app", module)


def test_serving_tree_warning_is_silent_when_the_gallery_serves_this_tree(
    isolated_repo, monkeypatch
):
    _stub_gallery(monkeypatch, {"service": "loramake-gallery",
                                "root": str(review_batch.ROOT),
                                "outputs": str(review_batch.OUTPUTS)})

    assert review_batch.serving_tree_warning("8771") is None


def test_serving_tree_warning_names_both_trees_on_a_mismatch(isolated_repo, monkeypatch):
    """The 2026-08-13 failure: the URL was printed against a gallery rooted elsewhere."""
    _stub_gallery(monkeypatch, {"service": "loramake-gallery",
                                "root": "/elsewhere/worktrees/other",
                                "outputs": "/elsewhere/worktrees/other/outputs"})

    warning = review_batch.serving_tree_warning("8771")

    assert warning is not None
    assert "/elsewhere/worktrees/other" in warning   # what actually answers
    assert str(review_batch.ROOT) in warning         # what the batch was written into
    assert "gallery_app.py restart" in warning       # and how to fix it


def test_serving_tree_warning_stays_quiet_when_nothing_answers(isolated_repo, monkeypatch):
    """Best-effort: nothing listening is not a problem worth a warning."""
    _stub_gallery(monkeypatch, None)

    assert review_batch.serving_tree_warning("8771") is None


def test_serving_tree_warning_flags_a_gallery_too_old_to_identify_itself(
    isolated_repo, monkeypatch
):
    """"Cannot verify" must not read as "verified fine" — that is how 2026-08-13 hid."""
    _stub_gallery(monkeypatch, None, legacy=True)

    warning = review_batch.serving_tree_warning("8771")

    assert warning is not None
    assert "predates /api/health" in warning
    assert "gallery_app.py restart" in warning


def test_registration_still_succeeds_and_exits_zero_despite_a_mismatch(
    isolated_repo, monkeypatch, capsys
):
    """The batch is written correctly; it is the server that is wrong, so stdout and
    the exit code must stay clean and the warning must go to stderr."""
    image = isolated_repo / "shot.png"
    image.write_bytes(b"shot")
    _stub_gallery(monkeypatch, {"service": "loramake-gallery", "root": "/elsewhere",
                                "outputs": "/elsewhere/outputs"})

    code = review_batch.main([str(image), "--batch", "mismatch", "--json"])

    captured = capsys.readouterr()
    assert code == 0
    assert json.loads(captured.out)["batch"] == "mismatch"
    assert "different tree" in captured.err
