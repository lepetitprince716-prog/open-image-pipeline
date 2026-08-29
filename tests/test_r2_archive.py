"""Staging, collision, and quota-gate tests for the R2 archiver (scripts/r2_archive.py)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "r2_archive.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("r2_archive", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # The module defines a dataclass, which resolves its own module via sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def archiver():
    return _load_module()


def test_ceiling_is_eighty_percent_of_a_decimal_free_tier(archiver):
    # Cloudflare states the R2 free tier in decimal GB; computing in GiB would
    # leave the gate ~7% looser than the code claims.
    assert archiver.FREE_TIER_BYTES == 10_000_000_000
    assert archiver.QUOTA_CEILING_BYTES == 8_000_000_000


def test_transparent_png_keeps_alpha_through_transcode(archiver, tmp_path):
    source = tmp_path / "transparent.png"
    Image.new("RGBA", (64, 64), (255, 0, 0, 128)).save(source)
    staged = tmp_path / "out" / "transparent.png.webp"

    assert archiver.stage_one((str(source), str(staged), "transcode")) is not None

    with Image.open(staged) as image:
        assert image.mode == "RGBA"


def test_opaque_png_is_not_promoted_to_rgba(archiver, tmp_path):
    source = tmp_path / "opaque.png"
    Image.new("RGB", (64, 64), (0, 128, 255)).save(source)
    staged = tmp_path / "out" / "opaque.png.webp"

    archiver.stage_one((str(source), str(staged), "transcode"))

    with Image.open(staged) as image:
        assert image.mode == "RGB"


def test_image_and_same_stem_thumbnail_get_distinct_staging_paths(archiver, tmp_path):
    src = tmp_path / "gallery"
    src.mkdir()
    Image.new("RGB", (8, 8)).save(src / "portrait.png")
    Image.new("RGB", (8, 8)).save(src / "portrait.webp", format="WEBP")

    jobs = archiver.build_jobs(src, "test")

    assert len(jobs) == 2
    assert len({job[1] for job in jobs}) == 2


def test_colliding_sources_raise_instead_of_dropping_one(archiver, tmp_path):
    src = tmp_path / "collide"
    src.mkdir()
    Image.new("RGB", (8, 8)).save(src / "a.png")
    Image.new("RGB", (8, 8)).save(src / "a.png.webp", format="WEBP")

    with pytest.raises(RuntimeError, match="staging path collision"):
        archiver.build_jobs(src, "test")


def test_prune_removes_staged_files_whose_source_is_gone(archiver, tmp_path, monkeypatch):
    monkeypatch.setattr(archiver, "STAGING_DIR", tmp_path / "staging")
    staged_root = tmp_path / "staging" / "test"
    staged_root.mkdir(parents=True)
    keep = staged_root / "kept.png.webp"
    drop = staged_root / "orphan.png.webp"
    keep.write_bytes(b"x")
    drop.write_bytes(b"y")

    removed = archiver.prune_staging("test", {str(keep)})

    assert removed == 1
    assert keep.exists()
    assert not drop.exists()


@pytest.mark.parametrize("bad", ["/etc", "../escape", "a/../../escape", "", " lead"])
def test_validate_prefix_rejects_paths_that_escape_the_staging_tree(archiver, bad):
    with pytest.raises(ValueError):
        archiver.validate_prefix(bad)


@pytest.mark.parametrize("bad", [".", "./."])
def test_validate_prefix_rejects_prefixes_naming_the_staging_root(archiver, bad):
    # These stay inside the tree but name all of it, which would widen a prune
    # from one prefix to every prefix.
    with pytest.raises(ValueError, match="staging root"):
        archiver.validate_prefix(bad)


@pytest.mark.parametrize("ok", ["a/.", "./a"])
def test_validate_prefix_accepts_dot_segments_that_normalise_to_a_subdirectory(archiver, ok):
    # pathlib folds these to "a", so they name one prefix and are harmless.
    assert archiver.validate_prefix(ok) == ok


def test_validate_prefix_accepts_ordinary_prefixes(archiver):
    assert archiver.validate_prefix("gallery") == "gallery"
    assert archiver.validate_prefix("gallery/2026-08") == "gallery/2026-08"


def test_prune_clears_symlinks_without_following_or_crashing(archiver, tmp_path, monkeypatch):
    monkeypatch.setattr(archiver, "STAGING_DIR", tmp_path / "staging")
    staged_root = tmp_path / "staging" / "test"
    staged_root.mkdir(parents=True)

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "precious.txt"
    outside_file.write_bytes(b"keep me")

    # A directory symlink used to reach the rmdir sweep and raise NotADirectoryError.
    (staged_root / "dirlink").symlink_to(outside_dir, target_is_directory=True)
    (staged_root / "filelink").symlink_to(outside_file)

    removed = archiver.prune_staging("test", set())

    assert removed == 2
    assert not (staged_root / "dirlink").exists()
    assert not (staged_root / "filelink").exists()
    assert outside_dir.is_dir()
    assert outside_file.read_bytes() == b"keep me"


@pytest.mark.parametrize("target", ["outside", "inside"])
@pytest.mark.parametrize("prefix", ["test", "test/deeper"])
def test_a_symlinked_path_component_is_refused_everywhere(
    archiver, tmp_path, monkeypatch, target, prefix
):
    """A link at any component — final or intermediate — must stop every entry point.

    A link pointing outside is already caught by the resolve() containment check, but
    one pointing at a sibling prefix inside the tree resolves cleanly, so only the
    per-component check stops it.
    """
    monkeypatch.setattr(archiver, "STAGING_DIR", tmp_path / "staging")
    staging = tmp_path / "staging"
    staging.mkdir()

    destination = tmp_path / "outside" if target == "outside" else staging / "other"
    destination.mkdir()
    planted = destination / "someone-elses-link"
    planted.symlink_to(tmp_path / "whatever")

    (staging / "test").symlink_to(destination, target_is_directory=True)

    with pytest.raises(ValueError, match="is a symlink"):
        archiver.validate_prefix(prefix)
    with pytest.raises(ValueError, match="is a symlink"):
        archiver.prune_staging(prefix, set())
    with pytest.raises(ValueError, match="is a symlink"):
        archiver.sync(prefix, None, False)

    assert planted.is_symlink()


def test_prune_refuses_an_escaping_prefix_before_unlinking(archiver, tmp_path, monkeypatch):
    monkeypatch.setattr(archiver, "STAGING_DIR", tmp_path / "staging")
    victim_dir = tmp_path / "victim"
    victim_dir.mkdir()
    victim = victim_dir / "precious.txt"
    victim.write_bytes(b"do not delete")

    with pytest.raises(ValueError):
        archiver.prune_staging("../victim", set())

    assert victim.exists()


def test_pending_upload_bytes_counts_only_what_sync_would_transfer(
    archiver, tmp_path, monkeypatch
):
    monkeypatch.setattr(archiver, "STAGING_DIR", tmp_path / "staging")
    staged_root = tmp_path / "staging" / "test"
    staged_root.mkdir(parents=True)
    transferred = staged_root / "new.png.webp"
    transferred.write_bytes(b"x" * 1500)
    skipped = staged_root / "already-there.png.webp"
    skipped.write_bytes(b"y" * 9999)

    class Result:
        returncode = 0
        stdout = f"(dryrun) upload: {transferred} to s3://bucket/test/new.png.webp\n"
        stderr = ""

    monkeypatch.setattr(archiver.subprocess, "run", lambda *a, **k: Result())

    # Only the file sync listed counts; the one it would skip is excluded.
    assert archiver.pending_upload_bytes("test", None) == 1500


def test_pending_upload_bytes_fails_closed_on_a_failed_dry_run(archiver, monkeypatch):
    class Result:
        returncode = 1
        stdout = ""
        stderr = "connection reset"

    monkeypatch.setattr(archiver.subprocess, "run", lambda *a, **k: Result())

    with pytest.raises(RuntimeError, match="dryrun failed"):
        archiver.pending_upload_bytes("test", None)


def _clear_aws_env(monkeypatch, tmp_path):
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "absent"))


def test_missing_credentials_are_detected(archiver, monkeypatch, tmp_path):
    _clear_aws_env(monkeypatch, tmp_path)
    assert archiver.has_credentials(None) is False
    assert archiver.has_credentials("some-profile") is True


def test_aws_profile_env_counts_as_credentials(archiver, monkeypatch, tmp_path):
    # Configuring the CLI through AWS_PROFILE is ordinary; refusing to run then
    # would be a false negative.
    _clear_aws_env(monkeypatch, tmp_path)
    monkeypatch.setenv("AWS_PROFILE", "r2")
    assert archiver.has_credentials(None) is True


def test_shared_credentials_file_counts_as_credentials(archiver, monkeypatch, tmp_path):
    _clear_aws_env(monkeypatch, tmp_path)
    shared = tmp_path / "credentials"
    shared.write_text("[r2]\n", encoding="utf-8")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(shared))
    assert archiver.has_credentials(None) is True


def test_empty_shared_credentials_env_still_falls_back_to_the_default_path(
    archiver, monkeypatch, tmp_path
):
    # An empty override is unset, not a path: Path("") is Path("."), which would
    # hide credentials sitting at the default location.
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "")
    monkeypatch.setenv("HOME", str(tmp_path))

    assert archiver.has_credentials(None) is False

    (tmp_path / ".aws").mkdir()
    (tmp_path / ".aws" / "credentials").write_text("[r2]\n", encoding="utf-8")
    assert archiver.has_credentials(None) is True


def test_tilde_in_the_shared_credentials_override_is_expanded(archiver, monkeypatch, tmp_path):
    # The CLI expands "~/..."; reading it literally would refuse a configured machine.
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "~/custom/credentials")

    assert archiver.has_credentials(None) is False

    creds = tmp_path / "custom" / "credentials"
    creds.parent.mkdir()
    creds.write_text("[r2]\n", encoding="utf-8")
    assert archiver.has_credentials(None) is True


def test_sync_reports_a_missing_aws_binary_as_a_clean_error(archiver, monkeypatch, tmp_path):
    monkeypatch.setattr(archiver, "STAGING_DIR", tmp_path / "staging")
    (tmp_path / "staging" / "test").mkdir(parents=True)

    def explode(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "aws")

    monkeypatch.setattr(archiver.subprocess, "run", explode)

    with pytest.raises(RuntimeError, match="could not run the AWS CLI"):
        archiver.sync("test", None, False)


def test_bucket_usage_fails_closed_when_summary_is_missing(archiver, monkeypatch):
    class Result:
        returncode = 0
        stdout = "2026-08-13 03:00:00  1234 some/key\n"
        stderr = ""

    monkeypatch.setattr(archiver.subprocess, "run", lambda *a, **k: Result())

    with pytest.raises(RuntimeError, match="refusing to guess"):
        archiver.bucket_usage(None)


def test_human_uses_decimal_units(archiver):
    assert archiver.human(10_000_000_000) == "10.0 GB"
    assert archiver.human(1_500_000) == "1.5 MB"


def _video(archiver, name: str, size: int):
    path = Path(f"/staged/{name}")
    return archiver.StagedFile(source=path, staged=path, source_bytes=size, staged_bytes=size)


def test_large_media_guard_passes_a_push_under_the_threshold(archiver, capsys):
    staged = [_video(archiver, "clip.mp4", 500_000_000)]

    archiver.check_large_media(staged, allow_large_media=False)

    assert "large-media guard" not in capsys.readouterr().out


def test_large_media_guard_refuses_a_video_heavy_push(archiver):
    staged = [_video(archiver, f"master{i}.mov", 400_000_000) for i in range(3)]

    with pytest.raises(RuntimeError, match="large-media threshold"):
        archiver.check_large_media(staged, allow_large_media=False)


def test_large_media_guard_yields_to_the_explicit_override(archiver, capsys):
    staged = [_video(archiver, f"master{i}.mov", 400_000_000) for i in range(3)]

    archiver.check_large_media(staged, allow_large_media=True)

    assert "overridden by --allow-large-media" in capsys.readouterr().out


def test_large_media_guard_ignores_stills_however_many(archiver):
    # Stills are transcoded to WebP and are not what gets to 8 GB in one command.
    stills = [_video(archiver, f"art{i}.webp", 100_000_000) for i in range(50)]

    archiver.check_large_media(stills, allow_large_media=False)


def test_large_media_guard_names_the_biggest_offenders(archiver, capsys):
    staged = [_video(archiver, "small.mp4", 10_000_000),
              _video(archiver, "huge.mov", 1_200_000_000)]

    with pytest.raises(RuntimeError):
        archiver.check_large_media(staged, allow_large_media=False)

    out = capsys.readouterr().out
    assert out.index("huge.mov") < out.index("small.mp4")
