#!/usr/bin/env python3
"""Unit tests for post_fx.py (image-space finishing pass)."""
from __future__ import annotations

import argparse
import stat
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import post_fx as fx


def _img(seed: int = 1) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (64, 48, 3), dtype=np.uint8))


def test_none_preset_is_identity() -> None:
    src = _img()
    out = fx.grade_image(src, fx.PRESETS["none"])
    assert np.array_equal(np.asarray(src), np.asarray(out))


def test_shape_and_mode_preserved() -> None:
    out = fx.grade_image(_img(), fx.PRESETS["film"])
    assert out.size == (48, 64) and out.mode == "RGB"


def test_grain_is_deterministic_per_seed() -> None:
    src = _img()
    a = np.asarray(fx.grade_image(src, fx.PRESETS["film"], seed=7))
    b = np.asarray(fx.grade_image(src, fx.PRESETS["film"], seed=7))
    c = np.asarray(fx.grade_image(src, fx.PRESETS["film"], seed=8))
    assert np.array_equal(a, b)          # same seed -> identical
    assert not np.array_equal(a, c)      # different seed -> grain differs


def test_muted_lowers_saturation() -> None:
    src = _img()
    base = np.asarray(src, dtype=np.float32)
    muted = np.asarray(fx.grade_image(src, fx.PRESETS["muted"]), dtype=np.float32)
    # per-pixel channel spread is a proxy for saturation; muted should reduce it on average
    assert muted.std(axis=2).mean() < base.std(axis=2).mean()


def test_output_in_range() -> None:
    out = np.asarray(fx.grade_image(_img(), fx.PRESETS["warm"]))
    assert out.min() >= 0 and out.max() <= 255


def test_cli_override_changes_grade() -> None:
    args = argparse.Namespace(preset="film", temp=None, tint=None, contrast=None, saturation=None,
                              halation=None, vignette=None, grain=0.0)
    g = fx._grade_from_args(args)
    assert g.grain == 0.0 and g.name == "film"  # override applied, rest inherited


@pytest.mark.parametrize("field", fx._SCALAR_RANGES)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_all_scalar_controls_reject_non_finite(field: str, bad: float) -> None:
    with pytest.raises(ValueError, match=field):
        fx.grade_image(_img(), replace(fx.PRESETS["none"], **{field: bad}))


@pytest.mark.parametrize(("field", "bad"), [
    ("temp", -1.01), ("temp", 1.01),
    ("tint", -1.01), ("tint", 1.01),
    ("contrast", -0.01), ("contrast", 2.01),
    ("saturation", -0.01), ("saturation", 2.01),
    ("halation", -0.01), ("halation", 1.01),
    ("vignette", -0.01), ("vignette", 1.01),
    ("grain", -0.001), ("grain", 0.151),
])
def test_scalar_controls_reject_values_outside_contract(field: str, bad: float) -> None:
    with pytest.raises(ValueError, match=field):
        fx.grade_image(_img(), replace(fx.PRESETS["none"], **{field: bad}))


@pytest.mark.parametrize(("field", "bad"), [
    ("shadow_tint", (0.0, 0.0)),
    ("highlight_tint", (0.0, float("nan"), 0.0)),
    ("shadow_tint", (0.0, 1.01, 0.0)),
])
def test_tint_triplets_reject_malformed_values(field: str, bad: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match=field):
        fx.grade_image(_img(), replace(fx.PRESETS["none"], **{field: bad}))


def test_negative_seed_is_rejected_even_without_grain() -> None:
    with pytest.raises(ValueError, match="seed"):
        fx.grade_image(_img(), fx.PRESETS["none"], seed=-1)


def _cli_namespace(preset: str = "film", **overrides: object) -> argparse.Namespace:
    fields: dict[str, object] = {name: None for name in fx._SCALAR_RANGES}
    fields.update(overrides)
    return argparse.Namespace(preset=preset, **fields)


def test_grade_from_args_retains_unset_preset_fields() -> None:
    grade = fx._grade_from_args(_cli_namespace("film", grain=0.0))
    assert grade.grain == 0.0
    assert grade.halation == fx.PRESETS["film"].halation


def test_cli_rejects_input_as_output_without_modifying_source(tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    _img().save(src)
    before = src.read_bytes()
    result = subprocess.run([sys.executable, str(Path(fx.__file__)), str(src), "--out", str(src),
                             "--overwrite"], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "input image" in result.stderr
    assert src.read_bytes() == before


def test_cli_rejects_hardlink_alias_of_input_without_modifying_source(tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    alias = tmp_path / "alias.png"
    _img().save(src)
    alias.hardlink_to(src)
    before = src.read_bytes()
    result = subprocess.run([sys.executable, str(Path(fx.__file__)), str(src), "--out", str(alias),
                             "--overwrite"], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "input image" in result.stderr
    assert src.read_bytes() == before


def test_cli_existing_output_requires_explicit_overwrite(tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    out = tmp_path / "out.png"
    _img().save(src)
    out.write_bytes(b"keep-me")
    script = str(Path(fx.__file__))
    blocked = subprocess.run([sys.executable, script, str(src), "--out", str(out)],
                             capture_output=True, text=True, timeout=30)
    assert blocked.returncode != 0
    assert "already exists" in blocked.stderr
    assert out.read_bytes() == b"keep-me"

    replaced = subprocess.run([sys.executable, script, str(src), "--out", str(out), "--overwrite"],
                              capture_output=True, text=True, timeout=30)
    assert replaced.returncode == 0
    with Image.open(src) as src_image, Image.open(out) as out_image:
        assert out_image.size == src_image.size


def test_cli_broken_output_symlink_is_rejected(tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    out = tmp_path / "dangling.png"
    target = tmp_path / "missing.png"
    _img().save(src)
    out.symlink_to(target)
    result = subprocess.run([sys.executable, str(Path(fx.__file__)), str(src), "--out", str(out)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "broken symlink" in result.stderr
    assert out.is_symlink()
    assert not target.exists()


def test_publish_race_does_not_replace_new_output(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "in.png"
    out = tmp_path / "out.png"
    _img().save(src)
    original_link = fx.os.link

    def raced_link(temp: Path, destination: Path) -> None:
        destination.write_bytes(b"racer")
        original_link(temp, destination)

    monkeypatch.setattr(fx.os, "link", raced_link)
    with pytest.raises(ValueError, match="appeared during rendering"):
        fx._publish_image(_img(seed=2), src, out, overwrite=False)
    assert out.read_bytes() == b"racer"
    assert list(tmp_path.glob(".out.tmp-*")) == []


def test_publish_race_with_input_hardlink_keeps_source_unchanged(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "in.png"
    out = tmp_path / "out.png"
    _img().save(src)
    before = src.read_bytes()
    original_link = fx.os.link

    def raced_link(temp: Path, destination: Path) -> None:
        original_link(src, destination)
        original_link(temp, destination)

    monkeypatch.setattr(fx.os, "link", raced_link)
    with pytest.raises(ValueError, match="appeared during rendering"):
        fx._publish_image(_img(seed=2), src, out, overwrite=False)
    assert src.read_bytes() == before
    assert out.read_bytes() == before


def test_publish_new_file_uses_normal_umask_permissions(tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    out = tmp_path / "out.png"
    control = tmp_path / "control.png"
    _img().save(src)
    control.write_bytes(b"control")
    fx._publish_image(_img(seed=2), src, out, overwrite=False)
    assert stat.S_IMODE(out.stat().st_mode) == stat.S_IMODE(control.stat().st_mode)


def test_overwrite_preserves_existing_output_permissions(tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    out = tmp_path / "out.png"
    _img().save(src)
    out.write_bytes(b"old")
    out.chmod(0o640)
    fx._publish_image(_img(seed=2), src, out, overwrite=True)
    assert stat.S_IMODE(out.stat().st_mode) == 0o640
