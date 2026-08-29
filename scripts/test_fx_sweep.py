#!/usr/bin/env python3
"""Unit tests for fx_sweep.py (parameter-sweep contact sheets)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import fx_sweep as sweep


def _img(seed: int = 1, size: tuple[int, int] = (64, 48)) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size[0], size[1], 3), dtype=np.uint8))


def _save_png(path: Path, seed: int = 1) -> Path:
    _img(seed).save(path)
    return path


# --- field classification & value parsing --------------------------------------------------------

def test_valid_fields_covers_expected_kinds() -> None:
    fields = sweep._valid_fields("retouch")
    assert fields["clarity"] == "float"
    assert fields["per_channel"] == "bool"
    assert fields["curve"] == "str"  # Optional[str] field, still parsed as a raw string


def test_valid_fields_excludes_unsupported_tuple_fields() -> None:
    fields = sweep._valid_fields("post_fx")
    assert "grain" in fields and fields["grain"] == "float"
    assert "shadow_tint" not in fields  # tuple[float, float, float] isn't a sweepable kind


def test_valid_fields_excludes_label_only_name_field() -> None:
    # Grade.name is a str field (would otherwise pass _field_kind) but has zero effect on the render;
    # sweeping it would produce visually-identical variants under different filenames.
    fields = sweep._valid_fields("post_fx")
    assert "name" not in fields


def test_parse_value_float() -> None:
    assert sweep._parse_value("float", "0.3") == pytest.approx(0.3)


@pytest.mark.parametrize("token", ["nan", "inf", "-inf"])
def test_parse_value_float_rejects_non_finite(token: str) -> None:
    with pytest.raises(ValueError, match="finite float"):
        sweep._parse_value("float", token)


@pytest.mark.parametrize("token,expected", [("true", True), ("1", True), ("false", False), ("0", False)])
def test_parse_value_bool_accepts_expected_tokens(token: str, expected: bool) -> None:
    assert sweep._parse_value("bool", token) is expected


def test_parse_value_bool_invalid_raises() -> None:
    with pytest.raises(ValueError):
        sweep._parse_value("bool", "yes")


def test_parse_value_str_passes_through() -> None:
    assert sweep._parse_value("str", "0:0,255:255") == "0:0,255:255"


def test_parse_param_unknown_field_raises_listing_valid_fields() -> None:
    fields = sweep._valid_fields("retouch")
    with pytest.raises(ValueError, match="clarity"):
        sweep._parse_param("retouch", "not_a_field", "1", fields, allow_preset=True)


def test_parse_param_preset_axis_accepts_known_preset() -> None:
    fields = sweep._valid_fields("retouch")
    assert sweep._parse_param("retouch", "preset", "crisp", fields, allow_preset=True) == "crisp"


def test_parse_param_preset_axis_rejects_unknown_preset() -> None:
    fields = sweep._valid_fields("retouch")
    with pytest.raises(ValueError, match="unknown preset"):
        sweep._parse_param("retouch", "preset", "bogus", fields, allow_preset=True)


def test_parse_param_rejects_preset_when_not_allowed() -> None:
    # --set doesn't special-case the "preset" literal — it's just an unknown field there.
    fields = sweep._valid_fields("retouch")
    with pytest.raises(ValueError, match="unknown param"):
        sweep._parse_param("retouch", "preset", "fix", fields, allow_preset=False)


# --- CLI token parsing -----------------------------------------------------------------------------

def test_parse_sweep_arg_splits_name_and_values() -> None:
    name, values = sweep._parse_sweep_arg("shadows=0.0,0.4,0.8")
    assert name == "shadows"
    assert values == ["0.0", "0.4", "0.8"]


def test_parse_sweep_arg_without_equals_raises() -> None:
    with pytest.raises(ValueError):
        sweep._parse_sweep_arg("shadows")


def test_parse_set_arg_splits_name_and_value() -> None:
    assert sweep._parse_set_arg("denoise=0.3") == ("denoise", "0.3")


def test_duplicate_sweep_param_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate --sweep"):
        sweep._collect_sweep_args(["shadows=0,0.4", "shadows=0.8"])


def test_duplicate_set_param_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate --set"):
        sweep._collect_set_args(["denoise=0.2", "denoise=0.4"])


# --- variant filename stems -------------------------------------------------------------------------

def test_variant_stem_joins_swept_params() -> None:
    stem = sweep._variant_stem({"shadows": "0.2", "clarity": "0.3"})
    assert stem == "shadows0.2_clarity0.3"


def test_variant_stem_abbreviates_when_too_long() -> None:
    combo = {"saturation": "1.2345", "highlights": "0.6789", "sharpen_radius": "2.5"}
    stem = sweep._variant_stem(combo)
    assert len(stem) <= sweep._MAX_STEM_LEN
    assert stem.startswith("satu1.2345")


def test_dedupe_stem_disambiguates_collisions() -> None:
    seen: dict[str, int] = {}
    assert sweep._dedupe_stem("x0.1", seen) == "x0.1"
    assert sweep._dedupe_stem("x0.1", seen) == "x0.1_2"


# --- grid size guard ---------------------------------------------------------------------------------

def test_grid_exceeding_max_combos_raises_with_count(tmp_path: Path) -> None:
    src = _save_png(tmp_path / "in.png")
    with pytest.raises(ValueError, match="25"):
        sweep.run_sweep(src, tool="retouch",
                        sweeps={"shadows": [str(v) for v in range(5)],
                                "clarity": [str(v) for v in range(5)]},
                        max_combos=10)


def test_no_sweep_axes_raises() -> None:
    with pytest.raises(ValueError):
        sweep.run_sweep("unused.png", tool="retouch", sweeps=None)


def test_unknown_tool_raises() -> None:
    with pytest.raises(ValueError):
        sweep.run_sweep("unused.png", tool="bogus", sweeps={"clarity": ["0.1"]})


# --- end-to-end rendering -----------------------------------------------------------------------------

def test_end_to_end_two_axis_grid_builds_sheet(tmp_path: Path) -> None:
    src = _save_png(tmp_path / "src.png")
    out_dir = tmp_path / "out"
    sheet = sweep.run_sweep(src, tool="retouch", preset="none",
                            sweeps={"shadows": ["0.0", "0.4"], "clarity": ["0.0", "0.3"]},
                            out_dir=out_dir, thumb=64)
    assert sheet == out_dir / "sheet.png"
    assert sheet.is_file()

    original = out_dir / "original.png"
    assert original.is_file()
    with Image.open(original) as im:
        assert im.convert("RGB").size == (48, 64)

    expected_variants = {
        "shadows0.0_clarity0.0.png", "shadows0.0_clarity0.3.png",
        "shadows0.4_clarity0.0.png", "shadows0.4_clarity0.3.png",
    }
    variant_names = {p.name for p in out_dir.glob("*.png")} - {"original.png", "sheet.png"}
    assert variant_names == expected_variants

    with Image.open(sheet) as im:
        sheet_rgb = im.convert("RGB")
        assert sheet_rgb.size[0] > 0 and sheet_rgb.size[1] > 0

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "loramake.fx_sweep.v1"
    assert manifest["input_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert manifest["sweeps"] == {"shadows": ["0.0", "0.4"], "clarity": ["0.0", "0.3"]}
    assert manifest["sheet_order"] == [
        "shadows0.0_clarity0.0.png",
        "shadows0.0_clarity0.3.png",
        "shadows0.4_clarity0.0.png",
        "shadows0.4_clarity0.3.png",
        "original.png",
    ]
    for variant in manifest["variants"]:
        artifact = out_dir / variant["file"]
        assert variant["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
        assert variant["effective_config"]["shadows"] in (0.0, 0.4)
        assert variant["effective_config"]["clarity"] in (0.0, 0.3)
    assert manifest["original_sha256"] == hashlib.sha256(
        (out_dir / manifest["original"]).read_bytes()).hexdigest()
    assert manifest["sheet_sha256"] == hashlib.sha256(sheet.read_bytes()).hexdigest()


def test_set_fixes_param_without_appearing_in_filename(tmp_path: Path) -> None:
    src = _save_png(tmp_path / "src.png")
    out_dir = tmp_path / "out"
    sweep.run_sweep(src, tool="retouch", preset="none", sweeps={"sharpen": ["0.0", "0.5"]},
                    fixed={"denoise": "0.2"}, out_dir=out_dir, thumb=64)
    variant_names = {p.name for p in out_dir.glob("*.png")} - {"original.png", "sheet.png"}
    assert variant_names == {"sharpen0.0.png", "sharpen0.5.png"}  # denoise not in the filenames


def test_post_fx_preset_axis_sweep(tmp_path: Path) -> None:
    src = _save_png(tmp_path / "src.png", seed=2)
    out_dir = tmp_path / "out_fx"
    sheet = sweep.run_sweep(src, tool="post_fx", sweeps={"preset": ["none", "film", "muted"]},
                            out_dir=out_dir, thumb=64)
    assert sheet.is_file()
    variant_names = {p.name for p in out_dir.glob("*.png")} - {"original.png", "sheet.png"}
    assert variant_names == {"presetnone.png", "presetfilm.png", "presetmuted.png"}


def test_run_sweep_return_value_is_path_and_exists(tmp_path: Path) -> None:
    src = _save_png(tmp_path / "src.png")
    sheet = sweep.run_sweep(src, tool="retouch", sweeps={"clarity": ["0.0", "0.2"]},
                            out_dir=tmp_path / "out2", thumb=64)
    assert isinstance(sheet, Path)
    assert sheet.exists()


def test_variants_actually_differ_by_pixel_value(tmp_path: Path) -> None:
    # A broken retouch/post_fx implementation that always returned the input unchanged would pass
    # every other end-to-end test here (they only check file existence, names, and dimensions).
    src = _save_png(tmp_path / "src.png")
    out_dir = tmp_path / "out"
    sweep.run_sweep(src, tool="retouch", preset="none", sweeps={"shadows": ["0.0", "0.8"]},
                    out_dir=out_dir, thumb=64)
    with Image.open(out_dir / "shadows0.0.png") as low, Image.open(out_dir / "shadows0.8.png") as high:
        low_arr = np.asarray(low.convert("RGB"), dtype=np.float32)
        high_arr = np.asarray(high.convert("RGB"), dtype=np.float32)
    assert not np.array_equal(low_arr, high_arr)
    # more shadow lift should raise, not lower, the overall mean brightness
    assert high_arr.mean() > low_arr.mean()


# --- regression: --sweep curve=... comma collision (P1) -------------------------------------------------

def test_sweep_curve_axis_raises_clear_error(tmp_path: Path) -> None:
    src = _save_png(tmp_path / "in.png")
    with pytest.raises(ValueError, match="cannot be swept via --sweep"):
        sweep.run_sweep(src, tool="retouch", sweeps={"curve": ["0:0,128:160,255:255"]})


def test_set_curve_still_works_as_a_single_fixed_value(tmp_path: Path) -> None:
    # curve stays usable via --set (a single raw string, never comma-split) even though it's blocked
    # from --sweep.
    src = _save_png(tmp_path / "in.png")
    out_dir = tmp_path / "out"
    sheet = sweep.run_sweep(src, tool="retouch", preset="none", sweeps={"sharpen": ["0.0", "0.3"]},
                            fixed={"curve": "0:0,128:160,255:255"}, out_dir=out_dir, thumb=64)
    assert sheet.is_file()


# --- regression: --set and --sweep on the same param (P2) -----------------------------------------------

def test_set_and_sweep_same_param_raises(tmp_path: Path) -> None:
    src = _save_png(tmp_path / "in.png")
    with pytest.raises(ValueError, match="shadows"):
        sweep.run_sweep(src, tool="retouch", sweeps={"shadows": ["0.0", "0.4"]},
                        fixed={"shadows": "0.9"})


def test_out_dir_containing_input_is_rejected(tmp_path: Path) -> None:
    src = _save_png(tmp_path / "original.png")
    with pytest.raises(ValueError, match="contains the input"):
        sweep.run_sweep(src, tool="retouch", sweeps={"shadows": ["0.0", "0.5"]}, out_dir=tmp_path)
    assert np.array_equal(np.asarray(Image.open(src)), np.asarray(_img()))  # input untouched


def test_existing_output_dir_is_rejected_without_overwriting(tmp_path: Path) -> None:
    src = _save_png(tmp_path / "input.png")
    out_dir = tmp_path / "existing"
    out_dir.mkdir()
    marker = out_dir / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        sweep.run_sweep(src, tool="retouch", sweeps={"shadows": ["0.0", "0.5"]},
                        out_dir=out_dir)
    assert marker.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(("control", "value", "message"), [
    ("thumb", 0, "--thumb"),
    ("thumb", -1, "--thumb"),
    ("cols", 0, "--cols"),
    ("cols", -1, "--cols"),
    ("max_combos", 0, "--max-combos"),
    ("max_combos", -1, "--max-combos"),
])
def test_invalid_layout_controls_fail_before_output(tmp_path: Path, control: str, value: int,
                                                    message: str) -> None:
    src = _save_png(tmp_path / "input.png")
    out_dir = tmp_path / "out"
    kwargs: dict[str, Any] = {control: value}
    with pytest.raises(ValueError, match=message):
        sweep.run_sweep(src, tool="retouch", sweeps={"shadows": ["0.0", "0.5"]},
                        out_dir=out_dir, **kwargs)
    assert not out_dir.exists()


def test_invalid_later_combo_fails_before_output(tmp_path: Path) -> None:
    src = _save_png(tmp_path / "input.png")
    out_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="shadows"):
        sweep.run_sweep(src, tool="retouch", sweeps={"shadows": ["0.2", "1.5"]},
                        out_dir=out_dir)
    assert not out_dir.exists()


def test_invalid_fixed_curve_fails_before_output(tmp_path: Path) -> None:
    src = _save_png(tmp_path / "input.png")
    out_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="curve"):
        sweep.run_sweep(src, tool="retouch", sweeps={"sharpen": ["0.0", "0.5"]},
                        fixed={"curve": "bad-curve"}, out_dir=out_dir)
    assert not out_dir.exists()


def test_invalid_post_fx_later_combo_fails_before_output(tmp_path: Path) -> None:
    src = _save_png(tmp_path / "input.png")
    out_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="vignette"):
        sweep.run_sweep(src, tool="post_fx", sweeps={"vignette": ["0.2", "1.1"]},
                        out_dir=out_dir)
    assert not out_dir.exists()


def test_negative_seed_fails_before_output(tmp_path: Path) -> None:
    src = _save_png(tmp_path / "input.png")
    out_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="--seed"):
        sweep.run_sweep(src, tool="post_fx", sweeps={"grain": ["0.0", "0.02"]},
                        out_dir=out_dir, seed=-1)
    assert not out_dir.exists()


def test_dangling_output_symlink_is_rejected_before_staging(tmp_path: Path) -> None:
    src = _save_png(tmp_path / "input.png")
    out_dir = tmp_path / "dangling"
    target = tmp_path / "missing"
    out_dir.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="already exists"):
        sweep.run_sweep(src, tool="retouch", sweeps={"shadows": ["0.0", "0.5"]},
                        out_dir=out_dir)
    assert out_dir.is_symlink()
    assert not target.exists()
    assert list(tmp_path.glob(".dangling.tmp-*")) == []


def test_render_failure_removes_staging_and_publishes_nothing(tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    src = _save_png(tmp_path / "input.png")
    out_dir = tmp_path / "out"
    calls = 0
    original_render = sweep._render

    def fail_second_render(tool: str, img: Image.Image, obj: object, seed: int) -> Image.Image:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic render failure")
        return original_render(tool, img, obj, seed)

    monkeypatch.setattr(sweep, "_render", fail_second_render)
    with pytest.raises(RuntimeError, match="synthetic render failure"):
        sweep.run_sweep(src, tool="retouch", sweeps={"shadows": ["0.0", "0.5"]},
                        out_dir=out_dir, thumb=64)
    assert not out_dir.exists()
    assert list(tmp_path.glob(".out.tmp-*")) == []


def test_keyboard_interrupt_removes_staging_and_publishes_nothing(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = _save_png(tmp_path / "input.png")
    out_dir = tmp_path / "out"

    def interrupt_render(tool: str, img: Image.Image, obj: object, seed: int) -> Image.Image:
        raise KeyboardInterrupt

    monkeypatch.setattr(sweep, "_render", interrupt_render)
    with pytest.raises(KeyboardInterrupt):
        sweep.run_sweep(src, tool="retouch", sweeps={"shadows": ["0.0", "0.5"]},
                        out_dir=out_dir, thumb=64)
    assert not out_dir.exists()
    assert list(tmp_path.glob(".out.tmp-*")) == []


def test_publish_race_preserves_new_empty_destination(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    src = _save_png(tmp_path / "input.png")
    out_dir = tmp_path / "out"
    original_build_sheet = sweep.contact_sheet.build_sheet

    def race_after_sheet(paths: list[str], out: str, cols: int | None = None, thumb: int = 512,
                         label: bool = True) -> str:
        sheet = original_build_sheet(paths, out, cols=cols, thumb=thumb, label=label)
        out_dir.mkdir()  # another process claims the canonical directory just before publish
        return sheet

    monkeypatch.setattr(sweep.contact_sheet, "build_sheet", race_after_sheet)
    with pytest.raises(ValueError, match="appeared during rendering"):
        sweep.run_sweep(src, tool="retouch", sweeps={"shadows": ["0.0", "0.5"]},
                        out_dir=out_dir, thumb=64)
    assert out_dir.is_dir()
    assert list(out_dir.iterdir()) == []
    assert list(tmp_path.glob(".out.tmp-*")) == []
