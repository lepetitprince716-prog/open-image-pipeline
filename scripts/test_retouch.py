#!/usr/bin/env python3
"""Unit tests for retouch.py (corrective retouch pass)."""
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

import retouch as rt


def _img(seed: int = 1, size: tuple[int, int] = (64, 48)) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size[0], size[1], 3), dtype=np.uint8))


def test_neutral_is_identity() -> None:
    src = _img()
    out = rt.retouch_image(src, rt.Retouch())
    assert np.array_equal(np.asarray(src), np.asarray(out))


def test_none_preset_is_identity() -> None:
    src = _img()
    out = rt.retouch_image(src, rt.PRESETS["none"])
    assert np.array_equal(np.asarray(src), np.asarray(out))


def test_shape_and_mode_preserved() -> None:
    out = rt.retouch_image(_img(), rt.PRESETS["fix"])
    assert out.size == (48, 64) and out.mode == "RGB"


def test_output_in_range() -> None:
    out = np.asarray(rt.retouch_image(_img(), rt.PRESETS["crisp"]))
    assert out.min() >= 0 and out.max() <= 255


def test_deterministic_same_input_same_params() -> None:
    src = _img()
    a = np.asarray(rt.retouch_image(src, rt.PRESETS["fix"]))
    b = np.asarray(rt.retouch_image(src, rt.PRESETS["fix"]))
    assert np.array_equal(a, b)


def test_auto_levels_expands_low_contrast_range() -> None:
    rng = np.random.default_rng(3)
    low_contrast = rng.integers(90, 161, (64, 64, 3), dtype=np.uint8)
    src = Image.fromarray(low_contrast)
    base_range = int(low_contrast.max()) - int(low_contrast.min())
    out = np.asarray(rt.retouch_image(src, rt.Retouch(auto_levels=1.0)))
    out_range = int(out.max()) - int(out.min())
    assert out_range > base_range


def test_shadows_raise_dark_half_mean() -> None:
    rng = np.random.default_rng(4)
    dark = rng.integers(10, 30, (32, 64, 3), dtype=np.uint8)
    bright = rng.integers(200, 230, (32, 64, 3), dtype=np.uint8)
    src = Image.fromarray(np.concatenate([dark, bright], axis=0))
    out = np.asarray(rt.retouch_image(src, rt.Retouch(shadows=0.8)), dtype=np.float32)
    assert out[:32].mean() > dark.astype(np.float32).mean()


def test_highlights_lower_bright_half_mean() -> None:
    rng = np.random.default_rng(4)
    dark = rng.integers(10, 30, (32, 64, 3), dtype=np.uint8)
    bright = rng.integers(200, 230, (32, 64, 3), dtype=np.uint8)
    src = Image.fromarray(np.concatenate([dark, bright], axis=0))
    out = np.asarray(rt.retouch_image(src, rt.Retouch(highlights=0.8)), dtype=np.float32)
    assert out[32:].mean() < bright.astype(np.float32).mean()


def test_auto_wb_reduces_channel_mean_imbalance() -> None:
    rng = np.random.default_rng(5)
    base = rng.integers(80, 160, (48, 48, 3), dtype=np.uint8).astype(np.float32)
    base[..., 0] = np.clip(base[..., 0] * 1.5, 0, 255)  # red-tinted
    src = Image.fromarray(base.astype(np.uint8))
    src_means = base.reshape(-1, 3).mean(axis=0)
    src_imbalance = src_means.max() - src_means.min()
    out = np.asarray(rt.retouch_image(src, rt.Retouch(auto_wb=1.0)), dtype=np.float32)
    out_means = out.reshape(-1, 3).mean(axis=0)
    out_imbalance = out_means.max() - out_means.min()
    assert out_imbalance < src_imbalance


def test_vibrance_boosts_low_saturation_more_than_high() -> None:
    # The spec's formula boosts (x - lum) by a multiplicative gain (1 + vibrance*w) where
    # w = (1-sat)**2 — i.e. the *proportional* boost is bigger for low-saturation pixels, even
    # though a high-saturation pixel's larger (x - lum) can still see a bigger absolute delta.
    # Test the gain directly on the two pixel types (near-gray vs. near-pure-red, both mid luma).
    arr = np.array([[[140, 130, 120], [220, 40, 40]]], dtype=np.float32) / 255.0
    out = rt._vibrance(arr, 0.8)
    lum = (arr @ rt._LUMA)[..., None]
    gain = (out - lum) / (arr - lum)
    low_gain = np.abs(gain[0, 0]).mean()
    high_gain = np.abs(gain[0, 1]).mean()
    assert low_gain > high_gain


def test_sharpen_increases_local_variance() -> None:
    src = _img(size=(64, 64))
    out = np.asarray(rt.retouch_image(src, rt.Retouch(sharpen=1.0)), dtype=np.float32)
    base = np.asarray(src, dtype=np.float32)
    blurred = rt._gaussian_blur(base / 255.0, 1.5) * 255.0
    base_local = (base - blurred).std()
    out_local = (out - blurred).std()
    assert out_local > base_local


@pytest.mark.parametrize("bad_spec", [
    "0:0,128:300,255:255",   # out-of-range output value
    "128:145",                # single point, needs at least 2
    "0:0,128:100,100:50,255:255",  # inputs not strictly increasing (100 < 128)
    "not-a-point",             # malformed token
])
def test_curve_validation_raises(bad_spec: str) -> None:
    with pytest.raises(ValueError):
        rt.retouch_image(_img(), rt.Retouch(curve=bad_spec))


def test_curve_valid_spec_applies_without_error() -> None:
    out = rt.retouch_image(_img(), rt.Retouch(curve="0:0,128:145,255:255"))
    assert out.size == (48, 64)


def _cli_args(preset: str, **overrides: object) -> argparse.Namespace:
    """Namespace as argparse would produce it: every override field None unless explicitly set."""
    fields: dict[str, object] = {name: None for name in (
        "denoise", "auto_wb", "wb_temp", "wb_tint", "auto_levels", "black", "white", "gamma",
        "scurve", "shadows", "highlights", "vibrance", "saturation", "clarity", "sharpen",
        "sharpen_radius", "curve", "per_channel")}
    fields.update(overrides)
    return argparse.Namespace(preset=preset, **fields)


def test_cli_override_inherits_preset_and_applies_override() -> None:
    r = rt._retouch_from_args(_cli_args("fix", sharpen=0.9))
    assert r.sharpen == 0.9                                  # override applied
    assert r.auto_wb == rt.PRESETS["fix"].auto_wb             # rest inherited from preset
    assert r.clarity == rt.PRESETS["fix"].clarity


def test_cli_override_per_channel_flag() -> None:
    r = rt._retouch_from_args(_cli_args("none", per_channel=True))
    assert r.per_channel is True


def test_cli_help_lists_no_per_channel_override() -> None:
    # Regression for the store_true/default=None asymmetry: --per-channel used to have no way back to
    # False. BooleanOptionalAction should surface both --per-channel and --no-per-channel.
    script = Path(__file__).parent / "retouch.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True,
                            timeout=10)
    assert "--no-per-channel" in result.stdout


# --- flat / near-flat image guard (auto_levels regression) ---------------------------------------------

def test_default_preset_flat_white_image_stays_white() -> None:
    src = Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8))
    out = np.asarray(rt.retouch_image(src, rt.PRESETS[rt.DEFAULT_PRESET]))
    assert out.min() >= 250  # not crushed toward black by auto_levels on a perfectly flat input


def test_default_preset_flat_black_image_stays_black() -> None:
    src = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    out = np.asarray(rt.retouch_image(src, rt.PRESETS[rt.DEFAULT_PRESET]))
    assert out.max() <= 5  # not blown toward white


def test_near_flat_image_with_outliers_does_not_crush_to_black() -> None:
    # Mirrors the confirmed bug report: a near-flat image (bulk at one value, a handful of outlier
    # pixels) used to have lo/hi percentiles both land in the bulk value, collapsing everything to 0.
    arr = np.full((64, 64, 3), 200, dtype=np.uint8)
    arr[0, :5, :] = 50
    src = Image.fromarray(arr)
    out = np.asarray(rt.retouch_image(src, rt.Retouch(auto_levels=0.3)))
    assert out[10:, 10:].mean() > 100  # bulk of the image stays near its original mid-tone, not black


def test_auto_levels_helper_flat_image_is_unchanged() -> None:
    arr = np.full((8, 8, 3), 0.5, dtype=np.float32)
    out = rt._auto_levels(arr, percent=0.3, per_channel=False)
    assert np.array_equal(out, arr)


def test_auto_levels_helper_per_channel_flat_image_is_unchanged() -> None:
    arr = np.full((8, 8, 3), 0.5, dtype=np.float32)
    out = rt._auto_levels(arr, percent=0.3, per_channel=True)
    assert np.array_equal(out, arr)


# --- image modes and shapes -----------------------------------------------------------------------------

def test_rgba_input_is_handled_via_rgb_conversion() -> None:
    rgba = np.zeros((16, 16, 4), dtype=np.uint8)
    rgba[..., :3] = 128
    rgba[..., 3] = 255
    src = Image.fromarray(rgba, mode="RGBA")
    out = rt.retouch_image(src, rt.PRESETS["fix"])
    assert out.mode == "RGB" and out.size == (16, 16)


def test_grayscale_l_mode_input_is_handled_via_rgb_conversion() -> None:
    gray = np.full((16, 16), 128, dtype=np.uint8)
    src = Image.fromarray(gray, mode="L")
    out = rt.retouch_image(src, rt.PRESETS["fix"])
    assert out.mode == "RGB" and out.size == (16, 16)


def test_1x1_image_does_not_crash() -> None:
    src = Image.fromarray(np.array([[[10, 20, 30]]], dtype=np.uint8))
    out = rt.retouch_image(src, rt.PRESETS["fix"])
    assert out.size == (1, 1)


def test_very_thin_image_does_not_crash() -> None:
    src = Image.fromarray(np.full((1, 200, 3), 100, dtype=np.uint8))
    out = rt.retouch_image(src, rt.PRESETS["crisp"])
    assert out.size == (200, 1)


# --- denoise: cv2 path vs. PIL fallback path ------------------------------------------------------------

def test_denoise_pil_fallback_changes_image(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rt, "_HAVE_CV2", False)
    src = _img(seed=7)
    out = np.asarray(rt.retouch_image(src, rt.Retouch(denoise=1.0)))
    assert not np.array_equal(out, np.asarray(src))


@pytest.mark.skipif(not rt._HAVE_CV2, reason="cv2 not installed")
def test_denoise_cv2_path_changes_image() -> None:
    src = _img(seed=7)
    out = np.asarray(rt.retouch_image(src, rt.Retouch(denoise=0.6)))
    assert not np.array_equal(out, np.asarray(src))


# --- white balance: temp / tint direction ---------------------------------------------------------------

def test_wb_temp_helper_warms_red_relative_to_blue() -> None:
    arr = np.full((4, 4, 3), 0.5, dtype=np.float32)
    out = rt._wb_temp_tint(arr, temp=0.5, tint=0.0)
    assert out[..., 0].mean() > out[..., 2].mean()


def test_wb_tint_helper_pushes_green_relative_to_red() -> None:
    arr = np.full((4, 4, 3), 0.5, dtype=np.float32)
    out = rt._wb_temp_tint(arr, temp=0.0, tint=0.5)
    assert out[..., 1].mean() > out[..., 0].mean()


# --- manual levels: black / white / gamma individually --------------------------------------------------

def test_manual_levels_black_point_crushes_below_threshold() -> None:
    arr = np.array([0.05, 0.5, 0.95], dtype=np.float32)
    out = rt._manual_levels(arr, black=0.2, white=1.0, gamma=1.0)
    assert out[0] == 0.0


def test_manual_levels_white_point_blows_above_threshold() -> None:
    arr = np.array([0.05, 0.5, 0.95], dtype=np.float32)
    out = rt._manual_levels(arr, black=0.0, white=0.8, gamma=1.0)
    assert out[2] == 1.0


def test_manual_levels_gamma_above_one_brightens_midtone() -> None:
    arr = np.array([0.5], dtype=np.float32)
    out = rt._manual_levels(arr, black=0.0, white=1.0, gamma=2.0)
    assert out[0] > 0.5


def test_gamma_above_one_brightens_full_pipeline() -> None:
    src = Image.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8))
    out = np.asarray(rt.retouch_image(src, rt.Retouch(gamma=2.0)), dtype=np.float32)
    assert out.mean() > 128.0


# --- saturation, clarity, scurve ------------------------------------------------------------------------

def test_saturation_helper_zero_collapses_to_luminance() -> None:
    arr = np.array([[[0.8, 0.2, 0.2]]], dtype=np.float32)
    out = rt._saturation(arr, 0.0)
    lum = float((arr @ rt._LUMA)[0, 0])
    assert out[0, 0] == pytest.approx([lum, lum, lum], abs=1e-5)


def test_clarity_helper_widens_a_step_edge() -> None:
    lum_like = np.zeros((32, 32, 3), dtype=np.float32)
    lum_like[:, :16] = 0.3
    lum_like[:, 16:] = 0.7
    out = rt._clarity(lum_like, 0.8)
    assert not np.array_equal(out, lum_like)
    base_edge_diff = abs(float(lum_like[16, 15, 0] - lum_like[16, 16, 0]))
    out_edge_diff = abs(float(out[16, 15, 0] - out[16, 16, 0]))
    assert out_edge_diff >= base_edge_diff


def test_scurve_pushes_shadows_darker_and_highlights_brighter() -> None:
    arr = np.array([0.2, 0.8], dtype=np.float32)
    out = rt._scurve(arr, 1.0)
    assert out[0] < arr[0]
    assert out[1] > arr[1]


# --- curve correctness and the scipy/no-scipy divergence guard ------------------------------------------

def test_curve_maps_known_control_points_to_expected_output() -> None:
    arr = np.array([0, 128, 255], dtype=np.float32) / 255.0
    out = rt._apply_curve(arr, "0:0,128:160,255:255")
    expected = np.array([0, 160, 255], dtype=np.float32) / 255.0
    assert np.allclose(out, expected, atol=1.0 / 255.0)


def test_curve_linear_interpolation_matches_control_points() -> None:
    arr = np.array([0, 128, 255], dtype=np.float32) / 255.0
    out = rt._apply_curve(arr, "0:0,128:160,255:255")
    expected = np.array([0, 160, 255], dtype=np.float32) / 255.0
    assert np.allclose(out, expected, atol=1.0 / 255.0)


def test_curve_linear_interpolation_is_environment_independent() -> None:
    # This curve used to diverge by >33/255 when scipy selected PCHIP but the fallback used np.interp.
    arr = np.arange(256, dtype=np.float32) / 255.0
    out = rt._apply_curve(arr, "0:0,64:200,192:210,255:255")
    xs = np.array([0, 64, 192, 255], dtype=np.float64) / 255.0
    ys = np.array([0, 200, 210, 255], dtype=np.float64) / 255.0
    expected = np.interp(arr.astype(np.float64), xs, ys).astype(np.float32)
    assert np.array_equal(out, expected)


def test_gamma_zero_or_negative_raises() -> None:
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="gamma"):
            rt.retouch_image(_img(), rt.Retouch(gamma=bad))


@pytest.mark.parametrize("field", rt._NUMERIC_FIELDS)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_all_numeric_controls_reject_non_finite(field: str, bad: float) -> None:
    with pytest.raises(ValueError, match=field):
        rt.retouch_image(_img(), replace(rt.Retouch(), **{field: bad}))


@pytest.mark.parametrize(("field", "bad"), [
    ("denoise", -0.01), ("denoise", 1.01),
    ("auto_wb", -0.01), ("auto_wb", 1.01),
    ("wb_temp", -1.01), ("wb_temp", 1.01),
    ("wb_tint", -1.01), ("wb_tint", 1.01),
    ("auto_levels", -0.01), ("auto_levels", 50.0),
    ("black", -0.01), ("black", 1.01),
    ("white", -0.01), ("white", 1.01),
    ("scurve", -1.01), ("scurve", 1.01),
    ("shadows", -2.0 / 3.0), ("shadows", 1.01),
    ("highlights", -0.01), ("highlights", 1.01),
    ("vibrance", -1.01), ("vibrance", 1.01),
    ("saturation", -0.01), ("saturation", 2.01),
    ("clarity", -1.01), ("clarity", 1.01),
    ("sharpen", -0.01), ("sharpen", 2.01),
    ("sharpen_radius", 0.0),
])
def test_numeric_controls_reject_values_outside_contract(field: str, bad: float) -> None:
    with pytest.raises(ValueError, match=field):
        rt.retouch_image(_img(), replace(rt.Retouch(), **{field: bad}))


@pytest.mark.parametrize(("black", "white"), [(0.5, 0.5), (0.8, 0.2)])
def test_manual_level_endpoints_must_be_ordered(black: float, white: float) -> None:
    with pytest.raises(ValueError, match="black must be less than white"):
        rt.retouch_image(_img(), rt.Retouch(black=black, white=white))


def test_cli_gamma_zero_exits_with_message_not_traceback(tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    _img().save(src)
    script = Path(rt.__file__)
    result = subprocess.run([sys.executable, str(script), str(src), "--gamma", "0",
                             "--out", str(tmp_path / "out.png")],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "gamma" in result.stderr and "Traceback" not in result.stderr


def test_cli_rejects_input_as_output_without_modifying_source(tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    _img().save(src)
    before = src.read_bytes()
    script = Path(rt.__file__)
    result = subprocess.run([sys.executable, str(script), str(src), "--out", str(src), "--overwrite"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "input image" in result.stderr
    assert src.read_bytes() == before


def test_cli_rejects_hardlink_alias_of_input_without_modifying_source(tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    alias = tmp_path / "alias.png"
    _img().save(src)
    alias.hardlink_to(src)
    before = src.read_bytes()
    script = Path(rt.__file__)
    result = subprocess.run([sys.executable, str(script), str(src), "--out", str(alias),
                             "--overwrite"], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "input image" in result.stderr
    assert src.read_bytes() == before
    assert alias.read_bytes() == before


def test_cli_broken_output_symlink_is_rejected(tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    out = tmp_path / "dangling.png"
    target = tmp_path / "missing.png"
    _img().save(src)
    out.symlink_to(target)
    script = Path(rt.__file__)
    result = subprocess.run([sys.executable, str(script), str(src), "--out", str(out)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "broken symlink" in result.stderr
    assert out.is_symlink()
    assert not target.exists()


def test_cli_existing_output_requires_explicit_overwrite(tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    out = tmp_path / "out.png"
    _img().save(src)
    out.write_bytes(b"keep-me")
    script = Path(rt.__file__)
    result = subprocess.run([sys.executable, str(script), str(src), "--out", str(out)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert out.read_bytes() == b"keep-me"

    replaced = subprocess.run([sys.executable, str(script), str(src), "--out", str(out), "--overwrite"],
                              capture_output=True, text=True, timeout=30)
    assert replaced.returncode == 0
    with Image.open(src) as src_image, Image.open(out) as out_image:
        assert out_image.size == src_image.size


def test_publish_race_does_not_replace_new_output(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "in.png"
    out = tmp_path / "out.png"
    _img().save(src)
    original_link = rt.os.link

    def raced_link(temp: Path, destination: Path) -> None:
        destination.write_bytes(b"racer")
        original_link(temp, destination)

    monkeypatch.setattr(rt.os, "link", raced_link)
    with pytest.raises(ValueError, match="appeared during rendering"):
        rt._publish_image(_img(seed=2), src, out, overwrite=False)
    assert out.read_bytes() == b"racer"
    assert list(tmp_path.glob(".out.tmp-*")) == []


def test_publish_new_file_uses_normal_umask_permissions(tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    out = tmp_path / "out.png"
    control = tmp_path / "control.png"
    _img().save(src)
    control.write_bytes(b"control")
    rt._publish_image(_img(seed=2), src, out, overwrite=False)
    assert stat.S_IMODE(out.stat().st_mode) == stat.S_IMODE(control.stat().st_mode)


def test_overwrite_preserves_existing_output_permissions(tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    out = tmp_path / "out.png"
    _img().save(src)
    out.write_bytes(b"old")
    out.chmod(0o640)
    rt._publish_image(_img(seed=2), src, out, overwrite=True)
    assert stat.S_IMODE(out.stat().st_mode) == 0o640
