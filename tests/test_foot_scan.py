"""Offline tests for the foot fissure scanner + two-evidence gate (foot_scan.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import foot_scan as fs  # noqa: E402


def _skin_image(size=(832, 1344), rng_seed=3):
    """Skin-toned gradient with high-frequency texture (no toe fissures)."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([
        150 + 40 * xx / w,
        100 + 25 * yy / h,
        90 + 20 * (xx + yy) / (w + h),
    ], axis=-1).astype(np.float32)
    rng = np.random.default_rng(rng_seed)
    base += rng.normal(0, 5.0, size=base.shape)
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def _fissure_image(fissures=(320, 352, 378), size=(832, 1344), rng_seed=5):
    """Skin image with vertical dark fissure lines (simulated toe seams)."""
    im = _skin_image(size=size, rng_seed=rng_seed)
    a = np.asarray(im).astype(np.float32)
    w, h = size
    band_y0, band_y1 = int(h * 0.7), h  # toe band
    for x in fissures:
        for dx in (-1, 0, 1):
            if 0 <= x + dx < w:
                a[band_y0:band_y1, x + dx] -= 45.0  # dark vertical line
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def test_clean_image_no_peaks(tmp_path):
    """纹理皮肤（无趾缝）: 检测不动声色或者低报，而非全图乱报。"""
    p = tmp_path / "clean.png"
    _skin_image().save(p)
    gray, _ = fs._load_gray(str(p))
    peaks = fs.scan_fissures(gray, threshold_sigma=1.4)
    # 纯纹理不应出现高 z 峰（1.4σ 下纹理会出少量边缘峰，但不能集中在某几列）
    assert len(peaks) <= 6, f"clean image too noisy: {[q['x'] for q in peaks]}"


def test_fissure_peaks_detected(tmp_path):
    """已知缝线位置必须被检出（目标峰集合 p）。"""
    p = tmp_path / "fissure.png"
    _fissure_image().save(p)
    gray, size = fs._load_gray(str(p))
    assert size == (832, 1344)
    peaks = fs.scan_fissures(gray, threshold_sigma=1.2, band=(940, 1344))
    xs = {q["x"] for q in peaks}
    # 三条缝 320/352/378 至少两条要出现（1.2σ + 局部极大，允许邻近融合）
    hit = [x for x in (320, 352, 378) if any(abs(x - q) <= 4 for q in xs)]
    assert len(hit) >= 2, f"fissures missed: wanted ~320/352/378, got {sorted(xs)}"
    assert all(q["z"] > 0 for q in peaks)


def test_peak_crops_written(tmp_path):
    """每峰一条竖条裁剪（复核介质）。"""
    p = tmp_path / "src.png"
    _fissure_image().save(p)
    gray, _ = fs._load_gray(str(p))
    peaks = fs.scan_fissures(gray, threshold_sigma=1.2, band=(940, 1344))
    assert peaks
    out = tmp_path / "crops"
    saved = fs.write_peak_crops(str(p), peaks, out, "stem", width_px=90)
    assert len(saved) == len(peaks)
    with Image.open(saved[0]) as im:
        assert im.width == 91  # x ±45 -> 90px + center
        assert im.height == 1344  # full height review strip


def test_gate_three_states():
    """双证据合流：defect→CONFIRMED, clean→DISMISSED, pending→SUSPECT；非点名峰不混入。"""
    scan_json = {"peaks": [
        {"x": 320, "strength": 4.0, "z": 3.0},
        {"x": 352, "strength": 4.0, "z": 3.0},
        {"x": 378, "strength": 4.0, "z": 3.0},
    ]}
    evidence = {"peaks": [
        {"x": 320, "verdict": "defect", "note": "vision: 缝"},
        {"x": 352, "verdict": "clean", "note": "vision: 高光带"},
        {"x": 378, "verdict": "pending"},
    ]}
    merged = fs.gate(scan_json, evidence)
    by_x = {q["x"]: q["gate"] for q in merged}
    assert by_x[320] == "CONFIRMED"
    assert by_x[352] == "DISMISSED"
    assert by_x[378] == "SUSPECT"
    assert {q["x"] for q in merged} == {320, 352, 378}


def test_gate_second_evidence_only_peak(tmp_path):
    """第二证据发现、检测未报警的峰也要合流进来（vision 独立发现路径）。"""
    scan_json = {"peaks": [{"x": 320, "strength": 1.0, "z": 0.5}]}
    evidence = {"peaks": [{"x": 500, "verdict": "defect", "note": "vision 独立发现"}]}
    merged = fs.gate(scan_json, evidence)
    by_x = {q["x"]: q["gate"] for q in merged}
    assert by_x[500] == "CONFIRMED"
    assert by_x.get(320) == "SUSPECT"


def test_evidence_format_validation(tmp_path):
    """坏 evidence JSON 必须报错，不静默通过。"""
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"not_peaks": []}), encoding="utf-8")
    import pytest
    with pytest.raises(ValueError):
        fs._load_evidence(str(bad))