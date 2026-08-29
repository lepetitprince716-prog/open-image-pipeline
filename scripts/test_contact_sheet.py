#!/usr/bin/env python3
"""Unit tests for contact_sheet.py (PIL montage builder)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import contact_sheet as cs


def _mk(tmp: Path, n: int) -> list[str]:
    paths = []
    for i in range(n):
        p = tmp / f"img{i}.png"
        Image.fromarray(np.full((40, 30, 3), min(i * 30, 255), dtype=np.uint8)).save(p)
        paths.append(str(p))
    return paths


def test_build_sheet_creates_grid(tmp_path: Path) -> None:
    out = tmp_path / "sheet.png"
    res = cs.build_sheet(_mk(tmp_path, 5), str(out), thumb=64)
    assert Path(res).is_file()
    cols = math.ceil(math.sqrt(5))  # 3
    im = Image.open(res)
    assert im.width >= cols * 64 and im.height > 0


def test_explicit_cols(tmp_path: Path) -> None:
    out = tmp_path / "s.png"
    cs.build_sheet(_mk(tmp_path, 4), str(out), cols=4, thumb=48)
    im = Image.open(out)
    assert im.width >= 4 * 48  # single row of 4


def test_missing_images_skipped(tmp_path: Path) -> None:
    paths = _mk(tmp_path, 2) + [str(tmp_path / "nope.png")]
    out = tmp_path / "s.png"
    cs.build_sheet(paths, str(out), thumb=64)
    assert out.is_file()


def test_no_images_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        cs.build_sheet([str(tmp_path / "none.png")], str(tmp_path / "o.png"))
