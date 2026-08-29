"""Shared test fixtures."""
from __future__ import annotations

import pytest

from app import recipe_store


@pytest.fixture(autouse=True)
def _isolate_recipe_library(tmp_path, monkeypatch):
    """Recipe-library writes (provenance capture, sync, runs) go to a scratch tree so
    no test can leave records under recipes/library/ or nsfw/recipes/library/."""
    monkeypatch.setenv(recipe_store.LIBRARY_ROOT_ENV, str(tmp_path / "recipe-store"))
