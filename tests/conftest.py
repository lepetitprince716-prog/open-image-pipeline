"""Shared test fixtures."""
from __future__ import annotations

import pytest

from app import jobs, recipe_store
from loramake import net_routes


@pytest.fixture(autouse=True)
def _isolate_tagdb(tmp_path, monkeypatch):
    """Default TagDB dir is missing so optional hooks do not see a developer extract."""
    monkeypatch.setenv("LORAMAKE_TAGDB", str(tmp_path / "no-tagdb"))


@pytest.fixture(autouse=True)
def _isolate_recipe_library(tmp_path, monkeypatch):
    """Recipe-library writes (provenance capture, sync, runs) go to a scratch tree so
    no test can leave records under recipes/library/ or nsfw/recipes/library/."""
    monkeypatch.setenv(recipe_store.LIBRARY_ROOT_ENV, str(tmp_path / "recipe-store"))


@pytest.fixture(autouse=True)
def _isolate_net_routes(tmp_path, monkeypatch):
    """Importing app.server wires a live route refresh into the job worker; keep workers off
    the network and route-cache writes out of app/data/net_routes.json."""
    # Import first so app.server's module-level wiring has already run and cannot overwrite
    # the patch below when a test imports it for the first time.
    try:
        import app.server  # noqa: F401
    except ModuleNotFoundError as exc:
        # The public tree does not publish the gallery scanner app.server imports, so
        # nothing wired a live route refresh into the job worker; the patch below still
        # holds. Any other import failure is a real regression and still raises.
        if exc.name != "gallery_scan":
            raise

    monkeypatch.setattr(jobs, "REFRESH_HOSTS", None)
    monkeypatch.setenv(net_routes.STATE_ENV, str(tmp_path / "net_routes.json"))
