"""Import-time configuration constants shared by loramake scripts."""
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse

from loramake.comfy_endpoints import DIRECT_COMFY_HOSTS, resolve_comfy_host


def _repo_root() -> Path:
    """Return the repository root containing ``loramake/__init__.py``.

    Walk upward from this file location with a small fixed limit.  This does not
    depend on ``.git`` because linked worktrees store ``.git`` as a file.
    """
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents[:5]):
        if (candidate / "loramake" / "__init__.py").is_file():
            return candidate
    raise RuntimeError("could not locate repo root containing loramake/__init__.py")


BOXA_HOST = resolve_comfy_host("boxa", explicit=os.environ.get("BOXA_HOST"))
BOXB_HOST = resolve_comfy_host("boxb", explicit=os.environ.get("BOXB_HOST"))
# Fall back to the canonical table rather than repeating its literal here:
# comfy_endpoints is meant to be the only place a host address is written down.
NP = urlparse(BOXA_HOST).hostname or urlparse(DIRECT_COMFY_HOSTS["boxa"]).hostname
POLL_TIMEOUT = float(os.environ.get("BOXA_POLL_TIMEOUT", "1800"))  # wall-clock seconds before a poll gives up

ROOT = _repo_root()
# Explicit repo-root alias for sibling-shim consumers: scripts/config.py re-exports this
# module; the nsfw mirror's generated config defines its own REPO_ROOT the same way.
REPO_ROOT = ROOT


def _path_from_env(name: str, default: Path) -> Path:
    """Resolve an optional project-local path override without changing defaults.

    Consumer projects can borrow loramake's engines while keeping every render in
    their own repository. Relative overrides remain anchored to the loramake root
    so subprocesses do not change behaviour merely because their cwd changed.
    """
    raw = os.environ.get(name)
    if not raw:
        return default
    path = Path(raw).expanduser()
    return path if path.is_absolute() else ROOT / path


OUTPUTS = _path_from_env("LORAMAKE_OUTPUTS", ROOT / "outputs")
# Shared read-only data (pose library) anchors here. It follows LORAMAKE_OUTPUTS by default,
# but a channel-scoped job (Studio stamps brief.channel for off-LAN callers) moves OUTPUTS to
# its own tree while LORAMAKE_REPO_OUTPUTS keeps shared reads on the real repo outputs.
REPO_OUTPUTS = _path_from_env("LORAMAKE_REPO_OUTPUTS", OUTPUTS)
OUTPUT_CHARACTER = os.environ.get("LORAMAKE_OUTPUT_CHARACTER", "character").strip() or "character"

# Local service addresses.  Two bases per service, on purpose:
#   api    — loopback + port, for scripts, health checks, notifications, tests (the agent
#            sandbox only forwards loopback, and portless' local CA is not trusted by
#            Python's certifi/httpx).
#   public — the portless name (https://<name>.localhost), for every link a person sees:
#            printed batch lines, chat hand-overs, Browser-pane navigation, cross-site
#            links inside the console.  LORAMAKE_PORTLESS=0 falls back to the api base.
# This is the only place these addresses are written down (comfy_endpoints.py plays the
# same role for the render boxes).
STUDIO_PORT = 8873
GALLERY_PORT_DEFAULT = "8771"
CHAT_PORT = 8874
STUDIO_PUBLIC_NAME = "loramake"
GALLERY_PUBLIC_NAME = "gallery"
CHAT_PUBLIC_NAME = "loramake-chat"


def _portless_enabled(env: Mapping[str, str]) -> bool:
    return env.get("LORAMAKE_PORTLESS", "1").strip().lower() not in {"0", "false", "no", "off"}


def _service_urls(env: Mapping[str, str], *, api_env: str, api_default: str,
                  public_env: str, public_name: str) -> dict[str, str]:
    api = (env.get(api_env) or api_default).strip().rstrip("/")
    public = (env.get(public_env) or "").strip().rstrip("/")
    if not public:
        public = f"https://{public_name}.localhost" if _portless_enabled(env) else api
    return {"api": api, "public": public}


def studio_urls(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """``{"api", "public"}`` for the Studio console (:8873 / https://loramake.localhost)."""
    values = os.environ if env is None else env
    return _service_urls(
        values,
        api_env="LORAMAKE_STUDIO_API",
        api_default=f"http://127.0.0.1:{STUDIO_PORT}",
        public_env="LORAMAKE_STUDIO_PUBLIC_URL",
        public_name=STUDIO_PUBLIC_NAME,
    )


def gallery_urls(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """``{"api", "public"}`` for the :8771 batch pages (https://gallery.localhost)."""
    values = os.environ if env is None else env
    port = (values.get("LORAMAKE_GALLERY_PORT") or GALLERY_PORT_DEFAULT).strip() or GALLERY_PORT_DEFAULT
    return _service_urls(
        values,
        api_env="LORAMAKE_GALLERY_API",
        api_default=f"http://127.0.0.1:{port}",
        public_env="LORAMAKE_GALLERY_PUBLIC_URL",
        public_name=GALLERY_PUBLIC_NAME,
    )


def chat_urls(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """``{"api", "public"}`` for the app-chat sidecar (:8874 / https://loramake-chat.localhost)."""
    values = os.environ if env is None else env
    return _service_urls(
        values,
        api_env="LORAMAKE_CHAT_API",
        api_default=f"http://127.0.0.1:{CHAT_PORT}",
        public_env="LORAMAKE_CHAT_PUBLIC_URL",
        public_name=CHAT_PUBLIC_NAME,
    )


def service_urls(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Flat map handed to the console (``/api/v1/capabilities`` → ``urls``)."""
    studio, gallery, chat = studio_urls(env), gallery_urls(env), chat_urls(env)
    return {
        "console_public": studio["public"],
        "console_api": studio["api"],
        "gallery_public": gallery["public"],
        "gallery_api": gallery["api"],
        "chat_public": chat["public"],
    }


def by_model_dir(character: str, line: str = "", *, model: str | None = None) -> Path:
    """Return ``outputs/<character>/by_model/<model>[/<line>]``.

    ``model=None`` reads ``DRIVING_MODEL`` at call time and defaults to
    ``claude``.  No normalization is applied here; callers pass their existing
    import-time model value when they need to preserve script-specific behavior.
    """
    m = model if model is not None else os.environ.get("DRIVING_MODEL", "claude")
    base = OUTPUTS / character / "by_model" / m
    return base / line if line else base
