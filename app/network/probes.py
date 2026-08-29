"""Reachability/credential probes for non-box services.

These run only inside ProbeCache's background threads — never in request
handlers. Every subprocess call carries the timeout backstop (--max-time only
caps the transfer, not a wedged curl). Credentials are reported as presence
booleans and travel to curl via `--config -` on stdin, never argv, and never
appear in details or errors.
"""
from __future__ import annotations

import os
import subprocess
import urllib.request
from pathlib import Path

CIVITAI_PROBE_URL = "https://civitai.com/api/v1/models?limit=1"
# Cheap authenticated probe: 200 proves the token works, 401 proves it is
# rejected; either way no Anlas is spent. Endpoint noted for live validation
# in the PR checklist (the probe fails closed to an error card if it moves).
NOVELAI_PROBE_URL = "https://api.novelai.net/user/subscription"
NAI_TOKEN_ENVS = ("NAI_ACCESS_TOKEN",)

ProbeResult = tuple[str, str, dict]


def _nai_token() -> str | None:
    for env_name in NAI_TOKEN_ENVS:
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    path = Path(os.path.expanduser(os.environ.get("NOVELAI_TOKEN_PATH", "~/.config/novelai/token")))
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def _http_code(url: str, *, cfg_text: str | None = None, max_time: float = 6.0) -> str:
    """Return the HTTP status code as text ('' when curl itself failed/hung)."""
    cmd = ["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}",
           "--max-time", f"{max_time:g}"]
    if cfg_text is not None:
        cmd += ["--config", "-"]
    cmd.append(url)
    try:
        proc = subprocess.run(cmd, input=cfg_text, capture_output=True, text=True,
                              timeout=max_time + 2)
    except subprocess.TimeoutExpired:
        return ""
    return (proc.stdout or "").strip()


def _civitai_credential_present() -> bool:
    try:
        from loramake.image_providers import _credential_available, settings
        return bool(_credential_available(settings("civitai")))
    except Exception:
        return False


def probe_civitai() -> ProbeResult:
    extra = {"credential": _civitai_credential_present(), "paid": True}
    code = _http_code(CIVITAI_PROBE_URL)
    if code == "200":
        return "ok", "API 可达", extra
    if not code or code == "000":
        return "error", "API 不可达（超时或网络失败）", extra
    return "error", f"API 返回 HTTP {code}", extra


def probe_novelai() -> ProbeResult:
    token = _nai_token()
    if token is None:
        return "unconfigured", "未配置 token（NAI_ACCESS_TOKEN 或 ~/.config/novelai/token）", \
            {"credential": False, "paid": True}
    extra = {"credential": True, "paid": True}
    # A quote/newline would terminate the curl config line and change its
    # meaning; NovelAI tokens never contain either, so refuse rather than escape.
    if any(ch in token for ch in ('"', "\n", "\r")):
        return "error", "token 含非法字符（引号/换行），请重新获取", extra
    cfg = f'header = "Authorization: Bearer {token}"\n'
    code = _http_code(NOVELAI_PROBE_URL, cfg_text=cfg)
    if code == "200":
        return "ok", "token 有效", extra
    if code == "401":
        return "error", "token 被拒（401，需重新获取）", extra
    if not code or code == "000":
        return "error", "API 不可达（超时或网络失败）", extra
    return "error", f"API 返回 HTTP {code}", extra


def probe_lora_manager(url: str) -> ProbeResult:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # LAN, proxy-free
    try:
        opener.open(url + "/", timeout=3)
    except Exception:
        return "error", "服务不可达", {"url": url}
    return "ok", "服务可达", {"url": url}
