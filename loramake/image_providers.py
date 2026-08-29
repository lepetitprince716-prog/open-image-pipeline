"""Paid cloud image-provider adapters used by :mod:`scripts.gen`.

The app never calls these providers directly.  A brief is persisted in SQLite, the
single worker launches ``gen.py``, and only then does this module perform a confirmed
provider request.  Credentials are loaded lazily, kept in request headers, and never
placed in argv, plans, events, results, or exception text.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

from .config import OUTPUT_CHARACTER, OUTPUTS

# httpx and PIL are imported inside the functions that use them: gen.py imports this
# module on every invocation (including --print planning and local-only runs), and the
# module-level imports were the single largest share of router cold-start time.

EXTERNAL_PROVIDERS = ("civitai", "gpt_image")
PROVIDERS = ("local", *EXTERNAL_PROVIDERS)
DEFAULT_CIVITAI_MODEL = "urn:air:sdxl:checkpoint:civitai:833294@1116447"
DEFAULT_GPT_IMAGE_MODEL = "gpt-image-2"
# Use the configured character rather than repeating its name: LORAMAKE_OUTPUT_CHARACTER
# is meant to move every render root, and a literal here silently ignored it.
DEFAULT_OUTPUT_ROOT = OUTPUTS / OUTPUT_CHARACTER / "by_model"
MAX_JSON_BYTES = 128 * 1024 * 1024
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
_TERMINAL_FAILURES = {"failed", "expired", "canceled", "cancelled", "blocked"}
_ACTIVE_STATUSES = {"unassigned", "preparing", "scheduled", "processing", "pending", "running"}
_SECRET_PATTERNS = (
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"(?i)(api[_-]?key|token|authorization)(\s*[:=]\s*)[^\s,;]+"),
)


class ProviderError(RuntimeError):
    """A safe, user-visible provider failure with a stable classification."""

    def __init__(self, error_class: str, message: str, *, http_status: int | None = None,
                 request_id: str | None = None, provider_status: str = "failed") -> None:
        super().__init__(redact(message))
        self.error_class = error_class
        self.http_status = http_status
        self.request_id = _safe_request_id(request_id)
        self.provider_status = provider_status


@dataclass(frozen=True)
class ProviderSettings:
    provider: str
    label: str
    endpoint: str
    model: str
    size: str
    quality: str | None
    timeout_s: float
    proxy: str | None
    token_envs: tuple[str, ...]
    token_path: Path | None


def redact(value: object, *, secrets: tuple[str, ...] = ()) -> str:
    """Bound and redact text before it can enter a result, event, or log."""
    text = str(value or "")
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda m: (
            f"{m.group(1)}{m.group(2)}[REDACTED]" if m.lastindex == 2 else "Bearer [REDACTED]"
        ), text)
    return text[:800]


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ProviderError("provider_config", f"{name} must be numeric") from exc
    if not minimum <= value <= maximum:
        raise ProviderError("provider_config", f"{name} must be in {minimum}..{maximum}")
    return value


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ProviderError("provider_config", f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ProviderError("provider_config", f"{name} must be in {minimum}..{maximum}")
    return value


def settings(provider: str) -> ProviderSettings:
    """Resolve non-secret provider configuration from environment defaults."""
    if provider == "civitai":
        token_path = Path(os.path.expanduser(
            os.environ.get("CIVITAI_TOKEN_PATH", "~/.config/civitai/token")
        ))
        proxy = os.environ.get(
            "LORAMAKE_CIVITAI_PROXY", os.environ.get("CIVITAI_PROXY", "http://127.0.0.1:1082")
        ).strip() or None
        return ProviderSettings(
            provider=provider,
            label="Civitai",
            endpoint=os.environ.get(
                "LORAMAKE_CIVITAI_ENDPOINT", "https://orchestration.civitai.com"
            ).rstrip("/"),
            model=os.environ.get("LORAMAKE_CIVITAI_MODEL", DEFAULT_CIVITAI_MODEL).strip(),
            size=os.environ.get("LORAMAKE_CIVITAI_SIZE", "832x1216").strip(),
            quality=None,
            timeout_s=_env_float("LORAMAKE_CIVITAI_TIMEOUT_S", 180.0, minimum=10.0, maximum=3600.0),
            proxy=proxy,
            token_envs=("CIVITAI_API_TOKEN", "CIVITAI_TOKEN"),
            token_path=token_path,
        )
    if provider == "gpt_image":
        token_env = os.environ.get("LORAMAKE_GPT_IMAGE_TOKEN_ENV", "OPENAI_API_KEY").strip()
        return ProviderSettings(
            provider=provider,
            label="GPT Image",
            endpoint=os.environ.get(
                "LORAMAKE_GPT_IMAGE_ENDPOINT", "https://api.openai.com/v1/images/generations"
            ).strip(),
            model=os.environ.get("LORAMAKE_GPT_IMAGE_MODEL", DEFAULT_GPT_IMAGE_MODEL).strip(),
            size=os.environ.get("LORAMAKE_GPT_IMAGE_SIZE", "1024x1536").strip(),
            quality=os.environ.get("LORAMAKE_GPT_IMAGE_QUALITY", "medium").strip(),
            timeout_s=_env_float("LORAMAKE_GPT_IMAGE_TIMEOUT_S", 600.0, minimum=10.0, maximum=3600.0),
            proxy=os.environ.get("LORAMAKE_GPT_IMAGE_PROXY", "").strip() or None,
            token_envs=(token_env,) if token_env else (),
            token_path=None,
        )
    raise ProviderError("provider_unknown", f"unknown image provider: {provider}")


def _credential_available(config: ProviderSettings) -> bool:
    if any(bool(os.environ.get(name, "").strip()) for name in config.token_envs):
        return True
    return bool(config.token_path and config.token_path.is_file())


def provider_catalog() -> list[dict[str, Any]]:
    """Public, non-secret channel metadata for ``/api/meta``."""
    out: list[dict[str, Any]] = [{
        "id": "local", "label": "本地 ComfyUI", "paid": False, "configured": True,
        "model": None, "size": None, "quality": None, "error": None,
    }]
    for provider in EXTERNAL_PROVIDERS:
        try:
            config = settings(provider)
            configured = bool(config.model and _credential_available(config))
            error = None
        except ProviderError as exc:
            config = None
            configured = False
            error = str(exc)
        out.append({
            "id": provider,
            "label": config.label if config else provider,
            "paid": True,
            "configured": configured,
            "model": config.model if config else None,
            "size": config.size if config else None,
            "quality": config.quality if config else None,
            "error": error,
        })
    return out


def _load_secret(config: ProviderSettings) -> str:
    for name in config.token_envs:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    if config.token_path:
        try:
            value = config.token_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ProviderError(
                "provider_not_configured", f"{config.label} credential file is unavailable"
            ) from exc
        if value:
            return value
    raise ProviderError("provider_not_configured", f"{config.label} credential is not configured")


def _validate_endpoint(url: str) -> None:
    parsed = urlsplit(url)
    loopback = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    if parsed.scheme not in ({"https"} if not loopback else {"http", "https"}):
        raise ProviderError("provider_config", "provider endpoint must use HTTPS (HTTP allowed on loopback)")
    if not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ProviderError("provider_config", "provider endpoint is malformed")


def _parse_size(value: str, provider: str) -> tuple[int, int] | None:
    if provider == "gpt_image" and value == "auto":
        return None
    match = re.fullmatch(r"([1-9]\d{1,4})x([1-9]\d{1,4})", value)
    if not match:
        raise ProviderError("provider_input", "provider size must be WIDTHxHEIGHT")
    width, height = int(match.group(1)), int(match.group(2))
    if provider == "civitai":
        if not (64 <= width <= 2048 and 64 <= height <= 2048 and width % 16 == 0 and height % 16 == 0):
            raise ProviderError(
                "provider_input", "Civitai size must be 64..2048 per side and divisible by 16"
            )
    elif not (256 <= width <= 4096 and 256 <= height <= 4096):
        raise ProviderError("provider_input", "GPT Image size must be 256..4096 per side")
    return width, height


def parse_passthrough(provider: str, extra: list[str]) -> dict[str, Any]:
    """Parse the deliberately tiny provider-specific subset after ``gen.py --``."""
    allowed = {"--res": "size"}
    if provider == "civitai":
        allowed.update({"--steps": "steps", "--cfg": "cfg"})
    options: dict[str, Any] = {}
    i = 0
    while i < len(extra):
        flag = extra[i]
        key = allowed.get(flag)
        if key is None:
            raise ProviderError("provider_input", f"{flag} is not supported by {provider}")
        if i + 1 >= len(extra):
            raise ProviderError("provider_input", f"{flag} requires a value")
        if key in options:
            raise ProviderError("provider_input", f"{flag} may be supplied only once")
        raw = extra[i + 1]
        try:
            options[key] = int(raw) if key == "steps" else float(raw) if key == "cfg" else raw
        except ValueError as exc:
            raise ProviderError("provider_input", f"{flag} has an invalid value") from exc
        i += 2
    return options


def plan(provider: str, prompt: str, *, n: int | None = None, seed: int | None = None,
         options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build and validate a secret-free request summary without network access."""
    config = settings(provider)
    _validate_endpoint(config.endpoint)
    prompt = str(prompt or "").strip()
    if not prompt:
        raise ProviderError("provider_input", "prompt is required")
    if len(prompt) > (10_000 if provider == "civitai" else 32_000):
        raise ProviderError("provider_input", f"prompt is too long for {config.label}")
    quantity = 1 if n is None else n
    maximum = 12 if provider == "civitai" else 10
    if isinstance(quantity, bool) or not isinstance(quantity, int) or not 1 <= quantity <= maximum:
        raise ProviderError("provider_input", f"{config.label} quantity must be in 1..{maximum}")
    if seed is not None and provider == "gpt_image":
        raise ProviderError("provider_input", "GPT Image does not accept seed on this adapter")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ProviderError("provider_input", "seed must be a non-negative integer")
    opts = dict(options or {})
    size = str(opts.get("size") or config.size)
    dimensions = _parse_size(size, provider)
    request: dict[str, Any] = {
        "provider": provider,
        "label": config.label,
        "paid": True,
        "model": config.model,
        "size": size,
        "quality": config.quality,
        "quantity": quantity,
    }
    if not config.model:
        raise ProviderError("provider_config", f"{config.label} model is not configured")
    if provider == "civitai":
        steps = opts.get("steps", _env_int("LORAMAKE_CIVITAI_STEPS", 30, minimum=1, maximum=150))
        cfg = opts.get("cfg", _env_float("LORAMAKE_CIVITAI_CFG", 5.0, minimum=0.0, maximum=30.0))
        if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 150:
            raise ProviderError("provider_input", "Civitai steps must be in 1..150")
        if isinstance(cfg, bool) or not isinstance(cfg, (int, float)) or not 0 <= float(cfg) <= 30:
            raise ProviderError("provider_input", "Civitai cfg must be in 0..30")
        request.update(width=dimensions[0], height=dimensions[1], steps=steps, cfg=float(cfg))
        if seed is not None:
            request["seed"] = seed
    return request


def _safe_request_id(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return re.sub(r"[^A-Za-z0-9_.:-]", "_", value.strip())[:160]


def _emit_event(provider: str, event: str, status: str, *, request_id: str | None = None,
                detail: str | None = None, elapsed_s: float | None = None) -> None:
    path = os.environ.get("GEN_EVENTS_OUT", "").strip()
    if not path:
        return
    payload: dict[str, Any] = {
        "event": event,
        "provider": provider,
        "provider_status": status,
        "at": time.time(),
    }
    safe_id = _safe_request_id(request_id)
    if safe_id:
        payload["provider_request_id"] = safe_id
        payload["prompt_id"] = safe_id  # existing attempt ledger identity field
    if detail:
        payload["detail"] = redact(detail)
    if elapsed_s is not None:
        payload["elapsed_s"] = round(elapsed_s, 3)
    event_path = Path(path)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(event_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
    finally:
        os.close(fd)


def _request_json(method: str, url: str, *, headers: dict[str, str], payload: dict[str, Any] | None,
                  timeout_s: float, proxy: str | None) -> tuple[int, Any, dict[str, str]]:
    import httpx

    try:
        with httpx.Client(timeout=timeout_s, follow_redirects=True, proxy=proxy) as client:
            response = client.request(method, url, headers=headers, json=payload)
    except httpx.HTTPError as exc:
        raise ProviderError("provider_transport", f"provider transport failed: {type(exc).__name__}") from exc
    limit = _env_int("LORAMAKE_PROVIDER_MAX_JSON_BYTES", MAX_JSON_BYTES,
                     minimum=1024, maximum=256 * 1024 * 1024)
    if len(response.content) > limit:
        raise ProviderError("provider_protocol", "provider JSON response exceeded the configured limit",
                            http_status=response.status_code)
    try:
        body = response.json()
    except ValueError as exc:
        raise ProviderError("provider_protocol", "provider returned non-JSON data",
                            http_status=response.status_code) from exc
    return response.status_code, body, {k.lower(): v for k, v in response.headers.items()}


def _download_bytes(url: str, *, timeout_s: float, proxy: str | None) -> bytes:
    import httpx

    limit = _env_int("LORAMAKE_PROVIDER_MAX_IMAGE_BYTES", MAX_IMAGE_BYTES,
                     minimum=1024, maximum=256 * 1024 * 1024)
    try:
        with httpx.Client(timeout=timeout_s, follow_redirects=True, proxy=proxy) as client:
            with client.stream("GET", url) as response:
                if not 200 <= response.status_code < 300:
                    raise ProviderError("provider_download", f"provider image download HTTP {response.status_code}",
                                        http_status=response.status_code)
                chunks: list[bytes] = []
                size = 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > limit:
                        raise ProviderError("provider_download", "provider image exceeded the configured limit")
                    chunks.append(chunk)
    except ProviderError:
        raise
    except httpx.HTTPError as exc:
        raise ProviderError("provider_download", f"provider image download failed: {type(exc).__name__}") from exc
    return b"".join(chunks)


def _remote_message(payload: Any, *, secrets: tuple[str, ...] = ()) -> str:
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return redact(error.get("message") or error.get("code") or "provider error",
                          secrets=secrets)
        if isinstance(error, str):
            return redact(error, secrets=secrets)
        return redact(
            payload.get("message") or payload.get("detail") or payload.get("status") or "provider error",
            secrets=secrets,
        )
    return "provider error"


def _require_success(provider: str, status: int, payload: Any, headers: dict[str, str],
                     *, secrets: tuple[str, ...] = ()) -> None:
    if 200 <= status < 300:
        return
    request_id = headers.get("x-request-id") or headers.get("request-id")
    raise ProviderError("provider_http", f"{provider} HTTP {status}: "
                        f"{_remote_message(payload, secrets=secrets)}",
                        http_status=status, request_id=request_id)


def _cost_entries(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    transactions = payload.get("transactions") or {}
    rows = transactions.get("list") if isinstance(transactions, dict) else transactions
    out = []
    for row in rows or []:
        if not isinstance(row, dict) or not isinstance(row.get("amount"), (int, float)):
            continue
        out.append({
            "amount": row["amount"],
            "currency": str(row.get("accountType") or row.get("currency") or "unknown")[:40],
        })
    return out


def _image_descriptors(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    out: list[dict[str, Any]] = []
    direct = payload.get("images") or payload.get("data") or []
    if isinstance(direct, list):
        out.extend(item for item in direct if isinstance(item, dict))
    for step in payload.get("steps") or []:
        if not isinstance(step, dict):
            continue
        output = step.get("output") or {}
        for item in output.get("images") or [] if isinstance(output, dict) else []:
            if isinstance(item, dict):
                out.append(item)
    return out


def _descriptor_bytes(item: dict[str, Any], config: ProviderSettings) -> bytes:
    encoded = item.get("b64_json")
    if isinstance(encoded, str) and encoded:
        try:
            return base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ProviderError("provider_protocol", "provider returned invalid base64 image data") from exc
    url = item.get("url") or item.get("previewUrl")
    if isinstance(url, str) and url.startswith(("https://", "http://")):
        return _download_bytes(url, timeout_s=config.timeout_s, proxy=config.proxy)
    raise ProviderError("provider_protocol", "provider response contained no usable image payload")


def _store_pngs(provider: str, descriptors: list[dict[str, Any]], config: ProviderSettings,
                quantity: int) -> list[dict[str, Any]]:
    from PIL import Image, UnidentifiedImageError

    if not descriptors:
        raise ProviderError("provider_no_output", "provider returned no images")
    attempt = re.sub(r"[^A-Za-z0-9_-]", "_", os.environ.get("LORAMAKE_ATTEMPT_ID", ""))[:80]
    run_id = attempt or uuid.uuid4().hex
    folder = "gpt-image" if provider == "gpt_image" else provider
    output_dir = DEFAULT_OUTPUT_ROOT / folder / "app"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    for index, item in enumerate(descriptors[:quantity], start=1):
        raw = _descriptor_bytes(item, config)
        if not raw:
            raise ProviderError("provider_protocol", "provider returned an empty image")
        try:
            with Image.open(io.BytesIO(raw)) as opened:
                width, height = opened.size
                if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                    raise ProviderError("provider_protocol", "provider image dimensions are unsafe")
                opened.load()
                transparent = opened.mode in {"RGBA", "LA"} or "transparency" in opened.info
                image = opened.convert("RGBA" if transparent else "RGB")
        except ProviderError:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ProviderError("provider_protocol", "provider returned an undecodable image") from exc
        target = output_dir / f"{run_id}_{index:02d}.png"
        if target.exists():
            raise ProviderError("provider_output_conflict", "provider output target already exists")
        temp = output_dir / f".{target.name}.{uuid.uuid4().hex}.tmp"
        try:
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                image.save(handle, format="PNG")
            os.link(temp, target)  # atomic no-clobber publish
        finally:
            temp.unlink(missing_ok=True)
        payload = target.read_bytes()
        artifacts.append({
            "path": str(target.resolve()),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "width": width,
            "height": height,
        })
    return artifacts


def _civitai_input(request: dict[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {
        "engine": "sdcpp",
        "ecosystem": "sdxl",
        "operation": "createImage",
        "model": request["model"],
        "prompt": request["prompt"],
        "negativePrompt": os.environ.get(
            "LORAMAKE_CIVITAI_NEGATIVE", "lowres, worst quality, low quality, bad anatomy"
        ),
        "width": request["width"],
        "height": request["height"],
        "cfgScale": request["cfg"],
        "steps": request["steps"],
        "quantity": request["quantity"],
    }
    if "seed" in request:
        body["seed"] = request["seed"]
    return body


def _run_civitai(config: ProviderSettings, request: dict[str, Any], secret: str,
                  started: float) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {secret}", "Content-Type": "application/json",
               "User-Agent": "loramake-provider-adapter/1.0"}
    image_input = _civitai_input(request)
    preview_url = (
        f"{config.endpoint}/v2/consumer/workflows?" +
        urlencode({"whatif": "true", "allowMatureContent": "false"})
    )
    status, preview, preview_headers = _request_json(
        "POST", preview_url, headers=headers,
        payload={"steps": [{"$type": "imageGen", "input": image_input}]},
        timeout_s=config.timeout_s, proxy=config.proxy,
    )
    _require_success("Civitai", status, preview, preview_headers, secrets=(secret,))
    cost = _cost_entries(preview)
    _emit_event("civitai", "provider_cost_previewed", "cost_previewed",
                detail=json.dumps(cost, ensure_ascii=False), elapsed_s=time.monotonic() - started)

    recipe_url = (
        f"{config.endpoint}/v2/consumer/recipes/imageGen?" +
        urlencode({"wait": "60", "allowMatureContent": "false"})
    )
    try:
        status, payload, response_headers = _request_json(
            "POST", recipe_url, headers=headers, payload=image_input,
            timeout_s=config.timeout_s, proxy=config.proxy,
        )
    except ProviderError as exc:
        raise ProviderError(
            "provider_outcome_unknown",
            "Civitai paid request outcome is unknown after a transport/protocol failure; do not resubmit blindly",
            provider_status="outcome_unknown",
        ) from exc
    try:
        _require_success("Civitai", status, payload, response_headers, secrets=(secret,))
    except ProviderError as exc:
        if status < 500:
            raise
        raise ProviderError(
            "provider_outcome_unknown",
            "Civitai paid request returned a server error; its outcome is unknown",
            http_status=status,
            request_id=exc.request_id,
            provider_status="outcome_unknown",
        ) from exc
    request_id = _safe_request_id(
        payload.get("id") if isinstance(payload, dict) else None
    ) or _safe_request_id(response_headers.get("x-request-id"))
    provider_status = str(payload.get("status") or "processing") if isinstance(payload, dict) else "processing"
    _emit_event("civitai", "provider_request_submitted", provider_status,
                request_id=request_id, elapsed_s=time.monotonic() - started)

    descriptors = _image_descriptors(payload)
    deadline = time.monotonic() + config.timeout_s
    while not descriptors and request_id and provider_status.lower() in _ACTIVE_STATUSES:
        if time.monotonic() >= deadline:
            raise ProviderError(
                "provider_outcome_unknown",
                "Civitai workflow polling timed out; the paid request may still complete",
                request_id=request_id,
                provider_status="outcome_unknown",
            )
        time.sleep(2.0)
        try:
            poll_status, payload, poll_headers = _request_json(
                "GET", f"{config.endpoint}/v2/consumer/workflows/{request_id}", headers=headers,
                payload=None, timeout_s=min(config.timeout_s, 60.0), proxy=config.proxy,
            )
        except ProviderError as exc:
            raise ProviderError(
                "provider_outcome_unknown",
                "Civitai workflow state is unknown after polling failed; do not resubmit blindly",
                request_id=request_id,
                provider_status="outcome_unknown",
            ) from exc
        try:
            _require_success("Civitai", poll_status, payload, poll_headers, secrets=(secret,))
        except ProviderError as exc:
            raise ProviderError(
                "provider_outcome_unknown",
                "Civitai workflow state is unknown after polling returned an error; do not resubmit blindly",
                http_status=poll_status,
                request_id=request_id,
                provider_status="outcome_unknown",
            ) from exc
        provider_status = str(payload.get("status") or provider_status).lower()
        _emit_event("civitai", "provider_status", provider_status, request_id=request_id,
                    elapsed_s=time.monotonic() - started)
        if provider_status in _TERMINAL_FAILURES:
            raise ProviderError("provider_failed", f"Civitai workflow ended as {provider_status}",
                                request_id=request_id)
        descriptors = _image_descriptors(payload)
    errors = payload.get("errors") if isinstance(payload, dict) else None
    if errors:
        raise ProviderError("provider_failed", f"Civitai generation failed: {redact(errors)}",
                            request_id=request_id)
    if not descriptors and provider_status.lower() in _ACTIVE_STATUSES:
        raise ProviderError(
            "provider_outcome_unknown",
            "Civitai accepted the paid request without a trackable result; do not resubmit blindly",
            request_id=request_id,
            provider_status="outcome_unknown",
        )
    try:
        artifacts = _store_pngs("civitai", descriptors, config, request["quantity"])
    except ProviderError as exc:
        raise ProviderError(
            exc.error_class,
            f"Civitai generation completed but its result could not be stored: {exc}",
            http_status=exc.http_status,
            request_id=request_id,
            provider_status="result_unavailable",
        ) from exc
    return {
        "provider_status": "succeeded",
        "provider_request_id": request_id,
        "cost_preview": cost,
        "charged": _cost_entries(payload),
        "artifacts": artifacts,
    }


def _numeric_usage(value: Any) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return {str(k)[:80]: item for k, v in value.items() if (item := _numeric_usage(v)) is not None}
    return None


def _run_gpt_image(config: ProviderSettings, request: dict[str, Any], secret: str,
                   started: float) -> dict[str, Any]:
    body = {
        "model": request["model"],
        "prompt": request["prompt"],
        "n": request["quantity"],
        "size": request["size"],
        "quality": request["quality"],
        "output_format": "png",
    }
    headers = {"Authorization": f"Bearer {secret}", "Content-Type": "application/json"}
    try:
        status, payload, response_headers = _request_json(
            "POST", config.endpoint, headers=headers, payload=body,
            timeout_s=config.timeout_s, proxy=config.proxy,
        )
    except ProviderError as exc:
        raise ProviderError(
            "provider_outcome_unknown",
            "GPT Image request outcome is unknown after a transport/protocol failure; do not resubmit blindly",
            provider_status="outcome_unknown",
        ) from exc
    request_id = _safe_request_id(response_headers.get("x-request-id") or response_headers.get("request-id"))
    try:
        _require_success("GPT Image", status, payload, response_headers, secrets=(secret,))
    except ProviderError as exc:
        if status < 500:
            raise
        raise ProviderError(
            "provider_outcome_unknown",
            "GPT Image returned a server error; the paid request outcome is unknown",
            http_status=status,
            request_id=exc.request_id,
            provider_status="outcome_unknown",
        ) from exc
    _emit_event("gpt_image", "provider_request_submitted", "processing", request_id=request_id,
                elapsed_s=time.monotonic() - started)
    try:
        artifacts = _store_pngs("gpt_image", _image_descriptors(payload), config, request["quantity"])
    except ProviderError as exc:
        raise ProviderError(
            exc.error_class,
            f"GPT Image generation completed but its result could not be stored: {exc}",
            http_status=exc.http_status,
            request_id=request_id,
            provider_status="result_unavailable",
        ) from exc
    return {
        "provider_status": "succeeded",
        "provider_request_id": request_id,
        "cost_preview": None,
        "charged": None,
        "usage": _numeric_usage(payload.get("usage")) if isinstance(payload, dict) else None,
        "artifacts": artifacts,
    }


def preview_cost(provider: str, prompt: str, *, n: int | None = None, seed: int | None = None,
                 options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Whatif-only cost preview: validates like plan(), asks the provider what the
    request would charge, and never submits paid work. Civitai only — GPT Image has
    no whatif endpoint on this adapter."""
    if provider != "civitai":
        raise ProviderError("provider_input", f"cost preview is not available for {provider}")
    request = plan(provider, prompt, n=n, seed=seed, options=options)
    request["prompt"] = str(prompt).strip()
    config = settings(provider)
    secret = _load_secret(config)
    headers = {"Authorization": f"Bearer {secret}", "Content-Type": "application/json",
               "User-Agent": "loramake-provider-adapter/1.0"}
    preview_url = (
        f"{config.endpoint}/v2/consumer/workflows?" +
        urlencode({"whatif": "true", "allowMatureContent": "false"})
    )
    status, preview, preview_headers = _request_json(
        "POST", preview_url, headers=headers,
        payload={"steps": [{"$type": "imageGen", "input": _civitai_input(request)}]},
        timeout_s=min(config.timeout_s, 30.0), proxy=config.proxy,
    )
    _require_success("Civitai", status, preview, preview_headers, secrets=(secret,))
    return {
        "provider": provider,
        "cost": _cost_entries(preview),
        "size": request["size"],
        "quantity": request["quantity"],
    }


def run_provider(provider: str, prompt: str, *, n: int | None = None, seed: int | None = None,
                 options: dict[str, Any] | None = None, confirmed: bool = False) -> dict[str, Any]:
    """Execute one confirmed paid request and return only normalized, non-secret data."""
    if confirmed is not True:
        raise ProviderError(
            "provider_confirmation_required", "paid generation requires explicit per-request confirmation"
        )
    request = plan(provider, prompt, n=n, seed=seed, options=options)
    request["prompt"] = str(prompt).strip()  # plan deliberately omits prompt text from public summaries
    config = settings(provider)
    secret = _load_secret(config)
    started = time.monotonic()
    _emit_event(provider, "provider_request_started", "starting")
    try:
        if provider == "civitai":
            outcome = _run_civitai(config, request, secret, started)
        elif provider == "gpt_image":
            outcome = _run_gpt_image(config, request, secret, started)
        else:  # guarded by plan(); retained as a fail-closed assertion at the mutation boundary
            raise ProviderError("provider_unknown", f"unknown image provider: {provider}")
    except ProviderError as exc:
        _emit_event(provider, "provider_failed", exc.provider_status, request_id=exc.request_id,
                    detail=str(exc), elapsed_s=time.monotonic() - started)
        raise
    request_summary = {k: v for k, v in request.items() if k != "prompt"}
    paths = [item["path"] for item in outcome["artifacts"]]
    _emit_event(provider, "provider_succeeded", "succeeded",
                request_id=outcome.get("provider_request_id"), elapsed_s=time.monotonic() - started)
    return {
        "provider": provider,
        "provider_status": outcome["provider_status"],
        "provider_request_id": outcome.get("provider_request_id"),
        "request_summary": request_summary,
        "cost_preview": outcome.get("cost_preview"),
        "charged": outcome.get("charged"),
        "usage": outcome.get("usage"),
        "artifacts": outcome["artifacts"],
        "new_files": paths,
        "images": paths,
    }
