"""Push notification for job terminal states (opt-in, fire-and-forget).

The SSE stream only reaches an open browser tab; this module closes the gap for
the away-from-keyboard case (blueprint R5: the phone is the primary remote
surface). Configure with the LORAMAKE_NOTIFY_URL environment variable:

- Bark:  https://api.day.app/<device_key>       (title/body appended as path segments)
- ntfy:  https://ntfy.sh/<topic> or self-hosted (POST body + Title header)

Unset (the default) disables notifications entirely. Delivery is best-effort:
one curl subprocess on a daemon thread, 8s timeout, failures print one WARN and
never affect the job lifecycle.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from urllib.parse import quote

NOTIFY_URL_ENV = "LORAMAKE_NOTIFY_URL"
NOTIFY_TIMEOUT_S = 8
# Terminal states worth a push: cancelled is excluded (the user did it themselves).
NOTIFY_STATUSES = {"done", "error", "interrupted"}


def _build_curl(url: str, title: str, body: str) -> list[str]:
    if "api.day.app" in url:
        target = f"{url.rstrip('/')}/{quote(title, safe='')}/{quote(body, safe='')}"
        return ["curl", "-sS", "--max-time", str(NOTIFY_TIMEOUT_S), "-o", os.devnull, target]
    return ["curl", "-sS", "--max-time", str(NOTIFY_TIMEOUT_S), "-o", os.devnull,
            "-H", f"Title: {title}", "-d", body, url.rstrip("/")]


def _send(url: str, title: str, body: str) -> None:
    try:
        subprocess.run(_build_curl(url, title, body), capture_output=True, text=True,
                       timeout=NOTIFY_TIMEOUT_S + 2)
    except Exception as exc:
        print(f"  notify WARN ({type(exc).__name__}: {exc})", file=sys.stderr)


def job_finished(job: dict | None) -> None:
    """Push one notification for a job that reached a terminal state. No-op unless
    LORAMAKE_NOTIFY_URL is set and the status is in NOTIFY_STATUSES."""
    url = os.environ.get(NOTIFY_URL_ENV, "").strip()
    if not url or not isinstance(job, dict) or job.get("status") not in NOTIFY_STATUSES:
        return
    status = job["status"]
    brief = job.get("brief") or {}
    tier = str(brief.get("tier") or "?")
    box = str(brief.get("box") or "?")
    prompt = str(brief.get("prompt") or "").strip()
    if len(prompt) > 60:
        prompt = prompt[:57] + "..."
    status_label = {"done": "完成", "error": "失败", "interrupted": "中断"}[status]
    title = f"出图{status_label} · {tier}@{box}"
    body = prompt or tier
    if status != "done":
        error = str(job.get("error") or "").strip()
        if error:
            body = f"{body}\n{error[:200]}"
    threading.Thread(target=_send, args=(url, title, body), daemon=True,
                     name="job-notify").start()
