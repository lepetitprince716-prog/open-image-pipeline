"""Job terminal-state push notifications (offline)."""
from __future__ import annotations

from app import notify


def _capture(monkeypatch):
    sent: list[tuple[str, str, str]] = []
    monkeypatch.setattr(notify.threading, "Thread",
                        lambda *, target, args, daemon, name: _SyncThread(target, args, sent))
    return sent


class _SyncThread:
    def __init__(self, target, args, sink):
        self._sink = sink
        self._args = args

    def start(self):
        self._sink.append(self._args)


def test_job_finished_noop_without_url_or_on_cancel(monkeypatch):
    sent = _capture(monkeypatch)
    monkeypatch.delenv(notify.NOTIFY_URL_ENV, raising=False)
    notify.job_finished({"status": "done", "brief": {"tier": "best"}})
    monkeypatch.setenv(notify.NOTIFY_URL_ENV, "https://ntfy.sh/topic")
    notify.job_finished({"status": "cancelled", "brief": {"tier": "best"}})
    notify.job_finished(None)
    assert sent == []


def test_job_finished_pushes_done_and_error_with_detail(monkeypatch):
    sent = _capture(monkeypatch)
    monkeypatch.setenv(notify.NOTIFY_URL_ENV, "https://ntfy.sh/topic")
    notify.job_finished({"status": "done",
                        "brief": {"tier": "best", "box": "zzz", "prompt": "1girl, silver hair"}})
    notify.job_finished({"status": "error", "error": "torch.OutOfMemoryError: CUDA out of memory",
                        "brief": {"tier": "artistic", "box": "z3", "prompt": "p"}})
    assert len(sent) == 2
    _, done_title, done_body = sent[0]
    assert done_title == "出图完成 · best@zzz"
    assert done_body == "1girl, silver hair"
    _, err_title, err_body = sent[1]
    assert err_title == "出图失败 · artistic@z3"
    assert "CUDA out of memory" in err_body


def test_build_curl_bark_vs_ntfy():
    bark = notify._build_curl("https://api.day.app/devkey", "标题 A", "正文/b")
    assert bark[-1].startswith("https://api.day.app/devkey/")
    assert "%2F" in bark[-1]  # body slash escaped into the path segment
    ntfy = notify._build_curl("https://ntfy.sh/topic", "T", "B")
    assert ntfy[-1] == "https://ntfy.sh/topic"
    assert "Title: T" in ntfy and "-d" in ntfy
