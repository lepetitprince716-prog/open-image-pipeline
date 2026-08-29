"""Offline contracts for the Studio ComfyUI progress relay.

The relay is deliberately strict: until the job attempt's remote prompt id is known it
publishes nothing, and once known it only forwards progress frames that belong to that
prompt.  This keeps external prompts on the same box from painting over the Studio job
card (documented 2026-08-10 crosstalk boundary).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from app import jobs


def _progress_frame(value: int, *, prompt_id: str | None = "ours") -> str:
    data: dict = {"value": value, "max": 10, "node": "7"}
    if prompt_id is not None:
        data["prompt_id"] = prompt_id
    return json.dumps({"type": "progress", "data": data})


def _execution_start(prompt_id: str, kind: str = "execution_start") -> str:
    return json.dumps({"type": kind, "data": {"prompt_id": prompt_id}})


def test_progress_event_historical_unfiltered_call() -> None:
    event = jobs._progress_event("j1", "z3", _progress_frame(5))
    assert event == {"type": "progress", "job_id": "j1", "box": "z3",
                     "value": 5, "max": 10, "node": "7"}


def test_progress_event_strict_prompt_filter() -> None:
    raw = _progress_frame(5)
    assert jobs._progress_event("j1", "z3", raw, expected_prompt_id="ours") == {
        "type": "progress", "job_id": "j1", "box": "z3",
        "value": 5, "max": 10, "node": "7",
    }
    assert jobs._progress_event("j1", "z3", raw, expected_prompt_id="other") is None
    # Nothing may be published before the attempt knows its own remote prompt id.
    assert jobs._progress_event("j1", "z3", raw, expected_prompt_id=None) is None


def test_progress_event_uses_execution_start_fallback() -> None:
    raw = _progress_frame(5, prompt_id=None)
    event = jobs._progress_event_from_message(
        "j1", "z3", json.loads(raw),
        expected_prompt_id="ours", executing_prompt_id="ours",
    )
    assert event is not None and event["value"] == 5
    assert jobs._progress_event_from_message(
        "j1", "z3", json.loads(raw),
        expected_prompt_id="ours", executing_prompt_id="other",
    ) is None
    # Embedded prompt id wins over the execution tracker.
    raw_with_id = _progress_frame(6, prompt_id="other")
    assert jobs._progress_event_from_message(
        "j1", "z3", json.loads(raw_with_id),
        expected_prompt_id="ours", executing_prompt_id="ours",
    ) is None


def test_progress_event_ignores_non_progress_and_binary_frames() -> None:
    assert jobs._progress_event("j1", "z3", _execution_start("ours")) is None
    assert jobs._progress_event("j1", "z3", b"\x89PNG") is None
    assert jobs._progress_event("j1", "z3", "not json") is None
    assert jobs._progress_event("j1", "z3", json.dumps({"type": "status"})) is None
    assert jobs._progress_event(
        "j1", "z3", json.dumps({"type": "progress", "data": ["malformed"]}),
    ) is None


class _FakeWebSocket:
    def __init__(self, frames: list, stop: threading.Event) -> None:
        self.frames = list(frames)
        self.stop = stop

    def recv(self, timeout: float):
        del timeout  # frames are deterministic and immediate
        if not self.frames:
            self.stop.set()
            raise TimeoutError
        return self.frames.pop(0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def test_progress_relay_forwards_only_the_current_attempt_prompt(monkeypatch, tmp_path) -> None:
    db = tmp_path / "jobs.sqlite3"
    job = jobs.submit({"tier": "best", "prompt": "1girl"}, db_path=db)
    conn = jobs._connect(db)
    attempt_id, _result_path, _event_path = jobs._new_attempt(conn, job, db)
    conn.execute(
        "UPDATE job_attempts SET remote_prompt_id='ours' WHERE attempt_id=?", (attempt_id,),
    )
    conn.commit()
    conn.close()

    stop = threading.Event()
    frames = [
        json.dumps({"type": "status", "data": {"status": {}}}),
        _execution_start("other"),
        _progress_frame(1, prompt_id="other"),
        _execution_start("ours"),
        _progress_frame(2, prompt_id="ours"),
        _progress_frame(3, prompt_id=None),  # falls back to executing prompt id
        _progress_frame(4, prompt_id="other"),
        b"\x89PNG binary preview",
        "not json",
    ]
    published: list[dict] = []
    monkeypatch.setattr(jobs, "HOSTS", {"zzz": "http://127.0.0.1:8188"})
    monkeypatch.setattr(jobs, "PROGRESS_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(jobs, "_publish", published.append)
    monkeypatch.setattr(
        jobs, "_ws_connect",
        lambda url, open_timeout: _FakeWebSocket(frames, stop),
    )

    jobs._progress_relay(job["id"], attempt_id, "zzz", stop, db)

    assert [event["value"] for event in published] == [2, 3]
    assert all(event["job_id"] == job["id"] for event in published)
    assert all(event["box"] == "zzz" for event in published)


def test_progress_relay_holds_until_the_attempt_prompt_is_known(monkeypatch, tmp_path) -> None:
    db = tmp_path / "jobs.sqlite3"
    job = jobs.submit({"tier": "best", "prompt": "1girl"}, db_path=db)
    conn = jobs._connect(db)
    attempt_id, _result_path, _event_path = jobs._new_attempt(conn, job, db)
    conn.commit()
    conn.close()

    stop = threading.Event()
    frames = [
        _execution_start("other"),
        _progress_frame(1, prompt_id="other"),
        _execution_start("ours"),
        _progress_frame(2, prompt_id="ours"),
    ]
    published: list[dict] = []
    monkeypatch.setattr(jobs, "HOSTS", {"zzz": "http://127.0.0.1:8188"})
    monkeypatch.setattr(jobs, "PROGRESS_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(jobs, "_publish", published.append)
    monkeypatch.setattr(
        jobs, "_ws_connect",
        lambda url, open_timeout: _FakeWebSocket(frames, stop),
    )

    jobs._progress_relay(job["id"], attempt_id, "zzz", stop, db)

    assert published == []


def test_progress_relay_picks_up_prompt_id_written_mid_stream(monkeypatch, tmp_path) -> None:
    """The production race: /ws progress can start before the event relay commits
    prompt_submitted.  The relay re-reads remote_prompt_id on every progress frame,
    so frames before the commit are dropped and frames after it are published."""
    db = tmp_path / "jobs.sqlite3"
    job = jobs.submit({"tier": "best", "prompt": "1girl"}, db_path=db)
    conn = jobs._connect(db)
    attempt_id, _result_path, _event_path = jobs._new_attempt(conn, job, db)
    conn.commit()
    conn.close()

    stop = threading.Event()
    frames = [
        _execution_start("ours"),
        _progress_frame(1, prompt_id="ours"),
        _progress_frame(2, prompt_id="ours"),
        _progress_frame(3, prompt_id="other"),
    ]

    class MidStreamWebSocket:
        def __init__(self) -> None:
            self.calls = 0

        def recv(self, timeout: float):
            del timeout
            if self.calls >= len(frames):
                stop.set()
                raise TimeoutError
            # Between the first dropped progress frame and the second, the attempt
            # event relay commits the prompt id — exactly the startup race.
            if self.calls == 2:
                write_conn = jobs._connect(db)
                write_conn.execute(
                    "UPDATE job_attempts SET remote_prompt_id='ours' WHERE attempt_id=?",
                    (attempt_id,),
                )
                write_conn.commit()
                write_conn.close()
            raw = frames[self.calls]
            self.calls += 1
            return raw

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    published: list[dict] = []
    monkeypatch.setattr(jobs, "HOSTS", {"zzz": "http://127.0.0.1:8188"})
    monkeypatch.setattr(jobs, "PROGRESS_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(jobs, "_publish", published.append)
    monkeypatch.setattr(jobs, "_ws_connect", lambda url, open_timeout: MidStreamWebSocket())

    jobs._progress_relay(job["id"], attempt_id, "zzz", stop, db)

    assert [event["value"] for event in published] == [2]


def test_progress_relay_is_best_effort_without_host_or_websocket(monkeypatch, tmp_path) -> None:
    db = tmp_path / "jobs.sqlite3"
    job = jobs.submit({"tier": "best", "prompt": "1girl"}, db_path=db)
    conn = jobs._connect(db)
    attempt_id, _result_path, _event_path = jobs._new_attempt(conn, job, db)
    conn.commit()
    conn.close()

    stop = threading.Event()
    monkeypatch.setattr(jobs, "HOSTS", {})
    jobs._progress_relay(job["id"], attempt_id, "zzz", stop, db)

    monkeypatch.setattr(jobs, "HOSTS", {"zzz": "http://127.0.0.1:8188"})
    monkeypatch.setattr(jobs, "_ws_connect", None)
    jobs._progress_relay(job["id"], attempt_id, "zzz", stop, db)


@pytest.mark.parametrize("bad_db", [False, True])
def test_progress_relay_does_not_raise_on_db_or_socket_errors(monkeypatch, tmp_path, bad_db) -> None:
    db = tmp_path / "jobs.sqlite3"
    job = jobs.submit({"tier": "best", "prompt": "1girl"}, db_path=db)
    conn = jobs._connect(db)
    attempt_id, _result_path, _event_path = jobs._new_attempt(conn, job, db)
    conn.commit()
    conn.close()

    stop = threading.Event()
    monkeypatch.setattr(jobs, "HOSTS", {"zzz": "http://127.0.0.1:8188"})

    class _ExplodingWebSocket:
        def __init__(self, *_args, **_kwargs) -> None:
            if bad_db:
                raise RuntimeError("socket open failed")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def recv(self, timeout: float):
            raise RuntimeError("socket read failed")

    monkeypatch.setattr(jobs, "_ws_connect", lambda url, open_timeout: _ExplodingWebSocket())

    jobs._progress_relay(job["id"], attempt_id, "zzz", stop, db)
