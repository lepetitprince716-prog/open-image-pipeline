"""Question inbox shared by the gallery ``/inbox`` page and ``scripts/inbox.py``.

An agent that needs a decision writes a question here instead of blocking a chat turn;
the person answers it from the gallery inbox page (or the CLI) and the agent reads the
answer back later. The store is one JSON file, rewritten atomically on every change::

    {"v": 1, "questions": [{"id", "title", "detail", "options", "answer",
                            "asked", "answered"}]}
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
TITLE_MAX = 120
DETAIL_MAX = 2000
OPTION_MAX = 60
OPTIONS_MAX = 6
ANSWER_MAX = 200


class InboxError(ValueError):
    """Raised for malformed store content or an invalid mutation."""


def empty() -> dict:
    return {"v": SCHEMA_VERSION, "questions": []}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _check_text(value: object, *, label: str, limit: int, required: bool) -> str:
    if value is None and not required:
        return ""
    if not isinstance(value, str):
        raise InboxError(f"{label} must be a string")
    text = value.strip()
    if required and not text:
        raise InboxError(f"{label} must not be empty")
    if len(text) > limit:
        raise InboxError(f"{label} exceeds {limit} characters")
    return text


def validate_question(question: object) -> dict:
    """Return a normalized copy of one question or raise ``InboxError``."""
    if not isinstance(question, dict):
        raise InboxError("question must be an object")
    qid = question.get("id")
    if not isinstance(qid, str) or not ID_RE.fullmatch(qid):
        raise InboxError("invalid question id")
    title = _check_text(question.get("title"), label="title", limit=TITLE_MAX, required=True)
    detail = _check_text(question.get("detail"), label="detail", limit=DETAIL_MAX, required=False)
    options = question.get("options")
    if not isinstance(options, list) or not options or len(options) > OPTIONS_MAX:
        raise InboxError(f"options must hold 1 to {OPTIONS_MAX} entries")
    clean_options = [
        _check_text(option, label="option", limit=OPTION_MAX, required=True) for option in options
    ]
    if len(set(clean_options)) != len(clean_options):
        raise InboxError("options must be distinct")
    answer = question.get("answer")
    if answer is not None:
        answer = _check_text(answer, label="answer", limit=ANSWER_MAX, required=True)
    asked = question.get("asked")
    if not isinstance(asked, str) or not asked:
        raise InboxError("asked timestamp missing")
    answered = question.get("answered")
    if (answer is None and answered is not None) or (answer is not None and not isinstance(answered, str)):
        raise InboxError("answered timestamp inconsistent with answer")
    return {
        "id": qid,
        "title": title,
        "detail": detail,
        "options": clean_options,
        "answer": answer,
        "asked": asked,
        "answered": answered,
    }


def load(path: Path) -> dict:
    """Read and validate the store; a missing file is an empty inbox."""
    if not path.is_file():
        return empty()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InboxError(f"unreadable inbox file: {path}") from exc
    if not isinstance(data, dict) or data.get("v") != SCHEMA_VERSION \
            or not isinstance(data.get("questions"), list):
        raise InboxError(f"invalid inbox file: {path}")
    questions = [validate_question(question) for question in data["questions"]]
    ids = [question["id"] for question in questions]
    if len(set(ids)) != len(ids):
        raise InboxError(f"duplicate question ids in {path}")
    return {"v": SCHEMA_VERSION, "questions": questions}


def save(path: Path, data: dict) -> None:
    """Rewrite the store through a same-directory temporary file and an atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".inbox.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def find(data: dict, qid: str) -> dict | None:
    for question in data["questions"]:
        if question["id"] == qid:
            return question
    return None


def ask(data: dict, *, qid: str, title: str, options: list[str], detail: str = "",
        replace: bool = False) -> dict:
    """Append a new unanswered question; ``replace`` drops an existing one with the same id."""
    question = validate_question({
        "id": qid, "title": title, "detail": detail, "options": options,
        "answer": None, "asked": _timestamp(), "answered": None,
    })
    existing = find(data, qid)
    if existing is not None:
        if not replace:
            raise InboxError(f"question already exists: {qid}")
        data["questions"].remove(existing)
    data["questions"].append(question)
    return question


def answer(data: dict, *, qid: str, value: object) -> dict:
    """Record an answer; the value may be one of the options or free text."""
    question = find(data, qid)
    if question is None:
        raise InboxError(f"unknown question: {qid}")
    text = _check_text(value, label="answer", limit=ANSWER_MAX, required=True)
    question["answer"] = text
    question["answered"] = _timestamp()
    return question


def reopen(data: dict, *, qid: str) -> dict:
    """Clear an answer so the question shows as pending again."""
    question = find(data, qid)
    if question is None:
        raise InboxError(f"unknown question: {qid}")
    question["answer"] = None
    question["answered"] = None
    return question


def drop(data: dict, *, qid: str) -> bool:
    question = find(data, qid)
    if question is None:
        return False
    data["questions"].remove(question)
    return True


def pending(data: dict) -> list[dict]:
    return [question for question in data["questions"] if question["answer"] is None]
