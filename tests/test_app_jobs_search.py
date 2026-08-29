"""Job history search and asset-to-job reverse lookup."""
from __future__ import annotations

import json
import sqlite3
import time

from app import jobs


def _seed(db, rows):
    """Insert bare job rows: (id, brief, status, created_at, result)."""
    conn = sqlite3.connect(db)
    conn.executescript(jobs._SCHEMA)
    conn.execute(f"PRAGMA user_version={jobs.SCHEMA_VERSION}")
    conn.executemany(
        "INSERT INTO jobs (id,brief,status,created_at,result) VALUES (?,?,?,?,?)",
        [
            (jid, json.dumps(brief, ensure_ascii=False), status, ts,
             json.dumps(result, ensure_ascii=False) if result is not None else None)
            for jid, brief, status, ts, result in rows
        ],
    )
    conn.commit()
    conn.close()


def test_search_filters_by_brief_substring_and_status(tmp_path):
    db = tmp_path / "jobs.sqlite3"
    _seed(db, [
        ("job1", {"tier": "best", "prompt": "library interior"}, "done", 100.0, None),
        ("job2", {"tier": "draft", "prompt": "beach sunset"}, "done", 200.0, None),
        ("job3", {"tier": "best", "prompt": "library at night"}, "error", 300.0, None),
    ])
    out = jobs.search_jobs(q="library", db_path=db)
    assert out["total"] == 2
    assert [j["id"] for j in out["jobs"]] == ["job3", "job1"]  # newest first

    out = jobs.search_jobs(q="library", status="done", db_path=db)
    assert out["total"] == 1 and out["jobs"][0]["id"] == "job1"

    out = jobs.search_jobs(status="error", db_path=db)
    assert [j["id"] for j in out["jobs"]] == ["job3"]


def test_search_pages_with_stable_total(tmp_path):
    db = tmp_path / "jobs.sqlite3"
    _seed(db, [
        (f"job{i}", {"tier": "best", "prompt": f"prompt {i}"}, "done", float(i), None)
        for i in range(5)
    ])
    page1 = jobs.search_jobs(limit=2, offset=0, db_path=db)
    page2 = jobs.search_jobs(limit=2, offset=2, db_path=db)
    assert page1["total"] == page2["total"] == 5
    assert [j["id"] for j in page1["jobs"]] == ["job4", "job3"]
    assert [j["id"] for j in page2["jobs"]] == ["job2", "job1"]


def test_search_escapes_like_metacharacters(tmp_path):
    db = tmp_path / "jobs.sqlite3"
    _seed(db, [
        ("job1", {"tier": "best", "prompt": "100% wool"}, "done", 100.0, None),
        ("job2", {"tier": "best", "prompt": "100x wool"}, "done", 200.0, None),
    ])
    out = jobs.search_jobs(q="100%", db_path=db)
    assert [j["id"] for j in out["jobs"]] == ["job1"]


def test_find_job_by_file_checks_new_files_membership(tmp_path):
    db = tmp_path / "jobs.sqlite3"
    hit = "/outputs/by_model/a/img_001.png"
    _seed(db, [
        # job2 is newer and mentions the path outside new_files (contact_sheet):
        # LIKE matches both rows, the membership check must skip job2 and
        # return the true owner
        ("job1", {"tier": "best", "prompt": "x"}, "done", 100.0,
         {"rc": 0, "new_files": [hit]}),
        ("job2", {"tier": "artistic", "prompt": "y"}, "done", 200.0,
         {"rc": 0, "new_files": ["/outputs/other.png"], "contact_sheet": hit}),
    ])
    job = jobs.find_job_by_file(hit, db_path=db)
    assert job is not None and job["id"] == "job1"


def test_find_job_by_file_returns_none_when_unclaimed(tmp_path):
    db = tmp_path / "jobs.sqlite3"
    _seed(db, [
        ("job1", {"tier": "best", "prompt": "x"}, "done", 100.0,
         {"rc": 0, "new_files": ["/outputs/other.png"]}),
    ])
    assert jobs.find_job_by_file("/outputs/missing.png", db_path=db) is None


def test_review_upsert_clear_and_surfacing(tmp_path):
    db = tmp_path / "jobs.sqlite3"
    _seed(db, [("job1", {"tier": "best", "prompt": "x"}, "done", 100.0, None)])

    job = jobs.set_review("job1", "up", note="keeper", db_path=db)
    assert job["review"]["verdict"] == "up" and job["review"]["note"] == "keeper"

    job = jobs.set_review("job1", "down", db_path=db)  # upsert overwrites
    assert job["review"]["verdict"] == "down" and job["review"]["note"] == ""

    # review rides along on list/search projections
    assert jobs.search_jobs(db_path=db)["jobs"][0]["review"]["verdict"] == "down"
    assert jobs.get_job("job1", db_path=db)["review"]["verdict"] == "down"

    job = jobs.set_review("job1", None, db_path=db)  # null clears
    assert job["review"] is None

    assert jobs.set_review("nope", "up", db_path=db) is None
    try:
        jobs.set_review("job1", "meh", db_path=db)
        raise AssertionError("invalid verdict must raise")
    except ValueError:
        pass
