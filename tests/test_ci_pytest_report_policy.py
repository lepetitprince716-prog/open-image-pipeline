from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "tools" / "ci" / "check_pytest_report.py"
SPEC = importlib.util.spec_from_file_location("ci_pytest_report_policy", POLICY_PATH)
assert SPEC and SPEC.loader
POLICY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(POLICY)


def _write_report(path: Path, cases: list[tuple[str, str, str | None]]) -> None:
    body = []
    for classname, name, skipped in cases:
        child = "" if skipped is None else f'<skipped message="{skipped}" />'
        body.append(f'<testcase classname="{classname}" name="{name}">{child}</testcase>')
    path.write_text(f"<testsuites><testsuite>{''.join(body)}</testsuite></testsuites>", encoding="utf-8")


def test_exact_skip_allowlist_passes(tmp_path: Path) -> None:
    report = tmp_path / "report.xml"
    allowlist = tmp_path / "allowlist.json"
    _write_report(report, [("tests.test_a", "test_optional", "optional dependency missing")])
    allowlist.write_text(
        json.dumps(
            {
                "version": 1,
                "skips": [
                    {
                        "node": "tests.test_a::test_optional",
                        "message_contains": "dependency missing",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert POLICY.check_report(report, allowlist=allowlist) == []


def test_unexpected_and_disappeared_skips_fail(tmp_path: Path) -> None:
    report = tmp_path / "report.xml"
    allowlist = tmp_path / "allowlist.json"
    _write_report(report, [("tests.test_new", "test_hidden", "new condition")])
    allowlist.write_text(
        json.dumps(
            {
                "version": 1,
                "skips": [
                    {"node": "tests.test_old::test_expected", "message_contains": "old condition"}
                ],
            }
        ),
        encoding="utf-8",
    )
    report_text = "\n".join(POLICY.check_report(report, allowlist=allowlist))
    assert "unexpected skip" in report_text
    assert "expected skip disappeared" in report_text


def test_local_mode_allows_expected_skips_to_disappear_but_rejects_unknown_skips(tmp_path: Path) -> None:
    report = tmp_path / "report.xml"
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(
        json.dumps(
            {
                "version": 1,
                "skips": [
                    {"node": "tests.test_optional::test_asset", "message_contains": "asset missing"}
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_report(report, [("tests.test_optional", "test_asset", None)])
    assert POLICY.check_report(report, allowlist=allowlist, allow_missing_expected=True) == []

    _write_report(report, [("tests.test_new", "test_hidden", "new condition")])
    problems = POLICY.check_report(report, allowlist=allowlist, allow_missing_expected=True)
    assert any("unexpected skip" in problem for problem in problems)


def test_zero_skip_mode_is_fail_closed(tmp_path: Path) -> None:
    report = tmp_path / "report.xml"
    _write_report(report, [("tests.test_a", "test_ok", None)])
    assert POLICY.check_report(report, allowlist=None, expected_skips=0) == []
    _write_report(report, [("tests.test_a", "test_hidden", "hidden")])
    assert POLICY.check_report(report, allowlist=None, expected_skips=0)
